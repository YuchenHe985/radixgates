#!/usr/bin/env bash
# Failure-injection benchmark: the ORIGINAL gateway (git tag upstream-snapshot) vs this fork,
# both driven by the same config file and the same load, in front of 4 simulated SGLang workers.
#   ./run_sim.sh crash      worker 31001 is SIGKILLed at FAULT_AT and restarted at RECOVER_AT
#   ./run_sim.sh brownout   worker 31001 gets +3 s time-to-first-token (gray failure; /health stays OK)
# The original ignores the "reliability" block of the config; the fork reads it.
# Workers are simulated (gateway-go/internal/mock): nothing here measures a GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/bin"
SCENARIO="${1:-crash}"
DURATION="${DURATION:-30s}"
FAULT_AT="${FAULT_AT:-8}"
RECOVER_AT="${RECOVER_AT:-20}"
CLIENTS="${CLIENTS:-16}"
GW_PORT=18080
BASE_PORT=31000
OUT="$ROOT/benchmarks/results/sim/$SCENARIO"
CONFIG="$ROOT/benchmarks/configs/sim.json"

mkdir -p "$BIN"
WT="$(mktemp -d)"
cleanup() { pkill -f "$BIN/mockworker" 2>/dev/null || true; pkill -f "$BIN/gateway-" 2>/dev/null || true; git -C "$ROOT" worktree remove --force "$WT/orig" 2>/dev/null || true; rm -rf "$WT"; }
trap cleanup EXIT

# Build the upgraded gateway, the tools, and the original gateway from the upstream-snapshot tag.
( cd "$ROOT/gateway-go" && go build -o "$BIN/gateway-upgraded" . && go build -o "$BIN/mockworker" ./cmd/mockworker && go build -o "$BIN/loadgen" ./cmd/loadgen )
git -C "$ROOT" worktree add --detach "$WT/orig" upstream-snapshot >/dev/null 2>&1
( cd "$WT/orig/gateway-go" && go build -o "$BIN/gateway-original" . )

start_worker() {
  "$BIN/mockworker" -addr ":$((BASE_PORT + $1))" -name "w$1" -ttft-cold 100ms -ttft-warm 50ms -tpot 15ms -tokens 16 -capacity 4 -cache 4 >/dev/null 2>&1 &
}
wait_http() { for _ in $(seq 1 100); do curl -fs -o /dev/null "$1" 2>/dev/null && return 0; sleep 0.1; done; echo "timeout waiting for $1" >&2; return 1; }
inject() {
  case "$SCENARIO" in
    crash)    pkill -9 -f "mockworker -addr :$((BASE_PORT + 1)) " || true ;;
    brownout) curl -fs -X POST "http://127.0.0.1:$((BASE_PORT + 1))/admin/fault" -d '{"mode":"slow","delay_ms":3000}' >/dev/null ;;
    *) echo "unknown scenario $SCENARIO" >&2; exit 1 ;;
  esac
}
recover() {
  case "$SCENARIO" in
    crash)    start_worker 1 ;;
    brownout) curl -fs -X POST "http://127.0.0.1:$((BASE_PORT + 1))/admin/fault" -d '{"mode":"none"}' >/dev/null ;;
  esac
}

mkdir -p "$OUT"
{
  echo "date_utc:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "machine:    $(uname -sm)"
  echo "cpu:        $( (sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) | sed 's/^ *//')"
  echo "go:         $(go version)"
  echo "original:   git tag upstream-snapshot ($(git -C "$ROOT" rev-parse --short upstream-snapshot))"
  echo "upgraded:   $(git -C "$ROOT" rev-parse --short HEAD)"
  echo "scenario:   $SCENARIO (fault at ${FAULT_AT}s, recover at ${RECOVER_AT}s, duration $DURATION, clients $CLIENTS)"
  echo "workers:    4 simulated (ttft cold 100ms / warm 50ms, tpot 15ms, 16 tokens, capacity 4, prefix cache 4)"
  echo "NOTE:       workers are simulated; numbers describe gateway behaviour, not GPU performance."
} > "$OUT/ENVIRONMENT.txt"

for MODE in original upgraded; do
  echo "=== $SCENARIO / $MODE ==="
  pkill -f "$BIN/mockworker" 2>/dev/null || true; pkill -f "$BIN/gateway-" 2>/dev/null || true; sleep 0.3
  for i in 0 1 2 3; do start_worker "$i"; done
  for i in 0 1 2 3; do wait_http "http://127.0.0.1:$((BASE_PORT + i))/health"; done
  CONFIG_PATH="$CONFIG" "$BIN/gateway-$MODE" 2> "$OUT/gateway_$MODE.log" &
  wait_http "http://127.0.0.1:$GW_PORT/health"
  sleep 1

  mkdir -p "$OUT/$MODE"
  ( sleep "$FAULT_AT"; inject; sleep $((RECOVER_AT - FAULT_AT)); recover ) &
  INJECTOR=$!
  "$BIN/loadgen" -url "http://127.0.0.1:$GW_PORT/v1/chat" -clients "$CLIENTS" -duration "$DURATION" \
    -label "$SCENARIO-$MODE" -out "$OUT/$MODE" | grep -E '"(total|ok|failed|success_rate|retried_requests)"|"p(50|95|99)"' | head -12
  wait "$INJECTOR"
  curl -fs "http://127.0.0.1:$GW_PORT/metrics" | grep -E '^radixgates_' > "$OUT/$MODE/metrics.prom" || true
done
echo "results in $OUT"

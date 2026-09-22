#!/usr/bin/env bash
# collect_docker_logs.sh — save logs from running PD containers to examples/logs/
#
# Usage:
#   bash scripts/collect_docker_logs.sh              # save all PD container logs
#   bash scripts/collect_docker_logs.sh --lines 200  # keep only the latest 200 lines

set -euo pipefail

LINES="${2:-}"          # optional: --lines N
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../examples/logs"
mkdir -p "$LOG_DIR"

TS="$(date '+%Y%m%d_%H%M%S')"

CONTAINERS=(gateway sglang-router sglang-prefill sglang-decode)

for cname in "${CONTAINERS[@]}"; do
    if ! docker ps --format '{{.Names}}' | grep -q "^${cname}$"; then
        echo "  [skip] ${cname} is not running"
        continue
    fi

    out="${LOG_DIR}/${cname}_${TS}.log"
    if [[ -n "$LINES" ]]; then
        docker logs --tail "$LINES" "$cname" > "$out" 2>&1
    else
        docker logs "$cname" > "$out" 2>&1
    fi
    echo "  [saved] ${cname} → ${out}"
done

echo ""
echo "Logs saved to ${LOG_DIR}/"
ls -lh "${LOG_DIR}/"*"${TS}"* 2>/dev/null || true

#!/bin/bash
# =============================================================================
# benchmark.sh — Pressure test the SGLang Inference Platform
# Requires: wrk2 (brew install wrk2) or ab (built-in on macOS)
# =============================================================================

GATEWAY_URL="http://localhost:8080"
DURATION="${1:-60s}"
CONCURRENCY="${2:-50}"
RATE="${3:-100}"  # requests per second (wrk2)

echo "======================================================="
echo "  SGLang Platform Benchmark"
echo "  URL: $GATEWAY_URL  | Duration: $DURATION"
echo "  Concurrency: $CONCURRENCY | Target Rate: $RATE req/s"
echo "======================================================="

# ── Test 1: Warm-up (verify gateway is up) ────────────────────────────────────
echo "[Bench] Sanity check..."
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$GATEWAY_URL/v1/chat" \
    -H "Content-Type: application/json" \
    -d '{"messages": [{"role": "user", "content": "Hello!"}]}')

if [ "$RESPONSE" != "202" ]; then
    echo "[Bench] ❌ Gateway not ready (HTTP $RESPONSE). Aborting."
    exit 1
fi
echo "[Bench] ✅ Gateway is up (HTTP 202)."

# ── Test 2: Throughput test (no prefix cache) ─────────────────────────────────
echo ""
echo "[Bench] ── Test 2: Baseline Throughput (random prompts) ──"
if command -v wrk2 &>/dev/null; then
    wrk2 -t4 -c"$CONCURRENCY" -d"$DURATION" -R"$RATE" \
        --script=scripts/wrk_random_prompt.lua \
        "$GATEWAY_URL/v1/chat" 2>&1 | tee /tmp/bench_baseline.txt
else
    echo "[Bench] wrk2 not found. Using ab (install wrk2 for more accurate results)."
    ab -n 500 -c "$CONCURRENCY" -p scripts/sample_payload.json \
        -T "application/json" "$GATEWAY_URL/v1/chat" 2>&1 | tee /tmp/bench_baseline.txt
fi

# ── Test 3: Prefix Cache Test (identical system prompt) ──────────────────────
echo ""
echo "[Bench] ── Test 3: Prefix Cache Test (same system prompt) ──"
echo "[Bench] Expect significantly lower TTFT for repeated prefixes."
if command -v wrk2 &>/dev/null; then
    wrk2 -t4 -c"$CONCURRENCY" -d"$DURATION" -R"$RATE" \
        --script=scripts/wrk_fixed_prefix.lua \
        "$GATEWAY_URL/v1/chat" 2>&1 | tee /tmp/bench_cached.txt
fi

echo ""
echo "[Bench] ── Results Summary ──"
echo "  Baseline: /tmp/bench_baseline.txt"
echo "  Cached:   /tmp/bench_cached.txt"
echo ""
echo "[Bench] SGLang Cache Metrics:"
curl -s http://localhost:30000/metrics 2>/dev/null | grep -E "cache|token|request" | head -20
echo ""
echo "[Bench] Done."

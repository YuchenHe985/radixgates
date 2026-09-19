#!/usr/bin/env bash
# =============================================================================
# start_sglang_pd_prefill.sh
# Launches SGLang in PREFILL-ONLY mode for PD disaggregation.
#
# In SGLang PD disaggregation:
#   - Prefill server: compute-intensive (high FLOPS GPU)
#   - Decode server:  memory-bandwidth-intensive (high HBM bandwidth GPU)
#   - Decode server is the REQUEST ENTRY POINT; Gateway routes to decode.
#   - Decode server bootstraps this prefill server via port 8998.
#
# Usage:
#   MODEL=NousResearch/Meta-Llama-3-8B-Instruct \
#   DECODE_HOST=decode-server-ip \
#   bash scripts/start_sglang_pd_prefill.sh
# =============================================================================

set -euo pipefail

MODEL="${MODEL:-NousResearch/Meta-Llama-3-8B-Instruct}"
PORT="${PORT:-30000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
DISAGG_BACKEND="${DISAGG_BACKEND:-mooncake}"
TENSOR_PARALLEL="${TENSOR_PARALLEL_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION_STATIC:-0.85}"

echo "╔══════════════════════════════════════════╗"
echo "║  SGLang PD — PREFILL-ONLY Server         ║"
echo "║  Model :  ${MODEL:0:40}"
echo "║  Port  :  ${PORT}                           ║"
echo "║  Bootstrap port: ${BOOTSTRAP_PORT}               ║"
echo "║  Transfer: ${DISAGG_BACKEND}                    ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "ℹ️  This is the PREFILL node. sglang_router routes prefill work here."
echo "   Requests do NOT arrive here directly — use sglang_router as entry point."
echo ""

python3 -m sglang.launch_server \
    --model-path "${MODEL}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
    --disaggregation-transfer-backend "${DISAGG_BACKEND}" \
    --host 0.0.0.0 \
    --trust-remote-code

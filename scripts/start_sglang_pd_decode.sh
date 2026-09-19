#!/usr/bin/env bash
# =============================================================================
# start_sglang_pd_decode.sh
# Launches SGLang in DECODE-ONLY mode for PD disaggregation.
#
# In SGLang PD disaggregation (v0.3+), the sglang_router is the request entry
# point. The decode server receives decode work routed by sglang_router.
# The decode node does NOT need --prefill-server-url; the router handles that.
#
# Usage (single machine, GPU 1):
#   CUDA_VISIBLE_DEVICES=1 \
#   bash scripts/start_sglang_pd_decode.sh
#
# Then launch the router:
#   python3 -m sglang_router.launch_router --pd-disaggregation \
#     --prefill http://localhost:30000 8998 \
#     --decode http://localhost:30001 \
#     --host 0.0.0.0 --port 9000
# =============================================================================

set -euo pipefail

MODEL="${MODEL:-NousResearch/Meta-Llama-3-8B-Instruct}"
PORT="${PORT:-30001}"
BASE_GPU_ID="${BASE_GPU_ID:-1}"
DISAGG_BACKEND="${DISAGG_BACKEND:-mooncake}"
TENSOR_PARALLEL="${TENSOR_PARALLEL_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION_STATIC:-0.85}"

echo "╔══════════════════════════════════════════╗"
echo "║  SGLang PD — DECODE-ONLY Server          ║"
echo "║  Model :  ${MODEL:0:40}"
echo "║  Port  :  ${PORT}                           ║"
echo "║  Base GPU ID: ${BASE_GPU_ID}                    ║"
echo "║  Transfer: ${DISAGG_BACKEND}                    ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "ℹ️  Entry point is sglang_router, NOT this server directly."
echo "   Add to router: --decode http://localhost:${PORT}"
echo ""

python3 -m sglang.launch_server \
    --model-path "${MODEL}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --base-gpu-id "${BASE_GPU_ID}" \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend "${DISAGG_BACKEND}" \
    --trust-remote-code

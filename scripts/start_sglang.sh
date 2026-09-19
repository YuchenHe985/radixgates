#!/bin/bash
# =============================================================================
# start_sglang.sh — Launch SGLang inference server instance(s)
# Usage: bash scripts/start_sglang.sh [model_path]
# =============================================================================

MODEL_PATH="${1:-NousResearch/Meta-Llama-3-8B-Instruct}"
LOG_DIR="logs/sglang"
mkdir -p "$LOG_DIR"

echo "[SGLang] Starting instance on GPU 0, port 30000..."
echo "[SGLang] Model: $MODEL_PATH"

# Launch SGLang server (GPU 0)
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port 30000 \
    --host 0.0.0.0 \
    --dp-size 1 \
    --enable-prefix-caching \
    --chunked-prefill-size 8192 \
    --mem-fraction-static 0.85 \
    --max-total-tokens 32768 \
    > "$LOG_DIR/sglang_gpu0.log" 2>&1 &

echo "[SGLang] Instance PID=$! started. Log: $LOG_DIR/sglang_gpu0.log"
echo "[SGLang] Waiting for server to be ready..."

# Poll until the server responds
for i in {1..60}; do
    if curl -s http://localhost:30000/health > /dev/null 2>&1; then
        echo "[SGLang] ✅ Server ready! Health check passed."
        break
    fi
    echo "[SGLang] Waiting... ($i/60)"
    sleep 5
done

echo "[SGLang] All instances started. To monitor logs:"
echo "  tail -f $LOG_DIR/sglang_gpu0.log"
echo ""
echo "[SGLang] To check prefix cache metrics:"
echo "  curl localhost:30000/metrics | grep cache"

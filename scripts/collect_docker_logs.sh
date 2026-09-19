#!/usr/bin/env bash
# collect_docker_logs.sh — 把当前运行的 PD 容器日志保存到 examples/logs/
#
# 用法：
#   bash scripts/collect_docker_logs.sh            # 保存所有 PD 容器日志
#   bash scripts/collect_docker_logs.sh --lines 200  # 只保留最近 200 行

set -euo pipefail

LINES="${2:-}"          # --lines N 可选
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../examples/logs"
mkdir -p "$LOG_DIR"

TS="$(date '+%Y%m%d_%H%M%S')"

CONTAINERS=(gateway sglang-router sglang-prefill sglang-decode)

for cname in "${CONTAINERS[@]}"; do
    if ! docker ps --format '{{.Names}}' | grep -q "^${cname}$"; then
        echo "  [skip] ${cname} 未运行"
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
echo "日志已保存到 ${LOG_DIR}/"
ls -lh "${LOG_DIR}/"*"${TS}"* 2>/dev/null || true

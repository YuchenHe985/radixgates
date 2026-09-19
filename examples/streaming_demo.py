"""
streaming_demo.py — RadixGates Gateway 流式响应演示（SSE / streaming）。

架构：Client → Gateway → SGLang（直连 SSE，无 Kafka / WebSocket / task_id）
与后端并行模式无关，适用于 DP / TP / EP 任意部署。

启动（本地 Docker）：
  docker compose -f docker-compose.yml up -d
  python3 examples/streaming_demo.py

启动（裸机，no-Docker）：
  GATEWAY_URL=http://localhost:8081 python3 examples/streaming_demo.py
"""

import json
import os
import sys
import time
from datetime import datetime

try:
    import requests
except ImportError:
    print("Please install requests: pip3 install requests")
    sys.exit(1)

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8080")

_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_log_path = os.path.join(_LOG_DIR, f"streaming_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
_log_file = open(_log_path, "w", buffering=1)

class _Tee:
    def __init__(self, *targets): self._targets = targets
    def write(self, s):
        for t in self._targets: t.write(s)
    def flush(self):
        for t in self._targets: t.flush()

sys.stdout = _Tee(sys.__stdout__, _log_file)
print(f"[log → {_log_path}]")


def check_mode():
    try:
        r = requests.get(f"{GATEWAY_URL}/health", timeout=5)
        mode = r.json().get("mode", "unknown")
        if mode != "direct":
            print(f"⚠️  Gateway 当前模式: {mode!r}，不是 'direct'")
            print("   请用 ROUTING_MODE=direct docker compose up 重启 Gateway")
            sys.exit(1)
    except Exception as e:
        print(f"❌ 无法连接 Gateway ({GATEWAY_URL}): {e}")
        sys.exit(1)


def stream_chat(question: str):
    """POST /v1/chat (stream=true) → 逐 token 打印到终端。"""
    payload = {
        "model": "NousResearch/Meta-Llama-3-8B-Instruct",
        "messages": [{"role": "user", "content": question}],
        "stream": True,
    }

    print("\n[Gateway → SGLang 直连，实时 SSE token 流]\n")
    print("─" * 50)

    start = time.time()
    token_count = 0

    try:
        with requests.post(
            f"{GATEWAY_URL}/v1/chat",
            json=payload,
            stream=True,
            timeout=300,
        ) as resp:
            if resp.status_code != 200:
                print(f"❌ HTTP {resp.status_code}: {resp.text[:200]}")
                return

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8")
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"].get("content") or ""
                    if delta:
                        print(delta, end="", flush=True)
                        token_count += 1
                except json.JSONDecodeError:
                    pass

    except KeyboardInterrupt:
        pass

    elapsed = time.time() - start
    print(f"\n{'─'*50}")
    print(f"✅ 完成  |  耗时: {elapsed:.2f}s  |  ~{token_count} tokens")
    if elapsed > 0:
        print(f"   吞吐: ~{token_count/elapsed:.1f} tokens/s")


from _log_utils import save_container_logs as _save_container_logs

def save_container_logs(ts: str):
    _save_container_logs(_LOG_DIR, ts)


def main():
    print("=" * 50)
    print(" RadixGates — 实时流式响应 Demo (Direct 模式)")
    print("=" * 50)

    check_mode()
    print("✅ Gateway: direct 模式（无 Kafka，无 WebSocket，纯 SSE）\n")

    question = input("请输入你的问题 (直接回车用默认问题): ").strip()
    if not question:
        question = "Write a short poem about GPU computing."

    stream_chat(question)

    _ts = os.path.basename(_log_path).replace("streaming_demo_", "").replace(".log", "")
    save_container_logs(_ts)


if __name__ == "__main__":
    main()

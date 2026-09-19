"""
pd_demo.py — PD (Prefill-Decode) 分离模式演示与基准测试

架构：
  GPU 0 → SGLang Prefill Server (:30000)  — 计算密集型，只做 prompt 处理
  GPU 1 → SGLang Decode Server  (:30001)  — 带宽密集型，只做 token 生成
  Gateway (:8080)               — 路由到 Decode Server（请求入口）

启动方式：
  docker compose -f docker-compose.pd.yml up -d                                 # mooncake 默认（推荐）
  PREFILL_BACKEND=mooncake DECODE_BACKEND=fake docker compose -f docker-compose.pd.yml up -d  # 快速路由验证

对比基准（同一模型，combined 模式 vs PD 分离）：
  Combined：prefill 和 decode 在同一 GPU 互相抢资源
  PD 分离： prefill 独占计算资源，decode 独占带宽资源
"""

import concurrent.futures
import json
import os
import statistics
import sys
import time
from datetime import datetime

try:
    import requests
except ImportError:
    print("pip3 install requests")
    sys.exit(1)

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8080")

_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_log_path = os.path.join(_LOG_DIR, f"pd_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
_log_file = open(_log_path, "w", buffering=1)

class _Tee:
    def __init__(self, *targets): self._targets = targets
    def write(self, s):
        for t in self._targets: t.write(s)
    def flush(self):
        for t in self._targets: t.flush()

sys.stdout = _Tee(sys.__stdout__, _log_file)
print(f"[log → {_log_path}]")

# 不同长度的 prompt 用于测试 TTFT（首 token 时间）
# PD 分离对长 prompt 的 TTFT 改善最明显（prefill 计算密集）
PROMPTS = {
    "short":  "Say hi.",
    "medium": "Explain what a GPU is in 3 sentences.",
    "long": (
        "The following is a detailed technical document about transformer architecture. "
        "Transformers use self-attention mechanisms to process sequential data. "
        "Each attention head computes queries, keys, and values from the input embeddings. "
        "The scaled dot-product attention formula is Attention(Q,K,V) = softmax(QK^T/√d_k)V. "
        "Multi-head attention allows the model to attend to different representation subspaces. "
        "Position encodings are added to preserve sequence order information. "
        "Feed-forward networks follow each attention layer with two linear transformations. "
        "Layer normalization and residual connections stabilize training. "
        "The encoder processes input sequences while the decoder generates output tokens. "
        "Based on this technical context, write a one-sentence summary of transformers."
    ),
}


def check_gateway():
    try:
        r = requests.get(f"{GATEWAY_URL}/health", timeout=5)
        health = r.json()
        mode = health.get("mode", "unknown")
        print(f"✅ Gateway: mode={mode!r}")
        if mode != "direct":
            print("⚠️  Gateway 需要 routing_mode=direct (配置中 sglang_instances 指向 decode server)")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Gateway 连不上: {e}")
        sys.exit(1)


def measure_ttft(prompt: str, label: str) -> float:
    """测量 TTFT（首 token 时间），使用流式 API。"""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 1,  # 只要第一个 token
    }
    t_start = time.time()
    ttft = None
    try:
        with requests.post(
            f"{GATEWAY_URL}/v1/chat",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"].get("content") or ""
                    if delta and ttft is None:
                        ttft = time.time() - t_start
                        break
                except Exception:
                    pass
    except Exception as e:
        print(f"  ❌ {label}: {e}")
        return -1.0
    return ttft or (time.time() - t_start)


def benchmark_ttft():
    """TTFT 基准：短/中/长 prompt，每个测 5 次取中位数。"""
    print("\n" + "=" * 60)
    print("⏱️  TTFT 基准测试（首 token 时间）")
    print("   PD 分离对长 prompt 的 TTFT 改善最显著")
    print("=" * 60)

    results = {}
    for name, prompt in PROMPTS.items():
        times = []
        token_count = len(prompt.split())
        print(f"\n[{name}] ~{token_count} 词 prompt", end="", flush=True)
        for _ in range(5):
            t = measure_ttft(prompt, name)
            if t > 0:
                times.append(t)
            print(".", end="", flush=True)
        if times:
            med = statistics.median(times)
            results[name] = med
            print(f"  中位 TTFT: {med*1000:.1f}ms  (min={min(times)*1000:.1f}ms)")
        else:
            print("  失败")

    return results


def benchmark_throughput():
    """并发吞吐：30 个请求同时发，测总耗时和成功率。"""
    print("\n" + "=" * 60)
    print("🚀 并发吞吐测试（30 并发，medium prompt）")
    print("=" * 60)

    prompt = PROMPTS["medium"]
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 64,
    }

    def single(i):
        t = time.time()
        try:
            r = requests.post(f"{GATEWAY_URL}/v1/chat", json=payload, timeout=120)
            r.raise_for_status()
            ans = r.json()["choices"][0]["message"]["content"].strip()[:40]
            return True, time.time() - t, ans
        except Exception as e:
            return False, time.time() - t, str(e)

    t_total = time.time()
    success = 0
    fail = 0
    latencies = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as ex:
        futs = [ex.submit(single, i) for i in range(30)]
        for f in concurrent.futures.as_completed(futs):
            ok, lat, ans = f.result()
            if ok:
                success += 1
                latencies.append(lat)
                print(f"  ✅ {lat:.1f}s → {ans}")
            else:
                fail += 1
                print(f"  ❌ {ans}")

    total = time.time() - t_total
    print(f"\n总耗时: {total:.1f}s  |  成功: {success}/30  |  失败: {fail}/30")
    if latencies:
        print(f"延迟: P50={statistics.median(latencies)*1000:.0f}ms  "
              f"P95={sorted(latencies)[int(len(latencies)*0.95)]*1000:.0f}ms")


from _log_utils import save_container_logs as _save_container_logs

def save_container_logs(ts: str):
    _save_container_logs(_LOG_DIR, ts)


def main():
    print("=" * 60)
    print("🔬 RadixGates — PD 分离模式演示")
    print("   Prefill: GPU 0 (:30000)  Decode: GPU 1 (:30001)")
    print(f"   Gateway: {GATEWAY_URL}")
    print("=" * 60)

    check_gateway()

    print("\n启动方式对照：")
    print("  docker compose -f docker-compose.pd.yml up -d")
    print("    → mooncake 默认，RTX 4090 via PCIe（推荐）")
    print("  PREFILL_BACKEND=mooncake DECODE_BACKEND=fake docker compose -f docker-compose.pd.yml up -d")
    print("    → 仅验证路由流程，不实际传输 KV cache")

    try:
        benchmark_ttft()
        benchmark_throughput()
    except KeyboardInterrupt:
        pass

    print("\n" + "=" * 60)
    print("💡 结果解读：")
    print("  long prompt TTFT 显著低于 combined 模式 → PD 分离有效")
    print("  throughput 与 combined 相当或更高 → GPU 利用率提升")
    print("=" * 60)

    _ts = os.path.basename(_log_path).replace("pd_demo_", "").replace(".log", "")
    save_container_logs(_ts)


if __name__ == "__main__":
    main()

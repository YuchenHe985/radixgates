"""
pd_demo.py — Prefill/decode (PD) disaggregation validation and benchmark.

Architecture:
  GPU 0 → SGLang prefill server (:30000) — prompt processing
  GPU 1 → SGLang decode server  (:30001) — token generation
  Gateway (:8080)               — request entry point

Launch:
  docker compose -f docker-compose.pd.yml up -d                                 # mooncake default
  PREFILL_BACKEND=mooncake DECODE_BACKEND=fake docker compose -f docker-compose.pd.yml up -d  # routing-only check

For a controlled comparison, run the same model and prompts in combined and
disaggregated modes and report both results.
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

# Prompt lengths used to measure time to first token (TTFT).
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
            print("⚠️  Gateway must use routing_mode=direct with sglang_instances pointing to the decode server")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot reach gateway: {e}")
        sys.exit(1)


def measure_ttft(prompt: str, label: str) -> float:
    """Measure TTFT with the streaming API."""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 1,  # Only the first token is needed for TTFT.
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
    """Measure median TTFT over five short, medium, and long prompt runs."""
    print("\n" + "=" * 60)
    print("⏱️  TTFT benchmark")
    print("   Report these values beside a separately measured combined-mode baseline")
    print("=" * 60)

    results = {}
    for name, prompt in PROMPTS.items():
        times = []
        token_count = len(prompt.split())
        print(f"\n[{name}] ~{token_count}-word prompt", end="", flush=True)
        for _ in range(5):
            t = measure_ttft(prompt, name)
            if t > 0:
                times.append(t)
            print(".", end="", flush=True)
        if times:
            med = statistics.median(times)
            results[name] = med
            print(f"  median TTFT: {med*1000:.1f}ms  (min={min(times)*1000:.1f}ms)")
        else:
            print("  failed")

    return results


def benchmark_throughput():
    """Send 30 concurrent requests and report wall time and success count."""
    print("\n" + "=" * 60)
    print("🚀 Concurrent load test (30 medium prompts)")
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
    print(f"\nWall time: {total:.1f}s  |  success: {success}/30  |  failed: {fail}/30")
    if latencies:
        print(f"Latency: P50={statistics.median(latencies)*1000:.0f}ms  "
              f"P95={sorted(latencies)[int(len(latencies)*0.95)]*1000:.0f}ms")


from _log_utils import save_container_logs as _save_container_logs

def save_container_logs(ts: str):
    _save_container_logs(_LOG_DIR, ts)


def main():
    print("=" * 60)
    print("🔬 RadixGates — prefill/decode disaggregation validation")
    print("   Prefill: GPU 0 (:30000)  Decode: GPU 1 (:30001)")
    print(f"   Gateway: {GATEWAY_URL}")
    print("=" * 60)

    check_gateway()

    print("\nLaunch options:")
    print("  docker compose -f docker-compose.pd.yml up -d")
    print("    → mooncake transfer backend")
    print("  PREFILL_BACKEND=mooncake DECODE_BACKEND=fake docker compose -f docker-compose.pd.yml up -d")
    print("    → routing-only validation; no real KV-cache transfer")

    try:
        benchmark_ttft()
        benchmark_throughput()
    except KeyboardInterrupt:
        pass

    print("\n" + "=" * 60)
    print("💡 Interpretation:")
    print("  Compare TTFT and throughput with a same-model combined-mode baseline")
    print("  This script alone validates the PD path; it does not establish a speedup")
    print("=" * 60)

    _ts = os.path.basename(_log_path).replace("pd_demo_", "").replace(".log", "")
    save_container_logs(_ts)


if __name__ == "__main__":
    main()

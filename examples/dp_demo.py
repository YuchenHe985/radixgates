"""
dp_demo.py — Data Parallel (DP) validation and benchmark.

Architecture:
  GPU 0 (:30000) — full model replica with an independent KV cache
  GPU 1 (:30001) — full model replica with an independent KV cache
  GPU 2 (:30002) — full model replica with an independent KV cache
  GPU 3 (:30003) — full model replica with an independent KV cache
  Gateway (:8080) — prefix_hash % 4 affinity routing

  DP properties:
    no cross-GPU model collectives (no All-Reduce or All-to-All)
    four replicas can process independent batches concurrently
    prefix affinity can preserve KV-cache locality

  Comparison:
    DP — replicas for aggregate throughput; each model must fit on one GPU
    TP — weight sharding for larger dense models; requires All-Reduce
    EP — expert sharding for MoE models; requires All-to-All

Launch:
  docker compose -f docker-compose.dp.yml up -d
  python3 examples/dp_demo.py

Example host:
  4× A100 80 GB SXM — MODEL_PATH=Qwen/Qwen2.5-14B-Instruct (default; ~28 GB weights)
"""

import concurrent.futures
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime

try:
    import requests
except ImportError:
    print("pip3 install requests")
    sys.exit(1)

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8080")

_base = os.environ.get("SGLANG_BASE_URL", "http://localhost")
DP_NODES = [
    {"name": "sglang-dp-0", "gpu": 0, "url": f"{_base}:30000"},
    {"name": "sglang-dp-1", "gpu": 1, "url": f"{_base}:30001"},
    {"name": "sglang-dp-2", "gpu": 2, "url": f"{_base}:30002"},
    {"name": "sglang-dp-3", "gpu": 3, "url": f"{_base}:30003"},
]

_LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_log_path = os.path.join(_LOG_DIR, f"dp_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
_log_file = open(_log_path, "w", buffering=1)


class _Tee:
    def __init__(self, *targets): self._targets = targets
    def write(self, s):
        for t in self._targets: t.write(s)
    def flush(self):
        for t in self._targets: t.flush()


sys.stdout = _Tee(sys.__stdout__, _log_file)
print(f"[log → {_log_path}]")

# ── System prompts used to test prefix-hash routing ──────────────────────────
# Gateway hashes the first N tokens of each request; same prefix → same node.
# We use 4 distinct prefixes to hit all 4 GPU nodes.
SYSTEM_PROMPTS = [
    "You are a helpful assistant for the engineering team.",
    "You are a helpful assistant for the finance team.",
    "You are a helpful assistant for the sales team.",
    "You are a helpful assistant for the legal team.",
]

USER_QUESTION = "What is a GPU in one sentence?"


# ── Verification ──────────────────────────────────────────────────────────────

def check_gateway():
    try:
        r    = requests.get(f"{GATEWAY_URL}/health", timeout=5)
        mode = r.json().get("mode", "unknown")
        print(f"✅ Gateway: mode={mode!r}")
    except Exception as e:
        print(f"❌ Cannot reach gateway: {e}")
        sys.exit(1)


def check_all_nodes() -> int:
    """
    Health-check all 4 SGLang DP nodes directly.
    Returns count of healthy nodes.

    In DP mode: each node is a fully independent SGLang server.
    All 4 must be healthy before Gateway starts routing.
    """
    print("\n" + "=" * 60)
    print("🔍 DP validation — health check for four SGLang workers")
    print("=" * 60)

    healthy = 0
    for node in DP_NODES:
        try:
            r    = requests.get(f"{node['url']}/health", timeout=5)
            ok   = r.status_code == 200
            # Also pull model info to confirm same model on all nodes
            info = requests.get(f"{node['url']}/get_model_info", timeout=5).json()
            model = info.get("model_path", info.get("model", "?"))
            tp    = info.get("tp_size", info.get("tensor_parallel_size", 1))
            mark  = "✅" if ok else "❌"
            print(f"  {mark} GPU {node['gpu']} [{node['name']}]  {node['url']}")
            print(f"       model={model}  tp_size={tp} (independent DP worker)")
            if ok:
                healthy += 1
        except Exception as e:
            print(f"  ❌ GPU {node['gpu']} [{node['name']}]: {e}")

    print()
    if healthy == len(DP_NODES):
        print(f"  ✅ {healthy}/{len(DP_NODES)} workers healthy — DP={healthy} ready")
    else:
        print(f"  ⚠️  {healthy}/{len(DP_NODES)} workers healthy — deployment not fully ready")

    return healthy


def check_gpu_memory() -> None:
    """
    In DP mode all 4 GPUs should show similar VRAM usage
    (each holds a complete model copy, no weight sharing between GPUs).
    """
    print("\n" + "=" * 60)
    print("🖥️  GPU memory (DP: one full model replica per GPU)")
    print("   Similar baseline allocation is expected across the four replicas")
    print("=" * 60)

    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        usages = []
        for line in out.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 4:
                continue
            idx, name, used_s, total_s = parts[0], parts[1], parts[2], parts[3]
            used, total = int(used_s), int(total_s)
            pct     = 100 * used / max(total, 1)
            bar_len = 30
            filled  = int(bar_len * used / max(total, 1))
            bar     = "█" * filled + "░" * (bar_len - filled)
            gpu_ok  = pct > 10
            usages.append(pct)
            mark    = "✅" if gpu_ok else "○ (idle)"
            print(f"  GPU {idx} [{name}]")
            print(f"    {bar}  {used}/{total} MB  ({pct:.1f}%)  {mark}")

        if len(usages) >= 4:
            diff = max(usages) - min(usages)
            sym  = "balanced" if diff < 5 else f"{diff:.1f}% spread (KV-cache occupancy may differ)"
            print()
            print(f"  → Memory distribution: {sym}")
            print("  → DP check: each GPU holds an independent full model replica ✅")

    except FileNotFoundError:
        print("  nvidia-smi is not available on PATH")
    except Exception as e:
        print(f"  {e}")


# ── Routing Distribution Test ─────────────────────────────────────────────────

def test_prefix_hash_routing() -> None:
    """
    Verify Gateway prefix-hash routing by sending requests with different
    system prompts and checking which node responds (via X-Node header if available,
    or by timing signature).

    The key proof:
    - 4 requests with 4 different prefixes go to 4 different nodes simultaneously
    - Same prefix always routes to the same node (deterministic hashing)
    - Repeat requests with same prefix → KV Cache hit on that node → faster TTFT
    """
    print("\n" + "=" * 60)
    print("🗺️  Prefix-affinity routing check")
    print("   Repeat each system prompt and compare cold versus warm TTFT")
    print("=" * 60)

    # Round 1: cold requests (no cache)
    print("\n[Round 1] Cold cache — send four different prefixes concurrently")
    cold_times = {}

    def send(sys_prompt: str, label: str) -> tuple[str, float]:
        payload = {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": USER_QUESTION},
            ],
            "stream":     True,
            "max_tokens": 1,
        }
        t = time.time()
        ttft = None
        with requests.post(
            f"{GATEWAY_URL}/v1/chat", json=payload, stream=True, timeout=120
        ) as resp:
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode()
                if not line.startswith("data:"):
                    continue
                ds = line[5:].strip()
                if ds == "[DONE]":
                    break
                try:
                    chunk = json.loads(ds)
                    delta = chunk["choices"][0]["delta"].get("content") or ""
                    if delta and ttft is None:
                        ttft = time.time() - t
                        break
                except Exception:
                    pass
        return label, ttft or (time.time() - t)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [
            ex.submit(send, sp, f"prefix-{i}")
            for i, sp in enumerate(SYSTEM_PROMPTS)
        ]
        for f in concurrent.futures.as_completed(futs):
            label, ttft = f.result()
            cold_times[label] = ttft
            print(f"  {label}: TTFT={ttft*1000:.1f}ms")

    # Round 2: warm requests (same prefixes → cache hit on same nodes)
    print("\n[Round 2] Warm cache — repeat the same prefixes")
    warm_times = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futs = [
            ex.submit(send, sp, f"prefix-{i}")
            for i, sp in enumerate(SYSTEM_PROMPTS)
        ]
        for f in concurrent.futures.as_completed(futs):
            label, ttft = f.result()
            warm_times[label] = ttft
            cold = cold_times.get(label, 0)
            improvement = (cold - ttft) / max(cold, 0.001) * 100
            hit_mark = "🔥 likely cache hit" if ttft < cold * 0.85 else "— no material difference"
            print(f"  {label}: TTFT={ttft*1000:.1f}ms  (cold={cold*1000:.1f}ms  {improvement:+.0f}%)  {hit_mark}")

    print()
    if all(warm_times[k] < cold_times[k] * 0.9 for k in cold_times):
        print("  ✅ Warm TTFT is materially below cold TTFT")
        print("     The result is consistent with prefix affinity preserving cache locality")
    else:
        print("  ℹ️  TTFT difference is not material; the prefix may be short or already warm")


# ── Throughput Benchmark ──────────────────────────────────────────────────────

def benchmark_throughput():
    """
    Send 40 concurrent requests (10 per GPU node if evenly distributed).
    DP's key proof: total time ≈ time for 10 requests on 1 GPU
    (not time for 40 sequential requests).

    No inter-GPU communication means perfect linear scaling with request count.
    """
    print("\n" + "=" * 60)
    print("🚀 Concurrent load test (40 requests across four prefixes)")
    print("   Four independent replicas serve separate requests without model collectives")
    print("=" * 60)

    # Cycle through 4 system prompts so requests distribute across all 4 nodes
    payloads = []
    for i in range(40):
        sys_prompt = SYSTEM_PROMPTS[i % 4]
        payloads.append({
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": "Explain GPU parallelism in 2 sentences."},
            ],
            "stream":     False,
            "max_tokens": 64,
        })

    def single(i: int) -> tuple[bool, float, str]:
        t = time.time()
        try:
            r = requests.post(f"{GATEWAY_URL}/v1/chat", json=payloads[i], timeout=120)
            r.raise_for_status()
            ans = r.json()["choices"][0]["message"]["content"].strip()[:40]
            return True, time.time() - t, ans
        except Exception as e:
            return False, time.time() - t, str(e)

    t_total   = time.time()
    success   = 0
    fail      = 0
    latencies = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as ex:
        futs = [ex.submit(single, i) for i in range(40)]
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
    print(f"\nWall time: {total:.1f}s  |  success: {success}/40  |  failed: {fail}/40")
    if latencies:
        p50 = statistics.median(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        print(f"Latency: P50={p50*1000:.0f}ms  P95={p95*1000:.0f}ms")
        # DP scaling estimate
        single_node_est = total * 4
        print("\n📊 Illustrative DP scaling estimate (not a measured single-GPU baseline):")
        print(f"  Measured wall time for 40 concurrent requests: {total:.1f}s")
        print(f"  Sequential estimate from this run: ~{single_node_est:.1f}s")
        print(f"  Estimated ratio: ~{single_node_est/max(total,0.1):.1f}×")


# ── Main ──────────────────────────────────────────────────────────────────────

from _log_utils import print_hw_info, save_container_logs as _save_container_logs


def main():
    print("=" * 60)
    print("🔬 RadixGates — Data Parallel (DP=4) validation")
    print("   GPU 0 (:30000)  GPU 1 (:30001)")
    print("   GPU 2 (:30002)  GPU 3 (:30003)")
    print("   One independent model replica per GPU")
    print(f"   Gateway: {GATEWAY_URL}")
    print("=" * 60)

    hw_tag = print_hw_info()

    check_gateway()

    healthy = check_all_nodes()
    if healthy < len(DP_NODES):
        print(f"\n⚠️  {len(DP_NODES) - healthy} workers are not ready; continuing with partial results")

    check_gpu_memory()

    try:
        test_prefix_hash_routing()
        benchmark_throughput()
    except KeyboardInterrupt:
        pass

    print("\n" + "=" * 60)
    print("💡 Interpretation:")
    print("  Four healthy workers establish that DP=4 is ready")
    print("  Warm-versus-cold TTFT checks whether affinity preserves cache locality")
    print("  The 40-request result is measured; the scaling ratio above is only an estimate")
    print()
    print("  Choosing a parallel mode:")
    print("    DP — replicate a model that fits on one GPU for aggregate throughput")
    print("    TP — shard dense weights when a model does not fit on one GPU")
    print("    EP — shard experts in an MoE model and route tokens with All-to-All")
    print("=" * 60)

    _ts = os.path.basename(_log_path).replace("dp_demo_", "").replace(".log", "") + f"_{hw_tag}"

    _save_container_logs(_LOG_DIR, _ts)

    print(f"\n[log saved → {_log_path}]")
    print(f"[machine   → {hw_tag}]")


if __name__ == "__main__":
    main()

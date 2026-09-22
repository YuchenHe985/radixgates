"""
ep_demo.py — Expert Parallel (EP) and All-to-All validation.

Architecture (Qwen1.5-MoE-A2.7B-Chat example, 64 experts per layer):
  GPU 0: experts 0-15; tokens are dispatched to selected experts with All-to-All
  GPU 1: Expert 16-31
  GPU 2: Expert 32-47
  GPU 3: Expert 48-63

  Per-layer flow:
    attention: TP collectives when TP is enabled
    expert FFN: All-to-All dispatch → local expert computation → All-to-All gather

  EP versus TP:
    TP — tensors or weights are partitioned across ranks
    EP — experts are partitioned across ranks and tokens are routed between them

Bare-metal launch:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \\
    --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat \\
    --tensor-parallel-size 4 --expert-parallel-size 4
  python3 examples/ep_demo.py

Example configuration:
  Model: Qwen/Qwen1.5-MoE-A2.7B-Chat (64 experts, top-4 routing, 14.3B total parameters)
  Storage: approximately 28 GB for fp16 weights
"""

import concurrent.futures
import json
import os
import re
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
SGLANG_URL  = os.environ.get("SGLANG_URL",  "http://localhost:30000")

_LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_log_path = os.path.join(_LOG_DIR, f"ep_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
_log_file = open(_log_path, "w", buffering=1)


class _Tee:
    def __init__(self, *targets): self._targets = targets
    def write(self, s):
        for t in self._targets: t.write(s)
    def flush(self):
        for t in self._targets: t.flush()


sys.stdout = _Tee(sys.__stdout__, _log_file)
print(f"[log → {_log_path}]")

PROMPTS = {
    "short":  "Say hi.",
    "medium": "Explain what Mixture of Experts is in 3 sentences.",
    "long": (
        "The following is a detailed technical document about Mixture of Experts (MoE) "
        "architecture in large language models. "
        "In MoE models, each transformer layer contains multiple expert FFN networks. "
        "A router network (gating function) selects which K experts process each token. "
        "Only the selected experts are activated, keeping compute proportional to K, not N. "
        "Expert Parallelism (EP) distributes these experts across multiple GPUs. "
        "When a token is routed to an expert on a different GPU, "
        "it must be transferred via All-to-All collective communication. "
        "This All-to-All has two phases: dispatch (send tokens to expert GPUs) "
        "and gather (return computed results to originating GPUs). "
        "Load imbalance occurs when some experts receive many more tokens than others, "
        "causing some GPUs to be idle while others are saturated. "
        "Based on this context, explain the trade-off between EP degree and All-to-All overhead."
    ),
}


# ── Verification ──────────────────────────────────────────────────────────────

def check_gateway():
    try:
        r    = requests.get(f"{GATEWAY_URL}/health", timeout=5)
        mode = r.json().get("mode", "unknown")
        print(f"✅ Gateway: mode={mode!r}")
    except Exception as e:
        print(f"❌ Cannot reach gateway: {e}")
        sys.exit(1)


def check_ep_active() -> tuple:
    """
    Verify EP is active via three sources (any one confirming is sufficient):
      1. /get_server_args  — SGLang 0.4.x startup flags
      2. Launch log file   — grep for 'ep_size=N' in stdout log (SGLang 0.5.x)
      3. /get_model_info   — MoE structure (num_experts, top_k)

    Returns (ep_confirmed: bool, ep_size: int|None, tp_size: int|None)
    """
    print("\n" + "=" * 60)
    print("🔍 EP validation — SGLang server parameters")
    print("=" * 60)

    ep_confirmed  = False
    moe_confirmed = False
    ep_size_found: int | None = None
    tp_size_found: int | None = None
    model_path = "?"

    # ── 1. /get_server_args (SGLang 0.4.x) ───────────────────────────────────
    try:
        r = requests.get(f"{SGLANG_URL}/get_server_args", timeout=10)
        if r.status_code == 200:
            args = r.json()
            model_path     = args.get("model_path", args.get("model", "?"))
            tp_size_found  = args.get("tensor_parallel_size", args.get("tp_size"))
            ep_size_raw    = args.get("expert_parallel_size",
                             args.get("ep_size",
                             args.get("enable_expert_parallel")))
            if ep_size_raw is not None:
                if isinstance(ep_size_raw, bool):
                    ep_size_found = tp_size_found if ep_size_raw else 1
                elif isinstance(ep_size_raw, int):
                    ep_size_found = ep_size_raw
                print(f"  source (ep_size)      : /get_server_args (SGLang 0.4.x)")
    except Exception:
        pass

    # ── 2. Launch log file (SGLang 0.5.x) ────────────────────────────────────
    # SGLang 0.5.x logs: ServerArgs(..., tp_size=4, ..., ep_size=4, ...)
    if ep_size_found is None or tp_size_found is None:
        log_candidates = [
            os.environ.get("SGLANG_LOG", ""),
            "/tmp/sglang_ep.log",
            "/tmp/sglang.log",
        ]
        for log_path in log_candidates:
            if not log_path or not os.path.isfile(log_path):
                continue
            try:
                with open(log_path) as f:
                    content = f.read(65536)
                if ep_size_found is None:
                    m = re.search(r"\bep_size=(\d+)|expert[_-]parallel[_-]size[=:\s]+(\d+)",
                                  content, re.IGNORECASE)
                    if m:
                        ep_size_found = int(m.group(1) or m.group(2))
                        print(f"  source (ep_size)      : log file ({log_path})")
                if tp_size_found is None:
                    m2 = re.search(r"\btp_size=(\d+)|tensor[_-]parallel[_-]size[=:\s]+(\d+)",
                                   content, re.IGNORECASE)
                    if m2:
                        tp_size_found = int(m2.group(1) or m2.group(2))
                if model_path == "?":
                    mp = re.search(r"model_path='([^']+)'", content)
                    if mp:
                        model_path = mp.group(1)
                if ep_size_found is not None:
                    break
            except Exception:
                pass

    # ── 3. /get_model_info — MoE structure ──────────────────────────────────
    num_experts = None
    top_k       = None
    print()
    print("  [MoE model structure]")
    try:
        r = requests.get(f"{SGLANG_URL}/get_model_info", timeout=10)
        if r.status_code == 200:
            info = r.json()
            if model_path == "?":
                model_path = info.get("model_path", "?")
            num_experts = info.get("num_experts",
                          info.get("n_routed_experts",
                          info.get("num_routed_experts")))
            top_k       = info.get("num_experts_per_tok",
                          info.get("top_k_experts",
                          info.get("n_top_k_experts")))
            num_hidden  = info.get("num_hidden_layers", "?")

            if num_experts:
                moe_confirmed = True
                ep_n = ep_size_found or 1
                experts_per_gpu = (
                    f"{num_experts // ep_n} per GPU"
                    if ep_n > 1 else "all on one GPU (EP=1)"
                )
                print(f"  num_experts (total)   : {num_experts}  → EP={ep_n}: {experts_per_gpu}")
            else:
                print("  num_experts           : not exposed; verify that an MoE model is loaded")

            if top_k:
                ep_n = ep_size_found or 1
                cross_gpu_pct = (
                    f"up to ~{100*(1 - 1/ep_n):.0f}% of placements may cross ranks"
                    if ep_n > 1 else ""
                )
                print(f"  experts_per_token (K) : {top_k}   {cross_gpu_pct}")
            print(f"  num_hidden_layers     : {num_hidden}")
    except Exception as e:
        print(f"  /get_model_info failed: {e}")

    # ── Print summary ─────────────────────────────────────────────────────────
    print()
    print(f"  endpoint              : {SGLANG_URL}")
    print(f"  model_path            : {model_path}")
    if tp_size_found is not None:
        print(f"  tensor_parallel_size  : {tp_size_found}")
    if ep_size_found is not None:
        ep_ok = ep_size_found > 1
        mark  = "✅ EP + All-to-All enabled" if ep_ok else "⚠️  EP=1 (single GPU)"
        print(f"  expert_parallel_size  : {ep_size_found}   {mark}")
        ep_confirmed = ep_ok
    else:
        print("  expert_parallel_size  : not found in the available sources")

    print()
    if ep_confirmed and moe_confirmed:
        ep_n = ep_size_found or 1
        print(f"  ✅ EP + All-to-All confirmed: ep_size={ep_n}, MoE model loaded")
        print("     forward path: dispatch All-to-All → expert computation → gather All-to-All")
    elif ep_confirmed and not moe_confirmed:
        print("  ✅ EP size confirmed; expert count was not exposed by this model-info format")
    elif moe_confirmed and not ep_confirmed:
        print("  ⚠️  MoE model loaded but ep_size was not found; EP may still be 1")
    else:
        print("  ⚠️  EP was not confirmed by API or logs; GPU allocation is supporting evidence only")

    return ep_confirmed, ep_size_found, tp_size_found


def check_gpu_memory(label: str = "") -> None:
    """
    Print VRAM usage per GPU.

    EP memory pattern differs from TP:
    - TP: perfectly symmetric (weight columns split evenly)
    - EP: Expert weights split, but shared weights (attention, embed) duplicated
          → both GPUs show similar baseline + Expert shard
          → slight asymmetry is normal (hot Expert imbalance)
    """
    title = f"🖥️  GPU memory{(' — ' + label) if label else ''}"
    print("\n" + "=" * 60)
    print(title)
    print("   EP should allocate expert shards across participating GPUs")
    print("   Runtime allocations may differ with expert routing and KV-cache occupancy")
    print("=" * 60)

    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        rows   = []
        loaded = []
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
            gpu_ok  = pct > 20
            loaded.append(gpu_ok)
            mark    = "✅" if gpu_ok else "○ (idle)"
            print(f"  GPU {idx} [{name}]")
            print(f"    {bar}  {used}/{total} MB  ({pct:.1f}%)  {mark}")
            rows.append((idx, used, total, pct))

        if len(rows) >= 2:
            usages = [r[3] for r in rows]
            diff   = max(usages) - min(usages)
            print()
            if all(loaded):
                sym = "balanced" if diff < 5 else f"{diff:.1f}% spread"
                print("  → Both GPUs have material allocation; this is consistent with expert sharding")
                print(f"  → Memory distribution: {sym}")
            else:
                print("  → ⚠️  Only one GPU has material allocation; EP may not be enabled")

    except FileNotFoundError:
        print("  nvidia-smi is not available on PATH")
    except Exception as e:
        print(f"  {e}")


def check_alltoall_in_logs() -> None:
    """
    Search the sglang-ep container logs for All-to-All related keywords.
    SGLang prints EP/All-to-All initialization info at startup.
    This is the most direct evidence that All-to-All communication is configured.
    """
    print("\n" + "=" * 60)
    print("📋 SGLang container logs — All-to-All keyword scan")
    print("=" * 60)

    keywords = [
        "expert_parallel",
        "all_to_all",
        "AllToAll",
        "all-to-all",
        "ep_size",
        "ExpertParallel",
        "dispatch",
        "gather",
    ]

    try:
        result = subprocess.run(
            ["docker", "logs", "sglang-ep", "--tail", "200"],
            capture_output=True, text=True,
        )
        log_text = result.stdout + result.stderr

        if not log_text.strip():
            print("  (container logs are empty or the container is not local)")
            return

        found = []
        for line in log_text.splitlines():
            line_lower = line.lower()
            if any(kw.lower() in line_lower for kw in keywords):
                found.append(line.strip())

        if found:
            print(f"  Found {len(found)} matching log lines:")
            for line in found[:20]:   # Display at most 20 lines.
                print(f"    {line}")
            if len(found) > 20:
                print(f"    ... {len(found)-20} additional lines in the saved container log")
            print()
            print("  ✅ SGLang logs contain Expert Parallel / All-to-All initialization evidence")
        else:
            print("  No All-to-All keyword found")
            print("  Log wording can vary by SGLang version")
            print("  Inspect manually: docker logs sglang-ep | grep -i expert")

    except FileNotFoundError:
        print("  docker is not available on PATH; run this check on the deployment host")
    except Exception as e:
        print(f"  {e}")


# ── Benchmarks ────────────────────────────────────────────────────────────────

def measure_ttft(prompt: str, label: str) -> float:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "stream":     True,
        "max_tokens": 1,
    }
    t_start = time.time()
    ttft    = None
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
    """
    TTFT benchmark for EP.

    EP's All-to-All adds latency per MoE layer (dispatch + gather round trip).
    With NVLink: All-to-All overhead is small (~microseconds per layer).
    With PCIe:   All-to-All overhead is larger, visible in long-prompt TTFT.

    Baseline comparison (same model, EP=1 single GPU vs EP=2):
      EP=2 TTFT may be slightly higher than EP=1 (All-to-All overhead)
      EP=2 Expert VRAM per GPU is halved → enables larger batch sizes
    """
    print("\n" + "=" * 60)
    print("⏱️  TTFT benchmark")
    print("   Interpret results with model, batch, topology, and NCCL settings recorded")
    print("=" * 60)

    results = {}
    for name, prompt in PROMPTS.items():
        times       = []
        token_count = len(prompt.split())
        print(f"\n[{name}] ~{token_count}-word prompt", end="", flush=True)
        for _ in range(5):
            t = measure_ttft(prompt, name)
            if t > 0:
                times.append(t)
            print(".", end="", flush=True)
        if times:
            med           = statistics.median(times)
            results[name] = med
            print(f"  median TTFT: {med*1000:.1f}ms  (min={min(times)*1000:.1f}ms)")
        else:
            print("  failed")

    return results


def benchmark_throughput():
    """
    Concurrent throughput test.

    EP benefit shows at high concurrency:
    - More requests → larger batch → more tokens per forward pass
    - More tokens per forward → All-to-All buckets fill up → better GPU utilization
    - Expert load balancing improves with larger batches (law of large numbers)
    """
    print("\n" + "=" * 60)
    print("🚀 Concurrent load test (20 medium prompts)")
    print("   Report batch and topology because collective efficiency depends on both")
    print("=" * 60)

    prompt  = PROMPTS["medium"]
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "stream":     False,
        "max_tokens": 64,
    }

    def single(_i):
        t = time.time()
        try:
            r   = requests.post(f"{GATEWAY_URL}/v1/chat", json=payload, timeout=120)
            r.raise_for_status()
            ans = r.json()["choices"][0]["message"]["content"].strip()[:40]
            return True, time.time() - t, ans
        except Exception as e:
            return False, time.time() - t, str(e)

    t_total   = time.time()
    success   = 0
    fail      = 0
    latencies = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
        futs = [ex.submit(single, i) for i in range(20)]
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
    print(f"\nWall time: {total:.1f}s  |  success: {success}/20  |  failed: {fail}/20")
    if latencies:
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        print(
            f"Latency: P50={statistics.median(latencies)*1000:.0f}ms  "
            f"P95={p95*1000:.0f}ms"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

from _log_utils import print_hw_info, save_container_logs as _save_container_logs


def main():
    print("=" * 60)
    print("🔬 RadixGates — Expert Parallel (EP) + All-to-All validation")
    print("   GPU 0: Expert  0-63")
    print("   GPU 1: Expert 64-127")
    print("   Token routing across expert ranks uses All-to-All dispatch and gather")
    print(f"   Gateway: {GATEWAY_URL}")
    print("=" * 60)

    hw_tag = print_hw_info()

    check_gateway()

    ep_ok, ep_size, tp_size = check_ep_active()

    # GPU memory: EP keeps attention weights on all GPUs, splits Expert weights
    check_gpu_memory("baseline after model load")

    # Scan SGLang container logs for All-to-All init messages
    check_alltoall_in_logs()

    if not ep_ok:
        print("\n⚠️  EP was not confirmed by API/log; continuing for diagnostics only")

    try:
        benchmark_ttft()
        benchmark_throughput()
    except KeyboardInterrupt:
        pass

    check_gpu_memory("after the load test")

    print("\n" + "=" * 60)
    print("💡 Interpretation:")
    if ep_ok:
        print(f"  expert_parallel_size={ep_size} confirmed ✅")
        print(f"  tensor_parallel_size={tp_size} (TP and EP may be composed)")
    else:
        print("  ep_size was not confirmed; memory distribution is supporting evidence only ⚠️")
    print()
    print("  MoE layers use All-to-All dispatch and gather collectives")
    print("  Measure collective cost on the target topology; this script does not isolate it")
    print()
    print("  EP trade-off:")
    print("    ✅ Expert weights are partitioned instead of fully replicated")
    print("    ⚠️  Token dispatch introduces communication overhead")
    print()
    print("  TP, EP, and TP+EP:")
    print("    TP partitions dense tensor operations")
    print("    EP partitions experts and routes tokens between ranks")
    print("    Compose them only when the model architecture and topology justify it")
    print("=" * 60)

    _ts = os.path.basename(_log_path).replace("ep_demo_", "").replace(".log", "") + f"_{hw_tag}"
    _save_container_logs(_LOG_DIR, _ts)
    print(f"\n[log saved → {_log_path}]")
    print(f"[machine   → {hw_tag}]")


if __name__ == "__main__":
    main()

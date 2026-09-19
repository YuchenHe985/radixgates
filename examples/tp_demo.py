"""
tp_demo.py — Tensor Parallel (TP) 模式演示与验证

架构：
  GPU 0 + GPU 1 → 单个 SGLang 实例 (:30000)，--tensor-parallel-size 2
  每张卡持有约 50% 的模型权重，每层计算后做 All-Reduce 同步激活值
  Gateway (:8080) 路由到该实例 — TP 对 Gateway 完全透明

  与 PD 分离的区别：
    PD  — 两个 SGLang 进程，GPU 0 做 prefill，GPU 1 做 decode
    TP  — 一个 SGLang 进程，GPU 0 + GPU 1 各持有半份权重

启动方式：
  docker compose -f docker-compose.tp.yml up -d
  python3 examples/tp_demo.py

推荐机器（Vast.ai）：
  2× RTX 4090 (24 GB each) + Qwen/Qwen2.5-14B-Instruct
  → 模型 28 GB fp16，单卡放不下，TP=2 是必须项，不是可选项
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
_log_path = os.path.join(_LOG_DIR, f"tp_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
    "medium": "Explain what Tensor Parallelism is in 3 sentences.",
    "long": (
        "The following is a detailed technical document about transformer architecture. "
        "Transformers use self-attention mechanisms to process sequential data. "
        "Each attention head computes queries, keys, and values from the input embeddings. "
        "The scaled dot-product attention formula is Attention(Q,K,V) = softmax(QK^T/√d_k)V. "
        "Multi-head attention allows the model to attend to different representation subspaces. "
        "Position encodings are added to preserve sequence order information. "
        "Feed-forward networks follow each attention layer with two linear transformations. "
        "Layer normalization and residual connections stabilize training. "
        "With Tensor Parallelism, attention heads are split across GPUs and FFN columns are "
        "partitioned, requiring an All-Reduce after each layer to synchronize activations. "
        "Based on this technical context, explain why NVLink is preferred over PCIe for TP."
    ),
}


# ── Verification ──────────────────────────────────────────────────────────────

def check_gateway():
    try:
        r = requests.get(f"{GATEWAY_URL}/health", timeout=5)
        mode = r.json().get("mode", "unknown")
        print(f"✅ Gateway: mode={mode!r}")
    except Exception as e:
        print(f"❌ Gateway 连不上: {e}")
        sys.exit(1)


def check_tp_active() -> bool:
    """
    Verify TP is running via three sources (any one confirming is sufficient):
      1. /get_server_args  — SGLang 0.4.x startup flags endpoint
      2. Launch log file   — grep for 'tensor_parallel_size=N' in stdout log
      3. /get_model_info   — SGLang 0.5.x runtime info (model path verification)

    GPU-memory symmetry is checked separately in check_gpu_memory().
    Returns True if TP is confirmed, False otherwise.
    """
    print("\n" + "=" * 60)
    print("🔍 TP 验证 — SGLang 服务器参数")
    print("=" * 60)

    tp_confirmed = False
    tp_size_found: int | None = None
    model_path = "?"

    # ── 1. /get_server_args (SGLang 0.4.x) ───────────────────────────────────
    try:
        r = requests.get(f"{SGLANG_URL}/get_server_args", timeout=10)
        if r.status_code == 200:
            args = r.json()
            tp_size_found = args.get("tensor_parallel_size", args.get("tp_size"))
            model_path    = args.get("model_path", args.get("model", "?"))
            if tp_size_found is not None:
                print(f"  source                : /get_server_args (SGLang 0.4.x)")
        # 404 on SGLang 0.5.x — silently fall through
    except Exception:
        pass

    # ── 2. Launch log file (SGLang 0.5.x compatible) ──────────────────────────
    # SGLang logs startup args at the beginning; we grep for tp_size there.
    if tp_size_found is None:
        log_candidates = [
            os.environ.get("SGLANG_LOG", ""),
            "/tmp/sglang_tp.log",
            "/tmp/sglang.log",
        ]
        for log_path in log_candidates:
            if not log_path or not os.path.isfile(log_path):
                continue
            try:
                with open(log_path) as f:
                    content = f.read(65536)   # first 64 KB covers all startup lines
                # matches: tp_size=4  OR  tensor_parallel_size=4  OR --tensor-parallel-size 4
                m = re.search(
                    r"\btp_size=(\d+)|tensor[_-]parallel[_-]size[=:\s]+(\d+)",
                    content, re.IGNORECASE
                )
                if m:
                    tp_size_found = int(m.group(1) or m.group(2))
                    print(f"  source                : log file ({log_path})")
                    # also try to grab model path from log
                    mp = re.search(r"model[_-]path[=:\s]+(\S+)", content, re.IGNORECASE)
                    if mp and model_path == "?":
                        model_path = mp.group(1).strip("'\"")
                    break
            except Exception:
                pass

    # ── 3. /get_model_info (SGLang 0.5.x) ────────────────────────────────────
    try:
        r = requests.get(f"{SGLANG_URL}/get_model_info", timeout=10)
        if r.status_code == 200:
            info = r.json()
            if model_path == "?":
                model_path = info.get("model_path", "?")
            tp_from_info = info.get("tp_size", info.get("tensor_parallel_size"))
            if tp_from_info is not None and tp_size_found is None:
                tp_size_found = tp_from_info
                print(f"  source                : /get_model_info")
    except Exception:
        pass

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"  endpoint              : {SGLANG_URL}")
    print(f"  model_path            : {model_path}")

    if tp_size_found is not None:
        tp_ok = isinstance(tp_size_found, int) and tp_size_found > 1
        mark  = "✅ TP 已激活" if tp_ok else "⚠️  单 GPU 模式（tp=1）"
        print(f"  tensor_parallel_size  : {tp_size_found}   {mark}")
        tp_confirmed = tp_ok
    else:
        print("  tensor_parallel_size  : 未找到（三个来源均无结果）")

    print()
    if tp_confirmed:
        print(f"  ✅ TP 运行确认：tensor_parallel_size={tp_size_found}，多 GPU 张量并行已激活")
    else:
        # GPU memory symmetry is still strong circumstantial evidence
        print("  ⚠️  API/log 未返回 tp_size，将以 GPU 显存对称性作为验证依据")
        print("     若所有 GPU 显存均 >50%（且模型超出单卡容量），TP 事实上已生效")

    return tp_confirmed, tp_size_found


def check_gpu_memory(label: str = "") -> None:
    """
    Print VRAM usage for all GPUs via nvidia-smi.
    In TP mode both GPUs should show comparable high usage
    (model weights split evenly). Prints a verdict line.
    """
    title = f"🖥️  GPU 显存占用{(' — ' + label) if label else ''}"
    print("\n" + "=" * 60)
    print(title)
    print("   TP 模式：两卡各持约 50% 权重，显存应均显著占用")
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
            # Consider a GPU "loaded" if it holds >20% VRAM (idle baseline is ~1-2%)
            gpu_ok  = pct > 20
            loaded.append(gpu_ok)
            mark    = "✅" if gpu_ok else "○ (空闲)"
            print(f"  GPU {idx} [{name}]")
            print(f"    {bar}  {used}/{total} MB  ({pct:.1f}%)  {mark}")
            rows.append((idx, name, used, total, pct))

        print()
        if len(rows) >= 2 and all(loaded):
            print("  → 所有 GPU 均有显著显存占用，权重已分片加载")
            print("  → TP 运行状态：✅ 两卡协同确认")
        elif len(rows) == 1 or (rows and not all(loaded)):
            print("  → ⚠️  仅 1 张 GPU 有显存占用，TP 可能未生效")
            print("       或 SGLang 尚未完成模型加载（等待后重试）")
        else:
            print("  → 未检测到 GPU 信息")

    except FileNotFoundError:
        print("  nvidia-smi 不在 PATH 中（在 GPU 机器上运行此脚本）")
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
    TTFT benchmark across short / medium / long prompts.
    TP splits attention heads and FFN columns across GPUs → each GPU does half
    the FLOPs per layer → prefill (compute-bound) is faster on long prompts.
    """
    print("\n" + "=" * 60)
    print("⏱️  TTFT 基准测试（首 token 时间）")
    print("   TP 对长 prompt prefill 加速最显著（矩阵乘各 GPU 算一半 FLOP）")
    print("=" * 60)

    results = {}
    for name, prompt in PROMPTS.items():
        times       = []
        token_count = len(prompt.split())
        print(f"\n[{name}] ~{token_count} 词 prompt", end="", flush=True)
        for _ in range(5):
            t = measure_ttft(prompt, name)
            if t > 0:
                times.append(t)
            print(".", end="", flush=True)
        if times:
            med           = statistics.median(times)
            results[name] = med
            print(f"  中位 TTFT: {med*1000:.1f}ms  (min={min(times)*1000:.1f}ms)")
        else:
            print("  失败")

    return results


def benchmark_throughput():
    print("\n" + "=" * 60)
    print("🚀 并发吞吐测试（20 并发，medium prompt）")
    print("   TP 主要解决「模型装得下」问题；decode 带宽收益有限")
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
    print(f"\n总耗时: {total:.1f}s  |  成功: {success}/20  |  失败: {fail}/20")
    if latencies:
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        print(
            f"延迟: P50={statistics.median(latencies)*1000:.0f}ms  "
            f"P95={p95*1000:.0f}ms"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

from _log_utils import print_hw_info, save_container_logs as _save_container_logs


def main():
    print("=" * 60)
    print("🔬 RadixGates — Tensor Parallel (TP) 模式演示")
    print("   SGLang TP instance: GPU 0 + GPU 1 (:30000)")
    print(f"   Gateway: {GATEWAY_URL}")
    print("=" * 60)

    hw_tag = print_hw_info()   # e.g. "4xRTX4090_PCIe" or "4xH100_NVLink"

    check_gateway()

    # Verify TP before running benchmarks — exits cleanly if clearly broken
    tp_ok, tp_size_found = check_tp_active()

    # GPU memory before benchmarks (baseline — model loaded, no active requests)
    check_gpu_memory("基准（模型已加载，无请求）")

    if not tp_ok:
        print("\n⚠️  TP 未确认，仍继续跑基准（结果仅供参考）")

    try:
        benchmark_ttft()
        benchmark_throughput()
    except KeyboardInterrupt:
        pass

    # GPU memory after benchmarks (should be stable — TP doesn't cause memory leaks)
    check_gpu_memory("测试后（应与基准相近）")

    print("\n" + "=" * 60)
    print("💡 结果解读：")
    if tp_ok:
        print(f"  tensor_parallel_size={tp_size_found} 已确认   → TP 生效 ✅")
    else:
        print("  API/log 未返回 tp_size，以显存分布作为依据：")
        print("  若4 卡各占 ~22GB（32B 模型 64GB / 4 = 16GB/卡权重）→ TP 事实上生效 ✅")
    print("  各卡显存均>50% → 权重已分片（模型超出单卡容量，必须开 TP）")
    print("  long prompt TTFT 改善 → prefill 矩阵乘法两卡各算一半 FLOP")
    print("  decode 带宽不翻倍   → All-Reduce 开销抵消部分收益，这是正常现象")
    print()
    print("  TP vs PD 定位：")
    print("    TP  — 模型大到单卡放不下时的必选项（解决显存问题）")
    print("    PD  — 模型放得下，但 prefill/decode 互相抢资源时的优化")
    print("    TP+PD — 工业级 70B+ 模型，两者叠加（如 DeepSeek-V3）")
    print("=" * 60)

    _ts = os.path.basename(_log_path).replace("tp_demo_", "").replace(".log", "") + f"_{hw_tag}"
    _save_container_logs(_LOG_DIR, _ts)
    print(f"\n[log saved → {_log_path}]")
    print(f"[machine   → {hw_tag}]")


if __name__ == "__main__":
    main()

"""
_log_utils.py — 容器日志收集 + 硬件信息打印（供 demo 脚本调用）

优先尝试本地 docker；本地不可用时通过 SSH_TARGET 环境变量 SSH 到远端收集。

SSH_TARGET 格式：user@host 或 user@host:port，例如：
  SSH_TARGET="root@203.0.113.10:22" python3 examples/pd_demo.py
"""

import os
import shutil
import subprocess

CONTAINERS = [
    "gateway",
    "sglang-dp-0", "sglang-dp-1", "sglang-dp-2", "sglang-dp-3",
    "sglang-ep",
    "sglang-tp",
    "sglang-router", "sglang-prefill", "sglang-decode",
]


def print_hw_info() -> str:
    """
    Print GPU hardware summary and return a short machine tag for log filenames.

    Detects:
      - GPU model and count (via nvidia-smi)
      - Interconnect type: NVLink vs PCIe (via nvidia-smi topo -m)

    Returned tag examples:
      "4xRTX4090_PCIe"   → 4× RTX 4090, PCIe interconnect
      "4xA100_NVLink"    → 4× A100, NVLink interconnect
      "4xH100_NVLink"    → 4× H100, NVLink interconnect

    All demo scripts print this block at startup so every log file records
    the exact hardware used — essential when comparing 4090 vs H100 benchmarks.
    """
    print("\n" + "=" * 60)
    print("🖥️  硬件信息")
    print("=" * 60)

    gpu_names  = []
    gpu_count  = 0
    link_type  = "Unknown"
    short_name = "UnknownGPU"

    # ── GPU names ─────────────────────────────────────────────────────────────
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        for line in out.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 3:
                continue
            idx, name, mem = parts[0], parts[1], parts[2]
            gpu_names.append(name)
            print(f"  GPU {idx}: {name}  ({int(mem)//1024} GB)")
        gpu_count = len(gpu_names)
    except Exception as e:
        print(f"  nvidia-smi 不可用: {e}")

    # ── Interconnect type ─────────────────────────────────────────────────────
    # nvidia-smi topo -m prints a matrix; "NV" prefix means NVLink,
    # "PHB" / "SYS" / "PIX" means PCIe topology.
    try:
        topo = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"], text=True, stderr=subprocess.DEVNULL
        )
        lines = [l for l in topo.splitlines() if l.strip() and not l.startswith("GPU")]
        nvlink_found = any("NV" in l for l in lines)
        link_type = "NVLink" if nvlink_found else "PCIe"
    except Exception:
        link_type = "PCIe (assumed)"

    print(f"\n  GPU 数量    : {gpu_count}")
    print(f"  互联方式    : {link_type}")

    if link_type.startswith("NVLink"):
        print("  → All-Reduce (TP) 和 All-to-All (EP) 带宽充足，加速比接近线性")
    else:
        print("  → PCIe 互联：TP All-Reduce 每层有额外延迟，加速比低于 NVLink")
        print("     EP All-to-All 影响相对小（只在 MoE Expert 层触发）")

    # ── Machine tag for log filenames ─────────────────────────────────────────
    if gpu_names:
        raw = gpu_names[0]
        if   "H100" in raw: short_name = "H100"
        elif "H800" in raw: short_name = "H800"
        elif "A100" in raw: short_name = "A100"
        elif "A10"  in raw: short_name = "A10"
        elif "4090" in raw: short_name = "RTX4090"
        elif "3090" in raw: short_name = "RTX3090"
        elif "L40"  in raw: short_name = "L40"
        else:               short_name = raw.replace(" ", "").replace("/", "")[:12]

    link_tag = "NVLink" if "NVLink" in link_type else "PCIe"
    tag      = f"{gpu_count}x{short_name}_{link_tag}"
    print(f"\n  机器标识    : {tag}  (写入 log 文件名)")
    print("=" * 60)
    return tag


def _docker_local_ok() -> bool:
    """检查本地 docker daemon 是否可用。"""
    if not shutil.which("docker"):
        return False
    r = subprocess.run(["docker", "info"], capture_output=True)
    return r.returncode == 0


def _save_via_local(log_dir: str, ts: str) -> list[str]:
    saved = []
    for name in CONTAINERS:
        check = subprocess.run(
            ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
            capture_output=True, text=True,
        )
        if name not in check.stdout:
            continue
        out_path = os.path.join(log_dir, f"{name}_{ts}.log")
        with open(out_path, "w") as f:
            subprocess.run(["docker", "logs", name], stdout=f, stderr=f)
        saved.append(f"  {name} → {out_path}")
    return saved


def _save_via_ssh(log_dir: str, ts: str, ssh_target: str) -> list[str]:
    """通过 SSH 在远端运行 docker logs，把输出拉到本地。"""
    parts = ssh_target.rsplit(":", 1)
    host = parts[0]
    port = parts[1] if len(parts) == 2 else "22"

    saved = []
    for name in CONTAINERS:
        check = subprocess.run(
            ["ssh", "-p", port, "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=5", host,
             f"docker ps --filter name=^{name}$ --format '{{{{.Names}}}}'"],
            capture_output=True, text=True,
        )
        if name not in check.stdout:
            continue
        out_path = os.path.join(log_dir, f"{name}_{ts}.log")
        with open(out_path, "w") as f:
            subprocess.run(
                ["ssh", "-p", port, "-o", "StrictHostKeyChecking=no", host,
                 f"docker logs {name}"],
                stdout=f, stderr=f,
            )
        saved.append(f"  {name} → {out_path}")
    return saved


def _load_env_file() -> None:
    """从项目根目录的 .env 文件加载 SSH_TARGET（若环境变量未设置）。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(root, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def save_container_logs(log_dir: str, ts: str) -> None:
    """
    把容器日志保存到 log_dir。
    本地 docker 可用时直接读；否则尝试 SSH_TARGET（环境变量或项目根 .env 文件）。
    """
    _load_env_file()

    if _docker_local_ok():
        saved = _save_via_local(log_dir, ts)
    else:
        ssh_target = os.environ.get("SSH_TARGET", "")
        if not ssh_target:
            print("\n[容器日志] 本地 docker 不可用，可通过以下任一方式配置：")
            print("  1. 永久：echo 'export SSH_TARGET=\"root@203.0.113.10:22\"' >> ~/.zshrc")
            print("  2. 项目：echo 'SSH_TARGET=root@203.0.113.10:22' >> .env")
            return
        saved = _save_via_ssh(log_dir, ts, ssh_target)

    if saved:
        print("\n[容器日志已保存]")
        print("\n".join(saved))
    else:
        print("\n[容器日志] 未找到运行中的 PD 容器，跳过。")

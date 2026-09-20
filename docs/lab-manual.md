# RadixGates 实验手册（可带学生逐步运行）

> 本手册把这个仓库**能做的所有实验**从零到跑通、逐步列出。
> 每个实验都写清楚：**运行什么命令 · 需要什么机器 · 会得到什么结果 · 对应生产里的什么应用**。
>
> 架构一句话：`Client → RadixGates Go 网关（前缀路由 + 信号量 + SSE）→ SGLang（多卡 GPU）`，全程 HTTP/JSON，直连、无消息队列、无 Redis。

---

## 目录

- [0. 实验总览（一张表）](#0-实验总览一张表)
- [1. 环境准备（所有实验通用）](#1-环境准备所有实验通用)
- [实验一：单卡流式 SSE（1 GPU，入门）](#实验一单卡流式-sse1-gpu入门)
- [实验二：PD 分离与前缀缓存（2 GPU，核心卖点 10.5×）](#实验二pd-分离与前缀缓存2-gpu核心卖点-105)
- [实验三：DP=4 数据并行与前缀路由（4 GPU）](#实验三dp4-数据并行与前缀路由4-gpu)
- [实验四：企业问答并发压测（复用实验三集群）](#实验四企业问答并发压测复用实验三集群)
- [实验五：TP=4 张量并行（4 GPU，跑大模型）](#实验五tp4-张量并行4-gpu跑大模型)
- [实验六：EP=4 专家并行（4 GPU，MoE）](#实验六ep4-专家并行4-gpumoe)
- [实验七：PD 双机横向扩展（2 机 × 2 GPU）](#实验七pd-双机横向扩展2-机--2-gpu)
- [复现论文基准表的确切数字](#复现论文基准表的确切数字)
- [机器采购 / 租用建议](#机器采购--租用建议)
- [常见问题排查](#常见问题排查)

---

## 0. 实验总览（一张表）

| # | 实验 | 最少 GPU | 启动文件 | 验证脚本 | 会看到什么 | 对应生产应用 |
|---|------|---------|---------|---------|-----------|-------------|
| 一 | 流式 SSE | **1** | 裸机手动 | `streaming_demo.py` | token 逐字流回（打字机效果） | 实时对话 / 代码助手的流式输出 |
| 二 | PD 分离 | **2** | `docker-compose.pd.yml` | `pd_demo.py` | 30/30 成功；前缀缓存 TTFT **861→82ms（10.5×）** | 金融实时风控：低延迟、prefill/decode 不抢资源 |
| 三 | DP=4 | **4** | `docker-compose.dp.yml` | `dp_demo.py` | 40/40；4× 线性 QPS；同上下文命中同一副本 | 企业内网 AI 助手：数百员工并发、按部门路由 |
| 四 | 企业问答压测 | **4**（复用三） | 同上 | `enterprise_qa_demo.py` | 30 员工 5 部门并发，信号量限流不打爆 | 企业问答生产稳定性验证 |
| 五 | TP 张量并行 | **2**（默认）/ **4**（基准） | `docker-compose.tp.yml` | `tp_demo.py` | 单实例多卡对称显存；跑得下 32B 大模型 | 金融合规大模型（32B+ 单卡放不下） |
| 六 | EP 专家并行 | **2**（默认）/ **4**（基准） | `docker-compose.ep.yml` | `ep_demo.py` | 日志 `[TP0 EP0]~[TP3 EP3]`；All-to-All | 医疗/法律 MoE 模型（DeepSeek-V3 类） |
| 七 | PD 双机 | **2 机 × 2** | `pd.yml` + `pd-node.yml` | `pd_demo.py` | `prefix_hash % 2` 分流两机，KV 局部性 | 从单节点到多节点的弹性伸缩 |

> **平台对照**：以上先在 **4× RTX 4090（PCIe）** 上跑一遍，再在 **4× A100（SXM NVLink）** 上重跑同样的实验，即可看到 NVLink 带来的 **1.3×–1.6×** 加速（通信越重的模式收益越大）。

---

## 1. 环境准备（所有实验通用）

### 1.1 硬件
- NVIDIA GPU（RTX 4090 / A100 均可），数量见每个实验要求。
- GPU 显存参考：8B 模型 fp16 约 16GB（单卡可放）；32B 约 64GB（需 TP 切 4 卡）。

### 1.2 软件（Docker 路径，推荐）
```bash
# 1) NVIDIA 驱动 + Docker + NVIDIA Container Toolkit（让容器能用 GPU）
nvidia-smi                      # 确认驱动 OK
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi   # 确认容器能看到 GPU

# 2) 构建本仓库的网关镜像（用当前代码，保证和仓库一致）
cd RadixGates            # 仓库根目录
docker build -t unicoregpu2020/radixgates:latest .

# 3) Python 依赖（跑 demo 脚本用）
pip install requests
```

> Vast.ai 等禁用 Docker 桥接网络的容器，改用**裸机 no-Docker 路径**：用 `tmux` 直接起 SGLang，`cd gateway-go && go build -o /usr/local/bin/sglang_gateway .` 编译网关。排障手册见 `docs/troubleshooting/`。

### 1.3 模型下载
- SGLang 会自动从 HuggingFace 拉模型；**Llama 系列是 gated**，需要先 `huggingface-cli login` 或设 `HF_TOKEN`。
- 想省事就用非 gated 的 Qwen 系列（compose 默认就是 Qwen）。

### 1.4 通用验证命令（每次起完集群都先跑）
```bash
curl http://localhost:8080/health        # 期望 {"status":"ok","mode":"direct"}
curl http://localhost:30000/health       # SGLang 节点健康
curl -s http://localhost:8080/metrics | grep radixgates   # Prometheus 指标
```

> **端口约定**：网关 `:8080`（demo 脚本默认就连它）；SGLang `:30000` 起；PD 的 sglang_router `:9000`。
> 全部 demo 都读环境变量 `GATEWAY_URL`（默认 `http://localhost:8080`），一般无需设置。

---

## 实验一：单卡流式 SSE（1 GPU，入门）

**目的**：用最小配置看懂“一个请求怎么被直连转发、token 怎么逐字流回”。
**机器**：1 张 GPU（任意，8B 模型即可）。

### 步骤（裸机 no-Docker，最小依赖）
```bash
# 1) 起一个 SGLang 实例（GPU 0）
python3 -m sglang.launch_server \
  --model-path NousResearch/Meta-Llama-3-8B-Instruct \
  --port 30000 --host 0.0.0.0 --mem-fraction-static 0.85 &

# 2) 编译并启动网关，指向这个实例
cd gateway-go
cat > config/config.json <<'EOF'
{ "port": 8080, "default_model": "NousResearch/Meta-Llama-3-8B-Instruct",
  "max_concurrent_per_node": 8,
  "sglang_instances": [ {"group":"local","host":"127.0.0.1","port":30000,"max_concurrent":8} ] }
EOF
go build -o /tmp/sglang_gateway . && /tmp/sglang_gateway &

# 3) 跑流式 demo
cd .. && python3 examples/streaming_demo.py
```

### 会得到什么
- 终端里模型回答**逐字（逐 token）实时蹦出来**，而不是等全部生成完一次性返回。
- 证明 `Gateway → SGLang → SSE` 直连链路通了，网关不缓存、不落盘。

### 对应生产应用
- **实时对话 / 代码助手的流式输出**：用户敲下问题后立刻看到“打字机”式回复，体验关键。

---

## 实验二：PD 分离与前缀缓存（2 GPU，核心卖点 10.5×）

**目的**：验证 Prefill/Decode 分卡 + RadixAttention 前缀缓存带来的巨大 TTFT 下降。
**机器**：**2 张 GPU**（GPU0 做 prefill，GPU1 做 decode）。基准用 2× RTX 4090。

### 步骤
```bash
# 1) 起 PD 集群（prefill:30000 + bootstrap:8998，decode:30001，router:9000，gateway:8080）
docker compose -f docker-compose.pd.yml up -d

# 2) 等健康检查通过
curl http://localhost:8080/health
curl http://localhost:9000/health        # sglang_router（PD 协调器）

# 3) 跑 PD 基准 demo
python3 examples/pd_demo.py
```

### 会得到什么（基准：2× RTX 4090）
- **30 并发全部成功（30/30，0 失败）**，总耗时约 **5.2s**，P50 ≈ 2.8s，P95 ≈ 5.2s。
- **前缀缓存对比**（7720-token 上下文）：
  - 冷启动（无缓存）TTFT ≈ **861ms**
  - 命中 RadixCache 后 TTFT ≈ **82ms** → **10.5× 加速** 🚀
- 会看到 KV cache 经 **mooncake（CUDA IPC）** 从 prefill 卡传到 decode 卡。

### 对应生产应用
- **金融实时风控 / 低延迟推理**：把计算密集的 prefill 和带宽密集的 decode 拆到不同 GPU，消除资源争抢；重复上下文（同一份合规文档）几乎“秒回”。

---

## 实验三：DP=4 数据并行与前缀路由（4 GPU）

**目的**：看懂“4 个独立副本 + 网关按前缀哈希一致性路由”，以及 QPS 线性扩展。
**机器**：**4 张 GPU**（每卡一个完整模型副本）。

> ⚠️ compose 默认模型是 `Qwen/Qwen2.5-14B`（约 28GB，**放不进单张 24GB 的 4090**）。
> 4090 上请把每副本换成 8B 模型（见下）。

### 步骤
```bash
# 4090（24GB）：换成 8B 模型，每卡一个副本
MODEL_PATH=NousResearch/Meta-Llama-3-8B-Instruct \
  docker compose -f docker-compose.dp.yml up -d

# 验证 4 个副本 + 网关
for p in 30000 30001 30002 30003; do curl -s localhost:$p/health; done
curl http://localhost:8080/health

# 跑 DP demo（40 并发）
python3 examples/dp_demo.py
```

### 会得到什么
- **40 并发全部成功（40/40）**，吞吐接近 **4.0× 线性扩展**（4 卡各跑各的，无跨卡通信）。
- demo 会演示：**相同 system prompt 的请求总是落到同一副本**（`prefix_hash % 4`），保持该副本 KV Cache 热度。

### 对应生产应用
- **企业内网 AI 助手**：数百名员工并发访问，用 DP 多副本线性扩 QPS；按部门 system prompt 路由，让同部门请求命中同一副本缓存，TTFT 大幅下降。

---

## 实验四：企业问答并发压测（复用实验三集群）

**目的**：在真实“多部门并发”场景下验证前缀路由 + 信号量背压（生产级稳定性）。
**机器**：复用实验三的 4 GPU 集群（DP 已经起好即可）。

### 步骤
```bash
# 集群沿用实验三（docker compose -f docker-compose.dp.yml 已 up）
python3 examples/enterprise_qa_demo.py
```

### 会得到什么
- **30 名员工、5 个部门、每部门 6 人**同时提问。
- 每个部门有独立 system prompt → 不同 `prefix_hash` → 一致性哈希把同部门请求路由到同一节点。
- Go 信号量（`max_concurrent=8`）**限流排队**，即使并发涌入也**不打爆 SGLang、零请求丢失**。
- 全部同步 HTTP，无 task_id 轮询。

### 对应生产应用
- **企业问答系统的生产验收**：多部门混合负载下，证明“数据不出内网 + GPU 不被打爆 + 请求零丢失”三条约束同时成立。

---

## 实验五：TP=4 张量并行（4 GPU，跑大模型）

**目的**：把单卡放不下的大模型权重切到多卡，理解每层 All-Reduce，以及“TP 对网关透明”。
**机器**：**4 张 GPU**（一个 SGLang 实例横跨 4 卡）。

### 步骤
```bash
# 默认 TP_SIZE=2、模型 Qwen2.5-14B（2 GPU 就能跑）：
docker compose -f docker-compose.tp.yml up -d

# 复现基准：TP=4 跑 32B 大模型（4 卡）
CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 \
  MODEL_PATH=Qwen/Qwen2.5-32B-Instruct \
  docker compose -f docker-compose.tp.yml up -d

# 验证权重确实切了 4 份（各卡显存对称）
nvidia-smi
curl http://localhost:30000/get_model_info | python3 -m json.tool

# 跑 TP demo
python3 examples/tp_demo.py
```

### 会得到什么
- 单个 SGLang 实例，**每张卡持有约 25% 权重**，`nvidia-smi` 看到 4 卡显存占用对称。
- 每个 Transformer 层后做 **All-Reduce** 同步激活值。
- 从网关视角，TP 完全透明——它看到的仍是一个 SGLang 端点。
- 基准（4× 4090）：Qwen2.5-32B，20/20，P50 ≈ **3851ms**。

### 对应生产应用
- **金融合规大模型**：32B+ 参数单卡放不下，用 TP 张量并行把权重切到多张 GPU 才能上线。

---

## 实验六：EP=4 专家并行（4 GPU，MoE）

**目的**：理解 MoE 的 Expert 分卡 + token 路由的 All-to-All 通信。
**机器**：**4 张 GPU**（Expert 分布在各卡）。

### 步骤
```bash
# 默认模型 Qwen/Qwen3-30B-A3B，TP_SIZE=2
docker compose -f docker-compose.ep.yml up -d

# 复现基准：EP=4 + MoE 模型（4 卡）
CUDA_VISIBLE_DEVICES=0,1,2,3 TP_SIZE=4 \
  MODEL_PATH=Qwen/Qwen1.5-MoE-A2.7B-Chat \
  docker compose -f docker-compose.ep.yml up -d

# 看日志确认 4 个 EP rank
docker logs sglang-ep 2>&1 | grep -E "\[TP[0-9] EP[0-9]\]"

# 跑 EP demo
python3 examples/ep_demo.py
```

### 会得到什么
- 日志出现 `[TP0 EP0]~[TP3 EP3]`，确认 **4 个 EP rank** 在跑。
- 每个 token 路由到 top-K Expert，跨卡时通过 **All-to-All** 搬运激活值；Attention 层走 All-Reduce（TP 部分）。
- 基准（4× 4090）：20/20，P50 ≈ **1099ms**。

### 对应生产应用
- **医疗 / 法律行业私有部署**：MoE 专家模型（如 DeepSeek-V3 类）在垂直领域表现优异，用 EP 专家并行高效部署。

---

## 实验七：PD 双机横向扩展（2 机 × 2 GPU）

**目的**：从单机 PD 扩到双机，理解 `prefix_hash % 2` 分流 + KV 局部性 + 跨机扩容。
**机器**：**2 台机器，每台 2 张 GPU**（各跑一套完整 PD 对）。

### 步骤
```bash
# 机器 2（Worker 节点）：只起 prefill + decode + router
docker compose -f docker-compose.pd-node.yml up -d

# 机器 1（含网关）：指向机器 2 的 router，网关按 prefix_hash % 2 分流
M2_HOST=<机器2的IP> docker compose -f docker-compose.pd.yml up -d

# 验证两机的 router
curl http://localhost:9000/health           # 机器 1 本地 PD 对
curl http://<机器2的IP>:9000/health          # 机器 2 PD 对

# 跑 demo（30 并发，观察被分到两台机器）
python3 examples/pd_demo.py
```

### 会得到什么
- 网关按 `prefix_hash % 2` 把流量分到两台机器：**同一 system prompt 总落到同一台**（KV Cache 局部性），两机分担总并发。
- 加一台机器**无需改网关代码**——只是节点列表多一项。

### 对应生产应用
- **弹性伸缩部署**：从单节点扩到多节点，Docker 容器化 + 多机 PD，支撑业务量增长。

---

## 复现论文基准表的确切数字

README 那张表是这样对齐的（**4× RTX 4090 PCIe** vs **4× A100 SXM NVLink**，SGLang 0.5.10）：

| 模式 | 模型 | 成功 | P50 (4090) | P50 (A100) | NVLink 加速 |
|------|------|------|-----------|-----------|------------|
| DP=4 | Llama-3-8B | 40/40 | 1174ms | **764ms** | **1.5×** |
| TP=4 | Qwen2.5-32B | 20/20 | 3851ms | **2422ms** | **1.6×** |
| EP=4 | Qwen1.5-MoE-A2.7B | 20/20 | 1099ms | **827ms** | **1.3×** |

- 用上面各实验里“复现基准”那条命令（指定 `MODEL_PATH` / `TP_SIZE` / `CUDA_VISIBLE_DEVICES`）。
- **先在 4× 4090 上跑一遍拿到 PCIe 列，再在 4× A100 NVLink 上跑同样命令拿到 A100 列**，两列一对比就是 NVLink 加速。
- PD 单机基准（2× 4090）：30/30、5.2s、前缀缓存 10.5×（见实验二）。
- 完整基准数据见 `docs/benchmark_parallel_modes.md`。

---

## 机器采购 / 租用建议

| 你有的机器 | 能跑哪些实验 |
|-----------|-------------|
| **1× GPU** | 实验一（流式 SSE） |
| **2× RTX 4090** | 实验二（PD 单机，含 10.5× 前缀缓存）；实验一；以及 TP=2 / EP=2 的默认（缩水）版 |
| **4× RTX 4090 PCIe** | 实验一~六（DP=4 / TP=4 / EP=4 + 企业问答），拿到基准表的 **4090 列** |
| **4× A100 SXM NVLink** | 同上，拿到基准表的 **A100 列**，验证 NVLink 加速 |
| **2 机 × 2 GPU** | 实验七（PD 双机横向扩展） |

> **建议节奏**：先租 **2× 4090** 把实验二跑通（成本低、能坐实核心卖点 10.5×）；再上 **4× 4090** 跑 DP/TP/EP；最后 **4× A100** 复现 NVLink 加速列。

---

## 常见问题排查

| 现象 | 原因 / 处理 |
|------|------------|
| 容器看不到 GPU | 没装 NVIDIA Container Toolkit；`docker run --gpus all ... nvidia-smi` 自查 |
| 模型下载 401/403 | Llama 是 gated 模型，需 `huggingface-cli login` 或设 `HF_TOKEN`；或改用 Qwen |
| SGLang OOM / 起不来 | 模型太大装不下：调低 `--mem-fraction-static`（如 0.80），或换更小模型 / 上 TP |
| 端口冲突 | 30000/8080/9000 被占：`docker compose down` 清理，或改端口映射 |
| `/health` 连不上 | SGLang 还在加载模型（大模型要几分钟）；等健康检查通过再跑 demo |
| 更多失败模式 | 见 `docs/troubleshooting/`（nvidia-docker、端口冲突、KV 传输后端等专项修复） |

---

*本手册对应清理后的 direct 架构版本：网关只有 `gateway-go/` 一个服务，全链路 HTTP + SSE，无消息队列 / Redis / gRPC。*

# RadixGates — High-Performance Private LLM Inference Gateway

**Prefix-Aware Routing · PD Disaggregation · Data / Tensor / Expert Parallelism · Direct SSE Streaming · Semaphore Backpressure · Prometheus Metrics**

---

## What Is This?

RadixGates is an **enterprise-grade private LLM deployment platform** built to solve a core problem in large-scale internal AI deployments:

> How do you serve hundreds of concurrent employees through a private LLM without exposing data to public APIs, dropping requests, or wasting GPU compute?

It sits in front of SGLang inference engines and provides:

1. **Multi-GPU Parallelism** — DP / TP / EP each targeting a different bottleneck; mix and stack as model size and QPS demand grows
2. **PD Disaggregation** — Prefill (compute-intensive) and Decode (memory-bandwidth-intensive) run on separate GPUs, removing resource contention
3. **Prefix-Aware Routing** — SHA-256 fingerprint routes same-context requests to the same node; SGLang's RadixAttention keeps the KV Cache hot
4. **Semaphore Backpressure** — Go semaphore limits concurrent load per node; excess requests wait in goroutines with zero message loss
5. **Direct SSE Streaming** — Token-by-token streaming from Gateway to client, no intermediate broker

**Typical use case**: 中大型企业内部私有部署 — 员工通过企业内网访问 LLM，网关保证数据不出内网、GPU 不被打爆、请求零丢失。

---

## Benchmark Results

### Multi-GPU Parallel Modes — Two Hardware Platforms Verified

**4× RTX 4090 PCIe** (SGLang 0.5.10, 2026-04-26) vs **4× A100 SXM NVLink** (SGLang 0.5.10.post1, 2026-04-27)

| Mode | Model | Success | P50 4090 PCIe | P50 A100 NVLink | NVLink Speedup |
|------|-------|---------|--------------|----------------|---------------|
| **DP=4** | Llama-3-8B | **40/40 ✅** | 1174ms | **764ms** | **1.5×** |
| **TP=4** | Qwen2.5-32B | **20/20 ✅** | 3851ms | **2422ms** | **1.6×** |
| **EP=4** | Qwen1.5-MoE-A2.7B | **20/20 ✅** | 1099ms | **827ms** | **1.3×** |

Key confirmations: `tensor_parallel_size=4`, `[TP0 EP0]~[TP3 EP3]` EP ranks, prefix-hash KV cache routing.

Full results: [`docs/benchmark_parallel_modes.md`](docs/benchmark_parallel_modes.md)

### PD Disaggregation (2× RTX 4090, verified 2026-04-25)

| Metric | Result |
|--------|--------|
| 30-concurrent stress test | **30/30, 0 failures** |
| Total time for 30 concurrent requests | **5.2 s** |
| P50 latency | **2.8 s** |
| P95 latency | **5.2 s** |
| RadixCache hit TTFT (7720-token context) | **82 ms** |
| Cold TTFT (no cache) | **861 ms** |
| Prefix-cache speedup | **10.5× 🚀** |

---

## Parallelism Strategy

```
模型放得进单卡?
  ├─ YES → DP=N  (多副本, 线性扩 QPS, 零跨卡通信)
  └─ NO  → TP=N  (权重切分, All-Reduce 每层)
               └─ MoE 模型? → +EP=N (Expert 切分, All-to-All)
                                └─ 超长流水线? → +PP (层间切分)

工业实践: DP + TP + EP 叠加 (DeepSeek-V3 / Qwen3-235B 等)
```

| Mode | Splits | Communication | When to Use |
|------|--------|--------------|-------------|
| **DP** | Requests across replicas | None | Model fits in one GPU; scale QPS |
| **TP** | Attention heads + FFN columns | All-Reduce (every layer) | Model too large for one GPU |
| **EP** | MoE Experts across GPUs | All-to-All (Expert layers only) | MoE model + high concurrency |
| **PP** | Layers across GPUs | Point-to-point | Pipeline across many nodes |

---

## Architecture

### Mode 1 — DP=4 (Data Parallel, 4 replicas)

Four independent SGLang instances, one per GPU. The Gateway routes by prefix-hash — same department/context always hits the same replica, keeping that replica's KV Cache hot.

```
Client → Go Gateway :8081
             │ prefix_hash(system_prompt) % 4
             ├─→ SGLang :30000  GPU 0  (replica 0 — e.g. Finance dept)
             ├─→ SGLang :30001  GPU 1  (replica 1 — e.g. HR dept)
             ├─→ SGLang :30002  GPU 2  (replica 2 — e.g. Engineering)
             └─→ SGLang :30003  GPU 3  (replica 3 — e.g. Marketing)

No cross-GPU communication. Linear QPS scaling (4.0× verified).
docker compose -f docker-compose.dp.yml up -d
```

---

### Mode 2 — TP=4 (Tensor Parallel, one model across 4 GPUs)

Single SGLang instance, model weights split across all GPUs. Each GPU holds ~25% of attention heads and FFN columns. All-Reduce synchronizes activations after every transformer layer.

```
Client → Go Gateway :8081 → SGLang :30000
                               │
                    ┌──────────┼──────────┐──────────┐
                    ▼          ▼          ▼          ▼
                  GPU 0      GPU 1      GPU 2      GPU 3
               (25% weights)(25% weights)(25% weights)(25% weights)
                    └──── All-Reduce after every layer ────┘

Required when model > single-GPU VRAM (e.g. Qwen2.5-32B: 64 GB fp16 on 4× 24 GB)
docker compose -f docker-compose.tp.yml up -d
```

---

### Mode 3 — EP=4 (Expert Parallel, MoE model)

MoE model with Experts distributed across GPUs. Each token is routed to top-K experts; when those experts live on other GPUs, All-to-All carries the activations across cards. Combines with TP for Attention layers.

```
Client → Go Gateway :8081 → SGLang :30000
                               │  TP=4 (Attention: All-Reduce)
                    ┌──────────┼──────────┐──────────┐
                    ▼          ▼          ▼          ▼
                  GPU 0      GPU 1      GPU 2      GPU 3
               Expert 0-15 Expert 16-31 Expert 32-47 Expert 48-63
                    └──── All-to-All (Expert FFN layers) ───┘

Log prefix [TP0 EP0]~[TP3 EP3] confirms 4 EP ranks running.
docker compose -f docker-compose.ep.yml up -d
```

---

### Mode 4 — PD Disaggregation, Single Machine

One machine, two GPUs. GPU 0 handles all prefill (compute-bound); GPU 1 handles all decode (memory-bandwidth-bound). KV cache is transferred via CUDA IPC (mooncake).

```
┌─────────────────────────────────────────────────────────────────┐
│  Client                                                         │
│    │  POST /v1/chat  (stream=true)                              │
│    ▼                                                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Go Gateway  :8080                                         │ │
│  │  • SHA-256 prefix_hash → consistent-hash node select      │ │
│  │  • Semaphore: max 8 concurrent per node                    │ │
│  │  • SSE proxy: streams tokens back to client                │ │
│  └──────────────────────────┬─────────────────────────────────┘ │
│                             │  HTTP POST /v1/chat               │
│                             ▼                                   │
│              ┌──────────────────────────┐                       │
│              │   sglang_router  :9000   │                       │
│              │   PD coordinator         │                       │
│              └────────┬────────┬────────┘                       │
│                       │        │                                │
│                 prefill│        │decode                          │
│                       ▼        ▼                                │
│         ┌─────────────────┐  ┌─────────────────┐               │
│         │ sglang-prefill  │  │ sglang-decode   │               │
│         │ :30000  GPU 0   │  │ :30001  GPU 1   │               │
│         │ Compute-bound   │  │ Memory-bw-bound │               │
│         └────────┬────────┘  └─────────────────┘               │
│                  │  KV cache transfer (mooncake · CUDA IPC)     │
│                  └──────────────────────────────►               │
└─────────────────────────────────────────────────────────────────┘

docker compose -f docker-compose.pd.yml up -d
```

---

### Mode 5 — PD Disaggregation, Dual Machine

Two machines, each running a full PD pair. Gateway prefix-hash partitions traffic — same system prompt always hits the same machine (KV Cache locality), while the two machines share total concurrency load.

```
                         Client
                           │  POST /v1/chat
                           ▼
           ┌───────────────────────────────────┐
           │  Go Gateway  :8080  (Machine 1)   │
           │  prefix_hash(system_prompt) % 2   │
           └───────────┬───────────────┬───────┘
                       │               │
              hash=0   │               │  hash=1
                       ▼               ▼
    ┌──────────────────────┐   ┌──────────────────────┐
    │  sglang-router M1    │   │  sglang-router M2    │
    └──────┬───────┬───────┘   └──────┬───────┬───────┘
           │       │                  │       │
     prefill│       │decode      prefill│       │decode
           ▼       ▼                  ▼       ▼
    ┌──────────┐ ┌──────────┐  ┌──────────┐ ┌──────────┐
    │ prefill  │ │ decode   │  │ prefill  │ │ decode   │
    │ GPU 0    │ │ GPU 1    │  │ GPU 0    │ │ GPU 1    │
    └──────────┘ └──────────┘  └──────────┘ └──────────┘
         KV transfer (mooncake)      KV transfer (mooncake)

docker compose -f docker-compose.pd-node.yml up -d   # Machine 2
M2_HOST=<m2-ip> docker compose -f docker-compose.pd.yml up -d   # Machine 1
```

---

## Demo Scripts

| Script | What It Tests | Command |
|--------|--------------|---------|
| `examples/dp_demo.py` | DP=4 prefix-hash routing, KV cache hits, 40-concurrent throughput | `GATEWAY_URL=http://localhost:8081 python3 examples/dp_demo.py` |
| `examples/tp_demo.py` | TP=4 weight sharding, symmetric GPU VRAM, TTFT + throughput | `GATEWAY_URL=http://localhost:8081 SGLANG_URL=http://localhost:30000 SGLANG_LOG=/tmp/sglang_tp.log python3 examples/tp_demo.py` |
| `examples/ep_demo.py` | EP=4 MoE Expert routing, All-to-All log rank confirmation | `GATEWAY_URL=http://localhost:8081 SGLANG_URL=http://localhost:30000 SGLANG_LOG=/tmp/sglang_ep.log python3 examples/ep_demo.py` |
| `examples/pd_demo.py` | PD KV transfer (`#transfer-req`), prefix-cache TTFT, 30-concurrent | `python3 examples/pd_demo.py` |
| `examples/enterprise_qa_demo.py` | 30-user multi-dept concurrent Q&A, prefix-hash partition LB | `GATEWAY_URL=http://localhost:8081 python3 examples/enterprise_qa_demo.py` |
| `examples/streaming_demo.py` | SSE token-by-token streaming, Gateway direct mode | `GATEWAY_URL=http://localhost:8081 python3 examples/streaming_demo.py` |

All scripts accept `GATEWAY_URL` via environment variable. Logs are auto-saved to `examples/logs/` with hardware tag in filename (e.g. `dp_demo_4xRTX4090_Llama3-8B_DP4.log`).

---

## Key Features

### Prefix-Aware Consistent Hash Routing
The Gateway extracts a SHA-256 fingerprint (`prefix_hash`) from each request's system prompt:
- Same department / same RAG context → same `prefix_hash` → same node → same SGLang instance
- SGLang's **RadixAttention** keeps that context in GPU SRAM across requests
- **Result**: TTFT drops from 861 ms → 82 ms (10.5×) on 7720-token context

### Semaphore-Based Backpressure
```go
sem := make(chan struct{}, maxConcurrent) // e.g. 8 slots per node
sem <- struct{}{}                         // blocks if all slots taken
defer func() { <-sem }()                 // always releases
```
Unlike a queue, goroutines waiting on the semaphore consume almost no memory and resume the instant a slot opens.

### Direct SSE Streaming
- Gateway acquires a semaphore slot → forwards request as HTTP streaming call
- Proxies Server-Sent Events token-by-token to the client
- No task_id, no polling, no intermediate broker — pure synchronous SSE, held open for the request

### Hardware-Tagged Log Files
`_log_utils.print_hw_info()` auto-detects GPU model, count, and interconnect type (NVLink vs PCIe) and appends a hardware tag to every log filename. Makes multi-machine benchmark comparison unambiguous.

### Prometheus Metrics
Built-in `/metrics` endpoint:

| Metric | Type | Meaning |
|--------|------|---------|
| `radixgates_requests_total{status}` | Counter | Requests by outcome |
| `radixgates_request_duration_seconds` | Histogram | Gateway-side P50/P95/P99 latency |
| `radixgates_active_requests` | Gauge | In-flight requests |

---

## Deployment

### Docker Path (standard VMs)

```bash
# DP=4
docker compose -f docker-compose.dp.yml up -d

# TP=4 (set MODEL_PATH and TP_SIZE)
TP_SIZE=4 MODEL_PATH=Qwen/Qwen2.5-32B-Instruct docker compose -f docker-compose.tp.yml up -d

# EP=4
EP_SIZE=4 MODEL_PATH=Qwen/Qwen1.5-MoE-A2.7B-Chat docker compose -f docker-compose.ep.yml up -d

# PD disaggregation
docker compose -f docker-compose.pd.yml up -d
```

### No-Docker Path (Vast.ai containers, overlayfs restricted)

Some cloud instances (e.g. Vast.ai containerized templates) block Docker's bridge networking and overlayfs. Run SGLang and Gateway natively with tmux:

```bash
# Install
pip install "sglang[all]" -q
wget -q https://go.dev/dl/go1.23.4.linux-amd64.tar.gz -O /tmp/go.tar.gz
tar -C /usr/local -xzf /tmp/go.tar.gz && export PATH=$PATH:/usr/local/go/bin

# Build Gateway
cd ~/RadixGates/gateway-go && go build -o /usr/local/bin/sglang_gateway .

# Launch DP=4 (one tmux session per GPU)
for GPU in 0 1 2 3; do
  tmux new-session -d -s sglang$GPU \
    "CUDA_VISIBLE_DEVICES=$GPU python3 -m sglang.launch_server \
       --model-path NousResearch/Meta-Llama-3-8B-Instruct \
       --port $((30000+GPU)) --host 0.0.0.0 --mem-fraction-static 0.85 \
       2>&1 | tee /tmp/sglang${GPU}.log"
done

# Launch TP=4 (single process, all GPUs)
tmux new-session -d -s sglang_tp \
  "CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
     --model-path Qwen/Qwen2.5-32B-Instruct \
     --tensor-parallel-size 4 --port 30000 --mem-fraction-static 0.80 \
     2>&1 | tee /tmp/sglang_tp.log"
```

Troubleshooting playbooks (port conflicts, EP setup, SSH tunnel, NVIDIA container runtime, API flag changes): [`docs/troubleshooting/`](troubleshooting/)

---

## Directory Structure

```text
RadixGates/
├── gateway-go/                    # Go Gateway — the only service (direct mode)
│   ├── main.go                    # Entry: load config → register route → HTTP server → graceful shutdown
│   ├── handler/
│   │   ├── direct.go              # Core: POST /v1/chat — parse → route → semaphore → forward → SSE proxy
│   │   ├── chat.go                # Shared types (ChatRequest/Message) + prefixHash + writeJSON helper
│   │   └── chat_test.go           # Unit test for computePrefixHash
│   ├── router/
│   │   └── router.go              # Node pool + consistent-hash node selection + per-node semaphore
│   ├── metrics/
│   │   └── metrics.go             # Prometheus metric definitions (requests_total / duration / active)
│   └── config/
│       ├── config.go              # Config struct + defaults + Load()
│       └── config.json            # Gateway runtime config (port, sglang_instances, max_concurrent)
├── examples/
│   ├── dp_demo.py                 # DP=4: prefix-hash routing + 40-concurrent benchmark
│   ├── tp_demo.py                 # TP=4: weight sharding + TTFT/throughput benchmark
│   ├── ep_demo.py                 # EP=4: MoE Expert routing + All-to-All verification
│   ├── pd_demo.py                 # PD: KV transfer + prefix-cache TTFT benchmark
│   ├── enterprise_qa_demo.py      # 30-user multi-dept concurrent Q&A
│   ├── streaming_demo.py          # SSE token-by-token streaming demo
│   ├── _log_utils.py              # Hardware detection + log collection utility
│   └── logs/                      # Auto-saved benchmark logs (hardware-tagged filenames)
├── scripts/                       # SGLang launch (incl. PD prefill/decode), benchmark, wrk load tests
├── mock_docs/                     # Sample documents used by the enterprise-QA demo
├── .env.example                   # Template for local environment variables (SSH_TARGET for log collection)
├── docs/
│   ├── benchmark_parallel_modes.md  # DP/TP/EP verified benchmark results
│   ├── troubleshooting/           # Failure-pattern playbooks (nvidia-docker, port conflicts, API flag changes, ...)
│   └── lab-notes/                 # Experiment logs (4x RTX 4090, 4x A100) and GPU sizing notes
├── docker-compose.dp.yml          # DP=4: 4 SGLang replicas, one per GPU  (gateway routing_mode=direct)
├── docker-compose.tp.yml          # TP=4: single SGLang, weights split across GPUs
├── docker-compose.ep.yml          # EP=4: MoE model, Expert routing across GPUs
├── docker-compose.pd.yml          # PD disaggregation — Machine 1 (gateway + PD pair)
├── docker-compose.pd-node.yml     # PD disaggregation — Machine 2+ (PD node)
└── Dockerfile                     # Multi-stage Go build → unicoregpu2020/radixgates:latest
```

---


*Built by the UnicoreGPU team.*

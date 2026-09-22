# Real multi-GPU measurements (SGLang, DP / TP / EP)

Data: [`benchmarks/results/real_gpu/sglang_parallelism_runs.csv`](../benchmarks/results/real_gpu/sglang_parallelism_runs.csv).
All runs were done by me on rented machines on 2026-07-27, with RadixGates as delivered (git tag
`upstream-snapshot`) in front of SGLang. The numbers come from my own experiment logs (one run per
cell, demo-script output); the condensed engineering reports are the [RTX 4090 deployment report](deployment-reports/rtx-4090.md) and [A100 deployment report](deployment-reports/a100-sxm4.md).

## Results

| Platform | Parallelism | Model | Concurrency | Success | P50 | P95 | Time to first token |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 4x RTX 4090, PCIe | DP=4 | Llama-3-8B-Instruct | 40 | 40/40 | 1,215 ms | 2,300 ms | warm 37.9-62.3 ms, cold 86.6-113.3 ms |
| 4x RTX 4090, PCIe | PD 1P+1D | Llama-3-8B-Instruct | 30 | 30/30 | 2,544 ms | 4,973 ms | median 94.7 / 93.9 / 117.3 ms (short / medium / long) |
| 4x RTX 4090, PCIe | TP=4 | Qwen2.5-32B-Instruct | 20 | 20/20 | 4,836 ms | 6,475 ms | median 87-89 ms |
| 4x RTX 4090, PCIe | TP=4 + EP=4 | Qwen1.5-MoE-A2.7B-Chat | 20 | 20/20 | 1,653 ms | 2,095 ms | median 63-70 ms |
| 4x A100-SXM4-40GB, NVLink | DP=4 | Llama-3-8B-Instruct | 40 | 40/40 | 921 ms | 1,749 ms | warm 31.1-62.6 ms |
| 4x A100-SXM4-40GB, NVLink | TP=4 | Qwen2.5-32B-Instruct | 20 | 20/20 | 2,754 ms | 3,856 ms | median 40-41 ms |
| 4x A100-SXM4-40GB, NVLink | TP=4 + EP=4 | Qwen1.5-MoE-A2.7B-Chat | 20 | 20/20 | 846 ms | 1,161 ms | median 18-19 ms |

Observed P50 ratio, 4090 box over A100 box: DP 1.32x, TP 1.76x, EP 1.95x. The ordering
(more inter-GPU communication per token, larger gap) is what interconnect bandwidth predicts
(PCIe about 64 GB/s vs NVLink 600 GB/s per the platform specs), but see the next section before
attributing the size of the gap to NVLink.

## Why the ratios are not a controlled comparison

| Variable | 4x RTX 4090 run | 4x A100 run |
| --- | --- | --- |
| SGLang | 0.5.16 (cu129 runtime image) | 0.5.15.post1 |
| Driver / CUDA | 575.51.03 / up to CUDA 12.9 | 580.65.06 / CUDA 13.0 |
| Deployment | Docker Compose, `ipc: host` | Bare metal in a non-privileged container |
| GPU memory | 24 GB | 40 GB |
| `--mem-fraction-static` | image default | 0.75 (0.85 caused an OOM on 40 GB cards) |
| NCCL P2P | **Disabled** for TP and EP (`NCCL_P2P_DISABLE=1`) after an NCCL init failure | Enabled |

The 4090 TP and EP runs were forced onto the slow shared-memory path, so they measure "PCIe
without P2P" rather than "PCIe". What these runs support is: the same parallel strategies work on
both platforms, DP scales without inter-GPU traffic, and the A100/NVLink configuration was faster
for the communication-heavy modes. They do not isolate NVLink as the cause of a specific factor.
The DP ~4.0x scaling in the demo output is the script's estimate (40 x single-request time), not a
measured single-GPU baseline. The A100 cold time-to-first-token (228-420 ms) likely includes
first-request warm-up and should not be compared with the 4090 cold numbers.

## Problems diagnosed during these runs (8)

| Symptom | Root cause | Fix |
| --- | --- | --- |
| Container will not start: `requirement error: unsatisfied condition: cuda>=13.0` | `latest` image built for CUDA 13.0, driver supports up to 12.9 | Use the `cu129` runtime image |
| `unrecognized arguments: --enable-prefix-caching` | Flag removed in newer SGLang (radix cache is on by default) | Drop the flag |
| TP container restart loop, `NCCL error: unhandled system error` at `ncclCommInitRank` | NCCL communicator creation failed in this Docker/host configuration; shared memory and P2P behavior were candidate factors, not independently isolated | `ipc: host`, `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1` (slower path) |
| `unrecognized arguments: --enable-expert-parallel` | Replaced by `--ep-size N` in newer SGLang | Use `--ep-size 4` |
| `torch.OutOfMemoryError` at startup on 40 GB GPUs | `--mem-fraction-static 0.85` too high once auxiliary processes are counted | 0.75 and `expandable_segments:True` |
| One GPU unusable (`cudaErrorDevicesUnavailable`) after starting four servers at once | The device remained unavailable after cleanup; the host-level cause could not be isolated without reset privileges | Run per-GPU preflight checks, stagger startups, and replace the host if the device remains unavailable |
| Gateway would not start on the A100 machine | Port 8080 was already taken by the machine image's Jupyter | Run the gateway on 8081 (`GATEWAY_URL=http://localhost:8081`) |
| Model download stalled | Several download processes fought over the same partial files | Kill duplicates, delete the `.incomplete` shards, download once |

## What I would rerun to make this a controlled study

Same SGLang version, driver, container setup and `--mem-fraction-static` on both platforms;
`NCCL_P2P_DISABLE` off on the 4090 where it can be made to work, and on/off on both as an
explicit factor; at least 5 repetitions per cell with confidence intervals; fixed input and
output token counts; a single-GPU baseline for the DP scaling claim; P99 and tokens/s next to
P50/P95.

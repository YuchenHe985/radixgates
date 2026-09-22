# RadixGates：面向多 GPU LLM 推理的容错网关

[English README](README.md)

> 这是中文概览。代码、配置、技术文档和可复现实验以英文主版本为准，避免中英文双份文档长期不同步。

RadixGates 是位于客户端与 SGLang 推理节点之间的 Go 网关。它利用系统提示词前缀进行缓存亲和路由，并在数据并行副本或 Prefill/Decode 节点之间分配请求。本仓库包含三个阶段：导师项目提供的初始网关、在真实 RTX 4090 与 A100 多 GPU 机器上的部署验证，以及我完成并用故障注入验证的可靠性升级。

## 主要结果

- **节点故障路由：**4 个节点损失 1 个时，Rendezvous Hash 对存活节点的 key 重映射为 **0%**；原始 `hash % N` 为 **75.3%**。
- **节点崩溃：**干净完成率从 **85.2% 提升到 99.7%**。
- **灰度故障：**当 `/health` 正常但推理挂起时，P99 从 **11.8 秒降至 0.6 秒**。
- **流式安全：**只在首字节返回前重试，避免重复输出；中途断流会返回明确的 `upstream_interrupted` 事件。
- **测试：**共 53 个测试，在 `go test -race` 下覆盖熔断器、健康检查、路由、准入队列、指标和故障 worker 集成测试。
- **真实 GPU：**在 4× RTX 4090 与 4× A100-SXM4 上完成 DP/TP/EP 部署验证，并在 4090 上完成 PD 分离验证。

真实 GPU 数据不是严格的显卡或互联 A/B 实验：两台机器的软件版本、部署方式与 NCCL 配置不同。因此仓库只陈述完整配置下的观测结果，不把差距单独归因于 NVLink。

## 我完成的工程化升级

- 主动健康检查和每节点熔断器；
- 首字节前的有界重试与流式中断处理；
- Rendezvous Hash、负载上限和缓存亲和等待；
- 有界准入队列以及 `429`/`503`/`Retry-After`；
- TTFT、队列等待、重试、熔断转换和节点状态等 Prometheus 指标；
- 可复现的崩溃、连接重置、首 token 超时和中途断流测试。

## 快速运行

```bash
make test   # go vet + go test -race
make bench  # 原始版本与升级版本的故障注入对比
make plots  # 生成结果图
```

## 技术资料

- [英文主 README](README.md)
- [可靠性设计](docs/RELIABILITY.md)
- [部署与运维](docs/OPERATING.md)
- [真实 GPU 结果与限制](docs/real-gpu-results.md)
- [RTX 4090 部署报告](docs/deployment-reports/rtx-4090.md)
- [A100-SXM4 部署报告](docs/deployment-reports/a100-sxm4.md)

本中文页面用于国内求职者快速了解项目；所有深入技术内容保持一套英文版本，确保事实和维护状态统一。

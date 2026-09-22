# Validation Guide

This page routes readers to the current, reproducible evidence for RadixGates. It replaces the earlier classroom-style lab manual and duplicated benchmark summary.

## Gateway reliability

```bash
make test   # go vet and 53 tests under go test -race
make bench  # original versus upgraded gateway under crash and gray-failure scenarios
make plots  # regenerate the benchmark figures
```

The methodology, phase definitions, raw-data paths, and limits are documented in [`RELIABILITY.md`](RELIABILITY.md). Simulated workers make failure timing repeatable; they do not represent GPU throughput.

## Real inference engines

The repository README describes the llama.cpp prefix-cache experiment and links its raw results. This workload measures routing and cache behavior on a real inference engine, but it is not a GPU benchmark.

## Multi-GPU deployments

- [Normalized results and comparison limits](real-gpu-results.md)
- [4× RTX 4090 deployment report](deployment-reports/rtx-4090.md)
- [4× A100-SXM4 deployment report](deployment-reports/a100-sxm4.md)

The GPU reports record runs against the delivered `upstream-snapshot`. Reliability features added later on `main` are evaluated separately. Do not combine the two evidence sets or attribute cross-platform differences to one variable without a controlled rerun.

## Operations

Use [`OPERATING.md`](OPERATING.md) for deployment boundaries, configuration, alerts, and troubleshooting. The project is a reference gateway for a protected internal network; authentication, authorization, TLS termination, tenant quotas, and an audit-log sink belong at the surrounding platform layer.

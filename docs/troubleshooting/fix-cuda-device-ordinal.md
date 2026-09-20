# Fix CUDA Device Ordinal Error

## Symptom

`sglang-decode` container is unhealthy or restarting. Logs show:

```
CUDA error: invalid device ordinal
```

## Root Cause

`CUDA_VISIBLE_DEVICES=1` restricts the container to see only one GPU, which it maps
internally as device 0. But `--base-gpu-id 1` in the launch command tries to access
device 1 — which doesn't exist inside the container.

## Fix

In `docker-compose.pd.yml`, change both prefill and decode services from:

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0   # or =1
```

to:

```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=all
```

Then use `--base-gpu-id` in the command to select the correct GPU:
- Prefill: `--base-gpu-id 0`
- Decode:  `--base-gpu-id 1`

`NVIDIA_VISIBLE_DEVICES=all` lets both containers see all GPUs on the host.
`--base-gpu-id` then tells SGLang which one to actually use.

## Apply

```bash
docker compose -f docker-compose.pd.yml down
docker compose -f docker-compose.pd.yml up -d
```

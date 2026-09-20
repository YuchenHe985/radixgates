# Fix Unrecognized SGLang Arguments

## Symptom

`sglang-prefill` container keeps restarting. Logs show:

```
unrecognized arguments: --bootstrap-port 8998
```

or similar `unrecognized arguments` errors for PD-related flags.

## Root Cause

SGLang's PD disaggregation API changed across versions. Older flag names are no
longer accepted.

## Current Correct Flags (as of SGLang 0.5+)

| Old (broken) | New (correct) |
|---|---|
| `--bootstrap-port` | `--disaggregation-bootstrap-port` |
| `--prefill-server-url` | removed — handled by `sglang_router` instead |
| `--transfer-backend` | `--disaggregation-transfer-backend` |

**Prefill server** launch command:
```bash
python3 -m sglang.launch_server \
  --model-path <model> \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port 8998 \
  --disaggregation-transfer-backend mooncake \
  --base-gpu-id 0
```

**Decode server** launch command:
```bash
python3 -m sglang.launch_server \
  --model-path <model> \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --base-gpu-id 1
```

**Router** launch command:
```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://sglang-prefill:30000 8998 \
  --decode http://sglang-decode:30001 \
  --host 0.0.0.0 \
  --port 9000
```

## Verify Current SGLang Flags

```bash
docker run --rm lmsysorg/sglang:latest \
  python3 -m sglang.launch_server --help | grep disaggregation
```

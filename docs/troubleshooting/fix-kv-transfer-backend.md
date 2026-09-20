# Fix KV Transfer Backend Mismatch

## Symptom

Inference request hangs and never returns. Prefill logs show:

```
Prefill bootstrap failed for request ... with exception KVTransferError
Aborted by AbortReq
```

## Root Cause

Prefill and decode are using different transfer backends:
- Prefill: `mooncake` (attempts real TCP/P2P KV transfer)
- Decode: `fake` (expects no actual KV transfer)

They can't handshake, so the request is stuck then aborted.

## Fix

Both prefill and decode must use the **same** backend.

For production (real KV transfer over PCIe/NVLink):
```bash
# default — both use mooncake
docker compose -f docker-compose.pd.yml up -d
```

For routing-only testing (no actual KV transfer):
```bash
# fake backend is only valid for decode, not prefill
# Do NOT set PREFILL_BACKEND=fake — it will cause an AssertionError
DECODE_BACKEND=fake docker compose -f docker-compose.pd.yml up -d
# But this will still mismatch. Use mooncake for both.
```

**The only working combinations:**

| PREFILL_BACKEND | DECODE_BACKEND | Works? |
|-----------------|----------------|--------|
| `mooncake`      | `mooncake`     | ✅ Yes |
| `mooncake`      | `fake`         | ❌ KVTransferError |
| `fake`          | anything       | ❌ AssertionError on prefill |

## Apply

```bash
docker compose -f docker-compose.pd.yml down
docker compose -f docker-compose.pd.yml up -d   # mooncake is the default
```

# Dual-Machine PD Disaggregation Setup

## Architecture

```
Local Mac
  │
  ├─ SSH tunnel ──→ M1 :8080 (gateway + PD pair: GPU0=prefill, GPU1=decode)
  │
  └─ SSH reverse tunnel (via Mac) ──→ M1 :9001 → M2 :9000 (PD pair: GPU0=prefill, GPU1=decode)
```

## Step 1 — Start PD Node on Machine 2

```bash
# On M2:
git clone <repo_url> ~/RadixGates
cd ~/RadixGates
docker compose -f docker-compose.pd-node.yml up -d

# Wait for healthy:
watch docker ps   # all three containers (prefill, decode, router) must be healthy
```

## Step 2 — Establish SSH Tunnels from Local Mac

Open **two terminals** on your Mac:

**Terminal A** — forward tunnel (Mac → M1 gateway):
```bash
ssh -p <M1-port> root@<M1-ip> -N -L 8080:localhost:8080
```

**Terminal B** — reverse tunnel (M2:9000 → Mac:9001 → M1:9001):
```bash
# First, open forward tunnel to M2:
ssh -p <M2-port> root@<M2-ip> -N -L 9001:localhost:9000 &

# Then, expose Mac's port 9001 as M1's port 9001 via reverse tunnel:
ssh -p <M1-port> root@<M1-ip> -N -R 0.0.0.0:9001:localhost:9001
```

## Step 3 — Ensure M1 Allows GatewayPorts

On M1, verify `/etc/ssh/sshd_config` contains:
```
GatewayPorts clientspecified
```

If not:
```bash
echo "GatewayPorts clientspecified" >> /etc/ssh/sshd_config
systemctl reload sshd
# Then restart the reverse tunnel (Terminal B above)
```

## Step 4 — Start M1 with M2_HOST

```bash
# On M1:
cd ~/RadixGates
M2_HOST=host.docker.internal M2_PORT=9001 docker compose -f docker-compose.pd.yml up -d
```

## Step 5 — Verify Dual-Machine Routing

```bash
# On M1:
docker logs gateway --tail=20 | grep DirectHandler
# Must show requests going to BOTH:
#   node=http://sglang-router:9000     ← M1
#   node=http://host.docker.internal:9001  ← M2
```

## Verify Tunnel is Accessible from Docker

```bash
# On M1:
ss -tlnp | grep 9001
# Must show: 0.0.0.0:9001  (not 127.0.0.1)

curl http://172.17.0.1:9001/health
# Must return healthy response from M2's sglang-router
```

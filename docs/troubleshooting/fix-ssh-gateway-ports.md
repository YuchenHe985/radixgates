# Bridging M2 sglang-router to M1 Gateway When Ports Are Firewall-Blocked

## Symptom

Dual-machine setup: M2's port 9000 is not reachable from M1 (cloud provider
firewall blocks it), so `M2_HOST=<m2-ip>` fails with 502.

```bash
# From M1:
curl http://<m2-ip>:9000/health   # → connection refused / timeout
```

Even with UFW inactive and the port listening on `0.0.0.0:9000`, the cloud
provider's perimeter firewall blocks all ports except SSH.

## Solution — SSH tunnel + socat bridge

Since the SSH port on M2 is open, build a tunnel from M1 to M2, then relay it
to Docker's bridge network so the gateway container can reach it.

### Step 1 — Generate and authorize an SSH key from M1 → M2

```bash
# On M1:
ssh-keygen -t ed25519 -f ~/.ssh/id_m2 -N "" -q
cat ~/.ssh/id_m2.pub
```

```bash
# From local Mac: add M1's public key to M2
ssh -p <m2-port> root@<m2-ip> \
  "echo '<M1_PUBKEY>' >> ~/.ssh/authorized_keys"
```

### Step 2 — Build the SSH tunnel on M1

```bash
# On M1: forward localhost:9001 → M2:9000 via SSH
ssh -i ~/.ssh/id_m2 -o StrictHostKeyChecking=no -p <m2-ssh-port> \
  -fN -L 127.0.0.1:9001:localhost:9000 root@<m2-ip>

# Verify:
curl -sf http://127.0.0.1:9001/health   # → {"status":"ok",...}
```

### Step 3 — Get Docker bridge IP and relay with socat

```bash
# On M1: find Docker bridge IP (usually 172.17.0.1)
docker network inspect bridge --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}'

# Write a relay script (avoids pkill killing the ssh session)
cat > /tmp/start_relay.sh << 'SCRIPT'
#!/bin/bash
PIDS=$(pgrep -f "/usr/bin/socat TCP-LISTEN:9001")
[ -n "$PIDS" ] && kill $PIDS 2>/dev/null
sleep 0.5
nohup /usr/bin/socat TCP-LISTEN:9001,fork,bind=172.17.0.1,reuseaddr \
  TCP:127.0.0.1:9001 > /tmp/socat.log 2>&1 &
echo "relay_pid=$!"
SCRIPT
chmod +x /tmp/start_relay.sh && /tmp/start_relay.sh

# Verify Docker bridge can reach M2 router:
curl -sf http://172.17.0.1:9001/health   # → {"status":"ok",...}
```

### Step 4 — Start gateway with M2_HOST pointing to Docker bridge IP

```bash
# On M1:
cd ~/RadixGates
M2_HOST=172.17.0.1 M2_PORT=9001 \
  docker compose -f docker-compose.pd.yml up -d --no-deps gateway

# Confirm config has both nodes:
docker exec gateway cat /app/gateway/config/config.json
# sglang_instances should show: m1 (sglang-router:9000) + m2 (172.17.0.1:9001)
```

## Verify

Run the partition load balancing test:
```bash
# On local Mac (with SSH tunnel to M1:8080 open):
python3 examples/enterprise_qa_demo.py
# Expected: 成功: 30  失败: 0
# Requests spread across both machines by prefix-hash
```

## Notes

- `pkill -f "socat"` will kill your own SSH session if "socat" appears in the
  SSH command line. Always use `pgrep -f "/usr/bin/socat"` or write a script
  to avoid this.
- The socat relay and SSH tunnel must be restarted after M1 reboots. Consider
  adding them to `/etc/rc.local` or a systemd service for persistence.
- This approach works for any number of additional machines: add more tunnels
  on unique local ports (9002, 9003…) and include them in M2_HOST/M3_HOST.

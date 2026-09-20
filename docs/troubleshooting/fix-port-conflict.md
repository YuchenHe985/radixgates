# Fix Local Port 8080 Already in Use

## Symptom

Running demo scripts locally returns:

```
Connection aborted. ConnectionResetError(54, 'Connection reset by peer')
```

Or `curl localhost:8080/health` returns a response from an unexpected service
(e.g. `drogon/1.9.12` 404 page instead of RadixGates JSON).

## Diagnosis

```bash
lsof -i :8080
# Shows a process (e.g. sglang_ga, another gateway instance) occupying port 8080
```

## Root Cause

A local process (often a leftover RadixGates binary or another service) is
listening on port 8080, preventing the SSH tunnel from binding there.
The demo scripts connect to this local process instead of M1's gateway.

## Fix

```bash
# Kill whatever is on port 8080
kill $(lsof -ti :8080)

# Re-establish the SSH tunnel
ssh -p <ssh-port> root@<server-ip> -N -L 8080:localhost:8080

# Verify tunnel is routing to M1
curl http://localhost:8080/health
# → {"status":"ok","mode":"direct"}
```

## Prevention

Do not run a local RadixGates binary while also using the SSH tunnel.
Use one or the other, not both simultaneously.

#!/usr/bin/env python3
"""Routing policies against real inference servers (llama.cpp), not simulated workers.

Three llama-server workers each have two slots, and llama.cpp keeps the KV cache of a slot's last
prompt, so a request whose prompt prefix is already in a slot skips most of its prefill. The experiment
sends requests that share long system prompts (a database schema followed by a question) through:

  original    the gateway built from the upstream-snapshot tag (FNV-32 of the system prompt, mod N)
  upgraded    this repository's gateway (rendezvous hashing with bounded load; a full preferred node
              spills to the next-ranked node at once)
  upgraded_wait  the same gateway with routing.affinity_wait set (a full preferred node is waited for)
  upgraded_place the same gateway with placement memory (a new prefix goes to the node holding the fewest
                 prefixes and stays there)
  upgraded_place_wait  placement memory and the affinity wait together
  roundrobin  no gateway: the client rotates over the workers

and reports, per policy, the fraction of prompt tokens served from the KV cache, prefill time and
end-to-end latency. Everything runs on one machine, so it shows how routing changes cache behaviour on
a real engine; it is not a GPU benchmark.

    python3 benchmarks/real_engine/run_real.py --model models/qwen2.5-0.5b-instruct-q4_k_m.gguf
"""
import argparse
import concurrent.futures as cf
import csv
import http.client
import json
import os
import random
import shutil
import signal
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKER_BASE_PORT = 9101
GATEWAY_PORT = 19090


def schema_prompt(pid):
    """A distinct ~600-token system prompt per id: a made-up database schema."""
    tables = []
    for t in range(12):
        cols = ", ".join(f"c{pid}_{t}_{c} {['INTEGER', 'TEXT', 'REAL', 'DATE'][c % 4]}" for c in range(8))
        tables.append(f"CREATE TABLE department{pid}_table{t} ({cols});")
    return ("You translate questions into SQL for the following schema. Answer with one SQL statement only.\n"
            + "\n".join(tables))


def wait_http(port, path, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            c.request("GET", path)
            if c.getresponse().status == 200:
                return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def post_json(port, path, body, timeout=120):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    return r.status, dict(r.getheaders()), data


def warm_workers(args):
    """One tiny request per worker so model load and first-run costs are not measured."""
    for i in range(args.workers):
        post_json(WORKER_BASE_PORT + i, "/v1/chat/completions",
                  {"model": "default", "max_tokens": 2, "messages": [{"role": "user", "content": "hi"}]})


def erase_caches(args):
    """Every measured run starts with empty KV caches on every worker, whatever ran before."""
    for i in range(args.workers):
        for slot in range(args.slots):
            c = http.client.HTTPConnection("127.0.0.1", WORKER_BASE_PORT + i, timeout=10)
            c.request("POST", f"/slots/{slot}?action=erase")
            c.getresponse().read()


def prefill_counters(args):
    """Prompt tokens each worker actually evaluated (cached tokens are not counted)."""
    out = []
    for i in range(args.workers):
        c = http.client.HTTPConnection("127.0.0.1", WORKER_BASE_PORT + i, timeout=10)
        c.request("GET", "/metrics")
        text = c.getresponse().read().decode()
        out.append(next(float(l.split()[-1]) for l in text.splitlines() if l.startswith("llamacpp:prompt_tokens_total")))
    return out


class Fleet:
    """Workers and gateways for one experiment, always cleaned up."""

    def __init__(self, args, tmp):
        self.args, self.tmp, self.procs = args, tmp, []

    def start_workers(self):
        for i in range(self.args.workers):
            port = WORKER_BASE_PORT + i
            os.makedirs(os.path.join(self.tmp, f"slots{i}"), exist_ok=True)
            extra = [] if self.args.cache_ram is None else ["--cache-ram", str(self.args.cache_ram)]
            self.procs.append(subprocess.Popen(
                ["llama-server", "-m", self.args.model, "--port", str(port), "-np", str(self.args.slots),
                 "-c", str(self.args.ctx), "-t", str(self.args.threads), "--metrics",
                 "--slot-save-path", os.path.join(self.tmp, f"slots{i}")] + extra,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for i in range(self.args.workers):
            if not wait_http(WORKER_BASE_PORT + i, "/health"):
                raise RuntimeError(f"worker {i} did not become healthy")

    def config(self, routing=None):
        path = os.path.join(self.tmp, "gateway.json")
        cfg = {"port": GATEWAY_PORT, "default_model": "default", "max_concurrent_per_node": self.args.slots,
               "sglang_instances": [{"group": "local", "host": "127.0.0.1", "port": WORKER_BASE_PORT + i,
                                     "max_concurrent": self.args.slots} for i in range(self.args.workers)]}
        if routing:
            cfg["routing"] = {"bounded_load_factor": 1.25, "affinity_floor": 4, **routing}
        with open(path, "w") as f:
            json.dump(cfg, f)
        return path

    def start_gateway(self, binary, routing=None):
        p = subprocess.Popen([binary], env=dict(os.environ, CONFIG_PATH=self.config(routing)),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(p)
        if not wait_http(GATEWAY_PORT, "/health"):
            raise RuntimeError("gateway did not start")
        return p

    def stop(self, p):
        p.send_signal(signal.SIGTERM)
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()
        self.procs.remove(p)

    def close(self):
        for p in list(self.procs):
            try:
                p.kill()
            except OSError:
                pass


def routing_for(policy, args):
    """routing options of the upgraded gateway for each policy (the original ignores them)."""
    return {"upgraded_wait": {"affinity_wait": args.affinity_wait},
            "upgraded_place": {"placement_size": 1024},
            "upgraded_place_wait": {"placement_size": 1024, "affinity_wait": args.affinity_wait}}.get(policy)


def build_gateways(tmp):
    """Build the upgraded gateway from the working tree and the original from the upstream-snapshot tag."""
    upgraded = os.path.join(tmp, "gateway-upgraded")
    subprocess.run(["go", "build", "-o", upgraded, "."], cwd=ROOT / "gateway-go", check=True)
    worktree = os.path.join(tmp, "orig")
    subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", worktree, "upstream-snapshot"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    original = os.path.join(tmp, "gateway-original")
    subprocess.run(["go", "build", "-o", original, "."], cwd=os.path.join(worktree, "gateway-go"), check=True)
    return original, upgraded, worktree


def workload(args, seed):
    rnd = random.Random(seed)
    prompts = [schema_prompt(i) for i in range(args.prompts)]
    reqs = [(pid, f"How many rows does department{pid}_table{rnd.randrange(12)} have where column c{pid}_0_{rnd.randrange(8)} "
                  f"is greater than {rnd.randrange(1000)}? (query {n})")
            for n in range(args.prompts * args.per_prompt) for pid in [n % args.prompts]]
    rnd.shuffle(reqs)
    return prompts, reqs


def run_policy(policy, args, prompts, reqs, gateway_port=None):
    def one(item):
        pid, question = item
        body = {"model": "default", "max_tokens": args.max_tokens, "temperature": 0, "stream": False,
                "messages": [{"role": "system", "content": prompts[pid]}, {"role": "user", "content": question}]}
        t0 = time.time()
        try:
            if policy == "roundrobin":
                with rr_lock:
                    n = rr_counter[0]
                    rr_counter[0] += 1
                port, path = WORKER_BASE_PORT + n % args.workers, "/v1/chat/completions"
            else:
                port, path = gateway_port, "/v1/chat"
            status, headers, data = post_json(port, path, body)
            lat = (time.time() - t0) * 1000
            if status != 200:
                return {"ok": 0, "lat": lat, "status": status}
            d = json.loads(data)
            tm = d.get("timings", {})
            node = headers.get("X-Gateway-Node") or f"127.0.0.1:{port}"
            return {"ok": 1, "lat": lat, "node": node, "prompt": d["usage"]["prompt_tokens"],
                    "cached": d["usage"]["prompt_tokens_details"]["cached_tokens"], "prefill_ms": tm.get("prompt_ms", 0.0)}
        except OSError:
            return {"ok": 0, "lat": (time.time() - t0) * 1000, "status": -1}

    rr_lock, rr_counter = threading.Lock(), [0]
    t0 = time.time()
    with cf.ThreadPoolExecutor(args.concurrency) as ex:
        rows = list(ex.map(one, reqs))
    wall = time.time() - t0
    return rows, wall


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def summarize(rows, wall):
    ok = [r for r in rows if r["ok"]]
    prompt, cached = sum(r["prompt"] for r in ok), sum(r["cached"] for r in ok)
    return {"requests": len(rows), "ok": len(ok), "cache_hit": cached / prompt if prompt else 0.0,
            "prefill_p50": pct([r["prefill_ms"] for r in ok], 0.5), "prefill_p95": pct([r["prefill_ms"] for r in ok], 0.95),
            "lat_p50": pct([r["lat"] for r in ok], 0.5), "lat_p95": pct([r["lat"] for r in ok], 0.95),
            "rps": len(ok) / wall, "share": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--prompts", type=int, default=6, help="distinct system prompts (the working set)")
    ap.add_argument("--per-prompt", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--policies", default="original,upgraded,upgraded_wait,upgraded_place,upgraded_place_wait,roundrobin")
    ap.add_argument("--affinity-wait", default="10s", help="routing.affinity_wait for the *_wait policies")
    ap.add_argument("--cache-ram", type=int, default=None,
                    help="llama-server host-memory prompt cache in MiB (its default is 8192; 0 disables it, so a "
                         "worker's cache capacity is just its slots)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "real_engine"))
    args = ap.parse_args()
    args.model = str(Path(args.model).expanduser())
    Path(args.out).mkdir(parents=True, exist_ok=True)

    tmp = tempfile.mkdtemp(prefix="rg-real-")
    worktree = None
    fleet = Fleet(args, tmp)
    policies = tuple(args.policies.split(","))
    results = {p: [] for p in policies}
    try:
        original, upgraded, worktree = build_gateways(tmp)
        fleet.start_workers()
        warm_workers(args)
        for rep in range(args.repeats):
            prompts, reqs = workload(args, seed=rep)
            for policy in policies:
                gw = None
                if policy != "roundrobin":
                    gw = fleet.start_gateway(original if policy == "original" else upgraded,
                                             routing_for(policy, args))
                erase_caches(args)
                before = prefill_counters(args)
                rows, wall = run_policy(policy, args, prompts, reqs, GATEWAY_PORT)
                after = prefill_counters(args)
                s = summarize(rows, wall)
                s["share"] = [int(b - a) for a, b in zip(before, after)]
                results[policy].append(s)
                with open(os.path.join(args.out, f"{policy}_run{rep}.csv"), "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["ok", "lat", "node", "prompt", "cached", "prefill_ms", "status"])
                    w.writeheader()
                    w.writerows([{k: r.get(k, "") for k in w.fieldnames} for r in rows])
                print(f"run {rep} {policy:10s} hit={s['cache_hit']:.1%} prefill p50={s['prefill_p50']:.0f}ms "
                      f"lat p50={s['lat_p50']:.0f}ms p95={s['lat_p95']:.0f}ms rps={s['rps']:.2f} prefill tokens/worker={s['share']}", flush=True)
                if gw:
                    fleet.stop(gw)
                    time.sleep(0.5)
    finally:
        fleet.close()
        if worktree:
            subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", worktree],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.rmtree(tmp, ignore_errors=True)

    med = lambda xs: statistics.median(xs)
    lines = ["| policy | cache hit (prompt tokens from KV cache) | prefill p50 / p95 (ms) | latency p50 / p95 (ms) | req/s | prompt tokens evaluated per worker |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for policy, runs in results.items():
        lines.append(f"| {policy} | {med([r['cache_hit'] for r in runs]):.1%} | {med([r['prefill_p50'] for r in runs]):.0f} / "
                     f"{med([r['prefill_p95'] for r in runs]):.0f} | {med([r['lat_p50'] for r in runs]):.0f} / "
                     f"{med([r['lat_p95'] for r in runs]):.0f} | {med([r['rps'] for r in runs]):.2f} | "
                     f"{' / '.join(str(x) for x in runs[len(runs) // 2]['share'])} |")
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(f"{args.workers} llama-server workers ({args.slots} slots each, host-memory prompt cache "
                f"{'default (8192 MiB)' if args.cache_ram is None else str(args.cache_ram) + ' MiB'}), {args.prompts} distinct system prompts, "
                f"{args.prompts * args.per_prompt} requests per run, concurrency {args.concurrency}, median of {args.repeats} runs.\n\n"
                + table + "\n")


if __name__ == "__main__":
    main()

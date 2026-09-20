# Operating the gateway

For whoever runs the gateway in front of a pool of SGLang (or other OpenAI-compatible) nodes: what to watch, how to choose the settings, and what to do when
something is wrong. Everything here refers to metrics and settings that exist in this repository; the numbers behind the tuning advice are in the
[README](../README.md#real-inference-engines-prefix-cache-behaviour-under-each-routing-policy) and [RELIABILITY.md](RELIABILITY.md).

## What to watch

| Question | Signal |
| --- | --- |
| Is every node in the pool serving? | `radixgates_node_up`, `radixgates_node_breaker_open` per node; `GET /admin/nodes` shows up, breaker state, in-flight and placed prefixes |
| Is the gateway rejecting or dropping work? | `radixgates_requests_total{status}`: `queue_full` and `queue_timeout` (capacity), `no_node` (nothing available), `upstream_failed`, `timeout`, `midstream_failed` |
| Are retries hiding a sick node? | `radixgates_retries_total{reason}` and `radixgates_upstream_attempts_total{node,result}` |
| How slow is the first token? | `radixgates_time_to_first_byte_seconds`, `radixgates_queue_wait_seconds` |
| Is one node doing all the work? | `radixgates_node_inflight` per node, `placed_prefixes` in `/admin/nodes` |
| Is the pool saturated? | `radixgates_queue_depth` above zero for minutes, not seconds |

`GET /readyz` is the load balancer check: it fails when no node is available.

## Alert rules

Starting points for Prometheus; thresholds depend on your traffic and should be tuned against a quiet week.

```yaml
groups:
  - name: radixgates
    rules:
      - alert: RadixGatesNodeDown
        expr: radixgates_node_up == 0
        for: 1m
        annotations: {summary: "{{ $labels.node }} failed its health checks"}
      - alert: RadixGatesBreakerOpen
        expr: radixgates_node_breaker_open == 1
        for: 2m
        annotations: {summary: "{{ $labels.node }} is failing real requests although its /health may be green"}
      - alert: RadixGatesRejectingRequests
        expr: sum(rate(radixgates_requests_total{status=~"queue_full|queue_timeout|no_node"}[5m])) / sum(rate(radixgates_requests_total[5m])) > 0.01
        for: 5m
        annotations: {summary: "more than 1% of requests are rejected for lack of capacity or nodes"}
      - alert: RadixGatesRetryRateHigh
        expr: sum(rate(radixgates_retries_total[10m])) / sum(rate(radixgates_requests_total[10m])) > 0.05
        for: 10m
        annotations: {summary: "more than 5% of requests needed a retry: a node is degrading"}
      - alert: RadixGatesStreamsBreaking
        expr: sum(rate(radixgates_midstream_failures_total[10m])) > 0
        for: 10m
        annotations: {summary: "streams are ending mid-response; clients see upstream_interrupted"}
      - alert: RadixGatesQueueBuilding
        expr: radixgates_queue_depth > 0
        for: 5m
        annotations: {summary: "requests have been waiting for a node for minutes: the pool is saturated"}
```

## Choosing the settings

| Setting | How to choose it |
| --- | --- |
| `max_concurrent_per_node` | The engine's own concurrency limit (running requests or slots). Above it, requests queue inside the engine where the gateway cannot see them |
| `reliability.attempt_timeout` | A few times the p99 time to first token you see under normal load. Too small turns slow prefill into retries |
| `reliability.stream_idle_timeout` | Longer than the slowest normal gap between tokens |
| `reliability.breaker.failure_threshold` / `open_duration` | Threshold: how many consecutive failures prove a node is bad (3-5); open duration: how long a bad node deserves before one probe (seconds, not minutes; it grows exponentially up to the maximum) |
| `reliability.health.interval` | Detection time is about `interval x unhealthy_threshold`; probing more often than the engine can answer just adds load |
| `admission.max_queue` / `queue_timeout` | Queue as long as a client is willing to wait, not as long as memory allows: a request that waits ten seconds is usually abandoned anyway |
| `routing.placement_size`, `routing.affinity_wait` | See below |

### Prefix routing: when to turn on placement memory and the affinity wait

The default (rendezvous hashing, spill to the next node when the preferred one is full) is right when each node's prefix cache holds all the
prefixes the traffic uses, or when a cache miss is cheap. It is wrong when a node's cache holds fewer prefixes than the traffic uses, because spilling
scatters one prefix over several nodes and evicts it everywhere. In the llama.cpp test with a cache of two prefixes per node and six prefixes in use, the
default reached 24% cache hits against 59% for queueing, and 61.5% with the two options below.

- Set `routing.placement_size` to at least the number of distinct hot prefixes per group. It spreads new prefixes evenly and keeps each one on its node.
- Set `routing.affinity_wait` to a small multiple of the cost of a cache miss (the extra prefill time), so waiting for the right node is cheaper than
  recomputing on another. The test used 10 s where a miss cost about 1.3 s; that was generous, and waiting longer than the queue timeout is pointless.
- Leave both off, as by default, when the engine can restore evicted prefixes cheaply (llama.cpp's default host-memory cache made every policy reach
  95-98% hits) or when prefills are so fast that recomputing is cheaper than waiting.

## When something is wrong

| Symptom | Likely cause | Check | Action |
| --- | --- | --- | --- |
| `NodeDown` or `BreakerOpen` for one node | node crashed, hung, or is failing requests while `/health` answers | `/admin/nodes`; the node's own logs; `radixgates_upstream_attempts_total{node,result}` for the failure kind | fix or replace the node; traffic already routes around it. A breaker that keeps re-opening means the node is still bad |
| Retry rate up, no node down | a node is slow rather than dead (first-token timeouts) | `radixgates_retries_total` by reason; time to first byte per node | look for a wedged GPU or memory pressure on that node; drain it |
| `queue_full` / `queue_timeout` rising | the pool is saturated | `radixgates_queue_depth`, `node_inflight` at their limits | add nodes, or lower `max_queue` so clients get a fast 429 with `Retry-After` and back off |
| Slow first token, requests otherwise fine | cache misses: prefixes scattered or evicted | per-node in-flight and `placed_prefixes`; the engine's cache hit counters | enable placement memory and the affinity wait (above), or add cache capacity |
| Everything 503 with `no_node` | every node is down or its breaker is open | `/admin/nodes` | restore at least one node; the gateway keeps probing and will return to service |
| Streams cut mid-response | a node died while generating | `radixgates_midstream_failures_total`; clients receive `upstream_interrupted` | clients may retry; only requests that have not started streaming are retried by the gateway |

## Rolling a node out of service

Stop sending it new work by failing its `/health`; requests already running finish. The gateway marks it down after `unhealthy_threshold` failed
probes and routes its prefixes elsewhere. With placement memory on, those prefixes move once and stay on their new node when the old one returns.

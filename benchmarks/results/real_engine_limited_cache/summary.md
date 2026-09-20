3 llama-server workers (2 slots each, host-memory prompt cache 0 MiB), 6 distinct system prompts, 120 requests per run, concurrency 6, median of 3 runs.

| policy | cache hit (prompt tokens from KV cache) | prefill p50 / p95 (ms) | latency p50 / p95 (ms) | req/s | prompt tokens evaluated per worker |
| --- | ---: | ---: | ---: | ---: | --- |
| original | 59.1% | 92 / 1260 | 1699 / 4532 | 3.13 | 2365 / 9938 / 33442 |
| upgraded | 24.4% | 1895 / 2764 | 3084 / 5246 | 1.80 | 23832 / 36769 / 23882 |
| upgraded_wait | 45.7% | 565 / 1081 | 1940 / 7320 | 2.60 | 0 / 48060 / 5542 |
| upgraded_place | 19.7% | 1634 / 3664 | 3070 / 5519 | 1.71 | 24760 / 39352 / 23886 |
| upgraded_place_wait | 61.5% | 110 / 1930 | 1517 / 5071 | 3.40 | 12609 / 12614 / 18783 |
| roundrobin | 22.0% | 1547 / 2278 | 3357 / 6396 | 1.68 | 31098 / 25803 / 31099 |

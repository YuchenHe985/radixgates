3 llama-server workers (2 slots each, host-memory prompt cache default (8192 MiB)), 6 distinct system prompts, 120 requests per run, concurrency 6, median of 3 runs.

| policy | cache hit (prompt tokens from KV cache) | prefill p50 / p95 (ms) | latency p50 / p95 (ms) | req/s | prompt tokens evaluated per worker |
| --- | ---: | ---: | ---: | ---: | --- |
| original | 96.2% | 95 / 236 | 601 / 1218 | 9.33 | 561 / 1083 / 1682 |
| upgraded | 96.5% | 181 / 353 | 547 / 777 | 10.77 | 1510 / 787 / 1630 |
| upgraded_wait | 97.3% | 98 / 179 | 530 / 1448 | 10.15 | 0 / 1993 / 1032 |
| upgraded_place | 96.8% | 174 / 438 | 513 / 840 | 11.40 | 1388 / 612 / 1577 |
| upgraded_place_wait | 97.3% | 92 / 219 | 483 / 1140 | 10.55 | 1014 / 1043 / 952 |
| roundrobin | 97.7% | 117 / 248 | 634 / 1323 | 9.59 | 963 / 699 / 887 |

# Notice

RadixGates was created by the UnicoreGPU team and handed to me (Yuchen He) as a project to extend.
The upstream snapshot carries no licence file. This repository is published with the project
owner's permission; the imported code is theirs and is tagged `upstream-snapshot`
(`git diff upstream-snapshot` shows exactly what I changed). The original README is kept at `docs/UPSTREAM_README.md`
(its embedded resume blurb is removed).

My additions and rewrites (`gateway-go/breaker`, `gateway-go/router`, `gateway-go/handler/direct.go`,
`gateway-go/metrics`, `gateway-go/config`, `gateway-go/main.go`, `gateway-go/cmd`, `gateway-go/internal`,
`benchmarks/`, `.github/`, `docs/real-gpu-results.md`, `README.md`) are offered under the MIT licence.

Rented-machine addresses and key names that appeared in the upstream docs and examples were replaced with documentation example
addresses (203.0.113.10). The accompanying slides, which embedded them, are omitted; the lab notes in `docs/lab-notes/` carry the same content.
The deployment runbooks in the original delivery's `agents/` directory are not included; their failure-pattern notes are kept as
troubleshooting guides in `docs/troubleshooting/`.

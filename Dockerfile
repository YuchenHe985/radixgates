# =============================================================================
#  RadixGates — Go Gateway (direct mode)
#  Multi-stage build:
#    builder  → compiles the Go gateway binary (pure Go, no CGO)
#    runtime  → slim image with only the binary + default config
# =============================================================================

# ─── Stage 1: Build Go binary ────────────────────────────────────────────────
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Install Go 1.22
RUN wget -q https://go.dev/dl/go1.22.5.linux-amd64.tar.gz -O /tmp/go.tar.gz \
    && tar -C /usr/local -xzf /tmp/go.tar.gz \
    && rm /tmp/go.tar.gz

ENV PATH="/usr/local/go/bin:${PATH}"
ENV CGO_ENABLED=0

WORKDIR /src

# ── Build Gateway ─────────────────────────────────────────────────────────────
COPY gateway-go/ ./gateway-go/
RUN cd gateway-go && go mod download && \
    go build -ldflags="-s -w" -o /out/sglang_gateway .

# =============================================================================
# ─── Stage 2: Slim runtime image ─────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled binary + default config
COPY --from=builder /out/sglang_gateway  /usr/local/bin/sglang_gateway
COPY gateway-go/config/config.json       /app/gateway/config/config.json

WORKDIR /app/gateway
EXPOSE 8080

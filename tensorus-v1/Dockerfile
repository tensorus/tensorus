# syntax=docker/dockerfile:1
# Multi-stage build for the Tensorus REST API server.
#
# The workspace's default features are pure-Rust (no C/protoc), so the standard
# Rust image builds it without extra system packages.

FROM rust:1-bookworm AS builder
WORKDIR /build
COPY . .
RUN cargo build --release -p tensorus-api --bin tensorus-server

FROM debian:bookworm-slim AS runtime
RUN useradd --system --uid 10001 tensorus \
    && mkdir -p /data && chown tensorus:tensorus /data
COPY --from=builder /build/target/release/tensorus-server /usr/local/bin/tensorus-server
COPY --from=builder /build/tensorus.toml /etc/tensorus/tensorus.toml

USER tensorus
ENV TENSORUS_DATA_DIR=/data \
    TENSORUS_HOST=0.0.0.0 \
    TENSORUS_REST_PORT=8080 \
    TENSORUS_LOG_JSON=true \
    RUST_LOG=info
EXPOSE 8080
VOLUME ["/data"]

# Kubernetes liveness/readiness probes hit the unauthenticated /health endpoint
# (see the Helm chart). No in-image healthcheck is defined to keep the runtime
# image minimal (no shell/curl).
ENTRYPOINT ["/usr/local/bin/tensorus-server"]

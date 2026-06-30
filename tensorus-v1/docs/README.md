# Tensorus v1.0 — Documentation

Tensorus is an agentic, **tensor-native database**: tensors are stored in their
native shape and made queryable by their mathematical properties and structural
similarity, with a built-in AI query/agent layer.

## Start here

| Document | What it covers |
|----------|----------------|
| [Architecture](./architecture.md) | Service map, data flow, design principles, documented substitutions |
| [Data model](./data-model.md) | `TensorId`, `DType`, `Shape`, `TensorDescriptor`, `TensorData`, `TensorRecord`, errors |
| [API reference](./api-reference.md) | REST endpoints (shapes, status codes, curl) + gRPC service |
| [Configuration](./configuration.md) | `tensorus.toml` + environment variables |
| [Benchmarks](../BENCHMARKS.md) | Measured vs. target performance |

## Service references

The system is a Cargo workspace of eight crates. Each has a detailed reference:

| Service | Responsibility |
|---------|----------------|
| [`tensorus-core`](./services/tensorus-core.md) | Shared types, traits, errors, Arrow bridge |
| [`tensorus-compute`](./services/tensorus-compute.md) | `TensorDescriptor` computation (norms, rank, eigenvalues, booleans) |
| [`tensorus-storage`](./services/tensorus-storage.md) | Crash-safe segment+WAL storage, adaptive compression, tiering |
| [`tensorus-index`](./services/tensorus-index.md) | Learned (PGM/ALEX), vector (HNSW/DiskANN), metadata indexes |
| [`tensorus-search`](./services/tensorus-search.md) | Tensor contraction similarity engine |
| [`tensorus-ai`](./services/tensorus-ai.md) | LLM router, NQL, ReAct agent, auto-optimizer |
| [`tensorus-api`](./services/tensorus-api.md) | REST API, server binary, telemetry |
| [`tensorus-python`](./services/tensorus-python.md) | PyO3 bindings + Python SDK |

## Other resources

- Project overview & quickstart: [`../README.md`](../README.md)
- Deployment (Docker + Helm): [`../deploy/README.md`](../deploy/README.md)
- Python SDK build notes: [`../python/README.md`](../python/README.md)
- API-level (rustdoc) docs: `cargo doc --workspace --no-deps --open`

## How the services compose (at a glance)

```
Client → tensorus-api (auth, rate limit, metrics)
       → TensorService
           ├─ tensorus-compute   (compute descriptor on insert)
           ├─ tensorus-storage   (durable segment+WAL, hot map, tiering, compression)
           ├─ tensorus-index     (PGM/ALEX/HNSW/DiskANN/metadata)
           ├─ tensorus-search    (contraction similarity)
           └─ tensorus-ai        (LLM router → NQL plan → execute; ReAct agent; optimizer)
       (all built on tensorus-core types/traits/errors)
```

See [Architecture](./architecture.md) for the full picture, including the
data-flow walkthroughs for insert, property search, contraction search, NQL, and
the ReAct loop.

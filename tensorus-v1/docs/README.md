# Tensorus v1.0 — Documentation

Tensorus is an agentic, **tensor-native database**: tensors are stored in their
native shape and made queryable by their mathematical properties and structural
similarity, with a built-in AI query/agent layer.

**Capabilities at a glance:** dataset CRUD; index-backed property search; HNSW
vector similarity search; tensor-contraction (structural) similarity; neural
query language (NQL) and a ReAct agent over a model-agnostic LLM router;
multi-tenancy with RBAC and quotas; durable storage with snapshot/restore,
persisted vector graphs, and single-leader replication; API-key auth, rate
limiting, Prometheus metrics, and structured tracing.

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
| [`tensorus-storage`](./services/tensorus-storage.md) | Crash-safe segment+WAL storage, compression, tiering, snapshot/restore, replication change-log |
| [`tensorus-index`](./services/tensorus-index.md) | Learned (PGM/ALEX), vector (HNSW + persistence/DiskANN), metadata indexes |
| [`tensorus-search`](./services/tensorus-search.md) | Tensor contraction similarity engine |
| [`tensorus-ai`](./services/tensorus-ai.md) | LLM router, NQL, ReAct agent, auto-optimizer |
| [`tensorus-api`](./services/tensorus-api.md) | REST API (search/NQL/agent), multi-tenancy + RBAC, replication, server binary, telemetry |
| [`tensorus-python`](./services/tensorus-python.md) | PyO3 bindings + Python SDK |

## Other resources

- Project overview & quickstart: [`../README.md`](../README.md)
- Deployment (Docker + Helm): [`../deploy/README.md`](../deploy/README.md)
- Python SDK build notes: [`../python/README.md`](../python/README.md)
- API-level (rustdoc) docs: `cargo doc --workspace --no-deps --open`

## How the services compose (at a glance)

```
Client → tensorus-api (auth/RBAC, rate limit, metrics; multi-tenant scoping)
       → TensorService
           ├─ tensorus-compute   (compute descriptor on insert)
           ├─ tensorus-storage   (durable segment+WAL, hot map, tiering, compression,
           │                      snapshot/restore, replication change-log)
           ├─ tensorus-index     (PGM/ALEX/HNSW[+persistence]/DiskANN/metadata)
           ├─ tensorus-search    (contraction similarity)
           └─ tensorus-ai        (LLM router → NQL plan → execute; ReAct agent; optimizer)
       + control plane: /admin (tenants, keys, snapshot/restore) · /replication (followers)
       (all built on tensorus-core types/traits/errors)
```

See [Architecture](./architecture.md) for the full picture, including the
data-flow walkthroughs for insert, property search, vector/contraction search,
NQL, the ReAct loop, auth/tenant scoping, and replication.

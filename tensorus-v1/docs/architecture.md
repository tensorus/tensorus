# Tensorus v1.0 — Architecture

Tensorus is an **agentic, tensor-native database**. Tensors (multi-dimensional
arrays) are first-class citizens: they are stored in their native shape and made
queryable by their *mathematical properties* (norm, rank, eigenvalues, symmetry,
sparsity) and by *structural similarity*, with a built-in AI layer that turns
natural language into executable query plans and drives autonomous workflows.

This document describes how the services fit together. Each service has its own
detailed reference under [`docs/services/`](./services/).

---

## 1. Layered service map

```
                         ┌──────────────────────────────────────────────┐
                         │                Clients                        │
                         │  Python SDK · REST · (gRPC) · CLI/HTTP         │
                         └───────────────┬──────────────────────────────┘
                                         │
                         ┌───────────────▼──────────────────────────────┐
       tensorus-api      │  API Gateway: auth · rate limit · metrics      │
                         │  TensorService (transport-agnostic logic)      │
                         └───────┬──────────────────┬─────────────┬──────┘
                                 │                  │             │
              ┌──────────────────▼───┐   ┌──────────▼───────┐  ┌──▼─────────────┐
 tensorus-ai  │ LLM router · NQL ·   │   │  tensorus-search │  │ tensorus-index │
              │ ReAct agent · opt.   │   │  contraction sim.│  │ PGM/ALEX/HNSW/ │
              └──────────────────┬───┘   └──────────┬───────┘  │ DiskANN/meta   │
                                 │                  │          └──┬─────────────┘
              ┌──────────────────▼──────────────────▼─────────────▼─────────────┐
 tensorus-    │  tensorus-compute (descriptors)   ·   tensorus-storage (durable │
 compute /    │  norms/rank/eigenvalues/booleans      segments + WAL + tiering  │
 storage      │                                       + adaptive compression)   │
              └──────────────────────────────┬───────────────────────────────┘
                                             │
                         ┌───────────────────▼──────────────────────────┐
 tensorus-core           │  Types · Traits · Errors · Arrow bridge        │
                         │  (the vocabulary every layer shares)           │
                         └────────────────────────────────────────────────┘
```

| Crate / service | Role | Reference |
|-----------------|------|-----------|
| `tensorus-core` | Shared types, traits, errors, Arrow bridge | [core](./services/tensorus-core.md) |
| `tensorus-compute` | Computes a tensor's mathematical descriptor | [compute](./services/tensorus-compute.md) |
| `tensorus-storage` | Durable storage (segments+WAL), compression, tiering | [storage](./services/tensorus-storage.md) |
| `tensorus-index` | Learned (PGM/ALEX), vector (HNSW/DiskANN), metadata indexes | [index](./services/tensorus-index.md) |
| `tensorus-search` | Tensor contraction similarity engine | [search](./services/tensorus-search.md) |
| `tensorus-ai` | LLM router, NQL, ReAct agent, auto-optimizer | [ai](./services/tensorus-ai.md) |
| `tensorus-api` | REST API, server binary, telemetry | [api](./services/tensorus-api.md) |
| `tensorus-python` | PyO3 bindings + Python SDK | [python](./services/tensorus-python.md) |

The dependency graph is acyclic and flows downward: every crate may depend on
`tensorus-core`; `search` depends on `index` and `compute`; `api` depends on
`storage`, `compute`, `index`, `search`, and `ai` (the engines are wired into the
running server, not merely test-only); nothing depends on `api` except the server
binary, and the Python bindings depend on `storage`/`compute`.

---

## 2. Design principles

1. **Tensor-native first.** Tensors are never opaque blobs. Their shape, dtype,
   and mathematical structure are preserved and indexed. The
   [`TensorDescriptor`](./data-model.md#tensordescriptor) is the queryable
   "fingerprint" carried through the whole system.
2. **Trait-decoupled layers.** Storage, indexing, and search are defined as
   traits in `tensorus-core` ([`Storage`], [`Index`], [`VectorIndex`],
   [`SearchEngine`]). Concrete engines implement them, so a component can be
   swapped without touching callers.
3. **Async I/O, sync compute.** Storage and the API are `async` (Tokio). Pure
   CPU work (descriptors, linear algebra, index math) is synchronous and called
   from async contexts directly.
4. **Adaptive & self-tuning.** Compression picks a codec per tensor; tiering
   moves hot data into memory; the auto-optimizer proposes indexes from observed
   workload.
5. **Model-agnostic intelligence.** No LLM vendor lock-in: providers sit behind
   a trait and a router picks among them by cost/latency/locality.
6. **Observability by default.** The API emits structured `tracing` spans and a
   Prometheus `/metrics` endpoint with a latency histogram.
7. **Safety.** Every crate is `#![forbid(unsafe_code)]`. Network endpoints
   require an API key by default.

---

## 3. Data flow for key operations

### 3.1 Insert a tensor

```
client → tensorus-api (auth, rate limit)
       → TensorService::insert(dataset, &[f32], shape, metadata)
       → tensorus-compute::compute_descriptor(...)        # norms, rank, eigvals, booleans
       → tensorus-storage: FileStorage::insert(...)
            → append to WAL (write-ahead)                  # crash-safe ordering
            → append Put frame to dataset segment
            → insert into in-memory hot map (serves reads)
            → group-commit fsync every N writes
       → IndexManager::on_insert(...)                      # update secondary indexes
            → HNSW vector index (adopts dim from first tensor)
            → contraction index (shape-locked, ndim ≥ 2)
            → property index (per-field ALEX + boolean posting sets)
       → return (TensorId, TensorDescriptor)
```

The in-memory indexes are rebuilt from durable storage at startup via
`TensorService::recover()`, so a restart loses no queryability.

### 3.2 Property search (index-backed)

```
client → POST /datasets/{ds}/search/property {is_symmetric, norm range, rank, ...}
       → TensorService::search_by_property
       → PropertyIndex: numeric predicates → per-field ALEX range/point lookup;
                        boolean predicates → posting sets; intersect candidates
       → verify each candidate exactly against its descriptor
       → matching (id, descriptor) list   # no full storage scan
```

### 3.2.1 Vector similarity search

```
client → POST /datasets/{ds}/search/similar {vector, k, metric?}
       → TensorService::search_similar
       → DatasetIndexes: HNSW search (metric fixed at dataset creation)
            → dimension checked against the dataset's adopted dimension (400 on mismatch)
       → hydrate descriptor + metadata, map distance → similarity score
       → ranked SimilarHit list (deleted tensors excluded via HNSW tombstones)
```

### 3.3 Structural similarity (contraction)

```
query tensor → tensorus-search::tucker_sketch (per-mode SVD factors + core)
            → ContractionIndex: probe one HNSW per tensor mode (factor vectors)
            → union candidates → exact rerank by sketch_similarity
              (0.6 · mean Grassmannian mode distance + 0.4 · aligned core distance)
            → top-k by structural similarity
```

### 3.4 Neural query (NQL)

```
natural language → tensorus-ai::Nql::query
               → LlmRouter selects a provider, returns a QueryPlan (validated JSON)
               → optimize(plan): order predicates by selectivity, cap scans
               → execute(plan, QueryContext)  # QueryContext is implemented by the serving layer over storage+index
               → on execution error: feed the error back to the LLM and retry (self-correction)
               → rows + executed plan
```

### 3.5 Agentic workflow (ReAct)

```
task → ReActAgent::run
    → loop (≤ max_steps, ≤ timeout, ≤ budget):
        LlmRouter → Decision { thought, action | final_answer }
        if action: ToolRegistry.execute(tool, args) → observation → append to transcript
        if final_answer: done
    → AgentOutcome { Success | MaxStepsReached | TimedOut | BudgetExceeded }
```

---

## 4. Storage and durability model

`tensorus-storage` implements the [`Storage`] trait with a self-contained,
crash-safe engine:

- **Segment files** (`{data_dir}/datasets/{name}/segment.dat`) hold an
  append-only log of `Put`/`Del` *frames* — the durable record of committed data.
- A **write-ahead log** (WAL) records each operation (with a sequence number and
  the target dataset) *before* it is applied to a segment. A **checkpoint**
  sidecar records the highest sequence number folded into segments.
- An **in-memory hot map** serves all reads (the "hot tier").
- **Group commit:** writes go to the OS immediately (surviving process crashes
  via the page cache) and are fsynced periodically and on shutdown (power-loss
  durability after `flush()`).

**Recovery** on `open()`: load segments (truncating any torn trailing frame),
read the checkpoint, replay WAL entries with `seq > checkpoint` (idempotently,
deduped by id), fsync, then truncate the WAL.

See [storage](./services/tensorus-storage.md) for the frame format and the
exact protocol.

---

## 5. Documented substitutions

This build targets a **pure-Rust, CPU-only** toolchain
(`x86_64-pc-windows-gnu`) with no C compiler dependency for the default feature
set and no GPU. Following the implementation plan's guidance to "implement a
simpler alternative that maintains the same interface" when a dependency is
infeasible, the following substitutions are made. Each preserves the public
trait/contract so the originally specified backend can be dropped in later.

| Area | Plan specified | This build uses | Why | Where documented |
|------|----------------|-----------------|-----|------------------|
| Warm storage | Lance columnar format | Custom append-only segment + WAL engine | Lance pulls DataFusion/object_store/protoc; heavy on the gnu toolchain | `tensorus-storage` lib docs |
| Metadata index | Tantivy | In-house inverted index | Keep build light/fast across remaining tasks | `tensorus-index::metadata` docs |
| Dense lossless compression | Zstd (C) | DEFLATE (pure-Rust `miniz_oxide`) | Avoid a C build dependency | `tensorus-storage::compression` docs |
| In-memory ANN codes | Product Quantization | Scalar Quantization (SQ8) | Simpler; same role (approx distance) | `tensorus-index::diskann` docs |
| SSD reads | `io_uring` async I/O | Synchronous `std::fs` reads | `io_uring` is Linux-only; host is Windows | `tensorus-index::diskann` docs |
| gRPC server | tonic (default) | tonic behind optional `grpc` feature | tonic build needs `protoc` | `tensorus-api` lib docs |
| LLM transport | HTTPS client (reqwest/native-TLS) | Built-in `http://` client (`HttpTransport`) | Pure-Rust TLS needs a C/asm crypto backend; local Ollama/vLLM use http | `tensorus-api::llm_http` docs |
| OTEL export | OpenTelemetry/OTLP | `tracing` to stdout + Prometheus `/metrics` | Avoid the heavy OTel tree; OTLP is additive | `tensorus-api::telemetry` docs |
| GPU paths | cuTENSOR/cuSOLVER/CUDA | CPU (`nalgebra`/`ndarray`) with `// TODO: GPU` markers | No GPU in environment | throughout compute/index/search |

See [BENCHMARKS.md](../BENCHMARKS.md) for where these choices meet or trail the
plan's performance targets.

---

## 6. Build, test, and toolchain

- **Toolchain:** `stable-x86_64-pc-windows-gnu` with self-contained `rust-lld`
  linking plus a MinGW-w64 `dlltool`/`gcc` for crates that need it.
- **Workspace:** a Cargo workspace of 8 crates; shared dependency versions are
  pinned in the root `[workspace.dependencies]`.
- **Commands:**
  ```bash
  cargo build  --workspace
  cargo test   --workspace
  cargo clippy --workspace --all-targets -- -D warnings
  cargo bench  -p tensorus-api
  cargo doc    --workspace --no-deps
  ```
- The Python extension is built separately with `maturin` (see
  [python](./services/tensorus-python.md)).

[`Storage`]: ./services/tensorus-core.md#storage
[`Index`]: ./services/tensorus-core.md#index
[`VectorIndex`]: ./services/tensorus-core.md#vectorindex
[`SearchEngine`]: ./services/tensorus-core.md#searchengine

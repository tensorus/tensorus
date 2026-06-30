# Benchmarks — measured vs. target

Measured on the development machine (Windows, `x86_64-pc-windows-gnu`, `--release`,
CPU-only, pure-Rust default features). Numbers are indicative single-run figures
from this build's perf tests and Criterion benches, not a controlled benchmark
lab. They are reported honestly, including where a CPU-only/pure-Rust choice
trails a target that assumes GPU/LAPACK/columnar backends.

## Summary

| Area | Metric | Target | Measured | Verdict |
|------|--------|--------|----------|---------|
| Storage | insert 10k × (128,) f32 | <2 s (≈ >100K/s headline) | ~252 ms (~40K/s) | meets task bound; below 100K/s headline |
| Storage | get by id | <0.1 ms | in-memory hot-map lookup (sub-µs) | meets |
| PGM index | point / narrow-range (10M keys) | <500 ns | ~308 ns | ✅ |
| PGM index | range query (10M keys) | <10 µs | ~299 ns | ✅ |
| ALEX index | insert throughput (1M random) | >500K/s | ~2.54M/s | ✅ |
| Metadata index | search over 100K docs | <5 ms | ~604 µs | ✅ |
| HNSW | search (10k × 128-d, k=10) | <1 ms | ~0.3–0.6 ms | ✅ |
| HNSW | recall@10 | ≥0.95 | ≥0.95 (low-dim/clustered); ~0.83 (uniform-random 128-d) | ✅ on structured data |
| HNSW | insert throughput | >50K/s | ~1.6K/s (ef_construction=200) | ❌ (scalar kernel; see notes) |
| DiskANN/Vamana | recall@10 (synthetic) | ≥0.90 | 1.0 (2,000 vectors) | ✅ |
| Contraction | Tucker sketch (64,64,3) | <10 ms | ~4.2 ms | ✅ |
| Compression | dense f32 ratio | ≥4× | SQ8 ~4× | ✅ |
| Compression | sparse ratio | ≥8× | COO 17.3× @95% sparsity (lossless) | ✅ at high sparsity |
| Compression | decompression throughput | >2 GB/s | SQ8 ~3.0 GB/s | ✅ |
| Compression | low-rank ratio | — | 31× (rank-1 64×64), DEFLATE 210× (structured) | — |
| Descriptor | compute (256×256) | <10 ms | ~65 ms | ❌ (general eigensolver; see notes) |
| LLM router | routing/fallback overhead | <5 ms | <5 ms (mock) | ✅ |
| Tiering | hot-tier hit rate (skewed) | >90% | ~96% | ✅ |

## Notes on the two misses

- **HNSW insert throughput (~1.6K/s vs 50K/s).** The implementation is correct
  and full-featured (cosine/L2/dot, dynamic insert/delete, neighbor-selection
  heuristic) but the distance kernel is **scalar** and `ef_construction=200`
  explores many candidates. Closing the gap needs SIMD (AVX-512/NEON) distance
  and a flat neighbor layout — marked `// TODO: GPU acceleration` / SIMD in
  `hnsw.rs`. Search latency and recall (on structured data) already meet target.
- **Descriptor (256×256) ~65 ms vs <10 ms.** Dominated by the **general
  (non-symmetric) eigendecomposition**; `nalgebra`'s pure-Rust QR eigensolver is
  slower than the LAPACK/cuSOLVER the target assumes. Correctness (the hard
  requirement, within 1e-6 of v0.1) is met; symmetric and norm-only paths are
  much faster.

## Scale caveats

Targets quoted at very large scale (e.g. PGM at 10M keys, vector search at 1M–1B
vectors, 10M-tensor tiering, SIFT1M/SIFT1B recall, a live K8s cluster) were
exercised at the largest scale feasible in the dev environment:
- PGM range/point latency was measured at **10M** keys.
- ALEX insert throughput at **1M** keys.
- Metadata search at **100K** documents.
- HNSW at **10K × 128-d**; DiskANN recall at **2,000** vectors (page-aligned
  on-disk layout verified). Billion-scale SSD runs and `io_uring` were not run
  (Windows host; `io_uring` is Linux-only).

## How to reproduce

```bash
# Criterion micro-benchmarks (PGM range, HNSW search, Tucker sketch, storage insert)
cargo bench -p tensorus-api

# Larger perf checks are #[ignore]d unit tests; run individually in release:
cargo test -p tensorus-index   --release pgm_10m_latency            -- --ignored --nocapture
cargo test -p tensorus-index   --release alex_insert_throughput     -- --ignored --nocapture
cargo test -p tensorus-index   --release metadata_100k_latency      -- --ignored --nocapture
cargo test -p tensorus-index   --release hnsw_50k_benchmark         -- --ignored --nocapture
cargo test -p tensorus-storage --release sq8_decompression_throughput -- --ignored --nocapture
cargo test -p tensorus-storage --release insert_10k_throughput      -- --release
cargo test -p tensorus-compute --release descriptor_256x256_timing  -- --nocapture
cargo test -p tensorus-search  --release sketch_timing_64x64x3      -- --ignored --nocapture
```

## Test inventory

The workspace has ~155+ passing unit/integration tests plus the `#[ignore]`d
performance tests above:

| Crate | Tests |
|-------|-------|
| `tensorus-core` | 10 |
| `tensorus-compute` | 10 |
| `tensorus-storage` | 25 (+ ignored benches) |
| `tensorus-index` | 24 (+ ignored benches) |
| `tensorus-search` | 6 |
| `tensorus-ai` | 27 |
| `tensorus-api` | 25 unit + 32 integration |

Run everything with `cargo test --workspace`. The `tensorus-api` suite covers the
full serving surface: index-backed property search, HNSW vector search (ranking +
dimension-mismatch handling), contraction search, index recovery after a
simulated restart, NQL `/query` and ReAct `/agent` over a mock LLM, an
`HttpTransport` round-trip, **multi-tenant isolation, RBAC, quotas, and
snapshot/restore**.

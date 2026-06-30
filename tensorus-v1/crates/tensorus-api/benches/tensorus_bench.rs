//! Criterion micro-benchmarks for the core hot paths: storage insert, storage
//! read, learned-index range query, HNSW vector search, and tensor contraction
//! sketching. Run with `cargo bench -p tensorus-api`.

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use tensorus_core::types::{DType, Shape, TensorDescriptor, TensorId};
use tensorus_core::Storage;
use tensorus_index::{Hnsw, Metric, PgmIndex};
use tensorus_search::tucker_sketch;
use tensorus_storage::FileStorage;

fn bench_pgm_range(c: &mut Criterion) {
    let entries: Vec<(f64, u64)> = (0..1_000_000u64).map(|i| (i as f64, i)).collect();
    let idx = PgmIndex::build(entries);
    c.bench_function("pgm_range_count_1M", |b| {
        b.iter(|| black_box(idx.range_count(black_box(100_000.0), black_box(200_000.0))))
    });
}

fn bench_hnsw_search(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(1);
    let dim = 64;
    let hnsw = Hnsw::with_metric(Metric::L2);
    for _ in 0..5_000 {
        let v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        hnsw.insert(TensorId::new(), &v);
    }
    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    c.bench_function("hnsw_search_5k_64d_k10", |b| {
        b.iter(|| black_box(hnsw.search(black_box(&query), 10)))
    });
}

fn bench_contraction_sketch(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(2);
    let shape = [16u64, 16, 3];
    let data: Vec<f32> = (0..16 * 16 * 3).map(|_| rng.gen_range(-1.0..1.0)).collect();
    c.bench_function("tucker_sketch_16x16x3", |b| {
        b.iter(|| black_box(tucker_sketch(black_box(&data), &shape, 8)))
    });
}

fn bench_storage_insert(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let tmp = tempfile::TempDir::new().unwrap();
    let storage =
        Arc::new(FileStorage::open(tmp.path().join("data"), tmp.path().join("wal")).unwrap());
    rt.block_on(storage.create_dataset("bench")).unwrap();
    let payload = vec![0u8; 128 * 4];
    let descriptor =
        TensorDescriptor::empty(TensorId::new(), Shape::new(vec![128]), DType::Float32);

    c.bench_function("storage_insert_128f32", |b| {
        b.iter(|| {
            rt.block_on(storage.insert(
                "bench",
                black_box(&payload),
                descriptor.clone(),
                serde_json::Value::Null,
            ))
            .unwrap()
        })
    });
}

criterion_group!(
    benches,
    bench_pgm_range,
    bench_hnsw_search,
    bench_contraction_sketch,
    bench_storage_insert
);
criterion_main!(benches);

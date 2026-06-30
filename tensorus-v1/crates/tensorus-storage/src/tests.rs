//! Storage engine tests: CRUD, throughput, durability, and crash recovery.

use super::*;
use crate::format::{Frame, StoredRecord};
use crate::wal::Wal;
use serde_json::json;
use tempfile::TempDir;
use tensorus_core::types::{DType, Shape, TensorDescriptor, TensorId};
use tensorus_core::Storage;

fn desc() -> TensorDescriptor {
    TensorDescriptor::empty(TensorId::new(), Shape::new(vec![2, 2]), DType::Float32)
}

fn dirs(t: &TempDir) -> (std::path::PathBuf, std::path::PathBuf) {
    (t.path().join("data"), t.path().join("wal"))
}

#[tokio::test]
async fn insert_get_roundtrip() {
    let t = TempDir::new().unwrap();
    let (d, w) = dirs(&t);
    let store = FileStorage::open(&d, &w).unwrap();
    store.create_dataset("weights").await.unwrap();

    let payload = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let id = store
        .insert("weights", &payload, desc(), json!({"name": "layer1"}))
        .await
        .unwrap();

    let rec = store.get("weights", id).await.unwrap();
    assert_eq!(rec.data, payload);
    assert_eq!(rec.metadata["name"], "layer1");
    assert_eq!(rec.descriptor.tensor_id, id);
    assert_eq!(rec.version, 1);
}

#[tokio::test]
async fn get_missing_is_not_found() {
    let t = TempDir::new().unwrap();
    let (d, w) = dirs(&t);
    let store = FileStorage::open(&d, &w).unwrap();
    store.create_dataset("d").await.unwrap();
    assert!(store.get("d", TensorId::new()).await.is_err());
    assert!(store.get("missing_ds", TensorId::new()).await.is_err());
}

#[tokio::test]
async fn scan_paginates_in_insertion_order() {
    let t = TempDir::new().unwrap();
    let (d, w) = dirs(&t);
    let store = FileStorage::open(&d, &w).unwrap();
    let mut ids = Vec::new();
    for i in 0..10u8 {
        let id = store
            .insert("d", &[i; 4], desc(), json!({ "i": i }))
            .await
            .unwrap();
        ids.push(id);
    }
    let page = store.scan("d", 3, 2).await.unwrap();
    assert_eq!(page.len(), 3);
    assert_eq!(page[0].id, ids[2]);
    assert_eq!(page[2].id, ids[4]);
}

#[tokio::test]
async fn delete_removes_record() {
    let t = TempDir::new().unwrap();
    let (d, w) = dirs(&t);
    let store = FileStorage::open(&d, &w).unwrap();
    let id = store.insert("d", &[0; 4], desc(), json!({})).await.unwrap();
    assert!(store.get("d", id).await.is_ok());
    store.delete("d", id).await.unwrap();
    assert!(store.get("d", id).await.is_err());
    // Deleting again is idempotent.
    store.delete("d", id).await.unwrap();
    assert_eq!(store.scan("d", 100, 0).await.unwrap().len(), 0);
}

#[tokio::test]
async fn list_datasets_sorted() {
    let t = TempDir::new().unwrap();
    let (d, w) = dirs(&t);
    let store = FileStorage::open(&d, &w).unwrap();
    store.create_dataset("zeta").await.unwrap();
    store.create_dataset("alpha").await.unwrap();
    assert_eq!(
        store.list_datasets().await.unwrap(),
        vec!["alpha".to_string(), "zeta".to_string()]
    );
}

#[tokio::test]
async fn persists_across_reopen() {
    let t = TempDir::new().unwrap();
    let (d, w) = dirs(&t);
    let id;
    {
        let store = FileStorage::open(&d, &w).unwrap();
        id = store
            .insert("d", &[7; 8], desc(), json!({"k": "v"}))
            .await
            .unwrap();
        store.flush().unwrap();
    }
    {
        let store = FileStorage::open(&d, &w).unwrap();
        let rec = store.get("d", id).await.unwrap();
        assert_eq!(rec.data, vec![7u8; 8]);
        assert_eq!(rec.metadata["k"], "v");
    }
}

/// Simulates a crash *after* a WAL append but *before* the segment write and
/// checkpoint advance, then verifies recovery replays the entry.
#[tokio::test]
async fn wal_replay_recovers_unapplied_entry() {
    let t = TempDir::new().unwrap();
    let (d, w) = dirs(&t);

    let id1;
    {
        let store = FileStorage::open(&d, &w).unwrap();
        store.create_dataset("d").await.unwrap();
        id1 = store.insert("d", &[1; 8], desc(), json!({})).await.unwrap();
        store.flush().unwrap();
    }

    // Hand-craft a WAL entry that never reached the segment.
    let id2 = TensorId::new();
    {
        let mut wal = Wal::open(&w).unwrap();
        let checkpoint = wal.read_checkpoint().unwrap();
        let rec = StoredRecord {
            id: id2,
            created_at_us: super::now_us(),
            version: 1,
            descriptor: TensorDescriptor::empty(id2, Shape::new(vec![2]), DType::Float32),
            metadata: json!({"recovered": true}),
            data: vec![9u8; 8],
        };
        wal.append(checkpoint + 1, "d", &Frame::Put(rec)).unwrap();
        wal.sync().unwrap();
    }

    {
        let store = FileStorage::open(&d, &w).unwrap();
        // The replayed record is retrievable.
        let r2 = store.get("d", id2).await.unwrap();
        assert_eq!(r2.data, vec![9u8; 8]);
        assert_eq!(r2.metadata["recovered"], true);
        // The originally committed record survived too.
        assert!(store.get("d", id1).await.is_ok());
        assert_eq!(store.scan("d", 100, 0).await.unwrap().len(), 2);
    }
}

#[test]
fn scan_segment_truncates_torn_tail() {
    let rec = StoredRecord {
        id: TensorId::new(),
        created_at_us: 0,
        version: 1,
        descriptor: TensorDescriptor::empty(TensorId::new(), Shape::new(vec![1]), DType::Float32),
        metadata: json!({}),
        data: vec![0u8; 4],
    };
    let payload = crate::format::encode_frame(&Frame::Put(rec)).unwrap();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&payload);
    let good_len = bytes.len();
    // Torn trailing frame: declares 100 bytes but only 3 follow.
    bytes.extend_from_slice(&100u32.to_le_bytes());
    bytes.extend_from_slice(&[1, 2, 3]);

    let scan = crate::segment::scan_segment(&bytes);
    assert_eq!(scan.frames.len(), 1);
    assert_eq!(scan.valid_len as usize, good_len);
}

#[tokio::test]
async fn insert_10k_throughput() {
    let t = TempDir::new().unwrap();
    let (d, w) = dirs(&t);
    let store = FileStorage::open(&d, &w).unwrap();
    store.create_dataset("bench").await.unwrap();

    // Shape (128,) float32 = 512 bytes per tensor.
    let payload = vec![0u8; 128 * 4];
    let descriptor =
        TensorDescriptor::empty(TensorId::new(), Shape::new(vec![128]), DType::Float32);

    let start = std::time::Instant::now();
    for _ in 0..10_000 {
        store
            .insert("bench", &payload, descriptor.clone(), json!({}))
            .await
            .unwrap();
    }
    store.flush().unwrap();
    let elapsed = start.elapsed();
    println!("inserted 10000 tensors in {:?}", elapsed);

    assert_eq!(store.scan("bench", 1, 0).await.unwrap().len(), 1);
    // Generous bound to stay reliable in debug/CI; release numbers are reported
    // separately. Target from the plan is < 2s.
    assert!(
        elapsed.as_secs() < 10,
        "10k insert took too long: {elapsed:?}"
    );
}

#[tokio::test]
async fn snapshot_then_restore_preserves_data() {
    let t = TempDir::new().unwrap();
    let (d, w) = dirs(&t);
    let id;
    {
        let store = FileStorage::open(&d, &w).unwrap();
        store.create_dataset("weights").await.unwrap();
        id = store
            .insert("weights", &[1u8; 16], desc(), json!({"name": "a"}))
            .await
            .unwrap();
        store
            .insert("weights", &[2u8; 16], desc(), json!({"name": "b"}))
            .await
            .unwrap();

        // Snapshot to a separate directory.
        let snap = t.path().join("snap");
        store.snapshot(&snap).unwrap();

        // Open the snapshot with a fresh WAL dir; data is intact.
        let restored = FileStorage::open(&snap, t.path().join("snap_wal")).unwrap();
        let rec = restored.get("weights", id).await.unwrap();
        assert_eq!(rec.data, vec![1u8; 16]);
        assert_eq!(rec.metadata["name"], "a");
        assert_eq!(restored.scan("weights", 100, 0).await.unwrap().len(), 2);
        assert_eq!(restored.list_datasets().await.unwrap(), vec!["weights"]);
    }
}

#[tokio::test]
async fn replication_changes_and_apply() {
    let t1 = TempDir::new().unwrap();
    let (d1, w1) = dirs(&t1);
    let primary = FileStorage::open(&d1, &w1).unwrap();
    let a = primary
        .insert("ds", &[1u8; 16], desc(), json!({"n": "a"}))
        .await
        .unwrap();
    let b = primary
        .insert("ds", &[2u8; 16], desc(), json!({"n": "b"}))
        .await
        .unwrap();
    primary.delete("ds", a).await.unwrap();

    // The change-log captures all three ops in order.
    let ops = primary.changes_since(0, 1000).unwrap();
    assert_eq!(ops.len(), 3);
    assert_eq!(
        primary.replication_head().unwrap(),
        ops.last().unwrap().seq()
    );

    // Apply them to a fresh replica; it converges to the same state.
    let t2 = TempDir::new().unwrap();
    let (d2, w2) = dirs(&t2);
    let replica = FileStorage::open(&d2, &w2).unwrap();
    for op in ops {
        replica.apply_replicated(op).unwrap();
    }
    assert!(replica.get("ds", b).await.is_ok());
    assert!(replica.get("ds", a).await.is_err()); // deleted on the primary
    assert_eq!(replica.scan("ds", 100, 0).await.unwrap().len(), 1);

    // Incremental catch-up: only the new op is returned past the head.
    let head = primary.replication_head().unwrap();
    let c = primary
        .insert("ds", &[3u8; 16], desc(), json!({"n": "c"}))
        .await
        .unwrap();
    let delta = primary.changes_since(head, 1000).unwrap();
    assert_eq!(delta.len(), 1);
    replica
        .apply_replicated(delta.into_iter().next().unwrap())
        .unwrap();
    assert!(replica.get("ds", c).await.is_ok());
}

#[tokio::test]
async fn replication_log_survives_reopen() {
    let t = TempDir::new().unwrap();
    let (d, w) = dirs(&t);
    {
        let store = FileStorage::open(&d, &w).unwrap();
        store
            .insert("ds", &[7u8; 16], desc(), json!({}))
            .await
            .unwrap();
    }
    // Reopen: the change-log is retained (not truncated like the WAL).
    let store = FileStorage::open(&d, &w).unwrap();
    assert_eq!(store.changes_since(0, 1000).unwrap().len(), 1);
}

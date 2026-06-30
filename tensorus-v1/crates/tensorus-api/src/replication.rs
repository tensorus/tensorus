//! Single-leader async replication helper.
//!
//! A **leader** records every committed operation in a durable change-log
//! ([`tensorus_storage::FileStorage::changes_since`]) and exposes it over REST
//! (`GET /replication/changes?since=N`). A **follower** runs a [`ReplicaSyncer`]:
//! it repeatedly fetches the leader's ops past its current head and applies them
//! (preserving ids and sequence), giving read-scaling and a warm standby with
//! bounded staleness.
//!
//! This module is transport-agnostic: the caller fetches a batch of [`ReplOp`]s
//! (e.g. via HTTP) and hands them to [`ReplicaSyncer::apply_batch`], which keeps
//! the local head in sync. Replication is **asynchronous and single-leader**
//! (no automatic failover or consensus).

use crate::service::TensorService;
use std::sync::Arc;
use tensorus_core::error::Result;
use tensorus_storage::ReplOp;

/// Applies replicated operations to a follower `TensorService`, tracking the
/// last applied change-log sequence (the "head").
pub struct ReplicaSyncer {
    replica: Arc<TensorService>,
    head: u64,
}

impl ReplicaSyncer {
    /// Create a syncer for `replica`, initialising the head from whatever the
    /// follower has already applied.
    pub fn new(replica: Arc<TensorService>) -> Result<Self> {
        let head = replica.replication_head()?;
        Ok(ReplicaSyncer { replica, head })
    }

    /// The last applied change-log sequence; pass this as `since` when fetching
    /// the next batch from the leader.
    pub fn head(&self) -> u64 {
        self.head
    }

    /// Apply a batch of operations (typically fetched from the leader's
    /// `/replication/changes`), advancing the head. Ops are applied in order;
    /// re-applying an already-seen op is harmless.
    pub async fn apply_batch(&mut self, ops: Vec<ReplOp>) -> Result<u64> {
        for op in ops {
            let seq = op.seq();
            self.replica.apply_replicated(op).await?;
            if seq > self.head {
                self.head = seq;
            }
        }
        Ok(self.head)
    }
}

//! Metadata index: full-text search plus structured filters over tensor
//! metadata and descriptor fields.
//!
//! ## Design note: Tantivy substitution
//!
//! The plan specifies [Tantivy](https://github.com/quickwit-oss/tantivy) (a Rust
//! Lucene). Tantivy is pure-Rust and would compile here, but it is a large
//! dependency that would substantially slow every build of this crate for the
//! remaining tasks. Per the plan's guidance on dependencies, this module
//! implements a self-contained inverted index behind a stable [`MetadataIndex`]
//! API offering the same capabilities the task requires: tokenized full-text
//! search over string fields, structured equality/range/text filters, combined
//! queries, and faceting. A Tantivy-backed implementation can replace it behind
//! the same API.

use parking_lot::RwLock;
use serde_json::{Map, Value};
use std::collections::HashMap;
use tensorus_core::types::TensorId;

/// Lowercase + split on non-alphanumeric boundaries.
fn tokenize(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// Collect tokens from every string value in a field map (used for the
/// document-level full-text field).
fn document_tokens(fields: &Map<String, Value>) -> Vec<String> {
    let mut tokens = Vec::new();
    for v in fields.values() {
        if let Value::String(s) = v {
            tokens.extend(tokenize(s));
        }
    }
    tokens
}

/// A document to index: a tensor's id, its dataset, and arbitrary metadata.
#[derive(Debug, Clone)]
pub struct Document {
    pub id: TensorId,
    pub dataset: String,
    pub fields: Map<String, Value>,
}

impl Document {
    /// Build a document, injecting `dataset` as a queryable field.
    pub fn new(id: TensorId, dataset: impl Into<String>, mut fields: Map<String, Value>) -> Self {
        let dataset = dataset.into();
        fields.insert("dataset".to_string(), Value::String(dataset.clone()));
        Document {
            id,
            dataset,
            fields,
        }
    }
}

/// A structured filter clause. Multiple clauses combine with AND.
#[derive(Debug, Clone)]
pub enum Filter {
    /// Field exactly equals a JSON value.
    Eq(String, Value),
    /// Numeric field lies in the inclusive range `[min, max]`.
    Range(String, f64, f64),
    /// A specific field's tokenized text contains `token`.
    TextMatch(String, String),
}

impl Filter {
    fn matches(&self, fields: &Map<String, Value>) -> bool {
        match self {
            Filter::Eq(field, value) => fields.get(field) == Some(value),
            Filter::Range(field, min, max) => fields
                .get(field)
                .and_then(|v| v.as_f64())
                .map(|x| x >= *min && x <= *max)
                .unwrap_or(false),
            Filter::TextMatch(field, token) => {
                let needle = token.to_lowercase();
                fields
                    .get(field)
                    .and_then(|v| v.as_str())
                    .map(|s| tokenize(s).contains(&needle))
                    .unwrap_or(false)
            }
        }
    }
}

/// A combined full-text + structured query.
#[derive(Debug, Clone, Default)]
pub struct MetadataQuery {
    /// Optional free-text query over all string fields.
    pub text: Option<String>,
    /// Structured filters, combined with AND.
    pub filters: Vec<Filter>,
    /// Maximum results to return (0 means unlimited).
    pub limit: usize,
}

impl MetadataQuery {
    /// A pure full-text query.
    pub fn text(q: impl Into<String>) -> Self {
        MetadataQuery {
            text: Some(q.into()),
            filters: Vec::new(),
            limit: 0,
        }
    }

    /// A pure structured-filter query.
    pub fn filtered(filters: Vec<Filter>) -> Self {
        MetadataQuery {
            text: None,
            filters,
            limit: 0,
        }
    }

    /// Set a result cap.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

struct StoredDoc {
    fields: Map<String, Value>,
}

#[derive(Default)]
struct Inner {
    docs: HashMap<TensorId, StoredDoc>,
    /// token -> (doc id -> term frequency).
    inverted: HashMap<String, HashMap<TensorId, u32>>,
}

/// A thread-safe metadata index.
#[derive(Default)]
pub struct MetadataIndex {
    inner: RwLock<Inner>,
}

impl MetadataIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.inner.read().docs.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert or replace a document.
    pub fn insert(&self, doc: Document) {
        let mut inner = self.inner.write();
        // Remove any prior version's postings first.
        Self::remove_postings(&mut inner, doc.id);
        for token in document_tokens(&doc.fields) {
            *inner
                .inverted
                .entry(token)
                .or_default()
                .entry(doc.id)
                .or_insert(0) += 1;
        }
        inner.docs.insert(doc.id, StoredDoc { fields: doc.fields });
    }

    /// Remove a document.
    pub fn delete(&self, id: TensorId) {
        let mut inner = self.inner.write();
        Self::remove_postings(&mut inner, id);
        inner.docs.remove(&id);
    }

    fn remove_postings(inner: &mut Inner, id: TensorId) {
        if let Some(doc) = inner.docs.get(&id) {
            let tokens: Vec<String> = document_tokens(&doc.fields);
            for token in tokens {
                if let Some(postings) = inner.inverted.get_mut(&token) {
                    postings.remove(&id);
                    if postings.is_empty() {
                        inner.inverted.remove(&token);
                    }
                }
            }
        }
    }

    /// Run a query, returning `(id, score)` ranked by relevance (score is the
    /// number of matched query tokens plus a small term-frequency bonus; it is
    /// `0.0` for pure structured queries).
    pub fn search(&self, query: &MetadataQuery) -> Vec<(TensorId, f32)> {
        let inner = self.inner.read();
        let has_filters = !query.filters.is_empty();
        let passes = |id: &TensorId| -> bool {
            inner
                .docs
                .get(id)
                .map(|d| query.filters.iter().all(|f| f.matches(&d.fields)))
                .unwrap_or(false)
        };

        let mut scored: Vec<(TensorId, f32)> = if let Some(text) = &query.text {
            let tokens = tokenize(text);
            let mut score_map: HashMap<TensorId, f32> = HashMap::new();
            for tok in &tokens {
                if let Some(postings) = inner.inverted.get(tok) {
                    for (id, tf) in postings {
                        let inc = 1.0 + (*tf as f32) * 0.01;
                        match score_map.entry(*id) {
                            std::collections::hash_map::Entry::Occupied(mut e) => {
                                // NEG_INFINITY marks a candidate that failed the filter.
                                if *e.get() > f32::NEG_INFINITY {
                                    *e.get_mut() += inc;
                                }
                            }
                            std::collections::hash_map::Entry::Vacant(e) => {
                                if !has_filters || passes(id) {
                                    e.insert(inc);
                                } else {
                                    e.insert(f32::NEG_INFINITY);
                                }
                            }
                        }
                    }
                }
            }
            score_map
                .into_iter()
                .filter(|(_, s)| *s > f32::NEG_INFINITY)
                .collect()
        } else {
            inner
                .docs
                .iter()
                .filter(|(_, d)| query.filters.iter().all(|f| f.matches(&d.fields)))
                .map(|(id, _)| (*id, 0.0))
                .collect()
        };

        let cmp = |a: &(TensorId, f32), b: &(TensorId, f32)| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.as_bytes().cmp(b.0.as_bytes()))
        };
        if query.limit > 0 && scored.len() > query.limit {
            // Partition so the top `limit` results are first, then sort only those.
            scored.select_nth_unstable_by(query.limit, cmp);
            scored.truncate(query.limit);
        }
        scored.sort_by(cmp);
        scored
    }

    /// Count documents grouped by the stringified value of `field` (faceting).
    pub fn facet_counts(&self, field: &str) -> HashMap<String, usize> {
        let inner = self.inner.read();
        let mut counts = HashMap::new();
        for doc in inner.docs.values() {
            if let Some(v) = doc.fields.get(field) {
                let key = match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                *counts.entry(key).or_insert(0) += 1;
            }
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn doc(dataset: &str, fields: Value) -> Document {
        let map = fields.as_object().unwrap().clone();
        Document::new(TensorId::new(), dataset, map)
    }

    #[test]
    fn full_text_search_ranks_relevant() {
        let idx = MetadataIndex::new();
        let a = doc("resnet50", json!({"name": "attention head weight matrix"}));
        let b = doc("resnet50", json!({"name": "convolution filter bias"}));
        let c = doc("resnet50", json!({"name": "attention output weight"}));
        idx.insert(a.clone());
        idx.insert(b.clone());
        idx.insert(c.clone());

        let res = idx.search(&MetadataQuery::text("attention weight"));
        // a (2 tokens) and c (2 tokens) match; b (0) excluded.
        let ids: Vec<TensorId> = res.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&a.id));
        assert!(ids.contains(&c.id));
        assert!(!ids.contains(&b.id));
    }

    #[test]
    fn structured_filter_eq_and_range() {
        let idx = MetadataIndex::new();
        let a = doc(
            "resnet50",
            json!({"name": "w1", "dtype": "float16", "norm": 5.0}),
        );
        let b = doc(
            "resnet50",
            json!({"name": "w2", "dtype": "float32", "norm": 12.0}),
        );
        let c = doc(
            "bert",
            json!({"name": "w3", "dtype": "float16", "norm": 8.0}),
        );
        idx.insert(a.clone());
        idx.insert(b.clone());
        idx.insert(c.clone());

        let res = idx.search(&MetadataQuery::filtered(vec![
            Filter::Eq("dataset".into(), json!("resnet50")),
            Filter::Eq("dtype".into(), json!("float16")),
        ]));
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, a.id);

        let res = idx.search(&MetadataQuery::filtered(vec![Filter::Range(
            "norm".into(),
            6.0,
            20.0,
        )]));
        let ids: Vec<TensorId> = res.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&b.id) && ids.contains(&c.id) && !ids.contains(&a.id));
    }

    #[test]
    fn combined_text_and_filter() {
        let idx = MetadataIndex::new();
        let a = doc(
            "resnet50",
            json!({"name": "attention weight", "dtype": "float16"}),
        );
        let b = doc(
            "bert",
            json!({"name": "attention weight", "dtype": "float16"}),
        );
        idx.insert(a.clone());
        idx.insert(b.clone());

        let q = MetadataQuery {
            text: Some("attention".into()),
            filters: vec![Filter::Eq("dataset".into(), json!("resnet50"))],
            limit: 0,
        };
        let res = idx.search(&q);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, a.id);
    }

    #[test]
    fn facet_counts_group_by_field() {
        let idx = MetadataIndex::new();
        idx.insert(doc("resnet50", json!({"layer": "conv1"})));
        idx.insert(doc("resnet50", json!({"layer": "conv1"})));
        idx.insert(doc("resnet50", json!({"layer": "fc"})));
        let facets = idx.facet_counts("layer");
        assert_eq!(facets.get("conv1"), Some(&2));
        assert_eq!(facets.get("fc"), Some(&1));
    }

    #[test]
    fn delete_removes_from_text_and_filters() {
        let idx = MetadataIndex::new();
        let a = doc("d", json!({"name": "unique token zebra"}));
        idx.insert(a.clone());
        assert_eq!(idx.search(&MetadataQuery::text("zebra")).len(), 1);
        idx.delete(a.id);
        assert_eq!(idx.search(&MetadataQuery::text("zebra")).len(), 0);
        assert!(idx.is_empty());
    }

    #[test]
    fn limit_truncates_results() {
        let idx = MetadataIndex::new();
        for i in 0..10 {
            idx.insert(doc("d", json!({"name": format!("token{i} common")})));
        }
        let res = idx.search(&MetadataQuery::text("common").with_limit(3));
        assert_eq!(res.len(), 3);
    }

    #[test]
    #[ignore]
    fn metadata_100k_latency() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let idx = MetadataIndex::new();
        let mut rng = StdRng::seed_from_u64(5);
        // A realistic high-cardinality vocabulary (tensor/layer names) so that
        // individual terms are selective, as in real metadata. Latency of this
        // inverted index scales with the size of the matched posting lists; for
        // pathologically common terms a production engine (Tantivy/BM25 with
        // block-max WAND) would be needed.
        let vocab: Vec<String> = (0..200).map(|i| format!("term{i}")).collect();
        for i in 0..100_000 {
            let name = format!(
                "{} {} {} attention weight layer{}",
                vocab[rng.gen_range(0..vocab.len())],
                vocab[rng.gen_range(0..vocab.len())],
                vocab[rng.gen_range(0..vocab.len())],
                i % 2000
            );
            let dataset = if i % 2 == 0 { "resnet50" } else { "bert" };
            idx.insert(doc(dataset, json!({ "name": name })));
        }
        // A moderately selective query (term-pair + dataset filter).
        let start = std::time::Instant::now();
        let mut total = 0;
        for _ in 0..100 {
            let res = idx.search(&MetadataQuery {
                text: Some("term7 term42 layer123".into()),
                filters: vec![Filter::Eq("dataset".into(), json!("resnet50"))],
                limit: 10,
            });
            total += res.len();
        }
        let per = start.elapsed().as_micros() as f64 / 100.0;
        println!("metadata search over 100k docs avg {per:.1} us (total={total})");
        assert!(per < 5000.0, "search too slow: {per} us");
    }
}

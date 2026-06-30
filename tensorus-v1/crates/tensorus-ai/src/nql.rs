//! Neural Query Language (NQL v2): natural language -> `QueryPlan` IR ->
//! optimization -> execution, with LLM-driven parsing and self-correction.
//!
//! The LLM emits a structured [`QueryPlan`] (validated by deserialization); the
//! plan is optimized with rule-based rewrites and executed against a
//! [`QueryContext`] (which the storage/index layer implements). If execution
//! fails, the error is fed back to the LLM to revise the plan.

use crate::llm::{InferenceConfig, LlmError, Message};
use crate::router::LlmRouter;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tensorus_core::types::TensorId;

/// Comparison operator for a property predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cmp {
    Eq,
    Gt,
    Lt,
    Ge,
    Le,
}

/// A single property predicate, e.g. `frobenius_norm > 5` or `is_symmetric = true`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Predicate {
    pub field: String,
    pub cmp: Cmp,
    pub value: serde_json::Value,
}

impl Predicate {
    /// Boolean/equality predicates are the most selective and should run first.
    fn selectivity_rank(&self) -> u8 {
        match self.cmp {
            Cmp::Eq => 0,
            _ => 1,
        }
    }
}

/// The query plan intermediate representation produced by the parser.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum QueryPlan {
    /// Full scan of a dataset.
    Scan {
        dataset: String,
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Property/index lookup with AND-combined predicates.
    IndexLookup {
        dataset: String,
        predicates: Vec<Predicate>,
        #[serde(default)]
        limit: Option<usize>,
    },
    /// Nearest-neighbour vector search.
    VectorSearch {
        dataset: String,
        k: usize,
        #[serde(default)]
        query: Option<Vec<f32>>,
    },
    /// Aggregation over a field.
    Aggregate {
        dataset: String,
        function: String,
        field: String,
    },
}

impl QueryPlan {
    /// The dataset this plan targets.
    pub fn dataset(&self) -> &str {
        match self {
            QueryPlan::Scan { dataset, .. }
            | QueryPlan::IndexLookup { dataset, .. }
            | QueryPlan::VectorSearch { dataset, .. }
            | QueryPlan::Aggregate { dataset, .. } => dataset,
        }
    }
}

/// Rule-based plan optimization: order predicates by selectivity (equality
/// first) so the most selective index probes run first; cap unbounded scans.
pub fn optimize(mut plan: QueryPlan, default_scan_limit: usize) -> QueryPlan {
    match &mut plan {
        QueryPlan::IndexLookup { predicates, .. } => {
            predicates.sort_by_key(|p| p.selectivity_rank());
        }
        QueryPlan::Scan { limit, .. } if limit.is_none() => {
            *limit = Some(default_scan_limit);
        }
        _ => {}
    }
    plan
}

/// A row returned by query execution.
#[derive(Debug, Clone, Serialize)]
pub struct QueryRow {
    pub id: TensorId,
    pub score: f64,
    pub metadata: serde_json::Value,
}

/// The storage/index backend that executes plans. Implemented by the serving
/// layer; tests use an in-memory mock.
#[async_trait]
pub trait QueryContext: Send + Sync {
    async fn scan(&self, dataset: &str, limit: usize) -> Result<Vec<QueryRow>, String>;
    async fn property_search(
        &self,
        dataset: &str,
        predicates: &[Predicate],
        limit: usize,
    ) -> Result<Vec<QueryRow>, String>;
    async fn vector_search(
        &self,
        dataset: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<QueryRow>, String>;
    async fn aggregate(
        &self,
        dataset: &str,
        function: &str,
        field: &str,
    ) -> Result<Vec<QueryRow>, String>;
}

/// Execute an optimized plan against a context.
pub async fn execute<C: QueryContext>(plan: &QueryPlan, ctx: &C) -> Result<Vec<QueryRow>, String> {
    match plan {
        QueryPlan::Scan { dataset, limit } => ctx.scan(dataset, limit.unwrap_or(100)).await,
        QueryPlan::IndexLookup {
            dataset,
            predicates,
            limit,
        } => {
            ctx.property_search(dataset, predicates, limit.unwrap_or(100))
                .await
        }
        QueryPlan::VectorSearch { dataset, k, query } => {
            let q = query
                .as_ref()
                .ok_or_else(|| "vector_search requires a query vector".to_string())?;
            ctx.vector_search(dataset, q, *k).await
        }
        QueryPlan::Aggregate {
            dataset,
            function,
            field,
        } => ctx.aggregate(dataset, function, field).await,
    }
}

const SCHEMA_HINT: &str = r#"{"op": one of "scan"|"index_lookup"|"vector_search"|"aggregate", "dataset": string, and op-specific fields: scan{limit?}, index_lookup{predicates:[{field,cmp:"eq"|"gt"|"lt"|"ge"|"le",value}],limit?}, vector_search{k,query?}, aggregate{function,field}}"#;

fn few_shot() -> Vec<Message> {
    vec![
        Message::user("get all tensors from dataset weights"),
        Message::assistant(r#"{"op":"scan","dataset":"weights","limit":100}"#),
        Message::user("find symmetric matrices with norm > 5 in dataset layers"),
        Message::assistant(
            r#"{"op":"index_lookup","dataset":"layers","predicates":[{"field":"is_symmetric","cmp":"eq","value":true},{"field":"frobenius_norm","cmp":"gt","value":5.0}]}"#,
        ),
    ]
}

/// The result of an NQL query.
#[derive(Debug, Clone)]
pub struct NqlResult {
    pub plan: QueryPlan,
    pub rows: Vec<QueryRow>,
    pub plan_json: String,
}

/// Neural query engine over an [`LlmRouter`].
pub struct Nql {
    router: LlmRouter,
    default_scan_limit: usize,
}

impl Nql {
    pub fn new(router: LlmRouter) -> Self {
        Nql {
            router,
            default_scan_limit: 100,
        }
    }

    /// Parse a natural-language query into a (validated) plan.
    pub async fn parse(&self, nl: &str) -> Result<QueryPlan, LlmError> {
        let mut messages = few_shot();
        messages.push(Message::user(nl.to_string()));
        self.router
            .complete_structured::<QueryPlan>(
                &messages,
                SCHEMA_HINT,
                &InferenceConfig::default(),
                3,
            )
            .await
    }

    /// Parse, optimize, and execute a natural-language query, self-correcting on
    /// execution failures by re-prompting the LLM with the error.
    pub async fn query<C: QueryContext>(
        &self,
        nl: &str,
        ctx: &C,
        max_correction: usize,
    ) -> Result<NqlResult, String> {
        let mut feedback: Option<String> = None;
        for _ in 0..=max_correction {
            let mut messages = few_shot();
            messages.push(Message::user(nl.to_string()));
            if let Some(err) = &feedback {
                messages.push(Message::user(format!(
                    "The previous plan failed during execution with: {err}. Produce a corrected plan."
                )));
            }
            let plan = self
                .router
                .complete_structured::<QueryPlan>(
                    &messages,
                    SCHEMA_HINT,
                    &InferenceConfig::default(),
                    3,
                )
                .await
                .map_err(|e| e.to_string())?;
            let plan = optimize(plan, self.default_scan_limit);
            match execute(&plan, ctx).await {
                Ok(rows) => {
                    let plan_json = serde_json::to_string(&plan).unwrap_or_default();
                    return Ok(NqlResult {
                        plan,
                        rows,
                        plan_json,
                    });
                }
                Err(e) => feedback = Some(e),
            }
        }
        Err(format!(
            "query failed after {} correction attempts: {}",
            max_correction,
            feedback.unwrap_or_default()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::MockProvider;
    use crate::router::RoutingStrategy;
    use std::collections::HashMap;
    use std::sync::Arc;

    struct Row {
        id: TensorId,
        nums: HashMap<String, f64>,
        bools: HashMap<String, bool>,
    }

    struct MockCtx {
        datasets: HashMap<String, Vec<Row>>,
    }

    impl MockCtx {
        fn eval(row: &Row, p: &Predicate) -> bool {
            if let Some(b) = p.value.as_bool() {
                return row.bools.get(&p.field).copied().unwrap_or(false) == b;
            }
            if let Some(target) = p.value.as_f64() {
                if let Some(&v) = row.nums.get(&p.field) {
                    return match p.cmp {
                        Cmp::Eq => (v - target).abs() < 1e-9,
                        Cmp::Gt => v > target,
                        Cmp::Lt => v < target,
                        Cmp::Ge => v >= target,
                        Cmp::Le => v <= target,
                    };
                }
            }
            false
        }
    }

    #[async_trait]
    impl QueryContext for MockCtx {
        async fn scan(&self, dataset: &str, limit: usize) -> Result<Vec<QueryRow>, String> {
            let rows = self
                .datasets
                .get(dataset)
                .ok_or_else(|| format!("unknown dataset '{dataset}'"))?;
            Ok(rows
                .iter()
                .take(limit)
                .map(|r| QueryRow {
                    id: r.id,
                    score: 1.0,
                    metadata: serde_json::Value::Null,
                })
                .collect())
        }

        async fn property_search(
            &self,
            dataset: &str,
            predicates: &[Predicate],
            limit: usize,
        ) -> Result<Vec<QueryRow>, String> {
            let rows = self
                .datasets
                .get(dataset)
                .ok_or_else(|| format!("unknown dataset '{dataset}'"))?;
            Ok(rows
                .iter()
                .filter(|r| predicates.iter().all(|p| MockCtx::eval(r, p)))
                .take(limit)
                .map(|r| QueryRow {
                    id: r.id,
                    score: 1.0,
                    metadata: serde_json::Value::Null,
                })
                .collect())
        }

        async fn vector_search(
            &self,
            _dataset: &str,
            _query: &[f32],
            _k: usize,
        ) -> Result<Vec<QueryRow>, String> {
            Ok(vec![])
        }

        async fn aggregate(
            &self,
            _dataset: &str,
            _function: &str,
            _field: &str,
        ) -> Result<Vec<QueryRow>, String> {
            Ok(vec![])
        }
    }

    fn nql_with(script: Vec<&str>) -> Nql {
        let provider = MockProvider::scripted(
            "mock",
            script.into_iter().map(|s| Ok(s.to_string())).collect(),
        );
        let router = LlmRouter::new(
            vec![Arc::new(provider)],
            RoutingStrategy::Fixed { provider_index: 0 },
            0.0,
        );
        Nql::new(router)
    }

    fn ctx() -> MockCtx {
        let mut datasets = HashMap::new();
        // Three matrices: two symmetric (one with norm 8, one with norm 3), one asymmetric norm 10.
        let layers = vec![
            Row {
                id: TensorId::new(),
                nums: HashMap::from([("frobenius_norm".to_string(), 8.0)]),
                bools: HashMap::from([("is_symmetric".to_string(), true)]),
            },
            Row {
                id: TensorId::new(),
                nums: HashMap::from([("frobenius_norm".to_string(), 3.0)]),
                bools: HashMap::from([("is_symmetric".to_string(), true)]),
            },
            Row {
                id: TensorId::new(),
                nums: HashMap::from([("frobenius_norm".to_string(), 10.0)]),
                bools: HashMap::from([("is_symmetric".to_string(), false)]),
            },
        ];
        datasets.insert("layers".to_string(), layers);
        datasets.insert(
            "weights".to_string(),
            (0..5)
                .map(|_| Row {
                    id: TensorId::new(),
                    nums: HashMap::new(),
                    bools: HashMap::new(),
                })
                .collect(),
        );
        MockCtx { datasets }
    }

    #[tokio::test]
    async fn parses_scan() {
        let nql = nql_with(vec![r#"{"op":"scan","dataset":"weights","limit":50}"#]);
        let plan = nql
            .parse("get all tensors from dataset weights")
            .await
            .unwrap();
        assert_eq!(
            plan,
            QueryPlan::Scan {
                dataset: "weights".into(),
                limit: Some(50)
            }
        );
    }

    #[tokio::test]
    async fn scan_executes() {
        let nql = nql_with(vec![r#"{"op":"scan","dataset":"weights"}"#]);
        let res = nql
            .query("get all tensors from dataset weights", &ctx(), 2)
            .await
            .unwrap();
        assert_eq!(res.rows.len(), 5);
        assert!(matches!(res.plan, QueryPlan::Scan { .. }));
    }

    #[tokio::test]
    async fn index_lookup_symmetric_norm_gt_5() {
        let nql = nql_with(vec![
            r#"{"op":"index_lookup","dataset":"layers","predicates":[{"field":"is_symmetric","cmp":"eq","value":true},{"field":"frobenius_norm","cmp":"gt","value":5.0}]}"#,
        ]);
        let res = nql
            .query("find symmetric matrices with norm > 5", &ctx(), 2)
            .await
            .unwrap();
        // Only the symmetric matrix with norm 8 qualifies.
        assert_eq!(res.rows.len(), 1);
    }

    #[tokio::test]
    async fn self_correction_recovers_from_bad_dataset() {
        // First plan references a misspelled dataset (execution error), second is correct.
        let nql = nql_with(vec![
            r#"{"op":"scan","dataset":"wieghts"}"#,
            r#"{"op":"scan","dataset":"weights"}"#,
        ]);
        let res = nql.query("get all from weights", &ctx(), 2).await.unwrap();
        assert_eq!(res.rows.len(), 5);
        assert_eq!(res.plan.dataset(), "weights");
    }

    #[tokio::test]
    async fn optimizer_orders_predicates() {
        let plan = QueryPlan::IndexLookup {
            dataset: "d".into(),
            predicates: vec![
                Predicate {
                    field: "norm".into(),
                    cmp: Cmp::Gt,
                    value: serde_json::json!(5.0),
                },
                Predicate {
                    field: "is_symmetric".into(),
                    cmp: Cmp::Eq,
                    value: serde_json::json!(true),
                },
            ],
            limit: None,
        };
        let opt = optimize(plan, 100);
        if let QueryPlan::IndexLookup { predicates, .. } = opt {
            // Equality predicate moved first.
            assert_eq!(predicates[0].cmp, Cmp::Eq);
        } else {
            panic!("expected index lookup");
        }
    }

    #[tokio::test]
    async fn scan_gets_default_limit() {
        let plan = QueryPlan::Scan {
            dataset: "d".into(),
            limit: None,
        };
        let opt = optimize(plan, 100);
        assert_eq!(
            opt,
            QueryPlan::Scan {
                dataset: "d".into(),
                limit: Some(100)
            }
        );
    }
}

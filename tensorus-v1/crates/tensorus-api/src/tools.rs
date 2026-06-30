//! Agent tools backed by [`TensorService`]. Unlike the synchronous
//! [`tensorus_ai::FnTool`], these implement the async [`Tool`] trait directly so
//! they can call into the async storage/index layer, letting the ReAct agent
//! actually explore and query real data.

use crate::service::{PropertyQuery, TensorService};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::str::FromStr;
use std::sync::Arc;
use tensorus_ai::{QueryContext, Tool, ToolRegistry};
use tensorus_core::types::TensorId;

/// Which service operation a [`ServiceTool`] performs.
enum ToolKind {
    ListDatasets,
    Scan,
    PropertySearch,
    Aggregate,
    Get,
}

/// A tool that performs one [`TensorService`] operation.
struct ServiceTool {
    service: Arc<TensorService>,
    kind: ToolKind,
    /// Tenant to scope dataset names to (`None` = legacy unscoped).
    tenant: Option<String>,
}

impl ServiceTool {
    fn arg_str<'a>(args: &'a Value, key: &str) -> Result<&'a str, String> {
        args.get(key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("missing string argument '{key}'"))
    }

    /// Scope a user-facing dataset name to its storage key.
    fn scoped(&self, ds: &str) -> String {
        match &self.tenant {
            Some(t) => format!("{t}.{ds}"),
            None => ds.to_string(),
        }
    }
}

#[async_trait]
impl Tool for ServiceTool {
    fn name(&self) -> &str {
        match self.kind {
            ToolKind::ListDatasets => "list_datasets",
            ToolKind::Scan => "tensor_scan",
            ToolKind::PropertySearch => "tensor_search",
            ToolKind::Aggregate => "tensor_aggregate",
            ToolKind::Get => "tensor_get",
        }
    }

    fn description(&self) -> &str {
        match self.kind {
            ToolKind::ListDatasets => "List all dataset names. No arguments.",
            ToolKind::Scan => "List tensor ids in a dataset. args: {dataset, limit?}",
            ToolKind::PropertySearch => {
                "Find tensors by mathematical property. args: {dataset, min_norm?, max_norm?, \
                 is_symmetric?, is_positive_definite?, rank?, max_condition_number?, limit?}"
            }
            ToolKind::Aggregate => {
                "Aggregate a numeric descriptor field. args: {dataset, function: \
                 count|min|max|sum|avg, field}"
            }
            ToolKind::Get => "Get a tensor's descriptor by id. args: {dataset, id}",
        }
    }

    fn schema(&self) -> Value {
        match self.kind {
            ToolKind::ListDatasets => json!({}),
            ToolKind::Scan => json!({"dataset": "string", "limit": "integer?"}),
            ToolKind::PropertySearch => json!({
                "dataset": "string", "min_norm": "number?", "max_norm": "number?",
                "is_symmetric": "bool?", "is_positive_definite": "bool?",
                "rank": "integer?", "max_condition_number": "number?", "limit": "integer?"
            }),
            ToolKind::Aggregate => {
                json!({"dataset": "string", "function": "string", "field": "string"})
            }
            ToolKind::Get => json!({"dataset": "string", "id": "string"}),
        }
    }

    async fn execute(&self, args: Value) -> Result<String, String> {
        match self.kind {
            ToolKind::ListDatasets => {
                let names = self
                    .service
                    .list_datasets()
                    .await
                    .map_err(|e| e.to_string())?;
                // In tenant scope, show only this tenant's datasets (unprefixed).
                let visible: Vec<String> = match &self.tenant {
                    Some(t) => {
                        let prefix = format!("{t}.");
                        names
                            .into_iter()
                            .filter_map(|k| k.strip_prefix(&prefix).map(|s| s.to_string()))
                            .collect()
                    }
                    None => names,
                };
                Ok(json!({ "datasets": visible }).to_string())
            }
            ToolKind::Scan => {
                let dataset = self.scoped(Self::arg_str(&args, "dataset")?);
                let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
                let recs = self
                    .service
                    .scan(&dataset, limit, 0)
                    .await
                    .map_err(|e| e.to_string())?;
                let ids: Vec<String> = recs.iter().map(|r| r.id.to_string()).collect();
                Ok(json!({ "count": ids.len(), "ids": ids }).to_string())
            }
            ToolKind::PropertySearch => {
                let dataset = self.scoped(Self::arg_str(&args, "dataset")?);
                let q: PropertyQuery =
                    serde_json::from_value(args.clone()).map_err(|e| e.to_string())?;
                let hits = self
                    .service
                    .search_by_property(&dataset, &q)
                    .await
                    .map_err(|e| e.to_string())?;
                let results: Vec<Value> = hits
                    .iter()
                    .map(|(id, d)| {
                        json!({
                            "id": id.to_string(),
                            "frobenius_norm": d.frobenius_norm,
                            "rank": d.rank,
                            "is_symmetric": d.is_symmetric,
                        })
                    })
                    .collect();
                Ok(json!({ "count": results.len(), "results": results }).to_string())
            }
            ToolKind::Aggregate => {
                let dataset = self.scoped(Self::arg_str(&args, "dataset")?);
                let function = Self::arg_str(&args, "function")?;
                let field = Self::arg_str(&args, "field")?;
                let rows =
                    QueryContext::aggregate(self.service.as_ref(), &dataset, function, field)
                        .await?;
                let value = rows.first().map(|r| r.score).unwrap_or(0.0);
                Ok(json!({ "function": function, "field": field, "value": value }).to_string())
            }
            ToolKind::Get => {
                let dataset = self.scoped(Self::arg_str(&args, "dataset")?);
                let id_str = Self::arg_str(&args, "id")?;
                let id = TensorId::from_str(id_str).map_err(|e| e.to_string())?;
                let rec = self
                    .service
                    .get(&dataset, id)
                    .await
                    .map_err(|e| e.to_string())?;
                Ok(json!({
                    "id": rec.id.to_string(),
                    "shape": rec.descriptor.shape.dims(),
                    "frobenius_norm": rec.descriptor.frobenius_norm,
                    "rank": rec.descriptor.rank,
                    "is_symmetric": rec.descriptor.is_symmetric,
                    "trace": rec.descriptor.trace,
                })
                .to_string())
            }
        }
    }
}

/// Build the default tool registry for the ReAct agent, backed by `service` and
/// scoped to `tenant` (`None` = legacy unscoped).
pub fn default_tool_registry(service: Arc<TensorService>, tenant: Option<String>) -> ToolRegistry {
    let mut reg = ToolRegistry::new();
    for kind in [
        ToolKind::ListDatasets,
        ToolKind::Scan,
        ToolKind::PropertySearch,
        ToolKind::Aggregate,
        ToolKind::Get,
    ] {
        reg.register(Arc::new(ServiceTool {
            service: service.clone(),
            kind,
            tenant: tenant.clone(),
        }));
    }
    reg
}

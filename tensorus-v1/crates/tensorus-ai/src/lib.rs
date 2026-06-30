//! # tensorus-ai
//!
//! The intelligence layer: a model-agnostic LLM router with pluggable
//! providers, the Neural Query Language (NQL) parser/planner/executor, a ReAct
//! agent orchestrator, and the workload auto-optimizer.

#![forbid(unsafe_code)]

pub mod agent;
pub mod llm;
pub mod nql;
pub mod optimizer;
pub mod providers;
pub mod router;

pub use agent::{AgentConfig, AgentOutcome, FnTool, ReActAgent, Tool, ToolRegistry};
pub use llm::{
    CompletionResponse, HttpClient, InferenceConfig, LlmError, LlmProvider, Message,
    ProviderCapabilities, Role, Usage,
};
pub use nql::{
    execute, optimize, Cmp, Nql, NqlResult, Predicate, QueryContext, QueryPlan, QueryRow,
};
pub use optimizer::{Action, AutoOptimizer, OptimizerConfig, WorkloadProfiler};
pub use providers::{AnthropicProvider, MockProvider, OpenAiCompatProvider};
pub use router::{LlmRouter, RoutingStrategy};

//! ReAct agent orchestrator: a Thought -> Action -> Observation loop driven by
//! the [`LlmRouter`], with a pluggable tool registry and guardrails (max steps,
//! timeout, and the router's cost budget).
//!
//! Reference: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language
//! Models" (2023).

use crate::llm::{InferenceConfig, LlmError, Message};
use crate::router::LlmRouter;
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A callable tool the agent can invoke.
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// JSON Schema for the tool's arguments (advisory; shown to the model).
    fn schema(&self) -> serde_json::Value;
    /// Execute the tool, returning an observation string or an error.
    async fn execute(&self, args: serde_json::Value) -> Result<String, String>;
}

/// A tool backed by a synchronous closure (convenient for wiring/tests).
pub struct FnTool {
    name: String,
    description: String,
    schema: serde_json::Value,
    #[allow(clippy::type_complexity)]
    func: Box<dyn Fn(serde_json::Value) -> Result<String, String> + Send + Sync>,
}

impl FnTool {
    pub fn new(
        name: &str,
        description: &str,
        schema: serde_json::Value,
        func: impl Fn(serde_json::Value) -> Result<String, String> + Send + Sync + 'static,
    ) -> Self {
        FnTool {
            name: name.to_string(),
            description: description.to_string(),
            schema,
            func: Box::new(func),
        }
    }
}

#[async_trait]
impl Tool for FnTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn schema(&self) -> serde_json::Value {
        self.schema.clone()
    }
    async fn execute(&self, args: serde_json::Value) -> Result<String, String> {
        (self.func)(args)
    }
}

/// Registry of tools available to an agent.
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// A human/LLM-readable catalog of tools.
    fn catalog(&self) -> String {
        let mut names: Vec<&String> = self.tools.keys().collect();
        names.sort();
        names
            .iter()
            .map(|n| {
                let t = &self.tools[*n];
                format!("- {}: {} args={}", t.name(), t.description(), t.schema())
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// One step of the agent's trace.
#[derive(Debug, Clone)]
pub struct Step {
    pub thought: String,
    pub tool: Option<String>,
    pub args: Option<serde_json::Value>,
    pub observation: Option<String>,
}

/// Outcome of an agent run.
#[derive(Debug, Clone)]
pub enum AgentOutcome {
    Success { answer: String, steps: Vec<Step> },
    MaxStepsReached { steps: Vec<Step> },
    TimedOut { steps: Vec<Step> },
    BudgetExceeded { steps: Vec<Step> },
}

/// Agent guardrails.
#[derive(Debug, Clone, Copy)]
pub struct AgentConfig {
    pub max_steps: usize,
    pub timeout: Duration,
}

impl Default for AgentConfig {
    fn default() -> Self {
        AgentConfig {
            max_steps: 10,
            timeout: Duration::from_secs(60),
        }
    }
}

/// The model's decision at each step.
#[derive(Debug, Deserialize)]
struct Decision {
    #[serde(default)]
    thought: String,
    #[serde(default)]
    action: Option<ToolCall>,
    #[serde(default)]
    final_answer: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ToolCall {
    tool: String,
    #[serde(default)]
    args: serde_json::Value,
}

const DECISION_SCHEMA: &str = r#"{"thought": string, and EITHER "action": {"tool": string, "args": object} OR "final_answer": string}"#;

/// A ReAct agent.
pub struct ReActAgent {
    router: Arc<LlmRouter>,
    tools: ToolRegistry,
    cfg: AgentConfig,
}

impl ReActAgent {
    pub fn new(router: Arc<LlmRouter>, tools: ToolRegistry, cfg: AgentConfig) -> Self {
        ReActAgent { router, tools, cfg }
    }

    /// Run the agent on a task.
    pub async fn run(&self, task: &str) -> AgentOutcome {
        let start = Instant::now();
        let mut steps: Vec<Step> = Vec::new();
        let mut transcript = vec![
            Message::system(format!(
                "You are a tensor-database agent. Use the ReAct pattern: reason, then either call a \
                 tool or give a final answer. Available tools:\n{}\n\nAt each step respond with JSON: \
                 either {{\"thought\":..., \"action\":{{\"tool\":..., \"args\":{{...}}}}}} or \
                 {{\"thought\":..., \"final_answer\":...}}.",
                self.tools.catalog()
            )),
            Message::user(format!("Task: {task}")),
        ];

        for _ in 0..self.cfg.max_steps {
            if start.elapsed() > self.cfg.timeout {
                return AgentOutcome::TimedOut { steps };
            }
            let decision = match self
                .router
                .complete_structured::<Decision>(
                    &transcript,
                    DECISION_SCHEMA,
                    &InferenceConfig::default(),
                    2,
                )
                .await
            {
                Ok(d) => d,
                Err(LlmError::BudgetExceeded) => return AgentOutcome::BudgetExceeded { steps },
                Err(_) => {
                    // Treat a malformed/failed decision as a step that observes the failure.
                    steps.push(Step {
                        thought: "(model error)".into(),
                        tool: None,
                        args: None,
                        observation: Some("model failed to produce a decision".into()),
                    });
                    continue;
                }
            };

            if let Some(answer) = decision.final_answer {
                steps.push(Step {
                    thought: decision.thought,
                    tool: None,
                    args: None,
                    observation: None,
                });
                return AgentOutcome::Success { answer, steps };
            }

            let (tool_name, args) = match decision.action {
                Some(tc) => (tc.tool, tc.args),
                None => {
                    steps.push(Step {
                        thought: decision.thought,
                        tool: None,
                        args: None,
                        observation: Some("no action or final answer provided".into()),
                    });
                    transcript.push(Message::user(
                        "You must provide either an action or a final_answer.",
                    ));
                    continue;
                }
            };

            let observation = match self.tools.get(&tool_name) {
                Some(tool) => tool
                    .execute(args.clone())
                    .await
                    .unwrap_or_else(|e| format!("tool error: {e}")),
                None => format!("unknown tool '{tool_name}'"),
            };

            transcript.push(Message::assistant(format!(
                "{{\"thought\":\"{}\",\"action\":{{\"tool\":\"{}\"}}}}",
                decision.thought.replace('"', "'"),
                tool_name
            )));
            transcript.push(Message::user(format!("Observation: {observation}")));

            steps.push(Step {
                thought: decision.thought,
                tool: Some(tool_name),
                args: Some(args),
                observation: Some(observation),
            });
        }

        AgentOutcome::MaxStepsReached { steps }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::MockProvider;
    use crate::router::RoutingStrategy;

    fn agent_with(script: Vec<&str>, tools: ToolRegistry, max_steps: usize) -> ReActAgent {
        let provider = MockProvider::scripted(
            "mock",
            script.into_iter().map(|s| Ok(s.to_string())).collect(),
        );
        let router = Arc::new(LlmRouter::new(
            vec![Arc::new(provider)],
            RoutingStrategy::Fixed { provider_index: 0 },
            0.0,
        ));
        ReActAgent::new(
            router,
            tools,
            AgentConfig {
                max_steps,
                timeout: Duration::from_secs(60),
            },
        )
    }

    fn compute_tool() -> Arc<dyn Tool> {
        Arc::new(FnTool::new(
            "tensor_compute",
            "Compute a tensor operation",
            serde_json::json!({"op": "string", "tensor": "string"}),
            |args| {
                let op = args.get("op").and_then(|v| v.as_str()).unwrap_or("");
                Ok(format!("computed {op}: result=[1.0, 2.0]"))
            },
        ))
    }

    fn search_tool() -> Arc<dyn Tool> {
        Arc::new(FnTool::new(
            "tensor_search",
            "Search tensors",
            serde_json::json!({"dataset": "string"}),
            |args| {
                let ds = args.get("dataset").and_then(|v| v.as_str()).unwrap_or("");
                Ok(format!("found 2 tensors in {ds}: [T1, T2]"))
            },
        ))
    }

    #[tokio::test]
    async fn single_tool_task() {
        let mut reg = ToolRegistry::new();
        reg.register(compute_tool());
        let agent = agent_with(
            vec![
                r#"{"thought":"I should compute the SVD","action":{"tool":"tensor_compute","args":{"op":"svd","tensor":"X"}}}"#,
                r#"{"thought":"I have the result","final_answer":"SVD of X computed successfully"}"#,
            ],
            reg,
            10,
        );
        let outcome = agent.run("compute SVD of tensor X").await;
        match outcome {
            AgentOutcome::Success { answer, steps } => {
                assert!(answer.contains("SVD"));
                // First step used the tool and recorded an observation.
                assert_eq!(steps[0].tool.as_deref(), Some("tensor_compute"));
                assert!(steps[0]
                    .observation
                    .as_ref()
                    .unwrap()
                    .contains("computed svd"));
            }
            other => panic!("expected success, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn multi_step_task() {
        let mut reg = ToolRegistry::new();
        reg.register(compute_tool());
        reg.register(search_tool());
        let agent = agent_with(
            vec![
                r#"{"thought":"find the largest tensor","action":{"tool":"tensor_search","args":{"dataset":"Y"}}}"#,
                r#"{"thought":"compute eigenvalues of it","action":{"tool":"tensor_compute","args":{"op":"eigenvalues","tensor":"T1"}}}"#,
                r#"{"thought":"done","final_answer":"The largest tensor T1 has eigenvalues [1.0, 2.0]"}"#,
            ],
            reg,
            10,
        );
        let outcome = agent
            .run("find the largest tensor in dataset Y and compute its eigenvalues")
            .await;
        match outcome {
            AgentOutcome::Success { answer, steps } => {
                assert!(answer.contains("eigenvalues"));
                assert_eq!(steps.len(), 3); // search, compute, final
                assert_eq!(steps[0].tool.as_deref(), Some("tensor_search"));
                assert_eq!(steps[1].tool.as_deref(), Some("tensor_compute"));
            }
            other => panic!("expected success, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stops_at_max_steps() {
        let mut reg = ToolRegistry::new();
        reg.register(compute_tool());
        // Always acts, never finishes.
        let action =
            r#"{"thought":"keep going","action":{"tool":"tensor_compute","args":{"op":"x"}}}"#;
        let agent = agent_with(vec![action, action, action, action, action], reg, 3);
        let outcome = agent.run("loop forever").await;
        match outcome {
            AgentOutcome::MaxStepsReached { steps } => assert_eq!(steps.len(), 3),
            other => panic!("expected max steps, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn unknown_tool_then_recovers() {
        let mut reg = ToolRegistry::new();
        reg.register(compute_tool());
        let agent = agent_with(
            vec![
                r#"{"thought":"try a tool","action":{"tool":"nonexistent","args":{}}}"#,
                r#"{"thought":"fall back to answer","final_answer":"done despite the error"}"#,
            ],
            reg,
            10,
        );
        let outcome = agent.run("do something").await;
        match outcome {
            AgentOutcome::Success { steps, .. } => {
                assert!(steps[0]
                    .observation
                    .as_ref()
                    .unwrap()
                    .contains("unknown tool"));
            }
            other => panic!("expected success, got {other:?}"),
        }
    }
}

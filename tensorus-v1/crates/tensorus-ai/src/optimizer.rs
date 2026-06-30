//! Self-tuning auto-optimizer: profiles the query workload, generates candidate
//! tuning actions (e.g. building an index on a hot field), and selects among
//! them with a UCB1 multi-armed bandit, subject to safety constraints (cooldown
//! and an action budget) to avoid oscillation.

use std::collections::{HashMap, HashSet, VecDeque};

/// A single observed query: which field it filtered on, and how long it took.
#[derive(Debug, Clone)]
pub struct QueryObservation {
    pub field: String,
    pub latency_ms: f64,
}

/// Sliding-window workload profiler (a ring buffer of recent observations).
pub struct WorkloadProfiler {
    window: VecDeque<QueryObservation>,
    capacity: usize,
}

impl WorkloadProfiler {
    pub fn new(capacity: usize) -> Self {
        WorkloadProfiler {
            window: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn record(&mut self, field: &str, latency_ms: f64) {
        if self.window.len() == self.capacity {
            self.window.pop_front();
        }
        self.window.push_back(QueryObservation {
            field: field.to_string(),
            latency_ms,
        });
    }

    /// Per-field access frequency within the window.
    pub fn field_frequency(&self) -> HashMap<String, usize> {
        let mut freq = HashMap::new();
        for o in &self.window {
            *freq.entry(o.field.clone()).or_insert(0) += 1;
        }
        freq
    }

    /// Fields ordered by descending frequency.
    pub fn hot_fields(&self, k: usize) -> Vec<String> {
        let mut v: Vec<(String, usize)> = self.field_frequency().into_iter().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.into_iter().take(k).map(|(f, _)| f).collect()
    }

    /// Mean latency of queries on a field within the window.
    pub fn mean_latency(&self, field: &str) -> Option<f64> {
        let xs: Vec<f64> = self
            .window
            .iter()
            .filter(|o| o.field == field)
            .map(|o| o.latency_ms)
            .collect();
        if xs.is_empty() {
            None
        } else {
            Some(xs.iter().sum::<f64>() / xs.len() as f64)
        }
    }
}

/// A tuning action the optimizer may take.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Action {
    /// Build a learned index on a property field.
    CreateIndex(String),
    /// Do nothing this round.
    NoOp,
}

#[derive(Default, Clone, Copy)]
struct Arm {
    count: u64,
    total_reward: f64,
}

impl Arm {
    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_reward / self.count as f64
        }
    }
}

/// Optimizer configuration.
#[derive(Debug, Clone, Copy)]
pub struct OptimizerConfig {
    /// Number of hot fields considered for indexing each round.
    pub top_k: usize,
    /// UCB1 exploration constant.
    pub exploration: f64,
    /// Minimum iterations before the same action may be taken again.
    pub cooldown_iters: u64,
    /// Maximum number of (non-NoOp) actions applied overall.
    pub action_budget: u64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        OptimizerConfig {
            top_k: 3,
            exploration: 1.0,
            cooldown_iters: 5,
            action_budget: 1_000,
        }
    }
}

/// The self-tuning optimizer.
pub struct AutoOptimizer {
    pub profiler: WorkloadProfiler,
    arms: HashMap<Action, Arm>,
    total_pulls: u64,
    last_applied: HashMap<Action, u64>,
    iter: u64,
    applied_count: u64,
    cfg: OptimizerConfig,
}

impl AutoOptimizer {
    pub fn new(cfg: OptimizerConfig) -> Self {
        AutoOptimizer {
            profiler: WorkloadProfiler::new(1024),
            arms: HashMap::new(),
            total_pulls: 0,
            last_applied: HashMap::new(),
            iter: 0,
            applied_count: 0,
            cfg,
        }
    }

    /// Record a query observation into the profiler.
    pub fn observe(&mut self, field: &str, latency_ms: f64) {
        self.profiler.record(field, latency_ms);
    }

    /// Candidate actions given the set of currently indexed fields.
    fn candidates(&self, indexed: &HashSet<String>) -> Vec<Action> {
        let mut acts = vec![Action::NoOp];
        for f in self.profiler.hot_fields(self.cfg.top_k) {
            if !indexed.contains(&f) {
                acts.push(Action::CreateIndex(f));
            }
        }
        acts
    }

    fn ucb_score(&self, action: &Action) -> f64 {
        let arm = self.arms.get(action).copied().unwrap_or_default();
        if arm.count == 0 {
            return f64::INFINITY; // explore unseen arms first
        }
        let total = self.total_pulls.max(1) as f64;
        arm.mean() + self.cfg.exploration * (total.ln() / arm.count as f64).sqrt()
    }

    fn on_cooldown(&self, action: &Action) -> bool {
        if let Action::NoOp = action {
            return false; // NoOp is never on cooldown
        }
        match self.last_applied.get(action) {
            Some(&last) => self.iter.saturating_sub(last) < self.cfg.cooldown_iters,
            None => false,
        }
    }

    /// Suggest the next action (or `None` if nothing is eligible). Respects the
    /// cooldown and action budget.
    pub fn suggest(&self, indexed: &HashSet<String>) -> Option<Action> {
        let candidates: Vec<Action> = self
            .candidates(indexed)
            .into_iter()
            .filter(|a| !self.on_cooldown(a))
            .collect();
        if self.applied_count >= self.cfg.action_budget {
            return Some(Action::NoOp);
        }
        candidates
            .into_iter()
            .max_by(|a, b| self.ucb_score(a).total_cmp(&self.ucb_score(b)))
    }

    /// Apply the outcome of taking `action`, recording its reward.
    pub fn apply(&mut self, action: Action, reward: f64) {
        let arm = self.arms.entry(action.clone()).or_default();
        arm.count += 1;
        arm.total_reward += reward;
        self.total_pulls += 1;
        self.last_applied.insert(action.clone(), self.iter);
        if !matches!(action, Action::NoOp) {
            self.applied_count += 1;
        }
        self.iter += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiler_identifies_hot_field() {
        let mut p = WorkloadProfiler::new(100);
        for _ in 0..10 {
            p.record("frobenius_norm", 50.0);
        }
        for _ in 0..2 {
            p.record("rank", 5.0);
        }
        assert_eq!(p.hot_fields(1), vec!["frobenius_norm".to_string()]);
        assert_eq!(p.field_frequency()["frobenius_norm"], 10);
    }

    #[test]
    fn suggests_index_for_hot_field() {
        let mut opt = AutoOptimizer::new(OptimizerConfig::default());
        for _ in 0..20 {
            opt.observe("frobenius_norm", 40.0);
        }
        let indexed = HashSet::new();
        let action = opt.suggest(&indexed).unwrap();
        assert_eq!(action, Action::CreateIndex("frobenius_norm".to_string()));
    }

    #[test]
    fn cooldown_prevents_oscillation() {
        let mut opt = AutoOptimizer::new(OptimizerConfig {
            cooldown_iters: 5,
            ..Default::default()
        });
        for _ in 0..20 {
            opt.observe("norm", 40.0);
        }
        let indexed = HashSet::new();
        let a1 = opt.suggest(&indexed).unwrap();
        assert_eq!(a1, Action::CreateIndex("norm".to_string()));
        opt.apply(a1.clone(), 39.0);
        // Immediately suggesting again (field still unindexed in this scenario)
        // must NOT repeat the action — it is on cooldown.
        let a2 = opt.suggest(&indexed).unwrap();
        assert_ne!(
            a2, a1,
            "should not re-suggest the same action during cooldown"
        );
    }

    #[test]
    fn improves_simulated_workload_latency() {
        let mut opt = AutoOptimizer::new(OptimizerConfig {
            cooldown_iters: 5,
            ..Default::default()
        });
        let mut indexed: HashSet<String> = HashSet::new();
        let mut latencies = Vec::new();
        let hot = "frobenius_norm";

        for it in 0..100 {
            // Latency depends on whether the hot field is indexed.
            let lat = if indexed.contains(hot) { 1.0 } else { 50.0 };
            opt.observe(hot, lat);
            // A little noise on another field.
            if it % 7 == 0 {
                opt.observe("rank", 3.0);
            }
            latencies.push(lat);

            if let Some(action) = opt.suggest(&indexed) {
                match &action {
                    Action::CreateIndex(f) => {
                        let before = opt.profiler.mean_latency(f).unwrap_or(50.0);
                        indexed.insert(f.clone());
                        let after = 1.0;
                        opt.apply(action, before - after); // reward = latency saved
                    }
                    Action::NoOp => opt.apply(action, 0.0),
                }
            }
        }

        assert!(
            indexed.contains(hot),
            "optimizer should have indexed the hot field"
        );
        let first: f64 = latencies[..50].iter().sum::<f64>() / 50.0;
        let second: f64 = latencies[50..].iter().sum::<f64>() / 50.0;
        println!("avg latency: first half {first:.1}ms, second half {second:.1}ms");
        assert!(
            second < first * 0.5,
            "second-half latency ({second}) should be much lower than first ({first})"
        );
    }

    #[test]
    fn action_budget_enforced() {
        let mut opt = AutoOptimizer::new(OptimizerConfig {
            action_budget: 1,
            cooldown_iters: 0,
            ..Default::default()
        });
        for _ in 0..20 {
            opt.observe("a", 40.0);
            opt.observe("b", 40.0);
        }
        let indexed = HashSet::new();
        let a1 = opt.suggest(&indexed).unwrap();
        opt.apply(a1, 30.0); // uses the only budgeted action
                             // Further suggestions are forced to NoOp by the budget.
        assert_eq!(opt.suggest(&indexed), Some(Action::NoOp));
    }
}

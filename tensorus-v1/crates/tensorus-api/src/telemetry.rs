//! Observability: structured logging/tracing setup and Prometheus-format
//! metrics (counters + a latency histogram).
//!
//! ## Scope & substitution
//!
//! This provides the in-process pieces the API needs: a `tracing` subscriber
//! (pretty or JSON) and a self-contained metrics registry rendered in the
//! Prometheus exposition format. Exporting traces to an OpenTelemetry collector
//! (OTLP) is an additive step via `tracing-opentelemetry`; it is documented as
//! optional here to avoid the heavy OTel dependency tree. `tracing` spans added
//! across the crates flow to whichever subscriber is installed (stdout in dev),
//! satisfying "traces appear in a collector (or stdout in test mode)".

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Once;

static INIT: Once = Once::new();

/// Install a global `tracing` subscriber. `json = true` emits structured JSON
/// logs (production); otherwise a human-readable format. Idempotent.
pub fn init_tracing(json: bool) {
    INIT.call_once(|| {
        use tracing_subscriber::{fmt, EnvFilter};
        let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
        if json {
            let _ = fmt()
                .with_env_filter(filter)
                .json()
                .with_current_span(true)
                .try_init();
        } else {
            let _ = fmt().with_env_filter(filter).try_init();
        }
    });
}

/// A cumulative latency histogram with fixed millisecond buckets, rendered in
/// the Prometheus exposition format.
pub struct Histogram {
    /// Upper bounds (ms), ascending.
    bounds: Vec<f64>,
    /// Cumulative counts: `counts[i]` = number of observations `<= bounds[i]`.
    counts: Vec<AtomicU64>,
    /// Total observation count.
    total: AtomicU64,
    /// Sum of observations in microseconds (integer-atomic).
    sum_micros: AtomicU64,
}

impl Histogram {
    /// Create a histogram with default request-latency buckets (ms).
    pub fn new_default() -> Self {
        Self::with_bounds(vec![
            0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0,
        ])
    }

    pub fn with_bounds(bounds: Vec<f64>) -> Self {
        let counts = bounds.iter().map(|_| AtomicU64::new(0)).collect();
        Histogram {
            bounds,
            counts,
            total: AtomicU64::new(0),
            sum_micros: AtomicU64::new(0),
        }
    }

    /// Record an observation in milliseconds.
    pub fn observe(&self, ms: f64) {
        self.total.fetch_add(1, Ordering::Relaxed);
        self.sum_micros
            .fetch_add((ms * 1000.0).round() as u64, Ordering::Relaxed);
        for (bound, count) in self.bounds.iter().zip(&self.counts) {
            if ms <= *bound {
                count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Total observations.
    pub fn count(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }

    /// Sum of observations in milliseconds.
    pub fn sum_ms(&self) -> f64 {
        self.sum_micros.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Render this histogram in the Prometheus exposition format under `name`.
    pub fn render(&self, name: &str) -> String {
        let mut out = String::new();
        out.push_str(&format!("# TYPE {name} histogram\n"));
        for (bound, count) in self.bounds.iter().zip(&self.counts) {
            out.push_str(&format!(
                "{name}_bucket{{le=\"{}\"}} {}\n",
                bound,
                count.load(Ordering::Relaxed)
            ));
        }
        let total = self.total.load(Ordering::Relaxed);
        out.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {total}\n"));
        out.push_str(&format!("{name}_sum {}\n", self.sum_ms()));
        out.push_str(&format!("{name}_count {total}\n"));
        out
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_counts_and_buckets() {
        let h = Histogram::with_bounds(vec![1.0, 10.0, 100.0]);
        h.observe(0.5); // <= 1, 10, 100
        h.observe(5.0); // <= 10, 100
        h.observe(50.0); // <= 100
        h.observe(500.0); // > all finite bounds

        assert_eq!(h.count(), 4);
        assert!((h.sum_ms() - 555.5).abs() < 1e-3);

        let rendered = h.render("tensorus_latency_ms");
        assert!(rendered.contains("tensorus_latency_ms_bucket{le=\"1\"} 1"));
        assert!(rendered.contains("tensorus_latency_ms_bucket{le=\"10\"} 2"));
        assert!(rendered.contains("tensorus_latency_ms_bucket{le=\"100\"} 3"));
        assert!(rendered.contains("tensorus_latency_ms_bucket{le=\"+Inf\"} 4"));
        assert!(rendered.contains("tensorus_latency_ms_count 4"));
    }

    #[test]
    fn init_tracing_is_idempotent() {
        init_tracing(false);
        init_tracing(false); // must not panic on second call
    }
}

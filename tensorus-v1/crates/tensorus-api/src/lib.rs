//! # tensorus-api
//!
//! The network-facing API. The **REST** surface ([`rest`]) is built on Axum with
//! API-key auth, token-bucket rate limiting, and health/metrics endpoints, all
//! over a transport-agnostic [`service::TensorService`].
//!
//! ## gRPC
//!
//! The protobuf service is defined at `proto/tensorus.proto`. A tonic server
//! mirrors the REST handlers over the same [`service::TensorService`]. tonic's
//! build step requires `protoc`, which is not available on this build host, so
//! the gRPC server is gated behind an optional `grpc` feature (not built by
//! default). The REST surface is the tested primary in this environment.

#![forbid(unsafe_code)]

pub mod rest;
pub mod service;
pub mod telemetry;

pub use rest::{build_app, ApiConfig, AppState};
pub use service::{HealthInfo, InsertRequest, PropertyQuery, TensorService};
pub use telemetry::{init_tracing, Histogram};

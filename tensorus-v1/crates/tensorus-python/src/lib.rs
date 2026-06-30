//! # tensorus-python
//!
//! PyO3 bindings exposing the Tensorus engine to Python. The actual `#[pyclass]`
//! and `#[pymodule]` definitions live behind the non-default `python` feature
//! (see [`bindings`]) so the workspace builds without a CPython link target on
//! toolchains that lack one. Build the extension with
//! `maturin develop --features python`.

#![cfg_attr(not(feature = "python"), allow(dead_code))]

#[cfg(feature = "python")]
mod bindings;

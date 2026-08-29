//! CLI command implementations for the `fuga` binary.
//!
//! Decomposed from the original 10.7K-line `src/main.rs` monolith.
//! Each submodule owns one command family. `main.rs` stays a thin
//! dispatcher over these modules.
//!
//! Phase 1 (this pass): argument parsing helpers.
//! Phase 1+ (next): analyze/train/tm/crystal/agent/query/reflect/sim/print.

pub mod args;
pub mod print;

//! CLI command implementations for the `fuga` binary.
//!
//! Decomposed from the original 10.7K-line `src/main.rs` monolith.
//! Each submodule owns one command family. `main.rs` stays a thin
//! dispatcher over these modules.

pub mod agent;
pub mod analyze;
pub mod args;
pub mod crystal;
pub mod inspect;
pub mod jepa;
pub mod print;
pub mod query;
pub mod sim;
pub mod tm_gen;
pub mod tools;
pub mod train;

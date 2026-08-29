//! CLI argument parsing helpers for the `fuga` binary.
//!
//! Extracted from `src/main.rs` during the monolith decomposition
//! (Phase 1 of docs/refactor-plan.md). Pure functions, no state.

use fuga::OutputFormat;

/// Parse `--dim` / `-d` from args starting at `start`.
pub fn parse_dim(args: &[String], start: usize) -> Option<usize> {
    parse_flag_value_in(args, start, &["--dim", "-d"]).and_then(|s| s.parse().ok())
}

/// Parse `--window` / `-w` from args starting at `start`.
pub fn parse_window(args: &[String], start: usize) -> Option<usize> {
    parse_flag_value_in(args, start, &["--window", "-w"]).and_then(|s| s.parse().ok())
}

/// Parse a single integer flag (`--flag N`).
pub fn parse_int(args: &[String], flag: &str) -> Option<usize> {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return args[i + 1].parse().ok();
        }
    }
    None
}

/// Parse a single float flag (`--flag N.N`).
pub fn parse_float(args: &[String], flag: &str) -> Option<f64> {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return args[i + 1].parse().ok();
        }
    }
    None
}

/// Parse `--output` / `-o` from args starting at `start`.
pub fn parse_output(args: &[String], start: usize) -> Option<String> {
    parse_flag_value_in(args, start, &["--output", "-o"]).map(|s| s.to_string())
}

/// Parse `--format` / `-f` from args starting at `start`.
pub fn parse_format(args: &[String], start: usize) -> Option<OutputFormat> {
    parse_flag_value_in(args, start, &["--format", "-f"])
        .and_then(OutputFormat::from_str)
}

/// Parse `--to` (translation target) from args starting at `start`.
pub fn parse_translate_target(args: &[String], start: usize) -> Option<&str> {
    parse_flag_value_in(args, start, &["--to"])
}

/// True if `flag` appears anywhere in args.
pub fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

/// Parse the value of a single flag, scanning from index 1 (skip program name).
pub fn parse_flag_value<'a>(args: &'a [String], _start: usize, flag: &str) -> Option<&'a str> {
    for i in 1..args.len().saturating_sub(1) {
        if args[i] == flag {
            return Some(&args[i + 1]);
        }
    }
    None
}

/// Parse all values of a repeated flag (e.g. `--jsonl a --jsonl b`).
pub fn parse_flag_values<'a>(args: &'a [String], flag: &str) -> Vec<&'a str> {
    let mut values = Vec::new();
    for i in 1..args.len().saturating_sub(1) {
        if args[i] == flag {
            values.push(args[i + 1].as_str());
        }
    }
    values
}

/// Shared helper: find the value of the first of `flags` present at or after `start`.
fn parse_flag_value_in<'a>(
    args: &'a [String],
    start: usize,
    flags: &[&str],
) -> Option<&'a str> {
    for i in start..args.len() {
        if flags.contains(&args[i].as_str()) && i + 1 < args.len() {
            return Some(&args[i + 1]);
        }
    }
    None
}

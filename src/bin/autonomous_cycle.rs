use std::env;
use std::fs;
use std::process;

use fuga::ai::mentalese::{extract_params, generate_body, Thought};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} \"<prompt>\"", args[0]);
        process::exit(1);
    }

    let prompt = &args[1];
    let thought = if prompt.contains("loop") || prompt.contains("for") {
        Thought::Loop
    } else if prompt.contains("match") || prompt.contains('{') {
        Thought::Match
    } else if prompt.contains("struct") || prompt.contains("type") {
        Thought::DeclareVar
    } else {
        Thought::BinaryOp
    };

    let body = generate_body(&format!("fn main {}", prompt), thought);
    let full_code = format!("fn main () {{ {} }}", body);

    let tmp_dir = env::temp_dir();
    let tmp_file_path = tmp_dir.join("autonomous_cycle_test.rs");
    fs::write(&tmp_file_path, full_code).expect("cannot write to temp file");

    let compile_status = process::Command::new("rustc")
        .arg(&tmp_file_path)
        .status()
        .expect("rustc command failed");

    let _ = fs::remove_file(&tmp_file_path);

    if compile_status.success() {
        println!("PASS");
        process::exit(0);
    } else {
        println!("FAIL");
        process::exit(1);
    }
}
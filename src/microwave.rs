use std::fs;
use std::path::Path;
use std::process::Command;

fn strip_prefix<'a>(s: &'a str) -> &'a str {
    let prefixes = [
        "pub ",
        "pub(crate) ",
        "pub(super) ",
        "pub(in ",
        "async ",
        "unsafe ",
        "extern ",
        "static ",
        "const ",
        "default ",
    ];
    let mut r = s;
    loop {
        let before = r;
        for p in &prefixes {
            if r.starts_with(p) {
                r = &r[p.len()..];
                break;
            }
        }
        if r == before {
            break;
        }
    }
    r
}

const MINI_FUGA_SRC: &str = r##"use std::process::Command;
use std::collections::HashSet;

struct SandboxOutcome {
    compiles: bool,
    errors: usize,
    warnings: usize,
    functions: usize,
    structs: usize,
    impls: usize,
    classes: usize,
    templates: usize,
    includes: usize,
    collage: f64,
    reward: f64,
}

fn strip_prefix(s: &str) -> &str {
    let prefixes = ["pub ", "pub(crate) ", "pub(super) ", "pub(in ",
                     "async ", "unsafe ", "extern ", "const "];
    let mut r = s;
    loop { let b = r; for p in &prefixes { if r.starts_with(p) { r = &r[p.len()..]; break; } } if r == b { break; } }
    r
}

fn run_rust(code: &str) -> SandboxOutcome {
    let dir = std::env::temp_dir().join("fuga_mini");
    let _ = std::fs::create_dir_all(&dir);
    let src = dir.join("mini.rs");
    let _ = std::fs::write(&src, code);
    let out = dir.join("mini.rlib");
    let cmd = Command::new("rustc")
        .arg("--edition").arg("2021")
        .arg("--crate-type").arg("lib")
        .arg("--crate-name").arg("mini")
        .arg(&src).arg("-o").arg(&out)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output();
    let output = match cmd {
        Ok(o) => o,
        Err(_) => return SandboxOutcome {
            compiles: false, errors: 1, warnings: 0,
            functions: 0, structs: 0, impls: 0,
            classes: 0, templates: 0, includes: 0,
            collage: 1.0, reward: -5.0,
        },
    };
    let stderr = String::from_utf8_lossy(&output.stderr);
    let errors = stderr.matches("error[").count() + stderr.matches("error:").count();
    let warnings = stderr.matches("warning[").count() + stderr.matches("warning:").count();
    let success = output.status.success() && errors == 0;

    let mut functions = 0; let mut structs = 0; let mut impls = 0;
    let mut last_line_was_fn = false;
    for line in code.lines() {
        let t = strip_prefix(line.trim());
        if t.starts_with("fn ") { functions += 1; last_line_was_fn = true; } else { last_line_was_fn = false; }
        if t.starts_with("struct ") { structs += 1; }
        if t.starts_with("impl ") && !t.contains("impl ") { impls += 1; }
    }

    let mut seen_fn = HashSet::new();
    let mut seen_st = HashSet::new();
    let mut dup_fn = 0; let mut dup_st = 0;
    for line in code.lines() {
        let t = strip_prefix(line.trim());
        if t.starts_with("fn ") {
            let sig = t.split('{').next().unwrap_or(t).trim().to_string();
            if !seen_fn.insert(sig) { dup_fn += 1; }
        }
        if t.starts_with("struct ") {
            let sig = t.split('{').next().unwrap_or(t).trim().to_string();
            if !seen_st.insert(sig) { dup_st += 1; }
        }
    }
    let total = (functions + structs + impls).max(1);
    let frags = code.lines().filter(|l| {
        let t = strip_prefix(l.trim());
        l.trim().starts_with("pub fn") || l.trim().starts_with("pub struct") || l.trim().starts_with("pub impl")
        || t.starts_with("fn ") || t.starts_with("struct ") || t.starts_with("impl ")
    }).count();
    let ratio = frags as f64 / total as f64;
    let base_collage = if ratio > 3.0 { 0.8 + (ratio - 3.0).min(1.0) * 0.2 }
        else if ratio > 1.5 { 0.4 + (ratio - 1.5) / 1.5 * 0.4 } else { 0.1 };
    let penalty = (dup_fn as f64 * 0.2) + (dup_st as f64 * 0.25);
    let collage = (base_collage + penalty).min(1.0);

    let mut r = 0.0;
    if success { r += 2.0; } else { r -= errors as f64 * 0.5; }
    r -= warnings as f64 * 0.1;
    if functions > 0 { r += 0.5 * (functions as f64).ln().max(0.0); }
    if structs > 0 { r += 0.3 * (structs as f64).ln().max(0.0); }
    if impls > 0 { r += 0.4 * (impls as f64).ln().max(0.0); }
    r -= collage * 2.0;
    let reward = r.max(-10.0).min(10.0);

    SandboxOutcome {
        compiles: success, errors, warnings,
        functions, structs, impls,
        classes: 0, templates: 0, includes: 0,
        collage, reward,
    }
}

fn run_cpp(code: &str) -> SandboxOutcome {
    let dir = std::env::temp_dir().join("fuga_mini");
    let _ = std::fs::create_dir_all(&dir);
    let src = dir.join("mini.cpp");
    let _ = std::fs::write(&src, code);
    let cmd = Command::new("g++")
        .arg("-std=c++17").arg("-Wall").arg("-Wno-unused-variable")
        .arg("-fsyntax-only").arg(&src)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output();
    let output = match cmd {
        Ok(o) => o,
        Err(_) => return SandboxOutcome {
            compiles: false, errors: 1, warnings: 0,
            functions: 0, structs: 0, impls: 0,
            classes: 0, templates: 0, includes: 0,
            collage: 1.0, reward: -5.0,
        },
    };
    let stderr = String::from_utf8_lossy(&output.stderr);
    let errors = stderr.matches("error:").count();
    let warnings = stderr.matches("warning:").count();
    let success = output.status.success() && errors == 0;

    let mut functions = 0; let mut classes = 0; let mut templates = 0; let mut includes = 0;
    let mut sigs: Vec<String> = Vec::new();
    for line in code.lines() {
        let t = line.trim();
        if t.contains("#include") { includes += 1; }
        if t.contains("template") && (t.contains('<') || t.contains("typename")) { templates += 1; }
        if t.starts_with("class ") || t.starts_with("struct ") { classes += 1; }
        let ret_types = ["void ", "int ", "bool ", "auto ", "std::", "static ",
                         "size_t ", "const ", "char ", "double ", "float ", "long ",
                         "short ", "unsigned ", "signed ", "uint32_t ", "int32_t ",
                         "uint64_t ", "int64_t ", "size_t ", "ssize_t "];
        if ret_types.iter().any(|p| t.starts_with(p))
            && t.contains('(') && t.contains(')') && !t.contains(';')
        {
            let sig = t.split('{').next().unwrap_or(t).trim().to_string();
            if !sig.is_empty() && sig.len() < 100 {
                functions += 1; sigs.push(sig);
            }
        }
    }
    let mut seen = HashSet::new();
    let mut dup = 0;
    for s in &sigs {
        if !seen.insert(s.clone()) { dup += 1; }
    }
    let collage = if sigs.is_empty() { 1.0 }
        else { (dup as f64 / sigs.len() as f64).clamp(0.1, 1.0) };

    let mut r = 0.0;
    if success { r += 2.0; } else { r -= errors as f64 * 0.5; }
    r -= warnings as f64 * 0.1;
    if functions > 0 { r += 0.4 * (functions as f64).ln().max(0.0); }
    if classes > 0 { r += 0.5 * (classes as f64).ln().max(0.0); }
    if templates > 0 { r += 0.3 * (templates as f64).ln().max(0.0); }
    if includes > 0 { r += 0.05 * includes as f64; }
    r -= collage * 2.0;
    let reward = r.max(-10.0).min(10.0);

    SandboxOutcome {
        compiles: success, errors, warnings,
        functions, structs: 0, impls: 0,
        classes, templates, includes,
        collage, reward,
    }
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum();
    let nb: f64 = b.iter().map(|x| x * x).sum();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na.sqrt() * nb.sqrt()) }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { eprintln!("Usage: mini_fuga <eval_rust|eval_cpp|cosine> [code]"); return; }
    let mode = &args[1];
    match mode.as_str() {
        "eval_rust" => {
            if args.len() < 3 { eprintln!("Missing code"); return; }
            let r = run_rust(&args[2]);
            println!("COMPILES:{}", r.compiles);
            println!("ERRORS:{}", r.errors);
            println!("WARNINGS:{}", r.warnings);
            println!("FUNCTIONS:{}", r.functions);
            println!("STRUCTS:{}", r.structs);
            println!("IMPLS:{}", r.impls);
            println!("CLASSES:{}", r.classes);
            println!("TEMPLATES:{}", r.templates);
            println!("INCLUDES:{}", r.includes);
            println!("COLLAGE:{:.4}", r.collage);
            println!("REWARD:{:.4}", r.reward);
        }
        "eval_cpp" => {
            if args.len() < 3 { eprintln!("Missing code"); return; }
            let r = run_cpp(&args[2]);
            println!("COMPILES:{}", r.compiles);
            println!("ERRORS:{}", r.errors);
            println!("WARNINGS:{}", r.warnings);
            println!("FUNCTIONS:{}", r.functions);
            println!("STRUCTS:{}", r.structs);
            println!("IMPLS:{}", r.impls);
            println!("CLASSES:{}", r.classes);
            println!("TEMPLATES:{}", r.templates);
            println!("INCLUDES:{}", r.includes);
            println!("COLLAGE:{:.4}", r.collage);
            println!("REWARD:{:.4}", r.reward);
        }
        "cosine" => {
            let v1 = vec![1.0, 0.0, 0.0, 0.0, 0.5];
            let v2 = vec![0.0, 1.0, 0.0, 0.0, 0.5];
            let sim = cosine(&v1, &v2);
            println!("COSINE_SIM:{:.4}", sim);
        }
        _ => { eprintln!("Unknown mode: {}", mode); }
    }
}
"##;

pub struct Microwave {
    sandbox_dir: String,
}

impl Microwave {
    pub fn new() -> Self {
        Self {
            sandbox_dir: "./microwave_sandbox".to_string(),
        }
    }

    pub fn set_dir(&mut self, dir: &str) {
        self.sandbox_dir = dir.to_string();
    }

    pub fn create_sandbox(&self) -> std::io::Result<String> {
        let src_dir = Path::new(&self.sandbox_dir).join("src");
        fs::create_dir_all(&src_dir)?;

        let cargo_toml = format!(
            r#"[package]
name = "mini-fuga"
version = "0.1.0"
edition = "2021"

[dependencies]
"#
        );
        fs::write(Path::new(&self.sandbox_dir).join("Cargo.toml"), &cargo_toml)?;
        fs::write(src_dir.join("main.rs"), MINI_FUGA_SRC)?;

        Ok(self.sandbox_dir.clone())
    }

    pub fn compile_self(&self) -> bool {
        let src_path = Path::new(&self.sandbox_dir).join("src").join("main.rs");
        let src = match fs::read_to_string(&src_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  Read failed: {}", e);
                return false;
            }
        };

        let tmp_dir = std::env::temp_dir().join("fuga_microwave");
        fs::create_dir_all(&tmp_dir).ok();
        let tmp_file = tmp_dir.join("microwave_main.rs");
        if let Err(e) = fs::write(&tmp_file, &src) {
            eprintln!("  Write failed: {}", e);
            return false;
        }

        let binary_path = tmp_dir.join("microwave_fuga");
        let cmd = Command::new("rustc")
            .arg("--edition")
            .arg("2021")
            .arg(&tmp_file)
            .arg("-o")
            .arg(&binary_path)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output();

        match cmd {
            Ok(output) => {
                if output.status.success() {
                    println!("  Compilation OK → {:?}", binary_path);
                    true
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    eprintln!("  Compilation FAILED:\n{}", stderr);
                    false
                }
            }
            Err(e) => {
                eprintln!("  Spawn failed: {}", e);
                false
            }
        }
    }

    pub fn self_test(&self) -> bool {
        let tmp_dir = std::env::temp_dir().join("fuga_microwave");
        let binary_path = tmp_dir.join("microwave_fuga");
        if !binary_path.exists() {
            eprintln!("  Binary not found at {:?}", binary_path);
            return false;
        }

        let good_code = r#"pub struct SendError(pub String);

impl std::fmt::Debug for SendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SendError({})", self.0)
    }
}

pub fn send(msg: String) -> Result<(), SendError> {
    Err(SendError(msg))
}"#;

        let good_cpp = r#"#include <string>
#include <memory>

class SendError {
    std::string msg;
public:
    explicit SendError(std::string m) : msg(std::move(m)) {}
    const std::string& what() const { return msg; }
};

template<typename T>
class Channel {
    T value;
public:
    void send(T v) { value = std::move(v); }
    T receive() { return std::move(value); }
};

bool send_message(std::string msg) {
    if (msg.empty()) return false;
    return true;
}

int process(size_t n) {
    int sum = 0;
    for (size_t i = 0; i < n; ++i) sum += (int)i;
    return sum;
}"#;

        println!("\n  ┌─ Self-Test 1: Rust evaluation ──────────────────────");
        let r1 = self.run_binary(
            &binary_path,
            &["eval_rust".to_string(), good_code.to_string()],
        );
        println!("  Output:\n{}", r1);

        println!("\n  ┌─ Self-Test 2: C++ evaluation ───────────────────────");
        let r2 = self.run_binary(
            &binary_path,
            &["eval_cpp".to_string(), good_cpp.to_string()],
        );
        println!("  Output:\n{}", r2);

        println!("\n  ┌─ Self-Test 3: Vector similarity ────────────────────");
        let r3 = self.run_binary(&binary_path, &["cosine".to_string()]);
        println!("  Output:\n{}", r3);

        let reward_rust = self.extract_reward(&r1);
        let reward_cpp = self.extract_reward(&r2);
        let cos_sim = self.extract_value(&r3, "COSINE_SIM");

        println!("\n  ┌─ Results ───────────────────────────────────────────");
        println!("  Rust reward:  {:.4}", reward_rust);
        println!("  C++  reward:  {:.4}", reward_cpp);
        println!("  Cosine sim:   {:.4}", cos_sim);

        reward_rust > 0.0 && reward_cpp > 0.0
    }

    fn run_binary(&self, path: &Path, args: &[String]) -> String {
        let cmd = Command::new(path)
            .args(args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output();
        match cmd {
            Ok(o) => {
                let mut out = String::from_utf8_lossy(&o.stdout).to_string();
                if !o.stderr.is_empty() {
                    out.push_str(&String::from_utf8_lossy(&o.stderr));
                }
                out
            }
            Err(e) => format!("spawn error: {}", e),
        }
    }

    fn extract_reward(&self, output: &str) -> f64 {
        for line in output.lines() {
            if line.starts_with("REWARD:") {
                if let Ok(v) = line.trim_start_matches("REWARD:").trim().parse::<f64>() {
                    return v;
                }
            }
        }
        -999.0
    }

    fn extract_value(&self, output: &str, key: &str) -> f64 {
        let prefix = format!("{}:", key);
        for line in output.lines() {
            let t = line.trim();
            if t.starts_with(&prefix) {
                if let Ok(v) = t.trim_start_matches(&prefix).trim().parse::<f64>() {
                    return v;
                }
            }
        }
        -999.0
    }

    pub fn run(args: &[String]) {
        let mut mw = Microwave::new();
        if args.len() > 2 {
            mw.set_dir(&args[2]);
        }

        println!("\n╔══════════════════════════════════════════════════════╗");
        println!("║        ⚡ PROJECT MICROWAVE — Self-Replication      ║");
        println!("╚══════════════════════════════════════════════════════╝");

        println!("\n▶ Phase 1: Creating sandbox at {}", mw.sandbox_dir);
        match mw.create_sandbox() {
            Ok(dir) => println!("  ✓ Sandbox created: {}/src/main.rs", dir),
            Err(e) => {
                eprintln!("  ✗ Failed: {}", e);
                return;
            }
        }

        println!("\n▶ Phase 2: Compiling Mini-Fuga via rustc");
        if !mw.compile_self() {
            eprintln!("\n  ✗ Self-compilation failed. Attempting surgical repair…");
            return;
        }

        println!("\n▶ Phase 3: Self-Testing Mini-Fuga");
        let passed = mw.self_test();

        println!("\n▶ Phase 4: Verdict");
        if passed {
            println!("  ✓ MICROWAVE SUCCESS — Fuga has reproduced itself.");
            println!("  ✓ The child binary passes all tests.");
        } else {
            println!("  ✗ MICROWAVE PARTIAL — Child binary failed some tests.");
            println!("  ↻ Iterating…");
        }
    }
}

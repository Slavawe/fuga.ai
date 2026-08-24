use std::process::Command;
use std::time::Duration;
use syn;

#[derive(Debug, Clone)]
pub struct SandboxResult {
    pub compiles: bool,
    pub error_count: usize,
    pub warning_count: usize,
    pub ast_nodes: usize,
    pub function_count: usize,
    pub struct_count: usize,
    pub impl_block_count: usize,
    pub collage_score: f64,
    pub reward: f64,
}

/// Raw rustc result including the full stderr text, so callers can extract the
/// failing token / expected token from diagnostics to drive targeted learning.
#[derive(Debug, Clone)]
pub struct SandboxDiagnostics {
    pub compiles: bool,
    pub error_count: usize,
    pub warning_count: usize,
    pub stderr: String,
}

#[derive(Debug, Clone)]
pub struct CppSandboxResult {
    pub compiles: bool,
    pub error_count: usize,
    pub warning_count: usize,
    pub function_count: usize,
    pub class_count: usize,
    pub template_count: usize,
    pub include_count: usize,
    pub collage_score: f64,
    pub reward: f64,
}

#[derive(Debug, Default)]
struct AstStats {
    functions: usize,
    structs: usize,
    enums: usize,
    impl_blocks: usize,
    traits: usize,
    modules: usize,
    macros: usize,
    total_nodes: usize,
}

pub struct SandboxHarness {
    temp_dir: std::path::PathBuf,
    rustc_timeout: Duration,
}

impl SandboxHarness {
    pub fn new() -> Self {
        let temp_dir = std::env::temp_dir().join("fuga_sandbox");
        std::fs::create_dir_all(&temp_dir).ok();
        Self {
            temp_dir,
            rustc_timeout: Duration::from_secs(30),
        }
    }

    pub fn evaluate(&self, code: &str, file_name: &str) -> SandboxResult {
        let file_path = self.temp_dir.join(file_name);
        let crate_name = file_name.trim_end_matches(".rs");

        if let Err(e) = std::fs::write(&file_path, code) {
            return SandboxResult::failed(format!("write error: {}", e));
        }

        let compile_result = self.compile_crate(&file_path, crate_name);
        let ast_stats = self.analyze_ast(code);
        let collage_score = self.compute_collage_score(&ast_stats, code);

        SandboxResult {
            compiles: compile_result.success,
            error_count: compile_result.errors,
            warning_count: compile_result.warnings,
            ast_nodes: ast_stats.total_nodes,
            function_count: ast_stats.functions,
            struct_count: ast_stats.structs,
            impl_block_count: ast_stats.impl_blocks,
            collage_score,
            reward: self.compute_reward(&compile_result, &ast_stats, collage_score),
        }
    }

    pub fn evaluate_cpp(&self, code: &str, file_name: &str) -> CppSandboxResult {
        let file_path = self.temp_dir.join(file_name);

        if let Err(e) = std::fs::write(&file_path, code) {
            return CppSandboxResult::failed(format!("write error: {}", e));
        }

        let compile_result = self.compile_cpp(&file_path);
        let stats = self.analyze_cpp_ast(code);
        let collage_score = self.compute_cpp_collage(&stats, code);

        CppSandboxResult {
            compiles: compile_result.success,
            error_count: compile_result.errors,
            warning_count: compile_result.warnings,
            function_count: stats.functions,
            class_count: stats.classes,
            template_count: stats.templates,
            include_count: stats.includes,
            collage_score,
            reward: self.compute_cpp_reward(&compile_result, &stats, collage_score),
        }
    }

    /// Compile-only check that also returns the raw stderr, so the caller can
    /// extract the expected token from diagnostics to target learning.
    pub fn evaluate_diagnostics(&self, code: &str, file_name: &str) -> SandboxDiagnostics {
        let file_path = self.temp_dir.join(file_name);
        let crate_name = file_name.trim_end_matches(".rs");

        if let Err(e) = std::fs::write(&file_path, code) {
            return SandboxDiagnostics {
                compiles: false,
                error_count: 1,
                warning_count: 0,
                stderr: format!("write error: {}", e),
            };
        }

        let compile_result = self.compile_crate(&file_path, crate_name);
        SandboxDiagnostics {
            compiles: compile_result.success,
            error_count: compile_result.errors,
            warning_count: compile_result.warnings,
            stderr: compile_result.stderr,
        }
    }

    fn compile_cpp(&self, file_path: &std::path::Path) -> CppCompileResult {
        let mut cmd = Command::new("g++");
        cmd.arg("-std=c++17")
            .arg("-Wall")
            .arg("-Wno-unused-variable")
            .arg("-fsyntax-only")
            .arg(file_path)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let output = match cmd.output() {
            Ok(o) => o,
            Err(e) => return CppCompileResult::failed(format!("spawn error: {}", e)),
        };

        let stderr = String::from_utf8_lossy(&output.stderr);
        let errors = stderr.matches("error:").count();
        let warnings = stderr.matches("warning:").count();

        CppCompileResult {
            success: output.status.success() && errors == 0,
            errors,
            warnings,
            stderr: stderr.to_string(),
        }
    }

    fn analyze_cpp_ast(&self, code: &str) -> CppAstStats {
        let mut stats = CppAstStats::default();
        for line in code.lines() {
            let t = line.trim();
            if t.contains("#include") {
                stats.includes += 1;
            }
            if t.contains("template") && (t.contains('<') || t.contains("typename")) {
                stats.templates += 1;
            }
            if t.starts_with("class ") || t.starts_with("struct ") {
                stats.classes += 1;
            }
            // Detect function signatures: type name(...) {
            if (t.starts_with("void ")
                || t.starts_with("int ")
                || t.starts_with("bool ")
                || t.starts_with("auto ")
                || t.starts_with("std::")
                || t.starts_with("static ")
                || t.starts_with("size_t ")
                || t.starts_with("const "))
                && t.contains('(')
                && t.contains(')')
                && !t.contains(";")
            {
                let sig_part = t.split('{').next().unwrap_or(t).trim();
                if !sig_part.is_empty() {
                    stats.functions += 1;
                    stats.signatures.push(sig_part.to_string());
                }
            }
        }
        stats
    }

    fn compute_cpp_collage(&self, stats: &CppAstStats, _code: &str) -> f64 {
        if stats.signatures.is_empty() {
            return 1.0;
        }
        let mut seen = std::collections::HashSet::new();
        let mut duplicates = 0;
        for sig in &stats.signatures {
            if !seen.insert(sig.clone()) {
                duplicates += 1;
            }
        }
        let dup_ratio = duplicates as f64 / stats.signatures.len() as f64;
        dup_ratio.clamp(0.1, 1.0)
    }

    fn compute_cpp_reward(
        &self,
        compile: &CppCompileResult,
        ast: &CppAstStats,
        collage: f64,
    ) -> f64 {
        let mut reward = 0.0;
        if compile.success {
            reward += 2.0;
        } else {
            reward -= compile.errors as f64 * 0.5;
        }
        reward -= compile.warnings as f64 * 0.1;
        if ast.functions > 0 {
            reward += 0.4 * (ast.functions as f64).ln().max(0.0);
        }
        if ast.classes > 0 {
            reward += 0.5 * (ast.classes as f64).ln().max(0.0);
        }
        if ast.templates > 0 {
            reward += 0.3 * (ast.templates as f64).ln().max(0.0);
        }
        if ast.includes > 0 {
            reward += 0.05 * ast.includes as f64;
        }
        reward -= collage * 2.0;
        reward.max(-10.0).min(10.0)
    }

    fn compile_crate(&self, file_path: &std::path::Path, crate_name: &str) -> CompileResult {
        let mut cmd = Command::new("rustc");
        cmd.arg("--edition")
            .arg("2021")
            .arg("--crate-type")
            .arg("lib")
            .arg("--crate-name")
            .arg(crate_name)
            .arg(file_path)
            .arg("-o")
            .arg(self.temp_dir.join(format!("{}.rlib", crate_name)))
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let output = match cmd.output() {
            Ok(o) => o,
            Err(e) => return CompileResult::failed(format!("spawn error: {}", e)),
        };

        let stderr = String::from_utf8_lossy(&output.stderr);
        let errors = stderr.matches("error[").count() + stderr.matches("error:").count();
        let warnings = stderr.matches("warning[").count() + stderr.matches("warning:").count();

        CompileResult {
            success: output.status.success() && errors == 0,
            errors,
            warnings,
            stderr: stderr.to_string(),
        }
    }

    fn analyze_ast(&self, code: &str) -> AstStats {
        let mut stats = AstStats::default();

        if let Ok(file) = syn::parse_file(code) {
            for item in file.items {
                match item {
                    syn::Item::Fn(_) => stats.functions += 1,
                    syn::Item::Struct(_) => stats.structs += 1,
                    syn::Item::Enum(_) => stats.enums += 1,
                    syn::Item::Impl(_) => stats.impl_blocks += 1,
                    syn::Item::Trait(_) => stats.traits += 1,
                    syn::Item::Mod(_) => stats.modules += 1,
                    syn::Item::Macro(_) => stats.macros += 1,
                    _ => {}
                }
                stats.total_nodes += 1;

                if let syn::Item::Impl(imp) = item {
                    stats.total_nodes += imp.items.len();
                }
            }
        }

        stats
    }

    fn compute_collage_score(&self, stats: &AstStats, code: &str) -> f64 {
        if stats.total_nodes == 0 {
            return 1.0;
        }

        let fragment_count = code
            .lines()
            .filter(|l| {
                l.trim().starts_with("fn ")
                    || l.trim().starts_with("struct ")
                    || l.trim().starts_with("impl ")
            })
            .count();
        let expected_structured = stats.functions + stats.structs + stats.impl_blocks;

        if expected_structured == 0 {
            return 1.0;
        }

        let ratio = fragment_count as f64 / expected_structured as f64;

        // Additional collage signals
        let lines: Vec<&str> = code.lines().collect();
        let semicolon_in_comments = lines
            .iter()
            .filter(|l| l.trim().starts_with("//") && l.contains(';'))
            .count();
        let duplicate_fn_sigs = self.count_duplicate_fn_signatures(&lines);
        let duplicate_structs = self.count_duplicate_structs(&lines);

        let base_score = if ratio > 3.0 {
            0.8 + (ratio - 3.0).min(1.0) * 0.2
        } else if ratio > 1.5 {
            0.4 + (ratio - 1.5) / 1.5 * 0.4
        } else {
            0.1
        };

        let penalty = (semicolon_in_comments as f64 * 0.15)
            + (duplicate_fn_sigs as f64 * 0.2)
            + (duplicate_structs as f64 * 0.25);

        (base_score + penalty).min(1.0)
    }

    fn count_duplicate_fn_signatures(&self, lines: &[&str]) -> usize {
        let mut seen = std::collections::HashSet::new();
        let mut duplicates = 0;
        for line in lines {
            let trimmed = line.trim();
            if trimmed.starts_with("fn ") {
                let sig = trimmed.split('{').next().unwrap_or(trimmed).trim();
                if !seen.insert(sig) {
                    duplicates += 1;
                }
            }
        }
        duplicates
    }

    fn count_duplicate_structs(&self, lines: &[&str]) -> usize {
        let mut seen = std::collections::HashSet::new();
        let mut duplicates = 0;
        for line in lines {
            let trimmed = line.trim();
            if trimmed.starts_with("struct ") {
                let sig = trimmed.split('{').next().unwrap_or(trimmed).trim();
                if !seen.insert(sig) {
                    duplicates += 1;
                }
            }
        }
        duplicates
    }

    fn compute_reward(&self, compile: &CompileResult, ast: &AstStats, collage: f64) -> f64 {
        let mut reward = 0.0;

        if compile.success {
            reward += 2.0;
        } else {
            reward -= compile.errors as f64 * 0.5;
        }

        reward -= compile.warnings as f64 * 0.1;

        if ast.functions > 0 {
            reward += 0.5 * (ast.functions as f64).ln().max(0.0);
        }
        if ast.structs > 0 {
            reward += 0.3 * (ast.structs as f64).ln().max(0.0);
        }
        if ast.impl_blocks > 0 {
            reward += 0.4 * (ast.impl_blocks as f64).ln().max(0.0);
        }

        reward -= collage * 2.0;

        reward.max(-10.0).min(10.0)
    }
}

#[derive(Debug)]
struct CompileResult {
    success: bool,
    errors: usize,
    warnings: usize,
    stderr: String,
}

impl CompileResult {
    fn failed(msg: String) -> Self {
        Self {
            success: false,
            errors: 1,
            warnings: 0,
            stderr: msg,
        }
    }
}

impl SandboxResult {
    fn failed(_msg: String) -> Self {
        Self {
            compiles: false,
            error_count: 1,
            warning_count: 0,
            ast_nodes: 0,
            function_count: 0,
            struct_count: 0,
            impl_block_count: 0,
            collage_score: 1.0,
            reward: -5.0,
        }
    }
}

#[derive(Debug, Default)]
struct CppAstStats {
    functions: usize,
    classes: usize,
    templates: usize,
    includes: usize,
    signatures: Vec<String>,
}

#[derive(Debug)]
struct CppCompileResult {
    success: bool,
    errors: usize,
    warnings: usize,
    stderr: String,
}

impl CppCompileResult {
    fn failed(msg: String) -> Self {
        Self {
            success: false,
            errors: 1,
            warnings: 0,
            stderr: msg,
        }
    }
}

impl CppSandboxResult {
    fn failed(_msg: String) -> Self {
        Self {
            compiles: false,
            error_count: 1,
            warning_count: 0,
            function_count: 0,
            class_count: 0,
            template_count: 0,
            include_count: 0,
            collage_score: 1.0,
            reward: -5.0,
        }
    }

    pub fn display(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("  compiles: {}\n", self.compiles));
        out.push_str(&format!("  errors: {}\n", self.error_count));
        out.push_str(&format!("  warnings: {}\n", self.warning_count));
        out.push_str(&format!("  functions: {}\n", self.function_count));
        out.push_str(&format!("  classes: {}\n", self.class_count));
        out.push_str(&format!("  templates: {}\n", self.template_count));
        out.push_str(&format!("  includes: {}\n", self.include_count));
        out.push_str(&format!("  collage: {:.2}\n", self.collage_score));
        out.push_str(&format!("  reward: {:.2}\n", self.reward));
        out
    }
}

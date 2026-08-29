//! Analyze / fix / translate command implementations.
//!
//! Extracted from `src/main.rs` during monolith decomposition
//! (Phase 1 of docs/refactor-plan.md). Self-contained cluster:
//! depends only on `fuga` crate types, std, and sibling functions
//! in this module.

use std::fs;
use std::path::Path;
use std::process;

use fuga::{
    AnalysisResult, CodeTranslator, FileAnalysisResult, FixProposal, FugaEngine, HtmlReporter,
    JsonReporter, LanguageId, MarkdownReporter, MultiEngine, MultiFixGenerator, OutputFormat,
    PatchGenerator, Reporter, ScanMode, WorkspaceScanner, WorkspaceStats,
};

/// Scan a path and emit an analysis report in the requested format.
pub fn run_analyze(path: &str, dim: usize, mode: ScanMode, format: OutputFormat, output: Option<&str>) {
    println!("Fuga 1.0 — Scanning...");

    let scanner = WorkspaceScanner::new();
    let files = match scanner.scan(path, mode) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Scan error: {}", e);
            process::exit(3);
        }
    };

    println!("   Found {} file(s)", files.len());
    println!("   Dimension: {}D", dim);
    println!();

    let mut results = Vec::new();
    let mut engine = FugaEngine::new(dim);
    let mut multi_engine = MultiEngine::new(dim);

    for file_path in &files {
        let path_str = file_path.display().to_string();
        let lang = LanguageId::from_path(file_path);
        print!("Analyzing {}... ", path_str);

        match lang {
            Some(LanguageId::Rust) => match engine.analyze_file(&path_str) {
                Ok(result) => {
                    println!("OK");
                    results.push(FileAnalysisResult {
                        file_path: path_str,
                        result: AnalysisResult::Rust(result),
                        error: None,
                    });
                }
                Err(e) => {
                    println!("ERR ({})", e);
                    results.push(FileAnalysisResult {
                        file_path: path_str,
                        result: AnalysisResult::Rust(
                            FugaEngine::new(dim)
                                .analyze("// dummy\nfn main() {}")
                                .unwrap(),
                        ),
                        error: Some(e.to_string()),
                    });
                }
            },
            Some(lang) => match multi_engine.analyze_file(&path_str) {
                Ok(result) => {
                    println!("OK");
                    results.push(FileAnalysisResult {
                        file_path: path_str,
                        result: AnalysisResult::Multi(result),
                        error: None,
                    });
                }
                Err(e) => {
                    println!("ERR ({})", e);
                    results.push(FileAnalysisResult {
                        file_path: path_str.clone(),
                        result: AnalysisResult::Multi(
                            multi_engine.analyze("// dummy", lang, &path_str),
                        ),
                        error: Some(e.to_string()),
                    });
                }
            },
            None => {
                println!("ERR (unsupported language)");
                results.push(FileAnalysisResult {
                    file_path: path_str.clone(),
                    result: AnalysisResult::Multi(multi_engine.analyze(
                        "// dummy",
                        LanguageId::Rust,
                        &path_str,
                    )),
                    error: Some("Unsupported file extension".to_string()),
                });
            }
        }
    }

    println!();

    let report = match format {
        OutputFormat::Text => generate_text_report(&results),
        OutputFormat::Json => JsonReporter::new().generate_report(&results),
        OutputFormat::Html => HtmlReporter::new().generate_report(&results),
        OutputFormat::Markdown => MarkdownReporter::new().generate_report(&results),
    };

    if let Some(output_path) = output {
        match fs::write(output_path, &report) {
            Ok(_) => println!("Report written to: {}", output_path),
            Err(e) => {
                eprintln!("Failed to write report: {}", e);
                process::exit(3);
            }
        }
    } else {
        println!("{}", report);
    }

    let had_errors = results.iter().any(|r| r.error.is_some());
    if had_errors {
        process::exit(3);
    }
    let stats = WorkspaceStats::from_results(&results);
    process::exit(stats.exit_code());
}

/// Build the plain-text workspace summary report.
fn generate_text_report(results: &[FileAnalysisResult]) -> String {
    let stats = WorkspaceStats::from_results(results);
    let mut report = String::new();

    report.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    report.push_str("  Fuga 1.0 — Workspace Analysis Summary\n");
    report.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    report.push_str(&format!("Files:       {}\n", stats.total_files));
    report.push_str(&format!("Lines:       {}\n", stats.total_lines));
    report.push_str(&format!("Functions:   {}\n", stats.total_functions));
    report.push_str(&format!("Violations:  {}\n", stats.total_violations));
    report.push_str(&format!("Bugs:        {}\n", stats.total_bugs));
    report.push_str(&format!(
        "Avg Safety:  {:.1}%\n",
        stats.avg_safety_score * 100.0
    ));

    if let Some((path, score)) = &stats.worst_file {
        report.push_str(&format!(
            "\nWorst file: {} (safety: {:.1}%)\n",
            path,
            score * 100.0
        ));
    }

    report.push_str("\n");

    let status = match stats.exit_code() {
        0 => "CLEAN",
        1 => "WARNINGS",
        2 => "BUGS DETECTED",
        _ => "ERRORS",
    };
    report.push_str(&format!("Status: {}\n", status));

    report.push_str("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    for file_result in results {
        if let Some(ref err) = file_result.error {
            report.push_str(&format!("ERR {}: {}\n", file_result.file_path, err));
            continue;
        }

        let result = &file_result.result;
        let safety = result.safety_score();
        let icon = if safety > 0.8 {
            "OK"
        } else if safety > 0.5 {
            "WARN"
        } else {
            "LOW"
        };

        report.push_str(&format!(
            "{} {} (safety: {:.1}%)\n",
            icon,
            file_result.file_path,
            safety * 100.0
        ));

        if !result.violations_is_empty() {
            report.push_str(&format!("   Violations: {}\n", result.violations_count()));
        }

        if result.bug_detected() {
            report.push_str(&format!(
                "   Bug detected (conf: {:.1}%)\n",
                result.bug_confidence() * 100.0
            ));
        }

        report.push_str("\n");
    }

    report
}

/// Autofix a single file: analyze, propose fixes, print/save the patch.
pub fn run_fix(path: &str, dim: usize, output: Option<&str>) {
    let lang = LanguageId::from_path(Path::new(path));

    match lang {
        Some(LanguageId::Rust) => {
            let mut engine = FugaEngine::new(dim);

            println!("Fuga Autofix — Analyzing: {}", path);
            println!("   Dimension: {}D", dim);
            println!();

            let result = match engine.analyze_file(path) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    process::exit(1);
                }
            };

            let source = match fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to read file: {}", e);
                    process::exit(1);
                }
            };

            let proposals = engine.generate_fixes(&source, &result);

            if proposals.is_empty() {
                println!("No fixes needed — code is clean!");
                return;
            }

            print_and_save_patch(path, &source, &proposals, output);
        }
        Some(lang) => {
            let mut multi_engine = MultiEngine::new(dim);

            println!("Fuga Autofix — Analyzing: {}", path);
            println!("   Language: {}", lang.name());
            println!("   Dimension: {}D", dim);
            println!();

            let result = match multi_engine.analyze_file(path) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    process::exit(1);
                }
            };

            let source = match fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to read file: {}", e);
                    process::exit(1);
                }
            };

            let fixer = MultiFixGenerator::new();
            let proposals = fixer.generate_fixes(&source, &result.syntax.violations, lang);

            if proposals.is_empty() {
                println!("No fixes needed — code is clean!");
                return;
            }

            print_and_save_patch(path, &source, &proposals, output);
        }
        None => {
            eprintln!("Unsupported file extension: {}", path);
            process::exit(1);
        }
    }
}

/// Print fix proposals and optionally save the generated diff.
fn print_and_save_patch(path: &str, source: &str, proposals: &[FixProposal], output: Option<&str>) {
    println!("Found {} fix proposals:", proposals.len());
    println!();

    for (i, proposal) in proposals.iter().enumerate() {
        println!(
            "{}. {} (confidence: {:.0}%)",
            i + 1,
            proposal.description,
            proposal.confidence * 100.0
        );
        if let Some(ref snippet) = proposal.location.code_snippet {
            println!("   Original: {}", snippet);
        }
        println!("   Strategy: {:?}", proposal.strategy);
        println!();
    }

    let patch_generator = PatchGenerator::new();
    let diff = patch_generator.generate_patch(path, source, proposals);

    if let Some(output_path) = output {
        match fs::write(output_path, &diff.diff_text) {
            Ok(_) => println!("Patch written to: {}", output_path),
            Err(e) => {
                eprintln!("Failed to write patch: {}", e);
                process::exit(1);
            }
        }
    } else {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("{}", diff.diff_text);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();
        println!("To apply: patch {} < patch.diff", path);
        println!("To save:  fuga fix {} --output patch.diff", path);
    }
}

/// Translate a file from its detected language to `target`.
pub fn run_translate(path: &str, target: &str) {
    let lang = LanguageId::from_path(Path::new(path)).unwrap_or(LanguageId::Rust);
    let target_lang = match target.to_lowercase().as_str() {
        "rust" | "rs" => LanguageId::Rust,
        "c" => LanguageId::C,
        "cpp" | "cxx" | "c++" => LanguageId::Cpp,
        "go" => LanguageId::Go,
        "python" | "py" => LanguageId::Python,
        "typescript" | "ts" => LanguageId::TypeScript,
        "javascript" | "js" => LanguageId::JavaScript,
        other => {
            eprintln!("Unsupported target language: {}", other);
            process::exit(1);
        }
    };

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read file: {}", e);
            process::exit(1);
        }
    };

    let translator = CodeTranslator::new();
    match translator.translate(&source, lang, target_lang) {
        Ok(output) => {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("  Translation: {:?} → {:?}", lang, target_lang);
            println!("  Source: {}", path);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("{}", output);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Note: Auto-translated code requires manual review");
        }
        Err(e) => {
            eprintln!("Translation failed: {}", e);
            process::exit(1);
        }
    }
}

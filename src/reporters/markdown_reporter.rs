use super::{Reporter, FileAnalysisResult, WorkspaceStats};

pub struct MarkdownReporter;

impl MarkdownReporter {
    pub fn new() -> Self {
        Self
    }
}

impl Reporter for MarkdownReporter {
    fn generate_report(&self, results: &[FileAnalysisResult]) -> String {
        let stats = WorkspaceStats::from_results(results);

        let mut md = String::new();

        // Header
        md.push_str("# 🎵 Fuga 1.0 — Analysis Report\n\n");

        // Summary
        md.push_str("## Summary\n\n");
        md.push_str("| Metric | Value |\n");
        md.push_str("|--------|-------|\n");
        md.push_str(&format!("| Files | {} |\n", stats.total_files));
        md.push_str(&format!("| Lines | {} |\n", stats.total_lines));
        md.push_str(&format!("| Functions | {} |\n", stats.total_functions));
        md.push_str(&format!("| Violations | {} |\n", stats.total_violations));
        md.push_str(&format!("| Bugs | {} |\n", stats.total_bugs));
        md.push_str(&format!("| Avg Safety | {:.1}% |\n", stats.avg_safety_score * 100.0));
        
        if let Some((path, score)) = &stats.worst_file {
            md.push_str(&format!("\n⚠️ **Worst file**: `{}` (safety: {:.1}%)\n", path, score * 100.0));
        }

        md.push_str("\n");

        // Exit code
        let exit_msg = match stats.exit_code() {
            0 => "✅ **Status**: Clean",
            1 => "⚠️ **Status**: Warnings found",
            2 => "🐛 **Status**: Bugs detected",
            _ => "❌ **Status**: Errors",
        };
        md.push_str(&format!("{}\n\n", exit_msg));

        // File details
        md.push_str("## File Details\n\n");

        for file_result in results {
            if let Some(ref err) = file_result.error {
                md.push_str(&format!("### ❌ `{}`\n\n", file_result.file_path));
                md.push_str(&format!("**Error**: {}\n\n", err));
                continue;
            }

            let result = &file_result.result;
            let safety = result.safety_score();
            let icon = if safety > 0.8 { "✅" } else if safety > 0.5 { "⚠️" } else { "🔴" };

            md.push_str(&format!("### {} `{}`\n\n", icon, file_result.file_path));
            md.push_str(&format!("- **Lines**: {}\n", result.lines()));
            md.push_str(&format!("- **Functions**: {}\n", result.functions()));
            md.push_str(&format!("- **Safety Score**: {:.1}%\n", safety * 100.0));
            md.push_str("\n");

            // Layer 1: Syntax
            md.push_str("#### Layer 1: Syntax & Invariants\n\n");
            let violations = result.violations();
            if violations.is_empty() {
                md.push_str("✅ No violations\n\n");
            } else {
                md.push_str(&format!("Found {} violation(s):\n\n", violations.len()));
                for v in &violations {
                    let sev_icon = match v.severity.as_str() {
                        "Critical" => "🔴",
                        "High" => "🟠",
                        "Medium" => "🟡",
                        "Low" => "🔵",
                        _ => "⚪",
                    };
                    md.push_str(&format!("- {} **{}**: {}\n", sev_icon, v.kind, v.message));
                }
                md.push_str("\n");
            }

            // Layer 2: Semantics
            md.push_str("#### Layer 2: Semantics & VSA\n\n");
            md.push_str(&format!("- **Coherence**: {:.1}%\n", result.coherence() * 100.0));
            md.push_str(&format!("- **Anomalies**: {}\n\n", result.anomalies_count()));

            // Layer 3: Chaos
            md.push_str("#### Layer 3: Chaos & Mutations\n\n");
            let attacks = result.attacks();
            if attacks.is_empty() {
                md.push_str("No attack vectors generated\n\n");
            } else {
                md.push_str(&format!("Generated {} attack(s):\n\n", attacks.len()));
                for a in &attacks {
                    md.push_str(&format!("- 🎯 **{}** (priority: {:.0}%): {}\n", a.kind, a.priority * 100.0, a.description));
                }
                md.push_str("\n");
            }

            // Fuga Synthesis
            if result.bug_confidence() > 0.0 || result.bug_detected() {
                md.push_str("#### Fuga Synthesis\n\n");
                if result.bug_detected() {
                    md.push_str(&format!("🐛 **BUG DETECTED** (confidence: {:.1}%)\n\n", result.bug_confidence() * 100.0));
                    let desc = result.counterpoint_description();
                    if !desc.is_empty() {
                        md.push_str(&format!("> {}\n\n", desc));
                    }
                } else {
                    md.push_str(&format!("✅ **CLEAN** (score: {:.1}%)\n\n", result.bug_confidence() * 100.0));
                }
            }

            md.push_str("---\n\n");
        }

        md
    }
}
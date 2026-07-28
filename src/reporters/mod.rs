pub mod json_reporter;
pub mod html_reporter;
pub mod markdown_reporter;
pub mod workspace;

use crate::engine::AnalysisResult;

/// Формат вывода отчёта
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    Text,
    Json,
    Html,
    Markdown,
}

impl OutputFormat {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "text" | "txt" => Some(Self::Text),
            "json" => Some(Self::Json),
            "html" | "htm" => Some(Self::Html),
            "markdown" | "md" => Some(Self::Markdown),
            _ => None,
        }
    }
}

/// Trait для всех репортеров
pub trait Reporter {
    fn generate_report(&self, results: &[FileAnalysisResult]) -> String;
}

/// Результат анализа одного файла
#[derive(Debug, Clone)]
pub struct FileAnalysisResult {
    pub file_path: String,
    pub result: AnalysisResult,
    pub error: Option<String>,
}

/// Агрегированная статистика workspace
#[derive(Debug, Clone, Default)]
pub struct WorkspaceStats {
    pub total_files: usize,
    pub total_lines: usize,
    pub total_functions: usize,
    pub total_violations: usize,
    pub total_bugs: usize,
    pub total_attacks: usize,
    pub avg_safety_score: f64,
    pub worst_file: Option<(String, f64)>, // (path, safety_score)
}

impl WorkspaceStats {
    pub fn from_results(results: &[FileAnalysisResult]) -> Self {
        let mut stats = Self::default();
        stats.total_files = results.len();

        let mut total_safety = 0.0;
        let mut worst_safety = 1.0;
        let mut worst_path = None;

        for file_result in results {
            if file_result.error.is_some() {
                continue;
            }

            let result = &file_result.result;
            stats.total_lines += result.lines();
            stats.total_functions += result.functions();
            stats.total_violations += result.violations_count();
            stats.total_attacks += result.attacks_count();

            if result.bug_detected() {
                stats.total_bugs += 1;
            }

            let safety = result.safety_score();
            total_safety += safety;

            if safety < worst_safety {
                worst_safety = safety;
                worst_path = Some(file_result.file_path.clone());
            }
        }

        stats.avg_safety_score = if stats.total_files > 0 {
            total_safety / stats.total_files as f64
        } else {
            0.0
        };

        if let Some(path) = worst_path {
            stats.worst_file = Some((path, worst_safety));
        }

        stats
    }

    pub fn exit_code(&self) -> i32 {
        if self.total_bugs > 0 {
            2 // Bugs found
        } else if self.total_violations > 0 {
            1 // Warnings (violations without bugs)
        } else {
            0 // Clean
        }
    }
}

pub use json_reporter::JsonReporter;
pub use html_reporter::HtmlReporter;
pub use markdown_reporter::MarkdownReporter;
pub use workspace::{WorkspaceScanner, ScanMode};
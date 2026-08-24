use super::{FileAnalysisResult, Reporter, WorkspaceStats};
use serde::{Deserialize, Serialize};

pub struct JsonReporter;

impl JsonReporter {
    pub fn new() -> Self {
        Self
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct JsonReport {
    fuga_version: String,
    summary: SummaryStats,
    files: Vec<JsonFileResult>,
}

#[derive(Serialize, Deserialize, Debug)]
struct SummaryStats {
    total_files: usize,
    total_lines: usize,
    total_functions: usize,
    total_violations: usize,
    total_bugs: usize,
    avg_safety_score: f64,
    worst_file: Option<String>,
    exit_code: i32,
}

#[derive(Serialize, Deserialize, Debug)]
struct JsonFileResult {
    path: String,
    lines: usize,
    functions: usize,
    safety_score: f64,
    violations: usize,
    anomalies: usize,
    attacks: usize,
    bug_detected: bool,
    bug_confidence: Option<f64>,
}

impl Reporter for JsonReporter {
    fn generate_report(&self, results: &[FileAnalysisResult]) -> String {
        let stats = WorkspaceStats::from_results(results);

        let files: Vec<JsonFileResult> = results
            .iter()
            .filter(|r| r.error.is_none())
            .map(|r| {
                let result = &r.result;
                JsonFileResult {
                    path: r.file_path.clone(),
                    lines: result.lines(),
                    functions: result.functions(),
                    safety_score: result.safety_score(),
                    violations: result.violations_count(),
                    anomalies: result.anomalies_count(),
                    attacks: result.attacks_count(),
                    bug_detected: result.bug_detected(),
                    bug_confidence: Some(result.bug_confidence()).filter(|&c| c > 0.0),
                }
            })
            .collect();

        let report = JsonReport {
            fuga_version: "0.1.0".to_string(),
            summary: SummaryStats {
                total_files: stats.total_files,
                total_lines: stats.total_lines,
                total_functions: stats.total_functions,
                total_violations: stats.total_violations,
                total_bugs: stats.total_bugs,
                avg_safety_score: stats.avg_safety_score,
                worst_file: stats.worst_file.as_ref().map(|(p, _)| p.clone()),
                exit_code: stats.exit_code(),
            },
            files,
        };

        serde_json::to_string_pretty(&report)
            .unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e))
    }
}

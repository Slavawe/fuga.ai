use super::{FileAnalysisResult, Reporter, WorkspaceStats};

pub struct HtmlReporter;

impl HtmlReporter {
    pub fn new() -> Self {
        Self
    }
}

impl Reporter for HtmlReporter {
    fn generate_report(&self, results: &[FileAnalysisResult]) -> String {
        let stats = WorkspaceStats::from_results(results);

        let mut html = String::new();
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<meta charset=\"UTF-8\">\n");
        html.push_str("<title>Fuga Analysis Report</title>\n");
        html.push_str("<style>\n");
        html.push_str(include_str!("html_style.css"));
        html.push_str("</style>\n</head>\n<body>\n");

        // Header
        html.push_str("<div class=\"header\">\n");
        html.push_str("<h1>🎵 Fuga 1.0 — Analysis Report</h1>\n");
        html.push_str(&format!(
            "<p class=\"subtitle\">Generated: {}</p>\n",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
        ));
        html.push_str("</div>\n");

        // Summary
        html.push_str("<div class=\"summary\">\n");
        html.push_str("<h2>Summary</h2>\n");
        html.push_str("<div class=\"stats-grid\">\n");
        html.push_str(&format!("<div class=\"stat\"><span class=\"stat-label\">Files</span><span class=\"stat-value\">{}</span></div>\n", stats.total_files));
        html.push_str(&format!("<div class=\"stat\"><span class=\"stat-label\">Lines</span><span class=\"stat-value\">{}</span></div>\n", stats.total_lines));
        html.push_str(&format!("<div class=\"stat\"><span class=\"stat-label\">Functions</span><span class=\"stat-value\">{}</span></div>\n", stats.total_functions));
        html.push_str(&format!("<div class=\"stat\"><span class=\"stat-label\">Violations</span><span class=\"stat-value\">{}</span></div>\n", stats.total_violations));
        html.push_str(&format!("<div class=\"stat\"><span class=\"stat-label\">Bugs</span><span class=\"stat-value {}\">{}</span></div>\n", 
            if stats.total_bugs > 0 { "stat-value-bad" } else { "stat-value-good" }, stats.total_bugs));
        html.push_str(&format!("<div class=\"stat\"><span class=\"stat-label\">Avg Safety</span><span class=\"stat-value\">{:.1}%</span></div>\n", stats.avg_safety_score * 100.0));
        html.push_str("</div>\n");

        if let Some((path, score)) = &stats.worst_file {
            html.push_str(&format!(
                "<p class=\"worst-file\">⚠️ Worst file: <code>{}</code> (safety: {:.1}%)</p>\n",
                path,
                score * 100.0
            ));
        }

        html.push_str("</div>\n");

        // File details
        html.push_str("<div class=\"files\">\n<h2>File Details</h2>\n");

        for file_result in results {
            if let Some(ref err) = file_result.error {
                html.push_str(&format!("<div class=\"file error\"><h3>{}</h3><p class=\"error-msg\">❌ Error: {}</p></div>\n", 
                    file_result.file_path, err));
                continue;
            }

            let result = &file_result.result;
            let safety = result.safety_score();
            let safety_class = if safety > 0.8 {
                "good"
            } else if safety > 0.5 {
                "warning"
            } else {
                "bad"
            };

            html.push_str(&format!("<div class=\"file {}\">\n", safety_class));
            html.push_str(&format!("<h3>{}</h3>\n", file_result.file_path));
            html.push_str("<div class=\"file-stats\">\n");
            html.push_str(&format!("<span>Lines: {}</span>\n", result.lines()));
            html.push_str(&format!("<span>Functions: {}</span>\n", result.functions()));
            html.push_str(&format!("<span>Safety: {:.1}%</span>\n", safety * 100.0));
            html.push_str("</div>\n");

            // Layer 1: Syntax
            html.push_str("<div class=\"layer\">\n");
            html.push_str("<h4>Layer 1: Syntax & Invariants</h4>\n");
            let violations = result.violations();
            html.push_str(&format!("<p>Violations: {}</p>\n", violations.len()));
            if !violations.is_empty() {
                html.push_str("<ul class=\"violations\">\n");
                for v in &violations {
                    html.push_str(&format!(
                        "<li><span class=\"sev-{}\">{}</span>: {}</li>\n",
                        v.severity, v.kind, v.message
                    ));
                }
                html.push_str("</ul>\n");
            }
            html.push_str("</div>\n");

            // Layer 2: Semantics
            html.push_str("<div class=\"layer\">\n");
            html.push_str("<h4>Layer 2: Semantics & VSA</h4>\n");
            html.push_str(&format!(
                "<p>Coherence: {:.1}%</p>\n",
                result.coherence() * 100.0
            ));
            html.push_str(&format!("<p>Anomalies: {}</p>\n", result.anomalies_count()));
            html.push_str("</div>\n");

            // Layer 3: Chaos
            html.push_str("<div class=\"layer\">\n");
            html.push_str("<h4>Layer 3: Chaos & Mutations</h4>\n");
            let attacks = result.attacks();
            html.push_str(&format!("<p>Attacks: {}</p>\n", attacks.len()));
            if !attacks.is_empty() {
                html.push_str("<ul class=\"attacks\">\n");
                for a in &attacks {
                    html.push_str(&format!(
                        "<li>{} (priority: {:.0}%): {}</li>\n",
                        a.kind,
                        a.priority * 100.0,
                        a.description
                    ));
                }
                html.push_str("</ul>\n");
            }
            html.push_str("</div>\n");

            // Fuga Synthesis
            if result.bug_confidence() > 0.0 || result.bug_detected() {
                html.push_str("<div class=\"synthesis\">\n");
                html.push_str("<h4>Fuga Synthesis</h4>\n");
                if result.bug_detected() {
                    html.push_str(&format!(
                        "<p class=\"bug-detected\">🐛 BUG DETECTED (confidence: {:.1}%)</p>\n",
                        result.bug_confidence() * 100.0
                    ));
                } else {
                    html.push_str(&format!(
                        "<p class=\"clean\">✅ CLEAN (score: {:.1}%)</p>\n",
                        result.bug_confidence() * 100.0
                    ));
                }
                let desc = result.counterpoint_description();
                if !desc.is_empty() {
                    html.push_str(&format!("<p class=\"counterpoint\">{}</p>\n", desc));
                }
                html.push_str("</div>\n");
            }

            html.push_str("</div>\n"); // file
        }

        html.push_str("</div>\n"); // files
        html.push_str("</body>\n</html>");
        html
    }
}

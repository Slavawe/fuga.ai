use crate::engine::FugaEngine;
use crate::multi::{
    LanguageId, MultiSyntaxLayer, MultiSemanticLayer, MultiChaosLayer,
};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct QualityScore {
    pub language: LanguageId,
    pub safety: f64,
    pub coherence: f64,
    pub violations: usize,
    pub attacks: usize,
    pub bugs_detected: bool,
    pub weight: f64,
    pub summary: String,
    pub path: String,
}

impl QualityScore {
    pub fn display(&self) -> String {
        let icon = if self.weight >= 0.8 { "✓" } else if self.weight > 0.0 { "~" } else { "✗" };
        format!(
            "[{}] {} w={:.2} safety={:.2} coherence={:.2} v={} a={} bugs={}",
            icon, self.language.name(), self.weight, self.safety, self.coherence,
            self.violations, self.attacks, self.bugs_detected,
        )
    }

    fn calculate_weight(safety: f64, coherence: f64, violations: usize, _attacks: usize, bugs_detected: bool, path: &str) -> f64 {
        if bugs_detected {
            return 0.0;
        }

        let is_core = path.contains("/core/") || path.contains("\\core\\");
        let is_weaver = path.contains("/weaver/") || path.contains("\\weaver\\");
        let is_ai = path.contains("/ai/") || path.contains("\\ai\\");
        let is_bin = path.contains("/bin/") || path.contains("\\bin\\");
        let is_system = is_core || is_weaver || is_ai;

        if is_system {
            if coherence > 0.90 && violations > 0 {
                return 0.5;
            }
            if coherence > 0.80 {
                return (0.5 + coherence * 0.3).min(0.9);
            }
        }

        if is_bin || path.contains("main.rs") {
            if !bugs_detected {
                return (safety * 0.4 + coherence * 0.4).min(0.7);
            }
            return 0.0;
        }

        if coherence > 0.90 && safety < 0.50 {
            return 0.5;
        }

        if safety < 0.3 && violations > 10 {
            return 0.0;
        }

        (safety * coherence).clamp(0.0, 1.0)
    }
}

pub struct CodeQualityFilter {
    dim: usize,
    fuga_engine: Option<FugaEngine>,
    multi_syntax: MultiSyntaxLayer,
    multi_semantic: MultiSemanticLayer,
    multi_chaos: MultiChaosLayer,
}

impl CodeQualityFilter {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            fuga_engine: None,
            multi_syntax: MultiSyntaxLayer::new(dim),
            multi_semantic: MultiSemanticLayer::new(dim),
            multi_chaos: MultiChaosLayer::new(dim),
        }
    }

    pub fn analyze_file(&mut self, path: &str) -> Result<QualityScore, String> {
        let lang = LanguageId::from_path(Path::new(path))
            .ok_or_else(|| format!("Unsupported language: {}", path))?;
        let source = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path, e))?;
        self.analyze(&source, lang, path)
    }

    pub fn analyze(&mut self, source: &str, lang: LanguageId, path: &str) -> Result<QualityScore, String> {
        match lang {
            LanguageId::Rust => self.analyze_rust(source, path),
            other => self.analyze_multi(source, other, path),
        }
    }

    fn analyze_rust(&mut self, source: &str, path: &str) -> Result<QualityScore, String> {
        let engine = self.fuga_engine.get_or_insert_with(|| FugaEngine::new(self.dim));
        let result = engine.analyze(source)
            .map_err(|e| format!("Rust analysis failed: {}", e))?;

        let safety = result.layer_results.syntax.safety_score;
        let violations = result.layer_results.syntax.violations.len();
        let attacks = result.layer_results.chaos.attack_vectors.len();
        let bug_detected = result.fuga_result.as_ref()
            .map(|r| r.bug_detected).unwrap_or(false);
        let coherence = result.layer_results.semantic.coherence;

        let weight = QualityScore::calculate_weight(safety, coherence, violations, attacks, bug_detected, path);
        let summary = format!(
            "Rust: safety={:.2} v={} a={} bug={} coh={:.2} → w={:.2}",
            safety, violations, attacks, bug_detected, coherence, weight,
        );

        Ok(QualityScore { language: LanguageId::Rust, safety, coherence, violations, attacks, bugs_detected: bug_detected, weight, summary, path: path.to_string() })
    }

    fn analyze_multi(&mut self, source: &str, lang: LanguageId, path: &str) -> Result<QualityScore, String> {
        let syntax = self.multi_syntax.analyze(source, lang, path);
        let semantic = self.multi_semantic.analyze(source, lang);
        let chaos = self.multi_chaos.analyze(source, lang);

        let safety = syntax.safety_score;
        let coherence = semantic.coherence;
        let violations = syntax.violations.len();
        let attacks = chaos.attack_vectors.len();
        let bug_detected = !chaos.attack_vectors.is_empty();

        let raw_weight = QualityScore::calculate_weight(safety, coherence, violations, attacks, bug_detected, path);
        let weight = match lang {
            LanguageId::Go => raw_weight.max(0.4),
            _ => raw_weight,
        };

        let summary = format!(
            "{}: safety={:.2} v={} a={} coh={:.2} → w={:.2}",
            lang.name(), safety, violations, attacks, coherence, weight,
        );

        Ok(QualityScore { language: lang, safety, coherence, violations, attacks, bugs_detected: bug_detected, weight, summary, path: path.to_string() })
    }

    pub fn scan_directory(&mut self, dir: &str, recursive: bool) -> Result<Vec<(String, QualityScore)>, String> {
        use walkdir::WalkDir;
        let mut results = Vec::new();
        let walker = if recursive {
            WalkDir::new(dir).follow_links(true).into_iter()
        } else {
            WalkDir::new(dir).follow_links(true).max_depth(1).into_iter()
        };

        for entry in walker.filter_map(|e| e.ok()) {
            let path = entry.path();
            if !path.is_file() { continue; }
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if crate::multi::language::is_supported(ext) {
                match self.analyze_file(path.to_str().unwrap()) {
                    Ok(score) => results.push((path.to_string_lossy().to_string(), score)),
                    Err(e) => eprintln!("  ⚠ {}: {}", path.display(), e),
                }
            }
        }
        results.sort_by(|a, b| b.1.weight.partial_cmp(&a.1.weight).unwrap());
        Ok(results)
    }
}

pub fn summarize_quality(results: &[(String, QualityScore)]) -> String {
    let total = results.len();
    let high = results.iter().filter(|(_, s)| s.weight >= 0.8).count();
    let medium = results.iter().filter(|(_, s)| s.weight >= 0.4 && s.weight < 0.8).count();
    let low = results.iter().filter(|(_, s)| s.weight > 0.0 && s.weight < 0.4).count();
    let blocked = results.iter().filter(|(_, s)| s.weight == 0.0).count();
    let avg_weight: f64 = results.iter().map(|(_, s)| s.weight).sum::<f64>() / total.max(1) as f64;
    let avg_safety: f64 = results.iter().map(|(_, s)| s.safety).sum::<f64>() / total.max(1) as f64;
    let avg_coherence: f64 = results.iter().map(|(_, s)| s.coherence).sum::<f64>() / total.max(1) as f64;
    let total_violations: usize = results.iter().map(|(_, s)| s.violations).sum();
    let total_attacks: usize = results.iter().map(|(_, s)| s.attacks).sum();

    let mut out = String::new();
    out.push_str(&format!("Files:     {}\n", total));
    out.push_str(&format!("  High (w≥0.8):   {}\n", high));
    out.push_str(&format!("  Med (0.4≤w<0.8): {}\n", medium));
    out.push_str(&format!("  Low (0<w<0.4):  {}\n", low));
    out.push_str(&format!("  Blocked (w=0):  {} ({:.0}%)\n", blocked, blocked as f64 / total.max(1) as f64 * 100.0));
    out.push_str(&format!("Avg weight:    {:.3}\n", avg_weight));
    out.push_str(&format!("Avg safety:    {:.3}\n", avg_safety));
    out.push_str(&format!("Avg coherence: {:.3}\n", avg_coherence));
    out.push_str(&format!("Violations:    {}\n", total_violations));
    out.push_str(&format!("Attacks:       {}\n", total_attacks));
    out.push('\n');
    for (path, score) in results {
        out.push_str(&format!("  {}  {}\n", score.display(), path));
    }
    out
}

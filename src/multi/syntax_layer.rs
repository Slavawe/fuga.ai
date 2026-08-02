use crate::core::hypervector::Hypervector;
use crate::multi::language::{
    LanguageId, count_nodes_by_kind, find_enclosing_function, parse_source, run_query,
};
use crate::multi::patterns::{Severity, ViolationPattern};
use rand::SeedableRng;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct MultiSyntaxResult {
    pub safety_score: f64,
    pub violation_vector: Hypervector,
    pub violations: Vec<MultiSyntaxViolation>,
    pub functions: usize,
    pub lines: usize,
}

#[derive(Debug, Clone)]
pub struct MultiSyntaxViolation {
    pub pattern: ViolationPattern,
    pub severity: Severity,
    pub file: String,
    pub line: usize,
    pub column: usize,
    pub start_byte: usize,
    pub end_byte: usize,
    pub function: Option<String>,
    pub code_snippet: Option<String>,
    pub message: String,
}

pub struct MultiSyntaxLayer {
    dim: usize,
    pattern_vectors: HashMap<ViolationPattern, Hypervector>,
}

impl MultiSyntaxLayer {
    pub fn new(dim: usize) -> Self {
        let mut pattern_vectors = HashMap::new();
        for p in ViolationPattern::iter_all() {
            let hv = deterministic_vector(dim, &format!("{:?}", p));
            pattern_vectors.insert(*p, hv);
        }
        Self {
            dim,
            pattern_vectors,
        }
    }

    pub fn analyze(&self, source: &str, lang: LanguageId, file_name: &str) -> MultiSyntaxResult {
        let lines = source.lines().count();
        let mut violations = Vec::new();
        let mut functions = 0;

        if let Some(tree) = parse_source(source, lang) {
            functions = count_nodes_by_kind(tree.root_node(), lang.function_kinds());

            for pattern in ViolationPattern::iter_all() {
                if let Some(query_str) = pattern.query(lang) {
                    if !query_str.is_empty() {
                        if let Some(results) = run_query(query_str, source, lang, &tree) {
                            for qr in results {
                                if qr.capture_name == "violation" {
                                    // Post-filter broad queries for Python (which lacks named fields)
                                    if lang == LanguageId::Python
                                        && !self.python_pattern_matches(*pattern, &qr.text)
                                    {
                                        continue;
                                    }
                                    let fn_name =
                                        find_enclosing_function(&tree, source, lang, qr.start_line);
                                    violations.push(MultiSyntaxViolation {
                                        pattern: *pattern,
                                        severity: pattern.default_severity(),
                                        file: file_name.to_string(),
                                        line: qr.start_line,
                                        column: qr.start_col,
                                        start_byte: qr.start_byte,
                                        end_byte: qr.end_byte,
                                        function: fn_name,
                                        code_snippet: Some(qr.text.clone()),
                                        message: format!("{:?} violation detected", pattern),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        let violation_vector = if violations.is_empty() {
            Hypervector::random(self.dim)
        } else {
            let vecs: Vec<&Hypervector> = violations
                .iter()
                .filter_map(|v| self.pattern_vectors.get(&v.pattern))
                .collect();
            if vecs.is_empty() {
                Hypervector::random(self.dim)
            } else {
                vecs[0].bundle(&vecs[1..])
            }
        };

        let safety_score = (1.0 - violations.len() as f64 * 0.08).max(0.0);

        MultiSyntaxResult {
            safety_score,
            violation_vector,
            violations,
            functions,
            lines,
        }
    }

    fn python_pattern_matches(&self, pattern: ViolationPattern, text: &str) -> bool {
        // Python attribute nodes don't use named fields, so queries match broadly.
        // We filter by function name in Rust code.
        let last_dot = text.rfind('.');
        let func_name = last_dot
            .map(|i| text[i + 1..].trim_end_matches(&['(', ' '][..]))
            .unwrap_or("");
        match pattern {
            ViolationPattern::UnwrapOrExpect => matches!(func_name, "get" | "pop" | "__getitem__"),
            ViolationPattern::CommandInjection => matches!(
                func_name,
                "run" | "Popen" | "call" | "check_output" | "system"
            ),
            ViolationPattern::FormatStringVulnerability => {
                matches!(func_name, "format" | "format_map")
            }
            ViolationPattern::SqlInjection => {
                matches!(func_name, "execute" | "executemany" | "cursor")
            }
            _ => true,
        }
    }
}

fn deterministic_vector(dim: usize, seed: &str) -> Hypervector {
    use rand::RngCore;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let mut rng = rand::rngs::StdRng::seed_from_u64(hasher.finish());
    let word_count = (dim + 63) / 64;
    let mut words = vec![0u64; word_count];
    for w in &mut words {
        *w = rng.next_u64();
    }
    let rem = dim % 64;
    if rem != 0 {
        words[word_count - 1] &= (1u64 << rem) - 1;
    }
    Hypervector { dim, words }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_syntax_rust() {
        let layer = MultiSyntaxLayer::new(4096);
        let code = r#"fn main() { let x = Some(42); x.unwrap(); }"#;
        let result = layer.analyze(code, LanguageId::Rust, "test.rs");
        assert!(
            result.safety_score > 0.0,
            "Should analyze, got {:.2}",
            result.safety_score
        );
    }

    #[test]
    fn test_multi_syntax_c() {
        let layer = MultiSyntaxLayer::new(4096);
        let code = r#"void buggy() { char buf[10]; gets(buf); }"#;
        let result = layer.analyze(code, LanguageId::C, "test.c");
        result
            .violations
            .iter()
            .any(|v| v.pattern == ViolationPattern::BufferOverflow);
        assert!(result.safety_score < 1.0 || !result.violations.is_empty());
    }
}

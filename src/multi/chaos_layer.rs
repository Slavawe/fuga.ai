use crate::core::hypervector::Hypervector;
use crate::multi::language::{LanguageId, count_nodes_by_kind, parse_source, run_query};
use crate::multi::patterns::ViolationPattern;
use rand::SeedableRng;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct MultiChaosResult {
    pub attack_vectors: Vec<MultiChaosAttack>,
    pub mutation_points: usize,
}

#[derive(Debug, Clone)]
pub struct MultiChaosAttack {
    pub pattern: ViolationPattern,
    pub vector: Hypervector,
    pub description: String,
    pub priority: f64,
    pub function: Option<String>,
    pub line: usize,
}

pub struct MultiChaosLayer {
    dim: usize,
    attack_vectors: HashMap<ViolationPattern, Hypervector>,
}

impl MultiChaosLayer {
    pub fn new(dim: usize) -> Self {
        let mut attack_vectors = HashMap::new();
        for p in ViolationPattern::iter_all() {
            let hv = deterministic_vector(dim, &format!("{:?}", p));
            attack_vectors.insert(*p, hv);
        }
        Self {
            dim,
            attack_vectors,
        }
    }

    pub fn analyze(&self, source: &str, lang: LanguageId) -> MultiChaosResult {
        let mut attacks = Vec::new();
        let mut mutation_points = 0;

        if let Some(tree) = parse_source(source, lang) {
            let risky_kinds: &[&str] = match lang {
                LanguageId::Rust => &["binary_expression", "call_expression"],
                LanguageId::C | LanguageId::Cpp => &["binary_expression", "call_expression"],
                LanguageId::Go => &["binary_expression", "call_expression"],
                LanguageId::Python => &["binary_operator", "call"],
                LanguageId::TypeScript | LanguageId::JavaScript => {
                    &["binary_expression", "call_expression"]
                }
            };
            if !risky_kinds.is_empty() {
                mutation_points = count_nodes_by_kind(tree.root_node(), risky_kinds);
            }

            for pattern in ViolationPattern::iter_all() {
                match pattern.query(lang) {
                    Some(query_str) if !query_str.is_empty() => {
                        match run_query(query_str, source, lang, &tree) {
                            Some(results) => {
                                for qr in results {
                                    attacks.push(MultiChaosAttack {
                                        pattern: *pattern,
                                        vector: self
                                            .attack_vectors
                                            .get(pattern)
                                            .cloned()
                                            .unwrap_or_else(|| Hypervector::random(self.dim)),
                                        description: format!("{:?} via {}", pattern, qr.text),
                                        priority: match pattern.default_severity() {
                                            crate::multi::patterns::Severity::Critical => 0.95,
                                            crate::multi::patterns::Severity::High => 0.8,
                                            crate::multi::patterns::Severity::Medium => 0.6,
                                            crate::multi::patterns::Severity::Low => 0.4,
                                            crate::multi::patterns::Severity::Info => 0.2,
                                        },
                                        function: None,
                                        line: qr.start_line,
                                    });
                                }
                            }
                            None => {}
                        }
                    }
                    _ => {}
                }
            }
        }

        attacks.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());

        MultiChaosResult {
            attack_vectors: attacks,
            mutation_points,
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
    fn test_multi_chaos_rust() {
        let layer = MultiChaosLayer::new(4096);
        let code = r#"fn main() { let x = 1 + 2; loop {} }"#;
        let result = layer.analyze(code, LanguageId::Rust);
        assert!(result.mutation_points > 0);
    }

    #[test]
    fn test_multi_chaos_c() {
        let layer = MultiChaosLayer::new(4096);
        let code = r#"void buggy() { char buf[10]; gets(buf); }"#;
        let result = layer.analyze(code, LanguageId::C);
        assert!(!result.attack_vectors.is_empty() || result.mutation_points > 0);
    }
}

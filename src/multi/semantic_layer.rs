use crate::core::hypervector::Hypervector;
use crate::multi::language::{LanguageId, collect_function_names, parse_source};
use rand::SeedableRng;

#[derive(Debug, Clone)]
pub struct MultiSemanticResult {
    pub semantic_vector: Hypervector,
    pub function_vectors: Vec<(String, Hypervector)>,
    pub coherence: f64,
    pub anomalies: Vec<MultiSemanticAnomaly>,
}

#[derive(Debug, Clone)]
pub struct MultiSemanticAnomaly {
    pub kind: AnomalyKind,
    pub location: String,
    pub description: String,
    pub severity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnomalyKind {
    SignatureMismatch,
    UnusedParameter,
    MissingBaseCase,
    HighComplexity,
    TypeInvariantViolation,
}

pub struct MultiSemanticLayer {
    pub dim: usize,
}

impl MultiSemanticLayer {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn analyze(&self, source: &str, lang: LanguageId) -> MultiSemanticResult {
        let tree = match parse_source(source, lang) {
            Some(t) => t,
            None => {
                return MultiSemanticResult {
                    semantic_vector: Hypervector::random(self.dim),
                    function_vectors: vec![],
                    coherence: 0.0,
                    anomalies: vec![MultiSemanticAnomaly {
                        kind: AnomalyKind::TypeInvariantViolation,
                        location: "root".into(),
                        description: "Failed to parse source".into(),
                        severity: 1.0,
                    }],
                };
            }
        };

        let functions = collect_function_names(&tree, source, lang);
        let mut function_vectors = Vec::new();
        let anomalies = Vec::new();

        for fn_name in &functions {
            let hv = encode_string(self.dim, &format!("fn:{}:{}", lang.name(), fn_name));
            function_vectors.push((fn_name.clone(), hv));
        }

        let semantic_vector = if function_vectors.is_empty() {
            Hypervector::random(self.dim)
        } else {
            let first = &function_vectors[0].1;
            let others: Vec<&Hypervector> =
                function_vectors.iter().skip(1).map(|(_, v)| v).collect();
            first.bundle(&others)
        };

        let coherence = if function_vectors.len() < 2 {
            1.0
        } else {
            let mut sims = Vec::new();
            for i in 0..function_vectors.len() {
                for j in i + 1..function_vectors.len() {
                    sims.push(function_vectors[i].1.similarity(&function_vectors[j].1));
                }
            }
            sims.iter().sum::<f64>() / sims.len() as f64
        };

        MultiSemanticResult {
            semantic_vector,
            function_vectors,
            coherence,
            anomalies,
        }
    }
}

fn encode_string(dim: usize, s: &str) -> Hypervector {
    use rand::RngCore;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
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
    fn test_multi_semantic_rust() {
        let layer = MultiSemanticLayer::new(4096);
        let code = r#"fn add(a: i32, b: i32) -> i32 { a + b }"#;
        let result = layer.analyze(code, LanguageId::Rust);
        assert!(!result.function_vectors.is_empty(), "Should find functions");
    }

    #[test]
    fn test_multi_semantic_python() {
        let layer = MultiSemanticLayer::new(4096);
        let code = "def add(a, b): pass";
        let result = layer.analyze(code, LanguageId::Python);
        assert!(!result.function_vectors.is_empty());
    }
}

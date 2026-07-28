use crate::core::hypervector::Hypervector;
use std::collections::HashMap;
use rand::RngCore;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::hash::{Hash, Hasher};

const PROMPT_MODES: &[&str] = &[
    "SAFETY", "EFFICIENT", "CONCISE", "EXPLAIN", "DRY_RUN",
];

pub struct PromptVectors {
    modes: HashMap<String, Hypervector>,
}

impl PromptVectors {
    pub fn new(dim: usize) -> Self {
        let mut modes = HashMap::new();
        for &name in PROMPT_MODES {
            let hv = Self::make_deterministic(name, dim);
            modes.insert(name.to_string(), hv);
        }
        Self { modes }
    }

    fn make_deterministic(name: &str, dim: usize) -> Hypervector {
        let wc = (dim + 63) / 64;
        let mut words = vec![0u64; wc];
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        name.hash(&mut hasher);
        let seed = hasher.finish();
        let mut rng = StdRng::seed_from_u64(seed);
        for w in &mut words {
            *w = rng.next_u64();
        }
        let rem = dim % 64;
        if rem != 0 {
            words[wc - 1] &= (1u64 << rem) - 1;
        }
        Hypervector { dim, words }
    }

    pub fn resolve(names: &[String], dim: usize) -> Vec<Hypervector> {
        names.iter()
            .map(|n| Self::make_deterministic(n, dim))
            .collect()
    }

    pub fn bind_all(query: &Hypervector, prompts: &[&Hypervector]) -> Hypervector {
        let mut result = query.clone();
        for p in prompts {
            result = result.bind(p);
        }
        result
    }

    pub fn all_modes(&self) -> Vec<String> {
        self.modes.keys().cloned().collect()
    }

    pub fn get(&self, name: &str) -> Option<&Hypervector> {
        self.modes.get(name)
    }
}

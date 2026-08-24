use super::TokenRole;
use super::deterministic_vector;
use super::vocabulary::TokenVocabulary;
use crate::core::hypervector::Hypervector;
use rand::Rng;

pub struct TokenExplorer {
    dim: usize,
}

impl TokenExplorer {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn synthesize(&self, concept: &str, _role: TokenRole) -> (u32, Hypervector) {
        let id = concept
            .chars()
            .fold(0u32, |acc, c| acc.wrapping_mul(31).wrapping_add(c as u32));
        let hv = deterministic_vector(self.dim, &format!("synth:{}", concept));
        (id, hv)
    }

    pub fn synthesize_concept_chain(
        &self,
        concepts: &[&str],
        role: TokenRole,
    ) -> (String, Hypervector) {
        let name = concepts.join("+");
        let mut acc = deterministic_vector(self.dim, "concept_chain_base");
        for (i, concept) in concepts.iter().enumerate() {
            let (_, hv) = self.synthesize(concept, role);
            let positioned = hv.permute(i % self.dim);
            acc = acc.bind(&positioned);
        }
        (name, acc)
    }

    pub fn explore_neighborhood(
        &self,
        query: &Hypervector,
        vocab: &TokenVocabulary,
        radius: usize,
    ) -> Vec<(u32, String, f64)> {
        let nearest = vocab.nearest_n(query, radius);
        let mut neighbors = Vec::new();

        for (id, text, _sim) in &nearest {
            let base = vocab.get_vector(*id).unwrap();
            for r in 1..=radius {
                let mut noise = base.clone();
                for _ in 0..r {
                    noise = self.mutate(&noise, 0.01);
                }
                let nsim = query.similarity(&noise);
                neighbors.push((*id, format!("{}_mut{}", text, r), nsim));
            }
        }
        neighbors
    }

    pub fn mutate(&self, hv: &Hypervector, rate: f64) -> Hypervector {
        let mut rng = rand::thread_rng();
        let mut words = hv.words.clone();
        for i in 0..hv.dim {
            if rng.gen_bool(rate) {
                let w = i / 64;
                let b = i % 64;
                words[w] ^= 1u64 << b;
            }
        }
        Hypervector {
            dim: self.dim,
            words,
        }
    }

    pub fn crossover(&self, a: &Hypervector, b: &Hypervector) -> Hypervector {
        let mut rng = rand::thread_rng();
        let mut words = a.words.clone();
        for i in 0..a.dim {
            if rng.gen_bool(0.5) {
                let w = i / 64;
                let bit = (b.words[w] >> (i % 64)) & 1;
                words[w] = (words[w] & !(1u64 << (i % 64))) | (bit << (i % 64));
            }
        }
        Hypervector {
            dim: self.dim,
            words,
        }
    }

    pub fn generate_new_tokens(
        &self,
        vocab: &TokenVocabulary,
        count: usize,
    ) -> Vec<(u32, String, Hypervector)> {
        let mut rng = rand::thread_rng();
        let mut new_tokens = Vec::new();

        for i in 0..count {
            let id = (vocab.size() + i) as u32;
            let base_id = rng.gen_range(0..vocab.size()) as u32;
            let base_text = vocab.search_by_id(base_id).unwrap_or("UNKNOWN");

            let mutation = format!("{}_v{}", base_text, i + 1);
            let base_hv = match vocab.get_vector(base_id) {
                Some(hv) => hv.clone(),
                None => deterministic_vector(self.dim, "fallback"),
            };
            let hv = self.mutate(&base_hv, 0.05);
            new_tokens.push((id, mutation, hv));
        }
        new_tokens
    }

    pub fn explore_tokens(&self, vocab: &TokenVocabulary) -> ExploreReport {
        let ids: Vec<_> = (0..vocab.size().min(100) as u32).collect();
        let sample: Vec<_> = ids
            .iter()
            .filter_map(|id| {
                let text = vocab.search_by_id(*id)?;
                let hv = vocab.get_vector(*id)?;
                Some((*id, text.to_string(), hv.entropy()))
            })
            .collect();

        let total = vocab.size();
        ExploreReport { total, sample }
    }
}

pub struct ExploreReport {
    pub total: usize,
    pub sample: Vec<(u32, String, f64)>,
}

impl ExploreReport {
    pub fn display(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Fuga Token Explorer ===\n");
        out.push_str(&format!("Vocab size: {}\n", self.total));
        out.push_str(&format!("Sample: {} tokens\n", self.sample.len()));
        out.push_str("\nToken sample (first 100):\n");
        for (id, text, entropy) in &self.sample {
            out.push_str(&format!("  {}: {:32} entropy={:.3}\n", id, text, entropy));
        }
        out
    }
}

pub mod explorer;
pub mod pattern_matcher;
pub mod super_token;
pub mod token_builder;
pub mod vocabulary;

use crate::core::hypervector::Hypervector;
use pattern_matcher::{PatternMatcher, TokenInfo};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use super_token::{SuperToken, TokenRole};
use vocabulary::TokenVocabulary;

pub fn token_id(text: &str) -> u32 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    (h.finish() % 900000) as u32 + 10
}

pub struct WeaverEngine {
    dim: usize,
    window_size: usize,
    pattern_matcher: PatternMatcher,
    vector_cache: HashMap<u32, Hypervector>,
}

impl WeaverEngine {
    pub fn new(dim: usize, window_size: usize) -> Self {
        Self {
            dim,
            window_size,
            pattern_matcher: PatternMatcher::new(window_size),
            vector_cache: HashMap::new(),
        }
    }

    pub fn cached_vector(&mut self, token_id: u32) -> &Hypervector {
        let dim = self.dim;
        self.vector_cache
            .entry(token_id)
            .or_insert_with(|| deterministic_vector(dim, &format!("token_{}", token_id)))
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn compress_stream(
        &mut self,
        tokens: &[TokenInfo],
        idf: Option<&HashMap<u32, f64>>,
    ) -> WeaverResult {
        let total_input = tokens.len();
        let boundaries = self.pattern_matcher.find_boundaries(tokens);
        let mut super_tokens = Vec::new();

        for boundary in &boundaries {
            let window = &tokens[boundary.start..boundary.end];
            let compressed = self.fuse_window(window, boundary.start, boundary.role, idf);
            super_tokens.push(compressed);
        }

        let output_count = super_tokens.len();
        let ratio = if output_count > 0 {
            total_input as f64 / output_count as f64
        } else {
            1.0
        };

        WeaverResult {
            super_tokens,
            input_tokens: total_input,
            output_tokens: output_count,
            compression_ratio: ratio,
        }
    }

    fn fuse_window(
        &mut self,
        tokens: &[TokenInfo],
        start_pos: usize,
        role: TokenRole,
        idf: Option<&HashMap<u32, f64>>,
    ) -> SuperToken {
        if tokens.is_empty() {
            return SuperToken::new(Hypervector::random(self.dim), start_pos);
        }

        let token_hvs: Vec<Hypervector> = tokens
            .iter()
            .enumerate()
            .map(|(pos, tok)| {
                let hv = self.cached_vector(tok.id).clone();
                hv.permute(pos % self.dim)
            })
            .collect();

        let mut items: Vec<&Hypervector> = Vec::new();
        for (i, hv) in token_hvs.iter().enumerate() {
            let repeats = idf
                .and_then(|w| w.get(&tokens[i].id))
                .map(|w| (w * 2.0).round() as usize)
                .unwrap_or(1)
                .max(1);
            for _ in 0..repeats {
                items.push(hv);
            }
        }

        let base = items[0].clone();
        let acc = if items.len() > 1 {
            base.bundle(&items[1..])
        } else {
            base
        };

        let mut st = SuperToken::new(acc, start_pos);
        st.token_count = tokens.len();
        st.role_flags = role;
        st.raw_tokens = tokens.iter().map(|t| t.id).collect();
        st
    }

    pub fn unweave_stream(
        &self,
        super_tokens: &[SuperToken],
        vocab: &TokenVocabulary,
    ) -> UnweaveResult {
        let mut recovered = Vec::new();
        let mut total_original = 0;
        let mut total_recovered = 0;
        let mut total_similarity = 0.0f64;

        for st in super_tokens {
            let window_size = st.raw_tokens.len();
            total_original += window_size;

            for pos in 0..window_size {
                let token = self.unweave_token(st, pos, vocab);
                let sim = token.2;
                total_similarity += sim;
                if sim > 0.5 {
                    total_recovered += 1;
                }
                recovered.push(TokenInfo {
                    id: token.0,
                    text: token.1,
                });
            }
        }

        let count = recovered.len();
        let avg_sim = if count > 0 {
            total_similarity / count as f64
        } else {
            0.0
        };
        let accuracy = if total_original > 0 {
            total_recovered as f64 / total_original as f64
        } else {
            0.0
        };

        UnweaveResult {
            recovered_tokens: recovered,
            total_original,
            total_recovered,
            accuracy,
            avg_similarity: avg_sim,
        }
    }

    pub fn unweave_stream_filtered(
        &self,
        super_tokens: &[SuperToken],
        vocab: &TokenVocabulary,
        candidates: &HashSet<u32>,
    ) -> UnweaveResult {
        let mut recovered = Vec::new();
        let mut total_original = 0;
        let mut total_recovered = 0;
        let mut total_similarity = 0.0f64;

        for st in super_tokens {
            let window_size = st.raw_tokens.len();
            total_original += window_size;

            for pos in 0..window_size {
                let shift = pos % self.dim;
                let unshifted = st.vector.permute(self.dim - shift);
                let token = vocab.nearest_in_set(&unshifted, candidates).unwrap_or((
                    0,
                    "<UNK>".to_string(),
                    0.0,
                ));
                let sim = token.2;
                total_similarity += sim;
                if sim > 0.5 {
                    total_recovered += 1;
                }
                recovered.push(TokenInfo {
                    id: token.0,
                    text: token.1,
                });
            }
        }

        let count = recovered.len();
        let avg_sim = if count > 0 {
            total_similarity / count as f64
        } else {
            0.0
        };
        let accuracy = if total_original > 0 {
            total_recovered as f64 / total_original as f64
        } else {
            0.0
        };

        UnweaveResult {
            recovered_tokens: recovered,
            total_original,
            total_recovered,
            accuracy,
            avg_similarity: avg_sim,
        }
    }

    fn unweave_token(
        &self,
        st: &SuperToken,
        position: usize,
        vocab: &TokenVocabulary,
    ) -> (u32, String, f64) {
        let shift = position % self.dim;
        let unshifted = st.vector.permute(self.dim - shift);

        let beam = vocab.nearest_beam(&unshifted, 8);
        beam.into_iter()
            .next()
            .unwrap_or((0, "<UNK>".to_string(), 0.0))
    }
}

pub struct WeaverResult {
    pub super_tokens: Vec<SuperToken>,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub compression_ratio: f64,
}

pub struct UnweaveResult {
    pub recovered_tokens: Vec<TokenInfo>,
    pub total_original: usize,
    pub total_recovered: usize,
    pub accuracy: f64,
    pub avg_similarity: f64,
}

impl UnweaveResult {
    pub fn display(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Fuga Unweave Result ===\n");
        out.push_str(&format!("Original tokens:  {}\n", self.total_original));
        out.push_str(&format!("Recovered tokens: {}\n", self.total_recovered));
        out.push_str(&format!(
            "Accuracy:         {:.1}%\n",
            self.accuracy * 100.0
        ));
        out.push_str(&format!("Avg similarity:   {:.4}\n", self.avg_similarity));
        out.push('\n');
        for (i, tok) in self.recovered_tokens.iter().enumerate().take(20) {
            out.push_str(&format!("  [{}] id={} text={:?}\n", i, tok.id, tok.text));
        }
        if self.recovered_tokens.len() > 20 {
            out.push_str(&format!(
                "  ... ({} more)\n",
                self.recovered_tokens.len() - 20
            ));
        }
        out
    }
}

impl WeaverResult {
    pub fn display(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Fuga Weaver Result ===\n");
        out.push_str(&format!("Input tokens:  {}\n", self.input_tokens));
        out.push_str(&format!("Output tokens: {}\n", self.output_tokens));
        out.push_str(&format!("Compression:   {:.2}x\n", self.compression_ratio));
        out.push('\n');
        for (i, st) in self.super_tokens.iter().enumerate() {
            out.push_str(&format!(
                "SuperToken [{}]: {} tokens compressed, role flags: {:08b}\n",
                i,
                st.raw_tokens.len(),
                st.role_flags.bits()
            ));
        }
        out
    }
}

pub(crate) fn deterministic_vector(dim: usize, seed: &str) -> Hypervector {
    use rand::{RngCore, SeedableRng};
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

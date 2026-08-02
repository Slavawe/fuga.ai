use super::TokenRole;
use super::deterministic_vector;
use super::token_builder::TokenBuilder;
use crate::core::hypervector::Hypervector;
use std::collections::{HashMap, HashSet};

pub struct TokenVocabulary {
    dim: usize,
    entries: Vec<VocabEntry>,
    by_id: HashMap<u32, usize>,
}

struct VocabEntry {
    pub id: u32,
    pub text: String,
    pub vector: Hypervector,
    pub role: TokenRole,
}

impl TokenVocabulary {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entries: Vec::new(),
            by_id: HashMap::new(),
        }
    }

    pub fn add(&mut self, id: u32, text: &str, role: TokenRole) {
        let hv = deterministic_vector(self.dim, &format!("token_{}", id));
        self.by_id.insert(id, self.entries.len());
        self.entries.push(VocabEntry {
            id,
            text: text.to_string(),
            vector: hv,
            role,
        });
    }

    pub fn add_custom(&mut self, id: u32, text: &str, hv: Hypervector, role: TokenRole) {
        self.by_id.insert(id, self.entries.len());
        self.entries.push(VocabEntry {
            id,
            text: text.to_string(),
            vector: hv,
            role,
        });
    }

    pub fn from_builder(builder: &TokenBuilder, dim: usize) -> Self {
        let mut vocab = Self::new(dim);
        let flat = builder.build_flat_vocab();
        for (id, text) in &flat {
            let role = if builder.merged_special().contains_key(id) {
                TokenRole::SPECIAL
            } else {
                TokenRole::NATURAL_LANGUAGE
            };
            vocab.add(*id, text, role);
        }
        vocab
    }

    pub fn nearest(&self, query: &Hypervector) -> Option<(u32, String, f64)> {
        self.entries
            .iter()
            .map(|e| (e.id, e.text.clone(), query.similarity(&e.vector)))
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
    }

    pub fn nearest_beam(&self, query: &Hypervector, beam: usize) -> Vec<(u32, String, f64)> {
        use rayon::prelude::*;
        let step = (self.entries.len() + beam - 1) / beam;
        let mut all: Vec<_> = self
            .entries
            .par_chunks(step)
            .flat_map(|chunk| {
                let mut best: Option<(u32, String, f64)> = None;
                for e in chunk {
                    let sim = query.similarity(&e.vector);
                    best = match best {
                        Some((_, _, best_sim)) if sim > best_sim => {
                            Some((e.id, e.text.clone(), sim))
                        }
                        Some(b) => Some(b),
                        None => Some((e.id, e.text.clone(), sim)),
                    };
                }
                best
            })
            .collect();
        all.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        all.truncate(beam.min(all.len()));
        all
    }

    pub fn nearest_in_set(
        &self,
        query: &Hypervector,
        ids: &HashSet<u32>,
    ) -> Option<(u32, String, f64)> {
        self.entries
            .iter()
            .filter(|e| ids.contains(&e.id))
            .map(|e| (e.id, e.text.clone(), query.similarity(&e.vector)))
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
    }

    pub fn nearest_n(&self, query: &Hypervector, n: usize) -> Vec<(u32, String, f64)> {
        let mut results: Vec<_> = self
            .entries
            .iter()
            .map(|e| (e.id, e.text.clone(), query.similarity(&e.vector)))
            .collect();
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        results.truncate(n);
        results
    }

    pub fn search_by_text(&self, text: &str) -> Option<(u32, f64)> {
        self.entries
            .iter()
            .find(|e| e.text == text)
            .map(|e| (e.id, 1.0))
    }

    pub fn search_by_id(&self, id: u32) -> Option<&str> {
        self.by_id
            .get(&id)
            .and_then(|&idx| self.entries.get(idx))
            .map(|e| e.text.as_str())
    }

    pub fn size(&self) -> usize {
        self.entries.len()
    }
    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn get_vector(&self, id: u32) -> Option<&Hypervector> {
        self.by_id
            .get(&id)
            .and_then(|&idx| self.entries.get(idx))
            .map(|e| &e.vector)
    }
}

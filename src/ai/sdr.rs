use crate::core::hypervector::Hypervector;

pub const SDR_DIM: usize = 8192;
pub const SDR_WORDS: usize = SDR_DIM / 64;
pub const SDR_DENSITY: f64 = 0.02;

#[derive(Clone, Debug)]
pub struct SdrVector {
    pub bits: [u64; SDR_WORDS],
}

impl SdrVector {
    pub fn zero() -> Self {
        SdrVector { bits: [0; SDR_WORDS] }
    }

    pub fn from_hypervector(hv: &Hypervector) -> Self {
        let mut sdr = SdrVector::zero();
        let dim = hv.dim.min(SDR_DIM);
        let target = (dim as f64 * SDR_DENSITY).ceil() as usize;

        let hv_seed: u64 = hv.words.iter().fold(0u64, |acc, w| acc.wrapping_mul(31).wrapping_add(*w));

        let mut scored: Vec<(usize, u64)> = hv.words.iter().enumerate().flat_map(|(wi, &w)| {
            let base = wi * 64;
            (0..64.min(dim - base)).filter_map(move |bi| {
                let bit = base + bi;
                if bit >= dim { return None; }
                if (w >> bi) & 1 == 0 { return None; }
                let score = hv_seed.wrapping_mul(bit as u64 + 1).wrapping_add(wi as u64 * 64 + bi as u64).reverse_bits();
                Some((bit, score))
            })
        }).collect();

        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored.truncate(target);

        for &(bit, _) in &scored {
            let wi = bit / 64;
            let bi = bit % 64;
            sdr.bits[wi] |= 1u64 << bi;
        }
        sdr
    }

    pub fn overlap(&self, other: &SdrVector) -> u32 {
        self.bits.iter().zip(other.bits.iter())
            .map(|(&a, &b)| (a & b).count_ones())
            .sum()
    }

    pub fn soft_overlap(&self, other: &SdrVector) -> f64 {
        let intersection = self.overlap(other) as f64;
        let denom = self.popcount().max(other.popcount()) as f64;
        if denom < 1.0 { 0.0 } else { intersection / denom }
    }

    pub fn popcount(&self) -> u32 {
        self.bits.iter().map(|w| w.count_ones()).sum()
    }

    pub fn bind(&self, other: &SdrVector) -> SdrVector {
        let mut out = SdrVector::zero();
        for i in 0..SDR_WORDS {
            out.bits[i] = self.bits[i] ^ other.bits[i];
        }
        out
    }

    pub fn bundle(&self, others: &[&SdrVector]) -> SdrVector {
        let mut out = SdrVector::zero();
        for i in 0..SDR_WORDS {
            let mid = (others.len() + 1) as u32 / 2;
            out.bits[i] = 0;
            for bi in 0..64 {
                let mut ones: u32 = ((self.bits[i] >> bi) & 1) as u32;
                for o in others {
                    ones += ((o.bits[i] >> bi) & 1) as u32;
                }
                if ones > mid {
                    out.bits[i] |= 1u64 << bi;
                }
            }
        }
        out
    }

    pub fn bundle_multi(sdrs: &[SdrVector]) -> SdrVector {
        if sdrs.is_empty() { return SdrVector::zero(); }
        let refs: Vec<&SdrVector> = sdrs.iter().collect();
        refs[0].bundle(&refs[1..])
    }

    pub fn to_hypervector(&self, dim: usize) -> Hypervector {
        let mut words = vec![0u64; dim / 64 + (dim % 64).min(1)];
        for (wi, &w) in self.bits.iter().enumerate() {
            if wi < words.len() {
                words[wi] = w;
            }
        }
        Hypervector { dim, words }
    }
}

pub fn sparsify(hv: &Hypervector) -> SdrVector {
    SdrVector::from_hypervector(hv)
}

use std::hash::{Hash, Hasher};

use rand::RngCore;
use rand::SeedableRng;

fn deterministic_hv(seed: u64) -> Hypervector {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let wc = 128;
    let mut words = vec![0u64; wc];
    for w in &mut words {
        *w = rng.next_u64();
    }
    Hypervector { dim: 8192, words }
}

pub fn encode_text(text: &str) -> SdrVector {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return SdrVector::zero();
    }
    let hvs: Vec<SdrVector> = words.iter().map(|w| {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        format!("token_{}", crate::weaver::token_id(w)).hash(&mut h);
        let hv = deterministic_hv(h.finish());
        sparsify(&hv)
    }).collect();
    let refs: Vec<&SdrVector> = hvs.iter().collect();
    refs[0].bundle(&refs[1..])
}

pub fn domain_sdr(name: &str) -> SdrVector {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    format!("sdr_domain_{}", name).hash(&mut h);
    let hv = deterministic_hv(h.finish());
    sparsify(&hv)
}

pub struct SdrIndex {
    pub nodes: Vec<SdrVector>,
    pub texts: Vec<String>,
}

impl SdrIndex {
    pub fn new() -> Self {
        SdrIndex { nodes: Vec::new(), texts: Vec::new() }
    }

    pub fn search(&self, query: &SdrVector, top_k: usize) -> Vec<(usize, f64, &str)> {
        let mut scored: Vec<(usize, f64)> = self.nodes.iter()
            .map(|n| query.soft_overlap(n))
            .enumerate()
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored.into_iter().map(|(i, s)| (i, s, self.texts[i].as_str())).collect()
    }

    pub fn search_cross(
        &self,
        query: &SdrVector,
        query_domain: &SdrVector,
        index_domain: &SdrVector,
        top_k: usize,
    ) -> Vec<(usize, f64, &str)> {
        let bound_query = query.bind(query_domain);
        let mut scored: Vec<(usize, f64)> = self.nodes.iter()
            .map(|n| {
                let bound_index = n.bind(index_domain);
                bound_query.soft_overlap(&bound_index)
            })
            .enumerate()
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored.into_iter().map(|(i, s)| (i, s, self.texts[i].as_str())).collect()
    }
}

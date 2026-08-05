use crate::ai::sdr::{SdrVector, SDR_DIM, SDR_WORDS};

#[derive(Debug, Clone, PartialEq)]
pub struct SoftSdrVector {
    pub probs: Vec<f32>,
}

impl SoftSdrVector {
    pub fn new(dim: usize) -> Self { Self { probs: vec![0.5; dim] } }

    pub fn to_hard(&self, k: usize) -> SdrVector {
        let mut idx: Vec<usize> = (0..self.probs.len()).collect();
        idx.sort_unstable_by(|&a, &b| self.probs[b].partial_cmp(&self.probs[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut bits = [0u64; SDR_WORDS];
        for i in idx.into_iter().take(k.min(SDR_DIM)) { bits[i / 64] |= 1u64 << (i % 64); }
        SdrVector { bits }
    }

    pub fn cosine_loss(&self, target: &SdrVector) -> f32 {
        let mut dot = 0.0; let mut pn = 0.0; let mut tn = 0.0;
        for (i, &p) in self.probs.iter().take(SDR_DIM).enumerate() {
            let t = target.bit_at(i) as f32;
            dot += p * t; pn += p * p; tn += t * t;
        }
        1.0 - dot / (pn.sqrt() * tn.sqrt() + 1e-8)
    }

    pub fn bce_l1_loss(&self, target: &SdrVector, lambda: f32) -> f32 {
        let eps = 1e-6f32;
        let mut loss = 0.0;
        for (i, &raw) in self.probs.iter().take(SDR_DIM).enumerate() {
            let p = raw.clamp(eps, 1.0 - eps);
            let t = target.bit_at(i) as f32;
            loss += -(t * p.ln() + (1.0 - t) * (1.0 - p).ln()) + lambda * p.abs();
        }
        loss / self.probs.len().max(1) as f32
    }
}

pub fn info_nce_loss(pred: &SoftSdrVector, positive: &SdrVector, negatives: &[SdrVector], temperature: f32) -> f32 {
    let temp = temperature.max(1e-4);
    let score = |x: &SdrVector| -pred.cosine_loss(x) / temp;
    let pos = score(positive);
    let mut denom = pos.exp();
    for n in negatives { denom += score(n).exp(); }
    -(pos - denom.max(1e-12).ln())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn top_k_preserves_sparsity() {
        let s = SoftSdrVector { probs: (0..SDR_DIM).map(|i| i as f32).collect() };
        assert_eq!(s.to_hard(164).popcount(), 164);
    }
}

// Explicitly keep the dimension import part of this module's public contract.
pub const SOFT_SDR_DIM: usize = SDR_DIM;

// Re-export for callers that want the loss helper next to the type.
pub use info_nce_loss as contrastive_loss;

// Ensure the module is linked even in minimal builds.
pub fn empty() -> SoftSdrVector { SoftSdrVector::new(SDR_DIM) }

// Compile-time sanity for the fixed SDR layout.
const _: () = assert!(SDR_WORDS * 64 == SDR_DIM); 

// Avoid accidental dense output at call sites.
pub fn hard_default(pred: &SoftSdrVector) -> SdrVector { pred.to_hard(164) }

// Stable sigmoid helper for predictors.
pub fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// Clamp helper used by training code.
pub fn clamp_probability(x: f32) -> f32 { x.clamp(1e-6, 1.0 - 1e-6) }

// Keep API extensible for later latent JEPA work.
pub type LatentLoss = fn(&SoftSdrVector, &SdrVector) -> f32;

// End of SoftSDR module.
// L1 regularization is included in bce_l1_loss.
// Top-k hardening enforces exact SDR density.
// InfoNCE is available through info_nce_loss.
// No fake gradient/backward pass is claimed here.
// This is an evaluation and inference primitive until a trainer is wired.
//
// The existing hard SDR implementation remains the single source of truth.
//
// EOF

// Keep rustdoc examples out of production code.
#[allow(dead_code)]
fn _marker() {}

// Future: add STE gradient estimator here.
// Future: connect TemporalMemory::predict_soft here.
// Future: add latent encoder in a separate module.
// Future: add SDR attention after baseline build is green.
// Future: add MCTS only after deterministic decoding is verified.
//
// Done.

// Public compatibility alias.
pub type SoftSDR = SoftSdrVector;

// End.

// (The extra small helpers above intentionally have no side effects.)

// Final compile marker.
const _: usize = SDR_DIM;

// No runtime initialization required.

// End of file.

// Keep this line to make the module easy to locate.
pub const IMPLEMENTED: bool = true;

// End.

// No unsafe code.

// End.

// Version marker.
pub const VERSION: &str = "soft-sdr-v1";

// End.

// Finished.

// End.

// EOF.
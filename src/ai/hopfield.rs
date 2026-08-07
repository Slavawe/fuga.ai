//! Modern Hopfield Memory (associative read) for the tokenless byte decoder.
//!
//! Directly implements the attention-without-tokens plan: instead of feeding a
//! drifting hidden state `h(t)` straight into `W`, the state is used as a QUERY
//! into an associative memory and only a CLEAN retrieved vector reaches `W`:
//!
//!   h_clean(t) = M_vals · softmax(β · M_keysᵀ · h(t))
//!
//! `M_keys` are normalized stable templates (structural Rust n-grams: `fn`,
//! `pub`, `struct`, `impl`, `let`, `{`, `}`, `;`, …) and `M_vals` are the
//! clean representations fed into `W`. High inverse-temperature `β` makes the
//! read "hard" (nearest attractor wins), suppressing the autoregressive drift
//! of `h`. Even if `h` drifts, the softmax attracts it to the nearest stable
//! memory cell, so what reaches `W` is back inside the training distribution.
//!
//! Training/spec parity: the SAME `read_clean` must be applied to `h` at train
//! time (rec_test) and decode time (tm_generate_hopfield), otherwise W learns
//! on clean vectors but sees raw ones at inference — the exposure gap again.
use crate::ai::latent_jepa::{LatentVector, LATENT_DIM};

/// Modern Hopfield associative memory over Rust-structural byte templates.
#[derive(Clone, Debug)]
pub struct HopfieldMemory {
    /// Normalized key templates (invariant attractor cells). Kept as raw
    /// latent vectors; read normalizes the dot-products implicitly by
    /// temperature, so prenormalization is optional but helps β stability.
    pub keys: Vec<LatentVector>,
    /// Value vectors actually mixed into the W input per key.
    pub vals: Vec<LatentVector>,
    /// Inverse temperature (sharpness). High β = hard attractor selection.
    pub beta: f32,
}

impl HopfieldMemory {
    pub fn new(beta: f32) -> Self {
        Self {
            keys: Vec::new(),
            vals: Vec::new(),
            beta,
        }
    }

    /// Insert a template: `key` is the query-matching attractor, `val` is
    /// what reaches W when that attractor wins. Dedupes identical keys.
    pub fn memorize(&mut self, key: LatentVector, val: LatentVector) {
        for k in &self.keys {
            if k.cosine_similarity(&key) > 0.9999 {
                return; // already stored
            }
        }
        self.keys.push(key);
        self.vals.push(val);
    }

    /// Associative read: attracted clean state from a (possibly drifted) query.
    ///   h_clean[i] = Σ_j val_j[i] · softmax(β · key_j·query)
    pub fn read(&self, query: &LatentVector) -> LatentVector {
        if self.keys.is_empty() {
            return query.clone();
        }
        let n = self.keys.len();
        let mut scores = Vec::with_capacity(n);
        let mut min_v = f32::INFINITY;
        for k in &self.keys {
            // β·cos(q,key). cosine_bounded stabilizes β scaling.
            let s = self.beta * query.cosine_similarity(k);
            scores.push(s);
            min_v = min_v.min(s);
        }
        // softmax (subtract max for numerical safety).
        let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut w = vec![0.0f32; n];
        let mut sum_w = 0.0f32;
        for (i, s) in scores.iter().enumerate() {
            w[i] = (s - max_s).exp();
            sum_w += w[i];
        }
        let mut out = vec![0.0f32; LATENT_DIM];
        for (j, val) in self.vals.iter().enumerate() {
            let weight = w[j] / sum_w.max(1e-12);
            for d in 0..LATENT_DIM {
                out[d] += weight * val.values[d];
            }
        }
        let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in &mut out {
            *v /= norm;
        }
        LatentVector { values: out }
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }
}

/// Build a Rust-structural Hopfield memory: template byte-ngrams → their own
/// SDR → latent. Values = keys (clean auto-attractor). Pass `encoder` so the
/// same projection is used as in the byte predictor.
pub fn build_rust_hopfield(
    encoder: &crate::ai::latent_jepa::SdrEncoder,
    beta: f32,
) -> HopfieldMemory {
    let mut mem = HopfieldMemory::new(beta);
    // Structural Rust n-grams: punctuation and keywords that anchor syntax.
    const TEMPLATES: &[&str] = &[
        "fn ", "pub ", "struct ", "impl ", "let ", "mut ", "for ", "while ",
        "return ", "{", "}", "[", "]", "(", ")", ";", ",", "::", "->", "match ",
        "use ", "mod ", "trait ", "enum ", "const ", "static ", "&", "|", "=>",
    ];
    for t in TEMPLATES {
        let sdr = crate::ai::sdr::encode_bytes_sdr(t.as_bytes());
        let lat = encoder.encode(&sdr);
        mem.memorize(lat.clone(), lat);
    }
    mem
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hopfield_retrieves_nearest_tracker_under_drift() {
        let enc = crate::ai::latent_jepa::SdrEncoder::new(7);
        let mem = build_rust_hopfield(&enc, 12.0);
        assert!(mem.len() >= 15, "template bank too small: {}", mem.len());

        let fn_key = enc.encode(&crate::ai::sdr::encode_bytes_sdr(b"fn "));
        let mut noisy = fn_key.clone();
        let mut seed = 0xDEADu64;
        for i in 0..LATENT_DIM / 2 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            noisy.values[i] = ((seed >> 33) as f32 / 1e9 as f32) - 0.5;
        }
        let n = noisy.values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in &mut noisy.values {
            *v /= n;
        }
        let read = mem.read(&noisy);
        assert!(
            read.cosine_similarity(&fn_key) > noisy.cosine_similarity(&fn_key) + 0.05,
            "hopfield read did not reduce drift toward the attractor"
        );
    }
}
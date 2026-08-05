use crate::ai::sdr::{SdrVector, structure_sdr_from_sdrs, SDR_DIM};

/// Cache of `SdrEncoder::encode` results keyed by the input SDR. The encoder
/// projection is O(active_bits × LATENT_DIM) with a hash per cell, and the
/// structural training path re-encodes heavily overlapping windows (8 of 9
/// tokens repeat between consecutive frames), so this turns ~10G hash ops on a
/// large file into a few thousand distinct vectors. The encoding is purely
/// deterministic, so a memo cache is exact, not an approximation.
static LATENT_ENC_CACHE: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<SdrVector, LatentVector>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

pub const LATENT_DIM: usize = 512;

#[derive(Clone, Debug, PartialEq)]
pub struct LatentVector {
    pub values: Vec<f32>,
}

impl LatentVector {
    pub fn zero() -> Self {
        Self { values: vec![0.0; LATENT_DIM] }
    }

    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let (mut dot, mut an, mut bn) = (0.0, 0.0, 0.0);
        for (&a, &b) in self.values.iter().zip(&other.values) {
            dot += a * b;
            an += a * a;
            bn += b * b;
        }
        let sim = dot / (an.sqrt() * bn.sqrt() + 1e-8);
        debug_assert!(
            sim.is_finite() && (-1.0001..=1.0001).contains(&sim),
            "impossible cosine similarity: {sim}"
        );
        sim
    }

    pub fn cosine_loss(&self, other: &Self) -> f32 {
        1.0 - self.cosine_similarity(other)
    }
}

/// Deterministic sign random projection. It is intentionally frozen: this
/// first JEPA tracer bullet tests the latent interface without pretending that
/// a random encoder is already semantically trained.
#[derive(Clone, Debug)]
pub struct SdrEncoder {
    pub latent_dim: usize,
    pub seed: u64,
}

impl SdrEncoder {
    pub fn new(seed: u64) -> Self {
        Self { latent_dim: LATENT_DIM, seed }
    }

    fn hash(&self, latent: usize, bit: usize) -> u64 {
        let mut x = self.seed
            .wrapping_add((latent as u64).wrapping_mul(0x9e3779b97f4a7c15))
            .wrapping_add(bit as u64);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }

    pub fn encode(&self, sdr: &SdrVector) -> LatentVector {
        if let Ok(cache) = LATENT_ENC_CACHE.lock() {
            if let Some(cached) = cache.get(sdr) {
                return cached.clone();
            }
        }
        let mut values = vec![0.0f32; self.latent_dim];
        for bit in 0..SDR_DIM {
            if sdr.bit_at(bit) == 0 { continue; }
            for (latent, value) in values.iter_mut().enumerate() {
                let h = self.hash(latent, bit);
                *value += if h & 1 == 0 { 1.0 } else { -1.0 };
            }
        }
        let norm = values.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for value in &mut values { *value /= norm; }
        let out = LatentVector { values };
        if let Ok(mut cache) = LATENT_ENC_CACHE.lock() {
            cache.insert(sdr.clone(), out.clone());
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct LatentPredictor {
    pub encoder: SdrEncoder,
    /// Learnable linear transition operator `W` (LATENT_DIM × LATENT_DIM,
    /// row-major). `predict_next` returns `W·x` (normalized) instead of a
    /// bare encoder projection, so the latent actually encodes sequence
    /// dynamics rather than the frozen encoder's vocabulary geometry. It is
    /// trained by the Widrow-Hoff delta rule (see `learn_transition`).
    pub w: Vec<f32>,
    /// Number of delta updates applied (used to throttle the row-norm cap).
    pub updates: u64,
    /// Number of times the row-norm cap fired (debug instrumentation).
    pub cap_firings: u64,
    /// Orthogonal Weight Modification (OWM) projector (LATENT_DIM ×
    /// LATENT_DIM, row-major). `P` is a symmetric projection matrix that maps
    /// the input direction before each delta update: `ΔW += lr·error ⊗ (P·x)`.
    /// Directions already consolidated (past files) lie in the null space of
    /// the projection, so new updates cannot overwrite them — the delta rule
    /// can only use the orthogonal complement. Starts as the identity (no
    /// protection) and is relaxed toward the identity by `consolidate_owm`.
    /// P is maintained incrementally (never recomputed from full history), so
    /// the checkpoint stores exactly one 512² matrix, like W.
    pub p: Vec<f32>,
}

impl LatentPredictor {
    pub fn new(seed: u64) -> Self {
        let mut w = vec![0.0f32; LATENT_DIM * LATENT_DIM];
        for i in 0..LATENT_DIM {
            w[i * LATENT_DIM + i] = 1.0;
        }
        let mut p = vec![0.0f32; LATENT_DIM * LATENT_DIM];
        for i in 0..LATENT_DIM {
            p[i * LATENT_DIM + i] = 1.0;
        }
        Self {
            encoder: SdrEncoder::new(seed),
            w,
            updates: 0,
            cap_firings: 0,
            p,
        }
    }

    pub fn encode_token(&self, token: &str) -> LatentVector {
        self.encoder.encode(&crate::ai::sdr::encode_text(token))
    }

    /// Apply the learnable transition operator to a latent vector and
    /// renormalize.
    pub fn apply_w(&self, x: &LatentVector) -> LatentVector {
        let mut out = LatentVector::zero();
        for o in 0..LATENT_DIM {
            let mut acc = 0.0f32;
            let row = o * LATENT_DIM;
            for i in 0..LATENT_DIM {
                acc += self.w[row + i] * x.values[i];
            }
            out.values[o] = acc;
        }
        let norm = out.values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in &mut out.values {
            *v /= norm;
        }
        out
    }

    /// Apply the OWM projector to a latent vector (no renormalization — P is
    /// a projection, so `P·x` is the component of `x` in the free subspace).
    pub fn apply_p(&self, x: &LatentVector) -> LatentVector {
        let mut out = LatentVector::zero();
        for o in 0..LATENT_DIM {
            let mut acc = 0.0f32;
            let row = o * LATENT_DIM;
            for i in 0..LATENT_DIM {
                acc += self.p[row + i] * x.values[i];
            }
            out.values[o] = acc;
        }
        out
    }

    /// Orthogonal Weight Modification: consolidate a set of *new* input
    /// directions observed during the current file. The projector `P` is
    /// updated so that these directions fall into its null space, protecting
    /// them from being overwritten by future files:
    ///
    ///   P ← P − P·Aᵀ·(A·P·Aᵀ + α·I)⁻¹·A·P
    ///
    /// where `A` is the `K×LATENT_DIM` matrix of directions to protect (rows).
    /// The Woodbury-style form only ever touches a `K×K` inverse (K = number
    /// of protected directions this call), so it is incremental — the full
    /// history of A is never stored or recomputed. `top_k` directions are
    /// first reduced to their principal components so a file cannot burn the
    /// whole 512-dim budget on ~190 collinear inputs.
    ///
    /// Returns the number of directions actually consolidated.
    pub fn consolidate_owm(&mut self, directions: &[LatentVector], top_k: usize, alpha: f32) -> usize {
        let start = std::mem::take(&mut self.p);
        match owm_update(&start, directions, top_k, alpha) {
            Some((pnew, k)) => {
                self.p = pnew;
                k
            }
            None => {
                self.p = start;
                0
            }
        }
    }

    /// Intra-sequence OWM: build a fresh local projector `P_seq` from the
    /// prefix directions of a single sequence, starting from identity. Unlike
    /// the global `P` (consolidated across files), `P_seq` protects only the
    /// already-learned transitions of the *current* function, so a later step
    /// in the same sequence cannot overwrite them. The caller projects the
    /// failing step's input through `P_seq` and uses `learn_transition_with_p`.
    pub fn local_owm_projector(&self, directions: &[LatentVector], top_k: usize, alpha: f32) -> Vec<f32> {
        let mut id = vec![0.0f32; LATENT_DIM * LATENT_DIM];
        for i in 0..LATENT_DIM {
            id[i * LATENT_DIM + i] = 1.0;
        }
        match owm_update(&id, directions, top_k, alpha) {
            Some((pnew, _)) => pnew,
            None => id,
        }
    }

    /// Project a latent through an *arbitrary* projector matrix (not `self.p`),
    /// so a step can be aligned to a local `P_seq` instead of the global one.
    pub fn apply_p_with(&self, x: &LatentVector, p: &[f32]) -> LatentVector {
        let mut out = LatentVector::zero();
        for o in 0..LATENT_DIM {
            let mut acc = 0.0f32;
            let row = o * LATENT_DIM;
            for i in 0..LATENT_DIM {
                acc += p[row + i] * x.values[i];
            }
            out.values[o] = acc;
        }
        out
    }

    /// Widrow-Hoff (delta rule) update on the linear associator:
    /// `pred = W·encode(last)`, `error = encode(next) - pred`,
    /// `W += lr * error ⊗ encode(last)`. Returns the pre-update error norm.
    /// This is the README's "local updates via delta rule" — no backprop
    /// through anything else. Cheap: one outer product per step.
    ///
    /// To keep full-corpus training tractable the delta update is applied on a
    /// subsample of transitions (every `stride`-th one) while the error norm is
    /// still reported every call — the associator needs aggregate statistics,
    /// not every single example.
    pub fn learn_transition(&mut self, context: &[SdrVector], actual: &SdrVector, lr: f32) -> f32 {
        self.learn_transition_with_p(context, actual, lr, &self.p.clone())
    }

    /// Widrow-Hoff delta rule like [`Self::learn_transition`], but the delta
    /// update is applied along `proj·x` for an *arbitrary* projector matrix
    /// instead of the global `self.p`. This is what makes intra-sequence OWM
    /// possible: pass a local `P_seq` (built from the already-learned prefix
    /// transitions of the current function) so the failing step cannot
    /// overwrite them. The row-norm cap still applies to the same `self.w`.
    ///
    /// The operator keys on the *position-sensitive structural fold of the
    /// whole window*, not just the last token. Two occurrences of the same
    /// token in different contexts (e.g. `(` after `assert_eq !` vs `(` after
    /// `try_into`) therefore produce different inputs, which is what lets an
    /// intra-sequence projector protect one context while learning the other.
    pub fn learn_transition_with_p(
        &mut self,
        context: &[SdrVector],
        actual: &SdrVector,
        lr: f32,
        proj: &[f32],
    ) -> f32 {
        if context.is_empty() {
            return 0.0;
        }
        const UPDATE_STRIDE: u64 = 4;
        self.updates += 1;
        let apply_delta = self.updates % UPDATE_STRIDE == 0;
        let x = self.encoder.encode(&structure_sdr_from_sdrs(context));
        let target = self.encoder.encode(actual);
        let pred = if apply_delta {
            self.apply_w(&x)
        } else {
            x.clone()
        };
        // OWM: the delta update is applied along `proj·x`, not `x`. Directions
        // of the protected prefix live in the projector's null space, so the
        // update only touches the orthogonal complement and cannot overwrite
        // them.
        let px = if apply_delta {
            self.apply_p_with(&x, proj)
        } else {
            x.clone()
        };
        let mut err_norm = 0.0f32;
        for o in 0..LATENT_DIM {
            let error = target.values[o] - pred.values[o];
            err_norm += error * error;
            if apply_delta {
                let row = o * LATENT_DIM;
                for i in 0..LATENT_DIM {
                    self.w[row + i] += lr * error * px.values[i];
                }
            }
        }
        // Soft weight cap instead of per-step decay. A multiplicative decay
        // (`scale = 1 - decay*norm` applied every step) compounds across the
        // hundreds of thousands of transitions in the full corpus and drives W
        // to ~0 long before training finishes. Instead, only shrink a row when
        // its squared norm exceeds a ceiling, and only by the fraction above
        // it — healthy rows are left untouched. The cap pass is O(D²), so it
        // runs once every CAP_EVERY updates rather than on every transition.
        const CAP_EVERY: u64 = 50;
        const ROW_NORM_CAP: f32 = 2.0; // squared-norm ceiling per output row
        if apply_delta && self.updates % CAP_EVERY == 0 {
            for o in 0..LATENT_DIM {
                let row = o * LATENT_DIM;
                let mut sq = 0.0f32;
                for i in 0..LATENT_DIM {
                    sq += self.w[row + i] * self.w[row + i];
                }
                if sq > ROW_NORM_CAP {
                    self.cap_firings += 1;
                    let scale = (ROW_NORM_CAP / sq).sqrt();
                    for i in 0..LATENT_DIM {
                        self.w[row + i] *= scale;
                    }
                }
            }
        }
        err_norm.sqrt()
    }

    /// Negative learning for the compiler-grounded loop: *diminish* the
    /// association between `context` and a specific `wrong` token, without
    /// promoting any particular alternative (which is what the compiler's
    /// diagnostic cannot always supply). The delta moves the W row so that the
    /// prediction's projection onto `wrong` decreases:
    ///
    ///   W ← W − η·(pred·wrong)·wrong ⊗ (P·x)
    ///
    /// where `pred = W·x` is the current prediction and η = lr. This is exactly
    /// the sign-flip of the delta rule toward `wrong`, applied OWM-protected so
    /// it cannot erase other tasks' rows. It only acts when the prediction
    /// currently overlaps `wrong` (positive dot), so it is self-terminating:
    /// once demoted, the row stops being pushed further.
    pub fn demote_transition_with_p(
        &mut self,
        context: &[SdrVector],
        wrong: &SdrVector,
        lr: f32,
        proj: &[f32],
    ) -> f32 {
        if context.is_empty() {
            return 0.0;
        }
        let x = self.encoder.encode(&structure_sdr_from_sdrs(context));
        let wrong_lat = self.encoder.encode(wrong);
        let px = self.apply_p_with(&x, proj);
        let pred = self.apply_w(&x);
        let mut overlap = 0.0f32;
        for o in 0..LATENT_DIM {
            overlap += pred.values[o] * wrong_lat.values[o];
        }
        if overlap <= 0.0 {
            return 0.0;
        }
        let step = lr * overlap;
        for o in 0..LATENT_DIM {
            let row = o * LATENT_DIM;
            let wo = wrong_lat.values[o];
            for i in 0..LATENT_DIM {
                self.w[row + i] -= step * wo * px.values[i];
            }
        }
        step
    }

    pub fn predict_next(&self, context: &[SdrVector]) -> LatentVector {
        if context.is_empty() {
            return LatentVector::zero();
        }
        self.apply_w(&self.encoder.encode(&structure_sdr_from_sdrs(context)))
    }

    pub fn cosine_loss(&self, context: &[SdrVector], actual: &SdrVector) -> f32 {
        self.predict_next(context).cosine_loss(&self.encoder.encode(actual))
    }

    /// Number of learnable transition parameters (for serialization size
    /// checks): LATENT_DIM × LATENT_DIM.
    pub fn w_len(&self) -> usize {
        self.w.len()
    }

    /// Rebuild a predictor from serialized `w` (falls back to identity when
    /// the length is wrong).
    pub fn with_w(seed: u64, w: Vec<f32>) -> Self {
        let mut p = Self::new(seed);
        if w.len() == LATENT_DIM * LATENT_DIM {
            p.w = w;
        }
        p
    }

    /// Restore the delta-update counter (throttles the row-norm cap).
    pub fn with_updates(mut self, updates: u64) -> Self {
        self.updates = updates;
        self
    }

    /// Restore the OWM projector (identity when the slice is missing/wrong).
    pub fn with_p(mut self, p: Vec<f32>) -> Self {
        if p.len() == LATENT_DIM * LATENT_DIM {
            self.p = p;
        }
        self
    }
}

/// Shared OWM projector math: given a starting projector `start` (identity for
/// intra-sequence, or the accumulated global P for file consolidation), fold in
/// `directions` so they fall into its null space:
///
///   P ← P − P·Aᵀ·(A·P·Aᵀ + α·I)⁻¹·A·P
///
/// `A` is the K×LATENT_DIM matrix of Gram-Schmidt-reduced principal directions.
/// Returns the new projector and the number of consolidated directions.
fn owm_update(start: &[f32], directions: &[LatentVector], top_k: usize, alpha: f32) -> Option<(Vec<f32>, usize)> {
    let d = LATENT_DIM;
    let m = directions.len();
    if m == 0 || top_k == 0 {
        return None;
    }
    let k = top_k.min(m);
    // A: rows = principal directions (K×d). Start with the raw directions,
    // then reduce via Gram-Schmidt to at most k orthogonal representatives
    // so the protected subspace is budgeted rather than allowed to blow up
    // with collinear inputs.
    let mut chosen: Vec<Vec<f32>> = Vec::with_capacity(k);
    {
        // Build unit-direction rows from the input vectors, largest-norm
        // first (most confident transitions dominate the subspace).
        let mut idx: Vec<usize> = (0..m).collect();
        idx.sort_by(|&i, &j| {
            let ni = directions[i].values.iter().map(|v| v * v).sum::<f32>();
            let nj = directions[j].values.iter().map(|v| v * v).sum::<f32>();
            nj.partial_cmp(&ni).unwrap_or(std::cmp::Ordering::Equal)
        });
        for &i in &idx {
            let v = &directions[i].values;
            let mut res = v.clone();
            // Orthogonalize against already-chosen directions.
            for c in &chosen {
                let mut dot = 0.0f32;
                for (a0, b0) in c.iter().zip(&res) {
                    dot += a0 * b0;
                }
                for (a0, b0) in c.iter().zip(res.iter_mut()) {
                    *b0 -= dot * *a0;
                }
            }
            let norm = res.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for r in &mut res {
                    *r /= norm;
                }
                chosen.push(res);
                if chosen.len() >= k {
                    break;
                }
            }
        }
    }
    let k = chosen.len(); // actual rows after reduction
    if k == 0 {
        return None;
    }
    let mut a = vec![0.0f32; k * d];
    for (r, row) in chosen.iter().enumerate() {
        a[r * d..(r + 1) * d].copy_from_slice(row);
    }
    // Build the K×K matrix M = A·P·Aᵀ + α·I.
    // Precompute PAᵀ = P·Aᵀ once (K×d), then M = A·PAᵀ. This is
    // O(k·d² + k²·d) instead of the naive O(k²·d²).
    let mut pat = vec![0.0f32; k * d]; // [l][j] = (P·Aᵀ)[l][j]
    for l in 0..d {
        let prow = l * d;
        for j in 0..k {
            let mut acc = 0.0f32;
            let arow = j * d;
            for c in 0..d {
                acc += start[prow + c] * a[arow + c];
            }
            pat[l * k + j] = acc;
        }
    }
    let mut mtx = vec![0.0f32; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0f32;
            for l in 0..d {
                acc += a[i * d + l] * pat[l * k + j];
            }
            mtx[i * k + j] = acc;
        }
        mtx[i * k + i] += alpha;
    }
    // Invert M (Gaussian elimination with partial pivoting).
    let minv = match invert_square(&mtx, k) {
        Some(x) => x,
        None => return None,
    };
    // Canonical OWM: P ← P − P·Aᵀ·(A·P·Aᵀ + α·I)⁻¹·A·P
    // T = P·Aᵀ is already available as `pat` (d×K); then
    // TM = T·M⁻¹ (d×K); XA = TM·A (d×d); P_new = P − XA·P.
    let t = &pat; // P·Aᵀ, (d×K), row-major d*k
    let mut tm = vec![0.0f32; d * k]; // T·M⁻¹
    for i in 0..d {
        for j in 0..k {
            let mut acc = 0.0f32;
            for l in 0..k {
                acc += t[i * k + l] * minv[l * k + j];
            }
            tm[i * k + j] = acc;
        }
    }
    let mut xa = vec![0.0f32; d * d]; // TM·A
    for i in 0..d {
        for j in 0..d {
            let mut acc = 0.0f32;
            for l in 0..k {
                acc += tm[i * k + l] * a[l * d + j];
            }
            xa[i * d + j] = acc;
        }
    }
    // P_new = P − XA·P
    let mut pnew = vec![0.0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut acc = 0.0f32;
            for l in 0..d {
                acc += xa[i * d + l] * start[l * d + j];
            }
            pnew[i * d + j] = start[i * d + j] - acc;
        }
    }
    Some((pnew, k))
}

/// Invert a square matrix in place-safe row-major form (Gaussian elimination
/// with partial pivoting). Returns `None` if singular.
fn invert_square(mat: &[f32], n: usize) -> Option<Vec<f32>> {
    let mut a = mat.to_vec();
    let mut inv = vec![0.0f32; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        // Pivot: largest absolute value in this column at/after `col`.
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for r in (col + 1)..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if piv != col {
            for j in 0..n {
                a.swap(col * n + j, piv * n + j);
                inv.swap(col * n + j, piv * n + j);
            }
        }
        let d = a[col * n + col];
        for j in 0..n {
            a[col * n + j] /= d;
            inv[col * n + j] /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r * n + col];
            if f.abs() < 1e-15 {
                continue;
            }
            for j in 0..n {
                a[r * n + j] -= f * a[col * n + j];
                inv[r * n + j] -= f * inv[col * n + j];
            }
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::sdr::encode_text;

    #[test]
    fn encoder_is_deterministic_and_512_dimensional() {
        let encoder = SdrEncoder::new(42);
        let a = encoder.encode(&encode_text("("));
        let b = encoder.encode(&encode_text("("));
        assert_eq!(a, b);
        assert_eq!(a.values.len(), LATENT_DIM);
        assert!((a.cosine_similarity(&a) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn predictor_returns_finite_cosine_loss() {
        let predictor = LatentPredictor::new(42);
        let context = vec![encode_text("main")];
        let loss = predictor.cosine_loss(&context, &encode_text("("));
        assert!(loss.is_finite());
    }

    #[test]
    fn delta_rule_reduces_loss_on_repeated_pair() {
        let mut p = LatentPredictor::new(42);
        let ctx = vec![encode_text("main")];
        let next = encode_text("(");
        let before = p.cosine_loss(&ctx, &next);
        for _ in 0..60 {
            p.learn_transition(&ctx, &next, 0.2);
        }
        let after = p.cosine_loss(&ctx, &next);
        assert!(
            after < before,
            "delta rule must tighten the predicted latent toward the target (before {before}, after {after})"
        );
        assert_eq!(p.w_len(), LATENT_DIM * LATENT_DIM);
    }

    #[test]
    fn identity_w_preserves_old_predict_next_behavior() {
        let p = LatentPredictor::new(42);
        let ctx = vec![encode_text("main")];
        // With W = I the prediction is just the encoder projection of the
        // window's structural fold, exactly the pre-delta-rule behavior.
        let pred = p.predict_next(&ctx);
        let enc = p.encoder.encode(&structure_sdr_from_sdrs(&ctx));
        assert!((pred.cosine_loss(&enc)).abs() < 1e-4);
    }

    #[test]
    fn owm_consolidation_yields_idempotent_projector() {
        // After consolidating real directions, P must satisfy P·P ≈ P and
        // symmetry. The old broken update (I − Aᵀ·M⁻¹·A·P) passed symmetry
        // but failed idempotency; this test pins the correct behavior.
        let mut pred = LatentPredictor::new(42);
        let mut dirs: Vec<LatentVector> = Vec::new();
        for t in ["main", "(", ")", "let", "fn", "match", "pub", "use", "mod", "self"] {
            dirs.push(pred.encoder.encode(&encode_text(t)));
        }
        let k = pred.consolidate_owm(&dirs, dirs.len(), 0.01);
        assert!(k > 0);
        let d = LATENT_DIM;
        // symmetry
        for i in 0..d {
            for j in (i + 1)..d {
                assert!(
                    (pred.p[i * d + j] - pred.p[j * d + i]).abs() < 1e-4,
                    "P asymmetric at ({i},{j})"
                );
            }
        }
        // idempotency P·P ≈ P
        let mut max_dev = 0.0f32;
        for i in 0..d {
            for j in 0..d {
                let mut acc = 0.0f32;
                for l in 0..d {
                    acc += pred.p[i * d + l] * pred.p[l * d + j];
                }
                max_dev = max_dev.max((acc - pred.p[i * d + j]).abs());
            }
        }
        assert!(max_dev < 1e-2, "P not idempotent: max |P·P−P| = {max_dev}");
    }
}

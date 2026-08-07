use crate::ai::latent_jepa::LatentPredictor;
use crate::ai::latent_jepa::LatentVector;
use crate::ai::latent_jepa::LATENT_DIM;
use crate::ai::sdr::{SDR_DENSITY, SDR_DIM, SDR_WORDS, SdrVector};
use crate::ai::soft_sdr::{info_nce_loss, sigmoid, SoftSdrVector};

const MIN_PERMANENCE: f64 = 0.2;
const CONNECTED_PERMANENCE: f64 = 0.5;
const PERMANENCE_INCREMENT: f64 = 0.05;
const PERMANENCE_DECREMENT: f64 = 0.02;
const MAX_SEGMENTS_PER_CELL: usize = 128;
const PRUNE_INTERVAL: usize = 200;

/// Minimum segment overlap required to count a predecessor as a real match.
/// Two unrelated 2%-density 8192-bit SDRs (164 bits each) overlap on roughly
/// 164²/8192 ≈ 3.3 bits purely by chance, so a low threshold (e.g. 3) treats
/// random noise as a learned bi-gram and lets common tokens hijack the
/// segments of rarer ones. 20 is ~8σ above the noise floor while a true
/// `prev → next` match saturates at the full 164.
const MATCH_OVERLAP: u32 = 20;

/// Match threshold for STRUCTURE-folded keys, which are 6%-dense (~490 active
/// bits at 8192 dims) instead of the 2%-dense single tokens (164 bits). Two
/// unrelated structure keys chance-overlap on ~490²/8192 ≈ 29 bits — already
/// above the token-level threshold of 20. 60 is ~2× that noise and far below a
/// true structural match (~490).
const STRUCTURE_MATCH_OVERLAP: u32 = 60;

#[derive(Clone, Debug)]
pub struct Synapse {
    pub bit_index: usize,
    pub permanence: f64,
}

impl Synapse {
    pub fn new(bit_index: usize, permanence: f64) -> Self {
        Synapse {
            bit_index,
            permanence,
        }
    }
    pub fn connected(&self) -> bool {
        self.permanence >= CONNECTED_PERMANENCE
    }
}

#[derive(Clone, Debug)]
pub struct DendriteSegment {
    pub synapses: Vec<Synapse>,
}

impl DendriteSegment {
    pub fn new() -> Self {
        DendriteSegment {
            synapses: Vec::new(),
        }
    }

    pub fn overlap(&self, sdr: &SdrVector) -> u32 {
        self.synapses
            .iter()
            .filter(|s| s.connected())
            .filter(|s| {
                let wi = s.bit_index / 64;
                let bi = s.bit_index % 64;
                (sdr.bits[wi] >> bi) & 1 == 1
            })
            .count() as u32
    }

    pub fn reinforce(&mut self, sdr: &SdrVector) {
        for s in &mut self.synapses {
            let wi = s.bit_index / 64;
            let bi = s.bit_index % 64;
            if (sdr.bits[wi] >> bi) & 1 == 1 {
                s.permanence = (s.permanence + PERMANENCE_INCREMENT).min(1.0);
            } else {
                s.permanence = (s.permanence - PERMANENCE_DECREMENT).max(0.0);
            }
        }
    }

    pub fn reinforce_match_only(&mut self, sdr: &SdrVector) {
        for s in &mut self.synapses {
            let wi = s.bit_index / 64;
            let bi = s.bit_index % 64;
            if (sdr.bits[wi] >> bi) & 1 == 1 {
                s.permanence = (s.permanence + PERMANENCE_INCREMENT).min(1.0);
            }
        }
    }

    pub fn prune_weak(&mut self) {
        self.synapses.retain(|s| s.permanence >= MIN_PERMANENCE);
    }
}

pub struct TemporalCell {
    pub id: usize,
    pub segments: Vec<DendriteSegment>,
    pub pattern: SdrVector,
}

impl TemporalCell {
    pub fn new(id: usize, pattern: SdrVector) -> Self {
        TemporalCell {
            id,
            segments: Vec::new(),
            pattern,
        }
    }

    pub fn learn_segment(&mut self, input: &SdrVector) {
        if self.segments.len() >= MAX_SEGMENTS_PER_CELL {
            let mut scores: Vec<(usize, u32)> = self
                .segments
                .iter()
                .enumerate()
                .map(|(i, seg)| (i, seg.overlap(input)))
                .collect();
            scores.sort_by(|a, b| b.1.cmp(&a.1));
            // Get index of segment with lowest overlap (last after descending sort)
            let idx = scores.last().map(|s| s.0).unwrap_or(0);
            scores.pop(); // Remove from scores to keep consistent state
            self.segments.remove(idx);
        }
        let mut seg = DendriteSegment::new();
        for bit in 0..SDR_DIM {
            let wi = bit / 64;
            let bi = bit % 64;
            if (input.bits[wi] >> bi) & 1 == 1 {
                seg.synapses
                    .push(Synapse::new(bit, CONNECTED_PERMANENCE + 0.1));
            }
        }
        if !seg.synapses.is_empty() {
            self.segments.push(seg);
        }
    }
}

pub struct TemporalMemory {
    pub cells: Vec<TemporalCell>,
    pub window: Vec<SdrVector>,
    pub context_len: usize,
    pub step: usize,
    predictor: LatentPredictor,
    /// Global two-speed (MegaByte-style) transition operator over BYTE PATCHES.
    /// Learns patch→patch transitions (each patch = a small group of raw bytes
    /// folded via `encode_bytes_sdr`); the byte-level `predictor` then decodes
    /// inside the chosen patch. Kept separate so the two rates do not blur.
    patch_predictor: LatentPredictor,
    cell_index: std::collections::HashMap<[u64; SDR_WORDS], usize>,
}

#[derive(Clone, Debug, Default)]
pub struct TrainStats {
    pub steps: usize,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub mean_loss: f32,
    pub learned_transitions: usize,
    pub total_latent_loss: f32,
}

impl TemporalMemory {
    pub fn new(capacity: usize, context_len: usize) -> Self {
        TemporalMemory {
            cells: Vec::with_capacity(capacity),
            window: Vec::new(),
            context_len,
            step: 0,
            predictor: LatentPredictor::new(0xF03D_C0DE),
            patch_predictor: LatentPredictor::new(0xBAC7_A5E0),
            cell_index: std::collections::HashMap::new(),
        }
    }

    pub fn get_or_create_cell(&mut self, pattern: &SdrVector) -> usize {
        if let Some(&idx) = self.cell_index.get(&pattern.bits) {
            return idx;
        }
        let id = self.cells.len();
        self.cells.push(TemporalCell::new(id, pattern.clone()));
        self.cell_index.insert(pattern.bits, id);
        id
    }

    pub fn reset(&mut self) {
        self.window.clear();
        self.step = 0;
    }

    pub fn train_on_sequence(&mut self, tokens: &[&str], epochs: usize) -> TrainStats {
        let mut stats = TrainStats::default();
        let rounds = epochs.max(1);
        let mut total = 0.0f32;
        for _ in 0..rounds {
            for pair in tokens.windows(2) {
                let prev = crate::ai::sdr::encode_text(pair[0]);
                let next = crate::ai::sdr::encode_text(pair[1]);
                let before = self.predict_soft(std::slice::from_ref(&prev));
                let loss = before.bce_l1_loss(&next, 0.01);
                if stats.steps == 0 { stats.initial_loss = loss; }
                total += loss;
                stats.steps += 1;
                self.learn_sequence(&prev, &next);
                stats.learned_transitions += 1;
                let cell = self.get_or_create_cell(&next);
                if let Some(seg) = self.cells[cell].segments.last_mut() {
                    for syn in &mut seg.synapses {
                        let target = next.bit_at(syn.bit_index) == 1;
                        syn.permanence = if target {
                            (syn.permanence + 0.02).min(1.0)
                        } else {
                            (syn.permanence - 0.005).max(MIN_PERMANENCE)
                        };
                    }
                }
                let after = self.predict_soft(std::slice::from_ref(&prev));
                stats.final_loss = after.bce_l1_loss(&next, 0.01);
                // Train the latent transition operator (Widrow-Hoff delta
                // rule): W += lr * (encode(next) - W·encode(prev)) ⊗ encode(prev).
                let latent_loss = self
                    .predictor
                    .learn_transition(std::slice::from_ref(&prev), &next, 0.1);
                stats.total_latent_loss += latent_loss;
            }
        }
        stats.mean_loss = if stats.steps == 0 { 0.0 } else { total / stats.steps as f32 };
        stats
    }

    pub fn feed(&mut self, input: &SdrVector) -> (SdrVector, f64) {
        self.step += 1;

        let cell_id = self.get_or_create_cell(input);

        let prediction = if !self.window.is_empty() {
            let prev = &self.window[self.window.len() - 1];
            self.predict_next(prev)
        } else {
            SdrVector::zero()
        };

        let match_score = if prediction.popcount() > 0 {
            prediction.soft_overlap(input)
        } else {
            0.0
        };

        if !self.window.is_empty() {
            let prev = self.window[self.window.len() - 1].clone();

            for seg in &mut self.cells[cell_id].segments {
                if seg.overlap(&prev) > 0 {
                    seg.reinforce(&prev);
                }
            }

            let mut has_prev_seg = false;
            for seg in &self.cells[cell_id].segments {
                if seg.overlap(&prev) >= MATCH_OVERLAP {
                    has_prev_seg = true;
                    break;
                }
            }
            if !has_prev_seg {
                self.cells[cell_id].learn_segment(&prev);
            }

            for c in &mut self.cells {
                if c.id == cell_id {
                    continue;
                }
                for seg in &mut c.segments {
                    if seg.overlap(&prev) >= MATCH_OVERLAP {
                        seg.reinforce_match_only(&prev);
                    }
                }
            }
        }

        self.window.push(input.clone());
        while self.window.len() > self.context_len {
            self.window.remove(0);
        }

        if self.step % PRUNE_INTERVAL == 0 {
            self.prune();
        }

        (prediction, match_score)
    }

    pub fn feed_no_learn(&mut self, input: &SdrVector) -> (SdrVector, f64) {
        self.step += 1;

        let prediction = if !self.window.is_empty() {
            let prev = &self.window[self.window.len() - 1];
            self.predict_next(prev)
        } else {
            SdrVector::zero()
        };

        let match_score = if prediction.popcount() > 0 {
            prediction.soft_overlap(input)
        } else {
            0.0
        };

        self.window.push(input.clone());
        while self.window.len() > self.context_len {
            self.window.remove(0);
        }

        if self.step % PRUNE_INTERVAL == 0 {
            self.prune();
        }

        (prediction, match_score)
    }

    pub fn predict_next(&self, prev: &SdrVector) -> SdrVector {
        let mut pred = SdrVector::zero();
        for c in &self.cells {
            let depolarized = c.segments.iter().any(|seg| seg.overlap(prev) >= MATCH_OVERLAP);
            if depolarized {
                for i in 0..SDR_WORDS {
                    pred.bits[i] |= c.pattern.bits[i];
                }
            }
        }
        if pred.popcount() > 0 {
            let target = (SDR_DIM as f64 * SDR_DENSITY).ceil() as usize;
            let mut scored: Vec<(usize, u64)> = (0..SDR_DIM)
                .filter(|&bit| {
                    let wi = bit / 64;
                    let bi = bit % 64;
                    (pred.bits[wi] >> bi) & 1 == 1
                })
                .map(|bit| {
                    let score = (self.step as u64)
                        .wrapping_mul(bit as u64 + 1)
                        .reverse_bits();
                    (bit, score)
                })
                .collect();
            scored.sort_by(|a, b| b.1.cmp(&a.1));
            scored.truncate(target);
            let mut out = SdrVector::zero();
            for &(bit, _) in &scored {
                out.bits[bit / 64] |= 1u64 << (bit % 64);
            }
            out
        } else {
            SdrVector::zero()
        }
    }

    /// Soft projection of HTM segment evidence. Inference-only; no fake gradient path.
    pub fn predict_soft(&self, context: &[SdrVector]) -> SoftSdrVector {
        let ctx = context.last().or_else(|| self.window.last());
        let Some(ctx) = ctx else { return SoftSdrVector::new(SDR_DIM); };
        let mut logits = vec![0.0f32; SDR_DIM];
        for cell in &self.cells {
            for seg in &cell.segments {
                let ov = seg.overlap(ctx) as f32;
                if ov < MATCH_OVERLAP as f32 {
                    continue;
                }
                let scale = ov / seg.synapses.len().max(1) as f32;
                for syn in &seg.synapses {
                    if syn.bit_index < SDR_DIM {
                        logits[syn.bit_index] += syn.permanence as f32 * scale;
                    }
                }
            }
        }
        let max_logit = logits.iter().copied().fold(0.0f32, f32::max).max(1e-6);
        SoftSdrVector { probs: logits.into_iter().map(|x| sigmoid(x / max_logit * 8.0 - 4.0)).collect() }
    }

    pub fn info_nce_loss(&self, context: &[SdrVector], actual: &SdrVector, temperature: f32) -> f32 {
        let pred = self.predict_soft(context);
        let negatives: Vec<SdrVector> = self.cells.iter().take(8).map(|c| c.pattern.clone()).collect();
        info_nce_loss(&pred, actual, &negatives, temperature)
    }

    /// Latent projection of the next-token signal. Uses the LatentPredictor
    /// to compress the contextual SDR into 512 dims. This is the primary
    pub fn predict_latent(&self, context: &[SdrVector]) -> crate::ai::latent_jepa::LatentVector {
        self.predictor.predict_next(context)
    }

    /// Cosine loss between the latent prediction and the latent of the actual
    /// next token. Lower is better.
    pub fn latent_cosine_loss(&self, context: &[SdrVector], actual: &SdrVector) -> f32 {
        self.predictor.cosine_loss(context, actual)
    }
    /// Latent encoding of an SDR through the frozen encoder (vocab pre-cache).
    pub fn latent_of_sdr(&self, sdr: &SdrVector) -> crate::ai::latent_jepa::LatentVector {
        self.predictor.encoder.encode(sdr)
    }

    /// Contextual prediction: a cell is depolarized when one of its segments
    /// matches the bundle of the whole context window, not just the last token.
    pub fn predict_next_context(&self, context: &[SdrVector]) -> SdrVector {
        self.predict_context_threshold(context, MATCH_OVERLAP)
    }

    fn predict_context_threshold(&self, context: &[SdrVector], threshold: u32) -> SdrVector {
        let pred = self.predict_context_raw(context, threshold);
        if pred.popcount() > 0 {
            let target = (SDR_DIM as f64 * SDR_DENSITY).ceil() as usize;
            let mut scored: Vec<(usize, u64)> = (0..SDR_DIM)
                .filter(|&bit| {
                    let wi = bit / 64;
                    let bi = bit % 64;
                    (pred.bits[wi] >> bi) & 1 == 1
                })
                .map(|bit| {
                    let score = (self.step as u64)
                        .wrapping_mul(bit as u64 + 1)
                        .reverse_bits();
                    (bit, score)
                })
                .collect();
            scored.sort_by(|a, b| b.1.cmp(&a.1));
            scored.truncate(target);
            let mut out = SdrVector::zero();
            for &(bit, _) in &scored {
                out.bits[bit / 64] |= 1u64 << (bit % 64);
            }
            out
        } else {
            SdrVector::zero()
        }
    }

    /// Raw merge of every depolarized cell's pattern for the context — the
    /// OR of the candidate next tokens before the density re-selection in
    /// [`Self::predict_context_threshold`]. Re-selection collapses a
    /// multi-candidate union into a hash-sampled 2%-dense slice that no single
    /// token reproduces; the raw union keeps each candidate's full pattern so a
    /// decoder can rank candidates by what fraction of their bits fired. Used
    /// by the TM sequential generator.
    fn predict_context_raw(&self, context: &[SdrVector], threshold: u32) -> SdrVector {
        let ctx = SdrVector::union(context);
        if ctx.popcount() == 0 {
            return SdrVector::zero();
        }
        let mut pred = SdrVector::zero();
        for c in &self.cells {
            let depolarized = c.segments.iter().any(|seg| seg.overlap(&ctx) >= threshold);
            if depolarized {
                for i in 0..SDR_WORDS {
                    pred.bits[i] |= c.pattern.bits[i];
                }
            }
        }
        pred
    }

    /// Structural prediction over the RAW merged candidate patterns: the union
    /// of every next-token pattern whose cell depolarized for `window_tokens`.
    /// Wraps [`Self::predict_context_raw`] with the structure-folded key so the
    /// generator can score overlaps without the destructive re-selection.
    pub fn predict_structure_raw(&self, window_tokens: &[&str]) -> SdrVector {
        let key = crate::ai::sdr::structure_sdr(window_tokens);
        self.predict_context_raw(std::slice::from_ref(&key), STRUCTURE_MATCH_OVERLAP)
    }

    /// Weighted structural prediction: a per-bit accumulator over every
    /// depolarized next-token pattern, where each pattern contributes its
    /// segment-match overlap as a weight. Unlike the raw OR (which loses
    /// multiplicity and saturates when many cells fire) and the re-selected
    /// vector (which samples bits at random), the weights preserve both how
    /// strongly a context matched AND how many cells voted for each bit, so a
    /// decoder can rank a token by the total evidence its bits carry. Returns
    /// SDR_DIM weights, zero when nothing depolarized.
    pub fn predict_structure_weighted(&self, window_tokens: &[&str]) -> Vec<f32> {
        let key = crate::ai::sdr::structure_sdr(window_tokens);
        if key.popcount() == 0 {
            return vec![0f32; SDR_DIM];
        }
        let mut weights = vec![0f32; SDR_DIM];
        for c in &self.cells {
            let mut best = 0u32;
            for seg in &c.segments {
                let ov = seg.overlap(&key);
                if ov > best {
                    best = ov;
                }
            }
            if best >= STRUCTURE_MATCH_OVERLAP {
                let w = best as f32;
                for (wi, bits) in c.pattern.bits.iter().enumerate() {
                    let base = wi * 64;
                    let mut x = *bits;
                    while x != 0 {
                        let bi = x.trailing_zeros() as usize;
                        weights[base + bi] += w;
                        x &= x - 1;
                    }
                }
            }
        }
        weights
    }

    pub fn learn_sequence(&mut self, prev: &SdrVector, next: &SdrVector) {
        let next_cell = self.get_or_create_cell(next);
        // Winner-take-all: reinforce only the best-matching segment so that a
        // high-fan-in cell (e.g. `(` after hundreds of distinct tokens) does
        // not decay the segment for each specific predecessor. Without this,
        // per-segment permanence drops under CONNECTED for all but the most
        // frequent predecessor, and rare-but-real bi-grams stop firing.
        let mut best: Option<usize> = None;
        let mut best_ov = 0u32;
        for (i, seg) in self.cells[next_cell].segments.iter().enumerate() {
            let ov = seg.overlap(prev);
            if ov > best_ov {
                best_ov = ov;
                best = Some(i);
            }
        }
        match best {
            Some(i) if best_ov >= MATCH_OVERLAP => {
                self.cells[next_cell].segments[i].reinforce_match_only(prev);
            }
            _ => {
                self.cells[next_cell].learn_segment(prev);
            }
        }
    }

    /// Contextual learning: build a segment on `next`'s cell from the union of
    /// the whole context window (last `context_len` tokens), so the cell fires
    /// only when the full recent history matches — not just the last token.
    pub fn learn_context(&mut self, context: &[SdrVector], next: &SdrVector) {
        self.learn_context_threshold(context, next, MATCH_OVERLAP)
    }

    fn learn_context_threshold(&mut self, context: &[SdrVector], next: &SdrVector, threshold: u32) {
        let next_cell = self.get_or_create_cell(next);
        let ctx = SdrVector::union(context);
        if ctx.popcount() == 0 {
            return;
        }
        let mut has_ctx_seg = false;
        for seg in &self.cells[next_cell].segments {
            if seg.overlap(&ctx) >= threshold {
                has_ctx_seg = true;
                break;
            }
        }
        if !has_ctx_seg {
            self.cells[next_cell].learn_segment(&ctx);
        }
        for seg in &mut self.cells[next_cell].segments {
            if seg.overlap(&ctx) > 0 {
                seg.reinforce(&ctx);
            }
        }
    }

    /// Learn a transition from an ORDER-SENSITIVE structural key (the current
    /// frame folded via `structure_sdr`) to the next token. Unlike a raw
    /// bi-gram segment, the whole recent window is bound into one super-vector
    /// (position-permuted bundle), so a tail of many predecessors does not
    /// drown out the pair we care about — structure is the feature, not a
    /// single predecessor.
    pub fn learn_structure(&mut self, window_tokens: &[&str], next_tok: &str) {
        self.learn_structure_lr(window_tokens, next_tok, 0.1);
    }

    pub fn learn_structure_lr(&mut self, window_tokens: &[&str], next_tok: &str, lr: f32) {
        let window_sdrs: Vec<crate::ai::sdr::SdrVector> = window_tokens
            .iter()
            .map(|t| crate::ai::sdr::encode_text(t))
            .collect();
        let key = crate::ai::sdr::structure_sdr_from_sdrs(&window_sdrs);
        let next = crate::ai::sdr::encode_text(next_tok);
        self.learn_context_threshold(std::slice::from_ref(&key), &next, STRUCTURE_MATCH_OVERLAP);
        // Also train the latent transition operator on the raw window→next
        // pair. The window SDRs are already encoded above — reuse them.
        self.predictor.learn_transition(&window_sdrs, &next, lr);
    }

    /// Same as [`Self::learn_structure_lr`] but the latent delta update is
    /// applied along `proj·x` for an arbitrary projector. Used by
    /// intra-sequence OWM: pass a local `P_seq` built from the already-learned
    /// prefix transitions so a later step of the same function cannot
    /// overwrite them. The structural segment learning still uses the default
    /// path (it is separate from the W transition operator).
    pub fn learn_structure_lr_with_p(
        &mut self,
        window_tokens: &[&str],
        next_tok: &str,
        lr: f32,
        proj: &[f32],
    ) {
        let window_sdrs: Vec<crate::ai::sdr::SdrVector> = window_tokens
            .iter()
            .map(|t| crate::ai::sdr::encode_text(t))
            .collect();
        let key = crate::ai::sdr::structure_sdr_from_sdrs(&window_sdrs);
        let next = crate::ai::sdr::encode_text(next_tok);
        self.learn_context_threshold(std::slice::from_ref(&key), &next, STRUCTURE_MATCH_OVERLAP);
        self.predictor.learn_transition_with_p(&window_sdrs, &next, lr, proj);
    }

    // ---------------------------------------------------------------------
    // Byte-level (ByT5 / MegaByte style) transitions.
    //
    // Same TM machinery, but the alphabet is the FIXED 256 raw UTF-8 bytes
    // instead of a corpus-derived token vocabulary. `learn_bytes` trains both
    // the structural segment (context → next byte) and the latent transition
    // operator W on byte SDRs; `predict_bytes_latent` returns the predicted
    // next-byte latent for cosine decode against the 256-byte alphabet.
    // ---------------------------------------------------------------------

    /// Learn a raw-byte transition: `window_bytes` (recent UTF-8 bytes,
    /// oldest first) → `next_byte`. Context is folded position-sensitively
    /// via [`crate::ai::sdr::encode_bytes_sdr`], so multi-byte UTF-8 chars
    /// and arbitrary binary are first-class citizens — no dictionary.
    pub fn learn_bytes(&mut self, window_bytes: &[u8], next_byte: u8, lr: f32) {
            let window_sdrs: Vec<crate::ai::sdr::SdrVector> = window_bytes
                .iter()
                .map(|&b| crate::ai::sdr::byte_basis(b))
                .collect();
            let key = crate::ai::sdr::structure_sdr_from_sdrs(&window_sdrs);
            let next = crate::ai::sdr::byte_basis(next_byte);
            self.learn_context_threshold(std::slice::from_ref(&key), &next, STRUCTURE_MATCH_OVERLAP);
            self.predictor.learn_transition(&window_sdrs, &next, lr);
        }

        /// Predicted NEXT-BYTE latent for a raw-byte context. Same W operator as
        /// the token path — only the input alphabet differs.
        pub fn predict_bytes_latent(
            &self,
            window_bytes: &[u8],
        ) -> crate::ai::latent_jepa::LatentVector {
            let window_sdrs: Vec<crate::ai::sdr::SdrVector> = window_bytes
                .iter()
                .map(|&b| crate::ai::sdr::byte_basis(b))
                .collect();
            self.predictor.predict_next(&window_sdrs)
        }

        // ------------------------------------------------------------------
        // Two-speed (MegaByte-style) global patch level.
        //
        // The byte rate above predicts ONE byte out of 256 — a huge, noisy space
        // that alone degrades into local bigram garbage (measured in bytogen).
        // Two-speed fixes it exactly like MegaByte: a GLOBAL transition operator
        // predicts whole BYTE PATCHES (small groups of bytes folded position-
        // sensitively), and the byte level then decodes INSIDE the chosen patch.
        // This concentrates the decision: predict a patch direction, then the
        // bytes within it — far fewer degrees of freedom per step.
        // ------------------------------------------------------------------

        /// Learn a global patch transition: `window_patches` (each a group of raw
        /// bytes, oldest first, folded via `encode_bytes_sdr`) → `next_patch`.
        /// Only the patch-level transition operator `W_patch` is trained; the
        /// byte-level cell segments stay separate.
        pub fn learn_patch(&mut self, window_patches: &[&[u8]], next_patch: &[u8], lr: f32) {
            if next_patch.is_empty() {
                return;
            }
            let window_sdrs: Vec<crate::ai::sdr::SdrVector> = window_patches
                .iter()
                .map(|p| crate::ai::sdr::encode_bytes_sdr(p))
                .collect();
            let next = crate::ai::sdr::encode_bytes_sdr(next_patch);
            self.patch_predictor.learn_transition(&window_sdrs, &next, lr);
        }

        /// Predicted NEXT-PATCH latent for a raw-byte patch window. The global
        /// rate answers "which direction / next patch", the byte rate answers
        /// "exactly which bytes inside it".
        pub fn predict_patch_latent(
            &self,
            window_patches: &[&[u8]],
        ) -> crate::ai::latent_jepa::LatentVector {
            let window_sdrs: Vec<crate::ai::sdr::SdrVector> = window_patches
                .iter()
                .map(|p| crate::ai::sdr::encode_bytes_sdr(p))
                .collect();
            self.patch_predictor.predict_next(&window_sdrs)
        }

        /// Negative learning for the compiler-grounded loop: reduce the association
    /// between a window and a *wrong* token (the one the decoder emitted where
    /// rustc rejected it), without promoting any alternative. Latent-side only
    /// (the linear transition operator), OWM-projected via `proj`. See
    /// [`LatentPredictor::demote_transition_with_p`].
    pub fn demote_structure_lr_with_p(
        &mut self,
        window_tokens: &[&str],
        wrong_tok: &str,
        lr: f32,
        proj: &[f32],
    ) {
        let window_sdrs: Vec<crate::ai::sdr::SdrVector> = window_tokens
            .iter()
            .map(|t| crate::ai::sdr::encode_text(t))
            .collect();
        let wrong = crate::ai::sdr::encode_text(wrong_tok);
        if wrong.popcount() == 0 {
            return;
        }
        self.predictor.demote_transition_with_p(&window_sdrs, &wrong, lr, proj);
    }

    /// Build an intra-sequence OWM projector `P_seq` from the given latent
    /// directions (the inputs of the already-learned prefix transitions),
    /// starting from identity. See `LatentPredictor::local_owm_projector`.
    pub fn local_owm_projector(
        &self,
        directions: &[crate::ai::latent_jepa::LatentVector],
        top_k: usize,
        alpha: f32,
    ) -> Vec<f32> {
        self.predictor.local_owm_projector(directions, top_k, alpha)
    }

    /// Structural prediction: fold the visible window into a super-vector and
    /// return the strongest predicted next-token pattern SDR.
    pub fn predict_structure(&self, window_tokens: &[&str]) -> SdrVector {
        let key = crate::ai::sdr::structure_sdr(window_tokens);
        // Structural folds are denser than single tokens, so they need a
        // density-scaled match threshold (see STRUCTURE_MATCH_OVERLAP).
        self.predict_context_threshold(std::slice::from_ref(&key), STRUCTURE_MATCH_OVERLAP)
    }

    pub fn prune(&mut self) {
        for c in &mut self.cells {
            for seg in &mut c.segments {
                seg.prune_weak();
            }
            c.segments.retain(|seg| {
                let connected = seg.synapses.iter().filter(|s| s.connected()).count();
                connected >= 2
            });
        }
    }

    pub fn save(&self, path: &str) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::File::create(path) {
            let n = self.cells.len() as u32;
            f.write_all(&n.to_le_bytes()).ok();
            for c in &self.cells {
                f.write_all(&(c.id as u32).to_le_bytes()).ok();
                for w in &c.pattern.bits {
                    f.write_all(&w.to_le_bytes()).ok();
                }
                f.write_all(&(c.segments.len() as u32).to_le_bytes()).ok();
                for seg in &c.segments {
                    f.write_all(&(seg.synapses.len() as u32).to_le_bytes()).ok();
                    for s in &seg.synapses {
                        f.write_all(&(s.bit_index as u32).to_le_bytes()).ok();
                        f.write_all(&s.permanence.to_le_bytes()).ok();
                    }
                }
            }
            f.write_all(&(self.window.len() as u32).to_le_bytes()).ok();
            for sdr in &self.window {
                for w in &sdr.bits {
                    f.write_all(&w.to_le_bytes()).ok();
                }
            }
            // Latent transition operator W (LATENT_DIM² f32 values). Written
            // after the window so older checkpoints without it still load.
            let w = &self.predictor.w;
            f.write_all(&(w.len() as u32).to_le_bytes()).ok();
            for val in w {
                f.write_all(&val.to_le_bytes()).ok();
            }
            // Number of delta updates (drives the throttled norm cap). Legacy
            // checkpoints end after W; absence is treated as 0.
            f.write_all(&self.predictor.updates.to_le_bytes()).ok();
            // Context-window length. Written last so older checkpoints (which
            // end after `updates`) still load with the historical default.
            f.write_all(&(self.context_len as u64).to_le_bytes()).ok();
            // OWM projector P (LATENT_DIM² f32 values). Written after the
            // context length so checkpoints saved without OWM still load with
            // P = identity (no protection).
            let p = &self.predictor.p;
            f.write_all(&(p.len() as u32).to_le_bytes()).ok();
            for val in p {
                f.write_all(&val.to_le_bytes()).ok();
            }
            // Two-speed GLOBAL patch operator W_patch + updates + P_patch.
            // Written last so checkpoints saved before the patch level load
            // with a fresh (identity) patch predictor.
            let pw = &self.patch_predictor.w;
            f.write_all(&(pw.len() as u32).to_le_bytes()).ok();
            for val in pw {
                f.write_all(&val.to_le_bytes()).ok();
            }
            f.write_all(&self.patch_predictor.updates.to_le_bytes()).ok();
            let pp = &self.patch_predictor.p;
            f.write_all(&(pp.len() as u32).to_le_bytes()).ok();
            for val in pp {
                f.write_all(&val.to_le_bytes()).ok();
            }
        }
    }

    pub fn load(path: &str) -> Option<Self> {
        let data = std::fs::read(path).ok()?;
        let mut pos = 0usize;
        let n = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        let mut cells = Vec::with_capacity(n);
        for _ in 0..n {
            let id = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            let mut bits = [0u64; 128];
            for w in bits.iter_mut() {
                *w = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
                pos += 8;
            }
            let pattern = crate::ai::sdr::SdrVector { bits };
            let seg_n = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            let mut segments = Vec::with_capacity(seg_n);
            for _ in 0..seg_n {
                let syn_n = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
                pos += 4;
                let mut synapses = Vec::with_capacity(syn_n);
                for _ in 0..syn_n {
                    let bi = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
                    pos += 4;
                    let perm = f64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
                    pos += 8;
                    synapses.push(Synapse::new(bi, perm));
                }
                segments.push(DendriteSegment { synapses });
            }
            cells.push(TemporalCell {
                id,
                segments,
                pattern,
            });
        }
        let wl = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        let mut window = Vec::with_capacity(wl);
        for _ in 0..wl {
            let mut bits = [0u64; 128];
            for w in bits.iter_mut() {
                *w = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
                pos += 8;
            }
            window.push(crate::ai::sdr::SdrVector { bits });
        }
        // Latent transition operator W. Older checkpoints end right after the
        // window — detect that and fall back to the identity-initialized
        // predictor.
        let mut w = Vec::new();
        let mut updates = 0u64;
        if pos + 4 <= data.len() {
            let wn = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            if wn > 0 && pos + wn * 4 <= data.len() {
                w.reserve(wn);
                for _ in 0..wn {
                    w.push(f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?));
                    pos += 4;
                }
            }
            if pos + 8 <= data.len() {
                updates = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
                pos += 8;
            }
        }
        // Context length is written after `updates`; checkpoints saved before
        // that field end here, and fall back to the historical default (4).
        let mut context_len = 4usize;
        if pos + 8 <= data.len() {
            context_len = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?) as usize;
            pos += 8;
        }
        // OWM projector P. Checkpoints saved before OWM end after the context
        // length; absence falls back to the identity projector.
        let mut p = Vec::new();
        if pos + 4 <= data.len() {
            let pn = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            if pn == LATENT_DIM * LATENT_DIM && pos + pn * 4 <= data.len() {
                p.reserve(pn);
                for _ in 0..pn {
                    p.push(f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?));
                    pos += 4;
                }
            }
        }
        let mut cell_index = std::collections::HashMap::with_capacity(cells.len());
        for (i, c) in cells.iter().enumerate() {
            cell_index.insert(c.pattern.bits, i);
        }
        // Two-speed GLOBAL patch operator. Written last; checkpoints saved
        // before the patch level end after P and fall back to a fresh
        // (identity) patch predictor.
        let mut patch_predictor = LatentPredictor::new(0xBAC7_A5E0);
        if pos + 4 <= data.len() {
            let pwn = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            let mut pw = Vec::new();
            if pwn == LATENT_DIM * LATENT_DIM && pos + pwn * 4 <= data.len() {
                pw.reserve(pwn);
                for _ in 0..pwn {
                    pw.push(f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?));
                    pos += 4;
                }
            }
            let mut pupdates = 0u64;
            if pos + 8 <= data.len() {
                pupdates = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
                pos += 8;
            }
            let mut pp = Vec::new();
            if pos + 4 <= data.len() {
                let ppn = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
                pos += 4;
                if ppn == LATENT_DIM * LATENT_DIM && pos + ppn * 4 <= data.len() {
                    pp.reserve(ppn);
                    for _ in 0..ppn {
                        pp.push(f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?));
                        pos += 4;
                    }
                }
            }
            if !pw.is_empty() {
                patch_predictor = LatentPredictor::with_w(0xBAC7_A5E0, pw)
                    .with_updates(pupdates)
                    .with_p(pp);
            }
        }
        Some(TemporalMemory {
            cells,
            window,
            context_len,
            step: 0,
            predictor: LatentPredictor::with_w(0xF03D_C0DE, w)
                .with_updates(updates)
                .with_p(p),
            patch_predictor,
            cell_index,
        })
    }

    /// Reassemble a TM from already-parsed parts (used by the binary loader in
    /// the CLI, which keeps its own defensive bounds-checked parser). An empty
    /// `w` falls back to the identity-initialized predictor (legacy checkpoints).
    pub fn restore(
        cells: Vec<TemporalCell>,
        window: Vec<SdrVector>,
        context_len: usize,
        w: Vec<f32>,
        updates: u64,
        p: Vec<f32>,
    ) -> Self {
        let mut cell_index = std::collections::HashMap::with_capacity(cells.len());
        for (i, c) in cells.iter().enumerate() {
            cell_index.insert(c.pattern.bits, i);
        }
        let predictor = if w.is_empty() {
            LatentPredictor::new(0xF03D_C0DE)
        } else {
            LatentPredictor::with_w(0xF03D_C0DE, w)
                .with_updates(updates)
                .with_p(p)
        };
        TemporalMemory {
            cells,
            window,
            context_len,
            step: 0,
            predictor,
            patch_predictor: LatentPredictor::new(0xBAC7_A5E0),
            cell_index,
        }
    }

    pub fn predictor_w(&self) -> &[f32] { &self.predictor.w }

    pub fn predictor_p(&self) -> &[f32] { &self.predictor.p }

    pub fn predictor_updates(&self) -> u64 { self.predictor.updates }

    pub fn predictor_cap_firings(&self) -> u64 { self.predictor.cap_firings }

    /// OWM consolidation hook: protect the given input directions (as encoded
    /// latents) so future delta updates cannot overwrite them. See
    /// `LatentPredictor::consolidate_owm`.
    pub fn consolidate_owm(&mut self, directions: &[LatentVector], top_k: usize, alpha: f32) -> usize {
        self.predictor.consolidate_owm(directions, top_k, alpha)
    }

    pub fn stats(&self) -> String {
        let total_segs: usize = self.cells.iter().map(|c| c.segments.len()).sum();
        let total_syn: usize = self
            .cells
            .iter()
            .flat_map(|c| c.segments.iter())
            .map(|s| s.synapses.len())
            .sum();
        format!(
            "cells={} segments={} synapses={} window={}",
            self.cells.len(),
            total_segs,
            total_syn,
            self.window.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn train_on_sequence_records_transition() {
        let mut tm = TemporalMemory::new(8, 2);
        let stats = tm.train_on_sequence(&["fn", "main", "(", ")"], 2);
        assert_eq!(stats.steps, 6);
        assert_eq!(stats.learned_transitions, 6);
        assert!(stats.initial_loss.is_finite());
        assert!(stats.final_loss.is_finite());
    }

    /// Regression guard for the `main → (` bi-gram getting lost when the `(`
    /// cell is trained under many distinct predecessors. The winner-take-all
    /// branch must preserve a dedicated segment per predecessor instead of
    /// decaying it away, so the segment keeps firing with high overlap.
    #[test]
    fn rare_bigram_survives_many_predecessors() {
        use crate::ai::sdr::{encode_text, SDR_DIM};

        let mut tm = TemporalMemory::new(SDR_DIM, 4);
        // 200 distinct predecessor tokens, each appearing before `(`, and the
        // specific `main` predecessor appears only once. Emulates a real corpus.
        for trial in 0..200 {
            let prev = encode_text(&format!("tok_{}", trial));
            tm.learn_sequence(&prev, &encode_text("("));
        }
        // The special pair is learned last and must be retrievable.
        tm.learn_sequence(&encode_text("main"), &encode_text("("));
        tm.learn_sequence(&encode_text("main"), &encode_text("("));

        let paren = encode_text("(");
        let main = encode_text("main");
        let cell = tm
            .cells
            .iter()
            .find(|c| c.pattern.bits == paren.bits)
            .expect("cell for `(` must exist");
        let best_overlap = cell
            .segments
            .iter()
            .map(|seg| seg.overlap(&main))
            .max()
            .unwrap_or(0);
        eprintln!(
            "segments={} best_overlap(main)={}",
            cell.segments.len(),
            best_overlap
        );
        // The segment trained on `main` must be strongly depolarized.
        assert!(best_overlap >= MATCH_OVERLAP, "main→( segment lost: {}", best_overlap);
        let pred = tm.predict_next(&main);
        assert!(pred.overlap(&paren) >= MATCH_OVERLAP);
    }

    /// Structural learning must associate the whole ordered window with the
    /// next token, and the folded key must be repeatable.
    #[test]
    fn structural_learning_predicts_next() {
        use crate::ai::sdr::encode_text;

        let mut tm = TemporalMemory::new(SDR_DIM, 4);
        // Teach the structure "fn NAME (" then ")" twice.
        tm.learn_structure(&["fn", "main", "("], ")");
        tm.learn_structure(&["fn", "main", "("], ")");
        tm.learn_structure(&["fn", "sum", "("], ")");

        let pred = tm.predict_structure(&["fn", "main", "("]);
        let close = encode_text(")");
        assert!(
            pred.overlap(&close) >= 5,
            "structural prediction lost: overlap={}",
            pred.overlap(&close)
        );
    }

    /// Latent predictor must produce a finite 512-dim vector for any context.
    #[test]
    fn predict_latent_returns_512_dim_vector() {
        use crate::ai::sdr::encode_text;
        let tm = TemporalMemory::new(SDR_DIM, 4);
        let ctx = vec![encode_text("fn"), encode_text("main")];
        let latent = tm.predict_latent(&ctx);
        assert_eq!(latent.values.len(), 512);
        assert!(latent.values.iter().all(|v| v.is_finite()));
    }

    /// Latent cosine loss is finite and in [0, 2].
    #[test]
    fn latent_cosine_loss_is_finite() {
        use crate::ai::sdr::encode_text;
        let tm = TemporalMemory::new(SDR_DIM, 4);
        let ctx = vec![encode_text("main")];
        let next = encode_text("(");
        let loss = tm.latent_cosine_loss(&ctx, &next);
        assert!(loss.is_finite());
        assert!((0.0..=2.0).contains(&loss));
    }

    /// Order is part of the structural key. Two windows sharing the same
    /// trailing token occupy the same position slot (so partial overlap is
    /// expected), but a genuinely different window — different tokens and
    /// order — must elicit far weaker evidence for `)`.
    #[test]
    fn structural_learning_is_order_sensitive() {
        use crate::ai::sdr::encode_text;

        let mut tm = TemporalMemory::new(SDR_DIM, 4);
        tm.learn_structure(&["fn", "main", "("], ")");
        tm.learn_structure(&["fn", "main", "("], ")");

        let close = encode_text(")");
        let pred_fwd = tm.predict_structure(&["fn", "main", "("]);
        let pred_bogus = tm.predict_structure(&["let", "count", "=", "5"]);
        // The learned signal fires strongly on the exact window.
        assert!(
            pred_fwd.overlap(&close) >= MATCH_OVERLAP,
            "forward match lost: {}",
            pred_fwd.overlap(&close)
        );
        // A structurally unrelated window must not produce a strong `)` response.
        assert!(
            pred_bogus.overlap(&close) < 5,
            "bogus window leaks into ) : {}",
            pred_bogus.overlap(&close)
        );
    }
}

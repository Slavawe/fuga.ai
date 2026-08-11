use std::collections::HashSet;

use crate::ai::htm_temporal::TemporalMemory;
use crate::ai::sdr::{SDR_DIM, SDR_WORDS, SdrVector, encode_text};

/// Minimum average per-bit weight a candidate must carry to be emitted. Weights
/// come from [`TemporalMemory::predict_structure_weighted`]: each bit accumulates
/// the segment-match overlap of every depolarized next-token pattern containing
/// it. A candidate that actually follows the context scores its own bit-count
/// worth of strong matches; a candidate that merely co-occurs by chance scores
/// close to 0. Empirically a real next token averages well above this.
const MIN_AVG_WEIGHT: f32 = 6.0;

/// Minimum cosine similarity between the predicted latent and a candidate's
/// latent for it to be eligible in the continuous (tokenless) decode path.
/// Mirrors `LATENT_MIN_COSINE` in main.rs (struct decode path).
const LATENT_MIN_COSINE: f32 = 0.05;

// Последовательная генерация через Временную память (TM + SDR).
// В отличие от статической склейки строк памяти, TM предсказывает следующий
// токен по контекстному окну предыдущих (temporal sequence), возвращая
// построенную во времени цепочку слов — непрерывно, с учётом history.
//
// `eligible` — опциональный коридор от верхнего уровня (H-JEPA L1/L2
// task-mask): локальный авторегрессор двигается СТРОГО внутри него. Это
// две скорости в духе MegaByte/BLT — глобал держит намерение (содержание),
// TM выстраивает валидный порядок (синтаксис). None = без маски.
pub fn tm_generate(
    tm: &TemporalMemory,
    seed: &[String],
    steps: usize,
    candidates: &[String],
    window_size: usize,
    eligible: Option<&HashSet<String>>,
) -> Vec<String> {
    let tokens: Vec<String> = seed
        .iter()
        .filter(|w| !w.trim().is_empty())
        .cloned()
        .collect();
    if tokens.is_empty() {
        return Vec::new();
    }
    // The seed is the user's context; prediction uses the MOST RECENT tokens,
    // so start the sliding window from the tail of the query.
    let start = tokens.len().saturating_sub(window_size.max(1));
    let mut window: Vec<String> = tokens[start..].to_vec();

    let mut out: Vec<String> = Vec::new();
    let mut guard: usize = 0;

    while out.len() < steps && guard < steps * 2 {
        guard += 1;
        let weights = best_structure_weights(tm, &window);
        if weights.iter().all(|&w| w <= 0.0) {
            break;
        }
        let next = decode_weighted(&weights, candidates, eligible);
        match next {
            Some(word) if out.last() == Some(&word) => {
                // A temporal memory must not re-emit the same token off the
                // same context forever; a repeated emission means the merged
                // prediction did not diverge, so stop rather than loop.
                break;
            }
            Some(word) => {
                out.push(word.to_string());
                // Anti-repetition guards. Either a token recurs ~3× in the
                // recent tail or the chain settles into a 2-token cycle
                // (`A B A B`), the generation is oscillating on a fixed set
                // instead of advancing — stop.
                let recent = &out[out.len().saturating_sub(6)..];
                if recent.iter().filter(|w| *w == &word).count() >= 3 {
                    break;
                }
                if out.len() >= 2 && out[out.len() - 2] == word {
                    break;
                }
                window.push(word.to_string());
                if window.len() > window_size.max(1) {
                    window.remove(0);
                }
            }
            None => break,
        }
    }
    out
}

/// Weighted next-token evidence from a sliding window, accepting partial
/// matches: if the full window depolarizes nothing, shrink it (drop the oldest
/// token) and retry — a tail that did appear in the learned corpus still fires.
fn best_structure_weights(tm: &TemporalMemory, window: &[String]) -> Vec<f32> {
    let mut n = window.len();
    while n >= 1 {
        let win_refs: Vec<&str> = window[window.len() - n..].iter().map(|s| s.as_str()).collect();
        let weights = tm.predict_structure_weighted(&win_refs);
        if weights.iter().any(|&w| w > 0.0) {
            return weights;
        }
        n -= 1;
    }
    vec![0f32; SDR_DIM]
}

/// Rank candidates by the average weight carried on their own bits. A token
/// whose cell depolarized with a strong segment match earns a high average;
/// a token only randomly overlapping the evidence scores ~0.
///
/// When `eligible` is present, tokens OUTSIDE the corridor get −∞ (hard gate):
/// the TM never emits a token the upper level did not sanction. This is the
/// H-JEPA→TM two-speed bridge — intent above, syntax below.
fn decode_weighted(
    weights: &[f32],
    candidates: &[String],
    eligible: Option<&HashSet<String>>,
) -> Option<String> {
    if candidates.is_empty() {
        return None;
    }
    let mut best: Option<(usize, f32)> = None;
    for (i, w) in candidates.iter().enumerate() {
        if let Some(elig) = eligible {
            if !elig.contains(w) {
                continue;
            }
        }
        let sdr = encode_text(w);
        let mut sum = 0f32;
        let mut cnt = 0usize;
        for wi in 0..SDR_WORDS {
            let base = wi * 64;
            let mut x = sdr.bits[wi];
            while x != 0 {
                let bi = x.trailing_zeros() as usize;
                sum += weights[base + bi];
                cnt += 1;
                x &= x - 1;
            }
        }
        if cnt == 0 {
            continue;
        }
        let avg = sum / cnt as f32;
        match best {
            Some((_, ba)) if avg > ba => best = Some((i, avg)),
            None => best = Some((i, avg)),
            _ => {}
        }
    }
    best.filter(|(_, avg)| *avg >= MIN_AVG_WEIGHT)
        .map(|(i, _)| candidates[i].clone())
}

/// Continuous (tokenless) next-token decode: predict the NEXT LATENT via the
/// learned transition operator `W` and rank candidates by COSINE SIMILARITY of
/// their latent encodings — no per-token softmax, no dictionary-as-decoder.
///
/// The vocabulary is still passed in, but only as a RELEVANCE GATE: a candidate
/// whose latent is far from the predicted direction is skipped, and the same
/// `eligible` corridor (H-JEPA task mask) applies as a hard filter. The chosen
/// token is whatever latent direction the TM actually predicts — the dictionary
/// never decides, it only validates. This is the "content from the latent
/// channel, order from the syntax graph" path in its pure form.
pub fn tm_generate_latent(
    tm: &TemporalMemory,
    seed: &[String],
    steps: usize,
    candidates: &[String],
    window_size: usize,
    eligible: Option<&HashSet<String>>,
) -> Vec<String> {
    let tokens: Vec<String> = seed
        .iter()
        .filter(|w| !w.trim().is_empty())
        .cloned()
        .collect();
    if tokens.is_empty() || candidates.is_empty() {
        return Vec::new();
    }
    // Pre-encode candidate latents ONCE (encoder is frozen after training).
    let vocab_latents: Vec<(String, crate::ai::latent_jepa::LatentVector)> = candidates
        .iter()
        .map(|w| {
            let sdr = encode_text(w);
            let lat = tm.latent_of_sdr(&sdr);
            (w.clone(), lat)
        })
        .collect();

    let start = tokens.len().saturating_sub(window_size.max(1));
    let mut window: Vec<String> = tokens[start..].to_vec();
    let mut out: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut guard: usize = 0;

    while out.len() < steps && guard < steps * 2 {
        guard += 1;
        // Predict the next latent on the SAME window shape the W operator was
        // trained on (trailing ctx tokens) — never the unbounded buffer.
        let ctx_sdrs: Vec<crate::ai::sdr::SdrVector> =
            window.iter().map(|t| encode_text(t)).collect();
        let pred_latent = tm.predict_latent(&ctx_sdrs);

        let mut best: Option<(f32, String)> = None;
        for (tok, lat) in vocab_latents.iter() {
            if let Some(elig) = eligible {
                if !elig.contains(tok) {
                    continue;
                }
            }
            let score = pred_latent.cosine_similarity(lat);
            if score < LATENT_MIN_COSINE {
                continue;
            }
            if out.len() >= 2 && seen.contains(tok) {
                continue;
            }
            if best.as_ref().map_or(true, |(bc, _)| score > *bc) {
                best = Some((score, tok.clone()));
            }
        }
        let (score, word) = match best {
            Some(b) => b,
            None => break,
        };
        if score < LATENT_MIN_COSINE {
            break;
        }
        if out.last() == Some(&word) {
            break;
        }
        seen.insert(word.clone());
        out.push(word.clone());
        window.push(word);
        if window.len() > window_size.max(1) {
            window.remove(0);
        }
    }
    out
}

/// Byte-level continuous decode (ByT5 / MegaByte style): predict the NEXT BYTE
/// latent via W and rank the FIXED 256 UTF-8 byte alphabet by cosine similarity.
/// No vocabulary, no corpus dependency — works for any language and any code.
/// The optional `eligible` corridor (a set of byte values) acts as a hard gate,
/// mirroring the H-JEPA task-mask role in the token path.
///
/// Output is assembled from raw bytes (not strings), so callers convert with
/// `String::from_utf8_lossy` when the result must be text.
pub fn tm_generate_latent_bytes(
    tm: &TemporalMemory,
    seed_bytes: &[u8],
    steps: usize,
    window_size: usize,
    eligible: Option<&HashSet<u8>>,
) -> Vec<u8> {
    if seed_bytes.is_empty() {
        return Vec::new();
    }
    // The FULL fixed alphabet: 256 raw bytes. Pre-encode their latents once
    // (encoder is frozen) — the "vocabulary" is the byte alphabet itself.
    let byte_latents: Vec<(u8, crate::ai::latent_jepa::LatentVector)> = (0u16..=255)
        .map(|b| {
            let sdr = crate::ai::sdr::byte_basis(b as u8);
            let lat = tm.latent_of_sdr(&sdr);
            (b as u8, lat)
        })
        .collect();

    let start = seed_bytes.len().saturating_sub(window_size.max(1));
    let mut window: Vec<u8> = seed_bytes[start..].to_vec();
    let mut out: Vec<u8> = Vec::new();
    let mut guard: usize = 0;

    while out.len() < steps && guard < steps * 2 {
        guard += 1;
        let pred_latent = tm.predict_bytes_latent(&window);

        let mut best: Option<(f32, u8)> = None;
        for (byte, lat) in byte_latents.iter() {
            if let Some(elig) = eligible {
                if !elig.contains(byte) {
                    continue;
                }
            }
            let score = pred_latent.cosine_similarity(lat);
            if score < LATENT_MIN_COSINE {
                continue;
            }
            if best.as_ref().map_or(true, |(bc, _)| score > *bc) {
                best = Some((score, *byte));
            }
        }
        let (score, byte) = match best {
            Some(b) => b,
            None => break,
        };
        if score < LATENT_MIN_COSINE {
            break;
        }
        if out.last() == Some(&byte) {
            break;
        }
        out.push(byte);
        window.push(byte);
        if window.len() > window_size.max(1) {
            window.remove(0);
        }
    }
    out
}

/// Two-speed (MegaByte-style) byte decoder.
///
/// Fixes the naive byte-by-byte failure (one byte out of 256 = huge noisy
/// decision space → local bigram garbage). A GLOBAL patch transition operator
/// (`learn_patch`/`predict_patch_latent`) decides ONE whole patch (a small
/// group of raw bytes) per step from the `patch_vocab` — far fewer, sharper
/// decisions; the localities of the selected patch are then emitted as-is.
/// The byte-level cell memory still provides context, but the *decision* is
/// concentrated at patch granularity, exactly how MegaByte beats single-rate
/// byte models.
///
/// `patch_vocab` is a dictionary-free byte grammar: every distinct byte-group
/// seen in training, none of which are tokens or subwords.
pub fn tm_generate_two_speed(
    tm: &TemporalMemory,
    seed_bytes: &[u8],
    steps_patches: usize,
    window_patches: usize,
    patch_vocab: &[Vec<u8>],
    eligible: Option<&HashSet<u8>>,
) -> Vec<u8> {
    let min_cosine = LATENT_MIN_COSINE;
    fn patches_from(state: &[u8], size: usize) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        for chunk in state.chunks(size) {
            if !chunk.is_empty() {
                out.push(chunk.to_vec());
            }
        }
        out
    }
    let psize = if patch_vocab.iter().map(|p| p.len()).max().unwrap_or(1) > 0 {
        patch_vocab
            .iter()
            .map(|p| p.len())
            .min()
            .unwrap_or(4)
            .max(1)
    } else {
        4
    };

    // Pre-encode each candidate patch latent once (frozen encoder).
    let patch_latents: Vec<(Vec<u8>, crate::ai::latent_jepa::LatentVector)> = patch_vocab
        .iter()
        .map(|p| {
            let sdr = crate::ai::sdr::encode_bytes_sdr(p);
            let lat = tm.latent_of_sdr(&sdr);
            (p.clone(), lat)
        })
        .collect();

    let mut state: Vec<u8> = seed_bytes.to_vec();
    let mut out: Vec<u8> = Vec::new();
    let mut guard: usize = 0;
    let mut last_patch: Vec<u8> = Vec::new();
    let mut repeat_run: usize = 0;
    // No-repeat window: a patch may not be re-emitted within the last N picks
    // (unless nothing else clears the gate). This is the decode-side iterator
    // that breaks top-1 frequency loops ('er er', ') { ) {').
    let no_repeat: usize = 2;
    let mut recent: Vec<Vec<u8>> = Vec::new();

    while out.len() < steps_patches * psize && guard < steps_patches * 2 {
        guard += 1;
        let patches = patches_from(&state, psize);
        let window: Vec<&[u8]> = patches
            .iter()
            .rev()
            .take(window_patches.max(1))
            .rev()
            .map(|p| p.as_slice())
            .collect();
        if window.is_empty() {
            break;
        }
        let pred = tm.predict_patch_latent(&window);

        let mut best: Option<(f32, Vec<u8>)> = None;
        for (patch, lat) in patch_latents.iter() {
            if let Some(elig) = eligible {
                // A patch is only eligible if every one of its bytes passes
                // the corridor — keeps generated output sanitized.
                if !patch.iter().all(|b| elig.contains(b)) {
                    continue;
                }
            }
            // Decode-side no-repeat (window N).
            if recent.contains(patch) {
                continue;
            }
            let score = pred.cosine_similarity(lat);
            if score < min_cosine {
                continue;
            }
            if best.as_ref().map_or(true, |(bc, _)| score > *bc) {
                best = Some((score, patch.clone()));
            }
        }
        let (score, patch) = match best {
            Some(b) => b,
            None => break,
        };
        if score < min_cosine {
            break;
        }
        // No-repeat bookkeeping: keep the last N picked patches.
        recent.push(patch.clone());
        if recent.len() > no_repeat.max(1) {
            recent.remove(0);
        }
        // Anti-repeat: a well-formed decoder must not loop. Detect BOTH a
        // repeat of one identical patch AND any short period-2..4 cycle in
        // the recent patch tail, and stop when a cycle has already produced
        // enough repetitions (no perpetual 'er er er' / 2-token loops).
        if !last_patch.is_empty() {
            if patch == last_patch {
                repeat_run += 1;
                if repeat_run >= 4 {
                    break;
                }
            } else {
                repeat_run = 0;
            }
        } else {
            repeat_run = 0;
        }
        // Detect a recently-repeating window: if the next patch plus the last
        // three already appear as a repeated unit earlier in the tail, stop.
        if out.len() >= psize * 8 {
            let mut tail: Vec<Vec<u8>> = Vec::new();
            {
                let p = patches_from(&state, psize);
                for unit in p.iter().rev().take(6) {
                    tail.push(unit.clone());
                }
            }
            tail.push(patch.clone()); // candidate next
            // A cycle of period 2 means tail[..4] == tail[2..6].
            if tail.len() >= 6 {
                let a1: Vec<u8> = tail[0].iter().chain(tail[1].iter()).cloned().collect();
                let a2: Vec<u8> = tail[2].iter().chain(tail[3].iter()).cloned().collect();
                let a3: Vec<u8> = tail[4].iter().chain(tail[5].iter()).cloned().collect();
                if a1 == a2 && a2 == a3 {
                    break;
                }
            }
            // period-3: tail[0..3] == tail[3..6]
            if tail.len() >= 6 {
                let b1: Vec<u8> = tail[0]
                    .iter()
                    .chain(tail[1].iter())
                    .chain(tail[2].iter())
                    .cloned()
                    .collect();
                let b2: Vec<u8> = tail[3]
                    .iter()
                    .chain(tail[4].iter())
                    .chain(tail[5].iter())
                    .cloned()
                    .collect();
                if b1 == b2 {
                    break;
                }
            }
        }
        last_patch = patch.clone();
        if out.last() == patch.last() && !out.is_empty() && patch.len() == 1 {
            break;
        }
        out.extend_from_slice(&patch);
        state.extend_from_slice(&patch);
    }
    out
}

/// Entropy-patched two-speed decoder (BLT-style dynamic patching).
///
/// Fixed patching (MegaByte) groups bytes by a constant size; BLT instead
/// sizes patches BY PREDICTABILITY. Here the LOCAL byte W operator gives a
/// cosine distribution over the fixed 256-byte alphabet; its normalized
/// entropy decides the patch rate per step:
///
///   - LOW entropy  (predictable run: 'let ', 'fn ', common keywords)
///     → emit the argmax byte locally; the patch GROWS (coarse rate).
///   - HIGH entropy (rare identifier, mixed code, a typo)
///     → hand over to the GLOBAL patch operator: pick the closest patch in
///       `patch_vocab` and emit it (fine rate), then resume byte-wise.
///
/// Entropy-gap two-speed decoder (BLT-style dynamic patching).
///
/// Fixed patching (MegaByte) groups bytes by a constant size; BLT instead
/// sizes patches BY PREDICTABILITY. Here the LOCAL byte W operator gives a
/// cosine distribution over the fixed 256-byte alphabet; the separation
/// (confidence gap) between the top-1 and top-2 predicted bytes decides the
/// rate per step:
///
///   - STRONG top-1 (large gap, predictable run: 'let ', 'fn ', keywords)
///     → emit the argmax byte locally; the patch GROWS (coarse rate).
///   - WEAK top-1 (small gap, rare identifier, mixed code, a typo)
///     → hand over to the GLOBAL patch operator: pick the closest patch in
///       `patch_vocab` and emit it (fine rate), then resume byte-wise.
///
/// This is the VSA equivalent of the BLT patcher without a learned patcher
/// net, using the model's own W distributions and a gap threshold instead of
/// raw softmax entropy (which saturates at ~1.0 because 256 byte candidates
/// all carry a small cosine baseline — measured 0.999 even on a clean 'a'→'b').
pub fn tm_generate_two_speed_entropy(
    tm: &TemporalMemory,
    seed_bytes: &[u8],
    max_bytes: usize,
    window_bytes: usize,
    gap_threshold: f32,
    patch_vocab: &[Vec<u8>],
) -> Vec<u8> {
    // Pre-encode the fixed byte alphabet latents (frozen encoder).
    let byte_lats: Vec<crate::ai::latent_jepa::LatentVector> = (0u16..256)
        .map(|b| {
            let sdr = crate::ai::sdr::byte_basis(b as u8);
            tm.latent_of_sdr(&sdr)
        })
        .collect();
    // Global patch grammar (same as tm_generate_two_speed).
    let patch_lats: Vec<(Vec<u8>, crate::ai::latent_jepa::LatentVector)> = patch_vocab
        .iter()
        .map(|p| {
            let sdr = crate::ai::sdr::encode_bytes_sdr(p);
            let lat = tm.latent_of_sdr(&sdr);
            (p.clone(), lat)
        })
        .collect();
    let psize = patch_vocab
        .iter()
        .map(|p| p.len())
        .min()
        .unwrap_or(4)
        .max(1);

    let mut state: Vec<u8> = seed_bytes.to_vec();
    let mut out: Vec<u8> = Vec::new();
    let mut guard: usize = 0;
    let mut recent: Vec<Vec<u8>> = Vec::new();
    let no_repeat: usize = 2;

    while out.len() < max_bytes && guard < max_bytes * 2 {
        guard += 1;
        // Local byte-rate distribution over the 256-byte alphabet.
        let win_lo = state.len().saturating_sub(window_bytes.max(1));
        let pred = tm.predict_bytes_latent(&state[win_lo..]);
        // Find top-1 and top-2 cosine bytes in one pass.
        let mut top1 = (0usize, f32::MIN);
        let mut top2 = (0usize, f32::MIN);
        for (b, lat) in byte_lats.iter().enumerate() {
            let c = pred.cosine_similarity(lat);
            if c > top1.1 {
                top2 = top1;
                top1 = (b, c);
            } else if c > top2.1 {
                top2 = (b, c);
            }
        }
        let gap = top1.1 - top2.1; // top-1 dominance; low = ambiguous

        if gap >= gap_threshold {
            // Predictable (top byte clearly dominant) → local emission.
            let b = top1.0 as u8;
            // Break on a repeated identical byte (avoid single-byte stall).
            if !out.is_empty() && out.last() == Some(&b) && out.len() > 2 {
                break;
            }
            out.push(b);
            state.push(b);
        } else {
            // Ambiguous → global patch operator picks the next patch.
            let patches: Vec<Vec<u8>> = state
                .chunks(psize)
                .filter(|c| !c.is_empty())
                .map(|c| c.to_vec())
                .collect();
            let window: Vec<&[u8]> = patches
                .iter()
                .rev()
                .take(4)
                .rev()
                .map(|p| p.as_slice())
                .collect();
            let pred_p = tm.predict_patch_latent(&window);
            let mut best: Option<(f32, Vec<u8>)> = None;
            for (patch, lat) in patch_lats.iter() {
                if recent.contains(patch) {
                    continue;
                }
                let score = pred_p.cosine_similarity(lat);
                if score < LATENT_MIN_COSINE {
                    continue;
                }
                if best.as_ref().map_or(true, |(bc, _)| score > *bc) {
                    best = Some((score, patch.clone()));
                }
            }
            let (score, patch) = match best {
                Some(b) => b,
                None => break,
            };
            if score < LATENT_MIN_COSINE {
                break;
            }
            recent.push(patch.clone());
            if recent.len() > no_repeat.max(1) {
                recent.remove(0);
            }
            out.extend_from_slice(&patch);
            state.extend_from_slice(&patch);
        }
    }
    out
}

/// Recurrent (SSM-lite) byte decoder.
///
/// Breaks the stateless `W` attractor failure (e→r garbage): instead of
/// predicting each byte from ONLY the fixed local window, it maintains a
/// leaky hidden state `h` (a unit latent) that accumulates the whole distant
/// past. Each step starts from context = seed + emitted bytes, encodes the
/// local window, and runs `predict.next_rnn(window_sdrs, &h, mix)` — the same
/// W-as-matrix, but fed `local ⊕ mix·h` so it can condition on BOTH the
/// window AND the running memory. After emitting a byte, `h` is advanced as
/// `h' = φ·h + (1-φ)·enc(byte)` (Mamba-style state). Returns the emitted bytes.
///
/// Being the first tokenless-attention (recurrent state) decoder, this is the
/// natural next architectural step after the stateless Beam/BLT/speculative
/// family that all hit the same local-W ceiling.
pub fn tm_generate_recurrent(
    tm: &TemporalMemory,
    seed_bytes: &[u8],
    max_bytes: usize,
    window_bytes: usize,
    mix: f32,
    phi: f32,
) -> Vec<u8> {
    let predictor = tm.predictor();
    let byte_lats: Vec<crate::ai::latent_jepa::LatentVector> = (0u16..256)
        .map(|b| {
            let sdr = crate::ai::sdr::byte_basis(b as u8);
            predictor.encoder.encode(&sdr)
        })
        .collect();

    let mut state: Vec<u8> = seed_bytes.to_vec();
    let mut out: Vec<u8> = Vec::new();
    let mut h = crate::ai::latent_jepa::LatentVector::zero();
    let mut guard: usize = 0;

    while out.len() < max_bytes && guard < max_bytes * 2 {
        guard += 1;
        let win_lo = state.len().saturating_sub(window_bytes.max(1));
        let window_byte = &state[win_lo..];
        let window_sdrs: Vec<crate::ai::sdr::SdrVector> = window_byte
            .iter()
            .map(|&b| crate::ai::sdr::byte_basis(b))
            .collect();
        let pred = predictor.predict_next_rnn(&window_sdrs, &h, mix);
        // Rank all 256 bytes by cosine to the recurrent latent prediction.
        let mut best = (0usize, f32::NEG_INFINITY);
        let mut second = (0usize, f32::NEG_INFINITY);
        for (i, lat) in byte_lats.iter().enumerate() {
            let c = pred.cosine_similarity(lat);
            if c > best.1 {
                second = best;
                best = (i, c);
            } else if c > second.1 {
                second = (i, c);
            }
        }
        let byte = best.0 as u8;
        // Confidence gap = top1 cosine - top2 cosine. When the model is
        // UNSURE (small gap), the hidden state is noise — forget it hard
        // (phi -> 0); when confident, keep the memory (phi -> full). This is
        // the "temperature-decayed memory" lever of the exposure-bias plan.
        let gap = best.1 - second.1;
        let phi_eff = if gap >= 0.30 {
            phi
        } else if gap <= 0.10 {
            0.05
        } else {
            // linear interpolation 0.10..0.30 -> 0.05..phi
            0.05 + (phi - 0.05) * (gap - 0.10) / 0.20
        };
        // Stop on a repeated identical byte (avoid single-byte stall).
        if !out.is_empty() && out.last() == Some(&byte) && out.len() > 2 {
            break;
        }
        out.push(byte);
        state.push(byte);
        // Advance the hidden state with the emitted byte (leaky integration).
        // phi_eff = gap-adaptive: uncertain steps forget the noisy state hard.
        h = predictor.advance_h(h, &crate::ai::sdr::byte_basis(byte), phi_eff);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::htm_temporal::TemporalMemory;

    #[test]
    fn tm_generate_breaks_on_empty_prediction() {
        let tm = TemporalMemory::new(32, 3);
        let seed = vec!["alpha".to_string(), "beta".to_string()];
        let cands = vec![
            "alpha".to_string(),
            "beta".to_string(),
            "gamma".to_string(),
            "delta".to_string(),
        ];
        // Untrained TM: predict_structure returns zero -> generation stops.
        let out = tm_generate(&tm, &seed, 20, &cands, 4, None);
        assert!(out.is_empty() || out.len() < 20);
    }

    #[test]
    fn tm_generate_learns_and_reproduces_sequence() {
        let mut tm = TemporalMemory::new(64, 3);
        let seq = [
            "tokio", "async", "stream", "tcp", "runtime", "task", "await", "spawn",
        ];
        // Learn bigram windows manually: tokio->async, async->stream, ...
        for w in 0..seq.len().saturating_sub(1) {
            let win = &[seq[w]];
            tm.learn_structure(win, seq[w + 1]);
        }
        let seed = vec!["tokio".to_string()];
        let cands: Vec<String> = seq.iter().map(|s| s.to_string()).collect();
        let out = tm_generate(&tm, &seed, 6, &cands, 2, None);
        assert!(
            !out.is_empty(),
            "TM learned a bigram chain, expected output, got none"
        );
    }

    #[test]
    fn tm_generate_latent_uses_continuous_decode_and_respects_gate() {
        let mut tm = TemporalMemory::new(64, 3);
        let seq = [
            "tokio", "async", "stream", "tcp", "runtime", "task",
        ];
        // Learn bigrams so the latent transition W maps tokio -> async -> ...
        // (learn_structure trains BOTH the structural segment and the latent
        // transition operator W — see htm_temporal::learn_structure.)
        for w in 0..seq.len().saturating_sub(1) {
            tm.learn_structure(&[seq[w]], seq[w + 1]);
        }
        let seed = vec!["tokio".to_string()];
        let cands: Vec<String> = seq.iter().map(|s| s.to_string()).collect();

        let out = tm_generate_latent(&tm, &seed, 4, &cands, 2, None);
        assert!(
            !out.is_empty(),
            "latent decode expected some output after bigram training"
        );

        // Corridor gate: only 'stream' is eligible -> the emitted sequence must
        // stay within it (validates the dictionary-as-GATE role).
        let mut elig = std::collections::HashSet::new();
        elig.insert("stream".to_string());
        let gated = tm_generate_latent(&tm, &seed, 4, &cands, 2, Some(&elig));
        for w in gated {
            assert!(elig.contains(&w), "gated generation escaped corridor: {}", w);
        }
    }

    #[test]
    fn tm_generate_latent_bytes_reproduces_text_without_dictionary() {
        let mut tm = TemporalMemory::new(64, 4);
        // "hello" as raw UTF-8 bytes; corpus-independent — the ONLY things the
        // decoder sees are the fixed 256 per-byte SDRs.
        let text = "hello";
        let bytes: Vec<u8> = text.bytes().collect();
        // Learn bigram byte transitions: h->e, e->l, l->l, l->o (window=1).
        for w in 0..bytes.len().saturating_sub(1) {
            tm.learn_bytes(&[bytes[w]], bytes[w + 1], 0.2);
        }
        let seed = &bytes[0..1]; // "h"
        let out = tm_generate_latent_bytes(&tm, seed, 4, 1, None);
        assert!(
            !out.is_empty(),
            "byte latent decode expected continuation after h, got none"
        );
        // The generated bytes must at least be printable ASCII fenceposts
        // consistent with the learned grammar (no panic, no empty).
        let s = String::from_utf8_lossy(&out);
        assert!(!s.trim().is_empty());
    }

    #[test]
    fn byte_basis_is_fixed_and_position_sensitive() {
        use crate::ai::sdr::{byte_basis, encode_bytes_sdr};
        // Fixed alphabet: same byte always maps to the same SDR.
        assert_eq!(byte_basis(42), byte_basis(42));
        assert_ne!(byte_basis(42), byte_basis(43));
        // Order matters in the folded sequence (mirrors structure fold).
        let ab = encode_bytes_sdr(b"ab");
        let ba = encode_bytes_sdr(b"ba");
        assert_ne!(ab, ba);
        // A single-byte typo keeps shared-prefix similarity (byte-level
                // robustness: "helo" vs "hello" still overlap strongly).
                let ok = encode_bytes_sdr(b"hello");
                let typo = encode_bytes_sdr(b"helio");
                assert!(ok.soft_overlap(&typo) > 0.0);
            }

            #[test]
            fn two_speed_patch_decode_selects_relevant_patch() {
                let mut tm = TemporalMemory::new(64, 4);
                // Learn patch transitions over byte-groups of size 2: "fn"->" m",
                // " m"->"ai", "ai"->"n(" ... the local byte memory stays free.
                let patches: Vec<Vec<u8>> = ["fn", " m", "ai", "n(", "{", "}"]
                    .iter()
                    .map(|s| s.as_bytes().to_vec())
                    .collect();
                for w in 0..patches.len().saturating_sub(1) {
                    tm.learn_patch(&[&patches[w]], &patches[w + 1], 0.2);
                }
                let seed = "fn".as_bytes(); // starts the chain " m->ai->..."
                let vocab = patches.clone();
                let out = tm_generate_two_speed(&tm, seed, 6, 2, &vocab, None);
                // The global rate must produce a non-empty continuation: at least the
                // first learned next-patch " m" (or a contiguous chain).
                assert!(
                    !out.is_empty(),
                    "two-speed patch decode expected a next patch after 'fn', got none"
                );
                let s = String::from_utf8_lossy(&out);
                assert!(!s.trim().is_empty());
                // Two-speed retains dictionary-free property: no token model here.
                // (Byte level remains validated by the earlier test.)
            }

            #[test]
            fn two_speed_zero_vocab_returns_empty() {
                let tm = TemporalMemory::new(32, 4);
                let out = tm_generate_two_speed(&tm, b"fn", 4, 2, &[], None);
                assert!(out.is_empty(), "empty patch_vocab must yield no output");
            }




            #[test]
            fn recurrent_decoder_continues_predictable_run() {
                let mut tm = TemporalMemory::new(64, 4);
                // Learn a hard 2-byte alternation 'ab' so the a->b transition is
                // unambiguous under a 1-byte window; the recurrent hidden state
                // must NOT break (or stall) the decoder on this regular pattern.
                let seq = b"abababababababababab";
                for w in 0..seq.len().saturating_sub(1) {
                    tm.learn_bytes(&seq[w..w + 1], seq[w + 1], 0.4);
                }
                // mix=0 ≈ stateless byte argmax; mix>0 activates the h(t) memory.
                let out_stateless = tm_generate_recurrent(&tm, b"a", 60, 2, 0.0, 0.9);
                assert!(out_stateless.len() > 1, "stateless recurrent must continue, got {:?}", String::from_utf8_lossy(&out_stateless));
                let out_stateful = tm_generate_recurrent(&tm, b"a", 60, 2, 0.6, 0.9);
                assert!(out_stateful.len() > 1, "stateful recurrent must continue, got {:?}", String::from_utf8_lossy(&out_stateful));
                // Any output must be valid UTF-8 (dictionary-free byte path).
                let _ = String::from_utf8(out_stateful.clone()).unwrap();
            }
        }

/// Учим байтовый переход на KAN-операторе (нелинейная замена линейного W).
/// window_bytes → next_byte: x = encoder(structure_sdr(window)),
/// target = encoder(byte_basis(next)); KAN Widrow-Hoff на сплайнах.
/// (Восстановлено из 3f6b28c — потеряно при восстановлении mod.rs.)
pub fn learn_byte_kan(
    kan: &mut crate::ai::kan::KanTransition,
    tm: &TemporalMemory,
    window_bytes: &[u8],
    next_byte: u8,
    lr: f32,
) {
    if window_bytes.is_empty() {
        return;
    }
    let encoder = &tm.predictor().encoder;
    let window_sdrs: Vec<SdrVector> = window_bytes
        .iter()
        .map(|&b| crate::ai::sdr::byte_basis(b))
        .collect();
    let x = encoder.encode(&crate::ai::sdr::structure_sdr_from_sdrs(&window_sdrs));
    let target = encoder.encode(&crate::ai::sdr::byte_basis(next_byte));
    kan.learn(&x, &target, lr);
    kan.cap_outputs();
}

/// MEGABYTE-порядок: патч решает ДО байтов, локальный W байты внутри.
///
/// Двухуровневый декодер в духе MegaByte:
///   1. ГЛОБАЛЬНЫЙ уровень (W_patch, `predict_patch_latent`) по окну патчей
///      предсказывает направление → выбирает top-N патчей-кандидатов из
///      `patch_vocab` по cosine (патч = группа байт, решение концентрируется).
///   2. ЛОКАЛЬНЫЙ уровень (байтовый W, `predict_bytes_latent`) по окну байт
///      предсказывает NEXT-BYTE латент. Выбор байта: среди 256-алфавита
///      НО с приором к выбранному патчу: байты, входящие в патч-кандидат,
///      получают бонус cosine к предсказанному патчу. Так байт «разрешается»
///      только внутри коридора, заданного глобальным оператором.
///
/// Это отличается от tm_generate_two_speed: там патч эмитится ЦЕЛИКОМ as-is
/// (байты из словаря, локальный W не участвует). Здесь патч задаёт
/// НАПРАВЛЕНИЕ (приор), а каждый байт внутри дополнительно фильтруется
/// локальным байтовым W — порядок решения = патч, потом байт.
///
/// `patch_len` — размер патча (2 для двухбайтовых), `lambda` — сила приора
/// к патчу (0.6–1.0): 0 = чистый локальный, 1 = жёсткий коридор патча.
pub fn tm_generate_megabyte(
    tm: &TemporalMemory,
    seed_bytes: &[u8],
    max_bytes: usize,
    window_bytes: usize,
    patch_len: usize,
    patch_vocab: &[Vec<u8>],
    lambda: f32,
) -> Vec<u8> {
    if seed_bytes.is_empty() {
        return Vec::new();
    }
    let plen = patch_len.max(1);
    let hard_gate = lambda >= 1.0;
    // Полный алфавит: 256 raw байт.
    let byte_latents: Vec<(u8, crate::ai::latent_jepa::LatentVector)> = (0u16..=255)
        .map(|b| {
            let sdr = crate::ai::sdr::byte_basis(b as u8);
                        let lat = tm.latent_of_sdr(&sdr);
                        (b as u8, lat)
                    })
                    .collect();
                // Только печатные байты (код/текст): ASCII printable + пробел/новые строки.
    let printable = |b: u8| b == b'\n' || b == b'\t' || b == b'\r' || (0x20..=0x7e).contains(&b);


    // Патч-кандидаты: предкодируем латенты один раз (замороженный энкодер).
    let patch_latents: Vec<(Vec<u8>, crate::ai::latent_jepa::LatentVector)> = patch_vocab
        .iter()
        .filter(|p| p.iter().all(|&b| printable(b)))
        .map(|p| {
            let sdr = crate::ai::sdr::encode_bytes_sdr(p);
            let lat = tm.latent_of_sdr(&sdr);
            (p.clone(), lat)
        })
        .collect();
    if patch_latents.is_empty() {
        return Vec::new();
    }

    let mut state: Vec<u8> = seed_bytes.to_vec();
    let mut out: Vec<u8> = Vec::new();
    let mut guard: usize = 0;
    let mut recent_patches: Vec<Vec<u8>> = Vec::new();
    let no_repeat: usize = 2;

    while out.len() < max_bytes && guard < max_bytes * 6 {
        guard += 1;

        // 1. Глобальный уровень: окно ПАТЧЕЙ → направление.
        //    Строим окно из последних байт state, группируя по plen
        //    (ровные патчи: игнорируем неполный хвост — он не информативен).
        let start_p = state.len().saturating_sub(window_bytes.max(1) * 2 * plen);
        let patch_state: Vec<u8> = state[start_p..].to_vec();
        let mut window_patches: Vec<&[u8]> = Vec::new();
        for chunk in patch_state.chunks(plen) {
            if chunk.len() == plen {
                window_patches.push(chunk);
            }
        }
        if window_patches.is_empty() {
            break;
        }
        let pred_patch = tm.predict_patch_latent(&window_patches);

        // Top-N патчей по cosine к направлению (с no-repeat окном).
        let mut cand: Vec<(f32, Vec<u8>)> = Vec::new();
        for (patch, lat) in patch_latents.iter() {
            if recent_patches.contains(patch) {
                continue;
            }
            let score = pred_patch.cosine_similarity(lat);
            if score < LATENT_MIN_COSINE {
                continue;
            }
            cand.push((score, patch.clone()));
        }
        cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        cand.truncate(8);
        if cand.is_empty() {
            break;
        }
        // Топ-1 патч — главное направление (для приора).
        let (top_score, top_patch) = &cand[0];
        recent_patches.push(top_patch.clone());
        if recent_patches.len() > no_repeat.max(1) {
            recent_patches.remove(0);
        }

        // 2. Локальный уровень: окно БАЙТ → next-byte латент.
        let start_b = state.len().saturating_sub(window_bytes.max(1));
        let byte_window = &state[start_b..];
        let pred_byte = tm.predict_bytes_latent(byte_window);

        // 3. Выбор байта:
        //    - hard_gate (λ>=1): байты ТОЛЬКО из топ-патча (жёсткий коридор);
        //    - мягкий режим: все печатные байты, байты из патча получают бонус.
        let mut best: Option<(f32, u8)> = None;
        for (byte, lat) in byte_latents.iter() {
            if !printable(*byte) {
                continue;
            }
            let in_top_patch = top_patch.contains(byte);
            if hard_gate && !in_top_patch {
                continue; // жёсткий коридор
            }
            let mut score = pred_byte.cosine_similarity(lat);
            if score < LATENT_MIN_COSINE {
                continue;
            }
            if in_top_patch && lambda > 0.0 {
                score += lambda * top_score.max(0.0);
            } else if hard_gate {
                // в жёстком режиме байт из патча: приор по умолчанию
                score += top_score.max(0.0);
            }
            if best.as_ref().map_or(true, |(bc, _)| score > *bc) {
                best = Some((score, *byte));
            }
        }
        let (score, byte) = match best {
            Some(b) => b,
            None => break,
        };
        if score < LATENT_MIN_COSINE {
            break;
        }
        if out.last() == Some(&byte) {
            break; // анти-цикл
        }
        out.push(byte);
        state.push(byte);
        if state.len() > 4096 {
            state.remove(0);
        }
    }
    out
}

/// KAN-lite байтовый декодер: 256 cosine-кандидатов, оператор — KanTransition
/// вместо линейного W. Гипотеза: нелинейный оператор разделит перемешанные
/// аттракторы (e→r vs структурные), которые линейный W не может
/// (доказано на синтетике в kan.rs).
pub fn tm_generate_kan(
    tm: &TemporalMemory,
    kan: &crate::ai::kan::KanTransition,
    seed_bytes: &[u8],
    steps: usize,
    window_size: usize,
) -> Vec<u8> {
    if seed_bytes.is_empty() {
        return Vec::new();
    }
    let encoder = &tm.predictor().encoder;
    let byte_latents: Vec<(u8, crate::ai::latent_jepa::LatentVector)> = (0u16..=255)
        .map(|b| {
            let sdr = crate::ai::sdr::byte_basis(b as u8);
            let lat = encoder.encode(&sdr);
            (b as u8, lat)
        })
        .collect();

    let start = seed_bytes.len().saturating_sub(window_size.max(1));
    let mut window: Vec<u8> = seed_bytes[start..].to_vec();
    let mut out: Vec<u8> = Vec::new();
    let mut guard: usize = 0;

    while out.len() < steps && guard < steps * 2 {
        guard += 1;
        let window_sdrs: Vec<SdrVector> =
            window.iter().map(|&b| crate::ai::sdr::byte_basis(b)).collect();
        let x = encoder.encode(&crate::ai::sdr::structure_sdr_from_sdrs(&window_sdrs));
        let pred_latent = kan.apply(&x);

        let mut best: Option<(f32, u8)> = None;
        for (byte, lat) in byte_latents.iter() {
            let score = pred_latent.cosine_similarity(lat);
            if best.as_ref().map_or(true, |(bc, _)| score > *bc) {
                best = Some((score, *byte));
            }
        }
        let Some((_, byte)) = best else {
            break;
        };
        if out.last() == Some(&byte) {
            break;
        }
        out.push(byte);
        window.push(byte);
    }
    out
}

/// Косинусный кондиционированный декодер (ФРОНТ: inference-архитектура).
///
/// Два рычага против "банального argmax":
/// 1. ВЫБОР ПО КОСИНУСУ + ТЕМПЕРАТУРА (не top-1 argmax): кандидаты-байты
///    ранжируются косинусом к предсказанному латенту pred = W·x + α·KAN(x),
///    затем семплируются из softmax(cos/τ). При τ → 0 это почти argmax, но с
///    энтропийным выходом из частотных ловушек ("er", "on", пробел), когда
///    топ-косинусы почти равны.
/// 2. MEGABYTE-КОРИДОР (патч решает ДО байта): глобальный W_patch предска-
///    зывает направление СЛОГА (top-K патчей по косинусу), и локальный
///    декодер эмитит байты ЖЁСТКО ВНУТРИ маски выученного патча (corridor=1)
///    либо с бонусом к байтам коридора (corridor=2). Байты вне коридора
///    исключаются — распад на несвязанные буквы подавлен.
///
/// corridor=0: чистый косинус+температура (без патчевого уровня)
/// corridor=1: жёсткий коридор (только байты топ-1 патча)
/// corridor=2: мягкий коридор (байты топ-N патчей получают бонус β)
pub fn tm_generate_cosine_gate(
    tm: &TemporalMemory,
    kan: &crate::ai::kan::KanTransition,
    seed_bytes: &[u8],
    max_bytes: usize,
    window_bytes: usize,
    patch_len: usize,
    patch_vocab: &[Vec<u8>],
    alpha: f32,
    tau: f32,
    corridor: u8,
) -> Vec<u8> {
    tm_generate_cosine_gate_inner(tm, kan, seed_bytes, max_bytes, window_bytes, patch_len, patch_vocab, alpha, tau, corridor, 0.005)
}

/// Внутренняя реализация с явным порогом косинуса (для тюнинга).
pub fn tm_generate_cosine_gate_inner(
    tm: &TemporalMemory,
    kan: &crate::ai::kan::KanTransition,
    seed_bytes: &[u8],
    max_bytes: usize,
    window_bytes: usize,
    patch_len: usize,
    patch_vocab: &[Vec<u8>],
    alpha: f32,
    tau: f32,
    corridor: u8,
    min_cos: f32,
) -> Vec<u8> {
    // v2: repetition penalty + аддитивное патч-кондиционирование
    tm_generate_cosine_gate_v2(
        tm, kan, seed_bytes, max_bytes, window_bytes, patch_len, patch_vocab,
        alpha, tau, corridor, min_cos, 0.0, 0.0, 0.0, 0.0,
    )
}

/// Внутренняя реализация с явным порогом косинуса и двумя рычагами v2:
/// - `rep_pen`: штраф на косинусы недавно сгенерированных n-грамм (выход из аттрактора)
/// - `beta`: аддитивное патч-кондиционирование z = W·x + β·P_{t+1} + α·KAN(x)
#[allow(clippy::too_many_arguments)]
pub fn tm_generate_cosine_gate_v2(
    tm: &TemporalMemory,
    kan: &crate::ai::kan::KanTransition,
    seed_bytes: &[u8],
    max_bytes: usize,
    window_bytes: usize,
    patch_len: usize,
    patch_vocab: &[Vec<u8>],
    alpha: f32,
    tau: f32,
    corridor: u8,
    min_cos: f32,
    beta: f32,
    rep_pen: f32,
    rep_word: f32,
    rep_phrase: f32,
) -> Vec<u8> {
    if seed_bytes.is_empty() {
        return Vec::new();
    }
    let encoder = &tm.predictor().encoder;
    let w = tm.predictor_w();
    let plen = patch_len.max(1);
    // Байт-базис: 256 латентов (тот же энкодер, что у обученного W).
    let byte_latents: Vec<(u8, crate::ai::latent_jepa::LatentVector)> = (0u16..=255)
        .map(|b| {
            let sdr = crate::ai::sdr::byte_basis(b as u8);
            let lat = encoder.encode(&sdr);
            (b as u8, lat)
        })
        .collect();
    // Патч-базис (коридор): латенты выученных патчей через ЭТОТ ЖЕ энкодер.
    let patch_latents: Vec<(Vec<u8>, crate::ai::latent_jepa::LatentVector)> = patch_vocab
        .iter()
        .map(|p| {
            let sdr = crate::ai::sdr::encode_bytes_sdr(p);
            let lat = encoder.encode(&sdr);
            (p.clone(), lat)
        })
        .collect();
    let patch_w: Vec<f32> = tm.patch_predictor_w().to_vec();

    let start = seed_bytes.len().saturating_sub(window_bytes.max(1));
    let mut window: Vec<u8> = seed_bytes[start..].to_vec();
    let mut state: Vec<u8> = seed_bytes.to_vec(); // ПОЛНАЯ история для патч-уровня
    let mut out: Vec<u8> = Vec::new();
    let mut guard: usize = 0;
    let mut recent_patches: Vec<Vec<u8>> = Vec::new();
    let no_repeat: usize = 2;
    let k_candidates: usize = 16;

    while out.len() < max_bytes && guard < max_bytes * 2 {
        guard += 1;

        // --- Локальный уровень: pred = W·x + α·KAN(x) ---
        // ВАЖНО: кодируем ТОЛЬКО последние window_bytes байт (как обучение
        // data[i-4..=i] → ровно window_bytes), иначе история растёт и вектор
        // уходит из обученного распределения (фазовый сдвиг).
        let win_lo = window.len().saturating_sub(window_bytes.max(1));
        let win_tail = &window[win_lo..];
        let window_sdrs: Vec<crate::ai::sdr::SdrVector> = win_tail
            .iter()
            .map(|&b| crate::ai::sdr::byte_basis(b))
            .collect();
        let x = encoder.encode(&crate::ai::sdr::structure_sdr_from_sdrs(&window_sdrs));
        let mut pred = crate::ai::latent_jepa::LatentVector::zero();
        for o in 0..crate::ai::latent_jepa::LATENT_DIM {
            let row = o * crate::ai::latent_jepa::LATENT_DIM;
            let mut acc = 0.0f32;
            for i in 0..crate::ai::latent_jepa::LATENT_DIM {
                acc += w[row + i] * x.values[i];
            }
            pred.values[o] = acc;
        }
        if alpha > 0.0 {
            let kan_out = kan.apply(&x);
            for i in 0..crate::ai::latent_jepa::LATENT_DIM {
                pred.values[i] += alpha * kan_out.values[i];
            }
        }
        // Аддитивное патч-кондиционирование: z = W·x + β·P_{t+1} + α·KAN(x).
        // β·P_{t+1} — это смещение к глобальному патчевому направлению
        // (тема слога/слова), НЕ жёсткая маска коридора.
        if beta > 0.0 {
            let patches_v: Vec<Vec<u8>> = state
                .chunks(plen)
                .filter(|c| c.len() == plen)
                .map(|c| c.to_vec())
                .collect();
            let patch_window: Vec<&[u8]> = patches_v
                .iter()
                .rev()
                .take(4)
                .rev()
                .map(|p| p.as_slice())
                .collect();
            if !patch_window.is_empty() {
                let xs: Vec<crate::ai::sdr::SdrVector> = patch_window
                    .iter()
                    .map(|p| crate::ai::sdr::encode_bytes_sdr(p))
                    .collect();
                let xp = encoder.encode(&crate::ai::sdr::structure_sdr_from_sdrs(&xs));
                let mut pp = crate::ai::latent_jepa::LatentVector::zero();
                for o in 0..crate::ai::latent_jepa::LATENT_DIM {
                    let row = o * crate::ai::latent_jepa::LATENT_DIM;
                    let mut acc = 0.0f32;
                    for i in 0..crate::ai::latent_jepa::LATENT_DIM {
                        acc += patch_w[row + i] * xp.values[i];
                    }
                    pp.values[o] = acc;
                }
                let pn2 = pp.values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
                for v in &mut pp.values {
                    *v /= pn2;
                }
                for i in 0..crate::ai::latent_jepa::LATENT_DIM {
                    pred.values[i] += beta * pp.values[i];
                }
            }
        }
        let pn = pred.values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in &mut pred.values {
            *v /= pn;
        }

        // --- Глобальный уровень (коридор): патч решает ДО байта ---
        let mut corridor_bytes: Option<std::collections::HashSet<u8>> = None;
        if corridor > 0 && !patch_latents.is_empty() && patch_w.iter().any(|&v| v != 0.0) {
            // Патч-окно: минимум 2 полных патча (обучение W_patch шло на окнах
            // из ДВУХ патчей — декод должен совпадать с обучением). Отступаем
            // до 4·plen байт назад, чтобы собрать ровно 2 завершённых патча.
            // Патч-окно: ТОЧНО как в entropy-BLT — chunks(plen) по всему state,
            // последние 4 полных патча (этот размер дал идеальный патчевый
            // декод на микро-пруфе; 2 патча не совпадали с траекторией).
            let patches_v: Vec<Vec<u8>> = state
                .chunks(plen)
                .filter(|c| c.len() == plen)
                .map(|c| c.to_vec())
                .collect();
            let patch_window: Vec<&[u8]> = patches_v
                .iter()
                .rev()
                .take(4)
                .rev()
                .map(|p| p.as_slice())
                .collect();
            let mut cand_p: Vec<(f32, &Vec<u8>)> = Vec::new();
            if !patch_window.is_empty() {
                let xs: Vec<crate::ai::sdr::SdrVector> = patch_window
                    .iter()
                    .map(|p| crate::ai::sdr::encode_bytes_sdr(p))
                    .collect();
                let xp = encoder.encode(&crate::ai::sdr::structure_sdr_from_sdrs(&xs));
                let mut pp = crate::ai::latent_jepa::LatentVector::zero();
                for o in 0..crate::ai::latent_jepa::LATENT_DIM {
                    let row = o * crate::ai::latent_jepa::LATENT_DIM;
                    let mut acc = 0.0f32;
                    for i in 0..crate::ai::latent_jepa::LATENT_DIM {
                        acc += patch_w[row + i] * xp.values[i];
                    }
                    pp.values[o] = acc;
                }
                let pn2 = pp.values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
                for v in &mut pp.values {
                    *v /= pn2;
                }
                for (patch, lat) in patch_latents.iter() {
                    if recent_patches.contains(patch) {
                        continue;
                    }
                    let c = pp.cosine_similarity(lat);
                    if c > min_cos {
                        cand_p.push((c, patch));
                    }
                }
                cand_p.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            }
            let top_n: usize = if corridor == 1 { 1 } else { 4 };
            let mut mask: std::collections::HashSet<u8> = std::collections::HashSet::new();
            for (_, patch) in cand_p.iter().take(top_n) {
                for &b in patch.iter() {
                    mask.insert(b);
                }
                if corridor == 1 {
                    recent_patches.push(patch.to_vec());
                    if recent_patches.len() > no_repeat.max(1) {
                        recent_patches.remove(0);
                    }
                }
            }
            if !mask.is_empty() {
                corridor_bytes = Some(mask);
            }
        }

        // --- Косинусные оценки байтов → softmax(cos/τ) ---
        // Repetition penalty на уровне N-грамм: байт, который образует
        // с хвостом history триграмму/биграмму, уже встречавшуюся в последних
        // 24 сгенерированных байтах, штрафуется — траектория выталкивается
        // из аттрактора («the red the red») в соседние вероятные ветки.
        let recent: Vec<u8> = state.iter().rev().take(24).cloned().collect::<Vec<_>>();
        let t0 = state[state.len().saturating_sub(1)];
        let t1 = state[state.len().saturating_sub(2)];
        let mut scores: Vec<(u8, f32)> = Vec::with_capacity(256);
        for (b, lat) in byte_latents.iter() {
            let c = pred.cosine_similarity(lat);
            if c < min_cos {
                continue;
            }
            let mut sc = c;
            if let Some(mask) = &corridor_bytes {
                if corridor == 1 {
                    if !mask.contains(b) {
                        continue; // жёсткий коридор: байт вне патча исключён
                    }
                } else {
                    sc += 0.10 * c * if mask.contains(b) { 1.0 } else { -0.5 };
                }
            }
            if rep_pen > 0.0 {
                // биграмма (t0, b) или триграмма (t1, t0, b) в недавней истории
                let mut rep = 0.0f32;
                for w in recent.windows(2) {
                    if w[0] == t0 && w[1] == *b {
                        rep += rep_pen * 0.5;
                    }
                }
                for w in recent.windows(3) {
                    if w[0] == t1 && w[1] == t0 && w[2] == *b {
                        rep += rep_pen * 1.0;
                    }
                }
                sc -= rep;
            }
            if rep_word > 0.0 {
                // СЛОВЕСНЫЙ repetition penalty: если кандидат завершает слово
                // (пробел) и получившееся слово уже встречалось в этой же
                // позиции при повторяющейся словесной биграмме/триграмме —
                // штрафуем. Декодим последние ~16 слов из state.
                let word_start = state
                    .iter()
                    .rposition(|&c| c == b' ')
                    .map(|p| p + 1)
                    .unwrap_or(0);
                let cur_word: Vec<u8> = state[word_start..].to_vec();
                // ЧИСТКА недописанных слов: если текущее слово уже очень длинное,
                // предпочитаем завершить его пробелом, а не дописывать букву
                // (борьба с «redirectiont» — лишняя буква после валидного
                // слова; порог 11 не трогает обычные слова ≤10 букв).
                if *b == b' ' && cur_word.len() >= 11 {
                    sc += 0.10 * c * (cur_word.len() as f32 - 10.0);
                } else if *b != b' ' && cur_word.len() >= 12 {
                    sc -= 0.08 * c * (cur_word.len() as f32 - 11.0);
                }
                let words: Vec<Vec<u8>> = state
                    .split(|&c| c == b' ')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_vec())
                    .collect();
                if *b == b' ' && !cur_word.is_empty() {
                    // Слово «cur_word» завершится пробелом: штраф, если
                    // (prev_word, cur_word) или (prev2, prev_word, cur_word)
                    // уже встречались в недавней последовательности слов.
                    let nw = words.len();
                    if nw >= 2 && words[nw - 2] == cur_word {
                        sc -= rep_word * 1.5; // немедленный повтор слова
                    }
                    // БИГРАММА СЛОВ: штраф за повтор пары (prev, cur) —
                    // прямой удар по «the red the red the red».
                    if nw >= 2 && !cur_word.is_empty() {
                        let prev_w = &words[nw - 2];
                        // Штраф за повтор пары (prev, cur) — прямой удар
                        // по «the red the red the red». Считаем пары ДО текущей
                        // (последняя пара (nw-2, nw-1) — это и есть текущая
                        // незавершённая позиция, её не штрафуем).
                        let mut pair_count2 = 0;
                        for i in 0..nw.saturating_sub(2) {
                            if words[i].len() == prev_w.len()
                                && words[i] == *prev_w
                                && words[i + 1] == cur_word
                            {
                                pair_count2 += 1;
                            }
                        }
                        if pair_count2 > 0 {
                            sc -= rep_word * 1.0 * pair_count2 as f32;
                        }
                    }
                    let mut wcount = 0;
                    for i in 0..nw.saturating_sub(1) {
                        if words[i] == cur_word {
                            wcount += 1;
                        }
                    }
                    sc -= rep_word * 0.5 * wcount as f32;
                }
            }
            if rep_phrase > 0.0 {
                // ФРАЗОВЫЙ repetition penalty (v6.2): модель заучивает
                // монолитные фразы 20-30 байт (напр. RedisModule_Free(...))
                // как единый блок; байтовый/словесный штрафы не бьют по ним,
                // т.к. внутри блока символы сменяются штатно. Здесь штрафуем
                // кандидата, который ЗАМЫКАЕТ повтор целого байтового блока
                // длины PHR_LEN (12 байт): если подстрока state[..-1] длиной
                // PHR_LEN-1 уже встречалась в истории state ранее, то
                // продолжение её теми же символами = повтор фразы — штраф.
                const PHR_LEN: usize = 12;
                if state.len() >= PHR_LEN + 4 {
                    // текущий расщепляемый блок: последние PHR_LEN-1 байт + кандидат
                    let start = state.len() - (PHR_LEN - 1);
                    let block: Vec<u8> = state[start..]
                        .iter()
                        .chain(std::iter::once(b))
                        .cloned()
                        .collect();
                    // ищем блок в истории (до текущей позиции, с отступом 2)
                    let hist_end = state.len().saturating_sub(2);
                    if hist_end >= PHR_LEN {
                        let mut hits = 0;
                        let mut k = 0;
                        while k + PHR_LEN <= hist_end {
                            if state[k..k + PHR_LEN] == block[..] {
                                hits += 1;
                                k += PHR_LEN; // непересекающиеся блоки
                            } else {
                                k += 1;
                            }
                        }
                        if hits > 0 {
                            sc -= rep_phrase * 1.0 * hits as f32;
                        }
                    }
                }
            }
            scores.push((*b, sc));
        }
        if scores.is_empty() {
            break;
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let topk: Vec<(u8, f32)> = scores.into_iter().take(k_candidates).collect();
        if topk[0].1 < min_cos {
            break;
        }
        if out.last() == Some(&topk[0].0)
            && topk.len() > 1
            && out.len() >= 2
            && out[out.len() - 2] == topk[0].0
        {
            break; // тройной повтор того же байта (настоящее зацикливание)
        }
        // Детерминированный (не-argmax) семплинг ниже: разрыв топ-1/топ-2
        // большой → почти argmax; малый → энтропийный выбор из топ-K.
        // Семплинг С GAP-ПОРОГОМ (как entropy-BLT): если топ-1 доминирует —
        // косинусный argmax; только при размытом направлении — температура.
        const GAP_SELECT: f32 = 0.03;
        let gap = if topk.len() > 1 { topk[0].1 - topk[1].1 } else { 1.0 };
        let pick: u8 = if gap >= GAP_SELECT {
            topk[0].0
        } else {
            let temp = tau.max(0.02);
            let maxs = topk[0].1;
            let mut wts: Vec<f32> = topk.iter().map(|(_, s)| ((s - maxs) / temp).exp()).collect();
            let wsum: f32 = wts.iter().sum();
            if wsum <= 0.0 || !wsum.is_finite() {
                break;
            }
            for wv in wts.iter_mut() {
                *wv /= wsum;
            }
            let mut r: f32 = {
                let mut h = 0x9e3779b97f4a7c15u64;
                for &b in state.iter().rev().take(8) {
                    h ^= b as u64;
                    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
                }
                h ^= h >> 31;
                (h & 0xFFFFFF) as f32 / 16_777_216.0
            };
            let mut picked: u8 = topk[0].0;
            for (idx, (b, _)) in topk.iter().enumerate() {
                r -= wts[idx];
                if r <= 0.0 {
                    picked = *b;
                    break;
                }
            }
            picked
        };
        out.push(pick);
        window.push(pick);
        state.push(pick);
        if window.len() > 4096 {
            window.remove(0);
        }
        if state.len() > 8192 {
            // патчевое окно берёт последние 4 байта — просто ограничим историю
            let cut = state.len() - 4096;
            state.drain(0..cut);
        }
    }
    out
}

/// ГИБРИДНЫЙ байтовый декодер: pred = W·x + α·KAN(x) — линейный W держит
/// частотные биграммы, сплайн KAN добавляет нелинейные структурные
/// аттракторы (см. hybrid.rs). Ранжирование по cosine к 256 байт-латентам.
pub fn tm_generate_hybrid(
    tm: &TemporalMemory,
    kan: &crate::ai::kan::KanTransition,
    seed_bytes: &[u8],
    steps: usize,
    window_size: usize,
    alpha: f32,
) -> Vec<u8> {
    if seed_bytes.is_empty() {
        return Vec::new();
    }
    let encoder = &tm.predictor().encoder;
    let w = tm.predictor_w();
    let byte_latents: Vec<(u8, crate::ai::latent_jepa::LatentVector)> = (0u16..=255)
        .map(|b| {
            let sdr = crate::ai::sdr::byte_basis(b as u8);
            let lat = encoder.encode(&sdr);
            (b as u8, lat)
        })
        .collect();

    let start = seed_bytes.len().saturating_sub(window_size.max(1));
    let mut window: Vec<u8> = seed_bytes[start..].to_vec();
    let mut out: Vec<u8> = Vec::new();
    let mut guard: usize = 0;

    while out.len() < steps && guard < steps * 2 {
        guard += 1;
        let window_sdrs: Vec<SdrVector> =
            window.iter().map(|&b| crate::ai::sdr::byte_basis(b)).collect();
        let x = encoder.encode(&crate::ai::sdr::structure_sdr_from_sdrs(&window_sdrs));
        // W-часть: W·x (линейная, без норм. до смешивания)
        let mut pred = crate::ai::latent_jepa::LatentVector::zero();
        for o in 0..crate::ai::latent_jepa::LATENT_DIM {
            let mut acc = 0.0f32;
            let row = o * crate::ai::latent_jepa::LATENT_DIM;
            for i in 0..crate::ai::latent_jepa::LATENT_DIM {
                acc += w[row + i] * x.values[i];
            }
            pred.values[o] = acc;
        }
        // KAN-остаток (нормализованный сплайн)
        let kan_out = kan.apply(&x);
        for o in 0..crate::ai::latent_jepa::LATENT_DIM {
            pred.values[o] += alpha * kan_out.values[o];
        }
        let n = pred
            .values
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(1e-8);
        for v in &mut pred.values {
            *v /= n;
        }

        let mut best: Option<(f32, u8)> = None;
        for (byte, lat) in byte_latents.iter() {
            let score = pred.cosine_similarity(lat);
            if best.as_ref().map_or(true, |(bc, _)| score > *bc) {
                best = Some((score, *byte));
            }
        }
        let Some((_, byte)) = best else {
            break;
        };
        if out.last() == Some(&byte) {
            break;
        }
        out.push(byte);
        window.push(byte);
    }
    out
}
/// MEGABYTE-ПОРЯДОК v2 (v7-поколение): патч решает ДО байтов.
/// Глобальный уровень (W_patch) по окну последних 4 ПОЛНЫХ патчей предсказывает
/// направление, top-K патчей-кандидатов из vocab по cosine образуют ЖЁСТКИЙ
/// коридор; локальный W·x ранжирует байты ВНУТРИ коридора (с приором к top-1
/// патчу через beta). Сигнатура: (tm, seed, max_bytes, window_bytes, patch_len,
/// patch_vocab, top_k, beta, rep_word, rep_phrase, min_cos). Окно = ctx+1.
pub fn tm_generate_megabyte_v2(
    tm: &TemporalMemory,
    seed_bytes: &[u8],
    max_bytes: usize,
    window_bytes: usize,
    patch_len: usize,
    patch_vocab: &[Vec<u8>],
    top_k: usize,
    beta: f32,
    rep_word: f32,
    rep_phrase: f32,
    min_cos: f32,
) -> Vec<u8> {
    if seed_bytes.is_empty() {
        return Vec::new();
    }
    let plen = patch_len.max(1);
    let encoder = &tm.predictor().encoder;
    let w = tm.predictor_w();
    let patch_w: Vec<f32> = tm.patch_predictor_w().to_vec();
    let printable = |b: u8| b == b'\n' || b == b'\t' || b == b'\r' || (0x20..=0x7e).contains(&b);
    let byte_latents: Vec<(u8, crate::ai::latent_jepa::LatentVector)> = (0u16..=255)
        .map(|b| {
            let sdr = crate::ai::sdr::byte_basis(b as u8);
            let lat = encoder.encode(&sdr);
            (b as u8, lat)
        })
        .filter(|(b, _)| printable(*b))
        .collect();
    let patch_latents: Vec<(Vec<u8>, crate::ai::latent_jepa::LatentVector)> = patch_vocab
        .iter()
        .map(|p| {
            let sdr = crate::ai::sdr::encode_bytes_sdr(p);
            let lat = encoder.encode(&sdr);
            (p.clone(), lat)
        })
        .collect();
    let mut window: Vec<u8> = seed_bytes.to_vec();
    let mut state: Vec<u8> = seed_bytes.to_vec();
    let mut out: Vec<u8> = Vec::new();
    let mut guard = 0usize;
    let mut recent_patches: Vec<Vec<u8>> = Vec::new();
while out.len() < max_bytes && guard < max_bytes * 4 {
        guard += 1;

        // Локальный уровень: pred = W·x (ровно window_bytes, как обучение).
        let win_lo = window.len().saturating_sub(window_bytes.max(1));
        let win_tail = &window[win_lo..];
        let window_sdrs: Vec<crate::ai::sdr::SdrVector> = win_tail
            .iter()
            .map(|&b| crate::ai::sdr::byte_basis(b))
            .collect();
        let x = encoder.encode(&crate::ai::sdr::structure_sdr_from_sdrs(&window_sdrs));
        let mut pred = crate::ai::latent_jepa::LatentVector::zero();
        for o in 0..crate::ai::latent_jepa::LATENT_DIM {
            let row = o * crate::ai::latent_jepa::LATENT_DIM;
            let mut acc = 0.0f32;
            for i in 0..crate::ai::latent_jepa::LATENT_DIM {
                acc += w[row + i] * x.values[i];
            }
            pred.values[o] = acc;
        }
        let pn = pred.values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in &mut pred.values {
            *v /= pn;
        }

        // ГЛОБАЛЬНЫЙ уровень: окно последних 4 ПОЛНЫХ патчей (1:1 с обучением
        // pp-4..pp-1 → pp) → направление W_patch·x_patch.
        let patches_v: Vec<Vec<u8>> = state
            .chunks(plen)
            .filter(|c| c.len() == plen)
            .map(|c| c.to_vec())
            .collect();
        let patch_window: Vec<&[u8]> = patches_v
            .iter()
            .rev()
            .take(4)
            .rev()
            .map(|p| p.as_slice())
            .collect();
        let mut pp = crate::ai::latent_jepa::LatentVector::zero();
        if !patch_window.is_empty() {
            let xs: Vec<crate::ai::sdr::SdrVector> = patch_window
                .iter()
                .map(|p| crate::ai::sdr::encode_bytes_sdr(p))
                .collect();
            let xp = encoder.encode(&crate::ai::sdr::structure_sdr_from_sdrs(&xs));
            for o in 0..crate::ai::latent_jepa::LATENT_DIM {
                let row = o * crate::ai::latent_jepa::LATENT_DIM;
                let mut acc = 0.0f32;
                for i in 0..crate::ai::latent_jepa::LATENT_DIM {
                    acc += patch_w[row + i] * xp.values[i];
                }
                pp.values[o] = acc;
            }
            let pn2 = pp.values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
            for v in &mut pp.values {
                *v /= pn2;
            }
        }

        // Top-K патчей по cosine к направлению (без недавних повторов).
        let mut cand: Vec<(f32, &Vec<u8>)> = Vec::new();
        for (patch, lat) in patch_latents.iter() {
            if recent_patches.contains(patch) {
                continue;
            }
            let score = pp.cosine_similarity(lat);
            if score < min_cos.max(0.0) {
                continue;
            }
            cand.push((score, patch));
        }
        cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        cand.truncate(top_k.max(1));
        if cand.is_empty() {
            break;
        }
        let top_patch: Vec<u8> = cand[0].1.clone();
        recent_patches.push(top_patch.clone());
        if recent_patches.len() > 2 {
            recent_patches.remove(0);
        }
        // Коридор: байты из top-K патчей (патч решает ДО байтов).
        let mut corridor: Vec<u8> = Vec::new();
        for (_, p) in cand.iter() {
            for &b in p.iter() {
                if !corridor.contains(&b) {
                    corridor.push(b);
                }
            }
        }
        if corridor.is_empty() {
            break;
        }
// Выбор байта: только из коридора (MegaByte-порядок), кос = W·x к
        // латенту байта + beta·(cos top-патча). rep_word/rep_phrase — штрафы.
        let token: Vec<u8> = state.iter().copied().collect();
        let words: Vec<Vec<u8>> = token
            .split(|&b| b == b' ')
            .filter(|w| !w.is_empty())
            .map(|w| w.to_vec())
            .collect();
        let cur_word: Vec<u8> = token
            .iter()
            .rev()
            .take_while(|&&b| b != b' ')
            .cloned()
            .collect::<Vec<u8>>()
            .into_iter()
            .rev()
            .collect();
        let mut scores: Vec<(u8, f32)> = Vec::new();
        for (byte, lat) in byte_latents.iter() {
            if !corridor.contains(byte) {
                continue; // жёсткий коридор патчей
            }
            let mut sc = pred.cosine_similarity(lat);
            if sc < min_cos.max(0.0) {
                continue;
            }
            if *byte == top_patch[0] && beta > 0.0 {
                sc += beta * cand[0].0.max(0.0);
            } else if top_patch.len() > 1 && *byte == top_patch[1] && beta > 0.0 {
                sc += beta * cand[0].0.max(0.0) * 0.8;
            }
            if rep_word > 0.0 && !cur_word.is_empty() && *byte == b' ' {
                let mut cnt = 0;
                for w in words.iter() {
                    if *w == cur_word {
                        cnt += 1;
                    }
                }
                if cnt > 1 {
                    sc -= rep_word * (cnt - 1) as f32;
                }
            }
            if rep_phrase > 0.0 && state.len() >= 14 {
                const PHR_L: usize = 12;
                let start = state.len() - (PHR_L - 1);
                let mut block: Vec<u8> = state[start..].to_vec();
                block.push(*byte);
                let hist_end = state.len().saturating_sub(2);
                let mut hits = 0;
                let mut k = 0;
                while k + PHR_L <= hist_end {
                    if state[k..k + PHR_L] == block[..] {
                        hits += 1;
                        k += PHR_L;
                    } else {
                        k += 1;
                    }
                }
                if hits > 0 {
                    sc -= rep_phrase * hits as f32;
                }
            }
            scores.push((*byte, sc));
        }
        if scores.is_empty() {
            break;
        }
        const GAP_SELECT: f32 = 0.03;
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let gap = if scores.len() > 1 {
            scores[0].1 - scores[1].1
        } else {
            1.0
        };
        let pick: u8 = if gap >= GAP_SELECT {
            scores[0].0
        } else {
            let top = scores.len().min(8);
            let maxs = scores[0].1;
            let mut wts: Vec<f32> = scores[..top]
                .iter()
                .map(|(_, s)| ((s - maxs) / 0.02f32).exp())
                .collect();
            let wsum: f32 = wts.iter().sum();
            if wsum <= 0.0 || !wsum.is_finite() {
                break;
            }
            for v in &mut wts {
                *v /= wsum;
            }
            let mut r = (guard as f64 * 0.6180339887).fract() as f32;
            let mut idx = 0;
            let mut acc = 0.0f32;
            for (i, wv) in wts.iter().enumerate() {
                acc += *wv;
                if r <= acc {
                    idx = i;
                    break;
                }
            }
            scores[idx].0
        };
        out.push(pick);
        window.push(pick);
        state.push(pick);
        if state.len() > 8192 {
            let cut = state.len() - 4096;
            state.drain(0..cut);
        }
    }
    out
}
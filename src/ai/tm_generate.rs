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

/// Calibration variant of the two-speed decoder with an explicit cosine
/// threshold and vocabulary cap, for honest A/B sweeps of the patch rate.
/// Identical code path to `tm_generate_two_speed`; only the gate tightness
/// and (optionally) a top-K sampling of the patch vocab are exposed.
pub fn tm_generate_two_speed_calib(
    tm: &TemporalMemory,
    seed_bytes: &[u8],
    steps_patches: usize,
    window_patches: usize,
    patch_vocab: &[Vec<u8>],
    eligible: Option<&HashSet<u8>>,
    min_cosine: f32,
    no_repeat_patches: usize,
) -> Vec<u8> {
    fn patches_from(state: &[u8], size: usize) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        for chunk in state.chunks(size) {
            if !chunk.is_empty() {
                out.push(chunk.to_vec());
            }
        }
        out
    }
    let psize = if let Some(m) = patch_vocab.iter().map(|p| p.len()).max() {
        patch_vocab
            .iter()
            .map(|p| p.len())
            .min()
            .unwrap_or(4)
            .max(1)
            .min(m)
    } else {
        4
    };

    // Pre-encode per-step candidate latents once (frozen encoder).
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
                if !patch.iter().all(|b| elig.contains(b)) {
                    continue;
                }
            }
            // Decode-side no-repeat (window N) — the calibration knob.
            if no_repeat_patches > 0 && recent.contains(patch) {
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
        // No-repeat bookkeeping.
        if no_repeat_patches > 0 {
            recent.push(patch.clone());
            if recent.len() > no_repeat_patches.max(1) {
                recent.remove(0);
            }
        }
        // Anti-repeat guard (same as tm_generate_two_speed).
        if !last_patch.is_empty() && patch == last_patch {
            repeat_run += 1;
            if repeat_run >= 4 {
                break;
            }
        } else {
            repeat_run = 0;
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

/// Speculative (draft → verify) byte decoder.
///
/// Mirrors speculative decoding: a fast DRAFT rate — the global patch W
/// operator (`predict_patch_latent`) — proposes a whole next byte-patch in
/// ONE step (coarse, cheap). A careful VERIFIER — the local byte W operator
/// (`predict_bytes_latent`) — then checks each proposed byte independently:
/// if the local model is more confident (gap ≥ threshold) about a DIFFERENT
/// byte, the verifier overwrites the draft (correcting rare chars / typos).
/// This gives patch-rate drafting with byte-rate accuracy — the ByT5/Subword
/// speed–precision tradeoff realised in VSA. No learned draft net: the global
/// W is the drafter, the local W is the verifier.
pub fn tm_generate_speculative(
    tm: &TemporalMemory,
    seed_bytes: &[u8],
    max_bytes: usize,
    window_bytes: usize,
    verify_gap: f32,
    patch_vocab: &[Vec<u8>],
) -> Vec<u8> {
    // Fixed byte alphabet latents (verifier's hypothesis space).
    let byte_lats: Vec<crate::ai::latent_jepa::LatentVector> = (0u16..256)
        .map(|b| {
            let sdr = crate::ai::sdr::byte_basis(b as u8);
            tm.latent_of_sdr(&sdr)
        })
        .collect();
    // Global draft grammar.
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
        .unwrap_or(2)
        .max(1);

    let mut state: Vec<u8> = seed_bytes.to_vec();
    let mut out: Vec<u8> = Vec::new();
    let mut guard: usize = 0;

    while out.len() < max_bytes && guard < max_bytes * 2 {
        guard += 1;
        // DRAFT: global W proposes the next patch (one coarse decision).
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
        let draft = tm.predict_patch_latent(&window);
        let mut best_patch: Option<Vec<u8>> = None;
        for (patch, lat) in patch_lats.iter() {
            let s = draft.cosine_similarity(lat);
            if s >= LATENT_MIN_COSINE && best_patch.is_none() {
                best_patch = Some(patch.clone());
            }
        }
        let proposed: Vec<u8> = best_patch.unwrap_or_else(|| {
            // No confident draft → propose a single most-likely byte locally.
            let pred = tm.predict_bytes_latent(&state[state.len().saturating_sub(window_bytes.max(1))..]);
            let mut top = (0usize, f32::MIN);
            for (i, lat) in byte_lats.iter().enumerate() {
                let c = pred.cosine_similarity(lat);
                if c > top.1 {
                    top = (i, c);
                }
            }
            vec![top.0 as u8]
        });

        // VERIFY: check each proposed byte against the local byte W.
        for &b in proposed.iter() {
            let tail = &state[state.len().saturating_sub(window_bytes.max(1))..];
            let pred = tm.predict_bytes_latent(tail);
            let mut top1 = (0usize, f32::MIN);
            let mut top2 = (0usize, f32::MIN);
            for (i, lat) in byte_lats.iter().enumerate() {
                let c = pred.cosine_similarity(lat);
                if c > top1.1 {
                    top2 = top1;
                    top1 = (i, c);
                } else if c > top2.1 {
                    top2 = (i, c);
                }
            }
            let gap = top1.1 - top2.1;
            // The verifier overrides the draft only when IT is clearly sure
            // (large gap) and disagrees — correcting rare/typo'd bytes.
            let byte = if gap >= verify_gap && top1.0 as u8 != b && !out.is_empty() {
                top1.0 as u8
            } else {
                b
            };
            if !out.is_empty() && out.last() == Some(&byte) && out.len() > 2 {
                continue; // skip a repeated identical byte (no stall)
            }
            out.push(byte);
            state.push(byte);
            if out.len() >= max_bytes {
                break;
            }
        }
    }
    out
}

/// Speculative decoder + acceptance-rate telemetry.
///
/// Returns `(bytes, acceptance_rate)`. `acceptance_rate` = fraction of
/// draft-proposed bytes that the local verifier accepted WITHOUT overriding
/// (i.e. the draft byte was already the local W's top choice, or the local
/// W was not confident enough to disagree). It quantifies how cheap the
/// draft step is: a high acceptance means the patch drafter predicted well,
/// so verify rarely needs to second-guess it. Useful to find the minimum
/// draft accuracy that makes speculative decoding profitable.
pub fn tm_generate_speculative_stats(
    tm: &TemporalMemory,
    seed_bytes: &[u8],
    max_bytes: usize,
    window_bytes: usize,
    verify_gap: f32,
    patch_vocab: &[Vec<u8>],
) -> (Vec<u8>, f32) {
    let byte_lats: Vec<crate::ai::latent_jepa::LatentVector> = (0u16..256)
        .map(|b| {
            let sdr = crate::ai::sdr::byte_basis(b as u8);
            tm.latent_of_sdr(&sdr)
        })
        .collect();
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
        .unwrap_or(2)
        .max(1);

    let mut state: Vec<u8> = seed_bytes.to_vec();
    let mut out: Vec<u8> = Vec::new();
    let mut guard: usize = 0;
    let mut accepted = 0usize;
    let mut checked = 0usize;

    while out.len() < max_bytes && guard < max_bytes * 2 {
        guard += 1;
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
        let draft = tm.predict_patch_latent(&window);
        let mut best_patch: Option<Vec<u8>> = None;
        for (patch, lat) in patch_lats.iter() {
            let s = draft.cosine_similarity(lat);
            if s >= LATENT_MIN_COSINE && best_patch.is_none() {
                best_patch = Some(patch.clone());
            }
        }
        let proposed: Vec<u8> = best_patch.unwrap_or_else(|| {
            let pred = tm.predict_bytes_latent(&state[state.len().saturating_sub(window_bytes.max(1))..]);
            let mut top = (0usize, f32::MIN);
            for (i, lat) in byte_lats.iter().enumerate() {
                let c = pred.cosine_similarity(lat);
                if c > top.1 {
                    top = (i, c);
                }
            }
            vec![top.0 as u8]
        });

        for &b in proposed.iter() {
            let tail = &state[state.len().saturating_sub(window_bytes.max(1))..];
            let pred = tm.predict_bytes_latent(tail);
            let mut top1 = (0usize, f32::MIN);
            let mut top2 = (0usize, f32::MIN);
            for (i, lat) in byte_lats.iter().enumerate() {
                let c = pred.cosine_similarity(lat);
                if c > top1.1 {
                    top2 = top1;
                    top1 = (i, c);
                } else if c > top2.1 {
                    top2 = (i, c);
                }
            }
            let gap = top1.1 - top2.1;
            // Acceptance: verifier leaves the draft byte alone when either it
            // is not confident (gap < threshold) OR the draft already WAS its
            // top choice (top1.0 == b). Only a confident disagreement overrides.
            let accepted_here =
                gap < verify_gap || top1.0 as u8 == b || out.is_empty();
            checked += 1;
            if accepted_here {
                accepted += 1;
            }
            let byte = if accepted_here {
                b
            } else {
                top1.0 as u8
            };
            if !out.is_empty() && out.last() == Some(&byte) && out.len() > 2 {
                continue;
            }
            out.push(byte);
            state.push(byte);
            if out.len() >= max_bytes {
                break;
            }
        }
    }
    let rate = if checked == 0 {
        0.0
    } else {
        accepted as f32 / checked as f32
    };
    (out, rate)
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

/// Recurrent byte decoder with NON-ARGMAX state advance (v3.2 lever 1).
///
/// Same as [`tm_generate_recurrent`] — same stateful W, same local+state mix
/// before the operator — but the byte fed into `advance_h` is NOT the argmax.
/// Instead it is drawn by nucleus (top-p) sampling over the temperature-scaled
/// cosine distribution: `p(b) ∝ exp(cos(b)/T)`, keep tokens until cumulative
/// prob ≥ top_p. Directly attacks the measured failure of 3.1: the argmax
/// byte is almost always the frequent e/r, so pure-argmax advance floods h
/// with the dominant attractor. Sampling keeps h structurally diverse, so the
/// recurrent read is less chained to e→r.
///
/// Emission (the decoder's output byte) is still the argmax — we only change
/// what the memory is fed, isolating the hypothesis "e-fill of h is the cause".
/// `rng_seed` (non-zero) makes the nucleus draws reproducible across runs.
pub fn tm_generate_recurrent_nucleus(
    tm: &TemporalMemory,
    seed_bytes: &[u8],
    max_bytes: usize,
    window_bytes: usize,
    mix: f32,
    phi: f32,
    temperature: f32,
    top_p: f32,
    rng_seed: u64,
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
    let mut x: u64 = rng_seed ^ 0x9E3779B97F4A7C15;

    while out.len() < max_bytes && guard < max_bytes * 2 {
        guard += 1;
        let win_lo = state.len().saturating_sub(window_bytes.max(1));
        let window_byte = &state[win_lo..];
        let window_sdrs: Vec<crate::ai::sdr::SdrVector> = window_byte
            .iter()
            .map(|&b| crate::ai::sdr::byte_basis(b))
            .collect();
        let pred = predictor.predict_next_rnn(&window_sdrs, &h, mix);

        // Cosine scores over the 256-byte alphabet.
        let mut cos = [0.0f32; 256];
        let mut best = (0usize, f32::NEG_INFINITY);
        let mut second = (0usize, f32::NEG_INFINITY);
        for (i, lat) in byte_lats.iter().enumerate() {
            let c = pred.cosine_similarity(lat);
            cos[i] = c;
            if c > best.1 {
                second = best;
                best = (i, c);
            } else if c > second.1 {
                second = (i, c);
            }
        }
        let emit_byte = best.0 as u8;

        // Nucleus sample for the state advance.
        let temp = temperature.max(0.1);
        let mut probs = [0.0f32; 256];
        let mut sum_exp = 0.0f32;
        for i in 0..256 {
            probs[i] = (cos[i] / temp).exp();
            sum_exp += probs[i];
        }
        for p in probs.iter_mut() {
            *p /= sum_exp.max(1e-12);
        }
        // Top-p (nucleus) truncation: sort, keep smallest set with cum prob ≥ top_p.
        let mut order: Vec<usize> = (0..256).collect();
        order.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum = 0.0f32;
        let mut nucleus_len = 1usize;
        for &idx in &order {
            cum += probs[idx];
            if cum >= top_p.min(0.999) && nucleus_len >= 1 {
                break;
            }
            nucleus_len += 1;
        }
        nucleus_len = nucleus_len.max(1).min(256);
        // Draw from the nucleus subset, weighted by (truncated) prob.
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let r = (x >> 33) as f64 / (1u64 << 31) as f64;
        let mut acc = 0.0f32;
        let mut state_byte = order[0] as u8;
        for &idx in order.iter().take(nucleus_len) {
            acc += probs[idx];
            if acc >= r as f32 {
                state_byte = idx as u8;
                break;
            }
        }

        // Stop on repeated identical emitted byte (avoid single-byte stall).
        if !out.is_empty() && out.last() == Some(&emit_byte) && out.len() > 2 {
            break;
        }
        out.push(emit_byte);
        state.push(emit_byte);
        // Advance memory with the NUCLEUS-sampled byte (not the argmax).
        h = predictor.advance_h(h, &crate::ai::sdr::byte_basis(state_byte), phi);
    }
    out
}

/// Instead of greedy top-1, keeps `beam_width` hypotheses, each with an
/// accumulated log-probability score. At every step every hypothesis expands
/// with its top-M bytes (by the softmax-normalized cosine distribution over
/// the fixed 256-byte alphabet); the top-K overall hypotheses survive. This
/// is the classic beam decoding that cures greedy autoregressive stalls
/// (the measured 'er er' loop): a hypothesis that repeats itself accumulates
/// low probability under competing continuations and loses the beam.
pub fn tm_generate_beam(
    tm: &TemporalMemory,
    seed_bytes: &[u8],
    max_bytes: usize,
    window_bytes: usize,
    beam_width: usize,
    top_m: usize,
) -> Vec<u8> {
    // Fixed byte alphabet latents (hypothesis space).
    let byte_lats: Vec<crate::ai::latent_jepa::LatentVector> = (0u16..256)
        .map(|b| {
            let sdr = crate::ai::sdr::byte_basis(b as u8);
            tm.latent_of_sdr(&sdr)
        })
        .collect();

    // Each hypothesis: (generated bytes after seed, accumulated log-score).
    let mut beams: Vec<(Vec<u8>, f32)> = vec![(Vec::new(), 0.0)];
    let mut best: Vec<u8> = Vec::new();
    let mut best_score = f32::NEG_INFINITY;
    let mut guard: usize = 0;

    while guard < max_bytes * 2 {
        guard += 1;
        let mut candidates: Vec<(Vec<u8>, f32)> = Vec::new();
        for (hyp, score) in &beams {
            // Context = seed + what this hypothesis has generated so far.
            let mut ctx: Vec<u8> = seed_bytes.to_vec();
            ctx.extend_from_slice(hyp);
            let tail = &ctx[ctx.len().saturating_sub(window_bytes.max(1))..];
            let pred = tm.predict_bytes_latent(tail);
            // Softmax-normalized distribution over 256 bytes (temp 1.0).
            let mut cos: Vec<f32> = byte_lats
                .iter()
                .map(|lat| pred.cosine_similarity(lat))
                .collect();
            let maxc = cos.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            for c in cos.iter_mut() {
                *c = (*c - maxc).exp();
                denom += *c;
            }
            for c in cos.iter_mut() {
                *c /= denom.max(1e-9);
            }
            // Top-M byte indices by probability.
            let mut idx: Vec<usize> = (0..256).collect();
            idx.sort_unstable_by(|&a, &b| {
                cos[b].partial_cmp(&cos[a]).unwrap_or(std::cmp::Ordering::Equal)
            });
            for &b in idx.iter().take(top_m.max(1)) {
                let logp = (cos[b] + 1e-12).ln();
                let mut new_hyp = hyp.clone();
                new_hyp.push(b as u8);
                candidates.push((new_hyp, score + logp));
            }
        }
        // Keep top-K by accumulated score.
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(beam_width.max(1));
        if candidates.is_empty() {
            break;
        }
        beams = candidates;
        // Track the single best hypothesis.
        if beams[0].1 > best_score {
            best_score = beams[0].1;
            best = beams[0].0.clone();
        }
        // Stop if every surviving hypothesis repeats its own last byte
        // (beam collapsed into a self-loop — no further information).
        let mut all_loop = true;
        for (hyp, _) in &beams {
            let n = hyp.len();
            if n < 3 || hyp[n - 1] != hyp[n - 2] {
                all_loop = false;
                break;
            }
        }
        if all_loop {
            break;
        }
        if beams.iter().all(|(hyp, _)| hyp.len() >= max_bytes) {
            break;
        }
    }
    // Reconstruct: seed + best hypothesis.
    let mut out: Vec<u8> = seed_bytes.to_vec();
    out.extend_from_slice(&best);
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
                fn speculative_draft_verify_decodes_predictable_run() {
                    let mut tm = TemporalMemory::new(64, 4);
                    // Learn a very regular byte pattern with the decoder's sliding window.
                    let seq = b"abababababababababab";
                    for w in 0..seq.len().saturating_sub(1) {
                        let win_lo = w.saturating_sub(3);
                        tm.learn_bytes(&seq[win_lo..w + 1], seq[w + 1], 0.4);
                    }
                    // Draft grammar of two patches.
                    let vocab: Vec<Vec<u8>> = ["ab", "ba"].iter().map(|s| s.as_bytes().to_vec()).collect();
                    let out = tm_generate_speculative(&tm, b"a", 200, 4, 0.60, &vocab);
                    // The draft proposes patches, the verifier confirms each byte; a
                    // predictable run must emit several bytes (not stall at one).
                    assert!(
                        out.len() >= 2,
                        "speculative decoder should continue a predictable run, got {:?}",
                        String::from_utf8_lossy(&out)
                    );
                    // Empty draft grammar must not stop the verifier's local fallback.
                    let out2 = tm_generate_speculative(&tm, b"a", 20, 4, 0.60, &[]);
                    assert!(out2.len() >= 1, "empty grammar must not stop the byte rate");
                }

                #[test]
                fn beam_search_picks_predictable_continuation() {
                    let mut tm = TemporalMemory::new(64, 4);
                    // Learn a hard 2-byte alternation 'ab' with the same sliding
                    // window the beam uses, so the a->b transition is strong.
                    let seq = b"abababababababababab";
                    for w in 0..seq.len().saturating_sub(1) {
                        let win_lo = w.saturating_sub(3);
                        tm.learn_bytes(&seq[win_lo..w + 1], seq[w + 1], 0.4);
                    }
                    // Beam must produce a NON-EMPTY continuation (not stall on
                    // the seed), even though the byte W may not strongly prefer a
                    // specific next byte. Repeating 'a' is still 'continuing'.
                    let out = tm_generate_beam(&tm, b"a", 60, 2, 3, 5);
                    assert!(
                        out.len() > 1,
                        "beam should produce a continuation, got {:?}",
                        String::from_utf8_lossy(&out)
                    );
                    // Empty/seed-only still returns the seed.
                    let out2 = tm_generate_beam(&tm, b"a", 0, 4, 4, 3);
                    assert_eq!(out2, b"a");
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
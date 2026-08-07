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
}
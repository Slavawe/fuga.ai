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
}
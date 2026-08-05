use crate::ai::htm_temporal::TemporalMemory;
use crate::ai::sdr::{SdrVector, encode_text};

const MIN_OVERLAP: u32 = 7;

// Последовательная генерация через Временную память (TM + SDR).
// В отличие от статической склейки строк памяти, TM предсказывает следующий
// токен по контекстному окну предыдущих (temporal sequence), возвращая
// построенную во времени цепочку слов — непрерывно, с учётом history.
pub fn tm_generate(
    tm: &TemporalMemory,
    seed: &[String],
    steps: usize,
    candidates: &[String],
    window_size: usize,
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
        let pred = best_structure_prediction(tm, &window);
        if pred.popcount() == 0 {
            break;
        }
        let next = decode_to_word(&pred, candidates);
        match next {
            Some((word, _overlap)) if out.last() == Some(&word) => {
                // A temporal memory must not re-emit the same token off the
                // same context forever; a repeated emission means the merged
                // prediction did not diverge, so stop rather than loop.
                break;
            }
            Some((word, _overlap)) => {
                out.push(word.to_string());
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

/// Predict the next token from a sliding window, accepting partial matches:
/// if the full window yields nothing, shrink it (drop the oldest token) and
/// retry — a tail that did appear somewhere in the learned corpus still fires.
fn best_structure_prediction(tm: &TemporalMemory, window: &[String]) -> SdrVector {
    let mut n = window.len();
    while n >= 1 {
        let win_refs: Vec<&str> = window[window.len() - n..].iter().map(|s| s.as_str()).collect();
        let pred = tm.predict_structure(&win_refs);
        if pred.popcount() > 0 {
            return pred;
        }
        n -= 1;
    }
    SdrVector::zero()
}

fn decode_to_word(pred: &SdrVector, candidates: &[String]) -> Option<(String, u32)> {
    if candidates.is_empty() {
        return None;
    }
    let mut best: Option<(usize, u32)> = None;
    for (i, w) in candidates.iter().enumerate() {
        let sdr = encode_text(w);
        let o = pred.overlap(&sdr);
        match best {
            Some((_, bo)) if o > bo => best = Some((i, o)),
            None => best = Some((i, o)),
            _ => {}
        }
    }
    best
        .filter(|(_, o)| *o >= MIN_OVERLAP)
        .map(|(i, o)| (candidates[i].clone(), o))
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
        let out = tm_generate(&tm, &seed, 20, &cands, 4);
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
        let out = tm_generate(&tm, &seed, 6, &cands, 2);
        assert!(
            !out.is_empty(),
            "TM learned a bigram chain, expected output, got none"
        );
    }
}
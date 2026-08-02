//! Tokenizer Bridge — semantic phase dictionary.
//!
//! Instead of storing 128K heavy PyTorch embed vectors, we keep only the
//! tokenizer vocabulary (plain JSON / text, a few MB) and *derive* a
//! deterministic VSA hypervector per token on the fly with an n-gram
//! permute-bind encoder. Decoding is a resonance scan over the generated
//! dictionary: O(vocab) XOR+popcount per query, no weights, no disk blob.
//!
//! Honest scope: this is a *text-space* phase bridge. The crystal's raw
//! expert bits live in tensor-weight space and do not share an alphabet with
//! VSA(token). To cross domains we bridge through the crystal's key_text
//! labels (real strings): label -> VSA(label) -> tokens.

use crate::ai::crystal::fnv1a;
use crate::core::hypervector::Hypervector;

/// How many ones each per-hash basis vector gets (sparse basis alphabet).
const BASIS_ONES: u32 = 16;
/// Number of bytes treated as a unit in the n-gram windows.
const NGRAM_MAX: usize = 3;
/// Deterministic per-token gram budget. Selecting the lowest-hashed grams
/// bounds every token's HV density (~<=19%), so long tokens do not drown out
/// short subword matches by pure bit-density.
const MAX_GRAMS: usize = 48;

/// Deterministic sparse basis vector for a 64-bit hash. Each hash maps to a
/// fixed sparse bit pattern, so encode() is stable across runs and cheap:
/// O(BASIS_ONES) bit-sets, no allocation-heavy permute.
fn hash_basis(h: u64, dim: usize) -> Hypervector {
    let mut hv = Hypervector::new(dim);
    let mut x = (h ^ 0x9E3779B97F4A7C15).wrapping_mul(0x5851F42D4C957F2D) as usize;
    let mut placed = 0u32;
    while placed < BASIS_ONES {
        x = x
            .wrapping_mul(0x5851F42D4C957F2D)
            .wrapping_add(0x14057B7EF767814F);
        let bit = x % dim;
        hv.words[bit / 64] |= 1u64 << (bit % 64);
        placed += 1;
    }
    hv
}

/// n-gram VSA encoder over raw bytes (sparse, bounded).
///
/// Every n-gram (size 1..=NGRAM_MAX) hashes to a sparse basis; only the
/// `MAX_GRAMS` lowest *content* hashes are kept (deterministic — the same
/// gram is always kept, independent of position/length), and position is
/// folded into the basis seed so order matters. Duplicate grams (repeated
/// words) collapse to one hash before selection, so frequent substrings
/// don't hog the bounded slot budget. Result: all tokens have a
/// bounded ones-count, so cosine resonance is content-driven, not density-
/// driven. O(len · NGRAM_MAX · log(len)) — usable for a 128K-token vocab.
pub fn encode_bytes(bytes: &[u8], dim: usize) -> Hypervector {
    encode_bytes_impl(bytes, dim, true)
}

/// Position-invariant n-gram VSA. Same content hash selection, but the basis
/// seed is the raw content hash (no fold of the gram start position). A query
/// fragment taken from the middle/end of a chunk still resonates, because the
/// same n-gram always maps to the same basis regardless of where it occurs.
pub fn encode_bytes_nopos(bytes: &[u8], dim: usize) -> Hypervector {
    encode_bytes_impl(bytes, dim, false)
}

/// nopos encoder that drops n-grams shorter than `min_gram` bytes. Kept
/// separate from the crystal path so the compact 1..=3-gram encoding (and all
/// stored phases) stays untouched. The Tri-Anchor framework uses this variant
/// because 1–2 byte grams (spaces, "a", "th") carry huge corpus-wide stopword
/// crosstalk — they always land in the top-MAX_GRAMS content hashes and give
/// any two unrelated phrases ~20% spurious overlap. Requiring >=3-byte grams
/// cuts that floor in half while preserving real word-level overlap.
pub fn encode_bytes_nopos_min3(bytes: &[u8], dim: usize) -> Hypervector {
    encode_bytes_impl_min(bytes, dim, 3)
}

fn encode_bytes_impl_min(bytes: &[u8], dim: usize, min_gram: usize) -> Hypervector {
    let mut acc = Hypervector::new(dim);
    if bytes.is_empty() {
        return acc;
    }
    let n = bytes.len();
    let mut grams: Vec<(u64, usize)> = Vec::new();
    for i in 0..n {
        for w in min_gram..=NGRAM_MAX {
            if i + w > n {
                break;
            }
            grams.push((fnv1a(&bytes[i..i + w]), i));
        }
    }
    grams.sort_by_key(|g| g.0);
    grams.dedup_by_key(|g| g.0);
    for &(hc, _s) in grams.iter().take(MAX_GRAMS) {
        acc = acc.bind(&hash_basis(hc, dim));
    }
    acc
}

fn encode_bytes_impl(bytes: &[u8], dim: usize, positional: bool) -> Hypervector {
    let mut acc = Hypervector::new(dim);
    if bytes.is_empty() {
        return acc;
    }
    let n = bytes.len();
    let mut grams: Vec<(u64, usize)> = Vec::new(); // (content hash, start)
    for i in 0..n {
        for w in 1..=NGRAM_MAX {
            if i + w > n {
                break;
            }
            grams.push((fnv1a(&bytes[i..i + w]), i));
        }
    }
    grams.sort_by_key(|g| g.0);
    grams.dedup_by_key(|g| g.0);
    for &(hc, s) in grams.iter().take(MAX_GRAMS) {
        let hb = if positional {
            hc ^ (s as u64).wrapping_mul(0x9E3779B97F4A7C15)
        } else {
            hc
        };
        acc = acc.bind(&hash_basis(hb, dim));
    }
    acc
}

pub fn encode_str(text: &str, dim: usize) -> Hypervector {
    encode_bytes(text.as_bytes(), dim)
}

/// Deterministic, cheap on-the-fly dictionary. Materializing all 128K
/// Hypervectors costs ~131 MB RAM (dim 8192); storage is just the vocab text.
pub struct TokenBridge {
    pub dim: usize,
    pub tokens: Vec<String>,
    pub hvs: Vec<Hypervector>,
}

impl TokenBridge {
    pub fn new(tokens: Vec<String>, dim: usize) -> Self {
        let hvs = tokens.iter().map(|t| encode_str(t, dim)).collect();
        TokenBridge { dim, tokens, hvs }
    }

    /// Resonance scan with a cosine-like score: overlap / sqrt(query_ones ·
    /// token_ones). Fully-contained subword tokens score high (~0.3–0.6),
    /// while random 2-byte tokens stay near 0.05 — clean separation.
    pub fn nearest(&self, query: &Hypervector, top_k: usize) -> Vec<(String, f64)> {
        let qw = query.words.len();
        let qones = query.words.iter().map(|w| w.count_ones()).sum::<u32>() as f64;
        let mut scored: Vec<(usize, f64)> = Vec::with_capacity(self.tokens.len());
        for (i, hv) in self.hvs.iter().enumerate() {
            if self.tokens[i].len() < 2 {
                continue;
            }
            let overlap: u32 = (0..qw)
                .map(|w| (query.words[w] & hv.words[w]).count_ones())
                .sum();
            let tones = hv.words.iter().map(|w| w.count_ones()).sum::<u32>() as f64;
            let denom = (qones * tones).sqrt();
            let res = if denom <= 0.0 {
                0.0
            } else {
                overlap as f64 / denom
            };
            if res > 0.0 {
                scored.push((i, res));
            }
        }
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(top_k)
            .map(|(i, res)| (self.tokens[i].clone(), res))
            .collect()
    }
}

/// Aggregate a set of label strings into one phase HV (OR of their encodes).
/// This is the cross-domain bridge entry point: labels are real text taken
/// from crystal resonance, so the result lives in text space.
pub fn aggregate_labels(labels: &[&str], dim: usize) -> Hypervector {
    let mut acc = Hypervector::new(dim);
    for l in labels {
        let hv = encode_str(l, dim);
        for (w, bit) in hv.words.iter().enumerate() {
            acc.words[w] |= *bit;
        }
    }
    acc
}

/// Load vocab keys from a HuggingFace `tokenizer.json` (model.vocab map).
pub fn load_vocab_from_tokenizer_json(path: &str) -> Result<Vec<String>, String> {
    let data = std::fs::read_to_string(path).map_err(|e| format!("read {}: {}", path, e))?;
    let parsed: serde_json::Value =
        serde_json::from_str(&data).map_err(|e| format!("parse {}: {}", path, e))?;
    let vocab = parsed
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(|v| v.as_object())
        .ok_or_else(|| format!("{}: no model.vocab map", path))?;
    let mut tokens: Vec<String> = vocab.keys().cloned().collect();
    // Drop pure whitespace/special markers? Keep everything; sorting keeps output stable.
    tokens.sort();
    Ok(tokens)
}

/// Load vocab from a plain text file, one token per line (whitespace stripped).
pub fn load_vocab_from_txt(path: &str) -> Result<Vec<String>, String> {
    let data = std::fs::read_to_string(path).map_err(|e| format!("read {}: {}", path, e))?;
    let mut tokens: Vec<String> = data
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    tokens.sort();
    tokens.dedup();
    Ok(tokens)
}

use crate::core::hypervector::Hypervector;
use std::collections::HashMap;
use std::sync::Mutex;

pub const SDR_DIM: usize = 8192;
pub const SDR_WORDS: usize = SDR_DIM / 64;
pub const SDR_DENSITY: f64 = 0.02;
pub const BOW_DENSITY: f64 = 0.20;
/// Density for structure-folded vectors: a whole-sequence cross-product needs
/// more active bits than a single token (0.02) to stay discriminative against
/// chance, but far fewer than bag-of-words (0.20) so distinct orderings remain
/// separable.
pub const STRUCTURE_DENSITY: f64 = 0.06;
/// Position stride for structure folding. Coprime with SDR_DIM so consecutive
/// positions map to distinct, well-mixed bit offsets and token order is
/// preserved in the super-vector.
///
/// # Invariant (mine)
/// `gcd(SDR_DIM, STRUCTURE_STRIDE) == 1`. If it ever breaks — SDR_DIM changes,
/// or the stride is edited — consecutive positions alias and folded vectors
/// collide (`fn main` == `main fn`), silently destroying order discriminability.
/// Enforced at the fold site via `structure_shift`.
const STRUCTURE_STRIDE: usize = 977;

/// Position-dependent bit offset for structure folding. Applies
/// `STRUCTURE_STRIDE` and guards the coprime invariant (see constant doc).
#[inline]
fn structure_shift(pos: usize) -> usize {
    debug_assert!(
        euclid_gcd(SDR_DIM, STRUCTURE_STRIDE) == 1,
        "STRUCTURE_STRIDE must be coprime with SDR_DIM (aliasing folds); now gcd={}",
        euclid_gcd(SDR_DIM, STRUCTURE_STRIDE)
    );
    (pos * STRUCTURE_STRIDE) % SDR_DIM
}

#[inline]
const fn euclid_gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

lazy_static::lazy_static! {
    static ref TOKEN_SDR_CACHE: Mutex<HashMap<u32, SdrVector>> = Mutex::new(HashMap::new());
}

pub fn clear_token_sdr_cache() {
    if let Ok(mut cache) = TOKEN_SDR_CACHE.lock() {
        cache.clear();
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SdrVector {
    pub bits: [u64; SDR_WORDS],
}

impl SdrVector {
    pub fn zero() -> Self {
        SdrVector {
            bits: [0; SDR_WORDS],
        }
    }

    pub fn from_hypervector(hv: &Hypervector) -> Self {
        let mut sdr = SdrVector::zero();
        let dim = hv.dim.min(SDR_DIM);
        let target = (dim as f64 * SDR_DENSITY).ceil() as usize;

        let hv_seed: u64 = hv
            .words
            .iter()
            .fold(0u64, |acc, w| acc.wrapping_mul(31).wrapping_add(*w));

        // Score each set bit with a well-mixed hash of (hv_seed, bit) so the
        // top-`target` selection is uniformly random per input. The previous
        // reverse_bits(hv_seed*(bit+1)) scoring was degenerate: it picked the
        // same low bit positions for every input, giving ~50% overlap between
        // unrelated vectors (see noise-regression: fact:first_code 0.575).
        let mut scored: Vec<(usize, u64)> = hv
            .words
            .iter()
            .enumerate()
            .flat_map(|(wi, &w)| {
                let base = wi * 64;
                (0..64.min(dim - base)).filter_map(move |bi| {
                    let bit = base + bi;
                    if bit >= dim {
                        return None;
                    }
                    if (w >> bi) & 1 == 0 {
                        return None;
                    }
                    let mut x = hv_seed.wrapping_add(bit as u64);
                    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
                    x = x ^ (x >> 31);
                    Some((bit, x))
                })
            })
            .collect();

        if scored.len() > target {
            scored.select_nth_unstable_by(target, |a, b| a.1.cmp(&b.1));
            scored.truncate(target);
        }
        scored.sort_unstable_by(|a, b| a.1.cmp(&b.1));

        for &(bit, _) in &scored {
            let wi = bit / 64;
            let bi = bit % 64;
            sdr.bits[wi] |= 1u64 << bi;
        }
        sdr
    }

    pub fn overlap(&self, other: &SdrVector) -> u32 {
        self.bits
            .iter()
            .zip(other.bits.iter())
            .map(|(&a, &b)| (a & b).count_ones())
            .sum()
    }

    pub fn soft_overlap(&self, other: &SdrVector) -> f64 {
        let intersection = self.overlap(other) as f64;
        let denom = self.popcount().max(other.popcount()) as f64;
        if denom < 1.0 {
            0.0
        } else {
            intersection / denom
        }
    }

    pub fn popcount(&self) -> u32 {
        self.bits.iter().map(|w| w.count_ones()).sum()
    }

    pub fn bit_at(&self, idx: usize) -> u64 {
        if idx >= SDR_DIM { 0 } else { (self.bits[idx / 64] >> (idx % 64)) & 1 }
    }

    pub fn bind(&self, other: &SdrVector) -> SdrVector {
        let mut out = SdrVector::zero();
        for i in 0..SDR_WORDS {
            out.bits[i] = self.bits[i] ^ other.bits[i];
        }
        out
    }

    pub fn bundle(&self, others: &[&SdrVector]) -> SdrVector {
        let mut out = SdrVector::zero();
        for i in 0..SDR_WORDS {
            let mid = (others.len() + 1) as u32 / 2;
            out.bits[i] = 0;
            for bi in 0..64 {
                let mut ones: u32 = ((self.bits[i] >> bi) & 1) as u32;
                for o in others {
                    ones += ((o.bits[i] >> bi) & 1) as u32;
                }
                if ones > mid {
                    out.bits[i] |= 1u64 << bi;
                }
            }
        }
        out
    }

    pub fn bundle_multi(sdrs: &[SdrVector]) -> SdrVector {
        if sdrs.is_empty() {
            return SdrVector::zero();
        }
        let refs: Vec<&SdrVector> = sdrs.iter().collect();
        refs[0].bundle(&refs[1..])
    }

    /// Bitwise union of a sequence of SDRs. Used to build a context vector that
    /// preserves every active bit from the recent window (unlike `bundle`,
    /// which applies a majority threshold and collapses on sparse inputs).
    pub fn union(sdrs: &[SdrVector]) -> SdrVector {
        let mut out = SdrVector::zero();
        for s in sdrs {
            for i in 0..SDR_WORDS {
                out.bits[i] |= s.bits[i];
            }
        }
        out
    }

    pub fn to_hypervector(&self, dim: usize) -> Hypervector {
        let mut words = vec![0u64; dim / 64 + (dim % 64).min(1)];
        for (wi, &w) in self.bits.iter().enumerate() {
            if wi < words.len() {
                words[wi] = w;
            }
        }
        Hypervector { dim, words }
    }
}

pub fn sparsify(hv: &Hypervector) -> SdrVector {
    SdrVector::from_hypervector(hv)
}

use std::hash::{Hash, Hasher};

use rand::RngCore;
use rand::SeedableRng;

fn deterministic_hv(seed: u64) -> Hypervector {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let wc = 128;
    let mut words = vec![0u64; wc];
    for w in &mut words {
        *w = rng.next_u64();
    }
    Hypervector { dim: 8192, words }
}

/// Stop words and noise tokens that dominate corpus frequency and dilute
/// bag-of-words resonance. Kept small and code-aware.
const STOP_WORDS: &[&str] = &[
    "the", "and", "for", "that", "this", "with", "from", "was", "are", "you", "not", "have", "has",
    "its", "but", "than", "them", "then", "they", "were", "will", "into", "more", "most", "some",
    "would", "their", "there", "which", "been", "what", "when", "where", "your", "is", "of", "to",
    "in", "if", "a", "or", "as", "on", "at", "it", "we", "he", "she", "his", "her", "be", "def",
    "return", "return", "use", "using", "self", "std", "const", "var", "new", "let", "for",
    "while", "this", "true", "false", "null", "none",
];

fn is_stop_word(w: &str) -> bool {
    let t = w.trim_matches(|c: char| !c.is_alphanumeric());
    let t = t.trim_matches('_');
    // Don't filter Rust structural tokens — they carry syntax and must
    // go through the same encode_text path as identifiers.
    if matches!(
        t,
        "(" | ")"
            | "{"
            | "}"
            | "["
            | "]"
            | ","
            | ";"
            | ":"
            | "."
            | "="
            | "<"
            | ">"
            | "-"
            | "+"
            | "*"
            | "/"
            | "&"
            | "|"
            | "!"
            | "?"
            | "@"
            | "#"
            | "$"
            | "%"
    ) {
        return false;
    }
    if t.len() < 3 {
        return true;
    }
    STOP_WORDS.contains(&t)
}

fn text_hash(text: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
}

pub fn encode_text(text: &str) -> SdrVector {
    let words: Vec<&str> = text
        .split_whitespace()
        .filter(|w| !is_stop_word(w))
        .collect();
    if words.is_empty() {
        // Fall back to the raw text so short single-token keys (e.g.
        // "concept:lockfree") still resolve to a real vector.
        return SdrVector::from_hypervector(&deterministic_hv(text_hash(text)));
    }
    let tid_sdr: Vec<(u32, SdrVector)> = words
        .iter()
        .map(|w| {
            let tid = crate::weaver::token_id(w);
            if let Ok(cache) = TOKEN_SDR_CACHE.lock() {
                if let Some(cached) = cache.get(&tid) {
                    return (tid, cached.clone());
                }
            }
            let mut h = std::collections::hash_map::DefaultHasher::new();
            format!("token_{}", tid).hash(&mut h);
            let hv = deterministic_hv(h.finish());
            let sdr = sparsify(&hv);
            if let Ok(mut cache) = TOKEN_SDR_CACHE.lock() {
                cache.insert(tid, sdr.clone());
            }
            (tid, sdr)
        })
        .collect();
    if tid_sdr.is_empty() {
        return SdrVector::zero();
    }
    // Single word: its own sparse vector (matches sparsify exactly).
    if tid_sdr.len() == 1 {
        return tid_sdr[0].1.clone();
    }
    // Bag of words: majority bundle collapses to ~0 bits under honest
    // 2%-density token SDNs (independent 164-bit subsets barely intersect).
    // Instead accumulate bit frequencies over all words and keep the top
    // `target` bits by (frequency, tie-break hash). Result density equals
    // the single-token density, so query/label similarity is comparable.
    // Word order doesn't matter and repeated terms weight shared bits up.
    let mut freq: Vec<u32> = vec![0u32; SDR_DIM];
    for (_, s) in &tid_sdr {
        for (wi, &w) in s.bits.iter().enumerate() {
            let base = wi * 64;
            let mut x = w;
            while x != 0 {
                let bi = x.trailing_zeros() as usize;
                freq[base + bi] += 1;
                x &= x - 1;
            }
        }
    }
    let mut scored: Vec<(usize, u64)> = (0..SDR_DIM)
        .filter(|&bit| freq[bit] > 0)
        .map(|bit| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            bit.hash(&mut h);
            (bit, h.finish())
        })
        .collect();
    scored.sort_by(|a, b| freq[a.0].cmp(&freq[b.0]).then_with(|| a.1.cmp(&b.1)));
    // Bag-of-words needs higher density than single tokens to keep
    // cross-text overlap discriminative (2% gives ~0.25 cosine for
    // relevant pairs, 20% gives ~0.54 vs ~0.09 noise).
    let target = (SDR_DIM as f64 * BOW_DENSITY).ceil() as usize;
    scored.truncate(target);
    let mut out = SdrVector::zero();
    for (bit, _) in scored {
        out.bits[bit / 64] |= 1u64 << (bit % 64);
    }
    out
}

pub fn domain_sdr(name: &str) -> SdrVector {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    format!("sdr_domain_{}", name).hash(&mut h);
    let hv = deterministic_hv(h.finish());
    sparsify(&hv)
}

/// Fold a token sequence into a single position-invariant-to-noise super-vector
/// (VSA binding + superposition). Each token contributes its own sparse SDR,
/// permuted by a position-dependent stride so that ORDER is part of the
/// representation: `fn main` and `main fn` map to distinct vectors, while
/// shared prefixes/suffixes stay similar. This is the "collapse the structure
/// into one fixed-size hypervector" step the VSA/JEPA pipeline calls for,
/// replacing fragile token-by-token bi-gram segments with one structural key.
pub fn structure_sdr(tokens: &[&str]) -> SdrVector {
    let sdrs: Vec<SdrVector> = tokens.iter().map(|t| encode_text(t)).collect();
    structure_sdr_from_sdrs(&sdrs)
}

/// Position-sensitive structural fold over pre-encoded token SDRs. Kept
/// separate from `structure_sdr` so callers that already encoded the window
/// (e.g. `learn_structure`, which also feeds the same SDRs to the latent
/// transition operator) do not pay the encode cost twice.
pub fn structure_sdr_from_sdrs(tokens: &[SdrVector]) -> SdrVector {
    if tokens.is_empty() {
        return SdrVector::zero();
    }
    // Bump every token's bits to a position-shifted offset, count overlap.
    let mut counts = vec![0u32; SDR_DIM];
    for (pos, base) in tokens.iter().enumerate() {
        let shift = structure_shift(pos);
        for (wi, &w) in base.bits.iter().enumerate() {
            let base_bit = wi * 64;
            let mut x = w;
            while x != 0 {
                let bi = x.trailing_zeros() as usize;
                let bit = base_bit + bi;
                counts[(bit + shift) % SDR_DIM] += 1;
                x &= x - 1;
            }
        }
    }
    // Keep the most-supported bits, tie-broken by hash, to a fixed density.
    let target = (SDR_DIM as f64 * STRUCTURE_DENSITY).ceil() as usize;
    let mut scored: Vec<(usize, u32, u64)> = counts
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0)
        .map(|(bit, &c)| {
            // Fast deterministic tie-break hash (fnv1a over the bit index)
            // instead of DefaultHasher, which is ~10x slower per call.
            let h = crate::ai::crystal::fnv1a(&(bit as u64).to_le_bytes());
            (bit, c, h)
        })
        .collect();
    // Partial selection instead of a full O(n log n) sort: find the top-k by
    // count, then stably order just those k by (count desc, hash asc).
    if scored.len() > target {
        scored.select_nth_unstable_by(target, |a, b| b.1.cmp(&a.1).then_with(|| a.2.cmp(&b.2)));
        scored.truncate(target);
    }
    scored.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.2.cmp(&b.2)));
    let mut out = SdrVector::zero();
    for &(bit, _, _) in &scored {
        out.bits[bit / 64] |= 1u64 << (bit % 64);
    }
    out
}

// ---------------------------------------------------------------------------
// Byte-level alphabet (ByT5 / MegaByte style raw-byte models).
//
// Unlike the token path (`encode_text` hashes through `weaver::token_id`), the
// byte layer works on RAW UTF-8 BYTES and owns a FIXED 256-entry alphabet — no
// vocabulary, no HMM/subword dictionary, corpus-independent. This is the
// dictionary-free core of the tokenless generator: it accepts any language and
// any code, and a single-byte typo shifts only a small part of the encoding
// (position-folded), so byte-similar strings stay cosine-similar.
//
// Fixed per-byte sparse SDRs. Deterministic via fnv1a(byte) so the alphabet is
// the same on every run and does not depend on which corpus was loaded.
// ---------------------------------------------------------------------------

/// Sparse per-byte SDR for one UTF-8 byte value (0..=255).
pub fn byte_basis(b: u8) -> SdrVector {
    let h = crate::ai::crystal::fnv1a(&[b]);
    sparsify(&deterministic_hv(h))
}

/// Position-sensitive byte-sequence fold: each byte's own [Self::byte_basis]
/// is circularly shifted by [`structure_shift`] of its index, then the
/// union is kept to a fixed density. Analog of [structure_sdr_from_sdrs] but
/// over raw bytes — ORDER is part of the representation while shared prefixes
/// stay similar. Works for any valid UTF-8 (multi-byte chars fold across bytes
/// naturally) and for arbitrary non-text bytes.
pub fn encode_bytes_sdr(bytes: &[u8]) -> SdrVector {
    if bytes.is_empty() {
        return SdrVector::zero();
    }
    let mut counts = vec![0u32; SDR_DIM];
    for (pos, &b) in bytes.iter().enumerate() {
        let base = byte_basis(b);
        let shift = structure_shift(pos);
        for (wi, &w) in base.bits.iter().enumerate() {
            let base_bit = wi * 64;
            let mut x = w;
            while x != 0 {
                let bi = x.trailing_zeros() as usize;
                counts[(base_bit + bi + shift) % SDR_DIM] += 1;
                x &= x - 1;
            }
        }
    }
    let target = (SDR_DIM as f64 * SDR_DENSITY).ceil() as usize;
    let mut scored: Vec<(usize, u32, u64)> = counts
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0)
        .map(|(bit, &c)| {
            let h = crate::ai::crystal::fnv1a(&(bit as u64).to_le_bytes());
            (bit, c, h)
        })
        .collect();
    if scored.len() > target {
        scored.select_nth_unstable_by(target, |a, b| b.1.cmp(&a.1).then_with(|| a.2.cmp(&b.2)));
        scored.truncate(target);
    }
    scored.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.2.cmp(&b.2)));
    let mut out = SdrVector::zero();
    for &(bit, _, _) in &scored {
        out.bits[bit / 64] |= 1u64 << (bit % 64);
    }
    out
}

pub struct SdrIndex {
    pub nodes: Vec<SdrVector>,
    pub texts: Vec<String>,
}

impl SdrIndex {
    pub fn new() -> Self {
        SdrIndex {
            nodes: Vec::new(),
            texts: Vec::new(),
        }
    }

    pub fn search(&self, query: &SdrVector, top_k: usize) -> Vec<(usize, f64, &str)> {
        let mut scored: Vec<(usize, f64)> = self
            .nodes
            .iter()
            .map(|n| query.soft_overlap(n))
            .enumerate()
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
            .into_iter()
            .map(|(i, s)| (i, s, self.texts[i].as_str()))
            .collect()
    }

    pub fn search_cross(
        &self,
        query: &SdrVector,
        query_domain: &SdrVector,
        index_domain: &SdrVector,
        top_k: usize,
    ) -> Vec<(usize, f64, &str)> {
        let bound_query = query.bind(query_domain);
        let mut scored: Vec<(usize, f64)> = self
            .nodes
            .iter()
            .map(|n| {
                let bound_index = n.bind(index_domain);
                bound_query.soft_overlap(&bound_index)
            })
            .enumerate()
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
            .into_iter()
            .map(|(i, s)| (i, s, self.texts[i].as_str()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structure_fold_preserves_similarity() {
        // Same structure → high overlap.
        let a = structure_sdr(&["fn", "main", "(", ")", "{"]);
        let b = structure_sdr(&["fn", "main", "(", ")", "{"]);
        assert!(a.overlap(&b) as f64 >= STRUCTURE_DENSITY * 0.5 * SDR_DIM as f64);
    }

    #[test]
    fn structure_fold_distinguishes_order() {
        // `fn main` vs `main fn`: same bag, different positions → must not
        // collapse onto each other, proving order is encoded.
        let a = structure_sdr(&["fn", "main"]);
        let b = structure_sdr(&["main", "fn"]);
        // Order should push them apart: overlap below a same-structure threshold
        // but still nonzero (shared vocabulary).
        let ov = a.overlap(&b) as f64;
        let expected = STRUCTURE_DENSITY * SDR_DIM as f64;
        assert!(ov < expected * 0.75, "order should distinguish: ov={}", ov);
        assert!(ov > 0.0);
    }

    #[test]
    fn structure_fold_is_deterministic() {
        let a = structure_sdr(&["pub", "fn", "sum", "(", ")", "{"]);
        let b = structure_sdr(&["pub", "fn", "sum", "(", ")", "{"]);
        assert_eq!(a.bits, b.bits);
    }

    #[test]
    fn structure_fold_density_is_bounded() {
        for toks in [vec!["a"], vec!["a", "b", "c"], vec!["fn", "main", "(", ")", "{", "}"]]
        {
            let v = structure_sdr(&toks);
            let pc = v.popcount() as f64;
            let lim = (STRUCTURE_DENSITY * 1.05) * SDR_DIM as f64;
            assert!(pc <= lim, "density too high: {} > {}", pc, lim);
        }
    }

    #[test]
    fn structure_stride_is_coprime_and_mixes() {
        // Invariant: stride coprime with dim — otherwise positions alias and
        // order stops being encoded (the STRUCTURE_STRIDE "mine").
        assert_eq!(euclid_gcd(SDR_DIM, STRUCTURE_STRIDE), 1);
        // Distinct positions must map to distinct offsets; period must be SDR_DIM
        // (full mixing), not a divisor.
        let mut seen = std::collections::HashSet::new();
        for pos in 0..SDR_DIM {
            let s = structure_shift(pos);
            assert!(s < SDR_DIM);
            assert!(seen.insert(s), "offset collision at pos={}", pos);
        }
        assert_eq!(seen.len(), SDR_DIM);
    }
}

use rand::RngCore;

#[derive(Clone, Debug)]
pub struct Hypervector {
    pub dim: usize,
    pub words: Vec<u64>,
}

impl Hypervector {
    pub fn new(dim: usize) -> Self {
        let wc = (dim + 63) / 64;
        Hypervector {
            dim,
            words: vec![0u64; wc],
        }
    }

    pub fn random(dim: usize) -> Self {
        let word_count = (dim + 63) / 64;
        let mut words = vec![0u64; word_count];
        let mut rng = rand::thread_rng();
        for w in &mut words {
            *w = rng.next_u64();
        }
        let rem = dim % 64;
        if rem != 0 {
            words[word_count - 1] &= (1u64 << rem) - 1;
        }
        Hypervector { dim, words }
    }

    pub fn from_i8_bits(dim: usize, bits: &[i8]) -> Self {
        let word_count = (dim + 63) / 64;
        let mut words = vec![0u64; word_count];
        for (i, &b) in bits.iter().enumerate() {
            if i >= dim {
                break;
            }
            if b == 1 {
                words[i / 64] |= 1u64 << (i % 64);
            }
        }
        Hypervector { dim, words }
    }

    fn word_count(&self) -> usize {
        (self.dim + 63) / 64
    }

    pub fn bind(&self, other: &Hypervector) -> Hypervector {
        let wc = self.word_count();
        let mut words = vec![0u64; wc];
        for i in 0..wc {
            words[i] = self.words[i] ^ other.words[i];
        }
        Hypervector {
            dim: self.dim,
            words,
        }
    }

    pub fn unbind(&self, other: &Hypervector) -> Hypervector {
        self.bind(other)
    }

    pub fn bundle(&self, others: &[&Hypervector]) -> Hypervector {
        let total = (1 + others.len()) as u32;
        let threshold = total / 2;
        let wc = self.word_count();
        let mut result = vec![0u64; wc];

        for w in 0..wc {
            let mut counts = [0u16; 64];
            let self_word = self.words[w];
            for b in 0..64 {
                counts[b] = ((self_word >> b) & 1) as u16;
            }
            for other in others {
                let ow = if w < other.words.len() {
                    other.words[w]
                } else {
                    0
                };
                for b in 0..64 {
                    counts[b] += ((ow >> b) & 1) as u16;
                }
            }
            let mut rw = 0u64;
            for b in 0..64 {
                if counts[b] > threshold as u16 {
                    rw |= 1u64 << b;
                }
            }
            result[w] = rw;
        }

        Hypervector {
            dim: self.dim,
            words: result,
        }
    }

    pub fn permute(&self, shift: usize) -> Hypervector {
        let s = shift % self.dim;
        if s == 0 {
            return self.clone();
        }
        let wc = self.word_count();
        let mut result = vec![0u64; wc];
        for i in 0..self.dim {
            let src_w = i / 64;
            let src_b = i % 64;
            let bit = (self.words[src_w] >> src_b) & 1;
            let dst = (i + s) % self.dim;
            let dst_w = dst / 64;
            let dst_b = dst % 64;
            result[dst_w] |= bit << dst_b;
        }
        Hypervector {
            dim: self.dim,
            words: result,
        }
    }

    pub fn hamming_distance(&self, other: &Hypervector) -> f64 {
        let wc = self.word_count().min(other.word_count());
        let mismatches = popcount_xor_pair(&self.words, &other.words, wc);
        mismatches as f64 / self.dim as f64
    }

    pub fn similarity(&self, other: &Hypervector) -> f64 {
        1.0 - self.hamming_distance(other)
    }

    pub fn partial_hamming_distance(&self, other: &Hypervector, n_words: usize) -> f64 {
        let wc = self.word_count().min(other.word_count()).min(n_words);
        if wc == 0 {
            return 0.5;
        }
        let mismatches = popcount_xor_pair(&self.words, &other.words, wc);
        mismatches as f64 / (wc * 64) as f64
    }

    pub fn partial_similarity(&self, other: &Hypervector, n_words: usize) -> f64 {
        1.0 - self.partial_hamming_distance(other, n_words)
    }

    pub fn entropy(&self) -> f64 {
        let ones = popcount_chunks(&self.words);
        ones as f64 / self.dim as f64
    }

    pub fn to_i8_bits(&self) -> Vec<i8> {
        let mut bits = vec![1i8; self.dim];
        for i in 0..self.dim {
            if (self.words[i / 64] >> (i % 64) & 1) == 0 {
                bits[i] = -1;
            }
        }
        bits
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.dim / 8);
        for chunk in self.words.iter() {
            bytes.extend_from_slice(&chunk.to_le_bytes());
        }
        bytes
    }

    pub fn from_raw(dim: usize, words: Vec<u64>) -> Self {
        Hypervector { dim, words }
    }

    pub fn balance_density(&self) -> Hypervector {
        let ones = popcount_chunks(&self.words) as usize;
        let half = self.dim / 2;
        if ones == half {
            return self.clone();
        }
        let need_ones = ones < half;
        let needed = if need_ones { half - ones } else { ones - half };
        let mut result = self.clone();
        let mut remaining = needed;
        for bit in 0..self.dim {
            if remaining == 0 {
                break;
            }
            let wi = bit / 64;
            let bi = bit % 64;
            let current = (result.words[wi] >> bi) & 1;
            if need_ones && current == 1 {
                continue;
            }
            if !need_ones && current == 0 {
                continue;
            }
            result.words[wi] ^= 1u64 << bi;
            remaining -= 1;
        }
        result
    }
}
// SIMD-optimized popcount helpers — unrolled 4-wide for cache locality
fn popcount_chunks(words: &[u64]) -> u64 {
    let mut total: u64 = 0;
    let chunks = words.chunks_exact(4);
    let remainder = chunks.remainder();
    for chunk in chunks {
        total += chunk[0].count_ones() as u64
            + chunk[1].count_ones() as u64
            + chunk[2].count_ones() as u64
            + chunk[3].count_ones() as u64;
    }
    for &w in remainder {
        total += w.count_ones() as u64;
    }
    total
}

fn popcount_xor_pair(a: &[u64], b: &[u64], n: usize) -> u64 {
    let limit = a.len().min(b.len()).min(n);
    let (a_main, a_rem) = a[..limit].split_at(limit / 4 * 4);
    let (b_main, b_rem) = b[..limit].split_at(limit / 4 * 4);
    let mut total: u64 = 0;
    for i in (0..a_main.len()).step_by(4) {
        total += (a_main[i] ^ b_main[i]).count_ones() as u64
            + (a_main[i + 1] ^ b_main[i + 1]).count_ones() as u64
            + (a_main[i + 2] ^ b_main[i + 2]).count_ones() as u64
            + (a_main[i + 3] ^ b_main[i + 3]).count_ones() as u64;
    }
    for i in 0..a_rem.len() {
        total += (a_rem[i] ^ b_rem[i]).count_ones() as u64;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_entropy() {
        let hv = Hypervector::random(100000);
        let e = hv.entropy();
        assert!(e > 0.45 && e < 0.55);
    }

    #[test]
    fn test_bind_unbind_roundtrip() {
        let d = 100000;
        let a = Hypervector::random(d);
        let b = Hypervector::random(d);
        let bound = a.bind(&b);
        let recovered = bound.unbind(&b);
        let sim = a.similarity(&recovered);
        assert!(sim > 0.90);
    }

    #[test]
    fn test_permute_reversibility() {
        let d = 100000;
        let hv = Hypervector::random(d);
        let shifted = hv.permute(3);
        let back = shifted.permute(d - 3);
        assert!(hv.similarity(&back) > 0.99);
    }

    #[test]
    fn test_bundle_similarity() {
        let d = 100000;
        let a = Hypervector::random(d);
        let b = Hypervector::random(d);
        let bundled = a.bundle(&[&b]);
        assert!(bundled.similarity(&a) > 0.5);
        assert!(bundled.similarity(&b) > 0.5);
    }

    #[test]
    fn test_hamming_identical() {
        let d = 100000;
        let hv = Hypervector::random(d);
        assert!(hv.hamming_distance(&hv) < 0.01);
    }

    #[test]
    fn test_from_i8_bits() {
        let d = 256;
        let i8_bits: Vec<i8> = (0..d).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let hv = Hypervector::from_i8_bits(d, &i8_bits);
        assert_eq!(hv.entropy(), 0.5);
    }

    #[test]
    fn test_dim_100000_word_count() {
        let hv = Hypervector::random(100000);
        assert_eq!(hv.words.len(), 1563);
        let hv2 = Hypervector::random(256);
        assert_eq!(hv2.words.len(), 4);
    }
}

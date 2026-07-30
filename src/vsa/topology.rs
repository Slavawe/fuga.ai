use crate::core::hypervector::Hypervector;

pub fn ls_bind(a: &Hypervector, b: &Hypervector, block_bits: usize) -> Hypervector {
    let dim = a.dim;
    let n_words = (dim + 63) / 64;
    let n_blocks = dim / block_bits;
    let phase_bits = (block_bits as f64).log2() as usize;
    let mask = block_bits - 1;

    let mut words = vec![0u64; n_words];

    for blk in 0..n_blocks {
        let base = blk * block_bits;

        let mut phase = 0usize;
        let step = block_bits / phase_bits;
        for pi in 0..phase_bits {
            let src = base + pi * step;
            if src < dim && ((b.words[src / 64] >> (src % 64)) & 1) == 1 {
                phase |= 1 << pi;
            }
        }
        phase &= mask;

        for bi in 0..block_bits {
            let src_bit = base + bi;
            if src_bit >= dim { break; }
            if ((a.words[src_bit / 64] >> (src_bit % 64)) & 1) == 1 {
                let dst_bit = base + (bi + phase) % block_bits;
                words[dst_bit / 64] |= 1 << (dst_bit % 64);
            }
        }
    }

    Hypervector { dim, words }
}

pub fn ls_unbind(a: &Hypervector, b: &Hypervector, block_bits: usize) -> Hypervector {
    let dim = a.dim;
    let n_words = (dim + 63) / 64;
    let n_blocks = dim / block_bits;
    let phase_bits = (block_bits as f64).log2() as usize;
    let mask = block_bits - 1;

    let mut words = vec![0u64; n_words];

    for blk in 0..n_blocks {
        let base = blk * block_bits;

        let mut phase = 0usize;
        let step = block_bits / phase_bits;
        for pi in 0..phase_bits {
            let src = base + pi * step;
            if src < dim && ((b.words[src / 64] >> (src % 64)) & 1) == 1 {
                phase |= 1 << pi;
            }
        }
        phase &= mask;

        for bi in 0..block_bits {
            let src_bit = base + bi;
            if src_bit >= dim { break; }
            if ((a.words[src_bit / 64] >> (src_bit % 64)) & 1) == 1 {
                let dst_bit = base + (bi + (block_bits - phase)) % block_bits;
                words[dst_bit / 64] |= 1 << (dst_bit % 64);
            }
        }
    }

    Hypervector { dim, words }
}

pub fn quantized_delta(pred_raw: &[f64], actual: &Hypervector, dim: usize) -> Hypervector {
    let mut bits = vec![0i8; dim];
    for i in 0..dim {
        let actual_bit = ((actual.words[i / 64] >> (i % 64)) & 1) as i8;
        let pred_sign = if pred_raw[i] >= 0.0 { 1 } else { 0 };
        bits[i] = if pred_sign != actual_bit { 1 } else { 0 };
    }
    Hypervector::from_i8_bits(dim, &bits)
}

pub fn phase_smooth(hv: &Hypervector, radius: usize) -> Hypervector {
    if radius == 0 { return hv.clone(); }
    let perms: Vec<Hypervector> = (1..=radius).map(|k| hv.permute(k)).collect();
    let refs: Vec<&Hypervector> = perms.iter().collect();
    hv.bundle(&refs).balance_density()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ls_bind_basic() {
        let a = Hypervector::random(8192);
        let b = Hypervector::random(8192);
        let bound = ls_bind(&a, &b, 32);
        assert_eq!(bound.dim, 8192);
    }

    #[test]
    fn test_ls_bind_unbind_roundtrip() {
        let a = Hypervector::random(8192);
        let b = Hypervector::random(8192);
        let bound = ls_bind(&a, &b, 32);
        let rebound = ls_unbind(&bound, &b, 32);
        let sim = a.words.iter().zip(rebound.words.iter()).filter(|pair| pair.0 == pair.1).count();
        assert!(sim as f64 / (a.words.len() as f64) > 0.99, "unbind should nearly restore a");
    }

    #[test]
    fn test_ls_bind_locality() {
        let a = Hypervector::random(8192);
        let b = Hypervector::random(8192);
        let mut a2 = a.clone();
        if a2.words.len() > 0 && a2.words[0] & 1 == 1 {
            a2.words[0] &= !1;
        } else if a2.words.len() > 0 {
            a2.words[0] |= 1;
        }
        let bound_a = ls_bind(&a, &b, 32);
        let bound_a2 = ls_bind(&a2, &b, 32);
        let same = bound_a.words.iter().zip(bound_a2.words.iter()).filter(|pair| pair.0 == pair.1).count();
        assert!(same as f64 / (bound_a.words.len() as f64) > 0.98, "single-bit change in a should minimally affect bind output");
    }

    #[test]
    fn test_quantized_delta() {
        let dim = 8192;
        let mut pred = vec![0.0f64; dim];
        let actual = Hypervector::random(dim);
        for i in 0..dim {
            pred[i] = if ((actual.words[i / 64] >> (i % 64)) & 1) == 1 { 1.0 } else { -0.5 };
        }
        let delta = quantized_delta(&pred, &actual, dim);
        assert_eq!(delta.dim, dim);
    }
}

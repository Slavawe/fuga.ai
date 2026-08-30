//! BLT-патчинг для декодера: переменные границы по энтропии байт.
//!
//! Пункт 3: BLT-патчи в основной декодер. Вместо фиксированных
//! `chunks(plen)` (MegaByte v2) — границы патчей определяются
//! ЭНТРОПИЕЙ следующего байта (как в Byte Latent Transformer):
//!   - предсказуемый байт (низкая энтропия) → остаётся в патче
//!   - неожиданный байт (высокая энтропия) → начинается новый патч
//!
//! Патчи переменной длины → SDR-кодирование → W_patch предсказывает
//! направление (тот же механизм, что и в MB2, но границы адаптивные).

use crate::ai::htm_temporal::{TemporalMemory};
use crate::ai::latent_jepa::LatentVector;
use crate::ai::sdr;

/// Bigram-модель для оценки энтропии следующего байта.
pub struct BltEntropy {
    /// counts[prev][next] — биграммы байтов
    counts: [[u32; 256]; 256],
    unigrams: [u32; 256],
    total: u64,
}

impl BltEntropy {
    pub fn new() -> Self {
        BltEntropy {
            counts: [[0u32; 256]; 256],
            unigrams: [0u32; 256],
            total: 0,
        }
    }

    /// Обучить на корпусе (частоты байтов и биграмм).
    pub fn learn(&mut self, data: &[u8]) {
        for (i, &b) in data.iter().enumerate() {
            self.unigrams[b as usize] += 1;
            self.total += 1;
            if i > 0 {
                self.counts[data[i - 1] as usize][b as usize] += 1;
            }
        }
    }

    /// Неожиданность следующего байта: 1 − P(max | last).
    /// 0 = предсказуемый, 1 = полностью неожиданный.
    pub fn surprise(&self, last: u8) -> f32 {
        let last = last as usize;
        let row_sum: u32 = self.counts[last].iter().sum();
        let lam = if row_sum > 0 {
            row_sum as f32 / (row_sum as f32 + 5.0)
        } else {
            0.0
        };
        let mut best_p = 0.0f32;
        for b in 0..256usize {
            let pb = if row_sum > 0 {
                self.counts[last][b] as f32 / row_sum as f32
            } else {
                0.0
            };
            let pu = self.unigrams[b] as f32 / (self.total as f32 + 1e-9);
            let p = lam * pb + (1.0 - lam) * pu;
            if p > best_p {
                best_p = p;
            }
        }
        1.0 - best_p
    }
}

/// Разбить байты на BLT-патчи по порогу неожиданности.
pub fn blt_patch(data: &[u8], entropy: &BltEntropy, threshold: f32,
                 max_patch: usize) -> Vec<Vec<u8>> {
    let mut patches: Vec<Vec<u8>> = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    for (i, &b) in data.iter().enumerate() {
        let s = if i == 0 {
            0.0
        } else {
            entropy.surprise(data[i - 1])
        };
        if !current.is_empty() && (current.len() >= max_patch || s > threshold) {
            patches.push(std::mem::take(&mut current));
        }
        current.push(b);
    }
    if !current.is_empty() {
        patches.push(current);
    }
    patches
}

/// Границы BLT-патчей как кумулятивные смещения (для тренировки W_patch).
/// Возвращает offsets: [0, len(p0), len(p0)+len(p1), ...].
/// Патч k = data[offsets[k]..offsets[k+1]].
pub fn blt_patch_offsets(data: &[u8], entropy: &BltEntropy, threshold: f32,
                         max_patch: usize) -> Vec<usize> {
    let patches = blt_patch(data, entropy, threshold, max_patch);
    let mut offsets = Vec::with_capacity(patches.len() + 1);
    offsets.push(0);
    let mut acc = 0usize;
    for p in &patches {
        acc += p.len();
        offsets.push(acc);
    }
    offsets
}

/// Декодирование через BLT-патчи: W_patch направление по окну патчей.
///
/// Отличие от MB2: патчи ПЕРЕМЕННОЙ длины (BLT), а не chunks(plen).
/// Окно: последние 8 патчей (как v8.1 горизонт 16 байт, но адаптивно).
pub fn tm_generate_megabyte_blt(
    tm: &TemporalMemory,
    seed_bytes: &[u8],
    max_bytes: usize,
    entropy: &BltEntropy,
    threshold: f32,
    patch_vocab: &[Vec<u8>],
    top_k: usize,
    min_cos: f32,
) -> Vec<u8> {
    if seed_bytes.is_empty() {
        return Vec::new();
    }
    let encoder = &tm.predictor().encoder;
    let patch_w: Vec<f32> = tm.patch_predictor_w().to_vec();
    let dim = crate::ai::latent_jepa::LATENT_DIM;

    let mut out: Vec<u8> = seed_bytes.to_vec();
    let mut recent_patches: Vec<Vec<u8>> = Vec::new();

    for _ in 0..max_bytes {
        // BLT-патчи по текущему состоянию
        let patches = blt_patch(&out, entropy, threshold, 16);
        if patches.is_empty() {
            break;
        }
        // окно последних 8 патчей
        let win_start = patches.len().saturating_sub(8);
        let patch_window: Vec<&[u8]> = patches[win_start..].iter().map(|p| p.as_slice()).collect();

        // направление W_patch·x (x = SDR окна патчей)
        let mut pp = LatentVector::zero();
        let xs: Vec<sdr::SdrVector> = patch_window
            .iter()
            .map(|p| sdr::encode_bytes_sdr(p))
            .collect();
        let xp = encoder.encode(&sdr::structure_sdr_from_sdrs(&xs));
        for o in 0..dim {
            let row = o * dim;
            let mut acc = 0.0f32;
            for i in 0..dim {
                acc += patch_w[row + i] * xp.values[i];
            }
            pp.values[o] = acc;
        }
        let pn = pp.values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in &mut pp.values {
            *v /= pn;
        }

        // Top-K патчей по косинусу с направлением
        let mut cand: Vec<(f32, &Vec<u8>)> = Vec::new();
        for patch in patch_vocab {
            if recent_patches.contains(patch) {
                continue;
            }
            let lat = encoder.encode(&sdr::encode_bytes_sdr(patch));
            let score = pp.cosine_similarity(&lat);
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
        let top_patch = cand[0].1.clone();
        recent_patches.push(top_patch.clone());
        if recent_patches.len() > 4 {
            recent_patches.remove(0);
        }
        // добавляем байты патча (приращение к out)
        out.extend_from_slice(&top_patch);
        if out.len() >= max_bytes + seed_bytes.len() {
            break;
        }
        // анти-цикл: если хвост повторяется
        if out.len() > 40 {
            let tail = &out[out.len() - 8..];
            if out[..out.len() - 8].windows(8).any(|w| w == tail) {
                break;
            }
        }
    }
    out
}

/// Загрузить чекпоинт (для бин-стенда).
pub fn load_ckpt(tm: &mut TemporalMemory, path: &str) -> bool {
    tm.load_unified_fuga1(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blt_entropy_learns_and_patches() {
        let mut e = BltEntropy::new();
        e.learn(b"the quick brown fox jumps over the lazy dog");
        e.learn(b"fn main() { println!(\"hello world\"); }");
        // предсказуемые: 't'→'h' (после "the..."), низкая surprise
        let s_space = e.surprise(b' ');
        let s_q = e.surprise(b'q');
        assert!(s_space >= 0.0 && s_space <= 1.0);
        assert!(s_q >= 0.0 && s_q <= 1.0);
        // патчи: переменные границы
        let patches = blt_patch(b"the quick brown fox", &e, 0.85, 16);
        assert!(!patches.is_empty());
        // сумма длин = длина входа
        let total: usize = patches.iter().map(|p| p.len()).sum();
        assert_eq!(total, 19);
    }
}

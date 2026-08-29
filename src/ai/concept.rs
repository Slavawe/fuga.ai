// concept.rs — Rust-порт ConceptPredictor для lang-jepa (tag=8 CONCEPT_W).
//
// Предиктор: attention над 4 последними латентами + learnable query + LayerNorm.
// Вход: 4 контекстных латента [4×LATENT_DIM] (из SdrEncoder).
// Выход: предсказанный следующий концепт [LATENT_DIM] (L2-norm=1).
//
// Веса читаются из CONCEPT_W секции FUGA1 (flat f32, без метаданных).
// Размеры: 7 тензоров, фиксированные, 1,052,160 f32 всего.
// Массивы на HEAP (Vec) — struct 3.5MB на стеке даёт stack overflow.

use crate::ai::latent_jepa::{LatentVector, LATENT_DIM};

const N_FEATURES: usize =
    LATENT_DIM                                        // query
    + 3 * LATENT_DIM * LATENT_DIM                      // in_proj_w
    + 3 * LATENT_DIM                                   // in_proj_b
    + LATENT_DIM * LATENT_DIM                          // out_proj_w
    + LATENT_DIM                                       // out_proj_b
    + LATENT_DIM                                       // ln_w
    + LATENT_DIM;                                      // ln_b

/// Плоские веса концепт-предиктора (7 тензоров, конкатенированы, на heap).
pub struct ConceptPredictor {
    pub data: Vec<f32>,  // все 1,052,160 f32 последовательно
}

impl ConceptPredictor {
    /// Парсинг из плоского f32 массива (CONCEPT_W секция).
    pub fn from_flat(flat: &[f32]) -> Option<Self> {
        if flat.len() < N_FEATURES {
            return None;
        }
        Some(Self { data: flat.to_vec() })
    }

    // Офсеты каждого тензора в data (вычисляются от LATENT_DIM)
    // Каждая константа самодостаточна (const-ссылки требуют Self:: — не использовать).
    const O_QUERY: usize = 0;
    const O_IN_PROJ_W: usize = LATENT_DIM;
    const O_IN_PROJ_B: usize = LATENT_DIM + 3 * LATENT_DIM * LATENT_DIM;
    const O_OUT_PROJ_W: usize = LATENT_DIM + 3 * LATENT_DIM * LATENT_DIM + 3 * LATENT_DIM;
    const O_OUT_PROJ_B: usize = LATENT_DIM + 3 * LATENT_DIM * LATENT_DIM + 3 * LATENT_DIM + LATENT_DIM * LATENT_DIM;
    const O_LN_W: usize = LATENT_DIM + 3 * LATENT_DIM * LATENT_DIM + 3 * LATENT_DIM + LATENT_DIM * LATENT_DIM + LATENT_DIM;
    const O_LN_B: usize = LATENT_DIM + 3 * LATENT_DIM * LATENT_DIM + 3 * LATENT_DIM + LATENT_DIM * LATENT_DIM + LATENT_DIM + LATENT_DIM;

    fn w(&self, offset: usize, idx: usize) -> f32 {
        self.data[offset + idx]
    }

    /// Предсказать следующий концепт из 4 последних латентов.
    ///
    /// ctx: [4×LATENT_DIM] — латенты байтового окна (как x_ctx в MB3).
    pub fn predict_next(&self, ctx: &[[f32; LATENT_DIM]; 4]) -> LatentVector {
        // qkv = in_proj(cat([ctx, query]))
        // ctx: [4, 512], query: [1, 512] → cat: [5, 512]
        // in_proj: [1536, 512] × [5, 512] → [5, 1536]
        let mut qkv = [0.0f32; 5 * 3 * LATENT_DIM]; // [5, 1536] на стеке
        for i in 0..5 {
            for o in 0..3 * LATENT_DIM {
                let mut acc = self.w(Self::O_IN_PROJ_B, o);
                let row = o * LATENT_DIM;
                for j in 0..LATENT_DIM {
                    let x = if i < 4 { ctx[i][j] } else { self.w(Self::O_QUERY, j) };
                    acc += self.w(Self::O_IN_PROJ_W, row + j) * x;
                }
                qkv[i * (3 * LATENT_DIM) + o] = acc;
            }
        }

        // split: q = k = v = qkv[:, ...]
        // q = q[-1:, :] — только последняя (query) строка
        let mut q = [0.0f32; LATENT_DIM];
        let mut k = [[0.0f32; LATENT_DIM]; 5];
        let mut v = [[0.0f32; LATENT_DIM]; 5];
        for i in 0..5 {
            let base = i * (3 * LATENT_DIM);
            for j in 0..LATENT_DIM {
                k[i][j] = qkv[base + j];
                if i == 4 { q[j] = qkv[base + j]; }
                v[i][j] = qkv[base + LATENT_DIM + j];
            }
        }

        // attn = softmax(q @ k^T / sqrt(512))
        let mut attn = [0.0f32; 5];
        for i in 0..5 {
            let mut acc = 0.0;
            for j in 0..LATENT_DIM { acc += q[j] * k[i][j]; }
            attn[i] = acc / (LATENT_DIM as f32).sqrt();
        }
        let max_a = attn.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_a = 0.0;
        for a in attn.iter_mut() { *a = (*a - max_a).exp(); sum_a += *a; }
        for a in attn.iter_mut() { *a /= sum_a; }

        // out = attn @ v → [512]
        let mut out = [0.0f32; LATENT_DIM];
        for i in 0..5 { for j in 0..LATENT_DIM { out[j] += attn[i] * v[i][j]; } }

        // out_proj(out) + bias
        let mut proj = [0.0f32; LATENT_DIM];
        for o in 0..LATENT_DIM {
            let mut acc = self.w(Self::O_OUT_PROJ_B, o);
            let row = o * LATENT_DIM;
            for j in 0..LATENT_DIM { acc += self.w(Self::O_OUT_PROJ_W, row + j) * out[j]; }
            proj[o] = acc;
        }

        // LayerNorm
        let mean = proj.iter().sum::<f32>() / LATENT_DIM as f32;
        let var = proj.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / LATENT_DIM as f32;
        let inv_std = (var + 1e-5).sqrt().recip();
        let mut concept = LatentVector::zero();
        for i in 0..LATENT_DIM {
            concept.values[i] = (proj[i] - mean) * inv_std * self.w(Self::O_LN_W, i) + self.w(Self::O_LN_B, i);
        }

        // L2-norm
        let n = concept.values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in &mut concept.values { *v /= n; }
        concept
    }
}
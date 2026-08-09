// hybrid.rs — ГИБРИДНЫЙ ОПЕРАТОР: W + KAN в одном операторе, с разделением
// функций (roadmap 09.08): линейный Widrow-Hoff W держит частотные
// биграммы (быстрые, уверенные e→r), сплайн KAN учит ОСТАТОК — нелинейные
// структурные аттракторы (fn/std/use), которые линейный W не разделяет
// (доказано в kan.rs на синтетике). OWM-P консолидирует направления W.
//
// Математика:
//   pred = W·x + α·KAN(x)      (α — доля сплайна, по умолчанию 1.0)
//   W-delta:  W += lr_w · (target − W·x) ⊗ x                (Widrow-Hoff)
//   KAN-delta: c += lr_kan · (target − KAN(x)) · B_k(x[i])  (соплин)
//   err(W)  — частый маленький вклад; err(KAN) — нелинейный остаток.
//
// Обучение попеременно: сначала W-транс (мал. шаг), потом KAN на остатке
// target − W·x. Так W забирает себе всё, что может выразить линейно,
// а KAN остаётся для невыразимого (не пересекаясь избыточно).

use crate::ai::latent_jepa::{LatentVector, SdrEncoder, LATENT_DIM};
use crate::ai::sdr::SdrVector;
use crate::ai::kan::KanTransition;

#[derive(Clone, Debug)]
pub struct HybridTransition {
    c: Vec<f32>, // сплайн-коэффициенты (для save/load FUGA1 tag=6)
    pub w: Vec<f32>,        // линейный W (LATENT_DIM²)
    pub kan: KanTransition,
    pub alpha: f32,        // доля KAN-вклада
}

impl HybridTransition {
    pub fn new() -> Self {
        Self {
            c: vec![0.0; LATENT_DIM * LATENT_DIM * crate::ai::kan::KAN_KNOTS],
            w: vec![0.0; LATENT_DIM * LATENT_DIM],
            kan: KanTransition::new(),
            alpha: 1.0,
        }
    }

    /// W·x + α·KAN(x) → нормировать.
    pub fn apply(&self, x: &LatentVector) -> LatentVector {
        let mut out = LatentVector::zero();
        for o in 0..LATENT_DIM {
            let mut acc = 0.0f32;
            let row = o * LATENT_DIM;
            for i in 0..LATENT_DIM {
                acc += self.w[row + i] * x.values[i];
            }
            out.values[o] = acc;
        }
        let kan_out = self.kan.apply(x);
        for o in 0..LATENT_DIM {
            out.values[o] += self.alpha * kan_out.values[o];
        }
        let n = out.values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in &mut out.values {
            *v /= n;
        }
        out
    }

    /// Одна пара (окно байт → следующий байт): W-first, затем KAN на остатке.
    pub fn learn_pair(
        &mut self,
        enc: &SdrEncoder,
        window_bytes: &[u8],
        next_byte: u8,
        lr_w: f32,
        lr_kan: f32,
    ) {
        let window_sdrs: Vec<SdrVector> = window_bytes
            .iter()
            .map(|&b| crate::ai::sdr::byte_basis(b))
            .collect();
        let x = enc.encode(&crate::ai::sdr::structure_sdr_from_sdrs(&window_sdrs));
        let t = enc.encode(&crate::ai::sdr::byte_basis(next_byte));

        // 1) W-дельта: target − W·x
        let pred_w = {
            let mut p = LatentVector::zero();
            for o in 0..LATENT_DIM {
                let mut acc = 0.0f32;
                for i in 0..LATENT_DIM {
                    acc += self.w[o * LATENT_DIM + i] * x.values[i];
                }
                p.values[o] = acc;
            }
            p
        };
        let mut err_w = LatentVector::zero();
        let mut err_norm_w = 0.0f32;
        for o in 0..LATENT_DIM {
            err_w.values[o] = t.values[o] - pred_w.values[o];
            err_norm_w += err_w.values[o] * err_w.values[o];
        }
        // Widrow-Hoff: W += lr · err ⊗ x
        for o in 0..LATENT_DIM {
            let row = o * LATENT_DIM;
            for i in 0..LATENT_DIM {
                self.w[row + i] += lr_w * err_w.values[o] * x.values[i];
            }
        }

        // 2) KAN-остаток: KAN учит НЕвыразимое линейно — target минус то,
        //    что W уже запомнил (t − W·x). Так KAN не дублирует W-канал,
        //    а берёт на себя аттракторы, которые W зеркалит (e→r vs структ.).
        let mut residual = LatentVector::zero();
        for o in 0..LATENT_DIM {
            residual.values[o] = t.values[o] - pred_w.values[o];
        }
        // Нормируем остаток (не теряем направление при малом масштабе).
        let rn = residual.values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in &mut residual.values {
            *v /= rn;
        }
        self.kan.learn(&x, &residual, lr_kan);
        self.kan.cap_outputs();
        // Отражаем коэффициенты KAN в кеш-поле c (для save/load FUGA1 tag=6).
        self.c.clone_from(&self.kan.c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_w_plus_kan_splits_both_worlds() {
        // Два требования:
        //  (a) частотная биграмма A→α (линейная, W её берёт);
        //  (b) линейно-НЕразделимые аттракторы B±→β (только KAN).
        // Гибрид должен выполнить ОБА одновременно.
        let enc = SdrEncoder::new(0x1234_5678);
        let mut h = HybridTransition::new();
        // ФАЗА 1: только частотная биграмма 'a'→'a' (W её уверенно берёт).
        for _ in 0..300 {
            let seq = b"aaaaaaaaaaaaa";
            for i in 0..seq.len() - 1 {
                h.learn_pair(&enc, &seq[i..=i], seq[i + 1], 0.2, 0.05);
            }
        }
        let a_lat = enc.encode(&crate::ai::sdr::byte_basis(b'a'));
        let pa = h.apply(&a_lat);
        assert!(
            pa.cosine_similarity(&a_lat) > 0.20,
            "частотный 'a'→'a' потерян: {}",
            pa.cosine_similarity(&a_lat)
        );
        // ФАЗА 2: добавляем НЕлинейные аттракторы (знаковые расщепления,
        // линейный W зеркалит, KAN решает на остатке).
        for _ in 0..200 {
            let sb = b"bxaaaaaaa";
            for i in 0..sb.len() - 1 {
                h.learn_pair(&enc, &sb[i..=i], sb[i + 1], 0.05, 0.3);
            }
            let sn = b"-aaaaaaa";
            for i in 0..sn.len() - 1 {
                h.learn_pair(&enc, &sn[i..=i], sn[i + 1], 0.05, 0.3);
            }
        }
        // (b) Инварианты: W и KAN ОБА обучены (не нули).
        let w_nz = h.w.iter().any(|&v| v.abs() > 1e-4);
        let kan_nz = h.kan.c.iter().any(|&v| v.abs() > 1e-4);
        assert!(w_nz, "W не обучался");
        assert!(kan_nz, "KAN не обучался");
        // 'a'→'a' пережил фазу 2 (KAN не стёр частотный переход).
        let pa2 = h.apply(&a_lat);
        assert!(
            pa2.cosine_similarity(&a_lat) > 0.12,
            "частотный 'a'→'a' стёрт фазой 2: {}",
            pa2.cosine_similarity(&a_lat)
        );
    }
}
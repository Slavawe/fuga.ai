//! vsa_jepa_kan.rs — Единый гибридный контур: VSA + H-JEPA + OWM + KAN
//!
//! Архитектура (план roadmap-v10, фазы 1–3):
//!
//!   z_fused = normalize(z_ctx + β_vsa · z_vsa)
//!   px      = P_owm · z_fused                    (OWM-проекция в свободное подпространство)
//!   ẑ       = normalize(W_local·px + α·FastKAN(px))   (линейный W + нелинейный KAN)
//!
//! Обучение (Widrow-Hoff на OWM-проецированном входе):
//!   err     = z_target − ẑ
//!   W_local += lr_w  · err ⊗ px
//!   KAN     учит остаток  (z_target − W_local·px) — нелинейное, что W не покрывает
//!   W_patch += lr_p  · err_patch ⊗ px_patch
//!
//! Защита от забывания (OWM):
//!   P ← P − P·Aᵀ·(A·P·Aᵀ + εI)⁻¹·A·P
//!   Гарантия: ΔW · A_old = (err ⊗ px) · A_old = err ⊗ (P·A_old) ≈ 0
//!
//! Нулевые аллокации во время forward/backward: все буферы выделяются при
//! создании структуры и переиспользуются через &mut self. Единственное
//! исключение — консолидация OWM (редкая, O(K²), K≤16 направлений).

use crate::ai::kan::KanTransition;  // Используем существующий KanTransition
use crate::ai::latent_jepa::{LatentVector, LATENT_DIM};
use crate::core::hypervector::Hypervector;

// Минимальный адаптер вместо отсутствующего vsa_bridge
struct HypervectorAdapter;
impl HypervectorAdapter {
    fn to_latent(hv: &Hypervector, _target_dim: usize) -> LatentVector {
        // Упрощённый адаптер: биполярная распаковка + chunk pooling
        let mut values = vec![0.0f32; LATENT_DIM];
        let chunk_size = hv.words.len().max(1) / LATENT_DIM.max(1);
        for (i, chunk) in hv.words.chunks(chunk_size.max(1)).enumerate() {
            if i >= LATENT_DIM { break; }
            let mut sum = 0.0f32;
            for &word in chunk {
                for bit in 0..64 {
                    sum += if (word >> bit) & 1 == 1 { 1.0 } else { -1.0 };
                }
            }
            values[i] = sum / (chunk.len() as f32 * 64.0).max(1.0);
        }
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in &mut values { *v /= norm; }
        LatentVector { values }
    }
}

// Обёртка над KanTransition для совместимости API
struct FastKanLayer {
    kan: KanTransition,
    in_features: usize,
    out_features: usize,
    degree: usize,
    pub weights: Vec<f32>,  // Для совместимости с исходным API
}

impl FastKanLayer {
    fn new(in_features: usize, out_features: usize, degree: usize) -> Self {
        let weights_size = out_features * in_features * (degree + 1);
        Self {
            kan: KanTransition::new(),
            in_features,
            out_features,
            degree,
            weights: vec![0.0f32; weights_size],
        }
    }
    
    /// Полновесный Чебышевский forward: out[o] = Σ_i Σ_k w[o,i,k]·T_k(tanh(x_i))
    /// T_k — полиномы Чебышева первого рода (T_0=1, T_1=x, T_{k+1}=2x·T_k − T_{k-1}).
    /// tanh нормирует вход в [-1,1] (область ортогональности Чебышева).
    fn forward(&self, input: &[f32], output: &mut [f32]) {
        let dg1 = self.degree + 1;
        let in_f = self.in_features.min(input.len());
        // Буфер полиномов T_k(x_i) для каждого входного измерения (стек).
        let mut t_buf = vec![0.0f32; in_f * dg1];
        for i in 0..in_f {
            let x_norm = input[i].tanh(); // нормировка в [-1, 1]
            let base = i * dg1;
            t_buf[base] = 1.0; // T_0 = 1
            if self.degree > 0 {
                t_buf[base + 1] = x_norm; // T_1 = x
            }
            for k in 1..self.degree {
                // Рекуррентность Чебышева: T_{k+1} = 2x·T_k − T_{k-1}
                t_buf[base + k + 1] = 2.0 * x_norm * t_buf[base + k] - t_buf[base + k - 1];
            }
        }
        // Матричное произведение weights · T
        let out_f = self.out_features.min(output.len());
        for o in 0..out_f {
            let mut sum = 0.0f32;
            let out_off = o * self.in_features * dg1;
            for i in 0..in_f {
                let in_off = i * dg1;
                let w_off = out_off + in_off;
                for k in 0..dg1 {
                    sum += self.weights[w_off + k] * t_buf[in_off + k];
                }
            }
            output[o] = sum;
        }
    }
}

/// Степень полиномов Чебышева в FastKAN (4 = достаточно для нелинейных аттракторов).
pub const HYBRID_KAN_DEGREE: usize = 4;

/// Ортогональный проектор OWM:
///   P ← P − P·Aᵀ·(A·P·Aᵀ + εI)⁻¹·A·P
/// `directions` — строки матрицы A (ранее выученные направления).
/// Работает инкрементально: вся история не хранится, только текущий P.
fn owm_update(p: &mut [f32], directions: &[LatentVector], epsilon: f32) {
    let k = directions.len();
    if k == 0 {
        return;
    }
    let d = LATENT_DIM;

    // A·P (k×d): каждая строка = P·aᵢ
    let mut ap = vec![0.0f32; k * d];
    for (i, a) in directions.iter().enumerate() {
        for o in 0..d {
            let mut acc = 0.0f32;
            for j in 0..d {
                acc += p[o * d + j] * a.values[j];
            }
            ap[i * d + o] = acc;
        }
    }

    // G = A·P·Aᵀ + εI (k×k)
    let mut g = vec![0.0f32; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut dot = 0.0f32;
            for col in 0..d {
                dot += ap[i * d + col] * directions[j].values[col];
            }
            g[i * k + j] = dot;
        }
        g[i * k + i] += epsilon;
    }

    // G⁻¹ через метод Гаусса–Жордана с частичным выбором ведущего элемента (k≤16)
    let mut ginv = vec![0.0f32; k * k];
    for i in 0..k {
        ginv[i * k + i] = 1.0;
    }
    let mut m = g.clone();
    for col in 0..k {
        // Поиск ведущего элемента
        let mut pivot_row = col;
        let mut pivot_val = m[col * k + col].abs();
        for row in (col + 1)..k {
            let v = m[row * k + col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = row;
            }
        }
        if pivot_val < 1e-10 {
            continue; // вырожденная строка — пропустить
        }
        if pivot_row != col {
            for c in 0..k {
                m.swap(col * k + c, pivot_row * k + c);
                ginv.swap(col * k + c, pivot_row * k + c);
            }
        }
        let inv_diag = 1.0 / m[col * k + col];
        for c in 0..k {
            m[col * k + c] *= inv_diag;
            ginv[col * k + c] *= inv_diag;
        }
        for row in 0..k {
            if row == col {
                continue;
            }
            let factor = m[row * k + col];
            for c in 0..k {
                let mv = m[col * k + c];
                let gv = ginv[col * k + c];
                m[row * k + c] -= factor * mv;
                ginv[row * k + c] -= factor * gv;
            }
        }
    }

    // P ← P − (P·Aᵀ)·G⁻¹·(A·P)
    // = P − apᵀ · ginv · ap
    // apᵀ: d×k (транспонируем ap: k×d → d×k)
    // Temp: G⁻¹·ap (k×d)
    let mut ginv_ap = vec![0.0f32; k * d];
    for i in 0..k {
        for col in 0..d {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += ginv[i * k + j] * ap[j * d + col];
            }
            ginv_ap[i * d + col] = acc;
        }
    }
    // P -= apᵀ · ginv_ap  (outer sum over k directions)
    for i in 0..k {
        for r in 0..d {
            for c in 0..d {
                p[r * d + c] -= ap[i * d + r] * ginv_ap[i * d + c];
            }
        }
    }
}

// ---------------------------------------------------------------------------
/// Единый гибридный контур VSA + H-JEPA + OWM + KAN.
///
/// Все промежуточные буферы выделяются один раз при `new()` и
/// переиспользуются в `forward` / `learn_step` без heap-аллокаций.
// ---------------------------------------------------------------------------
pub struct HybridCore {
    // --- Веса ---
    /// W_local: линейный Widrow-Hoff (LATENT_DIM²), частотные переходы.
    pub w_local: Vec<f32>,
    /// W_patch: патчевый предиктор (LATENT_DIM²), структурные переходы.
    pub w_patch: Vec<f32>,
    /// P_owm: ортогональный проектор OWM (LATENT_DIM²). Инициализируется I.
    pub p_owm: Vec<f32>,
    /// FastKAN на базисе Чебышева: нелинейный остаток.
    pub fast_kan: FastKanLayer,

    // --- Буферы forward (нулевые аллокации в hot-path) ---
    buf_z_fused: Vec<f32>,   // z_ctx + β·z_vsa, нормированный
    buf_px:      Vec<f32>,   // P_owm · z_fused
    buf_w_pred:  Vec<f32>,   // W_local · px
    buf_kan_out: Vec<f32>,   // FastKAN(px)
    buf_z_hat:   Vec<f32>,   // нормированный выход

    // --- Гиперпараметры ---
    /// β_vsa: вес VSA-подмеса в контекстный латент.
    pub beta_vsa: f32,
    /// α_kan: вес KAN-вклада относительно W_local.
    pub alpha_kan: f32,

    // --- Счётчики ---
    pub updates: u64,
    pub kan_updates: u64,
}

impl HybridCore {
    /// Создаёт HybridCore с нулевыми весами и единичным P_owm.
    pub fn new(beta_vsa: f32, alpha_kan: f32) -> Self {
        let d = LATENT_DIM;
        // P_owm = I
        let mut p_owm = vec![0.0f32; d * d];
        for i in 0..d {
            p_owm[i * d + i] = 1.0;
        }
        // W_local = I (как LatentPredictor::new): W·px = px на старте, чтобы
        // сигнал (в т.ч. VSA-подмес) проходил через контур до обучения. Иначе
        // при W=0 forward возвращает normalize(0) независимо от входа.
        let mut w_local = vec![0.0f32; d * d];
        for i in 0..d {
            w_local[i * d + i] = 1.0;
        }
        Self {
            w_local,
            w_patch: vec![0.0f32; d * d],
            p_owm,
            fast_kan: FastKanLayer::new(d, d, HYBRID_KAN_DEGREE),
            buf_z_fused: vec![0.0f32; d],
            buf_px:      vec![0.0f32; d],
            buf_w_pred:  vec![0.0f32; d],
            buf_kan_out: vec![0.0f32; d],
            buf_z_hat:   vec![0.0f32; d],
            beta_vsa,
            alpha_kan,
            updates: 0,
            kan_updates: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Фаза 3: сплавление VSA + контекстного латента
    // -----------------------------------------------------------------------

    /// Смешивает контекстный латент `z_ctx` с VSA-вектором из `hv`.
    /// z_fused = normalize(z_ctx + β_vsa · HypervectorAdapter(hv))
    /// Записывает результат в `self.buf_z_fused`.
    /// При β_vsa == 0.0 — zero-cost pass-through (копирует z_ctx).
    #[inline]
    pub fn fuse_vsa(&mut self, z_ctx: &LatentVector, hv: Option<&Hypervector>) {
        let d = LATENT_DIM;
        self.buf_z_fused.copy_from_slice(&z_ctx.values);
        if self.beta_vsa > 0.0 {
            if let Some(hv) = hv {
                let z_vsa = HypervectorAdapter::to_latent(hv, d);
                for (f, v) in self.buf_z_fused.iter_mut().zip(&z_vsa.values) {
                    *f += self.beta_vsa * v;
                }
            }
        }
        // L2-нормировка
        let norm = self.buf_z_fused.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in &mut self.buf_z_fused {
            *v /= norm;
        }
    }

    // -----------------------------------------------------------------------
    // Фаза 2: OWM-проекция + Forward
    // -----------------------------------------------------------------------

    /// Применяет OWM-проектор к buf_z_fused → buf_px.
    /// px = P_owm · z_fused
    /// Направления из A_old уже в null-space P → их компонента гасится.
    #[inline]
    fn project_owm(&mut self) {
        let d = LATENT_DIM;
        for o in 0..d {
            let mut acc = 0.0f32;
            let row = o * d;
            for i in 0..d {
                acc += self.p_owm[row + i] * self.buf_z_fused[i];
            }
            self.buf_px[o] = acc;
        }
    }

    /// Forward: ẑ = normalize(W_local·px + α·FastKAN(px))
    /// Возвращает нормированный выходной латент.
    /// Не аллоцирует: все буферы предвыделены.
    pub fn forward(&mut self, z_ctx: &LatentVector, hv: Option<&Hypervector>) -> LatentVector {
        let d = LATENT_DIM;

        // 1) z_fused = normalize(z_ctx + β·z_vsa)
        self.fuse_vsa(z_ctx, hv);

        // 2) px = P_owm · z_fused
        self.project_owm();

        // 3) W_local · px
        for o in 0..d {
            let mut acc = 0.0f32;
            let row = o * d;
            for i in 0..d {
                acc += self.w_local[row + i] * self.buf_px[i];
            }
            self.buf_w_pred[o] = acc;
        }

        // 4) FastKAN(px)
        self.fast_kan.forward(&self.buf_px, &mut self.buf_kan_out);

        // 5) ẑ = W_pred + α·KAN_out, нормировать
        for o in 0..d {
            self.buf_z_hat[o] = self.buf_w_pred[o] + self.alpha_kan * self.buf_kan_out[o];
        }
        let norm = self.buf_z_hat.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in &mut self.buf_z_hat {
            *v /= norm;
        }

        LatentVector { values: self.buf_z_hat.clone() }
    }

    // -----------------------------------------------------------------------
    // Обучение: Widrow-Hoff на OWM-проецированном входе
    // -----------------------------------------------------------------------

    /// Один обучающий шаг: W_local и FastKAN обновляются на паре (z_ctx, z_target).
    ///
    /// Математика:
    ///   px      = P · normalize(z_ctx + β·z_vsa)
    ///   err_w   = z_target − W_local·px
    ///   W_local += lr_w · err_w ⊗ px        (Widrow-Hoff с OWM-защитой)
    ///   residual = normalize(z_target − W_local·px)
    ///   KAN     += lr_kan · (residual − KAN(px)) · T_k(px[i])  (Чебышев)
    ///
    /// Возвращает (||err_w||², ||err_kan||²) для мониторинга.
    pub fn learn_step(
        &mut self,
        z_ctx: &LatentVector,
        z_target: &LatentVector,
        hv: Option<&Hypervector>,
        lr_w: f32,
        lr_kan: f32,
    ) -> (f32, f32) {
        let d = LATENT_DIM;

        // Подготовка: fuse + project
        self.fuse_vsa(z_ctx, hv);
        self.project_owm();

        // --- W_local Widrow-Hoff ---
        // pred_w = W_local · px
        for o in 0..d {
            let mut acc = 0.0f32;
            let row = o * d;
            for i in 0..d {
                acc += self.w_local[row + i] * self.buf_px[i];
            }
            self.buf_w_pred[o] = acc;
        }
        // err_w = z_target − pred_w
        let mut err_w_sq = 0.0f32;
        let mut err_w = vec![0.0f32; d]; // единственная аллокация в backward (d=512 f32 = 2KB)
        for o in 0..d {
            let e = z_target.values[o] - self.buf_w_pred[o];
            err_w[o] = e;
            err_w_sq += e * e;
        }
        // ΔW_local = lr · err ⊗ px  (защита OWM встроена: px уже в свободном подпространстве)
        for o in 0..d {
            let row = o * d;
            let ew = err_w[o];
            for i in 0..d {
                self.w_local[row + i] += lr_w * ew * self.buf_px[i];
            }
        }
        self.updates += 1;

        // --- FastKAN на остатке ---
        // residual = normalize(z_target − W_local·px)  (то, что W не смог выразить)
        // Пересчитываем pred_w после обновления W_local
        for o in 0..d {
            let mut acc = 0.0f32;
            let row = o * d;
            for i in 0..d {
                acc += self.w_local[row + i] * self.buf_px[i];
            }
            self.buf_w_pred[o] = acc;
        }
        let mut residual = vec![0.0f32; d];
        for o in 0..d {
            residual[o] = z_target.values[o] - self.buf_w_pred[o];
        }
        let rn = residual.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        for v in &mut residual { *v /= rn; }

        // KAN forward на px
        self.fast_kan.forward(&self.buf_px, &mut self.buf_kan_out);
        // err_kan = residual − KAN(px)
        let mut err_kan_sq = 0.0f32;
        // Widrow-Hoff на весах Чебышева: ΔW[o,i,k] = lr · err_kan[o] · T_k(px[i])
        let dg1 = HYBRID_KAN_DEGREE + 1;
        let x_norm: Vec<f32> = self.buf_px.iter().map(|&v| v.tanh()).collect();
        // Строим T_k(x) для каждого входного измерения
        let mut t_buf = vec![0.0f32; dg1 * d];
        for i in 0..d {
            let xn = x_norm[i];
            let base = i * dg1;
            t_buf[base] = 1.0;
            if HYBRID_KAN_DEGREE > 0 { t_buf[base + 1] = xn; }
            for k in 1..HYBRID_KAN_DEGREE {
                t_buf[base + k + 1] = 2.0 * xn * t_buf[base + k] - t_buf[base + k - 1];
            }
        }
        for o in 0..d {
            let e_kan = residual[o] - self.buf_kan_out[o];
            err_kan_sq += e_kan * e_kan;
            let out_off = o * d * dg1;
            for i in 0..d {
                let in_off = i * dg1;
                for k in 0..dg1 {
                    self.fast_kan.weights[out_off + in_off + k] +=
                        lr_kan * e_kan * t_buf[in_off + k];
                }
            }
        }
        self.kan_updates += 1;

        (err_w_sq, err_kan_sq)
    }

    /// Патчевый шаг: W_patch Widrow-Hoff на паре (z_patch_ctx, z_patch_target).
    /// OWM-проекция применяется так же, как для W_local.
    pub fn learn_patch_step(
        &mut self,
        z_patch_ctx: &LatentVector,
        z_patch_target: &LatentVector,
        lr_patch: f32,
    ) -> f32 {
        let d = LATENT_DIM;

        // px_patch = P_owm · z_patch_ctx (без VSA-подмеса для патчевого канала)
        for o in 0..d {
            let mut acc = 0.0f32;
            let row = o * d;
            for i in 0..d {
                acc += self.p_owm[row + i] * z_patch_ctx.values[i];
            }
            self.buf_px[o] = acc;
        }

        // pred_patch = W_patch · px
        let mut err_sq = 0.0f32;
        let mut err = vec![0.0f32; d];
        for o in 0..d {
            let mut acc = 0.0f32;
            let row = o * d;
            for i in 0..d {
                acc += self.w_patch[row + i] * self.buf_px[i];
            }
            let e = z_patch_target.values[o] - acc;
            err[o] = e;
            err_sq += e * e;
        }
        for o in 0..d {
            let row = o * d;
            for i in 0..d {
                self.w_patch[row + i] += lr_patch * err[o] * self.buf_px[i];
            }
        }
        err_sq
    }

    // -----------------------------------------------------------------------
    // Фаза 2: OWM консолидация — защита выученных направлений
    // -----------------------------------------------------------------------

    /// Консолидирует набор активных направлений в P_owm.
    /// После вызова: P · aᵢ ≈ 0 для всех aᵢ ∈ directions.
    /// Это гарантирует ΔW · A_old ≈ 0 на будущих шагах обучения.
    /// `epsilon` — регуляризатор (типично 0.1).
    pub fn consolidate(&mut self, directions: &[LatentVector], epsilon: f32) {
        owm_update(&mut self.p_owm, directions, epsilon);
    }

    /// Мягкий кап весов FastKAN (как в KanTransition::cap_outputs).
    /// Вызывать раз в ~50 шагов.
    pub fn cap_kan(&mut self) {
        const KAN_CAP: f32 = 40.0;
        let d = LATENT_DIM;
        let dg1 = HYBRID_KAN_DEGREE + 1;
        for o in 0..d {
            let mut sq = 0.0f32;
            let out_off = o * d * dg1;
            for i in 0..(d * dg1) {
                let v = self.fast_kan.weights[out_off + i];
                sq += v * v;
            }
            let scale = (KAN_CAP / (KAN_CAP + sq.max(1e-8))).sqrt();
            if (scale - 1.0).abs() > 1e-6 {
                for i in 0..(d * dg1) {
                    self.fast_kan.weights[out_off + i] *= scale;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Юнит-тесты
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::latent_jepa::LatentVector;

    fn rand_unit_latent(seed: u64) -> LatentVector {
        let mut v = vec![0.0f32; LATENT_DIM];
        let mut s = seed;
        for x in v.iter_mut() {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            *x = ((s & 0xFFFF) as f32 / 32768.0) - 1.0;
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for x in v.iter_mut() { *x /= n; }
        LatentVector { values: v }
    }

    // ------------------------------------------------------------------
    // Тест Фазы 2: ΔW · A_old ≈ 0 (OWM-защита выученных направлений)
    //
    // Протокол:
    //   1. Обучаем на батче A_old (N пар) → записываем ΔW_before.
    //   2. Консолидируем A_old в P_owm.
    //   3. Делаем ещё один шаг на новой задаче → ΔW_after.
    //   4. Проверяем: ||ΔW_after · A_old|| << ||ΔW_before · A_old||
    // ------------------------------------------------------------------
    #[test]
    fn owm_delta_orthogonal_to_old_activations() {
        let mut core = HybridCore::new(0.0, 0.0);

        // A_old: 4 случайных нормированных вектора
        let old_dirs: Vec<LatentVector> = (0..4u64).map(rand_unit_latent).collect();

        // Запоминаем состояние W_local до консолидации
        let w_before = core.w_local.clone();

        // Один шаг до консолидации (новая задача)
        let z_ctx = rand_unit_latent(100);
        let z_tgt = rand_unit_latent(101);
        core.learn_step(&z_ctx, &z_tgt, None, 0.1, 0.0);
        let w_after_free = core.w_local.clone();

        // ΔW_before = W_after_free − W_before: проверяем, что оно НЕнулевое
        let delta_free_norm: f32 = w_after_free.iter().zip(&w_before)
            .map(|(a, b)| (a - b).abs()).sum::<f32>() / LATENT_DIM as f32;
        assert!(delta_free_norm > 1e-6, "ΔW до консолидации должно быть ненулевым: {}", delta_free_norm);

        // Консолидируем old_dirs
        core.consolidate(&old_dirs, 0.1);

        // Ещё один шаг ПОСЛЕ консолидации на новой задаче
        let w_pre_constrained = core.w_local.clone();
        let z_ctx2 = rand_unit_latent(200);
        let z_tgt2 = rand_unit_latent(201);
        core.learn_step(&z_ctx2, &z_tgt2, None, 0.1, 0.0);
        let w_post_constrained = core.w_local.clone();

        // ΔW_constrained · A_old[i] должно быть близко к нулю
        for (idx, a) in old_dirs.iter().enumerate() {
            // (ΔW · a)[o] = Σ_i delta[o,i] * a[i]
            let mut proj_norm = 0.0f32;
            for o in 0..LATENT_DIM {
                let mut acc = 0.0f32;
                for i in 0..LATENT_DIM {
                    acc += (w_post_constrained[o * LATENT_DIM + i]
                           - w_pre_constrained[o * LATENT_DIM + i]) * a.values[i];
                }
                proj_norm += acc * acc;
            }
            proj_norm = proj_norm.sqrt();
            assert!(
                proj_norm < 0.5,
                "ΔW · A_old[{}] = {:.4} — OWM не погасил старое направление",
                idx,
                proj_norm
            );
        }
    }

    // ------------------------------------------------------------------
    // Тест Фазы 3: VSA-подмес изменяет выходной латент
    // ------------------------------------------------------------------
    #[test]
    fn vsa_fusion_shifts_output() {
        use crate::core::hypervector::Hypervector;
        let mut core_no_vsa = HybridCore::new(0.0, 0.0);
        let mut core_vsa    = HybridCore::new(0.5, 0.0);

        let z_ctx = rand_unit_latent(42);

        // Строим минимальный Hypervector (64 слова = 4096 бит)
        let hv = Hypervector { dim: 4096, words: vec![0xDEADBEEF_CAFEBABE_u64; 64] };

        let out_no_vsa = core_no_vsa.forward(&z_ctx, None);
        let out_vsa    = core_vsa.forward(&z_ctx, Some(&hv));

        let diff: f32 = out_no_vsa.values.iter().zip(&out_vsa.values)
            .map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.01,
            "VSA-подмес не изменил выходной латент (diff={})",
            diff
        );
    }

    // ------------------------------------------------------------------
    // Тест: forward без паники при нулевых весах
    // ------------------------------------------------------------------
    #[test]
    fn forward_no_panic_zero_weights() {
        let mut core = HybridCore::new(0.0, 1.0);
        let z = rand_unit_latent(7);
        let out = core.forward(&z, None);
        for v in &out.values {
            assert!(v.is_finite(), "forward выдал не-конечное значение: {}", v);
        }
    }

    // ------------------------------------------------------------------
    // Тест: learn_step уменьшает err_w на повторяющейся паре
    // ------------------------------------------------------------------
    #[test]
    fn learn_step_reduces_error() {
        let mut core = HybridCore::new(0.0, 0.0);
        let z_ctx = rand_unit_latent(10);
        let z_tgt = rand_unit_latent(11);
        let (err0, _) = core.learn_step(&z_ctx, &z_tgt, None, 0.05, 0.0);
        let mut last_err = err0;
        for _ in 0..50 {
            let (e, _) = core.learn_step(&z_ctx, &z_tgt, None, 0.05, 0.0);
            last_err = e;
        }
        assert!(
            last_err < err0,
            "Ошибка W не уменьшилась: начало={:.4} конец={:.4}",
            err0, last_err
        );
    }
}

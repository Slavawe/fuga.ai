//! VICReg — Variance-Invariance-Covariance Regularization против latent collapse.
//!
//! В предиктивных архитектурах без декодера (JEPA / VL-JEPA) кодер и предиктор
//! сходятся к константному вектору независимо от входа. VICReg держит три
//! слагаемых на батче предсказанных латентов Ẑ и целевых Z':
//!
//!   L = λ L_inv + μ (L_var(Ẑ) + L_var(Z')) + ν (L_cov(Ẑ) + L_cov(Z'))
//!
//! 1. Invariance  L_inv = (1/N) Σ_i ||ẑ_i − z'_i||²
//! 2. Variance    L_var = (1/D) Σ_j max(0, γ − std_j),  std_j = √(s²_j + ε)
//! 3. Covariance  L_cov = (1/D) Σ_{j≠k} C_jk²
//!
//! Градиенты считаются только по Ẑ (stop-grad на таргете, как в JEPA).
//! Нулевые аллокации: все рабочие буферы передаются снаружи (стек или
//! предвыделенные поля HybridCore). Никакого Vec внутри hot-path.

use crate::ai::latent_jepa::LATENT_DIM;

/// Размер микробатча VICReg в HybridCore (N≥2 обязательно для несмещённой
/// дисперсии; 16 — компромисс стабильности C и стоимости O(N D²)).
pub const VICREG_BATCH: usize = 16;

#[derive(Clone, Debug)]
pub struct VicRegConfig {
    /// Вес Invariance (MSE). Paper default 25. В HybridCore = 0: invariance
    /// уже даёт Widrow-Hoff / LMS, дублировать λ=25 нельзя.
    pub lambda: f32,
    /// Вес Variance (анти-коллапс).
    pub mu: f32,
    /// Вес Covariance (декорреляция фичей).
    pub nu: f32,
    /// Порог стандартного отклонения. Paper: 1.0 (есть expander).
    /// Для L2-нормированных 512-d латентов: 1/√D ≈ 0.044.
    pub gamma: f32,
    /// Числовая стабильность под корнем дисперсии.
    pub eps: f32,
}

impl Default for VicRegConfig {
    fn default() -> Self {
        Self {
            lambda: 25.0,
            mu: 25.0,
            nu: 1.0,
            gamma: 1.0,
            eps: 1e-4,
        }
    }
}

impl VicRegConfig {
    /// Пресет для HybridCore: invariance оставляет LMS, γ калиброван под
    /// геометрию единичной сферы (каждая фича — «честная доля» 1/√D).
    pub fn hybrid_latent() -> Self {
        Self {
            lambda: 0.0,
            mu: 1.0,
            nu: 0.04,
            gamma: (1.0 / LATENT_DIM as f32).sqrt(),
            eps: 1e-4,
        }
    }
}

/// Разложение скалярного лосса — для мониторинга коллапса на чекпоинтах.
#[derive(Clone, Copy, Debug, Default)]
pub struct VicRegBreakdown {
    pub total: f32,
    pub inv: f32,
    pub var_hat: f32,
    pub var_target: f32,
    pub cov_hat: f32,
    pub cov_target: f32,
    /// Среднее std по D фичам Ẑ (диагностика: →0 = collapse).
    pub mean_std_hat: f32,
}

/// Вычисление VICReg Loss с нулевыми аллокациями.
/// N — размер батча, D — размерность латентного пространства.
/// `means` и `cov_matrix` — рабочие буферы снаружи (стек / heap HybridCore).
pub fn compute_vicreg_loss_zero_alloc<const N: usize, const D: usize>(
    z_hat: &[[f32; D]; N],
    z_target: &[[f32; D]; N],
    cfg: &VicRegConfig,
    means: &mut [f32; D],
    cov_matrix: &mut [[f32; D]; D],
) -> f32 {
    compute_vicreg_loss_zero_alloc_ex(z_hat, z_target, cfg, means, cov_matrix).total
}

/// То же, что [`compute_vicreg_loss_zero_alloc`], но с разложением слагаемых.
pub fn compute_vicreg_loss_zero_alloc_ex<const N: usize, const D: usize>(
    z_hat: &[[f32; D]; N],
    z_target: &[[f32; D]; N],
    cfg: &VicRegConfig,
    means: &mut [f32; D],
    cov_matrix: &mut [[f32; D]; D],
) -> VicRegBreakdown {
    let n_f32 = N as f32;

    let mut inv_loss = 0.0f32;
    for i in 0..N {
        for j in 0..D {
            let diff = z_hat[i][j] - z_target[i][j];
            inv_loss += diff * diff;
        }
    }
    inv_loss /= n_f32.max(1.0);

    let (var_hat, cov_hat, _) = var_cov_const::<N, D>(z_hat, cfg, means, cov_matrix);
    let (var_target, cov_target, _) = var_cov_const::<N, D>(z_target, cfg, means, cov_matrix);

    let total = cfg.lambda * inv_loss
        + cfg.mu * (var_hat + var_target)
        + cfg.nu * (cov_hat + cov_target);

    VicRegBreakdown {
        total,
        inv: inv_loss,
        var_hat,
        var_target,
        cov_hat,
        cov_target,
        mean_std_hat: 0.0,
    }
}

/// L_var и L_cov + заполнение `means` / `cov` для набора Z.
fn var_cov_const<const N: usize, const D: usize>(
    z: &[[f32; D]; N],
    cfg: &VicRegConfig,
    means: &mut [f32; D],
    cov: &mut [[f32; D]; D],
) -> (f32, f32, f32) {
    let n_f32 = N as f32;
    let d_f32 = D as f32;
    if N < 2 {
        return (0.0, 0.0, 0.0);
    }

    means.fill(0.0);
    for i in 0..N {
        for j in 0..D {
            means[j] += z[i][j];
        }
    }
    for j in 0..D {
        means[j] /= n_f32;
    }

    let mut var_loss = 0.0f32;
    let mut std_sum = 0.0f32;
    let denom = n_f32 - 1.0;
    for j in 0..D {
        let mut variance = 0.0f32;
        for i in 0..N {
            let diff = z[i][j] - means[j];
            variance += diff * diff;
        }
        let std_dev = (variance / denom + cfg.eps).sqrt();
        std_sum += std_dev;
        let var_penalty = (cfg.gamma - std_dev).max(0.0);
        var_loss += var_penalty;
    }
    var_loss /= d_f32;

    let mut cov_loss = 0.0f32;
    for j in 0..D {
        for k in 0..D {
            let mut sum = 0.0f32;
            for i in 0..N {
                sum += (z[i][j] - means[j]) * (z[i][k] - means[k]);
            }
            cov[j][k] = sum / denom;
            if j != k {
                cov_loss += cov[j][k] * cov[j][k];
            }
        }
    }
    cov_loss /= d_f32;

    (var_loss, cov_loss, std_sum / d_f32)
}

/// Slice-API для HybridCore (D=512, N runtime, буферы предвыделены).
/// `z_hat`, `z_target`, `grad_zhat` — row-major N×D.
/// Градиенты только по Ẑ; Z' в stop-grad.
pub fn vicreg_loss_and_grad(
    z_hat: &[f32],
    z_target: &[f32],
    n: usize,
    d: usize,
    cfg: &VicRegConfig,
    means: &mut [f32],
    stds: &mut [f32],
    cov: &mut [f32],
    grad_zhat: &mut [f32],
) -> VicRegBreakdown {
    debug_assert_eq!(z_hat.len(), n * d);
    debug_assert_eq!(z_target.len(), n * d);
    debug_assert_eq!(grad_zhat.len(), n * d);
    debug_assert!(means.len() >= d && stds.len() >= d && cov.len() >= d * d);

    grad_zhat.fill(0.0);
    let n_f32 = n as f32;
    let d_f32 = d as f32;

    let mut inv_loss = 0.0f32;
    for i in 0..n {
        let row = i * d;
        for j in 0..d {
            let diff = z_hat[row + j] - z_target[row + j];
            inv_loss += diff * diff;
            // ∂L_inv/∂ẑ_ij = (2/N) (ẑ − z')
            if cfg.lambda != 0.0 {
                grad_zhat[row + j] += cfg.lambda * (2.0 / n_f32.max(1.0)) * diff;
            }
        }
    }
    inv_loss /= n_f32.max(1.0);

    let (var_hat, cov_hat, mean_std_hat) =
        var_cov_and_grad_hat(z_hat, n, d, cfg, means, stds, cov, grad_zhat);

    // Статистика таргета — только в скалярный лосс (stop-grad).
    let (var_target, cov_target, _) =
        var_cov_slice(z_target, n, d, cfg, means, stds, cov);

    let total = cfg.lambda * inv_loss
        + cfg.mu * (var_hat + var_target)
        + cfg.nu * (cov_hat + cov_target);

    VicRegBreakdown {
        total,
        inv: inv_loss,
        var_hat,
        var_target,
        cov_hat,
        cov_target,
        mean_std_hat,
    }
}

/// Const-generic градиент ∇_Ẑ L_VICReg. Буферы снаружи, heap нет.
pub fn compute_vicreg_grad_zero_alloc<const N: usize, const D: usize>(
    z_hat: &[[f32; D]; N],
    z_target: &[[f32; D]; N],
    cfg: &VicRegConfig,
    means: &mut [f32; D],
    stds: &mut [f32; D],
    cov_matrix: &mut [[f32; D]; D],
    grad_zhat: &mut [[f32; D]; N],
) -> VicRegBreakdown {
    // Плоские view без копирования: layout [[f32; D]; N] ≡ row-major N×D.
    let z_hat_flat: &[f32] =
        unsafe { std::slice::from_raw_parts(z_hat.as_ptr() as *const f32, N * D) };
    let z_tgt_flat: &[f32] =
        unsafe { std::slice::from_raw_parts(z_target.as_ptr() as *const f32, N * D) };
    let grad_flat: &mut [f32] =
        unsafe { std::slice::from_raw_parts_mut(grad_zhat.as_mut_ptr() as *mut f32, N * D) };
    let cov_flat: &mut [f32] =
        unsafe { std::slice::from_raw_parts_mut(cov_matrix.as_mut_ptr() as *mut f32, D * D) };
    vicreg_loss_and_grad(
        z_hat_flat,
        z_tgt_flat,
        N,
        D,
        cfg,
        means,
        stds,
        cov_flat,
        grad_flat,
    )
}

fn var_cov_slice(
    z: &[f32],
    n: usize,
    d: usize,
    cfg: &VicRegConfig,
    means: &mut [f32],
    stds: &mut [f32],
    cov: &mut [f32],
) -> (f32, f32, f32) {
    if n < 2 {
        return (0.0, 0.0, 0.0);
    }
    let n_f32 = n as f32;
    let d_f32 = d as f32;
    let denom = n_f32 - 1.0;

    means[..d].fill(0.0);
    for i in 0..n {
        let row = i * d;
        for j in 0..d {
            means[j] += z[row + j];
        }
    }
    for j in 0..d {
        means[j] /= n_f32;
    }

    let mut var_loss = 0.0f32;
    let mut std_sum = 0.0f32;
    for j in 0..d {
        let mut variance = 0.0f32;
        for i in 0..n {
            let diff = z[i * d + j] - means[j];
            variance += diff * diff;
        }
        let std_dev = (variance / denom + cfg.eps).sqrt();
        stds[j] = std_dev;
        std_sum += std_dev;
        var_loss += (cfg.gamma - std_dev).max(0.0);
    }
    var_loss /= d_f32;

    let mut cov_loss = 0.0f32;
    for j in 0..d {
        for k in j..d {
            let mut sum = 0.0f32;
            for i in 0..n {
                sum += (z[i * d + j] - means[j]) * (z[i * d + k] - means[k]);
            }
            let c = sum / denom;
            cov[j * d + k] = c;
            cov[k * d + j] = c;
            if j != k {
                // Оба внедиагональных элемента C_jk и C_kj.
                cov_loss += 2.0 * c * c;
            }
        }
    }
    cov_loss /= d_f32;

    (var_loss, cov_loss, std_sum / d_f32)
}

/// Variance + covariance Ẑ и начисление ∇_Ẑ (mean в stop-grad: μ считается
/// константой на шаге — стандартная аппроксимация при N≥8).
fn var_cov_and_grad_hat(
    z: &[f32],
    n: usize,
    d: usize,
    cfg: &VicRegConfig,
    means: &mut [f32],
    stds: &mut [f32],
    cov: &mut [f32],
    grad: &mut [f32],
) -> (f32, f32, f32) {
    let (var_loss, cov_loss, mean_std) = var_cov_slice(z, n, d, cfg, means, stds, cov);
    if n < 2 {
        return (var_loss, cov_loss, mean_std);
    }
    let n_f32 = n as f32;
    let d_f32 = d as f32;
    let denom = n_f32 - 1.0;

    // Variance: если std_j < γ,  ∂L_var/∂ẑ_ij = −(1/D) · (ẑ_ij−μ_j) / ((N−1) std_j)
    if cfg.mu != 0.0 {
        let scale = cfg.mu / d_f32;
        for j in 0..d {
            if stds[j] >= cfg.gamma {
                continue;
            }
            let inv_std = 1.0 / (denom * stds[j].max(1e-8));
            for i in 0..n {
                let centered = z[i * d + j] - means[j];
                grad[i * d + j] -= scale * centered * inv_std;
            }
        }
    }

    // Covariance: L_cov = (1/D) Σ_{j≠k} C_jk²
    // ∂L_cov/∂ẑ_aj = (2 /(D (N−1))) Σ_{k≠j} C_jk (ẑ_ak − μ_k)
    //              = (2 /(D (N−1))) [ (C ẑ̃_a)_j − C_jj ẑ̃_aj ]
    // (коэффициент 2 — точный градиент Σ_{j≠k}C_jk²; раньше был 4 — вдвое
    // сильнее минимизируемого лосса, аудит 22.08)
    if cfg.nu != 0.0 {
        let cov_scale = cfg.nu * 2.0 / (d_f32 * denom);
        for a in 0..n {
            let row = a * d;
            for j in 0..d {
                let mut cz = 0.0f32;
                for k in 0..d {
                    cz += cov[j * d + k] * (z[row + k] - means[k]);
                }
                let centered_j = z[row + j] - means[j];
                grad[row + j] += cov_scale * (cz - cov[j * d + j] * centered_j);
            }
        }
    }

    (var_loss, cov_loss, mean_std)
}

/// Правое умножение градиента строки W на P_owm: g ← gᵀ P  (row-major P).
/// Идемпотентно, если g уже лежит в образе P (px = P z), и явно выполняет
/// контракт «∇_W L_VICReg умножается на P_owm».
#[inline]
pub fn project_grad_row_owm(g_row: &mut [f32], p_owm: &[f32], tmp: &mut [f32]) {
    let d = g_row.len();
    debug_assert_eq!(tmp.len(), d);
    debug_assert_eq!(p_owm.len(), d * d);
    tmp.fill(0.0);
    // (g @ P)[i] = Σ_k g[k] P[k, i]
    for i in 0..d {
        let mut acc = 0.0f32;
        for k in 0..d {
            acc += g_row[k] * p_owm[k * d + i];
        }
        tmp[i] = acc;
    }
    g_row.copy_from_slice(tmp);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_unit() -> VicRegConfig {
        VicRegConfig {
            lambda: 1.0,
            mu: 1.0,
            nu: 1.0,
            gamma: 1.0,
            eps: 1e-4,
        }
    }

    #[test]
    fn constant_predictor_has_high_variance_loss() {
        // Коллапс: все ẑ одинаковы → std ≈ 0 → L_var ≈ γ = 1.
        const N: usize = 8;
        const D: usize = 4;
        let z_hat = [[0.25f32; D]; N];
        let mut z_target = [[0.0f32; D]; N];
        for i in 0..N {
            z_target[i][i % D] = 1.0;
        }
        let mut means = [0.0f32; D];
        let mut cov = [[0.0f32; D]; D];
        let br = compute_vicreg_loss_zero_alloc_ex(&z_hat, &z_target, &cfg_unit(), &mut means, &mut cov);
        assert!(
            br.var_hat > 0.9,
            "коллапс должен дать L_var ≈ 1, получили {}",
            br.var_hat
        );
    }

    #[test]
    fn diverse_features_zero_variance_hinge() {
        // Каждая фича имеет большой разброс по батчу → hinge не стреляет.
        const N: usize = 8;
        const D: usize = 4;
        let mut z_hat = [[0.0f32; D]; N];
        for i in 0..N {
            for j in 0..D {
                z_hat[i][j] = (i as f32 - 3.5) * (j as f32 + 1.0);
            }
        }
        let z_target = z_hat;
        let mut means = [0.0f32; D];
        let mut cov = [[0.0f32; D]; D];
        let br = compute_vicreg_loss_zero_alloc_ex(&z_hat, &z_target, &cfg_unit(), &mut means, &mut cov);
        assert!(
            br.var_hat < 1e-5,
            "при std≥γ variance hinge должен быть 0, получили {}",
            br.var_hat
        );
        assert!(br.inv < 1e-8, "ẑ = Z' → L_inv = 0, получили {}", br.inv);
    }

    #[test]
    fn correlated_features_raise_covariance_loss() {
        const N: usize = 8;
        const D: usize = 4;
        let mut z_hat = [[0.0f32; D]; N];
        for i in 0..N {
            let t = i as f32 - 3.5;
            z_hat[i][0] = t;
            z_hat[i][1] = t; // фичи 0 и 1 идентичны → C_01 велика
            z_hat[i][2] = if i % 2 == 0 { 1.0 } else { -1.0 };
            z_hat[i][3] = if i < 4 { 1.0 } else { -1.0 };
        }
        let z_target = z_hat;
        let mut means = [0.0f32; D];
        let mut cov = [[0.0f32; D]; D];
        let br = compute_vicreg_loss_zero_alloc_ex(&z_hat, &z_target, &cfg_unit(), &mut means, &mut cov);
        assert!(
            br.cov_hat > 0.1,
            "коррелированные фичи должны дать L_cov > 0, получили {}",
            br.cov_hat
        );
    }

    #[test]
    fn variance_grad_pushes_collapsed_feature_apart() {
        // Одна фича почти-коллапсирована (std ≪ γ) → hinge активен и градиент
        // по ней разводит батч. Точный коллапс (все значения равны) даёт
        // centered ≡ 0 и ∇_var ≡ 0 при любом ε — eps под корнем сокращается
        // в производной, это свойство формулы, а не баг.
        const N: usize = 8;
        const D: usize = 2;
        let mut z_hat = [[0.0f32; D]; N];
        for i in 0..N {
            z_hat[i][0] = if i % 2 == 0 { 1.0e-3 } else { -1.0e-3 }; // почти-коллапс
            z_hat[i][1] = (i as f32 - 3.5) * 3.0; // живая, std > 1
        }
        let z_target = z_hat;
        let cfg = VicRegConfig {
            lambda: 0.0,
            mu: 1.0,
            nu: 0.0,
            gamma: 1.0,
            eps: 1e-4,
        };
        let mut means = [0.0f32; D];
        let mut stds = [0.0f32; D];
        let mut cov = [[0.0f32; D]; D];
        let mut grad = [[0.0f32; D]; N];
        let br = compute_vicreg_grad_zero_alloc(
            &z_hat, &z_target, &cfg, &mut means, &mut stds, &mut cov, &mut grad,
        );
        assert!(br.var_hat > 0.0);
        // Живая фича (j=1) не в hinge → градиент ≈ 0.
        let g_live: f32 = (0..N).map(|i| grad[i][1].abs()).sum();
        assert!(g_live < 1e-5, "живая фича не должна получать var-град: {}", g_live);
        // Коллапсированная: градиенты не все нули (ε даёт крошечный std, hinge активен).
        let g_dead: f32 = (0..N).map(|i| grad[i][0].abs()).sum();
        assert!(g_dead > 0.0, "collapsed-фича должна получить ненулевой градиент");
    }

    #[test]
    fn invariance_grad_is_mse() {
        const N: usize = 4;
        const D: usize = 2;
        let z_hat = [[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]];
        let z_target = [[0.0f32, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        let cfg = VicRegConfig {
            lambda: 1.0,
            mu: 0.0,
            nu: 0.0,
            gamma: 1.0,
            eps: 1e-4,
        };
        let mut means = [0.0f32; D];
        let mut stds = [0.0f32; D];
        let mut cov = [[0.0f32; D]; D];
        let mut grad = [[0.0f32; D]; N];
        compute_vicreg_grad_zero_alloc(
            &z_hat, &z_target, &cfg, &mut means, &mut stds, &mut cov, &mut grad,
        );
        // ∂L_inv/∂ẑ_ij = (2/N)(ẑ − z') = 0.5 * ẑ
        for i in 0..N {
            for j in 0..D {
                let expected = 0.5 * z_hat[i][j];
                assert!(
                    (grad[i][j] - expected).abs() < 1e-5,
                    "grad[{i}][{j}]={} expected {}",
                    grad[i][j],
                    expected
                );
            }
        }
    }

    #[test]
    fn owm_right_multiply_identity_is_noop() {
        let mut g = [1.0f32, 2.0, 3.0];
        let mut p = [0.0f32; 9];
        for i in 0..3 {
            p[i * 3 + i] = 1.0;
        }
        let mut tmp = [0.0f32; 3];
        let before = g;
        project_grad_row_owm(&mut g, &p, &mut tmp);
        assert_eq!(g, before);
    }

    #[test]
    fn slice_and_const_generic_match() {
        const N: usize = 4;
        const D: usize = 3;
        let mut z_hat = [[0.0f32; D]; N];
        let mut z_tgt = [[0.0f32; D]; N];
        for i in 0..N {
            for j in 0..D {
                z_hat[i][j] = (i * D + j) as f32 * 0.1;
                z_tgt[i][j] = (j as f32) * 0.3 - 0.2;
            }
        }
        let cfg = cfg_unit();
        let mut means = [0.0f32; D];
        let mut cov = [[0.0f32; D]; D];
        let a = compute_vicreg_loss_zero_alloc(&z_hat, &z_tgt, &cfg, &mut means, &mut cov);

        let mut zhf = [0.0f32; N * D];
        let mut ztf = [0.0f32; N * D];
        for i in 0..N {
            for j in 0..D {
                zhf[i * D + j] = z_hat[i][j];
                ztf[i * D + j] = z_tgt[i][j];
            }
        }
        let mut means2 = [0.0f32; D];
        let mut stds = [0.0f32; D];
        let mut cov2 = [0.0f32; D * D];
        let mut grad = [0.0f32; N * D];
        let b = vicreg_loss_and_grad(&zhf, &ztf, N, D, &cfg, &mut means2, &mut stds, &mut cov2, &mut grad);
        assert!((a - b.total).abs() < 1e-4, "const={a} slice={}", b.total);
    }
}

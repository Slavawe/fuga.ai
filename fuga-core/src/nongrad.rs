//! NonGradientEngine — Rust-ядро безградиентного обучения.
//!
//! Перенос самых тяжёлых операций обучения из Python в Rust:
//!   - STDP + Oja update (Δw = lr·pre·post − lr·|post|²·w) — 512×512
//!   - predict (sign(weights @ hv)) — матричное умножение
//!   - LIF-спайк (утечка + порог) — пошаговый нейрон
//!   - fitness (косинус предсказания) — VSA-сравнение
//!
//! Python-обёртка: astral/nongradient_engine.py использует эти функции
//! через `fuga_core` (PyO3), обходя numpy-циклы.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Соревновательное Hebbian-правило: Δw = lr·pre·post − lr·|post|²·w
/// (Oja-нормализация: сильные каналы вытесняют слабые, веса не схлопываются).
#[pyfunction]
pub fn stdp_oja_update(
    weights: &Bound<'_, PyArray2<f32>>,
    pre: &Bound<'_, PyArray1<f32>>,
    post: &Bound<'_, PyArray1<f32>>,
    lr: f32,
    clip: f32,
) -> PyResult<()> {
    let shape = weights.shape();
    let rows = shape[0];
    let cols = shape[1];
    let w = unsafe { weights.as_slice_mut()? };
    let pre_s = unsafe { pre.as_slice()? };
    let post_s = unsafe { post.as_slice()? };

    // |post|²
    let post_norm2: f32 = post_s.iter().map(|x| x * x).sum();
    // pre·post^T + (−lr·|post|²)·w
    for i in 0..rows {
        for j in 0..cols {
            let hebb = lr * pre_s[i] * post_s[j];
            let oja = lr * post_norm2 * w[i * cols + j];
            let mut v = w[i * cols + j] + hebb - oja;
            if v > clip {
                v = clip;
            } else if v < -clip {
                v = -clip;
            }
            w[i * cols + j] = v;
        }
    }
    Ok(())
}

/// predict: pred = sign(hv) ⊗ sign(W·hv) — VSA-связывание предсказания.
/// Возвращает биполярный вектор-предсказание.
#[pyfunction]
pub fn vsa_predict<'py>(
    py: Python<'py>,
    weights: &Bound<'_, PyArray2<f32>>,
    hv: &Bound<'_, PyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let dim = hv.len();
    let w = unsafe { weights.as_slice()? };
    let hv_s = unsafe { hv.as_slice()? };

    let mut out = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut acc = 0.0f32;
        for j in 0..dim {
            acc += w[i * dim + j] * hv_s[j];
        }
        // pred[i] = sign(hv[i]) * sign(acc)
        let s1 = if hv_s[i] >= 0.0 { 1.0f32 } else { -1.0f32 };
        let s2 = if acc >= 0.0 { 1.0f32 } else { -1.0f32 };
        out.push(s1 * s2);
    }
    Ok(out.into_pyarray(py))
}

/// LIF-нейрон: один шаг (утечка e^{-dt/τ}, интеграция тока, порог).
/// Возвращает (spiked, новый потенциал).
#[pyfunction]
pub fn lif_step(v: f32, tau: f32, threshold: f32, current: f32, dt: f32) -> (bool, f32) {
    let v_new = v * (-dt / tau).exp() + current;
    let spiked = v_new >= threshold;
    if spiked {
        (true, 0.0)
    } else {
        (false, v_new)
    }
}

/// Косинус VSA: cos(a, b) = a·b / (|a|·|b|)
#[pyfunction]
pub fn vsa_cos(a: &Bound<'_, PyArray1<f32>>, b: &Bound<'_, PyArray1<f32>>) -> f32 {
    let a_s = unsafe { a.as_slice().unwrap() };
    let b_s = unsafe { b.as_slice().unwrap() };
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a_s.len() {
        dot += a_s[i] * b_s[i];
        na += a_s[i] * a_s[i];
        nb += b_s[i] * b_s[i];
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-9)
}

/// BATCH-косинус: cos между каждым вектором [B, dim] и одним [dim].
/// Возвращает [B] косинусов за ОДИН вызов PyO3 (для VSA-поиска учителя).
#[pyfunction]
pub fn vsa_cos_batch<'py>(
    py: Python<'py>,
    a: &Bound<'_, PyArray2<f32>>,
    b: &Bound<'_, PyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let shape = a.shape();
    let rows = shape[0];
    let cols = shape[1];
    let a_s = unsafe { a.as_slice()? };
    let b_s = unsafe { b.as_slice()? };
    let mut out = Vec::with_capacity(rows);
    // |b|
    let mut nb = 0.0f32;
    for j in 0..cols {
        nb += b_s[j] * b_s[j];
    }
    let nb = nb.sqrt() + 1e-9;
    for i in 0..rows {
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        for j in 0..cols {
            let v = a_s[i * cols + j];
            dot += v * b_s[j];
            na += v * v;
        }
        out.push(dot / (na.sqrt() * nb + 1e-9));
    }
    Ok(out.into_pyarray(py))
}

/// BATCH-обучение: весь цикл STDP/Oja за ОДИН вызов PyO3.
///
/// Принимает: weights [dim,dim], pre [B,dim], post [B,dim], lr, clip.
/// Для каждой пары: |post|² → hebb+oja → clip. 0 градиентов.
/// Это устраняет PyO3-overhead на каждый шаг (Python вызывает 1 раз
/// на батч вместо N раз).
#[pyfunction]
pub fn stdp_oja_batch(
    weights: &Bound<'_, PyArray2<f32>>,
    pre: &Bound<'_, PyArray2<f32>>,
    post: &Bound<'_, PyArray2<f32>>,
    lr: f32,
    clip: f32,
) -> PyResult<()> {
    let shape = weights.shape();
    let rows = shape[0];
    let cols = shape[1];
    let w = unsafe { weights.as_slice_mut()? };
    let pre_s = unsafe { pre.as_slice()? };
    let post_s = unsafe { post.as_slice()? };
    let b = pre_s.len() / cols; // batch size

    for batch_i in 0..b {
        // |post|² для этой пары
        let mut post_norm2: f32 = 0.0;
        for j in 0..cols {
            post_norm2 += post_s[batch_i * cols + j] * post_s[batch_i * cols + j];
        }
        for i in 0..rows {
            let pre_iv = pre_s[batch_i * cols + i];
            for j in 0..cols {
                let hebb = lr * pre_iv * post_s[batch_i * cols + j];
                let oja = lr * post_norm2 * w[i * cols + j];
                let mut v = w[i * cols + j] + hebb - oja;
                if v > clip {
                    v = clip;
                } else if v < -clip {
                    v = -clip;
                }
                w[i * cols + j] = v;
            }
        }
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════
// POINT-JEPA (Rust-порт): облако точек → фазовый HV, предиктор, обучение
// ═══════════════════════════════════════════════════════════════

/// Фазовый HV облака точек: sum exp(i·(x·ωx + y·ωy + z·ωz)) → sign → норм.
///
/// Args:
///   points: (N, 3) координаты точек
///   omega:  (3, dim) пространственные частоты [ωx; ωy; ωz]
///
/// Returns: биполярный латент (dim,), единичной нормы.
#[pyfunction]
pub fn point_cloud_encode<'py>(
    py: Python<'py>,
    points: &Bound<'_, PyArray2<f32>>,
    omega: &Bound<'_, PyArray2<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let pshape = points.shape();
    let n_pts = pshape[0];
    let oshape = omega.shape();
    let dim = oshape[1];
    let pts = unsafe { points.as_slice()? };
    let om = unsafe { omega.as_slice()? };

    let mut re = vec![0.0f32; dim];
    for p in 0..n_pts {
        let x = pts[p * 3];
        let y = pts[p * 3 + 1];
        let z = pts[p * 3 + 2];
        for d in 0..dim {
            let angle = x * om[d] + y * om[oshape[1] + d] + z * om[2 * oshape[1] + d];
            re[d] += angle.cos();
        }
    }
    let mut lat = vec![0.0f32; dim];
    for d in 0..dim {
        lat[d] = if re[d] >= 0.0 { 1.0 } else { -1.0 };
    }
    let norm: f32 = lat.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for v in &mut lat {
        *v /= norm;
    }
    Ok(lat.into_pyarray(py))
}

/// Point-JEPA предиктор: pred = W · lat.
#[pyfunction]
pub fn point_jepa_predict<'py>(
    py: Python<'py>,
    w: &Bound<'_, PyArray2<f32>>,
    lat: &Bound<'_, PyArray1<f32>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let shape = w.shape();
    let dim = shape[0];
    let w_s = unsafe { w.as_slice()? };
    let l_s = unsafe { lat.as_slice()? };

    let mut pred = vec![0.0f32; dim];
    for i in 0..dim {
        let mut acc = 0.0f32;
        for j in 0..dim {
            acc += w_s[i * dim + j] * l_s[j];
        }
        pred[i] = acc;
    }
    Ok(pred.into_pyarray(py))
}

/// Point-JEPA обучение: W += lr·(err⊗lat − |pred|²·W) (Widrow-Hoff + Oja).
#[pyfunction]
pub fn point_jepa_train(
    w: &Bound<'_, PyArray2<f32>>,
    lat: &Bound<'_, PyArray1<f32>>,
    target: &Bound<'_, PyArray1<f32>>,
    lr: f32,
) -> PyResult<()> {
    let shape = w.shape();
    let dim = shape[0];
    let w_s = unsafe { w.as_slice_mut()? };
    let l_s = unsafe { lat.as_slice()? };
    let t_s = unsafe { target.as_slice()? };

    let mut pred = vec![0.0f32; dim];
    for i in 0..dim {
        let mut acc = 0.0f32;
        for j in 0..dim {
            acc += w_s[i * dim + j] * l_s[j];
        }
        pred[i] = acc;
    }
    let pred_norm2: f32 = pred.iter().map(|x| x * x).sum();

    for i in 0..dim {
        let err = t_s[i] - pred[i];
        for j in 0..dim {
            let hebb = lr * err * l_s[j];
            let oja = lr * pred_norm2 * w_s[i * dim + j];
            w_s[i * dim + j] += hebb - oja;
        }
    }
    Ok(())
}

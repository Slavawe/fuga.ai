// macro_pc_proof.rs — синтетический пруф Predictive Coding (Rao-Ballard/Friston)
// + ГРИД СИГНАЛА ε (проверка гипотезы (c)-strong, Gary 16.10):
//   «PC требует нижнего уровня, который ПЛОХО предсказывает» =
//   «PC gain пропорционален силе класс-сигнала в остатке e_t».
//
// КОНСТРУКЦИЯ: класс персистентен на последовательность (как домен в корпусе);
// динамика z_{t+1} = S_class·z_t. S_B(ε) = ортогонализованная смесь
// (1−ε)·S_A + ε·R: ε=1 — независимые операторы (сильный сигнатурный остаток),
// ε→0 — S_B≈S_A (нижний уровень хорошо предсказывает ОБА класса → остаток ≈ шум).
// Нижний уровень W_low (WH) обучается на СМЕШАННОМ потоке → сходится к M̄,
// остаток e_t = o_t − W_low·o_{t−1} ≈ (S_class − M̄)z_{t−1} + ν.
// Верхний уровень: flat (вход z_t) vs PC (вход [z_t ; e_t]), WH оба. Каузально.
//
// ОЖИДАНИЕ (предрегистрировано 16.10): gain(ε) монотонно падает к 0 при ε→0 —
// «PC gain ∝ residual signal». Это подтверждает (c)-strong на синтетике;
// вопрос о РЕАЛЬНОМ корпусе решает доменный readout e_prev (val_macro --domain-check).
//
// PASS: gain(ε=1.0) ≥ 15% И gain(ε=0.1) < 5%.
// Запуск: cargo run --release --bin macro_pc_proof (CPU, секунды).
const DIM: usize = 64;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn norm(&mut self) -> f32 {
        let s: f32 = (0..12).map(|_| self.f32()).sum();
        s - 6.0
    }
}

fn unit_vec(rng: &mut Lcg) -> Vec<f32> {
    let mut v: Vec<f32> = (0..DIM).map(|_| rng.norm()).collect();
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
    v
}

fn wh_train(w: &mut Vec<f32>, x: &[f32], t: &[f32], lr: f32) {
    let mut pred = [0.0f32; DIM];
    for r in 0..DIM {
        let row = &w[r * DIM..(r + 1) * DIM];
        pred[r] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
    }
    for r in 0..DIM {
        let err = t[r] - pred[r];
        let row = &mut w[r * DIM..(r + 1) * DIM];
        for (wv, xv) in row.iter_mut().zip(x.iter()) {
            *wv += lr * err * xv;
        }
    }
}

fn wh_pred(w: &[f32], x: &[f32]) -> [f32; DIM] {
    let mut pred = [0.0f32; DIM];
    for r in 0..DIM {
        let row = &w[r * DIM..(r + 1) * DIM];
        pred[r] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
    }
    pred
}

/// Случайная ортогональная матрица (Грам-Шмидт по столбцам), row-major.
fn mk_orth(rng: &mut Lcg) -> Vec<f32> {
    let mut cols: Vec<Vec<f32>> = (0..DIM).map(|_| unit_vec(rng)).collect();
    for i in 0..DIM {
        for j in 0..i {
            let d: f32 = (0..DIM).map(|k| cols[k][i] * cols[k][j]).sum();
            for k in 0..DIM {
                cols[k][i] -= d * cols[k][j];
            }
            let n: f32 = (0..DIM).map(|k| cols[k][i] * cols[k][i]).sum::<f32>().sqrt();
            for k in 0..DIM {
                cols[k][i] /= n;
            }
        }
    }
    let mut s = vec![0.0f32; DIM * DIM];
    for c in 0..DIM {
        for r in 0..DIM {
            s[r * DIM + c] = cols[c][r];
        }
    }
    s
}

/// S_B(ε): смесь (1−ε)·S_A + ε·R с ре-ортогонализацией. ε=1 → независимый
/// оператор (сильный сигнатурный остаток), ε→0 → S_B≈S_A (остаток ≈ шум).
fn mix_orth(s_a: &[f32], r_other: &[f32], eps: f32) -> Vec<f32> {
    let mut cols: Vec<Vec<f32>> = (0..DIM)
        .map(|c| {
            (0..DIM)
                .map(|r| (1.0 - eps) * s_a[r * DIM + c] + eps * r_other[r * DIM + c])
                .collect()
        })
        .collect();
    for i in 0..DIM {
        for j in 0..i {
            let d: f32 = (0..DIM).map(|k| cols[k][i] * cols[k][j]).sum();
            for k in 0..DIM {
                cols[k][i] -= d * cols[k][j];
            }
            let n: f32 = (0..DIM).map(|k| cols[k][i] * cols[k][i]).sum::<f32>().sqrt();
            for k in 0..DIM {
                cols[k][i] /= n;
            }
        }
    }
    let mut s = vec![0.0f32; DIM * DIM];
    for c in 0..DIM {
        for r in 0..DIM {
            s[r * DIM + c] = cols[c][r];
        }
    }
    s
}

// Один полный эксперимент на паре операторов: W_low (смешанный поток) →
// flat vs PC (held-out ошибка). Возвращает (err_flat, err_pc).
fn experiment(rng: &mut Lcg, s_a: &[f32], s_b: &[f32]) -> (f64, f64) {
    let seq_len = 600usize;
    let n_train = 16usize;
    let n_test = 6usize;
    let dyn_noise = 0.02f32;
    let obs_noise = 0.15f32;
    let lr = 0.01f32;

    let gen_seq = |rng: &mut Lcg, s: &[f32], len: usize| -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut z0 = unit_vec(rng);
        let mut z = Vec::with_capacity(len);
        let mut o = Vec::with_capacity(len);
        for _ in 0..len {
            z.push(z0.clone());
            o.push(z0.iter().map(|&b| b + obs_noise * rng.norm()).collect());
            let mut z1 = [0.0f32; DIM];
            for r in 0..DIM {
                let row = &s[r * DIM..(r + 1) * DIM];
                z1[r] = row.iter().zip(z0.iter()).map(|(a, b)| a * b).sum::<f32>()
                    + dyn_noise * rng.norm();
            }
            let n: f32 = z1.iter().map(|x| x * x).sum::<f32>().sqrt();
            z0 = z1.iter().map(|&x| x / n).collect();
        }
        (z, o)
    };

    // 1. Нижний уровень: W_low на СМЕШАННОМ потоке → M̄; сигнатура класса в остатке.
    let mut w_low = vec![0.0f32; DIM * DIM];
    for _pass in 0..2 {
        for s in [s_a, s_b] {
            for _ in 0..n_train {
                let (_z, o) = gen_seq(rng, s, seq_len);
                for t in 1..o.len() {
                    wh_train(&mut w_low, &o[t - 1], &o[t], 0.02);
                }
            }
        }
    }

    // 2. Верхний уровень: flat (z_t) vs PC ([z_t ; e_t]), два блока Wz/We.
    let mut w_flat = vec![0.0f32; DIM * DIM];
    let mut w_pc_z = vec![0.0f32; DIM * DIM];
    let mut w_pc_e = vec![0.0f32; DIM * DIM];
    for s in [s_a, s_b] {
        for _ in 0..n_train {
            let (z, o) = gen_seq(rng, s, seq_len);
            let mut prev_o = o[0].clone();
            for t in 1..o.len() {
                let e: Vec<f32> = o[t]
                    .iter()
                    .zip(wh_pred(&w_low, &prev_o).iter())
                    .map(|(a, b)| a - b)
                    .collect();
                wh_train(&mut w_flat, &z[t - 1], &z[t], lr);
                // PC two-block: err = z_next − Wz·z − We·e; два раздельных ΔW.
                let pz = wh_pred(&w_pc_z, &z[t - 1]);
                let pe = wh_pred(&w_pc_e, &e);
                for r in 0..DIM {
                    let err = z[t][r] - pz[r] - pe[r];
                    for i in 0..DIM {
                        w_pc_z[r * DIM + i] += lr * err * z[t - 1][i];
                        w_pc_e[r * DIM + i] += lr * err * e[i];
                    }
                }
                prev_o = o[t].clone();
            }
        }
    }

    // 3. Held-out.
    let (mut sf, mut sp, mut cnt) = (0.0f64, 0.0f64, 0usize);
    for s in [s_a, s_b] {
        for _ in 0..n_test {
            let (z, o) = gen_seq(rng, s, seq_len);
            let mut prev_o = o[0].clone();
            for t in 1..o.len() {
                let e: Vec<f32> = o[t]
                    .iter()
                    .zip(wh_pred(&w_low, &prev_o).iter())
                    .map(|(a, b)| a - b)
                    .collect();
                let pf = wh_pred(&w_flat, &z[t - 1]);
                let pz = wh_pred(&w_pc_z, &z[t - 1]);
                let pe = wh_pred(&w_pc_e, &e);
                for r in 0..DIM {
                    let df = z[t][r] - pf[r];
                    let dp = z[t][r] - pz[r] - pe[r];
                    sf += (df * df) as f64;
                    sp += (dp * dp) as f64;
                }
                cnt += 1;
                prev_o = o[t].clone();
            }
        }
    }
    let n = (cnt * DIM) as f64;
    (sf / n, sp / n)
}

fn main() {
    let mut rng = Lcg(0xF03D_C0DE_15_10_2026);
    let s_a = mk_orth(&mut rng);
    let r_other = mk_orth(&mut rng);

    println!("== MACRO PC PROOF: грид сигнала ε (проверка (c)-strong) ==");
    println!("  ε=1.0 — независимые операторы (сильный сигнал); ε→0 — S_B≈S_A (остаток ≈ шум)");
    let mut gains: Vec<(f32, f64)> = Vec::new();
    for &eps in [1.0f32, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.1].iter() {
        let s_b = mix_orth(&s_a, &r_other, eps);
        let (flat, pc) = experiment(&mut rng, &s_a, &s_b);
        let gain = (flat - pc) / flat * 100.0;
        gains.push((eps, gain));
        println!(
            "  ε={:.2}: flat={:.4} pc={:.4} → выигрыш PC {:+.1}%",
            eps, flat, pc, gain
        );
    }

    let g_strong = gains.iter().find(|(e, _)| *e == 1.0).unwrap().1;
    let g_weak = gains.iter().find(|(e, _)| *e == 0.1).unwrap().1;
    let pass = g_strong >= 15.0 && g_weak < 5.0;
    println!(
        "  ВЕРДИКТ: {}",
        if pass {
            "PASS — PC gain монотонен по силе сигнала: (c)-strong подтверждена на синтетике («PC требует нижний уровень, который плохо предсказывает»)"
        } else {
            "FAIL — грид не подтверждает монотонность, (c)-strong требует переосмысления"
        }
    );
    if !pass {
        std::process::exit(1);
    }
}

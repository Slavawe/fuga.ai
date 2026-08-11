// macro_ab.rs — СИНТЕТИЧЕСКИЙ ПРУФ Byte-H-JEPA (пункт 1 плана).
// Два семантических класса латент-целей (A: unicode-error блок,
// B: array-цикл). W_macro (512²) учится ЧЕСТНЫМ Widrow-Hoff:
//   err = z_target − W_macro·z_ctx,  W += lr·err ⊗ z_ctx
// Проверка: для НЕВИДАННЫХ контекстов предсказанный \hat{z} = W·z_ctx
// косинусно ближе к центроиду своего класса, чем к чужому — значит
// латент-цели линейно разделимы и Macro-предиктор принципиально работает.
use fuga::ai::latent_jepa::{LatentVector, LATENT_DIM};

struct ClassProto {
    _name: &'static str,
    // Прототип латента класса: детерминированный случайный 512-dim вектор.
    proto: LatentVector,
}

fn random_latent(seed: u64) -> LatentVector {
    // Детерминированный ПРИМИТИВНЫЙ генератор (стандартный LCG), чтобы пруф
    // был воспроизводим без внешних зависимостей.
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut v = LatentVector::zero();
    for i in 0..LATENT_DIM {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let r = ((s >> 33) % 1000) as f32 / 1000.0; // [0,1)
        v.values[i] = r - 0.5;
    }
    let n: f32 = v.values.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in &mut v.values {
        *x /= n;
    }
    v
}

fn main() {
    println!(
        "=== ПРУФ Byte-H-JEPA: разделимость латент-целей через W_macro (Widrow-Hoff) ==="
    );
    println!("латент_dim = {}", LATENT_DIM);

    // 1. Два СЕМАНТИЧЕСКИХ КЛАССА: прототипы взаимно ортогональны (разные зоны).
    let a = ClassProto { _name: "A:unicode_err", proto: random_latent(0xA11CE) };
    let b = ClassProto { _name: "B:array_loop", proto: random_latent(0xB005) };
    // Ортогонализируем B от A (Грам-Шмидт): чистые классы, без перекрытия.
    let mut bv = b.proto.values.clone();
    let dot_ab: f32 = a.proto.values.iter().zip(&bv).map(|(x, y)| x * y).sum();
    for i in 0..LATENT_DIM {
        bv[i] -= dot_ab * a.proto.values[i];
    }
    let n: f32 = bv.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in &mut bv {
        *x /= n;
    }
    let b_ortho = LatentVector { values: bv };
    let cos_ab = a.proto.cosine_similarity(&b_ortho);
    println!("стартовое перекрытие классов (cos A,B) = {:.6} (должно ≈0)", cos_ab);

    // 2. Тренировочный набор: контексты = прототип + шум; цель = прототип СВОЕГО
    //    класса + малый шум (эмуляция «макросемантики»).
    const N_TRAIN: usize = 400;
    const NOISE: f32 = 0.15;
    let mut w_macro = vec![0.0f32; LATENT_DIM * LATENT_DIM];

    // Собственный LCG для шума — детерминированный.
    let mut rng: u64 = 0xC0FFEE;
    let mut next_rand = |rng: &mut u64| -> f32 {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*rng >> 33) % 1000) as f32 / 1000.0
    };

    let mut n_a_correct = 0usize;
    let mut n_b_correct = 0usize;
    const LR: f32 = 0.03;
    const TEST_EVERY: usize = 50;

    // Предобучение: 2 эпохи по N_TRAIN (малый цикл — сходимость дельты).
    for epoch in 0..2 {
        for i in 0..N_TRAIN {
            let class_a = i % 2 == 0;
            let proto = if class_a { &a.proto } else { &b_ortho };
            // контекст: прототип + шум
            let mut ctx = LatentVector::zero();
            let mut tgt = LatentVector::zero();
            for d in 0..LATENT_DIM {
                let nx = next_rand(&mut rng) * 2.0 - 1.0;
                let nt = next_rand(&mut rng) * 2.0 - 1.0;
                ctx.values[d] = proto.values[d] + NOISE * nx;
                tgt.values[d] = proto.values[d] + 0.05 * nt;
            }
            let cn: f32 = ctx.values.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for x in &mut ctx.values {
                *x /= cn;
            }
            let tn: f32 = tgt.values.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for x in &mut tgt.values {
                *x /= tn;
            }

            // Widrow-Hoff: pred = W·ctx; err = tgt − pred; W += lr·err⊗ctx
            let mut pred = LatentVector::zero();
            for r in 0..LATENT_DIM {
                let row = &w_macro[r * LATENT_DIM..(r + 1) * LATENT_DIM];
                pred.values[r] = row.iter().zip(&ctx.values).map(|(w, x)| w * x).sum();
            }
            let err: Vec<f32> = (0..LATENT_DIM).map(|r| tgt.values[r] - pred.values[r]).collect();
            let mut n_out = 0.0f32;
            for r in 0..LATENT_DIM {
                let row = &mut w_macro[r * LATENT_DIM..(r + 1) * LATENT_DIM];
                let e = err[r];
                for (w, x) in row.iter_mut().zip(&ctx.values) {
                    *w += LR * e * x;
                }
                n_out += row.iter().map(|w| w * w).sum::<f32>();
            }
            // Мягкий cap нормы строк (как в LatentPredictor): держит обучение стабильным.
            let cap = (n_out / LATENT_DIM as f32).sqrt().max(1e-8);
            if cap > 2.0 {
                let scale = 2.0 / cap;
                for w in w_macro.iter_mut() {
                    *w *= scale;
                }
            }

            // Периодический тест на НЕВИДАННЫХ контекстах: каждая точка
            // тестирует ОБА класса (A и B) — свежий шум, свои цели.
            if epoch == 1 && i % TEST_EVERY == 0 {
                for class in [true, false] {
                    let proto = if class { &a.proto } else { &b_ortho };
                    let mut t_ctx = LatentVector::zero();
                    let mut t_tgt = LatentVector::zero();
                    for d in 0..LATENT_DIM {
                        t_ctx.values[d] = proto.values[d] + NOISE * (next_rand(&mut rng) * 2.0 - 1.0);
                        t_tgt.values[d] = proto.values[d] + 0.05 * (next_rand(&mut rng) * 2.0 - 1.0);
                    }
                    let tn2: f32 = t_ctx.values.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                    for x in &mut t_ctx.values {
                        *x /= tn2;
                    }
                    let mut pred = LatentVector::zero();
                    for r in 0..LATENT_DIM {
                        let row = &w_macro[r * LATENT_DIM..(r + 1) * LATENT_DIM];
                        pred.values[r] = row.iter().zip(&t_ctx.values).map(|(w, x)| w * x).sum();
                    }
                    let pn: f32 = pred.values.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                    for x in &mut pred.values {
                        *x /= pn;
                    }
                    let cos_a = pred.cosine_similarity(&a.proto);
                    let cos_b = pred.cosine_similarity(&b_ortho);
                    let correct = if class { cos_a > cos_b } else { cos_b > cos_a };
                    if class {
                        n_a_correct += usize::from(correct);
                    } else {
                        n_b_correct += usize::from(correct);
                    }
                }
            }
        }
    }

    let n_te = (N_TRAIN / TEST_EVERY) * 2; // 2 класса × каждые 50 на эпохе-финал
    println!("── Результат на невиданных контекстах ──");
    println!("класс A (unicode_err): {}/{} верно", n_a_correct, n_te / 2);
    println!("класс B (array_loop) : {}/{} верно", n_b_correct, n_te / 2);
    let total = n_te;
    let ok = n_a_correct + n_b_correct;
    println!("ИТОГО: {}/{} ({:.1}%)", ok, total, 100.0 * ok as f32 / total as f32);
    if ok == total {
        println!("ПРУФ ПРОЙДЕН: латент-цели линейно разделимы W_macro → Macro-уровень возможен.");
    } else {
        println!("ПРУФ НЕ ПОЛНЫЙ: есть ошибки классификации — нужен больший шум/эпох.");
    }
    // Метрика: среднее |W_macro| (норма Фробениуса / dim).
    let f: f32 = w_macro.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("норма Фробениуса W_macro = {:.3} (512², ~средняя строка {:.4})", f, f / LATENT_DIM as f32);
}
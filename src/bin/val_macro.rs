// val_macro.rs — валидатор МАКРО-КАНАЛА (урок 15.10: val_domains слеп к W_macro).
//
// Для каждой позиции i валидационного потока:
//   z_t    = enc(окно data[i-ctx..=i])                  — вход flat-макро
//   target = enc(окно data[i+1..=i+1+ctx])              — цель макро (как в тренере)
//   e_prev = enc(byte_i) − W_local·enc(окно data[i-1-ctx..=i-1])
//          — каузальный остаток байтового уровня предыдущей пары (PC-вход)
//   s = flat: z_t            |  pc: z_t + k·e_prev     (вход, КОТОРЫМ модель обучалась)
// Метрики: mean ||target − W_macro·s|| и mean cos(target, W_macro·s) —
// та же схема, что [val] в тренере. A/B корректен только при СВОЁМ входе каждой руки.
//
// Usage: val_macro <ckpt.fuga> <val.bin> [--ctx 8] [--mode flat|pc] [--k 1.0]
use fuga::ai::htm_temporal::{load_unified_macro, load_unified_macro2, TemporalMemory};
use fuga::ai::sdr::{byte_basis, structure_sdr_from_sdrs};

const DIM: usize = 512;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: val_macro <ckpt.fuga> <val.bin> [--ctx 8] [--mode flat|pc] [--k 1.0]");
        std::process::exit(1);
    }
    let ckpt = args[1].clone();
    let mut val_path = String::new();
    let mut ctxw = 8usize;
    let mut mode = "flat".to_string();
    let mut k = 1.0f32;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--ctx" => {
                ctxw = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(8);
                i += 2;
            }
            "--mode" => {
                mode = args.get(i + 1).cloned().unwrap_or_else(|| "flat".into());
                i += 2;
            }
            "--k" => {
                k = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(1.0);
                i += 2;
            }
            pos => {
                if val_path.is_empty() {
                    val_path = pos.to_string();
                }
                i += 1;
            }
        }
    }
    let mode = mode.as_str();

    // РЕЖИМ domain: есть ли доменный сигнал в e_prev? (решающая проверка
    // (c)-strong на реальном корпусе). Nearest-centroid: центроиды e_prev
    // первых половин трёх доменных файлов, классификация вторых половин.
    // Шанс = 33.3%. Если ≈шанс — доменного сигнала в остатке нет → PC в этой
    // форме архитектурно мёртв на этом стеке (W_local насыщен).
    if mode == "domain" {
        let w_macro_flat = load_unified_macro(&ckpt); // не используется для e_prev
        let _ = w_macro_flat;
        let mut tm = TemporalMemory::new(64, ctxw);
        assert!(tm.load_unified_fuga1(&ckpt), "FUGA1 не читается: {}", ckpt);
        let w_local = tm.predictor().w.clone();
        let enc = tm.predictor().encoder.clone();
        let byte_cache: Vec<fuga::SdrVector> = (0..=255u8).map(byte_basis).collect();
        let e_prevs_of = |path: &str| -> Vec<Vec<f32>> {
            let vd = std::fs::read(path).unwrap_or_default();
            let win_latent = |lo: usize, hi: usize| -> fuga::LatentVector {
                let sdrs: Vec<fuga::SdrVector> = vd[lo..=hi]
                    .iter()
                    .map(|&c| byte_cache[c as usize].clone())
                    .collect();
                enc.encode(&structure_sdr_from_sdrs(&sdrs))
            };
            let mut out = Vec::new();
            let mut norms: Vec<f32> = Vec::new(); // сырые нормы до нормировки (диагностика norm-сигнала)
            let mut t0 = ctxw + 1;
            while t0 + 1 + ctxw < vd.len() && out.len() < 1024 {
                let b = enc.encode(&byte_cache[vd[t0] as usize]);
                let x_prev = win_latent(t0 - 1 - ctxw, t0 - 1);
                let mut e = vec![0.0f32; DIM];
                for o in 0..DIM {
                    let mut acc = 0.0f32;
                    let row = o * DIM;
                    for j in 0..DIM {
                        acc += w_local[row + j] * x_prev.values[j];
                    }
                    e[o] = b.values[o] - acc;
                }
                // нормировка на единицу: читается НАПРАВЛЕНИЕ остатка
                let nrm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
                norms.push(nrm);
                if nrm > 1e-6 {
                    for x in e.iter_mut() {
                        *x /= nrm;
                    }
                }
                out.push(e);
                t0 += ctxw + 1;
            }
            let mn: f32 = norms.iter().sum::<f32>() / norms.len().max(1) as f32;
            println!("    [норма ||e|| {}]: средняя={:.3} (диагностика norm-сигнала)", path, mn);
            out
        };
        let domains = [
            ("code", "train_fuga1/corpus/val_code.bin"),
            ("en", "train_fuga1/corpus/val_en.bin"),
            ("ru", "train_fuga1/corpus/val_ru.bin"),
        ];
        // СТРАТИФИЦИРОВАННЫЙ сплит (Gary 03:26 п.1): у КАЖДОГО класса первая
        // половина → центроид, вторая → тест. Размеры классов печатаются.
        let per_class: Vec<(String, Vec<Vec<f32>>, Vec<Vec<f32>>)> = domains
            .iter()
            .map(|(n, p)| {
                let mut es = e_prevs_of(p);
                es.truncate(1024);
                let half = es.len() / 2;
                (n.to_string(), es[..half].to_vec(), es[half..].to_vec())
            })
            .collect();

        // центроид = нормированное среднее e половины класса
        let centroid = |es: &[Vec<f32>]| -> Vec<f64> {
            let mut c = vec![0.0f64; DIM];
            for e in es {
                for d in 0..DIM {
                    c[d] += e[d] as f64;
                }
            }
            let nrm: f64 = c.iter().map(|x| x * x).sum::<f64>().sqrt();
            if nrm > 1e-9 {
                c.iter_mut().for_each(|x| *x /= nrm);
            }
            c
        };
        let cents: Vec<Vec<f64>> = per_class.iter().map(|(_, tr, _)| centroid(tr)).collect();
        let cosc = |a: &[f64], b: &[f64]| -> f64 {
            let d: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            d / (na * nb + 1e-12)
        };

        println!("== DOMAIN CHECK e_prev (стратифицированный nearest-centroid, шанс 33.3%) ==");
        // ДИАГНОСТИКА ЦЕНТРОИДОВ (Gary п.1): попарные cos — схлопывание видно сразу
        for i in 0..3 {
            for j in (i + 1)..3 {
                println!(
                    "  cos(центроид {}, {}): {:+.3}",
                    per_class[i].0,
                    per_class[j].0,
                    cosc(&cents[i], &cents[j])
                );
            }
        }
        let (mut hit, mut tot) = (0usize, 0usize);
        let (mut own_sum, mut other_sum) = (0.0f64, 0.0f64);
        let mut per = [0usize; 3];
        let mut per_tot = [0usize; 3];
        for (ci, (_n, _tr, te)) in per_class.iter().enumerate() {
            for e in te {
                let coses: Vec<f64> = (0..3)
                    .map(|k| {
                        let d: f64 = e
                            .iter()
                            .zip(cents[k].iter())
                            .map(|(x, y)| *x as f64 * y)
                            .sum();
                        let ne: f64 = e
                            .iter()
                            .map(|x| (*x as f64) * (*x as f64))
                            .sum::<f64>()
                            .sqrt();
                        d / (ne + 1e-12)
                    })
                    .collect();
                own_sum += coses[ci];
                other_sum += (coses[(ci + 1) % 3] + coses[(ci + 2) % 3]) / 2.0;
                let best = (0..3)
                    .max_by(|a, b| coses[*a].total_cmp(&coses[*b]))
                    .unwrap();
                tot += 1;
                per_tot[ci] += 1;
                if best == ci {
                    hit += 1;
                    per[ci] += 1;
                }
            }
        }
        for (k, (n, tr, te)) in per_class.iter().enumerate() {
            println!(
                "  {:<5}: train={} test={} точность {}/{} = {:.1}%",
                n,
                tr.len(),
                te.len(),
                per[k],
                per_tot[k],
                per[k] as f64 / per_tot[k].max(1) as f64 * 100.0
            );
        }
        println!(
            "  средний cos к своему центроиду: {:+.4}; к чужому: {:+.4}",
            own_sum / tot.max(1) as f64,
            other_sum / tot.max(1) as f64
        );
        let acc = hit as f64 / tot.max(1) as f64;
        println!(
            "  ИТОГО: {}/{} = {:.1}% — {}",
            hit,
            tot,
            acc * 100.0,
            if acc > 0.5 {
                "ДОМЕННЫЙ СИГНАЛ В e_prev ЕСТЬ — PC архитектурно жив, вопрос калибровки"
            } else if acc >= 0.30 {
                "около шанса — наивный readout не выявляет сигнал; (c)-strong на корпусе не доказана и не опровергнута"
            } else {
                "НИЖЕ ШАНСА — анти-сигнал или схлопывание центроидов; смотреть диагностику cos выше"
            }
        );
        // LOGISTIC REGRESSION (Gary 17.10 п.3): ru 49.8% (+16.5 п.п.) — значимо;
        // softmax-regression на тех же признаках — прямая оценка отделимости.
        // Train: первые половины, Test: вторые (те же, что для центроидов).
        let mut ws = [[0.0f64; DIM]; 3]; // 3 класса × 512 весов (без bias — данные нормированы)
        let lr_lg = 0.5f64;
        let epochs = 120usize;
        let mut train_all: Vec<(usize, &Vec<f32>)> = Vec::new();
        for (ci, (_n, tr, _te)) in per_class.iter().enumerate() {
            for e in tr {
                train_all.push((ci, e));
            }
        }
        for _ep in 0..epochs {
            for (ci, e) in &train_all {
                // softmax
                let mut scores = [0.0f64; 3];
                for k in 0..3 {
                    scores[k] = ws[k].iter().zip(e.iter()).map(|(w, x)| w * (*x as f64)).sum::<f64>();
                }
                let mx = scores.iter().cloned().fold(f64::MIN, f64::max);
                let exps: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
                let sum: f64 = exps.iter().sum();
                for k in 0..3 {
                    let p = exps[k] / sum;
                    let g = p - if k == *ci { 1.0 } else { 0.0 };
                    for d in 0..DIM {
                        ws[k][d] -= lr_lg * g * e[d] as f64;
                    }
                }
            }
        }
        let (mut lg_hit, mut lg_tot) = (0usize, 0usize);
        let mut lg_per = [0usize; 3];
        let mut lg_per_tot = [0usize; 3];
        for (ci, (_n, _tr, te)) in per_class.iter().enumerate() {
            for e in te {
                let mut scores = [0.0f64; 3];
                for k in 0..3 {
                    scores[k] = ws[k].iter().zip(e.iter()).map(|(w, x)| w * (*x as f64)).sum::<f64>();
                }
                let best = (0..3).max_by(|a, b| scores[*a].total_cmp(&scores[*b])).unwrap();
                lg_tot += 1;
                lg_per_tot[ci] += 1;
                if best == ci {
                    lg_hit += 1;
                    lg_per[ci] += 1;
                }
            }
        }
        println!(
            "  LOGREG (softmax, {} epochs): {}/{} = {:.1}% — {}",
            epochs,
            lg_hit,
            lg_tot,
            lg_hit as f64 / lg_tot.max(1) as f64 * 100.0,
            if (lg_hit as f64 / lg_tot.max(1) as f64) > 0.6 {
                "сигнал в e_prev ЕСТЬ (наивный центроид — плохой readout)"
            } else {
                "сигнала нет и для линейного классификатора — (c)-strong на корпусе подтверждена"
            }
        );
        for (k, (n, _tr, _te)) in per_class.iter().enumerate() {
            if lg_per_tot[k] > 0 {
                println!(
                    "    logreg {:<5}: {}/{} = {:.1}%",
                    n,
                    lg_per[k],
                    lg_per_tot[k],
                    lg_per[k] as f64 / lg_per_tot[k] as f64 * 100.0
                );
            }
        }
        return;
    }

    // W_macro — из tag=7; W_local и энкодер — из единого чекпоинта.
    // W_macro: pc2 → tag=9 (Wz+We); flat/pc → tag=7.
    let (wz, we_opt) = if mode == "pc2" {
        let r = load_unified_macro2(&ckpt).expect("tag=9 MACRO_W2 не читается");
        (r.0, Some(r.1))
    } else {
        (load_unified_macro(&ckpt).expect("tag=7 MACRO_W не читается"), None::<Vec<f32>>)
    };
    let w_macro = wz;
    let mut tm = TemporalMemory::new(64, ctxw);
    assert!(tm.load_unified_fuga1(&ckpt), "FUGA1 не читается: {}", ckpt);
    let w_local = tm.predictor().w.clone();
    let enc = tm.predictor().encoder.clone();
    let byte_cache: Vec<fuga::SdrVector> = (0..=255u8).map(byte_basis).collect();

    let vd = std::fs::read(&val_path).expect("val.bin не читается");
    let win_latent = |lo: usize, hi: usize| -> fuga::LatentVector {
        let sdrs: Vec<fuga::SdrVector> = vd[lo..=hi]
            .iter()
            .map(|&c| byte_cache[c as usize].clone())
            .collect();
        enc.encode(&structure_sdr_from_sdrs(&sdrs))
    };

    let n_max = 4096usize;
    let span = vd.len().saturating_sub(ctxw + ctxw + 2);
    let stride = (span / n_max).max(1);
    let (mut se, mut st, mut sc, mut cnt) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    let mut cos_vals: Vec<f64> = Vec::new(); // для noise floor (чанковый SE)
    let mut t0 = ctxw + 1; // индекс последнего байта текущего окна (t0-1-ctxw ≥ 0 для e_prev)
    while t0 + 1 + ctxw < vd.len() {
        // z_t: окно, заканчивающееся на t0
        let z_t = win_latent(t0 - ctxw, t0);
        // цель: следующее окно
        let target = win_latent(t0 + 1, t0 + 1 + ctxw);
        // s: вход макро по режиму
        let e_prev: Vec<f32> = if mode == "pc" || mode == "pc2" {
            // e_prev = enc(byte t0) − W_local·enc(окно, заканчивающееся на t0−1)
            let b = enc.encode(&byte_cache[vd[t0] as usize]);
            let x_prev = win_latent(t0 - 1 - ctxw, t0 - 1);
            let mut e = vec![0.0f32; DIM];
            for o in 0..DIM {
                let mut acc = 0.0f32;
                let row = o * DIM;
                for j in 0..DIM {
                    acc += w_local[row + j] * x_prev.values[j];
                }
                e[o] = b.values[o] - acc;
            }
            e
        } else {
            Vec::new()
        };
        // pred = W_macro·s (pc2: ДВА блока Wz·z + We·e — архитектура пруфа)
        let mut pred = [0.0f32; DIM];
        if mode == "pc2" {
            for o in 0..DIM {
                let row = o * DIM;
                pred[o] = w_macro[row..row + DIM]
                    .iter()
                    .zip(z_t.values.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    + we_opt
                        .as_ref()
                        .unwrap()[row..row + DIM]
                        .iter()
                        .zip(e_prev.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>();
            }
        } else {
            let s: Vec<f32> = if mode == "pc" {
                z_t.values.iter().zip(e_prev.iter()).map(|(a, b)| a + k * b).collect()
            } else {
                z_t.values.clone()
            };
            for o in 0..DIM {
                let row = o * DIM;
                pred[o] = w_macro[row..row + DIM]
                    .iter()
                    .zip(s.iter())
                    .map(|(a, b)| a * b)
                    .sum();
            }
        }
        let (mut e2, mut t2n, mut d, mut pn) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for o in 0..DIM {
            let dlt = target.values[o] - pred[o];
            e2 += dlt * dlt;
            t2n += target.values[o] * target.values[o];
            d += target.values[o] * pred[o];
            pn += pred[o] * pred[o];
        }
        se += e2.sqrt() as f64;
        st += t2n.sqrt() as f64;
        let cos_sample = (d / (t2n.sqrt() * pn.sqrt() + 1e-8)) as f64;
        sc += cos_sample;
        cos_vals.push(cos_sample);
        cnt += 1;
        // шаг со stride (равномерное покрытие потока)
        let step = stride * (ctxw + 1);
        t0 += step.max(1);
        if cnt >= n_max {
            break;
        }
    }
    println!(
        "== VAL MACRO {} | mode={} k={:.2} | ||W_macro||={:.3} ==",
        ckpt,
        mode,
        k,
        w_macro.iter().map(|x| x * x).sum::<f32>().sqrt()
    );
    if cnt > 0 {
        println!(
            "  {} пар (val): ||z_next − W_macro·s||={:.4} (||z_next||={:.4}) cos={:.4}",
            cnt,
            se / cnt as f64,
            st / cnt as f64,
            sc / cnt as f64
        );
        // NOISE FLOOR (Gary 16.10, вопрос 2): чанковый разброс cos — SE среднего.
        // Критерий A/B: |Δcos| > 3·SE — содержательный; ниже — неразличим.
        let chunk = 8usize;
        let per = cos_vals.len() / chunk;
        if per > 0 {
            let mut chunk_means = Vec::with_capacity(chunk);
            for c in 0..chunk {
                let lo = c * per;
                let m = cos_vals[lo..lo + per].iter().sum::<f64>() / per as f64;
                chunk_means.push(m);
            }
            let mean_cm = chunk_means.iter().sum::<f64>() / chunk as f64;
            let var = chunk_means
                .iter()
                .map(|m| (m - mean_cm) * (m - mean_cm))
                .sum::<f64>()
                / (chunk - 1) as f64;
            let se = (var / chunk as f64).sqrt();
            println!("  noise floor: chunk-cos std={:.4}, SE(среднего)={:.4} → критерий |Δcos| > {:.4} (3·SE)", var.sqrt(), se, 3.0 * se);
        }
    } else {
        println!("  val слишком короткий");
    }
}

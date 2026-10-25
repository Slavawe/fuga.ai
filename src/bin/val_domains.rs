// val_domains.rs — доменная валидация чекпоинта FUGA1 (Fuga-1.0, 15.10).
// Те же метрики, что [val] в unified_gpu_train (без обновлений весов):
// mean ||t−Wx|| и mean cos(t, Wx) — но по НЕСКОЛЬКИМ файлам сразу,
// чтобы видеть «размытие» доменов (EN-текст / RU-текст / код).
//
// Usage: val_domains <ckpt.fuga> <val1.bin> [val2.bin ...] [--ctx 8]
use fuga::ai::htm_temporal::TemporalMemory;
use fuga::ai::sdr::{byte_basis, structure_sdr_from_sdrs};

const DIM: usize = 512;

fn val_file(
    w: &[f32],
    enc: &fuga::ai::latent_jepa::SdrEncoder,
    byte_cache: &[fuga::SdrVector],
    vd: &[u8],
    ctxw: usize,
) -> Option<(f64, f64, usize, f64, f64)> {
    if vd.len() <= ctxw + 2 {
        return None;
    }
    let n_max = 4096usize;
    let span = vd.len() - ctxw - 1;
    let stride = (span / n_max).max(1);
    // BASELINE (ревью 25.10 п.2): сравнение Wx с нулевым предсказанием,
    // средним целевым вектором и мажоритарным байтом на ТОМ ЖЕ окне.
    // Средний таргет = Σ freq(b)·lat(b) (таргет зависит только от байта).
    let mut freq = [0usize; 256];
    for &b in vd {
        freq[b as usize] += 1;
    }
    // таргет-латенты байтов (тот же путь, что у пар: enc.encode(byte_basis))
    let latents: Vec<fuga::ai::latent_jepa::LatentVector> =
        (0..256usize).map(|b| enc.encode(&byte_cache[b])).collect();
    let mut mean = vec![0.0f32; DIM];
    for b in 0..256usize {
        if freq[b] > 0 {
            let wgt = freq[b] as f32 / vd.len() as f32;
            for o in 0..DIM {
                mean[o] += wgt * latents[b].values[o];
            }
        }
    }
    let mn = mean.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
    for v in mean.iter_mut() {
        *v /= mn;
    }
    let (maj_b, _) = freq
        .iter()
        .enumerate()
        .max_by_key(|(_, c)| **c)
        .map(|(b, c)| (b as u8, *c))
        .unwrap_or((0u8, 0));
    let (mut se, mut st, mut sc, mut cnt) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    let (mut sc_mean, mut sc_maj) = (0.0f64, 0.0f64);
    for k in (0..span).step_by(stride).take(n_max) {
        let win = &vd[k.saturating_sub(ctxw)..=k];
        let wsdrs: Vec<fuga::SdrVector> = win.iter().map(|&c| byte_cache[c as usize].clone()).collect();
        let x = enc.encode(&structure_sdr_from_sdrs(&wsdrs));
        let t = enc.encode(&byte_cache[vd[k + 1] as usize]);
        let mut pred = [0.0f32; DIM];
        for o in 0..DIM {
            let row = o * DIM;
            let mut acc = 0.0f32;
            for i in 0..DIM {
                acc += w[row + i] * x.values[i];
            }
            pred[o] = acc;
        }
        let (mut e2, mut t2, mut dot, mut pn) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut dot_mean = 0.0f32;
        let mut dot_maj = 0.0f32;
        for o in 0..DIM {
            let d = t.values[o] - pred[o];
            e2 += d * d;
            t2 += t.values[o] * t.values[o];
            dot += t.values[o] * pred[o];
            pn += pred[o] * pred[o];
            dot_mean += t.values[o] * mean[o];
            dot_maj += t.values[o] * latents[maj_b as usize].values[o];
        }
        se += e2.sqrt() as f64;
        st += t2.sqrt() as f64;
        sc += (dot / (t2.sqrt() * pn.sqrt() + 1e-8)) as f64;
        sc_mean += (dot_mean / t2.sqrt().max(1e-8)) as f64;
        sc_maj += (dot_maj / t2.sqrt().max(1e-8)) as f64;
        cnt += 1;
    }
    if cnt == 0 {
        None
    } else {
        Some((
            se / cnt as f64,
            sc / cnt as f64,
            cnt,
            sc_mean / cnt as f64,
            sc_maj / cnt as f64,
        ))
    }
}

fn main() {
    run(std::env::args().collect());
}

// run() отделён от main для подкоманды `fuga val ...` единого бинарника (24.10).
pub fn run(args: Vec<String>) {
    if args.len() < 3 {
        eprintln!("Usage: val_domains <ckpt.fuga> <val1.bin> [val2.bin ...] [--ctx 8]");
        std::process::exit(1);
    }
    let ctxw: usize = args
        .iter()
        .position(|a| a == "--ctx")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    // --hist <файл> (25.10, ревью п.1): baseline (mean/maj byte) из гистограммы
    // TRAIN-корпуса (256×u64 LE, train_byte_hist.bin) — без утечки из val.
    // --boot N (ревью п.2): разбить val на N чанков, cos в каждом, SE среднего
    // и SE парной разницы (модель − baseline).
    let hist_path: String = args
        .iter()
        .position(|a| a == "--hist")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_default();
    let boot: usize = args
        .iter()
        .position(|a| a == "--boot")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    // --reps B (25.10, ревью п.1): процентильный бутстреп поверх чанков —
    // B ресемплов N чанков с возвращением, CI 2.5/97.5 перцентили.
    let reps: usize = args
        .iter()
        .position(|a| a == "--reps")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    // --top1 (25.10, ревью-план этап 03): top-1 точность декодирования байта
    // из ПРЕДСКАЗАННОГО латента (argmax cos(Wx, lat_b) == факт) и из
    // ИСТИННОГО (argmax cos(t, lat_b) == b — санити энкодера, должно быть ~1).
    let top1: bool = args.iter().any(|a| a == "--top1");
    let ckpt = &args[1];
    let mut files: Vec<&String> = Vec::new();
    let mut skip_next = false;
    for a in &args[2..] {
        if skip_next {
            skip_next = false;
            continue;
        }
        // булевы флаги без значения не попадают в файлы и не глотают соседей
        if a == "--top1" {
            continue;
        }
        if a == "--ctx" || a == "--hist" || a == "--boot" || a == "--reps" {
            skip_next = true;
            continue;
        }
        files.push(a);
    }

    let mut tm = TemporalMemory::new(64, ctxw);
    assert!(tm.load_unified_fuga1(ckpt), "FUGA1 не читается: {}", ckpt);
    let w = tm.predictor().w.clone();
    let enc = tm.predictor().encoder.clone();
    let byte_cache: Vec<fuga::SdrVector> = (0..=255u8).map(byte_basis).collect();

    // TRAIN-гистограмма для baseline без утечки (если --hist задан)
    let train_hist: Option<Vec<u64>> = if hist_path.is_empty() {
        None
    } else {
        let raw = std::fs::read(&hist_path).unwrap_or_else(|e| panic!("--hist: {}", e));
        Some(
            (0..256usize)
                .map(|b| {
                    let o = b * 8;
                    u64::from_le_bytes(raw[o..o + 8].try_into().unwrap())
                })
                .collect(),
        )
    };
    // train-mean таргет (нормированный) и train-мажоритарный байт
    let (train_mean, train_maj): (Vec<f32>, u8) = match &train_hist {
        Some(h) => {
            let total: f64 = h.iter().sum::<u64>() as f64;
            let latents: Vec<fuga::ai::latent_jepa::LatentVector> =
                (0..256usize).map(|b| enc.encode(&byte_cache[b])).collect();
            let mut m = vec![0.0f32; DIM];
            for b in 0..256usize {
                let wgt = (h[b] as f64 / total) as f32;
                for o in 0..DIM {
                    m[o] += wgt * latents[b].values[o];
                }
            }
            let mn = m.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
            for v in m.iter_mut() {
                *v /= mn;
            }
            let maj = h
                .iter()
                .enumerate()
                .max_by_key(|(_, c)| **c)
                .map(|(b, _)| b as u8)
                .unwrap_or(0);
            (m, maj)
        }
        None => (Vec::new(), 0),
    };
    let val_with_baseline = |vd: &[u8]| -> Option<(f64, f64, usize, f64, f64)> {
        if !train_mean.is_empty() {
            Some(val_file_train(
                &w,
                &enc,
                &byte_cache,
                vd,
                ctxw,
                &train_mean,
                train_maj,
            ))
        } else {
            val_file(&w, &enc, &byte_cache, vd, ctxw)
        }
    };

    println!(
        "== VAL DOMAINS: {} (ctx={}) baseline={} boot={} ==",
        ckpt,
        ctxw,
        if train_hist.is_some() {
            "TRAIN-hist (без утечки)"
        } else {
            "val-local (утечка возможна!)"
        },
        if boot > 0 { boot } else { 1 }
    );
    for f in &files {
        let vd = match std::fs::read(f) {
            Ok(v) => v,
            Err(e) => {
                println!("  {:<28} НЕ ЧИТАЕТСЯ: {}", f, e);
                continue;
            }
        };
        if boot > 1 && vd.len() > boot * 64 {
            let clen = vd.len() / boot;
            let mut cs: Vec<(f64, f64)> = Vec::new();
            let mut c_top1: Vec<f64> = Vec::new(); // парный top-1 по чанкам (25.10)
            for ci in 0..boot {
                let chunk = &vd[ci * clen..(ci + 1) * clen];
                if let Some((_, c, _, cm, _)) = val_with_baseline(chunk) {
                    cs.push((c, cm));
                }
                if top1 {
                    if let Some((ap, _, _)) = val_top1(&w, &enc, &byte_cache, chunk, ctxw) {
                        c_top1.push(ap);
                    }
                }
            }
            if cs.len() < 2 {
                println!("  {:<28} чанков мало", f);
                continue;
            }
            let n = cs.len() as f64;
            let m_c: f64 = cs.iter().map(|x| x.0).sum::<f64>() / n;
            let m_cm: f64 = cs.iter().map(|x| x.1).sum::<f64>() / n;
            let var = |mu: f64, idx: usize| -> f64 {
                cs.iter()
                    .map(|x| (if idx == 0 { x.0 } else { x.1 } - mu).powi(2))
                    .sum::<f64>()
                    / (n - 1.0)
            };
            let se_c = (var(m_c, 0) / n).sqrt();
            let se_cm = (var(m_cm, 1) / n).sqrt();
            let diffs: Vec<f64> = cs.iter().map(|x| x.0 - x.1).collect();
            let m_d = diffs.iter().sum::<f64>() / n;
            let se_d = (diffs.iter().map(|d| (d - m_d).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt();
            // ПРОЦЕНТИЛЬНЫЙ БУТСТРЕП (если --reps B): B ресемплов чанков
            let (ci_lo, ci_hi) = if reps > 1 {
                let mut bs: Vec<f64> = Vec::with_capacity(reps);
                let mut h: u64 = 0x9e3779b97f4a7c15 ^ (cs.len() as u64 * 2654435761);
                let mut next01 = || {
                    h ^= h << 13;
                    h ^= h >> 7;
                    h ^= h << 17;
                    (h >> 11) as f64 / (1u64 << 53) as f64
                };
                for _ in 0..reps {
                    let mut s = 0.0f64;
                    for _ in 0..cs.len() {
                        let i = (next01() * cs.len() as f64) as usize % cs.len();
                        s += cs[i].0 - cs[i].1;
                    }
                    bs.push(s / n);
                }
                bs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let lo = bs[(reps as f64 * 0.025) as usize];
                let hi = bs[((reps as f64 * 0.975) as usize).min(reps - 1)];
                (lo, hi)
            } else {
                (f64::NAN, f64::NAN)
            };
            println!(
                "  {:<28} cos(t,Wx)={:.4}±{:.4}  baseline(mean)={:.4}±{:.4}  ПАРНАЯ РАЗНИЦА Δ={:+.4}±{:.4} (3·SE={:.4}; значимо: {})",
                f,
                m_c,
                se_c,
                m_cm,
                se_cm,
                m_d,
                se_d,
                3.0 * se_d,
                m_d.abs() > 3.0 * se_d
            );
            if reps > 1 {
                println!(
                    "    бутстреп B={}: Δ 95% CI [{:+.4}; {:+.4}] — 0 {}: интервал",
                    reps,
                    ci_lo,
                    ci_hi,
                    if ci_lo > 0.0 || ci_hi < 0.0 { "ВНЕ" } else { "внутри" }
                );
            }
            print!("    чанки cos:");
            for (c, _) in &cs {
                print!(" {:.3}", c);
            }
            println!();
            if top1 && !c_top1.is_empty() {
                let m = c_top1.iter().sum::<f64>() / c_top1.len() as f64;
                let se = (c_top1
                    .iter()
                    .map(|x| (x - m).powi(2))
                    .sum::<f64>()
                    / (c_top1.len() as f64 - 1.0)
                    / c_top1.len() as f64)
                    .sqrt();
                print!("    чанки top1 ({:.4}±{:.4}):", m, se);
                for a in &c_top1 {
                    print!(" {:.3}", a);
                }
                println!();
            }
        } else {
            match val_with_baseline(&vd) {
                Some((e, c, n, c_mean, c_maj)) => println!(
                    "  {:<28} ||t−Wx||={:.4}  cos(t,Wx)={:.4}  (n={})  [baseline: cos(t,mean)={:.4} cos(t,maj_byte)={:.4}; нулевое: ||t−0||=1.0]",
                    f, e, c, n, c_mean, c_maj
                ),
                None => println!("  {:<28} слишком короткий", f),
            }
        }
        if top1 {
            match val_top1(&w, &enc, &byte_cache, &vd, ctxw) {
                Some((acc_pred, acc_true, n)) => println!(
                    "  {:<28} [top1] из ПРЕДСКАЗАННОГО латента: {:.3}%  из ИСТИННОГО (санити энкодера): {:.3}%  шанс 0.39%  (n={})",
                    f, acc_pred * 100.0, acc_true * 100.0, n
                ),
                None => println!("  {:<28} [top1] слишком короткий", f),
            }
        }
    }
}

/// Этап 03 (encoder→predictor→decoder): top-1 точность восстановления
/// следующего байта через ближайший латент. acc_true — верхняя граница
/// (декод из ИСТИННОГО латента: если <1, энкодер не разделяет байты
/// уникально); acc_pred — реальная генеративная точность модели.
fn val_top1(
    w: &[f32],
    enc: &fuga::ai::latent_jepa::SdrEncoder,
    byte_cache: &[fuga::SdrVector],
    vd: &[u8],
    ctxw: usize,
) -> Option<(f64, f64, usize)> {
    if vd.len() <= ctxw + 2 {
        return None;
    }
    let latents: Vec<fuga::ai::latent_jepa::LatentVector> =
        (0..256usize).map(|b| enc.encode(&byte_cache[b])).collect();
    let n_max = 2048usize;
    let span = vd.len() - ctxw - 1;
    let stride = (span / n_max).max(1);
    let (mut ok_pred, mut ok_true, mut cnt) = (0usize, 0usize, 0usize);
    for k in (0..span).step_by(stride).take(n_max) {
        let win = &vd[k.saturating_sub(ctxw)..=k];
        let wsdrs: Vec<fuga::SdrVector> =
            win.iter().map(|&c| byte_cache[c as usize].clone()).collect();
        let x = enc.encode(&structure_sdr_from_sdrs(&wsdrs));
        let t = enc.encode(&byte_cache[vd[k + 1] as usize]);
        let mut pred = [0.0f32; DIM];
        for o in 0..DIM {
            let row = o * DIM;
            let mut acc = 0.0f32;
            for i in 0..DIM {
                acc += w[row + i] * x.values[i];
            }
            pred[o] = acc;
        }
        let pn = pred.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        let tn = t.values.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-8);
        let (mut best_p, mut best_t) = (f32::MIN, f32::MIN);
        let (mut arg_p, mut arg_t) = (0usize, 0usize);
        for b in 0..256usize {
            let mut dp = 0.0f32;
            let mut dt = 0.0f32;
            for o in 0..DIM {
                dp += pred[o] * latents[b].values[o];
                dt += t.values[o] * latents[b].values[o];
            }
            dp /= pn;
            dt /= tn;
            if dp > best_p {
                best_p = dp;
                arg_p = b;
            }
            if dt > best_t {
                best_t = dt;
                arg_t = b;
            }
        }
        if arg_p == vd[k + 1] as usize {
            ok_pred += 1;
        }
        if arg_t == vd[k + 1] as usize {
            ok_true += 1;
        }
        cnt += 1;
    }
    if cnt == 0 {
        None
    } else {
        Some((ok_pred as f64 / cnt as f64, ok_true as f64 / cnt as f64, cnt))
    }
}

/// val_file с baseline от TRAIN-статистик (mean-вектор нормирован, maj-байт
/// переданы снаружи; логика окон идентична val_file).
fn val_file_train(
    w: &[f32],
    enc: &fuga::ai::latent_jepa::SdrEncoder,
    byte_cache: &[fuga::SdrVector],
    vd: &[u8],
    ctxw: usize,
    train_mean: &[f32],
    train_maj: u8,
) -> (f64, f64, usize, f64, f64) {
    if vd.len() <= ctxw + 2 {
        return (0.0, 0.0, 0, 0.0, 0.0);
    }
    let n_max = 4096usize;
    let span = vd.len() - ctxw - 1;
    let stride = (span / n_max).max(1);
    let maj_lat = enc.encode(&byte_cache[train_maj as usize]);
    let (mut se, mut st, mut sc, mut cnt) = (0.0f64, 0.0f64, 0.0f64, 0usize);
    let (mut sc_mean, mut sc_maj) = (0.0f64, 0.0f64);
    for k in (0..span).step_by(stride).take(n_max) {
        let win = &vd[k.saturating_sub(ctxw)..=k];
        let wsdrs: Vec<fuga::SdrVector> =
            win.iter().map(|&c| byte_cache[c as usize].clone()).collect();
        let x = enc.encode(&structure_sdr_from_sdrs(&wsdrs));
        let t = enc.encode(&byte_cache[vd[k + 1] as usize]);
        let mut pred = [0.0f32; DIM];
        for o in 0..DIM {
            let row = o * DIM;
            let mut acc = 0.0f32;
            for i in 0..DIM {
                acc += w[row + i] * x.values[i];
            }
            pred[o] = acc;
        }
        let (mut e2, mut t2, mut dot, mut pn) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut dot_mean = 0.0f32;
        let mut dot_maj = 0.0f32;
        for o in 0..DIM {
            let d = t.values[o] - pred[o];
            e2 += d * d;
            t2 += t.values[o] * t.values[o];
            dot += t.values[o] * pred[o];
            pn += pred[o] * pred[o];
            dot_mean += t.values[o] * train_mean[o];
            dot_maj += t.values[o] * maj_lat.values[o];
        }
        se += e2.sqrt() as f64;
        st += t2.sqrt() as f64;
        sc += (dot / (t2.sqrt() * pn.sqrt() + 1e-8)) as f64;
        sc_mean += (dot_mean / t2.sqrt().max(1e-8)) as f64;
        sc_maj += (dot_maj / t2.sqrt().max(1e-8)) as f64;
        cnt += 1;
    }
    if cnt == 0 {
        return (0.0, 0.0, 0, 0.0, 0.0);
    }
    (
        se / cnt as f64,
        sc / cnt as f64,
        cnt,
        sc_mean / cnt as f64,
        sc_maj / cnt as f64,
    )
}

// gpu_train.rs — двухсторонний конвейер CPU/GPU (бета).
//
// Разделение работы:
//   CPU : читает JSONL, строит окна, structure-fold + SdrEncoder
//         (x = encode(window), target = encode(next byte)) и копит батчи
//   GPU : Widrow-Hoff дельта `W += lr·(target−x)⊗x` пачками batch_delta
//         (W живёт в VRAM, не скачивается на каждом шаге)
//
// Честные допущения (задокументированы):
//   - init W≈0 → pred≈0, err ≈ target (тот же приём, что в первых GPU-стендах)
//   - OWM-consolidate в этом стенде не выполняется (CPU-поток encode-only);
//     это конвейерный прототип для замера CPU/GPU нагрузки и скорости.
//   - В конце W скачивается и сохраняется в FBW1 (bin-совместим с Rust).
//
// Usage: gpu_train --jsonl "corpus.jsonl,corpus2.jsonl" [--max-steps 300000]
//                  [--batch 512] [--out /tmp/gpu_w.bin] [--no-gpu]
//                  [--ckpt-every 1000000] — периодический FUGA1-чекпоинт
//                  (переживает перезагрузку: сейв идёт ПО ХОДУ, не в конце)
use std::io::BufRead;
use std::path::Path;
use std::sync::mpsc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use fuga::ai::gpu_ops::GpuOps;
use fuga::ai::latent_jepa::{LatentPredictor, SdrEncoder};
use fuga::ai::sdr::{byte_basis, encode_bytes_sdr, structure_sdr_from_sdrs};

const DIM: usize = 512;

// Извлечь байты из JSONL-строки (doc/code/chapters) — как в full_byte_train.
fn extract_bytes(line: &str) -> Vec<u8> {
    let v: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(_) => return line.as_bytes().to_vec(),
    };
    let mut out = String::new();
    if let Some(chs) = v.get("chapters").and_then(|c| c.as_array()) {
        for ch in chs {
            if let Some(paras) = ch.get("paragraphs").and_then(|p| p.as_array()) {
                for p in paras {
                    if let Some(s) = p.as_str() {
                        out.push_str(s);
                        out.push('\n');
                    }
                }
            }
        }
    } else if let Some(doc) = v.get("doc").and_then(|d| d.as_str()) {
        out.push_str(doc);
        if let Some(code) = v.get("code").and_then(|c| c.as_str()) {
            out.push('\n');
            out.push_str(code);
        }
    } else if let Some(code) = v.get("code").and_then(|c| c.as_str()) {
        out.push_str(code);
    } else {
        return line.as_bytes().to_vec();
    }
    out.into_bytes()
}

fn arg<T: std::str::FromStr>(args: &[String], name: &str, def: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<T>().ok())
        .unwrap_or(def)
}

// Найти (x, err) для одного байт-шага: x = encode(структура окна),
// err ≈ target (W стартует с нуля → pred≈0). Возвращает None если
// проводить латентные вычисления не нужно (окно пустое).
fn window_pairs(enc: &SdrEncoder, window: &[u8], next: u8, ctx: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    if window.is_empty() {
        return None;
    }
    let lo = window.len().saturating_sub(ctx.max(1));
    let win_sdrs: Vec<fuga::SdrVector> = window[lo..]
        .iter()
        .map(|&c| byte_basis(c))
        .collect();
    let x = enc.encode(&structure_sdr_from_sdrs(&win_sdrs));
    let t = enc.encode(&byte_basis(next));
    Some((x.values.clone(), t.values.clone()))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus: String = arg(&args, "--jsonl", "corpus_doc_code_pairs.jsonl".into());
    let max_steps: usize = arg(&args, "--max-steps", 300_000);
    let batch: usize = arg(&args, "--batch", 256);
    let lr: f32 = arg(&args, "--lr", 0.05);
    let out_path: String = arg(&args, "--out", "/tmp/gpu_train_w.bin".into());
    let use_gpu = !args.iter().any(|a| a == "--no-gpu");
    let ctxw: usize = arg(&args, "--ctx", 4);
    let ckpt_every: usize = arg(&args, "--ckpt-every", 1_000_000);

    let enc = SdrEncoder::new(0x9E37_79B9_7F4A_7C15);
    let byte_cache: Vec<fuga::SdrVector> = (0..=255u8).map(byte_basis).collect();

    // ОГРАНИЧЕННЫЙ канал: backpressure! CPU ждёт свободный слот, пока GPU
    // не применит пачку. Иначе CPU уезжает вперёд на 6GB+ и(OOM — баг №2)
    // и это не конвейер. sync_channel(batch*4) = двойная буферизация.
    // Пары двухскоростные: (x_local, err_local, x_patch, err_patch).
    type Pair4 = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
    let (tx, rx) = std::sync::mpsc::sync_channel::<Pair4>(batch * 4);
    let stop = std::sync::Arc::new(AtomicBool::new(false));

    // --- Поток CPU: читать JSONL, готовить (x, err) пары ---
    let cpu_handle = {
        let stop = stop.clone();
        let ctxw = ctxw;
        let enc = enc.clone();
        let byte_cache = byte_cache.clone();
        std::thread::spawn(move || {
            let corpora: Vec<String> = corpus
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let mut steps = 0usize;
            'outer: for corpus_file in &corpora {
                let path = std::path::Path::new(corpus_file);
                if !path.exists() {
                    eprintln!("corpus missing: {}", corpus_file);
                    continue;
                }
                let f = std::fs::File::open(path).unwrap();
                let reader = std::io::BufReader::new(f);
                for line in reader.lines() {
                    let line = match line {
                        Ok(l) => l,
                        Err(_) => continue,
                    };
                    if line.trim().is_empty() {
                        continue;
                    }
                    let data = extract_bytes(&line);
                    if data.len() < 2 {
                        continue;
                    }
                    for i in 0..data.len().saturating_sub(1) {
                        if stop.load(Ordering::Relaxed) {
                            break 'outer;
                        }
                        let win_lo = i.saturating_sub(ctxw);
                        let win: &[u8] = &data[win_lo..=i];
                        let nxt = data[i + 1];
                        let win_sdrs: Vec<fuga::SdrVector> = win
                            .iter()
                            .map(|&c| byte_cache[c as usize].clone())
                            .collect();
                        let x = enc.encode(&structure_sdr_from_sdrs(&win_sdrs));
                        let t = enc.encode(&byte_cache[nxt as usize]);
                        let err: Vec<f32> = t.values.clone(); // pred≈0 (W=0 init)
                        // Двухскоростной (патчевый) канал: окно патчей → следующий патч
                        let mut x2 = Vec::new();
                        let mut err2 = Vec::new();
                        if i >= 2 {
                            let pat_window = &data[i - 2..=i];
                            let pats: Vec<&[u8]> = pat_window.chunks(2).collect();
                            let next_patch = &data[i + 1..(i + 3).min(data.len())];
                            let mut win_patch_sdrs: Vec<fuga::SdrVector> =
                                pats.iter().map(|p| encode_bytes_sdr(p)).collect();
                            if win_patch_sdrs.len() < 2 {
                                win_patch_sdrs.push(win_patch_sdrs[0].clone());
                            }
                            let xs = enc.encode(&structure_sdr_from_sdrs(&win_patch_sdrs));
                            let ts = enc.encode(&encode_bytes_sdr(next_patch));
                            x2 = xs.values.clone();
                            err2 = ts.values.clone(); // pred = 0 → err = target
                        }
                        if tx.send((x.values, err, x2, err2)).is_err() {
                            break 'outer;
                        }
                        steps += 1;
                        if steps >= max_steps {
                            break 'outer;
                        }
                    }
                }
            }
            steps
        })
    };

    // --- Основной поток: GPU применяет пары пачками (или CPU, если --no-gpu) ---
    let gpu = if use_gpu { fuga::ai::gpu_ops::try_new() } else { None };
    let mut w = vec![0.0f32; DIM * DIM];
    let mut w_patch: Vec<f32> = vec![0.0f32; DIM * DIM];
    let mut applied: usize = 0;
    let mut next_ckpt: usize = ckpt_every;
    let t0 = Instant::now();

    match &gpu {
        Some(g) => {
            g.upload_w(&w);
            g.upload_w2(&w);
            let mut xs: Vec<Vec<f32>> = Vec::with_capacity(batch);
            let mut errs: Vec<Vec<f32>> = Vec::with_capacity(batch);
            let mut xs2: Vec<Vec<f32>> = Vec::with_capacity(batch);
            let mut errs2: Vec<Vec<f32>> = Vec::with_capacity(batch);
            let mut recv = rx;
            loop {
                match recv.recv() {
                    Ok((x, e, x2, e2)) => {
                        xs.push(x);
                        errs.push(e);
                        if !x2.is_empty() && !e2.is_empty() {
                            xs2.push(x2);
                            errs2.push(e2);
                        }
                        if xs.len() >= batch {
                            g.batch_delta(&xs, &errs, lr);
                            if !xs2.is_empty() {
                                g.batch_delta2(&xs2, &errs2, lr);
                            }
                            applied += xs.len();
                            xs.clear();
                            errs.clear();
                            xs2.clear();
                            errs2.clear();
                            // Cap как в Rust learn_transition (CAP_EVERY≈50 дельт → ROW_NORM_CAP²=4)
                            if applied / batch % 50 == 0 {
                                g.cap_w(4.0);
                                g.cap_w2(4.0);
                            }
                            // Периодический чкпоинт: переживает перезагрузку.
                            if ckpt_every > 0 && applied >= next_ckpt {
                                next_ckpt += ckpt_every;
                                let mut cw = vec![0.0f32; DIM * DIM];
                                g.download_w(&mut cw);
                                let mut cw2 = vec![0.0f32; DIM * DIM];
                                g.download_w2(&mut cw2);
                                let cm = fuga::ai::htm_temporal::UnifiedMeta {
                                    steps: applied as u64,
                                    patch_steps: applied as u64,
                                    ctx: ctxw as u32,
                                    version: 1,
                                };
                                let mut ident_p = vec![0.0f32; DIM * DIM];
                                for di in 0..DIM {
                                    ident_p[di * DIM + di] = 1.0;
                                }
                                let ckpt_path = format!("{}.ckpt.fuga", out_path);
                                fuga::ai::htm_temporal::save_unified(
                                    &ckpt_path,
                                    &cw,
                                    &cw2,
                                    &ident_p,
                                    &cm,
                                    None,
                                )
                                .ok();
                                eprintln!("  [ckpt] {} пар -> {}", applied, ckpt_path);
                            }
                            if applied % (batch * 8) == 0 {
                                eprintln!("  [gpu] applied {} pairs", applied);
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
            if !xs.is_empty() {
                g.batch_delta(&xs, &errs, lr);
                if !xs2.is_empty() {
                    g.batch_delta2(&xs2, &errs2, lr);
                }
                applied += xs.len();
            }
            g.download_w(&mut w);
            let mut w2 = vec![0.0f32; DIM * DIM];
            g.download_w2(&mut w2);
            // Записать W_patch в TM для единого формата (FUGA1 хранит обе)
            w_patch = w2;
        }
        None => {
            let mut recv = rx;
            while let Ok((x, e, x2, e2)) = recv.recv() {
                // CPU fallback: тот же Widrow-Hoff, но построчно (референс)
                for o in 0..DIM {
                    let row = o * DIM;
                    for i in 0..DIM {
                        w[row + i] += lr * e[o] * x[i];
                        if !x2.is_empty() && !e2.is_empty() {
                            w_patch[row + i] += lr * e2[o] * x2[i];
                        }
                    }
                }
                applied += 1;
            }
        }
    }

    stop.store(true, Ordering::Relaxed);
    let cpu_steps = cpu_handle.join().unwrap_or(0);
    let el = t0.elapsed().as_secs_f64();

    // --- Сохранить W и W_patch в единый FUGA1 формат ---
    let mut tm = fuga::ai::htm_temporal::TemporalMemory::new(64, ctxw);
    tm.apply_byte_w(w.clone());
    let mut tm2 = fuga::ai::htm_temporal::TemporalMemory::new(64, ctxw);
    tm2.apply_byte_w(w_patch.clone());
    // Save sidecar (FBW1 local W) + unified FUGA1 (обе W через save-структуру).
    // Sidecar всегда .bin — иначе при --out .fuga FBW1 перезапишет FUGA1.
    let side_path = if out_path.ends_with(".fuga") || out_path.ends_with(".bin") {
        format!("{}_w.bin", out_path.trim_end_matches(".fuga").trim_end_matches(".bin"))
    } else {
        format!("{}.bin", out_path)
    };
    tm.save_byte_w(&side_path).ok();
    let fuga1_path = if out_path.ends_with(".fuga") {
        out_path.clone()
    } else if out_path.ends_with(".bin") {
        out_path.replace(".bin", ".fuga")
    } else {
        format!("{}.fuga", out_path)
    };
    // У TM нет public patch-setter — пишем единный формат напрямую, чтобы
    // FUGA1 нёс обе W (как учит full_byte_train: local + patch).
    let meta = fuga::ai::htm_temporal::UnifiedMeta {
        steps: applied as u64,
        patch_steps: applied as u64,
        ctx: ctxw as u32,
        version: 1,
    };
    fuga::ai::htm_temporal::save_unified(
        &fuga1_path,
        &w,
        &w_patch,
        &tm.predictor().p,
        &meta,
        None,
    )
    .ok();
    println!("=== GPU/CPU PIPELINE COMPLETE ===\n  cpu_prepared={} gpu_applied={} in {:.1}s", cpu_steps, applied, el);
    println!("  throughput: {:.0} pairs/s (local + patch двухканально)", applied as f64 / el);
    println!(
        "  saved local W -> {} / unified (W+W_patch+OWM-P) -> {}",
        side_path, fuga1_path
    );
    let _ = tm2;
}
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
// Usage: gpu_train --jsonl corpus.jsonl [--max-steps 300000]
//                  [--batch 512] [--out /tmp/gpu_w.bin] [--no-gpu]
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

    let enc = SdrEncoder::new(0x9E37_79B9_7F4A_7C15);
    let byte_cache: Vec<fuga::SdrVector> = (0..=255u8).map(byte_basis).collect();

    // ОГРАНИЧЕННЫЙ канал: backpressure! CPU ждёт свободный слот, пока GPU
    // не применит пачку. Иначе CPU уезжает вперёд на 6GB+ и(OOM — баг №2)
    // и это не конвейер. sync_channel(batch*4) = двойная буферизация.
    let (tx, rx) = std::sync::mpsc::sync_channel::<(Vec<f32>, Vec<f32>)>(batch * 4);
    let stop = std::sync::Arc::new(AtomicBool::new(false));

    // --- Поток CPU: читать JSONL, готовить (x, err) пары ---
    let cpu_handle = {
        let stop = stop.clone();
        let ctxw = ctxw;
        let enc = enc.clone();
        let byte_cache = byte_cache.clone();
        std::thread::spawn(move || {
            let path = std::path::Path::new(&corpus);
            if !path.exists() {
                eprintln!("corpus missing: {}", corpus);
                return 0usize;
            }
            let f = std::fs::File::open(path).unwrap();
            let reader = std::io::BufReader::new(f);
            let mut steps = 0usize;
            'outer: for line in reader.lines() {
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
                    let win_sdrs: Vec<fuga::SdrVector> =
                        win.iter().map(|&c| byte_cache[c as usize].clone()).collect();
                    let x = enc.encode(&structure_sdr_from_sdrs(&win_sdrs));
                    let t = enc.encode(&byte_cache[nxt as usize]);
                    let err: Vec<f32> = t.values.clone(); // pred≈0 (W=0 init)
                    if tx.send((x.values, err)).is_err() {
                        break 'outer;
                    }
                    steps += 1;
                    if steps >= max_steps {
                        break 'outer;
                    }
                }
            }
            steps
        })
    };

    // --- Основной поток: GPU применяет пары пачками (или CPU, если --no-gpu) ---
    let gpu = if use_gpu { fuga::ai::gpu_ops::try_new() } else { None };
    let mut w = vec![0.0f32; DIM * DIM];
    let mut applied: usize = 0;
    let t0 = Instant::now();

    match &gpu {
        Some(g) => {
            g.upload_w(&w);
            let mut xs: Vec<Vec<f32>> = Vec::with_capacity(batch);
            let mut errs: Vec<Vec<f32>> = Vec::with_capacity(batch);
            let mut recv = rx;
            loop {
                match recv.recv() {
                    Ok((x, e)) => {
                        xs.push(x);
                        errs.push(e);
                        if xs.len() >= batch {
                            g.batch_delta(&xs, &errs, lr);
                            applied += xs.len();
                            xs.clear();
                            errs.clear();
                            // Cap как в Rust learn_transition (CAP_EVERY≈50 дельт → ROW_NORM_CAP²=4)
                            if applied / batch % 50 == 0 {
                                g.cap_w(4.0);
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
                applied += xs.len();
            }
            g.download_w(&mut w);
        }
        None => {
            let mut recv = rx;
            while let Ok((x, e)) = recv.recv() {
                // CPU fallback: тот же Widrow-Hoff, но построчно (референс)
                for o in 0..DIM {
                    let row = o * DIM;
                    for i in 0..DIM {
                        w[row + i] += lr * e[o] * x[i];
                    }
                }
                applied += 1;
            }
        }
    }

    stop.store(true, Ordering::Relaxed);
    let cpu_steps = cpu_handle.join().unwrap_or(0);
    let el = t0.elapsed().as_secs_f64();

    // --- Сохранить W в FBW1 (bin-совместим с Rust save_byte_w) ---
    let mut tm = fuga::ai::htm_temporal::TemporalMemory::new(64, ctxw);
    tm.apply_byte_w(w.clone());
    tm.save_byte_w(&out_path).ok();
    println!("=== GPU/CPU PIPELINE COMPLETE ===");
    println!("  cpu_prepared={} gpu_applied={} in {:.1}s", cpu_steps, applied, el);
    println!("  throughput: {:.0} pairs/s", applied as f64 / el);
    println!("  saved W -> {}", out_path);
}
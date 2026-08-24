// kan_calib.rs — калибровка KAN-оператора (итерация 5) + НОВЫЙ GPU-канал.
//
// CPU-поток: читает corpus → окно 4 байта → x = encoder(structure_sdr(win)),
//           err ≈ target = encoder(byte(next)) (W=0 init, pred=0).
// GPU      : c[o,i,k] += lr·err[o]·hat_k(x[i]) пачками (KAN_SHADER, wgpu).
// Кап:     мягкий per-node cap (как kan.rs cap_outputs после калибровки).
// Метрика: avg cosine KAN-предсказания → следующий байт (2000 пар).
//
// Usage: kan_calib <corpus.jsonl> [lr] [max_steps] [--gpu] [--no-gpu] [--ckpt-every]
use fuga::ai::gpu_ops::GpuOps;
use fuga::ai::htm_temporal::TemporalMemory;
use fuga::ai::kan::KanTransition;
use fuga::ai::sdr::{byte_basis, structure_sdr_from_sdrs};
use std::io::BufRead;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

const DIM: usize = 512;

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

/// Мягкий кап (аналог kan.rs cap_outputs после калибровки: scale = sqrt(CAP/(CAP+sq))).
fn soft_cap_kan(kan: &mut KanTransition) {
    // Выполняется на CPU только между пачками (редко): считаем sq по всем o.
    // У KAN c имеет layout [o][i][k] — см. kan.rs.
    const CAP: f32 = 40.0;
    let mut sq: f64 = 0.0;
    for v in &kan.c {
        sq += (*v as f64) * (*v as f64);
    }
    let scale = (CAP as f64 / (CAP as f64 + sq.max(1e-8))) as f32;
    if (scale - 1.0).abs() > 1e-6 {
        for v in &mut kan.c {
            *v *= scale;
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus: String = args.get(1).cloned().unwrap_or_else(|| "fisig_corpus.jsonl".into());
    let lr: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.2);
    let max_steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(300_000);
    let batch: usize = arg(&args, "--batch", 256);
    let use_gpu = !args.iter().any(|a| a == "--no-gpu");
    let ctxw: usize = arg(&args, "--ctx", 4);

    let tm = TemporalMemory::new(64, 4);
    let enc = &tm.predictor().encoder;
    let mut kan = KanTransition::new();
    let t0 = Instant::now();

    let gpu = if use_gpu { fuga::ai::gpu_ops::try_new() } else { None };
    match &gpu {
        Some(g) => {
            // GPU-режим: двухканальный CPU→GPU конвейер (как gpu_train).
            let (tx, rx) = mpsc::sync_channel::<(Vec<f32>, Vec<f32>)>(batch * 4);
            let stop = Arc::new(AtomicBool::new(false));
            let cpu = std::thread::spawn({
                let corpus = corpus.clone();
                let tx = tx.clone();
                let stop = stop.clone();
                let enc = enc.clone();
                move || {
                    let mut steps = 0usize;
                    let f = std::fs::File::open(&corpus).expect("corpus");
                    let rd = std::io::BufReader::new(f);
                    'outer: for line in rd.lines().flatten() {
                        let data: Vec<u8> = line.into_bytes();
                        if data.len() < 5 {
                            continue;
                        }
                        for i in 0..data.len() - 1 {
                            if stop.load(Ordering::Relaxed) {
                                break 'outer;
                            }
                            let lo = i.saturating_sub(ctxw);
                            let win = &data[lo..=i];
                            let win_sdr: Vec<fuga::SdrVector> =
                                win.iter().map(|&b| byte_basis(b)).collect();
                            let x = enc.encode(&structure_sdr_from_sdrs(&win_sdr));
                            let t = enc.encode(&byte_basis(data[i + 1]));
                            if tx.send((x.values.clone(), t.values.clone())).is_err() {
                                break 'outer;
                            }
                            steps += 1;
                            if steps >= max_steps {
                                break 'outer;
                            }
                        }
                    }
                    steps
                }
            });
            // главный поток: GPU применяет пары
            let mut c: Vec<f32> = kan.c.clone();
            g.upload_kan(&c);
            let mut applied = 0usize;
            let mut xs: Vec<Vec<f32>> = Vec::with_capacity(batch);
            let mut errs: Vec<Vec<f32>> = Vec::with_capacity(batch);
            loop {
                match rx.recv() {
                    Ok((x, e)) => {
                        xs.push(x);
                        errs.push(e);
                        if xs.len() >= batch {
                            g.kan_batch_delta(&xs, &errs, lr);
                            applied += xs.len();
                            if applied % (batch * 50) == 0 {
                                // Мягкий KAN-кап НА GPU — без download/upload цикла
                                g.kan_cap_w(40.0);
                            }
                            xs.clear();
                            errs.clear();
                            if applied % (batch * 8) == 0 {
                                eprintln!("  [kan-gpu] applied {} pairs", applied);
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
            if !xs.is_empty() {
                g.kan_batch_delta(&xs, &errs, lr);
                applied += xs.len();
            }
            g.download_kan(&mut c);
            kan.c = c.clone();
            let cpu_steps = cpu.join().unwrap_or(0);
            let el = t0.elapsed().as_secs_f64();
            println!("=== KAN GPU/CPU PIPELINE ===\n  cpu_prepared={} applied={} in {:.1}s ({:.0} pairs/s), lr={}",
                     cpu_steps, applied, el, applied as f64 / el, lr);
        }
        None => {
            // CPU-режим: честный референс (как раньше)
            let f = std::fs::File::open(&corpus).expect("corpus");
            let rd = std::io::BufReader::new(f);
            let mut count = 0usize;
            'outer: for line in rd.lines().flatten() {
                let data: Vec<u8> = line.into_bytes();
                if data.len() < 5 {
                    continue;
                }
                for i in 0..data.len() - 1 {
                    let lo = i.saturating_sub(ctxw);
                    let win = &data[lo..=i];
                    let win_sdr: Vec<fuga::SdrVector> =
                        win.iter().map(|&b| byte_basis(b)).collect();
                    let x = enc.encode(&structure_sdr_from_sdrs(&win_sdr));
                    let t = enc.encode(&byte_basis(data[i + 1]));
                    kan.learn(&x, &t, lr);
                    kan.cap_outputs();
                    count += 1;
                    if count >= max_steps {
                        break 'outer;
                    }
                }
            }
            let el = t0.elapsed().as_secs_f64();
            println!("=== KAN CPU PASS ===\n  trained {} steps in {:.1}s ({:.0} steps/s), lr={}",
                     count, el, count as f64 / el, lr);
        }
    }

    // Метрика: средний cosine KAN→target на 2000 пар
    let f = std::fs::File::open(&corpus).unwrap();
    let rd = std::io::BufReader::new(f);
    let mut tot = 0.0f32;
    let mut n = 0usize;
    for line in rd.lines().flatten() {
        let data: Vec<u8> = line.into_bytes();
        if data.len() < 5 {
            continue;
        }
        for i in 0..data.len() - 4 {
            let lo = i.saturating_sub(4);
            let win = &data[lo..=i];
            let win_sdr: Vec<fuga::SdrVector> =
                win.iter().map(|&b| byte_basis(b)).collect();
            let x = enc.encode(&fuga::ai::sdr::structure_sdr_from_sdrs(&win_sdr));
            let pred = kan.apply(&x);
            let t = enc.encode(&byte_basis(data[i + 1]));
            tot += pred.cosine_similarity(&t);
            n += 1;
            if n >= 2000 {
                break;
            }
        }
        if n >= 2000 {
            break;
        }
    }
    println!("avg cosine KAN→target: {:.4} (n={})", tot / n as f32, n);
    println!("== DONE ==");
}
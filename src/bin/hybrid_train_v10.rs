// hybrid_train_v10.rs — CPU-трейнер на HybridCore::step() (Фаза 5).
//
// Единый гибридный контур VSA + H-JEPA + OWM + KAN через ОДИН вызов
// HybridCore::step() вместо раздельных g.hybrid_step/hybrid_step2.
//
// Usage:
//   hybrid_train_v10 --jsonl "a.jsonl,b.jsonl" --max-steps 1500000
//                    --out fuga_hybrid_v10.fuga [--ckpt-every 500000]
//                    [--lr-w 0.05] [--lr-patch 0.1] [--lr-kan 0.3] [--ctx 4]
//
// Честный CPU-путь: реально исполняется на этой машине (без CUDA).

use fuga::ai::vsa_jepa_kan::HybridCore;
use fuga::ai::latent_jepa::{LatentVector, SdrEncoder, LATENT_DIM};
use fuga::ai::sdr::{byte_basis, encode_bytes_sdr, structure_sdr_from_sdrs};
use std::io::{BufRead, Write};
use std::time::Instant;

fn arg<T: std::str::FromStr>(args: &[String], name: &str, def: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<T>().ok())
        .unwrap_or(def)
}

fn extract_bytes(line: &str) -> Vec<u8> {
    let v: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(_) => return line.as_bytes().to_vec(),
    };
    let mut out = String::new();
    if let Some(doc) = v.get("doc").and_then(|d| d.as_str()) {
        out.push_str(doc);
        if let Some(code) = v.get("code").and_then(|c| c.as_str()) {
            out.push('\n');
            out.push_str(code);
        }
    } else if let Some(code) = v.get("code").and_then(|c| c.as_str()) {
        out.push_str(code);
    } else if let Some(text) = v.get("text").and_then(|t| t.as_str()) {
        out.push_str(text);
    } else {
        return line.as_bytes().to_vec();
    }
    out.into_bytes()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus: String = arg(&args, "--jsonl", "corpus.jsonl".into());
    let max_steps: usize = arg(&args, "--max-steps", 1_500_000);
    let lr_w: f32 = arg(&args, "--lr-w", 0.05);
    let lr_patch: f32 = arg(&args, "--lr-patch", 0.1);
    let lr_kan: f32 = arg(&args, "--lr-kan", 0.3);
    let ctxw: usize = arg(&args, "--ctx", 4);
    let patch_ctx: usize = arg(&args, "--patch-ctx", 8);
    let ckpt_every: usize = arg(&args, "--ckpt-every", 500_000);
    let out_path: String = arg(&args, "--out", "/tmp/fuga_hybrid_v10.fuga".into());
    let beta_vsa: f32 = arg(&args, "--beta-vsa", 0.0);
    let alpha_kan: f32 = arg(&args, "--alpha-kan", 1.0);

    // Энкодер: тот же seed, что декодер TM (0xF03D_C0DE) — иначе чужой базис.
    let tm = fuga::ai::htm_temporal::TemporalMemory::new(64, ctxw);
    let enc = tm.predictor().encoder.clone();
    let enc_patch = enc.clone();
    let byte_cache: Vec<fuga::SdrVector> = (0..=255u8).map(byte_basis).collect();

    let mut core = HybridCore::new(beta_vsa, alpha_kan);
    println!("=== HYBRID_TRAIN_V10 (HybridCore::step, CPU) ===");
    println!("  corpus={} max_steps={} lr_w={} lr_patch={} lr_kan={}", corpus, max_steps, lr_w, lr_patch, lr_kan);
    println!("  ctx={} patch_ctx={} beta_vsa={} alpha_kan={}", ctxw, patch_ctx, beta_vsa, alpha_kan);

    let corpora: Vec<String> = corpus.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    let start = Instant::now();
    let mut applied = 0usize;
    let mut consolidations = 0usize;
    let mut replay: Vec<LatentVector> = Vec::new();
    let mut next_ckpt = ckpt_every;
    // Буфер последних (x,t) для метрики остатка.
    let mut probe: std::collections::VecDeque<(LatentVector, LatentVector)> = std::collections::VecDeque::new();

    'outer: for corpus_file in &corpora {
        let path = std::path::Path::new(corpus_file);
        if !path.exists() {
            eprintln!("corpus missing: {}", corpus_file);
            continue;
        }
        let f = std::fs::File::open(path).unwrap();
        let reader = std::io::BufReader::new(f);
        for line in reader.lines() {
            let line = match line { Ok(l) => l, Err(_) => continue };
            if line.trim().is_empty() { continue; }
            let data = extract_bytes(&line);
            if data.len() < 2 || data.len() > 65_536 { continue; }

            for i in 0..data.len().saturating_sub(1) {
                let win_lo = i.saturating_sub(ctxw);
                let win: &[u8] = &data[win_lo..=i];
                let nxt = data[i + 1];
                let win_sdrs: Vec<fuga::SdrVector> = win.iter().map(|&c| byte_cache[c as usize].clone()).collect();
                let z_ctx = enc.encode(&structure_sdr_from_sdrs(&win_sdrs));
                let z_target = enc.encode(&byte_cache[nxt as usize]);

                // Патчевый канал (два-байтовые патчи, окно patch_ctx).
                let pp = (i + 1) / 2;
                let z_patch: Option<(LatentVector, LatentVector)> = if i + 3 <= data.len() && pp >= patch_ctx {
                    let mut pw: Vec<&[u8]> = Vec::with_capacity(patch_ctx);
                    for k in 0..patch_ctx {
                        pw.push(&data[(pp - patch_ctx + k) * 2..(pp - patch_ctx + k + 1) * 2]);
                    }
                    let next_patch = &data[pp * 2..(pp + 1) * 2];
                    let win_p: Vec<fuga::SdrVector> = pw.iter().map(|p| encode_bytes_sdr(p)).collect();
                    let xs = enc_patch.encode(&structure_sdr_from_sdrs(&win_p));
                    let ts = enc_patch.encode(&encode_bytes_sdr(next_patch));
                    Some((xs, ts))
                } else { None };

                let zp_ref = z_patch.as_ref().map(|(c, t)| (c, t));
                let (err_w, _err_kan, _err_p) = core.step(&z_ctx, &z_target, None, zp_ref, lr_w, lr_kan, lr_patch);

                // ReplayBuffer для OWM: аномальные (большой err) направления.
                if err_w > 0.5 {
                    replay.push(z_ctx.clone());
                    if replay.len() > 64 { replay.remove(0); }
                }
                // Метрика остатка.
                probe.push_back((z_ctx.clone(), z_target.clone()));
                if probe.len() > 64 { probe.pop_front(); }

                applied += 1;

                // OWM-консолидация (Sleep) раз в 2048 шагов.
                if applied % 2048 == 0 {
                    let dirs: Vec<LatentVector> = replay.iter().take(16).cloned().collect();
                    if !dirs.is_empty() {
                        core.consolidate(&dirs, 0.1);
                        consolidations += 1;
                        replay.clear();
                    }
                }

                // Прогресс.
                if applied % 50_000 == 0 {
                    let el = start.elapsed().as_secs_f32();
                    let pps = applied as f32 / el;
                    print!("\r  step {}/{} ({:.0} pairs/s, {} consolid.)   ", applied, max_steps, pps, consolidations);
                    std::io::stdout().flush().ok();
                }

                // Чекпоинт.
                if ckpt_every > 0 && applied >= next_ckpt {
                    next_ckpt += ckpt_every;
                    save_ckpt(&core, applied, ctxw, &format!("{}.ckpt", out_path));
                    // Метрика остатка на чекпоинте.
                    let mut me = 0.0f32; let mut mt = 0.0f32; let mut n = 0;
                    for (xv, tv) in probe.iter() {
                        let mut sq = 0.0f32; let mut tsq = 0.0f32;
                        let mut pred = vec![0.0f32; LATENT_DIM];
                        for o in 0..LATENT_DIM {
                            let row = o * LATENT_DIM;
                            let mut acc = 0.0f32;
                            for k in 0..LATENT_DIM { acc += core.w_local[row + k] * xv.values[k]; }
                            pred[o] = acc;
                        }
                        for o in 0..LATENT_DIM { let d = tv.values[o] - pred[o]; sq += d*d; tsq += tv.values[o]*tv.values[o]; }
                        me += sq.sqrt(); mt += tsq.sqrt(); n += 1;
                    }
                    if n > 0 {
                        println!("\n[ckpt] {} пар: ||t-Wx||_ср={:.4} (||t||={:.4}) consolid={}",
                                 applied, me / n as f32, mt / n as f32, consolidations);
                    }
                }

                if applied >= max_steps { break 'outer; }
            }
        }
    }

    let el = start.elapsed().as_secs_f32();
    save_ckpt(&core, applied, ctxw, &out_path);
    let wl_norm: f32 = core.w_local.iter().map(|v| v*v).sum::<f32>().sqrt();
    let wp_norm: f32 = core.w_patch.iter().map(|v| v*v).sum::<f32>().sqrt();
    let p_diag: f32 = (0..LATENT_DIM).map(|i| core.p_owm[i*LATENT_DIM+i]).sum::<f32>() / LATENT_DIM as f32;
    println!("\n=== COMPLETE ===");
    println!("  {} пар за {:.1}s ({:.0} pairs/s)", applied, el, applied as f32 / el);
    println!("  |W_local|={:.3} |W_patch|={:.3} P_owm_diag={:.4} consolid={}", wl_norm, wp_norm, p_diag, consolidations);
    println!("  файл: {}", out_path);
}

/// Сохранение весов HybridCore в FUGA1 (tag=1 LOCAL_W, tag=2 PATCH_W, tag=3 OWM_P, tag=4 META).
fn save_ckpt(core: &HybridCore, steps: usize, ctxw: usize, path: &str) {
    use fuga::ai::htm_temporal::{save_unified_with_kan, UnifiedMeta};
    let meta = UnifiedMeta { steps: steps as u64, patch_steps: steps as u64, ctx: ctxw as u32, version: 10 };
    // KAN веса (fast_kan.weights) не сериализуем в старый формат KAN_C (другой layout);
    // сохраняем W_local + W_patch + P_owm — ядро HybridCore.
    save_unified_with_kan(path, &core.w_local, &core.w_patch, &core.p_owm, &meta, None, None, None).ok();
}

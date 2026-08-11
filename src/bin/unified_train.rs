// unified_train.rs — ЕДИНЫЙ КОНТУР «все технологии → один файл».
//
// В одном прогоне над ВСЕМИ корпусами (код + тексты) обучаются и
// сохраняются в один FUGA1-файл:
//   1. LOCAL W   — локальный байтовый Widrow-Hoff (частотные биграммы)
//   2. KAN_C     — сплайны на остатке (нелинейные структурные аттракторы)
//   3. PATCH_W   — глобальный патчевый W_patch (two-speed), патчи 2 байта
//   4. OWM_P     — Woodbury-консолидация ключевых направлений
//   5. META      — steps / patch_steps / ctx / version
// После обучения — декоды ОДНИМ чекпоинтом и текстовыми, и код-сидами
// всеми декодерами главного пути (naive, entropy-BLT, hybrid W+KAN).
//
// Usage: unified_train [--jsonl a,b,c] [--max-steps N] [--patch-lr ..]
//                      [--out model.fuga] [--text-seed "..."] [--code-seed "..."]
use fuga::ai::htm_temporal::{save_unified_with_kan, TemporalMemory, UnifiedMeta};
use fuga::ai::hybrid::HybridTransition;
use fuga::ai::tm_generate::{
    tm_generate_hybrid, tm_generate_latent_bytes, tm_generate_two_speed_entropy,
};
use std::collections::HashSet;
use std::io::BufRead;
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpora: Vec<String> = args
        .iter()
        .position(|a| a == "--jsonl")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
        .unwrap_or_else(|| {
            vec![
                "fisig_corpus.jsonl".into(),
                "corpus_doc_code_pairs.jsonl".into(),
                "training_stack.jsonl".into(),
                "corpus.jsonl".into(),
            ]
        });
    let max_steps: usize = arg(&args, "--max-steps", 200_000);
    let patch_len: usize = arg(&args, "--patch-len", 2);
    let lr_w: f32 = arg(&args, "--lr-w", 0.05);
    let lr_kan: f32 = arg(&args, "--lr-kan", 0.3);
    let lr_patch: f32 = arg(&args, "--lr-patch", 0.1);
    let alpha: f32 = arg(&args, "--alpha", 1.0);
    let ctxw: usize = arg(&args, "--ctx", 4);
    let ckpt_every: usize = arg(&args, "--ckpt-every", 0); // 0 = сохранить только в конце
    let out_path: String = arg(&args, "--out", "/tmp/unified.fuga".into());
    let code_seed: Vec<u8> = args
        .iter()
        .position(|a| a == "--code-seed")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_bytes().to_vec())
        .unwrap_or_else(|| b"fn main() {".to_vec());
    let text_seed: Vec<u8> = args
        .iter()
        .position(|a| a == "--text-seed")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_bytes().to_vec())
        .unwrap_or_else(|| b"the force of gravity is".to_vec());

    let t0 = Instant::now();
    let mut hyb = HybridTransition::new();
    let mut tm = TemporalMemory::new(64, ctxw);

    // Патч-словарь (2-байтовые патчи, cap 512) из тех же корпусов.
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut patch_vocab: Vec<Vec<u8>> = Vec::new();
    for corp in &corpora {
        let Ok(f) = std::fs::File::open(corp) else { continue };
        let rd = std::io::BufReader::new(f);
        for line in rd.lines().flatten() {
            let bytes = line.as_bytes();
            for w in bytes.windows(patch_len) {
                if seen.insert(w.to_vec()) && patch_vocab.len() < 5000 {
                    patch_vocab.push(w.to_vec());
                }
            }
            if patch_vocab.len() >= 5000 {
                break;
            }
        }
        if patch_vocab.len() >= 5000 {
            break;
        }
    }
    println!("patch_vocab={} (patch_len={})", patch_vocab.len(), patch_len);

    // ── ПРОХОД 1+2: байтовые W+KAN и патчевый W_patch одновременно.
    let mut steps = 0usize;
    let mut patch_steps = 0usize;
    let mut bytes_seen = 0usize;
    'outer: for corp in &corpora {
        let Ok(f) = std::fs::File::open(corp) else { continue };
        let rd = std::io::BufReader::new(f);
        for line in rd.lines().flatten() {
            let data = extract_bytes(&line);
            if data.len() < ctxw + 1 {
                continue;
            }
            // Локальный байтовый W + KAN на остатка.
            for i in 0..data.len() - 1 {
                let lo = i.saturating_sub(ctxw);
                let win = &data[lo..=i];
                let enc = &tm.predictor().encoder;
                hyb.learn_pair(enc, win, data[i + 1], lr_w, lr_kan);
                steps += 1;
                bytes_seen += 1;
                if ckpt_every > 0 && steps % ckpt_every == 0 {
                    tm.apply_byte_w(hyb.w.clone());
                    let meta_c = UnifiedMeta {
                        steps: steps as u64,
                        patch_steps: patch_steps as u64,
                        ctx: ctxw as u32,
                        version: 2,
                    };
                    if save_unified_with_kan(
                        &out_path,
                        &hyb.w,
                        &tm.patch_predictor().w,
                        &tm.predictor().p,
                        &meta_c,
                        None,
                        None,
                        Some(&hyb.kan.c)
                    )
                    .is_ok()
                    {
                        println!(
                            "  [ckpt] {} byte-steps ({} patch) -> {}",
                            steps, patch_steps, out_path
                        );
                    }
                }
                if steps >= max_steps {
                    break 'outer;
                }
            }
            // Глобальный патчевый W_patch (two-speed) — патчи по патч-словарь.
            let npatches = data.len() / patch_len;
            if npatches < 2 {
                continue;
            }
            let mut pat: Vec<&[u8]> = Vec::with_capacity(npatches);
            for p in 0..npatches {
                pat.push(&data[p * patch_len..(p + 1) * patch_len]);
            }
            for p in 1..npatches {
                tm.learn_patch(&pat[p.saturating_sub(ctxw)..p], pat[p], lr_patch);
                patch_steps += 1;
            }
        }
        println!("  corpus done: {}", corp);
    }
    let el = t0.elapsed().as_secs_f64();
    println!(
        "trained unified: {} byte-steps + {} patch-steps, {} bytes in {:.1}s ({:.0} bs)",
        steps,
        patch_steps,
        bytes_seen,
        el,
        steps as f64 / el
    );

    // ── OWM-консолидация ключевых направлений (Woodbury, tag=3).
    let owm_dirs = [
        "fn", "main", "(", ")", "let", "mut", "use", "std", "impl", "if", "the", "and",
        "gravity", "force", "return", "struct",
    ];
    let dirs: Vec<fuga::ai::latent_jepa::LatentVector> = owm_dirs
        .iter()
        .map(|s| tm.predictor().encoder.encode(&fuga::ai::sdr::encode_text(s)))
        .collect();
    let k = tm.consolidate_owm(&dirs, 16, 0.01);
    println!("OWM: consolidated {} directions", k);

    // ── Сохраняем ЕДИНЫЙ FUGA1: LOCAL_W + PATCH_W + OWM_P + META + KAN_C.
    tm.apply_byte_w(hyb.w.clone());
    let meta = UnifiedMeta {
        steps: steps as u64,
        patch_steps: patch_steps as u64,
        ctx: ctxw as u32,
        version: 2,
    };
    let patch_w: Vec<f32> = tm.patch_predictor().w.clone();
    let owm_p: Vec<f32> = tm.predictor().p.clone();
    save_unified_with_kan(
        &out_path,
        &hyb.w,
        &patch_w,
        &owm_p,
        &meta,
        None,
        None,
        Some(&hyb.kan.c)
    )
    .expect("save unified+kan");
    println!(
        "saved {} (LOCAL_W {} + PATCH_W {} + OWM_P {} + KAN_C {})",
        out_path,
        hyb.w.len(),
        patch_w.len(),
        owm_p.len(),
        hyb.kan.c.len()
    );

    // ── Декодим ОДНИМ чекпоинтом: и текст, и код.
    let kan = &hyb.kan;
    for (label, seed) in [("TEXT", &text_seed), ("CODE", &code_seed)] {
        println!("\n--- {} SEED {:?} ---", label, String::from_utf8_lossy(seed));
        let o1 = tm_generate_latent_bytes(&tm, seed, 120, ctxw, None);
        println!(
            "  naive       ({} B): {:?}",
            o1.len(),
            String::from_utf8_lossy(&o1).chars().take(60).collect::<String>()
        );
        let o2 = tm_generate_two_speed_entropy(&tm, seed, 200, patch_len, 0.60, &patch_vocab);
        println!(
            "  entropy-BLT ({} B): {:?}",
            o2.len(),
            String::from_utf8_lossy(&o2).chars().take(60).collect::<String>()
        );
        let o3 = tm_generate_hybrid(&tm, kan, seed, 120, ctxw, alpha);
        println!(
            "  hybrid W+K  ({} B): {:?}",
            o3.len(),
            String::from_utf8_lossy(&o3).chars().take(60).collect::<String>()
        );
    }
    println!("\n== UNIFIED TRAIN DONE ==  все технологии в одном файле: {}", out_path);
}
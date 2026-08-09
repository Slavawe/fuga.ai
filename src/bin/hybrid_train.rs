// hybrid_train.rs — ПОЛНОЕ гибридное обучение W+KAN (разделение функций).
//
// 1. Читает корпус(ы): каждый байт-шаг → (window, next).
// 2. Учит W: Widrow-Hoff (частотные биграммы) — пре-существующий путь.
// 3. Учит KAN: сплайн на ОСТАТКЕ (target − W·x) — нелинейные аттракторы.
// 4. Сохраняет единый FUGA1 С СЕКЦИЕЙ KAN_C (tag=6) — save_unified_with_kan.
// 5. Декодит: tm_generate_hybrid (W·x + α·KAN(x)) + сравнение с W-only.
//
// Usage: hybrid_train [--jsonl a,b,c] [--max-steps N] [--lr-w ..] [--lr-kan ..]
//                     [--alpha ..] [--out model.fuga] [--seed "fn main() {"]
use fuga::ai::htm_temporal::{
    save_unified_with_kan, TemporalMemory, UnifiedMeta,
};
use fuga::ai::hybrid::HybridTransition;
use fuga::ai::kan::KanTransition;
use fuga::ai::tm_generate::{tm_generate_hybrid, tm_generate_latent_bytes};
use fuga::ai::sdr::{byte_basis, structure_sdr_from_sdrs};
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
            ]
        });
    let max_steps: usize = arg(&args, "--max-steps", 500_000);
    let lr_w: f32 = arg(&args, "--lr-w", 0.05);
    let lr_kan: f32 = arg(&args, "--lr-kan", 0.2);
    let alpha: f32 = arg(&args, "--alpha", 1.0);
    let out_path: String = arg(&args, "--out", "/tmp/hybrid.fuga".into());
    let seed: Vec<u8> = args
        .iter()
        .position(|a| a == "--seed")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_bytes().to_vec())
        .unwrap_or_else(|| b"fn main() {".to_vec());
    let ctxw: usize = arg(&args, "--ctx", 4);
    let ckpt_every: usize = arg(&args, "--ckpt-every", 0);

    let t0 = Instant::now();
    // Гибридный оператор: W (внутри) + KAN. Учим напрямую на байт-шагах.
    let mut hyb = HybridTransition::new();
    let mut tm = TemporalMemory::new(64, ctxw); // TM нужен как обёртка для encoder/декодера
    let enc = &tm.predictor().encoder;
    let mut steps = 0usize;
    let mut bytes_seen = 0usize;

    'outer: for corp in &corpora {
        let f = std::fs::File::open(corp).expect("corpus");
        let rd = std::io::BufReader::new(f);
        for line in rd.lines().flatten() {
            let data = extract_bytes(&line);
            if data.len() < ctxw + 1 {
                continue;
            }
            for i in 0..data.len() - 1 {
                let lo = i.saturating_sub(ctxw);
                let win = &data[lo..=i];
                let next = data[i + 1];
                // W-дельта + KAN на остатке — внутри hybrid.learn_pair
                hyb.learn_pair(enc, win, next, lr_w, lr_kan);
                steps += 1;
                bytes_seen += 1;
                // Периодический чкпоинт: переживает убийство/ребут (урок 09.08).
                if ckpt_every > 0 && steps % ckpt_every == 0 {
                    let cm = UnifiedMeta {
                        steps: steps as u64,
                        patch_steps: 0,
                        ctx: ctxw as u32,
                        version: 1,
                    };
                    let ckpt_path = format!("{}.ckpt.fuga", out_path);
                    save_unified_with_kan(
                        &ckpt_path,
                        &hyb.w,
                        &vec![0.0f32; 512 * 512],
                        &vec![0.0f32; 512 * 512],
                        &cm,
                        None,
                        Some(&hyb.kan.c),
                    )
                    .ok();
                    eprintln!("  [ckpt] {} шагов -> {}", steps, ckpt_path);
                }
                if steps >= max_steps {
                    break 'outer;
                }
            }
        }
        println!("  corpus done: {}", corp);
    }
    let el = t0.elapsed().as_secs_f64();
    println!(
        "trained hybrid: {} steps, {} bytes in {:.1}s ({:.0} steps/s)",
        steps,
        bytes_seen,
        el,
        steps as f64 / el
    );

    // Применяем W из гибрида в TM (декодер tm_generate_hybrid использует w из TM).
    tm.apply_byte_w(hyb.w.clone());

    // --- Сохраняем единый FUGA1 с KAN_C (tag=6) ---
    let meta = UnifiedMeta {
        steps: steps as u64,
        patch_steps: 0,
        ctx: ctxw as u32,
        version: 1,
    };
    let owm_p = tm.predictor().p.clone();
    // patch-секция для совместимости: пустой W_patch (не использовался)
    let empty_patch = vec![0.0f32; 512 * 512];
    save_unified_with_kan(
        &out_path,
        &hyb.w,
        &empty_patch,
        &owm_p,
        &meta,
        None,
        Some(&hyb.kan.c),
    )
    .expect("save unified+kan");
    println!("saved {} (FUGA1 + KAN_C {} f32)", out_path, hyb.kan.c.len());

    // --- Декодирование: гибрид vs чистый W ---
    let w_only = tm_generate_latent_bytes(&tm, &seed, 120, ctxw, None);
    let w_text = String::from_utf8_lossy(&w_only);
    println!("  W-only   ({:>3} B): {:?}", w_only.len(), w_text.chars().take(60).collect::<String>());

    let kan = &hyb.kan;
    let hyb_out = tm_generate_hybrid(&tm, kan, &seed, 120, ctxw, alpha);
    let hyb_text = String::from_utf8_lossy(&hyb_out);
    println!("  HYBRID   ({:>3} B): {:?}", hyb_out.len(), hyb_text.chars().take(60).collect::<String>());
    println!("== HYBRID TRAIN DONE ==");
}
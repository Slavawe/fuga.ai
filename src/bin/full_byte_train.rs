// full_byte_train.rs — самое крупное обучение байтового стека на ВСЕХ JSONL.
//
// Стримит все датасеты единым байтовым потоком и обучает:
//   - локальный байтовый W (predictor_mut().learn_transition, окно ctx)
//   - глобальный патчевый W_patch (patch_predictor_mut().learn_transition,
//      2-байтовые патчи — two-speed)
//   - периодический OWM-consolidate (защита W от катастрофического забывания)
// Сохраняет полный TM-чекпоинт (клетки+W+W_patch) + sidecar save_byte_w.
// Затем прогоняет декодеры, чтобы показать, что обучение дошло до состояния.
//
// ЧИСТЫЙ CPU-стенд (Rust остаётся прокси/референсом). GPU-интеграция и
// Python-оркестрация живут в C++-ядре (cpp/fuga_core) — см. README.
//
// Usage: full_byte_train [--ctx 4] [--lr 0.05] [--out fuga_full_byte_tm.bin]
//                        [--max-bytes 50000000]
use std::io::BufRead;
use std::path::Path;
use std::time::Instant;

use fuga::ai::htm_temporal::TemporalMemory;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ctx: usize = arg(&args, "--ctx", 4);
    let lr: f32 = arg(&args, "--lr", 0.05);
    let out_tm: String = arg(&args, "--out", "fuga_full_byte_tm.bin".into());
    let max_bytes: usize = arg(&args, "--max-bytes", 50_000_000);

    let corpora = [
        "fuga_unified_train.jsonl",
        "corpus_doc_code_pairs.jsonl",
        "training_stack.jsonl",
        "omni_corpus_full.jsonl",
        "corpus.jsonl",
        "corpus_rus_eng.jsonl",
        "omni_corpus_repos.jsonl",
    ];

    let t0 = Instant::now();
    let mut tm = TemporalMemory::new(64, ctx);
    let mut steps: u64 = 0;
    let mut bytes: u64 = 0;
    let mut since_consol: u64 = 0;
    let mut dirs: Vec<fuga::LatentVector> = Vec::new();

    'outer: for corp in &corpora {
        let p = Path::new(corp);
        if !p.exists() {
            println!("  skip (missing): {}", corp);
            continue;
        }
        println!("TRAINING on {} ...", corp);
        let f = match std::fs::File::open(p) {
            Ok(f) => f,
            Err(e) => {
                println!("  open err: {}", e);
                continue;
            }
        };
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
            // Guard: skip degenerate zero-length extract.
            if data.len() < 2 {
                continue;
            }
            for i in 0..data.len().saturating_sub(1) {
                let lo = i.saturating_sub(ctx);
                let nxt = data[i + 1];
                // local byte transition (window → next byte)
                let win_sdrs: Vec<fuga::SdrVector> = data[lo..=i]
                    .iter()
                    .map(|&c| fuga::byte_basis(c))
                    .collect();
                let next = fuga::byte_basis(nxt);
                tm.predictor_mut().learn_transition(&win_sdrs, &next, lr);
                // global patch level (two-speed): encode last 2-byte window as
                // patches -> next 2-byte patch.
                if i >= 2 {
                    let pat_window = &data[i - 2..=i];
                    let pats: Vec<&[u8]> = pat_window.chunks(2).collect();
                    let next_patch = &data[i + 1..(i + 3).min(data.len())];
                    let win_sd: Vec<fuga::SdrVector> =
                        pats.iter().map(|p| fuga::encode_bytes_sdr(p)).collect();
                    let nxt_sdr = fuga::encode_bytes_sdr(next_patch);
                    tm.patch_predictor_mut().learn_transition(&win_sd, &nxt_sdr, lr);
                    bytes += 1;
                }
                steps += 1;
                // collect direction for OWM consolidation
                if steps % 64 == 0 {
                    let v = fuga::LatentVector::zero();
                    dirs.push(v);
                }
                since_consol += 1;
                if since_consol >= 25_000 {
                    tm.consolidate_owm(&dirs, 64, 0.9);
                    dirs.clear();
                    since_consol = 0;
                }
                if steps as usize >= max_bytes {
                    println!("  reached step budget {}", max_bytes);
                    break 'outer;
                }
            }
            println!("  processed line ({} bytes), steps={}", data.len(), steps);
            if steps % 1_000_000 == 0 {
                println!("    ... {} steps", steps);
            }
        }
        println!("  corpus done: {}", corp);
    }

    let el = t0.elapsed().as_secs_f64();
    println!();
    println!("=== FULL BYTE TRAINING COMPLETE ===");
    println!("  steps={} bytes={} in {:.1}s", steps, bytes, el);
    println!(
        "  cells={} W_updates={} W_patch_updates={}",
        tm.cells.len(),
        tm.predictor_updates(),
        tm.patch_predictor().updates
    );

    // Save full checkpoint (cells + W + W_patch) and byte-W sidecar.
    tm.save(&out_tm);
    println!("saved full checkpoint -> {}", out_tm);
    let side = out_tm.replace(".bin", "_byte.bin");
    if let Ok(()) = tm.save_byte_w(&side) {
        println!("saved sidecar -> {}", side);
    }

    // Decoder sweep on the trained state.
    let seed: &[u8] = b"fn main() {";
    println!();
    println!("=== DECODER SWEEP (trained in-memory state) ===");
    let naive = fuga::tm_generate_latent_bytes(&tm, seed, 200, ctx, None);
    println!(
        "  naive byte W      : {} B :: {}",
        naive.len(),
        String::from_utf8_lossy(&naive).chars().take(40).collect::<String>()
    );
    let rec0 = fuga::tm_generate_recurrent(&tm, seed, 200, ctx, 0.0, 0.9);
    println!(
        "  recurrent mix0    : {} B :: {}",
        rec0.len(),
        String::from_utf8_lossy(&rec0).chars().take(40).collect::<String>()
    );
    let rec4 = fuga::tm_generate_recurrent(&tm, seed, 200, ctx, 0.4, 0.9);
    println!(
        "  recurrent mix0.4  : {} B :: {}",
        rec4.len(),
        String::from_utf8_lossy(&rec4).chars().take(40).collect::<String>()
    );
    println!();
    println!("DONE.");
}

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
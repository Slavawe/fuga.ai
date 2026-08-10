// w_diag.rs — диагностика: загрузить FUGA1 и проверить, что W/P/patch реально
// те же, что писал unified_gpu_train. Печатаем нормы и пробуем декод.
use fuga::ai::htm_temporal::TemporalMemory;

// Копия extract_bytes из unified_gpu_train (JSON → doc/code/chapters текст).
fn unified_gpu_train_extract(line: &str) -> Vec<u8> {
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
    let ckpt = args.get(1).cloned().unwrap_or_else(|| "/tmp/ab_test.fuga".into());
    let raw = fuga::ai::htm_temporal::load_unified(&ckpt);
    match raw {
        Some((w, pw, p, meta, hj, kan)) => {
            let norm = |v: &[f32]| v.iter().map(|x| (x * x) as f64).sum::<f64>().sqrt();
            let s1: f64 = w.iter().take(64).map(|x| *x as f64).sum();
            let s2: f64 = pw.iter().take(64).map(|x| *x as f64).sum();
            println!("local W: len={} norm={:.3} sum64={:.4}", w.len(), norm(&w), s1);
            println!("patch W: len={} norm={:.3} sum64={:.4}", pw.len(), norm(&pw), s2);
            println!("patch W: len={} norm={:.3}", pw.len(), norm(&pw));
            println!("OWM P  : len={} norm={:.3}", p.len(), norm(&p));
            println!("meta   : steps={} patch_steps={} ctx={} ver={}", meta.steps, meta.patch_steps, meta.ctx, meta.version);
            println!("hjepa  : {:?}", hj.map(|v| v.len()));
            println!("kan_c  : {:?}", kan.map(|v| v.len()));

            // Декод через talk_model-стиль (load_unified_fuga1)
            let mut tm = TemporalMemory::new(64, 4);
            assert!(tm.load_unified_fuga1(&ckpt), "load fail");
            println!("after load: local_updates={} patch_updates={}", tm.predictor().updates, tm.patch_predictor().updates);
            // Наивный декод
            let out = fuga::tm_generate_latent_bytes(&tm, b"the force of gravity is", 40, 4, None);
            println!("naive (loaded): {} B -> {:?}", out.len(), String::from_utf8_lossy(&out).chars().take(40).collect::<String>());
            let out2 = fuga::tm_generate_latent_bytes(&tm, b"fn main() {", 40, 4, None);
            println!("naive (loaded, code): {} B -> {:?}", out2.len(), String::from_utf8_lossy(&out2).chars().take(40).collect::<String>());

            // Декод через unified-стиль: свежий TM + только apply_byte_w (P identity, patch identity)
            let mut tm2 = TemporalMemory::new(64, 4);
            tm2.apply_byte_w(w.clone());
            let out3 = fuga::tm_generate_latent_bytes(&tm2, b"the force of gravity is", 40, 4, None);
            println!("naive (fresh+W): {} B -> {:?}", out3.len(), String::from_utf8_lossy(&out3).chars().take(40).collect::<String>());
            let out4 = fuga::tm_generate_latent_bytes(&tm2, b"fn main() {", 40, 4, None);
            println!("naive (fresh+W, code): {} B -> {:?}", out4.len(), String::from_utf8_lossy(&out4).chars().take(40).collect::<String>());

            // Предзаполнение кеша, ИМИТИРУЯ обучение: прогнать байт-окна корпуса
            // через predictor.encoder (0xF03D) и патч-окна через patch_predictor
            // (0xBAC7) — как это делал unified_gpu_train до декода. Гипотеза:
            // LATENT_ENC_CACHE коллизирует между двумя энкодерами (ключ =
            // SdrVector без seed!) → порядок заполнения меняет результат.
            if std::env::args().any(|a| a == "--warm-cache") {
                use std::io::BufRead as _;
                if let Ok(f) = std::fs::File::open("fisig_corpus.jsonl") {
                    let rd = std::io::BufReader::new(f);
                    let mut n = 0usize;
                    for line in rd.lines().flatten() {
                        let b = line.as_bytes();
                        if b.len() >= 4 {
                            let _ = tm.predictor().encoder.encode(&fuga::ai::sdr::byte_basis(b[0]));
                            let _ = tm.patch_predictor().encoder.encode(&fuga::ai::sdr::encode_bytes_sdr(&b[0..2]));
                            n += 1;
                            if n > 2000 { break; }
                        }
                    }
                }
                println!("  [A/B] кеш прогрет (имитация обучения)");
            }

            // entropy-BLT в ОБОИХ стилях (как в talk_model vs unified)
            let mut seen: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
            let mut vocab: Vec<Vec<u8>> = Vec::new();
            // A/B vocab: raw JSON-строки vs extract_bytes (извлечённый контент).
            let use_extract = std::env::args().any(|a| a == "--extract-vocab");
            if let Ok(f) = std::fs::File::open("fisig_corpus.jsonl") {
                use std::io::BufRead;
                let rd = std::io::BufReader::new(f);
                for line in rd.lines().flatten() {
                    let src = if use_extract {
                        unified_gpu_train_extract(&line)
                    } else {
                        line.as_bytes().to_vec()
                    };
                    for w2 in src.windows(2) {
                        if seen.insert(w2.to_vec()) && vocab.len() < 5000 {
                            vocab.push(w2.to_vec());
                        }
                    }
                    if vocab.len() >= 5000 { break; }
                }
            }
            println!("  [A/B] vocab={} (extract={})", vocab.len(), use_extract);
            let e1 = fuga::tm_generate_two_speed_entropy(&tm, b"the force of gravity is", 200, 2, 0.60, &vocab);
            println!("entropy (loaded)  : {} B -> {:?}", e1.len(), String::from_utf8_lossy(&e1).chars().take(60).collect::<String>());
            let e2 = fuga::tm_generate_two_speed_entropy(&tm2, b"the force of gravity is", 200, 2, 0.60, &vocab);
            println!("entropy (fresh+W) : {} B -> {:?}", e2.len(), String::from_utf8_lossy(&e2).chars().take(60).collect::<String>());
            let e3 = fuga::tm_generate_two_speed_entropy(&tm2, b"fn main() {", 200, 2, 0.60, &vocab);
            println!("entropy (fresh+W, code): {} B -> {:?}", e3.len(), String::from_utf8_lossy(&e3).chars().take(60).collect::<String>());
        }
        None => println!("load_unified failed for {}", ckpt),
    }
}
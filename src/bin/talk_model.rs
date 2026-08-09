// talk_model.rs — «говорящая модель»: загружает единый FUGA1 чекпоинт,
// строит patch-vocab из ТЕХ ЖЕ корпусов, что и обучение, и декодит
// текстовые seed'ы всеми декодерами главного пути.
//
// Usage: talk_model <checkpoint.fuga> [corpus1.jsonl ...]
use fuga::ai::htm_temporal::TemporalMemory;
use fuga::ai::sdr::encode_bytes_sdr;
use fuga::tm_generate_two_speed;
use fuga::tm_generate_two_speed_entropy;
use fuga::tm_generate_hybrid;
use fuga::tm_generate_recurrent;
use fuga::tm_generate_latent_bytes;
use std::collections::HashSet;
use std::io::BufRead;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ckpt = args.get(1).cloned().unwrap_or_else(|| "fuga_talk_10m.fuga".into());
    let corpora: Vec<&str> = if args.len() > 2 {
        args[2..].iter().map(|s| s.as_str()).collect()
    } else {
        vec![
            "fuga_unified_train.jsonl",
            "corpus_doc_code_pairs.jsonl",
            "training_stack.jsonl",
            "corpus.jsonl",
            "omni_corpus_full.jsonl",
        ]
    };

    // 1. Загружаем единый чекпоинт (все секции: W + W_patch + OWM-P)
    let mut tm = TemporalMemory::new(64, 4);
    assert!(tm.load_unified_fuga1(&ckpt), "FUGA1 не читается: {}", ckpt);
    println!("== TALK MODEL: {} ==", ckpt);
    println!(
        "  W_updates={} W_patch_updates={} ctx={} OWM_P={}",
        tm.predictor().updates,
        tm.patch_predictor().updates,
        tm.context_len,
        tm.predictor().p.len()
    );

    // 2. Патч-словарь из корпусов обучения (2-байтовые патчи, cap 512)
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut patch_vocab: Vec<Vec<u8>> = Vec::new();
    for corp in &corpora {
        let Ok(f) = std::fs::File::open(corp) else { continue };
        use std::io::BufRead;
        let rd = std::io::BufReader::new(f);
        for line in rd.lines().flatten() {
            if line.len() < 4 { continue; }
            let bytes = line.as_bytes();
            for w in bytes.windows(2) {
                if seen.insert(w.to_vec()) && patch_vocab.len() < 512 {
                    patch_vocab.push(w.to_vec());
                }
            }
            if patch_vocab.len() >= 512 { break; }
        }
        if patch_vocab.len() >= 512 { break; }
    }
    println!("  patch_vocab={} (из {} корпусов)", patch_vocab.len(), corpora.len());
    let _ = encode_bytes_sdr; // патч-кодирование внутри декодера
    let _ = tm_generate_latent_bytes;

    // 3. ГИБРИД: KAN_C из чекпоинта + tm_generate_hybrid (W·x + α·KAN(x))
    let kan_c: Option<Vec<f32>> =
        fuga::ai::htm_temporal::load_unified(&ckpt).and_then(|t| t.5);
    let mut kan = fuga::ai::kan::KanTransition::new();
    if let Some(c) = &kan_c {
        if c.len() == kan.c.len() {
            kan.c.clone_from(c);
            println!("  KAN_C  загружен: {} f32 (гибридный декодер активен)", c.len());
        } else {
            println!("  KAN_C  не совпал по длине ({} != {}) — гибрид НЕ активен", c.len(), kan.c.len());
        }
    } else {
        println!("  KAN_C  в чекпоинте НЕТ — гибрид НЕ активен");
    }
    let _ = &kan_c; // Option в кортеже (индексы: 0=w 1=patch 2=owm 3=meta 4=hjepa 5=kan)

    // 4. «Говорящие» seed'ы: обычные фразы, не только код
    let seeds: Vec<Vec<u8>> = vec![
        b"fn main() {".to_vec(),
        b"the force of gravity is".to_vec(),
        b"in the beginning".to_vec(),
        b"let x = 4".to_vec(),
    ];

    for seed in &seeds {
        let s = String::from_utf8_lossy(seed);
        println!("\n--- SEED {:?} ---", s);

        let out1 = tm_generate_latent_bytes(&tm, seed, 200, 4, None);
        println!("  naive byte  ({} B): {:?}", out1.len(), String::from_utf8_lossy(&out1).chars().take(60).collect::<String>());

        let out2 = tm_generate_two_speed(&tm, seed, 60, 2, &patch_vocab, None);
        println!("  two-speed   ({} B): {:?}", out2.len(), String::from_utf8_lossy(&out2).chars().take(60).collect::<String>());

        let out3 = tm_generate_two_speed_entropy(&tm, seed, 200, 4, 0.60, &patch_vocab);
        println!("  entropy-BLT ({} B): {:?}", out3.len(), String::from_utf8_lossy(&out3).chars().take(60).collect::<String>());

        let out4 = tm_generate_recurrent(&tm, seed, 200, 4, 0.0, 0.9);
        println!("  recurrent   ({} B): {:?}", out4.len(), String::from_utf8_lossy(&out4).chars().take(60).collect::<String>());

        if kan.c.iter().any(|&v| v.abs() > 1e-6) {
            let out5 = tm_generate_hybrid(&tm, &kan, seed, 200, 4, 1.0);
            println!("  HYBRID      ({} B): {:?}", out5.len(), String::from_utf8_lossy(&out5).chars().take(60).collect::<String>());
        }
    }
    println!("\n== TALK-декодеры отработали ==");
}

// алиасы помощи: чтобы не дублировать типы
fn _unused(_x: usize) {}
#[allow(dead_code)]
fn tmdb_latent_sdr(_b: &[u8]) {}
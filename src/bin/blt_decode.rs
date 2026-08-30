//! BLT-декодер: проверка переменных патчей на реальном чекпоинте.
//!
//! Использование: blt_decode <ckpt.fuga> <корпус.jsonl> [threshold]
//!
//! 1. Обучает BLT-энтропию (bigram) на корпусе
//! 2. Загружает чекпоинт (W_patch)
//! 3. Декодирует сиды через BLT-патчи переменной длины
//! 4. Сравнивает с фиксированными патчами (MB2-логика)

use std::io::BufRead;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <ckpt.fuga> <corpus.jsonl> [threshold]", args[0]);
        std::process::exit(1);
    }
    let ckpt_path = &args[1];
    let corpus_path = &args[2];
    let threshold: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.85);

    // 1. Обучаем энтропию на корпусе
    println!("[BLT] обучение энтропии на корпусе...");
    let mut entropy = fuga::ai::blt_patch::BltEntropy::new();
    let file = match std::fs::File::open(corpus_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("не могу открыть корпус: {}", e);
            std::process::exit(1);
        }
    };
    let reader = std::io::BufReader::new(file);
    let mut n_lines = 0u64;
    for line in reader.lines() {
        if let Ok(l) = line {
            entropy.learn(l.as_bytes());
            n_lines += 1;
            if n_lines >= 200_000 {
                break;
            }
        }
    }
    println!("[BLT] строк корпуса: {}", n_lines);

    // 2. Загружаем чекпоинт
    println!("[BLT] загрузка чекпоинта: {}", ckpt_path);
    let mut tm = fuga::ai::htm_temporal::TemporalMemory::new(64, 8);
    if !tm.load_unified_fuga1(ckpt_path) {
        eprintln!("не удалось загрузить чекпоинт");
        std::process::exit(1);
    }
    let w = tm.predictor_w();
    let wp = tm.patch_predictor_w();
    println!("[BLT] |W|={:.3} |Wp|={:.3}",
             (w.iter().map(|v| v * v).sum::<f32>()).sqrt(),
             (wp.iter().map(|v| v * v).sum::<f32>()).sqrt());

    // 3. Patch-vocab: 2-байтовые окна корпуса (как трейнер)
    let mut patch_vocab: Vec<Vec<u8>> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let file = std::fs::File::open(corpus_path).unwrap();
    let reader = std::io::BufReader::new(file);
    'outer: for line in reader.lines() {
        if let Ok(l) = line {
            let bytes = l.as_bytes();
            for w2 in bytes.windows(2) {
                if seen.insert(w2.to_vec()) && patch_vocab.len() < 5000 {
                    patch_vocab.push(w2.to_vec());
                }
                if patch_vocab.len() >= 5000 {
                    break 'outer;
                }
            }
        }
    }
    println!("[BLT] patch_vocab: {} (2-байтовые)", patch_vocab.len());

    // 4. BLT-патчи на сидах
    let seeds = [
        "fn main() {",
        "the force of gravity is",
        "in the beginning",
        "let x = 4",
    ];
    println!("\n=== BLT-ДЕКОДЕР (threshold={}) ===", threshold);
    for seed in seeds {
        // статистика BLT-патчей на сиде
        let pats = fuga::ai::blt_patch::blt_patch(seed.as_bytes(), &entropy, threshold, 16);
        let lens: Vec<usize> = pats.iter().map(|p| p.len()).collect();
        println!("\nseed: {:?}", seed);
        println!("  BLT-патчи: {:?} (сумма={})", lens, lens.iter().sum::<usize>());

        // BLT-декодирование
        let out = fuga::ai::blt_patch::tm_generate_megabyte_blt(
            &tm, seed.as_bytes(), 100, &entropy, threshold, &patch_vocab, 8, 0.05,
        );
        let text: String = out.iter().map(|&b| {
            if b == b'\n' { '\n' } else if b == b'\t' { '\t' }
            else if (0x20..=0x7e).contains(&b) { b as char } else { '·' }
        }).collect();
        println!("  BLT: {}B → {:?}", out.len(), &text[..text.len().min(120)]);
    }
    println!("\n=== BLT-ДЕКОДЕР ГОТОВ ===");
}

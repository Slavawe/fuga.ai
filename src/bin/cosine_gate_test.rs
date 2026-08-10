// cosine_gate_test.rs — A/B новых рычагов декодирования на реальном чекпоинте:
//   1) косинус+температура (corridor=0, τ варьируется)
//   2) MegaByte-коридор жёсткий (corridor=1) и мягкий (corridor=2)
// против старого entropy-BLT (argmax-семейство). Один чекпоинт, одни сиды.
use fuga::ai::htm_temporal::TemporalMemory;
use fuga::ai::kan::KanTransition;
use fuga::tm_generate_cosine_gate;
use std::collections::HashSet;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ckpt = args.get(1).cloned().unwrap_or_else(|| "fuga_unified_v4.fuga".into());
    let corpora: Vec<&str> = if args.len() > 2 {
        args[2..].iter().map(|s| s.as_str()).collect()
    } else {
        vec!["fisig_corpus.jsonl", "corpus_doc_code_pairs.jsonl", "training_stack.jsonl", "corpus.jsonl"]
    };

    let mut tm = TemporalMemory::new(64, 4);
    assert!(tm.load_unified_fuga1(&ckpt), "load fail");

    // KAN_C из чекпоинта
    let kan_c: Option<Vec<f32>> = fuga::ai::htm_temporal::load_unified(&ckpt).and_then(|t| t.5);
    let mut kan = KanTransition::new();
    if let Some(c) = &kan_c {
        if c.len() == kan.c.len() {
            kan.c.clone_from(c);
            println!("KAN_C загружен: {} f32", c.len());
        }
    }

    // Патч-словарь (2-байтовые патчи из корпусов, cap 5000)
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut patch_vocab: Vec<Vec<u8>> = Vec::new();
    for corp in &corpora {
        let Ok(f) = std::fs::File::open(corp) else { continue };
        use std::io::BufRead;
        let rd = std::io::BufReader::new(f);
        for line in rd.lines().flatten() {
            if line.len() < 4 { continue; }
            for w in line.as_bytes().windows(2) {
                if seen.insert(w.to_vec()) && patch_vocab.len() < 5000 {
                    patch_vocab.push(w.to_vec());
                }
            }
            if patch_vocab.len() >= 5000 { break; }
        }
        if patch_vocab.len() >= 5000 { break; }
    }
    println!("patch_vocab={}", patch_vocab.len());

    let seeds: Vec<Vec<u8>> = vec![
        b"fn main() {".to_vec(),
        b"the force of gravity is".to_vec(),
        b"in the beginning".to_vec(),
        b"let x = 4".to_vec(),
        b"in the beginning".to_vec(),
    ];

    for seed in &seeds {
        println!("\n--- SEED {:?} ---", String::from_utf8_lossy(seed));
        // Базовый: entropy-BLT (старый путь, argmax-хэндовер)
        let out_blt = fuga::tm_generate_two_speed_entropy(&tm, seed, 160, 5, 0.60, &patch_vocab);
        println!("  entropy-BLT : {:?}", String::from_utf8_lossy(&out_blt).chars().take(80).collect::<String>());

        // Новый рычаг 1: косинус+температура без коридора
        let out_t0 = fuga::tm_generate_cosine_gate_inner(&tm, &kan, seed, 160, 5, 2, &patch_vocab, 0.0, 0.01, 0, 0.001);
        println!("  cos α=0 τ=0.01  : {:?}", String::from_utf8_lossy(&out_t0).chars().take(80).collect::<String>());
        let out_t2 = tm_generate_cosine_gate(&tm, &kan, seed, 160, 5, 2, &patch_vocab, 1.0, 0.05, 0);
        println!("  cos+τ=0.05   : {:?}", String::from_utf8_lossy(&out_t2).chars().take(80).collect::<String>());
        let out_t3 = tm_generate_cosine_gate(&tm, &kan, seed, 160, 5, 2, &patch_vocab, 1.0, 0.12, 0);
        println!("  cos+τ=0.12   : {:?}", String::from_utf8_lossy(&out_t3).chars().take(80).collect::<String>());

        // Новый рычаг 2: MegaByte-коридор (жёсткий / мягкий), α=0.3 (KAN-коррекция)
        let out_hard3 = fuga::tm_generate_cosine_gate_inner(
            &tm, &kan, seed, 160, 5, 2, &patch_vocab, 0.3, 0.08, 1, 0.003,
        );
        println!(
            "  corridor=1 α=0.3   : {:?}",
            String::from_utf8_lossy(&out_hard3)
                .chars()
                .take(80)
                .collect::<String>()
        );
        let out_hard = tm_generate_cosine_gate(&tm, &kan, seed, 160, 5, 2, &patch_vocab, 1.0, 0.05, 1);
        println!("  corridor=1 (hard) : {:?}", String::from_utf8_lossy(&out_hard).chars().take(80).collect::<String>());
        let out_soft = tm_generate_cosine_gate(&tm, &kan, seed, 160, 5, 2, &patch_vocab, 1.0, 0.05, 2);
        println!("  corridor=2 (soft) : {:?}", String::from_utf8_lossy(&out_soft).chars().take(80).collect::<String>());

        // === РЫЧАГИ v2: repetition penalty + аддитивное патч-кондиционирование ===
        // v2 базовый (α=0, β=0, rep=0): должен совпасть с naive-потоком
        let o_v2  = fuga::tm_generate_cosine_gate_v2(&tm, &kan, seed, 160, 5, 2, &patch_vocab, 0.0, 0.01, 0, 0.001, 0.0, 0.0, 0.0, 0.0);
        println!("  v2 base          : {:?}", String::from_utf8_lossy(&o_v2).chars().take(80).collect::<String>());
        // rep_pen: байтовый штраф на недавние n-граммы
        let o_rep = fuga::tm_generate_cosine_gate_v2(&tm, &kan, seed, 160, 5, 2, &patch_vocab, 0.0, 0.01, 0, 0.001, 0.0, 0.20, 0.0, 0.0);
        println!("  v2 rep=0.20      : {:?}", String::from_utf8_lossy(&o_rep).chars().take(80).collect::<String>());
        // rep_word: СЛОВЕСНЫЙ штраф на повторяющиеся слова
        let o_w1 = fuga::tm_generate_cosine_gate_v2(&tm, &kan, seed, 160, 5, 2, &patch_vocab, 0.0, 0.01, 0, 0.001, 0.0, 0.0, 0.20, 0.0);
        println!("  v2 word=0.20     : {:?}", String::from_utf8_lossy(&o_w1).chars().take(80).collect::<String>());
        let o_w2 = fuga::tm_generate_cosine_gate_v2(&tm, &kan, seed, 160, 5, 2, &patch_vocab, 0.0, 0.01, 0, 0.001, 0.0, 0.0, 0.50, 0.0);
        println!("  v2 word=0.50     : {:?}", String::from_utf8_lossy(&o_w2).chars().take(80).collect::<String>());
        // beta: аддитивный патчевый сдвиг темы
        let o_beta = fuga::tm_generate_cosine_gate_v2(&tm, &kan, seed, 160, 5, 2, &patch_vocab, 0.0, 0.01, 0, 0.001, 0.30, 0.0, 0.0, 0.0);
        println!("  v2 β=0.30        : {:?}", String::from_utf8_lossy(&o_beta).chars().take(80).collect::<String>());
        // комбинированный (байтовый + словесный штраф)
        let o_both = fuga::tm_generate_cosine_gate_v2(&tm, &kan, seed, 160, 5, 2, &patch_vocab, 0.0, 0.01, 0, 0.001, 0.0, 0.20, 0.50, 0.0);
        println!("  v2 rep+word      : {:?}", String::from_utf8_lossy(&o_both).chars().take(80).collect::<String>());
    }
    println!("\n== A/B готов ==");
}
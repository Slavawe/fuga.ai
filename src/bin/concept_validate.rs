// concept_validate.rs — A/B: MB2 (без макро/концепта) vs MB3 (макро)
// vs MB4 (макро + концепт lang-jepa, tag=8 CONCEPT_W).
//
// Использование: concept_validate <ckpt.fuga> [корпуса...]
// Сравнивает длину и содержимое потоков. При concept_flat=[]/βc=0
// MB4 бит-в-бит == MB3 (детерминизм проверки).
use fuga::ai::htm_temporal::{load_unified, load_unified_concept};
use std::collections::HashSet;
use std::io::BufRead;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ckpt = args.get(1).cloned().unwrap_or_else(|| "fuga_unified_v6.fuga".into());
    let corpora: Vec<&str> = if args.len() > 2 {
        args[2..].iter().map(|s| s.as_str()).collect()
    } else {
        vec!["corpus_doc_code_pairs.jsonl"]
    };

    let (w, w_patch, owm_p, meta, _, _) = load_unified(&ckpt).expect("load_unified");
    let w_macro = fuga::ai::htm_temporal::load_unified_macro(&ckpt)
        .unwrap_or_else(|| vec![0.0f32; 512 * 512]);
    let concept = load_unified_concept(&ckpt).unwrap_or_default();
    println!(
        "ckpt={} ctx={} steps={} |W|={:.3} |Wp|={:.3} macro={} concept={}",
        ckpt,
        meta.ctx,
        meta.steps,
        (w.iter().map(|x| x * x).sum::<f32>()).sqrt(),
        (w_patch.iter().map(|x| x * x).sum::<f32>()).sqrt(),
        if w_macro.iter().any(|&v| v != 0.0) { "ОБУЧЕН" } else { "нулевой" },
        if concept.len() >= 1_052_160 { "ОБУЧЕН" } else { "нет" }
    );

    // Патч-вокабуляр из корпусов.
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut patch_vocab: Vec<Vec<u8>> = Vec::new();
    for corp in &corpora {
        let Ok(f) = std::fs::File::open(corp) else { continue };
        let rd = std::io::BufReader::new(f);
        for line in rd.lines().flatten() {
            if line.len() < 4 { continue; }
            for w2 in line.as_bytes().windows(2) {
                if seen.insert(w2.to_vec()) && patch_vocab.len() < 5000 {
                    patch_vocab.push(w2.to_vec());
                }
            }
            if patch_vocab.len() >= 5000 { break; }
        }
        if patch_vocab.len() >= 5000 { break; }
    }

    let mut tm = fuga::ai::htm_temporal::TemporalMemory::new(64, meta.ctx as usize);
    tm.apply_byte_w(w.clone());
    let _ = owm_p;

    let seeds: Vec<Vec<u8>> = vec![
        b"the force of gravity is".to_vec(),
        b"in the beginning".to_vec(),
        b"fn main() {".to_vec(),
        b"let x = 4".to_vec(),
    ];
    let win = meta.ctx as usize + 1;

    println!("── A/B: MB2 | MB3(макро) | MB4(макро+концепт) ──");
    for seed in &seeds {
        let out2 = fuga::tm_generate_megabyte_v3(
            &tm, seed, 200, win, 2, &patch_vocab, 8, 0.3, 0.20, 0.8, 0.001, 0.30, 0.05,
            &[], 0.0, &[], 0.0,  // MB2: без макро, без концепта
        );
        let out3 = fuga::tm_generate_megabyte_v3(
            &tm, seed, 200, win, 2, &patch_vocab, 8, 0.3, 0.20, 0.8, 0.001, 0.30, 0.05,
            &w_macro, 0.3, &[], 0.0,  // MB3: макро, без концепта
        );
        let out4 = fuga::tm_generate_megabyte_v3(
            &tm, seed, 200, win, 2, &patch_vocab, 8, 0.3, 0.20, 0.8, 0.001, 0.30, 0.05,
            &w_macro, 0.3, &concept, 0.9,  // MB4: макро + концепт βc=0.9
        );
        let s = |b: &[u8]| String::from_utf8_lossy(b).chars().take(50).collect::<String>();
        println!(
            "[{}] MB2={}B | MB3={}B{} | MB4={}B{}",
            String::from_utf8_lossy(seed).chars().take(16).collect::<String>(),
            out2.len(), out3.len(),
            if out3 == out2 { "" } else { " ≠MB2" },
            out4.len(),
            if out4 == out3 { "" } else { " ≠MB3" },
        );
        if out4 != out3 {
            println!("    MB3: {}", s(&out3));
            println!("    MB4: {}", s(&out4));
        }
    }
    println!("== concept validate done ==");
}

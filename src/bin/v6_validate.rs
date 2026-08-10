// v6_validate.rs — валидация v6: ctx=8 → window=9, β=0.3 (явный Patch Loss)
// загружает чекпоинт FUGA1, прогоняет V2-декодер (конфигурация из AGENTS.md:
// α=0, τ=0.01, corridor=0, β=0.3, rep_word=0.20, window=9) на 4 сидах,
// печатает норму остатка и декоды — единая метрика для всех чекпоинтов v5.
use fuga::ai::htm_temporal::{load_unified, TemporalMemory};
use fuga::ai::kan::KanTransition;
use std::collections::HashSet;
use std::io::BufRead;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ckpt = args.get(1).cloned().unwrap_or_else(|| "fuga_unified_v4.fuga".into());
    let corpora: Vec<&str> = if args.len() > 2 {
        args[2..].iter().map(|s| s.as_str()).collect()
    } else {
        vec!["fisig_corpus.jsonl", "corpus_doc_code_pairs.jsonl", "training_stack.jsonl", "corpus.jsonl"]
    };

    let data = load_unified(&ckpt);
    let Some((w, pw, _owm, meta, _hj, kan_c)) = data else {
        println!("НЕ ЗАГРУЖЕН");
        return;
    };
    let _ = &meta;
    println!(
        "ckpt={} ctx={} steps={} | |W|={:.3} |Wp|={:.3}",
        ckpt,
        meta.ctx,
        meta.steps,
        w.iter().map(|v| v * v).sum::<f32>().sqrt(),
        pw.iter().map(|v| v * v).sum::<f32>().sqrt()
    );

    let mut tm = TemporalMemory::new(64, 8);
    assert!(tm.load_unified_fuga1(&ckpt), "load FUGA1");
    let mut kan = KanTransition::new();
    if let Some(c) = &kan_c {
        if c.len() == kan.c.len() {
            kan.c.clone_from(c);
        }
    }

    // patch_vocab из корпусов (тот же способ, что в трейнере)
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut patch_vocab: Vec<Vec<u8>> = Vec::new();
    for corp in &corpora {
        let Ok(f) = std::fs::File::open(corp) else { continue };
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

    let seeds: Vec<Vec<u8>> = vec![
        b"the force of gravity is".to_vec(),
        b"in the beginning".to_vec(),
        b"let x = 4".to_vec(),
        b"fn main() {".to_vec(),
    ];
    for seed in &seeds {
        let out = fuga::tm_generate_cosine_gate_v2(
            &tm, &kan, seed, 200, 9, 2, &patch_vocab, 0.0, 0.01, 0, 0.001, 0.30, 0.0, 0.20,
        );
        println!(
            "[V2] {:?} ({}B): {:?}",
            String::from_utf8_lossy(seed),
            out.len(),
            String::from_utf8_lossy(&out).chars().take(90).collect::<String>()
        );
    }
    println!("== v6 validate done ==");
}
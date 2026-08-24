// macro_validate.rs — A/B-стенд Byte-H-JEPA: MB2 (без макро) vs MB3 (с макро).
// Использование: macro_validate <ckpt.fuga> [корпуса...]
// Читает MACRO_W (tag=7) через load_unified_macro; если секции нет — w_macro
// нулевой, и MB3 должен быть БИТ-В-БИТ равен MB2 (проверка детерминизма).
use fuga::ai::htm_temporal::{load_unified, load_unified_macro};
use std::collections::HashSet;
use std::io::BufRead;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ckpt = args.get(1).cloned().unwrap_or_else(|| "fuga_v8_1_m1.fuga".into());
    let corpora: Vec<&str> = if args.len() > 2 {
        args[2..].iter().map(|s| s.as_str()).collect()
    } else {
        vec![
            "fisig_corpus.jsonl",
            "corpus_doc_code_pairs.jsonl",
            "training_stack.jsonl",
            "corpus.jsonl",
        ]
    };

    let (w, w_patch, owm_p, meta, _, _) = load_unified(&ckpt).expect("load_unified");
    let w_macro = load_unified_macro(&ckpt).unwrap_or_else(|| vec![0.0f32; 512 * 512]);
    println!(
        "ckpt={} ctx={} steps={} |W|={:.3} |Wp|={:.3} macro={}",
        ckpt,
        meta.ctx,
        meta.steps,
        (w.iter().map(|x| x * x).sum::<f32>()).sqrt(),
        (w_patch.iter().map(|x| x * x).sum::<f32>()).sqrt(),
        if w_macro.iter().any(|&v| v != 0.0) { "ОБУЧЕН" } else { "нулевой" }
    );

    // Патч-вокабуляр из корпусов (те же 5000 2-байт патчей, что в обучении).
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut patch_vocab: Vec<Vec<u8>> = Vec::new();
    for corp in &corpora {
        let Ok(f) = std::fs::File::open(corp) else { continue };
        let rd = std::io::BufReader::new(f);
        for line in rd.lines().flatten() {
            if line.len() < 4 {
                continue;
            }
            for w2 in line.as_bytes().windows(2) {
                if seen.insert(w2.to_vec()) && patch_vocab.len() < 5000 {
                    patch_vocab.push(w2.to_vec());
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

    let mut tm = fuga::ai::htm_temporal::TemporalMemory::new(64, meta.ctx as usize);
    tm.apply_byte_w(w.clone());
    let kan = fuga::ai::kan::KanTransition::new();
    let _ = owm_p;

    let seeds: Vec<Vec<u8>> = vec![
        b"the force of gravity is".to_vec(),
        b"in the beginning".to_vec(),
        b"let x = 4".to_vec(),
        b"fn main() {".to_vec(),
    ];
    let win = meta.ctx as usize + 1; // окно декодера = ctx+1

    println!("── A/B: MB2 (без макро) vs MB3 (βm=0.3) vs MB3 (βm=0.6) ──");
    for seed in &seeds {
        let out2 = fuga::tm_generate_megabyte_v2(
            &tm, seed, 200, win, 2, &patch_vocab, 8, 0.3, 0.20, 0.8, 0.001, 0.30, 0.05,
        );
        let out3a = fuga::tm_generate_megabyte_v3(
            &tm, seed, 200, win, 2, &patch_vocab, 8, 0.3, 0.20, 0.8, 0.001, 0.30, 0.05,
            &w_macro, 0.3,
        );
        let out3b = fuga::tm_generate_megabyte_v3(
            &tm, seed, 200, win, 2, &patch_vocab, 8, 0.3, 0.20, 0.8, 0.001, 0.30, 0.05,
            &w_macro, 0.6,
        );
        let same = out2 == out3a;
        let same2 = out3a == out3b;
        println!(
            "[{}] MB2={}B  MB3(0.3)={}B{}  MB3(0.6)={}B{}",
            String::from_utf8_lossy(seed).chars().take(18).collect::<String>(),
            out2.len(),
            out3a.len(),
            if same { " [==MB2]" } else { " [≠MB2]" },
            out3b.len(),
            if same2 { " [==MB3(0.3)]" } else { " [≠MB3(0.3)]" },
        );
        if !same {
            println!(
                "    MB2 : {:?}",
                String::from_utf8_lossy(&out2).chars().take(60).collect::<String>()
            );
            println!(
                "    MB3 : {:?}",
                String::from_utf8_lossy(&out3a).chars().take(60).collect::<String>()
            );
        }
        let _ = kan;
    }
    println!("== macro validate done ==");
}
// fuga1.rs — ЕДИНЫЙ БИНАРНИК Fuga (24.10, «всё в одном exe»).
//
// Один файл = весь цикл: обучение → валидация → декод → диалог.
// Подкоманды (аргументы 1:1 с исходными бинами):
//   fuga1 train [флаги unified_gpu_train]   — единое обучение FUGA1+KAN (GPU/CPU)
//   fuga1 talk  <ckpt> [corpora...]         — декоды всеми декодерами
//   fuga1 val   <ckpt> <val1.bin> [...]     — доменная валидация (val_domains)
//   fuga1 cap   <ckpt> [corpus]             — тест способностей (capability_test)
//   fuga1 chat  <ckpt> [--mode v2|mb3|all]  — интерактивный bilingual-чат
//              [--corpus c.jsonl] [--len N] [--ru|--en|--code]
//              [--say "фраза"] [--beta-macro X]
//   fuga1 help
//
// Режим chat — документированные условия декода (как в capability_test):
// OWM-P выгружен (P=identity), patch_w=0; V2 = v6.2-конфиг (window=9, plen=12,
// rep_word=0.20, rep_phrase=0.8, β=0); MB3 = top_k=8, β=0.3, βm=0.3.
// Маркеры доменов (--ru/--en/--code) — control-input для моделей, обученных
// с маркерами корпуса v1.1 ([RU] / [EN] / [CODE] в начале документа).
use fuga::ai::htm_temporal::{load_unified, load_unified_macro2, TemporalMemory};
use fuga::ai::kan::KanTransition;
use std::collections::HashSet;
use std::io::{BufRead, Write};

#[path = "unified_gpu_train.rs"]
#[allow(dead_code)]
mod train_cmd;
#[path = "talk_model.rs"]
#[allow(dead_code)]
mod talk_cmd;
#[path = "val_domains.rs"]
#[allow(dead_code)]
mod val_cmd;
#[path = "capability_test.rs"]
#[allow(dead_code)]
mod cap_cmd;
#[path = "filter_corpus.rs"]
#[allow(dead_code)]
mod filter_cmd;
#[path = "codegen_harness.rs"]
#[allow(dead_code)]
mod codetest_cmd;

fn print_help() {
    println!(
        "fuga — единый бинарник. Подкоманды:\n  \
         fuga train [флаги unified_gpu_train]\n  \
         fuga talk  <ckpt.fuga> [corpora...]\n  \
         fuga val   <ckpt.fuga> <val1.bin> [val2.bin ...] [--ctx 8]\n  \
         fuga cap   <ckpt.fuga> [corpus.jsonl] [--beta-macro X]\n  \
         fuga filter <in.jsonl> <out.jsonl> [--drop-quantile 0.15] [--stats-only]\n  \
         fuga chat  <ckpt.fuga> [--mode v2|mb3|v3|all] [--corpus c.jsonl] [--len N]\n  \
         \x20           [--ru|--en|--code] [--say \"фраза\"] [--beta-macro X]\n  \
         \x20           v3: [--temp T] [--top-k K] [--top-p P] [--rep R] [--ngram N] [--macro-boost M]
                       [--syntax] [--syntax-pen P] — Syntax Loss (tree-sitter-python)"
    );
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let sub = argv.get(1).cloned().unwrap_or_else(|| "help".into());
    // rest сохраняет раскладку argv: [0]=имя программы, [1..]=аргументы
    let mut rest: Vec<String> = vec!["fuga".to_string()];
    rest.extend(argv.into_iter().skip(2));
    match sub.as_str() {
        "train" => train_cmd::run(rest),
        "talk" => talk_cmd::run(rest),
        "val" => val_cmd::run(rest),
        "cap" => cap_cmd::run(rest),
        "chat" => chat(&rest),
        "filter" => filter_cmd::run(rest),
        "codetest" => codetest_cmd::run(rest),
        _ => print_help(),
    }
}

/// cyr% и code% декода (та же формула, что capability_test 16.10).
fn cyr_code_pct(text: &str) -> (f64, f64) {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len().max(1) as f64;
    let cyr = chars
        .iter()
        .filter(|c| ('А'..='я').contains(&**c) || **c == 'Ё' || **c == 'ё')
        .count() as f64
        / n
        * 100.0;
    let lower = text.to_lowercase();
    let tokens = ["def ", "fn ", "import ", "let ", "return ", "(", ")", "{", "}", "=", ";"];
    let hits: usize = tokens.iter().map(|t| lower.matches(t).count()).sum();
    (cyr, hits as f64 / n * 100.0)
}

// ─── chat: интерактивный bilingual-чат с моделью ───
fn chat(args: &[String]) {
    if args.len() < 2 {
        eprintln!("Usage: fuga chat <ckpt.fuga> [--mode v2|mb3|all] [--corpus c.jsonl] [--len N] [--ru|--en|--code] [--say \"...\"] [--beta-macro X]");
        std::process::exit(1);
    }
    let ckpt = args[1].clone();
    let mut mode = "v2".to_string();
    let mut corpus = String::new();
    let mut len = 200usize;
    let mut marker: &[u8] = b"";
    let mut beta_macro = 0.3f32;
    let mut says: Vec<String> = Vec::new();
    // Параметры v3 (Top-K/Top-P + температура, динамический rep-penalty,
    // n-gram blocking, усиление W_macro) — запрос пользователя 24.10.
    let mut temperature = 0.02f32;
    let mut top_k = 8usize;
    let mut top_p = 0.95f32;
    let mut rep_penalty = 0.0f32;
    let mut ngram_block = 0usize;
    let mut macro_boost = 0.0f32;
    // Syntax Loss на декоде (25.10): tree-sitter-python демоция кандидатов
    let mut syntax = false;
    let mut syntax_pen = 0.3f32;
    let mut i = 2;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "--mode" => {
                mode = args.get(i + 1).cloned().unwrap_or_else(|| "v2".into());
                i += 2;
            }
            "--corpus" => {
                corpus = args.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            "--len" => {
                len = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(200);
                i += 2;
            }
            "--ru" => {
                marker = b"[RU] ";
                i += 1;
            }
            "--en" => {
                marker = b"[EN] ";
                i += 1;
            }
            "--code" => {
                marker = b"[CODE] ";
                i += 1;
            }
            "--say" => {
                if let Some(s) = args.get(i + 1) {
                    says.push(s.clone());
                }
                i += 2;
            }
            "--beta-macro" => {
                beta_macro = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.3);
                i += 2;
            }
            "--temp" => {
                temperature = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.02);
                i += 2;
            }
            "--top-k" => {
                top_k = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(8);
                i += 2;
            }
            "--top-p" => {
                top_p = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.95);
                i += 2;
            }
            "--rep" => {
                rep_penalty = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.0);
                i += 2;
            }
            "--ngram" => {
                ngram_block = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0);
                i += 2;
            }
            "--macro-boost" => {
                macro_boost = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.0);
                i += 2;
            }
            "--syntax" => {
                syntax = true;
                i += 1;
            }
            "--syntax-pen" => {
                syntax_pen = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.3);
                i += 2;
            }
            _ => i += 1,
        }
    }
    if syntax && marker.is_empty() {
        marker = b"[CODE] ";
        println!("[syntax] Syntax Guard включён → маркер домена [CODE] (python-грамматика)");
    }
    if corpus.is_empty() {
        // корпуса v1.2 / v1.1 / v1.0 — первый существующий
        for c in [
            "train_fuga1/corpus/fuga12_train.jsonl",
            "train_fuga1/corpus/fuga11_train.jsonl",
            "train_fuga1/corpus/fuga1_train.jsonl",
        ] {
            if std::path::Path::new(c).exists() {
                corpus = c.to_string();
                break;
            }
        }
    }

    let mut tm = TemporalMemory::new(64, 4);
    assert!(tm.load_unified_fuga1(&ckpt), "FUGA1 не читается: {}", ckpt);
    // Документированные условия декода (см. capability_test / AGENTS «OWM-P
    // decode-time projector»): проекция W коллапсирует декод → P=identity,
    // patch_w=0.
    let np = tm.patch_predictor().w.len();
    tm.apply_patch_w(vec![0.0f32; np]);
    let mut ident_p = vec![0.0f32; tm.predictor().p.len()];
    for d in 0..512 {
        ident_p[d * 512 + d] = 1.0;
    }
    tm.predictor_mut().p = ident_p;

    // патч-словарь из корпуса (2-байтовые патчи, cap 5000)
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut patch_vocab: Vec<Vec<u8>> = Vec::new();
    if !corpus.is_empty() {
        if let Ok(f) = std::fs::File::open(&corpus) {
            for line in std::io::BufReader::new(f).lines().flatten() {
                if line.len() < 4 {
                    continue;
                }
                for w in line.as_bytes().windows(2) {
                    if seen.insert(w.to_vec()) && patch_vocab.len() < 5000 {
                        patch_vocab.push(w.to_vec());
                    }
                }
                if patch_vocab.len() >= 5000 {
                    break;
                }
            }
        }
    }

    let kan_c: Option<Vec<f32>> = load_unified(&ckpt).and_then(|t| t.5);
    let mut kan = KanTransition::new();
    if let Some(c) = &kan_c {
        if c.len() == kan.c.len() {
            kan.c.clone_from(c);
        }
    }
    // MACRO для MB3: tag=9 (Wz/We) или tag=7 (flat) — fallback без паник
    let (macro_wz, _macro_we) = load_unified_macro2(&ckpt)
        .unwrap_or_else(|| (vec![0.0f32; 512 * 512], vec![0.0f32; 512 * 512]));

    println!(
        "== FUGA CHAT: {} | mode={} len={} corpus={} marker={}=",
        ckpt,
        mode,
        len,
        if corpus.is_empty() { "<нет>" } else { &corpus },
        String::from_utf8_lossy(marker)
    );

    let syntax_guard = if syntax {
        Some(fuga::ai::syntax_guard::SyntaxGuard::new(true, syntax_pen))
    } else {
        None
    };

    let turn = |seed: &[u8]| {
        let mut outs: Vec<(String, Vec<u8>)> = Vec::new();
        if mode == "v2" || mode == "all" {
            let out = fuga::tm_generate_cosine_gate_v2(
                &tm, &kan, seed, len, 9, 12, &patch_vocab, 0.0, 0.01, 0, 0.001, 0.0, 0.0, 0.20, 0.8,
            );
            outs.push(("v2".into(), out));
        }
        if mode == "mb3" || mode == "all" {
            let out = fuga::tm_generate_megabyte_v3(
                &tm,
                seed,
                len,
                9,
                2,
                &patch_vocab,
                8,
                0.3,
                0.20,
                0.8,
                0.0,
                0.30,
                0.0,
                &macro_wz,
                beta_macro,
            );
            outs.push(("mb3".into(), out));
        }
        if mode == "v3" || mode == "all" {
            // V3: Top-K/Top-P + температура + динамический rep-penalty +
            // n-gram blocking + усиление W_macro (параметры через CLI-флаги)
            let gp = fuga::GenParams {
                temperature,
                top_k,
                top_p,
                rep_penalty,
                ngram_block,
                macro_boost,
                rep_word: 0.0,
                rep_phrase: 0.0,
                min_cos: 0.001,
                beta: 0.0,
                alpha: 0.0,
            };
            let out = fuga::tm_generate_cosine_gate_v3(
                &tm,
                &kan,
                seed,
                len,
                9,
                12,
                &patch_vocab,
                &macro_wz,
                &gp,
                syntax_guard.as_ref(),
            );
            outs.push(("v3".into(), out));
        }
        outs
    };

    if !says.is_empty() {
        for s in &says {
            let mut seed = marker.to_vec();
            seed.extend_from_slice(s.as_bytes());
            println!("вы> {}", s);
            for (name, out) in turn(&seed) {
                let text = String::from_utf8_lossy(&out);
                let (cy, cd) = cyr_code_pct(&text);
                println!("fuga[{}]> {}   [len={} cyr%={:.1} code%={:.1}]",
                    name, text, out.len(), cy, cd);
            }
        }
        return;
    }

    println!("пустая строка или exit — выход");
    let stdin = std::io::stdin();
    loop {
        print!("вы> ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let q = line.trim_end().to_string();
        if q.is_empty() || q == "exit" || q == "quit" {
            break;
        }
        let mut seed = marker.to_vec();
        seed.extend_from_slice(q.as_bytes());
        for (name, out) in turn(&seed) {
            println!("fuga[{}]> {}", name, String::from_utf8_lossy(&out));
        }
    }
}

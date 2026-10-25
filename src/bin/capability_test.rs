// capability_test.rs — НОВЫЙ тест способностей модели (директива пользователя 16.10).
//
// Батарея сидов по доменам на каноническом V2-декоде (v6.2: window=9, plen=12,
// rep_word=0.20, rep_phrase=0.8, β=0, OWM-P выгрузлен — документированные условия
// декода). Автоматические метрики на каждый декод (200B):
//   len     — байт
//   utf8%   — доля байт в валидных UTF-8 последовательностях
//   cyr%    — доля кириллицы (RU-индикатор; на EN-сиде — кросс-доменный перенос)
//   code%   — плотность код-токенов (def/fn/import/let/return/(){}=;/) на 100 символов
//   uniq2%  — доля уникальных байтовых биграмм (анти-монотония)
//   top4%   — доля самого частого 4-грамма (сила аттрактора; ниже — разнообразнее)
//
// Usage: capability_test <ckpt.fuga> [corpus.jsonl]
use fuga::ai::kan::KanTransition;
use fuga::ai::htm_temporal::{load_unified_macro2, TemporalMemory};
use std::collections::{HashMap, HashSet};

const MAX: usize = 200;

fn utf8_frac(b: &[u8]) -> f64 {
    // стандартный валидатор UTF-8: доля байт, входящих в валидные последовательности
    let mut i = 0usize;
    let mut ok = 0usize;
    while i < b.len() {
        let c = b[i];
        let len = if c < 0x80 {
            1
        } else if c >= 0xC2 && c < 0xE0 {
            2
        } else if c >= 0xE0 && c < 0xF0 {
            3
        } else if c >= 0xF0 && c < 0xF5 {
            4
        } else {
            0
        };
        if len == 0 {
            i += 1;
            continue;
        }
        if i + len <= b.len() && b[i + 1..i + len].iter().all(|&x| (x & 0xC0) == 0x80) {
            ok += len;
            i += len;
        } else {
            i += 1;
        }
    }
    ok as f64 / b.len().max(1) as f64 * 100.0
}

fn metrics(b: &[u8]) -> (f64, f64, f64, f64, f64) {
    let utf8 = utf8_frac(b);
    let text = String::from_utf8_lossy(b);
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len().max(1) as f64;
    let cyr = chars
        .iter()
        .filter(|c| {
            let c = **c;
            ('А'..='я').contains(&c) || c == 'Ё' || c == 'ё'
        })
        .count() as f64
        / n
        * 100.0;
    let lower = text.to_lowercase();
    let tokens = ["def ", "fn ", "import ", "let ", "return ", "(", ")", "{", "}", "=", ";"];
    let hits: usize = tokens.iter().map(|t| lower.matches(t).count()).sum();
    let code = hits as f64 / n * 100.0;
    // биграммы
    let mut bi: HashSet<Vec<u8>> = HashSet::new();
    let mut total = 0usize;
    for w in b.windows(2) {
        bi.insert(w.to_vec());
        total += 1;
    }
    let uniq2 = bi.len() as f64 / total.max(1) as f64 * 100.0;
    // топ-4грамм
    let mut quads: HashMap<[u8; 4], usize> = HashMap::new();
    let mut qtotal = 0usize;
    for w in b.windows(4) {
        let mut key = [0u8; 4];
        key.copy_from_slice(w);
        *quads.entry(key).or_insert(0) += 1;
        qtotal += 1;
    }
    let top4 = quads.values().max().copied().unwrap_or(0) as f64 / qtotal.max(1) as f64 * 100.0;
    (utf8, cyr, code, uniq2, top4)
}

fn decode(tm: &TemporalMemory, kan: &KanTransition, patch_vocab: &[Vec<u8>], seed: &[u8]) -> Vec<u8> {
    fuga::tm_generate_cosine_gate_v2(
        tm, kan, seed, MAX, 9, 12, patch_vocab, 0.0, 0.01, 0, 0.001, 0.0, 0.0, 0.20, 0.8,
    )
}

fn avg(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

fn main() {
    run(std::env::args().collect());
}

// run() отделён от main для подкоманды `fuga cap ...` единого бинарника (24.10).
pub fn run(args: Vec<String>) {
    if args.len() < 2 {
        eprintln!("Usage: capability_test <ckpt.fuga> [corpus.jsonl]");
        std::process::exit(1);
    }
    let ckpt = &args[1];
    let corpus = args.get(2).cloned().unwrap_or_else(|| "train_fuga1/corpus/fuga1_train.jsonl".into());
    // Контроль βm (Gary 17.10): --beta-macro 0 отключает макро-кондиционирование
    // при прочих равных → отделяет «MB3 не умеет кириллицу» от «макро переключает домен».
    let mut beta_macro = 0.3f32;
    let mut i = 3;
    while i < args.len() {
        if args[i] == "--beta-macro" {
            beta_macro = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(0.3);
            i += 2;
        } else {
            i += 1;
        }
    }

    let mut tm = TemporalMemory::new(64, 4);
    assert!(tm.load_unified_fuga1(ckpt), "FUGA1 не читается: {}", ckpt);
    // Документированные условия декода: OWM-P выгружен (иначе проекция W
    // коллапсирует декод — см. AGENTS «OWM-P decode-time projector»).
    let n = tm.patch_predictor().w.len();
    tm.apply_patch_w(vec![0.0f32; n]);
    let mut ident_p = vec![0.0f32; tm.predictor().p.len()];
    for d in 0..512 {
        ident_p[d * 512 + d] = 1.0;
    }
    tm.predictor_mut().p = ident_p;

    // патч-словарь из корпуса (2-байтовые патчи, cap 5000)
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut patch_vocab: Vec<Vec<u8>> = Vec::new();
    if let Ok(f) = std::fs::File::open(&corpus) {
        use std::io::BufRead;
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

    let kan_c: Option<Vec<f32>> = fuga::ai::htm_temporal::load_unified(ckpt).and_then(|t| t.5);
    let mut kan = KanTransition::new();
    if let Some(c) = &kan_c {
        if c.len() == kan.c.len() {
            kan.c.clone_from(c);
        }
    }
    // MACRO для MB3: tag=9 (Wz) или tag=7 (flat) — fallback без паник
    let (macro_wz, macro_we) = load_unified_macro2(ckpt)
        .unwrap_or_else(|| (vec![0.0f32; 512 * 512], vec![0.0f32; 512 * 512]));

    // Батарея: (домен, сид)
    let battery: Vec<(&str, &[u8])> = vec![
        ("RU", "привет, как дела".as_bytes()),
        ("RU", "модель учится".as_bytes()),
        ("RU", "что такое программирование".as_bytes()),
        ("EN", "the model learns".as_bytes()),
        ("EN", "hello world".as_bytes()),
        ("EN", "in the beginning".as_bytes()),
        ("PY", "def absorb_result".as_bytes()),
        ("PY", "import json, os".as_bytes()),
        ("RS", "fn main() {".as_bytes()),
        ("RS", "let x = 4".as_bytes()),
    ];

    println!("== CAPABILITY TEST: {} | βm={:.2} ==", ckpt, beta_macro);
    println!(
        "{:<6} {:<28} {:>5} {:>6} {:>6} {:>6} {:>6} {:>6}   (MB3: {:>5} {:>6} {:>6})",
        "домен", "сид", "len", "utf8%", "cyr%", "code%", "uniq2%", "top4%", "len", "cyr%", "top4%"
    );
    let mut by_domain: HashMap<&str, Vec<f64>> = HashMap::new(); // cyr%
    let mut code_fracs: Vec<f64> = Vec::new();
    let mut top4_all: Vec<f64> = Vec::new();
    for (domain, seed) in &battery {
        let out = decode(&tm, &kan, &patch_vocab, seed);
        let (u, cyr, code, uniq2, top4) = metrics(&out);
        // MB3: патч-каркас с макро-кондиционированием (W_macro участвует!)
        let out_mb3 = fuga::tm_generate_megabyte_v3(
            &tm,
            seed,
            MAX,
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
        let (_u3, cyr3, _c3, _q3, top43) = metrics(&out_mb3);
        println!(
            "{:<6} {:<28} {:>5} {:>6.1} {:>6.1} {:>6.1} {:>6.1} {:>6.1}   ({:>5} {:>6.1} {:>6.1}) {:?}",
            domain,
            String::from_utf8_lossy(seed),
            out.len(),
            u,
            cyr,
            code,
            uniq2,
            top4,
            out_mb3.len(),
            cyr3,
            top43,
            String::from_utf8_lossy(&out_mb3).chars().take(48).collect::<String>()
        );
        by_domain.entry(domain).or_default().push(cyr);
        if *domain == "PY" || *domain == "RS" {
            code_fracs.push(code);
        }
        top4_all.push(top4);
    }

    // Скоркарта доменов
    println!("== СКОРКАРТА ==");
    for d in ["RU", "EN", "PY", "RS"] {
        if let Some(v) = by_domain.get(d) {
            println!("  {}: средний cyr% = {:.1}", d, avg(v));
        }
    }
    println!("  код-токены на код-сидах: {:.1} / 100 симв.", avg(&code_fracs));
    println!("  средний top4% (аттрактор; ниже — разнообразнее): {:.1}", avg(&top4_all));
    println!("== КОНЕЦ ==");
}

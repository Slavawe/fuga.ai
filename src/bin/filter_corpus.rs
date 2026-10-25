// filter_corpus.rs — Syntax Guard как ФИЛЬТР ОБУЧАЮЩИХ ДАННЫХ (25.10).
//
// Третий уровень Syntax Loss (25.10): (1) метрика v8, (2) демоция на декоде,
// (3) фильтр корпуса — обучение только на синтаксически валидных код-доках.
// Ломающие синтаксис байты в корпусе учат модель невалидным переходам;
// фильтр их убирает ДО обучения.
//
// Scope: «чистые» код-доки (.rs → tree-sitter-rust, .py → tree-sitter-python).
// rc-пары в FUGA-корпусе двойное-кодированы (модель учит JSON-обёртку —
// формат консистентен train/decode) — парсить их байты бессмысленно,
// помечаются json-wrapped и проходят без фильтра. Текстовые домены ([RU]/[EN])
// проходят без фильтра (python/rust грамматика к ним неприменима).
//
// Извлечение текста ЗЕРКАЛИТ extract_bytes тренера (doc + "\n" + code) —
// фильтруется ровно то, что видит модель.
//
// Usage:
//   fuga1 filter <in.jsonl> <out.jsonl> --stats-only
//   fuga1 filter <in.jsonl> <out.jsonl> [--drop-quantile 0.15] [--max-err-kb 3.0]
use serde_json::Value;
use std::io::{BufRead, Write};
use tree_sitter::Parser;

#[derive(Clone, Copy, PartialEq)]
enum Lang {
    Python,
    Rust,
}

/// Извлечение текста дока — зеркало extract_bytes тренера (doc + \n + code).
fn extract(line: &str) -> Option<String> {
    let v: Value = serde_json::from_str(line).ok()?;
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
        return None;
    }
    Some(out)
}

/// Язык код-дока: СНАЧАЛА маркер домена из JSONL (урок 25.10 — эвристики
/// по сигнатрам misclassified [EN]/[RU]-тексты как код: AGENTS.md полон
/// rust-сниппетов, статьи пестрят «def » в кавычках). Код = только [CODE].
fn detect_lang(text: &str) -> Option<Lang> {
    // ПОРЯДОК по журналу IV: СНАЧАЛА маркер [CODE], затем FILE-маркер
    // (code-review 25.10: [EN]/[RU]-док, цитирующий FILE-сниппет, ошибочно
    // классифицировался Rust и фильтровался).
    let head: String = text.chars().take(64).collect();
    if !head.contains("[CODE]") {
        return None;
    }
    const FILE_MARK: &str = "// ==== FILE: ";
    if let Some(pos) = text.find(FILE_MARK) {
        let rest = &text[pos + FILE_MARK.len()..];
        let path = rest.split(" ====").next().unwrap_or("");
        if path.ends_with(".rs") {
            return Some(Lang::Rust);
        }
        if path.ends_with(".py") {
            return Some(Lang::Python);
        }
    }
    let py_signals = ["def ", "import ", "from ", "class ", "self.", "print("];
    let rs_signals = ["fn ", "impl ", "let mut ", "->", "::", "match ", "use "];
    let py: usize = py_signals.iter().map(|s| text.matches(s).count()).sum();
    let rs: usize = rs_signals.iter().map(|s| text.matches(s).count()).sum();
    if py >= 3 && py > rs * 2 {
        return Some(Lang::Python);
    }
    if rs >= 3 && rs > py * 2 {
        return Some(Lang::Rust);
    }
    // [CODE] без ясного языка: по FILE-маркерам внутри (rc-пары — python)
    if text.contains("Задача:") {
        return Some(Lang::Python);
    }
    None
}

fn error_count(lang: Lang, src: &str) -> usize {
    // хвост до границы валидного UTF-8 (как в syntax_guard)
    let bytes = match std::str::from_utf8(src.as_bytes()) {
        Ok(_) => src.as_bytes(),
        Err(e) => &src.as_bytes()[..e.valid_up_to()],
    };
    let mut parser = Parser::new();
    let ok = match lang {
        Lang::Python => parser.set_language(&tree_sitter_python::LANGUAGE.into()),
        Lang::Rust => parser.set_language(&tree_sitter_rust::LANGUAGE.into()),
    };
    if ok.is_err() {
        return 0;
    }
    match parser.parse(bytes, None) {
        Some(tree) => {
            let mut errs = 0usize;
            let mut cursor = tree.walk();
            loop {
                let node = cursor.node();
                if node.is_error() || node.is_missing() {
                    errs += 1;
                }
                if cursor.goto_first_child() {
                    continue;
                }
                while !cursor.goto_next_sibling() {
                    if !cursor.goto_parent() {
                        return errs;
                    }
                }
            }
        }
        None => 0,
    }
}

struct Verdict {
    lang: Option<Lang>, // None = текст или json-wrapped — без фильтра
    bytes: usize,
    err_kb: f64,
}

pub fn run(args: Vec<String>) {
    if args.len() < 3 {
        eprintln!("Usage: fuga1 filter <in.jsonl> <out.jsonl> [--stats-only] [--drop-quantile 0.15] [--max-err-kb 3.0]");
        std::process::exit(1);
    }
    let inp = &args[1];
    let outp = &args[2];
    let mut stats_only = false;
    let mut drop_quantile: Option<f64> = None;
    let mut max_err_kb: Option<f64> = None;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--stats-only" => stats_only = true,
            "--drop-quantile" => {
                drop_quantile = args.get(i + 1).and_then(|v| v.parse().ok());
                i += 1;
            }
            "--max-err-kb" => {
                max_err_kb = args.get(i + 1).and_then(|v| v.parse().ok());
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    let f = std::fs::File::open(inp).unwrap_or_else(|e| panic!("не читается {}: {}", inp, e));
    let mut raw_lines: Vec<String> = Vec::new();
    for line in std::io::BufReader::new(f).lines() {
        if let Ok(l) = line {
            raw_lines.push(l);
        }
    }

    // вердикты по всем докам
    let mut verdicts: Vec<Verdict> = Vec::with_capacity(raw_lines.len());
    for line in &raw_lines {
        let v = match extract(line) {
            Some(text) => {
                if text.trim_start().starts_with("{\"doc\"") {
                    // двойное-кодированная rc-пара: модель учит JSON-обёртку —
                    // фильтр неприменим (формат консистентен train/decode)
                    Verdict { lang: None, bytes: text.len(), err_kb: 0.0 }
                } else {
                    match detect_lang(&text) {
                        Some(lang) => {
                            let bytes = text.len().max(1);
                            let errs = error_count(lang, &text);
                            Verdict { lang: Some(lang), bytes, err_kb: errs as f64 * 1024.0 / bytes as f64 }
                        }
                        None => Verdict { lang: None, bytes: text.len(), err_kb: 0.0 },
                    }
                }
            }
            None => Verdict { lang: None, bytes: 0, err_kb: 0.0 },
        };
        verdicts.push(v);
    }

    // порог: квантиль худших код-доков (по языку) или абсолютный
    fn quantile_threshold(vs: &mut Vec<f64>, q: f64) -> f64 {
        if vs.is_empty() {
            return f64::INFINITY;
        }
        vs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((vs.len() as f64 - 1.0) * (1.0 - q)).round() as usize;
        vs[idx.min(vs.len() - 1)]
    }
    let mut thresholds = [f64::INFINITY, f64::INFINITY]; // [python, rust]
    for (li, lang) in [Lang::Python, Lang::Rust].iter().enumerate() {
        let mut errs: Vec<f64> = verdicts
            .iter()
            .filter(|v| v.lang == Some(*lang))
            .map(|v| v.err_kb)
            .collect();
        if let Some(q) = drop_quantile {
            thresholds[li] = quantile_threshold(&mut errs, q);
        } else if let Some(x) = max_err_kb {
            thresholds[li] = x;
        }
    }

    // статистика
    println!("== FILTER: {} ==", inp);
    for (name, lang) in [("python", Lang::Python), ("rust", Lang::Rust)] {
        let docs: Vec<&Verdict> = verdicts.iter().filter(|v| v.lang == Some(lang)).collect();
        if docs.is_empty() {
            println!("  {}: 0 доков", name);
            continue;
        }
        let mut e: Vec<f64> = docs.iter().map(|v| v.err_kb).collect();
        e.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let med = e[e.len() / 2];
        let p90 = e[e.len() * 9 / 10];
        let bad = e.iter().filter(|&&x| x > thresholds[if lang == Lang::Python { 0 } else { 1 }]).count();
        println!(
            "  {}: {} доков ({:.0} KB), err/KB: median {:.2} p90 {:.2} max {:.2}; порог {:.2} → убрать {}",
            name,
            docs.len(),
            docs.iter().map(|v| v.bytes as f64).sum::<f64>() / 1024.0,
            med,
            p90,
            e[e.len() - 1],
            thresholds[if lang == Lang::Python { 0 } else { 1 }],
            bad
        );
    }
    let text_n = verdicts.iter().filter(|v| v.lang.is_none()).count();
    println!("  текст/пропущено (без фильтра): {} доков", text_n);

    if stats_only {
        return;
    }

    // запись: код-доки выше порога выбрасываются, остальное 1:1
    let out = std::fs::File::create(outp).unwrap_or_else(|e| panic!("не создать {}: {}", outp, e));
    let mut w = std::io::BufWriter::new(out);
    let (mut kept, mut dropped) = (0usize, 0usize);
    let (mut dropped_b, mut kept_b) = (0usize, 0usize);
    for (line, v) in raw_lines.iter().zip(verdicts.iter()) {
        let drop = match v.lang {
            Some(Lang::Python) => v.err_kb > thresholds[0],
            Some(Lang::Rust) => v.err_kb > thresholds[1],
            None => false,
        };
        if drop {
            dropped += 1;
            dropped_b += line.len();
        } else {
            kept += 1;
            kept_b += line.len();
            writeln!(w, "{}", line).unwrap();
        }
    }
    w.flush().unwrap();
    println!(
        "== итог: {} → {} (убрано {} доков / {:.0} KB) ==",
        inp,
        outp,
        dropped,
        dropped_b as f64 / 1024.0
    );
    let _ = (kept, kept_b);
}

fn main() {
    run(std::env::args().collect());
}

// data_prep.rs — подготовка данных для unified_gpu_train (план 22.08, этап 1).
//
// Очистка сырого корпуса → train/val сплит → бинарный поток .bin
// (сырые UTF-8 байты, документы через '\n') — на трейне НЕТ JSON-парсинга,
// файл читается чанками (см. bin-режим unified_gpu_train).
//
// Форматы входа (по расширению, список через запятую):
//   .jsonl — старый формат ({"doc":..,"code":..} / {"chapters":[..]});
//            не-JSON строка берётся как сырой текст.
//   .csv/.tsv — tatoeba-подобный TSV: id \t lang \t text [\t date];
//            --lang фильтрует по колонке 2 (без фильтра — все строки).
//   .txt — по строке на документ.
//
// Выход: <prefix>_train.bin, <prefix>_val.bin + отчёт.
// Сплит детерминированный: каждый round(1/val_frac)-й документ → val.
//
// Usage:
//   data_prep --in "a.jsonl,b.tsv" --out-prefix datasets/v9 --val-frac 0.05 [--lang eng]

use std::io::{BufRead, BufReader, Write};

fn arg(args: &[String], name: &str, def: &str) -> String {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| def.to_string())
}

/// Совместимо с unified_gpu_train::extract_bytes (тот же разбор JSONL).
fn extract_bytes(line: &str) -> Vec<u8> {
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
    } else if let Some(s) = v.as_str() {
        out.push_str(s);
    }
    out.into_bytes()
}

/// Нормализация (план, этап 1.1): UTF-8 уже валиден (String), выкидываем \0 и
/// прочие управляющие кроме \n/\t — чтобы исключить сюрпризы в downstream.
fn clean(s: &str) -> String {
    s.chars()
        .filter(|&c| c == '\n' || c == '\t' || !c.is_control())
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let inputs = arg(&args, "--in", "");
    let prefix = arg(&args, "--out-prefix", "datasets/prep");
    let val_frac: f64 = arg(&args, "--val-frac", "0.05").parse().unwrap_or(0.05);
    let lang = arg(&args, "--lang", ""); // пусто = без фильтра (только для tsv/csv)

    if inputs.is_empty() {
        eprintln!("usage: data_prep --in \"a.jsonl,b.tsv\" --out-prefix datasets/v9 [--val-frac 0.05] [--lang eng]");
        std::process::exit(1);
    }
    let every = val_frac.recip().round().max(2.0) as usize; // каждый N-й док → val

    let mut train_bytes: usize = 0;
    let mut val_bytes: usize = 0;
    let mut train_docs: usize = 0;
    let mut val_docs: usize = 0;
    let mut skipped: usize = 0;
    let mut doc_idx: usize = 0;

    let mut train = std::io::BufWriter::new(std::fs::File::create(format!("{}_train.bin", prefix)).expect("create train"));
    let mut val = std::io::BufWriter::new(std::fs::File::create(format!("{}_val.bin", prefix)).expect("create val"));

    let files: Vec<&str> = inputs.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    for path in &files {
        let ext = std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let f = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("skip {}: {}", path, e);
                continue;
            }
        };
        eprintln!("reading {} [{}]", path, ext);
        for line in BufReader::new(f).lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => {
                    skipped += 1;
                    continue;
                }
            };
            if line.trim().is_empty() {
                continue;
            }
            let doc: Vec<u8> = match ext.as_str() {
                "jsonl" => extract_bytes(&line),
                "csv" | "tsv" => {
                    // tatoeba TSV: id \t lang \t text [\t date]
                    let mut it = line.splitn(4, '\t');
                    let _id = it.next();
                    let l = it.next().unwrap_or("");
                    if !lang.is_empty() && l != lang {
                        skipped += 1;
                        continue;
                    }
                    match it.next() {
                        Some(text) => text.as_bytes().to_vec(),
                        None => {
                            skipped += 1;
                            continue;
                        }
                    }
                }
                _ => line.as_bytes().to_vec(),
            };
            if doc.len() < 16 {
                skipped += 1;
                continue; // мусорные огрызки
            }
            let doc = clean(&String::from_utf8_lossy(&doc)).into_bytes();
            if doc.len() < 16 {
                skipped += 1;
                continue;
            }
            let to_val = doc_idx % every == 0;
            doc_idx += 1;
            use std::io::Write;
            if to_val {
                val.write_all(&doc).and_then(|_| val.write_all(b"\n")).expect("write val");
                val_bytes += doc.len() + 1;
                val_docs += 1;
            } else {
                train.write_all(&doc).and_then(|_| train.write_all(b"\n")).expect("write train");
                train_bytes += doc.len() + 1;
                train_docs += 1;
            }
        }
    }

    train.flush().expect("flush train");
    val.flush().expect("flush val");

    println!("== data_prep: {} → {}", inputs, prefix);
    println!(
        "  train: {} доков, {} байт ({:.1} MB)",
        train_docs,
        train_bytes,
        train_bytes as f64 / 1e6
    );
    println!(
        "  val:   {} доков, {} байт ({:.1} MB)",
        val_docs,
        val_bytes,
        val_bytes as f64 / 1e6
    );
    println!("  skipped: {} (пустые/чужой язык/огрызки <16B)", skipped);
    println!(
        "  средняя длина документа: {:.0} B",
        (train_bytes + val_bytes) as f64 / (train_docs + val_docs).max(1) as f64
    );
    println!("запуск обучения:");
    println!(
        "  unified_gpu_train --jsonl {}_train.bin --val {}_val.bin --out fuga_unified_v9.fuga ...",
        prefix, prefix
    );
}

//! Query / think / ask / readout commands.
//!
//! Extracted from `src/main.rs` during monolith decomposition.

use std::path::Path;
use std::process;

use crate::cli::args::{has_flag, parse_dim, parse_flag_value, parse_flag_values};
use crate::cli::print::is_name_token;
use crate::cli::tm_gen::{lex_rust_code, run_tm_gen};
use crate::cli::inspect::print_usage;
use fuga::core::wave_cube::peek_cube_header;
use fuga::weaver::token_id;
use fuga::{FugaAI, MemoryStore, TokenBuilder, TokenInfo, WaveCube};

pub fn run_think(args: &[String]) {
    let dim = parse_dim(args, 3).unwrap_or(8192);
    let window = 3;

    let text = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("Hello world from Fuga AI");

    let tokens: Vec<TokenInfo> = text
        .split_whitespace()
        .enumerate()
        .map(|(_, word)| TokenInfo {
            id: token_id(&word),
            text: word.to_string(),
        })
        .collect();

    let mut ai = FugaAI::<3, 4>::new(dim, window);

    let output = ai.think(&tokens);
    println!("{}", output.display());

    ai.absorb_knowledge(&output.super_tokens);
    println!(
        "  -> absorbed {} SuperTokens into cube",
        output.super_tokens.len()
    );
    println!(
        "  -> cube entropy after absorb: {:.4}",
        ai.cube.global_entropy()
    );
}

fn run_ask<const N: usize, const S: usize>(
    question: &str,
    cube_path: &str,
    explain: bool,
    summary: bool,
    prompts: &[String],
) {
    if explain || summary {
        let engine = match fuga::AnswerEngine::<N, S>::load(cube_path) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Failed to load cube: {}", e);
                return;
            }
        };
        let result = if prompts.is_empty() {
            engine.search(question)
        } else {
            engine.search_with_prompts(question, prompts)
        };
        let output = if explain {
            engine.format_explain(&result)
        } else {
            engine.format_summary(&result)
        };
        println!("{}", output);
        return;
    }

    let mem_path = cube_path.replace(".bin", "_mem.bin");
    let cube = match WaveCube::<N, S>::load_bin(cube_path) {
        Ok(c) => {
            println!(
                "Cube: {}x{}x{} dim={} ({} cells)",
                S,
                S,
                S,
                c.dim,
                WaveCube::<N, S>::TOTAL_CELLS
            );
            c
        }
        Err(e) => {
            eprintln!("Failed to load cube: {}", e);
            return;
        }
    };
    let memory = match fuga::MemoryStore::load_bin(&mem_path) {
        Ok(m) => {
            println!("Memory: {} entries", m.size());
            m
        }
        Err(e) => {
            eprintln!("Failed to load memory: {}", e);
            return;
        }
    };

    let mut ai = FugaAI::<N, S>::new(cube.dim, 3);
    ai.cube = cube;
    ai.memory = memory;

    let answer = ai.answer(question);
    println!("{}", answer);
}

pub fn run_ask_entry(args: &[String]) {
    let question = args.get(2).map(|s| s.as_str()).unwrap_or("What is light?");
    let cube_path = parse_flag_value(args, 3, "--cube").unwrap_or("fuga_cube.bin");
    let explain = has_flag(args, "--explain") || has_flag(args, "--answer") || has_flag(args, "-e");
    let summary = has_flag(args, "--summary") || has_flag(args, "-s");
    let prompts: Vec<String> = parse_flag_values(args, "--prompt")
        .into_iter()
        .flat_map(|v| {
            v.split(',')
                .map(|s| s.trim().to_uppercase())
                .collect::<Vec<_>>()
        })
        .collect();

    let (ndim, side_len, _dim) = match peek_cube_header(cube_path) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("{}", e);
            return;
        }
    };
    if !prompts.is_empty() {
        println!("  Active prompts: {:?}", prompts);
    }
    match (ndim, side_len) {
        (3, 4) => run_ask::<3, 4>(question, cube_path, explain, summary, &prompts),
        (4, 4) => run_ask::<4, 4>(question, cube_path, explain, summary, &prompts),
        (3, 8) => run_ask::<3, 8>(question, cube_path, explain, summary, &prompts),
        (4, 8) => run_ask::<4, 8>(question, cube_path, explain, summary, &prompts),
        (5, 2) => run_ask::<5, 2>(question, cube_path, explain, summary, &prompts),
        (5, 4) => run_ask::<5, 4>(question, cube_path, explain, summary, &prompts),
        _ => eprintln!("Unsupported cube dimensions: {}×{}", side_len, ndim),
    }
}

pub fn run_readout<const N: usize, const S: usize>(
    query: &str,
    cube_path: &str,
    beam: usize,
    cells_k: usize,
) {
    let cube = match WaveCube::<N, S>::load_bin(cube_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to load cube: {}", e);
            return;
        }
    };
    let mem_path = cube_path.replace(".bin", "_mem.bin");
    let memory = match fuga::MemoryStore::load_bin(&mem_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load memory: {}", e);
            return;
        }
    };

    let mut ai = FugaAI::<N, S>::new(cube.dim, 3);
    ai.cube = cube;
    ai.memory = memory;

    let out = fuga::logit_lens(&mut ai, query, beam, cells_k);

    println!("╔══════════════════════════════════════════════╗");
    println!("║  VSA Logit-Lens Decoder (readout of cube)   ║");
    println!("╚══════════════════════════════════════════════╝");
    println!("  Query:        {:?}", query);
    println!("  Thought cells: {}", out.thought_cells);
    println!("  Cube entropy: {:.4}", out.entropy);
    println!();
    println!("  CONCEPT BEAM (W_unembed projection):");
    for (i, (text, sim)) in out.concepts.iter().enumerate() {
        println!("    {:>2}. {:<24} sim={:.4}", i + 1, text, sim);
    }
    println!();
    if !out.concepts.is_empty() {
        println!("  Readout: {}", out.concepts.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>().join(" "));
    }
}

pub fn run_tm_gen_entry(args: &[String]) {
    if args.len() < 3 {
        eprintln!("Error: missing query");
        print_usage(&args[0]);
        process::exit(1);
    }
    let query = args[2..]
        .iter()
        .take_while(|a| !a.starts_with("--"))
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    let cube_path = parse_flag_value(args, 3, "--cube").unwrap_or("fuga_cube.bin");
    let steps = parse_flag_value(args, 3, "--steps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);
    let window = parse_flag_value(args, 3, "--window")
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let debug = args.iter().any(|a| a == "--debug");

    let tm_path = cube_path.replace(".bin", "_tm.bin");
    let Some(tm) = fuga::TemporalMemory::load(&tm_path) else {
        eprintln!("Failed to load temporal memory: {}", tm_path);
        return;
    };
    if debug {
        let cells: usize = tm.cells.len();
        let synapses: usize = tm.cells.iter().map(|c| c.segments.iter().map(|s| s.synapses.len()).sum::<usize>()).sum();
        println!("[debug] tm={} cells, {} synapses, ctx", cells, synapses);
        let toks: Vec<&str> = query.split_whitespace().collect();
        for t in toks.iter().rev().take(5) {
            let pc = tm.predict_structure(&[*t]).popcount();
            let raw = tm.predict_structure_raw(&[*t]).popcount();
            println!("[debug] predict_structure([{:?}]) pc={} raw_pc={}", t, pc, raw);
        }
    }

    // Кандидаты для декода: слова запроса + слова ближайшей памяти.
    let mem_path = cube_path.replace(".bin", "_mem.bin");
    // `--vocab-dir DIR` overrides the memory source: build candidates from all
    // .rs files in DIR (the lesson/corpus vocab), so the TM can only ever emit
    // tokens that actually appear in that source.
    let mut cands: Vec<String> = query
        .split_whitespace()
        .map(|s| s.to_lowercase())
        .filter(|w| w.len() >= 2)
        .collect();
    if let Some(vocab_dir) = parse_flag_value(args, 3, "--vocab-dir") {
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        fn walk_rs_dir(d: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            if let Ok(entries) = std::fs::read_dir(d) {
                for ent in entries.flatten() {
                    let p = ent.path();
                    let meta = match ent.metadata() {
                        Ok(m) => m,
                        Err(_) => continue,
                    };
                    if meta.is_file() {
                        if p.extension().map(|e| e == "rs").unwrap_or(false) {
                            out.push(p);
                        }
                    } else if meta.is_dir() {
                        walk_rs_dir(&p, out);
                    }
                }
            }
        }
        walk_rs_dir(std::path::Path::new(&vocab_dir), &mut files);
        files.sort();
        let single_ok = |t: &str| {
            matches!(
                t,
                "(" | ")" | "{" | "}" | "[" | "]" | "," | ";" | ":" | "." | "=" | "<" | ">"
            )
        };
        let mut seen: std::collections::HashSet<String> = cands.iter().cloned().collect();
        for f in files {
            if let Ok(content) = std::fs::read_to_string(&f) {
                for tok in lex_rust_code(&content) {
                    let tl = tok.to_lowercase();
                    if (tl.len() >= 2 || single_ok(&tl)) && seen.insert(tl.clone()) {
                        cands.push(tl);
                    }
                }
            }
        }
    } else if let Ok(memory) = fuga::MemoryStore::load_bin(&mem_path) {
        let hits = memory.search_by_text(&query, 10);
        for (_i, _s, e) in &hits {
            for w in e.text.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-') {
                let wl = w.to_lowercase();
                if wl.len() >= 2 && !cands.contains(&wl) {
                    cands.push(wl);
                }
            }
        }
    }
    if cands.len() > 3000 {
        cands.truncate(3000);
    }

    let seed: Vec<String> = query.split_whitespace().map(|s| s.to_lowercase()).collect();

    // ── Two-speed bridge: H-JEPA task-mask → TM corridor ────────────────
    // Optional `--task "<words>"` + `--task-sim <floor>`: the upper level
    // sanctions a corridor (eligible Set) of tokens whose SDR is cosine-close
    // to the task union; the local TM autoregressor may only emit inside it.
    // This is the architectural fix for "right tokens, wrong order": content
    // comes from the task/identity channel, ORDER still from the TM syntax
    // graph. Enabled only when a task text is given.
    let eligible: Option<std::collections::HashSet<String>> =
        if let Some(task_text) = parse_flag_value(args, 3, "--task") {
            let floor = parse_flag_value(args, 3, "--task-sim")
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0);
            if task_text.trim().is_empty() {
                None
            } else {
                let task_words: Vec<String> = task_text
                    .split_whitespace()
                    .filter(|w| w.len() >= 2)
                    .map(|s| s.to_lowercase())
                    .collect();
                let task_sdrs: Vec<_> = task_words
                    .iter()
                    .map(|w| fuga::encode_text(w))
                    .collect();
                let mut task_bits = [0u64; fuga::SDR_WORDS];
                for s in &task_sdrs {
                    for i in 0..fuga::SDR_WORDS {
                        task_bits[i] |= s.bits[i];
                    }
                }
                let task_pop: f64 = task_bits
                    .iter()
                    .map(|w| w.count_ones() as f64)
                    .sum();
                let mut elig = std::collections::HashSet::new();
                for w in &cands {
                    let sdr = fuga::encode_text(w);
                    let cand_pop: f64 = sdr.bits.iter().map(|b| b.count_ones() as f64).sum();
                    let shared: f64 = sdr
                        .bits
                        .iter()
                        .zip(task_bits.iter())
                        .map(|(a, b)| (a & b).count_ones() as f64)
                        .sum();
                    let sim = if cand_pop * task_pop > 0.0 {
                        shared / (cand_pop * task_pop).sqrt()
                    } else {
                        0.0
                    };
                    if sim >= floor {
                        elig.insert(w.clone());
                    }
                }
                println!("  Eligible corridor: {} tokens (task-sim ≥ {})", elig.len(), floor);
                Some(elig)
            }
        } else {
            None
        };

    let out = fuga::tm_generate(&tm, &seed, steps, &cands, window, eligible.as_ref());

    println!("╔══════════════════════════════════════════════╗");
    println!("║  Temporal-Memory Sequential Generator       ║");
    println!("╚══════════════════════════════════════════════╝");
    println!("  Query:   {:?}", query);
    println!("  Steps:   {} (generated {})", steps, out.len());
    println!("  Cand:    {} words", cands.len());
    println!();
    if out.is_empty() {
        println!("  (нет временных предсказаний — память не дала следующего токена)");
    } else {
        println!("  Sequence: {}", out.join(" "));
    }
}

pub fn run_jepa_train_entry(args: &[String]) {
    let tm_path = parse_flag_value(args, 2, "--tm").unwrap_or("fuga_stack_tm.bin");
    let corpus_path = parse_flag_value(args, 2, "--corpus").unwrap_or("training_stack.jsonl");
    let out_path = parse_flag_value(args, 2, "--out").unwrap_or("fuga_hjepa.bin");
    let dim = parse_flag_value(args, 2, "--dim")
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    let limit = parse_flag_value(args, 2, "--limit")
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);

    // --load <path> — warm-start: продолжить обучение с уже обученного чекпоинта
    // (саморекурсивный цикл агента: впитал новые уроки → дотянул модель).
    let load_path = parse_flag_value(args, 2, "--load");

    // feed_learn_hv_only учит JEPA в пространстве сырых word-SDR и TM не
    // трогает — тяжёлый чекпоинт грузим только если явно запрошен.
    let tm = match parse_flag_value(args, 2, "--tm") {
        Some(p) => match fuga::TemporalMemory::load(&p) {
            Some(t) => t,
            None => {
                eprintln!("No temporal memory at {}", p);
                return;
            }
        },
        None => fuga::TemporalMemory::new(32, 3),
    };
    let hjepa = match &load_path {
        Some(p) => match fuga::HierarchicalJEPA::load(p) {
            Ok(h) => {
                println!("  Warm-start: continued from {}", p);
                h
            }
            Err(e) => {
                eprintln!("  Failed to load {}: {} — starting fresh", p, e);
                fuga::HierarchicalJEPA::new(dim)
            }
        },
        None => fuga::HierarchicalJEPA::new(dim),
    };
    let mut tp = fuga::TemporalPredictor::new(tm, hjepa);
    println!("═══ H-JEPA corpus training ═══");
    println!("  TM:     {} ({} cells)", tm_path, tp.tm.cells.len());
    println!("  JEPA:   dim={}", dim);
    println!("  Corpus: {}", corpus_path);

    let docs = match fuga::load_corpus(&corpus_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to load corpus: {}", e);
            return;
        }
    };
    println!("  Docs:   {} ({})", docs.len(), docs.len().min(limit));

    let mut builder = TokenBuilder::new();
    let _ = builder.load_configs_from_dir("tikones");
    let flat_vocab = builder.build_flat_vocab();

    let progress_every = parse_flag_value(args, 2, "--progress-every")
        .and_then(|v| v.parse().ok())
        .unwrap_or(250);

    let mut n_tokens = 0usize;
    let mut n_learns = 0usize;
    let mut last_errs: [f64; 3] = [1.0; 3];
    let t0 = std::time::Instant::now();
    let n = docs.len().min(limit);
    for (di, doc) in docs.iter().enumerate() {
        if di >= n {
            break;
        }
        for ch in &doc.chapters {
            let heading = ch.heading.as_deref().unwrap_or("");
            for para in &ch.paragraphs {
                let combined = format!("{}: {}", heading, para);
                let tokens = fuga::tokenize_corpus_text(&combined, &flat_vocab);
                for t in &tokens {
                    n_tokens += 1;
                    let errs = tp.feed_learn_hv_only(&t.text);
                    for (i, &e) in errs.iter().enumerate().take(3) {
                        last_errs[i] = e;
                    }
                    if errs.iter().any(|e| *e < 1.0) {
                        n_learns += 1;
                    }
                }
            }
        }
        // Print on a document cadence so progress is visible and never looks
        // frozen (the old learns-milestone log went silent for ~320k tokens).
        if (di + 1) % progress_every == 0 || di + 1 >= n {
            println!(
                "  [{}/{}] tokens={} learns={} err=({:.3},{:.3},{:.3}) {:.0}s",
                di + 1,
                n,
                n_tokens,
                n_learns,
                last_errs[0],
                last_errs.get(1).copied().unwrap_or(1.0),
                last_errs.get(2).copied().unwrap_or(1.0),
                t0.elapsed().as_secs_f64()
            );
        }
    }
    println!("\n  Tokens fed: {} · learns: {}", n_tokens, n_learns);
    println!("  Elapsed: {:.1}s", t0.elapsed().as_secs_f64());

    match tp.hjepa.save(&out_path) {
        Ok(()) => println!("✓ H-JEPA -> {}", out_path),
        Err(e) => println!("  Save failed: {}", e),
    }
    println!("{}", tp.stats());
}

pub fn run_hjepa_gen_entry(args: &[String]) {
    if args.len() < 3 {
        eprintln!("Error: missing query");
        print_usage(&args[0]);
        process::exit(1);
    }
    let query = args[2..]
        .iter()
        .take_while(|a| !a.starts_with("--"))
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    let steps = parse_flag_value(args, 3, "--steps")
        .and_then(|v| v.parse().ok())
        .unwrap_or(24);
    let min_sim = parse_flag_value(args, 3, "--min-sim")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.5);

    let tm_path = "fuga_stack_tm.bin";
    let jepa_path = "fuga_hjepa.bin";
    // Корпус-обученная пара (fuga_stack) — предпочтительна; mirror — фолбэк.
    let (tm_path, jepa_path) = if std::path::Path::new(tm_path).exists()
        && std::path::Path::new(jepa_path).exists()
    {
        (tm_path, jepa_path)
    } else {
        ("fuga_mirror_tm.bin", "fuga_mirror_jepa.bin")
    };
    let Some(tm) = fuga::TemporalMemory::load(tm_path) else {
        eprintln!("No {}", tm_path);
        return;
    };
    let hjepa = match fuga::HierarchicalJEPA::load(jepa_path) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("No {} ({})", jepa_path, e);
            return;
        }
    };
    let mut tp = fuga::TemporalPredictor::new(tm, hjepa);

    let mut words: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|w| w.len() >= 2)
        .map(|w| w.to_lowercase())
        .collect();
    const KEYWORDS: &[&str] = &[
        "tokio", "async", "await", "runtime", "task", "spawn", "stream", "tcp", "net", "io",
        "read", "write", "buffer", "channel", "mutex", "future", "poll", "fn", "use",
        "let", "mut", "impl", "struct", "trait", "result", "error", "ok", "code", "main",
        "join", "select", "timeout", "sleep", "interval", "socket", "connection", "handler",
    ];
    for k in KEYWORDS {
        words.push(k.to_string());
    }
    if words.len() > 800 {
        words.truncate(800);
    }
    let vocab = tp.word_vocab(&words);
    let seq = tp.generate_words(&query, steps, &vocab, min_sim);

    println!("╔══════════════════════════════════════════════╗");
    println!("║  H-JEPA Latent Sequence Generator           ║");
    println!("╚══════════════════════════════════════════════╝");
    println!("  Query:  {:?}", query);
    println!("  Steps:  {} (decoded {})", steps, seq.len());
    println!("  Vocab:  {} words", vocab.len());
    println!();
    if seq.is_empty() {
        println!("  (латентный ролл-аут не декодировался ни в одно слово)");
    } else {
        println!("  Sequence: {}", seq.join(" "));
    }
}

pub fn run_readout_entry(args: &[String]) {
    if args.len() < 3 {
        eprintln!("Error: missing query");
        print_usage(&args[0]);
        process::exit(1);
    }
    let query = args[2..]
        .iter()
        .filter(|a| !a.starts_with("--"))
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    let cube_path = parse_flag_value(args, 3, "--cube").unwrap_or("fuga_cube.bin");
    let beam = parse_flag_value(args, 3, "--beam")
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let cells = parse_flag_value(args, 3, "--cells")
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);

    let (ndim, side_len, _dim) = match peek_cube_header(cube_path) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("{}", e);
            return;
        }
    };
    match (ndim, side_len) {
        (3, 4) => run_readout::<3, 4>(&query, cube_path, beam, cells),
        (4, 4) => run_readout::<4, 4>(&query, cube_path, beam, cells),
        (3, 8) => run_readout::<3, 8>(&query, cube_path, beam, cells),
        (4, 8) => run_readout::<4, 8>(&query, cube_path, beam, cells),
        (5, 2) => run_readout::<5, 2>(&query, cube_path, beam, cells),
        (5, 4) => run_readout::<5, 4>(&query, cube_path, beam, cells),
        _ => eprintln!("Unsupported cube dimensions: {}×{}", side_len, ndim),
    }
}


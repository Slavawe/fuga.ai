//! Training command implementations.
//!
//! Extracted from `src/main.rs` during monolith decomposition
//! (Phase 1 of docs/refactor-plan.md). Fisig/unified/stack/omni
//! training pipelines.

use std::fs;

use fuga::{
    CodeQualityFilter, CorpusDoc, FugaAI, MemoryStore, TemporalMemory, TokenBuilder, TokenInfo,
    WaveCube,
};

use crate::cli::args::{has_flag, parse_dim, parse_flag_value, parse_float, parse_int};
use fuga::core::wave_cube::peek_cube_header;
use fuga::weaver::token_id;

pub fn run_fisig_train(args: &[String]) {
    let corpus_path = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("fisig_corpus.jsonl");
    let dim = args
        .get(3)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(8192);
    let window = 3;
    let save_path = "fisig_cube.bin";

    let mut builder = TokenBuilder::new();
    let _ = builder.load_configs_from_dir("tikones");
    let flat_vocab = builder.build_flat_vocab();

    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Fuga Fisig — Field Physics Model               ║");
    println!("║  Aether · warp · Mach effect · ZPF · GEM        ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();
    println!("  Corpus:     {}", corpus_path);
    println!("  Dimension:  {}D", dim);
    println!("  Cube side:  4");
    println!("  Window:     {}", window);
    println!("  Save to:    {}\n", save_path);

    let content = match std::fs::read_to_string(corpus_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read {}: {}", corpus_path, e);
            return;
        }
    };

    let docs: Vec<CorpusDoc> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();

    println!("  Documents:  {}\n", docs.len());

    let mut ai = FugaAI::<3, 4>::new(dim, window);
    let mut total_paras = 0;

    for (di, doc) in docs.iter().enumerate() {
        let title = doc.title.as_deref().unwrap_or("Untitled");
        let author = doc.author.as_deref().unwrap_or("Unknown");
        let ch_count: usize = doc.chapters.iter().map(|ch| ch.paragraphs.len()).sum();
        println!(
            "  [{}/{}] {} — {} ({} paragraphs)",
            di + 1,
            docs.len(),
            title,
            author,
            ch_count
        );

        for ch in &doc.chapters {
            for para in &ch.paragraphs {
                let tokens = fuga::tokenize_corpus_text(para, &flat_vocab);
                if tokens.len() < 3 {
                    continue;
                }
                total_paras += 1;
                ai.absorb_with_source(&tokens, title);
            }
        }
        println!(
            "    entropy={:.4} mem={}",
            ai.cube.global_entropy(),
            ai.memory.size()
        );
    }

    if let Err(e) = ai.cube.save_bin(save_path) {
        eprintln!("Cube save failed: {}", e);
    } else {
        println!("\n  Cube saved to {}", save_path);
    }
    let mem_path = save_path.replace(".bin", "_mem.bin");
    if let Err(e) = ai.memory.save_bin(&mem_path) {
        eprintln!("Memory save failed: {}", e);
    } else {
        println!(
            "  Memory saved to {} ({} entries)",
            mem_path,
            ai.memory.size()
        );
    }

    println!("\n  === Fuga Fisig Training Complete ===");
    println!("  Paragraphs: {}", total_paras);
    println!("  Entropy:    {:.4}", ai.cube.global_entropy());
    println!("  Coherence:  {:.4}", ai.cube.coherence());

    println!("\n  --- Probe: aether density gradient ---");
    let answer = ai.answer("aether density gradient gravity");
    for line in answer.lines().take(20) {
        println!("  {}", line);
    }
}

pub fn run_train_unified(args: &[String]) {
    if args.len() < 4 {
        eprintln!("Usage: fuga train-unified <source1>[,<source2>,...] [options]");
        eprintln!("  Sources: code:<dir>, corpus:<jsonl>, omni:<jsonl>, fisig:<jsonl>");
        eprintln!("  Options: --dim <N> --ndim <N> --side <N> --save <path> --window <N>");
        eprintln!(
            "  Example: fuga train-unified code:src,corpus:corpus.jsonl --dim 1024 --ndim 5 --side 4 --save unified.bin"
        );
        return;
    }

    let dim = parse_dim(&args, 3).unwrap_or(1024);
    let ndim = args
        .iter()
        .find(|a| a.starts_with("--ndim"))
        .and_then(|a| a.split('=').nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let side = args
        .iter()
        .find(|a| a.starts_with("--side"))
        .and_then(|a| a.split('=').nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let window = args
        .iter()
        .find(|a| a.starts_with("--window"))
        .and_then(|a| a.split('=').nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let save_path = args
        .iter()
        .find(|a| a.starts_with("--save"))
        .and_then(|a| a.split('=').nth(1))
        .unwrap_or("unified_cube.bin");

    let sources: Vec<&str> = args[2].split(',').collect();

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  Fuga Unified Training — Multi-Source Fusion               ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!(
        "  Dimension: {}D | Cube: {}^{} ({} cells) | Window: {}",
        dim,
        side,
        ndim,
        (side as u32).pow(ndim as u32),
        window
    );
    println!("  Sources: {}", sources.join(", "));
    println!("  Save to:  {}\n", save_path);

    let mem_path = save_path.replace(".bin", "_mem.bin");
    let (mut ai, start_mem) = if std::path::Path::new(save_path).exists() {
        println!("  Loading existing cube from {}", save_path);
        let cube = match WaveCube::<5, 4>::load_bin(save_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to load cube: {}", e);
                return;
            }
        };
        let memory = if std::path::Path::new(&mem_path).exists() {
            match fuga::MemoryStore::load_bin(&mem_path) {
                Ok(m) => {
                    println!("  Loaded memory: {} entries", m.size());
                    m
                }
                Err(e) => {
                    eprintln!("Memory load failed: {}", e);
                    fuga::MemoryStore::new()
                }
            }
        } else {
            fuga::MemoryStore::new()
        };
        let memory_size = memory.size();
        let mut ai = FugaAI::<5, 4>::new(cube.dim, 3);
        ai.cube = cube;
        ai.memory = memory;
        (ai, memory_size)
    } else {
        (FugaAI::<5, 4>::new(dim, 3), 0)
    };

    println!("  Start memory: {} entries\n", start_mem);

    for src in sources {
        let (kind, path) = match src.split_once(':') {
            Some((k, p)) => (k, p),
            None => {
                eprintln!("Source format: kind:path (e.g., code:src)");
                continue;
            }
        };

        match kind {
            "code" => {
                println!("  📁 Code source: {}", path);
                train_code_source(&mut ai, path);
            }
            "corpus" => {
                println!("  📄 Corpus source: {}", path);
                train_corpus_source(&mut ai, path);
            }
            "omni" => {
                println!("  🧠 Omni source: {}", path);
                train_omni_source(&mut ai, path);
            }
            "fisig" => {
                println!("  ⚛️  Fisig source: {}", path);
                train_fisig_source(&mut ai, path);
            }
            _ => eprintln!("  Unknown source kind: {}", kind),
        }
        println!(
            "    → entropy={:.4} mem={}",
            ai.cube.global_entropy(),
            ai.memory.size()
        );
    }

    if let Err(e) = ai.cube.save_bin(save_path) {
        eprintln!("Cube save failed: {}", e);
    } else {
        println!("\nCube saved to {}", save_path);
    }
    if let Err(e) = ai.memory.save_bin(&mem_path) {
        eprintln!("Memory save failed: {}", e);
    } else {
        println!(
            "Memory saved to {} ({} entries)",
            mem_path,
            ai.memory.size()
        );
    }

    println!("\n=== Unified Training Complete ===");
    println!("  Entropy:   {:.4}", ai.cube.global_entropy());
    println!("  Coherence: {:.4}", ai.cube.coherence());
    println!("  Memory:    {} entries", ai.memory.size());
}

pub fn run_train_stack(args: &[String]) {
    let corpus_path = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("training_stack.jsonl");
    let dim = parse_dim(&args, 3).unwrap_or(8192);
    let save_path = args
        .iter()
        .position(|a| a == "--save")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "fuga_stack.bin".to_string());
    let tm_cap = parse_int(&args, "--tm-cap").unwrap_or(8192);
    let ctx = parse_int(&args, "--ctx").unwrap_or(4);
    let tm_per_doc = parse_int(&args, "--tm-per-doc").unwrap_or(512);
    let lr = args
        .iter()
        .position(|a| a == "--lr")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(0.1);
    let ndim = parse_int(&args, "--ndim").unwrap_or(5);
    let side = parse_int(&args, "--side").unwrap_or(4);
    if !(3..=5).contains(&ndim) || !(2..=8).contains(&side) {
        eprintln!("Unsupported ndim/side: {}/{}", ndim, side);
        return;
    }
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  Fuga 2.0 Unified Training Stack — TM · SDR · VSA · H-JEPA   ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!(
        "  Corpus:     {} | Cube: {}^{} dim={}",
        corpus_path, side, ndim, dim
    );
    println!(
        "  TM: cap={} ctx={} lr={} | Save: {}",
        tm_cap, ctx, lr, save_path
    );

    let docs = match fuga::load_corpus(corpus_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to load corpus: {}", e);
            return;
        }
    };
    println!("  Loaded {} documents\n", docs.len());

    let mut ai = FugaAI::<5, 4>::new(dim, 3);
    let mut tm = fuga::TemporalMemory::new(tm_cap, ctx);
    let mut builder = TokenBuilder::new();
    let _ = builder.load_configs_from_dir("tikones");
    let flat_vocab = builder.build_flat_vocab();

    let mut paras = 0usize;
    let mut seq_count = 0usize;
    let t0 = std::time::Instant::now();
    for (di, doc) in docs.iter().enumerate() {
        let mut doc_seqs = 0usize;
        for ch in &doc.chapters {
            let heading = ch.heading.as_deref().unwrap_or("");
            for para in &ch.paragraphs {
                let combined = format!("{}: {}", heading, para);
                let tokens = fuga::tokenize_corpus_text(&combined, &flat_vocab);
                // VSA: WaveCube + MemoryStore absorb — full coverage of every doc.
                ai.absorb_with_source(&tokens, doc.title.as_deref().unwrap_or("untitled"));
                // TM/HTM + SDR + H-JEPA: sliding-window structure learning, capped per doc
                // so total synapse growth stays bounded for the full unified corpus.
                if doc_seqs < tm_per_doc {
                    let refs: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
                    let mut w: Vec<&str> = Vec::with_capacity(ctx);
                    for i in 0..refs.len() {
                        w.push(refs[i]);
                        if w.len() > ctx {
                            w.remove(0);
                        }
                        if i + 1 < refs.len() {
                            tm.learn_structure_lr(&w, refs[i + 1], lr);
                            doc_seqs += 1;
                            seq_count += 1;
                            if doc_seqs >= tm_per_doc {
                                break;
                            }
                        }
                    }
                }
                paras += 1;
            }
        }
        if (di + 1) % 500 == 0 || di + 1 == docs.len() {
            print!(
                "\r  [{}/{}] docs · {} paras · TM cells={} seqs={} · mem={} entropy={:.4}",
                di + 1,
                docs.len(),
                paras,
                tm.cells.len(),
                seq_count,
                ai.memory.size(),
                ai.cube.global_entropy()
            );
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }
    println!();

    let mem_path = save_path.replace(".bin", "_mem.bin");
    let tm_path = save_path.replace(".bin", "_tm.bin");
    if let Err(e) = ai.cube.save_bin(&save_path) {
        eprintln!("Cube save failed: {}", e);
    } else {
        println!("✓ VSA cube -> {}", save_path);
    }
    if let Err(e) = ai.memory.save_bin(&mem_path) {
        eprintln!("Memory save failed: {}", e);
    } else {
        println!("✓ VSA memory -> {} ({} entries)", mem_path, ai.memory.size());
    }
    tm.save(&tm_path);
    println!("✓ TM/HTM (+SDR cells, +H-JEPA latent W) -> {}", tm_path);

    println!("\n=== Unified Training Stack Complete ===");
    println!("  Docs: {} | Paras: {} | TM sequences: {}", docs.len(), paras, seq_count);
    println!("  TM cells: {} | {}", tm.cells.len(), tm.stats());
    println!("  Entropy:   {:.4}", ai.cube.global_entropy());
    println!("  Coherence: {:.4}", ai.cube.coherence());
    println!("  Memory:    {} entries", ai.memory.size());
    println!("  Time:      {:.1}s", t0.elapsed().as_secs_f64());
}

fn train_code_source(ai: &mut FugaAI<5, 4>, dir: &str) {
    let mut filter = CodeQualityFilter::new(ai.dim);
    let results = match filter.scan_directory(dir, true) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Scan failed: {}", e);
            return;
        }
    };
    println!("  Found {} supported files", results.len());

    for (path, score) in &results {
        if score.weight <= 0.0 {
            continue;
        }
        let source = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let tokens: Vec<TokenInfo> = source
            .split_whitespace()
            .enumerate()
            .map(|(_, w)| TokenInfo {
                id: token_id(&w),
                text: w.to_string(),
            })
            .collect();
        ai.accumulate_df(&tokens);
    }
    ai.compute_idf();
    println!(
        "  IDF: {} terms, {} docs",
        ai.idf_weights.len(),
        ai.total_docs
    );

    for (path, score) in &results {
        if score.weight <= 0.0 {
            continue;
        }
        let source = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let tokens: Vec<TokenInfo> = source
            .split_whitespace()
            .enumerate()
            .map(|(_, w)| TokenInfo {
                id: token_id(&w),
                text: w.to_string(),
            })
            .collect();
        ai.absorb_with_quality(&tokens, path, score, &source);
    }
}

fn train_corpus_source(ai: &mut FugaAI<5, 4>, path: &str) {
    let content = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read: {}", e);
            return;
        }
    };
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let tokens: Vec<TokenInfo> = line
            .split_whitespace()
            .enumerate()
            .map(|(_, w)| TokenInfo {
                id: token_id(&w),
                text: w.to_string(),
            })
            .collect();
        ai.absorb_with_source(&tokens, path);
    }
}

fn train_omni_source(ai: &mut FugaAI<5, 4>, path: &str) {
    let content = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read: {}", e);
            return;
        }
    };
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let tokens: Vec<TokenInfo> = line
            .split_whitespace()
            .enumerate()
            .map(|(_, w)| TokenInfo {
                id: token_id(&w),
                text: w.to_string(),
            })
            .collect();
        ai.absorb_with_source(&tokens, path);
    }
}

fn train_fisig_source(ai: &mut FugaAI<5, 4>, path: &str) {
    let content = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read: {}", e);
            return;
        }
    };
    let mut paras = 0;
    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let tokens: Vec<TokenInfo> = line
            .split_whitespace()
            .enumerate()
            .map(|(_, w)| TokenInfo {
                id: token_id(&w),
                text: w.to_string(),
            })
            .collect();
        ai.absorb_with_source(&tokens, path);
        paras += 1;
    }
    println!("  Absorbed {} paragraphs", paras);
}

fn run_fisig_query<const N: usize, const S: usize>(query: &str, cube_path: &str, window: usize) {
    let cube = match WaveCube::<N, S>::load_bin(cube_path) {
        Ok(c) => {
            println!("Fisig cube: {}x{}x{} dim={}", S, S, S, c.dim);
            c
        }
        Err(e) => {
            eprintln!("Failed to load cube: {}", e);
            return;
        }
    };

    let mem_path = cube_path.replace(".bin", "_mem.bin");
    let memory = match fuga::MemoryStore::load_bin(&mem_path) {
        Ok(m) => {
            println!("Memory: {} entries\n", m.size());
            m
        }
        Err(e) => {
            eprintln!("Memory: {}", e);
            fuga::MemoryStore::new()
        }
    };

    let mut builder = TokenBuilder::new();
    let _ = builder.load_configs_from_dir("tikones");
    let flat_vocab = builder.build_flat_vocab();
    let tokens = fuga::tokenize_corpus_text(query, &flat_vocab);

    let dim = cube.dim;
    let mut ai = FugaAI::<N, S>::new(dim, window);
    ai.cube = cube;
    ai.memory = memory;

    println!("  Query: {}\n", query);

    let answer = fuga::fisig_formatter::format_answer(&mut ai, query, &tokens);
    println!("{}", fuga::fisig_formatter::render_fisig_answer(&answer));
}

pub fn run_fisig_query_entry(args: &[String]) {
    let query = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("Tesla ether theory");
    let cube_path = parse_flag_value(args, 3, "--cube").unwrap_or("fisig_cube.bin");
    let window = 3;
    run_fisig_query::<3, 4>(query, cube_path, window);
}

pub fn run_omni_train(args: &[String]) {
    let corpus_path = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("omni_corpus.jsonl");
    let dim = args
        .get(3)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(8192);
    let save_path = args.get(4).map(|s| s.as_str()).unwrap_or("omni_cube.bin");
    let ndim = args
        .get(5)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4);

    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Fuga Omni 1.0 — Unified Brain Training         ║");
    println!("║  Physics · Code · Spatial · Reactor · Cross     ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();
    let side: usize = args
        .get(6)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4);
    println!("  Corpus:     {}", corpus_path);
    println!("  Dim:        {}", dim);
    println!("  Cube:       {}x{}x{} dim={}", side, side, side, dim);
    println!("  Save to:    {}\n", save_path);

    match (ndim, side) {
        (3, 4) => run_omni_train_inner::<3, 4>(corpus_path, dim, save_path),
        (4, 4) => run_omni_train_inner::<4, 4>(corpus_path, dim, save_path),
        (3, 8) => run_omni_train_inner::<3, 8>(corpus_path, dim, save_path),
        (4, 8) => run_omni_train_inner::<4, 8>(corpus_path, dim, save_path),
        (5, 2) => run_omni_train_inner::<5, 2>(corpus_path, dim, save_path),
        (5, 4) => run_omni_train_inner::<5, 4>(corpus_path, dim, save_path),
        _ => eprintln!("Unsupported ndim/side: {}/{}", ndim, side),
    }
}

fn run_omni_train_inner<const N: usize, const S: usize>(
    corpus_path: &str,
    dim: usize,
    save_path: &str,
) {
    let mut engine = fuga::omni::OmniEngine::<N, S>::new(dim, 3);
    match fuga::omni::omni_train(&mut engine.ai, corpus_path, save_path) {
        Ok((paras, entropy, coherence)) => {
            println!();
            println!("  === Fuga Omni 1.0 Training Complete ===");
            println!("  Paragraphs: {}", paras);
            println!("  Entropy:    {:.4}", entropy);
            println!("  Coherence:  {:.4}", coherence);
        }
        Err(e) => eprintln!("  Training failed: {}", e),
    }
}

fn run_omni_query<const N: usize, const S: usize>(query: &str, cube_path: &str) {
    let dim = 8192usize;
    let mut engine = fuga::omni::OmniEngine::<N, S>::new(dim, 3);
    if let Err(e) = engine.load_cube(cube_path) {
        eprintln!("Failed to load cube: {}", e);
        return;
    }

    let mut builder = TokenBuilder::new();
    let _ = builder.load_configs_from_dir("tikones");
    let flat_vocab = builder.build_flat_vocab();
    let tokens = fuga::tokenize_corpus_text(query, &flat_vocab);

    let result = engine.query(query, &tokens);
    println!("{}", fuga::omni::render_omni_result(&result));
}

pub fn run_omni(args: &[String]) {
    let query = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("Fuga Omni architecture");
    let cube_path = args
        .get(3)
        .filter(|s| !s.starts_with("--"))
        .or_else(|| args.get(2))
        .map(|s| s.as_str())
        .unwrap_or("omni_cube.bin");

    let (ndim, side_len, _dim) = match peek_cube_header(cube_path) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("{}", e);
            return;
        }
    };
    match (ndim, side_len) {
        (4, 4) => run_omni_query::<4, 4>(query, cube_path),
        (3, 4) => run_omni_query::<3, 4>(query, cube_path),
        (3, 8) => run_omni_query::<3, 8>(query, cube_path),
        (4, 8) => run_omni_query::<4, 8>(query, cube_path),
        (5, 2) => run_omni_query::<5, 2>(query, cube_path),
        (5, 4) => run_omni_query::<5, 4>(query, cube_path),
        _ => eprintln!("Unsupported cube dimensions: {}×{}", side_len, ndim),
    }
}


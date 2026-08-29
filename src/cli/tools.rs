//! Solve / query / codegen / weave / tokenize / scan / UI tools.
//!
//! Extracted from `src/main.rs` during monolith decomposition.

use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::process;

use crate::cli::args::{parse_dim, parse_flag_value, parse_flag_values, has_flag, parse_output, parse_window};
use crate::cli::print::is_name_token;
use crate::cli::tm_gen::lex_rust_code;
use crate::cli::inspect::print_usage;
use fuga::core::wave_cube::peek_cube_header;
use fuga::weaver::token_id;
use fuga::{
    CodeQualityFilter, FugaAI, MemoryStore, OutputFormat, TokenBuilder, TokenExplorer, TokenInfo,
    TokenVocabulary, WaveCube, WeaverEngine, summarize_quality,
};

fn run_solve<const N: usize, const S: usize>(problem: &str, cube_path: &str) {
    let mem_path = cube_path.replace(".bin", "_mem.bin");

    let cube = match WaveCube::<N, S>::load_bin(cube_path) {
        Ok(c) => {
            println!("Cube: {}x{}x{} dim={}", S, S, S, c.dim);
            c
        }
        Err(e) => {
            eprintln!("Failed to load cube: {}", e);
            return;
        }
    };
    let memory = match fuga::MemoryStore::load_bin(&mem_path) {
        Ok(m) => {
            println!("Memory: {} entries\n", m.size());
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

    let solution = ai.solve(problem);
    println!("{}", solution);
}

pub fn run_solve_entry(args: &[String]) {
    let problem = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("What forces act on a body in motion and how does light refract?");
    let cube_path = parse_flag_value(args, 3, "--cube").unwrap_or("fuga_cube.bin");
    let (ndim, side_len, _dim) = match peek_cube_header(cube_path) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("{}", e);
            return;
        }
    };
    match (ndim, side_len) {
        (3, 4) => run_solve::<3, 4>(problem, cube_path),
        (4, 4) => run_solve::<4, 4>(problem, cube_path),
        (3, 8) => run_solve::<3, 8>(problem, cube_path),
        (4, 8) => run_solve::<4, 8>(problem, cube_path),
        (5, 2) => run_solve::<5, 2>(problem, cube_path),
        (5, 4) => run_solve::<5, 4>(problem, cube_path),
        _ => eprintln!("Unsupported cube dimensions: {}x{}", side_len, ndim),
    }
}

fn run_query<const N: usize, const S: usize>(text: &str, cube_path: &str, window: usize) {
    let cube = match WaveCube::<N, S>::load_bin(cube_path) {
        Ok(c) => {
            println!("Cube loaded: {}x{}x{} dim={}", S, S, S, c.dim);
            c
        }
        Err(e) => {
            eprintln!("Failed to load cube: {}", e);
            return;
        }
    };

    let mut builder = TokenBuilder::new();
    let _ = builder.load_configs_from_dir("tikones");
    let flat_vocab = builder.build_flat_vocab();

    let tokens = fuga::tokenize_corpus_text(text, &flat_vocab);
    println!("Query: {:?} ({} tokens)", text, tokens.len());

    let mut ai = FugaAI::<N, S>::new(cube.dim, window);
    ai.cube = cube;

    let output = ai.think(&tokens);
    println!("Route: {}", output.route.name());

    for st in &output.super_tokens {
        let cells = ai.query_memory(st);
        println!(
            "\nSuperToken ({} raw tokens): {} resonance hits",
            st.raw_tokens.len(),
            cells.len()
        );
        for cell in cells.iter().take(10) {
            println!(
                "  Cell ({},{},{}): score={:.4}",
                cell.x, cell.y, cell.z, cell.score
            );
        }
    }

    println!("\nCube entropy:  {:.4}", ai.cube.global_entropy());
    println!("Cube coherence: {:.4}", ai.cube.coherence());
}

pub fn run_query_entry(args: &[String]) {
    let text = args.get(2).map(|s| s.as_str()).unwrap_or("Newton");
    let _dim = parse_dim(args, 3).unwrap_or(8192);
    let cube_path = parse_flag_value(args, 3, "--cube").unwrap_or("fuga_cube.bin");
    let window = 3;

    let (ndim, side_len, _dim) = match peek_cube_header(cube_path) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("{}", e);
            return;
        }
    };
    match (ndim, side_len) {
        (3, 4) => run_query::<3, 4>(text, cube_path, window),
        (4, 4) => run_query::<4, 4>(text, cube_path, window),
        (3, 8) => run_query::<3, 8>(text, cube_path, window),
        (4, 8) => run_query::<4, 8>(text, cube_path, window),
        (5, 2) => run_query::<5, 2>(text, cube_path, window),
        (5, 4) => run_query::<5, 4>(text, cube_path, window),
        _ => eprintln!("Unsupported cube dimensions: {}x{}", side_len, ndim),
    }
}

fn run_codegen<const N: usize, const S: usize>(
    seed: &str,
    cube_path: &str,
    max_tokens: usize,
    temperature: f64,
) {
    let cube = match WaveCube::<N, S>::load_bin(cube_path) {
        Ok(c) => {
            println!(
                "Cube: {} dim={} ({} cells)",
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

    let mem_path = cube_path.replace(".bin", "_mem.bin");

    let mut ai = FugaAI::<N, S>::new(cube.dim, 3);
    ai.cube = cube;
    if let Ok(m) = fuga::MemoryStore::load_bin(&mem_path) {
        println!("Memory: {} entries", m.size());
        ai.memory = m;
    } else {
        println!("No memory file found at {}", mem_path);
    }

    println!("=== Fuga CodeGen ===");
    println!("Seed:     {}", seed);
    println!("Max tokens: {}", max_tokens);
    println!("Temperature: {:.2}\n", temperature);

    let result = fuga::ai::codegen::generate::<N, S>(&mut ai, seed, max_tokens, temperature);
    println!("{}", result.display());
    println!("\nGenerated text:");
    println!("{}", result.to_text());
}

pub fn run_codegen_entry(args: &[String]) {
    let seed = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("write a PID controller");
    let cube_path = parse_flag_value(args, 3, "--cube").unwrap_or("fuga_cube.bin");
    let max_tokens = parse_flag_value(args, 3, "--max-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let temperature = parse_flag_value(args, 3, "--temperature")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.6);

    let (ndim, side_len, _dim) = match peek_cube_header(cube_path) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("{}", e);
            return;
        }
    };
    match (ndim, side_len) {
        (3, 4) => run_codegen::<3, 4>(seed, cube_path, max_tokens, temperature),
        (4, 4) => run_codegen::<4, 4>(seed, cube_path, max_tokens, temperature),
        (3, 8) => run_codegen::<3, 8>(seed, cube_path, max_tokens, temperature),
        (4, 8) => run_codegen::<4, 8>(seed, cube_path, max_tokens, temperature),
        (5, 2) => run_codegen::<5, 2>(seed, cube_path, max_tokens, temperature),
        (5, 4) => run_codegen::<5, 4>(seed, cube_path, max_tokens, temperature),
        _ => eprintln!("Unsupported cube dimensions: {}×{}", side_len, ndim),
    }
}

pub fn run_weave(args: &[String]) {
    let dim = parse_dim(args, 3).unwrap_or(8192);
    let window = parse_window(args, 3).unwrap_or(4);
    let path = args.get(2);

    println!("Fuga Weaver — VSA Token Compression Engine");
    println!();
    println!("  Configuration:");
    println!("    Dimension:  {}D", dim);
    println!("    Window size: {}", window);
    if let Some(p) = path {
        println!("    Source:      {}", p);
    }
    println!();

    let mut builder = TokenBuilder::new();
    match builder.load_configs_from_dir("tikones") {
        Ok(count) => println!("  Loaded {} tokenizer config(s) from tikones/\n", count),
        Err(e) => eprintln!("  Config load: {}\n", e),
    }

    println!("{}", builder.report());

    let vocab = builder.build_flat_vocab();
    println!("  Flat vocab size: {}\n", vocab.len());

    let source = if let Some(p) = path {
        match std::fs::read_to_string(p) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to read {}: {}", p, e);
                return;
            }
        }
    } else {
        "Hello world, this is a test of the Fuga Weaver engine!".to_string()
    };

    let tokens: Vec<TokenInfo> = source
        .split_whitespace()
        .enumerate()
        .map(|(_, word)| TokenInfo {
            id: token_id(&word),
            text: word.to_string(),
        })
        .collect();

    println!("  Source chars: {}", source.len());
    println!("  Raw tokens:   {}", tokens.len());
    println!();

    let mut weaver = WeaverEngine::new(dim, window);
    let result = weaver.compress_stream(&tokens, None);
    println!("{}", result.display());

    if !result.super_tokens.is_empty() {
        let vocab = TokenVocabulary::from_builder(&builder, dim);
        let ids: HashSet<u32> = tokens.iter().map(|t| t.id).collect();
        let unweave_result = weaver.unweave_stream_filtered(&result.super_tokens, &vocab, &ids);
        println!("{}", unweave_result.display());
    }
}

pub fn run_unweave(args: &[String]) {
    let dim = parse_dim(args, 3).unwrap_or(8192);
    let window = parse_window(args, 3).unwrap_or(4);
    let path = args.get(2);

    let mut builder = TokenBuilder::new();
    let _ = builder.load_configs_from_dir("tikones");

    let source = if let Some(p) = path {
        match std::fs::read_to_string(p) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to read {}: {}", p, e);
                return;
            }
        }
    } else {
        "Hello world, this is a test of the Fuga Weaver engine!".to_string()
    };

    let tokens: Vec<TokenInfo> = source
        .split_whitespace()
        .enumerate()
        .map(|(_, word)| TokenInfo {
            id: token_id(&word),
            text: word.to_string(),
        })
        .collect();

    let mut weaver = WeaverEngine::new(dim, window);
    let result = weaver.compress_stream(&tokens, None);

    println!("Fuga Unweave — Resonance Token Recovery");
    println!();
    println!(
        "  Input: {} tokens → {} SuperTokens ({:.2}x)",
        tokens.len(),
        result.super_tokens.len(),
        result.compression_ratio
    );

    let vocab = TokenVocabulary::from_builder(&builder, dim);
    let ids: HashSet<u32> = tokens.iter().map(|t| t.id).collect();
    let unweave = weaver.unweave_stream_filtered(&result.super_tokens, &vocab, &ids);

    println!(
        "  Recovered: {} / {} tokens",
        unweave.recovered_tokens.len(),
        unweave.total_original
    );
    println!("  Accuracy:  {:.1}%", unweave.accuracy * 100.0);
    println!("  Avg sim:   {:.4}", unweave.avg_similarity);
    println!();
    println!("Sample recovery:");
    for i in 0..tokens.len().min(unweave.recovered_tokens.len()).min(10) {
        let orig = &tokens[i];
        let rec = &unweave.recovered_tokens[i];
        let match_str = if orig.id == rec.id { "OK" } else { "MIS" };
        println!(
            "  [{}] orig={}:{:20} → rec={}:{:20} {}",
            i, orig.id, orig.text, rec.id, rec.text, match_str
        );
    }
}

pub fn run_tokenize(args: &[String]) {
    let dim = parse_dim(args, 3).unwrap_or(8192);
    let count = parse_window(args, 3).unwrap_or(10);

    let mut builder = TokenBuilder::new();
    let _ = builder.load_configs_from_dir("tikones");
    let vocab = TokenVocabulary::from_builder(&builder, dim);

    let explorer = TokenExplorer::new(dim);

    println!("Fuga Tokenizer — VSA Token Synthesis & Exploration");
    println!();

    let report = explorer.explore_tokens(&vocab);
    println!("{}", report.display());

    println!("Generating {} new synthetic tokens...", count);
    let new_tokens = explorer.generate_new_tokens(&vocab, count);
    for (id, text, hv) in &new_tokens {
        println!(
            "  New token {}: {:32} entropy={:.3}",
            id,
            text,
            hv.entropy()
        );
    }

    println!();
    let (name, hv) = explorer.synthesize_concept_chain(
        &["fuga", "weaver", "vsa", "token"],
        fuga::TokenRole::SPECIAL,
    );
    println!(
        "Synthesized concept '{}': entropy={:.3}",
        name,
        hv.entropy()
    );

    let nearest = vocab.nearest(&hv);
    if let Some((id, text, sim)) = nearest {
        println!("  Nearest vocab match: {}:{} (sim={:.4})", id, text, sim);
    }
}

pub fn run_code_quality(path: &str, dim: usize, recursive: bool) {
    println!("Fuga Code Quality Filter — Tree-Sitter Analysis");
    println!("  Path:  {}", path);
    println!("  Dim:   {}", dim);
    if recursive {
        println!("  Mode:  recursive");
    }
    println!();

    let mut filter = CodeQualityFilter::new(dim);
    let results = match filter.scan_directory(path, recursive) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{}", e);
            return;
        }
    };

    println!("{}", summarize_quality(&results));

    let absorbed: Vec<_> = results.iter().filter(|(_, s)| s.weight > 0.0).collect();
    let blocked: Vec<_> = results.iter().filter(|(_, s)| s.weight <= 0.0).collect();

    if !blocked.is_empty() {
        println!("\n--- Blocked (w=0.00) ---");
        for (path, score) in blocked.iter().take(20) {
            println!("  {}  {}", score.display(), path);
        }
        if blocked.len() > 20 {
            println!("  ... ({} more)", blocked.len() - 20);
        }
    }
    if !absorbed.is_empty() {
        println!("\n--- Absorbed (w>0.00) ---");
        for (path, score) in absorbed.iter().take(10) {
            println!("  {}  {}", score.display(), path);
        }
        if absorbed.len() > 10 {
            println!("  ... ({} more)", absorbed.len() - 10);
        }
    }
}

pub fn run_scan(path: &str, dim: usize, output: Option<&str>) {
    println!("Fuga Security Scan — AST Pattern Audit");
    println!("  Target: {}", path);
    println!("  Dim:    {}", dim);
    println!();

    let mut filter = CodeQualityFilter::new(dim);
    let results = match filter.scan_directory(path, true) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{}", e);
            return;
        }
    };

    let mut findings = Vec::new();
    let dangerous_patterns = [
        ("eval()", "eval("),
        ("innerHTML", ".innerHTML"),
        ("exec()", ".exec("),
        ("unsafe-eval", "unsafe-eval"),
        ("Function()", "new Function("),
        ("document.write", "document.write"),
        ("localStorage", "localStorage"),
        ("shell injection", "$("),
    ];

    for (path, score) in &results {
        if score.weight <= 0.0 {
            continue;
        }
        let source = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        for (name, pattern) in &dangerous_patterns {
            for (i, line) in source.lines().enumerate() {
                if line.contains(pattern) {
                    findings.push((
                        path.clone(),
                        name.to_string(),
                        i + 1,
                        line.trim().to_string(),
                    ));
                }
            }
        }
    }

    if findings.is_empty() {
        println!("✓ No security issues found.");
        return;
    }

    let report = findings
        .iter()
        .map(|(p, n, l, s)| format!("  {}:{}  [{}]  {}", p, l, n, s))
        .collect::<Vec<_>>()
        .join("\n");

    match output {
        Some(file) => {
            std::fs::write(file, &report).unwrap_or_else(|e| eprintln!("Write error: {}", e));
            println!("Report saved to {}", file);
        }
        None => {
            println!("Security findings ({} total):\n{}", findings.len(), report);
        }
    }
}

pub fn run_ui(prompt: &str, _dim: usize, output: Option<&str>) {
    println!("Fuga UI Generator — expert_code synthesis\n");
    println!("Prompt: {}\n", prompt);

    let mut moe = fuga::MoEStore::new("fuga_code_cube");
    match moe.load_domain("code") {
        Ok(()) => {}
        Err(e) => {
            eprintln!("Failed to load code domain: {}", e);
            return;
        }
    }

    println!("Searching {} code patterns...\n", moe.domain_size("code"));

    let results = moe.search_by_text("code", prompt, 10);
    if results.is_empty() {
        println!("No matching UI patterns found in memory.");
        return;
    }
    let mut seen = std::collections::HashSet::new();
    let html: String = results
        .iter()
        .filter(|(_, _, e)| seen.insert(e.text.as_str()))
        .map(|(_, _, e)| format!("// {}\n{}", e.source_doc, e.text))
        .collect::<Vec<_>>()
        .join("\n---\n");
    match output {
        Some(f) => {
            std::fs::write(f, &html).unwrap_or_else(|e| eprintln!("Write error: {}", e));
            println!("Saved to {}", f);
        }
        None => println!("{}", html),
    }
}

pub fn save_agent_result(content: &str) -> String {
    let dir = "agent_results";
    std::fs::create_dir_all(dir).ok();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let path = format!("{}/agent_{}.txt", dir, ts);
    std::fs::write(&path, content).ok();
    path
}


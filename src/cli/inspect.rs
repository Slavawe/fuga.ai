//! Inspect / self-mirror / JEPA set commands.
//!
//! Extracted from `src/main.rs`.

use std::fs;
use std::process;

use crate::cli::args::{parse_dim, parse_int, has_flag, parse_float};
use crate::cli::jepa::load_sdr_store;
use crate::cli::tm_gen::{load_tm, load_tm_from};
use crate::cli::jepa::encode_chunk;
use fuga::weaver::token_id;
use fuga::{
    FugaAI, HierarchicalJEPA, SDR_DIM, TemporalMemory, TokenBuilder, TokenInfo, WaveCube,
};

pub fn run_inspect_text(text: &str) {
    println!("═══ Fuga Inspect ═══\n");
    let profile = fuga::StyloProfile::compute(text);
    println!("  Unique ratio:    {:.2}%", profile.unique_ratio * 100.0);
    println!("  Token entropy:   {:.4} bits", profile.token_entropy);
    println!("  Struct entropy:  {:.4} bits", profile.structural_entropy);
    println!("  Avg line len:    {:.1} chars", profile.avg_line_len);
    println!();

    let sdr = fuga::encode_text(text);
    let pop = sdr.popcount();
    let overlap = if pop > 0 {
        sdr.overlap(&sdr) as f64 / SDR_DIM as f64
    } else {
        0.0
    };
    println!("  SDR popcount:    {}", pop);
    println!(
        "  Density:         {:.2}% (target 2%)",
        pop as f64 / SDR_DIM as f64 * 100.0
    );
    println!("  Self-overlap:    {:.4}", overlap);

    let mem = load_sdr_store("fuga_sdr_index.bin");
    if let Some(ref store) = mem {
        let results = store.index.search(&sdr, 5);
        if results.is_empty() {
            println!("\n  Top-0 matches — context is novel");
        } else {
            println!("\n  Top-{} SDR matches:", results.len());
            for (_i, score, snippet) in &results {
                println!("    [{:.4}] {}", score, snippet);
            }
        }
    }

    let tm = load_tm();
    if let Some(mut tm) = tm {
        let (_pred, match_score) = tm.feed(&sdr);
        println!("\n  TM match_score:  {:.4}", match_score);
        println!("  TM resonance:    {:.1}%", match_score * 100.0);
    }

    println!("\n═══ end inspect ═══");
}

pub fn run_inspect_file(path: &str) {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  Can't read {}: {}", path, e);
            return;
        }
    };
    println!("═══ Fuga Inspect: {} ═══\n", path);
    let lines = content.lines().count();
    let bytes = content.len();
    let profile = fuga::StyloProfile::compute(&content);
    println!("  Lines:           {}", lines);
    println!("  Bytes:           {}", bytes);
    println!("  Unique ratio:    {:.2}%", profile.unique_ratio * 100.0);
    println!("  Token entropy:   {:.4} bits", profile.token_entropy);
    println!("  Struct entropy:  {:.4} bits", profile.structural_entropy);
    println!("  Avg line len:    {:.1} chars", profile.avg_line_len);
    println!();

    let sdr = fuga::encode_text(&content);
    let mem = load_sdr_store("fuga_sdr_index.bin");
    if let Some(ref store) = mem {
        let results = store.index.search(&sdr, 5);
        if results.is_empty() {
            println!("  No matching nodes in SDR index");
        } else {
            println!("  Top-{} SDR matches:", results.len());
            for (_i, score, snippet) in &results {
                println!("    [{:.4}] {}", score, snippet);
            }
        }
    }

    println!("\n═══ end inspect ═══");
}

pub fn load_hjepa() -> Option<fuga::HierarchicalJEPA> {
    let model_path = "fuga_hjepa.bin";
    if std::path::Path::new(model_path).exists() {
        match fuga::HierarchicalJEPA::load(model_path) {
            Ok(h) => Some(h),
            Err(e) => {
                eprintln!("  H-JEPA load failed: {}", e);
                None
            }
        }
    } else {
        let hjepa = fuga::HierarchicalJEPA::new(8192);
        println!("  Created fresh H-JEPA (dim=8192)");
        Some(hjepa)
    }
}

pub fn run_tm_jepa_repl() {
    let tm = match load_tm() {
        Some(t) => t,
        None => {
            eprintln!("No fuga_htm.bin. Run 'fuga htm-train' first.");
            return;
        }
    };
    let hjepa = match load_hjepa() {
        Some(h) => h,
        None => return,
    };
    let mut tp = fuga::TemporalPredictor::new(tm, hjepa);

    println!("═══ Fuga TM→JEPA Bridge ═══");
    println!("  Enter text lines (one = one step). Empty line to quit.");
    println!("  Each line → TM feed + H-JEPA learn\n");

    loop {
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).is_err() || line.trim().is_empty() {
            break;
        }
        let (tm_match, errors) = tp.feed_learn(line.trim());
        print!("  tm={:.4}  err=[", tm_match);
        for (i, e) in errors.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!("{:.4}", e);
        }
        println!("]  {}", tp.stats());
    }
    println!("\n  Session done. {}", tp.stats());
}

pub fn run_self_mirror() {
    let tm = match load_tm() {
        Some(t) => t,
        None => {
            eprintln!("No fuga_htm.bin. Run 'fuga htm-train' first.");
            return;
        }
    };
    let hjepa = match load_hjepa() {
        Some(h) => h,
        None => return,
    };
    let mut mirror = fuga::SelfMirror::new(tm, hjepa);
    println!("═══ Fuga Self-Mirror ═══");
    println!("  Indexing src/ai/ ...\n");
    let total = mirror.index_dir("src/ai");
    println!("\n  Indexed {} phase nodes", total);
    mirror.save();
}

pub fn run_mirror_index(dir: &str) {
    let mut mirror = match fuga::SelfMirror::load() {
        Some(m) => {
            println!("  Loaded existing mirror ({} nodes)", m.nodes.len());
            m
        }
        None => {
            let tm = load_tm().unwrap_or_else(|| {
                println!("  Creating fresh TemporalMemory");
                fuga::TemporalMemory::new(2000, 4)
            });
            let hjepa = load_hjepa().unwrap_or_else(|| {
                println!("  Creating fresh H-JEPA (dim=8192)");
                fuga::HierarchicalJEPA::new(8192)
            });
            fuga::SelfMirror::new(tm, hjepa)
        }
    };
    println!("═══ Fuga Mirror-Index: {} ═══\n", dir);
    let total = mirror.index_dir_fast(dir);
    println!(
        "\n  Indexed {} phase nodes from {} (total: {})",
        total,
        dir,
        mirror.nodes.len()
    );
    mirror.save();
}

pub fn run_inspect_dir(dir: &str) {
    let mirror = match fuga::SelfMirror::load() {
        Some(m) => m,
        None => {
            eprintln!("No mirror data. Run 'fuga mirror-index' first.");
            return;
        }
    };
    println!("═══ Fuga Inspect-Dir: {} ═══\n", dir);
    let reports = mirror.inspect_dir(dir);
    println!("\n  Total files: {}", reports.len());
    let anomalies: Vec<_> = reports.iter().filter(|r| r.anomaly_score > 0.5).collect();
    if anomalies.is_empty() {
        println!("  No anomalous files found.");
    } else {
        println!("\n  ⚠ Anomalous files (score > 0.5):");
        for r in &anomalies {
            println!(
                "    {:.3}  {}  (entropy={:.2} density={:.3})",
                r.anomaly_score, r.path, r.entropy, r.sdr_density
            );
        }
    }
}

pub fn run_auto_correct(path: &str) {
    let mut engine = match fuga::AutoCorrectEngine::load() {
        Some(e) => e,
        None => {
            eprintln!("No mirror data. Run 'fuga mirror-index' first.");
            return;
        }
    };
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Can't read {}: {}", path, e);
            return;
        }
    };
    println!("═══ Fuga Auto-Correct: {} ═══\n", path);
    let report = engine.mirror.inspect_file(path);
    println!(
        "  Lines: {}  Entropy: {:.2}  Density: {:.3}  Anomaly: {:.3}",
        report.lines, report.entropy, report.sdr_density, report.anomaly_score
    );
    if report.anomaly_score > 0.5 {
        println!("\n  ⚠ High anomaly score, scanning blocks for corrections...\n");
        let mut total_patches = 0;
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.len() > 10 {
                let (_text, patches) = engine.apply_correction(trimmed);
                if !patches.is_empty() {
                    println!("  {}", patches.join("\n  "));
                    total_patches += 1;
                }
            }
        }
        if total_patches == 0 {
            println!(
                "  No L2 divergences detected in {} blocks.",
                content.lines().count()
            );
        } else {
            println!("\n  {} corrections generated", total_patches);
        }
    } else {
        println!("  No anomalies detected.");
    }
    println!("\n  {}", engine.stats());
}

pub fn run_train_predictor(epochs: usize, chunk_size: usize, use_ff: bool) {
    let mut mirror = match fuga::SelfMirror::load() {
        Some(m) => m,
        None => {
            eprintln!("No mirror data. Run 'fuga mirror-index' first.");
            return;
        }
    };
    println!("═══ Fuga Train Predictor ═══\n");
    if use_ff {
        mirror.train_predictor_ff(epochs, chunk_size);
    } else {
        mirror.train_predictor_chunked(epochs, chunk_size);
    }
    mirror.save();
    println!("  Mirror saved.");
}

pub fn run_generate_code(
    text: &str,
    beam_width: usize,
    temperature: f64,
    gen_mode: bool,
    token_mode: bool,
) {
    let mut mirror = match fuga::SelfMirror::load() {
        Some(m) => m,
        None => {
            eprintln!("No mirror data. Run 'fuga mirror-index' first.");
            return;
        }
    };
    mirror.predictor.load_buffer();
    let steps = 5;
    println!("═══ Fuga Generate ═══\n");
    println!("  Input: {}\n", text);
    if beam_width > 1 {
        println!("  Beam: {}  Temp: {}\n", beam_width, temperature);
    }
    let node_count = mirror.nodes.len();
    let total_cells = mirror.predictor.tm.cells.len();
    let total_seg: usize = mirror
        .predictor
        .tm
        .cells
        .iter()
        .map(|c| c.segments.len())
        .sum();
    let raw_preds = mirror.generate_code_beam(text, steps, beam_width, temperature);
    if raw_preds.is_empty() || raw_preds.iter().all(|s| s.is_empty()) {
        println!("  Not enough context to predict. Provide more text.");
        return;
    }
    let preds: Vec<(String, String, String, usize, u32)> = raw_preds
        .iter()
        .filter_map(|step| {
            step.first().map(|(node, overlap)| {
                (
                    node.path.clone(),
                    node.kind.clone(),
                    node.name.clone(),
                    node.line,
                    *overlap,
                )
            })
        })
        .collect();
    mirror.predictor.save_buffer();

    if token_mode {
        let vocab_count = mirror.build_token_vocab_from_files();
        println!("  Token vocab: {} entries", vocab_count);
        let trained = mirror.train_token_sequences(2000, 10);
        println!("  TM trained: {} token steps", trained);
        mirror.save();
        println!("  VSA token-level generation...\n");
        let tokens = mirror.generate_tokens(text, 100);
        if tokens.is_empty() {
            eprintln!("  Generation failed (no tokens produced)");
            return;
        }
        for chunk in tokens.chunks(12) {
            println!("{}", chunk.join(" "));
        }
    } else if gen_mode {
        println!("  VSA autoregressive PhaseNode generation...\n");
        let snippets = mirror.generate_code_autoregressive(text, 20, temperature);
        if snippets.is_empty() {
            eprintln!("  Generation failed (no snippets produced)");
            return;
        }
        for (si, snippet) in snippets.iter().enumerate() {
            println!("  // Step {}:\n{}\n", si + 1, snippet);
        }
        let indexed = mirror.index_generated_snippets(&snippets);
        if indexed > 0 {
            mirror.save();
            println!("  Self-indexed {} new phase nodes", indexed);
        }
    } else {
        println!("  Generated code:\n");
        for (si, (path, kind, name, line, overlap)) in preds.iter().enumerate() {
            let code = mirror.source_snippet_for_path(path, *line, 5);
            let snippet_str = if code.is_empty() {
                "// <source not available>".to_string()
            } else {
                code
            };
            println!(
                "  // Step {} — matched {}::{} ({})\n{}\n",
                si + 1,
                kind,
                name,
                overlap,
                snippet_str
            );
        }
    }
    println!(
        "  Mirror: {} phase nodes, TM: {} cells, {} segments",
        node_count, total_cells, total_seg
    );
}

pub fn run_evaluate() {
    let mut mirror = match fuga::SelfMirror::load() {
        Some(m) => m,
        None => {
            eprintln!("No mirror data. Run 'fuga mirror-index' first.");
            return;
        }
    };
    println!("═══ Fuga Evaluate ═══\n");
    let result = mirror.evaluate();
    println!("  {}", result);
    mirror.save();
    println!("  Mirror saved.");
}

pub fn run_reinit_jepa() {
    let mut mirror = match fuga::SelfMirror::load() {
        Some(m) => m,
        None => {
            println!("  No mirror data — creating fresh TM + HJEPA");
            let tm = fuga::TemporalMemory::new(2000, 4);
            let hjepa = fuga::HierarchicalJEPA::new(8192);
            fuga::SelfMirror::new(tm, hjepa)
        }
    };
    println!("═══ Reinit HJEPA ═══\n");
    let dim = mirror.predictor.hjepa.dim;
    mirror.predictor.hjepa = fuga::HierarchicalJEPA::new(dim);
    println!(
        "  Created fresh HJEPA (dim={}) with PERM_EXPANSION={}",
        dim,
        mirror.predictor.hjepa.levels[0].perm_offsets.len()
            / mirror.predictor.hjepa.levels[0].context_len
    );
    mirror.save();
    println!("  Mirror saved.");
}

pub fn run_set_mode(mode: &str) {
    let mut mirror = match fuga::SelfMirror::load() {
        Some(m) => m,
        None => {
            eprintln!("No mirror data.");
            return;
        }
    };
    let parse_code = |s: &str| -> Option<u8> {
        match s {
            "linear" | "lin" | "0" => Some(0u8),
            "bundled" | "bundle" | "bund" | "1" => Some(1u8),
            "phase" | "2" => Some(2u8),
            _ => None,
        }
    };
    // Per-level syntax: "l0:phase,l1:linear,l2:bundled" or "phase" (all)
    let parts: Vec<&str> = mode.split(',').collect();
    let mut applied = false;
    if parts.len() >= 3 && parts.iter().all(|p| p.contains(':')) {
        for p in parts {
            let kv: Vec<&str> = p.split(':').collect();
            if kv.len() != 2 {
                continue;
            }
            let level = match kv[0].trim().to_lowercase().as_str() {
                "l0" | "0" => 0usize,
                "l1" | "1" => 1usize,
                "l2" | "2" => 2usize,
                _ => continue,
            };
            let code = match parse_code(kv[1].trim()) {
                Some(c) => c,
                None => {
                    eprintln!("  Bad mode '{}' (linear|bundled|phase)", kv[1]);
                    continue;
                }
            };
            if let Some(lvl) = mirror.predictor.hjepa.levels.get_mut(level) {
                lvl.mode = code;
                applied = true;
            }
        }
    } else if parts.len() == 3 && parts.iter().all(|p| parse_code(p).is_some()) {
        // positional syntax: set-mode phase,linear,phase  (L0,L1,L2)
        for (i, p) in parts.iter().enumerate() {
            if let (Some(code), Some(lvl)) =
                (parse_code(p), mirror.predictor.hjepa.levels.get_mut(i))
            {
                lvl.mode = code;
                applied = true;
            }
        }
    } else if let Some(code) = parse_code(mode) {
        for lvl in &mut mirror.predictor.hjepa.levels {
            lvl.mode = code;
        }
        applied = true;
    }
    if !applied {
        eprintln!(
            "  Usage: set-mode linear|bundled|phase | <l0>:<m>,<l1>:<m>,<l2>:<m> | <m>,<m>,<m>"
        );
        return;
    }
    let codes: Vec<u8> = mirror
        .predictor
        .hjepa
        .levels
        .iter()
        .map(|l| l.mode)
        .collect();
    println!("  Set HJEPA modes to {:?} (L0,L1,L2)", codes);
    mirror.save();
    println!("  Mirror saved.");
}

pub fn run_set_topk(n: usize) {
    let mut mirror = match fuga::SelfMirror::load() {
        Some(m) => m,
        None => {
            eprintln!("No mirror data.");
            return;
        }
    };
    for lvl in &mut mirror.predictor.hjepa.levels {
        lvl.top_k = n;
    }
    println!(
        "  Set HJEPA sparse phase router top_k = {} (0 = dense, all projections)",
        n
    );
    mirror.save();
    println!("  Mirror saved.");
}

pub fn run_set_router(topk: usize, num_expert_group: usize, topk_group: usize) {
    let mut mirror = match fuga::SelfMirror::load() {
        Some(m) => m,
        None => {
            eprintln!("No mirror data.");
            return;
        }
    };
    for lvl in &mut mirror.predictor.hjepa.levels {
        lvl.top_k = topk;
        lvl.num_expert_group = num_expert_group;
        lvl.topk_group = topk_group;
    }
    println!(
        "  Set grouped phase router: top_k={} num_expert_group={} topk_group={}",
        topk, num_expert_group, topk_group
    );
    mirror.save();
    println!("  Mirror saved.");
}

pub fn print_usage(program: &str) {
    println!("Fuga 1.0 — Polyphonic Code Analysis via VSA Geometry");
    println!();
    println!("Usage: {} <command> [options]", program);
    println!();
    println!("Commands:");
    println!("  analyze <path>        Analyze file, directory, or workspace");
    println!("  check <path>          Alias for analyze");
    println!("  scan <path>           Security AST audit (eval, XSS, injection patterns)");
    println!("  ui <prompt>           Generate UI from expert_code memory patterns");
    println!("  agent <task>          Autonomous task executor using Fuga memory");
    println!("  fix <file>            Generate autofix patch (all supported languages)");
    println!("  translate <file>      Translate code between languages");
    println!("  weave [path] [--dim N] [--window N]  Compress tokens with VSA Weaver Engine");
    println!("  unweave [path]        Unweave SuperTokens back to token stream");
    println!("  tokenize [--count N]  Explore and synthesize new VSA tokens");
    println!("  think [text]          Run Fuga AI core: attend → route → absorb");
    println!("  train [corpus.jsonl]   Train AI on corpus: tokenize → weave → absorb → cube");
    println!("  query <text>           Resonance search over trained cube");
    println!("                         (loads ./fuga_cube.bin, ./tikones/ configs)");
    println!("  ask <question>         Answer question from trained memory");
    println!("                         (--explain/--answer for structured response with snippets)");
    println!("                         (--prompt SAFETY,EFFICIENT,... to modulate VSA search)");
    println!("  solve <problem>        Multi-step problem solving with decomposition");
    println!("  codegen <seed>          Generate novel text from cube knowledge");
    println!("                           --cube <path> (default: fuga_cube.bin)");
    println!("                           --max-tokens <N> (default: 100)");
    println!("                           --temperature <T> (default: 0.6)");
    println!("  code-quality <path>    Analyze code quality (safety, UB, violations)");
    println!("  scan <path>             Security AST audit (eval, XSS, injection)");
    println!("  ui <prompt> [-o file]   Generate UI patterns from expert_code");
    println!("  generate <prompt> [-o file]  Synthesize HTML page from memory patterns");
    println!("  agent <task>            Autonomous task using Fuga memory + zx");
    println!(
        "  stream-train <dir...>    Train code repos without loading full memory (streaming append)"
    );
    println!("  absorb-agent            Absorb agent results (success/failure) into MoE");
    println!(
        "  refactor <file> <desc> [max_iter]  Self-refactoring loop: generate → check → fix → absorb"
    );
    println!("  jepa-train <dir> [dim] [ctx] [epochs]  Train JEPA predictor on corpus");
    println!("  jepa-predict <text> [dim] [ctx]  Predict next state via JEPA");
    println!(
        "  h-jepa-train <dir> [dim] [epochs]  Train hierarchical JEPA (3-level) on code repos"
    );
    println!("  h-jepa-predict <text> [dim]       Predict at all 3 hierarchical levels");
    println!(
        "  sdr-build [path] [max]  Build SDR index from .mem.bin (path: fuga_code_cube_mem.bin)"
    );
    println!("  sdr-query <text>        SDR popcount search (Fuga 1.3)");
    println!("  sdr-query-cross <text>  Cross-SDR bridge doc→code (Fuga 1.4)");
    println!("  baby                    Interactive H-JEPA REPL (embryo)");
    println!("  prompts               List available VSA prompt modes (SAFETY, EFFICIENT, ...)");
    println!(
        "  train | train-code <dir>  Train on source code (new cube: --side N --ndim N --dim N)"
    );
    println!("  train-text <dir>          Train text corpus into existing cube");
    println!("  moe-add <domain>          Create new MoE domain");
    println!("  moe-list                  List all MoE domains");
    println!(
        "  docs [--cube path] [--output path]  Generate self-documentation from trained memory"
    );
    println!("  merge <cube> <mem1,mem2,...>  Transfer memory entries into cube");
    println!("  room [dim] [steps]     Room phase lock (headless)");
    println!("  room-view               Room 3D visualizer with LIDAR");
    println!("  reactor [steps]         Reactor point kinetics (headless)");
    println!("  reactor-view            Reactor core 3D viewer");
    println!("  fisig [corpus] [dim]   Train Fuga Fisig physics model");
    println!("  fisig-query <text>      Query the Fuga Fisig model");
    println!("  sim                     Run physics simulation (valve/heater/boiler stages)");
    println!("  rebuild-moe [--save path]  Rebuild MoE domains from memory file");
    println!("  version               Show version");
    println!("  help                  Show this help");
    println!();
    println!("Options:");
    println!("  --dim, -d <N>         Hypervector dimension (default: 8192)");
    println!("  --side <N>            Cube side length for new cubes (default: 8)");
    println!("  --ndim <N>            Cube dimension count for new cubes (default: 3)");
    println!("  --epochs, -e <N>      Training epochs (default: 1, use 30+ for deep absorption)");
    println!("  --recursive, -r       Scan directory recursively");
    println!("  --workspace, -w       Scan Cargo workspace");
    println!("  --format, -f <fmt>    Output format: text|json|html|markdown (default: text)");
    println!("  --output, -o <file>   Write report to file");
    println!();
    println!("Supported languages:");
    println!("  Rust (.rs), C (.c, .h), C++ (.cpp, .cc, .hpp), Go (.go),");
    println!("  Python (.py), TypeScript (.ts, .tsx), JavaScript (.js, .jsx)");
    println!();
    println!("Exit codes:");
    println!("  0  Clean (no issues)");
    println!("  1  Warnings (violations found)");
    println!("  2  Bugs detected");
    println!("  3  Errors (parse/IO failures)");
    println!();
    println!("Examples:");
    println!("  {} analyze src/main.rs", program);
    println!(
        "  {} analyze src/ --recursive --format json -o report.json",
        program
    );
    println!(
        "  {} analyze . --workspace --format html -o report.html",
        program
    );
    println!("  {} fix src/main.rs --output fix.patch", program);
    println!("  {} analyze app.py", program);
}


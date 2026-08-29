mod cli;

use fuga::core::wave_cube::peek_cube_header;
use fuga::weaver::token_id;
use fuga::{
    AnalysisResult, CodeQualityFilter, CodeTranslator, CorpusDoc, FileAnalysisResult, FixProposal,
    FugaAI, FugaEngine, HtmlReporter, JsonReporter, LanguageId, MarkdownReporter, MultiEngine,
    MultiFixGenerator, OutputFormat, PatchGenerator, Reporter, SDR_DIM, ScanMode, TokenBuilder,
    TokenExplorer, TokenInfo, TokenVocabulary, WaveCube, WeaverEngine, WorkspaceScanner,
    WorkspaceStats, summarize_quality,
};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::Path;
use std::process;

use cli::agent::{
    run_absorb_agent, run_agent, run_agent_loop, run_generate, run_merge, run_moe_split,
    run_rebuild_moe, run_stream_train, run_train_autofix, run_train_code, run_train_text,
    human_size,
};
use cli::analyze::{run_analyze, run_fix, run_translate};
use cli::args::{
    has_flag, parse_dim, parse_flag_value, parse_flag_values, parse_float, parse_format, parse_int,
    parse_output, parse_translate_target, parse_window,
};
use cli::crystal::{
    run_crystal_build, run_crystal_forget, run_crystal_hippo, run_crystal_learn,
    run_crystal_learn_dir, run_crystal_popcount, run_crystal_query, run_crystal_reencode,
    run_crystal_stats, run_crystal_test, run_crystal_triad, run_decode, run_docs_entry,
    run_hierarchical_jepa_predict, run_htm_predict, run_phase_codegen, run_phase_trajectory,
    run_refactor, run_reflect_repl, run_self_query, run_transpile, run_cross_domain,
    run_crystal_2_init, run_crystal_2_learn,
};
use cli::inspect::{
    print_usage, run_auto_correct, run_evaluate, run_generate_code, run_inspect_dir,
    run_inspect_file, run_inspect_text, run_mirror_index, run_reinit_jepa, run_self_mirror,
    run_set_mode, run_set_router, run_set_topk, run_tm_jepa_repl, run_train_predictor,
};
use cli::jepa::{
    load_sdr_store, run_baby_repl, run_hierarchical_jepa_train, run_htm_feed, run_htm_train,
    run_jepa_predict, run_jepa_train, run_sdr_query, run_sdr_query_cross, run_train_tm,
};
use cli::query::{
    run_ask_entry, run_hjepa_gen_entry, run_jepa_train_entry, run_readout, run_readout_entry,
    run_think, run_tm_gen_entry,
};
use cli::sim::{
    run_perceive, run_reactor, run_reactor_view_3d, run_room_phase_lock, run_room_view_3d,
    run_sim, run_view_3d,
};
use cli::tools::{
    run_codegen_entry, run_code_quality, run_query_entry, run_scan, run_solve_entry,
    run_tokenize, run_ui, run_unweave, run_weave, save_agent_result,
};
use cli::print::{capitalize, is_name_token, print_cpp_code, print_rust_code};
use cli::tm_gen::{lex_rust_code, load_tm, load_tm_from, run_tm_gen};
use cli::train::{
    run_fisig_query_entry, run_fisig_train, run_omni, run_omni_train, run_train_stack,
    run_train_unified,
};

fn main() {
    fuga::gpu::init_gpu();

    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        process::exit(1);
    }

    let command = &args[1];

    match command.as_str() {
        "analyze" | "check" => {
            if args.len() < 3 {
                eprintln!("Error: missing path");
                print_usage(&args[0]);
                process::exit(1);
            }
            let path = &args[2];
            let dim = parse_dim(&args, 3).unwrap_or(8192);
            let recursive = has_flag(&args, "--recursive") || has_flag(&args, "-r");
            let workspace = has_flag(&args, "--workspace") || has_flag(&args, "-w");
            let format = parse_format(&args, 3).unwrap_or(OutputFormat::Text);
            let output = parse_output(&args, 3);

            let mode = if workspace {
                ScanMode::Workspace
            } else if recursive {
                ScanMode::Recursive
            } else {
                ScanMode::SingleFile
            };

            run_analyze(path, dim, mode, format, output.as_deref());
        }
        "fix" => {
            if args.len() < 3 {
                eprintln!("Error: missing file path");
                print_usage(&args[0]);
                process::exit(1);
            }
            let path = &args[2];
            let dim = parse_dim(&args, 3).unwrap_or(8192);
            let output = parse_output(&args, 3);
            run_fix(path, dim, output.as_deref());
        }
        "translate" => {
            if args.len() < 3 {
                eprintln!("Error: missing file path");
                print_usage(&args[0]);
                process::exit(1);
            }
            let path = &args[2];
            let target = parse_translate_target(&args, 3).unwrap_or("rust");
            run_translate(path, target);
        }
        "weave" => {
            run_weave(&args);
        }
        "unweave" => {
            run_unweave(&args);
        }
        "tokenize" => {
            run_tokenize(&args);
        }
        "think" => {
            run_think(&args);
        }
        "query" => {
            run_query_entry(&args);
        }
        "ask" => {
            run_ask_entry(&args);
        }
        "readout" => {
            run_readout_entry(&args);
        }
        "tm-gen" => {
            run_tm_gen_entry(&args);
        }
        "w-gen" => {
            // Latent W-operator generation (predict_latent), the OLD generative
            // path that `tm-gen` no longer reaches. Seeded probe for testing
            // whether an accepted lesson actually shifted the W operator.
            let text = args.get(2).cloned().unwrap_or_default();
            if text.is_empty() {
                eprintln!("Usage: fuga w-gen <prompt> [--file TM] [--vocab-dir DIR] [--steps N] [--structure]");
                return;
            }
            run_tm_gen(&text, &args);
        }
        "hjepa" => {
            run_hjepa_gen_entry(&args);
        }
        "jepa-train" => {
            run_jepa_train_entry(&args);
        }
        "solve" => {
            run_solve_entry(&args);
        }
        "code-quality" | "quality" => {
            if args.len() < 3 {
                eprintln!("Error: missing path");
                print_usage(&args[0]);
                process::exit(1);
            }
            let path = &args[2];
            let dim = parse_dim(&args, 3).unwrap_or(8192);
            let recursive = has_flag(&args, "--recursive") || has_flag(&args, "-r");
            run_code_quality(path, dim, recursive);
        }
        "scan" => {
            if args.len() < 3 {
                eprintln!("Error: missing path");
                print_usage(&args[0]);
                process::exit(1);
            }
            let path = &args[2];
            let dim = parse_dim(&args, 3).unwrap_or(8192);
            let output = parse_output(&args, 3);
            run_scan(path, dim, output.as_deref());
        }
        "ui" => {
            if args.len() < 3 {
                eprintln!("Error: missing prompt");
                print_usage(&args[0]);
                process::exit(1);
            }
            let prompt = args[2..].join(" ");
            let dim = parse_dim(&args, 3).unwrap_or(8192);
            let output = parse_output(&args, 3);
            run_ui(&prompt, dim, output.as_deref());
        }
        "generate" => {
            if args.len() < 3 {
                eprintln!("Error: missing prompt");
                print_usage(&args[0]);
                process::exit(1);
            }
            let prompt = args[2..].join(" ");
            let dim = parse_dim(&args, 3).unwrap_or(8192);
            let output = parse_output(&args, 3);
            run_generate(&prompt, dim, output.as_deref());
        }
        "agent" => {
            if args.len() < 3 {
                eprintln!("Error: missing task");
                print_usage(&args[0]);
                process::exit(1);
            }
            let task = args[2..]
                .iter()
                .filter(|a| !a.starts_with("--"))
                .cloned()
                .collect::<Vec<_>>()
                .join(" ");
            let force = args.iter().any(|a| a == "--force");
            let agent_prompts: Vec<String> = parse_flag_values(&args, "--prompt")
                .into_iter()
                .flat_map(|v| {
                    v.split(',')
                        .map(|s| s.trim().to_uppercase())
                        .collect::<Vec<_>>()
                })
                .collect();
            if !agent_prompts.is_empty() {
                println!("  Active prompts: {:?}", agent_prompts);
            }
            let dim = parse_dim(&args, 3).unwrap_or(8192);
            if force {
                run_agent(&task, dim, true);
            } else {
                run_agent(&task, dim, false);
            }
        }
        "agent-loop" => {
            // Полный агентский стек + саморекурсивное обучение через necli
            // (Fuga-мозг через собственный OpenAI-совместимый маск, не внешний).
            run_agent_loop(&args);
        }
        "sim" => {
            run_sim(&args);
        }
        "room" => {
            let dim = parse_dim(&args, 2).unwrap_or(8192);
            let steps = args
                .get(3)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(500);
            run_room_phase_lock(dim, steps);
        }
        "room-view" => {
            run_room_view_3d();
        }
        "reactor" => {
            let steps = args
                .get(2)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(500);
            run_reactor(steps);
        }
        "reactor-view" => {
            run_reactor_view_3d();
        }
        "fisig" | "fisig-train" => {
            run_fisig_train(&args);
        }
        "fisig-query" => {
            run_fisig_query_entry(&args);
        }
        "omni-train" => {
            run_omni_train(&args);
        }
        "train-unified" => {
            run_train_unified(&args);
        }
        "train-stack" => {
            run_train_stack(&args);
        }
        "omni" => {
            run_omni(&args);
        }
        "microwave" => {
            fuga::microwave::Microwave::run(&args);
        }
        "codegen" => {
            run_codegen_entry(&args);
        }
        "docs" => {
            run_docs_entry(&args);
        }
        "view" => {
            run_view_3d();
        }
        "perceive" => {
            let dim = parse_dim(&args, 2).unwrap_or(8192);
            let steps = args
                .get(3)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(500);
            run_perceive(dim, steps);
        }
        "merge" => {
            if args.len() < 3 {
                eprintln!("Error: missing cube path");
                print_usage(&args[0]);
                process::exit(1);
            }
            let target_path = &args[2];
            if std::path::Path::new(target_path).exists() {
                let (ndim, side_len, _) = match peek_cube_header(target_path) {
                    Ok(h) => h,
                    Err(e) => {
                        eprintln!("{}", e);
                        return;
                    }
                };
                match (ndim, side_len) {
                    (3, 4) => run_merge::<3, 4>(&args),
                    (4, 4) => run_merge::<4, 4>(&args),
                    (3, 8) => run_merge::<3, 8>(&args),
                    (4, 8) => run_merge::<4, 8>(&args),
                    (5, 2) => run_merge::<5, 2>(&args),
                    (5, 4) => run_merge::<5, 4>(&args),
                    _ => eprintln!("Unsupported cube dims: {}x{}", side_len, ndim),
                }
            } else {
                eprintln!("Cube file not found: {}", target_path);
                print_usage(&args[0]);
                process::exit(1);
            }
        }
        "train" | "train-code" => {
            let dir = args.get(2).map(|s| s.as_str()).unwrap_or("src");
            let dim = parse_dim(&args, 3).unwrap_or(100000);
            let save_path = parse_flag_value(&args, 3, "--save").unwrap_or("fuga_code_cube.bin");
            let epochs = args
                .iter()
                .position(|a| a == "--epochs")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(1);
            let side = args
                .iter()
                .position(|a| a == "--side")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(8);
            let ndim = args
                .iter()
                .position(|a| a == "--ndim")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(3);

            let cube_spec = if std::path::Path::new(save_path).exists() {
                match peek_cube_header(save_path) {
                    Ok(h) => h,
                    Err(e) => {
                        eprintln!("{}", e);
                        return;
                    }
                }
            } else {
                (ndim, side, dim)
            };
            match cube_spec {
                (3, 4, _) => run_train_code::<3, 4>(dir, dim, save_path, epochs, &args),
                (4, 4, _) => run_train_code::<4, 4>(dir, dim, save_path, epochs, &args),
                (3, 8, _) => run_train_code::<3, 8>(dir, dim, save_path, epochs, &args),
                (4, 8, _) => run_train_code::<4, 8>(dir, dim, save_path, epochs, &args),
                (5, 2, _) => run_train_code::<5, 2>(dir, dim, save_path, epochs, &args),
                (5, 4, _) => run_train_code::<5, 4>(dir, dim, save_path, epochs, &args),
                (3, 5, _) => run_train_code::<3, 5>(dir, dim, save_path, epochs, &args),
                (3, 6, _) => run_train_code::<3, 6>(dir, dim, save_path, epochs, &args),
                (3, 7, _) => run_train_code::<3, 7>(dir, dim, save_path, epochs, &args),
                _ => eprintln!("Unsupported cube dims: {}x{}", cube_spec.0, cube_spec.1),
            }
        }
        "train-text" => {
            if args.len() < 3 {
                eprintln!("Error: missing text corpus directory");
                print_usage(&args[0]);
                process::exit(1);
            }
            let dir = &args[2];
            let dim = parse_dim(&args, 3).unwrap_or(100000);
            let save_path = parse_flag_value(&args, 3, "--save").unwrap_or("fuga_code_cube.bin");

            if std::path::Path::new(save_path).exists() {
                let (ndim, side_len, _) = match peek_cube_header(save_path) {
                    Ok(h) => h,
                    Err(e) => {
                        eprintln!("{}", e);
                        return;
                    }
                };
                match (ndim, side_len) {
                    (3, 4) => run_train_text::<3, 4>(dir, dim, save_path, &args),
                    (4, 4) => run_train_text::<4, 4>(dir, dim, save_path, &args),
                    (3, 8) => run_train_text::<3, 8>(dir, dim, save_path, &args),
                    (4, 8) => run_train_text::<4, 8>(dir, dim, save_path, &args),
                    (5, 2) => run_train_text::<5, 2>(dir, dim, save_path, &args),
                    (5, 4) => run_train_text::<5, 4>(dir, dim, save_path, &args),
                    (3, 5) => run_train_text::<3, 5>(dir, dim, save_path, &args),
                    (3, 6) => run_train_text::<3, 6>(dir, dim, save_path, &args),
                    (3, 7) => run_train_text::<3, 7>(dir, dim, save_path, &args),
                    _ => eprintln!("Unsupported cube dims: {}x{}", side_len, ndim),
                }
            } else {
                eprintln!(
                    "Error: existing cube required for training. Run 'fuga train-code' first."
                );
                print_usage(&args[0]);
                process::exit(1);
            }
        }
        "train-autofix" => {
            if args.len() < 3 {
                eprintln!("Error: missing source directory");
                print_usage(&args[0]);
                process::exit(1);
            }
            let dir = &args[2];
            let dim = parse_dim(&args, 3).unwrap_or(100000);
            let save_path = parse_flag_value(&args, 3, "--save").unwrap_or("fuga_code_cube.bin");
            let mw_path = parse_flag_value(&args, 3, "--microwave")
                .unwrap_or("microwave_sandbox/target/release/mini-fuga");

            if std::path::Path::new(save_path).exists() {
                let (ndim, side_len, _) = match peek_cube_header(save_path) {
                    Ok(h) => h,
                    Err(e) => {
                        eprintln!("{}", e);
                        return;
                    }
                };
                match (ndim, side_len) {
                    (3, 4) => run_train_autofix::<3, 4>(dir, dim, save_path, mw_path, &args),
                    (4, 4) => run_train_autofix::<4, 4>(dir, dim, save_path, mw_path, &args),
                    (3, 8) => run_train_autofix::<3, 8>(dir, dim, save_path, mw_path, &args),
                    (4, 8) => run_train_autofix::<4, 8>(dir, dim, save_path, mw_path, &args),
                    (5, 2) => run_train_autofix::<5, 2>(dir, dim, save_path, mw_path, &args),
                    (5, 4) => run_train_autofix::<5, 4>(dir, dim, save_path, mw_path, &args),
                    (3, 5) => run_train_autofix::<3, 5>(dir, dim, save_path, mw_path, &args),
                    (3, 6) => run_train_autofix::<3, 6>(dir, dim, save_path, mw_path, &args),
                    (3, 7) => run_train_autofix::<3, 7>(dir, dim, save_path, mw_path, &args),
                    _ => eprintln!("Unsupported cube dims: {}x{}", side_len, ndim),
                }
            } else {
                eprintln!(
                    "Error: cube {} not found. Train first with 'train-code'",
                    save_path
                );
            }
        }
        "moe-split" => {
            let save_path = parse_flag_value(&args, 2, "--save").unwrap_or("fuga_code_cube.bin");
            if !std::path::Path::new(save_path).exists() {
                eprintln!("Cube not found: {}", save_path);
                process::exit(1);
            }
            let (ndim, side_len, _) = match peek_cube_header(save_path) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("{}", e);
                    return;
                }
            };
            match (ndim, side_len) {
                (3, 4) => run_moe_split::<3, 4>(save_path),
                (4, 4) => run_moe_split::<4, 4>(save_path),
                (3, 8) => run_moe_split::<3, 8>(save_path),
                (4, 8) => run_moe_split::<4, 8>(save_path),
                (5, 2) => run_moe_split::<5, 2>(save_path),
                (5, 4) => run_moe_split::<5, 4>(save_path),
                _ => eprintln!("Unsupported cube dims: {}x{}", side_len, ndim),
            }
        }
        "version" | "--version" | "-v" => {
            println!("fuga 0.1.0");
        }
        "absorb-agent" => {
            run_absorb_agent();
        }
        "stream-train" => {
            let dirs: Vec<&str> = if args.len() > 2 {
                args[2..]
                    .iter()
                    .filter(|a| !a.starts_with("--"))
                    .map(|s| s.as_str())
                    .collect()
            } else {
                vec!["temp_repos"]
            };
            let save_path = parse_flag_value(&args, 2, "--save").unwrap_or("fuga_code_cube.bin");
            let batch_size = args
                .iter()
                .position(|a| a == "--batch-size")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(100);
            if std::path::Path::new(save_path).exists() {
                if let Ok((ndim, side, _)) = peek_cube_header(save_path) {
                    match (ndim, side) {
                        (3, 4) => run_stream_train::<3, 4>(&dirs, save_path, batch_size),
                        (4, 4) => run_stream_train::<4, 4>(&dirs, save_path, batch_size),
                        _ => eprintln!("Unsupported cube dims: {}x{}", side, ndim),
                    }
                } else {
                    eprintln!("Invalid cube: {}", save_path);
                }
            } else {
                eprintln!(
                    "Cube not found: {}. Train with 'train-code' first.",
                    save_path
                );
            }
        }
        "refactor" => {
            let file = args.get(2).map(|s| s.as_str()).unwrap_or("");
            let desc = args.get(3).map(|s| s.as_str()).unwrap_or("refactor");
            let max_iter = args
                .get(4)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(5);
            run_refactor(file, desc, max_iter);
        }
        "jepa-train" => {
            let dir = args.get(2).map(|s| s.as_str()).unwrap_or(".");
            let dim = parse_dim(&args, 3).unwrap_or(8192);
            let ctx = args
                .get(4)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(4);
            let epochs = args
                .get(5)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(100);
            run_jepa_train(dir, dim, ctx, epochs);
        }
        "jepa-predict" => {
            let text = args.get(2).map(|s| s.as_str()).unwrap_or("");
            let dim = parse_dim(&args, 3).unwrap_or(8192);
            let ctx = args
                .get(4)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(4);
            run_jepa_predict(text, dim, ctx);
        }
        "h-jepa-train" => {
            let dir = args.get(2).map(|s| s.as_str()).unwrap_or("temp_repos");
            let dim = parse_dim(&args, 3).unwrap_or(8192);
            let epochs = args
                .get(4)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(100);
            run_hierarchical_jepa_train(dir, dim, epochs);
        }
        "h-jepa-predict" => {
            let text = args.get(2).map(|s| s.as_str()).unwrap_or("");
            let dim = parse_dim(&args, 3).unwrap_or(8192);
            run_hierarchical_jepa_predict(text, dim);
        }
        "cross-domain" => {
            let dim = parse_dim(&args, 2).unwrap_or(8192);
            let epochs = args
                .get(3)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(50);
            run_cross_domain(dim, epochs);
        }
        "sdr-build" => {
            let path = args
                .get(2)
                .map(|s| s.as_str())
                .unwrap_or("fuga_code_cube_mem.bin");
            let max_entries = args
                .get(3)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(100000);
            println!(
                "Building SDR index from {} (max {} entries)",
                path, max_entries
            );
            match fuga::SdrStore::build_from_mem(path, max_entries) {
                Ok(store) => {
                    let sdr_path = "fuga_sdr_index.bin";
                    let mut f = std::fs::File::create(sdr_path).unwrap();
                    use std::io::Write;
                    let count = store.index.nodes.len() as u32;
                    f.write_all(&count.to_le_bytes()).ok();
                    for node in &store.index.nodes {
                        for w in &node.bits {
                            f.write_all(&w.to_le_bytes()).ok();
                        }
                    }
                    let tcount = store.index.texts.len() as u32;
                    f.write_all(&tcount.to_le_bytes()).ok();
                    for t in &store.index.texts {
                        let tb = t.as_bytes();
                        f.write_all(&(tb.len() as u32).to_le_bytes()).ok();
                        f.write_all(tb).ok();
                    }
                    println!(
                        "  Saved {} SDR nodes + {} texts to {}",
                        count, tcount, sdr_path
                    );
                }
                Err(e) => eprintln!("  SDR build failed: {}", e),
            }
        }
        "sdr-query" => {
            let text = args[2..].join(" ");
            if text.is_empty() {
                eprintln!("Usage: fuga sdr-query <text>");
                return;
            }
            run_sdr_query(&text);
        }
        "sdr-query-cross" => {
            let text = args[2..].join(" ");
            if text.is_empty() {
                eprintln!("Usage: fuga sdr-query-cross <text>");
                return;
            }
            run_sdr_query_cross(&text);
        }
        "htm-train" => {
            let path = args
                .get(2)
                .map(|s| s.as_str())
                .unwrap_or("fuga_code_cube_code_mem.bin");
            let steps = args
                .get(3)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(1000);
            run_htm_train(path, steps);
        }
        "htm-predict" => {
            let text = args[2..].join(" ");
            if text.is_empty() {
                eprintln!("Usage: fuga htm-predict <text>");
                return;
            }
            run_htm_predict(&text);
        }
        "htm-feed" => {
            let text = args[2..].join(" ");
            if text.is_empty() {
                eprintln!("Usage: fuga htm-feed <text>");
                return;
            }
            run_htm_feed(&text);
        }
        "train-tm" => {
            let dir = args.get(2).map(|s| s.as_str()).unwrap_or("corpus_src");
            let cap = parse_int(&args, "--cap").unwrap_or(8192);
            let ctx = parse_int(&args, "--ctx").unwrap_or(4);
            let max_files = parse_int(&args, "--max-files").unwrap_or(usize::MAX);
            let out = parse_flag_value(&args, 2, "--out")
                .map(|s| s.to_string())
                .unwrap_or_else(|| "fuga_htm.bin".to_string());
            let structure = args.iter().any(|a| a == "--structure");
            run_train_tm(dir, cap, ctx, max_files, &out, structure);
        }
        "tm-gen" => {
            let text = args.get(2).cloned().unwrap_or_default();
            if text.is_empty() {
                eprintln!("Usage: fuga tm-gen <prompt> [--steps N] [--file FILE]");
                return;
            }
            run_tm_gen(&text, &args);
        }
        "mirror-index" => {
            let dir = if args.len() > 2 {
                args[2].as_str()
            } else {
                "src/ai"
            };
            run_mirror_index(dir);
        }
        "inspect-dir" | "id" => {
            let dir = if args.len() > 2 {
                args[2].as_str()
            } else {
                "."
            };
            run_inspect_dir(dir);
        }
        "auto-correct" | "ac" => {
            let path = if args.len() > 2 { args[2].as_str() } else { "" };
            if path.is_empty() {
                eprintln!("Usage: fuga auto-correct <file>");
                return;
            }
            run_auto_correct(path);
        }
        "generate-code" | "gen-code" => {
            let beam = parse_int(&args, "--beam").unwrap_or(1);
            let temp = parse_float(&args, "--temp").unwrap_or(1.0);
            let gen_mode = args.iter().any(|a| a == "--gen");
            let token_mode = args.iter().any(|a| a == "--tokens");
            let mut text_parts: Vec<String> = Vec::new();
            let mut skip_next = false;
            for s in args[2..].iter() {
                if skip_next {
                    skip_next = false;
                    continue;
                }
                if s == "--beam" || s == "--temp" || s == "--model" {
                    skip_next = true;
                    continue;
                }
                if s == "--gen" || s == "--tokens" || s.starts_with("--") {
                    continue;
                }
                text_parts.push(s.clone());
            }
            let text = text_parts.join(" ");
            if text.is_empty() {
                eprintln!(
                    "Usage: fuga generate-code <text> [--beam N] [--temp T] [--gen] [--tokens]"
                );
                return;
            }
            run_generate_code(&text, beam, temp, gen_mode, token_mode);
        }
        "train-predictor" | "tp" => {
            let epochs = if args.len() > 2 {
                args[2].parse().unwrap_or(5)
            } else {
                5
            };
            let chunk_size = parse_int(&args, "--chunk").unwrap_or(1);
            let use_ff = args.iter().any(|a| a == "--ff");
            run_train_predictor(epochs, chunk_size, use_ff);
        }
        "evaluate" | "eval" => {
            run_evaluate();
        }
        "eval-debug" => {
            let mut mirror = match fuga::SelfMirror::load() {
                Some(m) => m,
                None => {
                    eprintln!("No mirror data.");
                    return;
                }
            };
            println!("{}", mirror.evaluate_debug());
        }
        "reinit-jepa" => {
            run_reinit_jepa();
        }
        "set-mode" => {
            let mode = args.get(2).map(|s| s.as_str()).unwrap_or("");
            run_set_mode(mode);
        }
        "set-topk" => {
            let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
            run_set_topk(n);
        }
        "set-router" => {
            let topk: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
            let groups: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
            let topk_groups: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
            run_set_router(topk, groups, topk_groups);
        }
        "crystal-build" | "crystalb" => {
            let out = args
                .get(2)
                .map(|s| s.as_str())
                .unwrap_or("fuga_crystal.bin");
            let max: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1000);
            run_crystal_build(out, max);
        }
        "crystal-query" | "crystalq" => {
            let raw: Vec<&String> = args[2..].iter().collect();
            let mut text_parts = Vec::new();
            let mut skip = false;
            for s in &raw {
                if skip {
                    skip = false;
                    continue;
                }
                if *s == "--from" || *s == "--scale" || *s == "--gate" {
                    skip = true;
                    continue;
                }
                if s.starts_with("--from=") || s.starts_with("--scale=") {
                    continue;
                }
                text_parts.push(s.as_str());
            }
            let text = text_parts.join(" ");
            if text.is_empty() {
                eprintln!(
                    "Usage: fuga crystal-query \"<query>\" [--from crystal.bin] [--scale 0.5] [--gate]"
                );
                return;
            }
            run_crystal_query(&text, &args);
        }
        "crystal-reencode" => {
            let path = parse_flag_value(&args, 1, "--from")
                .map(|s| s.to_string())
                .unwrap_or_else(|| "fuga_crystal.bin".to_string());
            run_crystal_reencode(&path);
        }
        "phase-trajectory" | "phase" | "trajectory" => {
            let text = args[2..].join(" ");
            if text.is_empty() {
                eprintln!("Usage: fuga phase-trajectory \"<query>\"");
                return;
            }
            run_phase_trajectory(&text);
        }
        "decode" => {
            let text = args[2..]
                .iter()
                .filter(|a| !a.starts_with("--"))
                .cloned()
                .collect::<Vec<_>>()
                .join(" ");
            if text.is_empty() {
                eprintln!("Usage: fuga decode \"<text>\" [--vocab tokenizer.json] [--k N]");
                return;
            }
            run_decode(&text, &args);
        }
        "phase-codegen" | "pgen" => {
            let text = args[2..]
                .iter()
                .filter(|a| !a.starts_with("--"))
                .cloned()
                .collect::<Vec<_>>()
                .join(" ");
            if text.is_empty() {
                eprintln!(
                    "Usage: fuga phase-codegen \"<prompt>\" [--lang rust|cpp] [--vocab tokenizer.json] [--k N]"
                );
                return;
            }
            run_phase_codegen(&text, &args);
        }
        "crystal-test" | "crystalt" => {
            run_crystal_test();
        }
        "crystal-stats" | "crystals" => {
            let path = args
                .get(2)
                .map(|s| s.as_str())
                .unwrap_or("fuga_crystal.bin");
            run_crystal_stats(path);
        }
        "crystal-learn" | "learn" => {
            let key = args.get(2).map(|s| s.as_str()).unwrap_or("");
            let text = args.get(3).map(|s| s.as_str()).unwrap_or("");
            if key.is_empty() || text.is_empty() {
                eprintln!(
                    "Usage: fuga crystal-learn \"<key>\" \"<text>\" [--alpha 0.2] [--out crystal.bin]"
                );
                return;
            }
            run_crystal_learn(key, text, &args);
        }
        "crystal-learn-dir" | "learndir" => {
            let dir = args.get(2).map(|s| s.as_str()).unwrap_or("");
            if dir.is_empty() {
                eprintln!(
                    "Usage: fuga crystal-learn-dir \"<dir>\" [--alpha 0.2] [--chunk 512] [--max-chunks 32] [--out crystal.bin]"
                );
                return;
            }
            run_crystal_learn_dir(dir, &args);
        }
        "crystal-forget" | "forget" => {
            let key = args.get(2).map(|s| s.as_str()).unwrap_or("");
            if key.is_empty() {
                eprintln!("Usage: fuga crystal-forget \"<key>\"");
                return;
            }
            run_crystal_forget(key, &args);
        }
        "crystal-popcount" | "crystalp" => {
            let text = args[2..].join(" ");
            if text.is_empty() {
                eprintln!("Usage: fuga crystal-popcount \"<query>\"");
                return;
            }
            run_crystal_popcount(&text);
        }
        "crystal-2-init" | "c2init" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or("fuga_cortex.bin");
            run_crystal_2_init(path);
        }
        "crystal-2-learn" | "c2learn" => {
            let key = args.get(2).map(|s| s.to_string());
            let text = args.get(3).map(|s| s.to_string());
            if key.is_none() || text.is_none() {
                eprintln!(
                    "Usage: fuga crystal-2-learn \"<key>\" \"<episode>\" [--from cortex.bin] [--alpha 0.2] [--out cortex.bin]"
                );
                return;
            }
            run_crystal_2_learn(&key.unwrap(), &text.unwrap(), &args);
        }
        "crystal-hippo" | "hippo" => {
            let raw: Vec<&String> = args[2..].iter().collect();
            let mut text_parts = Vec::new();
            let mut skip = false;
            for s in &raw {
                if skip {
                    skip = false;
                    continue;
                }
                if s.starts_with("--from")
                    || s.starts_with("--cortex")
                    || s.starts_with("--alpha")
                    || s.starts_with("--scale")
                    || s.starts_with("--gate")
                {
                    skip = true;
                    continue;
                }
                text_parts.push(s.as_str());
            }
            let text = text_parts.join(" ");
            if text.is_empty() {
                eprintln!(
                    "Usage: fuga crystal-hippo \"<query>\" [--from static.bin] [--cortex cortex.bin] [--alpha 0.2] [--scale 0.5] [--gate]"
                );
                return;
            }
            run_crystal_hippo(&text, &args);
        }
        "crystal-triad" | "triad" => {
            let raw: Vec<&String> = args[2..].iter().collect();
            let mut text_parts = Vec::new();
            let mut skip = false;
            for s in &raw {
                if skip {
                    skip = false;
                    continue;
                }
                if s.starts_with("--from")
                    || s.starts_with("--cortex")
                    || s.starts_with("--intent")
                    || s.starts_with("--candidate")
                {
                    skip = true;
                    continue;
                }
                text_parts.push(s.as_str());
            }
            let text = text_parts.join(" ");
            if text.is_empty() {
                eprintln!(
                    "Usage: fuga crystal-triad \"<candidate>\" [--from static.bin] [--cortex cortex.bin] [--intent \"<goal>\"]"
                );
                return;
            }
            run_crystal_triad(&text, &args);
        }
        "transpile" | "xpl" => {
            run_transpile(&args);
        }
        "export-gguf" | "gguf" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or("fuga.gguf");
            match fuga::gguf::export_gguf(path) {
                Ok(_) => {
                    let ggen = fuga::gguf::read_gguf_version(path).unwrap_or(0);
                    println!("  GGUF exported to {} (generation {})", path, ggen);
                    if ggen > 0 && ggen % 5 == 0 {
                        let tag = format!("{}", ggen);
                        let _ = fuga::gguf::snapshot(path, &tag);
                        println!("  Snapshot: fuga_{}.gguf", tag);
                    }
                }
                Err(e) => eprintln!("  GGUF export failed: {}", e),
            }
        }
        "snapshot-gguf" | "ggufs" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or("fuga.gguf");
            let tag = args.get(3).map(|s| s.as_str()).unwrap_or("snap");
            match fuga::gguf::snapshot(path, tag) {
                Ok(_) => println!("  Snapshot: fuga_{}.gguf <- {}", tag, path),
                Err(e) => eprintln!("  Snapshot failed: {}", e),
            }
        }
        "inspect-gguf" | "ggufi" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or("fuga.gguf");
            println!("═══ GGUF Inspect ═══\n");
            let _ = fuga::gguf::inspect_gguf(path);
        }
        "self-mirror" | "sm" => {
            if args.get(2).map(|s| s.as_str()) == Some("query") {
                let text = args[3..].join(" ");
                run_self_query(&text);
            } else {
                run_self_mirror();
            }
        }
        "self-query" | "sq" => {
            let text = args[2..].join(" ");
            run_self_query(&text);
        }
        "htm-jepa" | "tm-jepa" => {
            run_tm_jepa_repl();
        }
        "reflect" | "reflect-repl" => {
            run_reflect_repl();
        }
        "inspect" | "recon" => {
            let path = args.get(2).map(|s| s.as_str()).unwrap_or("");
            if path.is_empty() {
                let text = args[2..].join(" ");
                if text.is_empty() {
                    eprintln!("Usage: fuga inspect <file|text>");
                    return;
                }
                run_inspect_text(&text);
            } else if std::path::Path::new(path).is_file() {
                run_inspect_file(path);
            } else {
                run_inspect_text(path);
            }
        }
        "baby" => {
            let dim = parse_dim(&args, 2).unwrap_or(8192);
            run_baby_repl(dim);
        }
        "prompts" => {
            let dim = parse_dim(&args, 2).unwrap_or(8192);
            let pv = fuga::PromptVectors::new(dim);
            println!("Available VSA prompt modes:");
            for name in pv.all_modes() {
                if let Some(hv) = pv.get(&name) {
                    println!("  [{:12}] entropy={:.4} dim={}", name, hv.entropy(), hv.dim);
                }
            }
            println!("\nUsage: --prompt MODE1,MODE2 (e.g. --prompt SAFETY,CONCISE)");
        }
        "moe-add" => {
            let domain = args.get(2).map(|s| s.as_str()).unwrap_or("");
            if domain.is_empty() {
                eprintln!("Usage: fuga moe-add <domain>");
                return;
            }
            let mut moe = fuga::MoEStore::new("fuga_code_cube");
            match moe.add_domain(domain) {
                Ok(()) => println!(
                    "  Domain '{}' added. Train with: fuga train-{} <dir>",
                    domain, domain
                ),
                Err(e) => eprintln!("  Error: {}", e),
            }
        }
        "moe-list" => {
            let domains = fuga::MoEStore::discover_domains();
            println!("MoE domains ({} found):", domains.len());
            for d in &domains {
                let path = fuga::MoEStore::mem_path(d);
                let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                println!("  {}  ({} bytes)", d, size);
            }
        }
        "rebuild-moe" => {
            let save_path = parse_flag_value(&args, 2, "--save").unwrap_or("fuga_code_cube.bin");
            run_rebuild_moe(save_path);
        }
        "help" | "--help" | "-h" => {
            print_usage(&args[0]);
        }
        _ => {
            eprintln!("Unknown command: {}", command);
            print_usage(&args[0]);
            process::exit(1);
        }
    }
}

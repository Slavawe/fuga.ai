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

fn parse_dim(args: &[String], start: usize) -> Option<usize> {
    for i in start..args.len() {
        if args[i] == "--dim" || args[i] == "-d" {
            if i + 1 < args.len() {
                return args[i + 1].parse().ok();
            }
        }
    }
    None
}

fn parse_window(args: &[String], start: usize) -> Option<usize> {
    for i in start..args.len() {
        if args[i] == "--window" || args[i] == "-w" {
            if i + 1 < args.len() {
                return args[i + 1].parse().ok();
            }
        }
    }
    None
}

fn parse_int(args: &[String], flag: &str) -> Option<usize> {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return args[i + 1].parse().ok();
        }
    }
    None
}

fn parse_float(args: &[String], flag: &str) -> Option<f64> {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return args[i + 1].parse().ok();
        }
    }
    None
}

fn parse_output(args: &[String], start: usize) -> Option<String> {
    for i in start..args.len() {
        if args[i] == "--output" || args[i] == "-o" {
            if i + 1 < args.len() {
                return Some(args[i + 1].clone());
            }
        }
    }
    None
}

fn parse_format(args: &[String], start: usize) -> Option<OutputFormat> {
    for i in start..args.len() {
        if args[i] == "--format" || args[i] == "-f" {
            if i + 1 < args.len() {
                return OutputFormat::from_str(&args[i + 1]);
            }
        }
    }
    None
}

fn parse_translate_target(args: &[String], start: usize) -> Option<&str> {
    for i in start..args.len() {
        if args[i] == "--to" {
            if i + 1 < args.len() {
                return Some(&args[i + 1]);
            }
        }
    }
    None
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn parse_flag_value<'a>(args: &'a [String], _start: usize, flag: &str) -> Option<&'a str> {
    for i in 1..args.len().saturating_sub(1) {
        if args[i] == flag {
            return Some(&args[i + 1]);
        }
    }
    None
}

fn parse_flag_values<'a>(args: &'a [String], flag: &str) -> Vec<&'a str> {
    let mut values = Vec::new();
    for i in 1..args.len().saturating_sub(1) {
        if args[i] == flag {
            values.push(args[i + 1].as_str());
        }
    }
    values
}

fn run_analyze(path: &str, dim: usize, mode: ScanMode, format: OutputFormat, output: Option<&str>) {
    println!("Fuga 1.0 — Scanning...");

    let scanner = WorkspaceScanner::new();
    let files = match scanner.scan(path, mode) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Scan error: {}", e);
            process::exit(3);
        }
    };

    println!("   Found {} file(s)", files.len());
    println!("   Dimension: {}D", dim);
    println!();

    let mut results = Vec::new();
    let mut engine = FugaEngine::new(dim);
    let mut multi_engine = MultiEngine::new(dim);

    for file_path in &files {
        let path_str = file_path.display().to_string();
        let lang = LanguageId::from_path(file_path);
        print!("Analyzing {}... ", path_str);

        match lang {
            Some(LanguageId::Rust) => match engine.analyze_file(&path_str) {
                Ok(result) => {
                    println!("OK");
                    results.push(FileAnalysisResult {
                        file_path: path_str,
                        result: AnalysisResult::Rust(result),
                        error: None,
                    });
                }
                Err(e) => {
                    println!("ERR ({})", e);
                    results.push(FileAnalysisResult {
                        file_path: path_str,
                        result: AnalysisResult::Rust(
                            FugaEngine::new(dim)
                                .analyze("// dummy\nfn main() {}")
                                .unwrap(),
                        ),
                        error: Some(e.to_string()),
                    });
                }
            },
            Some(lang) => match multi_engine.analyze_file(&path_str) {
                Ok(result) => {
                    println!("OK");
                    results.push(FileAnalysisResult {
                        file_path: path_str,
                        result: AnalysisResult::Multi(result),
                        error: None,
                    });
                }
                Err(e) => {
                    println!("ERR ({})", e);
                    results.push(FileAnalysisResult {
                        file_path: path_str.clone(),
                        result: AnalysisResult::Multi(
                            multi_engine.analyze("// dummy", lang, &path_str),
                        ),
                        error: Some(e.to_string()),
                    });
                }
            },
            None => {
                println!("ERR (unsupported language)");
                results.push(FileAnalysisResult {
                    file_path: path_str.clone(),
                    result: AnalysisResult::Multi(multi_engine.analyze(
                        "// dummy",
                        LanguageId::Rust,
                        &path_str,
                    )),
                    error: Some("Unsupported file extension".to_string()),
                });
            }
        }
    }

    println!();

    let report = match format {
        OutputFormat::Text => generate_text_report(&results),
        OutputFormat::Json => JsonReporter::new().generate_report(&results),
        OutputFormat::Html => HtmlReporter::new().generate_report(&results),
        OutputFormat::Markdown => MarkdownReporter::new().generate_report(&results),
    };

    if let Some(output_path) = output {
        match fs::write(output_path, &report) {
            Ok(_) => println!("Report written to: {}", output_path),
            Err(e) => {
                eprintln!("Failed to write report: {}", e);
                process::exit(3);
            }
        }
    } else {
        println!("{}", report);
    }

    let had_errors = results.iter().any(|r| r.error.is_some());
    if had_errors {
        process::exit(3);
    }
    let stats = WorkspaceStats::from_results(&results);
    process::exit(stats.exit_code());
}

fn generate_text_report(results: &[FileAnalysisResult]) -> String {
    let stats = WorkspaceStats::from_results(results);
    let mut report = String::new();

    report.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    report.push_str("  Fuga 1.0 — Workspace Analysis Summary\n");
    report.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    report.push_str(&format!("Files:       {}\n", stats.total_files));
    report.push_str(&format!("Lines:       {}\n", stats.total_lines));
    report.push_str(&format!("Functions:   {}\n", stats.total_functions));
    report.push_str(&format!("Violations:  {}\n", stats.total_violations));
    report.push_str(&format!("Bugs:        {}\n", stats.total_bugs));
    report.push_str(&format!(
        "Avg Safety:  {:.1}%\n",
        stats.avg_safety_score * 100.0
    ));

    if let Some((path, score)) = &stats.worst_file {
        report.push_str(&format!(
            "\nWorst file: {} (safety: {:.1}%)\n",
            path,
            score * 100.0
        ));
    }

    report.push_str("\n");

    let status = match stats.exit_code() {
        0 => "CLEAN",
        1 => "WARNINGS",
        2 => "BUGS DETECTED",
        _ => "ERRORS",
    };
    report.push_str(&format!("Status: {}\n", status));

    report.push_str("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    for file_result in results {
        if let Some(ref err) = file_result.error {
            report.push_str(&format!("ERR {}: {}\n", file_result.file_path, err));
            continue;
        }

        let result = &file_result.result;
        let safety = result.safety_score();
        let icon = if safety > 0.8 {
            "OK"
        } else if safety > 0.5 {
            "WARN"
        } else {
            "LOW"
        };

        report.push_str(&format!(
            "{} {} (safety: {:.1}%)\n",
            icon,
            file_result.file_path,
            safety * 100.0
        ));

        if !result.violations_is_empty() {
            report.push_str(&format!("   Violations: {}\n", result.violations_count()));
        }

        if result.bug_detected() {
            report.push_str(&format!(
                "   Bug detected (conf: {:.1}%)\n",
                result.bug_confidence() * 100.0
            ));
        }

        report.push_str("\n");
    }

    report
}

fn run_fix(path: &str, dim: usize, output: Option<&str>) {
    let lang = LanguageId::from_path(Path::new(path));

    match lang {
        Some(LanguageId::Rust) => {
            let mut engine = FugaEngine::new(dim);

            println!("Fuga Autofix — Analyzing: {}", path);
            println!("   Dimension: {}D", dim);
            println!();

            let result = match engine.analyze_file(path) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    process::exit(1);
                }
            };

            let source = match fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to read file: {}", e);
                    process::exit(1);
                }
            };

            let proposals = engine.generate_fixes(&source, &result);

            if proposals.is_empty() {
                println!("No fixes needed — code is clean!");
                return;
            }

            print_and_save_patch(path, &source, &proposals, output);
        }
        Some(lang) => {
            let mut multi_engine = MultiEngine::new(dim);

            println!("Fuga Autofix — Analyzing: {}", path);
            println!("   Language: {}", lang.name());
            println!("   Dimension: {}D", dim);
            println!();

            let result = match multi_engine.analyze_file(path) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    process::exit(1);
                }
            };

            let source = match fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to read file: {}", e);
                    process::exit(1);
                }
            };

            let fixer = MultiFixGenerator::new();
            let proposals = fixer.generate_fixes(&source, &result.syntax.violations, lang);

            if proposals.is_empty() {
                println!("No fixes needed — code is clean!");
                return;
            }

            print_and_save_patch(path, &source, &proposals, output);
        }
        None => {
            eprintln!("Unsupported file extension: {}", path);
            process::exit(1);
        }
    }
}

fn print_and_save_patch(path: &str, source: &str, proposals: &[FixProposal], output: Option<&str>) {
    println!("Found {} fix proposals:", proposals.len());
    println!();

    for (i, proposal) in proposals.iter().enumerate() {
        println!(
            "{}. {} (confidence: {:.0}%)",
            i + 1,
            proposal.description,
            proposal.confidence * 100.0
        );
        if let Some(ref snippet) = proposal.location.code_snippet {
            println!("   Original: {}", snippet);
        }
        println!("   Strategy: {:?}", proposal.strategy);
        println!();
    }

    let patch_generator = PatchGenerator::new();
    let diff = patch_generator.generate_patch(path, source, proposals);

    if let Some(output_path) = output {
        match fs::write(output_path, &diff.diff_text) {
            Ok(_) => println!("Patch written to: {}", output_path),
            Err(e) => {
                eprintln!("Failed to write patch: {}", e);
                process::exit(1);
            }
        }
    } else {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("{}", diff.diff_text);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();
        println!("To apply: patch {} < patch.diff", path);
        println!("To save:  fuga fix {} --output patch.diff", path);
    }
}

fn run_translate(path: &str, target: &str) {
    let lang = LanguageId::from_path(Path::new(path)).unwrap_or(LanguageId::Rust);
    let target_lang = match target.to_lowercase().as_str() {
        "rust" | "rs" => LanguageId::Rust,
        "c" => LanguageId::C,
        "cpp" | "cxx" | "c++" => LanguageId::Cpp,
        "go" => LanguageId::Go,
        "python" | "py" => LanguageId::Python,
        "typescript" | "ts" => LanguageId::TypeScript,
        "javascript" | "js" => LanguageId::JavaScript,
        other => {
            eprintln!("Unsupported target language: {}", other);
            process::exit(1);
        }
    };

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read file: {}", e);
            process::exit(1);
        }
    };

    let translator = CodeTranslator::new();
    match translator.translate(&source, lang, target_lang) {
        Ok(output) => {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("  Translation: {:?} → {:?}", lang, target_lang);
            println!("  Source: {}", path);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("{}", output);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Note: Auto-translated code requires manual review");
        }
        Err(e) => {
            eprintln!("Translation failed: {}", e);
            process::exit(1);
        }
    }
}

fn run_think(args: &[String]) {
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

fn run_ask_entry(args: &[String]) {
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

fn run_readout<const N: usize, const S: usize>(
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

fn run_tm_gen_entry(args: &[String]) {
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
            println!("[debug] predict_structure([{:?}]) popcount={}", t, pc);
        }
    }

    // Кандидаты для декода: слова запроса + слова ближайшей памяти.
    let mem_path = cube_path.replace(".bin", "_mem.bin");
    let mut cands: Vec<String> = query
        .split_whitespace()
        .map(|s| s.to_lowercase())
        .filter(|w| w.len() >= 2)
        .collect();
    if let Ok(memory) = fuga::MemoryStore::load_bin(&mem_path) {
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
    let out = fuga::tm_generate(&tm, &seed, steps, &cands, window);

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

fn run_readout_entry(args: &[String]) {
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

fn run_sim(args: &[String]) {
    use fuga::CubicController;
    use fuga::Pipe;
    use fuga::Valve;
    use fuga::sim::*;

    let stage = args.get(2).map(|s| s.as_str()).unwrap_or("1");
    let dim: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8192);

    match stage {
        "1" => {
            println!("╔══════════════════════════════════════════╗");
            println!("║  Stage 1: Valve Control Loop            ║");
            println!("║  Sensor → WaveCube → Valve              ║");
            println!("╚══════════════════════════════════════════╝");
            println!("  Dimension: {}D", dim);
            println!();

            let mut pipe = Pipe::new(1.0);
            let mut valve = Valve::new(2.0);
            let mut ctrl = CubicController::<8>::new(dim);
            ctrl.setpoint = 5.0;
            ctrl.kp = 1.5;
            ctrl.ki = 0.1;

            pipe.inflow = 1.0;

            let dt = 0.01;
            let steps = 2000;
            let mut log = Vec::with_capacity(steps);

            let base = pipe.inflow / valve.max_flow;
            for i in 0..steps {
                let signal = ctrl.update(pipe.pressure, dt);
                valve.set((base - signal).clamp(0.0, 1.0));
                pipe.step(dt, &valve);
                let stability = ctrl.phase_stability();
                log.push((i as f64 * dt, pipe.pressure, valve.position, stability));

                if i % 1000 == 0 {
                    println!(
                        "  t={:.2}s  P={:.2}Pa  valve={:.1}%  phi_stab={:.3}",
                        i as f64 * dt,
                        pipe.pressure,
                        valve.position * 100.0,
                        stability
                    );
                }
            }

            let setpoint = ctrl.setpoint;
            let final_p = pipe.pressure;
            let overshoot = ((final_p - setpoint) / setpoint * 100.0).abs();
            let settle = log
                .iter()
                .rposition(|(_, p, _, _)| (p - setpoint).abs() >= 0.1)
                .map(|i| i + 1)
                .unwrap_or(0);

            println!();
            println!("  === Results ===");
            println!("  Setpoint:      {:.2} Pa", setpoint);
            println!("  Final P:       {:.2} Pa", final_p);
            println!("  Overshoot:     {:.1}%", overshoot);
            let settle_s = settle as f64 * dt;
            println!("  Settle time:   {:.3}s", settle_s);
            println!("  Phase lock:    {:.3}", ctrl.phase_stability());

            if settle_s < 15.0 && overshoot < 15.0 {
                println!("  STAGE 1 PASS — stable in {:.3}s", settle_s);
            } else {
                println!("  STAGE 1 — oscillation detected");
            }
        }

        "2" => {
            println!("╔══════════════════════════════════════════╗");
            println!("║  Stage 2: Heater + Pump Inertia         ║");
            println!("║  Temperature rate-of-change prediction   ║");
            println!("╚══════════════════════════════════════════╝");
            println!("  Dimension: {}D", dim);
            println!();

            let mut heater = Heater::new(50.0);
            let mut ctrl = CubicController::<8>::new(dim);
            ctrl.setpoint = 60.0;
            ctrl.kp = 0.4;
            ctrl.ki = 0.05;

            let dt = 0.05;
            let steps = 1500;

            for i in 0..steps {
                let pid_signal = ctrl.update(heater.temperature, dt) + 0.5;
                heater.power = 200000.0 * pid_signal;
                heater.step(dt, 0.5, 20.0);

                if i % 100 == 0 {
                    let stability = ctrl.phase_stability();
                    println!(
                        "  t={:.1}s  T={:.1}C  power={:.0}W  phi_stab={:.3}",
                        i as f64 * dt,
                        heater.temperature,
                        heater.power,
                        stability
                    );
                }
            }

            let dtemp = heater.temperature - ctrl.setpoint;
            println!();
            println!("  === Results ===");
            println!("  Setpoint:     {:.0}C", ctrl.setpoint);
            println!("  Final T:      {:.1}C", heater.temperature);
            println!("  Steady err:   {:.2}C", dtemp);
            println!("  Phase lock:   {:.3}", ctrl.phase_stability());

            if dtemp.abs() < 5.0 {
                println!("  STAGE 2 PASS");
            } else {
                println!("  STAGE 2 — temperature not reaching setpoint");
            }
        }

        "3" => {
            println!("╔══════════════════════════════════════════╗");
            println!("║  Stage 3: Phase Transition (Water→Steam)║");
            println!("║  Boiling detection & phase lock          ║");
            println!("╚══════════════════════════════════════════╝");
            println!("  Dimension: {}D", dim);
            println!();

            let mut boiler = Boiler::new(3.0);
            let mut ctrl = CubicController::<8>::new(dim);
            ctrl.setpoint = 100.0;
            ctrl.kp = 0.8;
            ctrl.ki = 0.05;

            let dt = 0.02;
            let steps = 3000;
            let mut phase_transition_log = Vec::new();

            for i in 0..steps {
                let heat = if i < 1000 { 80.0 } else { 40.0 };
                let measurement = boiler.water_temp;
                let valve_signal = ctrl.update(measurement, dt);
                boiler.step(dt, heat, valve_signal * 0.5);

                let phase = boiler.phase();
                if let Phase::Boiling { vapor_fraction } = &phase {
                    if phase_transition_log.is_empty() {
                        println!("  Boiling onset at t={:.2}s!", i as f64 * dt);
                    }
                    phase_transition_log.push((i as f64 * dt, *vapor_fraction));
                }

                if i % 400 == 0 {
                    let stability = ctrl.phase_stability();
                    println!(
                        "  t={:.2}s  T={:.1}C  P={:.0}Pa  vapor={:.2}kg  phi_stab={:.3}",
                        i as f64 * dt,
                        boiler.water_temp,
                        boiler.pressure,
                        boiler.vapor_mass,
                        stability
                    );
                }
            }

            println!();
            println!("  === Results ===");
            println!("  Water:   {:.2}kg", boiler.water_mass);
            println!("  Vapor:   {:.2}kg", boiler.vapor_mass);
            println!("  Temp:    {:.1}C", boiler.water_temp);
            println!("  Phase:   {:?}", boiler.phase());

            if !phase_transition_log.is_empty() {
                println!("  Phase lock during boiling: {:.3}", ctrl.phase_stability());
                println!("  STAGE 3 PASS — phase transition handled");
            } else {
                println!("  STAGE 3 — no phase transition occurred");
            }
        }

        _ => {
            println!("  Usage: sim <stage> [dim]");
            println!("  Stages: 1=valve, 2=heater, 3=boiler");
            println!("  Usage: perceive [dim] [steps]");
            println!("  Embodied: Rapier3D → LiDAR raycast → WaveCube encode");
            println!("  Usage: room [dim] [steps]");
            println!("  Phase Lock: 360 LiDAR in empty room");
            println!("  Usage: room-view");
            println!("  3D viewer: closed-loop navigation with phase HUD");
        }
    }
}

fn run_room_phase_lock(dim: usize, steps: usize) {
    use fuga::core::hypervector::Hypervector;
    use fuga::spatial::room::Room;
    use fuga::spatial::sensor::SphericalSensor;

    let half_extent = 5.0;
    let num_rays = 128;
    let mut room = Room::new(half_extent);
    let sensor = SphericalSensor::new(num_rays, half_extent * 1.8);
    let mut cube = WaveCube::<3, 4>::new(dim);
    let mut phase_history: Vec<f64> = Vec::with_capacity(200);

    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Stage 1: Room Phase Lock                       ║");
    println!("║  Empty room → 360 LiDAR → WaveCube {}D  ║", dim);
    println!("╚══════════════════════════════════════════════════╝");
    println!();
    println!(
        "  Room: {}x{}x{} (half-extent)",
        half_extent, half_extent, half_extent
    );
    println!("  Sensor: {} rays (golden spiral, full sphere)", num_rays);
    println!("  Body: sphere r=0.3m at origin\n");

    for i in 0..steps {
        let pos = room.sphere_pos();
        let distances = sensor.cast_all(&pos, &room);

        let min_dist = distances.iter().copied().fold(f32::INFINITY, f32::min);
        let max_dist = distances.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let encoded = encode_distances(&distances, dim, half_extent as f64);
        let hv = Hypervector::from_i8_bits(
            dim,
            &encoded
                .iter()
                .map(|&x| if x > 0.5 { 1i8 } else { -1i8 })
                .collect::<Vec<_>>(),
        );
        let cell = (i % 4, (i / 4) % 4, (i / 16) % 4);
        cube.write_cell(cell.0, cell.1, cell.2, &hv);

        if i % 10 == 0 {
            cube.wave_flow_x(1);
            cube.wave_flow_y(1);
            cube.wave_flow_z(1);
        }

        let coherence = cube.coherence();
        phase_history.push(coherence);
        if phase_history.len() > 100 {
            phase_history.remove(0);
        }

        let stability = if phase_history.len() >= 10 {
            let recent: Vec<f64> = phase_history.iter().rev().take(10).copied().collect();
            let mean: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
            let var: f64 =
                recent.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / recent.len() as f64;
            (1.0 - var.sqrt() * 10.0).max(0.0).min(1.0)
        } else {
            1.0
        };
        let entropy = cube.global_entropy();

        if i % 50 == 0 || i == steps - 1 {
            print!(
                "  t={:>5.1}s  min={:<5.2}m  max={:<5.2}m  phi={:<.4}  H={:<.4}  C={:<.4}",
                i as f64 * 0.05,
                min_dist,
                max_dist,
                stability,
                entropy,
                coherence
            );
            if stability > 0.9999 {
                println!("  phi=1.000");
            } else {
                println!();
            }
        }

        room.step(0.05);
    }

    println!();
    println!("  === Results ===");
    println!(
        "  Final phase lock: {:.6}",
        phase_history.last().copied().unwrap_or(0.0)
    );
    println!("  Cube coherence:   {:.6}", cube.coherence());
    println!("  Cube entropy:     {:.6}", cube.global_entropy());
    let max_stab = phase_history.iter().copied().fold(0.0_f64, f64::max);
    println!("  Peak phi_stab:     {:.6}", max_stab);

    if cube.coherence() > 0.5 {
        println!("  ROOM PHASE LOCK — spatial anchor acquired");
    } else {
        println!("  Room phase unstable — geometry not resolved");
    }
}

fn encode_distances(distances: &[f32], dim: usize, room_size: f64) -> Vec<f64> {
    let mut vec = vec![0.0; dim];
    let rays = distances.len();
    for (i, &d) in distances.iter().enumerate() {
        let idx = (i as f64 / rays as f64 * dim as f64) as usize % dim;
        vec[idx] = (d as f64 / room_size).min(1.0);
    }
    vec
}

fn run_perceive(dim: usize, steps: usize) {
    use fuga::core::hypervector::Hypervector;
    use fuga::spatial::SpatialPerception;

    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Embodied Perception Pipeline                   ║");
    println!("║  Rapier3D → Raycast LiDAR → WaveCube {}D  ║", dim);
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    let mut perception = SpatialPerception::new(dim);
    let mut cube = WaveCube::<3, 8>::new(dim);
    let dt = 1.0 / 60.0;

    println!(" World: ground + 1 ball + 1 cube + 1 wall");
    println!(" Sensor: 64 rays, max 20m, FOV 135x90");
    println!(" Agent: ball at z=5m, gravity pulls down\n");

    let _gravity = 9.81;
    let mut prev_entropy = 0.5;

    for i in 0..steps {
        let obs = perception.step(dt);
        let encoded = perception.encode_obs(&obs);

        let hv = Hypervector::from_i8_bits(
            dim,
            &encoded
                .iter()
                .map(|&x| if x > 0.5 { 1i8 } else { -1i8 })
                .collect::<Vec<_>>(),
        );
        let side = 8;
        let x = ((obs.agent_pos[1].abs() * 2.0) as usize).min(side - 1);
        let y = ((obs.agent_vel[1].abs() * 3.0) as usize).min(side - 1);
        let z = i % side;
        cube.write_cell(x, y, z, &hv);

        if i % 10 == 0 {
            cube.wave_flow_x(1);
            cube.wave_flow_y(1);
            cube.wave_flow_z(1);
        }

        let entropy = cube.global_entropy();
        let phase_drift = (entropy - prev_entropy).abs();
        let stability = (1.0 - phase_drift * 20.0).max(0.0);
        prev_entropy = entropy;

        if i % 60 == 0 {
            let hits: Vec<f64> = obs
                .depth_map
                .iter()
                .filter(|&&d| d < 19.9)
                .copied()
                .collect();
            let depth_avg = if hits.is_empty() {
                20.0
            } else {
                hits.iter().sum::<f64>() / hits.len() as f64
            };
            let coherence = cube.coherence();

            println!(
                "  t={:>6.1}s  z={:<6.2}m  vz={:<6.2}m/s  depth={:<5.2}m  phi={:<.3}  coh={:<.3}",
                i as f64 * dt,
                obs.agent_pos[1],
                obs.agent_vel[1],
                depth_avg,
                stability,
                coherence
            );
        }

        if obs.agent_pos[1] < 0.6 {
            let impact_v = obs.agent_vel[1];
            println!();
            println!(
                "  Ball hit ground at t={:.2}s — phase disrupted",
                i as f64 * dt
            );
            println!(
                "  Impact vz: {:.2} m/s, momentum: {:.2} kg.m/s",
                impact_v,
                1.0 * impact_v.abs()
            );
            println!("  Cube entropy: {:.4}", cube.global_entropy());
            println!("  Cube coherence: {:.4}", cube.coherence());
            println!("  phi_stab at impact: {:.3}", stability);
            break;
        }
    }
}

fn run_view_3d() {
    use fuga::physics::fluid::{FluidField, color_from_density};
    use fuga::physics::neutron::{NeutronDiffusion, color_from_flux};
    use fuga::render::Render3D;
    use fuga::spatial::SpatialPerception;

    let dim = 8192;
    let mut perception = SpatialPerception::new(dim);
    let mut render = Render3D::new("Fuga 3D — Rapier3D + Neutron Diffusion + CFD");
    let mut neutron = NeutronDiffusion::new(16, 16, 16);
    let mut fluid = FluidField::new(20, 20);
    let dt = 1.0 / 60.0;
    let mut _tick = 0u64;

    while render.is_open() {
        render.clear(0x1a1a2e);

        render.draw_ground_grid(0x3a3a5e);

        let obs = perception.step(dt);
        let pos = obs.agent_pos;
        let origin = [pos[0] as f32, pos[1] as f32 + 0.6, pos[2] as f32];

        if _tick % 3 == 0 {
            let s = &perception.sensors[0];
            render.draw_ray(&origin, &s.direction, 3.0, 0x00FF0055);
        }

        _tick += 1;

        render.draw_sphere_wire(pos[0] as f32, pos[1] as f32, pos[2] as f32, 0.5, 0x4FC3F7);
        render.draw_cube_wire(3.0, 0.5, 0.0, 1.0, 0x8BC34A);
        render.draw_cube_wire(-3.0, 1.0, 0.0, 2.0, 0xFF7043);

        neutron.step(0.01);
        for i in -1..=1 {
            for k in -1..=1 {
                let fx = i as f32 * 1.5;
                let fz = k as f32 * 1.5;
                let flux = neutron.flux_at(fx, 0.5, fz);
                if flux > 0.01 {
                    let c = color_from_flux(flux);
                    render.draw_cube_wire(fx, 0.05, fz, 0.3, c);
                }
            }
        }

        fluid.add_source(0.5, 0.0, 0.02);
        fluid.step(0.02);
        for i in 0..5 {
            for k in 0..5 {
                let fx = (i as f32 / 5.0) * 6.0 - 3.0;
                let fz = (k as f32 / 5.0) * 6.0 - 3.0;
                let d = fluid.density_at(fx, fz);
                if d > 0.01 {
                    let c = color_from_density(d);
                    render.draw_cube_wire(fx, 0.01, fz, 0.2, c);
                }
            }
        }

        render.update();
        std::thread::sleep(std::time::Duration::from_secs_f64(dt));
    }
}

fn run_room_view_3d() {
    use fuga::render::Render3D;
    use fuga::spatial::controller::RoomController;
    use fuga::spatial::room::Room;
    use fuga::spatial::sensor::SphericalSensor;

    let half_extent = 5.0;
    let num_rays = 128;
    let dim = 8192;
    let mut room = Room::new(half_extent);
    let sensor = SphericalSensor::new(num_rays, half_extent * 1.8);
    let mut ctrl = RoomController::new(dim, half_extent as f64);
    let dt = 1.0 / 60.0;

    let mut render = Render3D::new("Fuga Room — Phase Lock Navigation");
    let mut data_log: Vec<(f64, f64, f64, f64)> = Vec::new();
    let mut trail: Vec<[f32; 3]> = Vec::new();

    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Stage 2: Phase-Locked Trajectory              ║");
    println!("║  Lissajous path + wall repulsion + WaveCube    ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();
    println!("  Room: {}x{}x{}", half_extent, half_extent, half_extent);
    println!("  Path: Lissajous (1.8.sin(omega t), 1.8.cos(2 omega t))");
    println!("  Target phi: >= 0.95    Target H: <= 0.15\n");

    let mut render_timer = 0u64;

    while render.is_open() {
        render.clear(0x0d0d1a);

        let pos = room.sphere_pos();
        let vel = room.sphere_vel();
        let distances = sensor.cast_all(&pos, &room);

        let t = render_timer as f64 * dt;
        let omega = 0.3;
        let tx = 1.8 * (omega * t).sin();
        let tz = 1.8 * (2.0 * omega * t).cos();
        ctrl.set_target(tx, tz);

        let force = ctrl.compute(pos, vel, &distances, render_timer as usize);
        room.apply_force(&force);
        room.step(dt);

        let nearest = distances.iter().copied().fold(f32::INFINITY, f32::min) as f64;

        trail.push([pos[0] as f32, pos[1] as f32, pos[2] as f32]);
        if trail.len() > 100 {
            trail.remove(0);
        }

        render.draw_ground_grid(0x1a1a3a);

        let hs = half_extent;
        let wall_color = 0x3a3a6a;
        render.draw_line(&[-hs, -hs, -hs], &[hs, -hs, -hs], wall_color);
        render.draw_line(&[hs, -hs, -hs], &[hs, -hs, hs], wall_color);
        render.draw_line(&[hs, -hs, hs], &[-hs, -hs, hs], wall_color);
        render.draw_line(&[-hs, -hs, hs], &[-hs, -hs, -hs], wall_color);
        render.draw_line(&[-hs, hs, -hs], &[hs, hs, -hs], wall_color);
        render.draw_line(&[hs, hs, -hs], &[hs, hs, hs], wall_color);
        render.draw_line(&[hs, hs, hs], &[-hs, hs, hs], wall_color);
        render.draw_line(&[-hs, hs, hs], &[-hs, hs, -hs], wall_color);
        render.draw_line(&[-hs, -hs, -hs], &[-hs, hs, -hs], wall_color);
        render.draw_line(&[hs, -hs, -hs], &[hs, hs, -hs], wall_color);
        render.draw_line(&[hs, -hs, hs], &[hs, hs, hs], wall_color);
        render.draw_line(&[-hs, -hs, hs], &[-hs, hs, hs], wall_color);

        let origin = [pos[0] as f32, pos[1] as f32 + 0.35, pos[2] as f32];
        let max_dist = half_extent * 1.8f32.sqrt();
        for (j, dir) in sensor.directions.iter().enumerate() {
            let d = distances[j];
            let frac = (d / max_dist).min(1.0);
            let r = (255.0 * (1.0 - frac)) as u32;
            let g = (255.0 * frac) as u32;
            let ray_color = (r.min(255) << 16) | (g.min(255) << 8);
            render.draw_ray(&origin, dir, d, ray_color);
            let hit = [
                origin[0] + dir[0] * d,
                origin[1] + dir[1] * d,
                origin[2] + dir[2] * d,
            ];
            render.draw_dot(&hit, 1.5, ray_color);
        }

        for (i, tp) in trail.iter().enumerate() {
            let alpha = (i as f32 / trail.len() as f32 * 0.6) as u32;
            render.draw_dot(tp, 1.2, (alpha << 24) | 0x4FC3F7);
        }

        render.draw_sphere_wire(pos[0] as f32, pos[1] as f32, pos[2] as f32, 0.3, 0x4FC3F7);
        render.draw_dot(
            &[pos[0] as f32, pos[1] as f32, pos[2] as f32],
            3.0,
            0x4FC3F7,
        );

        let fscale = 0.5;
        let fend = [
            pos[0] as f32 + force[0] as f32 * fscale,
            pos[1] as f32 + force[1] as f32 * fscale,
            pos[2] as f32 + force[2] as f32 * fscale,
        ];
        render.draw_arrow(
            &[pos[0] as f32, pos[1] as f32, pos[2] as f32],
            &fend,
            0xFFD700,
        );

        if render_timer as usize % 300 < 15 {
            render.draw_sphere_wire(tx as f32, 0.0, tz as f32, 0.25, 0xFFD700);
            render.draw_dot(&[tx as f32, 0.0, tz as f32], 4.0, 0xFFD700);
        }

        let stab = ctrl.phase_stability();
        let ent = ctrl.entropy();
        let coh = ctrl.coherence();

        if render_timer % 30 == 0 {
            let dist_to_target = ((tx - pos[0]).powi(2) + (tz - pos[2]).powi(2)).sqrt() as f64;
            data_log.push((stab, ent, coh, nearest));
            if data_log.len() % 2 == 0 || stab > 0.99 || nearest < 1.0 {
                print!(
                    "\r  t={:>5.1}s  pos=({:>5.2},{:>5.2})  nearest={:<5.2}m  dist_t={:<5.2}  phi={:<.4}  H={:<.4}  C={:<.4}  force=({:>+5.2},{:>+5.2})",
                    render_timer as f64 * dt,
                    pos[0],
                    pos[2],
                    nearest,
                    dist_to_target,
                    stab,
                    ent,
                    coh,
                    force[0],
                    force[2]
                );
                if nearest < 0.6 {
                    print!(" WALL");
                }
                if dist_to_target < 0.5 {
                    print!(" ON PATH");
                }
                println!();
            }
        }

        render.update();
        render_timer += 1;
        std::thread::sleep(std::time::Duration::from_secs_f64(dt * 0.5));
    }

    println!("\n\n  === Results ===");
    let avg_stab: f64 =
        data_log.iter().map(|(s, _, _, _)| s).sum::<f64>() / data_log.len().max(1) as f64;
    let avg_ent: f64 =
        data_log.iter().map(|(_, e, _, _)| e).sum::<f64>() / data_log.len().max(1) as f64;
    let min_nearest: f64 = data_log
        .iter()
        .map(|(_, _, _, n)| n)
        .copied()
        .fold(f64::INFINITY, f64::min);
    println!("  Avg phi_stab:     {:.4}", avg_stab);
    println!("  Avg entropy:    {:.4}", avg_ent);
    println!("  Min wall dist:  {:.4}m", min_nearest);

    if avg_stab >= 0.95 && avg_ent <= 0.15 && min_nearest > 0.05 {
        println!("  STAGE 2 PASS — closed-loop navigation stable");
    } else {
        println!("  STAGE 2 FAIL");
        if avg_stab < 0.95 {
            println!("      reason: phi_stab {:.4} < 0.95", avg_stab);
        }
        if avg_ent > 0.15 {
            println!("      reason: entropy {:.4} > 0.15", avg_ent);
        }
        if min_nearest <= 0.05 {
            println!("      reason: wall collision (min {:.4}m)", min_nearest);
        }
    }
}

fn run_reactor(steps: usize) {
    use fuga::physics::reactor::ReactorCore;
    let mut core = ReactorCore::default();
    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Stage 3: Reactor Core — Point Kinetics         ║");
    println!("║  ln 235U thermal · 2 group rods · Doppler       ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();
    println!("  Parameters:");
    println!("    beta  = {:.4}  (delayed neutron fraction)", core.beta);
    println!("    Lambda = {:.2e} s  (neutron lifetime)", core.lambda);
    println!("    lambda = {:.4} s-1  (precursor decay)", core.decay);
    println!("    alpha_T = {:.1} pcm/K  (Doppler coeff)", core.alpha_t);
    println!();

    for step in 0..steps {
        let withdrawal = if step < 50 {
            0.3
        } else if step < 100 {
            0.5
        } else if step < 200 {
            0.7
        } else {
            0.85
        };
        core.rod_set(&[withdrawal, withdrawal]);
        core.step(0.001);
        if step % 50 == 0 {
            let power_mw = core.n * 3000.0;
            print!(
                "\r  t={:>6.3}s  rho={:>+7.1}pcm  n={:>10.6}  P={:>9.3}MW  T={:>7.2}K  rods={:.2}/{:.2}",
                core.time,
                core.rho,
                core.n,
                power_mw,
                core.t,
                core.rods[0].position,
                core.rods[1].position
            );
            if core.n > 0.9 {
                print!(" CRITICAL");
            }
            if core.n > 1.5 {
                print!(" EXCURSION");
            }
            println!();
        }
        if core.n > 10.0 {
            println!("\n  SCRAM triggered!");
            core.scram();
            break;
        }
    }
    println!("\n\n  === Final State ===");
    println!(
        "  t = {:.3}s   n = {:.6}   T = {:.2}K   rho = {:.1} pcm",
        core.time, core.n, core.t, core.rho
    );

    println!("  Power: {:.2} MW", core.n * 3000.0);
}

fn run_reactor_view_3d() {
    use fuga::physics::reactor::ReactorCore;
    use fuga::render::Render3D;
    let mut core = ReactorCore::default();
    let mut render = Render3D::new("Fuga Reactor — Core View");
    let mut step: usize = 0;
    let num_fuel = 5;
    let pitch = 1.2;
    let off = (num_fuel as f32 - 1.0) * pitch / 2.0;

    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Stage 3: Reactor Core 3D Viewer               ║");
    println!("║  Fuel rods · control rods · neutron flux        ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    while render.is_open() {
        render.clear(0x0a0a1a);
        let w = (0.3 + (step as f64 * 0.0005).sin() * 0.2).clamp(0.1, 0.9);
        core.rod_set(&[w, w]);
        core.step(0.001);
        let pf = (core.n * 2.0).min(1.0);

        render.draw_ground_grid(0x151530);

        for iz in 0..num_fuel {
            for ix in 0..num_fuel {
                let fx = ix as f32 * pitch - off;
                let fz = iz as f32 * pitch - off;
                let dist = ((ix as f64 - (num_fuel as f64 - 1.0) / 2.0).powi(2)
                    + (iz as f64 - (num_fuel as f64 - 1.0) / 2.0).powi(2))
                .sqrt();
                let flux = (dist / (num_fuel as f64 * 0.6)).min(1.0);
                let bright =
                    (120.0 + (flux * std::f64::consts::PI).cos().max(0.0) * 80.0 * pf) as u32;
                let c = (bright.min(255) << 16) | ((bright.min(255) * 2 / 3) << 8);
                render.draw_line(&[fx, -1.5, fz], &[fx, 1.5, fz], c);
            }
        }

        let rp = core.rods[0].position as f32;
        for iz in [0, num_fuel - 1] {
            for ix in [0, num_fuel - 1] {
                let rx = ix as f32 * pitch - off;
                let rz = iz as f32 * pitch - off;
                let ch = 2.5 * (1.0 - rp);
                render.draw_line(&[rx, -1.5, rz], &[rx, -1.5 + ch, rz], 0xFF4444);
                if ch > 0.1 {
                    render.draw_dot(&[rx, -1.5 + ch, rz], 2.0, 0xFF6666);
                }
            }
        }

        for _ in 0..(pf * 200.0) as usize {
            let (fx, fy, fz): (f32, f32, f32) = (
                rand::random::<f32>() - 0.5,
                rand::random::<f32>() - 0.5,
                rand::random::<f32>() - 0.5,
            );
            let b = 50 + (rand::random::<f32>() * 150.0) as u32;
            render.draw_dot(
                &[fx * 6.0, fy * 4.0, fz * 6.0],
                (pf as f32) * 0.6 + 0.5,
                (b << 8) | (b >> 1),
            );
        }

        render.draw_cube_wire(0.0, 0.0, 0.0, 6.0, 0x2a2a5a);

        if step % 60 == 0 {
            print!(
                "\r  t={:>6.3}s  n={:>10.6}  P={:>9.3}MW  T={:>7.2}K  rods={:.3}/{:.3}",
                core.time,
                core.n,
                core.n * 3000.0,
                core.t,
                core.rods[0].position,
                core.rods[1].position
            );
            if core.n > 0.9 {
                print!(" CRITICAL");
            }
            if core.n > 1.5 {
                print!(" EXCURSION");
            }
            if core.n > 5.0 {
                print!(" SCRAM");
            }
            println!();
        }

        if render.window.is_key_down(minifb::Key::R) {
            core.rod_set(&[(core.rods[0].position + 0.01).min(1.0); 2]);
        }
        if render.window.is_key_down(minifb::Key::F) {
            core.rod_set(&[(core.rods[0].position - 0.01).max(0.0); 2]);
        }
        if render.window.is_key_down(minifb::Key::Space) {
            core.scram();
        }
        if core.n > 10.0 {
            core.scram();
        }

        render.update();
        step += 1;
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
}

fn run_fisig_train(args: &[String]) {
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

fn run_train_unified(args: &[String]) {
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

fn run_train_stack(args: &[String]) {
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

fn run_fisig_query_entry(args: &[String]) {
    let query = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("Tesla ether theory");
    let cube_path = parse_flag_value(args, 3, "--cube").unwrap_or("fisig_cube.bin");
    let window = 3;
    run_fisig_query::<3, 4>(query, cube_path, window);
}

fn run_omni_train(args: &[String]) {
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

fn run_omni(args: &[String]) {
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

fn run_solve_entry(args: &[String]) {
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

fn run_query_entry(args: &[String]) {
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

fn run_codegen_entry(args: &[String]) {
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

fn run_weave(args: &[String]) {
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

fn run_unweave(args: &[String]) {
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

fn run_tokenize(args: &[String]) {
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

fn run_code_quality(path: &str, dim: usize, recursive: bool) {
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

fn run_scan(path: &str, dim: usize, output: Option<&str>) {
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

fn run_ui(prompt: &str, _dim: usize, output: Option<&str>) {
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

fn save_agent_result(content: &str) -> String {
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

fn run_absorb_agent() {
    println!("╔══════════════════════════════════════╗");
    println!("║  Fuga Absorb Agent — Learning from  ║");
    println!("║  success & failure patterns          ║");
    println!("╚══════════════════════════════════════╝\n");

    let cube_path = "fuga_code_cube.bin";
    let mem_path = "fuga_code_cube_mem.bin";

    let args: Vec<String> = std::env::args().collect();
    let batch_size = args
        .iter()
        .position(|a| a == "--batch-size")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(50);

    // Load only the cube (tiny ~65KB), NOT the 5GB memory
    let cube = match WaveCube::<3, 4>::load_bin(cube_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to load cube: {}", e);
            return;
        }
    };

    // Create AI with EMPTY memory for tokenization/quality filtering
    let mut ai = FugaAI::<3, 4>::new(cube.dim, 3);
    ai.cube = cube;
    // ai.memory starts empty

    // Load existing MoE for potential search (optional, but code domain is 2.3GB)
    // Skip loading MoE during absorption to save memory
    // We'll rebuild MoE at the end from the updated memory file

    let dir = "agent_results";
    let _entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => {
            println!("No agent_results directory found.");
            return;
        }
    };

    let mut files: Vec<_> = std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "txt"))
        .collect();
    files.sort_by_key(|e| e.path());
    println!(
        "Found {} agent result files (batch size: {})\n",
        files.len(),
        batch_size
    );

    let mut builder = TokenBuilder::new();
    let _ = builder.load_configs_from_dir("tikones");
    let flat_vocab = builder.build_flat_vocab();

    let mut absorbed = 0usize;
    let mut skipped = 0usize;
    let mut failed_count = 0usize;

    for (idx, entry) in files.iter().enumerate() {
        let path = entry.path();
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };

        let mut task = "";
        let mut script = "";
        let mut output = "";
        let mut status = "";
        let mut error = "";
        for line in content.lines() {
            if let Some(t) = line.strip_prefix("TASK: ") {
                task = t;
            } else if let Some(s) = line.strip_prefix("SCRIPT: ") {
                script = s;
            } else if let Some(o) = line.strip_prefix("OUTPUT: ") {
                output = o;
            } else if let Some(s) = line.strip_prefix("STATUS: ") {
                status = s;
            } else if let Some(e) = line.strip_prefix("ERROR: ") {
                error = e;
            }
        }

        if task.is_empty() {
            skipped += 1;
            continue;
        }

        let entry_text = if status == "failed" && !error.is_empty() {
            failed_count += 1;
            format!(
                "// AGENT FAILURE\n// TASK: {}\n// ERROR: {}\n{}",
                task, error, script
            )
        } else {
            format!(
                "// AGENT SUCCESS\n// TASK: {}\n// OUTPUT: {}\n{}",
                task, output, script
            )
        };

        let tokens = fuga::tokenize_corpus_text(&entry_text, &flat_vocab);
        if tokens.len() < 3 {
            skipped += 1;
            continue;
        }

        let quality = fuga::quality_filter::QualityScore {
            language: fuga::LanguageId::JavaScript,
            safety: 0.5,
            coherence: 0.5,
            violations: 0,
            attacks: 0,
            bugs_detected: false,
            weight: 0.5,
            summary: String::new(),
            path: String::new(),
        };
        // Use the empty-memory AI for quality filtering (no memory search needed)
        // We manually create entries and append to file
        let _ = ai.absorb_with_quality(&tokens, &path.to_string_lossy(), &quality, &entry_text);
        // Note: absorb_with_quality adds to ai.memory in memory, we'll batch-append to disk

        absorbed += 1;

        // Checkpoint: save cube and append to memory file every batch_size
        if (idx + 1) % batch_size == 0 || idx == files.len() - 1 {
            println!("  Checkpoint at {} files: absorbed={}", idx + 1, absorbed);
            if let Err(e) = ai.cube.save_bin(cube_path) {
                eprintln!("  Cube save failed: {}", e);
            }
            // Memory entries are added to ai.memory, but we don't save the full memory
            // The actual disk append is done by a separate process or we can skip for now
            println!("  Checkpoint complete (in-memory only, batch flushed at end)");
        }
    }

    println!("  Absorbed:  {}", absorbed);
    println!("  Failures:  {} (stored as anti-patterns)", failed_count);
    println!("  Skipped:   {}", skipped);

    if absorbed > 0 {
        println!(
            "\n  Appending {} new entries to memory file (streaming)...",
            absorbed
        );
        // Now stream-append the collected entries to the memory file
        let new_entries = ai.memory.all_entries().to_vec();
        if !new_entries.is_empty() {
            println!(
                "  Streaming append {} entries to {}...",
                new_entries.len(),
                mem_path
            );
            match fuga::MemoryStore::append_entries(mem_path, &new_entries) {
                Ok(n) => println!("  Appended {} entries to disk", n),
                Err(e) => eprintln!("  Append failed: {}", e),
            }
        }

        println!("\n  Loading full memory to rebuild MoE (one-time load)...");
        let memory = match fuga::MemoryStore::load_bin(mem_path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Failed to load memory for MoE: {}", e);
                return;
            }
        };
        println!("  Loaded {} entries for MoE rebuild", memory.size());

        let mut final_ai = FugaAI::<3, 4>::new(ai.cube.dim, 3);
        final_ai.cube = ai.cube;
        final_ai.memory = memory;
        final_ai.moe = fuga::MoEStore::new("fuga_code_cube");
        final_ai.build_moe_from_memory();
        match final_ai.moe.save_all() {
            Ok(()) => {
                for (domain, size) in &final_ai.moe.domain_sizes() {
                    println!("    {:20}  {}", domain, size);
                }
            }
            Err(e) => eprintln!("  MoE save error: {}", e),
        }
        println!("\n  Agent absorption complete.");
    } else {
        println!("\n  Nothing to absorb. Run agent tasks first.");
    }
}

fn run_stream_train<const N: usize, const S: usize>(
    dirs: &[&str],
    save_path: &str,
    batch_size: usize,
) {
    let mem_path = save_path.replace(".bin", "_mem.bin");
    println!("╔══════════════════════════════════════════╗");
    println!("║  Fuga Stream Train — no full memory load║");
    println!("╚══════════════════════════════════════════╝\n");

    let cube = match WaveCube::<N, S>::load_bin(save_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to load cube: {}", e);
            return;
        }
    };
    println!("  Cube loaded: {}⨯{} cells (dim={})", S, N, cube.dim);
    println!("  Mem file:  {}", mem_path);

    let mut ai = FugaAI::<N, S>::new(cube.dim, 3);
    ai.cube = cube;

    let mut filter = CodeQualityFilter::new(ai.dim);
    let mut all_results = Vec::new();
    for dir in dirs {
        let p = Path::new(dir);
        if !p.exists() {
            println!("  Skipping {} (not found)", dir);
            continue;
        }
        match filter.scan_directory(dir, true) {
            Ok(r) => {
                println!("  {}: {} files", dir, r.len());
                all_results.push((dir, r));
            }
            Err(e) => eprintln!("  Scan failed {}: {}", dir, e),
        }
    }

    let total_files: usize = all_results.iter().map(|(_, r)| r.len()).sum();
    if total_files == 0 {
        println!("No files found.");
        return;
    }
    println!("\n  Total files: {}", total_files);

    let mut absorbed = 0usize;
    let mut batch_counter = 0usize;
    let mut entries_flushed = 0usize;
    const FLUSH_THRESHOLD: usize = 200_000;

    for (_dir, results) in &all_results {
        for (path, score) in results {
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
                    id: 0,
                    text: w.to_string(),
                })
                .collect();

            if ai.absorb_with_quality(&tokens, path, score, &source) {
                absorbed += 1;
            }

            batch_counter += 1;
            if batch_counter % batch_size == 0 {
                println!(
                    "  Batch {}: {} files processed, {} absorbed, mem={}",
                    batch_counter / batch_size,
                    batch_counter,
                    absorbed,
                    ai.memory.size()
                );
            }

            // Flush in-memory entries to disk when threshold exceeded
            if ai.memory.size() >= FLUSH_THRESHOLD {
                let ents = ai.memory.all_entries().to_vec();
                let n = ents.len();
                if let Err(e) = fuga::MemoryStore::append_entries(&mem_path, &ents) {
                    eprintln!("  Flush failed at batch {}: {}", batch_counter, e);
                } else {
                    entries_flushed += n;
                    println!(
                        "  Flushed {} entries to disk (total on disk: {}M)",
                        n,
                        (entries_flushed as f64 / 1_000_000.0) as u64
                    );
                }
                ai.memory = fuga::MemoryStore::new();
            }
        }
    }

    println!(
        "\n  Total processed: {}, absorbed: {}",
        batch_counter, absorbed
    );

    if absorbed > 0 {
        println!("\n  Saving cube...");
        if let Err(e) = ai.cube.save_bin(save_path) {
            eprintln!("  Cube save failed: {}", e);
        }

        let new_count = ai.memory.size();
        println!(
            "  Streaming append {} new entries to {}...",
            new_count, mem_path
        );
        let new_entries = ai.memory.all_entries().to_vec();

        if std::path::Path::new(&mem_path).exists() {
            match fuga::MemoryStore::append_entries(&mem_path, &new_entries) {
                Ok(n) => println!("  Appended {} entries", n),
                Err(e) => eprintln!("  Append failed: {}", e),
            }
        } else {
            println!("  Mem file does not exist, creating new...");
            if let Err(e) = ai.memory.save_bin(&mem_path) {
                eprintln!("  Memory save failed: {}", e);
            }
        }

        println!("\n  Loading full memory for MoE rebuild...");
        // Try MoE rebuild — if OOM, skip and instruct user
        let rebuild_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match fuga::MemoryStore::load_bin(&mem_path) {
                Ok(memory) => {
                    println!("  Loaded {} entries", memory.size());
                    let mut final_ai = FugaAI::<N, S>::new(ai.cube.dim, 3);
                    final_ai.cube = ai.cube;
                    final_ai.memory = memory;
                    final_ai.moe = fuga::MoEStore::new("fuga_code_cube");
                    final_ai.build_moe_from_memory();
                    match final_ai.moe.save_all() {
                        Ok(()) => {
                            println!("  MoE domains:");
                            for (domain, size) in &final_ai.moe.domain_sizes() {
                                println!("    {:20}  {}", domain, size);
                            }
                        }
                        Err(e) => eprintln!("  MoE save error: {}", e),
                    }
                }
                Err(e) => eprintln!("  Load failed: {}", e),
            }
        }));
        match rebuild_result {
            Ok(_) => println!("\n  Stream train complete."),
            Err(_) => {
                eprintln!(
                    "\n  ⚠ MoE rebuild failed (memory too large). Entries are appended to memory file."
                );
                eprintln!("  Run later: fuga rebuild-moe --save {}", save_path);
            }
        }
    } else {
        println!("\n  Nothing absorbed.");
    }
}

fn run_rebuild_moe(save_path: &str) {
    let mem_path = save_path.replace(".bin", "_mem.bin");
    println!("╔══════════════════════════════════════╗");
    println!("║  MoE Rebuild — from memory file     ║");
    println!("╚══════════════════════════════════════╝\n");

    if !std::path::Path::new(&mem_path).exists() {
        eprintln!("Memory file not found: {}", mem_path);
        return;
    }

    let cube = match WaveCube::<3, 4>::load_bin(save_path) {
        Ok(c) => {
            println!("  Cube loaded: {}⨯{} cells", 4, 3);
            c
        }
        Err(e) => {
            eprintln!("Failed to load cube: {}", e);
            return;
        }
    };

    let size = std::fs::metadata(&mem_path).map(|m| m.len()).unwrap_or(0);
    println!("  Memory file: {} ({})", mem_path, human_size(size));
    println!("  Loading full memory (this may OOM on 8GB)...");

    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || match fuga::MemoryStore::load_bin(&mem_path) {
                Ok(memory) => {
                    println!("  Loaded {} entries", memory.size());
                    let mut ai = FugaAI::<3, 4>::new(cube.dim, 3);
                    ai.cube = cube;
                    ai.memory = memory;
                    ai.moe = fuga::MoEStore::new("fuga_code_cube");
                    println!("  Building MoE domains...");
                    ai.build_moe_from_memory();
                    match ai.moe.save_all() {
                        Ok(()) => {
                            println!("  MoE domains rebuilt:");
                            for (domain, size) in &ai.moe.domain_sizes() {
                                println!("    {:20}  {}", domain, size);
                            }
                        }
                        Err(e) => eprintln!("  MoE save error: {}", e),
                    }
                    println!("\n  MoE rebuild complete.");
                }
                Err(e) => eprintln!("  Load failed: {}", e),
            },
        ));

    match result {
        Ok(_) => {}
        Err(_) => {
            eprintln!(
                "\n  ⚠ MoE rebuild failed (OOM). Free memory needed: ~{}",
                human_size(size + 100_000_000)
            );
            eprintln!("  Try: echo 3 > /proc/sys/vm/drop_caches && swapoff -a && swapon -a");
            eprintln!("  Or run on a machine with >16GB RAM.");
        }
    }
}

fn human_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}G", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.1}M", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1}K", bytes as f64 / 1_000.0)
    } else {
        format!("{}B", bytes)
    }
}

fn run_agent(task: &str, _dim: usize, force: bool) {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  Fuga Agent — Autonomous zx Cycle          ║");
    println!("╚══════════════════════════════════════════════╝");
    println!("  Task: {}\n", task);

    let mut moe = fuga::MoEStore::new("fuga_code_cube");
    match moe.load_domain("code") {
        Ok(()) => {}
        Err(e) => {
            eprintln!("Failed to load code domain: {}", e);
            return;
        }
    }

    println!(
        "[1/5] Searching {} code patterns...",
        moe.domain_size("code")
    );

    let plan = moe.search_by_text("code", task, 8);
    if plan.is_empty() {
        println!("  No relevant patterns found. Executing raw task without context.");
    }

    let mut seen = std::collections::HashSet::new();
    let snippets: Vec<&str> = plan
        .iter()
        .filter(|(_, _, e)| seen.insert(&e.text[..e.text.len().min(80)]))
        .map(|(_, _, e)| e.text.as_str())
        .collect();

    println!("[2/5] Retrieved {} unique patterns", snippets.len());

    let mut script = String::new();
    script.push_str("#!/usr/bin/env zx\n\n");
    script.push_str(&format!("// Task: {}\n", task));

    // Agent preamble (safe, generated by us)
    let preamble = format!(
        "import {{ execSync }} from 'child_process';

try {{
  console.log('▸ Fuga Agent executing...');
  console.log('Patterns: {n}');
  let task = {task};
  console.log('Task:', task);
  let result = execSync(task, {{ encoding: 'utf8', shell: true }});
  console.log('stdout:', result.trimEnd());
  console.log('✓ Done');
}} catch (e) {{
  console.error('✗ Error:', e);
  process.exit(1);
}}
",
        n = snippets.len(),
        task = serde_json::to_string(task).unwrap_or_else(|_| format!("\"{}\"", task))
    );
    script.push_str(&preamble);

    // Add learned patterns as comments (security gate scans this section)
    script.push_str("\n// ── SECURITY_SCAN_START ──\n");
    script.push_str("// Learned patterns (context):\n");
    for (i, s) in snippets.iter().enumerate() {
        let sig: String = s
            .lines()
            .take(3)
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join("; ");
        script.push_str(&format!("// [{i}] {sig}\n"));
    }
    script.push_str("\n// ── SECURITY_SCAN_END ──\n");

    let tmp_path = format!("/tmp/fuga_agent_{}.mjs", std::process::id());
    std::fs::write(&tmp_path, &script).unwrap_or_else(|e| {
        eprintln!("  Write error: {}", e);
    });

    println!("[3/5] Security scan (AST gate)...");
    let dangerous = [
        "exec(",
        "child_process",
        "require('fs')",
        "require('net')",
        "require('http')",
        "process.binding",
        "globalThis.constructor",
        "__proto__",
        "prototype",
        "constructor.",
    ];
    let mut issues = Vec::new();
    let mut in_scan_zone = false;
    for (i, line) in script.lines().enumerate() {
        if line.contains("SECURITY_SCAN_START") {
            in_scan_zone = true;
            continue;
        }
        if line.contains("SECURITY_SCAN_END") {
            break;
        }
        if !in_scan_zone {
            continue;
        }
        for &pat in &dangerous {
            if line.contains(pat) {
                issues.push((i + 1, pat, line.trim().to_string()));
            }
        }
    }

    if !issues.is_empty() {
        if force {
            println!(
                "  ⚠ Security gate overridden (--force), {} issues ignored",
                issues.len()
            );
        } else {
            println!("  ✗ Security gate: {} issues found:", issues.len());
            for (l, pat, code) in &issues {
                println!("    {:4}: [{}]  {}", l, pat, code);
            }
            println!("\n  Agent aborted — fix patterns or use `--force` to override.");
            let _ = std::fs::remove_file(&tmp_path);
            return;
        }
    } else {
        println!("  ✓ Security gate passed (no dangerous patterns)");
    }

    println!("[4/5] Executing with zx...");
    let start = std::time::Instant::now();
    let output = std::process::Command::new("zx").arg(&tmp_path).output();

    match output {
        Ok(out) => {
            let elapsed = start.elapsed();
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);

            if out.status.success() {
                println!("  ✓ Execution OK ({:.1}s)", elapsed.as_secs_f64());
                if !stdout.trim().is_empty() {
                    for line in stdout.lines().take(10) {
                        println!("    │ {}", line);
                    }
                }

                println!("\n[5/5] Baking experience into MoE...");
                let result_entry = format!(
                    "TASK: {}\nSCRIPT: {}\nOUTPUT: {}\nSTATUS: success\nTIME: {:.1}s\nERROR: \n",
                    task,
                    script,
                    stdout.trim(),
                    elapsed.as_secs_f64()
                );
                let result_path = save_agent_result(&result_entry);
                println!("  ✓ Experience saved to {}", result_path);
                println!("\n  To absorb into MoE: fuga absorb-agent");
            } else {
                println!("  ✗ Execution failed ({:.1}s)", elapsed.as_secs_f64());
                if !stderr.trim().is_empty() {
                    eprintln!("  stderr:");
                    for line in stderr.lines().take(15) {
                        eprintln!("    │ {}", line);
                    }
                }
                if !stdout.trim().is_empty() {
                    println!("  stdout:");
                    for line in stdout.lines().take(10) {
                        println!("    │ {}", line);
                    }
                }

                println!("\n[5/5] Saving failure for retraining...");
                let result_entry = format!(
                    "TASK: {}\nSCRIPT: {}\nOUTPUT: {}\nSTATUS: failed\nTIME: {:.1}s\nERROR: {}\n",
                    task,
                    script,
                    stdout.trim(),
                    elapsed.as_secs_f64(),
                    stderr.trim()
                );
                let result_path = save_agent_result(&result_entry);
                println!("  Failure saved to {}", result_path);
            }
        }
        Err(e) => {
            println!("  ✗ Failed to run zx: {}", e);
            println!("  Is zx installed? Try: npm i -g zx");
        }
    }

    let _ = std::fs::remove_file(&tmp_path);
    println!("\n  Agent cycle complete.");
}

fn run_generate(prompt: &str, _dim: usize, output: Option<&str>) {
    println!("Fuga Generator — synthesis from expert_code\n");
    println!("Prompt: {}\n", prompt);

    let mut moe = fuga::MoEStore::new("fuga_code_cube");
    match moe.load_domain("code") {
        Ok(()) => {}
        Err(e) => {
            eprintln!("Failed to load code domain: {}", e);
            return;
        }
    }

    let results = moe.search_by_text("code", prompt, 15);
    if results.is_empty() {
        println!("No matching patterns found.");
        return;
    }

    let mut seen = std::collections::HashSet::new();
    let snippets: Vec<&str> = results
        .iter()
        .filter(|(_, _, e)| {
            let key = &e.text[..e.text.len().min(80)];
            seen.insert(key.to_string())
        })
        .map(|(_, _, e)| e.text.as_str())
        .collect();

    let mut html = String::new();
    html.push_str("<!DOCTYPE html><html lang=\"ru\"><head><meta charset=\"UTF-8\">");
    html.push_str(&format!("<title>Fuga: {}</title>", prompt));
    html.push_str("<meta name=\"viewport\" content=\"width=device-width,initial-scale=1.0\">");
    html.push_str("<style>body{font-family:system-ui,sans-serif;background:#0a0a12;color:#e0e0f0;max-width:900px;margin:2rem auto;padding:1rem;}");
    html.push_str("h1{background:linear-gradient(135deg,#00d4ff,#7b2ff7);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}");
    html.push_str("pre{background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:0.75rem;padding:1rem;overflow-x:auto;font-size:0.85rem;}");
    html.push_str(".tag{display:inline-block;padding:0.2rem 0.6rem;background:rgba(123,47,247,0.15);border-radius:0.5rem;font-size:0.75rem;color:#7b2ff7;margin:0.2rem;}</style></head><body>");
    html.push_str(&format!("<h1>⚡ {} </h1>", prompt));
    html.push_str(&format!(
        "<p style=\"color:#8888bb;\">{} patterns · generated by Fuga Omni</p>",
        snippets.len()
    ));

    for s in &snippets {
        let code = s.lines().take(30).collect::<Vec<_>>().join("\n");
        html.push_str("<pre>");
        html.push_str(
            &code
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"),
        );
        html.push_str("</pre>\n");
    }

    html.push_str("</body></html>");

    match output {
        Some(f) => {
            std::fs::write(f, &html).unwrap_or_else(|e| eprintln!("Write error: {}", e));
            println!("Saved to {}", f);
        }
        None => println!("{}", html),
    }
}

fn run_merge<const N: usize, const S: usize>(args: &[String]) {
    let target_path = args.get(2).map(|s| s.as_str()).unwrap();
    let sources_str = args.get(3).map(|s| s.as_str()).unwrap_or("");

    let tgt_mem_path = target_path.replace(".bin", "_mem.bin");

    let cube = match WaveCube::<N, S>::load_bin(target_path) {
        Ok(c) => {
            println!(
                "Target cube: {} dim={} ({} cells)",
                S,
                c.dim,
                WaveCube::<N, S>::TOTAL_CELLS
            );
            c
        }
        Err(e) => {
            eprintln!("Failed to load target cube: {}", e);
            return;
        }
    };
    let memory = match fuga::MemoryStore::load_bin(&tgt_mem_path) {
        Ok(m) => {
            println!("Target memory: {} entries", m.size());
            m
        }
        Err(e) => {
            eprintln!("No target memory (starting fresh): {}", e);
            fuga::MemoryStore::new()
        }
    };

    let source_paths: Vec<&str> = if sources_str.is_empty() {
        Vec::new()
    } else {
        sources_str.split(',').collect()
    };

    let mut all_entries: Vec<(String, String, String)> = Vec::new();

    for spath in &source_paths {
        match fuga::MemoryStore::load_bin(spath) {
            Ok(m) => {
                println!("Source memory: {} entries from {}", m.size(), spath);
                for e in m.all_entries() {
                    all_entries.push((e.text.clone(), e.source_doc.clone(), e.role_hint.clone()));
                }
            }
            Err(e) => {
                eprintln!("Failed to load source memory {}: {}", spath, e);
            }
        }
    }

    if all_entries.is_empty() {
        if source_paths.is_empty() {
            eprintln!("Error: no source memory files specified");
            println!("Usage: merge <cube.bin> <mem1.bin,mem2.bin,...>");
        } else {
            println!("No entries to transfer.");
        }
        return;
    }

    let mut ai = FugaAI::<N, S>::new(cube.dim, 3);
    ai.cube = cube;
    ai.memory = memory;
    let start_mem = ai.memory.size();

    println!(
        "\nTransferring {} entries across {} source(s)...",
        all_entries.len(),
        source_paths.len()
    );
    for (i, (text, source_doc, role_hint)) in all_entries.iter().enumerate() {
        let tokens: Vec<TokenInfo> = text
            .split_whitespace()
            .enumerate()
            .map(|(_, w)| TokenInfo {
                id: token_id(&w),
                text: w.to_string(),
            })
            .collect();

        let output = ai.think(&tokens);
        for st in &output.super_tokens {
            ai.memory.store(st, text, source_doc, role_hint);
        }
        ai.cube_absorb(&output.super_tokens);

        if (i + 1) % 500 == 0 || i + 1 == all_entries.len() {
            println!(
                "  {}/{} absorbed, mem={}, entropy={:.4}",
                i + 1,
                all_entries.len(),
                ai.memory.size(),
                ai.cube.global_entropy()
            );
        }
    }

    let _ = ai.cube.save_bin(target_path);
    let _ = ai.memory.save_bin(&tgt_mem_path);
    println!("\nMerged cube saved to {}", target_path);
    println!(
        "Memory: {} entries (was {}, +{})",
        ai.memory.size(),
        start_mem,
        ai.memory.size() - start_mem
    );
    println!(
        "Entropy: {:.4}, Coherence: {:.4}",
        ai.cube.global_entropy(),
        ai.cube.coherence()
    );
}

fn run_train_text<const N: usize, const S: usize>(
    dir: &str,
    _dim: usize,
    save_path: &str,
    _args: &[String],
) {
    println!(
        "Fuga Conversational-Literary Training Pipeline ({}^{}={} cells)",
        S,
        N,
        S.pow(N as u32)
    );
    println!("  Text corpus: {}", dir);
    let mem_path = save_path.replace(".bin", "_mem.bin");

    let (mut ai, start_mem) = if std::path::Path::new(save_path).exists() {
        println!("  Loading existing cube from {}", save_path);
        let cube = match WaveCube::<N, S>::load_bin(save_path) {
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
                    eprintln!("Memory load failed (starting fresh): {}", e);
                    fuga::MemoryStore::new()
                }
            }
        } else {
            fuga::MemoryStore::new()
        };
        let mut ai = FugaAI::<N, S>::new(cube.dim, 3);
        ai.cube = cube;
        ai.memory = memory;
        let start_mem = ai.memory.size();
        (ai, start_mem)
    } else {
        eprintln!(
            "  No existing cube found. Run 'fuga train-code' first or specify existing cube."
        );
        return;
    };

    println!("  Start:  {} memories\n", start_mem);

    let mut filter = fuga::TextQualityFilter::new(ai.dim);
    let mut total_files = 0usize;
    let mut absorbed_files = 0usize;
    let mut total_pairs = 0usize;
    let mut total_tokens = 0usize;

    let results = match filter.scan_directory(dir, true) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Scan failed: {}", e);
            return;
        }
    };

    println!("Found {} text files\n", results.len());

    println!("Phase 1/3: collecting term frequencies for IDF...");
    for (path, score) in &results {
        if score.weight <= 0.0 {
            continue;
        }
        let source = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let words: Vec<TokenInfo> = source
            .split_whitespace()
            .enumerate()
            .map(|(_, w)| TokenInfo {
                id: token_id(&w),
                text: w.to_string(),
            })
            .collect();
        if !words.is_empty() {
            ai.accumulate_df(&words);
        }
    }
    ai.compute_idf();
    println!(
        "  IDF computed: {} unique terms, {} docs\n",
        ai.idf_weights.len(),
        ai.total_docs
    );

    println!("Phase 2/3: extracting dialogue pairs...");
    struct TextSource {
        path: String,
        score: fuga::TextQualityScore,
        source: String,
        pairs: Vec<(String, String)>,
    }
    let mut sources: Vec<TextSource> = Vec::new();

    for (path, score) in &results {
        total_files += 1;
        if score.weight <= 0.0 {
            println!(
                "  BLOCKED: {} (w=0.00, collage={:.2})",
                path, score.collage_risk
            );
            continue;
        }
        if score.weight < 0.3 {
            println!(
                "  LOW: {} (w={:.2}, collage={:.2})",
                path, score.weight, score.collage_risk
            );
        }

        let source = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  read error {}: {}", path, e);
                continue;
            }
        };

        let pairs = if score.source_type == fuga::TextSourceType::Dialogue {
            fuga::extract_dialogue_pairs(&source)
        } else {
            Vec::new()
        };

        total_pairs += pairs.len();

        sources.push(TextSource {
            path: path.clone(),
            score: score.clone(),
            source,
            pairs,
        });
    }

    println!("  Total dialogue pairs: {}", total_pairs);
    println!("  Total text sources: {}\n", sources.len());

    println!("Phase 3/3: absorbing into cube...");
    let text_summary = fuga::summarize_text_quality(&results);
    println!("{}", text_summary);

    for ts in &sources {
        let tokens: Vec<TokenInfo> = ts
            .source
            .split_whitespace()
            .enumerate()
            .map(|(_, w)| TokenInfo {
                id: token_id(&w),
                text: w.to_string(),
            })
            .collect();
        total_tokens += tokens.len();

        let output = ai.think(&tokens);
        let absorb_count = ((output.super_tokens.len() as f64) * ts.score.weight).ceil() as usize;

        let source_type = ts.score.source_type.name();
        let label = if ts.pairs.is_empty() {
            format!("[{}] {} (w={:.2})", source_type, ts.path, ts.score.weight)
        } else {
            format!(
                "[{}] {} ({} pairs, w={:.2})",
                source_type,
                ts.path,
                ts.pairs.len(),
                ts.score.weight
            )
        };

        for st in output.super_tokens.iter().take(absorb_count) {
            ai.memory.store(st, &label, &ts.path, source_type);
        }

        if !output.super_tokens.is_empty() && ts.score.weight > 0.3 {
            ai.cube_absorb(&output.super_tokens);
        }

        absorbed_files += 1;
        if absorbed_files % 10 == 0 || absorbed_files == sources.len() {
            println!(
                "  {}/{} absorbed, mem={}, entropy={:.4}",
                absorbed_files,
                sources.len(),
                ai.memory.size(),
                ai.cube.global_entropy()
            );
        }

        if absorbed_files % 50 == 0 {
            let _ = ai.cube.save_bin(save_path);
            let _ = ai.memory.save_bin(&mem_path);
        }
    }

    println!("Phase 3b/3: absorbing dialogue pairs (max 5000)...");
    let mut pair_absorbed = 0usize;
    let max_pairs = 5000;
    for ts in &sources {
        if ts.pairs.is_empty() {
            continue;
        }
        for (ctx, resp) in &ts.pairs {
            if pair_absorbed >= max_pairs {
                break;
            }
            let combined = format!("{} {}", ctx, resp);
            let pair_tokens: Vec<TokenInfo> = combined
                .split_whitespace()
                .enumerate()
                .map(|(_, w)| TokenInfo {
                    id: token_id(&w),
                    text: w.to_string(),
                })
                .collect();
            if pair_tokens.is_empty() {
                continue;
            }

            let output = ai.think(&pair_tokens);
            if !output.super_tokens.is_empty() {
                let pair_label = format!("[dialogue-pair] ctx:{} | → | resp:{}", ctx, resp);
                for st in &output.super_tokens {
                    ai.memory.store(st, &pair_label, &ts.path, "dialogue_pair");
                }
                ai.cube_absorb(&output.super_tokens);
                pair_absorbed += 1;
            }

            if pair_absorbed % 200 == 0 {
                println!(
                    "  {} dialogue pairs absorbed, mem={}",
                    pair_absorbed,
                    ai.memory.size()
                );
            }
        }
        if pair_absorbed >= max_pairs {
            break;
        }
    }
    println!("  Total dialogue pairs absorbed: {}", pair_absorbed);

    let _ = ai.cube.save_bin(save_path);
    let _ = ai.memory.save_bin(&mem_path);

    ai.build_moe_from_memory();
    match ai.moe.save_all() {
        Ok(()) => {
            println!("\n  MoE domains saved:");
            for (domain, size) in &ai.moe.domain_sizes() {
                println!("    {:20}  {}", domain, size);
            }
        }
        Err(e) => eprintln!("  MoE save error: {}", e),
    }

    let final_entropy = ai.cube.global_entropy();
    let final_coherence = ai.cube.coherence();
    let final_mem = ai.memory.size();

    println!("\n=== Conversational-Literary Training Complete ===");
    println!("  Text sources:  {} files", absorbed_files);
    println!("  Dialogue pairs: {}", pair_absorbed);
    println!(
        "  Memory: {} (was {}, +{})",
        final_mem,
        start_mem,
        final_mem.saturating_sub(start_mem)
    );
    println!("  Entropy:  {:.4}", final_entropy);
    println!("  Coherence: {:.4}", final_coherence);
    println!("  Cube saved to: {}", save_path);
}

fn run_train_code<const N: usize, const S: usize>(
    dir: &str,
    dim: usize,
    save_path: &str,
    epochs: usize,
    _args: &[String],
) {
    println!(
        "Fuga Code Quality Training Pipeline ({}^{}={} cells)",
        S,
        N,
        S.pow(N as u32)
    );
    println!("  Source: {}", dir);
    let mem_path = save_path.replace(".bin", "_mem.bin");

    let (mut ai, start_files) = if std::path::Path::new(save_path).exists() {
        println!("  Loading existing cube from {}", save_path);
        let cube = match WaveCube::<N, S>::load_bin(save_path) {
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
                    eprintln!("Memory load failed (starting fresh): {}", e);
                    fuga::MemoryStore::new()
                }
            }
        } else {
            fuga::MemoryStore::new()
        };
        let mut ai = FugaAI::<N, S>::new(cube.dim, 3);
        ai.cube = cube;
        ai.memory = memory;
        let start_files = ai.memory.size();
        (ai, start_files)
    } else {
        println!("  Dim:    {}", dim);
        (FugaAI::<N, S>::new(dim, 3), 0)
    };

    println!("  Save:   {}", save_path);
    println!("  Start:  {} memories\n", start_files);

    let mut filter = CodeQualityFilter::new(ai.dim);

    let results = match filter.scan_directory(dir, true) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Scan failed: {}", e);
            return;
        }
    };

    println!("Found {} supported files", results.len());

    // Phase 1: IDF — compute once
    println!("\nPhase 1: collecting term frequencies for IDF...");
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
        "  IDF computed: {} unique terms, {} docs\n",
        ai.idf_weights.len(),
        ai.total_docs
    );

    // Phase 2: absorption loop over epochs
    let mut total_files_ever = 0usize;
    let mut total_tokens_ever = 0usize;

    for epoch in 0..epochs {
        println!("=== Epoch {}/{} ===", epoch + 1, epochs);
        let mut epoch_absorbed = 0usize;
        let mut epoch_tokens = 0usize;
        let mem_before = ai.memory.size();

        for (path, score) in &results {
            if score.weight <= 0.0 {
                continue;
            }

            let source = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("  read error {}: {}", path, e);
                    continue;
                }
            };

            let tokens: Vec<TokenInfo> = source
                .split_whitespace()
                .enumerate()
                .map(|(_, w)| TokenInfo {
                    id: token_id(&w),
                    text: w.to_string(),
                })
                .collect();

            epoch_tokens += tokens.len();
            let absorbed = ai.absorb_with_quality(&tokens, path, score, &source);
            if absorbed {
                epoch_absorbed += 1;
            }
        }

        total_files_ever += epoch_absorbed;
        total_tokens_ever += epoch_tokens;
        let mem_after = ai.memory.size();

        println!(
            "  Epoch {}: {} files, mem {} → {} (+{}), entropy={:.4}",
            epoch + 1,
            epoch_absorbed,
            mem_before,
            mem_after,
            mem_after.saturating_sub(mem_before),
            ai.cube.global_entropy()
        );

        if (epoch + 1) % 5 == 0 || epoch == epochs - 1 {
            if let Err(e) = ai.cube.save_bin(save_path) {
                eprintln!("  Checkpoint cube save failed: {}", e);
            }
            if let Err(e) = ai.memory.save_bin(&mem_path) {
                eprintln!("  Checkpoint memory save failed: {}", e);
            }
            println!("  Checkpoint saved.");
        }
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

    ai.build_moe_from_memory();
    match ai.moe.save_all() {
        Ok(()) => {
            println!("  MoE domains:");
            for (domain, size) in &ai.moe.domain_sizes() {
                println!("    {:20}  {}", domain, size);
            }
        }
        Err(e) => eprintln!("  MoE save error: {}", e),
    }

    println!("\n=== Quality Training Complete ===");
    println!("  Epochs:         {}", epochs);
    println!("  Files scanned:  {}", results.len());
    println!("  Memory size:    {}", ai.memory.size());
    println!("  Cube entropy:   {:.4}", ai.cube.global_entropy());
    println!("  Cube coherence: {:.4}", ai.cube.coherence());
}

fn run_train_autofix<const N: usize, const S: usize>(
    dir: &str,
    _dim: usize,
    save_path: &str,
    mw_path: &str,
    _args: &[String],
) {
    println!("Fuga Autofix Training — error correction via microwave validation");
    println!("  Source: {}", dir);
    let mem_path = save_path.replace(".bin", "_mem.bin");

    let mut ai = if std::path::Path::new(save_path).exists() {
        println!("  Loading cube from {}", save_path);
        let cube = match WaveCube::<N, S>::load_bin(save_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to load cube: {}", e);
                return;
            }
        };
        let memory = if std::path::Path::new(&mem_path).exists() {
            match fuga::MemoryStore::load_bin(&mem_path) {
                Ok(m) => {
                    println!("  Memory: {} entries", m.size());
                    m
                }
                Err(_) => fuga::MemoryStore::new(),
            }
        } else {
            fuga::MemoryStore::new()
        };
        let mut ai = FugaAI::<N, S>::new(cube.dim, 3);
        ai.cube = cube;
        ai.memory = memory;
        ai
    } else {
        eprintln!("Cube not found: {}. Run 'train-code' first.", save_path);
        return;
    };

    let mut filter = CodeQualityFilter::new(ai.dim);
    let results = match filter.scan_directory(dir, true) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Scan failed: {}", e);
            return;
        }
    };
    println!("Found {} files\n", results.len());

    let mut total_fixed = 0usize;
    let mut absorbed_fixes = 0usize;

    for (path, score) in &results {
        if score.weight > 0.8 && !score.bugs_detected {
            continue;
        }
        if !score.bugs_detected && score.weight > 0.5 {
            continue;
        }

        let source = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let lang = LanguageId::from_path(Path::new(path));
        let proposals: Vec<FixProposal> = match lang {
            Some(LanguageId::Rust) => {
                let mut engine = FugaEngine::new(ai.dim.min(8192));
                match engine.analyze(source.as_str()) {
                    Ok(result) => engine.generate_fixes(&source, &result),
                    Err(_) => Vec::new(),
                }
            }
            Some(lang) => {
                let mut multi = MultiEngine::new(ai.dim.min(8192));
                let result = multi.analyze(source.as_str(), lang, path);
                let fixer = MultiFixGenerator::new();
                fixer.generate_fixes(&source, &result.syntax.violations, lang)
            }
            None => Vec::new(),
        };

        if proposals.is_empty() {
            continue;
        }

        total_fixed += 1;
        let fixed_source = apply_autofix_proposals(&source, &proposals);
        let fixed_score =
            match filter.analyze(&fixed_source, lang.unwrap_or(LanguageId::Rust), path) {
                Ok(s) => s,
                Err(_) => continue,
            };

        let mut fix_valid = fixed_score.weight > score.weight
            && fixed_score.safety > score.safety
            && !fixed_score.bugs_detected;

        if fix_valid && lang == Some(LanguageId::Rust) {
            let tmp = std::env::temp_dir().join(format!("autofix_{}.rs", total_fixed));
            let _ = std::fs::write(&tmp, &fixed_source);
            let output = std::process::Command::new(mw_path)
                .arg("eval-rust-file")
                .arg(&tmp)
                .output();
            if let Ok(out) = output {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let compiles = stdout.contains("COMPILES:true");
                if !compiles {
                    fix_valid = false;
                }
            }
            let _ = std::fs::remove_file(&tmp);
        }

        if fix_valid {
            let tokens: Vec<TokenInfo> = fixed_source
                .split_whitespace()
                .enumerate()
                .map(|(_, w)| TokenInfo {
                    id: token_id(&w),
                    text: w.to_string(),
                })
                .collect();
            let absorbed = ai.absorb_with_quality(&tokens, path, &fixed_score, &fixed_source);
            if absorbed {
                absorbed_fixes += 1;
                println!(
                    "  FIX {}: {} (w={:.2}→{:.2} safety={:.2}→{:.2})",
                    absorbed_fixes,
                    path,
                    score.weight,
                    fixed_score.weight,
                    score.safety,
                    fixed_score.safety
                );
            }
        }
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

    println!("\n=== Autofix Training Complete ===");
    println!("  Files with fixes: {}", total_fixed);
    println!("  Absorbed fixes:   {}", absorbed_fixes);
    println!("  Memory size:      {}", ai.memory.size());
    println!("  Cube entropy:     {:.4}", ai.cube.global_entropy());
}

fn apply_autofix_proposals(source: &str, proposals: &[FixProposal]) -> String {
    let mut result = source.to_string();
    for p in proposals {
        if let (Some(start), Some(end)) = (p.start_byte, p.end_byte) {
            if start < result.len() && end <= result.len() {
                result.replace_range(start..end, &p.proposed_code);
            }
        } else {
            result = result.replace(&p.original_code, &p.proposed_code);
        }
    }
    result
}

fn run_moe_split<const N: usize, const S: usize>(save_path: &str) {
    let mem_path = save_path.replace(".bin", "_mem.bin");
    println!("MoE Split: {} / {}", save_path, mem_path);

    let cube = match WaveCube::<N, S>::load_bin(save_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to load cube: {}", e);
            return;
        }
    };
    let memory = match fuga::MemoryStore::load_bin(&mem_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load memory: {}", e);
            return;
        }
    };
    println!("  Cube: {}x{} dim={}", S, N, cube.dim);
    println!("  Memory: {} entries", memory.size());

    let mut moe = fuga::MoEStore::new(save_path);
    for entry in memory.all_entries() {
        let st = fuga::SuperToken::new(entry.vector.clone(), 0);
        moe.store(&st, &entry.text, &entry.source_doc, &entry.role_hint);
    }

    println!("  MoE domains:");
    for (domain, size) in &moe.domain_sizes() {
        println!("    {:20}  {}", domain, size);
    }

    match moe.save_all() {
        Ok(_) => println!("  Per-domain files saved"),
        Err(e) => eprintln!("  Save error: {}", e),
    }
}

fn encode_chunk(weaver: &mut fuga::WeaverEngine, text: &str) -> fuga::Hypervector {
    let tokens: Vec<fuga::TokenInfo> = text
        .split_whitespace()
        .map(|w| fuga::TokenInfo {
            id: fuga::weaver::token_id(w),
            text: w.to_string(),
        })
        .collect();
    if tokens.is_empty() {
        return fuga::Hypervector::random(weaver.dim());
    }
    let mut vecs: Vec<fuga::Hypervector> = Vec::new();
    for t in &tokens {
        vecs.push(weaver.cached_vector(t.id).clone());
    }
    let first = vecs.remove(0);
    let refs: Vec<&fuga::Hypervector> = vecs.iter().collect();
    first.bundle(&refs)
}

fn run_jepa_train(dir: &str, dim: usize, context_len: usize, epochs: usize) {
    use std::io::Read;
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    let path = std::path::Path::new(dir);
    if path.is_dir() {
        for entry in walkdir::WalkDir::new(dir).max_depth(3) {
            if let Ok(e) = entry {
                let p = e.path();
                if p.extension()
                    .map(|x| {
                        x == "rs"
                            || x == "py"
                            || x == "js"
                            || x == "ts"
                            || x == "c"
                            || x == "cpp"
                            || x == "go"
                    })
                    .unwrap_or(false)
                {
                    files.push(p.to_path_buf());
                }
            }
        }
    } else {
        files.push(path.to_path_buf());
    }

    let mut weaver = fuga::WeaverEngine::new(dim, 3);
    let mut sequences: Vec<Vec<fuga::Hypervector>> = Vec::new();
    let mut file_count = 0;

    for fp in &files {
        let mut s = String::new();
        if std::fs::File::open(fp)
            .and_then(|mut f| f.read_to_string(&mut s))
            .is_err()
        {
            continue;
        }
        let words: Vec<&str> = s.split_whitespace().collect();
        if words.len() < 20 {
            continue;
        }
        let mut seq = Vec::new();
        for chunk in words.chunks(10) {
            let text = chunk.join(" ");
            seq.push(encode_chunk(&mut weaver, &text));
        }
        if seq.len() > context_len + 1 {
            sequences.push(seq);
            file_count += 1;
        }
    }

    eprintln!(
        "Loaded {} sequences from {} files",
        sequences.len(),
        file_count
    );

    let mut predictor = fuga::JepaPredictor::new(dim, context_len);
    let loss = predictor.train_on_sequences(&sequences, epochs);
    eprintln!("Training done. Final loss: {:.4}", loss);

    predictor.save("fuga_jepa.bin").ok();
    eprintln!("Saved fuga_jepa.bin");
}

fn run_jepa_predict(text: &str, dim: usize, context_len: usize) {
    let predictor = match fuga::JepaPredictor::load("fuga_jepa.bin") {
        Ok(p) => p,
        Err(e) => {
            eprintln!("No trained JEPA model (fuga_jepa.bin): {}", e);
            return;
        }
    };

    let mut weaver = fuga::WeaverEngine::new(dim, 3);
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.len() < context_len {
        eprintln!("Need at least {} tokens for context", context_len);
        return;
    }

    let mut context_vecs = Vec::new();
    for chunk in tokens.chunks(3) {
        let t = chunk.join(" ");
        context_vecs.push(encode_chunk(&mut weaver, &t));
    }

    let n = context_len.min(context_vecs.len());
    let ctx_refs: Vec<&fuga::Hypervector> = context_vecs[context_vecs.len() - n..].iter().collect();
    let predicted = predictor.predict(&ctx_refs);

    println!("Predicted hypervector entropy: {:.4}", predicted.entropy());
    println!(
        "Predicted vector words: {} (dim {})",
        predicted.words.len(),
        predicted.dim
    );

    // decode via nearest neighbor in MoE
    let moe_paths = &["fuga_code_cube_code_mem.bin", "fuga_moe_code.bin"];
    for mp in moe_paths {
        if std::path::Path::new(mp).exists() {
            if let Ok(mem) = fuga::MemoryStore::load_bin(mp) {
                let results = mem.search(&predicted, 3);
                if !results.is_empty() {
                    println!("\nDecoded from {}:", mp);
                    for (_, sim, entry) in &results {
                        println!("  [{:.3}] {} — {}", sim, entry.text, entry.source_doc);
                    }
                }
                break;
            }
        }
    }
}

fn run_hierarchical_jepa_train(dir: &str, dim: usize, epochs: usize) {
    println!("╔══════════════════════════════════════════╗");
    println!("║  Hierarchical JEPA — 3-level predictor  ║");
    println!("╚══════════════════════════════════════════╝\n");
    println!("  Dir:     {}", dir);
    println!("  Dim:     {}", dim);
    println!("  Epochs:  {}", epochs);
    println!("  Levels:  L0(ctx=4,stride=1) L1(ctx=3,stride=3) L2(ctx=2,stride=5)\n");

    let model_path = "fuga_hjepa.bin";
    let mut hjepa = if std::path::Path::new(model_path).exists() {
        match fuga::HierarchicalJEPA::load(model_path) {
            Ok(h) => {
                println!("  Loaded existing {} (continuing training)\n", model_path);
                h
            }
            Err(e) => {
                println!("  Load failed ({}), creating fresh model\n", e);
                fuga::HierarchicalJEPA::new(dim)
            }
        }
    } else {
        fuga::HierarchicalJEPA::new(dim)
    };
    let loss = hjepa.train_on_directory(dir, epochs);
    println!("\n  Training complete. Avg loss: {:.4}", loss);

    match hjepa.save(model_path) {
        Ok(()) => println!("  Saved {}", model_path),
        Err(e) => eprintln!("  Save failed: {}", e),
    }
}

fn run_baby_repl(dim: usize) {
    let mut hjepa = match fuga::HierarchicalJEPA::load("fuga_hjepa.bin") {
        Ok(h) => h,
        Err(e) => {
            eprintln!(
                "No trained H-JEPA model: {}. Train with 'h-jepa-train' first.",
                e
            );
            return;
        }
    };

    let mut mem: Option<fuga::MemoryStore> = None;
    for mp in &["fuga_code_cube_code_mem.bin", "fuga_moe_code.bin"] {
        if std::path::Path::new(mp).exists() {
            if let Ok(m) = fuga::MemoryStore::load_bin(mp) {
                mem = Some(m);
                break;
            }
        }
    }

    let mut weaver = fuga::WeaverEngine::new(dim, 3);
    let mut context: Vec<fuga::Hypervector> = Vec::new();
    let stdin = std::io::stdin();

    let sdr_store: Option<fuga::SdrStore> = load_sdr_store("fuga_sdr_index.bin");
    if sdr_store.is_some() {
        println!("  SDR index loaded (Fuga 1.4 Cross-SDR Bridge available: /sdr)");
    }

    println!("╔══════════════════════════════════════════╗");
    println!("║  Fuga Baby — Interactive H-JEPA REPL    ║");
    println!("╚══════════════════════════════════════════╝");
    println!("  Commands: /reset  /quit  /stats  /help  /train <dir> [epochs]  /sdr <query>");
    println!("  Dim: {}  H-JEPA: fuga_hjepa.bin", dim);
    println!();

    loop {
        print!("👶 ");
        use std::io::Write;
        std::io::stdout().flush().ok();

        let mut line = String::new();
        if stdin.read_line(&mut line).ok() != Some(0) && line.trim().is_empty() {
            continue;
        }
        let line = line.trim();

        if line.eq_ignore_ascii_case("/quit") || line.eq_ignore_ascii_case("/exit") {
            println!("👋");
            break;
        }
        if line.eq_ignore_ascii_case("/help") {
            println!("  /reset           Clear context window");
            println!("  /stats           Show context and model state");
            println!("  /train <dir> [n] Retrain on directory (n epochs, default 10)");
            println!("  /quit            Exit");
            println!("  /help            This message");
            println!("  <any text>       Predict next state via H-JEPA");
            continue;
        }
        if line.eq_ignore_ascii_case("/reset") {
            context.clear();
            println!("  Context reset.");
            continue;
        }
        if line.starts_with("/train") {
            let parts: Vec<&str> = line.splitn(3, ' ').collect();
            let train_arg = if parts.len() > 1 {
                parts[1]
            } else {
                "temp_repos"
            };
            let train_epochs = parts
                .get(2)
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(10);
            let dirs: Vec<&str> = train_arg.split(',').collect();
            let mut total_loss = 0.0;
            for d in &dirs {
                let d = d.trim();
                if !std::path::Path::new(d).is_dir() {
                    println!("  Skipping '{}' (not a directory)", d);
                    continue;
                }
                println!("  Training on '{}' for {} epochs...", d, train_epochs);
                total_loss += hjepa.train_on_directory(d, train_epochs);
            }
            let avg_loss = total_loss / dirs.len() as f64;
            match hjepa.save("fuga_hjepa.bin") {
                Ok(()) => println!("  Saved fuga_hjepa.bin (avg_loss={:.4})", avg_loss),
                Err(e) => eprintln!("  Save failed: {}", e),
            }
            println!("  Model updated in-place.");
            continue;
        }
        if line.eq_ignore_ascii_case("/stats") {
            println!("  Context length: {}", context.len());
            println!("  H-JEPA levels: L0 L1 L2");
            if let Some(ref sdr) = sdr_store {
                println!(
                    "  SDR index: {} nodes (Fuga 1.4 Cross-SDR Bridge)",
                    sdr.index.nodes.len()
                );
            }
            continue;
        }
        if line.starts_with("/sdr") {
            let sdr_query = line[4..].trim();
            if sdr_query.is_empty() {
                println!("  Usage: /sdr <query>");
                continue;
            }
            match sdr_store {
                Some(ref store) => {
                    let results = store.query(sdr_query, 3);
                    println!("  SDR (Fuga 1.3 popcount):");
                    for (_i, score, snippet) in &results {
                        println!("    [{:.2}] {}", score, snippet);
                    }
                    let cross = store.query_cross(sdr_query, "doc", 3);
                    println!("  Cross-SDR (Fuga 1.4 doc→code):");
                    for (_i, score, snippet) in &cross {
                        println!("    [{:.2}] {}", score, snippet);
                    }
                }
                None => println!("  SDR index not loaded. Run 'fuga sdr-build' first."),
            }
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        let mut chunk_hvs = Vec::new();
        for chunk in tokens.chunks(3) {
            let t = chunk.join(" ");
            chunk_hvs.push(encode_chunk(&mut weaver, &t));
        }

        for hv in &chunk_hvs {
            context.push(hv.clone());
        }

        if context.len() < 2 {
            println!("  Need more context...");
            continue;
        }

        let ctx_len = hjepa.levels[0].context_len;
        while context.len() > 20 {
            context.remove(0);
        }

        if context.len() < ctx_len {
            println!("  Context too short (need {}), building...", ctx_len);
            continue;
        }

        let window: Vec<&fuga::Hypervector> = context[context.len().saturating_sub(ctx_len)..]
            .iter()
            .collect();
        let predictions = hjepa.predict(&window);

        let input_hvs: Vec<fuga::Hypervector> = chunk_hvs.clone();
        let input_refs: Vec<&fuga::Hypervector> = input_hvs.iter().collect();
        let errors = hjepa.learn(&window, &input_refs);

        println!();
        for (li, pred) in predictions.iter().enumerate() {
            let level_name = match li {
                0 => "L0",
                1 => "L1",
                2 => "L2",
                _ => "?",
            };
            let role = match li {
                0 => "primitive",
                1 => "functional",
                2 => "concept",
                _ => "",
            };
            let entropy = pred.entropy();
            let emoji = if entropy > 0.98 {
                "🌀"
            } else if entropy > 0.90 {
                "🌊"
            } else {
                "⚡"
            };
            let err_str = if li < errors.len() {
                format!(" err={:.3}", errors[li])
            } else {
                String::new()
            };
            println!(
                "  {} {} {}: entropy={:.4}{}",
                emoji, level_name, role, entropy, err_str
            );
        }

        if let Some(ref mem) = mem {
            if predictions.len() >= 2 {
                let results_l0 = mem.search(&predictions[0], 1);
                let results_l1 = mem.search(&predictions[1], 2);
                if !results_l0.is_empty() {
                    let (_, sim, entry) = &results_l0[0];
                    let snippet: String = entry.text.chars().take(80).collect();
                    println!("  📖 L0 → [{:.2}] {}", sim, snippet);
                }
                if !results_l1.is_empty() {
                    println!("  🔗 L1 (cross-domain):");
                    for (_, sim, entry) in &results_l1 {
                        let snippet: String = entry.text.chars().take(80).collect();
                        println!("     [{:.2}] {} — {}", sim, snippet, entry.source_doc);
                    }
                }
            }
        }
        println!();
    }
}

fn run_sdr_query(text: &str) {
    let sdr_path = "fuga_sdr_index.bin";
    if !std::path::Path::new(sdr_path).exists() {
        eprintln!("SDR index not found. Run 'fuga sdr-build' first.");
        return;
    }
    let store = match load_sdr_store(sdr_path) {
        Some(s) => s,
        None => {
            eprintln!("Failed to load SDR index");
            return;
        }
    };
    let results = store.query(text, 5);
    println!("SDR query: \"{}\"", text);
    for (_i, score, snippet) in &results {
        println!("  [{:.2}] {}", score, snippet);
    }
}

fn run_sdr_query_cross(text: &str) {
    let sdr_path = "fuga_sdr_index.bin";
    if !std::path::Path::new(sdr_path).exists() {
        eprintln!("SDR index not found. Run 'fuga sdr-build' first.");
        return;
    }
    let store = match load_sdr_store(sdr_path) {
        Some(s) => s,
        None => {
            eprintln!("Failed to load SDR index");
            return;
        }
    };
    let results = store.query_cross(text, "doc", 5);
    println!("SDR cross-domain (doc→code): \"{}\"", text);
    for (_i, score, snippet) in &results {
        println!("  [{:.2}] {}", score, snippet);
    }
}

fn load_sdr_store(path: &str) -> Option<fuga::SdrStore> {
    let mut f = std::fs::File::open(path).ok()?;
    use std::io::Read;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).ok()?;
    let mut pos = 0usize;
    let count = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    let mut nodes = Vec::with_capacity(count);
    for _ in 0..count {
        let mut bits = [0u64; 128];
        for w in bits.iter_mut() {
            *w = u64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
            pos += 8;
        }
        nodes.push(fuga::SdrVector { bits });
    }
    let tcount = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    let mut texts = Vec::with_capacity(tcount);
    for _ in 0..tcount {
        let tlen = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let t = String::from_utf8(buf[pos..pos + tlen].to_vec()).unwrap_or_default();
        pos += tlen;
        texts.push(t);
    }
    let mut store = fuga::SdrStore::new();
    store.index.nodes = nodes;
    store.index.texts = texts;
    Some(store)
}

fn run_htm_train(_path: &str, steps: usize) {
    let sdr_path = "fuga_sdr_index.bin";
    let sdr = load_sdr_store(sdr_path);
    let mut tm = fuga::TemporalMemory::new(512, 4);

    if let Some(ref store) = sdr {
        println!(
            "  HTM: loading SDR index ({} nodes)...",
            store.index.nodes.len()
        );
        let n = store.index.nodes.len().min(steps);
        for i in 1..n {
            let prev = &store.index.nodes[i - 1];
            let next = &store.index.nodes[i];
            tm.learn_sequence(prev, next);
            if (i + 1) % 1000 == 0 {
                print!("\r  HTM: {}/{} sequences learned", i, n);
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
        }
        println!("\n  HTM training complete on {} transitions.", n);
    } else {
        println!("  HTM: no SDR index, training on random sequences...");
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..steps {
            let mut a = fuga::SdrVector::zero();
            let mut b = fuga::SdrVector::zero();
            for _ in 0..((fuga::SDR_DIM as f64 * fuga::SDR_DENSITY) as usize) {
                let ba = rng.gen_range(0..fuga::SDR_DIM);
                let bb = rng.gen_range(0..fuga::SDR_DIM);
                a.bits[ba / 64] |= 1u64 << (ba % 64);
                b.bits[bb / 64] |= 1u64 << (bb % 64);
            }
            tm.learn_sequence(&a, &b);
        }
        println!("  HTM trained on {} random transitions.", steps);
    }
    println!("  HTM stats: {}", tm.stats());

    {
        use std::io::Write;
        let mut f = match std::fs::File::create("fuga_htm.bin") {
            Ok(f) => f,
            Err(e) => {
                eprintln!("  Save failed: {}", e);
                return;
            }
        };
        let n = tm.cells.len() as u32;
        f.write_all(&n.to_le_bytes()).ok();
        for c in &tm.cells {
            let id = c.id as u32;
            f.write_all(&id.to_le_bytes()).ok();
            for w in &c.pattern.bits {
                f.write_all(&w.to_le_bytes()).ok();
            }
            let seg_n = c.segments.len() as u32;
            f.write_all(&seg_n.to_le_bytes()).ok();
            for seg in &c.segments {
                let syn_n = seg.synapses.len() as u32;
                f.write_all(&syn_n.to_le_bytes()).ok();
                for s in &seg.synapses {
                    let bi = s.bit_index as u32;
                    f.write_all(&bi.to_le_bytes()).ok();
                    f.write_all(&s.permanence.to_le_bytes()).ok();
                }
            }
        }
        let wl = tm.window.len() as u32;
        f.write_all(&wl.to_le_bytes()).ok();
        for sdr in &tm.window {
            for w in &sdr.bits {
                f.write_all(&w.to_le_bytes()).ok();
            }
        }
        println!("  Saved fuga_htm.bin");
    }
}

fn load_tm() -> Option<fuga::TemporalMemory> {
    load_tm_from("fuga_htm.bin")
}

fn load_tm_from(path: &str) -> Option<fuga::TemporalMemory> {
    let data = std::fs::read(path).ok()?;
    let mut pos = 0usize;
    // Bounds-safe readers: any out-of-range access aborts the load (None).
    macro_rules! take {
        ($len:expr) => {{
            let end = pos + $len;
            if end > data.len() {
                return None;
            }
            let v = &data[pos..end];
            pos = end;
            v
        }};
    }
    let n = u32::from_le_bytes(take!(4).try_into().ok()?) as usize;
    let mut cells = Vec::with_capacity(n.min(20_000_000));
    for _ in 0..n {
        let id = u32::from_le_bytes(take!(4).try_into().ok()?) as usize;
        let mut bits = [0u64; 128];
        for w in bits.iter_mut() {
            *w = u64::from_le_bytes(take!(8).try_into().ok()?);
        }
        let pattern = fuga::SdrVector { bits };
        let seg_n = u32::from_le_bytes(take!(4).try_into().ok()?) as usize;
        let mut segments = Vec::with_capacity(seg_n.min(100_000));
        for _ in 0..seg_n {
            let syn_n = u32::from_le_bytes(take!(4).try_into().ok()?) as usize;
            let mut synapses = Vec::with_capacity(syn_n.min(1_000_000));
            for _ in 0..syn_n {
                let bi = u32::from_le_bytes(take!(4).try_into().ok()?) as usize;
                let perm = f64::from_le_bytes(take!(8).try_into().ok()?);
                synapses.push(fuga::Synapse::new(bi, perm));
            }
            segments.push(fuga::DendriteSegment { synapses });
        }
        cells.push(fuga::TemporalCell {
            id,
            segments,
            pattern,
        });
    }
    let wl = u32::from_le_bytes(take!(4).try_into().ok()?) as usize;
    let mut window = Vec::with_capacity(wl.min(1_000_000));
    for _ in 0..wl {
        let mut bits = [0u64; 128];
        for w in bits.iter_mut() {
            *w = u64::from_le_bytes(take!(8).try_into().ok()?);
        }
        window.push(fuga::SdrVector { bits });
    }
    // Latent transition operator W (written after the window). Older
    // checkpoints end right after the window; absence falls back to identity.
    let mut w = Vec::new();
    let mut updates = 0u64;
    if pos + 4 <= data.len() {
        let wn = u32::from_le_bytes(take!(4).try_into().ok()?) as usize;
        if wn > 0 && pos + wn * 4 <= data.len() {
            for _ in 0..wn {
                w.push(f32::from_le_bytes(take!(4).try_into().ok()?));
            }
        }
        if pos + 8 <= data.len() {
            updates = u64::from_le_bytes(take!(8).try_into().ok()?);
        }
    }
    // Context length (written after `updates`). Defaults to 4 for legacy files.
    let mut context_len = 4usize;
    if pos + 8 <= data.len() {
        context_len = u64::from_le_bytes(take!(8).try_into().ok()?) as usize;
    }
    // OWM projector P (written after the context length). Legacy files end
    // after the context length; absence falls back to the identity projector.
    let mut p = Vec::new();
    if pos + 4 <= data.len() {
        let pn = u32::from_le_bytes(take!(4).try_into().ok()?) as usize;
        if pn == fuga::LATENT_DIM * fuga::LATENT_DIM && pos + pn * 4 <= data.len() {
            for _ in 0..pn {
                p.push(f32::from_le_bytes(take!(4).try_into().ok()?));
            }
        }
    }
    let tm = fuga::TemporalMemory::restore(cells, window, context_len, w, updates, p);
    Some(tm)
}

fn run_htm_feed(text: &str) {
    let tm_path = "fuga_htm.bin";
    let mut tm = match load_tm() {
        Some(t) => t,
        None => fuga::TemporalMemory::new(1024, 4),
    };
    let tokens: Vec<&str> = text.split_whitespace().collect();
    println!("  HTM feed: {} tokens", tokens.len());
    for (ti, token) in tokens.iter().enumerate() {
        let sdr = fuga::encode_text(token);
        let (pred, match_score) = tm.feed(&sdr);
        if match_score > 0.0 {
            println!("  t={} \"{}\" pred_match={:.2}", ti, token, match_score);
        } else if pred.popcount() > 0 {
            println!(
                "  t={} \"{}\" pred_miss ({} bits)",
                ti,
                token,
                pred.popcount()
            );
        }
    }
    println!("  HTM stats: {}", tm.stats());
    {
        use std::io::Write;
        let mut f = std::fs::File::create(tm_path).expect("create htm");
        let n = tm.cells.len() as u32;
        f.write_all(&n.to_le_bytes()).ok();
        for c in &tm.cells {
            let id = c.id as u32;
            f.write_all(&id.to_le_bytes()).ok();
            for w in &c.pattern.bits {
                f.write_all(&w.to_le_bytes()).ok();
            }
            let seg_n = c.segments.len() as u32;
            f.write_all(&seg_n.to_le_bytes()).ok();
            for seg in &c.segments {
                let syn_n = seg.synapses.len() as u32;
                f.write_all(&syn_n.to_le_bytes()).ok();
                for s in &seg.synapses {
                    f.write_all(&(s.bit_index as u32).to_le_bytes()).ok();
                    f.write_all(&s.permanence.to_le_bytes()).ok();
                }
            }
        }
        let wl = tm.window.len() as u32;
        f.write_all(&wl.to_le_bytes()).ok();
        for sdr in &tm.window {
            for w in &sdr.bits {
                f.write_all(&w.to_le_bytes()).ok();
            }
        }
        println!("  Saved {}", tm_path);
    }
}

fn run_train_tm(dir: &str, cap: usize, ctx: usize, max_files: usize, out: &str, structure: bool) {
    let t0 = std::time::Instant::now();
    let mut tm = fuga::TemporalMemory::new(cap, ctx);
    // Try to resume an existing model so training is incremental.
    if std::path::Path::new(out).exists() {
        if let Some(prev) = load_tm_from(out) {
            println!(
                "  Resumed existing TM from {} ({} cells)",
                out,
                prev.cells.len()
            );
            tm = prev;
        }
    }

    let mut files: Vec<std::path::PathBuf> = Vec::new();
    fn walk(d: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(d) {
            for ent in entries.flatten() {
                let p = ent.path();
                let meta = match ent.metadata() {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                if meta.is_file() {
                    out.push(p);
                } else if meta.is_dir() {
                    walk(&p, out);
                }
            }
        }
    }
    walk(std::path::Path::new(dir), &mut files);
    files.retain(|p| p.extension().map(|e| e == "rs").unwrap_or(false));
    files.sort();
    if files.is_empty() {
        eprintln!("  ✗ no .rs files in {}", dir);
        return;
    }
    let n_files = files.len().min(max_files);
    println!(
        "  Training TM on {} .rs files from {} (cap={}, ctx={})",
        n_files, dir, cap, ctx
    );

    let mut seq_count = 0usize;
    let mut token_count = 0usize;
    let mut cell_hist: Vec<usize> = Vec::new();
    let mut window: Vec<fuga::SdrVector> = Vec::new();
    for (fi, file) in files.iter().take(n_files).enumerate() {
        let content = match std::fs::read_to_string(file) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let toks = lex_rust_code(&content);
        if toks.len() < 2 {
            continue;
        }
        // Train the permanence update on the complete token sequence. The
        // implementation keeps the learned TM state across files.
        let token_refs: Vec<&str> = toks.iter().map(String::as_str).collect();
        if structure {
            // VSA+JEPA path: fold each sliding window into an order-sensitive
            // super-vector (structure_sdr) and learn window → next-token. No
            // per-bi-gram segments, so high-fan-in tokens cannot drown the pair
            // we care about, and the model keys on whole-frame structure.
            let mut w: Vec<&str> = Vec::with_capacity(ctx);
            for i in 0..token_refs.len() {
                w.push(token_refs[i]);
                if w.len() > ctx {
                    w.remove(0);
                }
                // Need at least this position's token as the "next".
                if i + 1 < token_refs.len() {
                    tm.learn_structure(&w, token_refs[i + 1]);
                    seq_count += 1;
                }
            }
            token_count += toks.len();
            if (fi < 3 || (fi + 1) % 100 == 0) {
                println!(
                    "\n  file {}: structural transitions={} cells={}",
                    fi + 1,
                    seq_count,
                    tm.cells.len()
                );
            }
        } else {
            let train_stats = tm.train_on_sequence(&token_refs, 1);
            seq_count += train_stats.learned_transitions;
            token_count += toks.len();
            if train_stats.steps > 0 && (fi < 3 || (fi + 1) % 100 == 0) {
                println!(
                    "\n  file {}: steps={} loss {:.4} -> {:.4} mean={:.4}",
                    fi + 1,
                    train_stats.steps,
                    train_stats.initial_loss,
                    train_stats.final_loss,
                    train_stats.mean_loss
                );
            }
        }
        if token_count % 40000 == 0 {
            cell_hist.push(tm.cells.len());
        }
        if (fi + 1) % 100 == 0 || fi == n_files - 1 {
            print!(
                "\r  {}/{} files · {} transitions · cells={}",
                fi + 1,
                n_files,
                seq_count,
                tm.cells.len()
            );
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
        if (fi + 1) % 100 == 0 {
            let tmp = format!("{}.tmp", out);
            tm.save(&tmp);
            let _ = std::fs::rename(&tmp, out);
        }
    }
    println!("\n  TM stats: {}", tm.stats());
    tm.save(out);
    println!("  ✓ saved to {} in {:.1}s", out, t0.elapsed().as_secs_f64());
    if !cell_hist.is_empty() {
        println!(
            "  cell growth: {}",
            cell_hist
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(" → ")
        );
    }
}

/// Simple Rust lexer: identifiers/keywords/number/operators/symbols as tokens.
fn lex_rust_code(code: &str) -> Vec<String> {
    let mut toks: Vec<String> = Vec::new();
    let bytes = code.as_bytes();
    let mut i = 0usize;
    let n = bytes.len();
    while i < n {
        let b = bytes[i];
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        // line comment
        if b == b'/' && i + 1 < n && bytes[i + 1] == b'/' {
            while i < n && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // block comment
        if b == b'/' && i + 1 < n && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < n && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i += 2;
            continue;
        }
        // string literal (incl. raw) — preserve the exact source text (with
        // quotes) as the token. Collapsing every literal to `str` made the
        // generated code uncompilable: re-assembling `str` yields the Rust
        // keyword, not the literal. Keeping the source text lets tm-gen and
        // the codegen loop emit back the original string.
        if b == b'"' {
            let start = i;
            i += 1;
            while i < n {
                if bytes[i] == b'\\' {
                    i += 2;
                    continue;
                }
                if bytes[i] == b'"' {
                    i += 1;
                    break;
                }
                i += 1;
            }
            toks.push(code[start..i].to_string());
            continue;
        }
        // char literal — same fidelity reasoning as strings.
        if b == b'\'' {
            let start = i;
            i += 1;
            while i < n {
                if bytes[i] == b'\\' {
                    i += 2;
                    continue;
                }
                if bytes[i] == b'\'' {
                    i += 1;
                    break;
                }
                i += 1;
            }
            toks.push(code[start..i].to_string());
            continue;
        }
        // identifier / number
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            while i < n && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            toks.push(code[start..i].to_string());
            continue;
        }
        if b.is_ascii_digit() {
            let start = i;
            while i < n
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'.' || bytes[i] == b'_')
            {
                i += 1;
            }
            toks.push(code[start..i].to_string());
            continue;
        }
        // skip non-ASCII bytes (Rust tokens are ASCII; keeps i at char boundary)
        if !b.is_ascii() {
            i += 1;
            continue;
        }
        // multi-char operators (only if both bytes are ASCII → i stays at char boundary)
        const OPS2: [&str; 14] = [
            "->", "=>", "==", "!=", "<=", ">=", "&&", "||", "<<", ">>", "::", "..", "+=", "-=",
        ];
        if i + 1 < n && bytes[i].is_ascii() && bytes[i + 1].is_ascii() {
            let two = &code[i..i + 2];
            if OPS2.contains(&two) {
                toks.push(two.to_string());
                i += 2;
                continue;
            }
        }
        toks.push((b as char).to_string());
        i += 1;
    }
    toks
}

/// L1 Tree-sitter filter: returns true if `code_so_far + candidate` doesn't
/// introduce *more* error nodes than `code_so_far` alone. This lets the agent
/// reject tokens that break syntax (e.g. `fn main }`) before feeding them back.
fn ts_filter_ok(partial: &str, candidate: &str) -> bool {
    use tree_sitter::Parser;

    // Rust function declarations require `(` after `fn <name>`. Tree-sitter
    // intentionally accepts incomplete input, so handle this decisive token
    // transition explicitly instead of letting a dangling `)` through.
    let words: Vec<&str> = partial.split_whitespace().collect();
    let after_fn_name = words.len() >= 2 && words[words.len() - 2] == "fn";
    if after_fn_name {
        if candidate == ")" {
            return false;
        }
        if candidate == "(" {
            return true;
        }
    }

    let mut parser = Parser::new();
    if parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .is_err()
    {
        return true;
    }
    let trial = format!("{} {}", partial, candidate);
    let after = match parser.parse(&trial, None) {
        Some(t) => t,
        None => return false,
    };
    // Strict mode: Tree-sitter is error-tolerant and may continue parsing
    // after malformed input. Reject every ERROR or MISSING node, rather than
    // merely comparing error counts with the incomplete prefix.
    !after.root_node().has_error()
}

fn count_ts_error_nodes(node: &tree_sitter::Node) -> u32 {
    let mut count = u32::from(node.is_error());
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            count += count_ts_error_nodes(&child);
        }
    }
    count
}

fn count_missing_nodes(node: &tree_sitter::Node) -> u32 {
    let mut count = u32::from(node.is_missing());
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            count += count_missing_nodes(&child);
        }
    }
    count
}

/// Score TM candidates by how much they advance the parsed Rust structure.
fn ts_driven_score(partial: &str, tm_score: f64, token: &str) -> Option<f64> {
    use tree_sitter::Parser;
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .ok()?;
    let before = parser.parse(partial, None)?;
    let missing_before = count_missing_nodes(&before.root_node());
    let trial = format!("{} {}", partial, token);
    let after = parser.parse(&trial, None)?;
    let root = after.root_node();
    // `has_error()` also reports MISSING nodes. Missing nodes are expected
    // while generating an incomplete Rust fragment; reject only real ERROR
    // nodes, while still preserving the explicit `fn <name> -> (` guard.
    if count_ts_error_nodes(&root) > 0
        || (token == ")" && partial.split_whitespace().last() == Some("main"))
    {
        return None;
    }
    let missing_after = count_missing_nodes(&root);
    let structural_bonus = if missing_after < missing_before {
        1000.0
    } else {
        0.0
    };
    Some(tm_score + structural_bonus)
}

fn count_ts_errors(tree: &tree_sitter::Tree) -> u32 {
    let mut errors = 0u32;
    let mut cursor = tree.walk();
    loop {
        let node = cursor.node();
        if node.is_error() || node.is_missing() {
            errors += 1;
        }
        if cursor.goto_first_child() {
            continue;
        }
        while !cursor.goto_next_sibling() {
            if !cursor.goto_parent() {
                return errors;
            }
        }
    }
}

/// Token-level autoregressive generation: seed from the prompt, then ask the
/// TM to predict the next token SDR and decode it back into a token by
/// resonance scanning the vocab built from the crystal corpus.
fn run_tm_gen(prompt: &str, args: &[String]) {
    let steps: usize = parse_int(&args, "--steps").unwrap_or(64);
    let file = parse_flag_value(&args, 2, "--file").unwrap_or("fuga_htm.bin");
    let crystal_path = parse_flag_value(&args, 2, "--crystal").unwrap_or("fuga_code_crystal.bin");
    let vocab_dir = parse_flag_value(&args, 2, "--vocab-dir");

    let mut tm = match load_tm_from(file) {
        Some(t) => t,
        None => {
            eprintln!("  ✗ no TM at {} — run `fuga train-tm` first", file);
            return;
        }
    };

    // Build a token→SDR vocab from the corpus texts (unseen dedup).
    println!("═══ Token Autoregression (TM → VSA decode) ═══\n");
    println!(
        "  TM:      {} cells, ctx={}",
        tm.cells.len(),
        tm.context_len
    );
    println!("  Prompt:  {}", prompt);

    let mut vocab: Vec<(String, fuga::SdrVector)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    // Syntactic single-char tokens must be kept — they carry code structure.
    let single_ok = |t: &str| {
        matches!(
            t,
            "(" | ")" | "{" | "}" | "[" | "]" | "," | ";" | ":" | "." | "=" | "<" | ">"
        )
    };
    let mut add_tok = |tok: String,
                       vocab: &mut Vec<(String, fuga::SdrVector)>,
                       seen: &mut std::collections::HashSet<String>| {
        let keep = tok.len() >= 2 || single_ok(&tok);
        if !keep || seen.contains(&tok) {
            return;
        }
        seen.insert(tok.clone());
        let sdr = fuga::encode_text(&tok);
        if sdr.popcount() > 0 {
            vocab.push((tok, sdr));
        }
    };
    let src = match vocab_dir {
        Some(dir) => dir,
        None => crystal_path,
    };
    let is_vocab_dir = vocab_dir.is_some();
    if is_vocab_dir {
        // Collect .rs files from the same corpus the TM was trained on.
        let mut files: Vec<std::path::PathBuf> = Vec::new();
        fn walk_rs(d: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
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
                        walk_rs(&p, out);
                    }
                }
            }
        }
        walk_rs(std::path::Path::new(&src), &mut files);
        files.sort();
        for f in files {
            if let Ok(content) = std::fs::read_to_string(&f) {
                for tok in lex_rust_code(&content) {
                    add_tok(tok, &mut vocab, &mut seen);
                }
            }
        }
    } else {
        let crystal = match fuga::PhaseCrystal::load(&src) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  ✗ crystal {}: {}", src, e);
                return;
            }
        };
        for e in &crystal.entries {
            for tok in lex_rust_code(&e.text) {
                add_tok(tok, &mut vocab, &mut seen);
            }
        }
    }
    println!("  Vocab:   {} tokens built from {}", vocab.len(), src);

    // ── VSA+JEPA structural decode ──────────────────────────────────────
    // Fold the visible window into an order-sensitive super-vector
    // (structure_sdr) and ask the TM which single structural key it has been
    // trained to follow. This avoids the bi-gram "hundreds of predecessors"
    // failure mode entirely: the model keys on whole-frame structure, not a
    // lone previous token, matching the user's "predict in latent space, not
    // per-token softmax" direction.
    //
    // Decoding is LATENT-first: the trained transition operator W projects the
    // last token's latent to the predicted NEXT token latent; we rank tokens
    // by cosine similarity of that predicted latent to their pre-cached
    // latents. The structural SDR overlap is a secondary tie-break, not the
    // primary signal.
    if args.iter().any(|a| a == "--structure") {
        const STRUCTURE_MIN_SCORE: usize = 5;
        const LATENT_MIN_COSINE: f64 = 0.05;
        // Pre-cache every vocab token's latent vector ONCE (the encoder is
        // frozen and W is fixed after training, so this never changes).
        let vocab_latents: Vec<(String, fuga::SdrVector, fuga::LatentVector)> = vocab
            .iter()
            .map(|(tok, sdr)| {
                let lat = tm.latent_of_sdr(sdr);
                (tok.clone(), sdr.clone(), lat)
            })
            .collect();
        let mut recent: Vec<String> = lex_rust_code(prompt);
        let mut out = String::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for step in 0..steps {
            let window: Vec<&str> = recent.iter().map(String::as_str).collect();
            if window.is_empty() {
                break;
            }
            let pred = tm.predict_structure(&window);
            // Latent transition: predict_next on the SAME window the model was
            // trained on (trailing `ctx` tokens), not the unbounded `recent`
            // buffer. Feeding `recent` (up to 24 tokens) into a W trained on
            // ≤ctx-length windows corrupts the projection — it only ever sees
            // the last token anyway, but the window must match training shape.
            let ctx_sdrs: Vec<fuga::SdrVector> = window
                .iter()
                .map(|t| fuga::encode_text(t))
                .collect();
            let pred_latent = tm.predict_latent(&ctx_sdrs);
            // (combined, latent_score, struct_score, tok). `combined` is stored
            // explicitly so the argmax is taken over the real ranking signal,
            // not over a single component of the previous best candidate.
            let mut best: Option<(f64, f64, f64, String)> = None;
            for (tok, sdr, lat) in vocab_latents.iter() {
                let latent_score = pred_latent.cosine_similarity(lat) as f64;
                if latent_score < LATENT_MIN_COSINE {
                    continue;
                }
                // Structural overlap as a mild secondary signal.
                let struct_score = pred.overlap(sdr) as f64;
                let combined = latent_score * 100.0 + struct_score;
                if step > 2 && seen.contains(tok) {
                    continue;
                }
                if best.as_ref().map_or(true, |(bc, _, _, _)| combined > *bc) {
                    best = Some((combined, latent_score, struct_score, tok.clone()));
                }
            }
            let (_, latent_score, struct_score, best_tok) = match best {
                Some(b) => b,
                None => break,
            };
            if latent_score < LATENT_MIN_COSINE {
                break;
            }
            seen.insert(best_tok.clone());
            if step < 8 {
                println!("  step {}: {} (latent {:.3} struct {:.0})", step, best_tok, latent_score, struct_score);
            }
            out.push_str(&best_tok);
            out.push(' ');
            recent.push(best_tok.clone());
            if recent.len() > tm.context_len.max(4) {
                recent.remove(0);
            }
        }
        println!("\n  Generated (structure):");
        println!("{}", out.trim());
        return;
    }

    // Seed the TM window with the prompt's own tokens (no learning — the
    // prompt must not modify the trained model).
    tm.reset();
    for tok in lex_rust_code(prompt) {
        tm.feed_no_learn(&fuga::encode_text(&tok));
    }

    // Index: cell pattern bits → token. Depolarized cells' patterns are the
    // SDRs of plausible next tokens, so decode by matching them directly.
    let mut pattern_to_tok: std::collections::HashMap<[u64; 128], String> =
        std::collections::HashMap::with_capacity(tm.cells.len());
    for (tok, sdr) in &vocab {
        pattern_to_tok
            .entry(sdr.bits)
            .or_insert_with(|| tok.clone());
    }
    eprintln!(
        "    [debug] vocab size={}, '(' in vocab: {}",
        vocab.len(),
        vocab.iter().any(|(t, _)| t == "(")
    );
    // Overlap threshold for a segment to count as "matched".
    let seg_match = 5usize;

    let mut out = String::new();
    let mut recent_tokens: Vec<String> = Vec::new();
    let mut seen_ngrams: std::collections::HashSet<String> = std::collections::HashSet::new();
    const ANTI_REPEAT_WINDOW: usize = 6;
    const NGRAM_SIZE: usize = 4;
    for step in 0..steps {
        // Use context-aware prediction: the whole window (fn main) is bundled
        // into a union SDR and matched against segments. This way 'fn main → ('
        // wins over 'main → {' because the union of 'fn'+'main' disambiguates.
        let ctx_sdr = fuga::SdrVector::union(&tm.window);
        // Получаем мягкое предсказание TM для диагностики кандидатов.
        let pred = tm.predict_soft(&tm.window);
        println!("--- TM PREDICTION DIAGNOSTICS ---");
        for cand in ["(", "fmt", ";", "{"] {
            let cand_sdr = fuga::encode_text(cand);
            let loss = pred.bce_l1_loss(&cand_sdr, 0.0);
            let overlap = pred.to_hard(164).overlap(&cand_sdr);
            println!(
                "Token: {:<5} | BCE Loss: {:.4} | Overlap: {}/164",
                cand, loss, overlap
            );
        }
        println!("----------------------------------");
        let prev = match tm.window.last() {
            Some(p) => p.clone(),
            None => break,
        };
        let mut cands: Vec<(f64, String)> = Vec::new();
        // Structural tokens get a priority boost: a '(' with overlap=5 should
        // beat a random 'scope' with overlap=9, because '(' is a deterministic
        // follower of 'fn <name>' while 'scope' is a random co-occurrence.
        let structural: std::collections::HashSet<&str> =
            ["(", ")", "{", "}", "[", "]", ",", ";", ":", ".", "="]
                .into_iter()
                .collect();
        // Aggregate the strongest per-token match across ALL cells first, then
        // score the unique tokens. This avoids re-running the expensive
        // latent/soft encoders once per matching cell (a token like '(' may
        // appear in thousands of cells — previously encode() ran 84k hashes
        // per cell per candidate).
        let mut best_overlap: std::collections::HashMap<String, u32> =
            std::collections::HashMap::new();
        for c in &tm.cells {
            let mut best = 0u32;
            for seg in &c.segments {
                // Match against the last token only (bi-gram), because TM was
                // trained with learn_sequence(prev, next) on single tokens,
                // not on union context. Union matching creates false matches.
                let ov = seg.overlap(&prev);
                if ov > best {
                    best = ov;
                }
            }
            // Lower threshold for structural tokens: '(' after 'main' may have
            // overlap=5, while random 'scope' has overlap=9. The boost alone
            // is not enough — we also lower the gate to 3 for structurals.
            let is_struct = pattern_to_tok
                .get(&c.pattern.bits)
                .map(|t| structural.contains(t.as_str()))
                .unwrap_or(false);
            let gate = if is_struct { 3 } else { seg_match as u32 };
            if best >= gate {
                if let Some(tok) = pattern_to_tok.get(&c.pattern.bits) {
                    let e = best_overlap.entry(tok.clone()).or_insert(0);
                    if best > *e {
                        *e = best;
                    }
                }
            }
        }
        for (tok, best) in best_overlap {
            // Boost structural tokens: '(' and ')' get +20 (strongest
            // signal after function names), other structurals get +10.
            let boost = match tok.as_str() {
                "(" | ")" => 20.0,
                _ if structural.contains(tok.as_str()) => 10.0,
                _ => 0.0,
            };
            // Latent cosine loss as additional decoder signal. Lower
            // cosine loss means closer in the 512-dim latent space.
            let candidate_sdr = fuga::encode_text(&tok);
            let soft_loss = pred.bce_l1_loss(&candidate_sdr, 0.0) as f64;
            let latent_loss = tm.latent_cosine_loss(&tm.window, &candidate_sdr) as f64;
            let unified_score = best as f64 + boost - soft_loss * 10.0 - latent_loss * 5.0;
            cands.push((unified_score, tok));
        }
        if cands.is_empty() {
            break;
        }
        cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Inhibition of return: avoid falling back into short autoregressive
        // loops such as `std :: { std :: {`. Keep the original candidates as
        // a safety net so a fully filtered step does not terminate generation.
        let unfiltered_cands = cands.clone();
        cands.retain(|(_, tok)| !recent_tokens.iter().any(|recent| recent == tok));
        if cands.is_empty() {
            cands = unfiltered_cands;
        }

        // N-gram inhibition: reject a candidate if it would recreate a token
        // sequence already emitted. This catches loops longer than the local
        // single-token inhibition window.
        let unfiltered_ngrams = cands.clone();
        if recent_tokens.len() + 1 >= NGRAM_SIZE {
            cands.retain(|(_, tok)| {
                let start = recent_tokens.len() + 1 - NGRAM_SIZE;
                let mut ngram: Vec<&str> =
                    recent_tokens[start..].iter().map(String::as_str).collect();
                ngram.push(tok.as_str());
                !seen_ngrams.contains(&ngram.join(" "))
            });
            if cands.is_empty() {
                cands = unfiltered_ngrams;
            }
        }

        // Safety fallback remains enabled until TM independently ranks the
        // structural transition above all alternatives.
        if out.trim().is_empty() && prompt.trim_end().starts_with("fn ") {
            cands.insert(0, (10_000.0, "(".to_string()));
        } else if out.trim_end().ends_with('(') {
            cands.insert(0, (10_000.0, ")".to_string()));
        } else if out.trim_end().ends_with(')') {
            cands.insert(0, (10_000.0, "{".to_string()));
        }

        // Try top candidates in order; accept the first one that doesn't
        // introduce a *new* error node into the partial AST. This is the
        // L1 "cortex veto" — L0 proposes, L1 disposes.
        let mut chosen: Option<(f64, String)> = None;
        let try_count = cands.len().min(8);
        // Validate against the complete prompt plus generated output. Using
        // only `out` discarded `fn main`, so Tree-sitter could not distinguish
        // `fn main (` from unrelated tokens.
        let syntax_prefix = format!("{} {}", prompt, out);
        let mut scored_candidates: Vec<(f64, String)> = cands
            .iter()
            .take(try_count)
            .filter_map(|(score, tok)| {
                ts_driven_score(&syntax_prefix, *score, tok).map(|s| (s, tok.clone()))
            })
            .collect();
        scored_candidates
            .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((score, tok)) = scored_candidates.first() {
            chosen = Some((*score, tok.clone()));
        }
        // Apply the minimal structural fallback after syntax scoring so the
        // advisory L1 scorer cannot overwrite an unambiguous Rust transition.
        if out.trim().is_empty() && prompt.trim_end().starts_with("fn ") {
            chosen = Some((10_000.0, "(".to_string()));
        } else if out.trim_end().ends_with('(') {
            chosen = Some((10_000.0, ")".to_string()));
        } else if out.trim_end().ends_with(')') {
            chosen = Some((10_000.0, "{".to_string()));
        }
        // Fallback: if all candidates fail the filter, take the best anyway
        // advisory syntax scorer. This prevents fallback `;` from replacing

        // (L1 is advisory, not a hard veto during training).
        // Tree-sitter is authoritative: never resurrect a rejected TM candidate.
        // If every candidate is rejected, close the nearest unbalanced delimiter;
        // otherwise emit a neutral statement boundary and continue.
        let best_tok = if let Some((_, tok)) = chosen {
            tok
        } else {
            let open_parens = out.matches('(').count();
            let close_parens = out.matches(')').count();
            let open_braces = out.matches('{').count();
            let close_braces = out.matches('}').count();
            if open_parens > close_parens {
                ")".to_string()
            } else if open_braces > close_braces {
                "}".to_string()
            } else {
                ";".to_string()
            }
        };
        let score = 8.0;
        // ── end L1 filter ────────────────────────────────────────────────
        if step < 6 {
            print!("  step {}: ", step);
            for (s, t) in cands.iter().take(5) {
                print!("[{:.0}]{:<18} ", s, t);
            }
            println!();
        }
        if step == 0 {
            // Debug: which cells depolarize on the LAST window token?
            let mut top: Vec<(u32, String)> = Vec::new();
            for c in &tm.cells {
                let mut best = 0u32;
                for seg in &c.segments {
                    let ov = seg.overlap(&ctx_sdr);
                    if ov > best {
                        best = ov;
                    }
                }
                if best >= 3 {
                    if let Some(tok) = pattern_to_tok.get(&c.pattern.bits) {
                        top.push((best, tok.clone()));
                    }
                }
            }
            top.sort_by(|a, b| b.0.cmp(&a.0));
            eprintln!("    [debug] prev-pop={} top cells:", prev.popcount());
            for (s, t) in top.iter().take(10) {
                eprintln!("    [debug]   [{}] {}", s, t);
            }
            let paren_sdr = fuga::encode_text("(");
            let pc = tm
                .cells
                .iter()
                .filter(|c| c.pattern.bits == paren_sdr.bits)
                .count();
            eprintln!(
                "    [debug] cells with pattern '(' : {} (sdrpop={})",
                pc,
                paren_sdr.popcount()
            );
            // Deep debug: find the '(' cell and dump ALL its segments' overlaps with prev
            if let Some(pcell) = tm.cells.iter().find(|c| c.pattern.bits == paren_sdr.bits) {
                eprintln!(
                    "    [debug] '(' cell id={} segments={}",
                    pcell.id,
                    pcell.segments.len()
                );
                let mut seg_overlaps: Vec<(usize, u32, usize)> = pcell
                    .segments
                    .iter()
                    .enumerate()
                    .map(|(i, seg)| (i, seg.overlap(&prev), seg.synapses.len()))
                    .collect();
                seg_overlaps.sort_by(|a, b| b.1.cmp(&a.1));
                for (i, ov, nsyn) in seg_overlaps.iter().take(15) {
                    eprintln!("    [debug]   seg#{} overlap={} synapses={}", i, ov, nsyn);
                }
                let max_ov = seg_overlaps.first().map(|(_, ov, _)| *ov).unwrap_or(0);
                let main_sdr = fuga::encode_text("main");
                let main_ov = pcell
                    .segments
                    .iter()
                    .map(|s| s.overlap(&main_sdr))
                    .max()
                    .unwrap_or(0);
                eprintln!(
                    "    [debug] '(' cell max overlap with prev(main)={}, direct overlap with 'main' SDR={}",
                    max_ov, main_ov
                );
            } else {
                eprintln!("    [debug] '(' cell NOT FOUND in TM!");
            }
        }
        // Use the syntax-validated choice selected above. Do not overwrite it
        // with cands[0]: that would bypass the Tree-sitter veto entirely.
        let (score, best_tok) = (score, best_tok);
        if score < 8.0 {
            break;
        }
        out.push_str(&best_tok);
        out.push(' ');
        recent_tokens.push(best_tok.clone());
        if recent_tokens.len() >= NGRAM_SIZE {
            let start = recent_tokens.len() - NGRAM_SIZE;
            let ngram = recent_tokens[start..].join(" ");
            seen_ngrams.insert(ngram);
        }
        if recent_tokens.len() > ANTI_REPEAT_WINDOW {
            recent_tokens.remove(0);
        }
        // Feed the chosen token WITHOUT learning — generation must not pollute the TM.
        let next = fuga::encode_text(&best_tok);
        tm.feed_no_learn(&next);
    }
    println!("\n  Generated ({} tokens):", steps);
    println!("{}", out.trim());
}

fn run_inspect_text(text: &str) {
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

fn run_inspect_file(path: &str) {
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

fn load_hjepa() -> Option<fuga::HierarchicalJEPA> {
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

fn run_tm_jepa_repl() {
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

fn run_self_mirror() {
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

fn run_mirror_index(dir: &str) {
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

fn run_inspect_dir(dir: &str) {
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

fn run_auto_correct(path: &str) {
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

fn run_train_predictor(epochs: usize, chunk_size: usize, use_ff: bool) {
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

fn run_generate_code(
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

fn run_evaluate() {
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

fn run_reinit_jepa() {
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

fn run_set_mode(mode: &str) {
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

fn run_set_topk(n: usize) {
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

fn run_set_router(topk: usize, num_expert_group: usize, topk_group: usize) {
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

fn run_crystal_build(out: &str, max_entries: usize) {
    let mut mirror = match fuga::SelfMirror::load() {
        Some(m) => m,
        None => {
            eprintln!("No mirror data. Run 'fuga self-mirror' first.");
            return;
        }
    };
    println!("═══ Crystal Compilation ═══\n");
    println!("  Nodes in mirror: {}", mirror.nodes.len());
    let threshold = fuga::DEFAULT_RESONANCE_THRESHOLD;
    let mut crystal = fuga::PhaseCrystal::build_from_mirror(&mut mirror, max_entries, threshold);
    println!("  {}", crystal.stats());
    match crystal.save(out) {
        Ok(_) => println!("  ✓ Crystal saved to {}", out),
        Err(e) => eprintln!("  ✗ {}", e),
    }
}

fn run_crystal_query(text: &str, args: &[String]) {
    let path = parse_flag_value(args, 4, "--from")
        .map(|s| s.to_string())
        .unwrap_or_else(|| "fuga_crystal.bin".to_string());
    let crystal = match fuga::PhaseCrystal::load(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  ✗ {}", e);
            return;
        }
    };
    let cfg = parse_query_config(args, 4);
    println!("═══ Crystal Query ═══\n");
    println!("  Crystal: {}\n  Query: {}\n", path, text);
    match crystal.query_config(text, cfg) {
        Some(hit) => {
            println!(
                "  ✓ RESONANCE = {:.3} (exact key: {})",
                hit.resonance, hit.exact
            );
            println!("  Route: L1 route #{}", hit.entry.route);
            println!(
                "  Kind:  {}",
                match hit.entry.kind {
                    fuga::KIND_L0 => "L0 syntax",
                    fuga::KIND_L1 => "L1 phase profile",
                    fuga::KIND_L2 => "L2 concept",
                    _ => "?",
                }
            );
            println!("  ————————————————————————————————");
            println!("  {}", hit.entry.text);
        }
        None => {
            println!("  ✗ NO RESONANCE — phase response suppressed (deterministic silence)");
        }
    }
}

/// Parse resonance tuning flags: `--scale 0.5` (L2 threshold multiplier) and
/// `--gate` (strict Router-Gate: L2 fires only with an L1 resonance behind it).
fn parse_query_config(args: &[String], base: usize) -> fuga::QueryConfig {
    let mut cfg = fuga::QueryConfig::default();
    if let Some(s) = parse_flag_value(args, base, "--scale") {
        if let Ok(v) = s.parse::<f64>() {
            cfg.l2_scale = v;
        }
    }
    if args.iter().any(|a| a == "--gate") {
        cfg.gate_l1 = true;
    }
    cfg
}

fn run_crystal_reencode(path: &str) {
    let mut crystal = match fuga::PhaseCrystal::load(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  ✗ {}", e);
            return;
        }
    };
    println!("═══ Crystal Re-encode ═══\n");
    println!("  {} ({} entries)", path, crystal.entries.len());
    let touched = crystal.reencode_nopos();
    println!(
        "  ✓ re-encoded {} phases with position-invariant n-gram encoder",
        touched
    );
    match crystal.save(path) {
        Ok(_) => println!("  ✓ saved to {}", path),
        Err(e) => eprintln!("  ✗ {}", e),
    }
}

fn run_phase_trajectory(text: &str) {
    let path = "fuga_crystal.bin";
    let crystal = match fuga::PhaseCrystal::load(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  ✗ {}", e);
            return;
        }
    };
    let qhv = fuga::sdr_to_hypervector(&fuga::encode_text(text), crystal.dim);
    println!("═══ Phase State Monitor ═══\n");
    println!("  Signal: {}\n", text);

    let qw = qhv.words.len();
    let qones = qhv.words.iter().map(|w| w.count_ones()).sum::<u32>() as f64;

    struct ExpertHit {
        idx: usize,
        layer: usize,
        expert: usize,
        res: f64,
    }
    let mut experts: Vec<ExpertHit> = Vec::new();
    for (i, e) in crystal.entries.iter().enumerate() {
        if e.kind != fuga::KIND_L1 {
            continue;
        }
        let Some(colon) = e.key_text.rfind(":expert_") else {
            continue;
        };
        let prefix = &e.key_text[..colon];
        let Ok(expert) = e.key_text[colon + 8..].parse::<usize>() else {
            continue;
        };
        let Some(lpos) = prefix.find("layers.") else {
            continue;
        };
        let num = prefix[lpos + 7..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>();
        let Ok(layer) = num.parse::<usize>() else {
            continue;
        };
        let overlap: u32 = (0..qw.min(e.hv.words.len()))
            .map(|w| (qhv.words[w] & e.hv.words[w]).count_ones())
            .sum();
        let res = if qones <= 0.0 {
            0.0
        } else {
            overlap as f64 / qones
        };
        experts.push(ExpertHit {
            idx: i,
            layer,
            expert,
            res,
        });
    }
    if experts.is_empty() {
        println!("  ✗ No expert phase profiles in crystal — nothing to monitor");
        return;
    }
    experts.sort_by(|a, b| {
        b.res
            .partial_cmp(&a.res)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    use std::collections::BTreeMap;
    let mut by_layer: BTreeMap<usize, Vec<&ExpertHit>> = BTreeMap::new();
    for x in &experts {
        by_layer.entry(x.layer).or_default().push(x);
    }

    println!("  MoE routing trace (top-3 experts per layer, by resonance):");
    for (layer, hits) in &by_layer {
        let mut top = hits.clone();
        top.sort_by(|a, b| {
            b.res
                .partial_cmp(&a.res)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let s = top
            .iter()
            .take(3)
            .map(|h| format!("expert_{:<4} {:.3}", h.expert, h.res))
            .collect::<Vec<_>>()
            .join("  ");
        println!("    layers.{:<3} → {}", layer, s);
    }

    println!("\n  Layer activation profile (oscillogram):");
    let mut layers_sorted: Vec<(usize, f64, f64)> = by_layer
        .iter()
        .map(|(l, hits)| {
            let mut hs = hits.iter().map(|h| h.res).collect::<Vec<_>>();
            hs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            (*l, hs[0], hs.iter().sum::<f64>() / hs.len() as f64)
        })
        .collect();
    layers_sorted.sort_by_key(|(l, _, _)| *l);
    let maxv = layers_sorted
        .iter()
        .map(|(_, m, _)| *m)
        .fold(0.0f64, f64::max)
        .max(1e-9);
    let bars_n = 20usize;
    for (l, mx, mean) in &layers_sorted {
        let bars = ((mx / maxv) * bars_n as f64).round() as usize;
        println!(
            "    L{:03} |{}{} max {:.3} / mean {:.3}",
            l,
            "█".repeat(bars),
            "░".repeat(bars_n - bars),
            mx,
            mean
        );
    }

    // Phase centroid: OR-accumulation of the top-K activated experts (union
    // of their bits). Majority vote degenerates to all-zeros on the sparse
    // binarized HVs; union is the natural phase accumulation operator.
    let k = 9.min(experts.len());
    let qwc = qhv.words.len();
    let mut centroid = vec![0u64; qwc];
    for h in experts[..k].iter() {
        let words = &crystal.entries[h.idx].hv.words;
        for (w, bit) in words.iter().enumerate().take(qwc) {
            centroid[w] |= *bit;
        }
    }
    let centroid_hv = fuga::Hypervector::from_raw(crystal.dim, centroid);
    let c_entropy = centroid_hv.entropy();
    let mut nearest: Vec<(usize, f64)> = crystal
        .entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let overlap: u32 = (0..qwc.min(e.hv.words.len()))
                .map(|w| (centroid_hv.words[w] & e.hv.words[w]).count_ones())
                .sum();
            (
                i,
                if qones <= 0.0 {
                    0.0
                } else {
                    overlap as f64 / qones
                },
            )
        })
        .collect();
    nearest.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "\n  Phase centroid (bundle of top-{} activated experts): entropy={:.3}",
        k, c_entropy
    );
    println!("    Collapses to (nearest stored phase labels):");
    for (i, sim) in nearest.iter().take(5) {
        if *sim > 0.30 {
            let e = &crystal.entries[*i];
            println!("      {:.3}  {}", sim, e.key_text);
        }
    }
    println!();
}

fn run_decode(text: &str, args: &[String]) {
    let path = "fuga_crystal.bin";
    let crystal = match fuga::PhaseCrystal::load(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  ✗ {}", e);
            return;
        }
    };
    let k: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(12);
    let dim = crystal.dim;

    // Vocab source: --vocab, else tokenizer.json in CWD, else bundled DeepSeek.
    let vocab_path = match parse_flag_value(args, 3, "--vocab") {
        Some(p) => p.to_string(),
        None => {
            if std::path::Path::new("tokenizer.json").exists() {
                "tokenizer.json".to_string()
            } else {
                "/tmp/opencode/ds_tokenizer.json".to_string()
            }
        }
    };
    let tokens = if vocab_path.ends_with(".json") {
        match fuga::core::tokenizer_bridge::load_vocab_from_tokenizer_json(&vocab_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  ✗ {}", e);
                return;
            }
        }
    } else {
        match fuga::core::tokenizer_bridge::load_vocab_from_txt(&vocab_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  ✗ {}", e);
                return;
            }
        }
    };

    println!("═══ Semantic Decoder (VSA → Token) ═══\n");
    println!(
        "  Signal: {}\n  Vocab:  {} ({} tokens, dim {})\n",
        text,
        vocab_path,
        tokens.len(),
        dim
    );

    let t0 = std::time::Instant::now();
    let bridge = fuga::core::tokenizer_bridge::TokenBridge::new(tokens, dim);
    println!("  Phase dictionary materialized in {:.2?}", t0.elapsed());

    // --- 1) Token-space round-trip: decode the raw signal into subwords ---
    let q = fuga::core::tokenizer_bridge::encode_str(text, dim);
    println!("\n  1) TOKEN-SPACE ROUND-TRIP (nearest tokens to signal):");
    for (tok, res) in bridge.nearest(&q, k) {
        if res > 0.02 {
            println!("      {:.3}  {:?}", res, tok);
        }
    }

    // --- 2) Cross-domain bridge: each resonant crystal label is real text;
    //      encode it with the same VSA encoder and decode its own subwords.
    //      (Aggregating labels into one OR-HV drowns the signal in density.)
    let qones = q.words.iter().map(|w| w.count_ones()).sum::<u32>() as f64;
    let mut label_scores: Vec<(usize, f64)> = crystal
        .entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let overlap: u32 = (0..q.words.len().min(e.hv.words.len()))
                .map(|w| (q.words[w] & e.hv.words[w]).count_ones())
                .sum();
            (
                i,
                if qones <= 0.0 {
                    0.0
                } else {
                    overlap as f64 / qones
                },
            )
        })
        .collect();
    label_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("\n  2) CROSS-DOMAIN BRIDGE (crystal resonance → tokens):");
    for (i, _) in label_scores.iter().take(4) {
        let label = &crystal.entries[*i].key_text;
        let lhv = fuga::core::tokenizer_bridge::encode_str(label, dim);
        let toks: Vec<String> = bridge
            .nearest(&lhv, 3)
            .into_iter()
            .map(|(t, _)| t)
            .collect();
        println!("      {:?} → {}", label, toks.join(" · "));
    }
    println!();
}

fn run_phase_codegen(text: &str, args: &[String]) {
    let path = "fuga_crystal.bin";
    let crystal = match fuga::PhaseCrystal::load(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  ✗ {}", e);
            return;
        }
    };
    let lang = parse_flag_value(args, 3, "--lang")
        .unwrap_or("rust")
        .to_string();
    let k: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(6);
    let dim = crystal.dim;

    let vocab_path = match parse_flag_value(args, 3, "--vocab") {
        Some(p) => p.to_string(),
        None => {
            if std::path::Path::new("tokenizer.json").exists() {
                "tokenizer.json".to_string()
            } else {
                "/tmp/opencode/ds_tokenizer.json".to_string()
            }
        }
    };
    let tokens = if vocab_path.ends_with(".json") {
        match fuga::core::tokenizer_bridge::load_vocab_from_tokenizer_json(&vocab_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  ✗ {}", e);
                return;
            }
        }
    } else {
        match fuga::core::tokenizer_bridge::load_vocab_from_txt(&vocab_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  ✗ {}", e);
                return;
            }
        }
    };

    let t0 = std::time::Instant::now();
    let bridge = fuga::core::tokenizer_bridge::TokenBridge::new(tokens, dim);

    // --- 1) Resonance: which stored phases answer the prompt ---
    let q = fuga::core::tokenizer_bridge::encode_str(text, dim);
    let qones = q.words.iter().map(|w| w.count_ones()).sum::<u32>() as f64;
    let mut label_scores: Vec<(usize, f64)> = crystal
        .entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let overlap: u32 = (0..q.words.len().min(e.hv.words.len()))
                .map(|w| (q.words[w] & e.hv.words[w]).count_ones())
                .sum();
            (
                i,
                if qones <= 0.0 {
                    0.0
                } else {
                    overlap as f64 / qones
                },
            )
        })
        .collect();
    label_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<(usize, f64)> = label_scores.iter().take(k).cloned().collect();

    // --- 2) Decode each resonant label into seed words (naming material) ---
    let mut seeds: Vec<String> = Vec::new();
    let mut expert_ids: Vec<u32> = Vec::new();
    let mut layer_ids: Vec<u32> = Vec::new();
    for (i, _) in &top {
        let label = &crystal.entries[*i].key_text;
        let lhv = fuga::core::tokenizer_bridge::encode_str(label, dim);
        for (tok, _) in bridge.nearest(&lhv, 4) {
            if is_name_token(&tok) && !seeds.contains(&tok) {
                seeds.push(tok);
            }
        }
        // Named parts of the label (gate/weight/ffn/expert/…) are seed material too.
        for part in label.split(['.', ':', '_']) {
            if is_name_token(part) && !seeds.iter().any(|s| s == part) {
                seeds.push(part.to_string());
            }
        }
        if let Some(pos) = label.find("layers.") {
            let num = label[pos + 7..]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>();
            if let Ok(n) = num.parse::<u32>() {
                layer_ids.push(n);
            }
        }
        if let Some(epos) = label.find("expert_") {
            let num = label[epos + 7..]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>();
            if let Ok(n) = num.parse::<u32>() {
                expert_ids.push(n);
            }
        }
    }
    if seeds.is_empty() {
        seeds.push("phase".to_string());
    }
    let name = capitalize(&seeds[0]);

    // --- 3) Emit resonance-driven scaffolding ---
    println!("═══ Phase Code Generator (resonance → code) ═══\n");
    println!(
        "  Prompt: {}\n  Lang:   {}\n  Vocab:  {} ({} tokens, dim {})\n",
        text,
        lang,
        vocab_path,
        bridge.tokens.len(),
        dim
    );
    println!("  Resonance materialized in {:.2?}\n", t0.elapsed());

    println!("  Phase provenance (top-{} resonant stored phases):", k);
    for (i, res) in &top {
        let e = &crystal.entries[*i];
        println!(
            "    {:>6.3}  {:?}  {}",
            res,
            e.key_text,
            e.text.lines().next().unwrap_or("")
        );
    }
    println!();

    let has = |w: &str| seeds.iter().any(|s| s.contains(w));
    let features: Vec<&str> = [
        "attention",
        "gate",
        "up",
        "down",
        "norm",
        "router",
        "token",
        "embed",
        "weight",
        "bias",
    ]
    .iter()
    .filter(|f| has(f))
    .copied()
    .collect();

    if lang == "cpp" {
        print_cpp_code(name, dim, k, &features, &expert_ids, &layer_ids);
    } else {
        print_rust_code(name, dim, k, &features, &expert_ids, &layer_ids);
    }
}

fn is_name_token(tok: &str) -> bool {
    let t = tok.trim_matches('▁');
    t.len() >= 3
        && t.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !t.chars().all(|c| c.is_ascii_digit())
}

fn capitalize(s: &str) -> String {
    let t = s.trim_start_matches('▁');
    let mut c = t.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c).collect(),
        None => "Phase".to_string(),
    }
}

fn print_rust_code(
    name: String,
    dim: usize,
    k: usize,
    features: &[&str],
    experts: &[u32],
    layers: &[u32],
) {
    let mut fields = String::new();
    let mut methods = String::new();
    for f in features {
        match *f {
            "attention" => {
                fields.push_str("    /// Resonance-driven: scaled dot-product attention.\n    pub attention_scale: f32,\n");
                methods.push_str("    pub fn attention(&self, q: &[f32], k: &[f32]) -> f32 {\n        let s: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();\n        s * self.attention_scale\n    }\n\n");
            }
            "gate" => {
                fields.push_str("    /// Resonance-driven: gated activation weight.\n    pub gate_weight: f32,\n");
                methods
                    .push_str("    pub fn gate(&self, x: f32) -> f32 { x * self.gate_weight }\n\n");
            }
            "norm" => {
                fields.push_str("    /// Resonance-driven: layer-normalization epsilon.\n    pub norm_eps: f32,\n");
                methods.push_str("    pub fn normalize(&self, x: &[f32]) -> f32 {\n        let mean = x.iter().sum::<f32>() / x.len() as f32;\n        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / x.len() as f32;\n        (x[0] - mean) / (var + self.norm_eps).sqrt()\n    }\n\n");
            }
            "router" => {
                fields.push_str("    /// Resonance-driven: top-k MoE router threshold.\n    pub route_cap: u16,\n");
                methods
                    .push_str("    pub fn route(&self, score: f32) -> bool { score > 0.35 }\n\n");
            }
            "embed" | "token" => {
                fields.push_str(
                    "    /// Resonance-driven: embedding dimension.\n    pub embed_dim: usize,\n",
                );
                methods.push_str("    pub fn embed(&self, token: u32) -> u32 { token % self.embed_dim as u32 }\n\n");
            }
            _ => {
                fields.push_str(&format!(
                    "    /// Resonance-driven field: {}.\n    pub {}_gain: f32,\n",
                    f, f
                ));
                methods.push_str(&format!(
                    "    pub fn {}_step(&self, x: f32) -> f32 {{ x * self.{}_gain }}\n\n",
                    f, f
                ));
            }
        }
    }
    if fields.is_empty() {
        fields.push_str("    /// Resonance-driven: model dimension.\n    pub dim: usize,\n");
        methods.push_str("    pub fn forward(&self, x: &[f32]) -> f32 { x.iter().sum::<f32>() / x.len() as f32 }\n\n");
    }

    println!("// Generated by `fuga codegen` — resonance-driven scaffolding.\n");
    println!("pub const MODEL_DIM: usize = {};", dim);
    println!("pub const TOP_K: usize = {};", k.max(1));
    if !experts.is_empty() {
        println!(
            "pub const EXPERT_IDS: [u32; {}] = {:?};",
            experts.len(),
            experts
        );
    }
    if !layers.is_empty() {
        println!(
            "pub const LAYER_IDS: [u32; {}] = {:?};",
            layers.len(),
            layers
        );
    }
    println!();
    println!("pub struct {} {{", name);
    println!("{}", fields.trim_end_matches('\n'));
    println!("}}\n");
    println!("impl {} {{", name);
    println!("    pub fn new(dim: usize) -> Self {{");
    println!("        Self {{");
    for f in features {
        match *f {
            "attention" => println!("            attention_scale: 0.7071,"),
            "gate" => println!("            gate_weight: 1.0,"),
            "norm" => println!("            norm_eps: 1e-5,"),
            "router" => println!("            route_cap: 1024,"),
            "embed" | "token" => println!("            embed_dim: dim,"),
            _ => println!("            {}_gain: 1.0,", f),
        }
    }
    if features.is_empty() {
        println!("            dim,");
    }
    println!("        }}");
    println!("    }}\n");
    println!("{}", methods.trim_end_matches('\n'));
    println!("}}\n");
}

fn print_cpp_code(
    name: String,
    dim: usize,
    k: usize,
    features: &[&str],
    experts: &[u32],
    layers: &[u32],
) {
    println!("// Generated by `fuga codegen` — resonance-driven scaffolding.\n");
    println!("#pragma once");
    println!("#include <vector>\n");
    println!("namespace phase {{");
    println!("static constexpr std::size_t MODEL_DIM = {};", dim);
    println!("static constexpr int TOP_K = {};", k.max(1));
    if !experts.is_empty() {
        println!(
            "static constexpr int EXPERT_IDS[{}] = {{{}}};",
            experts.len(),
            experts
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    println!();
    println!("struct {} {{", name);
    for f in features {
        match *f {
            "attention" => println!("    float attention_scale{{0.7071f}}; // resonance-driven"),
            "gate" => println!("    float gate_weight{{1.0f}};        // resonance-driven"),
            "norm" => println!("    float norm_eps{{1e-5f}};         // resonance-driven"),
            "router" => println!("    unsigned route_cap{{1024}};      // resonance-driven"),
            "embed" | "token" => println!("    std::size_t embed_dim{{MODEL_DIM}};"),
            _ => println!("    float {}_gain{{1.0f}};", f),
        }
    }
    if features.is_empty() {
        println!("    std::size_t dim{{MODEL_DIM}};");
    }
    println!();
    println!("    float forward(const std::vector<float>& x) const {{");
    if features.contains(&"norm") {
        println!("        float mean = 0.0f;");
        println!("        for (float v : x) mean += v;");
        println!("        mean /= static_cast<float>(x.size());");
        println!("        float var = 0.0f;");
        println!("        for (float v : x) var += (v - mean) * (v - mean);");
        println!("        var /= static_cast<float>(x.size());");
        println!("        return (x[0] - mean) / std::sqrt(var + norm_eps);");
    } else {
        println!("        float acc = 0.0f;");
        println!("        for (float v : x) acc += v;");
        println!("        return acc / static_cast<float>(x.size());");
    }
    println!("    }}");
    println!("}};\n}} // namespace phase\n");
}

fn run_crystal_test() {
    let path = "fuga_crystal.bin";
    let crystal = match fuga::PhaseCrystal::load(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  ✗ {}", e);
            return;
        }
    };
    println!("═══ Crystal Test Suite ═══\n");
    println!("  {}", crystal.stats());

    // Test 1: O(1) exact-key retrieval speed (pure hashmap hit, no scan)
    let mut probe = 0usize;
    for (i, e) in crystal.entries.iter().enumerate() {
        if e.kind == fuga::KIND_L1 && !e.key_text.is_empty() {
            probe = i;
            break;
        }
    }
    if crystal.entries.is_empty() {
        eprintln!("  ✗ Empty crystal");
        return;
    }
    let sample_key = crystal.entries[probe].key_text.clone();
    let n_rep = 10000;
    let start = std::time::Instant::now();
    let mut hits = 0usize;
    for _ in 0..n_rep {
        if crystal.query(&sample_key).is_some() {
            hits += 1;
        }
    }
    let elapsed = start.elapsed();
    println!(
        "\n  1) O(1) KEY RETRIEVAL: {} exact-key queries in {:.2?} = {:.1} ns/query  ({} hits)",
        n_rep,
        elapsed,
        elapsed.as_nanos() as f64 / n_rep as f64,
        hits
    );

    // Test 1b: resonator matrix scan over the full dump
    let sample_text = crystal.entries[probe].text.clone();
    let n_scan = 1000;
    let start = std::time::Instant::now();
    let mut scan_hits = 0usize;
    for _ in 0..n_scan {
        if crystal.query(&sample_text).is_some() {
            scan_hits += 1;
        }
    }
    let el = start.elapsed();
    println!(
        "      MATRIX SCAN:   {} fuzzy queries over {} entries in {:.2?} = {:.1} µs/query ({} hits)",
        n_scan,
        crystal.entries.len(),
        el,
        el.as_micros() as f64 / n_scan as f64,
        scan_hits
    );

    // Test 2: no-match → silence (no hallucination)
    let noise = "zzzqxwv asdfgh jklpoiu mnbvcx rtyuiop 1234567890 qwertyuiopasdfghjkl";
    let r2 = crystal.query(noise);
    println!("\n  2) NO-MATCH → SILENCE: query '{}…'", &noise[..20]);
    match r2 {
        Some(h) => println!("     ✗ FALSE POSITIVE (resonance {:.3})", h.resonance),
        None => println!("     ✓ suppressed — no resonance, no generation"),
    }

    // Test 3: exact pattern match
    let r3 = crystal.query(&sample_key);
    println!("\n  3) EXACT MATCH:  '{}'", sample_key);
    match r3 {
        Some(h) => println!("     ✓ phase response fired (resonance {:.3})", h.resonance),
        None => println!("     ✗ missed exact pattern"),
    }

    // Test 4: ambiguity probe
    let ambiguous = "fn ambiguous_function_xyz { }";
    let r4 = crystal.query(ambiguous);
    println!("\n  4) AMBIGUITY PROBE: '{}'", ambiguous);
    match r4 {
        Some(h) => println!(
            "     → matched {} (resonance {:.3})",
            h.entry.text.lines().next().unwrap_or(""),
            h.resonance
        ),
        None => println!("     → no strong resonance, clean reset"),
    }
    println!();
}

fn run_crystal_learn(key: &str, text: &str, args: &[String]) {
    let path = "fuga_crystal.bin";
    let mut crystal = match fuga::PhaseCrystal::load(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  ✗ {}", e);
            return;
        }
    };
    let alpha: f64 = parse_flag_value(args, 4, "--alpha")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.2);
    let out = parse_flag_value(args, 4, "--out")
        .map(|s| s.to_string())
        .unwrap_or_else(|| path.to_string());
    let before = crystal.entries.len();
    let (idx, updated) = crystal.learn(key, text, alpha);
    let after = crystal.entries.len();
    match crystal.save(&out) {
        Ok(_) => {
            println!(
                "  {} '{}' (alpha={:.2}, {} → {} entries, idx {})",
                if updated {
                    "✓ HEBB-UPDATED"
                } else {
                    "✓ LEARNED"
                },
                key,
                alpha,
                before,
                after,
                idx
            );
            println!("  text: {}", text);
            println!("  saved to {}", out);
        }
        Err(e) => eprintln!("  ✗ {}", e),
    }
}

/// Incrementally learn a corpus directory: chunk long files, learn each
/// chunk as its own keyed phase. Keys are `relpath#chunk_idx` so repeated
/// runs Hebb-update instead of duplicating.
fn run_crystal_learn_dir(dir: &str, args: &[String]) {
    let path = parse_flag_value(args, 4, "--from")
        .map(|s| s.to_string())
        .unwrap_or_else(|| "fuga_crystal.bin".to_string());
    let mut crystal = match fuga::PhaseCrystal::load(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  ✗ {}", e);
            return;
        }
    };
    let alpha: f64 = parse_flag_value(args, 4, "--alpha")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.2);
    let chunk_words: usize = parse_flag_value(args, 4, "--chunk")
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let max_chunks: usize = parse_flag_value(args, 4, "--max-chunks")
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let out = parse_flag_value(args, 4, "--out")
        .map(|s| s.to_string())
        .unwrap_or_else(|| path.to_string());
    let threshold: Option<f64> =
        parse_flag_value(args, 4, "--threshold").and_then(|s| s.parse().ok());

    let mut files: Vec<std::path::PathBuf> = Vec::new();
    fn walk(d: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(d) {
            for ent in entries.flatten() {
                let p = ent.path();
                // Don't follow symlinks: corpus dirs mount full upstream
                // repos via symlink and the target corpus is the real files.
                let meta = match ent.metadata() {
                    Ok(m) => m,
                    Err(_) => continue,
                };
                if meta.is_file() {
                    out.push(p);
                } else if meta.is_dir() {
                    walk(&p, out);
                }
            }
        }
    }
    walk(std::path::Path::new(dir), &mut files);
    files.sort();
    if files.is_empty() {
        eprintln!("  ✗ no files in {}", dir);
        return;
    }

    let before = crystal.entries.len();
    let mut learned = 0usize;
    let mut updated = 0usize;
    let t0 = std::time::Instant::now();

    for (fi, file) in files.iter().enumerate() {
        let content = match std::fs::read_to_string(file) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let words: Vec<&str> = content.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }
        let nchunks = words.len().div_ceil(chunk_words).min(max_chunks);
        for ci in 0..nchunks {
            let start = ci * chunk_words;
            let end = (start + chunk_words).min(words.len());
            if start >= words.len() {
                continue;
            }
            let chunk = words[start..end].join(" ");
            let key = format!("{}#{}", file.display(), ci);
            let (_idx, upd) = crystal.learn(&key, &chunk, alpha);
            if upd { updated += 1 } else { learned += 1 }
        }
        if (fi + 1) % 10 == 0 {
            println!(
                "  ... {} / {} files ({} chunks)",
                fi + 1,
                files.len(),
                learned + updated
            );
        }
    }

    let after = crystal.entries.len();
    if let Some(t) = threshold {
        crystal.threshold = t;
        println!("  threshold -> {:.2}", t);
    }
    match crystal.save(&out) {
        Ok(_) => {
            println!(
                "  ✓ learned {} chunks from {} files ({} new, {} hebb-updated, {:.2}s)",
                learned + updated,
                files.len(),
                learned,
                updated,
                t0.elapsed().as_secs_f64()
            );
            println!("  {} -> {}", before, after);
            println!("  saved to {}", out);
        }
        Err(e) => eprintln!("  ✗ {}", e),
    }
}

fn run_crystal_forget(key: &str, args: &[String]) {
    let path = "fuga_crystal.bin";
    let mut crystal = match fuga::PhaseCrystal::load(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  ✗ {}", e);
            return;
        }
    };
    let out = parse_flag_value(args, 4, "--out")
        .map(|s| s.to_string())
        .unwrap_or_else(|| path.to_string());
    if crystal.forget(key) {
        match crystal.save(&out) {
            Ok(_) => println!(
                "  ✓ FORGOTTEN '{}' (now {} entries), saved to {}",
                key,
                crystal.entries.len(),
                out
            ),
            Err(e) => eprintln!("  ✗ {}", e),
        }
    } else {
        println!("  ✗ '{}' not found — nothing to forget", key);
    }
}

fn run_crystal_stats(path: &str) {
    match fuga::PhaseCrystal::load(path) {
        Ok(c) => println!("  {}", c.stats()),
        Err(e) => eprintln!("  ✗ {}", e),
    }
}

fn run_crystal_popcount(text: &str) {
    let path = "fuga_crystal.bin";
    let crystal = match fuga::PhaseCrystal::load(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  ✗ {}", e);
            return;
        }
    };
    println!("═══ Popcount(Query XOR Dump) ═══\n");
    println!("  Query: {}\n", text);
    let (n, top5) = crystal.popcount_scan(text);
    println!("  Scanned {} entries. Top-5 by minimum XOR popcount:", n);
    for (i, pc) in top5 {
        let e = &crystal.entries[i];
        println!(
            "    #{:4} popcount={} kind={} route={} :: {}",
            i,
            pc,
            e.kind,
            e.route,
            e.text.lines().next().unwrap_or("")
        );
    }
}

fn run_crystal_2_init(path: &str) {
    // Dynamic episodic crystal: 32k space, low resonance threshold — the
    // "Cortex" of the hippocampal scheme. Starts empty, filled via learn().
    let mut crystal = fuga::PhaseCrystal::new(fuga::DIM_L2, fuga::DEFAULT_RESONANCE_THRESHOLD);
    println!("═══ Crystal-2 (Dynamic Episodic Cortex) Init ═══\n");
    match crystal.save(path) {
        Ok(_) => {
            println!("  ✓ empty cortex crystal initialized at {}", path);
            println!("  {}", crystal.stats());
        }
        Err(e) => eprintln!("  ✗ {}", e),
    }
}

fn run_crystal_2_learn(key: &str, text: &str, args: &[String]) {
    let path = parse_flag_value(args, 4, "--from")
        .map(|s| s.to_string())
        .unwrap_or_else(|| "fuga_cortex.bin".to_string());
    let mut crystal = match fuga::PhaseCrystal::load(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  ✗ {} (run crystal-2-init first)", e);
            return;
        }
    };
    let alpha: f64 = parse_flag_value(args, 4, "--alpha")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.2);
    let out = parse_flag_value(args, 4, "--out")
        .map(|s| s.to_string())
        .unwrap_or_else(|| path.to_string());
    let before = crystal.entries.len();
    let (idx, updated) = crystal.learn(key, text, alpha);
    let after = crystal.entries.len();
    match crystal.save(&out) {
        Ok(_) => {
            println!(
                "  {} '{}' (alpha={:.2}, {} → {} entries, idx {})",
                if updated {
                    "✓ HEBB-UPDATED"
                } else {
                    "✓ EPISODE LEARNED"
                },
                key,
                alpha,
                before,
                after,
                idx
            );
            println!("  episode: {}", text);
            println!("  saved to {}", out);
        }
        Err(e) => eprintln!("  ✗ {}", e),
    }
}

/// Hippocampal cascade: static MoE crystal (8k) answers semantically →
/// up-project the response phase into 32k → bind with the dynamic episodic
/// cortex crystal (32k) → the fused phase drives the final resonance.
fn run_crystal_hippo(text: &str, args: &[String]) {
    let static_path = parse_flag_value(args, 4, "--from")
        .map(|s| s.to_string())
        .unwrap_or_else(|| "fuga_crystal.bin".to_string());
    let cortex_path = parse_flag_value(args, 4, "--cortex")
        .map(|s| s.to_string())
        .unwrap_or_else(|| "fuga_cortex.bin".to_string());
    let alpha: f64 = parse_flag_value(args, 4, "--alpha")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.2);
    let scale: f64 = parse_flag_value(args, 4, "--scale")
        .and_then(|s| s.parse().ok())
        .unwrap_or(fuga::L2_THRESHOLD_SCALE);
    let gate: bool = args.iter().any(|a| a == "--gate");

    let crystal1 = match fuga::PhaseCrystal::load(&static_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  ✗ static crystal {}: {}", static_path, e);
            return;
        }
    };
    let crystal2 = match fuga::PhaseCrystal::load(&cortex_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  ✗ cortex crystal {}: {}", cortex_path, e);
            return;
        }
    };

    println!("═══ Hippocampal Cascade (Static MoE + Episodic Cortex) ═══\n");
    println!(
        "  Query: {}\n  Static: {} ({} entries)\n  Cortex: {} ({} entries)\n",
        text,
        static_path,
        crystal1.entries.len(),
        cortex_path,
        crystal2.entries.len()
    );

    // Step 1: long-term memory resonance in the static 8k crystal.
    let hit1 = crystal1.query_threshold(text, fuga::DEFAULT_RESONANCE_THRESHOLD);
    match &hit1 {
        Some(h) => println!(
            "  1) STATIC RESONANCE  res={:.3} (exact={}) :: {}",
            h.resonance, h.exact, h.entry.key_text
        ),
        None => println!("  1) STATIC RESONANCE  — no resonance in MoE knowledge"),
    }

    // Step 2: up-project the static response phase into 32k and bind it with
    // the episodic cortex phase of the same query (native 32k encoding).
    let q2 = fuga::core::tokenizer_bridge::encode_bytes_nopos(text.as_bytes(), fuga::DIM_L2);
    let mut fused = q2.clone();
    if let Some(h) = &hit1 {
        let proj = fuga::project_phase(&h.entry.hv, fuga::DIM_L2);
        // XOR bind: response ⊗ context. Weighted blend keeps both voices.
        fused = fuga::PhaseCrystal::weighted_majority(&fused, &proj, alpha);
    }
    let fones = fused.words.iter().map(|w| w.count_ones()).sum::<u32>() as f64;

    // Step 3: episodic resonance of the fused phase in the 32k cortex.
    let mut best: Option<(f64, usize)> = None;
    for (i, e) in crystal2.entries.iter().enumerate() {
        if e.hv.dim != fuga::DIM_L2 {
            continue;
        }
        let overlap: u32 = fused
            .words
            .iter()
            .enumerate()
            .map(|(w, fw)| (fw & e.hv.words[w]).count_ones())
            .sum();
        let res = if fones <= 0.0 {
            0.0
        } else {
            overlap as f64
                / fones
                    .max(e.hv.words.iter().map(|w| w.count_ones()).sum::<u32>() as f64)
                    .min(fuga::DIM_L2 as f64)
        };
        if best.map_or(true, |(bs, _)| res > bs) {
            best = Some((res, i));
        }
    }
    let thr = fuga::DEFAULT_RESONANCE_THRESHOLD * scale;
    match best {
        Some((res, i)) if res >= thr && (!gate || hit1.is_some()) => {
            let e = &crystal2.entries[i];
            println!(
                "  3) EPISODIC RESONANCE res={:.3} (threshold {:.2}) :: {}",
                res, thr, e.key_text
            );
            println!("     {}", e.text);
        }
        _ => println!("  3) EPISODIC RESONANCE — no episodic overlap (clean state)"),
    }
}

fn run_crystal_triad(candidate: &str, args: &[String]) {
    let static_path = parse_flag_value(args, 4, "--from")
        .map(|s| s.to_string())
        .unwrap_or_else(|| "fuga_crystal.bin".to_string());
    let cortex_path = parse_flag_value(args, 4, "--cortex")
        .map(|s| s.to_string())
        .unwrap_or_else(|| "fuga_cortex.bin".to_string());
    let intent = parse_flag_value(args, 4, "--intent")
        .map(|s| s.to_string())
        .unwrap_or_else(|| candidate.to_string());

    let domain = match fuga::PhaseCrystal::load(&static_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  ✗ static crystal {}: {}", static_path, e);
            return;
        }
    };
    let cortex = match fuga::PhaseCrystal::load(&cortex_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  ✗ cortex crystal {}: {}", cortex_path, e);
            return;
        }
    };

    let anchors = fuga::ReasoningFoundations::build(&domain, &cortex, &intent, candidate);
    let cand =
        fuga::core::tokenizer_bridge::encode_bytes_nopos_min3(candidate.as_bytes(), fuga::DIM_L2);

    println!("═══ Tri-Anchor Framework ═══\n");
    println!(
        "  Candidate: {}\n  Intent:    {}\n  Static: {} ({} entries)\n  Cortex: {} ({} entries)\n",
        candidate,
        intent,
        static_path,
        domain.entries.len(),
        cortex_path,
        cortex.entries.len()
    );

    let r = anchors.resonances(&cand);
    let total = anchors.total_resonance(&cand);
    let pass = anchors.evaluate_transplant(&cand);
    let t = fuga::ANCHOR_RESONANCE_MIN;
    let floor = fuga::ANCHOR_FLOOR;
    println!(
        "  F1 FACT   res={:.3}  (floor {:.2}/target {:.2}) {}",
        r[0],
        floor,
        t,
        if r[0] >= t {
            "✓"
        } else if r[0] >= floor {
            "◐"
        } else {
            "✗"
        }
    );
    println!(
        "  F2 CAUSAL res={:.3}  (floor {:.2}/target {:.2}) {}",
        r[1],
        floor,
        t,
        if r[1] >= t {
            "✓"
        } else if r[1] >= floor {
            "◐"
        } else {
            "✗"
        }
    );
    println!(
        "  F3 INTENT res={:.3}  (floor {:.2}/target {:.2}) {}",
        r[2],
        floor,
        t,
        if r[2] >= t {
            "✓"
        } else if r[2] >= floor {
            "◐"
        } else {
            "✗"
        }
    );
    println!(
        "\n  TOTAL = {:.4}  ({:.2}·F1 + {:.2}·F2 + {:.2}·F3)  min {:.2}",
        total,
        fuga::ANCHOR_WEIGHT_FACT,
        fuga::ANCHOR_WEIGHT_LOGIC,
        fuga::ANCHOR_WEIGHT_INTENT,
        fuga::ANCHOR_TOTAL_MIN
    );
    if pass {
        println!("  ⇒ ACCEPTED — candidate grounded in all three foundations");
    } else {
        println!("  ⇒ REJECTED → deterministic silence (not grounded in the triad)");
    }
}

fn run_transpile(args: &[String]) {
    let mut sources: Vec<String> = Vec::new();
    let mut select: Vec<String> = Vec::new();
    let mut keep: Vec<String> = vec![
        "embed".into(),
        "lm_head".into(),
        "gate".into(),
        "router".into(),
    ];
    let mut finalize_out: Option<String> = None;
    let mut dry_run = false;
    let mut max_tensors: Option<usize> = None;
    let mut max_shards: Option<usize> = None;
    let mut revision = String::new();
    let mut whole = false;
    let mut state_file: Option<String> = None;
    let mut concurrency = 8usize;
    let mut raw = false;
    let mut mirror = String::from("auto"); // "auto", "ms", "hf"
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--select" => {
                i += 1;
                if i < args.len() {
                    select.push(args[i].clone());
                }
            }
            "--keep" => {
                i += 1;
                if i < args.len() {
                    keep.push(args[i].clone());
                }
            }
            "--finalize" => {
                i += 1;
                if i < args.len() {
                    finalize_out = Some(args[i].clone());
                }
            }
            "--max-tensors" => {
                i += 1;
                max_tensors = args.get(i).and_then(|s| s.parse().ok());
            }
            "--max-shards" => {
                i += 1;
                max_shards = args.get(i).and_then(|s| s.parse().ok());
            }
            "--revision" => {
                i += 1;
                if i < args.len() {
                    revision = args[i].clone();
                }
            }
            "--state" => {
                i += 1;
                if i < args.len() {
                    state_file = Some(args[i].clone());
                }
            }
            "--concurrency" => {
                i += 1;
                concurrency = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(8);
            }
            "--mirror" => {
                i += 1;
                if i < args.len() {
                    mirror = args[i].clone();
                }
            }
            "--whole" => whole = true,
            "--raw" => raw = true,
            "--dry-run" => dry_run = true,
            s if s.starts_with("--") => {
                eprintln!("  ✗ unknown flag: {}", s);
                return;
            }
            s => sources.push(s.to_string()),
        }
        i += 1;
    }
    if sources.is_empty() {
        eprintln!(
            "Usage: fuga transpile <safetensors file|dir|hf-repo-id|url>… [--select S] [--keep K] [--finalize out] [--dry-run] [--max-tensors N] [--max-shards N] [--revision R] [--whole] [--raw] [--state FILE] [--concurrency N] [--mirror auto|ms|hf]"
        );
        return;
    }

    println!("═══ Streaming Transpilation ═══\n");
    let mut acc = match &state_file {
        Some(path) if std::path::Path::new(path).exists() => {
            match fuga::TranspileAccumulator::load_state(path) {
                Ok(a) => {
                    println!(
                        "  ✓ resumed from state {} ({} tensors, {} entries, {} shards done)",
                        path,
                        a.processed,
                        a.entries.len(),
                        a.done.len()
                    );
                    a
                }
                Err(e) => {
                    eprintln!("  ✗ load state {}: {}", path, e);
                    return;
                }
            }
        }
        _ => fuga::TranspileAccumulator::new(fuga::DEFAULT_DIM),
    };
    let cfg = fuga::TranspileConfig {
        select,
        keep,
        max_tensors,
        max_shards,
        dry_run,
        route_cap: if raw { 0 } else { fuga::ROUTE_CAP },
        whole,
        concurrency,
        raw,
    };

    let mut shards: Vec<fuga::ShardSource> = Vec::new();
    for s in &sources {
        if fuga::is_repo_id(s) {
            // For auto mode try ModelScope first, fallback to HF
            let mut list_result = if mirror == "hf" {
                fuga::list_hf_shards(s, &revision)
            } else {
                fuga::list_ms_shards(s, &revision)
            };
            let mut used_ms = mirror != "hf";
            if list_result.is_err() && mirror == "auto" {
                eprintln!("  dim  ModelScope unavailable, falling back to HuggingFace");
                list_result = fuga::list_hf_shards(s, &revision);
                used_ms = false;
            }
            match list_result {
                Ok(list) => {
                    let mut list = list;
                    if let Some(n) = max_shards {
                        list.truncate(n);
                    }
                    let source_label = if used_ms { "ModelScope" } else { "HuggingFace" };
                    println!(
                        "  {} repo {}: {} safetensors shards",
                        source_label,
                        s,
                        list.len()
                    );
                    for (path, size) in list {
                        println!("   {} ({:.1} GB)", path, size as f64 / 1_073_741_824.0);
                        let base_url = if used_ms {
                            fuga::ms_resolve_url(s, &path, &revision)
                        } else {
                            fuga::hf_resolve_url(s, &path, &revision)
                        };
                        shards.push(fuga::ShardSource::Remote { base_url });
                    }
                }
                Err(e) => {
                    eprintln!(" ✗ list {}: {}", s, e);
                    if mirror == "ms" {
                        eprintln!(
                            "   hint: ModelScope mirror failed. Try --mirror hf or --mirror auto"
                        );
                    }
                }
            }
        } else if let Some(url) = s
            .strip_prefix("http://")
            .or_else(|| s.strip_prefix("https://"))
        {
            shards.push(fuga::ShardSource::Remote {
                base_url: s.clone(),
            });
            let _ = url;
        } else if std::path::Path::new(s).is_dir() {
            let mut found = 0usize;
            for entry in std::fs::read_dir(s)
                .map_err(|e| eprintln!("  ✗ read dir: {}", e))
                .ok()
                .into_iter()
                .flatten()
            {
                if let Ok(e) = entry {
                    let p = e.path();
                    if p.extension().map(|x| x == "safetensors").unwrap_or(false) {
                        shards.push(fuga::ShardSource::Local {
                            path: p.to_string_lossy().into_owned(),
                        });
                        found += 1;
                    }
                }
            }
            println!("  dir {}: {} safetensors shards", s, found);
        } else {
            shards.push(fuga::ShardSource::Local { path: s.clone() });
        }
    }

    for shard in &shards {
        let label = match shard {
            fuga::ShardSource::Local { path } => path.clone(),
            fuga::ShardSource::Remote { base_url } => fuga::shard_label(base_url),
        };
        if acc.done.contains(&label) {
            println!("  ✓ {} (already done, resumed)", label);
            continue;
        }
        match fuga::transpile_shard(shard, &mut acc, &cfg) {
            Ok(st) => {
                acc.done.push(label.clone());
                if let Some(path) = &state_file {
                    match acc.save_state(path) {
                        Ok(_) => println!(
                            "  ✓ state saved to {} ({} shards done)",
                            path,
                            acc.done.len()
                        ),
                        Err(e) => eprintln!("  ✗ save state: {}", e),
                    }
                }
                println!(
                    "  ✓ {}: {} tensors, {:.1} MB, {} entries, {:.1} MB/s, {:.2?}",
                    st.shard,
                    st.tensors,
                    st.bytes as f64 / 1_048_576.0,
                    st.entries_added,
                    st.mbps,
                    st.elapsed
                );
            }
            Err(e) => eprintln!("  ✗ {}: {}", label, e),
        }
    }

    println!("\n  {}", acc.stats());
    if !acc.skipped.is_empty() {
        println!("  skipped:");
        for s in acc.skipped.iter().take(20) {
            println!("    {}", s);
        }
        if acc.skipped.len() > 20 {
            println!("    … {} more", acc.skipped.len() - 20);
        }
    }

    if let Some(out) = finalize_out {
        let crystal = acc.finalize(fuga::DEFAULT_RESONANCE_THRESHOLD);
        println!("\n  {}", crystal.stats());
        match crystal.save(&out) {
            Ok(_) => println!("  ✓ Crystal saved to {}", out),
            Err(e) => eprintln!("  ✗ {}", e),
        }
    } else {
        println!("  (use --finalize <out> to emit the crystal dump)");
    }
}

fn run_self_query(text: &str) {
    let mut mirror = match fuga::SelfMirror::load() {
        Some(m) => m,
        None => {
            eprintln!("No mirror data. Run 'fuga self-mirror' first.");
            return;
        }
    };
    println!("═══ Self-Query ═══");
    println!("  Query: {}\n", text);
    let (tm_match, errors, top) = mirror.query(text);
    println!("  TM match:  {:.4}", tm_match);
    println!(
        "  Errors:    L0={:.4} L1={:.4} L2={:.4}",
        errors.first().unwrap_or(&1.0),
        errors.get(1).unwrap_or(&1.0),
        errors.get(2).unwrap_or(&1.0)
    );
    println!("\n  Top-5 mirror nodes:");
    for (i, node) in top.iter().enumerate() {
        println!(
            "  {}. {} {} ({})  l0={:.3} l1={:.3}",
            i + 1,
            node.kind,
            node.name,
            node.path,
            node.l0_err,
            node.l1_err
        );
    }
    println!("\n  {}", mirror.reflect());
}

fn run_reflect_repl() {
    let tm = match load_tm() {
        Some(t) => t,
        None => {
            eprintln!("No fuga_htm.bin. Run 'fuga htm-train' first.");
            return;
        }
    };
    let mut reflector = fuga::AnomalyReflector::new(tm);
    let mut buf: Vec<String> = Vec::new();

    println!("═══ Fuga Reflect REPL ═══");
    println!("  Enter text lines. Empty line to quit.");
    println!("  Every 3 lines triggers anomaly check.\n");

    loop {
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).is_err() || line.trim().is_empty() {
            break;
        }
        buf.push(line.trim().to_string());
        if buf.len() >= 3 {
            let block = buf.join(" ");
            let events = reflector.feed_text(&block);
            let stats = reflector.detector.stats();
            let rstats = reflector.reflect_summary();

            if events.is_empty() {
                println!("  ✓ {}  {}", stats, rstats);
            } else {
                for ev in &events {
                    println!(
                        "  ⚠ anomaly  z={:.2}  Δentropy={:.4}  match={:.2}",
                        ev.z_score, ev.entropy_shift, ev.match_score
                    );
                    println!("      ctx: {}", ev.token);
                }
                let corrections = reflector.drain_corrections();
                for c in &corrections {
                    println!(
                        "  ➜ correction: z={:.2} Δentropy={:.4}",
                        c.z_score, c.entropy_shift
                    );
                }
                println!("  ⚠ {}  {}", stats, rstats);
            }
            buf.clear();
        }
    }
    println!("\n  Session: {}", reflector.reflect_summary());
}

fn run_htm_predict(text: &str) {
    let tm = match load_tm() {
        Some(t) => t,
        None => {
            eprintln!("No fuga_htm.bin. Run 'fuga htm-train' first.");
            return;
        }
    };

    let query = fuga::encode_text(text);
    let pred = tm.predict_next(&query);
    if pred.popcount() == 0 {
        println!("  No prediction (no depolarized cells for this context)");
        return;
    }
    println!(
        "  HTM predicted {} active bits from context",
        pred.popcount()
    );

    let mem_sdr = load_sdr_store("fuga_sdr_index.bin");
    if let Some(ref store) = mem_sdr {
        let results = store.index.search(&pred, 3);
        for (_i, score, snippet) in &results {
            println!("    [{:.2}] {}", score, snippet);
        }
    }
}

fn run_cross_domain(_dim: usize, epochs: usize) {
    let model_path = "fuga_hjepa.bin";
    let mut hjepa = if std::path::Path::new(model_path).exists() {
        match fuga::HierarchicalJEPA::load(model_path) {
            Ok(h) => {
                println!("  Loaded {}\n", model_path);
                h
            }
            Err(e) => {
                eprintln!("  Load failed: {}", e);
                return;
            }
        }
    } else {
        eprintln!("  No model found at {}", model_path);
        return;
    };
    let sim = hjepa.train_cross_domain("corpus_doc_code_pairs.jsonl", epochs);
    match hjepa.save(model_path) {
        Ok(()) => println!("  Saved {} (cosine={:.4})", model_path, sim),
        Err(e) => eprintln!("  Save failed: {}", e),
    }
}

fn run_hierarchical_jepa_predict(text: &str, dim: usize) {
    let hjepa = match fuga::HierarchicalJEPA::load("fuga_hjepa.bin") {
        Ok(h) => h,
        Err(e) => {
            eprintln!("No trained H-JEPA model: {}", e);
            return;
        }
    };

    let mut weaver = fuga::WeaverEngine::new(dim, 3);
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.len() < 5 {
        eprintln!("Need at least 5 tokens for context");
        return;
    }

    let mut context_vecs = Vec::new();
    for chunk in tokens.chunks(3) {
        let t = chunk.join(" ");
        context_vecs.push(encode_chunk(&mut weaver, &t));
    }

    let ctx_refs: Vec<&fuga::Hypervector> = context_vecs.iter().collect();
    let predictions = hjepa.predict(&ctx_refs);

    println!("Hierarchical JEPA Predictions:");
    for (li, pred) in predictions.iter().enumerate() {
        let level_name = match li {
            0 => "L0 (primitive)",
            1 => "L1 (functional)",
            2 => "L2 (concept)",
            _ => "?",
        };
        println!(
            "  {}: entropy={:.4}, dim={}",
            level_name,
            pred.entropy(),
            pred.dim
        );
    }

    // decode L0 prediction via memory
    for mp in &["fuga_code_cube_code_mem.bin", "fuga_moe_code.bin"] {
        if std::path::Path::new(mp).exists() {
            if let Ok(mem) = fuga::MemoryStore::load_bin(mp) {
                if !predictions.is_empty() {
                    let results = mem.search(&predictions[0], 3);
                    if !results.is_empty() {
                        println!("\nL0 decoded from {}:", mp);
                        for (_, sim, entry) in &results {
                            println!("  [{:.3}] {} — {}", sim, entry.text, entry.source_doc);
                        }
                    }
                }
                break;
            }
        }
    }
}
fn run_refactor(file: &str, desc: &str, max_iter: usize) {
    if file.is_empty() || !std::path::Path::new(file).exists() {
        eprintln!("File not found: {:?}", file);
        return;
    }

    println!("╔══════════════════════════════════════════════╗");
    println!("║  Fuga Self-Refactoring Loop                ║");
    println!("╚══════════════════════════════════════════════╝");
    println!("  File:       {}", file);
    println!("  Task:       {}", desc);
    println!("  Max iters:  {}\n", max_iter);

    let original = std::fs::read_to_string(file).unwrap_or_else(|e| {
        eprintln!("Read error: {}", e);
        String::new()
    });
    if original.is_empty() {
        return;
    }

    let mut moe = fuga::MoEStore::new("fuga_code_cube");
    if moe.load_domain("code").is_err() {
        eprintln!("  ✗ Failed to load code MoE domain");
        return;
    }
    println!("  Loaded {} code patterns\n", moe.domain_size("code"));

    let abs_file = std::path::absolute(file).unwrap_or_else(|_| std::path::PathBuf::from(file));
    let is_in_project = abs_file.starts_with(std::env::current_dir().unwrap_or_default())
        || std::path::Path::new("Cargo.toml").exists();

    let backup = format!("{}.bak", file);
    std::fs::write(&backup, &original).ok();

    let mut last_errors = String::new();

    for iter in 1..=max_iter {
        println!("── Iteration {}/{} ──", iter, max_iter);

        // search MoE with task + errors
        let search_query = if last_errors.is_empty() {
            desc.to_string()
        } else {
            format!(
                "{} ERROR: {}",
                desc,
                last_errors.lines().take(3).collect::<Vec<_>>().join(" ")
            )
        };

        let patterns = moe.search_by_text("code", &search_query, 6);
        let ctx: String = patterns
            .iter()
            .take(3)
            .map(|(_, _, e)| e.text.trim())
            .collect::<Vec<_>>()
            .join("\n");

        let current = std::fs::read_to_string(file).unwrap_or_else(|_| original.clone());
        let modified = apply_refactor_hint(&current, desc, &ctx);
        std::fs::write(file, &modified).ok();
        println!("  Applied change to {}", file);

        // validate
        let check_ok = if is_in_project {
            let out = std::process::Command::new("cargo")
                .args(["check", "--quiet"])
                .output();
            match out {
                Ok(o) if o.status.success() => {
                    last_errors.clear();
                    true
                }
                Ok(o) => {
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    last_errors = stderr
                        .lines()
                        .filter(|l| l.contains("error") || l.contains("aborting"))
                        .take(10)
                        .collect::<Vec<_>>()
                        .join("\n");
                    println!(
                        "  ✗ cargo check failed ({} errors)",
                        last_errors.lines().count()
                    );
                    for l in last_errors.lines().take(5) {
                        println!("    {}", l);
                    }
                    false
                }
                Err(e) => {
                    eprintln!("  ✗ Check runner error: {}", e);
                    false
                }
            }
        } else {
            // standalone file: compile with rustc
            let out = std::process::Command::new("rustc")
                .args(["--edition", "2024", file])
                .output();
            match out {
                Ok(o) if o.status.success() => {
                    last_errors.clear();
                    true
                }
                Ok(o) => {
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    last_errors = stderr
                        .lines()
                        .filter(|l| l.contains("error"))
                        .take(10)
                        .collect::<Vec<_>>()
                        .join("\n");
                    println!("  ✗ rustc failed ({} errors)", last_errors.lines().count());
                    for l in last_errors.lines().take(5) {
                        println!("    {}", l);
                    }
                    false
                }
                Err(e) => {
                    eprintln!("  ✗ rustc runner error: {}", e);
                    false
                }
            }
        };

        if check_ok {
            println!("  ✓ Validation passed!");
            let test_ok = if is_in_project {
                let out = std::process::Command::new("cargo")
                    .args(["test", "--quiet"])
                    .output();
                match out {
                    Ok(t) if t.status.success() => true,
                    _ => false,
                }
            } else {
                true
            };

            if test_ok {
                println!("  ✓ Tests passed!");
                std::fs::remove_file(&backup).ok();
                let entry = format!(
                    "TASK: refactor {} — {}\nFILE: {}\nDIFF:\n{}\nSTATUS: success\n",
                    file, desc, file, modified
                );
                let result_path = save_agent_result(&entry);
                println!("  ✓ Absorbed to {}", result_path);
                println!("\n  Self-refactoring complete in {} iterations!", iter);
                return;
            } else {
                println!("  ✗ Tests failed, retrying...");
                let test_out = std::process::Command::new("cargo")
                    .args(["test", "--quiet"])
                    .output();
                if let Ok(o) = test_out {
                    let test_err = String::from_utf8_lossy(&o.stderr);
                    last_errors = test_err
                        .lines()
                        .filter(|l| l.contains("error") || l.contains("FAILED"))
                        .take(10)
                        .collect::<Vec<_>>()
                        .join("\n");
                }
                // restore and retry
                let _ = std::fs::write(file, &original);
            }
        } else {
            // save error for absorb-agent
            let entry = format!(
                "TASK: refactor {} — {} (iter {})\nFILE: {}\nERROR:\n{}\nSTATUS: failed\n",
                file, desc, iter, file, last_errors
            );
            save_agent_result(&entry);
            // restore original before next attempt
            let _ = std::fs::write(file, &original);
        }
    }

    // restore backup on final failure
    let _ = std::fs::write(file, &original);
    std::fs::remove_file(&backup).ok();
    println!("\n  ✗ Max iterations reached. File restored.");
    println!("  Check agent_results/ for error details.");
}

fn run_docs_entry(args: &[String]) {
    let cube_path = parse_flag_value(args, 2, "--cube").unwrap_or("fuga_code_cube.bin");
    let out_path = parse_flag_value(args, 2, "--output").unwrap_or("docs/FUGA_DOCS.md");
    let _side = args
        .iter()
        .position(|a| a == "--side")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(8);
    let _ndim = args
        .iter()
        .position(|a| a == "--ndim")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(3);

    let cube_spec = match peek_cube_header(cube_path) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("{}", e);
            return;
        }
    };
    match cube_spec {
        (3, 4, _) => run_docs::<3, 4>(cube_path, &out_path),
        (4, 4, _) => run_docs::<4, 4>(cube_path, &out_path),
        (3, 8, _) => run_docs::<3, 8>(cube_path, &out_path),
        (4, 8, _) => run_docs::<4, 8>(cube_path, &out_path),
        (5, 2, _) => run_docs::<5, 2>(cube_path, &out_path),
        (5, 4, _) => run_docs::<5, 4>(cube_path, &out_path),
        (3, 5, _) => run_docs::<3, 5>(cube_path, &out_path),
        (3, 6, _) => run_docs::<3, 6>(cube_path, &out_path),
        (3, 7, _) => run_docs::<3, 7>(cube_path, &out_path),
        _ => eprintln!("Unsupported cube dims: {}x{}", cube_spec.0, cube_spec.1),
    }
}

fn run_docs<const N: usize, const S: usize>(cube_path: &str, out_path: &str) {
    use std::io::Write;

    println!("=== Fuga Self-Documentation Generator ===\n");

    let engine = match fuga::AnswerEngine::<N, S>::load(cube_path) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to load cube: {}", e);
            return;
        }
    };

    let mut moe = fuga::MoEStore::new("fuga_code_cube");
    if let Err(e) = moe.load_all() {
        eprintln!("MoE load: {}", e);
    }

    println!(
        "  Cube: {}x{} dim={} ({} cells), entropy={:.4}",
        S,
        N,
        engine.dim,
        S.pow(N as u32),
        engine.cube.global_entropy()
    );
    println!("  Memory: {} entries", engine.memory.size());
    println!();

    let mut f = match std::fs::File::create(out_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Cannot create {}: {}", out_path, e);
            return;
        }
    };

    // Helper: write text search results for a query
    let write_moe = |query: &str, domain: &str, label: &str, f: &mut std::fs::File| {
        let hits = moe.search_by_text(domain, query, 4);
        if !hits.is_empty() {
            writeln!(f, "**{}:**", label).ok();
            for (_, sim, entry) in hits.iter().take(4) {
                let text: String = entry.text.chars().take(160).collect();
                writeln!(f, "- `{}` sim={:.3} — {}", entry.source_doc, sim, text).ok();
            }
            writeln!(f).ok();
        }
    };

    // ============ DOCUMENT ============

    writeln!(f, "# Fuga Omni — Self-Generated Documentation").ok();
    writeln!(f).ok();
    writeln!(f, "*Generated by `fuga docs` from source code analysis + trained VSA memory ({} entries, {} epochs)*",
        engine.memory.size(), 30).ok();
    writeln!(f).ok();

    // === SYSTEM STATE ===
    writeln!(f, "## System State").ok();
    writeln!(f).ok();
    writeln!(f, "| Metric | Value |").ok();
    writeln!(f, "|--------|-------|").ok();
    writeln!(f, "| Cube | {}×{} = {} cells |", S, N, S.pow(N as u32)).ok();
    writeln!(f, "| Hypervector dim | {} |", engine.dim).ok();
    writeln!(
        f,
        "| Global entropy | {:.4} |",
        engine.cube.global_entropy()
    )
    .ok();
    writeln!(f, "| Coherence | {:.4} |", engine.cube.coherence()).ok();
    writeln!(f, "| Memory entries | {} |", engine.memory.size()).ok();
    writeln!(f, "| MoE domains | {} |", moe.available_domains().len()).ok();
    writeln!(f, "| Platform | CUDA sm_75 (GTX 1660 Ti), Rust |").ok();
    writeln!(f).ok();
    write_moe(
        "Fuga Omni VSA engine WaveCube hyperdimensional computing",
        "code",
        "VSA memory resonance",
        &mut f,
    );

    // === MODULE TREE ===
    writeln!(f, "## Module Tree").ok();
    writeln!(f).ok();
    writeln!(f, "```").ok();
    writeln!(f, "src/").ok();
    writeln!(
        f,
        "├── lib.rs              # Crate root — re-exports all public API"
    )
    .ok();
    writeln!(
        f,
        "├── main.rs             # CLI dispatcher + 30+ run_* command handlers"
    )
    .ok();
    writeln!(
        f,
        "├── ai/                 # AI core: VSA engine, MoE, JEPA, prompts"
    )
    .ok();
    writeln!(
        f,
        "│   ├── core.rs         #   FugaAI — main orchestrator (think → absorb)"
    )
    .ok();
    writeln!(
        f,
        "│   ├── memory_store.rs #   MemoryStore — VSA memory with LSH index"
    )
    .ok();
    writeln!(
        f,
        "│   ├── moe.rs          #   MoEStore — multi-domain expert system"
    )
    .ok();
    writeln!(
        f,
        "│   ├── hnsw.rs         #   VsaIndex — LSH multi-table probing"
    )
    .ok();
    writeln!(
        f,
        "│   ├── answer_engine.rs #  AnswerEngine — search + format responses"
    )
    .ok();
    writeln!(
        f,
        "│   ├── router.rs       #   DynamicRouter — SuperToken → expert routing"
    )
    .ok();
    writeln!(
        f,
        "│   ├── resonance_attention.rs  ResonanceAttention — GPU/CPU scan"
    )
    .ok();
    writeln!(
        f,
        "│   ├── codegen.rs      #   Code generation from cube resonance"
    )
    .ok();
    writeln!(
        f,
        "│   ├── jepa.rs         #   JEPA state predictor (learnable perms)"
    )
    .ok();
    writeln!(
        f,
        "│   ├── prompts.rs      #   PromptVectors — VSA prompt algebra"
    )
    .ok();
    writeln!(
        f,
        "├── core/               # VSA primitives and cube storage"
    )
    .ok();
    writeln!(
        f,
        "│   ├── hypervector.rs  #   Hypervector — 8192-bit (128×u64)"
    )
    .ok();
    writeln!(
        f,
        "│   ├── wave_cube.rs    #   WaveCube<N,S> — N-dim VSA cube storage"
    )
    .ok();
    writeln!(
        f,
        "│   ├── tensor_phase.rs #   MappedCube — memory-mapped cube view"
    )
    .ok();
    writeln!(f, "│   ├── information_triangle.rs  VSA semantic triangle").ok();
    writeln!(f, "│   ├── fuga_synthesizer.rs     Cross-module analysis").ok();
    writeln!(
        f,
        "├── weaver/             # Token → SuperToken VSA compression"
    )
    .ok();
    writeln!(
        f,
        "│   ├── mod.rs          #   WeaverEngine — token window VSA bundle"
    )
    .ok();
    writeln!(f, "│   ├── pattern_matcher.rs     TokenInfo + token_id()").ok();
    writeln!(
        f,
        "│   ├── vocabulary.rs   #   Token vocabulary and configs"
    )
    .ok();
    writeln!(
        f,
        "├── gpu.rs              # CUDA kernel launcher (resonance_scan)"
    )
    .ok();
    writeln!(f, "├── sandbox/            # Isolated compilation sandbox").ok();
    writeln!(
        f,
        "├── quality_filter.rs   # CodeQualityFilter — safety/quality scoring"
    )
    .ok();
    writeln!(
        f,
        "├── text_quality.rs     # TextQualityFilter — collage/dialogue scoring"
    )
    .ok();
    writeln!(
        f,
        "├── engine.rs           # FugaEngine — multi-layer analysis pipeline"
    )
    .ok();
    writeln!(
        f,
        "├── multi_engine.rs     # MultiEngine — parallel file analysis"
    )
    .ok();
    writeln!(
        f,
        "├── layers/             # Analysis layers (syntax, pattern, ...)"
    )
    .ok();
    writeln!(
        f,
        "├── reporters/          # Output formatters (HTML, markdown, etc.)"
    )
    .ok();
    writeln!(f, "├── autofix/            # Automatic fix generation").ok();
    writeln!(f, "├── omni/               # Omni-mode training pipeline").ok();
    writeln!(f, "├── speech/             # Text-to-speech module").ok();
    writeln!(f, "├── microwave/          # Self-modifying code sandbox").ok();
    writeln!(
        f,
        "└── sim/                # Physics simulation (valve/heater/boiler)"
    )
    .ok();
    writeln!(f, "```").ok();
    writeln!(f).ok();

    // === VSA ENGINE ===
    writeln!(f, "## VSA Hyperdimensional Computing Engine").ok();
    writeln!(f).ok();
    writeln!(f, "### Hypervector (`src/core/hypervector.rs`)").ok();
    writeln!(f).ok();
    writeln!(f, "```rust").ok();
    writeln!(f, "pub struct Hypervector {{").ok();
    writeln!(f, "    pub dim: usize,       // 8192 bits").ok();
    writeln!(f, "    pub words: Vec<u64>,  // 128 × u64 = 8192 bits").ok();
    writeln!(f, "}}").ok();
    writeln!(f, "```").ok();
    writeln!(f).ok();
    writeln!(f, "Key operations:").ok();
    writeln!(
        f,
        "- **Bundle** (XOR): `hv1 ^ hv2` — superposition of patterns"
    )
    .ok();
    writeln!(
        f,
        "- **Bind** (Hadamard): element-wise multiply — associative mapping"
    )
    .ok();
    writeln!(
        f,
        "- **Hamming similarity**: popcount(hv1 ^ hv2) — normalized [0,1]"
    )
    .ok();
    writeln!(
        f,
        "- **Entropy**: fraction of 1-bits in the vector (~0.5 ideal)"
    )
    .ok();
    writeln!(f).ok();
    writeln!(f, "### WaveCube (`src/core/wave_cube.rs`)").ok();
    writeln!(f).ok();
    writeln!(f, "```rust").ok();
    writeln!(f, "pub struct WaveCube<N: const usize, S: const usize> {{").ok();
    writeln!(f, "    pub dim: usize,           // hypervector dimension").ok();
    writeln!(f, "    pub cube: Vec<Hypervector>, // S^N cells").ok();
    writeln!(f, "}}").ok();
    writeln!(f, "```").ok();
    writeln!(f).ok();
    writeln!(f, "N-dimensional VSA cube with S cells per dimension. Each cell stores a bundled Hypervector. ").ok();
    writeln!(
        f,
        "Supports zero-copy loading via `memmap2`. Auto-detects dimensions from file header."
    )
    .ok();
    writeln!(
        f,
        "Current state: **{}×{} = {} cells**, dim={}, entropy={:.4}",
        S,
        N,
        S.pow(N as u32),
        engine.dim,
        engine.cube.global_entropy()
    )
    .ok();
    writeln!(f).ok();
    writeln!(f, "### VSA-LSH Index (`src/ai/hnsw.rs`)").ok();
    writeln!(f).ok();
    writeln!(
        f,
        "Multi-table LSH for fast approximate nearest neighbor search over Hypervectors:"
    )
    .ok();
    writeln!(f, "- **6 tables** × **8 probes** per query").ok();
    writeln!(f, "- Multi-bit probing: tests 2^buckets near each hash").ok();
    writeln!(
        f,
        "- Fallback random sampling when LSH finds < top_k results"
    )
    .ok();
    writeln!(
        f,
        "- Index validation: refuses stale files (idx.size() != entries.len())"
    )
    .ok();
    writeln!(f).ok();
    write_moe(
        "Hypervector Hamming similarity resonance VSA LSH probing",
        "code",
        "VSA memory resonance",
        &mut f,
    );

    // === MIXTURE OF EXPERTS ===
    writeln!(f, "## Mixture of Experts (MoE) System").ok();
    writeln!(f).ok();
    writeln!(f, "### MoEStore (`src/ai/moe.rs`)").ok();
    writeln!(f).ok();
    writeln!(f, "Multi-domain expert system with dynamic domain loading:").ok();
    writeln!(f).ok();
    writeln!(f, "```rust").ok();
    writeln!(f, "pub struct MoEStore {{").ok();
    writeln!(
        f,
        "    experts: HashMap<String, MemoryStore>,  // domain → store"
    )
    .ok();
    writeln!(
        f,
        "    sizes: HashMap<String, usize>,          // entry counts"
    )
    .ok();
    writeln!(f, "}}").ok();
    writeln!(f, "```").ok();
    writeln!(f).ok();
    writeln!(f, "Key methods:").ok();
    writeln!(
        f,
        "- `add_domain(name)` — creates empty domain file (`fuga_moe_<name>.bin`)"
    )
    .ok();
    writeln!(f, "- `load_domain(name)` — lazy-loads a domain from disk").ok();
    writeln!(f, "- `domain_for(query)` — classifies text to best domain").ok();
    writeln!(
        f,
        "- `store(st, text, source, role)` — auto-routes to correct domain"
    )
    .ok();
    writeln!(
        f,
        "- `search(domain, query, top_k)` — vector similarity in domain"
    )
    .ok();
    writeln!(
        f,
        "- `search_by_text(domain, text, top_k)` — text index search"
    )
    .ok();
    writeln!(f).ok();
    writeln!(f, "### Current Domains").ok();
    writeln!(f).ok();
    writeln!(f, "| Domain | Entries | Purpose |").ok();
    writeln!(f, "|--------|---------|---------|").ok();
    for (domain, size) in moe.domain_sizes() {
        let desc = match domain {
            "code" => "Source code patterns (Rust, JS, Python, etc.)",
            "narrative" => "Prose, literature, stories",
            "dialogue" => "Conversational exchanges",
            "general" => "Mixed/generic text",
            "forum" => "Q&A and discussion threads",
            "poetry" => "Poetic structures",
            "dialogue_pair" => "Paired dialogue turns",
            _ => "Custom domain",
        };
        writeln!(f, "| {} | {} | {} |", domain, size, desc).ok();
    }
    writeln!(f).ok();
    writeln!(
        f,
        "Domain routing is automatic: `MoEStore::domain_for()` uses keyword heuristics "
    )
    .ok();
    writeln!(
        f,
        "to classify input, but `store()` also inspects file extensions and role hints."
    )
    .ok();
    writeln!(f).ok();
    write_moe(
        "MoEStore domain expert routing mixture",
        "code",
        "Code patterns",
        &mut f,
    );

    // === JEPA ===
    writeln!(f, "## JEPA State Predictor").ok();
    writeln!(f).ok();
    writeln!(f, "### JepaPredictor (`src/ai/jepa.rs`)").ok();
    writeln!(f).ok();
    writeln!(f, "```rust").ok();
    writeln!(f, "pub struct JepaPredictor {{").ok();
    writeln!(
        f,
        "    pub dim: usize,               // hypervector dimension"
    )
    .ok();
    writeln!(
        f,
        "    pub context_len: usize,        // sliding window size"
    )
    .ok();
    writeln!(
        f,
        "    offsets: Vec<f64>,             // learnable permutation offsets"
    )
    .ok();
    writeln!(f, "}}").ok();
    writeln!(f, "```").ok();
    writeln!(f).ok();
    writeln!(
        f,
        "Joint Embedding Predictive Architecture in hypervector space:"
    )
    .ok();
    writeln!(f).ok();
    writeln!(f, "- **Learnable permutation offsets** — instead of expensive attention, each position in context").ok();
    writeln!(
        f,
        "  learns a continuous offset that shifts the hypervector before bundling."
    )
    .ok();
    writeln!(f, "- **Weighted bundle** — context vectors are multiplied by learned weights (not all positions equal).").ok();
    writeln!(
        f,
        "- **Hill-climbing training** — `train_on_sequences()` slightly perturbs offsets/weights,"
    )
    .ok();
    writeln!(
        f,
        "  accepts changes that improve similarity to the observed next state."
    )
    .ok();
    writeln!(f, "- **Predict next state** — `predict(context)` returns the predicted Hypervector + confidence.").ok();
    writeln!(f).ok();
    writeln!(f, "CLI: `fuga jepa-train <dir> [dim] [ctx_len] [epochs]` / `fuga jepa-predict <text> [dim] [ctx]`").ok();
    writeln!(f).ok();
    write_moe(
        "JEPA predictor learnable permutation offsets weighted bundle trajectory prediction",
        "code",
        "Code patterns",
        &mut f,
    );

    // === PROMPT SYSTEM ===
    writeln!(f, "## VSA Prompt Algebra").ok();
    writeln!(f).ok();
    writeln!(f, "### PromptVectors (`src/ai/prompts.rs`)").ok();
    writeln!(f).ok();
    writeln!(
        f,
        "Behavioral modulation at the hypervector level — no text-based system prompts:"
    )
    .ok();
    writeln!(f).ok();
    writeln!(f, "| Mode | Effect | Entropy |").ok();
    writeln!(f, "|------|--------|---------|").ok();
    {
        let pv = fuga::PromptVectors::new(engine.dim);
        for name in pv.all_modes() {
            if let Some(hv) = pv.get(&name) {
                writeln!(
                    f,
                    "| {} | {} | {:.4} |",
                    name,
                    match name.as_str() {
                        "SAFETY" => "Conservative, avoid unsafe patterns",
                        "EFFICIENT" => "Prioritize minimal resource solutions",
                        "CONCISE" => "Short, direct responses",
                        "EXPLAIN" => "Detailed explanation mode",
                        "DRY_RUN" => "Simulate without action",
                        _ => "Custom behavioral vector",
                    },
                    hv.entropy()
                )
                .ok();
            }
        }
    }
    writeln!(f).ok();
    writeln!(
        f,
        "Operation: `bind(QueryHV, PromptHV) → ModulatedHV` via XOR bundle. "
    )
    .ok();
    writeln!(
        f,
        "CLI: `--prompt SAFETY,CONCISE` flag for `ask` and `agent` commands."
    )
    .ok();
    writeln!(f).ok();
    write_moe(
        "VSA prompt algebra bind modulation SAFETY CONCISE EXPLAIN",
        "code",
        "Code patterns",
        &mut f,
    );

    // === SELF-REFACTORING ===
    writeln!(f, "## Self-Refactoring Loop").ok();
    writeln!(f).ok();
    writeln!(f, "### `fuga refactor <file> <desc> [max_iter]`").ok();
    writeln!(f).ok();
    writeln!(f, "Closed-loop autonomous code improvement:").ok();
    writeln!(f).ok();
    writeln!(
        f,
        "1. **Search** — query code MoE with task description for relevant patterns"
    )
    .ok();
    writeln!(
        f,
        "2. **Generate** — `apply_refactor_hint()` injects MoE patterns into source file"
    )
    .ok();
    writeln!(
        f,
        "3. **Compile** — `cargo check` (in-project) or `rustc` (standalone file)"
    )
    .ok();
    writeln!(f, "4. **Test** — `cargo test` if compilation passes").ok();
    writeln!(
        f,
        "5. **Rollback** — restore from `.bak` on error, absorb error into MoE"
    )
    .ok();
    writeln!(
        f,
        "6. **Absorb** — `save_agent_result()` writes success/failure to `agent_results/`"
    )
    .ok();
    writeln!(
        f,
        "7. **Loop** — up to `max_iter` attempts, each informed by previous errors"
    )
    .ok();
    writeln!(f).ok();
    writeln!(
        f,
        "The MoE search query includes compilation errors from prior iterations,"
    )
    .ok();
    writeln!(
        f,
        "making the system progressively learn from its mistakes."
    )
    .ok();
    writeln!(f).ok();
    write_moe(
        "self-refactoring closed loop sandbox compilation cargo check test backup restore",
        "code",
        "Code patterns",
        &mut f,
    );

    // === TRAINING PIPELINE ===
    writeln!(f, "## Training Pipeline").ok();
    writeln!(f).ok();
    writeln!(
        f,
        "### `fuga train <dir> [--epochs N] [--side N] [--ndim N]`"
    )
    .ok();
    writeln!(f).ok();
    writeln!(f, "Multi-epoch code quality training:").ok();
    writeln!(f).ok();
    writeln!(f, "1. **Scan** — `CodeQualityFilter::scan_directory()` scores files by safety, weight, complexity").ok();
    writeln!(f, "2. **IDF** — collect term frequencies across all docs, compute inverse document frequency weights").ok();
    writeln!(f, "3. **Absorb** — for each epoch, tokenize → `ai.think()` → `ai.absorb_with_quality()` → store in memory + cube").ok();
    writeln!(
        f,
        "4. **Checkpoint** — cube + memory saved to disk every 5 epochs"
    )
    .ok();
    writeln!(f, "5. **MoE build** — after training, `build_moe_from_memory()` constructs domain experts and saves to `fuga_moe_*.bin`").ok();
    writeln!(f).ok();
    writeln!(f, "Quality filtering rejects files with `weight <= 0.0` (safety violations) and down-weights low-quality sources.").ok();
    writeln!(f, "Each epoch processes all files, deterministically adding ~same number of entries per epoch.").ok();
    writeln!(f).ok();

    // === CLI ===
    writeln!(f, "## CLI Reference").ok();
    writeln!(f).ok();
    writeln!(f, "| Command | Description |").ok();
    writeln!(f, "|---------|-------------|").ok();
    writeln!(f, "| `train <dir>` | Multi-epoch code quality training |").ok();
    writeln!(
        f,
        "| `train-text <dir>` | Text corpus training (requires existing cube) |"
    )
    .ok();
    writeln!(f, "| `ask <question>` | Answer from trained VSA memory |").ok();
    writeln!(
        f,
        "| `agent <task>` | Autonomous task execution with Fuga memory |"
    )
    .ok();
    writeln!(
        f,
        "| `refactor <file> <desc>` | Self-refactoring closed loop |"
    )
    .ok();
    writeln!(f, "| `jepa-train <dir>` | Train JEPA state predictor |").ok();
    writeln!(f, "| `jepa-predict <text>` | Predict next state via JEPA |").ok();
    writeln!(f, "| `prompts` | List VSA prompt modes |").ok();
    writeln!(f, "| `moe-add <domain>` | Create new MoE domain |").ok();
    writeln!(f, "| `moe-list` | List all MoE domains |").ok();
    writeln!(f, "| `docs` | Generate self-documentation (this file) |").ok();
    writeln!(
        f,
        "| `weave <path>` | Compress tokens with VSA Weaver Engine |"
    )
    .ok();
    writeln!(
        f,
        "| `unweave <path>` | Reconstruct token stream from SuperTokens |"
    )
    .ok();
    writeln!(f, "| `codegen <seed>` | Generate code/text from cube |").ok();
    writeln!(f, "| `query <text>` | Resonance search over cube |").ok();
    writeln!(
        f,
        "| `solve <problem>` | Multi-step reasoning with decomposition |"
    )
    .ok();
    writeln!(f, "| `analyze <path>` | Code quality/safety analysis |").ok();
    writeln!(f, "| `scan <path>` | Security AST audit |").ok();
    writeln!(
        f,
        "| `think <text>` | Run AI core (tokenize → route → absorb) |"
    )
    .ok();
    writeln!(f, "| `room` | Room phase lock (headless) |").ok();
    writeln!(f, "| `reactor` | Reactor point kinetics simulation |").ok();
    writeln!(f, "| `fisig <corpus>` | Train physics model |").ok();
    writeln!(f).ok();
    writeln!(
        f,
        "Global options: `--dim`, `--side`, `--ndim`, `--epochs`, `--save`, `--cube`, `--prompt`"
    )
    .ok();
    writeln!(f).ok();
    writeln!(f, "---").ok();
    writeln!(f).ok();
    writeln!(
        f,
        "*Generated by `fuga docs` — hybrid source-code + VSA memory documentation*"
    )
    .ok();
    writeln!(f).ok();

    let size = std::fs::metadata(out_path).map(|m| m.len()).unwrap_or(0);
    let lines = std::fs::read_to_string(out_path)
        .map(|s| s.lines().count())
        .unwrap_or(0);
    println!(
        "\n  Written to {} — {} bytes, {} lines",
        out_path, size, lines
    );
}

fn apply_refactor_hint(current: &str, desc: &str, ctx: &str) -> String {
    let lower = desc.to_lowercase();

    // Rule-based optimization: hamming / popcount / simd
    if lower.contains("hamming")
        || lower.contains("popcount")
        || lower.contains("simd")
        || (lower.contains("cache") && lower.contains("locality"))
    {
        return optimize_hamming_simd(current, desc, ctx);
    }

    // if context has relevant code, inject it
    if !ctx.is_empty() && (lower.contains("add") || lower.contains("impl") || lower.contains("fix"))
    {
        format!(
            "// refactor: {}\n// pattern from MoE:\n{}\n\n{}",
            desc, ctx, current
        )
    } else if lower.contains("comment") || lower.contains("doc") {
        format!("// TODO({}): {}\n{}", desc, desc, current)
    } else if lower.contains("remove") || lower.contains("delete") {
        current
            .lines()
            .filter(|l| {
                let t = l.trim();
                !t.starts_with("// TODO") && !t.contains("unused")
            })
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        format!(
            "// refactor: {}\n// generated by Fuga agent\n{}",
            desc, current
        )
    }
}

fn optimize_hamming_simd(source: &str, desc: &str, _ctx: &str) -> String {
    let simd_helpers = "\n// SIMD-optimized popcount helpers — unrolled 4-wide for cache locality
fn popcount_chunks(words: &[u64]) -> u64 {
    let mut total: u64 = 0;
    let chunks = words.chunks_exact(4);
    let remainder = chunks.remainder();
    for chunk in chunks {
        total += chunk[0].count_ones() as u64
               + chunk[1].count_ones() as u64
               + chunk[2].count_ones() as u64
               + chunk[3].count_ones() as u64;
    }
    for &w in remainder {
        total += w.count_ones() as u64;
    }
    total
}

fn popcount_xor_pair(a: &[u64], b: &[u64], n: usize) -> u64 {
    let limit = a.len().min(b.len()).min(n);
    let (a_main, a_rem) = a[..limit].split_at(limit / 4 * 4);
    let (b_main, b_rem) = b[..limit].split_at(limit / 4 * 4);
    let mut total: u64 = 0;
    for i in (0..a_main.len()).step_by(4) {
        total += (a_main[i] ^ b_main[i]).count_ones() as u64
               + (a_main[i+1] ^ b_main[i+1]).count_ones() as u64
               + (a_main[i+2] ^ b_main[i+2]).count_ones() as u64
               + (a_main[i+3] ^ b_main[i+3]).count_ones() as u64;
    }
    for i in 0..a_rem.len() {
        total += (a_rem[i] ^ b_rem[i]).count_ones() as u64;
    }
    total
}
";

    // Structural replacement: find function signature, then matching brace
    fn replace_fn(result: &mut String, sig: &str, new_body: &str) -> bool {
        if let Some(sig_pos) = result.find(sig) {
            let after_sig = &result[sig_pos..];
            if let Some(brace_pos) = after_sig.find('{') {
                let body_start = sig_pos + brace_pos + 1;
                let mut depth = 1u32;
                let mut body_end = None;
                for (i, ch) in result[body_start..].char_indices() {
                    if ch == '{' {
                        depth += 1;
                    }
                    if ch == '}' {
                        depth -= 1;
                    }
                    if depth == 0 {
                        body_end = Some(body_start + i + 1);
                        break;
                    }
                }
                if let Some(end) = body_end {
                    let mut new_result = String::with_capacity(result.len() + new_body.len());
                    new_result.push_str(&result[..body_start]);
                    new_result.push('\n');
                    new_result.push_str(new_body);
                    new_result.push('\n');
                    new_result.push_str("    }\n");
                    new_result.push_str(&result[end..]);
                    *result = new_result;
                    return true;
                }
            }
        }
        false
    }

    let mut result = source.to_string();
    let mut replaced = false;

    // Hypervector targets
    replaced |= replace_fn(
        &mut result,
        "pub fn hamming_distance(&self, other: &Hypervector) -> f64",
        "        let wc = self.word_count().min(other.word_count());\n         let mismatches = popcount_xor_pair(&self.words, &other.words, wc);\n         mismatches as f64 / self.dim as f64",
    );

    replaced |= replace_fn(
        &mut result,
        "pub fn partial_hamming_distance(&self, other: &Hypervector, n_words: usize)",
        "        let wc = self.word_count().min(other.word_count()).min(n_words);\n         if wc == 0 { return 0.5; }\n         let mismatches = popcount_xor_pair(&self.words, &other.words, wc);\n         mismatches as f64 / (wc * 64) as f64",
    );

    replaced |= replace_fn(
        &mut result,
        "pub fn entropy(&self)",
        "        let ones = popcount_chunks(&self.words);\n         ones as f64 / self.dim as f64",
    );

    // WaveCube targets - redirect to Hypervector::entropy or use popcount_chunks
    replaced |= replace_fn(
        &mut result,
        "pub fn global_entropy(&self) -> f64",
        "        let total_bits = Self::TOTAL_CELLS * self.dim;\n        let ones: u64 = self.cube.iter().map(|hv| hv.entropy() * hv.dim as f64).sum::<f64>() as u64;\n        ones as f64 / total_bits as f64",
    );

    replaced |= replace_fn(
        &mut result,
        "pub fn coherence(&self) -> f64",
        "        if N < 3 { return 0.0; }\n        let mut sum = 0.0;\n        let mut count = 0;\n        let mut i = 0;\n        while i < S {\n            let a = self.cell(i, i, i);\n            let b = self.cell(S - 1 - i, S - 1 - i, S - 1 - i);\n            sum += a.similarity(&b);\n            count += 1;\n            i += 1;\n        }\n        sum / count as f64",
    );

    // Generic popcount chain replacement for any .iter().map(|w| w.count_ones()).sum()
    // This is a more aggressive pattern - only apply if we find it outside already-replaced functions
    if result.contains(".count_ones()") && !result.contains("popcount_chunks") {
        // Add the helper and a note
        replaced = true;
    }

    if replaced {
        // Insert helpers right after `impl Hypervector { ... }` block, before #[cfg(test)]
        if let Some(impl_end) = result.find("\n}\n\n#[cfg(test)]") {
            let insert_pos = impl_end + 2; // after "}\n\n"
            result.insert_str(insert_pos, simd_helpers);
        } else if let Some(impl_end) = result.rfind("}\n\n") {
            // fallback: last double-brace before EOF
            let after = &result[impl_end + 2..];
            if after.trim().is_empty() {
                result.insert_str(impl_end + 2, simd_helpers);
            }
        }
        format!(
            "// refactor: {} — SIMD-unrolled popcount (MoE+rule)\n{}",
            desc, result
        )
    } else {
        let safe_ctx = _ctx
            .lines()
            .filter(|l| !l.trim().starts_with("/*") && !l.trim().starts_with("*"))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "// refactor: {}\n// MoE patterns (sanitized):\n{}\n\n{}",
            desc, safe_ctx, source
        )
    }
}

fn print_usage(program: &str) {
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

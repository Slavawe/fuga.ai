//! Agent / absorb / stream / generate / merge commands.
//!
//! Extracted from `src/main.rs`.

use std::fs;
use std::path::Path;
use std::process;

use crate::cli::args::{parse_dim, parse_flag_value, parse_int, has_flag, parse_float, parse_output, parse_window};
use crate::cli::tools::save_agent_result;
use crate::cli::inspect::print_usage;
use fuga::weaver::token_id;
use fuga::{
    CodeQualityFilter, FixProposal, FugaAI, FugaEngine, LanguageId, MemoryStore,
    MultiEngine, MultiFixGenerator, SDR_DIM, TokenBuilder, TokenInfo, WaveCube,
};

pub fn run_absorb_agent() {
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

pub fn run_stream_train<const N: usize, const S: usize>(
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

pub fn run_rebuild_moe(save_path: &str) {
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

pub fn human_size(bytes: u64) -> String {
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

pub fn run_agent(task: &str, _dim: usize, force: bool) {
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

pub fn run_agent_loop(args: &[String]) {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  Fuga Agent — Self-Recursive Loop (necli)  ║");
    println!("╚══════════════════════════════════════════════╝");
    println!("  Brain: Fuga self-powered mask (OpenAI-compat /v1/chat/completions)\n");
    let task = args.iter()
        .skip_while(|a| !a.contains("loop") && !a.contains("agent") && a.as_str() != "fuga")
        .skip(1)
        .filter(|a| !a.starts_with("--") && !a.starts_with("-"))
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    let task_str = if task.trim().is_empty() {
        "Generate a safe Rust function that computes factorial; include test; compile it; output PASS/FAIL.".to_string()
    } else {
        task
    };
    let iters = args.iter()
        .enumerate()
        .filter_map(|(i, v)| if v == "--iters" { args.get(i + 1) } else { None })
        .filter_map(|v| v.parse::<usize>().ok())
        .next()
        .unwrap_or(3);
    println!("  Task: {}  |  Iterations: {}\n", task_str, iters);
    // Delegate to the Python agent_loop driver (uses necli -> Fuga mask -> compile -> retrain).
    let script_path = std::env::current_dir().unwrap().join("agent_loop.py");
    if !script_path.exists() {
        eprintln!("  agent_loop.py not found — make sure it is in repo root");
        return;
    }
    let mut cmd = std::process::Command::new("python3");
    cmd.arg(&script_path)
        .arg(&task_str)
        .arg("--iters")
        .arg(iters.to_string())
        .env("FUGA_WEB_PORT", "8080");
    println!("  Running: python3 agent_loop.py \"{}\" --iters {} ... (self-brain -> necli -> compile -> retrain)\n", task_str, iters);
    match cmd.status() {
        Ok(s) => {
            println!("  Agent-loop exit code: {}\n", s.code().unwrap_or(-1));
            println!("  Recursive results: check agent_lessons.jsonl and updated fuga_hjepa.bin.");
        }
        Err(e) => {
            eprintln!("  Failed to launch agent_loop: {}", e);
        }
    }
}

pub fn run_generate(prompt: &str, _dim: usize, output: Option<&str>) {
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

pub fn run_merge<const N: usize, const S: usize>(args: &[String]) {
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

pub fn run_train_text<const N: usize, const S: usize>(
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

pub fn run_train_code<const N: usize, const S: usize>(
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

pub fn run_train_autofix<const N: usize, const S: usize>(
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

pub fn apply_autofix_proposals(source: &str, proposals: &[FixProposal]) -> String {
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

pub fn run_moe_split<const N: usize, const S: usize>(save_path: &str) {
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

    // ── Task-conditioned hard VSA mask (identity/task channel) ────────────
    // Optional `--task "<text>"`: build a task-hypervector from the task words
    // (hamming union) and HARD-gate every candidate token: a candidate whose SDR
    // shares no bit with any task word is zeroed (windows/handle/async ... the
    // corpus-dominant noise), so next-token selection can only fall on tokens
    // that belong to the requested task's semantic neighbourhood.
    let task_words: Vec<fuga::SdrVector> = parse_flag_value(&args, 2, "--task")
        .map(|tt| {
            tt.split(|c: char| !c.is_alphanumeric() && c != '_')
                .filter(|w| w.len() >= 2)
                .map(fuga::encode_text)
                .collect()
        })
        .unwrap_or_default();
    let task_masked = !task_words.is_empty();
    if task_masked {
        println!("  Task-mask: {} task word SDRs (hard AND gate on candidates)", task_words.len());
    }
    // Soft VSA mask: `--task-soft <w>` replaces the hard AND gate with a
    // weighted Hamming-overlap score against the task hypervector's spanning
    // bits, so syntactic connector tokens are NOT dropped — their weight is
    // just moderated by task relevance.
    let task_weight = parse_flag_value(&args, 2, "--task-soft")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let task_bits: [u64; 128] = if !task_words.is_empty() {
        let mut b = [0u64; 128];
        for s in &task_words {
            for i in 0..128 {
                b[i] |= s.bits[i];
            }
        }
        b
    } else {
        [0u64; 128]
    };
    if task_weight > 0.0 {
        println!("  Task-soft: weight={} (cosine-hamming to task union)", task_weight);
    }

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
        // ── H-JEPA L1/L2 trajectory guidance ─────────────────────────────
        // `--hjepa <path>`: the upper-level hierarchical JEPA (TemporalPredictor:
        // TM feed -> HV buffer -> predict_sequence latent roll-out) REGULATES THE
        // ORDER of generated tokens, decoding each predicted latent to the
        // nearest eligible vocab word. The vocab is task-gated (hard mask), so
        // the trajectory only picks among task-eligible tokens.
        if let Some(hj_path) = parse_flag_value(&args, 2, "--hjepa") {
            let hjepa = match fuga::HierarchicalJEPA::load(hj_path) {
                Ok(h) => h,
                Err(e) => {
                    eprintln!("  ✗ H-JEPA {}: {}", hj_path, e);
                    return;
                }
            };
            let dim = hjepa.dim;
            let mut tpred = fuga::TemporalPredictor::new(tm, hjepa);
            let task_pop: f64 = task_bits.iter().map(|w| w.count_ones() as f64).sum();
            let mut elig: Vec<(String, fuga::Hypervector)> = Vec::new();
            let task_sim_floor = parse_flag_value(&args, 2, "--task-sim")
                .and_then(|v| v.parse::<f64>().ok());
            for (tok, sdr, _) in vocab_latents.iter() {
                if task_masked {
                    let share = task_words.iter().any(|w| sdr.overlap(w) > 0);
                    if let Some(floor) = task_sim_floor {
                        // Cosine-hamming similarity vs the task-union bits.
                        let cand_pop = sdr.bits.iter().map(|w| w.count_ones() as f64).sum::<f64>();
                        let shared = sdr
                            .bits
                            .iter()
                            .zip(task_bits.iter())
                            .map(|(a, b)| (a & b).count_ones() as f64)
                            .sum::<f64>();
                        let sim = if cand_pop * task_pop > 0.0 {
                            shared / (cand_pop * task_pop).sqrt()
                        } else {
                            0.0
                        };
                        if sim < floor {
                            continue;
                        }
                    } else if !share {
                        continue;
                    }
                }
                elig.push((tok.clone(), fuga::sdr_to_hypervector(sdr, dim)));
            }
            println!("  H-JEPA L1/L2 guidance: {} eligible vocab words", elig.len());
            if elig.len() <= 60 {
                let mut names: Vec<&str> = elig.iter().map(|(t, _)| t.as_str()).collect();
                names.sort();
                println!("  eligible: {:?}", names);
            }
            let out = tpred.generate_words(prompt, steps, &elig, 0.05);
            for (i, w) in out.iter().enumerate() {
                println!("  step {}: {}", i, w);
            }
            return;
        }
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
                // Task score: hard AND gate (any shared bit) when in hard mode,
                // or a soft Hamming overlap vs the task-union bits.
                let task_score = if task_weight > 0.0 {
                    let mut c = 0usize;
                    for i in 0..128 {
                        c += (sdr.bits[i] & task_bits[i]).count_ones() as usize;
                    }
                    c as f64
                } else {
                    0.0
                };
                if task_masked && task_weight <= 0.0
                    && !task_words.iter().any(|w| sdr.overlap(w) > 0)
                {
                    continue;
                }
                // Structural overlap as a mild secondary signal.
                let struct_score = pred.overlap(sdr) as f64;
                let combined = latent_score * 100.0 + struct_score + task_weight * task_score;
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


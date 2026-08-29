//! JEPA / HTM / SDR training and prediction commands.
//!
//! Extracted from `src/main.rs` during monolith decomposition.

use std::fs;
use std::process;

use crate::cli::args::{parse_dim, parse_int, parse_flag_value, has_flag};
use crate::cli::tm_gen::{lex_rust_code, load_tm, load_tm_from};

use fuga::weaver::token_id;

pub fn run_jepa_train(dir: &str, dim: usize, context_len: usize, epochs: usize) {
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

pub fn run_jepa_predict(text: &str, dim: usize, context_len: usize) {
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

pub fn run_hierarchical_jepa_train(dir: &str, dim: usize, epochs: usize) {
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

pub fn run_baby_repl(dim: usize) {
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

pub fn run_sdr_query(text: &str) {
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

pub fn run_sdr_query_cross(text: &str) {
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

pub fn load_sdr_store(path: &str) -> Option<fuga::SdrStore> {
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

pub fn run_htm_train(_path: &str, steps: usize) {
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


pub fn run_htm_feed(text: &str) {
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

pub fn run_train_tm(dir: &str, cap: usize, ctx: usize, max_files: usize, out: &str, structure: bool) {
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

pub fn encode_chunk(weaver: &mut fuga::WeaverEngine, text: &str) -> fuga::Hypervector {
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


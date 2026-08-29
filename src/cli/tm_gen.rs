//! TM Generation and tree-sitter helpers.
//!
//! Extracted from `src/main.rs` during monolith decomposition.
//! Run_tm_gen entry is in main.rs, core logic lives here.

use std::path::Path;
use std::process;

use crate::cli::args::{parse_flag_value, parse_int};

pub fn lex_rust_code(code: &str) -> Vec<String> {
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
pub fn ts_filter_ok(partial: &str, candidate: &str) -> bool {
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

pub fn count_ts_error_nodes(node: &tree_sitter::Node) -> u32 {
    let mut count = u32::from(node.is_error());
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            count += count_ts_error_nodes(&child);
        }
    }
    count
}

pub fn count_missing_nodes(node: &tree_sitter::Node) -> u32 {
    let mut count = u32::from(node.is_missing());
    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            count += count_missing_nodes(&child);
        }
    }
    count
}

/// Score TM candidates by how much they advance the parsed Rust structure.
pub fn ts_driven_score(partial: &str, tm_score: f64, token: &str) -> Option<f64> {
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

pub fn count_ts_errors(tree: &tree_sitter::Tree) -> u32 {
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
pub fn run_tm_gen(prompt: &str, args: &[String]) {
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


pub fn load_tm() -> Option<fuga::TemporalMemory> {
    load_tm_from("fuga_htm.bin")
}
pub fn load_tm_from(path: &str) -> Option<fuga::TemporalMemory> {
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


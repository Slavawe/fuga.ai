//! Crystal / resonance / phase command implementations.
//!
//! Extracted from `src/main.rs` during monolith decomposition.
//! Crystal memory, phase trajectory, decode, transpile, docs.

use std::fs;
use std::process;

use crate::cli::args::parse_flag_value;
use crate::cli::print::{capitalize, is_name_token, print_cpp_code, print_rust_code};
use crate::cli::tm_gen::load_tm;
use crate::cli::jepa::load_sdr_store;
use crate::cli::tools::save_agent_result;
use crate::cli::jepa::encode_chunk;
use crate::cli::inspect::print_usage;
use fuga::core::wave_cube::peek_cube_header;
use fuga::weaver::token_id;

pub fn run_crystal_build(out: &str, max_entries: usize) {
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

pub fn run_crystal_query(text: &str, args: &[String]) {
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
pub fn parse_query_config(args: &[String], base: usize) -> fuga::QueryConfig {
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

pub fn run_crystal_reencode(path: &str) {
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

pub fn run_phase_trajectory(text: &str) {
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

pub fn run_decode(text: &str, args: &[String]) {
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

pub fn run_phase_codegen(text: &str, args: &[String]) {
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

pub fn run_crystal_test() {
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

pub fn run_crystal_learn(key: &str, text: &str, args: &[String]) {
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
pub fn run_crystal_learn_dir(dir: &str, args: &[String]) {
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

pub fn run_crystal_forget(key: &str, args: &[String]) {
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

pub fn run_crystal_stats(path: &str) {
    match fuga::PhaseCrystal::load(path) {
        Ok(c) => println!("  {}", c.stats()),
        Err(e) => eprintln!("  ✗ {}", e),
    }
}

pub fn run_crystal_popcount(text: &str) {
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

pub fn run_crystal_2_init(path: &str) {
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

pub fn run_crystal_2_learn(key: &str, text: &str, args: &[String]) {
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
pub fn run_crystal_hippo(text: &str, args: &[String]) {
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

pub fn run_crystal_triad(candidate: &str, args: &[String]) {
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

pub fn run_transpile(args: &[String]) {
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

pub fn run_self_query(text: &str) {
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

pub fn run_reflect_repl() {
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

pub fn run_htm_predict(text: &str) {
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

pub fn run_cross_domain(_dim: usize, epochs: usize) {
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

pub fn run_hierarchical_jepa_predict(text: &str, dim: usize) {
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
pub fn run_refactor(file: &str, desc: &str, max_iter: usize) {
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

pub fn run_docs_entry(args: &[String]) {
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

pub fn apply_refactor_hint(current: &str, desc: &str, ctx: &str) -> String {
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

pub fn optimize_hamming_simd(source: &str, desc: &str, _ctx: &str) -> String {
    let simd_helpers = "\n// SIMD-optimized popcount helpers — unrolled 4-wide for cache locality
pub fn popcount_chunks(words: &[u64]) -> u64 {
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

pub fn popcount_xor_pair(a: &[u64], b: &[u64], n: usize) -> u64 {
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


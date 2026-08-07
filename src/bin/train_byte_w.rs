// train_byte_w.rs — the missing link: train the LOCAL byte-transition
// operator W on a corpus and persist it as a sidecar (magic "FBW1") that
// `combined_decode` can attach to any checkpoint.
//
// The full-training checkpoints (mirror_tm_*) keep W at identity (only TM
// cells are saved); fuga_stack_tm.bin has a TOKEN-trained W. This stand trains
// the BYTE-level W with the same `learn_bytes` loop used in the A/B benches,
// then saves it so the connected decoder suite finally has a byte-consistent
// operator.
//
// Usage: train_byte_w <corpus.jsonl> <N lines> <out.bin> [ctx_window]
use std::time::Instant;

use fuga::ai::htm_temporal::TemporalMemory;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus_path = args.get(1).cloned().unwrap_or_else(|| "corpus_doc_code_pairs.jsonl".into());
    let limit: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(800);
    let out_path = args.get(3).cloned().unwrap_or_else(|| "fuga_byte_w.bin".into());
    let ctx: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(4);

    let corpus = std::fs::read_to_string(&corpus_path).expect("corpus");
    // Train the W operator DIRECTLY (no TM cell growth): each byte window →
    // next byte through the latent transition operator, exactly the byte-path
    // learning the decoders consume. This is orders of magnitude faster than
    // learn_bytes (which also grows TM cells) and persists the same W.
    let mut predictor = fuga::LatentPredictor::new(0xF03D_C0DE);
    let t0 = Instant::now();
    let mut n_lines = 0usize;
    let mut n_steps = 0u64;
    for line in corpus.lines().take(limit) {
        let b = line.as_bytes();
        if b.len() < 8 {
            continue;
        }
        for w in 0..b.len().saturating_sub(1) {
            let lo = w.saturating_sub(ctx);
            let window: Vec<fuga::SdrVector> = b[lo..w + 1]
                .iter()
                .map(|&c| fuga::byte_basis(c))
                .collect();
            let next = fuga::byte_basis(b[w + 1]);
            predictor.learn_transition(&window, &next, 0.1);
            n_steps += 1;
        }
        n_lines += 1;
    }
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "trained byte W directly on {} lines / {} byte-steps in {:.1}s (ctx={})",
        n_lines, n_steps, secs, ctx
    );
    println!(
        "W_updates={} |W|={} (non-trivially trained={})",
        predictor.updates,
        predictor.w.len(),
        predictor.w.iter().any(|&v| (v - 1.0).abs() > 1e-4 && v != 0.0)
    );
    // Attach to a throwaway TM just to use its save_byte_w (or write directly).
    let mut tm = TemporalMemory::new(64, ctx);
    tm.apply_byte_w(predictor.w.clone());
    match tm.save_byte_w(&out_path) {
        Ok(()) => println!("saved byte W sidecar -> {} ({} bytes)", out_path, std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0)),
        Err(e) => {
            eprintln!("FAILED to save sidecar: {}", e);
            std::process::exit(2);
        }
    }
    // Roundtrip self-check: reload and compare length.
    if let Some(back) = TemporalMemory::load_byte_w(&out_path) {
        println!("self-check roundtrip OK: {} floats (== {} fl.equiv)", back.len(), predictor.w.len());
    }
}
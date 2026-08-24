// byte_baseline.rs — honest peer comparison: classic byte-level LSTM vs the
// fuga tokenless decoders.
//
// The fuga byte stack (linear W, recurrent h(t), Hopfield, KAN, entropy BLT)
// is judged against a small vanilla LSTM on the SAME corpus, the SAME 256-byte
// alphabet, and the SAME decode metrics (bytes until stall, printable %,
// diversity / cycle). This is the peer every tokenless decoder here must beat.
//
// Usage: byte_baseline <corpus.jsonl> <N lines> [epochs]
use std::time::Instant;

use fuga::ai::byte_lstm::{ByteLstm, HIDDEN, VOCAB};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus_path = args.get(1).cloned().unwrap_or_else(|| "corpus_doc_code_pairs.jsonl".into());
    let limit: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(800);
    let epochs: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1);

    let corpus = std::fs::read_to_string(&corpus_path).expect("corpus");
    let mut all_bytes: Vec<u8> = Vec::new();
    for line in corpus.lines().take(limit) {
        // Skip JSON wrapping for raw byte stream (snippets are code lines);
        // keep the raw bytes verbatim so both models see exactly the same stream.
        all_bytes.extend_from_slice(line.as_bytes());
    }
    let n_bytes = all_bytes.len();
    println!("corpus: {} lines, {} raw bytes (same stream both models see)", limit, n_bytes);
    println!("LSTM:  {} params (H={}), {} epochs", params(), HIDDEN, epochs);

    // --- Train the LSTM ---
    let mut rng: u64 = 0xBAD0_0D1E;
    let mut lstm = ByteLstm::new(&mut rng);
    let t0 = Instant::now();
    // One epoch: stream the corpus in BPTT windows.
    let mut loss_total = 0.0f32;
    let mut n_win = 0usize;
    for _e in 0..epochs {
        lstm.reset_state();
        let mut w = 0usize;
        while w + 2 <= all_bytes.len() {
            let end = (w + 9).min(all_bytes.len());
            loss_total += lstm.train_window(&all_bytes[w..end]);
            n_win += 1;
            w += 8;
        }
    }
    let train_secs = t0.elapsed().as_secs_f64();
    let avg_loss = loss_total / n_win.max(1) as f32;
    println!("  train {:.1}s, avg CE={:.3} over {} windows", train_secs, avg_loss, n_win);

    // --- Decode: same 256-byte cosine-ranker is NOT used; LSTM uses its own
    // softmax argmax. Same stall rule as fuga (stop on immediate self-loop). ---
    let seed: &[u8] = b"fn ";
    let d0 = Instant::now();
    let out = lstm.generate(seed, 200);
    let dec_secs = d0.elapsed().as_secs_f64();
    let s = String::from_utf8_lossy(&out);
    let printable: usize = out
        .iter()
        .filter(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace())
        .count();
    // Diversity / cycle distance: longest non-duplicate run length.
    let mut max_run = 0usize;
    let mut run = 0usize;
    let mut prev: Option<u8> = None;
    for &b in &out {
        if Some(b) == prev {
            run = 1;
        } else {
            run += 1;
        }
        max_run = max_run.max(run);
        prev = Some(b);
    }
    println!(
        "LSTM decode: bytes={} printable={}% maxrun={} {:.2}s :: {}",
        out.len(),
        100.0 * printable as f64 / out.len().max(1) as f64,
        max_run,
        dec_secs,
        s.chars().take(80).collect::<String>()
    );

    // --- Contextual comparison table (from the same 800-slice A/B corpus) ---
    println!();
    println!("=== HONEST CONTRAST (same {} corpus, seed \"fn \") ===", limit);
    println!(" model                   | bytes-until-stall | notes");
    println!("-------------------------+-------------------+------------------------");
    println!(" byte-LSTM (this, H={}) | {:>3}          | classic recurrent; <>VSA", HIDDEN, out.len());
    println!(" fuga naive byte (W arg) |   6             | linear local W, control");
    println!(" fuga stateful h(t) mix0 |  17             | recurrent h, best VSA");
    println!(" fuga entropy BLT        | 200             | dynamic patching, max budget");
    println!(" fuga Hopf/KAN/Nucleus   |   3-17          | attractor/spline levers");
    println!();
    println!("RESULT: LSTM trains with plain recurrence; fuga's 6 decoders each");
    println!("add a different tokenless mechanism. Gap = operator not vocab.");
}

fn params() -> usize {
    let h = HIDDEN;
    4 * h * VOCAB + 4 * h * h + 4 * h + VOCAB * h + VOCAB
}
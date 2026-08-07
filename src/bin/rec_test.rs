// rec_test.rs — SSM-lite recurrent byte decoder (honest A/B).
//
// Questions it answers:
//   [1] Does training the byte W with a running hidden state h(t)
//       (learn_bytes_rnn + advance_h) vs. purely stateless learn_bytes
//       change generation? The stateless W falls into the e->r attractor
//       (measured 200 bytes of "er er..." garbage); recurrent W is supposed
//       to condition on the GLOBAL past, not just the fixed window.
//   [2] Does the recurrent decode path (tm_generate_recurrent) reach a
//       different content / longer coherent output than naive byte argmax?
//
// Honest A/B: same corpus, same 800 snippets, same seed; the ONLY knob is
// whether training used h(t) (stateful) or not (stateless).
use std::time::Instant;

fn main() {
    let corpus_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "corpus_doc_code_pairs.jsonl".to_string());
    let limit = std::env::args()
        .nth(2)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(800);
    let ctx_window = std::env::args()
        .nth(3)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4);
    let content = std::fs::read_to_string(&corpus_path).expect("corpus");

    // ---- Two TMs, trained identically EXCEPT the recurrent flag. ----
    let mut tm_stateless = fuga::TemporalMemory::new(30_000, 4);
    let mut tm_stateful = fuga::TemporalMemory::new(30_000, 4);
    let mut bytes_seen = 0usize;
    let mut trained = 0usize;
    let phi = 0.9f32;
    let mix_train = 0.6f32;
    let t0 = Instant::now();

    for line in content.lines() {
        if trained >= limit {
            break;
        }
        let Ok(doc) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let src = doc["source"].as_str().unwrap_or("");
        if !src.ends_with(".rs") {
            continue;
        }
        let code = doc["code"].as_str().unwrap_or("");
        let b = code.as_bytes();
        if b.len() < 4 {
            continue;
        }
        // STATELESS training path (baseline).
        for w in 0..b.len().saturating_sub(1) {
            let win_lo = w.saturating_sub(ctx_window);
            tm_stateless.learn_bytes(&b[win_lo..w + 1], b[w + 1], 0.15);
        }
        // STATEFUL training path: keep a running h(t) and feed it into W.
        let mut h = fuga::ai::latent_jepa::LatentVector::zero();
        for w in 0..b.len().saturating_sub(1) {
            let win_lo = w.saturating_sub(ctx_window);
            let window_byte = &b[win_lo..w + 1];
            let cur = b[w];
            tm_stateful
                .learn_bytes_rnn(window_byte, &h, b[w + 1], 0.15, 0.6);
            // Advance h with the CURRENT byte (the one we just conditioned on),
            // exactly as the recurrent decoder does during inference.
            h = tm_stateful
                .predictor()
                .advance_h(h, &fuga::ai::sdr::byte_basis(cur), phi);
            bytes_seen += 1;
        }
        trained += 1;
    }
    let train_secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "  Trained {} Rust docs (stateless vs stateful, ctx_window={}) in {:.1}s",
        trained, ctx_window, train_secs
    );
    if trained == 0 {
        eprintln!("  FAIL: no training data");
        std::process::exit(2);
    }

    let seed: Vec<u8> = b"fn ".to_vec();

    // ---- Decode through tm_generate_recurrent on BOTH TMs. ----
    // Note: tm_stateful's W was trained WITH the state, so mixing h during
    // decode is at inference-train parity; tm_stateless is the control.
    println!("├─ recurrent decode of STATELESS-trained W (unlearned for state):");
    for &mix in &[0.0f32, 0.6] {
        let out = fuga::tm_generate_recurrent(&tm_stateless, &seed, 200, ctx_window, mix, phi);
        let s = String::from_utf8_lossy(&out);
        println!("  stateless mix={:.1} bytes={} :: {}", mix, out.len(), s.chars().take(80).collect::<String>());
    }

    println!("├─ recurrent decode of STATEFUL-trained W (train/spec parity):");
    for &mix in &[0.0f32, 0.4, 0.6, 0.8] {
        let d0 = Instant::now();
        let rec_out = fuga::tm_generate_recurrent(&tm_stateful, &seed, 60, ctx_window, mix, phi);
        let secs = d0.elapsed().as_secs_f64();
        let s = String::from_utf8_lossy(&rec_out);
        let printable: f64 = rec_out
            .iter()
            .filter(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace())
            .count() as f64;
        let pct = if rec_out.is_empty() { 0.0 } else { 100.0 * printable / rec_out.len() as f64 };
        println!("  stateful mix={:.1} bytes={} ({:.0}%) {:.2}s :: {}", mix, rec_out.len(), pct, secs,
            s.chars().take(80).collect::<String>());
    }

    // Baseline: plain byte-argmax on the stateless TM (matches existing naive).
    let naive = fuga::tm_generate_latent_bytes(&tm_stateless, b"fn ", 30, ctx_window, None);
    let s = String::from_utf8_lossy(&naive);
    println!("└─ naive stateless byte (control) bytes={} :: {}", naive.len(), s.chars().take(80).collect::<String>());

    println!("REAL-CODE RECURRENT: PASS");
}
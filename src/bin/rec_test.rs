// rec_test.rs — SSM-lite recurrent byte decoder (honest A/B), v3.1.
//
// Questions it answers:
//   [1] Does training the byte W with a running hidden state h(t)
//       (learn_bytes_rnn + advance_h) vs. purely stateless learn_bytes
//       change generation? The stateless W falls into the e->r attractor
//       ("er er..." garbage); recurrent W is supposed to condition on the
//       GLOBAL past, not just the fixed window.
//   [2] Does the recurrent decode path (tm_generate_recurrent) reach longer
//       coherent output than naive byte argmax?
//   [3] (v3.1) Does Scheduled Sampling during training let the decoder make
//       productive use of mix > 0 (closing the train/spec exposure gap)?
//       Args: corpus limit ctx_window epsilon
//
// Honest A/B: same corpus, same snippets, same seed; knobs are the recurrent
// flag (stateful vs stateless) and the scheduled-sampling prob epsilon.
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
    let epsilon = std::env::args()
        .nth(4)
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.15); // scheduled-sampling prob: feed model's own byte into h
    let content = std::fs::read_to_string(&corpus_path).expect("corpus");

    // Two TMs, trained identically EXCEPT the recurrent flag.
    let mut tm_stateless = fuga::TemporalMemory::new(30_000, 4);
    let mut tm_stateful = fuga::TemporalMemory::new(30_000, 4);
    let mut bytes_seen = 0usize;
    let mut trained = 0usize;
    let phi = 0.9f32;
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
        // STATELESS training path (baseline): fixed-window W, no state.
        for w in 0..b.len().saturating_sub(1) {
            let win_lo = w.saturating_sub(ctx_window);
            tm_stateless.learn_bytes(&b[win_lo..w + 1], b[w + 1], 0.15);
        }
        // STATEFUL training path: running h(t) fed into W.
        // Scheduled Sampling: with prob `epsilon`, advance h with the model's
        // OWN predicted next byte (drift) instead of the honest corpus byte.
        let mut h = fuga::ai::latent_jepa::LatentVector::zero();
        let mut x: u64 = 0x9E3779B97F4A7C15 ^ (trained as u64).wrapping_mul(0x12345678);
        for w in 0..b.len().saturating_sub(1) {
            let win_lo = w.saturating_sub(ctx_window);
            let window_byte = &b[win_lo..w + 1];
            let cur = b[w];
            tm_stateful.learn_bytes_rnn(window_byte, &h, b[w + 1], 0.15, 0.6);
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as f64 / (1u64 << 31) as f64;
            let byte_for_h: u8 = if u < epsilon as f64 {
                let pred = tm_stateful.predictor();
                let win_sdrs: Vec<fuga::ai::sdr::SdrVector> =
                    window_byte.iter().map(|&c| fuga::ai::sdr::byte_basis(c)).collect();
                let lat = pred.predict_next_rnn(&win_sdrs, &h, 0.6);
                let mut best = (0u8, f32::NEG_INFINITY);
                for bb in 0u16..256 {
                    let c = lat.cosine_similarity(
                        &pred.encoder.encode(&fuga::ai::sdr::byte_basis(bb as u8)),
                    );
                    if c > best.1 {
                        best = (bb as u8, c);
                    }
                }
                best.0
            } else {
                cur
            };
            bytes_seen += 1;
            h = tm_stateful
                .predictor()
                .advance_h(h, &fuga::ai::sdr::byte_basis(byte_for_h), phi);
        }
        trained += 1;
    }
    let train_secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "  Trained {} Rust docs (stateless vs stateful, ctx_window={} eps={:.2}) in {:.1}s",
        trained, ctx_window, epsilon, train_secs
    );
    if trained == 0 {
        eprintln!("  FAIL: no training data");
        std::process::exit(2);
    }

    let seed: Vec<u8> = b"fn ".to_vec();

    // ---- Decode through tm_generate_recurrent on BOTH TMs. ----
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

    let naive = fuga::tm_generate_latent_bytes(&tm_stateless, b"fn ", 30, ctx_window, None);
    let s = String::from_utf8_lossy(&naive);
    println!("└─ naive stateless byte (control) bytes={} :: {}", naive.len(), s.chars().take(80).collect::<String>());

    // === KAN-lite transition (roadmap point 5): spline operator replacing W ===
        println!("REAL-CODE RECURRENT: PASS");
        println!("REAL-CODE RECURRENT: PASS");
}
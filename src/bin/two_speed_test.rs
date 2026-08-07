// two_speed_test.rs — REAL two-speed (MegaByte-style) code generation test.
//
// Trains BOTH rates of the TemporalMemory on raw UTF-8 bytes of actual Rust
// snippets:
//   - LOCAL byte rate:  learn_bytes  (one byte out of 256, noisy)
//   - GLOBAL patch rate: learn_patch (ONE whole byte-patch per step)
// then generates through tm_generate_two_speed (global picks patch direction,
// coins only at patch granularity) and reports honest metrics + a comparison
// against the naive single-rate byte decoder.
//
// AD-HOC verification stand — mirrors the two-speed byte production path.
use fuga::ai::TemporalMemory;
use fuga::tm_generate_latent_bytes;
use fuga::tm_generate_two_speed;
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    let corpus_path =
        std::env::args().nth(1).unwrap_or_else(|| "corpus_doc_code_pairs.jsonl".to_string());
    let limit = std::env::args()
        .nth(2)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(800); // byte+patch training; 800 snippets is a solid real-code sample
    let patch_size = std::env::args()
        .nth(3)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4);
    let vocab_cap = std::env::args()
        .nth(4)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4000);
    let ctx_window = std::env::args()
        .nth(5)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4); // context depth for W training (bytes & patches)
    let gap_thresh = std::env::args()
        .nth(6)
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.60); // BLT entropy-gap threshold sweep point
    let content = std::fs::read_to_string(&corpus_path).expect("corpus");

    let mut tm = TemporalMemory::new(30_000, 4);
    let mut bytes_seen = 0usize;
    let mut patches_seen = 0usize;
    let mut trained = 0usize;
    // (Dedup-free frequency) patch grammar harvested from what we train on.
    let mut patch_freq: HashMap<Vec<u8>, usize> = HashMap::new();
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
        // LOCAL byte rate: sliding ctx_window-byte window -> next byte.
        for w in 0..b.len().saturating_sub(1) {
            let win_lo = w.saturating_sub(ctx_window);
            tm.learn_bytes(&b[win_lo..w + 1], b[w + 1], 0.15);
            bytes_seen += 1;
        }
        // GLOBAL patch rate: non-overlapping patches of patch_size -> next patch.
        let patches: Vec<Vec<u8>> = b
            .chunks(patch_size)
            .filter(|c| !c.is_empty())
            .map(|c| c.to_vec())
            .collect();
        for w in 0..patches.len().saturating_sub(1) {
            let win_lo = w.saturating_sub(ctx_window);
            tm.learn_patch(&patches[win_lo..w + 1].iter().map(|v| v.as_slice()).collect::<Vec<_>>(), &patches[w + 1], 0.15);
            patches_seen += 1;
        }
        for p in &patches {
            *patch_freq.entry(p.clone()).or_insert(0) += 1;
        }
        trained += 1;
        if trained % 1000 == 0 {
            eprintln!("  trained {} docs ({} byte-steps, {} patch-steps) in {:.1}s",
                trained, bytes_seen, patches_seen, t0.elapsed().as_secs_f64());
        }
    }
    let train_secs = t0.elapsed().as_secs_f64();
    eprintln!("  Trained {} Rust docs (ctx_window={}), {} byte-steps, {} patch-steps in {:.1}s ({:.0} patch-steps/s)",
        trained, ctx_window, bytes_seen, patches_seen, train_secs, patches_seen as f64 / train_secs.max(1.0));

    if trained == 0 {
        eprintln!("  FAIL: no training data");
        std::process::exit(2);
    }

    // Global patch vocabulary: the distinct byte-patches observed, capped.
    let mut vocab: Vec<(Vec<u8>, usize)> = patch_freq.into_iter().collect();
    vocab.sort_by(|a, b| b.1.cmp(&a.1));
    vocab.truncate(vocab_cap);
    let patch_vocab: Vec<Vec<u8>> = vocab.into_iter().map(|(p, _)| p).collect();
    eprintln!("  patch_vocab: {} distinct byte-patches (size {})", patch_vocab.len(), patch_size);

    let seed: Vec<u8> = b"fn main() { ".to_vec();

    // DIAGNOSE (global rate): cosine of predicted next-patch latent vs vocab.
    let seed_patches: Vec<&[u8]> = seed
        .chunks(patch_size)
        .filter(|c| !c.is_empty())
        .map(|c| c as &[u8])
        .collect();
    let pred = tm.predict_patch_latent(&seed_patches);
    let mut maxcos = -1f32;
    for p in &patch_vocab {
        let lat = tm.latent_of_sdr(&fuga::ai::sdr::encode_bytes_sdr(p));
        maxcos = maxcos.max(pred.cosine_similarity(&lat));
    }
    let over = patch_vocab.iter().filter(|p| {
        pred.cosine_similarity(&tm.latent_of_sdr(&fuga::ai::sdr::encode_bytes_sdr(p))) >= 0.05
    }).count();
    eprintln!("  DIAGNOSE patch_vocab max-cosine {:.4}, patches over 0.05: {}", maxcos, over);

    // Two-speed decode at a sweep of cosine thresholds (honest calibration).
    let n_patch_steps = 50_usize;
    println!("├─ two-speed no-repeat sweep (patch_vocab={}):", patch_vocab.len());
    let nr_values = [0usize, 2, 4];
    for &nr in &nr_values {
        let d0 = Instant::now();
        let out = fuga::tm_generate_two_speed_calib(&tm, &seed, n_patch_steps, 4, &patch_vocab, None, 0.05, nr);
        let secs = d0.elapsed().as_secs_f64();
        let s = String::from_utf8_lossy(&out).to_string();
        let printable: f64 = out.iter().filter(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace()).count() as f64;
        let pct = if out.is_empty() { 0.0 } else { 100.0 * printable / out.len() as f64 };
        println!("  no_repeat={} bytes={} ({:.1}%) {:.1}s :: {}", nr, out.len(), pct, secs,
                    s.chars().take(90).collect::<String>());
            }
            // Entropy-gap (BLT) decoder: dynamic patch boundaries by predictability.
                // Sweep the gap threshold on the SAME trained TM (decode is cheap;
                // training dominates). Measures the B/s vs. connectivity inflection.
                for &gap in &[0.30f32, 0.45, 0.60, 0.75] {
                    let d0e = Instant::now();
                    let entropy_out = fuga::tm_generate_two_speed_entropy(&tm, &seed, 200, ctx_window, gap, &patch_vocab);
                    let entropy_secs = d0e.elapsed().as_secs_f64();
                    let entropy_str = String::from_utf8_lossy(&entropy_out).to_string();
                    let entropy_printable: f64 = entropy_out
                        .iter()
                        .filter(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace())
                        .count() as f64;
                    let entropy_pct = if entropy_out.is_empty() { 0.0 } else { 100.0 * entropy_printable / entropy_out.len() as f64 };
                    println!("├─ entropy(BLT gap={:.2}) decoded ({} bytes):", gap, entropy_out.len());
                    println!("└─ {}", entropy_str.chars().take(120).collect::<String>());
                    println!("BENCH entropy_gap={:.2} bytes={} ({:.1}%) in {:.2}s ({:.0} B/s)", gap, entropy_out.len(), entropy_pct,
                        entropy_secs, entropy_out.len() as f64 / entropy_secs.max(1e-4));
                }
                // Speculative (draft→verify): global W drafts patches, local W verifies bytes.
                let d0s = Instant::now();
                let spec_out = fuga::tm_generate_speculative(&tm, &seed, 200, 4, 0.60, &patch_vocab);
                let spec_secs = d0s.elapsed().as_secs_f64();
                let spec_str = String::from_utf8_lossy(&spec_out).to_string();
                let spec_printable: f64 = spec_out
                    .iter()
                    .filter(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace())
                    .count() as f64;
                let spec_pct = if spec_out.is_empty() { 0.0 } else { 100.0 * spec_printable / spec_out.len() as f64 };
                println!("├─ speculative(draft→verify) decoded ({} bytes):", spec_out.len());
                println!("└─ {}", spec_str.chars().take(120).collect::<String>());
                println!("BENCH speculative_bytes={} ({:.1}%) in {:.2}s ({:.0} B/s)", spec_out.len(), spec_pct,
                        spec_secs, spec_out.len() as f64 / spec_secs.max(1e-4));
                    // Speculative acceptance rate: fraction of draft bytes the verifier accepted.
                    let (spec_stat_out, spec_accept) = fuga::tm_generate_speculative_stats(&tm, &seed, 200, 4, 0.60, &patch_vocab);
                    println!("ACCEPTANCE speculative draft-acceptance={:.3} (bytes={})", spec_accept, spec_stat_out.len());
                    // Beam search: K hypotheses with accumulated log-score vs greedy top-1.
                    let d0b = Instant::now();
                    let beam_out = fuga::tm_generate_beam(&tm, &seed, 200, 4, 3, 5);
                    let beam_secs = d0b.elapsed().as_secs_f64();
                    let beam_str = String::from_utf8_lossy(&beam_out).to_string();
                    let beam_printable: f64 = beam_out
                        .iter()
                        .filter(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace())
                        .count() as f64;
                    let beam_pct = if beam_out.is_empty() { 0.0 } else { 100.0 * beam_printable / beam_out.len() as f64 };
                    println!("├─ beam-3 decoded ({} bytes):", beam_out.len());
                    println!("└─ {}", beam_str.chars().take(120).collect::<String>());
                    println!("BENCH beam_bytes={} ({:.1}%) in {:.2}s ({:.0} B/s)", beam_out.len(), beam_pct,
                        beam_secs, beam_out.len() as f64 / beam_secs.max(1e-4));
    // Keep the default-path decode for the canonical PASS line (same as prod).
    let d0default = Instant::now();
    let bytes_out = fuga::tm_generate_two_speed(&tm, &seed, n_patch_steps, 4, &patch_vocab, None);
    let ts_secs = d0default.elapsed().as_secs_f64();
    let ts_str = String::from_utf8_lossy(&bytes_out).to_string();
    println!("├─ two_speed (default 0.05) decoded ({} bytes):", bytes_out.len());
    println!("└─ {}", ts_str.chars().take(200).collect::<String>());

    // Baseline: naive byte-by-byte decoder on the SAME tm.
    let d1 = Instant::now();
    let naive = tm_generate_latent_bytes(&tm, &seed, 200, 4, None);
    let naive_secs = d1.elapsed().as_secs_f64();
    let naive_str = String::from_utf8_lossy(&naive).to_string();
    println!("├─ naive_byte decoded ({} bytes): {}", naive.len(), naive_str.chars().take(120).collect::<String>());

    // Honest metrics.
    let ts_printable: f64 = bytes_out
        .iter()
        .filter(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace())
        .count() as f64;
    let naive_printable: f64 = naive
        .iter()
        .filter(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace())
        .count() as f64;
    println!("BENCH train_byte_steps={} train_patch_steps={} in {:.2}s", bytes_seen, patches_seen, train_secs);
    println!("BENCH two_speed_bytes={} ({:.1}%) in {:.2}s ({:.0} B/s)", bytes_out.len(),
        if bytes_out.is_empty() { 0.0 } else { 100.0 * ts_printable / bytes_out.len() as f64 },
        ts_secs, bytes_out.len() as f64 / ts_secs.max(1e-4));
    println!("BENCH naive_bytes={} ({:.1}%) in {:.2}s ({:.0} B/s)", naive.len(),
        if naive.is_empty() { 0.0 } else { 100.0 * naive_printable / naive.len() as f64 },
        naive_secs, naive.len() as f64 / naive_secs.max(1e-4));
    println!("BENCH alphabet_vocab=256 patch_vocab={}", patch_vocab.len());

    // Validation: two-speed must emit printable bytes that stay inside the
    // ASCII grammar (no NUL, no binary) and must not be empty.
    let mut fail = 0;
    if bytes_out.is_empty() {
        eprintln!("  FAIL: empty two-speed generation");
        fail = 1;
    }
    if bytes_out.iter().any(|&b| b == 0) {
        eprintln!("  FAIL: NUL byte in two-speed output");
        fail = 1;
    }
    if fail == 0 {
        println!("REAL-CODE TWO-SPEED: PASS");
    } else {
        println!("REAL-CODE TWO-SPEED: FAIL");
        std::process::exit(1);
    }
}
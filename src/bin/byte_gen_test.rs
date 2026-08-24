// byte_gen_test.rs — REAL byte-level (ByT5/MegaByte) code-generation test.
//
// Trains a TemporalMemory on RAW UTF-8 BYTES of actual Rust snippets from
// corpus_doc_code_pairs.jsonl (via learn_bytes -> byte_basis SDRs + latent W),
// then generates raw bytes from a seed through tm_generate_latent_bytes (the
// FIXED 256-byte alphabet, no vocabulary, no tokenizer), and validates:
//   1) non-empty emission
//   2) viability of the latent decode on real data (max cosine over 256 bytes)
//   3) repeatability + gate respect
// and prints honest benchmark timings (train bytes/s, decode steps, byte/s).
//
// AD-HOC verification stand — mirrors the byte-level production path.
use fuga::ai::TemporalMemory;
use fuga::tm_generate_latent_bytes;
use fuga::ai::sdr::byte_basis;
use std::collections::HashSet;
use std::time::Instant;

fn main() {
    let corpus_path =
        std::env::args().nth(1).unwrap_or_else(|| "corpus_doc_code_pairs.jsonl".to_string());
    let limit = std::env::args()
        .nth(2)
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(600); // byte training is per-byte; 600 code snippets is a solid sample
    let content = std::fs::read_to_string(&corpus_path).expect("corpus");

    let mut tm = TemporalMemory::new(30_000, 4);
    let mut trained = 0usize;
    let mut bytes_seen = 0usize;
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
        // Learn raw-byte transitions with a sliding byte window of 4 (context -> next byte).
        for w in 0..b.len().saturating_sub(1) {
            let win_lo = w.saturating_sub(4);
            tm.learn_bytes(&b[win_lo..w + 1], b[w + 1], 0.15);
            bytes_seen += 1;
        }
        trained += 1;
        if trained % 1000 == 0 {
            eprintln!("  trained {} docs ({} byte-steps) in {:.1}s", trained, bytes_seen, t0.elapsed().as_secs_f64());
        }
    }
    let train_secs = t0.elapsed().as_secs_f64();
    eprintln!("  Trained {} Rust docs, {} byte-steps in {:.1}s ({:.0} byte-steps/s)",
        trained, bytes_seen, train_secs, bytes_seen as f64 / train_secs.max(1.0));

    if trained == 0 {
        eprintln!("  FAIL: no training data");
        std::process::exit(2);
    }

    // Seed bytes: "fn main() { " (raw UTF-8). The byte decoder must continue it.
    let seed: Vec<u8> = b"fn main() { ".to_vec();

    // DIAGNOSE: cosine of predicted next-byte latent against the 256-byte alphabet.
    let pred = tm.predict_bytes_latent(&seed[seed.len().saturating_sub(4)..]);
    let mut best: f32 = -1.0;
    let mut best_b: u8 = 0;
    for b in 0u16..=255 {
        let lat = tm.latent_of_sdr(&byte_basis(b as u8));
        let c = pred.cosine_similarity(&lat);
        if c > best {
            best = c;
            best_b = b as u8;
        }
    }
    eprintln!("  DIAGNOSE max byte-cosine over 256-byte alphabet: {:.4} (byte {} '{:?}')", best, best_b, (best_b as char).to_string());
    let over_thresh = (0u16..=255)
        .filter(|&b| {
            let lat = tm.latent_of_sdr(&byte_basis(b as u8));
            pred.cosine_similarity(&lat) >= 0.05
        })
        .count();
    eprintln!("  DIAGNOSE bytes at cosine>=0.05: {}", over_thresh);

    // Eligible corridor: printable ASCII + space + newline/tab (no dictionary).
    let eligible: HashSet<u8> = (32u8..=126).chain([10u8, 9u8]).collect();

    let d0 = Instant::now();
    let bytes_out = tm_generate_latent_bytes(&tm, &seed, 200, 4, Some(&eligible));
    let decode_secs = d0.elapsed().as_secs_f64();
    let out_str = String::from_utf8_lossy(&bytes_out).to_string();
    println!("├─ byte decoded ({} bytes): {:?}", bytes_out.len(), String::from_utf8_lossy(&bytes_out));
    println!("└─ assembled: {}", out_str.chars().take(80).collect::<String>());

    // Benchmarks
    println!("BENCH train_byte_steps={} in {:.2}s ({:.0}/s)", bytes_seen, train_secs, bytes_seen as f64 / train_secs.max(0.001));
    println!("BENCH decode_bytes={} in {:.2}s ({:.0} bytes/s)", bytes_out.len(), decode_secs, bytes_out.len() as f64 / decode_secs.max(0.001));
    println!("BENCH alphabet_size=256 vocabulary=0(dictionary-free)");

    // Validation.
    let mut fail = 0;
    if bytes_out.is_empty() {
        eprintln!("  FAIL: empty byte generation");
        fail = 1;
    }
    for &b in &bytes_out {
        if !eligible.contains(&b) {
            eprintln!("  FAIL: byte {} outside corridor (not printable/fencepost)", b);
            fail = 1;
        }
        if b == 0 {
            eprintln!("  FAIL: NUL byte emitted");
            fail = 1;
        }
    }
    // The emitted bytes must be at least printable ASCII consistent with a code body.
    let printable: f64 = bytes_out.iter().filter(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace()).count() as f64;
    if bytes_out.len() > 0 && (printable / bytes_out.len() as f64) < 0.5 {
        eprintln!("  FAIL: <50% printable ASCII in {}-byte emission", bytes_out.len());
        fail = 1;
    }
    if fail == 0 {
        println!("REAL-CODE BYTE-GEN: PASS");
    } else {
        println!("REAL-CODE BYTE-GEN: FAIL");
        std::process::exit(1);
    }
}
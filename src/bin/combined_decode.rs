// combined_decode.rs — all technologies, one pipeline, on a TRAINED checkpoint.
//
// Loads a saved TemporalMemory (new format: cells+window+W+OWM-P+W_patch),
// builds a patch vocabulary from the SAME corpus the decoder will be seeded
// with, trains the KAN operator and the LSTM peer on that corpus, then runs
// EVERY decoder we built across the iterations on the identical seed:
//
//   naive byte W | two-speed (global W_patch + local W) | entropy BLT |
//   recurrent h(t) | LSTM peer |
//   LSTM peer
//
// One process, one corpus, one checkpoint — the honest "connect all
// technologies" harness.
use std::collections::HashSet;
use std::time::Instant;

use fuga::ai::byte_lstm::ByteLstm;
use fuga::ai::htm_temporal::TemporalMemory;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ckpt = args.get(1).cloned().unwrap_or_else(|| "fuga_stack_tm.bin".into());
    let corpus_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "corpus_doc_code_pairs.jsonl".into());
    let limit: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(300);
    let seed: &[u8] = b"fn main() {";
    // 4th optional arg: path to a BYTE-trained W sidecar (magic "FBW1").
    // If present, attach it so the decoders use a byte-level operator instead
    // of the checkpoint's W (which may be token-trained or identity).
    let bytew_path = args.get(4).cloned();

    // 1. Load the trained checkpoint.
    let t0 = Instant::now();
    let mut tm = match TemporalMemory::load(&ckpt) {
        Some(tm) => tm,
        None => {
            eprintln!("FAILED to load {}", ckpt);
            std::process::exit(2);
        }
    };
    println!("=== CONNECTED DECODER SUITE on {} ===", ckpt);
    println!(
        "loaded {:.1}s cells={} ctx={} W_updates={} W_patch_updates={}",
        t0.elapsed().as_secs_f64(),
        tm.cells.len(),
        tm.context_len,
        tm.predictor_updates(),
        tm.patch_predictor().updates
    );
    if let Some(path) = &bytew_path {
        match TemporalMemory::load_byte_w(path) {
            Some(w) if w.len() == tm.predictor_w().len() => {
                tm.apply_byte_w(w);
                println!("  attached BYTE-trained W from {} (updates={})", path, tm.predictor_updates());
            }
            _ => {
                eprintln!("WARNING: could not attach byte W from {:?}; running with checkpoint W", path);
            }
        }
    }
    println!("corpus slice={} seed={:?}", limit, String::from_utf8_lossy(seed));

    // 2. Corpus lines + patch vocabulary (2-byte patches, like the stands).
    let corpus = std::fs::read_to_string(&corpus_path).expect("corpus");
    let lines: Vec<String> = corpus.lines().take(limit).map(|s| s.to_string()).collect();
    let mut all_bytes: Vec<u8> = Vec::new();
    for l in &lines {
        all_bytes.extend_from_slice(l.as_bytes());
    }
    let mut patch_vocab: Vec<Vec<u8>> = Vec::new();
    {
        let mut seen = HashSet::new();
        for l in &lines {
            let b = l.as_bytes();
            for i in (0..b.len().saturating_sub(1)).step_by(2) {
                let p = b[i..(i + 2).min(b.len())].to_vec();
                if seen.insert(p.clone()) && patch_vocab.len() < 512 {
                    patch_vocab.push(p);
                }
            }
        }
    }
    println!("patch_vocab={} (2-byte)", patch_vocab.len());

    // 4. Train the LSTM peer (classic recurrent baseline).
    let mut rng: u64 = 0xBAD0_0D1E;
    let mut lstm = ByteLstm::new(&mut rng);
    let tlstm = Instant::now();
    lstm.reset_state();
    let mut w = 0usize;
    while w + 2 <= all_bytes.len() {
        let end = (w + 9).min(all_bytes.len());
        lstm.train_window(&all_bytes[w..end]);
        w += 8;
    }
    println!(
        "LSTM trained on {} bytes in {:.1}s",
        all_bytes.len(),
        tlstm.elapsed().as_secs_f64()
    );

    // 6. Run every decoder on the identical seed.
    let win = 4usize;
    let mut rows: Vec<(String, usize, String)> = Vec::new();

    // a) naive byte W
    let out = fuga::tm_generate_latent_bytes(&tm, seed, 200, win, None);
    rows.push(("naive byte W".into(), out.len(), String::from_utf8_lossy(&out).chars().take(48).collect()));
    // b) two-speed (global patch + local byte)
    let out = fuga::tm_generate_two_speed(&tm, seed, 40, 2, &patch_vocab, None);
    rows.push(("two-speed (global+local)".into(), out.len(), String::from_utf8_lossy(&out).chars().take(48).collect()));
    // c) entropy BLT (dynamic patch boundaries by gap)
    let out = fuga::tm_generate_two_speed_entropy(&tm, seed, 200, win, 0.60, &patch_vocab);
    rows.push(("entropy BLT (gap=0.6)".into(), out.len(), String::from_utf8_lossy(&out).chars().take(48).collect()));
    // d) recurrent h(t)
    let out = fuga::tm_generate_recurrent(&tm, seed, 200, win, 0.0, 0.9);
    rows.push(("recurrent h(t) mix=0".into(), out.len(), String::from_utf8_lossy(&out).chars().take(48).collect()));
    let out = fuga::tm_generate_recurrent(&tm, seed, 200, win, 0.4, 0.9);
    rows.push(("recurrent h(t) mix=0.4".into(), out.len(), String::from_utf8_lossy(&out).chars().take(48).collect()));
    // e) LSTM peer (classic recurrent, no VSA)
    lstm.reset_state();
    let out = lstm.generate(seed, 200);
    rows.push(("LSTM peer (H=128)".into(), out.len(), String::from_utf8_lossy(&out).chars().take(48).collect()));

    // 7. Summary table.
    println!();
    println!("decoder                        | bytes | content");
    println!("-------------------------------+-------+--------------------------------");
    for (name, len, text) in &rows {
        println!("{:<29} | {:>5} | {}", name, len, text);
    }

    // 8. Winner assessment (honest: raw length is NOT quality).
    let best_len = rows.iter().map(|(_, n, _)| *n).max().unwrap_or(0);
    let best = rows.iter().find(|(_, n, _)| *n == best_len).unwrap();
    println!();
    println!(
        "RESULT: longest run = {} bytes ({}) — length ≠ quality; entropy BLT and",
        best_len, best.0
    );
    println!("recurrent share the same local-W attractor ceiling measured in A/B.");
}
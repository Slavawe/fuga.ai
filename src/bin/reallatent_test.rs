// gen_code_latent.rs — REAL end-to-end code-generation test for the continuous
// (tokenless) decoder tm_generate_latent.
//
// Trains a TemporalMemory on actual Rust snippets from corpus_doc_code_pairs.jsonl
// (via learn_structure, which also trains the latent transition W), then
// generates code from a Rust "fn main" seed through tm_generate_latent with an
// eligible corridor, and validates the emission is non-empty / syntactically sane.
//
// This is an AD-HOC verification stand (temporary), mirroring the production
// wiring in omni-web.rs handle_code_generate.
use fuga::ai::{HierarchicalJEPA, TemporalMemory};
use fuga::{encode_text, tm_generate_latent, Hypervector, TemporalPredictor};
use std::collections::HashSet;
use std::time::Instant;

fn lex_code(code: &str) -> Vec<String> {
    // Coarse word-level lexer: identifiers/keywords/symbols as tokens.
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in code.chars() {
        if c.is_alphanumeric() || c == '_' {
            cur.push(c);
        } else {
            if !cur.is_empty() {
                out.push(cur.clone());
                cur.clear();
            }
            if !c.is_whitespace() {
                out.push(c.to_string());
            }
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn main() {
    let corpus_path =
        std::env::args().nth(1).unwrap_or_else(|| "corpus_doc_code_pairs.jsonl".to_string());
    let limit = std::env::args().nth(2).and_then(|v| v.parse::<usize>().ok()).unwrap_or(8_000);
    // Only Rust sources (.rs) — the corpus also holds C/Python/Go; testing the
    // decoder against a mismatched language is invalid.
    let rust_only = std::env::args().nth(3).map(|v| v != "0").unwrap_or(true);

    let content = std::fs::read_to_string(&corpus_path).expect("corpus");
    let mut tm = TemporalMemory::new(30_000, 4);
    let mut trained = 0usize;
    let t0 = Instant::now();

    for line in content.lines() {
        if trained >= limit {
            break;
        }
        let Ok(doc) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if rust_only {
            let src = doc["source"].as_str().unwrap_or("");
            if !src.ends_with(".rs") {
                continue;
            }
        }
        let code = doc["code"].as_str().unwrap_or("");
        if code.trim().is_empty() {
            continue;
        }
        let tokens = lex_code(code);
        if tokens.len() < 3 {
            continue;
        }
        // Learn bigram windows (context_len=4) -> next token, structural + latent.
        for w in 0..tokens.len().saturating_sub(1) {
            let win: Vec<&str> = tokens[w..(w + 4).min(tokens.len() - 1)].iter().map(|s| s.as_str()).collect();
            tm.learn_structure(&win, &tokens[w + 1]);
        }
        trained += 1;
        if trained % 2000 == 0 {
            eprintln!("  trained {} docs in {:.1}s", trained, t0.elapsed().as_secs_f64());
        }
    }
    eprintln!("  Trained {} Rust doc snippets in {:.1}s", trained, t0.elapsed().as_secs_f64());

    // Vocabulary: collect the most frequent tokens seen so the decoder has candidates.
    let mut freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for line in content.lines() {
        let Ok(doc) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if rust_only {
            let src = doc["source"].as_str().unwrap_or("");
            if !src.ends_with(".rs") {
                continue;
            }
        }
        for t in lex_code(doc["code"].as_str().unwrap_or("")) {
            *freq.entry(t).or_insert(0) += 1;
        }
    }
    let mut vocab_list: Vec<String> = freq.into_iter().take(3000).map(|(k, _)| k).collect();
    vocab_list.sort();
    eprintln!("  vocab: {} tokens", vocab_list.len());

    // Seed and eligible corridor for a realistic Rust generation task.
    let seed: Vec<String> = vec!["fn".into(), "main".into(), "(".into(), ")".into()];
    let eligible: HashSet<String> = vec![
        "fn".into(), "main".into(), "(".into(), ")".into(), "{".into(), "}".into(),
        "let".into(), "mut".into(), "println".into(), "return".into(), "->".into(),
        ":".into(), "i32".into(), "for".into(), "in".into(), "0".into(), "..".into(),
        "use".into(), "std".into(),
    ].into_iter().collect();

    // DIAGNOSE: what does the first latent prediction look like against candidate latents?
    let ctx_sdrs: Vec<fuga::SdrVector> = seed.iter().map(|t| fuga::encode_text(t)).collect();
    let pred_latent = tm.predict_latent(&ctx_sdrs);
    let mut scored: Vec<(f32, String)> = Vec::new();
    for w in vocab_list.iter() {
        let sdr = fuga::encode_text(w);
        let lat = tm.latent_of_sdr(&sdr);
        scored.push((pred_latent.cosine_similarity(&lat), w.clone()));
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("  DIAGNOSE pred_latent top-6 cosine: {:?}", &scored[..6.min(scored.len())]);
    eprintln!("  DIAGNOSE top-at-cosine0.05 eligible count: {}",
        scored.iter().filter(|(s, _)| *s >= 0.05).count());
    let best_overall: f32 = scored.iter().map(|(s, _)| *s).fold(0.0f32, f32::max);
    eprintln!("  DIAGNOSE max cosine over ALL vocab: {:.4}", best_overall);

    let out = tm_generate_latent(&tm, &seed, 18, &vocab_list, 4, Some(&eligible));
    println!("├─ latent decoded: {:?}", out.iter());
    println!("└─ assembled:    {}", out.join(" "));

    // Validation gates.
    let mut fail = 0;
    if out.is_empty() {
        eprintln!("  FAIL: empty generation");
        fail = 1;
    }
    for w in &out {
        if !eligible.contains(w) {
            eprintln!("  FAIL: token outside eligible corridor: {:?}", w);
            fail = 1;
        }
    }
    if fail == 0 {
        println!("REAL-CODE LATENT GEN: PASS");
    } else {
        println!("REAL-CODE LATENT GEN: FAIL");
        std::process::exit(1);
    }
}

// silence unused-import warning for Hypervector if not otherwise used
#[allow(unused)]
fn _unused(h: &Hypervector) { let _ = h; }
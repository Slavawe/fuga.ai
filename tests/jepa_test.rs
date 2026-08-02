use fuga::Hypervector;
use fuga::ai::jepa::JepaPredictor;

#[test]
fn test_jepa_new_defaults() {
    let jepa = JepaPredictor::new(8192, 4);
    assert_eq!(jepa.dim, 8192);
    assert_eq!(jepa.context_len, 4);
    assert_eq!(jepa.perm_offsets.len(), 4);
    assert_eq!(jepa.weights.len(), 4);
    let sum: f64 = jepa.weights.iter().sum();
    assert!((sum - 1.0).abs() < 1e-12);
}

#[test]
fn test_jepa_context_len_clamped() {
    let jepa = JepaPredictor::new(8192, 0);
    assert_eq!(jepa.context_len, 1);
    let jepa2 = JepaPredictor::new(8192, 100);
    assert_eq!(jepa2.context_len, 8);
}

#[test]
fn test_jepa_predict_empty_context() {
    let jepa = JepaPredictor::new(8192, 4);
    let ctx: Vec<&Hypervector> = vec![];
    let result = jepa.predict(&ctx);
    assert_eq!(result.dim, 8192);
    let entropy = result.entropy();
    assert!((entropy - 0.5).abs() < 0.05);
}

#[test]
fn test_jepa_predict_single() {
    let jepa = JepaPredictor::new(8192, 4);
    let hv = Hypervector::random(8192);
    let result = jepa.predict(&[&hv]);
    assert_eq!(result.dim, 8192);
    let sim = result.similarity(&hv);
    assert!(sim > 0.49);
}

#[test]
fn test_jepa_predict_deterministic() {
    let jepa = JepaPredictor::new(8192, 4);
    let a = Hypervector::random(8192);
    let b = Hypervector::random(8192);
    let r1 = jepa.predict(&[&a, &b]);
    let r2 = jepa.predict(&[&a, &b]);
    assert_eq!(r1.words, r2.words);
}

#[test]
fn test_jepa_train_reduces_loss() {
    let mut jepa = JepaPredictor::new(8192, 3);
    let seq: Vec<Hypervector> = (0..10).map(|_| Hypervector::random(8192)).collect();
    let sequences = vec![seq];
    let loss_before = {
        let mut total = 0.0;
        let mut count = 0;
        for seq in &sequences {
            if seq.len() < jepa.context_len + 1 {
                continue;
            }
            for i in 0..seq.len() - jepa.context_len {
                let ctx: Vec<&Hypervector> = seq[i..i + jepa.context_len].iter().collect();
                let pred = jepa.predict(&ctx);
                let sim = pred.similarity(&seq[i + jepa.context_len]);
                total += 1.0 - sim;
                count += 1;
            }
        }
        if count > 0 { total / count as f64 } else { 1.0 }
    };
    let loss_after = jepa.train_on_sequences(&sequences, 50);
    assert!(loss_after <= loss_before + 0.01);
}

#[test]
fn test_jepa_save_load_roundtrip() {
    let path = "/tmp/test_jepa_save.bin";
    let jepa = JepaPredictor::new(8192, 4);
    jepa.save(path).unwrap();
    let loaded = JepaPredictor::load(path).unwrap();
    assert_eq!(jepa.dim, loaded.dim);
    assert_eq!(jepa.context_len, loaded.context_len);
    assert_eq!(jepa.perm_offsets, loaded.perm_offsets);
    for (a, b) in jepa.weights.iter().zip(loaded.weights.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
    std::fs::remove_file(path).ok();
}

#[test]
fn test_jepa_similarity_to_expected() {
    let jepa = JepaPredictor::new(8192, 4);
    let seq: Vec<Hypervector> = (0..5).map(|_| Hypervector::random(8192)).collect();
    let ctx: Vec<&Hypervector> = seq[..4].iter().collect();
    let sim = jepa.similarity_to_expected(&ctx, &seq[4]);
    assert!(sim >= 0.0 && sim <= 1.0);
}

#[test]
fn test_jepa_train_convergence() {
    let mut jepa = JepaPredictor::new(8192, 2);
    let a = Hypervector::random(8192);
    let b = Hypervector::random(8192);
    let c = Hypervector::random(8192);
    let sequences = vec![vec![a, b, c]];
    let loss = jepa.train_on_sequences(&sequences, 100);
    assert!(loss < 0.55);
}

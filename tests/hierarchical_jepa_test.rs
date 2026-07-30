use fuga::HierarchicalJEPA;

#[test]
fn test_hjepa_new_defaults() {
    let hj = HierarchicalJEPA::new(8192);
    assert_eq!(hj.dim, 8192);
    assert_eq!(hj.levels.len(), 3);
    assert_eq!(hj.levels[0].name, "L0");
    assert_eq!(hj.levels[1].name, "L1");
    assert_eq!(hj.levels[2].name, "L2");
    assert_eq!(hj.levels[0].context_len, 4);
    assert_eq!(hj.levels[1].context_len, 3);
    assert_eq!(hj.levels[2].context_len, 2);
    assert_eq!(hj.levels[0].stride, 1);
    assert_eq!(hj.levels[1].stride, 3);
    assert_eq!(hj.levels[2].stride, 5);
    assert_eq!(hj.levels[0].perm_offsets.len(), 4);
    assert_eq!(hj.levels[1].weights.len(), 3 * 8192);
}

#[test]
fn test_hjepa_level_predict() {
    let hj = HierarchicalJEPA::new(8192);
    let hv1 = fuga::Hypervector::random(8192);
    let hv2 = fuga::Hypervector::random(8192);
    let hv3 = fuga::Hypervector::random(8192);
    let hv4 = fuga::Hypervector::random(8192);
    let ctx = vec![&hv1, &hv2, &hv3, &hv4];

    let preds = hj.predict(&ctx);
    assert_eq!(preds.len(), 3);
    for p in &preds {
        assert_eq!(p.dim, 8192);
        assert!(p.entropy() > 0.15);
    }
}

#[test]
fn test_hjepa_level_similarity() {
    let hj = HierarchicalJEPA::new(8192);
    let hv1 = fuga::Hypervector::random(8192);
    let hv2 = fuga::Hypervector::random(8192);
    let hv3 = fuga::Hypervector::random(8192);
    let hv4 = fuga::Hypervector::random(8192);
    let ctx = vec![&hv1, &hv2, &hv3];

    for lvl in &hj.levels {
        let sim = lvl.similarity_to_expected(&ctx, &hv4);
        assert!(sim >= -1.0 && sim <= 1.0, "sim={}", sim);
    }
}

#[test]
fn test_hjepa_save_load_roundtrip() {
    let hj = HierarchicalJEPA::new(8192);
    let path = "/tmp/test_hjepa.bin";
    hj.save(path).unwrap();
    let loaded = HierarchicalJEPA::load(path).unwrap();
    assert_eq!(loaded.dim, hj.dim);
    assert_eq!(loaded.levels.len(), hj.levels.len());
    for i in 0..3 {
        assert_eq!(loaded.levels[i].name, hj.levels[i].name);
        assert_eq!(loaded.levels[i].context_len, hj.levels[i].context_len);
        assert_eq!(loaded.levels[i].stride, hj.levels[i].stride);
        assert_eq!(loaded.levels[i].perm_offsets, hj.levels[i].perm_offsets);
        assert_eq!(loaded.levels[i].weights, hj.levels[i].weights);
    }
    std::fs::remove_file(path).ok();
}

#[test]
fn test_hjepa_predict_entropy_range() {
    let hj = HierarchicalJEPA::new(8192);
    let hvs: Vec<fuga::Hypervector> = (0..6).map(|_| fuga::Hypervector::random(8192)).collect();
    let ctx: Vec<&fuga::Hypervector> = hvs.iter().collect();
    let preds = hj.predict(&ctx);

    for p in &preds {
        // Entropy should be in reasonable range for binary HV (0.0-1.0)
        let e = p.entropy();
        assert!(e > 0.05, "entropy too low: {}", e);
    }
}

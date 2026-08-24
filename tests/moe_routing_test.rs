use fuga::MoEStore;

#[test]
fn test_moe_routing_purity() {
    let tests = vec![
        ("fn hamming_distance(a: &Hypervector) -> usize", "code"),
        (
            "impl MemoryStore for WaveCube { fn store(&mut self) {} }",
            "code",
        ),
        ("struct Vec3 { x: f32, y: f32, z: f32 }", "code"),
        ("python script with def function", "code"),
        ("rust async tokio runtime", "code"),
        ("javascript closure arrow function", "code"),
        ("Привет, как дела?", "dialogue"),
        ("hello world hi there", "dialogue"),
        ("расскажи историю про кота", "narrative"),
        ("this novel has three chapters", "narrative"),
        ("deploy nginx docker container", "sysadmin"),
        ("сервер админ конфиг", "sysadmin"),
        ("this is a general question about weather", "general"),
        ("что такое энтропия термодинамика", "general"),
    ];

    let mut correct = 0;
    for (query, expected) in &tests {
        let got = MoEStore::domain_for(query);
        let ok = got == *expected;
        if ok {
            correct += 1;
        }
        eprintln!(
            "  [{}] '{:.50}' → {:?} (expected {:?})",
            if ok { "✓" } else { "✗" },
            query,
            got,
            expected
        );
        assert_eq!(got, *expected, "routing mismatch for: {}", query);
    }
    eprintln!("\n  Accuracy: {}/{}", correct, tests.len());
}

#[test]
fn test_moe_routing_no_overlap_code_vs_dialogue() {
    let code_queries = vec![
        "fn sort_array(v: &mut [i32])",
        "impl Trait for Struct",
        "python async def handler():",
        "rust closure move |x| x + 1",
    ];
    let dialogue_queries = vec![
        "привет как ты",
        "hi how are you today",
        "hello world greeting",
        "добрый день собеседник",
    ];

    for q in &code_queries {
        assert_eq!(
            MoEStore::domain_for(q),
            "code",
            "code query misrouted: {}",
            q
        );
    }
    for q in &dialogue_queries {
        assert_eq!(
            MoEStore::domain_for(q),
            "dialogue",
            "dialogue query misrouted: {}",
            q
        );
    }
}

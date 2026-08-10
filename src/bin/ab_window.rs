// window A/B: один чекпоинт, один сид, окно 2..6
use fuga::ai::htm_temporal::TemporalMemory;
fn main() {
    let mut tm = TemporalMemory::new(64, 4);
    assert!(tm.load_unified_fuga1("fuga_unified_v4.fuga"), "load");
    for w in [2usize, 3, 4, 5, 6] {
        let out = fuga::tm_generate_latent_bytes(&tm, b"in the beginning", 60, w, None);
        println!(
            "window={} -> {}B: {:?}",
            w,
            out.len(),
            String::from_utf8_lossy(&out).chars().take(60).collect::<String>()
        );
    }
}
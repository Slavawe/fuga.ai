// unified_roundtrip_cpp.rs — Rust пишет FUGA1 → C++ должен прочитать.
// Usage: unified_roundtrip_cpp <out.fuga>
use fuga::ai::htm_temporal::{save_unified, UnifiedMeta};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/rust_written.fuga".into());
    let n = 512 * 512;
    // Различимые веса: local = 0.5*i/n, patch = -0.5*i/n, OWM-P = диагональ 2.0
    let local_w: Vec<f32> = (0..n).map(|i| 0.5 * (i as f32) / (n as f32)).collect();
    let patch_w: Vec<f32> = (0..n).map(|i| -0.5 * (i as f32) / (n as f32)).collect();
    let owm_p: Vec<f32> = (0..n)
        .map(|i| if i % (512 + 1) == 0 { 2.0 } else { 0.0 })
        .collect();
    let meta = UnifiedMeta {
        steps: 777,
        patch_steps: 888,
        ctx: 4,
        version: 1,
    };
    save_unified(&path, &local_w, &patch_w, &owm_p, &meta, None).expect("save");
    println!("wrote {} ({} bytes)", path, n * 3 * 4 + 64);
    // Проверка на лету тем же Rust-кодом
    if let Some((l, p, o, m, _)) = fuga::ai::htm_temporal::load_unified(&path) {
        println!(
            "self-check: local[0]={:.4} patch[0]={:.4} owm[0]={:.4} steps={} (OWM diag=2.0)",
            l[0], p[0], o[0], m.steps
        );
    }
}
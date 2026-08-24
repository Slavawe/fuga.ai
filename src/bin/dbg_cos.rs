// Отладка: топ-3 косинуса на первых 4 шагах, сравнение molecule с naive-потоком
use fuga::ai::htm_temporal::TemporalMemory;
fn main() {
    let mut tm = TemporalMemory::new(64, 4);
    assert!(tm.load_unified_fuga1("fuga_unified_v4.fuga"), "load");
    let seed = b"in the beginning";
    let enc = &tm.predictor().encoder;

    // Реплика naive шаг за шагом
    let mut state: Vec<u8> = seed.to_vec();
    let byte_lats: Vec<(u8, fuga::LatentVector)> = (0u16..=255).map(|b| {
        let sdr = fuga::ai::sdr::byte_basis(b as u8);
        let lat = enc.encode(&sdr);
        (b as u8, lat)
    }).collect();

    for step in 0..6 {
        let win_lo = state.len().saturating_sub(5);
        let win = &state[win_lo..];
        let wsdrs: Vec<fuga::SdrVector> = win.iter().map(|&b| fuga::ai::sdr::byte_basis(b)).collect();
        let pred = tm.predict_bytes_latent(win);
        let mut scores: Vec<(u8, f32)> = byte_lats.iter()
            .map(|(b, l)| (*b, pred.cosine_similarity(l)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top: String = scores.iter().take(5)
            .map(|(b, s)| format!("{}:{:.3}", *b as char, s)).collect::<Vec<_>>().join(" ");
        println!("step{} win={:?} | top: {}", step, String::from_utf8_lossy(win), top);
        let pick = scores[0].0;
        state.push(pick);
    }
}

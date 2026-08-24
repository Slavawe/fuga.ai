// Быстрая проверка: обучить tm на короткой последовательности патчей,
// потом прогнать tm_generate_megabyte и сравнить с two_speed.
use fuga::ai::htm_temporal::TemporalMemory;
use fuga::tm_generate_megabyte;
use fuga::tm_generate_two_speed;

fn main() {
    let mut tm = TemporalMemory::new(64, 4);
    // Патчи по 2 байта: "fn", " m", "ai", "n(", "{\n", "  ", "le", "t " ...
    let patches: Vec<Vec<u8>> = [
        "fn", " m", "ai", "n(", "{", "}", "le", "t ", "x ", "= ", "4;", "\n",
    ]
    .iter()
    .map(|s| s.as_bytes().to_vec())
    .collect();
    // Учим патчевый оператор: окно 2 патча → следующий патч
    for w in 0..patches.len().saturating_sub(2) {
        let win: Vec<&[u8]> = patches[w..w + 2].iter().map(|p| p.as_slice()).collect();
        tm.learn_patch(&win, &patches[w + 2], 0.2);
    }
    // Учим байтовые переходы внутри патчей
    for p in &patches {
        for i in 0..p.len().saturating_sub(1) {
            tm.learn_bytes(&p[..=i], p[i + 1], 0.3);
        }
    }
    let seed = b"fn m";
    let vocab = patches.clone();
    println!("seed: {:?}", String::from_utf8_lossy(seed));

    let two = tm_generate_two_speed(&tm, seed, 8, 2, &vocab, None);
    println!("two_speed   ({} B): {:?}", two.len(), String::from_utf8_lossy(&two));

    for lambda in [0.0f32, 0.5, 1.0] {
        let mb = tm_generate_megabyte(&tm, seed, 40, 4, 2, &vocab, lambda);
        println!("megabyte λ={:.1} ({} B): {:?}", lambda, mb.len(), String::from_utf8_lossy(&mb));
    }
}

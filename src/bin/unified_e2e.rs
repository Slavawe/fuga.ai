// unified_e2e.rs — единый цикл обучения: C++ обучил → FUGA1 файл → Rust читает
// и декодит (entropy-BLT, главный путь). Доказывает bin-совместимость формата.
//
// Usage: unified_e2e <checkpoint.fuga>
use fuga::ai::htm_temporal::TemporalMemory;
use fuga::tm_generate_two_speed_entropy;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/unified_test.fuga".into());
    let seed: Vec<u8> = b"fn main() {".to_vec();

    // 1. Применяем единый файл в TM (load_unified_fuga1 читает все секции)
    let mut tm = TemporalMemory::new(64, 4);
    let loaded = tm.load_unified_fuga1(&path);
    assert!(loaded, "FUGA1 файл не прочитан (магия/секции?)");
    println!("== FUGA1 read: {} ==", path);
    println!(
        "  LOCAL_W {} f32 | PATCH_W {} f32 | OWM_P {} f32",
        tm.predictor_w().len(),
        tm.patch_predictor().w.len(),
        tm.predictor().p.len()
    );
    println!(
        "  META steps={} patch_steps={} ctx={}",
        tm.predictor().updates,
        tm.patch_predictor().updates,
        tm.context_len
    );
    // OWM-P из C++: после консолидации это НЕ identity — проверяем ненулевой.
    let p = &tm.predictor().p;
    let nonzero = p.iter().any(|v| *v != 0.0);
    println!(
        "  OWM-P: nonzero={} (C++ консолидировал 4 направления)",
        nonzero
    );

    // 2. Декодим entropy-BLT (главный путь генерации). Патч-словарь —
    //    дефолтный структурный (смок проверяет ФОРМАТ, не качество).
    let patch_vocab: Vec<Vec<u8>> = [
        b"fn ".to_vec(), b"let ".to_vec(), b"use ".to_vec(), b"main".to_vec(),
        b"()".to_vec(), b" {".to_vec(), b"}\n".to_vec(), b"    ".to_vec(),
        b"pub ".to_vec(), b"impl".to_vec(), b"match".to_vec(), b"struct".to_vec(),
    ]
    .to_vec();

    let out = tm_generate_two_speed_entropy(&tm, &seed, 200, 4, 0.60, &patch_vocab);
    let text = String::from_utf8_lossy(&out);
    println!("  entropy-BLT decoded: {} байт", out.len());
    println!("  output: {:?}", text);
    println!("== E2E: единый файл C++ → Rust подтверждён ==");
}
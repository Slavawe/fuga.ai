// v6_validate.rs — валидация чекпоинта ПОЛНОЙ конфигурацией декодера v2
// (AGENTS.md v6.2: α=0, τ=0.01, corridor=0, min_cos=0.001, β=0,
//  rep_word=0.20, rep_phrase=0.8, window=9, PHR_LEN=12)
use fuga::ai::htm_temporal::{load_unified, UnifiedMeta};
use fuga::tm_generate_cosine_gate_v2;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ckpt = args.get(1).cloned().unwrap_or_else(|| "fuga_unified_v6.fuga".into());
    let corpus_path = args.get(2).cloned().unwrap_or_else(|| "corpus.jsonl".into());

    // Загрузка чекпоинта (FUGA1)
    let (local_w, patch_w, owm_p, meta, _hj, kan_c) =
        load_unified(&ckpt).expect("не FUGA1 чекпоинт");
    let _ = &meta;
    println!("== V6.2 VALIDATE: {} ==", ckpt);
    println!("  local_w={} patch_w={} owm_p={} kan={:?}",
             local_w.len(), patch_w.len(), owm_p.len(),
             kan_c.as_ref().map(|k| k.len()));

    // TemporalMemory с энкодером (как в talk_model)
    let mut tm = fuga::ai::htm_temporal::TemporalMemory::new(64, 4);
    tm.apply_byte_w(local_w.clone());
    tm.apply_patch_w(patch_w.clone());

    // Патч-словарь из корпуса (как в unified_gpu_train: чанки по 2 байта)
    let mut patch_vocab: Vec<Vec<u8>> = Vec::new();
    if let Ok(text) = std::fs::read_to_string(&corpus_path) {
        let mut seen = std::collections::HashSet::new();
        for line in text.lines() {
            let bytes = line.as_bytes();
            for chunk in bytes.chunks(2) {
                if chunk.len() == 2 && seen.insert(chunk.to_vec()) {
                    patch_vocab.push(chunk.to_vec());
                }
            }
            if patch_vocab.len() > 1000 {
                break;
            }
        }
    }
    println!("  patch_vocab={} (из {})", patch_vocab.len(), corpus_path);

    // KAN (пустой — α=0)
    let mut kan = fuga::ai::kan::KanTransition::new();
    if let Some(c) = &kan_c {
        if c.len() == kan.c.len() {
            kan.c.clone_from(c);
        }
    }

    // Полная конфигурация v6.2 из AGENTS.md
    let (alpha, tau, corridor, min_cos, beta) = (0.0f32, 0.01f32, 0u8, 0.001f32, 0.0f32);
    let (rep_word, rep_phrase) = (0.20f32, 0.8f32);

    let seeds: Vec<&[u8]> = vec![
        b"fn main() {",
        b"the force of gravity is",
        b"in the beginning",
        b"let x = 4",
    ];
    for seed in seeds {
        let out = tm_generate_cosine_gate_v2(
            &tm, &kan, seed, 200, 9, 2, &patch_vocab,
            alpha, tau, corridor, min_cos, beta, 0.0, rep_word, rep_phrase,
        );
        let s = String::from_utf8_lossy(&out);
        println!("\n--- SEED {:?} ---", String::from_utf8_lossy(seed));
        println!("  V2 rep_phrase={} ({} B): {:?}", rep_phrase, out.len(),
                 s.chars().take(90).collect::<String>());
    }
    println!("\n== v6.2 VALIDATE DONE ==");
    // Удержать типы (UnifiedMeta не используется явно)
    let _ = UnifiedMeta::default();
}

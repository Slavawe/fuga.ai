// dialogue_test.rs — НАСТОЯЩАЯ проверка обучения: диалоговый e2e на
// обученном стеке (cube.bin + _mem.bin).
//
// Что проверяет:
//   1. «Распознавание речи» — VSA-роутинг запроса и извлечение релевантных
//      записей из обученной памяти (sim>).
//   2. «Поддержка разговора» — ответы СОДЕРЖИМЫМ памяти (не шаблоном).
//
// Usage: dialogue_test [--cube omni_cube.bin] [--mem omni_cube_mem.bin]
use std::time::Instant;

use fuga::{
    MemoryStore, WaveCube,
    omni::OmniEngine,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cube_path = args
        .iter()
        .position(|a| a == "--cube")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "omni_cube.bin".into());

    // Размерности из заголовка куба (как omni-web main).
    let (ndim, side_len, dim) = match fuga::core::wave_cube::peek_cube_header(&cube_path) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };
    eprintln!("header: ndim={} side={} dim={}", ndim, side_len, dim);
    match (ndim, side_len) {
        (4, 4) => run::<4, 4>(&cube_path, dim),
        (3, 4) => run::<3, 4>(&cube_path, dim),
        (3, 8) => run::<3, 8>(&cube_path, dim),
        (4, 8) => run::<4, 8>(&cube_path, dim),
        (5, 2) => run::<5, 2>(&cube_path, dim),
        (5, 4) => run::<5, 4>(&cube_path, dim),
        other => {
            eprintln!("Unsupported cube: {:?}", other);
            std::process::exit(1);
        }
    }
}

fn run<const N: usize, const S: usize>(cube_path: &str, dim: usize) {
    let t0 = Instant::now();
    let cube = WaveCube::<N, S>::load_bin(cube_path).expect("cube load");
    let mut omni = OmniEngine::<N, S>::new(dim, 3);
    omni.ai.cube = cube;
    let mem_path = cube_path.replace(".bin", "_mem.bin");
    if let Ok(mem) = MemoryStore::load_bin(&mem_path) {
        omni.ai.memory = mem;
    }
    // Строим текстовый индекс: без него search_by_text идёт линейным сканом
    // по всем 604K записей (медленно и без filename-буста). Индекс даёт
    // ИНВЕРТИРОВАННЫЙ поиск по словам — это и есть лексика-first канал.
    omni.ai.memory.build_text_index();
    let mem_size = omni.ai.memory.size();
    println!(
        "Cube {}^{} dim={} | Memory: {} entries | load {:.1}s",
        S,
        N,
        dim,
        mem_size,
        t0.elapsed().as_secs_f64()
    );

    // Серия запросов НА ЯЗЫКЕ КОРПУСА (idf-куб: англ. код): лексика-панк
    // заработает только если слова запроса встречаются в тексте записей.
    let queries = [
        "hello how are you",
        "what is vector symbolic architecture", // частично англ; слова есть в текстах
        "write a function in rust that sorts an array", // "array", "sort" есть в коде
        "explain backpropagation neural network",
        "how does temporal memory work",
        "goodbye",
    ];

    let mut answered = 0usize;
    for q in queries {
        println!("\n────────────────────────────────────────────");
        println!("ВОПРОС: {}", q);
        let out = omni.ai.answer(q);
        let has_content = out.contains("sim=") || out.contains("(text):") || out.contains("lex=");
        if has_content {
            answered += 1;
        }
        println!("ОТВЕТ ({} байт):", out.len());
        for line in out.lines().take(8) {
            println!("  {}", line);
        }
        if !has_content {
            println!("  ⚠ НЕТ retrieval-контента — ответ пустой/шаблонный");
        }
    }

    println!("\n════════════════════════════════════════════");
    println!(
        "ИТОГ: {} из {} запросов получили контент из памяти",
        answered,
        queries.len()
    );
}
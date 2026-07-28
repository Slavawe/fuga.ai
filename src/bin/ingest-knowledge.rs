use std::env;
use std::collections::HashSet;
use fuga::{
    FugaAI, MemoryStore, WaveCube,
    ai::world::{self, chunk_text, make_tokens, SEED_TOPICS},
    core::wave_cube::peek_cube_header,
};

fn main() {
    let cube_path = env::var("FUGA_CUBE_PATH").unwrap_or_else(|_| "fuga_code_cube.bin".into());
    let mem_path = env::var("FUGA_MEM_PATH").unwrap_or_else(|_| "fuga_code_cube_mem.bin".into());
    let dim = env::var("FUGA_DIM").unwrap_or_else(|_| "8192".into()).parse().unwrap_or(8192);
    let out_cube = env::var("FUGA_OUT_CUBE").unwrap_or_else(|_| "fuga_knowledge_cube.bin".into());
    let out_mem = env::var("FUGA_OUT_MEM").unwrap_or_else(|_| "fuga_knowledge_mem.bin".into());
    let topic_limit = env::var("FUGA_TOPIC_LIMIT").ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(SEED_TOPICS.len());

    let (ndim, side_len, _) = peek_cube_header(&cube_path)
        .expect("Failed to read cube header");
    println!("Cube: {}×{}, dim={}", side_len, ndim, dim);

    let mut seen_sources: HashSet<String> = HashSet::new();

    let result = match (ndim, side_len) {
        (3, 4) => ingest::<3, 4>(&cube_path, &mem_path, dim, &out_cube, &out_mem, &mut seen_sources, topic_limit),
        (4, 4) => ingest::<4, 4>(&cube_path, &mem_path, dim, &out_cube, &out_mem, &mut seen_sources, topic_limit),
        (3, 5) => ingest::<3, 5>(&cube_path, &mem_path, dim, &out_cube, &out_mem, &mut seen_sources, topic_limit),
        (3, 6) => ingest::<3, 6>(&cube_path, &mem_path, dim, &out_cube, &out_mem, &mut seen_sources, topic_limit),
        (3, 7) => ingest::<3, 7>(&cube_path, &mem_path, dim, &out_cube, &out_mem, &mut seen_sources, topic_limit),
        (3, 8) => ingest::<3, 8>(&cube_path, &mem_path, dim, &out_cube, &out_mem, &mut seen_sources, topic_limit),
        (4, 8) => ingest::<4, 8>(&cube_path, &mem_path, dim, &out_cube, &out_mem, &mut seen_sources, topic_limit),
        (5, 2) => ingest::<5, 2>(&cube_path, &mem_path, dim, &out_cube, &out_mem, &mut seen_sources, topic_limit),
        (5, 4) => ingest::<5, 4>(&cube_path, &mem_path, dim, &out_cube, &out_mem, &mut seen_sources, topic_limit),
        _ => panic!("Unsupported cube: {}×{}", side_len, ndim),
    };

    match result {
        Ok(n) => println!("\nDone. Ingested {} chunks.", n),
        Err(e) => eprintln!("Error: {}", e),
    }
}

fn ingest<const N: usize, const S: usize>(
    cube_path: &str, mem_path: &str, dim: usize,
    out_cube: &str, out_mem: &str,
    seen_sources: &mut HashSet<String>,
    topic_limit: usize,
) -> Result<usize, String> {
    let (cube, memory) = if std::path::Path::new(out_cube).exists() && std::path::Path::new(out_mem).exists() {
        println!("Resuming from previous output files...");
        let c = WaveCube::<N, S>::load_bin(out_cube)
            .map_err(|e| format!("Resume cube load: {}", e))?;
        let m = MemoryStore::load_bin(out_mem)
            .map_err(|e| format!("Resume memory load: {}", e))?;
        (c, m)
    } else {
        let c = WaveCube::<N, S>::load_bin(cube_path)
            .map_err(|e| format!("Cube load: {}", e))?;
        let m = MemoryStore::load_bin(mem_path)
            .map_err(|e| format!("Memory load: {}", e))?;
        (c, m)
    };

    let mut ai = FugaAI::<N, S>::new(dim, 3);
    ai.cube = cube;
    ai.memory = memory;

    println!("Memory: {} entries", ai.memory.size());

    // Populate seen_sources from existing memory to avoid re-ingestion
    for entry in ai.memory.all_entries() {
        if !entry.source_doc.is_empty() {
            seen_sources.insert(entry.source_doc.clone());
        }
    }
    println!("Already have {} unique sources in memory", seen_sources.len());

    let total = SEED_TOPICS.len().min(topic_limit);
    let mut ingested = 0usize;
    let mut errors = 0usize;

    for (i, topic) in SEED_TOPICS.iter().enumerate().take(topic_limit) {
        if seen_sources.contains(*topic) {
            println!("  [{}/{}] Skipping {} (already ingested)", i+1, total, topic);
            continue;
        }

        print!("  [{}/{}] Fetching {}... ", i+1, total, topic);
        let result = world::fetch_wikipedia(topic);
        let (title, extract) = match result {
            Ok(pair) => pair,
            Err(e) => {
                println!("ERR: {}", e);
                errors += 1;
                continue;
            }
        };

        let chunks = chunk_text(&extract, 256);
        println!("{} chunks", chunks.len());

        for chunk in &chunks {
            let tokens = make_tokens(chunk);
            if tokens.is_empty() { continue; }
            ai.absorb_with_source(&tokens, &title);
            ingested += 1;
        }

        seen_sources.insert(title);

        std::thread::sleep(std::time::Duration::from_millis(500));

        if ingested % 50 == 0 && ingested > 0 {
            println!("  [checkpoint] Saving...");
            ai.cube.save_bin(out_cube).ok();
            ai.memory.save_bin(out_mem).ok();
        }
    }

    println!("\nSaving final cube -> {} and memory -> {}", out_cube, out_mem);
    ai.cube.save_bin(out_cube).map_err(|e| format!("Save cube: {}", e))?;
    ai.memory.save_bin(out_mem).map_err(|e| format!("Save memory: {}", e))?;

    println!("Sources ingested: {}, errors: {}", ingested, errors);
    Ok(ingested)
}

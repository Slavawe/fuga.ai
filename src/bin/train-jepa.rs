use fuga::ai::jepa::JepaPredictor;
use fuga::ai::memory_store::MemoryStore;
use fuga::Hypervector;
use std::env;

const MEM_PATH: &str = "fuga_knowledge_mem.bin";
const JEPA_PATH: &str = "fuga_jepa.bin";

fn main() {
    let args: Vec<String> = env::args().collect();
    let mem_path = args.get(1).map(|s| s.as_str()).unwrap_or(MEM_PATH);
    let jepa_path = args.get(2).map(|s| s.as_str()).unwrap_or(JEPA_PATH);
    let epochs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50);
    let n_seqs: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(50);

    eprintln!("Loading memory from {}...", mem_path);
    let mem = MemoryStore::load_bin(mem_path).expect("Failed to load memory");
    let mem_size = mem.size();
    eprintln!("Loaded {} entries", mem_size);

    if mem_size < 10 {
        eprintln!("Too few entries ({}), need at least 10", mem_size);
        std::process::exit(1);
    }

    let dim = 8192;
    let cl = 4usize;
    let mut jepa = JepaPredictor::new(dim, cl);

    let mut seqs: Vec<Vec<Hypervector>> = Vec::new();
    for _ in 0..n_seqs {
        let rv = Hypervector::random(dim);
        let nearby: Vec<Hypervector> = mem.search(&rv, cl + 1)
            .into_iter().map(|(_, _, e)| e.vector.clone()).collect();
        if nearby.len() >= cl + 1 {
            seqs.push(nearby);
        }
    }

    if seqs.is_empty() {
        eprintln!("Could not build any sequences");
        std::process::exit(1);
    }

    eprintln!("Training JEPA on {} sequences, {} epochs...", seqs.len(), epochs);
    let loss = jepa.train_on_sequences(&seqs, epochs);
    eprintln!("Loss: {:.4}", loss);

    jepa.save(jepa_path).expect("Failed to save JEPA");
    eprintln!("JEPA saved to {}", jepa_path);
}

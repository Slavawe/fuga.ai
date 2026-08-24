// txt_search_debug.rs — диагностика лексического канала на живом кубе.
use fuga::{MemoryStore, WaveCube};

fn main() {
    let cube_path = std::env::args().nth(1).unwrap_or_else(|| "omni_cube_idf.bin".into());
    let cube = WaveCube::<4, 8>::load_bin(&cube_path).expect("cube");
    let dim = cube.dim;
    let _ = dim;
    let mut mem = MemoryStore::load_bin(&cube_path.replace(".bin", "_mem.bin")).expect("mem");
    println!("entries: {}", mem.size());
    mem.build_text_index();
    let queries = ["hello how are you", "write a function in rust that sorts an array",
                   "explain backpropagation neural network"];
    for q in queries {
        let hits = mem.search_by_text(q, 5);
        println!("\nQ: {}", q);
        for (i, s, e) in &hits {
            let fname = std::path::Path::new(&e.source_doc)
                .file_name().and_then(|x| x.to_str()).unwrap_or("");
            println!("  lex={:.2} [{}] ({} байт)", s, fname, e.text.len());
        }
    }
}
use std::collections::HashMap;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mem_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "fuga_code_cube_mem.bin".into());
    let out_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "fuga_code_crystal.bin".into());

    let f = std::fs::File::open(&mem_path).unwrap_or_else(|e| panic!("open {}: {}", mem_path, e));
    let mmap = unsafe { memmap2::Mmap::map(&f).expect("mmap") };
    let data = &mmap[..];
    let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    println!("scanning {} entries from {}", count, mem_path);

    let mut pos = 4usize;
    let mut seen: HashMap<u64, ()> = HashMap::new();
    let mut entries: Vec<fuga::CrystalEntry> = Vec::new();
    let mut dups = 0usize;
    let mut max_text = 0usize;

    for _ in 0..count {
        let dim = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let wc = (dim + 63) / 64;
        let mut words = vec![0u64; wc];
        for i in 0..wc {
            words[i] = u64::from_le_bytes(data[pos + i * 8..pos + (i + 1) * 8].try_into().unwrap());
        }
        pos += wc * 8;
        let hv = fuga::core::hypervector::Hypervector::from_raw(dim, words);

        let tl = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let text = String::from_utf8(data[pos..pos + tl].to_vec()).unwrap_or_default();
        pos += tl;
        if tl > max_text {
            max_text = tl;
        }

        let dl = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let doc = String::from_utf8(data[pos..pos + dl].to_vec()).unwrap_or_default();
        pos += dl;

        let rl = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let role = String::from_utf8(data[pos..pos + rl].to_vec()).unwrap_or_default();
        pos += rl;

        let key = fuga::fnv1a(text.as_bytes());
        if seen.insert(key, ()).is_some() {
            dups += 1;
            continue;
        }

        let entry = fuga::CrystalEntry {
            hv,
            key,
            key_text: text.chars().take(80).collect(),
            resonance: 1.0,
            route: ((key >> 8) & 0xFF) as u16,
            kind: fuga::KIND_L1,
            text: format!("{}\n---\nsource: {}\nrole: {}", text, doc, role),
        };
        entries.push(entry);
        if entries.len() % 1000 == 0 {
            println!(
                "  {} unique ({} dups, {:.1}% of file)",
                entries.len(),
                dups,
                100.0 * pos as f64 / data.len() as f64
            );
        }
    }

    println!(
        "done: {} unique entries, {} duplicates skipped",
        entries.len(),
        dups
    );
    println!("max text len: {} B", max_text);

    let mut l0_index: HashMap<u64, usize> = HashMap::with_capacity(entries.len());
    for (i, e) in entries.iter().enumerate() {
        l0_index.insert(e.key, i);
    }

    let crystal = fuga::PhaseCrystal {
        dim: 8192,
        entries,
        l0_index,
        threshold: 0.35,
    };
    crystal.save(&out_path).unwrap();
    println!("saved {} entries to {}", crystal.entries.len(), out_path);

    let v = fuga::PhaseCrystal::load(&out_path).unwrap();
    let mut bad = 0;
    for (i, e) in v.entries.iter().enumerate() {
        if v.l0_index.get(&e.key).map(|&j| j) != Some(i) {
            bad += 1;
        }
    }
    println!(
        "verify: {} entries, {} l0_keys, {} mismatches",
        v.entries.len(),
        v.l0_index.len(),
        bad
    );
}

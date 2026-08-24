const GGUF_MAGIC: u32 = 0x46475547;
const GGUF_VERSION: u32 = 3;
const GGUF_ALIGN: usize = 32;

fn pad_to(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

pub fn read_gguf_version(path: &str) -> Option<u64> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 24 {
        return None;
    }
    if u32::from_le_bytes(data[0..4].try_into().ok()?) != GGUF_MAGIC {
        return None;
    }
    let kv_count = u64::from_le_bytes(data[16..24].try_into().ok()?);
    let mut off = 24usize;
    for _ in 0..kv_count {
        if off + 8 > data.len() {
            return None;
        }
        let klen = u64::from_le_bytes(data[off..off + 8].try_into().ok()?) as usize;
        off += 8;
        if off + klen > data.len() {
            return None;
        }
        let key = String::from_utf8_lossy(&data[off..off + klen]);
        off += klen;
        if off + 4 > data.len() {
            return None;
        }
        let vtype = u32::from_le_bytes(data[off..off + 4].try_into().ok()?);
        off += 4;
        if key == "fuga.generation" && vtype == 10 {
            return Some(u64::from_le_bytes(data[off..off + 8].try_into().ok()?));
        }
        // skip value
        match vtype {
            8 => {
                if off + 8 <= data.len() {
                    let slen = u64::from_le_bytes(data[off..off + 8].try_into().ok()?) as usize;
                    off += 8 + slen;
                }
            }
            9 => {
                if off + 12 <= data.len() {
                    let elem_type = u32::from_le_bytes(data[off..off + 4].try_into().ok()?);
                    off += 4;
                    let count = u64::from_le_bytes(data[off..off + 8].try_into().ok()?) as usize;
                    off += 8 + if elem_type == 0 { count } else { 0 };
                } else {
                    off += 12;
                }
            }
            10 | 11 => off += 8,
            12 => off += 8,
            _ => off += 4,
        }
    }
    None
}

pub fn export_gguf(path: &str) -> Result<(), String> {
    let files: Vec<(&str, &str)> = vec![
        ("jepa", "fuga_mirror_jepa.bin"),
        ("tm", "fuga_mirror_tm.bin"),
        ("nodes", "fuga_mirror_nodes.bin"),
        ("buffer", "fuga_buffer.bin"),
    ];

    let old_gen = read_gguf_version(path).unwrap_or(0);
    let new_gen = old_gen + 1;

    let mut kv: Vec<(String, Vec<u8>, u32)> = Vec::new();

    // version string = "1.0.0"
    let ver_str = format!("1.0.0");
    kv.push((
        "fuga.version".into(),
        {
            let mut v = Vec::new();
            v.extend_from_slice(&(ver_str.len() as u64).to_le_bytes());
            v.extend_from_slice(ver_str.as_bytes());
            v
        },
        8,
    ));
    // generation counter
    kv.push(("fuga.generation".into(), new_gen.to_le_bytes().to_vec(), 10));
    // timestamp
    let ts = format!(
        "{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );
    kv.push((
        "fuga.timestamp".into(),
        {
            let mut v = Vec::new();
            v.extend_from_slice(&(ts.len() as u64).to_le_bytes());
            v.extend_from_slice(ts.as_bytes());
            v
        },
        8,
    ));
    kv.push(("fuga.dim".into(), 8192u64.to_le_bytes().to_vec(), 10));

    // read and store eval result if available
    if let Ok(e) = std::fs::read_to_string("goals.md") {
        for line in e.lines() {
            if line.starts_with("eval:") {
                let eval_str = line[5..].trim();
                kv.push((
                    "fuga.eval".into(),
                    {
                        let mut v = Vec::new();
                        v.extend_from_slice(&(eval_str.len() as u64).to_le_bytes());
                        v.extend_from_slice(eval_str.as_bytes());
                        v
                    },
                    8,
                ));
                break;
            }
        }
    }

    for (key, filepath) in &files {
        let data = match std::fs::read(filepath) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("  warn: {} not found ({}), skipping", filepath, e);
                continue;
            }
        };
        let len = data.len() as u64;
        let mut blob = Vec::new();
        blob.extend_from_slice(&0u32.to_le_bytes()); // array of uint8
        blob.extend_from_slice(&len.to_le_bytes());
        blob.extend_from_slice(&data);
        kv.push((format!("fuga.{}", key), blob, 9));
    }

    let mut out = Vec::new();
    out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    out.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    out.extend_from_slice(&(kv.len() as u64).to_le_bytes());

    for (key, val_bytes, vtype) in &kv {
        let key_bytes = key.as_bytes();
        out.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(key_bytes);
        out.extend_from_slice(&vtype.to_le_bytes());
        out.extend_from_slice(val_bytes);
    }

    let data_off = pad_to(out.len(), GGUF_ALIGN);
    out.resize(data_off, 0);

    std::fs::write(path, &out).map_err(|e| format!("write {}: {}", path, e))?;
    // also keep the main fuga.gguf if this is a different path
    if path != "fuga.gguf" {
        let _ = std::fs::write("fuga.gguf", &out);
    }
    Ok(())
}

pub fn snapshot(path: &str, tag: &str) -> Result<(), String> {
    let snap = format!("fuga_{}.gguf", tag);
    let data = std::fs::read(path).map_err(|e| format!("read {}: {}", path, e))?;
    std::fs::write(&snap, &data).map_err(|e| format!("write {}: {}", snap, e))?;
    Ok(())
}

pub fn inspect_gguf(path: &str) -> Result<(), String> {
    let data = std::fs::read(path).map_err(|e| format!("read {}: {}", path, e))?;
    if data.len() < 20 {
        return Err("too short".into());
    }
    let magic = u32::from_le_bytes(data[0..4].try_into().unwrap());
    if magic != GGUF_MAGIC {
        return Err("bad magic".into());
    }
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let _tensor_count = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let kv_count = u64::from_le_bytes(data[16..24].try_into().unwrap());
    println!("  GGUF version={} kv_count={}", version, kv_count);
    let mut off = 24usize;
    for _ in 0..kv_count {
        if off + 8 > data.len() {
            break;
        }
        let klen = u64::from_le_bytes(data[off..off + 8].try_into().unwrap()) as usize;
        off += 8;
        if off + klen > data.len() {
            break;
        }
        let key = String::from_utf8_lossy(&data[off..off + klen]).to_string();
        off += klen;
        if off + 4 > data.len() {
            break;
        }
        let vtype = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
        off += 4;
        match vtype {
            8 => {
                if off + 8 > data.len() {
                    break;
                }
                let slen = u64::from_le_bytes(data[off..off + 8].try_into().unwrap()) as usize;
                off += 8;
                let s = if off + slen <= data.len() {
                    String::from_utf8_lossy(&data[off..off + slen]).to_string()
                } else {
                    "(truncated)".into()
                };
                println!("    {} (string) = {}", key, s);
                off += slen;
            }
            9 => {
                if off + 4 > data.len() {
                    break;
                }
                let elem_type = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
                off += 4;
                if off + 8 > data.len() {
                    break;
                }
                let count = u64::from_le_bytes(data[off..off + 8].try_into().unwrap()) as usize;
                off += 8;
                let total_bytes = if elem_type == 0 { count } else { 0 };
                println!(
                    "    {} (array[{}] of type {}  ~{} bytes)",
                    key, count, elem_type, total_bytes
                );
                off += total_bytes;
            }
            10 => {
                if off + 8 <= data.len() {
                    let v = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
                    println!("    {} (uint64) = {}", key, v);
                    off += 8;
                }
            }
            _ => {
                println!("    {} (type {})", key, vtype);
            }
        }
    }
    Ok(())
}

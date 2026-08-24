use std::env;
use std::process::Command;
use std::thread;
use std::time::Duration;

use fuga::{Hypervector, MemoryStore, ai::wave_mesh::cosine_similarity};

fn main() {
    let args: Vec<String> = env::args().collect();
    let interface = args.get(1).map(|s| s.as_str()).unwrap_or("wlan0");
    let mem_path = env::var("FUGA_MEM_PATH").unwrap_or_else(|_| "fuga_knowledge_mem.bin".into());

    println!("[Fuga-Wave Recv] Listening on {}...", interface);

    let memory = match MemoryStore::load_bin(&mem_path) {
        Ok(m) => {
            println!("Memory: {} entries", m.size());
            m
        }
        Err(e) => {
            eprintln!("Memory load: {}", e);
            return;
        }
    };

    let _ = Command::new("ip")
        .args(["link", "set", interface, "down"])
        .status();
    let _ = Command::new("iw")
        .args([interface, "set", "type", "monitor"])
        .status();
    let _ = Command::new("ip")
        .args(["link", "set", interface, "up"])
        .status();
    println!("[Fuga-Wave Recv] {} is in monitor mode.", interface);

    let mut seen_frames: u64 = 0;
    loop {
        let output = Command::new("tcpdump")
            .args([
                "-i", interface, "-c", "1", "-XX", "-n", "-e", "-t", "-s", "0", "-l",
            ])
            .output();

        match output {
            Ok(out) => {
                let text = String::from_utf8_lossy(&out.stdout);
                let mut hex_bytes = Vec::new();
                for line in text.lines() {
                    if let Some(hex_part) = line.split('|').next() {
                        for word in hex_part.split_whitespace().skip(1) {
                            if word.len() == 2 {
                                if let Ok(b) = u8::from_str_radix(word, 16) {
                                    hex_bytes.push(b);
                                }
                            }
                        }
                    }
                }
                if hex_bytes.len() > 40 {
                    seen_frames += 1;
                    let radiotap_len = u16::from_le_bytes([hex_bytes[2], hex_bytes[3]]) as usize;
                    let data_start = radiotap_len + 24;
                    if data_start < hex_bytes.len() {
                        let payload = &hex_bytes[data_start..];
                        let dim = payload.len() * 8;
                        let mut bits = Vec::with_capacity(dim);
                        for &b in payload {
                            for i in 0..8 {
                                bits.push(if (b >> i) & 1 == 1 { 1i8 } else { -1i8 });
                            }
                        }
                        let received = Hypervector::from_i8_bits(dim, &bits);
                        let entries = memory.all_entries();
                        let n = entries.len().min(5000);
                        let mut best_sim = 0.0f64;
                        let mut best_text = "";
                        for e in entries[..n].iter() {
                            let sim = cosine_similarity(&received, &e.vector);
                            if sim > best_sim {
                                best_sim = sim;
                                best_text = &e.text;
                            }
                        }
                        println!(
                            "[Fuga-Wave Recv] Frame #{} | payload={}B | best_sim={:.4} | \"{:.80}\"",
                            seen_frames,
                            payload.len(),
                            best_sim,
                            best_text
                        );
                    }
                }
            }
            Err(e) => {
                eprintln!("[Fuga-Wave Recv] tcpdump error: {}", e);
                thread::sleep(Duration::from_secs(5));
            }
        }
    }
}

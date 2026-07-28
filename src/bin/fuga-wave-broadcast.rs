use std::env;
use std::process::Command;
use std::fs;
use std::thread;
use std::time::Duration;

use fuga::{
    FugaAI, WaveCube, MemoryStore, Hypervector,
    ai::wave_mesh::{self, hypervector_to_byte_payload, build_radiotap_frame},
    core::wave_cube::peek_cube_header,
};

fn main() {
    let args: Vec<String> = env::args().collect();
    let interface = args.iter().skip(1)
        .find(|a| !a.starts_with("--") && !a.contains('/') && !a.contains('\\'))
        .map(|s| s.as_str())
        .unwrap_or("wlan0");
    let write_path = if args.contains(&"--write".to_string()) {
        args.iter().position(|a| a == "--write").and_then(|i| args.get(i + 1))
    } else { None };

    let cube_path = env::var("FUGA_CUBE_PATH").unwrap_or_else(|_| "fuga_knowledge_cube.bin".into());
    let mem_path = env::var("FUGA_MEM_PATH").unwrap_or_else(|_| "fuga_knowledge_mem.bin".into());

    let (ndim, side_len, _) = peek_cube_header(&cube_path)
        .expect("Failed to read cube header");
    println!("Cube: {}×{}, loaded", side_len, ndim);

    let result = match (ndim, side_len) {
        (3, 4) => broadcast::<3, 4>(interface, &cube_path, &mem_path, write_path),
        (4, 4) => broadcast::<4, 4>(interface, &cube_path, &mem_path, write_path),
        (3, 5) => broadcast::<3, 5>(interface, &cube_path, &mem_path, write_path),
        (3, 6) => broadcast::<3, 6>(interface, &cube_path, &mem_path, write_path),
        (3, 7) => broadcast::<3, 7>(interface, &cube_path, &mem_path, write_path),
        (3, 8) => broadcast::<3, 8>(interface, &cube_path, &mem_path, write_path),
        (4, 8) => broadcast::<4, 8>(interface, &cube_path, &mem_path, write_path),
        (5, 2) => broadcast::<5, 2>(interface, &cube_path, &mem_path, write_path),
        (5, 4) => broadcast::<5, 4>(interface, &cube_path, &mem_path, write_path),
        _ => panic!("Unsupported cube: {}×{}", side_len, ndim),
    };

    if let Err(e) = result {
        eprintln!("Fuga-Wave error: {}", e);
    }
}

fn setup_monitor(interface: &str) -> bool {
    if !std::path::Path::new(&format!("/sys/class/net/{}", interface)).exists() {
        return false;
    }
    let down = Command::new("ip")
        .args(["link", "set", interface, "down"])
        .output();
    if down.map_or(true, |o| !o.status.success()) { return false; }
    let set_mon = Command::new("iw")
        .args([interface, "set", "type", "monitor"])
        .output();
    if set_mon.map_or(true, |o| !o.status.success()) { return false; }
    let up = Command::new("ip")
        .args(["link", "set", interface, "up"])
        .output();
    up.map_or(false, |o| o.status.success())
}

fn broadcast<const N: usize, const S: usize>(
    interface: &str, cube_path: &str, mem_path: &str,
    write_path: Option<&String>,
) -> Result<(), String> {
    let cube = WaveCube::<N, S>::load_bin(cube_path)
        .map_err(|e| format!("Cube: {}", e))?;
    let memory = MemoryStore::load_bin(mem_path)
        .map_err(|e| format!("Memory: {}", e))?;

    let mut ai = FugaAI::<N, S>::new(cube.dim, 3);
    ai.cube = cube;
    ai.memory = memory;

    let has_radio = setup_monitor(interface);
    if has_radio {
        println!("[Fuga-Wave] {} is in monitor mode. Broadcasting VSA phase state...", interface);
    } else {
        println!("[Fuga-Wave] Radio unavailable (need root?). Running in simulation mode.");
        println!("[Fuga-Wave] Use --write <file> to capture frames to disk.");
    }

    let cube_entropy = ai.cube.global_entropy();
    let mem_size = ai.memory.size();
    println!("[Fuga-Wave] Cube entropy={:.4}, Memory={} entries", cube_entropy, mem_size);

    let mut tick: u64 = 0;
    let mut capture_file = write_path.and_then(|p| fs::File::create(p).ok());

    loop {
        let cx = (tick as usize) % 4;
        let cy = (tick as usize / 4) % 4;
        let cz = (tick as usize / 16) % 4;

        let state_vec: Hypervector = ai.cube.cell(cx, cy, cz);
        let phasors = wave_mesh::hypervector_to_phasors(&state_vec);
        let spread = wave_mesh::spectral_spread(&state_vec.to_i8_bits());

        let payload = hypervector_to_byte_payload(&state_vec);
        let frame = build_radiotap_frame(&payload);

        let top_match = ai.memory.search(&state_vec, 1)
            .first().map(|(_, s, e)| (*s, e.text.as_str())).unwrap_or((0.0, ""));

        println!("[Fuga-Wave] Tick {} | cell=({},{},{}) | phasor_symbols={} | spread={:.4} | frame={}B | top_sim={:.4} | \"{:.60}\"",
            tick, cx, cy, cz, phasors.len(), spread, frame.len(), top_match.0, top_match.1);

        if let Some(ref mut file) = capture_file {
            use std::io::Write;
            let header = format!("TICK={} CELL=({},{},{}) SIM={:.4} LEN={}\n",
                tick, cx, cy, cz, top_match.0, frame.len());
            file.write_all(header.as_bytes()).ok();
            file.write_all(&frame).ok();
            file.write_all(b"\n---END---\n").ok();
            file.flush().ok();
        }

        thread::sleep(Duration::from_millis(1000));
        tick += 1;
    }
}

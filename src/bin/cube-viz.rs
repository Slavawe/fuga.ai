// fuga cube-viz — exports 4³ cube cell data as JSON for 3D rendering
// Usage: fuga-cube-viz [cube_path] > cube_data.json

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "fuga_code_cube.bin".to_string());
    let cube = match fuga::WaveCube::<3, 4>::load_bin(&path) {
        Ok(c) => c,
        Err(e) => { eprintln!("Cube load error: {}", e); std::process::exit(1); }
    };

    let mut cells = Vec::with_capacity(64);
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                let hv = cube.cell(x, y, z);
                let ones: u64 = hv.words.iter().map(|w| w.count_ones() as u64).sum();
                let total_bits = hv.dim as u64;
                let density = ones as f64 / total_bits as f64;
                let entropy = if density == 0.0 || density == 1.0 {
                    0.0
                } else {
                    -density * density.log2() - (1.0 - density) * (1.0 - density).log2()
                };
                cells.push(serde_json::json!({
                    "x": x, "y": y, "z": z,
                    "density": (density * 1000.0).round() / 1000.0,
                    "entropy": (entropy * 1000.0).round() / 1000.0,
                    "ones": ones,
                    "dim": hv.dim,
                }));
            }
        }
    }

    let out = serde_json::json!({
        "n": 3, "s": 4, "dim": cube.dim, "cells": cells
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}

// gpu_bench.rs — подключение видеокарты: честное A/B CPU vs GPU для
// Widrow-Hoff дельты байтового W-оператора (главная операция обучения).
//
// Замеряет шагов/с одного и того же `W += lr · err ⊗ x` (512²):
//   - CPU : нативный Rust цикл, как в learn_transition
//   - GPU : wgpu/Vulkan compute-шейдер, W держится в GPU-буфере
// Печатает factor ускорения. Если Vulkan недоступен — честно сообщает,
// что GPU не поднялся и стенд возвращает CPU-показатель.
//
// Usage: gpu_bench [--steps 200000]
use std::time::Instant;

use fuga::ai::gpu_ops::{GpuOps, DIM};
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let steps: usize = args
        .iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000);

    let n = (DIM * DIM) as usize;
    let mut w = vec![0.0f32; n];           // W 512²
    let mut w_gpu = vec![0.0f32; n];       // CPU-копия для GPU-стенки (download compare)
    let x: Vec<f32> = (0..DIM).map(|i| ((i as f32) / DIM as f32) - 0.5).collect();
    let err: Vec<f32> = (0..DIM).map(|i| ((i * 7 % DIM) as f32 / DIM as f32) - 0.5).collect();
    let lr = 0.05f32;
    w_gpu.copy_from_slice(&w);

    // ---- CPU baseline ----
    let t0 = std::time::Instant::now();
    let mut cpu_ops: u64 = 0;
    for _ in 0..steps {
        for o in 0..DIM as usize {
            let e = err[o];
            let row = &mut w[o * DIM as usize..(o + 1) * DIM as usize];
            for (j, rv) in row.iter_mut().enumerate() {
                *rv += lr * e * x[j];
            }
        }
        cpu_ops += 1;
    }
    let cpu_secs = t0.elapsed().as_secs_f64();

    // ---- GPU ----
    eprintln!("[DEBUG] Attempting GPU init...");
    let gpu = fuga::ai::gpu_ops::try_new();
    eprintln!("[DEBUG] try_new() returned: {}", if gpu.is_some() { "Some" } else { "None" });
    let (gpu_secs, ok) = match &gpu {
        Some(g) => {
            g.upload_w(&w_gpu);
            let tt = std::time::Instant::now();
            for _ in 0..steps {
                g.delta(&x, &err, lr);
            }
            let s = tt.elapsed().as_secs_f64();
            // verify GPU W matches CPU-updated W within float tolerance
            g.download_w(&mut w_gpu);
            (s, true)
        }
        None => (0.0, false),
    };

    // ---- report ----
    println!("=== GPU BENCH (Widrow-Hoff delta, DIM=512) ===");
    println!("steps: {}", steps);
    println!("CPU : {:.4}s  ({:.0} step/s)", cpu_secs, steps as f64 / cpu_secs);
    if ok {
        let g = gpu_secs;
        println!("GPU : {:.4}s  ({:.0} step/s)", g, steps as f64 / g);
        println!("speedup: {:.2}×", (cpu_secs / g).clamp(0.0, 1e9));
        // consistency: GPU W must match CPU W (same updates applied)
        let maxdiff = w
            .iter()
            .zip(w_gpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("max |W_GPU - W_CPU| = {:.4e} ({} exact)", maxdiff, if maxdiff < 1e-2 { "≈OK" } else { "MISMATCH" });
    } else {
        println!("GPU:  Vulkan adapter NOT available — CPU fallback only");
        println!("speedup: 1.00× (GPU disabled)");
        println!("РЕЗУЛЬТАТ: GPU не подключён (нет Vulkan-адаптера на этой машине)");
    }
}
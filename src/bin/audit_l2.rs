use fuga::ai::{HierarchicalJEPA, TemporalMemory};
use fuga::TemporalPredictor;
use std::time::Instant;

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    let mut tp = TemporalPredictor::new(
        TemporalMemory::new(30_000, 4),
        HierarchicalJEPA::new(8192),
    );

    let start = Instant::now();
    let mut fed = 0usize;
    let mut l2_resets = 0usize;
    let mut ema = [0.0f64; 3];
    // гистограмма: 20 корзин от 0.4 до 1.6
    let bins: usize = 24;
    let lo: f64 = 0.4;
    let hi: f64 = 1.6;
    let mut hist: Vec<[u64; 3]> = vec![[0u64; 3]; bins];
    let mut ema_dbg = [[0.0f64; 3]; 60];

    for p in &paths {
        let Ok(content) = std::fs::read_to_string(p) else {
            eprintln!("cannot read {}", p);
            continue;
        };
        for line in content.lines() {
            let l = line.trim();
            if l.len() < 8 {
                continue;
            }
            let (_tm, errors) = tp.feed_learn(l);
            fed += 1;
            if errors.len() == 3 {
                if errors[2] == 1.0 {
                    l2_resets += 1;
                }
                for (i, v) in errors.iter().enumerate() {
                    ema[i] = ema[i] * 0.99 + v * 0.01;
                    let b = (((v - lo) / (hi - lo)) * bins as f64) as isize;
                    if b >= 0 && (b as usize) < bins {
                        hist[b as usize][i] += 1;
                    }
                }
                if fed % 500 == 0 {
                    eprintln!("[EMA] steps={} L0={:.4} L1={:.4} L2={:.4}", fed, ema[0], ema[1], ema[2]);
                }
            }
            let _ = &mut ema_dbg;
        }
        eprintln!("[{}] fed so far {} in {:.1}s", p, fed, start.elapsed().as_secs_f64());
    }

    eprintln!(
        "[SUMMARY] fed={} L2 resets (breaker critical)={}  L2_reset_rate={:.2}%",
        fed,
        l2_resets,
        l2_resets as f64 / fed as f64 * 100.0
    );
    eprintln!("[HISTOGRAM] fed={} (bins {:.2}-{:.2} step {:.4})", fed, lo, hi, (hi - lo) / bins as f64);
    for b in 0..bins {
        let v = lo + (hi - lo) * (b as f64 + 0.5) / bins as f64;
        eprintln!("  {:.2}: L0={:6} L1={:6} L2={:6}", v, hist[b][0], hist[b][1], hist[b][2]);
    }
}

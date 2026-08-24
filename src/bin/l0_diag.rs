// L0/L1 convergence diagnostic stand v3 (temporary, ad-hoc).
// Honest measurement:
//  - L0: ctx = last 4 buffer HVs (L0_CTX), target = next buffer HV.
//        sim = cosine(thresholded pred, actual)  [pred = baseline⊗raw]
//  - L1: ctx = 3 buffer HVs shifted by L0_CTX (same as feed_learn l1_ctx),
//        target = l1_pred computed the same way feed_learn does.
//        sim = cosine(thresholded pred from shifted window, l1_pred target).
// This tells us whether L1 actually learns to reproduce ITS OWN target
// (training convergence) — not whether it predicts buffer[-1].
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
    let mut loss_sum = [0.0f64; 3];
    let mut sim_l0 = 0.0f64;
    let mut n_l0 = 0usize;
    let mut sim_l1 = 0.0f64;
    let mut n_l1 = 0usize;

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
            for (i, v) in errors.iter().enumerate() {
                loss_sum[i] += v;
            }
            let buf = &tp.buffer;
            // L0: ctx = last L0_CTX(4) buffer HVs (positions before last), target = last.
            if buf.len() >= 5 {
                let ctx0: Vec<&fuga::Hypervector> =
                    buf[buf.len() - 4..buf.len() - 1].iter().collect();
                let actual = &buf[buf.len() - 1];
                let l0 = &tp.hjepa.levels[0];
                sim_l0 += l0.similarity_to_expected(&ctx0, actual);
                n_l0 += 1;
            }
            // L1: L1 is trained (learn win = last 3 buffer HVs before last) to
            // reproduce l1_pred (computed by feed_learn from the window shifted
            // by L0_CTX). Honest check: sim(predict(win), l1_pred).
            if buf.len() >= 9 {
                let win: Vec<&fuga::Hypervector> =
                    buf[buf.len() - 4..buf.len() - 1].iter().collect();
                let l1_ctx_end = buf.len() - 1 - 4; // buf_minus_1 - L0_CTX
                let l1_ctx: Vec<&fuga::Hypervector> =
                    buf[l1_ctx_end - 3..l1_ctx_end].iter().collect();
                let l1_pred = tp.hjepa.levels[1].predict(&l1_ctx);
                let l1 = &tp.hjepa.levels[1];
                let sim = l1.similarity_to_expected(&win, &l1_pred);
                sim_l1 += sim;
                n_l1 += 1;
            }
            if fed % 500 == 0 {
                eprintln!(
                    "[diag] steps={} loss L0={:.4} L1={:.4} L2={:.4} | sim L0={:.4} L1(own target)={:.4}",
                    fed,
                    loss_sum[0] / fed as f64,
                    loss_sum[1] / fed as f64,
                    loss_sum[2] / fed as f64,
                    if n_l0 > 0 { sim_l0 / n_l0 as f64 } else { f64::NAN },
                    if n_l1 > 0 { sim_l1 / n_l1 as f64 } else { f64::NAN },
                );
            }
        }
        eprintln!("[{}] fed so far {} in {:.1}s", p, fed, start.elapsed().as_secs_f64());
    }

    eprintln!(
        "[SUMMARY] fed={} loss L0={:.4} L1={:.4} L2={:.4}",
        fed,
        loss_sum[0] / fed as f64,
        loss_sum[1] / fed as f64,
        loss_sum[2] / fed as f64
    );
    eprintln!(
        "[SIM] L0 pred~actual={:.4} (n={}) | L1 pred~own_target={:.4} (n={})",
        if n_l0 > 0 { sim_l0 / n_l0 as f64 } else { f64::NAN },
        n_l0,
        if n_l1 > 0 { sim_l1 / n_l1 as f64 } else { f64::NAN },
        n_l1,
    );
}

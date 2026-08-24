// l2_ab.rs — честное A/B Задачи 2 vs Задачи 5 (temporary, ad-hoc).
//
// Логическая петля из ревью: диагноз Задачи 5 («все уровни учатся») был сделан
// стендом, который видел только ПОСТ-фиксное состояние (corr-таргет уже
// применён во всех 4 путях). Здесь разрываем петлю: ОДИН код, ОДИН корпус,
// ДВА параллельных TemporalPredictor с разным L2-таргетом:
//   - old: actuals[2] = l0_pred       (старый, «self-reinforcement»)
//   - new: actuals[2] = corr(l1,act)  (новый, фикс Задачи 2)
// Оба получают те же строки в том же порядке. Сравниваем loss и честную
// метрику pred~actual по уровням (similarity_to_expected).
use fuga::ai::{HierarchicalJEPA, TemporalMemory};
use fuga::vsa::topology::{ls_bind, phase_smooth};
use fuga::{Hypervector, TemporalPredictor, encode_text, sdr_to_hypervector};
use std::time::Instant;

fn feed_with_l2_target(
    tp: &mut TemporalPredictor,
    text: &str,
    l2_mode: u8,
) -> (f64, Vec<f64>) {
    // Копия внутренностей feed_learn (temporal_predictor.rs:128-178) с
    // переключаемым actuals[2]. l2_mode: 0 = old (l0_pred), 1 = new (corr).
    let sdr = encode_text(text);
    let (tm_pred, tm_match) = tp.tm.feed(&sdr);
    let hv = sdr_to_hypervector(&tm_pred, tp.hjepa.dim);
    tp.buffer.push(hv);
    if tp.buffer.len() > tp.buf_capacity {
        tp.buffer.remove(0);
    }
    if tp.buffer.len() < tp.hjepa.levels[0].context_len + 1 {
        return (tm_match, vec![1.0; 3]);
    }
    let buf_minus_1 = tp.buffer.len() - 1;
    let ctx: Vec<&Hypervector> = tp.buffer[..buf_minus_1].iter().collect();
    let actual = &tp.buffer[buf_minus_1];

    let l0_ctx_end = buf_minus_1;
    let l0_start = l0_ctx_end.saturating_sub(tp.hjepa.levels[0].context_len);
    let l0_ctx: Vec<&Hypervector> = tp.buffer[l0_start..l0_ctx_end].iter().collect();
    let l0_pred = tp.hjepa.levels[0].predict(&l0_ctx);

    let l1_pred =
        if buf_minus_1 >= tp.hjepa.levels[0].context_len + tp.hjepa.levels[1].context_len {
            let l1_ctx_end = buf_minus_1 - tp.hjepa.levels[0].context_len;
            let l1_start = l1_ctx_end.saturating_sub(tp.hjepa.levels[1].context_len);
            let l1_ctx: Vec<&Hypervector> = tp.buffer[l1_start..l1_ctx_end].iter().collect();
            Some(tp.hjepa.levels[1].predict(&l1_ctx))
        } else {
            None
        };

    let mut actuals: Vec<Hypervector> = Vec::new();
    actuals.push(actual.clone());
    if let Some(ref l1) = l1_pred {
        actuals.push(l1.clone());
    } else {
        actuals.push(l0_pred.clone());
    }
    if l2_mode == 0 {
        // OLD target: L2 learns to predict L0's own prediction (self-reinforcement).
        actuals.push(l0_pred.clone());
    } else {
        // NEW target: L2 learns the L1-error correction (fix from Task 2).
        let l1_for_corr = l1_pred.clone().unwrap_or_else(|| l0_pred.clone());
        actuals.push(phase_smooth(&ls_bind(&l1_for_corr, actual, 32), 2));
    }

    let actual_refs: Vec<&Hypervector> = actuals.iter().collect();
    let errors = tp.hjepa.learn(&ctx, &actual_refs);
    (tm_match, errors)
}

fn run(paths: &[String], l2_mode: u8, label: &str) {
    let mut tp = TemporalPredictor::new(
        TemporalMemory::new(30_000, 4),
        HierarchicalJEPA::new(8192),
    );
    let start = Instant::now();
    let mut fed = 0usize;
    let mut loss_sum = [0.0f64; 3];
    let mut sim_l0 = 0.0f64;
    let mut n_l0 = 0usize;
    let mut resets = 0usize;

    for p in paths {
        let Ok(content) = std::fs::read_to_string(p) else {
            eprintln!("cannot read {}", p);
            continue;
        };
        for line in content.lines() {
            let l = line.trim();
            if l.len() < 8 {
                continue;
            }
            let (_tm, errors) = feed_with_l2_target(&mut tp, l, l2_mode);
            fed += 1;
            if errors.len() == 3 {
                if errors[2] == 1.0 {
                    resets += 1;
                }
                for (i, v) in errors.iter().enumerate() {
                    loss_sum[i] += v;
                }
            }
            // Honest L0 quality: thresholded pred vs actual (contrast per-step).
            let buf = &tp.buffer;
            if buf.len() >= 5 {
                let ctx0: Vec<&Hypervector> =
                    buf[buf.len() - 4..buf.len() - 1].iter().collect();
                let actual = &buf[buf.len() - 1];
                let l0 = &tp.hjepa.levels[0];
                sim_l0 += l0.similarity_to_expected(&ctx0, actual);
                n_l0 += 1;
            }
            if fed % 1000 == 0 {
                eprintln!(
                    "[{}][{}] steps={} loss L0={:.4} L1={:.4} L2={:.4} | simL0={:.4}",
                    label, l2_mode, fed,
                    loss_sum[0] / fed as f64,
                    loss_sum[1] / fed as f64,
                    loss_sum[2] / fed as f64,
                    if n_l0 > 0 { sim_l0 / n_l0 as f64 } else { f64::NAN },
                );
            }
        }
        eprintln!("[{}] fed so far {} in {:.1}s", p, fed, start.elapsed().as_secs_f64());
    }

    eprintln!(
        "[{}][SUMMARY] fed={} resets={} loss L0={:.4} L1={:.4} L2={:.4} | simL0(pred~actual)={:.4}",
        label,
        fed,
        resets,
        loss_sum[0] / fed as f64,
        loss_sum[1] / fed as f64,
        loss_sum[2] / fed as f64,
        if n_l0 > 0 { sim_l0 / n_l0 as f64 } else { f64::NAN },
    );
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: l2_ab <corpus.jsonl...>");
        return;
    }
    // Same corpus, same code path, two target regimes — interleaved in ONE run
    // so TM-randomization is the only difference, both see identical input.
    run(&paths, 0, "OLD");
    run(&paths, 1, "NEW");
    eprintln!("[DONE] both regimes finished on same corpus");
}

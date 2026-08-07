// checkpoint_check.rs — verify loading trained checkpoints of the NEW format
// (cells + window + W + OWM-P + global W_patch) and probe what survived.
//
// The byte-generation techs added across iterations (recurrent h(t), Hopfield,
// KAN, LSTM peer) are DECODERS/operators built on top of a loaded TemporalMemory;
// only W (local byte transitions) and W_patch (global two-speed) are persisted.
// This stand loads a checkpoint and reports which sections actually present.
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "fuga_mirror_tm.bin".into());

    let t0 = Instant::now();
    match fuga::ai::htm_temporal::TemporalMemory::load(&path) {
        None => {
            eprintln!("FAILED to load checkpoint: {}", path);
            std::process::exit(2);
        }
        Some(tm) => {
            let secs = t0.elapsed().as_secs_f64();
            let w_len = tm.predictor_w().len();
            let p_len = tm.predictor_p().len();
            let pw_len = tm.patch_predictor().w.len();
            let pp_len = tm.patch_predictor().p.len();
            let up = tm.predictor_updates();
            println!("LOADED {} in {:.2}s", path, secs);
            println!(
                "  cells={} window={} context_len={}",
                tm.cells.len(),
                tm.window.len(),
                tm.context_len
            );
            println!(
                "  local W len={} (LATENT_DIM²={}), OWM P len={}, updates={}",
                w_len,
                512 * 512,
                p_len,
                up
            );
            println!(
                "  global W_patch len={}, P_patch len={}, patch_updates={}",
                pw_len,
                pp_len,
                tm.patch_predictor().updates
            );
            let trained = w_len == 512 * 512
                && tm
                    .predictor_w()
                    .iter()
                    .any(|&v| (v - 1.0).abs() > 1e-4 && v != 0.0);
            let patch_trained = pw_len == 512 * 512
                && tm
                    .patch_predictor()
                    .w
                    .iter()
                    .any(|&v| (v - 1.0).abs() > 1e-4 && v != 0.0);
            println!("  W trained = {}", trained);
            println!("  W_patch trained = {}", patch_trained);
        }
    }
}
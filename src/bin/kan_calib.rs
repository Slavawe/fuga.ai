// kan_calib.rs — калибровка KAN-оператора (итерация 5, диагноз:
// lr=0.05/stride=4/cap=4 сжимали обучение → 1 байт за 431s).
// После калибровки (stride=1, мягкий cap=40) — sweep lr на РЕАЛЬНОМ корпусе:
// метрика = средний cosine KAN-предсказания к следующему байту + длина декода.
//
// Usage: kan_calib <corpus.jsonl> [lr] [max_steps]
use fuga::ai::htm_temporal::TemporalMemory;
use fuga::ai::kan::KanTransition;
use fuga::ai::sdr::{byte_basis, structure_sdr_from_sdrs};
use std::io::BufRead;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus = args.get(1).cloned().unwrap_or_else(|| "fisig_corpus.jsonl".into());
    let lr: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.2);
    let max_steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(300_000);

    let tm = TemporalMemory::new(64, 4);
    let enc = &tm.predictor().encoder;

    let mut kan = KanTransition::new();
    let mut count = 0usize;
    let t0 = std::time::Instant::now();

    // 1. Обучаем: окно 4 байта → следующий байт (как learn_bytes, но на KAN)
    let f = std::fs::File::open(&corpus).expect("corpus");
    let rd = std::io::BufReader::new(f);
    'outer: for line in rd.lines().flatten() {
        let data: Vec<u8> = line.into_bytes();
        if data.len() < 5 {
            continue;
        }
        for i in 0..data.len() - 1 {
            let lo = i.saturating_sub(4);
            let win = &data[lo..=i];
            let win_sdr: Vec<fuga::SdrVector> =
                win.iter().map(|&b| byte_basis(b)).collect();
            let x = enc.encode(&fuga::ai::sdr::structure_sdr_from_sdrs(&win_sdr));
            let t = enc.encode(&byte_basis(data[i + 1]));
            kan.learn(&x, &t, lr);
            kan.cap_outputs();
            count += 1;
            if count >= max_steps {
                break 'outer;
            }
        }
    }
    let el = t0.elapsed().as_secs_f64();
    println!(
        "trained KAN: {} steps in {:.1}s ({:.0} steps/s), lr={}",
        count,
        el,
        count as f64 / el,
        lr
    );

    // 2. Метрика: средний cosine KAN-pred → target на 2000 новых пар
    let f = std::fs::File::open(&corpus).unwrap();
    let rd = std::io::BufReader::new(f);
    let mut tot = 0.0f32;
    let mut n = 0usize;
    for line in rd.lines().flatten() {
        let data: Vec<u8> = line.into_bytes();
        if data.len() < 5 {
            continue;
        }
        for i in 0..data.len() - 4 {
            let lo = i.saturating_sub(4);
            let win = &data[lo..=i];
            let win_sdr: Vec<fuga::SdrVector> =
                win.iter().map(|&b| byte_basis(b)).collect();
            let x = enc.encode(&fuga::ai::sdr::structure_sdr_from_sdrs(&win_sdr));
            let pred = kan.apply(&x);
            let t = enc.encode(&byte_basis(data[i + 1]));
            tot += pred.cosine_similarity(&t);
            n += 1;
            if n >= 2000 {
                break;
            }
        }
        if n >= 2000 {
            break;
        }
    }
    println!("avg cosine KAN→target: {:.4} (n={})", tot / n as f32, n);
    println!("== DONE ==");
}

// (костыль для сборки: пути-алиасы, реальные функции подставляются при сборке)
use fuga::ai::sdr::byte_basis as byte_basis_import;
fn _alias() {
    let _ = byte_basis_import;
}
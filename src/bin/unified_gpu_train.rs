// unified_gpu_train.rs — ДВУХСКОРОСТНОЙ CPU/GPU конвейер всех технологий
// в ОДИН единый файл (FUGA1 + KAN_C).
//
// Разделение работы:
//   CPU : читает JSONL, строит окна, structure-fold + SdrEncoder
//         (x = encode(window), target = encode(next byte)) и копит батчи
//   GPU : три канала в VRAM:
//         W_local  — Widrow-Hoff дельта (batch_delta, tag=1 LOCAL_W)
//         W_patch  — глобальный two-speed патч (batch_delta2, tag=2 PATCH_W)
//         KAN      — сплайны на остатке (kan_batch_delta, tag=6 KAN_C)
//         Капы: cap_w / cap_w2 / kan_cap_w (мягкие, периодические)
//
// Честные допущения (задокументированы, как в gpu_train/kan_calib):
//   - init W≈0 → pred≈0, err ≈ target (тот же приём в первых GPU-стендах;
//     для KAN это означает целевой канал без вычитания W·x — CPU-гибрид
//     hybrid.rs делает W-первый шаг; GPU-версия декомпозирует: W-канал
//     берёт частотные пары, KAN учится на том же целевом направлении)
//   - OWM-consolidate считается на CPU в конце (16 направлений, дёшево)
//   - В конце всё скачивается и пишется ЕДИНЫМ save_unified_with_kan →
//     один файл со всеми секциями (LOCAL_W + PATCH_W + OWM_P + META + KAN_C).
//
// Usage:
//   unified_gpu_train --jsonl "a.jsonl,b.jsonl" --max-steps 2000000
//                     --out fuga_unified_gpu.fuga [--ckpt-every 500000]
//                     [--batch 256] [--lr-w 0.05] [--lr-patch 0.1]
//                     [--lr-kan 0.3] [--ctx 4] [--no-gpu]
use fuga::ai::gpu_ops::GpuOps;
use fuga::ai::htm_temporal::{save_unified_with_kan, UnifiedMeta};

/// v8 Syntax Loss: число ERROR-узлов в декоде по tree-sitter C.
/// 0 = валидное AST; >0 = синтаксис сломан. Compiler-in-the-Loop метрика.
fn syntax_error_count(code: &[u8]) -> usize {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_c::LANGUAGE.into())
        .ok();
    match parser.parse(&code[..], None) {
        Some(tree) => {
            let mut errs = 0;
            let mut cursor = tree.walk();
            loop {
                let node = cursor.node();
                if node.is_error() || node.is_missing() {
                    errs += 1;
                }
                if cursor.goto_first_child() {
                    continue;
                }
                while !cursor.goto_next_sibling() {
                    if !cursor.goto_parent() {
                        return errs;
                    }
                }
            }
        }
        None => code.len().max(1),
    }
}
use fuga::ai::sdr::{byte_basis, encode_bytes_sdr, structure_sdr_from_sdrs};
use std::io::BufRead;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

const DIM: usize = 512;

fn extract_bytes(line: &str) -> Vec<u8> {
    let v: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(_) => return line.as_bytes().to_vec(),
    };
    let mut out = String::new();
    if let Some(chs) = v.get("chapters").and_then(|c| c.as_array()) {
        for ch in chs {
            if let Some(paras) = ch.get("paragraphs").and_then(|p| p.as_array()) {
                for p in paras {
                    if let Some(s) = p.as_str() {
                        out.push_str(s);
                        out.push('\n');
                    }
                }
            }
        }
    } else if let Some(doc) = v.get("doc").and_then(|d| d.as_str()) {
        out.push_str(doc);
        if let Some(code) = v.get("code").and_then(|c| c.as_str()) {
            out.push('\n');
            out.push_str(code);
        }
    } else if let Some(code) = v.get("code").and_then(|c| c.as_str()) {
        out.push_str(code);
    } else {
        return line.as_bytes().to_vec();
    }
    out.into_bytes()
}

fn arg<T: std::str::FromStr>(args: &[String], name: &str, def: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<T>().ok())
        .unwrap_or(def)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus: String = arg(&args, "--jsonl", "corpus_doc_code_pairs.jsonl".into());
    let max_steps: usize = arg(&args, "--max-steps", 300_000);
    let batch: usize = arg(&args, "--batch", 256);
    let lr_w: f32 = arg(&args, "--lr-w", 0.05);
    let lr_patch: f32 = arg(&args, "--lr-patch", 0.1);
    let lr_macro: f32 = arg(&args, "--lr-macro", 0.05); // Byte-H-JEPA канал
    // v7 Patch Graph Curriculum: λ-старт 0.4 → экспоненциальное затухание к
    // floor 0.10 (~1.5M). τ в шагах: λ(t) = floor + (start−floor)·exp(−t/τ).
    let lambda_patch: f32 = arg(&args, "--lambda-patch", 0.4); // стартовый вес Patch Loss
    let lambda_floor: f32 = arg(&args, "--lambda-floor", 0.10);
    let lambda_tau: f32 = arg(&args, "--lambda-tau", 1_000_000.0);
    // v7 Curriculum: λ(t) = floor + (start−floor)·exp(−t/τ) — жёсткая сшивка
    // соседних API-узлов в начале, мягкое кондиционирование к 1.5M.
    let lambda_now = |applied: usize| {
        let t = applied as f32;
        let l = lambda_floor + (lambda_patch - lambda_floor) * (-t / lambda_tau).exp();
        l.max(lambda_floor).min(lambda_patch)
    };
    let lr_kan: f32 = arg(&args, "--lr-kan", 0.3);
    let out_path: String = arg(&args, "--out", "/tmp/unified_gpu.fuga".into());
    let use_gpu = !args.iter().any(|a| a == "--no-gpu");
    let ctxw: usize = arg(&args, "--ctx", 4);
    let patch_ctx: usize = arg(&args, "--patch-ctx", 8); // v8.1 горизонт 16 байт (компромисс)
    let ckpt_every: usize = arg(&args, "--ckpt-every", 500_000);
    let seed_text: Vec<u8> = args
        .iter()
        .position(|a| a == "--text-seed")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_bytes().to_vec())
        .unwrap_or_else(|| b"the force of gravity is".to_vec());
    let seed_code: Vec<u8> = args
        .iter()
        .position(|a| a == "--code-seed")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_bytes().to_vec())
        .unwrap_or_else(|| b"fn main() {".to_vec());

    // ВАЖНО: энкодер должен быть ТОЧНО тот же, что у декодера (TM predictor seed
    // 0xF03D_C0DE), иначе W обучается в чужом базисе → декод = мусор.
    let mut tm_seed_anchor = fuga::ai::htm_temporal::TemporalMemory::new(64, ctxw);
    let enc = tm_seed_anchor.predictor().encoder.clone();
    let enc_patch = tm_seed_anchor.patch_predictor().encoder.clone();
    let byte_cache: Vec<fuga::SdrVector> = (0..=255u8).map(byte_basis).collect();

    // ОГРАНИЧЕННЫЙ канал (backpressure, sync_channel — урок OOM).
    // Трёхканальная пара: (x_local, err_local, x_patch, err_patch).
    type Pair5 = (
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    ); // x, t, x2, t2, x2b, xm, tm (Byte-H-JEPA: макро-латент пара)
    let (tx, rx) = mpsc::sync_channel::<Pair5>(batch * 4);
    let stop = Arc::new(AtomicBool::new(false));
    // Кольцевой буфер последних (x, t) пар для метрики остатка на чекпоинтах.
    let probe_buf: Arc<std::sync::Mutex<std::collections::VecDeque<(Vec<f32>, Vec<f32>)>>> =
        Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new()));

    // --- Поток CPU: читать JSONL, готовить (x, err) пары, патч-канал ---
    let cpu_handle = {
        let stop = stop.clone();
        let ctxw = ctxw;
        let patch_ctx = patch_ctx;
        let lambda_patch = lambda_patch;
        let enc = enc.clone();
        let enc_patch = enc_patch.clone();
        let byte_cache = byte_cache.clone();
        let corpus = corpus.clone();
        let probe_buf = probe_buf.clone();
        std::thread::spawn(move || {
            let corpora: Vec<String> = corpus
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let mut steps = 0usize;
            'outer: for corpus_file in &corpora {
                let path = Path::new(corpus_file);
                if !path.exists() {
                    eprintln!("corpus missing: {}", corpus_file);
                    continue;
                }
                let f = std::fs::File::open(path).unwrap();
                let reader = std::io::BufReader::new(f);
                for line in reader.lines() {
                    let line = match line {
                        Ok(l) => l,
                        Err(_) => continue,
                    };
                    if line.trim().is_empty() {
                        continue;
                    }
                    let data = extract_bytes(&line);
                    if data.len() < 2 {
                        continue;
                    }
                    // AST-маска: интервалы семантичных C-узлов строки (один парсинг).
                    // ОГРАНИЧЕНИЕ: строки >50KB (весь файл/функция в одной строке
                    // JSONL) не парсим — дерево и копии текстов узлов велики и
                    // вели к OOM (4.4G за 29s на 7.5GB-машине). Такие строки
                    // дают фон-цель (след. окно), макро учится только на
                    // разумных по размеру узлах.
                    let ast_ranges = if data.len() <= 8_000 {
                        ast_node_ranges(&data)
                    } else {
                        Vec::new()
                    };
                    for i in 0..data.len().saturating_sub(1) {
                        if stop.load(Ordering::Relaxed) {
                            break 'outer;
                        }
                        let win_lo = i.saturating_sub(ctxw);
                        let win: &[u8] = &data[win_lo..=i];
                        let nxt = data[i + 1];
                        let win_sdrs: Vec<fuga::SdrVector> = win
                            .iter()
                            .map(|&c| byte_cache[c as usize].clone())
                            .collect();
                        let x = enc.encode(&structure_sdr_from_sdrs(&win_sdrs));
                        let t = enc.encode(&byte_cache[nxt as usize]);
                        // Byte-H-JEPA МАКРО: z_ctx = латент окна (тот же x),
                        // z_tgt = латент СЛЕДУЮЩЕГО окна (data[i+1..=i+1+ctxw]):
                        // семантический переход состояний в эмбеддингах.
                        let mut xm: Vec<f32> = x.values.clone();
                        let mut tm: Vec<f32> = Vec::new();
                        // Sparse AST-маска: макро-уровню не нужна микро-разметка
                        // на каждом байте. Берём AST-цель раз в 8 байт (граница
                        // патчевого блока), остальные шаги остаются на фон-цели.
                        // Это снимает 7/8 дорогих AST-целей с CPU hot path.
                        let mut ast_target: Option<&[u8]> = None;
                        if i % 1_000_000 == 0 { // BYPASS-режим: AST почти никогда
                            for (s, _, text) in ast_ranges.iter() {
                                if *s == i + 1 {
                                    ast_target = Some(text.as_slice());
                                    break;
                                }
                            }
                        }
                        if let Some(node_text) = ast_target {
                            // Цель — семантический вектор ВСЕГО будущего узла.
                            // encode_NO_CACHE: AST-узлы уникальны и огромны,
                            // кэш до этого доводил до OOM (4.4G за 29s).
                            let nsdrs: Vec<fuga::SdrVector> = node_text
                                .iter()
                                .map(|&c| byte_cache[c as usize].clone())
                                .collect();
                            let t_m = enc.encode_no_cache(&structure_sdr_from_sdrs(&nsdrs));
                            tm = t_m.values.clone();
                        } else if i + 1 + ctxw < data.len() {
                            // Фон: латент следующего окна (скользящий переход).
                            let nxt_win: &[u8] = &data[i + 1..=i + 1 + ctxw];
                            let nxt_sdrs: Vec<fuga::SdrVector> = nxt_win
                                .iter()
                                .map(|&c| byte_cache[c as usize].clone())
                                .collect();
                            let t_m = enc.encode(&structure_sdr_from_sdrs(&nxt_sdrs));
                            tm = t_m.values.clone();
                        }
                        let tv: Vec<f32> = t.values.clone(); // сырой таргет; остаток считает GPU
                        // v6: ПАТЧЕВОЕ ОКНО 1:1 С V2-ДЕКОДЕРОМ (последние 4 ПОЛНЫХ патча →
                        // следующий полный патч), 100% градиентный поток
                        // (БЕЗ условия i%2==1 — W_patch учится на КАЖДОМ шаге).
                        // Явный Patch Loss: ||P_{t+1} − W_patch·x||² с весом
                        // lambda_patch (масштаб сходимости vs байтовый W).
                        let mut x2 = Vec::new();
                        let mut t2 = Vec::new();
                        let mut x2b = Vec::new(); // перекрывающееся окно (сдвиг −1 патч)
                        let pp = (i + 1) / 2; // номер патча, в который входит data[i+1]
                        if i + 3 <= data.len() && pp >= patch_ctx {
                            // v8 ГОРИЗОНТ: последние patch_ctx полных патчей
                            // ДО целевого: pp-patch_ctx .. pp-1 (32 байта при 16)
                            let mut pw: Vec<&[u8]> = Vec::with_capacity(patch_ctx);
                            for k in 0..patch_ctx {
                                pw.push(&data[(pp - patch_ctx + k) * 2..(pp - patch_ctx + k + 1) * 2]);
                            }
                            let next_patch = &data[pp * 2..(pp + 1) * 2];
                            let win_patch_sdrs: Vec<fuga::SdrVector> =
                                pw.iter().map(|p| encode_bytes_sdr(p)).collect();
                            let xs = enc_patch.encode(&structure_sdr_from_sdrs(&win_patch_sdrs));
                            let ts = enc_patch.encode(&encode_bytes_sdr(next_patch));
                            x2 = xs.values.clone();
                            t2 = ts.values.clone(); // сырой таргет патча
                            // v8 УПЛОТНЕНИЕ: перекрывающееся окно со сдвигом −1 патч
                            // (pp-patch_ctx-1 .. pp-2 → pp) — альтернативные рёбра.
                            if pp >= patch_ctx + 1 {
                                let mut pv: Vec<&[u8]> = Vec::with_capacity(patch_ctx);
                                for k in 0..patch_ctx {
                                    pv.push(&data[(pp - patch_ctx - 1 + k) * 2..(pp - patch_ctx + k) * 2]);
                                }
                                let win2_sdrs: Vec<fuga::SdrVector> =
                                    pv.iter().map(|p| encode_bytes_sdr(p)).collect();
                                let xs2 = enc_patch.encode(&structure_sdr_from_sdrs(&win2_sdrs));
                                x2b = xs2.values.clone();
                            }
                        }
                        // Кольцевой буфер для метрики остатка (последние 512 пар).
                        let xv: Vec<f32> = x.values.clone();
                        let pv: Vec<f32> = t.values.clone();
                        if tx.send((x.values, tv, x2, t2, x2b, xm, tm)).is_err() {
                            break 'outer;
                        }
                        if let Ok(mut pb) = probe_buf.lock() {
                            pb.push_back((xv, pv));
                            while pb.len() > 512 {
                                pb.pop_front();
                            }
                        }
                        steps += 1;
                        if steps >= max_steps {
                            break 'outer;
                        }
                    }
                }
            }
            steps
        })
    };

    // --- Основной поток: GPU применяет ТРИ канала пачками ---
    let gpu = if use_gpu { fuga::ai::gpu_ops::try_new() } else { None };
    let mut w = vec![0.0f32; DIM * DIM];
    let mut w_patch: Vec<f32> = vec![0.0f32; DIM * DIM];
    let mut w_macro: Vec<f32> = vec![0.0f32; DIM * DIM]; // Byte-H-JEPA: латент-предиктор
    let mut kan_c: Vec<f32> = vec![0.0f32; DIM * DIM * 6];
    let mut applied: usize = 0;
    let mut next_ckpt: usize = ckpt_every;
    let t0 = Instant::now();
    let mut ident_p: Vec<f32> = vec![0.0f32; DIM * DIM];
    for di in 0..DIM {
        ident_p[di * DIM + di] = 1.0;
    }

    match &gpu {
        Some(g) => {
            g.upload_w(&w);
            g.upload_w2(&w);
            g.upload_kan(&kan_c);
            let mut xs: Vec<Vec<f32>> = Vec::with_capacity(batch);
            let mut ts: Vec<Vec<f32>> = Vec::with_capacity(batch);
            let mut xs2: Vec<Vec<f32>> = Vec::with_capacity(batch);
            let mut ts2: Vec<Vec<f32>> = Vec::with_capacity(batch);
            let mut xs2b: Vec<Vec<f32>> = Vec::with_capacity(batch); // перекрывающиеся окна
            let mut xms: Vec<Vec<f32>> = Vec::with_capacity(batch);
            let mut tms: Vec<Vec<f32>> = Vec::with_capacity(batch);
            let mut recv = rx;
            loop {
                match recv.recv() {
                    Ok((x, t, x2, t2, x2b, xm, tm)) => {
                        xs.push(x);
                        ts.push(t);
                        if !xm.is_empty() && !tm.is_empty() {
                            xms.push(xm);
                            tms.push(tm);
                        }
                        if !x2.is_empty() && !t2.is_empty() {
                            xs2.push(x2);
                            ts2.push(t2);
                            if !x2b.is_empty() {
                                xs2b.push(x2b.clone());
                            }
                        }
                        if xs.len() >= batch {
                            // Честный LMS: err = t − W·x на GPU, затем W+KAN-дельты.
                            for (xi, ti) in xs.iter().zip(ts.iter()) {
                                g.hybrid_step(xi, ti, lr_w, lr_kan);
                            }
                            for (xi, ti) in xs2.iter().zip(ts2.iter()) {
                                g.hybrid_step2(xi, ti, lr_patch * lambda_now(applied));
                            }
                            // v7 УПЛОТНЕНИЕ: перекрывающиеся окна с половинным λ
                            // (мягче, чтобы не перекачать W_patch на альт. рёбрах).
                            let lam2 = lr_patch * lambda_now(applied) * 0.5;
                            for xi in xs2b.iter() {
                                if !ts2.is_empty() {
                                    g.hybrid_step2(xi, &ts2[0], lam2);
                                }
                            }
                            applied += xs.len();
                            // Byte-H-JEPA MACRO: Widrow-Hoff в эмбеддингах
                            // err = z_tgt − W_macro·z_ctx; λ_macro масштабирует.
                            if !xms.is_empty() {
                                let lam_m = lr_macro * lambda_now(applied);
                                for (xm_i, tm_i) in xms.iter().zip(tms.iter()) {
                                    let mut pred = vec![0.0f32; DIM];
                                    for r in 0..DIM {
                                        let row = &w_macro[r * DIM..(r + 1) * DIM];
                                        pred[r] = row.iter().zip(xm_i.iter()).map(|(a, b)| a * b).sum();
                                    }
                                    for r in 0..DIM {
                                        let e = tm_i[r] - pred[r];
                                        let row = &mut w_macro[r * DIM..(r + 1) * DIM];
                                        for (w_, xv) in row.iter_mut().zip(xm_i.iter()) {
                                            *w_ += lam_m * e * xv;
                                        }
                                    }
                                }
                            }
                            xs.clear();
                            ts.clear();
                            xs2.clear();
                            ts2.clear();
                            xs2b.clear();
                            xms.clear();
                            tms.clear();
                            // Капы (как в learn_transition: CAP_EVERY≈50 → soft).
                            if applied / batch % 50 == 0 {
                                g.cap_w(4.0);
                                g.cap_w2(4.0);
                                g.kan_cap_w(40.0);
                            }
                            // Периодический ЕДИНЫЙ чекпоинт (переживает ребут).
                            if ckpt_every > 0 && applied >= next_ckpt {
                                next_ckpt += ckpt_every;
                                g.sync(); // batch dispatch: один poll после пачки
                                let mut cw = vec![0.0f32; DIM * DIM];
                                let mut cw2 = vec![0.0f32; DIM * DIM];
                                let mut ck = vec![0.0f32; DIM * DIM * 6];
                                g.download_w(&mut cw);
                                g.download_w2(&mut cw2);
                                g.download_kan(&mut ck);
                                let cm = UnifiedMeta {
                                    steps: applied as u64,
                                    patch_steps: applied as u64,
                                    ctx: ctxw as u32,
                                    version: 2,
                                };
                                let ckpt_path = format!("{}.ckpt.fuga", out_path);
                                save_unified_with_kan(
                                    &ckpt_path,
                                    &cw,
                                    &cw2,
                                    &ident_p,
                                    &cm,
                                    None,
                                    Some(&w_macro),
                                    Some(&ck)
                                )
                                .ok();
                                // Метрика LMS-остатка: mean ||t − W·x|| по свежим парам.
                                let residual = {
                                    let mut pr = Vec::new();
                                    if let Ok(pb) = probe_buf.lock() {
                                        for (xv, tv) in pb.iter().take(64) {
                                            let mut pred = [0.0f32; DIM];
                                            for o in 0..DIM {
                                                let row = o * DIM;
                                                let mut acc = 0.0f32;
                                                for i in 0..DIM {
                                                    acc += cw[row + i] * xv[i];
                                                }
                                                pred[o] = acc;
                                            }
                                            let mut sq = 0.0f32;
                                            let mut tsq = 0.0f32;
                                            for o in 0..DIM {
                                                let d = tv[o] - pred[o];
                                                sq += d * d;
                                                tsq += tv[o] * tv[o];
                                            }
                                            pr.push((sq.sqrt(), tsq.sqrt()));
                                        }
                                    }
                                    pr
                                };
                                if !residual.is_empty() {
                                    let mean_e: f32 =
                                        residual.iter().map(|(e, _)| e).sum::<f32>() / residual.len() as f32;
                                    let mean_t: f32 =
                                        residual.iter().map(|(_, t)| t).sum::<f32>() / residual.len() as f32;
                                    println!(
                                        "[ckpt] {} пар: ||t−Wx||_среднее={:.4} (||t||={:.4})",
                                        applied, mean_e, mean_t
                                    );
                                }
                                // Мини-декод naive 40B на чекпоинте: N-граммная связность.
                                let mut tm_p = fuga::ai::htm_temporal::TemporalMemory::new(64, ctxw);
                                tm_p.apply_byte_w(cw);
                                let dec_t = fuga::tm_generate_latent_bytes(
                                    &tm_p,
                                    b"the force of gravity is",
                                    40,
                                    ctxw,
                                    None,
                                );
                                let dec_c = fuga::tm_generate_latent_bytes(
                                    &tm_p,
                                    b"fn main() {",
                                    40,
                                    ctxw,
                                    None,
                                );
                                println!(
                                    "  [ckpt-decode] naive TEXT={}B {:?} | CODE={}B {:?}",
                                    dec_t.len(),
                                    String::from_utf8_lossy(&dec_t)
                                        .chars()
                                        .take(36)
                                        .collect::<String>(),
                                    dec_c.len(),
                                    String::from_utf8_lossy(&dec_c)
                                        .chars()
                                        .take(36)
                                        .collect::<String>()
                                );
                                eprintln!("  [ckpt] {} пар -> {}", applied, ckpt_path);
                                // v8 Syntax Loss: число ERROR-узлов C-AST в декоде
                                let errs = syntax_error_count(&dec_c);
                                eprintln!(
                                    "  [syntax] C-AST errors={} (0 = валидное дерево)",
                                    errs
                                );
                            }
                            if applied % (batch * 8) == 0 {
                                eprintln!("  [gpu] applied {} pairs", applied);
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
            if !xs.is_empty() {
                for (xi, ti) in xs.iter().zip(ts.iter()) {
                    g.hybrid_step(xi, ti, lr_w, lr_kan);
                }
                for (xi, ti) in xs2.iter().zip(ts2.iter()) {
                    g.hybrid_step2(xi, ti, lr_patch * lambda_now(applied));
                }
                applied += xs.len();
            }
            g.sync(); // все пачки применены — один poll перед readback
            g.download_w(&mut w);
            g.download_w2(&mut w_patch);
            g.download_kan(&mut kan_c);
        }
        None => {
            // CPU fallback: честный Widrow-Hoff (reĭенс) + KAN на остатке.
            let mut kan = fuga::ai::kan::KanTransition::new();
            let mut recv = rx;
            while let Ok((x, t, x2, t2, x2b, xm, tm)) = recv.recv() {
                // pred = W·x (линейный вклад)
                let mut pred = vec![0.0f32; DIM];
                for o in 0..DIM {
                    let row = o * DIM;
                    let mut acc = 0.0f32;
                    for i in 0..DIM {
                        acc += w[row + i] * x[i];
                    }
                    pred[o] = acc;
                }
                for o in 0..DIM {
                    let err = t[o] - pred[o];
                    let row = o * DIM;
                    for i in 0..DIM {
                        w[row + i] += lr_w * err * x[i];
                    }
                }
                // Byte-H-JEPA MACRO: Widrow-Hoff в эмбеддингах (CPU fallback)
                if !xm.is_empty() && !tm.is_empty() {
                    let lam_m = lr_macro * lambda_now(applied);
                    let mut pm = vec![0.0f32; DIM];
                    for o in 0..DIM {
                        let row = o * DIM;
                        let mut acc = 0.0f32;
                        for i in 0..DIM {
                            acc += w_macro[row + i] * xm[i];
                        }
                        pm[o] = acc;
                    }
                    for o in 0..DIM {
                        let err = tm[o] - pm[o];
                        let row = o * DIM;
                        for i in 0..DIM {
                            w_macro[row + i] += lam_m * err * xm[i];
                        }
                    }
                }
                // KAN на честном остатке t − W·x (как hybrid.learn_pair).
                let xv = fuga::ai::latent_jepa::LatentVector { values: x.clone() };
                let tv = fuga::ai::latent_jepa::LatentVector { values: t.clone() };
                let pv = fuga::ai::latent_jepa::LatentVector { values: pred };
                let mut res = fuga::ai::latent_jepa::LatentVector::zero();
                for o in 0..DIM {
                    res.values[o] = tv.values[o] - pv.values[o];
                }
                let rn = res
                    .values
                    .iter()
                    .map(|v| v * v)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-8);
                for v in &mut res.values {
                    *v /= rn;
                }
                kan.learn(&xv, &res, lr_kan);
                kan.cap_outputs();
                if !x2.is_empty() && !t2.is_empty() {
                    for o in 0..DIM {
                        let row = o * DIM;
                        let mut acc = 0.0f32;
                        for i in 0..DIM {
                            acc += w_patch[row + i] * x2[i];
                        }
                        let err = t2[o] - acc;
                        for i in 0..DIM {
                            w_patch[row + i] += lr_patch * lambda_now(applied) * err * x2[i];
                        }
                    }
                    // v7 УПЛОТНЕНИЕ: перекрывающееся окно (сдвиг −1) с половинным λ
                    if !x2b.is_empty() && !t2.is_empty() {
                        for o in 0..DIM {
                            let row = o * DIM;
                            let mut acc = 0.0f32;
                            for i in 0..DIM {
                                acc += w_patch[row + i] * x2b[i];
                            }
                            let err = t2[o] - acc;
                            for i in 0..DIM {
                                w_patch[row + i] +=
                                    lr_patch * lambda_now(applied) * 0.5 * err * x2b[i];
                            }
                        }
                    }
                }
                applied += 1;
            }
            kan_c = kan.c.clone();
        }
    }

    stop.store(true, Ordering::Relaxed);
    let cpu_steps = cpu_handle.join().unwrap_or(0);
    let el = t0.elapsed().as_secs_f64();

    // --- OWM-консолидация на CPU (16 направлений, дёшево) ---
    let mut tm = fuga::ai::htm_temporal::TemporalMemory::new(64, ctxw);
    tm.apply_byte_w(w.clone());
    let dirs: Vec<fuga::ai::latent_jepa::LatentVector> = [
        "fn", "main", "(", ")", "let", "mut", "use", "std", "impl", "if", "the", "and",
        "gravity", "force", "return", "struct",
    ]
    .iter()
    .map(|s| tm.predictor().encoder.encode(&fuga::ai::sdr::encode_text(s)))
    .collect();
    let k = tm.consolidate_owm(&dirs, 16, 0.01);
    let owm_p: Vec<f32> = tm.predictor().p.clone();
    println!("OWM: consolidated {} directions", k);

    // --- Сохраняем ЕДИНЫЙ FUGA1: LOCAL_W + PATCH_W + OWM_P + META + KAN_C ---
    let meta = UnifiedMeta {
        steps: applied as u64,
        patch_steps: applied as u64,
        ctx: ctxw as u32,
        version: 2,
    };
    {
        let n1: f64 = w.iter().map(|x| (x*x) as f64).sum::<f64>().sqrt();
        let n2: f64 = w_patch.iter().map(|x| (x*x) as f64).sum::<f64>().sqrt();
        let s1: f64 = w.iter().take(64).map(|x| *x as f64).sum();
        let s2: f64 = w_patch.iter().take(64).map(|x| *x as f64).sum();
        println!("[diag] нормы перед save: local_W={:.3} patch_W={:.3} sum64={:.4}/{:.4}", n1, n2, s1, s2);
    }
    save_unified_with_kan(&out_path, &w, &w_patch, &owm_p, &meta, None, Some(&w_macro), Some(&kan_c))
        .expect("save unified+kan");

    // ТОЧЕЧНЫЙ ТЕСТ СЕРИАЛИЗАЦИИ (по пользовательской методике):
    // бинарное равенство w_gpu_mem (в памяти) vs w_disk_mem (из файла).
    {
        let mut tm_chk = fuga::ai::htm_temporal::TemporalMemory::new(64, ctxw);
        assert!(tm_chk.load_unified_fuga1(&out_path), "reload fail");
        let disk = tm_chk.predictor_w();
        let mut mse = 0.0f64;
        let mut maxdiff = 0.0f64;
        let mut neq = 0usize;
        for i in 0..w.len() {
            let d = (w[i] - disk[i]) as f64;
            mse += d * d;
            if d.abs() > maxdiff {
                maxdiff = d.abs();
            }
            if w[i] != disk[i] {
                neq += 1;
            }
        }
        mse /= w.len() as f64;
        println!(
            "[SERIAL] local_W: MSE={:.3e} maxdiff={:.3e} несовпадающих={}/{} — {}",
            mse,
            maxdiff,
            neq,
            w.len(),
            if neq == 0 { "БИНАРНО ИДЕНТИЧЕН" } else { "РАСХОЖДЕНИЕ" }
        );
        let disk_p = tm_chk.patch_predictor().w.clone();
        let mut mse2 = 0.0f64;
        let mut neq2 = 0usize;
        for i in 0..w_patch.len() {
            let d = (w_patch[i] - disk_p[i]) as f64;
            mse2 += d * d;
            if w_patch[i] != disk_p[i] {
                neq2 += 1;
            }
        }
        mse2 /= w_patch.len() as f64;
        println!(
            "[SERIAL] patch_W: MSE={:.3e} несовпадающих={}/{} — {}",
            mse2,
            neq2,
            w_patch.len(),
            if neq2 == 0 { "БИНАРНО ИДЕНТИЧЕН" } else { "РАСХОЖДЕНИЕ" }
        );
    }
    println!(
        "saved {} (LOCAL_W {} + PATCH_W {} + OWM_P {} + KAN_C {})",
        out_path,
        w.len(),
        w_patch.len(),
        owm_p.len(),
        kan_c.len()
    );

    // --- Декодим ОДНИМ чекпоинтом: и текст, и код ---
    let mut tm_d = fuga::ai::htm_temporal::TemporalMemory::new(64, ctxw);
    tm_d.apply_byte_w(w);
    let kan_d = fuga::ai::kan::KanTransition {
        c: kan_c,
        updates: 0,
    };
    let mut seen: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
    let mut patch_vocab: Vec<Vec<u8>> = Vec::new();
    for corp in corpus.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
        let Ok(f) = std::fs::File::open(corp) else { continue };
        let rd = std::io::BufReader::new(f);
        for line in rd.lines().flatten() {
            for w2 in line.as_bytes().windows(2) {
                if seen.insert(w2.to_vec()) && patch_vocab.len() < 5000 {
                    patch_vocab.push(w2.to_vec());
                }
            }
            if patch_vocab.len() >= 5000 {
                break;
            }
        }
        if patch_vocab.len() >= 5000 {
            break;
        }
    }
    println!("[diag] patch_vocab.size={}", patch_vocab.len());
    for (label, seed) in [("TEXT", &seed_text), ("CODE", &seed_code)] {
        println!(
            "\n--- {} SEED {:?} ---",
            label,
            String::from_utf8_lossy(seed)
        );
        let o1 = fuga::tm_generate_latent_bytes(&tm_d, seed, 120, ctxw, None);
        println!(
            "  naive       ({} B): {:?}",
            o1.len(),
            String::from_utf8_lossy(&o1).chars().take(60).collect::<String>()
        );
        // РЕШАЮЩИЙ A/B: entropy на tm_d (identity patch) vs tm_reload (обученный patch).
        let mut tm_reload = fuga::ai::htm_temporal::TemporalMemory::new(64, ctxw);
        tm_reload.load_unified_fuga1(&out_path);
        let o2a = fuga::tm_generate_two_speed_entropy(&tm_d, seed, 200, 2, 0.60, &patch_vocab);
        println!(
            "  entropy-mem   ({} B): {:?}",
            o2a.len(),
            String::from_utf8_lossy(&o2a).chars().take(60).collect::<String>()
        );
        let o2b = fuga::tm_generate_two_speed_entropy(&tm_reload, seed, 200, 2, 0.60, &patch_vocab);
        println!(
            "  entropy-file  ({} B): {:?}",
            o2b.len(),
            String::from_utf8_lossy(&o2b).chars().take(60).collect::<String>()
        );
        let o3 = fuga::ai::tm_generate::tm_generate_hybrid(&tm_d, &kan_d, seed, 120, ctxw, 1.0);
        println!(
            "  hybrid W+K  ({} B): {:?}",
            o3.len(),
            String::from_utf8_lossy(&o3).chars().take(60).collect::<String>()
        );
    }

    println!("=== UNIFIED GPU/CPU PIPELINE COMPLETE ===\n  cpu_prepared={} gpu_applied={} in {:.1}s ({:.0} pairs/s)", cpu_steps, applied, el, applied as f64 / el);
    println!("  единый файл всех технологий: {}", out_path);
    use std::path::Path;
    let _ = Path::new("");
}
/// AST-маска (пункт 5 Byte-H-JEPA): интервалы семантичных узлов C-AST строки.
/// Возвращает (start, end, текст) для узлов с len >= MIN_LEN — «будущие узлы»
/// для макро-целей. Парсим C один раз на строку (tree-sitter).
fn ast_node_ranges(code: &[u8]) -> Vec<(usize, usize, Vec<u8>)> {
    const MIN_LEN: usize = 6;
    let src = String::from_utf8_lossy(code);
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_c::LANGUAGE.into())
        .ok();
    let Some(tree) = parser.parse(&src[..], None) else {
        return Vec::new();
    };
    let mut out: Vec<(usize, usize, Vec<u8>)> = Vec::new();
    const MAX_NODES: usize = 200; // OOM-защита: хватит на семантику, не на взрыв
    let mut cursor = tree.walk();
    loop {
        let n = cursor.node();
        let (s, e) = (n.start_byte(), n.end_byte());
        // Только узлы не ERROR, без пробельных хвостов, семантичный размер.
        if !n.is_error() && !n.is_missing() && e - s >= MIN_LEN {
            let text = src[s..e].as_bytes().to_vec();
            if text.iter().any(|&b| b.is_ascii_alphabetic() || b == b'_') {
                out.push((s, e, text));
                if out.len() >= MAX_NODES {
                    break;
                }
            }
        }
        if cursor.goto_first_child() {
            continue;
        }
        while !cursor.goto_next_sibling() {
            if !cursor.goto_parent() {
                break;
            }
        }
    }
    out
}

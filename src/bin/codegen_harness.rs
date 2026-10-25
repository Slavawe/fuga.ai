// codegen_harness.rs — решающий тест кодогенерации (25.10, ревью п.3/4).
//
// Протокол: кодовые промпты на ОТЛОЖЕННЫХ задачах (heldout_tasks.jsonl —
// решения не встречаются в train, проверено 80-символьным префиксом),
// генерация MB3 (документированные условия capability_test), затем:
//   1) синтаксис: tree_sitter_python error_count (syntax_guard);
//   2) компиляция: python compile() всего текста через subprocess.
//
// ДВА режима на каждую задачу:
//   cold — промпт «[CODE] Задача: X\n» (модель пишет с нуля);
//   warm — промпт = задача + ПЕРВАЯ ПОЛОВИНА решения (модель дописывает
//          валидную программу; честная форма «продолжи код»).
// Baseline-рука: --base-ckpt (слабый чекпоинт) на тех же задачах/сидах.
//
// Usage:
//   fuga1 codetest <ckpt> [--n 20] [--len 300] [--corpus c.jsonl] [--base-ckpt <ckpt2>]
use fuga::ai::htm_temporal::{load_unified, load_unified_macro2, TemporalMemory};
use fuga::ai::kan::KanTransition;
use serde_json::Value;
use std::collections::HashSet;
use std::io::BufRead;

struct Task {
    problem: String,
    solution: String,
}

fn load_tasks(path: &str) -> Vec<Task> {
    let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("tasks: {}", e));
    let mut out = Vec::new();
    for line in std::io::BufReader::new(f).lines().flatten() {
        let v: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let problem = v.get("problem").and_then(|x| x.as_str()).unwrap_or("").to_string();
        let solution = v.get("solution").and_then(|x| x.as_str()).unwrap_or("").to_string();
        if !problem.is_empty() && !solution.is_empty() {
            out.push(Task { problem, solution });
        }
    }
    out
}

struct Model {
    tm: TemporalMemory,
    kan: KanTransition,
    macro_wz: Vec<f32>,
    patch_vocab: Vec<Vec<u8>>,
}

fn load_model(ckpt: &str, corpus: &str) -> Model {
    let mut tm = TemporalMemory::new(64, 4);
    assert!(tm.load_unified_fuga1(ckpt), "FUGA1 не читается: {}", ckpt);
    // Документированные условия декода (capability_test): P=identity, patch_w=0
    let np = tm.patch_predictor().w.len();
    tm.apply_patch_w(vec![0.0f32; np]);
    let mut ident_p = vec![0.0f32; tm.predictor().p.len()];
    for d in 0..512 {
        ident_p[d * 512 + d] = 1.0;
    }
    tm.predictor_mut().p = ident_p;
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut patch_vocab: Vec<Vec<u8>> = Vec::new();
    if let Ok(f) = std::fs::File::open(corpus) {
        for line in std::io::BufReader::new(f).lines().flatten() {
            if line.len() < 4 {
                continue;
            }
            for w in line.as_bytes().windows(2) {
                if seen.insert(w.to_vec()) && patch_vocab.len() < 5000 {
                    patch_vocab.push(w.to_vec());
                }
            }
            if patch_vocab.len() >= 5000 {
                break;
            }
        }
    }
    let kan_c: Option<Vec<f32>> = load_unified(ckpt).and_then(|t| t.5);
    let mut kan = KanTransition::new();
    if let Some(c) = &kan_c {
        if c.len() == kan.c.len() {
            kan.c.clone_from(c);
        }
    }
    let (macro_wz, _) = load_unified_macro2(ckpt)
        .unwrap_or_else(|| (vec![0.0f32; 512 * 512], vec![0.0f32; 512 * 512]));
    Model { tm, kan, macro_wz, patch_vocab }
}

fn mb3(m: &Model, seed: &[u8], len: usize) -> Vec<u8> {
    fuga::tm_generate_megabyte_v3(
        &m.tm, seed, len, 9, 2, &m.patch_vocab, 8, 0.3, 0.20, 0.8, 0.0, 0.30, 0.0,
        &m.macro_wz, 0.3,
    )
}

/// Оценка одного полного текста: (errors, compiles)
fn assess(text: &str, tmp: &str) -> (usize, bool) {
    let errors = fuga::ai::syntax_guard::python_error_count(text.as_bytes());
    std::fs::write(tmp, text).unwrap();
    let ok = std::process::Command::new("python")
        .args([
            "-c",
            "import sys; compile(open(sys.argv[1], encoding='utf-8').read(), 'g', 'exec')",
            tmp,
        ])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    (errors, ok)
}

fn run_arm(name: &str, ckpt: &str, tasks: &[Task], corpus: &str, n: usize, len: usize) {
    let m = load_model(ckpt, corpus);
    println!("== CODETEST рука '{}': {} | задач={} len={} ==", name, ckpt, tasks.len(), len);
    println!(
        "{:<4} {:<7} {:>7} {:>8} {:>9}",
        "task", "режим", "errors", "compiles", "gen_B"
    );
    let tmp = format!("codetest_{}.py", name);
    let (mut cold_ok, mut warm_ok, mut cold_syn, mut warm_syn) = (0usize, 0usize, 0usize, 0usize);
    for (ti, task) in tasks.iter().take(n).enumerate() {
        // cold: задача с нуля
        let prompt_c = format!("[CODE] Задача: {}\n", task.problem);
        let mut seed = prompt_c.as_bytes().to_vec();
        let gen = mb3(&m, &seed, len);
        seed.extend_from_slice(&gen);
        let full = String::from_utf8_lossy(&seed).to_string();
        let (errs, comp) = assess(&full, &tmp);
        if errs == 0 {
            cold_syn += 1;
        }
        if comp {
            cold_ok += 1;
        }
        println!("{:<4} {:<7} {:>7} {:>8} {:>9}", ti, "cold", errs, comp, gen.len());

        // warm: дописать первую половину решения
        let sol_bytes: Vec<char> = task.solution.chars().collect();
        let half: String = sol_bytes[..sol_bytes.len() / 2].iter().collect();
        let prompt_w = format!("[CODE] Задача: {}\n{}", task.problem, half);
        let mut seed = prompt_w.as_bytes().to_vec();
        let gen = mb3(&m, &seed, len);
        seed.extend_from_slice(&gen);
        let full = String::from_utf8_lossy(&seed).to_string();
        let (errs, comp) = assess(&full, &tmp);
        if errs == 0 {
            warm_syn += 1;
        }
        if comp {
            warm_ok += 1;
        }
        println!("{:<4} {:<7} {:>7} {:>8} {:>9}", ti, "warm", errs, comp, gen.len());
    }
    let nn = tasks.len().min(n);
    println!(
        "== '{}': cold: синтаксис-чисто {}/{}, компилируется {}/{}; warm: {}/{}, {}/{} ==",
        name, cold_syn, nn, cold_ok, nn, warm_syn, nn, warm_ok, nn
    );
    let _ = std::fs::remove_file(&tmp);
}

pub fn run(args: Vec<String>) {
    if args.len() < 2 {
        eprintln!("Usage: fuga1 codetest <ckpt> [--n 20] [--len 300] [--corpus c.jsonl] [--base-ckpt <ckpt2>] [--tasks heldout_tasks.jsonl]");
        std::process::exit(1);
    }
    let ckpt = args[1].clone();
    let mut n = 20usize;
    let mut len = 300usize;
    let mut corpus = String::new();
    let mut base = String::new();
    let mut tasks_path = "train_fuga1/corpus/heldout_tasks.jsonl".to_string();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--n" => {
                n = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(20);
                i += 2;
            }
            "--len" => {
                len = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(300);
                i += 2;
            }
            "--corpus" => {
                corpus = args.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            "--base-ckpt" => {
                base = args.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            "--tasks" => {
                tasks_path = args.get(i + 1).cloned().unwrap_or_else(|| tasks_path.clone());
                i += 2;
            }
            _ => i += 1,
        }
    }
    if corpus.is_empty() {
        for c in ["train_fuga1/corpus/fuga13_train.jsonl", "train_fuga1/corpus/fuga12_train.jsonl"] {
            if std::path::Path::new(c).exists() {
                corpus = c.to_string();
                break;
            }
        }
    }
    let tasks = load_tasks(&tasks_path);
    if tasks.is_empty() {
        eprintln!("codetest: задач нет в {}", tasks_path);
        std::process::exit(1);
    }
    run_arm("main", &ckpt, &tasks, &corpus, n, len);
    if !base.is_empty() {
        run_arm("base", &base, &tasks, &corpus, n, len);
    }
}

fn main() {
    run(std::env::args().collect());
}

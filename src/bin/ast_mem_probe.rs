// ast_mem_probe.rs — ИЗОЛИРОВАННЫЙ замер памяти ast_node_ranges.
// Читает строки из корпуса, вызывает ту же логику, что в трейнере,
// считает узлы и суммарный объём текстов — проверка, что НЕ растёт
// бесконечно. Аналог ast_node_ranges из unified_gpu_train.
use std::io::BufRead;

fn ast_node_ranges_probe(code: &[u8]) -> Vec<(usize, usize, Vec<u8>)> {
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
    let mut cursor = tree.walk();
    loop {
        let n = cursor.node();
        let (s, e) = (n.start_byte(), n.end_byte());
        if !n.is_error() && !n.is_missing() && e - s >= MIN_LEN {
            let text = src[s..e].as_bytes().to_vec();
            if text.iter().any(|&b| b.is_ascii_alphabetic() || b == b'_') {
                out.push((s, e, text));
            }
        }
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                return out;
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corp = args.get(1).cloned().unwrap_or_else(|| "corpus_doc_code_pairs.jsonl".into());
    let limit = args.get(2).and_then(|s| s.parse::<usize>().ok()).unwrap_or(200);
    let f = std::fs::File::open(&corp).unwrap();
    let rd = std::io::BufReader::new(f);
    let mut n_lines = 0usize;
    let mut total_nodes = 0usize;
    let mut total_text_bytes = 0usize;
    let mut max_nodes = 0usize;
    let mut max_text = 0usize;
    for line in rd.lines().flatten() {
        if n_lines >= limit {
            break;
        }
        n_lines += 1;
        // Та же логика extract_bytes минимальная: сырые байты строки кода.
        let data: Vec<u8> = line.into_bytes();
        if data.len() <= 50_000 {
            let ranges = ast_node_ranges_probe(&data);
            let nt: usize = ranges.iter().map(|(_, _, t)| t.len()).sum();
            total_nodes += ranges.len();
            total_text_bytes += nt;
            max_nodes = max_nodes.max(ranges.len());
            max_text = max_text.max(nt);
        }
    }
    println!(
        "строк: {} | узлов всего: {} | сумма текстов: {} B ({:.1} MB) | макс узлов/строка: {} | макс текст/строка: {} B",
        n_lines,
        total_nodes,
        total_text_bytes,
        total_text_bytes as f64 / 1_048_576.0,
        max_nodes,
        max_text
    );
}
use std::fs;
use std::path::Path;

fn main() {
    let dirs = vec!["temp_repos", "text_corpus_processed", "text_batches", "text_done"];
    let exts = &[".rs", ".py", ".js", ".ts", ".c", ".cpp", ".h", ".go", ".java"];
    let mut pairs = Vec::new();

    for dir in &dirs {
        let mut files = Vec::new();
        collect_files(dir, exts, &mut files);
        for fp in &files {
            let src = match fs::read_to_string(fp) { Ok(s) => s, Err(_) => continue };
            let lines: Vec<&str> = src.lines().collect();
            let mut i = 0;
            while i < lines.len() {
                let line = lines[i].trim();
                if line.starts_with("///") || line.starts_with("//!") || line.starts_with("/**") || line.starts_with("/*") || line.starts_with("# ") || line.starts_with("\"\"\"") {
                    let mut doc_lines = Vec::new();
                    while i < lines.len() {
                        let l = lines[i].trim();
                        if l.starts_with("///") { doc_lines.push(l.trim_start_matches("///").trim()); }
                        else if l.starts_with("//!") { doc_lines.push(l.trim_start_matches("//!").trim()); }
                        else if l.starts_with("/*") || l.starts_with("/**") { doc_lines.push(l.trim_start_matches("/*").trim_start_matches("/**").trim()); }
                        else if l.starts_with("# ") { doc_lines.push(l.trim_start_matches("# ")); }
                        else if l == "*/" || l.ends_with("*/") { if l == "*/" { doc_lines.push(""); } else { doc_lines.push(l.trim_end_matches("*/").trim()); } i += 1; break; }
                        else if l.starts_with("\"\"\"") { doc_lines.push(l.trim_start_matches("\"\"\"")); i += 1; while i < lines.len() && !lines[i].trim().ends_with("\"\"\"") { doc_lines.push(lines[i].trim()); i += 1; } break; }
                        else if l.is_empty() || l.starts_with("//") || l.starts_with("use ") || l.starts_with("import ") || l.starts_with("fn ") || l.starts_with("pub ") || l.starts_with("struct ") || l.starts_with("enum ") || l.starts_with("impl ") || l.starts_with("trait ") || l.starts_with("def ") || l.starts_with("class ") || l.starts_with("function ") || l.starts_with("int ") || l.starts_with("void ") || l.starts_with("char ") || l.starts_with("static ") { break; }
                        else { doc_lines.push(l); }
                        i += 1;
                    }
                    let doc = doc_lines.join(" ").trim().to_string();
                    if doc.len() < 10 { continue; }

                    let mut code_lines = Vec::new();
                    let mut brace_depth = 0i32;
                    let start = i;
                    while i < lines.len() && (brace_depth > 0 || code_lines.is_empty() || i - start < 3) {
                        let cl = lines[i];
                        brace_depth += cl.matches('{').count() as i32;
                        brace_depth -= cl.matches('}').count() as i32;
                        code_lines.push(cl);
                        i += 1;
                        if brace_depth <= 0 && code_lines.len() >= 5 { break; }
                    }
                    let code = code_lines.join("\n").trim().to_string();
                    if code.len() < 20 || code.len() > 2000 { continue; }
                    pairs.push((doc, code, fp.clone()));
                } else {
                    i += 1;
                }
            }
        }
    }

    let out_path = "corpus_doc_code_pairs.jsonl";
    let mut out = fs::File::create(out_path).unwrap();
    for (doc, code, src) in &pairs {
        let entry = serde_json::json!({
            "doc": doc,
            "code": code,
            "source": src
        });
        writeln!(out, "{}", entry).unwrap();
    }
    println!("Generated {} docstring→code pairs → {}", pairs.len(), out_path);
}

fn collect_files(dir: &str, exts: &[&str], out: &mut Vec<String>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if n.starts_with('.') || n == "node_modules" || n == "target" || n == "thirdparty" { continue; }
                collect_files(&p.to_string_lossy(), exts, out);
            } else if let Some(e) = p.extension().and_then(|e| e.to_str()) {
                if exts.iter().any(|x| *x == format!(".{}", e).as_str()) {
                    out.push(p.to_string_lossy().to_string());
                }
            }
        }
    }
}

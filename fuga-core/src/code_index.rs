use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[pyclass]
/// Параллельный AST-индексатор на rayon (6 языков).
pub struct CodeIndexer;

fn lang_for(ext: &str) -> Option<i32> {
    match ext {
        "c" | "h" => Some(0),
        "cpp" | "hpp" | "cc" | "hxx" => Some(1),
        "rs" => Some(2),
        "py" => Some(3),
        "go" => Some(4),
        "java" => Some(5),
        _ => None,
    }
}

fn make_parser(lang_id: i32) -> Option<tree_sitter::Parser> {
    let mut p = tree_sitter::Parser::new();
    let language: tree_sitter::Language = match lang_id {
        0 => tree_sitter_c::LANGUAGE.into(),
        1 => tree_sitter_cpp::LANGUAGE.into(),
        2 => tree_sitter_rust::LANGUAGE.into(),
        3 => tree_sitter_python::LANGUAGE.into(),
        4 => tree_sitter_go::LANGUAGE.into(),
        5 => tree_sitter_java::LANGUAGE.into(),
        _ => return None,
    };
    p.set_language(&language).ok()?;
    Some(p)
}

fn walk_files(root: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![std::path::PathBuf::from(root)];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let ext = path
                    .extension()
                    .map(|e| e.to_string_lossy().to_lowercase())
                    .unwrap_or_default();
                if lang_for(&ext).is_some() {
                    out.push(path.to_string_lossy().into_owned());
                }
            }
        }
    }
    out
}

fn extract_names<'a>(node: tree_sitter::Node, source: &'a [u8],
                     out: &mut Vec<(String, String)>) {
    let kind = node.kind();
    let is_target = matches!(
        kind,
        "function_definition" | "function_declaration" | "method_definition"
            | "function_item" | "method_declaration" | "class_declaration"
            | "struct_specifier" | "type_definition" | "interface_declaration"
    );
    if is_target {
        let text = node.utf8_text(source).unwrap_or("").to_string();
        // имя: для функций — последнее слово до '(' ; иначе — до пробела/} 
        let name = if text.contains('(') {
            text.split('(').next().unwrap_or("").split_whitespace().last()
                .unwrap_or("").to_string()
        } else {
            text.split_whitespace().last().unwrap_or("").to_string()
        };
        if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            out.push((kind.to_string(), name));
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        extract_names(child, source, out);
    }
}

#[pymethods]
impl CodeIndexer {
    #[new]
    pub fn new() -> Self {
        CodeIndexer
    }

    /// (paths, total_lines) для интроспекции.
    pub fn index_dir(
        &self,
        py: Python<'_>,
        root: &str,
        max_files: usize,
    ) -> (Vec<(String, String, String)>, usize) {
        let files = walk_files(root);
        let total_lines = AtomicUsize::new(0);
        let items: Vec<(String, String, String)> = py.allow_threads(|| {
            files
                .par_iter()
                .take(max_files)
                .filter_map(|path| {
                    let ext = std::path::Path::new(path)
                        .extension()?
                        .to_str()?
                        .to_lowercase();
                    let lang_id = lang_for(&ext)?;
                    let bytes = std::fs::read(path).ok()?;
                    if bytes.len() > 3_000_000 {
                        return None;
                    }
                    let lines = bytes.iter().filter(|&&b| b == b'\n').count();
                    total_lines.fetch_add(lines, Ordering::Relaxed);
                    let mut parser = make_parser(lang_id)?;
                    let tree = parser.parse(&bytes, None)?;
                    let mut names: Vec<(String, String)> = Vec::new();
                    extract_names(tree.root_node(), &bytes, &mut names);
                    let file = path.rsplit('/').next().unwrap_or(path).to_string();
                    Some(names.into_iter().take(256).map(|(k, n)| {
                        (k, n, file.clone())
                    }).collect::<Vec<_>>())
                })
                .flatten()
                .collect()
        });
        (items, total_lines.load(Ordering::Relaxed))
    }

    pub fn threads(&self) -> usize {
        rayon::current_num_threads()
    }
}

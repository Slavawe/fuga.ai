use tree_sitter::{Language, Parser, StreamingIterator};
use std::path::Path;

/// Поддерживаемые языки
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum LanguageId {
    Rust,
    C,
    Cpp,
    Go,
    Python,
    TypeScript,
    JavaScript,
}

impl LanguageId {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "rs" => Some(Self::Rust),
            "c" | "h" => Some(Self::C),
            "cpp" | "cc" | "cxx" | "hpp" | "hh" => Some(Self::Cpp),
            "go" => Some(Self::Go),
            "py" => Some(Self::Python),
            "ts" | "tsx" => Some(Self::TypeScript),
            "js" | "jsx" => Some(Self::JavaScript),
            _ => None,
        }
    }

    pub fn from_path(path: &Path) -> Option<Self> {
        path.extension().and_then(|e| e.to_str()).and_then(Self::from_extension)
    }

    pub fn tree_sitter_language(&self) -> Language {
        match self {
            LanguageId::Rust => tree_sitter_rust::LANGUAGE.into(),
            LanguageId::C => tree_sitter_c::LANGUAGE.into(),
            LanguageId::Cpp => tree_sitter_cpp::LANGUAGE.into(),
            LanguageId::Go => tree_sitter_go::LANGUAGE.into(),
            LanguageId::Python => tree_sitter_python::LANGUAGE.into(),
            LanguageId::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            LanguageId::JavaScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            LanguageId::Rust => "Rust",
            LanguageId::C => "C",
            LanguageId::Cpp => "C++",
            LanguageId::Go => "Go",
            LanguageId::Python => "Python",
            LanguageId::TypeScript => "TypeScript",
            LanguageId::JavaScript => "JavaScript",
        }
    }

    pub fn function_kinds(&self) -> &'static [&'static str] {
        match self {
            LanguageId::Rust => &["function_item"],
            LanguageId::C | LanguageId::Cpp => &["function_definition"],
            LanguageId::Go => &["function_declaration", "method_declaration"],
            LanguageId::Python => &["function_definition"],
            LanguageId::TypeScript | LanguageId::JavaScript => &["function_declaration", "arrow_function", "method_definition"],
        }
    }

    pub fn function_name_kinds(&self) -> &'static [&'static str] {
        match self {
            LanguageId::Rust => &["identifier"],
            LanguageId::C | LanguageId::Cpp => &["identifier"],
            LanguageId::Go => &["field_identifier", "identifier"],
            LanguageId::Python => &["identifier"],
            LanguageId::TypeScript | LanguageId::JavaScript => &["property_identifier", "identifier"],
        }
    }
}

pub fn is_supported(ext: &str) -> bool {
    LanguageId::from_extension(ext).is_some()
}

/// Парсит исходный код, возвращает Tree
pub fn parse_source(source: &str, lang: LanguageId) -> Option<tree_sitter::Tree> {
    let mut parser = Parser::new();
    parser.set_language(&lang.tree_sitter_language()).ok()?;
    parser.parse(source, None)
}

/// Запускает tree-sitter query и возвращает совпадения
/// Принимает source + Tree (Tree создаётся отдельно, чтобы lifetime совпадал)
pub fn run_query<'a>(query_str: &str, source: &'a str, lang: LanguageId, tree: &'a tree_sitter::Tree) -> Option<Vec<QueryResult>> {
    let lang_inst = lang.tree_sitter_language();
    let query = tree_sitter::Query::new(&lang_inst, query_str).ok()?;
    let root = tree.root_node();
    let mut cursor = tree_sitter::QueryCursor::new();

    let mut results = Vec::new();
    let mut matches = cursor.matches(&query, root, source.as_bytes());
    while let Some(m) = matches.next() {
        for c in m.captures.iter() {
            let start = c.node.start_position();
            let end = c.node.end_position();
            let byte_range = c.node.byte_range();
            let text = &source[byte_range.clone()];
            results.push(QueryResult {
                capture_name: query.capture_names()[c.index as usize].to_string(),
                text: text.to_string(),
                start_byte: byte_range.start,
                end_byte: byte_range.end,
                start_line: start.row + 1,
                start_col: start.column + 1,
                end_line: end.row + 1,
                end_col: end.column + 1,
            });
        }
    }

    Some(results)
}

#[derive(Debug, Clone)]
pub struct QueryResult {
    pub capture_name: String,
    pub text: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
}

/// Traverse all nodes recursively with a callback
pub fn traverse_tree<F>(node: tree_sitter::Node, f: &F)
where
    F: Fn(tree_sitter::Node),
{
    f(node);
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        traverse_tree(child, f);
    }
}

/// Count nodes matching given kinds
pub fn count_nodes_by_kind(node: tree_sitter::Node, kinds: &[&str]) -> usize {
    let mut count = if kinds.contains(&node.kind()) { 1 } else { 0 };
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        count += count_nodes_by_kind(child, kinds);
    }
    count
}

/// Collect function names from a tree
pub fn collect_function_names(tree: &tree_sitter::Tree, source: &str, lang: LanguageId) -> Vec<String> {
    let root = tree.root_node();
    let mut names = Vec::new();
    collect_func_names_recursive(root, source, lang.function_kinds(), lang.function_name_kinds(), &mut names);
    names
}

fn collect_func_names_recursive(
    node: tree_sitter::Node,
    source: &str,
    func_kinds: &[&str],
    name_kinds: &[&str],
    names: &mut Vec<String>,
) {
    if func_kinds.contains(&node.kind()) {
        let mut found = false;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if name_kinds.contains(&child.kind()) {
                names.push(source[child.byte_range()].to_string());
                found = true;
            }
        }
        // If name not found as direct child, search recursively
        if !found {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                if let Some(name) = find_name_recursive(child, source, name_kinds) {
                    names.push(name);
                    break;
                }
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_func_names_recursive(child, source, func_kinds, name_kinds, names);
    }
}

fn find_name_recursive(node: tree_sitter::Node, source: &str, name_kinds: &[&str]) -> Option<String> {
    if name_kinds.contains(&node.kind()) {
        return Some(source[node.byte_range()].to_string());
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if let Some(name) = find_name_recursive(child, source, name_kinds) {
            return Some(name);
        }
    }
    None
}

/// Find enclosing function for a given line number
pub fn find_enclosing_function(tree: &tree_sitter::Tree, source: &str, lang: LanguageId, target_line: usize) -> Option<String> {
    let root = tree.root_node();
    let func_kinds = lang.function_kinds();
    let name_kinds = lang.function_name_kinds();
    find_enc_fn_recursive(root, source, func_kinds, name_kinds, target_line)
}

fn find_enc_fn_recursive(
    node: tree_sitter::Node,
    source: &str,
    func_kinds: &[&str],
    name_kinds: &[&str],
    target_line: usize,
) -> Option<String> {
    let node_start = node.start_position().row + 1;
    let node_end = node.end_position().row + 1;
    if target_line < node_start || target_line > node_end {
        return None;
    }
    if func_kinds.contains(&node.kind()) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if name_kinds.contains(&child.kind()) {
                return Some(source[child.byte_range()].to_string());
            }
        }
        return Some(node.kind().to_string());
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if let Some(name) = find_enc_fn_recursive(child, source, func_kinds, name_kinds, target_line) {
            return Some(name);
        }
    }
    None
}
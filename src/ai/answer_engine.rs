use crate::ai::FugaAI;
use crate::ai::memory_store::{MemoryEntry, MemoryStore};
use crate::core::hypervector::Hypervector;
use crate::core::wave_cube::WaveCube;
use crate::weaver::pattern_matcher::TokenInfo;
use std::path::Path;

pub struct AnswerEngine<const N: usize, const S: usize> {
    pub cube: WaveCube<N, S>,
    pub memory: MemoryStore,
    pub dim: usize,
}

pub struct AnswerResult {
    pub query: String,
    pub route: String,
    pub hits: Vec<AnswerHit>,
    pub cube_entropy: f64,
}

pub struct AnswerHit {
    pub source_doc: String,
    pub similarity: f64,
    pub text: String,
    pub snippet: Option<String>,
}

impl<const N: usize, const S: usize> AnswerEngine<N, S> {
    pub fn load(cube_path: &str) -> Result<Self, String> {
        let cube = WaveCube::<N, S>::load_bin(cube_path)?;
        let mem_path = cube_path.replace(".bin", "_mem.bin");
        let memory = if std::path::Path::new(&mem_path).exists() {
            MemoryStore::load_bin(&mem_path)?
        } else {
            MemoryStore::new()
        };
        let dim = cube.dim;
        Ok(Self { cube, memory, dim })
    }

    pub fn search(&self, query: &str) -> AnswerResult {
        let route = self.route_query(query);

        let tokens: Vec<TokenInfo> = query
            .split_whitespace()
            .enumerate()
            .map(|(_, w)| TokenInfo {
                id: crate::weaver::token_id(&w),
                text: w.to_string(),
            })
            .collect();

        let mut ai = FugaAI::<N, S>::new(self.dim, 3);
        ai.cube = self.cube.clone();

        let output = ai.think(&tokens);
        let entropy = ai.cube.global_entropy();

        let mut all_hits: Vec<(f64, String, String)> = Vec::new();

        let text_results = search_memory_by_words(&self.memory, query, 30);
        for (_idx, sim, entry) in &text_results {
            all_hits.push((*sim, entry.source_doc.clone(), entry.text.clone()));
        }

        if all_hits.len() < 5 {
            for st in &output.super_tokens {
                let results = self.memory.search(&st.vector, 10);
                for (_idx, sim, entry) in &results {
                    all_hits.push((*sim, entry.source_doc.clone(), entry.text.clone()));
                }
            }
        }

        all_hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let query_owned = query.to_string();
        let mut seen = std::collections::HashSet::new();
        let hits: Vec<AnswerHit> = all_hits
            .into_iter()
            .filter(|(_, doc, text)| {
                let key = format!("{}::{}", doc, text);
                seen.insert(key)
            })
            .take(10)
            .map(|(sim, doc, text)| {
                let snippet = if sim > 0.35 {
                    extract_relevant_snippet(&doc, &query_owned)
                } else {
                    None
                };
                let func_match = text
                    .split(';')
                    .find(|part| {
                        query_owned
                            .split_whitespace()
                            .any(|w| part.to_lowercase().contains(&w.to_lowercase()))
                    })
                    .unwrap_or("")
                    .trim()
                    .to_string();
                AnswerHit {
                    source_doc: doc,
                    similarity: sim,
                    text: func_match,
                    snippet,
                }
            })
            .collect();

        AnswerResult {
            query: query_owned,
            route,
            hits,
            cube_entropy: entropy,
        }
    }

    pub fn search_with_prompts(&self, query: &str, prompt_names: &[String]) -> AnswerResult {
        if prompt_names.is_empty() {
            return self.search(query);
        }
        let dim = self.dim;
        let prompt_vecs = crate::ai::prompts::PromptVectors::resolve(prompt_names, dim);
        let prompt_refs: Vec<&Hypervector> = prompt_vecs.iter().collect();

        let route = self.route_query(query);

        let tokens: Vec<TokenInfo> = query
            .split_whitespace()
            .enumerate()
            .map(|(_, w)| TokenInfo {
                id: crate::weaver::token_id(&w),
                text: w.to_string(),
            })
            .collect();

        let mut ai = FugaAI::<N, S>::new(self.dim, 3);
        ai.cube = self.cube.clone();

        let output = ai.think(&tokens);
        let entropy = ai.cube.global_entropy();

        let mut all_hits: Vec<(f64, String, String)> = Vec::new();

        for st in &output.super_tokens {
            let modulated = st.vector.bind(prompt_refs[0]);
            let results = self.memory.search(&modulated, 10);
            for (_idx, sim, entry) in &results {
                all_hits.push((*sim, entry.source_doc.clone(), entry.text.clone()));
            }
        }

        let text_results = search_memory_by_words(&self.memory, query, 30);
        for (_idx, sim, entry) in &text_results {
            all_hits.push((*sim, entry.source_doc.clone(), entry.text.clone()));
        }

        all_hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let query_owned = query.to_string();
        let mut seen = std::collections::HashSet::new();
        let hits: Vec<AnswerHit> = all_hits
            .into_iter()
            .filter(|(_, doc, text)| {
                let key = format!("{}::{}", doc, text);
                seen.insert(key)
            })
            .take(10)
            .map(|(sim, doc, text)| AnswerHit {
                source_doc: doc,
                similarity: sim,
                text,
                snippet: None,
            })
            .collect();

        AnswerResult {
            query: query_owned,
            route,
            hits,
            cube_entropy: entropy,
        }
    }

    fn route_query(&self, query: &str) -> String {
        let code_keywords = [
            "fn ", "func ", "def ", "struct ", "class ", "impl ", "int ", "void ", "char ",
            "return ", "static ", "pub ", "const ", "let ", "mut ", "->", "=>", "::", "#include",
            "import ",
        ];
        let has_code = code_keywords.iter().any(|kw| query.contains(kw));
        if has_code {
            "CodeLogic".to_string()
        } else {
            "GeneralLanguage".to_string()
        }
    }

    pub fn format_summary(&self, result: &AnswerResult) -> String {
        let mut out = String::new();
        out.push_str(&format!("Fuga AI — Structured Answer\n"));
        out.push_str(&format!("Query:   {}\n", result.query));
        out.push_str(&format!("Route:   {}\n", result.route));
        out.push_str(&format!("Matches: {}\n", result.hits.len()));
        out.push_str(&format!("Entropy: {:.4}\n", result.cube_entropy));
        out.push('\n');

        if result.hits.is_empty() {
            out.push_str("No resonance in cube.\n");
            return out;
        }

        let top = &result.hits[0];
        out.push_str(&format!(
            "Top match: {} (sim={:.3})\n",
            top.source_doc, top.similarity
        ));
        if let Some(ref snippet) = top.snippet {
            out.push_str("```\n");
            out.push_str(snippet);
            out.push_str("\n```\n");
        } else {
            let truncated = if top.text.len() > 200 {
                format!("{}...", &top.text[..200])
            } else {
                top.text.clone()
            };
            out.push_str(&truncated);
            out.push('\n');
        }
        out.push('\n');

        if result.hits.len() > 1 {
            out.push_str("Related:\n");
            for hit in result.hits.iter().skip(1).take(5) {
                let label = Path::new(&hit.source_doc)
                    .file_name()
                    .map(|f| f.to_string_lossy())
                    .unwrap_or(std::borrow::Cow::Borrowed(&hit.source_doc));
                out.push_str(&format!("  {} (sim={:.3})\n", label, hit.similarity));
            }
        }

        out
    }

    pub fn format_explain(&self, result: &AnswerResult) -> String {
        let mut out = String::new();
        out.push_str("**Fuga AI — Resonance Response**\n\n");

        if result.hits.is_empty() {
            out.push_str("No resonant context found in cube.\n");
            return out;
        }

        let top = &result.hits[0];

        let file_name = Path::new(&top.source_doc)
            .file_name()
            .map(|f| f.to_string_lossy())
            .unwrap_or(std::borrow::Cow::Borrowed(&top.source_doc));
        let parent = Path::new(&top.source_doc)
            .parent()
            .and_then(|p| p.file_name())
            .map(|f| f.to_string_lossy())
            .unwrap_or(std::borrow::Cow::Borrowed("?"));

        out.push_str(&format!("**Module:** `{}` ({})\n", file_name, parent));
        out.push_str(&format!(
            "**Resonance:** {:.3} (phase lock)\n",
            top.similarity
        ));

        if let Some(ref snippet) = top.snippet {
            let lang_hint = match Path::new(&top.source_doc)
                .extension()
                .and_then(|e| e.to_str())
            {
                Some("rs") => "rust",
                Some("c") | Some("h") => "c",
                Some("py") => "python",
                Some("go") => "go",
                Some("ts") | Some("tsx") => "typescript",
                Some("js") | Some("jsx") => "javascript",
                Some("cpp") | Some("cc") | Some("hpp") => "cpp",
                _ => "",
            };
            out.push_str(&format!(
                "**Key fragment:**\n```{}\n{}\n```\n",
                lang_hint, snippet
            ));
        }

        if result.hits.len() > 1 {
            out.push_str("---\n**Related modules:**\n");
            for hit in result.hits.iter().skip(1).take(4) {
                let label = Path::new(&hit.source_doc)
                    .file_name()
                    .map(|f| f.to_string_lossy())
                    .unwrap_or(std::borrow::Cow::Borrowed(&hit.source_doc));
                out.push_str(&format!("- `{}` (sim={:.3})\n", label, hit.similarity));
            }
        }

        out.push_str(&format!(
            "\n*Status: phase-locked at entropy {:.4}*\n",
            result.cube_entropy
        ));
        out
    }
}

fn search_memory_by_words<'a>(
    memory: &'a MemoryStore,
    query: &str,
    top_k: usize,
) -> Vec<(usize, f64, &'a MemoryEntry)> {
    let query_lower = query.to_lowercase();
    let words: Vec<&str> = query_lower
        .split_whitespace()
        .filter(|w| w.len() > 2)
        .collect();

    if words.is_empty() {
        return memory.search_by_text(query, top_k);
    }

    let mut scores: Vec<(usize, f64, usize)> = memory
        .all_entries()
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let source_lower = e.source_doc.to_lowercase();
            let fname = std::path::Path::new(&e.source_doc)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_lowercase();
            let mut match_count = 0;
            let is_source = source_lower.ends_with(".c")
                || source_lower.ends_with(".rs")
                || source_lower.ends_with(".go")
                || source_lower.ends_with(".py");
            for w in &words {
                if fname.contains(w) {
                    match_count += 5;
                } else if source_lower.contains(w) {
                    match_count += 2;
                }
                let text_lower = e.text.to_lowercase();
                let text_hits = text_lower.matches(w).count();
                if text_hits > 0 {
                    match_count += 3 + text_hits.min(3);
                }
            }
            if is_source {
                match_count += 2;
            }
            (
                i,
                match_count as f64 / (words.len() as f64 * 11.0 + 2.0),
                match_count,
            )
        })
        .collect();

    scores.sort_by(|a, b| b.2.cmp(&a.2));
    scores.truncate(top_k);
    let mut results: Vec<(usize, f64, &MemoryEntry)> = scores
        .into_iter()
        .filter(|(_, _, total)| *total > 0)
        .map(|(i, s, _)| (i, s.min(1.0), &memory.all_entries()[i]))
        .collect();
    results.sort_by(|a, b| {
        b.1.partial_cmp(&a.1).unwrap().then_with(|| {
            let a_src = a.2.source_doc.ends_with(".c")
                || a.2.source_doc.ends_with(".rs")
                || a.2.source_doc.ends_with(".go")
                || a.2.source_doc.ends_with(".py");
            let b_src = b.2.source_doc.ends_with(".c")
                || b.2.source_doc.ends_with(".rs")
                || b.2.source_doc.ends_with(".go")
                || b.2.source_doc.ends_with(".py");
            b_src.cmp(&a_src)
        })
    });
    results
}

fn extract_relevant_snippet(file_path: &str, query_text: &str) -> Option<String> {
    let content = std::fs::read_to_string(file_path).ok()?;

    let lang = lang_from_extension(file_path);
    let query_lower = query_text.to_lowercase();
    let query_words: Vec<&str> = query_lower
        .split_whitespace()
        .filter(|w| w.len() > 1)
        .collect();
    if query_words.is_empty() {
        return None;
    }

    if let Some(lang_id) = lang {
        if let Some(snippet) = extract_ast_node(&content, lang_id, &query_words) {
            return Some(snippet);
        }
    }

    let lines: Vec<&str> = content.lines().collect();

    let mut best_line = None;
    let mut best_score = 0usize;
    for (i, l) in lines.iter().enumerate() {
        let lower = l.to_lowercase();
        let score: usize = query_words.iter().map(|kw| lower.matches(kw).count()).sum();
        let is_comment = lower.trim_start().starts_with("//")
            || lower.trim_start().starts_with("/*")
            || lower.trim_start().starts_with("#")
            || lower.trim_start().starts_with('*');
        let adjusted = if is_comment { score / 2 } else { score * 2 };
        if adjusted > best_score {
            best_score = adjusted;
            best_line = Some(i);
        }
    }
    let match_line = best_line?;

    let context_start = match_line.saturating_sub(3);
    let mut context_end = lines.len().min(match_line + 12);

    for i in match_line..lines.len() {
        let trimmed = lines[i].trim();
        if trimmed.is_empty() && i > match_line + 3 {
            context_end = i + 1;
            break;
        }
        if trimmed.starts_with('}') || (trimmed.len() < 2 && i > match_line + 2) {
            context_end = i + 1;
            break;
        }
    }

    let snippet: String = lines[context_start..context_end]
        .iter()
        .map(|l| l.to_string())
        .collect::<Vec<_>>()
        .join("\n");

    if snippet.len() > 2000 {
        let truncated: String = snippet.chars().take(2000).collect();
        return Some(format!("{}...\n--- (truncated) ---", truncated));
    }
    if snippet.trim().is_empty() {
        None
    } else {
        Some(snippet)
    }
}

fn lang_from_extension(path: &str) -> Option<crate::multi::LanguageId> {
    let ext = std::path::Path::new(path).extension()?.to_str()?;
    crate::multi::LanguageId::from_extension(ext)
}

fn extract_ast_node(
    content: &str,
    lang: crate::multi::LanguageId,
    query_words: &[&str],
) -> Option<String> {
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&lang.tree_sitter_language()).ok()?;
    let tree = parser.parse(content, None)?;
    let root = tree.root_node();

    let func_kinds = lang.function_kinds();
    let struct_kinds: &[&str] = match lang {
        crate::multi::LanguageId::Rust => &["struct_item", "impl_item", "trait_item", "enum_item"],
        crate::multi::LanguageId::C | crate::multi::LanguageId::Cpp => {
            &["struct_specifier", "enum_specifier"]
        }
        _ => &["class_declaration", "struct_specifier"],
    };

    let mut candidates: Vec<(String, usize)> = Vec::new();
    collect_ast_nodes_recursive(
        root,
        content,
        func_kinds,
        struct_kinds,
        query_words,
        &mut candidates,
    );

    if candidates.is_empty() {
        return None;
    }

    candidates.sort_by(|a, b| b.1.cmp(&a.1));
    let (best_text, _) = &candidates[0];

    let lines: Vec<&str> = best_text.lines().collect();
    let display_lines = lines.len().min(40);
    let display: String = lines[..display_lines].join("\n");

    if display.len() > 2000 {
        Some(format!(
            "{}...\n--- (truncated, {} lines) ---",
            &display[..2000],
            lines.len()
        ))
    } else {
        Some(display)
    }
}

fn collect_ast_nodes_recursive(
    node: tree_sitter::Node,
    source: &str,
    func_kinds: &[&str],
    struct_kinds: &[&str],
    query_words: &[&str],
    candidates: &mut Vec<(String, usize)>,
) {
    let kind = node.kind();
    let is_target = func_kinds.contains(&kind) || struct_kinds.contains(&kind);
    if is_target {
        let byte_range = node.byte_range();
        if byte_range.start < byte_range.end && byte_range.end <= source.len() {
            let text = &source[byte_range.clone()];
            let score = count_keyword_matches(text, query_words);
            if score > 0 {
                candidates.push((text.to_string(), score));
            }
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_ast_nodes_recursive(
            child,
            source,
            func_kinds,
            struct_kinds,
            query_words,
            candidates,
        );
    }
}

fn count_keyword_matches(text: &str, query_words: &[&str]) -> usize {
    let lower = text.to_lowercase();
    query_words.iter().filter(|w| lower.contains(*w)).count()
}

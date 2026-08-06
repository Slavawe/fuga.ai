use fuga::{
    CodeQualityFilter, FixProposal, FugaEngine, LanguageId, MultiEngine, MultiFixGenerator,
    TokenBuilder, TokenInfo, WaveCube,
    ai::FugaAI,
    core::wave_cube::peek_cube_header,
    omni::{self, OmniEngine},
    speech::{self, FugaText},
    tokenize_corpus_text,
};
use std::env;
use std::sync::{Arc, Mutex};
use tiny_http::{Header, Response, Server};

const SITE_HTML: &str = include_str!("site.html");

struct WebState<const N: usize, const S: usize> {
    omni: OmniEngine<N, S>,
    speech: FugaText,
    qual: CodeQualityFilter,
    dim: usize,
    jepa: Option<fuga::TemporalPredictor>,
    /// Explicit, separately-ranked lesson tier. Gate-passed lessons only
    /// (compilable AND task-relevant), matched by task-token overlap BEFORE any
    /// competition against the 587K raw-corpus entries — so a single fresh
    /// lesson is observable by construction, not by fighting similarity.
    lessons: Vec<AgentLesson>,
}

/// A gated self-improvement lesson: the task it was generated for + the code
/// that passed compile AND relevance gates.
struct AgentLesson {
    task: String,
    code: String,
}

fn run_microwave(mw_path: &str, mode: &str, code: &str) -> String {
    let tmp = std::env::temp_dir().join("mw_eval.rs");
    let _ = std::fs::write(&tmp, code);
    let output = std::process::Command::new(mw_path)
        .arg(mode)
        .arg(&tmp)
        .output();
    let _ = std::fs::remove_file(&tmp);
    match output {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(e) => format!("ERROR:{}", e),
    }
}

fn parse_mw_output(output: &str) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for line in output.lines() {
        if let Some((k, v)) = line.split_once(':') {
            map.insert(k.to_string(), serde_json::Value::String(v.to_string()));
        }
    }
    serde_json::Value::Object(map)
}

fn handle_code_analyze<const N: usize, const S: usize>(
    state: &mut WebState<N, S>,
    body: &str,
) -> String {
    let req: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let code = req.get("code").and_then(|v| v.as_str()).unwrap_or("");
    let lang_name = req
        .get("language")
        .and_then(|v| v.as_str())
        .unwrap_or("rust");
    let path = req
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or("web_input.rs");

    let lang = match lang_name {
        "rust" => LanguageId::Rust,
        "c" => LanguageId::C,
        "cpp" => LanguageId::Cpp,
        "go" => LanguageId::Go,
        "python" => LanguageId::Python,
        "ts" | "typescript" => LanguageId::TypeScript,
        "js" => LanguageId::JavaScript,
        _ => LanguageId::Rust,
    };

    match state.qual.analyze(code, lang, path) {
        Ok(score) => {
            let mw_out = if lang == LanguageId::Rust || lang_name == "rust" {
                parse_mw_output(&run_microwave(
                    "microwave_sandbox/target/release/mini-fuga",
                    "eval-rust-file",
                    code,
                ))
            } else if lang == LanguageId::Cpp || lang == LanguageId::C {
                parse_mw_output(&run_microwave(
                    "microwave_sandbox/target/release/mini-fuga",
                    "eval-cpp-file",
                    code,
                ))
            } else {
                serde_json::Value::Null
            };

            serde_json::json!({
                "weight": score.weight,
                "safety": score.safety,
                "coherence": score.coherence,
                "violations": score.violations,
                "attacks": score.attacks,
                "bugs": score.bugs_detected,
                "summary": score.summary,
                "microwave": mw_out,
            })
            .to_string()
        }
        Err(e) => serde_json::json!({"error": e}).to_string(),
    }
}

fn handle_code_fix<const N: usize, const S: usize>(
    state: &mut WebState<N, S>,
    body: &str,
) -> String {
    let req: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let code = req.get("code").and_then(|v| v.as_str()).unwrap_or("");
    let lang_name = req
        .get("language")
        .and_then(|v| v.as_str())
        .unwrap_or("rust");
    let path = req
        .get("path")
        .and_then(|v| v.as_str())
        .unwrap_or("web_input.rs");

    let lang = match lang_name {
        "rust" => LanguageId::Rust,
        "c" => LanguageId::C,
        "cpp" => LanguageId::Cpp,
        "go" => LanguageId::Go,
        "python" => LanguageId::Python,
        "ts" | "typescript" => LanguageId::TypeScript,
        "js" => LanguageId::JavaScript,
        _ => LanguageId::Rust,
    };

    let orig_score = match state.qual.analyze(code, lang, path) {
        Ok(s) => s,
        Err(e) => return serde_json::json!({"error": e}).to_string(),
    };

    let proposals: Vec<FixProposal> = match lang {
        LanguageId::Rust => {
            let mut engine = FugaEngine::new(state.dim.min(8192));
            match engine.analyze(code) {
                Ok(result) => engine.generate_fixes(code, &result),
                Err(_) => Vec::new(),
            }
        }
        lang => {
            let mut multi = MultiEngine::new(state.dim.min(8192));
            let result = multi.analyze(code, lang, path);
            let fixer = MultiFixGenerator::new();
            fixer.generate_fixes(code, &result.syntax.violations, lang)
        }
    };

    if proposals.is_empty() {
        return serde_json::json!({
            "fixed": false,
            "message": "No fixes needed",
            "original_score": {
                "weight": orig_score.weight,
                "safety": orig_score.safety,
                "violations": orig_score.violations,
                "bugs": orig_score.bugs_detected,
            }
        })
        .to_string();
    }

    let mut fixed = code.to_string();
    for p in &proposals {
        if let (Some(start), Some(end)) = (p.start_byte, p.end_byte) {
            if start < fixed.len() && end <= fixed.len() {
                fixed.replace_range(start..end, &p.proposed_code);
            }
        } else {
            fixed = fixed.replace(&p.original_code, &p.proposed_code);
        }
    }

    let fixed_score = match state.qual.analyze(&fixed, lang, path) {
        Ok(s) => s,
        Err(_) => return serde_json::json!({"error": "fix analysis failed"}).to_string(),
    };

    let mw_orig = parse_mw_output(&run_microwave(
        "microwave_sandbox/target/release/mini-fuga",
        "eval-rust-file",
        code,
    ));
    let mw_fixed = parse_mw_output(&run_microwave(
        "microwave_sandbox/target/release/mini-fuga",
        "eval-rust-file",
        &fixed,
    ));

    serde_json::json!({
        "fixed": true,
        "proposals": proposals.iter().map(|p| serde_json::json!({
            "description": p.description,
            "confidence": p.confidence,
            "strategy": format!("{:?}", p.strategy),
        })).collect::<Vec<_>>(),
        "fixed_code": fixed,
        "original_score": {
            "weight": orig_score.weight,
            "safety": orig_score.safety,
            "violations": orig_score.violations,
            "bugs": orig_score.bugs_detected,
        },
        "fixed_score": {
            "weight": fixed_score.weight,
            "safety": fixed_score.safety,
            "violations": fixed_score.violations,
            "bugs": fixed_score.bugs_detected,
        },
        "microwave_original": mw_orig,
        "microwave_fixed": mw_fixed,
    })
    .to_string()
}

/// Sanitized source-doc name for a gated agent lesson. The filename boost in
/// `search_by_text` (+2 when a query word appears in the source filename) makes
/// an "agent_lesson_<task>.rs" entry structurally more discoverable than the
/// 587K raw corpus entries — the honest priority layer for absorbed lessons.
fn agent_lesson_source(task: &str) -> String {
    let slug: String = task
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' || c == '.' { c } else { '_' })
        .collect();
    format!(
        "agent_lesson_{}.rs",
        slug.trim_matches('_').chars().take(48).collect::<String>()
    )
}

/// Store a gate-passed lesson directly into the LIVE searchable memory store
/// (the one the answer tiers and code generation actually query), tagged with
/// the agent_lesson_* source doc so it wins the filename boost in text search.
fn store_lesson_live<const N: usize, const S: usize>(
    ai: &mut FugaAI<N, S>,
    task: &str,
    code: &str,
) -> serde_json::Value {
    let mut builder = TokenBuilder::new();
    let _ = builder.load_configs_from_dir("tikones");
    let flat_vocab = builder.build_flat_vocab();
    let combined = format!("TASK: {}\nCODE:\n{}", task, code);
    let tokens = tokenize_corpus_text(&combined, &flat_vocab);
    let src = agent_lesson_source(task);
    ai.absorb_with_source(&tokens, &src);
    serde_json::json!({
        "ok": true,
        "source_doc": src,
        "memory_size": ai.memory.size(),
    })
}

/// Re-inject durable lessons (agent_lessons.jsonl, CorpusDoc JSONL) into live
/// memory at startup, so gated lessons survive a mask restart.
fn load_lessons_into_memory<const N: usize, const S: usize>(ai: &mut FugaAI<N, S>, path: &str) {
    let Ok(content) = std::fs::read_to_string(path) else { return; };
    let mut n = 0usize;
    for line in content.lines() {
        let Ok(doc) = serde_json::from_str::<serde_json::Value>(line) else { continue; };
        let task = doc.get("title").and_then(|t| t.as_str()).unwrap_or("task");
        let code = doc
            .get("chapters")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first())
            .and_then(|ch| ch.get("paragraphs"))
            .and_then(|p| p.as_array())
            .and_then(|p| {
                p.iter().find(|par| {
                    par.as_str()
                        .map_or(false, |s| s.starts_with("CODE: ") || s.starts_with("CODE:"))
                })
            })
            .and_then(|par| par.as_str())
            .map(|s| s.trim_start_matches("CODE: ").to_string())
            .unwrap_or_default();
        if !code.is_empty() {
            store_lesson_live(ai, task, &code);
            n += 1;
        }
    }
    if n > 0 {
        println!("  Lessons injected into live memory: {} (source={})", n, path);
    }
}

/// Meaningful query/task tokens (mirror of agent_loop.relevance_gate stopwords).
fn meaningful_tokens(text: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "the", "a", "an", "and", "or", "of", "to", "in", "that", "it", "for",
        "generate", "safe", "rust", "function", "with", "compile", "report",
        "pass", "fail", "this", "is", "output", "only", "write", "into", "as",
        "on", "by", "from", "at",
    ];
    text.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .map(|w| w.to_lowercase())
        .filter(|w| w.len() >= 3 && !STOP.contains(&w.as_str()))
        .collect()
}

/// Find the best matching gated lesson for a query. The query must contain at
/// least one meaningful token of the lesson's task and cover >=25% of the
/// task's meaningful vocabulary — the same thresholds the gate uses, so a
/// lesson only wins when the query genuinely asks for its task. Returns None to
/// fall through to the general corpus when nothing matches.
fn find_lesson<const N: usize, const S: usize>(state: &WebState<N, S>, query: &str) -> Option<String> {
    let q: Vec<String> = meaningful_tokens(query);
    if q.is_empty() {
        return None;
    }
    let mut best: Option<(f64, &AgentLesson)> = None;
    for lesson in &state.lessons {
        let lt = meaningful_tokens(&lesson.task);
        if lt.is_empty() {
            continue;
        }
        let mut matched: Vec<String> = Vec::new();
        for t in &q {
            if lt.contains(t) && !matched.contains(t) {
                matched.push(t.clone());
            }
        }
        let hits = matched.len() as f64;
        let ratio = hits / lt.len() as f64;
        // A single shared stopword-ish token (e.g. "via") must not steal a
        // lesson: require at least two overlapping tokens AND a solid overlap
        // fraction so only queries genuinely about the lesson's task win.
        if hits >= 2.0 && ratio >= 0.5 {
            match best {
                Some((r, _)) if r >= ratio => {}
                _ => best = Some((ratio, lesson)),
            }
        }
    }
    best.map(|(_, l)| l.code.clone())
}

/// Parse durable lessons (agent_lessons.jsonl, CorpusDoc JSONL) into the
/// explicit lesson tier.
fn parse_lessons(path: &str) -> Vec<AgentLesson> {
    let Ok(content) = std::fs::read_to_string(path) else { return vec![]; };
    let mut lessons = Vec::new();
    for line in content.lines() {
        let Ok(doc) = serde_json::from_str::<serde_json::Value>(line) else { continue; };
        let task = doc.get("title").and_then(|t| t.as_str()).unwrap_or("task");
        let code = doc
            .get("chapters")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first())
            .and_then(|ch| ch.get("paragraphs"))
            .and_then(|p| p.as_array())
            .and_then(|p| {
                p.iter().find(|par| {
                    par.as_str()
                        .map_or(false, |s| s.starts_with("CODE: ") || s.starts_with("CODE:"))
                })
            })
            .and_then(|par| par.as_str())
            .map(|s| s.trim_start_matches("CODE: ").trim().to_string())
            .unwrap_or_default();
        if !code.is_empty() {
            lessons.push(AgentLesson { task: task.to_string(), code });
        }
    }
    lessons
}

fn handle_code_generate<const N: usize, const S: usize>(
    state: &mut WebState<N, S>,
    body: &str,
) -> String {
    let req: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let prompt = req.get("prompt").and_then(|v| v.as_str()).unwrap_or("");

    let mut builder = TokenBuilder::new();
    let _ = builder.load_configs_from_dir("tikones");
    let flat_vocab = builder.build_flat_vocab();
    let tokens = tokenize_corpus_text(prompt, &flat_vocab);

    let domain = OmniEngine::<N, S>::detect_domain(prompt);
    let lang = detect_lang_from_prompt(prompt);

    // C: explicit lesson tier — checked BEFORE the general corpus competition.
    // A gate-passed lesson whose task matches the prompt wins directly, so a
    // single fresh lesson is observable by construction (not by boost against
    // 587K entries). Non-matching lessons never shadow the corpus.
    if let Some(lesson_code) = find_lesson(state, prompt) {
        return serde_json::json!({
            "domain": domain,
            "generated_code": format!("// agent lesson\n{}", lesson_code),
            "language": lang,
            "hits": 1,
            "entropy": format!("{:.4}", state.omni.ai.cube.global_entropy()),
        })
        .to_string();
    }

    // C2: two-speed bridge (MegaByte/BLT split) — the H-JEPA task corridor
    // gates the TM autoregressor. When no stored lesson matches but a trained
    // TM/W exists, GENERATE a fresh code path instead of falling back to
    // corpus retrieval: the corridor (eligible tokens whose SDR is cosine-close
    // to the task words) supplies CONTENT, the TM syntax graph supplies ORDER.
    // This is what lets the mask produce NEW code for unseen tasks.
    if let Some(tp) = state.jepa.as_ref() {
        if tp.tm.cells.len() > 0 {
            let words: Vec<String> = prompt
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .filter(|w| w.len() >= 2)
                .map(|w| w.to_lowercase())
                .collect();
            if !words.is_empty() {
                let task_sdrs: Vec<fuga::SdrVector> =
                    words.iter().map(|w| fuga::encode_text(w)).collect();
                let mut task_bits = [0u64; fuga::SDR_WORDS];
                for s in &task_sdrs {
                    for i in 0..fuga::SDR_WORDS {
                        task_bits[i] |= s.bits[i];
                    }
                }
                let task_pop: f64 = task_bits.iter().map(|w| w.count_ones() as f64).sum();
                // Vocabulary: lesson words + task words (small, so content
                // dominates the TM weights — corpus-scale cands let corpus
                // noise win, as the CLI experiments showed).
                let mut vocab: Vec<String> = Vec::new();
                let mut seen_vocab = std::collections::HashSet::new();
                for l in &state.lessons {
                    for w in l
                        .code
                        .split(|c: char| !c.is_alphanumeric() && c != '_')
                        .filter(|w| w.len() >= 2)
                    {
                        let wl = w.to_lowercase();
                        if seen_vocab.insert(wl.clone()) {
                            vocab.push(wl);
                        }
                    }
                }
                for w in &words {
                    if seen_vocab.insert(w.clone()) {
                        vocab.push(w.clone());
                    }
                }
                if !vocab.is_empty() {
                    let mut eligible = std::collections::HashSet::new();
                    for w in &vocab {
                        let sdr = fuga::encode_text(w);
                        let cand_pop: f64 = sdr.bits.iter().map(|b| b.count_ones() as f64).sum();
                        let shared: f64 = sdr
                            .bits
                            .iter()
                            .zip(task_bits.iter())
                            .map(|(a, b)| (a & b).count_ones() as f64)
                            .sum();
                        let sim = if cand_pop * task_pop > 0.0 {
                            shared / (cand_pop * task_pop).sqrt()
                        } else {
                            0.0
                        };
                        // 0.05: permissive enough to include lesson body tokens
                        // (which carry the task's words) yet reject corpus noise.
                        if sim >= 0.05 {
                            eligible.insert(w.clone());
                        }
                    }
                    if !eligible.is_empty() {
                        let seed: Vec<String> =
                            prompt.split_whitespace().map(|s| s.to_lowercase()).collect();
                        let tok_seq = fuga::tm_generate(&tp.tm, &seed, 24, &vocab, 4, Some(&eligible));
                        // Guard: a bridge that degenerates to a 1-2 token echo
                        // (TM has no chain for this task) must NOT be emitted
                        // as an answer — fall through to corpus retrieval.
                        if tok_seq.len() >= 3 {
                            let generated: String = tok_seq.join(" ");
                            return serde_json::json!({
                                "domain": domain,
                                "generated_code": format!(
                                    "// fuga-bridge tm_generate (corridor {} tokens)\n{}",
                                    eligible.len(),
                                    generated
                                ),
                                "language": lang,
                                "hits": eligible.len(),
                                "entropy": format!("{:.4}", state.omni.ai.cube.global_entropy()),
                                "bridge": true,
                            })
                            .to_string();
                        }
                    }
                }
            }
        }
    }

    let output = state.omni.ai.think(&tokens);

    let mut hits: Vec<(f64, String)> = Vec::new();
    for st in &output.super_tokens {
        let results = state.omni.ai.memory.search(&st.vector, 5);
        for (_idx, sim, entry) in &results {
            hits.push((*sim, entry.text.clone()));
        }
    }

    hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut seen = std::collections::HashSet::new();
    hits.retain(|(_, t)| seen.insert(t.clone()));
    hits.truncate(5);

    let mut generated = assemble_code(prompt, &hits);

    serde_json::json!({
        "domain": domain,
        "generated_code": generated,
        "language": lang,
        "hits": hits.len(),
        "entropy": format!("{:.4}", state.omni.ai.cube.global_entropy()),
    })
    .to_string()
}

fn detect_lang_from_prompt(prompt: &str) -> &'static str {
    let p = prompt.to_lowercase();
    if p.contains("rust") || p.contains("cargo") || p.contains("impl") || p.contains("mut") {
        "rust"
    } else if p.contains("python") || p.contains("def ") || p.contains("import") {
        "python"
    } else if p.contains("go ") || p.contains("golang") || p.contains("goroutine") {
        "go"
    } else if p.contains("c++") || p.contains("cpp") || p.contains("template") {
        "cpp"
    } else if p.contains("c ") || p.contains("malloc") || p.contains("printf") {
        "c"
    } else if p.contains("typescript")
        || p.contains("ts ")
        || p.contains("react")
        || p.contains("angular")
    {
        "typescript"
    } else if p.contains("javascript") || p.contains("js ") || p.contains("node") {
        "javascript"
    } else {
        "rust"
    }
}

fn extract_fn_name(prompt: &str) -> String {
    let lower = prompt.to_lowercase();
    let stop = [
        "которая", "который", "которые", "которую", "что", "для", "код",
        "функцию", "функции", "функция", "функций", "rust", "python",
    ];
    for kw in ["функцию", "функции", "функция", "function"] {
        if let Some(pos) = lower.find(kw) {
            let rest = &prompt[pos + kw.len()..];
            for w in rest.split(|c: char| !c.is_alphanumeric() && c != '_') {
                let w = w.trim();
                if !w.is_empty() && !stop.contains(&w.to_lowercase().as_str()) {
                    return w.to_lowercase();
                }
            }
        }
    }
    "generated".to_string()
}

fn strip_code_prefix(mut s: &str) -> &str {
    if let Some(i) = s.find("Code: code_") {
        let rest = &s[i + "Code: code_".len()..];
        s = match rest.find(": ") {
            Some(j) => rest[j + 2..].trim_start(),
            None => rest,
        };
    }
    s
}

fn assemble_code(prompt: &str, hits: &[(f64, String)]) -> String {
    let lang = detect_lang_from_prompt(prompt);
    let mut lines: Vec<String> = Vec::new();
    for (_, text) in hits {
        for line in text.lines() {
            let l = strip_code_prefix(line.trim());
            if !l.is_empty() && l.len() > 8 && !lines.contains(&l.to_string()) {
                lines.push(l.to_string());
            }
        }
    }

    let fname = extract_fn_name(prompt);
    let mut out = match lang {
        "rust" => {
            let mut s = format!("fn {}() {{\n", fname);
            for l in &lines {
                if l.contains("fn ") || l.contains("struct ") || l.contains("impl ") {
                    s.push_str("    ");
                    s.push_str(l);
                    s.push('\n');
                }
            }
            s.push_str("}\n");
            s
        }
        "python" => {
            let mut s = format!("def {}():\n", fname);
            for l in &lines {
                if !l.starts_with("def ") && !l.starts_with("import ") {
                    s.push_str("    ");
                    s.push_str(l);
                    s.push('\n');
                }
            }
            s
        }
        "go" => {
            let mut s = format!("func {}() {{\n", fname);
            for l in &lines {
                if l.contains("func ") || l.contains("var ") || l.contains("type ") {
                    s.push_str("\t");
                    s.push_str(l);
                    s.push('\n');
                }
            }
            s.push_str("}\n");
            s
        }
        _ => lines.join("\n"),
    };
    out
}

fn take_lines(text: &str, n: usize) -> String {
    let mut seen = std::collections::HashSet::new();
    let parts: Vec<&str> = text
        .lines()
        .filter(|l| !l.contains("(sim=") && !l.starts_with("Answer") && !l.starts_with("Route"))
        .filter(|l| !l.trim().is_empty())
        .filter(|l| seen.insert(l.trim()))
        .take(n)
        .collect();
    parts.join(" ")
}

// Компактный ответ по обученной памяти: топ-записи по векторной близости,
// с обрезкой до читаемого размера и пометкой источника.
fn answer_from_memory<const N: usize, const S: usize>(
    ai: &mut FugaAI<N, S>,
    query: &str,
) -> String {
    let mut all: Vec<(f64, String, String)> = Vec::new();
    let direct = ai.memory.search_by_text(query, 12);
    for (_idx, sim, e) in &direct {
        all.push((*sim, e.source_doc.clone(), e.text.clone()));
    }
    if all.len() < 4 {
        let tokens: Vec<TokenInfo> = query
            .split_whitespace()
            .enumerate()
            .map(|(_, w)| TokenInfo {
                id: w.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32)),
                text: w.to_string(),
            })
            .collect();
        let output = ai.think(&tokens);
        for st in &output.super_tokens {
            let results = ai.memory.search(&st.vector, 8);
            for (_idx, sim, e) in &results {
                all.push((*sim, e.source_doc.clone(), e.text.clone()));
            }
        }
    }
    all.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let q = query.to_lowercase();
    let wants_code = q.contains("код")
        || q.contains("коди")
        || q.contains("function")
        || q.contains("fn ")
        || q.contains("generate")
        || q.contains("write ")
        || q.contains("напиши")
        || q.contains("сгенерируй")
        || q.contains("создай")
        || q.contains("реализуй")
        || q.contains("переведи")
        || q.contains("скрипт")
        || q.contains("import")
        || q.contains("class ")
        || q.contains("rust")
        || q.contains("tokio")
        || q.contains("hyper")
        || q.contains("python")
        || q.contains("http")
        || q.contains("async")
        || q.contains("api")
        || q.contains("функци");
    let mut seen = std::collections::HashSet::new();
    let mut chosen: Vec<(String, String)> = Vec::new();
    for (_sim, doc, text) in all {
        let t = text.trim();
        if t.len() < 20 {
            continue;
        }
        // Корпусные записи кода ("Code: code_<lang>: ...") не даём в ответ
        // на прозаический вопрос — иначе маск отвечает чужим дампом.
        if t.starts_with("Code: code_") && !wants_code {
            continue;
        }
        if !seen.insert(doc.clone()) {
            continue;
        }
        chosen.push((truncate_snippet(&text, 300), doc));
        if chosen.len() >= 3 {
            break;
        }
    }
    if chosen.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    for (i, (snippet, doc)) in chosen.iter().enumerate() {
        let label = source_label(doc);
        if i > 0 {
            out.push_str("\n\n");
        }
        out.push_str(&format!("[{}] {}", label, snippet));
    }
    out
}

fn truncate_snippet(text: &str, max: usize) -> String {
    let t = text.trim();
    if t.chars().count() <= max {
        return t.to_string();
    }
    let cut: String = t.chars().take(max).collect();
    format!("{}…", cut.trim_end())
}

fn source_label(doc: &str) -> &str {
    std::path::Path::new(doc)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(doc)
}

fn answer_from_flat<const N: usize, const S: usize>(ai: &mut FugaAI<N, S>, query: &str) -> String {
    let tokens: Vec<TokenInfo> = query
        .split_whitespace()
        .enumerate()
        .map(|(_, w)| TokenInfo {
            id: w
                .bytes()
                .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32)),
            text: w.to_string(),
        })
        .collect();
    let output = ai.think(&tokens);
    let answer = ai.answer_from_output(&output, query);
    if answer.contains("No knowledge stored") || answer.len() < 20 {
        String::new()
    } else {
        take_lines(&answer, 3)
    }
}

fn handle_openai_chat<const N: usize, const S: usize>(
    state: &mut WebState<N, S>,
    body: &str,
) -> String {
    let req: serde_json::Value = serde_json::from_str(body).unwrap_or(serde_json::json!({}));
    let model = req.get("model").and_then(|v| v.as_str()).unwrap_or("fuga-2.0");
    let stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let messages = req.get("messages").and_then(|v| v.as_array()).cloned();
    let temperature =
        req.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.6);
    let _max_tokens = req.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(512);

    // Собираем контекст: последнее user-сообщение + всё после прошлого tool-цикла,
    // чтобы агентный цикл (fenced `:::call` у Fuga) мог продолжать с результатами.
    let mut prompt_parts: Vec<String> = Vec::new();
    if let Some(msgs) = messages {
        for m in msgs {
            let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("");
            let mut content = m
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            if let Some(tool_calls) = m.get("tool_calls").and_then(|v| v.as_array()) {
                for tc in tool_calls {
                    if let Some(args) = tc.get("function").and_then(|f| f.get("arguments")) {
                        content.push_str(&format!("\n[want_tool] {}", args));
                    }
                }
            }
            if !content.trim().is_empty() {
                match role {
                    "user" | "system" | "tool" | "assistant" => {
                        prompt_parts.push(format!("[{}]\n{}", role, content))
                    }
                    _ => {}
                }
            }
        }
    }
    let query = if prompt_parts.is_empty() {
        "Hello".to_string()
    } else {
        prompt_parts.join("\n")
    };

    let chat_body = serde_json::json!({"message": query}).to_string();
    let _ = chat_body; // ответ строится полностью из handle_chat ниже
    let full = handle_chat::<N, S>(state, &query);
    let v: serde_json::Value = serde_json::from_str(&full).unwrap_or_default();

    // Fuga-ответ = свой контекст: конверсионный + omni + сгенерированный код.
    let conversation = v
        .get("conversational")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    let omni_text = v
        .get("omni_output")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    let generated_code = v
        .get("generated_code")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    let mut content = conversation;
    if !omni_text.trim().is_empty() {
        content.push_str("\n\n");
        content.push_str(&omni_text);
    }
    if !generated_code.trim().is_empty() {
        content.push_str("\n\n```\n");
        content.push_str(&generated_code);
        content.push_str("\n```\n");
    }

    let _ = temperature; // VSA-температура пока задаётся внутри handle_chat
    let id = format!("chatcmpl-fuga-{}", uuid_like());
    let usage = serde_json::json!({
        "prompt_tokens": query.split_whitespace().count(),
        "completion_tokens": content.split_whitespace().count(),
        "total_tokens": query.split_whitespace().count() + content.split_whitespace().count(),
    });
    let base = serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "created": crate::unix_now(),
        "model": model,
        "system_fingerprint": "fuga-self-jepa-vsa",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": usage,
    });
    if stream {
        let mut out = String::new();
        let chunk = serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": crate::unix_now(),
            "model": model,
            "system_fingerprint": "fuga-self-jepa-vsa",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": null}]
        });
        out.push_str("data: ");
        out.push_str(&chunk.to_string());
        out.push_str("\n\n");
        out.push_str(&serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": crate::unix_now(),
            "model": model,
            "system_fingerprint": "fuga-self-jepa-vsa",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }).to_string());
        out.push_str("\n\ndata: [DONE]\n\n");
        out
    } else {
        base.to_string()
    }
}

/// H-JEPA continuation tier: seed the self-mirror TemporalPredictor with the
/// query, roll out a latent sequence, and decode each latent to the nearest
/// vocabulary word (query words + words from retrieved memory). Returns "" when
/// the checkpoint is absent or the roll-out is too weak to decode anything.
fn answer_from_jepa<const N: usize, const S: usize>(
    state: &mut WebState<N, S>,
    query: &str,
) -> String {
    let Some(tp) = state.jepa.as_mut() else {
        return String::new();
    };

    let mut words: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|w| w.len() >= 2)
        .map(|w| w.to_lowercase())
        .collect();

    let hits = state.omni.ai.memory.search_by_text(query, 6);
    for (_score, _src, e) in &hits {
        for w in e.text.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-') {
            if w.len() >= 2 {
                words.push(w.to_lowercase());
            }
        }
    }
    if words.len() > 800 {
        words.truncate(800);
    }

    let vocab = tp.word_vocab(&words);
    let seq = tp.generate_words(query, 24, &vocab, 0.5);
    let text = seq.join(" ");
    let trimmed = text.trim();
    if trimmed.len() < 2 {
        String::new()
    } else {
        trimmed.to_string()
    }
}

fn handle_chat<const N: usize, const S: usize>(state: &mut WebState<N, S>, query: &str) -> String {
    let mut builder = TokenBuilder::new();
    let _ = builder.load_configs_from_dir("tikones");
    let flat_vocab = builder.build_flat_vocab();
    let tokens = tokenize_corpus_text(query, &flat_vocab);

    let domain = OmniEngine::<N, S>::detect_domain(query);
    let result = state.omni.query(query, &tokens);
    let omni_text = omni::render_omni_result(&result);

    let sys_vec: &[f64] = match &result.domain_result {
        omni::OmniDomainResult::Physics(a) => &a.system_vector,
        _ => &[],
    };

    // Для ЛЮБОГО домена сначала пробуем вытащить ответ из обученной памяти
    // (MoE → VSA-flat → цитатный retrieval → шаблон), чтобы маск отвечал
    // содержимым, а не шаблоном.
    let conv = {
        // C: when a gated lesson matches, the authoritative answer is its code
        // (emitted as generated_code). Suppress the corpus conversational dump
        // so the response leads with the real lesson, not raw-corpus noise.
        if find_lesson(state, query).is_some() {
            String::new()
        } else {
        // Цитатный retrieval по текстовому индексу (с бустом по имени файла)
        // — лучший источник; далее moe, flat и H-JEPA как фолбэки.
        let mem = answer_from_memory(&mut state.omni.ai, query);
        if !mem.is_empty() {
            mem
        } else {
            let moe_text = state.omni.ai.answer_from_moe(query);
            let answer = take_lines(&moe_text, 3);
            if !answer.is_empty() {
                answer
            } else {
                let flat = answer_from_flat(&mut state.omni.ai, query);
                if !flat.is_empty() {
                    flat
                } else {
                    let hjepa_text = answer_from_jepa(state, query);
                    if !hjepa_text.is_empty() {
                        hjepa_text
                    } else {
                        speech::conversational_reply(query, domain, sys_vec)
                    }
                }
            }
        }
        }
    };

    let sv = match &result.domain_result {
        omni::OmniDomainResult::Physics(a) => format!(
            "[{}]",
            a.system_vector
                .iter()
                .map(|v| format!("{:.4e}", v))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => "[]".to_string(),
    };

    let is_code_request = query.contains("write ")
        || query.contains("write a ")
        || query.contains("generate ")
        || query.contains("generate a ")
        || query.contains("create ")
        || query.contains("implement ")
        || query.contains("make ")
        || query.contains("code for ")
        || query.contains("sample")
        || query.contains("snippet")
        || query.contains("fn ")
        || query.contains("function")
        || query.contains("программа")
        || query.contains("программу")
        || query.contains("код ")
        || query.contains("функцию")
        || query.contains("функция")
        || query.contains("напиши")
        || query.contains("напишите")
        || query.contains("создай")
        || query.contains("создайте")
        || query.contains("сгенерируй")
        || query.contains("сгенерируйте")
        || query.contains("реализуй")
        || query.contains("реализуйте")
        || query.contains("скрипт")
        || query.contains("translate")
        || query.contains("переведи")
        || query.contains("переведи на")
        || query.contains("refactor")
        || query.contains("рефакторинг");

    let generated_code = if is_code_request {
        let body = serde_json::json!({"prompt": query}).to_string();
        let gend = handle_code_generate::<N, S>(state, &body);
        let v: serde_json::Value = serde_json::from_str(&gend).unwrap_or_default();
        v.get("generated_code")
            .and_then(|g| g.as_str())
            .unwrap_or("")
            .to_string()
    } else {
        String::new()
    };

    let code_analysis = detect_and_analyze_code::<N, S>(state, query);

    serde_json::json!({
        "domain": domain,
        "conversational": conv,
        "omni_output": omni_text,
        "system_vector": sv,
        "entropy": format!("{:.4}", result.entropy),
        "coherence": format!("{:.4}", result.coherence),
        "memory": result.memory_size,
        "generated_code": generated_code,
        "code_analysis": code_analysis,
    })
    .to_string()
}

fn detect_and_analyze_code<const N: usize, const S: usize>(
    state: &mut WebState<N, S>,
    query: &str,
) -> serde_json::Value {
    let extracted: Vec<(String, String)> = {
        let mut blocks = Vec::new();
        let mut in_block = false;
        let mut lang = String::new();
        let mut code = String::new();
        for line in query.lines() {
            if line.starts_with("```") {
                if in_block {
                    if !code.trim().is_empty() {
                        blocks.push((lang.clone(), code.trim().to_string()));
                    }
                    in_block = false;
                    lang.clear();
                    code.clear();
                } else {
                    in_block = true;
                    lang = line.trim_start_matches("```").trim().to_string();
                }
            } else if in_block {
                code.push_str(line);
                code.push('\n');
            }
        }
        if in_block && !code.trim().is_empty() {
            blocks.push((lang, code.trim().to_string()));
        }

        if blocks.is_empty() && query.contains('\n') {
            let trimmed = query.trim();
            let looks_like_code = trimmed.starts_with("fn ")
                || trimmed.starts_with("pub ")
                || trimmed.starts_with("use ")
                || trimmed.starts_with("impl ")
                || trimmed.starts_with("struct ")
                || trimmed.starts_with("enum ")
                || trimmed.starts_with("trait ")
                || trimmed.starts_with("#include")
                || trimmed.starts_with("import ")
                || trimmed.starts_with("package ")
                || trimmed.starts_with("func ")
                || trimmed.starts_with("def ")
                || trimmed.starts_with("class ")
                || trimmed.starts_with("const ")
                || trimmed.starts_with("let ")
                || trimmed.starts_with("int ")
                || trimmed.starts_with("void ")
                || trimmed.starts_with("std::");
            if looks_like_code {
                let lang = if query.contains("fn ")
                    || query.contains("impl ")
                    || query.contains("let mut")
                {
                    "rust"
                } else if query.contains("#include") || query.contains("int main") {
                    "c"
                } else if query.contains("def ") || query.contains("import ") {
                    "python"
                } else if query.contains("func ") || query.contains("package ") {
                    "go"
                } else {
                    "rust"
                };
                blocks.push((lang.to_string(), trimmed.to_string()));
            }
        }
        blocks
    };

    if extracted.is_empty() {
        return serde_json::Value::Null;
    }

    let (lang_name, code) = &extracted[0];
    let path = format!(
        "chat_code.{}",
        match lang_name.as_str() {
            "rust" | "rs" => "rs",
            "c" => "c",
            "cpp" | "c++" => "cpp",
            "go" => "go",
            "python" | "py" => "py",
            "ts" | "typescript" => "ts",
            "js" => "js",
            _ => "rs",
        }
    );

    let lang = match lang_name.as_str() {
        "rust" | "rs" => LanguageId::Rust,
        "c" => LanguageId::C,
        "cpp" | "c++" => LanguageId::Cpp,
        "go" => LanguageId::Go,
        "python" | "py" => LanguageId::Python,
        "ts" | "typescript" => LanguageId::TypeScript,
        "js" | "javascript" => LanguageId::JavaScript,
        _ => LanguageId::Rust,
    };

    let qual_result = match state.qual.analyze(code, lang, &path) {
        Ok(s) => s,
        Err(e) => return serde_json::json!({"error": e}),
    };

    let mw_out = if lang == LanguageId::Rust || lang_name == "rust" {
        parse_mw_output(&run_microwave(
            "microwave_sandbox/target/release/mini-fuga",
            "eval-rust-file",
            code,
        ))
    } else if lang == LanguageId::Cpp || lang == LanguageId::C {
        parse_mw_output(&run_microwave(
            "microwave_sandbox/target/release/mini-fuga",
            "eval-cpp-file",
            code,
        ))
    } else {
        serde_json::Value::Null
    };

    let bugs_text = if qual_result.bugs_detected {
        "⚠️ BUGS DETECTED"
    } else {
        "✓ no bugs"
    };
    let violations_text = if qual_result.violations > 0 {
        format!("⚠ {} violations", qual_result.violations)
    } else {
        "✓ no violations".to_string()
    };

    serde_json::json!({
        "language": lang_name,
        "weight": format!("{:.2}", qual_result.weight),
        "safety": format!("{:.2}", qual_result.safety),
        "violations": qual_result.violations,
        "bugs": qual_result.bugs_detected,
        "bugs_text": bugs_text,
        "violations_text": violations_text,
        "summary": qual_result.summary,
        "microwave": mw_out,
    })
}

fn handle_speak<const N: usize, const S: usize>(state: &WebState<N, S>, text: &str) -> Vec<u8> {
    state.speech.speak(text)
}

fn run_server<const N: usize, const S: usize>(cube_path: &str, port: u16) {
    println!("╔══════════════════════════════════════════════════╗");
    println!("║  Fuga Omni + Code Analysis Web Server          ║");
    println!("╚══════════════════════════════════════════════════╝");

    let cube = WaveCube::<N, S>::load_bin(cube_path)
        .expect(&format!("Failed to load cube: {}", cube_path));
    let cube_dim = cube.dim;
    println!("  Cube: {}^{} dim={}", S, N, cube_dim);

    let mut omni = OmniEngine::<N, S>::new(cube_dim, 3);
    omni.ai.cube = cube;
    let mem_path = cube_path.replace(".bin", "_mem.bin");
    if let Ok(mem) = fuga::MemoryStore::load_bin(&mem_path) {
        omni.ai.memory = mem;
    }
    println!("  Memory: {} entries", omni.ai.memory.size());

    // Re-inject durable gated lessons into live memory so self-improvement
    // survives a restart (lessons are the ONLY store tagged agent_lesson_*).
    load_lessons_into_memory::<N, S>(&mut omni.ai, "agent_lessons.jsonl");

    // Init the multi-engine so the code-domain query renders a real status
    // instead of leaking "No multi-engine initialized..." into the answer.
    omni = omni.with_multi(cube_dim);

    omni.ai.moe = fuga::MoEStore::new(cube_path);
    match omni.ai.moe.load_all() {
        Ok(()) => {
            println!("  MoE loaded:");
            for (domain, size) in &omni.ai.moe.domain_sizes() {
                println!("    {:20}  {}", domain, size);
            }
        }
        Err(e) => println!("  MoE warning: {}", e),
    }

    let qual = CodeQualityFilter::new(cube_dim);

    // H-JEPA sequential generator: предпочитаем корпус-обученную пару
    // (fuga_stack_tm.bin + fuga_hjepa.bin), иначе self-mirror чекпоинт.
    let jepa = {
        let candidates = [
            ("fuga_stack_tm.bin", "fuga_hjepa.bin"),
            ("fuga_mirror_tm.bin", "fuga_mirror_jepa.bin"),
        ];
        let mut loaded = None;
        for (tm_path, jepa_path) in candidates {
            if std::path::Path::new(tm_path).exists() && std::path::Path::new(jepa_path).exists() {
                match (
                    fuga::TemporalMemory::load(tm_path),
                    fuga::HierarchicalJEPA::load(jepa_path),
                ) {
                    (Some(tm), Ok(hj)) => {
                        println!("  H-JEPA: loaded {} / {}", tm_path, jepa_path);
                        loaded = Some(fuga::TemporalPredictor::new(tm, hj));
                        break;
                    }
                    _ => {}
                }
            }
        }
        match loaded {
            Some(tp) => Some(tp),
            None => {
                println!("  H-JEPA: skipped (no checkpoints)");
                None
            }
        }
    };

    let state = Arc::new(Mutex::new(WebState {
        omni,
        speech: FugaText::new(),
        qual,
        dim: cube_dim,
        jepa,
        lessons: parse_lessons("agent_lessons.jsonl"),
    }));

    let server =
        Server::http(&format!("0.0.0.0:{}", port)).expect(&format!("Failed to bind port {}", port));
    let ip = std::net::TcpStream::connect("8.8.8.8:53")
        .ok()
        .and_then(|s| s.local_addr().ok())
        .map(|a| a.ip().to_string())
        .unwrap_or_else(|| "0.0.0.0".to_string());
    println!("  LAN: http://{}:{}", ip, port);

    if let Ok(url) = std::fs::read_to_string("tunnel_url.txt") {
        let url = url.trim();
        if !url.is_empty() {
            println!("  🌐 Public: {}", url);
        }
    }
    println!();

    for mut request in server.incoming_requests() {
        let url = request.url().to_string();
        let method = request.method().as_str().to_string();

        let response = match (method.as_str(), url.as_str()) {
            ("GET", "/") | ("GET", "/index.html") => Response::from_string(SITE_HTML).with_header(
                Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap(),
            ),
            ("GET", "/api/stats") => {
                let st = state.lock().unwrap();
                let moe_total = st.omni.ai.moe.total_size();
                let domains: Vec<serde_json::Value> = st
                    .omni
                    .ai
                    .moe
                    .domain_sizes()
                    .iter()
                    .map(|(d, s)| serde_json::json!({"name": d, "size": s}))
                    .collect();
                let j = serde_json::json!({
                    "entropy": format!("{:.4}", st.omni.ai.cube.global_entropy()),
                    "coherence": format!("{:.4}", st.omni.ai.cube.coherence()),
                    "memory": st.omni.ai.memory.size(),
                    "moe_total": moe_total,
                    "dim": st.omni.ai.cube.dim,
                    "cube_side": S,
                    "ndim": N,
                    "domains": domains,
                });
                Response::from_string(j.to_string()).with_header(
                    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
                )
            }
            ("POST", "/api/readout") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap_or(0);
                let query: serde_json::Value =
                    serde_json::from_str(&body).unwrap_or(serde_json::json!({"query": ""}));
                let text = query.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let beam = query
                    .get("beam")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(12)
                    .min(40) as usize;
                let resp = {
                    let mut st = state.lock().unwrap();
                    if text.is_empty() {
                        serde_json::json!({"concepts": [], "thought_cells": 0, "entropy": ""})
                    } else {
                        let out = fuga::logit_lens::<N, S>(&mut st.omni.ai, text, beam, 6);
                        serde_json::json!({
                            "query": text,
                            "concepts": out.concepts.iter().map(|(t, s)| serde_json::json!({"word": t, "sim": format!("{:.4}", s)})).collect::<Vec<_>>(),
                            "thought_cells": out.thought_cells,
                            "entropy": format!("{:.4}", out.entropy),
                        })
                    }
                };
                Response::from_string(resp.to_string()).with_header(
                    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
                )
            }
            ("POST", "/api/chat") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap_or(0);
                let query: serde_json::Value =
                    serde_json::from_str(&body).unwrap_or(serde_json::json!({"message": ""}));
                let text = query
                    .get("message")
                    .or_else(|| query.get("query"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let reply = {
                    let mut st = state.lock().unwrap();
                    handle_chat::<N, S>(&mut st, text)
                };
                let resp = serde_json::json!({"reply": reply});
                Response::from_string(resp.to_string()).with_header(
                    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
                )
            }
            ("POST", "/api/retrieve") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap_or(0);
                let query: serde_json::Value =
                    serde_json::from_str(&body).unwrap_or(serde_json::json!({"query": ""}));
                let text = query.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let k = query
                    .get("top_k")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(8)
                    .min(25) as usize;
                let st = state.lock().unwrap();
                let results: Vec<serde_json::Value> = if text.is_empty() {
                    vec![]
                } else {
                    st.omni
                        .ai
                        .memory
                        .search_by_text(text, k)
                        .into_iter()
                        .map(|(_, s, e)| {
                            serde_json::json!({
                                "score": format!("{:.4}", s),
                                "source": e.source_doc,
                                "text": e.text.chars().take(1200).collect::<String>(),
                            })
                        })
                        .collect()
                };
                let resp = serde_json::json!({"query": text, "results": results});
                Response::from_string(resp.to_string()).with_header(
                    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
                )
            }
            ("GET", "/v1/models") => {
                let models = serde_json::json!({
                    "object": "list",
                    "data": [
                        {
                            "id": "fuga-2.0",
                            "object": "model",
                            "created": 0,
                            "owned_by": "fuga",
                            "permission": [],
                            "description": "Fuga self-powered brain (JEPA/VSA) — NOT an external LLM. Local inference, own learned model.",
                            "system_fingerprint": "fuga-self-jepa-vsa",
                        }
                    ]
                });
                Response::from_string(models.to_string()).with_header(
                    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
                )
            }
            ("POST", "/v1/chat/completions") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap_or(0);
                let resp = {
                    let mut st = state.lock().unwrap();
                    handle_openai_chat::<N, S>(&mut st, &body)
                };
                Response::from_string(resp).with_header(
                    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
                )
            }
            ("POST", "/v1/fuga/lesson") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap_or(0);
                let req: serde_json::Value = serde_json::from_str(&body).unwrap_or_default();
                let task = req.get("task").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let code = req.get("code").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let resp = if code.is_empty() {
                    serde_json::json!({"ok": false, "error": "empty code"}).to_string()
                } else {
                    let mut st = state.lock().unwrap();
                    // C: register in the explicit lesson tier (checked before
                    // corpus retrieval) + keep the live-memory copy as a bonus.
                    st.lessons.push(AgentLesson { task: task.clone(), code: code.clone() });
                    store_lesson_live::<N, S>(&mut st.omni.ai, &task, &code).to_string()
                };
                Response::from_string(resp).with_header(
                    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
                )
            }
            ("POST", "/api/speak") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap_or(0);
                let query: serde_json::Value =
                    serde_json::from_str(&body).unwrap_or(serde_json::json!({"text": ""}));
                let text = query.get("text").and_then(|v| v.as_str()).unwrap_or("");
                let wav = {
                    let st = state.lock().unwrap();
                    handle_speak::<N, S>(&st, text)
                };
                Response::from_data(wav).with_header(
                    Header::from_bytes(&b"Content-Type"[..], &b"audio/wav"[..]).unwrap(),
                )
            }
            ("POST", "/api/code-analyze") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap_or(0);
                let result = {
                    let mut st = state.lock().unwrap();
                    handle_code_analyze::<N, S>(&mut st, &body)
                };
                Response::from_string(result).with_header(
                    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
                )
            }
            ("POST", "/api/code-fix") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap_or(0);
                let result = {
                    let mut st = state.lock().unwrap();
                    handle_code_fix::<N, S>(&mut st, &body)
                };
                Response::from_string(result).with_header(
                    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
                )
            }
            ("POST", "/api/code-generate") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap_or(0);
                let result = {
                    let mut st = state.lock().unwrap();
                    handle_code_generate::<N, S>(&mut st, &body)
                };
                Response::from_string(result).with_header(
                    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap(),
                )
            }
            _ => Response::from_string("404 Not Found").with_status_code(404),
        };

        if let Err(e) = request.respond(response) {
            eprintln!("Response error: {}", e);
        }
    }
}

fn main() {
    let cube_path = env::var("FUGA_CUBE_PATH").unwrap_or_else(|_| "fuga_stack.bin".into());
    let port: u16 = env::var("FUGA_WEB_PORT")
        .unwrap_or_else(|_| "8080".into())
        .parse()
        .unwrap_or(8080);

    let (ndim, side_len, _dim) = match peek_cube_header(&cube_path) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };
    match (ndim, side_len) {
        (4, 4) => run_server::<4, 4>(&cube_path, port),
        (3, 4) => run_server::<3, 4>(&cube_path, port),
        (3, 8) => run_server::<3, 8>(&cube_path, port),
        (4, 8) => run_server::<4, 8>(&cube_path, port),
        (5, 2) => run_server::<5, 2>(&cube_path, port),
        (5, 4) => run_server::<5, 4>(&cube_path, port),
        _ => eprintln!("Unsupported cube dimensions: {}×{}", side_len, ndim),
    }
}

fn unix_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn uuid_like() -> String {
    use fastrand;
    let mut h = String::with_capacity(24);
    let hex = b"0123456789abcdef";
    for i in 0..24 {
        h.push(hex[fastrand::usize(..16)] as char);
    }
    h
}

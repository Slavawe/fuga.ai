use fuga::{
    WaveCube, TokenBuilder,
    ai::FugaAI,
    omni::{self, OmniEngine},
    speech::{self, FugaText},
    tokenize_corpus_text,
    core::wave_cube::peek_cube_header,
    CodeQualityFilter, FugaEngine, MultiEngine, MultiFixGenerator,
    LanguageId, FixProposal, TokenInfo,
};
use tiny_http::{Server, Response, Header};
use std::sync::{Arc, Mutex};
use std::env;

const SITE_HTML: &str = include_str!("site.html");

struct WebState<const N: usize, const S: usize> {
    omni: OmniEngine<N, S>,
    speech: FugaText,
    qual: CodeQualityFilter,
    dim: usize,
}

fn run_microwave(mw_path: &str, mode: &str, code: &str) -> String {
    let tmp = std::env::temp_dir().join("mw_eval.rs");
    let _ = std::fs::write(&tmp, code);
    let output = std::process::Command::new(mw_path)
        .arg(mode).arg(&tmp)
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

fn handle_code_analyze<const N: usize, const S: usize>(state: &mut WebState<N, S>, body: &str) -> String {
    let req: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let code = req.get("code").and_then(|v| v.as_str()).unwrap_or("");
    let lang_name = req.get("language").and_then(|v| v.as_str()).unwrap_or("rust");
    let path = req.get("path").and_then(|v| v.as_str()).unwrap_or("web_input.rs");

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
                parse_mw_output(&run_microwave("microwave_sandbox/target/release/mini-fuga", "eval-rust-file", code))
            } else if lang == LanguageId::Cpp || lang == LanguageId::C {
                parse_mw_output(&run_microwave("microwave_sandbox/target/release/mini-fuga", "eval-cpp-file", code))
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
            }).to_string()
        }
        Err(e) => serde_json::json!({"error": e}).to_string(),
    }
}

fn handle_code_fix<const N: usize, const S: usize>(state: &mut WebState<N, S>, body: &str) -> String {
    let req: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let code = req.get("code").and_then(|v| v.as_str()).unwrap_or("");
    let lang_name = req.get("language").and_then(|v| v.as_str()).unwrap_or("rust");
    let path = req.get("path").and_then(|v| v.as_str()).unwrap_or("web_input.rs");

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
        }).to_string();
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
        "microwave_sandbox/target/release/mini-fuga", "eval-rust-file", code));
    let mw_fixed = parse_mw_output(&run_microwave(
        "microwave_sandbox/target/release/mini-fuga", "eval-rust-file", &fixed));

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
    }).to_string()
}

fn handle_code_generate<const N: usize, const S: usize>(state: &mut WebState<N, S>, body: &str) -> String {
    let req: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let prompt = req.get("prompt").and_then(|v| v.as_str()).unwrap_or("");

    let mut builder = TokenBuilder::new();
    let _ = builder.load_configs_from_dir("tikones");
    let flat_vocab = builder.build_flat_vocab();
    let tokens = tokenize_corpus_text(prompt, &flat_vocab);

    let domain = OmniEngine::<N, S>::detect_domain(prompt);
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
    let lang = detect_lang_from_prompt(prompt);

    serde_json::json!({
        "domain": domain,
        "generated_code": generated,
        "language": lang,
        "hits": hits.len(),
        "entropy": format!("{:.4}", state.omni.ai.cube.global_entropy()),
    }).to_string()
}

fn detect_lang_from_prompt(prompt: &str) -> &'static str {
    let p = prompt.to_lowercase();
    if p.contains("rust") || p.contains("cargo") || p.contains("impl") || p.contains("mut") { "rust" }
    else if p.contains("python") || p.contains("def ") || p.contains("import") { "python" }
    else if p.contains("go ") || p.contains("golang") || p.contains("goroutine") { "go" }
    else if p.contains("c++") || p.contains("cpp") || p.contains("template") { "cpp" }
    else if p.contains("c ") || p.contains("malloc") || p.contains("printf") { "c" }
    else if p.contains("typescript") || p.contains("ts ") || p.contains("react") || p.contains("angular") { "typescript" }
    else if p.contains("javascript") || p.contains("js ") || p.contains("node") { "javascript" }
    else { "rust" }
}

fn assemble_code(prompt: &str, hits: &[(f64, String)]) -> String {
    let lang = detect_lang_from_prompt(prompt);
    let mut lines: Vec<String> = Vec::new();
    for (_, text) in hits {
        for line in text.lines() {
            let l = line.trim();
            if !l.is_empty() && l.len() > 8 && !lines.contains(&l.to_string()) {
                lines.push(l.to_string());
            }
        }
    }

    let mut out = match lang {
        "rust" => {
            let mut s = String::from("fn generated() {\n");
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
            let mut s = String::from("def generated():\n");
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
            let mut s = String::from("func Generated() {\n");
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
    let parts: Vec<&str> = text.lines()
        .filter(|l| !l.contains("(sim=") && !l.starts_with("Answer") && !l.starts_with("Route"))
        .filter(|l| !l.trim().is_empty())
        .filter(|l| seen.insert(l.trim()))
        .take(n)
        .collect();
    parts.join(" ")
}

fn answer_from_flat<const N: usize, const S: usize>(ai: &mut FugaAI<N, S>, query: &str) -> String {
    let tokens: Vec<TokenInfo> = query.split_whitespace().enumerate().map(|(_, w)| TokenInfo {
        id: w.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32)),
        text: w.to_string(),
    }).collect();
    let output = ai.think(&tokens);
    let answer = ai.answer_from_output(&output, query);
    if answer.contains("No knowledge stored") || answer.len() < 20 {
        String::new()
    } else {
        take_lines(&answer, 3)
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

    let conv = if domain == "general" || domain == "text" || domain == "dialogue" || domain == "narrative" {
        let moe_text = state.omni.ai.answer_from_moe(query);
        let answer = take_lines(&moe_text, 3);
        if !answer.is_empty() {
            answer
        } else {
            let flat = answer_from_flat(&mut state.omni.ai, query);
            if !flat.is_empty() { flat } else { speech::conversational_reply(query, domain, &[]) }
        }
    } else {
        speech::conversational_reply(query, domain, match &result.domain_result {
            omni::OmniDomainResult::Physics(a) => &a.system_vector,
            _ => &[],
        })
    };

    let sv = match &result.domain_result {
        omni::OmniDomainResult::Physics(a) => format!("[{}]", a.system_vector.iter()
            .map(|v| format!("{:.4e}", v)).collect::<Vec<_>>().join(", ")),
        _ => "[]".to_string(),
    };

    let is_code_request = query.contains("write ") || query.contains("generate ")
        || query.contains("create ") || query.contains("implement ")
        || query.contains("make ") || query.contains("code for ")
        || query.contains("fn ") || query.contains("function")
        || query.contains("программа") || query.contains("код ") || query.contains("функция")
        || query.contains("напиши") || query.contains("создай");

    let generated_code = if is_code_request {
        let body = serde_json::json!({"prompt": query}).to_string();
        let gend = handle_code_generate::<N, S>(state, &body);
        let v: serde_json::Value = serde_json::from_str(&gend).unwrap_or_default();
        v.get("generated_code").and_then(|g| g.as_str()).unwrap_or("").to_string()
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
    }).to_string()
}

fn detect_and_analyze_code<const N: usize, const S: usize>(state: &mut WebState<N, S>, query: &str) -> serde_json::Value {
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
                let lang = if query.contains("fn ") || query.contains("impl ") || query.contains("let mut") { "rust" }
                    else if query.contains("#include") || query.contains("int main") { "c" }
                    else if query.contains("def ") || query.contains("import ") { "python" }
                    else if query.contains("func ") || query.contains("package ") { "go" }
                    else { "rust" };
                blocks.push((lang.to_string(), trimmed.to_string()));
            }
        }
        blocks
    };

    if extracted.is_empty() {
        return serde_json::Value::Null;
    }

    let (lang_name, code) = &extracted[0];
    let path = format!("chat_code.{}", match lang_name.as_str() {
        "rust" | "rs" => "rs",
        "c" => "c",
        "cpp" | "c++" => "cpp",
        "go" => "go",
        "python" | "py" => "py",
        "ts" | "typescript" => "ts",
        "js" => "js",
        _ => "rs",
    });

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
        parse_mw_output(&run_microwave("microwave_sandbox/target/release/mini-fuga", "eval-rust-file", code))
    } else if lang == LanguageId::Cpp || lang == LanguageId::C {
        parse_mw_output(&run_microwave("microwave_sandbox/target/release/mini-fuga", "eval-cpp-file", code))
    } else {
        serde_json::Value::Null
    };

    let bugs_text = if qual_result.bugs_detected { "⚠️ BUGS DETECTED" } else { "✓ no bugs" };
    let violations_text = if qual_result.violations > 0 { format!("⚠ {} violations", qual_result.violations) } else { "✓ no violations".to_string() };

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

    let state = Arc::new(Mutex::new(WebState {
        omni,
        speech: FugaText::new(),
        qual,
        dim: cube_dim,
    }));

    let server = Server::http(&format!("0.0.0.0:{}", port))
        .expect(&format!("Failed to bind port {}", port));
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
            ("GET", "/") | ("GET", "/index.html") => {
                Response::from_string(SITE_HTML)
                    .with_header(Header::from_bytes(&b"Content-Type"[..], &b"text/html; charset=utf-8"[..]).unwrap())
            }
            ("GET", "/api/stats") => {
                let st = state.lock().unwrap();
                let moe_total = st.omni.ai.moe.total_size();
                let domains: Vec<serde_json::Value> = st.omni.ai.moe.domain_sizes().iter()
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
                Response::from_string(j.to_string())
                    .with_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap())
            }
            ("POST", "/api/chat") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap_or(0);
                let query: serde_json::Value = serde_json::from_str(&body).unwrap_or(serde_json::json!({"message": ""}));
                let text = query.get("message").or_else(|| query.get("query")).and_then(|v| v.as_str()).unwrap_or("");
                let reply = {
                    let mut st = state.lock().unwrap();
                    handle_chat::<N, S>(&mut st, text)
                };
                let resp = serde_json::json!({"reply": reply});
                Response::from_string(resp.to_string())
                    .with_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap())
            }
            ("POST", "/api/speak") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap_or(0);
                let query: serde_json::Value = serde_json::from_str(&body).unwrap_or(serde_json::json!({"text": ""}));
                let text = query.get("text").and_then(|v| v.as_str()).unwrap_or("");
                let wav = {
                    let st = state.lock().unwrap();
                    handle_speak::<N, S>(&st, text)
                };
                Response::from_data(wav)
                    .with_header(Header::from_bytes(&b"Content-Type"[..], &b"audio/wav"[..]).unwrap())
            }
            ("POST", "/api/code-analyze") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap_or(0);
                let result = {
                    let mut st = state.lock().unwrap();
                    handle_code_analyze::<N, S>(&mut st, &body)
                };
                Response::from_string(result)
                    .with_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap())
            }
            ("POST", "/api/code-fix") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap_or(0);
                let result = {
                    let mut st = state.lock().unwrap();
                    handle_code_fix::<N, S>(&mut st, &body)
                };
                Response::from_string(result)
                    .with_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap())
            }
            ("POST", "/api/code-generate") => {
                let mut body = String::new();
                request.as_reader().read_to_string(&mut body).unwrap_or(0);
                let result = {
                    let mut st = state.lock().unwrap();
                    handle_code_generate::<N, S>(&mut st, &body)
                };
                Response::from_string(result)
                    .with_header(Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap())
            }
            _ => {
                Response::from_string("404 Not Found").with_status_code(404)
            }
        };

        if let Err(e) = request.respond(response) {
            eprintln!("Response error: {}", e);
        }
    }
}

fn main() {
    let cube_path = env::var("FUGA_CUBE_PATH").unwrap_or_else(|_| "fuga_code_cube.bin".into());
    let port: u16 = env::var("FUGA_WEB_PORT").unwrap_or_else(|_| "8080".into()).parse().unwrap_or(8080);

    let (ndim, side_len, _dim) = match peek_cube_header(&cube_path) {
        Ok(h) => h,
        Err(e) => { eprintln!("{}", e); std::process::exit(1); }
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

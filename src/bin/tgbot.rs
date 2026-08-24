use fuga::{
    FugaAI, HierarchicalJEPA, Hypervector, JepaPredictor, MemoryStore, MoEStore, PromptVectors,
    TokenInfo, WaveCube, core::wave_cube::peek_cube_header, speech::FugaText, weaver::token_id,
};
use rand;
use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::hash::{Hash, Hasher};
use ureq;

const TELEGRAM_API: &str = "https://api.telegram.org/bot";
const JEPA_PATH: &str = "fuga_jepa.bin";
const HJEPA_PATH: &str = "fuga_hjepa.bin";

fn is_training_artifact(text: &str) -> bool {
    text.contains("[dialogue-pair]")
        || text.contains("[narrative]")
        || text.contains("[code]")
        || text.contains("ctx:")
        || text.contains("resp:")
        || text.contains("| → |")
        || text.starts_with("TASK:")
        || text.starts_with("FILE:")
        || text.starts_with("DIFF:")
        || text.starts_with("STATUS:")
        || text.contains("source_doc:")
        || text.starts_with("// /")
}

fn is_greeting(q: &str) -> bool {
    let g = q.to_lowercase();
    g == "привет"
        || g == "hello"
        || g == "hi"
        || g == "здравствуй"
        || g == "здравствуйте"
        || g == "даров"
        || g == "ку"
        || g.starts_with("привет ")
        || g.starts_with("hello ")
        || g.starts_with("hi ")
}

fn looks_like_code_request(text: &str) -> bool {
    let t = text.to_lowercase();
    t.len() >= 5
        && (t.contains("python")
            || t.contains("rust")
            || t.contains("javascript")
            || t.contains("js")
            || t.contains("go ")
            || t.contains("golang")
            || t.contains("c++")
            || t.contains("cpp")
            || t.contains("java")
            || t.contains("typescript")
            || t.contains("ts ")
            || t.contains("функция")
            || t.contains("класс")
            || t.contains("метод")
            || t.contains("code")
            || t.contains("program")
            || t.contains("script")
            || t.contains("напиши")
            || t.contains("создай")
            || t.contains("написать")
            || t.contains("файл")
            || t.contains("алгоритм")
            || t.contains("сортировк")
            || t.contains("server")
            || t.contains("api")
            || t.contains("endpoint")
            || t.contains("bot")
            || t.contains("функц")
            || t.contains("парс")
            || t.contains("regex")
            || t.contains("sql")
            || t.contains("запрос")
            || t.contains("db")
            || t.contains("баз")
            || t.contains("html")
            || t.contains("css")
            || t.contains("стил")
            || t.contains("апи")
            || t.contains("код"))
}

fn is_binary_question(text: &str) -> bool {
    let t = text.to_lowercase();
    t.contains("да или нет")
        || t.contains("yes or no")
        || t.contains("скажи да")
        || t.contains("ответь прямо")
        || t.contains("просто скажи")
        || t.contains("только да")
        || t.contains("только нет")
        || t == "да"
        || t == "нет"
        || t == "yes"
        || t == "no"
}

fn need_context_persistence(query: &str) -> bool {
    query.trim().len() < 15 && !query.starts_with('/')
}

fn guess_language(text: &str) -> &'static str {
    let t = text.to_lowercase();
    if t.contains("fn ")
        || t.contains("impl ")
        || t.contains("let mut ")
        || t.contains(".rs")
        || t.contains("pub ")
        || t.contains("-> ")
    {
        "rust"
    } else if t.contains("def ")
        || t.contains("import ")
        || t.contains("print(")
        || t.contains("class ")
        || t.contains("self")
        || t.contains(".py")
    {
        "python"
    } else if t.contains("function")
        || t.contains("const ")
        || t.contains("let ")
        || t.contains("=>")
        || t.contains("export")
        || t.contains(".js")
    {
        "javascript"
    } else if t.contains("go ") || t.contains("func ") || t.contains(".go") {
        "go"
    } else if t.contains("int ")
        || t.contains("void ")
        || t.contains("printf")
        || t.contains("std::")
        || t.contains("#include")
    {
        "cpp"
    } else {
        ""
    }
}

fn matches_language(query: &str, entry_text: &str) -> bool {
    let q = query.to_lowercase();
    let want_rust = q.contains("rust") || q.contains(".rs");
    let want_py = q.contains("python") || q.contains(".py") || q.contains("питон");
    let want_js = q.contains("javascript") || q.contains(".js") || q.contains("js");
    let want_go = q.contains("go") || q.contains("golang");
    let want_cpp = q.contains("c++") || q.contains("cpp");

    if !want_rust && !want_py && !want_js && !want_go && !want_cpp {
        return true;
    }

    let lang = guess_language(entry_text);
    if want_rust && lang == "rust" {
        return true;
    }
    if want_py && lang == "python" {
        return true;
    }
    if want_js && lang == "javascript" {
        return true;
    }
    if want_go && lang == "go" {
        return true;
    }
    if want_cpp && lang == "cpp" {
        return true;
    }

    false
}

struct BabyContext {
    text: String,
    vecs: Vec<Hypervector>,
}

struct TgBot<const N: usize, const S: usize> {
    token: String,
    offset: i64,
    ai: FugaAI<N, S>,
    speech: FugaText,
    cube_path: String,
    mem_path: String,
    moe: MoEStore,
    prompts: PromptVectors,
    jepa: Option<JepaPredictor>,
    hjepa: Option<HierarchicalJEPA>,
    last_query: String,
    contexts: HashMap<i64, BabyContext>,
}

impl<const N: usize, const S: usize> TgBot<N, S> {
    fn new(token: String, cube_path: String, mem_path: String, dim: usize) -> Result<Self, String> {
        let cube = WaveCube::<N, S>::load_bin(&cube_path).map_err(|e| format!("Cube: {}", e))?;
        let memory = MemoryStore::load_bin(&mem_path).map_err(|e| format!("Memory: {}", e))?;

        let cube_dim = cube.dim;

        let mut ai = FugaAI::<N, S>::new(cube_dim, 3);
        ai.cube = cube;
        ai.memory = memory;

        let speech = FugaText::new();

        let mut moe = MoEStore::new("fuga_code_cube");
        let _ = moe.load_all();

        let prompts = PromptVectors::new(dim);

        let moe_total = moe.total_size();
        println!(
            "Bot: {}^{} dim={}, mem={}, moe={}",
            S,
            N,
            cube_dim,
            ai.memory.size(),
            moe_total
        );
        for (dom, sz) in moe.domain_sizes() {
            println!("  {}: {}", dom, sz);
        }

        let jepa = if std::path::Path::new(JEPA_PATH).exists() {
            match JepaPredictor::load(JEPA_PATH) {
                Ok(j) => {
                    println!("JEPA loaded: ctx_len={}, dim={}", j.context_len, j.dim);
                    Some(j)
                }
                Err(e) => {
                    println!("JEPA load failed: {} (will use bundle)", e);
                    None
                }
            }
        } else {
            println!("JEPA: no weights file ({}), using bundle blend", JEPA_PATH);
            None
        };

        let hjepa = if std::path::Path::new(HJEPA_PATH).exists() {
            match HierarchicalJEPA::load(HJEPA_PATH) {
                Ok(h) => {
                    println!("H-JEPA loaded: dim={}, levels={}", h.dim, h.levels.len());
                    Some(h)
                }
                Err(e) => {
                    println!("H-JEPA load failed: {} (baby mode unavailable)", e);
                    None
                }
            }
        } else {
            println!("H-JEPA: no model ({}), baby mode unavailable", HJEPA_PATH);
            None
        };

        Ok(Self {
            token,
            offset: 0,
            ai,
            speech,
            cube_path,
            mem_path,
            moe,
            prompts,
            jepa,
            hjepa,
            last_query: String::new(),
            contexts: HashMap::new(),
        })
    }

    fn api_url(&self, method: &str) -> String {
        format!("{}{}/{}", TELEGRAM_API, self.token, method)
    }

    /// Маскирует токен в тексте ошибки: ureq::Error содержит полный URL
    /// api.telegram.org/bot<TOKEN>/method — без маски токен утекает в
    /// stderr/system journal (аудит 22.08).
    fn mask(&self, e: &dyn std::fmt::Display) -> String {
        let s = e.to_string();
        if self.token.len() > 8 {
            s.replace(&self.token, "***")
        } else {
            s
        }
    }

    fn get_updates(&mut self) -> Result<serde_json::Value, String> {
        let url = format!(
            "{}?offset={}&timeout=30&limit=100",
            self.api_url("getUpdates"),
            self.offset + 1
        );
        let resp = ureq::get(&url).call().map_err(|e| format!("HTTP: {}", self.mask(&e)))?;
        resp.into_json().map_err(|e| format!("JSON: {}", e))
    }

    fn send_message(&self, chat_id: i64, text: &str) -> Result<(), String> {
        let safe_text = if text.len() > 4000 {
            // Срез по границе символа: байтовый индекс 3990 может попасть
            // в середину многобайтового UTF-8 (кириллица/эмодзи) → паника.
            let mut cut = 3990.min(text.len());
            while cut > 0 && !text.is_char_boundary(cut) {
                cut -= 1;
            }
            format!("{}...\n\n[truncated]", &text[..cut])
        } else {
            text.to_string()
        };
        let payload = serde_json::json!({"chat_id": chat_id, "text": safe_text});
        match ureq::post(&self.api_url("sendMessage")).send_json(&payload) {
            Ok(_) => Ok(()),
            Err(e) => {
                eprintln!("sendMessage: {}", self.mask(&e));
                Err(format!("Send: {}", self.mask(&e)))
            }
        }
    }

    fn send_voice(&self, chat_id: i64, text: &str) -> Result<(), String> {
        let wav_bytes = self.speech.speak(text);
        let boundary = "----FormBoundary7MA4YWxkTrZu0gW";
        let mut body = Vec::new();
        body.extend_from_slice(b"--");
        body.extend_from_slice(boundary.as_bytes());
        body.extend_from_slice(b"\r\nContent-Disposition: form-data; name=\"chat_id\"\r\n\r\n");
        body.extend_from_slice(format!("{}", chat_id).as_bytes());
        body.extend_from_slice(b"\r\n--");
        body.extend_from_slice(boundary.as_bytes());
        body.extend_from_slice(
            b"\r\nContent-Disposition: form-data; name=\"voice\"; filename=\"fuga_speech.wav\"\r\n",
        );
        body.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
        body.extend_from_slice(&wav_bytes);
        body.extend_from_slice(b"\r\n--");
        body.extend_from_slice(boundary.as_bytes());
        body.extend_from_slice(b"--\r\n");
        match ureq::post(&self.api_url("sendVoice"))
            .set(
                "Content-Type",
                &format!("multipart/form-data; boundary={}", boundary),
            )
            .send_bytes(&body)
        {
            Ok(_) => Ok(()),
            Err(e) => {
                if format!("{}", e).contains("400") {
                    self.send_message(chat_id, &format!("(voice: {})", text))
                } else {
                    eprintln!("sendVoice: {}", self.mask(&e));
                    self.send_message(chat_id, text)
                }
            }
        }
    }

    fn respond(&mut self, query: &str) -> String {
        let q = query.trim().to_lowercase();

        if is_greeting(&q) {
            self.last_query = q;
            return "Привет! Я Fuga — VSA-нейросеть на гипервекторной памяти. Спрашивай что угодно.".to_string();
        }
        if q.contains("кто ты") || q.contains("who are you") || q.contains("что ты") {
            self.last_query = q;
            return "Я Fuga Omni — нейросеть на гипервекторной памяти (VSA). Могу отвечать на вопросы, писать код, вести диалог.".to_string();
        }
        if q.contains("как дела") || q.contains("how are you") || q.contains("what's up") {
            self.last_query = q;
            return "Всё отлично! Гипервекторы резонируют, куб активен. Чем могу помочь?"
                .to_string();
        }
        if q.contains("спасибо") || q.contains("thanks") || q.contains("благодарю")
        {
            self.last_query = q;
            return "Пожалуйста! Обращайся.".to_string();
        }
        if q.contains("пока")
            || q.contains("bye")
            || q.contains("goodbye")
            || q.contains("до свидания")
        {
            self.last_query = q;
            return "Пока! Заходи ещё.".to_string();
        }
        if is_binary_question(&q) {
            let ctx = &self.last_query;
            if !ctx.is_empty() && !ctx.contains("да или нет") {
                let answer = format!(
                    "По твоему предыдущему вопросу «{}» — да. Но если коротко: уточни, что именно тебя интересует.",
                    ctx
                );
                self.last_query = q;
                return answer;
            }
            self.last_query = q;
            return "Сформулируй вопрос точнее, и я отвечу «да» или «нет».".to_string();
        }

        let search_query = if need_context_persistence(&q) && !self.last_query.is_empty() {
            format!("{} {}", self.last_query, q)
        } else {
            q.clone()
        };

        let tokens: Vec<TokenInfo> = search_query
            .split_whitespace()
            .map(|w| TokenInfo {
                id: token_id(&w),
                text: w.to_string(),
            })
            .collect();
        let output = self.ai.think(&tokens);

        if !output.super_tokens.is_empty() {
            let query_vec = &output.super_tokens[0].vector;
            let mut pv: Vec<&Hypervector> = Vec::new();
            if let Some(c) = self.prompts.get("CONCISE") {
                pv.push(c);
            }
            if let Some(e) = self.prompts.get("EFFICIENT") {
                pv.push(e);
            }

            let results = self.ai.memory.search_with_prompts(query_vec, &pv, 5);
            let mut seen = HashSet::new();
            let mut parts: Vec<String> = Vec::new();
            let mut blend_hvs: Vec<Hypervector> = Vec::new();

            for (_, _, entry) in &results {
                let t = entry.text.trim();
                if t.len() < 10 || is_training_artifact(t) {
                    continue;
                }
                let mut h = std::collections::hash_map::DefaultHasher::new();
                t.hash(&mut h);
                if !seen.insert(h.finish()) {
                    continue;
                }
                parts.push(t.to_string());
                if parts.len() <= 3 {
                    blend_hvs.push(entry.vector.clone());
                }
            }

            if blend_hvs.len() >= 2 {
                let refs: Vec<&Hypervector> = blend_hvs[1..].iter().collect();
                let hybrid = if let Some(ref jepa) = self.jepa {
                    let all_refs: Vec<&Hypervector> = blend_hvs.iter().collect();
                    jepa.predict(&all_refs)
                } else {
                    blend_hvs[0].bundle(&refs)
                };
                let blend_results = self.ai.memory.search(&hybrid, 3);
                for (_, _, entry) in &blend_results {
                    let t = entry.text.trim();
                    if t.len() < 10 || is_training_artifact(t) {
                        continue;
                    }
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    t.hash(&mut h);
                    if !seen.insert(h.finish()) {
                        continue;
                    }
                    parts.push(t.to_string());
                }
            }

            if !parts.is_empty() {
                self.last_query = q;
                return parts.join("\n");
            }
        }

        let raw = self.ai.answer(query);
        let mut parts: Vec<String> = Vec::new();
        for line in raw.lines() {
            let t = line.trim();
            if t.is_empty()
                || t.starts_with('[')
                || t.starts_with("Answer for:")
                || t.starts_with("Route:")
                || t.starts_with("//")
                || t.starts_with('#')
            {
                continue;
            }
            if is_training_artifact(t) {
                continue;
            }
            parts.push(t.to_string());
        }
        if !parts.is_empty() {
            let result = parts.join("\n");
            if result.len() > 10 {
                self.last_query = q;
                return result;
            }
        }

        if !need_context_persistence(&q) || looks_like_code_request(&q) {
            let domain = MoEStore::domain_for(query);
            let hits = self.moe.search_by_text(domain, query, 8);
            for (_, _, entry) in hits.iter() {
                let t = entry.text.trim();
                if t.len() > 10 && !is_training_artifact(t) {
                    self.last_query = q;
                    return t.to_string();
                }
            }

            let all = self.moe.search_all_by_text(query, 8);
            for (_, _, entry, _) in all.iter() {
                let t = entry.text.trim();
                if t.len() > 10 && !is_training_artifact(t) {
                    self.last_query = q;
                    return t.to_string();
                }
            }
        }

        self.last_query = q;
        "Я тебя слушаю. Расскажи подробнее.".to_string()
    }

    fn respond_baby(&mut self, chat_id: i64, text: &str) -> String {
        let hjepa = match self.hjepa {
            Some(ref mut h) => h,
            None => return "H-JEPA не загружен".to_string(),
        };
        let ctx_len = hjepa.levels[0].context_len;
        let tokens: Vec<&str> = text.split_whitespace().collect();
        if tokens.is_empty() {
            return "Напиши что-нибудь".to_string();
        }

        let ctx = self.contexts.entry(chat_id).or_insert(BabyContext {
            text: String::new(),
            vecs: Vec::new(),
        });

        for chunk in tokens.chunks(3) {
            let token_infos: Vec<TokenInfo> = chunk
                .iter()
                .map(|w| TokenInfo {
                    id: token_id(w),
                    text: w.to_string(),
                })
                .collect();
            let token_hvs: Vec<Hypervector> = token_infos
                .iter()
                .map(|ti| {
                    use std::hash::{Hash, Hasher};
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    format!("token_{}", ti.id).hash(&mut h);
                    Hypervector::random(hjepa.dim)
                })
                .collect();
            let hv = if token_hvs.is_empty() {
                Hypervector::random(hjepa.dim)
            } else {
                let refs: Vec<&Hypervector> = token_hvs.iter().collect();
                refs[0].bundle(&refs[1..]).balance_density()
            };
            ctx.vecs.push(hv);
        }
        ctx.text.push_str(text);
        ctx.text.push(' ');

        while ctx.vecs.len() > 20 {
            ctx.vecs.remove(0);
        }

        if ctx.vecs.len() < ctx_len {
            return format!("Строю контекст... ({}/{})", ctx.vecs.len(), ctx_len);
        }

        let window: Vec<&Hypervector> = ctx.vecs[ctx.vecs.len().saturating_sub(ctx_len)..]
            .iter()
            .collect();
        let predictions = hjepa.predict(&window);

        let input_hvs: Vec<Hypervector> =
            ctx.vecs[ctx.vecs.len().saturating_sub(ctx_len)..].to_vec();
        let input_refs: Vec<&Hypervector> = input_hvs.iter().collect();
        let errors = hjepa.learn(&window, &input_refs);

        let mut response = String::new();

        for (li, pred) in predictions.iter().enumerate() {
            let name = match li {
                0 => "L0",
                1 => "L1",
                2 => "L2",
                _ => "",
            };
            let entropy = pred.entropy();
            let emoji = if entropy > 0.98 {
                "\u{1F300}"
            } else if entropy > 0.90 {
                "\u{1F30A}"
            } else {
                "\u{26A1}"
            };
            let err_str = if li < errors.len() {
                format!(" err={:.3}", errors[li])
            } else {
                String::new()
            };
            response.push_str(&format!(
                "{} {}: entropy={:.4}{}\n",
                emoji, name, entropy, err_str
            ));
        }

        if !predictions.is_empty() {
            let results = self.ai.memory.search(&predictions[0], 1);
            if !results.is_empty() {
                let (_, sim, entry) = &results[0];
                let snippet: String = entry.text.chars().take(100).collect();
                response.push_str(&format!("\u{1F4D6} L0 \u{2192} [{:.2}] {}\n", sim, snippet));
            }
        }
        response.push_str(&format!("\u{1F916} ctx_len={}", ctx.vecs.len()));
        response
    }

    fn handle_message(&mut self, chat_id: i64, text: &str) -> Result<(), String> {
        let parts: Vec<&str> = text.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(());
        }

        let cmd = parts[0];
        let args = parts[1..].join(" ");

        match cmd {
            "/start" | "/help" => {
                let help = "\u{1F916} Fuga Omni — живой диалог на VSA-памяти\n\n\
                \u{1F4AC} Просто напиши что-нибудь — я обучен на коде и текстах\n\
                \u{1F4DD} /code <описание> — сгенерировать код\n\
                \u{1F3A4} /speak <текст> — озвучить\n\
                \u{1F4A1} /stats — состояние\n\
                \u{1F3AF} /prompt <MODE> — установить режим (SAFETY, EFFICIENT, CONCISE, EXPLAIN)";
                self.send_message(chat_id, help)
            }
            "/code" => {
                if args.trim().is_empty() {
                    return self.send_message(chat_id, "Опиши, какой код нужен");
                }
                if !looks_like_code_request(&args) {
                    return self.send_message(
                        chat_id,
                        "Напиши, какой именно код нужен. Например: /code python hello world",
                    );
                }
                self.send_message(chat_id, "\u{1F52E} Генерирую код...")?;
                let domain = MoEStore::domain_for(&args);
                let hits = self.moe.search_by_text(domain, &args, 12);

                let mut seen = HashSet::new();
                let mut code_parts: Vec<String> = Vec::new();

                for (_, _, entry) in hits.iter() {
                    let raw = entry.text.trim();
                    if raw.len() < 10 || is_training_artifact(raw) {
                        continue;
                    }
                    if !matches_language(&args, raw) {
                        continue;
                    }
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    raw.hash(&mut h);
                    let hash = h.finish();
                    if seen.contains(&hash) {
                        continue;
                    }
                    seen.insert(hash);
                    let excerpt: String = raw.chars().take(600).collect();
                    if excerpt.len() > 20 {
                        code_parts.push(excerpt);
                        if code_parts.len() >= 4 {
                            break;
                        }
                    }
                }

                let lang = if args.contains("rust") || args.contains("rs") {
                    "rust"
                } else if args.contains("py") || args.contains("python") {
                    "python"
                } else if args.contains("js") || args.contains("javascript") {
                    "javascript"
                } else if args.contains("ts") {
                    "typescript"
                } else if args.contains("go") {
                    "go"
                } else if args.contains("c++") || args.contains("cpp") {
                    "cpp"
                } else if args.contains("c ") || args.contains("cargo") {
                    "c"
                } else {
                    "rust"
                };

                let response = if code_parts.is_empty() {
                    "Не нашлось кода по запросу. Уточни.".to_string()
                } else {
                    format!("```{}\n{}```", lang, code_parts.join("\n"))
                };

                self.send_message(chat_id, &response)
            }
            "/prompt" => {
                let mode = args.trim().to_uppercase();
                if mode.is_empty() {
                    let mut info = "\u{1F3AF} Доступные режимы:\n".to_string();
                    for name in self.prompts.all_modes() {
                        if let Some(hv) = self.prompts.get(&name) {
                            info.push_str(&format!(
                                "  {} (dim={}, entropy={:.4})\n",
                                name,
                                hv.dim,
                                hv.entropy()
                            ));
                        }
                    }
                    self.send_message(chat_id, &info)
                } else {
                    let mode_names: Vec<&str> = mode.split(',').collect();
                    let base = self.prompts.get(mode_names[0]);
                    match base {
                        Some(bv) => {
                            let rest: Vec<&Hypervector> = mode_names[1..].iter()
                                .filter_map(|n| self.prompts.get(n))
                                .collect();
                            let modulated = PromptVectors::bind_all(bv, &rest);
                            let e = modulated.entropy();
                            self.send_message(chat_id, &format!(
                                "\u{2705} Режим {} → dim={}, entropy={:.4}", mode, modulated.dim, e))
                        }
                        None => self.send_message(chat_id, &format!(
                            "Неизвестный режим: {}. Доступные: SAFETY, EFFICIENT, CONCISE, EXPLAIN, DRY_RUN", mode))
                    }
                }
            }
            "/jepa" => {
                let sub = args.trim().to_lowercase();
                match sub.as_str() {
                    "" | "status" => match self.jepa {
                        Some(ref j) => {
                            let ler = if j.weights.len() > 1 {
                                format!("{:.4}", j.weights[0] / (1.0 / j.weights.len() as f64))
                            } else {
                                "N/A".into()
                            };
                            self.send_message(
                                chat_id,
                                &format!(
                                    "\u{269B} JEPA: dim={}, ctx_len={}\n\
                                     weights: {:?}\n\
                                     loading ratio: {}",
                                    j.dim, j.context_len, j.weights, ler
                                ),
                            )
                        }
                        None => self
                            .send_message(chat_id, "\u{26A0} JEPA не загружен (нет fuga_jepa.bin)"),
                    },
                    "train" => {
                        let mem_size = self.ai.memory.size();
                        if mem_size < 10 {
                            return self.send_message(chat_id, "Слишком мало записей для обучения");
                        }
                        self.send_message(chat_id, "\u{1F9E0} Обучаю JEPA на памяти...")?;

                        let dim = 8192;
                        let cl = 4usize;
                        let mut jepa = JepaPredictor::new(dim, cl);

                        let mut seqs: Vec<Vec<Hypervector>> = Vec::new();
                        let mut rng = rand::thread_rng();
                        for _ in 0..50 {
                            let rv = Hypervector::random(dim);
                            let nearby: Vec<Hypervector> = self
                                .ai
                                .memory
                                .search(&rv, cl + 1)
                                .into_iter()
                                .map(|(_, _, e)| e.vector.clone())
                                .collect();
                            if nearby.len() >= cl + 1 {
                                seqs.push(nearby);
                            }
                        }

                        if seqs.is_empty() {
                            return self
                                .send_message(chat_id, "Не удалось собрать последовательности");
                        }

                        let loss = jepa.train_on_sequences(&seqs, 50);
                        let _ = jepa.save(JEPA_PATH);
                        self.jepa = Some(jepa);

                        self.send_message(
                            chat_id,
                            &format!(
                                "\u{2705} JEPA обучен: loss={:.4}, sequences={}",
                                loss,
                                seqs.len()
                            ),
                        )
                    }
                    _ => self.send_message(chat_id, "Использование: /jepa [status|train]"),
                }
            }
            "/speak" => {
                if args.trim().is_empty() {
                    return self.send_message(chat_id, "Напиши текст для озвучки");
                }
                self.send_voice(chat_id, &args)
            }
            "/baby" => {
                let hjepa =
                    match self.hjepa {
                        Some(ref h) => h,
                        None => return self.send_message(
                            chat_id,
                            "H-JEPA не загружен (нет fuga_hjepa.bin). Сначала обучи: h-jepa-train",
                        ),
                    };
                let sub = args.trim().to_lowercase();
                match sub.as_str() {
                    "" | "status" => {
                        let msg = format!(
                            "\u{1F476} H-JEPA Baby: dim={}, L0(ctx={}) L1(ctx={}) L2(ctx={})",
                            hjepa.dim,
                            hjepa.levels[0].context_len,
                            hjepa.levels[1].context_len,
                            hjepa.levels[2].context_len
                        );
                        self.send_message(chat_id, &msg)
                    }
                    "on" => {
                        self.contexts.entry(chat_id).or_insert(BabyContext {
                            text: String::new(),
                            vecs: Vec::new(),
                        });
                        self.send_message(chat_id, "\u{1F476} Baby mode ON. Теперь каждое сообщение проходит через H-JEPA.")
                    }
                    "off" => {
                        self.contexts.remove(&chat_id);
                        self.send_message(chat_id, "\u{1F476} Baby mode OFF.")
                    }
                    "reset" => {
                        if let Some(ctx) = self.contexts.get_mut(&chat_id) {
                            ctx.text.clear();
                            ctx.vecs.clear();
                            self.send_message(chat_id, "\u{1F476} Context reset.")
                        } else {
                            self.send_message(chat_id, "Baby mode not active.")
                        }
                    }
                    _ => self.send_message(chat_id, "/baby [status|on|off|reset]"),
                }
            }
            "/stats" => {
                let entropy = self.ai.cube.global_entropy();
                let mem = self.ai.memory.size();
                let mut info = format!(
                    "\u{1F4CA} Записей: {} | Энтропия: {:.4}\n\nДомены:\n",
                    mem, entropy
                );
                for (dom, sz) in self.moe.domain_sizes() {
                    if sz > 0 {
                        info.push_str(&format!("  {}: {}\n", dom, sz));
                    }
                }
                self.send_message(chat_id, &info)
            }
            _ => {
                if cmd.starts_with('/') {
                    self.send_message(chat_id, &format!("Неизвестная команда: {}", cmd))
                } else if self.contexts.contains_key(&chat_id) {
                    let answer = self.respond_baby(chat_id, text);
                    self.send_message(chat_id, &answer)
                } else {
                    let answer = self.respond(text);
                    self.send_message(chat_id, &answer)
                }
            }
        }
    }

    fn run(&mut self) -> Result<(), String> {
        println!("Bot polling...");
        loop {
            match self.get_updates() {
                Ok(updates) => {
                    if let Some(results) = updates.get("result").and_then(|v| v.as_array()) {
                        for upd in results {
                            if let Some(update_id) = upd.get("update_id").and_then(|v| v.as_i64()) {
                                self.offset = update_id;
                            }
                            if let Some(msg) = upd.get("message") {
                                if let Some(chat) = msg
                                    .get("chat")
                                    .and_then(|c| c.get("id").and_then(|v| v.as_i64()))
                                {
                                    if let Some(text) = msg.get("text").and_then(|v| v.as_str()) {
                                        if let Err(e) = self.handle_message(chat, text) {
                                            eprintln!("Handle: {}", e);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Updates: {}", e);
                    std::thread::sleep(std::time::Duration::from_secs(if e.contains("409") {
                        5
                    } else {
                        3
                    }));
                }
            }
        }
    }
}

fn run_bot<const N: usize, const S: usize>(
    token: String,
    cube_path: String,
    mem_path: String,
    dim: usize,
) {
    let mut bot = TgBot::<N, S>::new(token, cube_path, mem_path, dim).expect("Bot init failed");
    if let Err(e) = bot.run() {
        eprintln!("Bot error: {}", e);
    }
}

fn main() {
    let token = env::var("FUGA_TG_TOKEN").unwrap_or_else(|_| {
        if let Ok(t) = std::fs::read_to_string("fuga.token") {
            return t.trim().to_string();
        }
        eprintln!("Set FUGA_TG_TOKEN or create fuga.token");
        std::process::exit(1);
    });
    let cube_path = env::var("FUGA_CUBE_PATH").unwrap_or_else(|_| "fuga_code_cube.bin".into());
    let mem_path = env::var("FUGA_MEM_PATH").unwrap_or_else(|_| "fuga_code_cube_mem.bin".into());
    let dim = env::var("FUGA_DIM")
        .unwrap_or_else(|_| "8192".into())
        .parse()
        .unwrap_or(8192);

    let (ndim, side_len, _) = match peek_cube_header(&cube_path) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };
    match (ndim, side_len) {
        (3, 4) => run_bot::<3, 4>(token, cube_path, mem_path, dim),
        (4, 4) => run_bot::<4, 4>(token, cube_path, mem_path, dim),
        (3, 5) => run_bot::<3, 5>(token, cube_path, mem_path, dim),
        (3, 6) => run_bot::<3, 6>(token, cube_path, mem_path, dim),
        (3, 7) => run_bot::<3, 7>(token, cube_path, mem_path, dim),
        (3, 8) => run_bot::<3, 8>(token, cube_path, mem_path, dim),
        (4, 8) => run_bot::<4, 8>(token, cube_path, mem_path, dim),
        (5, 2) => run_bot::<5, 2>(token, cube_path, mem_path, dim),
        (5, 4) => run_bot::<5, 4>(token, cube_path, mem_path, dim),
        _ => eprintln!("Unsupported cube: {}×{}", side_len, ndim),
    }
}

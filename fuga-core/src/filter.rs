use std::collections::{HashMap, HashSet};

use pyo3::prelude::*;
use rayon::prelude::*;

/// Лингвистический валидатор перед VSA-кодированием.
///
/// Два канала:
///  1. Лексикон (Wiktionary/корпусные словоформы) — O(1) проверка атомов.
///  2. Биграммные переходы (RuCoLA ok-предложения) — допустимость связей.
///
/// Вместо жёсткого "все слова и все пары должны быть известны" считаем
/// ДОЛЮ покрытия: реальный лексикон всегда дырявый, тотальный мэтч
/// отбраковал бы большинство грамматичных фраз.
#[pyclass]
pub struct RustLinguisticFilter {
    valid_vocab: HashSet<String>,
    transitions: HashMap<String, HashSet<String>>,
    // Морфология из Wiktionary-дампа: словоформа -> множество POS-тегов.
    word_pos: HashMap<String, Vec<String>>,
    // Статистика для отладки порогов (AtomicU32: фильтр шарится между потоками).
    checked: std::sync::atomic::AtomicUsize,
    accepted: std::sync::atomic::AtomicUsize,
}

fn normalize(w: &str) -> String {
    w.to_lowercase()
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '-' || c == '_'))
        .filter(|s| !s.is_empty())
        .map(normalize)
        .collect()
}

#[pymethods]
impl RustLinguisticFilter {
    #[new]
    pub fn new() -> Self {
        RustLinguisticFilter {
            valid_vocab: HashSet::new(),
            transitions: HashMap::new(),
            word_pos: HashMap::new(),
            checked: std::sync::atomic::AtomicUsize::new(0),
            accepted: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Словоформы/леммы лексикона (Wiktionary или корпусная частотная база).
    pub fn load_wiktionary_vocab(&mut self, words: Vec<String>) {
        for w in &words {
            self.valid_vocab.insert(normalize(w));
        }
    }

    /// Валидные биграммы: только из предложений с acceptable=1 (RuCoLA).
    pub fn load_rucola_transitions(&mut self, transitions: Vec<(String, String)>) {
        for (w1, w2) in &transitions {
            let k = normalize(w1);
            let v = normalize(w2);
            if !k.is_empty() && !v.is_empty() {
                self.transitions.entry(k).or_default().insert(v);
            }
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.valid_vocab.len()
    }

    pub fn transitions_size(&self) -> usize {
        self.transitions.values().map(|s| s.len()).sum()
    }

    /// Скор приемлемости: среднее двух покрытий (лексикон, биграммы).
    /// Пустые каналы дают нейтральную 1.0 по своей части.
    pub fn score(&self, text: &str) -> (f32, f32) {
        let words = tokenize(text);
        if words.is_empty() {
            return (0.0, 0.0);
        }
        let vocab_hits = words.iter().filter(|w| self.valid_vocab.contains(*w)).count();
        let vocab_cov = vocab_hits as f32 / words.len() as f32;

        if words.len() < 2 {
            return (vocab_cov, 1.0);
        }
        let mut known = 0usize;
        let mut total = 0usize;
        for pair in words.windows(2) {
            total += 1;
            if let Some(nexts) = self.transitions.get(&pair[0]) {
                if nexts.contains(&pair[1]) {
                    known += 1;
                }
            }
        }
        let trans_cov = known as f32 / total.max(1) as f32;
        (vocab_cov, trans_cov)
    }

    /// Жёсткая версия из спецификации: все слова в лексиконе И все биграммы.
    pub fn is_acceptable_strict(&self, text: &str) -> bool {
        let (v, t) = self.score(text);
        v >= 1.0 && t >= 1.0
    }

    /// Пороговая версия: покрытие лексикона >= vocab_cov И биграмм >= trans_cov.
    #[pyo3(signature = (text, vocab_cov=0.8, trans_cov=0.5))]
    pub fn is_acceptable(&self, text: &str, vocab_cov: f32, trans_cov: f32) -> bool {
        let (v, t) = self.score(text);
        v >= vocab_cov && t >= trans_cov
    }

    /// Батч-фильтрация до bind_batch: rayon, без GIL на горячем пути.
    /// Возвращает (valid_texts, rejected_texts).
    #[pyo3(signature = (batch, vocab_cov=0.8, trans_cov=0.5))]
    pub fn filter_batch(
        &self,
        py: Python<'_>,
        batch: Vec<String>,
        vocab_cov: f32,
        trans_cov: f32,
    ) -> (Vec<String>, Vec<String>) {
        py.allow_threads(|| {
            let flags: Vec<bool> = batch
                .par_iter()
                .map(|text| {
                    let (v, t) = self.score(text);
                    v >= vocab_cov && t >= trans_cov
                })
                .collect();
            let mut ok = Vec::new();
            let mut bad = Vec::new();
            for (text, good) in batch.into_iter().zip(flags) {
                if good {
                    ok.push(text)
                } else {
                    bad.push(text)
                }
            }
            use std::sync::atomic::Ordering;
            self.checked.fetch_add(ok.len() + bad.len(), Ordering::Relaxed);
            self.accepted.fetch_add(ok.len(), Ordering::Relaxed);
            (ok, bad)
        })
    }

    /// Негативы для VICReg/InfoNCE: перемешанные слова реальных фраз
    /// (синтаксический мусор при сохранении лексики) — hard negatives.
    #[pyo3(signature = (batch, n_shuffles=1))]
    pub fn make_word_salad_negatives(
        &self,
        py: Python<'_>,
        batch: Vec<String>,
        n_shuffles: usize,
    ) -> Vec<String> {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        py.allow_threads(|| {
            let mut out = Vec::with_capacity(batch.len() * n_shuffles.max(1));
            let mut rng = rand::rngs::StdRng::seed_from_u64(0xF00D);
            for text in &batch {
                let mut words = tokenize(text);
                if words.len() < 3 {
                    continue;
                }
                for _ in 0..n_shuffles.max(1) {
                    words.shuffle(&mut rng);
                    out.push(words.join(" "));
                }
            }
            out
        })
    }
    /// Потоковая загрузка дампа Wiktionary (kaikki.org JSONL) без Python GIL:
    /// каждая строка разбирается serde_json на лету, словоформа летит в HashSet.
    pub fn load_wiktionary_dump_jsonl(&mut self, file_path: &str) -> PyResult<usize> {
        use std::io::{BufRead, BufReader};
        let file = std::fs::File::open(file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{e}")))?;
        let reader = BufReader::with_capacity(1 << 20, file);
        let mut loaded = 0usize;
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) {
                if let Some(word) = json.get("word").and_then(|w| w.as_str()) {
                    if !word.is_empty() {
                        self.valid_vocab.insert(normalize(word));
                        loaded += 1;
                    }
                }
            }
        }
        Ok(loaded)
    }

    /// POS-теги словоформы (из дампа Wiktionary).
    pub fn pos_of(&self, py: Python<'_>, word: String) -> Vec<String> {
        let _ = py;
        self.word_pos
            .get(&normalize(&word))
            .cloned()
            .unwrap_or_default()
    }

    /// Потоковая загрузка дампа с морфологией: word + pos (kaikki JSONL).
    /// Заполняет и лексикон, и карту POS за один проход файла.
    pub fn load_wiktionary_pos_jsonl(&mut self, file_path: &str) -> PyResult<usize> {
        use std::io::{BufRead, BufReader};
        let file = std::fs::File::open(file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{e}")))?;
        let reader = BufReader::with_capacity(1 << 20, file);
        let mut loaded = 0usize;
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            let Ok(json) = serde_json::from_str::<serde_json::Value>(&line) else {
                continue;
            };
            let Some(word) = json.get("word").and_then(|w| w.as_str()) else {
                continue;
            };
            if word.is_empty() {
                continue;
            }
            let key = normalize(word);
            self.valid_vocab.insert(key.clone());
            if let Some(pos) = json.get("pos").and_then(|p| p.as_str()) {
                if !pos.is_empty() {
                    let entry = self.word_pos.entry(key).or_default();
                    if !entry.iter().any(|p| p == pos) {
                        entry.push(pos.to_string());
                    }
                }
            }
            loaded += 1;
        }
        Ok(loaded)
    }
}

// ---------------------------------------------------------------------------
// AST Grammar Filter: валидность рёбер (parent_kind, child_kind) tree-sitter.
// ---------------------------------------------------------------------------

use tree_sitter::Parser as TsParser;

fn make_ts_parser(lang: i32) -> Option<TsParser> {
    let mut p = TsParser::new();
    let language: tree_sitter::Language = match lang {
        0 => tree_sitter_python::LANGUAGE.into(),
        1 => tree_sitter_c::LANGUAGE.into(),
        _ => return None,
    };
    p.set_language(&language).ok()?;
    Some(p)
}

/// Итеративный обход дерева (без рекурсии — глубокие деревья не рвут стек).
fn collect_edges(root: tree_sitter::Node, edges: &mut Vec<(String, String)>) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        let parent_kind = node.kind().to_string();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            edges.push((parent_kind.clone(), child.kind().to_string()));
            stack.push(child);
        }
    }
}
#[pyclass]
pub struct RustASTGrammarFilter {
    valid_ast_edges: HashSet<(String, String)>,
}

#[pymethods]
impl RustASTGrammarFilter {
    #[new]
    pub fn new() -> Self {
        RustASTGrammarFilter {
            valid_ast_edges: HashSet::new(),
        }
    }

    /// Правила грамматики: пары (родитель, потомок), извлечённые из
    /// эталонного корпуса (treebank / валидный код).
    pub fn load_ast_grammar_rules(&mut self, rules: Vec<(String, String)>) {
        for (p, c) in rules {
            self.valid_ast_edges.insert((p, c));
        }
    }

    pub fn rules_size(&self) -> usize {
        self.valid_ast_edges.len()
    }

    /// Извлечение рёбер из корпуса эталонных текстов (для обучения фильтра).
    #[pyo3(signature = (texts, lang=0))]
    pub fn collect_ast_edges(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        lang: i32,
    ) -> Vec<(String, String)> {
        py.allow_threads(|| {
            let mut parser = match make_ts_parser(lang) {
                Some(p) => p,
                None => return Vec::new(),
            };
            let mut out = Vec::new();
            for text in &texts {
                if let Some(tree) = parser.parse(text.as_bytes(), None) {
                    collect_edges(tree.root_node(), &mut out);
                }
            }
            out
        })
    }

    /// Скор приемлемости: доля рёбер AST, известных грамматике.
    /// Незнакомое ребро = структурный мусор. Пустой текст -> 0.0.
    #[pyo3(signature = (text, lang=0))]
    pub fn score_ast_acceptability(&self, text: &str, lang: i32) -> f32 {
        let mut parser = match make_ts_parser(lang) {
            Some(p) => p,
            None => return 0.0,
        };
        let tree = match parser.parse(text.as_bytes(), None) {
            Some(t) => t,
            None => return 0.0,
        };
        let mut edges = Vec::new();
        collect_edges(tree.root_node(), &mut edges);
        if edges.is_empty() {
            return 0.0;
        }
        let valid = edges.iter().filter(|e| self.valid_ast_edges.contains(e)).count();
        valid as f32 / edges.len() as f32
    }
}


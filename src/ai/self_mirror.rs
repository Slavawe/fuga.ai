use std::collections::{HashMap, BTreeSet};
use std::path::Path;

use crate::ai::sdr::{encode_text, sparsify, SdrVector};
use crate::ai::htm_temporal::TemporalMemory;
use crate::ai::hierarchical_jepa::HierarchicalJEPA;
use crate::ai::temporal_predictor::TemporalPredictor;
use crate::core::hypervector::Hypervector;
use crate::vsa::topology::ls_bind;

const MIRROR_TM_PATH: &str = "fuga_mirror_tm.bin";
const MIRROR_JEPA_PATH: &str = "fuga_mirror_jepa.bin";

#[derive(Clone, Debug)]
pub struct RawChunk {
    pub path: String,
    pub line: usize,
    pub kind: String,
    pub name: String,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct PhaseNode {
    pub path: String,
    pub line: usize,
    pub kind: String,
    pub name: String,
    pub tm_match: f64,
    pub l0_err: f64,
    pub l1_err: f64,
    pub language: String,
    pub file_order: usize,
}

fn file_language(path: &str) -> String {
    if path.ends_with(".java") { "java".into() }
    else if path.ends_with(".rs") { "rust".into() }
    else if path.ends_with(".c") || path.ends_with(".cpp") || path.ends_with(".h") || path.ends_with(".hpp") { "cpp".into() }
    else if path.ends_with(".py") { "python".into() }
    else if path.ends_with(".go") { "go".into() }
    else if path.ends_with(".js") || path.ends_with(".ts") { "js".into() }
    else { "other".into() }
}

pub struct SelfMirror {
    pub predictor: TemporalPredictor,
    pub nodes: Vec<PhaseNode>,
    pub cache: HashMap<String, Vec<PhaseNode>>,
    name_sdr_cache: HashMap<String, SdrVector>,
    pub token_vocab: Vec<(String, SdrVector)>,
}

impl SelfMirror {
    pub fn new(tm: TemporalMemory, hjepa: HierarchicalJEPA) -> Self {
        SelfMirror {
            predictor: TemporalPredictor::new(tm, hjepa),
            nodes: Vec::new(),
            cache: HashMap::new(),
            name_sdr_cache: HashMap::new(),
            token_vocab: Vec::new(),
        }
    }

    pub fn load() -> Option<Self> {
        let tm_path = Path::new(MIRROR_TM_PATH);
        let jepa_path = Path::new(MIRROR_JEPA_PATH);
        if !tm_path.exists() || !jepa_path.exists() {
            return None;
        }
        let tm = TemporalMemory::load(MIRROR_TM_PATH)?;
        let hjepa = HierarchicalJEPA::load(MIRROR_JEPA_PATH).ok()?;
        let mut mirror = SelfMirror::new(tm, hjepa);
        let node_path = "fuga_mirror_nodes.bin";
        if let Ok(data) = std::fs::read(node_path) {
            let mut pos = 0usize;
            if pos + 4 <= data.len() {
                let n = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
                pos += 4;
                for _ in 0..n {
                    if pos + 4 > data.len() { break; }
                    let plen = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
                    pos += 4;
                    if pos + plen > data.len() { break; }
                    let path = String::from_utf8(data[pos..pos+plen].to_vec()).unwrap_or_default();
                    pos += plen;
                    if pos + 4 > data.len() { break; }
                    let line = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
                    pos += 4;
                    if pos + 4 > data.len() { break; }
                    let klen = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
                    pos += 4;
                    if pos + klen > data.len() { break; }
                    let kind = String::from_utf8(data[pos..pos+klen].to_vec()).unwrap_or_default();
                    pos += klen;
                    if pos + 4 > data.len() { break; }
                    let nlen = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
                    pos += 4;
                    if pos + nlen > data.len() { break; }
                    let name = String::from_utf8(data[pos..pos+nlen].to_vec()).unwrap_or_default();
                    pos += nlen;
                    if pos + 24 > data.len() { break; }
                    let tm_match = f64::from_le_bytes(data[pos..pos+8].try_into().ok()?);
                    pos += 8;
                    let l0_err = f64::from_le_bytes(data[pos..pos+8].try_into().ok()?);
                    pos += 8;
                    let l1_err = f64::from_le_bytes(data[pos..pos+8].try_into().ok()?);
                    pos += 8;
                    let language = if pos + 4 <= data.len() {
                        let llen = u32::from_le_bytes(data[pos..pos+4].try_into().ok()?) as usize;
                        pos += 4;
                        if pos + llen <= data.len() {
                            let lang = String::from_utf8(data[pos..pos+llen].to_vec()).unwrap_or_default();
                            pos += llen;
                            lang
                        } else { String::new() }
                    } else { String::new() };
                    let file_order = if pos + 8 <= data.len() {
                        let fo = u64::from_le_bytes(data[pos..pos+8].try_into().ok()?);
                        pos += 8;
                        fo as usize
                    } else { 0 };
                    mirror.nodes.push(PhaseNode { path, line, kind, name, tm_match, l0_err, l1_err, language, file_order });
                }
            }
        }
        if let Ok(data) = std::fs::read("fuga_token_vocab.bin") {
            let mut pos = 0usize;
            if pos + 4 <= data.len() {
                let n = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap_or([0u8; 4])) as usize;
                pos += 4;
                for _ in 0..n {
                    if pos + 4 > data.len() { break; }
                    let tlen = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap_or([0u8; 4])) as usize;
                    pos += 4;
                    if pos + tlen > data.len() { break; }
                    let tok = String::from_utf8(data[pos..pos+tlen].to_vec()).unwrap_or_default();
                    pos += tlen;
                    let mut bits = [0u64; 128];
                    for i in 0..128 {
                        if pos + 8 > data.len() { break; }
                        bits[i] = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap_or([0u8; 8]));
                        pos += 8;
                    }
                    mirror.token_vocab.push((tok, SdrVector { bits }));
                }
            }
        }
        Some(mirror)
    }

    pub fn save(&self) {
        self.predictor.tm.save(MIRROR_TM_PATH);
        let _ = self.predictor.hjepa.save(MIRROR_JEPA_PATH);
        self.predictor.save_buffer();
        use std::io::Write;
        if let Ok(mut f) = std::fs::File::create("fuga_token_vocab.bin") {
            let n = self.token_vocab.len() as u32;
            f.write_all(&n.to_le_bytes()).ok();
            for (tok, sdr) in &self.token_vocab {
                let tb = tok.as_bytes();
                f.write_all(&(tb.len() as u32).to_le_bytes()).ok();
                f.write_all(tb).ok();
                for word in &sdr.bits {
                    f.write_all(&word.to_le_bytes()).ok();
                }
            }
        }
        if let Ok(mut f) = std::fs::File::create("fuga_mirror_nodes.bin") {
            let n = self.nodes.len() as u32;
            f.write_all(&n.to_le_bytes()).ok();
            for node in &self.nodes {
                let pb = node.path.as_bytes();
                f.write_all(&(pb.len() as u32).to_le_bytes()).ok();
                f.write_all(pb).ok();
                f.write_all(&(node.line as u32).to_le_bytes()).ok();
                let kb = node.kind.as_bytes();
                f.write_all(&(kb.len() as u32).to_le_bytes()).ok();
                f.write_all(kb).ok();
                let nb = node.name.as_bytes();
                f.write_all(&(nb.len() as u32).to_le_bytes()).ok();
                f.write_all(nb).ok();
                f.write_all(&node.tm_match.to_le_bytes()).ok();
                f.write_all(&node.l0_err.to_le_bytes()).ok();
                f.write_all(&node.l1_err.to_le_bytes()).ok();
                let lb = node.language.as_bytes();
                f.write_all(&(lb.len() as u32).to_le_bytes()).ok();
                f.write_all(lb).ok();
                f.write_all(&(node.file_order as u64).to_le_bytes()).ok();
            }
        }
    }

    pub fn source_snippet(&self, node: &PhaseNode, context_lines: usize) -> String {
        self.source_snippet_for_path(&node.path, node.line, context_lines)
    }

    fn node_sdr(&mut self, name: &str) -> SdrVector {
        self.name_sdr_cache.entry(name.to_string())
            .or_insert_with(|| encode_text(name))
            .clone()
    }

    pub fn source_snippet_for_path(&self, path: &str, line: usize, context_lines: usize) -> String {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return String::new(),
        };
        let lines: Vec<&str> = content.lines().collect();
        if line == 0 || line > lines.len() {
            return String::new();
        }
        let start = line.saturating_sub(context_lines);
        let end = (line + context_lines).min(lines.len());
        let mut snippet = String::new();
        for i in start..end {
            if !snippet.is_empty() { snippet.push('\n'); }
            snippet.push_str(lines[i]);
        }
        snippet
    }

    pub fn source_snippet_for_prediction(&self, pred_sdr: &SdrVector) -> (String, usize) {
        let mut best_overlap = 0u32;
        let mut best_node: Option<&PhaseNode> = None;
        for node in &self.nodes {
            let nsdr = encode_text(&format!("{} {}", node.kind, node.name));
            let o = pred_sdr.overlap(&nsdr);
            if o > best_overlap {
                best_overlap = o;
                best_node = Some(node);
            }
        }
        if let Some(node) = best_node {
            (self.source_snippet(node, 3), best_overlap as usize)
        } else {
            (String::new(), 0)
        }
    }

    pub fn evaluate(&mut self) -> String {
        let mut total_l0 = 0.0f64;
        let mut total_l1 = 0.0f64;
        let mut total_l2 = 0.0f64;
        let mut total_cnt = 0usize;
        let mut sim_l0 = 0.0f64;
        let mut sim_l1 = 0.0f64;
        let mut dbg_modes = self.predictor.hjepa.levels.iter().map(|l| l.mode).collect::<Vec<_>>();

        let paths: Vec<String> = self.nodes.iter().map(|n| n.path.clone()).collect::<BTreeSet<_>>().into_iter().collect();
        let max_chunks = 1000usize;
        let mut processed = 0usize;

        for fp in &paths {
            if processed >= max_chunks { break; }
            let content = match std::fs::read_to_string(fp) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let words: Vec<&str> = content.split_whitespace().collect();
            for chunk in words.chunks(10) {
                if processed >= max_chunks { break; }
                let text = chunk.join(" ");
                let sdr = encode_text(&text);
                let (tm_pred, _) = self.predictor.tm.feed(&sdr);
                let hv = crate::ai::temporal_predictor::sdr_to_hypervector(&tm_pred, self.predictor.hjepa.dim);
                self.predictor.buffer.push(hv);
                if self.predictor.buffer.len() > self.predictor.buf_capacity {
                    self.predictor.buffer.remove(0);
                }
                if self.predictor.buffer.len() <= self.predictor.hjepa.levels[0].context_len {
                    continue;
                }
                let ctx: Vec<&Hypervector> = self.predictor.buffer[..self.predictor.buffer.len()-1].iter().collect();
                let actual = self.predictor.buffer.last().unwrap();

                let l0_sim = self.predictor.hjepa.levels[0].similarity_to_expected(&ctx, actual);
                total_l0 += 1.0 - l0_sim;
                sim_l0 += l0_sim;
                let full_pred = self.predictor.hjepa.predict(&ctx);
                if full_pred.len() > 2 {
                    let l1_ctx_len = self.predictor.hjepa.levels[1].context_len;
                    let l1_start = ctx.len().saturating_sub(l1_ctx_len);
                    let l1_sim = self.predictor.hjepa.levels[1].similarity_to_expected(&ctx[l1_start..], actual);
                    total_l1 += 1.0 - l1_sim;
                    sim_l1 += l1_sim;
                    let corrected = &full_pred[2];
                    let c_sim = 1.0 - corrected.hamming_distance(actual);
                    total_l2 += 1.0 - c_sim;
                }
                total_cnt += 1;
                processed += 1;
            }
            if processed % 200 == 0 {
                print!("\r  Eval: {}/{} chunks", processed, max_chunks);
                use std::io::{Write, stdout};
                stdout().flush().ok();
            }
        }

        let a = |v: f64, c: usize| if c > 0 { v / c as f64 } else { 1.0 };
        println!("\r  Eval: {} chunks", processed);
        println!("  modes={:?} avg_sim_L0={:.4} avg_sim_L1={:.4}", dbg_modes, a(sim_l0, total_cnt), a(sim_l1, total_cnt));
        format!("L0={:.4} L1={:.4} L2={:.4} samples={}", a(total_l0, total_cnt), a(total_l1, total_cnt), a(total_l2, total_cnt), total_cnt)
    }

    pub fn index_file(&mut self, path: &str) -> Vec<PhaseNode> {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };
        let mut nodes = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;
        let mut file_counter = 0usize;
        let plang = file_language(path);
        while i < lines.len() {
            let line = lines[i];
            let trimmed = line.trim();
            if let Some(name) = Self::extract_fn(trimmed) {
                let mut block = String::new();
                let start = i;
                while i < lines.len() && !lines[i].trim().starts_with('}') {
                    block.push_str(lines[i]);
                    block.push('\n');
                    i += 1;
                }
                if i < lines.len() {
                    block.push_str(lines[i]);
                }
                let (tm_match, errors) = self.predictor.feed_learn(&block);
                file_counter += 1;
                let node = PhaseNode {
                    path: path.to_string(),
                    line: start + 1,
                    kind: "fn".into(),
                    name,
                    tm_match,
                    l0_err: errors.first().copied().unwrap_or(1.0),
                    l1_err: errors.get(1).copied().unwrap_or(1.0),
                    language: plang.clone(),
                    file_order: file_counter,
                };
                nodes.push(node);
                i += 1;
            } else if let Some(name) = Self::extract_struct(trimmed) {
                let mut block = String::new();
                let start = i;
                while i < lines.len() && !lines[i].trim().starts_with('}') {
                    block.push_str(lines[i]);
                    block.push('\n');
                    i += 1;
                }
                if i < lines.len() {
                    block.push_str(lines[i]);
                }
                let (tm_match, errors) = self.predictor.feed_learn(&block);
                file_counter += 1;
                let node = PhaseNode {
                    path: path.to_string(),
                    line: start + 1,
                    kind: "struct".into(),
                    name,
                    tm_match,
                    l0_err: errors.first().copied().unwrap_or(1.0),
                    l1_err: errors.get(1).copied().unwrap_or(1.0),
                    language: plang.clone(),
                    file_order: file_counter,
                };
                nodes.push(node);
                i += 1;
            } else if let Some(name) = Self::extract_impl(trimmed) {
                let mut block = String::new();
                let start = i;
                let mut brace = 1usize;
                block.push_str(lines[i]);
                block.push('\n');
                i += 1;
                while i < lines.len() && brace > 0 {
                    let l = lines[i].trim();
                    brace += l.matches('{').count();
                    brace -= l.matches('}').count();
                    block.push_str(lines[i]);
                    block.push('\n');
                    i += 1;
                }
                let (tm_match, errors) = self.predictor.feed_learn(&block);
                file_counter += 1;
                let node = PhaseNode {
                    path: path.to_string(),
                    line: start + 1,
                    kind: "impl".into(),
                    name,
                    tm_match,
                    l0_err: errors.first().copied().unwrap_or(1.0),
                    l1_err: errors.get(1).copied().unwrap_or(1.0),
                    language: plang.clone(),
                    file_order: file_counter,
                };
                nodes.push(node);
            } else {
                i += 1;
            }
        }
        self.cache.entry(path.to_string()).or_default().extend(nodes.clone());
        self.nodes.extend(nodes.clone());
        nodes
    }

    pub fn index_dir(&mut self, dir: &str) -> usize {
        let mut total = 0;
        let walk = match std::fs::read_dir(dir) {
            Ok(d) => d,
            Err(_) => return 0,
        };
        let mut entries: Vec<_> = walk.filter_map(|e| e.ok()).collect();
        entries.sort_by_key(|e| e.path());
        for entry in entries {
            let p = entry.path();
            let ps = p.to_str().unwrap_or("");
            let ext = p.extension().map(|e| e.to_str().unwrap_or("")).unwrap_or("");
            if ext == "rs" {
                let nodes = self.index_file(ps);
                total += nodes.len();
                println!("  {} → {} phase nodes", p.display(), nodes.len());
            } else if Self::is_c_cpp(ps) {
                let nodes = self.index_file_c(ps);
                total += nodes.len();
                println!("  {} → {} phase nodes", p.display(), nodes.len());
            } else if p.is_dir() {
                total += self.index_dir(ps);
            }
        }
        total
    }



    pub fn generate_code(&mut self, text: &str, steps: usize) -> Vec<Vec<(PhaseNode, u32)>> {
        self.generate_code_beam(text, steps, 1, 1.0)
    }

    pub fn generate_code_autoregressive(&mut self, seed: &str, max_steps: usize, _temperature: f64) -> Vec<String> {
        for word in seed.split_whitespace() {
            let sdr = encode_text(word);
            self.predictor.tm.feed(&sdr);
            let hv = crate::ai::temporal_predictor::sdr_to_hypervector(&sdr, self.predictor.hjepa.dim);
            self.predictor.buffer.push(hv);
            if self.predictor.buffer.len() > self.predictor.buf_capacity {
                self.predictor.buffer.remove(0);
            }
        }
        let goal_sdr = encode_text(seed);
        let goal_hv = crate::ai::temporal_predictor::sdr_to_hypervector(&goal_sdr, self.predictor.hjepa.dim);
        let mut output: Vec<String> = Vec::new();
        let mut last_nodes: Vec<String> = Vec::new();
        let mut recent_node_keys: Vec<String> = Vec::new();
        let mut dominant_lang: Option<String> = None;
        let mut prev_node: Option<PhaseNode> = None;
        let is_rust_seed = seed.contains("fn ") || seed.contains("impl ") || seed.contains("struct ") || seed.contains("async ");
        let node_sdr_cache: Vec<(String, SdrVector, String, usize, usize)> = self.nodes.iter()
            .map(|n| (n.path.clone(), encode_text(&n.name), n.language.clone(), n.file_order, n.line))
            .filter(|(_, _, lang, _, _)| !(is_rust_seed && (lang == "cpp" || lang == "C")))
            .collect();
        if self.predictor.buffer.len() < self.predictor.hjepa.levels[0].context_len {
            let pad_hv = Hypervector::random(self.predictor.hjepa.dim);
            while self.predictor.buffer.len() < self.predictor.hjepa.levels[0].context_len {
                self.predictor.buffer.push(pad_hv.clone());
            }
        }
        for _ in 0..max_steps {
            let ctx: Vec<&Hypervector> = self.predictor.buffer.iter().collect();
            if ctx.is_empty() { break; }
            let temps = [0.8, 1.0, 1.2, 1.5];
            let (preds, _converge) = self.predictor.hjepa.predict_refined(&ctx, &temps);
            let goal_dense = goal_hv.bundle(&[
                &goal_hv.permute(1), &goal_hv.permute(2), &goal_hv.permute(4), &goal_hv.permute(8),
                &goal_hv.permute(16), &goal_hv.permute(32), &goal_hv.permute(64), &goal_hv.permute(128),
            ]);
            let goal_bound_pred = ls_bind(&preds[2], &goal_dense, 32);
            let pred_sdr = sparsify(&goal_bound_pred);
            let mut best_score = 0f64;
            let mut best_overlap_raw = 0u32;
            let mut best_lineno = 0usize;
            let mut best_path = String::new();
            for (npath, nsdr, nlang, nforder, nline) in &node_sdr_cache {
                let node_key = format!("{}:{}", npath, nline);
                if recent_node_keys.iter().any(|k| *k == node_key) { continue; }
                let overlap = pred_sdr.overlap(nsdr) as f64;
                let mut score = overlap;
                if let Some(ref dl) = dominant_lang {
                    if nlang != dl { score *= 0.3; } else { score *= 1.5; }
                }
                if let Some(ref prev) = prev_node {
                    if prev.path == *npath && prev.file_order + 1 == *nforder {
                        score += overlap * 0.8;
                    }
                }
                if score > best_score {
                    best_score = score;
                    best_overlap_raw = overlap as u32;
                    best_path = npath.clone();
                    best_lineno = *nline;
                }
            }
            if best_score < 2.0 { break; }
            if best_overlap_raw < 10 {
                let retry = self.predictor.hjepa.levels[0].predict_with_temp(&ctx, 2.0);
                let retry_sdr = sparsify(&retry);
                let mut retry_overlap = 0u32;
                let mut retry_lineno = 0usize;
                let mut retry_path = String::new();
                for (npath, nsdr, _nlang, _nforder, nline) in &node_sdr_cache {
                    let node_key = format!("{}:{}", npath, nline);
                    if recent_node_keys.iter().any(|k| *k == node_key) { continue; }
                    let o = retry_sdr.overlap(nsdr);
                    if o > retry_overlap { retry_overlap = o; retry_path = npath.clone(); retry_lineno = *nline; }
                }
                if retry_overlap > best_overlap_raw { best_overlap_raw = retry_overlap; best_path = retry_path; best_lineno = retry_lineno; }
            }
            if dominant_lang.is_none() {
                let candidate = self.nodes.iter().find(|n| n.path == best_path);
                if let Some(n) = candidate { dominant_lang = Some(n.language.clone()); }
            }
            prev_node = self.nodes.iter().find(|n| n.path == best_path).cloned();
            let snippet = self.source_snippet_for_path(&best_path, best_lineno, 8);
            let combined = if snippet.is_empty() {
                format!("// (source not available) {} overlap={} score={:.0}", best_path, best_overlap_raw, best_score)
            } else {
                format!("// overlap={} score={:.0}\n{}", best_overlap_raw, best_score, snippet)
            };
            output.push(combined.clone());
            last_nodes.push(combined.clone());
            if last_nodes.len() > 6 { last_nodes.remove(0); }
            if last_nodes.iter().filter(|x| *x == &combined).count() >= 3 { break; }
            recent_node_keys.push(format!("{}:{}", best_path, best_lineno));
            if recent_node_keys.len() > 8 { recent_node_keys.remove(0); }
            let node_sdr = encode_text(&snippet);
            self.predictor.tm.feed(&node_sdr);
            let node_hv = crate::ai::temporal_predictor::sdr_to_hypervector(&node_sdr, self.predictor.hjepa.dim);
            self.predictor.buffer.push(node_hv);
            if self.predictor.buffer.len() > self.predictor.buf_capacity {
                self.predictor.buffer.remove(0);
            }
        }
        output
    }

    pub fn build_token_vocab_from_files(&mut self) -> usize {
        let top_n = 4000;
        let allowed_dirs = ["/home/slava/fuga/src/", "/home/slava/neural-engine/"];
        let multi_ops: std::collections::HashSet<String> = ["->", "::", "=>", "!=", "==", ">=", "<=", "+=", "-=", "&&", "||"].into_iter().map(|s| s.to_string()).collect();
        let mut freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for p in &self.nodes.iter().map(|n| n.path.clone()).collect::<Vec<_>>() {
            if !p.ends_with(".rs") { continue; }
            if !allowed_dirs.iter().any(|d| p.contains(d)) { continue; }
            if let Ok(text) = std::fs::read_to_string(p) {
                for word in text.split_whitespace() {
                    let chars: Vec<char> = word.chars().collect();
                    let mut i = 0;
                    while i < chars.len() {
                        if chars[i].is_alphanumeric() || chars[i] == '_' {
                            let mut acc = String::new();
                            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                                acc.push(chars[i]);
                                i += 1;
                            }
                            if acc.len() <= 32 { *freq.entry(acc).or_insert(0) += 1; }
                        } else {
                            let two: String = if i + 1 < chars.len() { chars[i..=i+1].iter().collect() } else { String::new() };
                            if !two.is_empty() && multi_ops.contains(&two) {
                                *freq.entry(two).or_insert(0) += 1;
                                i += 2;
                            } else {
                                *freq.entry(chars[i].to_string()).or_insert(0) += 1;
                                i += 1;
                            }
                        }
                    }
                }
            }
        }
        let mut sorted: Vec<(String, usize)> = freq.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.truncate(top_n);
        let mut vocab: Vec<(String, SdrVector)> = Vec::new();
        for (tok, _) in &sorted {
            vocab.push((tok.clone(), encode_text(tok)));
        }
        self.token_vocab = vocab;
        self.token_vocab.len()
    }

    pub fn generate_tokens(&mut self, seed: &str, max_tokens: usize) -> Vec<String> {
        let mut last_sdr = SdrVector::zero();
        for word in seed.split_whitespace() {
            let sdr = encode_text(word);
            let (pred, _) = self.predictor.tm.feed(&sdr);
            last_sdr = sdr;
            let _ = pred;
        }
        let mut output: Vec<String> = Vec::new();
        let mut recent_set: Vec<String> = Vec::new();
        let mut cell_fatigue: Vec<u32> = vec![0; self.predictor.tm.cells.len()];
        let mut fatigue_decay = 0u32;
        for _ in 0..max_tokens {
            while cell_fatigue.len() < self.predictor.tm.cells.len() {
                cell_fatigue.push(0);
            }
            let tm_pred = {
                let mut best_score = 0i32;
                let mut best_pat = SdrVector::zero();
                let mut best_cell_idx = 0usize;
                for (ci, c) in self.predictor.tm.cells.iter().enumerate() {
                    for seg in &c.segments {
                        let so = seg.overlap(&last_sdr);
                        let fatigue_penalty = (cell_fatigue[ci] * 10) as i32;
                        let adjusted = (so as i32) - fatigue_penalty;
                        if adjusted > best_score {
                            best_score = adjusted;
                            best_pat = c.pattern.clone();
                            best_cell_idx = ci;
                        }
                    }
                }
                if best_score >= 3 {
                    cell_fatigue[best_cell_idx] += 1;
                    fatigue_decay += 1;
                    if fatigue_decay >= 10 {
                        for f in &mut cell_fatigue { *f = f.saturating_sub(1); }
                        fatigue_decay = 0;
                    }
                    best_pat
                } else {
                    self.predictor.tm.predict_next(&last_sdr)
                }
            };
            let mut best_overlap = 0u32;
            let mut best_token = String::new();
            for (tok, tsdr) in &self.token_vocab {
                if recent_set.iter().any(|t| t == tok) { continue; }
                let o = tm_pred.overlap(tsdr);
                if o > best_overlap { best_overlap = o; best_token = tok.clone(); }
            }
            if best_overlap < 8 || best_token.is_empty() { break; }
            output.push(best_token.clone());
            recent_set.push(best_token.clone());
            if recent_set.len() > 16 { recent_set.remove(0); }
            let next_sdr = encode_text(&best_token);
            let (_pred, _) = self.predictor.tm.feed(&next_sdr);
            last_sdr = next_sdr;
        }
        output
    }

    pub fn train_token_sequences(&mut self, max_sequences: usize, steps_per_seq: usize) -> usize {
        self.predictor.tm.cells.clear();
        self.predictor.tm.window.clear();
        self.predictor.tm.step = 0;
        let allowed_dirs = ["/home/slava/fuga/src/", "/home/slava/neural-engine/"];
        let rust_keywords: std::collections::HashSet<&str> = [
            "fn", "let", "mut", "pub", "struct", "impl", "use", "mod", "type", "enum",
            "match", "if", "else", "while", "loop", "for", "in", "return", "self",
            "Self", "as", "break", "continue", "const", "static", "ref", "where",
            "unsafe", "trait", "impl", "dyn", "move", "box", "await", "async",
        ].into_iter().collect();
        let mut paths: std::collections::BTreeSet<String> = self.nodes.iter()
            .map(|n| n.path.clone()).filter(|p| p.ends_with(".rs") && allowed_dirs.iter().any(|d| p.contains(d)))
            .collect();
        for corpus_dir in ["crosvm", "sled", "smoltcp"] {
            let base = format!("/home/slava/fuga/corpus_combined/{}", corpus_dir);
            for entry in walkdir::WalkDir::new(&base).into_iter().filter_map(|e| e.ok()) {
                let fp = entry.path();
                if fp.extension().map(|x| x == "rs").unwrap_or(false) {
                    paths.insert(fp.to_string_lossy().to_string());
                }
            }
        }
        let paths_vec: Vec<String> = paths.into_iter().collect();
        let multi_ops: std::collections::HashSet<String> = ["->", "::", "=>", "!=", "==", ">=", "<=", "+=", "-=", "&&", "||"].into_iter().map(|s| s.to_string()).collect();
        let mut sequences: Vec<Vec<String>> = Vec::new();
        for p in paths_vec.iter().take(200) {
            if let Ok(text) = std::fs::read_to_string(p) {
                let mut tokens: Vec<String> = Vec::new();
                for word in text.split_whitespace() {
                    let mut acc = String::new();
                    let chars: Vec<char> = word.chars().collect();
                    let mut i = 0;
                    while i < chars.len() {
                        if chars[i].is_alphanumeric() || chars[i] == '_' {
                            acc.push(chars[i]);
                            i += 1;
                        } else {
                            if !acc.is_empty() { tokens.push(acc.clone()); acc.clear(); }
                            let two: String = if i + 1 < chars.len() { chars[i..=i+1].iter().collect() } else { String::new() };
                            if !two.is_empty() && multi_ops.contains(&two) {
                                tokens.push(two);
                                i += 2;
                            } else {
                                tokens.push(chars[i].to_string());
                                i += 1;
                            }
                        }
                    }
                    if !acc.is_empty() { tokens.push(acc.clone()); }
                }
                for chunk in tokens.windows(steps_per_seq).step_by(steps_per_seq) {
                    sequences.push(chunk.to_vec());
                    if sequences.len() >= max_sequences { break; }
                }
                if sequences.len() >= max_sequences { break; }
            }
        }
        let mut trained = 0usize;
        for seq in &sequences {
            for tok_str in seq {
                let sdr = encode_text(tok_str);
                let (pred, _) = self.predictor.tm.feed(&sdr);
                let _ = pred;
                trained += 1;
            }
        }
        let syntax_patterns: Vec<Vec<&str>> = vec![
            vec!["fn", "new", "(", ")", "{", "self", ".", "init", "(", ")", ";", "}"],
            vec!["pub", "fn", "handle", "(", "&", "self", ")", "->", "Result", "{", "}"],
            vec!["let", "mut", "x", "=", "self", ".", "get", "(", ")", ";"],
            vec!["if", "let", "Some", "(", "val", ")", "=", "opt", "{", "}"],
            vec!["match", "res", "{", "Ok", "(", "v", ")", "=>", "v", ",", "Err", "(", "e", ")", "=>", "e", "}"],
            vec!["impl", "Foo", "{", "pub", "fn", "bar", "(", ")", "->", "i32", "{", "0", "}", "}"],
            vec!["for", "item", "in", "list", ".", "iter", "(", ")", "{", "println", "!", "(", ")", ";", "}"],
            vec!["struct", "Config", "{", "host", ":", "String", ",", "port", ":", "u16", "}"],
            vec!["pub", "fn", "new", "(", ")", "->", "Self", "{", "Self", "{", "field", ":", "0", "}", "}"],
            vec!["fn", "audit", "(", "&", "self", ")", "{", "}"],
            vec!["fn", "process", "(", "input", ":", "&", "str", ")", "->", "bool", "{", "true", "}"],
            vec!["let", "result", "=", "self", ".", "check", "(", ")", ";"],
            vec!["if", "x", ">", "0", "{", "return", "x", ";", "}"],
            vec!["while", "idx", "<", "len", "{", "idx", "+=", "1", ";", "}"],
        ];
        for _ in 0..5 {
            for pat in &syntax_patterns {
                for tok_str in pat {
                    let sdr = encode_text(tok_str);
                    let (pred, _) = self.predictor.tm.feed(&sdr);
                    let _ = pred;
                    trained += 1;
                }
            }
        }
        trained
    }

    pub fn index_generated_snippets(&mut self, snippets: &[String]) -> usize {
        let mut count = 0usize;
        for snippet in snippets {
            let lines: Vec<&str> = snippet.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                let trimmed = line.trim();
                if let Some(name) = Self::extract_fn(trimmed) {
                    let text = lines[i..].join("\n");
                    let (tm_match, errors) = self.predictor.feed_learn(&text);
                    let node = PhaseNode {
                        path: "_generated".to_string(),
                        line: i + 1, kind: "fn".into(), name,
                        tm_match, l0_err: errors.first().copied().unwrap_or(1.0),
                        l1_err: errors.get(1).copied().unwrap_or(1.0),
                        language: "rust".into(), file_order: 0,
                    };
                    self.nodes.push(node);
                    count += 1;
                } else if let Some(name) = Self::extract_struct(trimmed) {
                    let text = lines[i..].join("\n");
                    let (tm_match, errors) = self.predictor.feed_learn(&text);
                    let node = PhaseNode {
                        path: "_generated".to_string(),
                        line: i + 1, kind: "struct".into(), name,
                        tm_match, l0_err: errors.first().copied().unwrap_or(1.0),
                        l1_err: errors.get(1).copied().unwrap_or(1.0),
                        language: "rust".into(), file_order: 0,
                    };
                    self.nodes.push(node);
                    count += 1;
                } else if let Some(name) = Self::extract_impl(trimmed) {
                    let text = lines[i..].join("\n");
                    let (tm_match, errors) = self.predictor.feed_learn(&text);
                    let node = PhaseNode {
                        path: "_generated".to_string(),
                        line: i + 1, kind: "impl".into(), name,
                        tm_match, l0_err: errors.first().copied().unwrap_or(1.0),
                        l1_err: errors.get(1).copied().unwrap_or(1.0),
                        language: "rust".into(), file_order: 0,
                    };
                    self.nodes.push(node);
                    count += 1;
                }
            }
        }
        count
    }

    pub fn generate_code_beam(&mut self, text: &str, steps: usize, beam_width: usize, temperature: f64) -> Vec<Vec<(PhaseNode, u32)>> {
        let words: Vec<&str> = text.split_whitespace().collect();
        for chunk in words.chunks(10) {
            self.predictor.feed(&chunk.join(" "));
        }
        if self.predictor.buffer.len() < self.predictor.hjepa.levels[0].context_len {
            let pad_hv = Hypervector::random(self.predictor.hjepa.dim);
            while self.predictor.buffer.len() < self.predictor.hjepa.levels[0].context_len {
                self.predictor.buffer.push(pad_hv.clone());
            }
        }
        let ctx: Vec<&Hypervector> = self.predictor.buffer.iter().collect();
        let seqs = self.predictor.hjepa.predict_sequence_beam(&ctx, steps, beam_width, temperature);
        let best_seq = seqs.first().cloned().unwrap_or_default();
        let mut results = Vec::new();
        for pred in &best_seq {
            let pred_sdr = sparsify(pred);
            let mut scored: Vec<(PhaseNode, u32)> = self.nodes.iter()
                .map(|n| {
                    let nsdr = encode_text(&format!("{} {}", n.kind, n.name));
                    (n.clone(), pred_sdr.overlap(&nsdr))
                })
                .collect();
            scored.sort_by(|a, b| b.1.cmp(&a.1));
            scored.truncate(5);
            results.push(scored);
        }
        results
    }

    pub fn query(&mut self, text: &str) -> (f64, Vec<f64>, Vec<&PhaseNode>) {
        let (tm_match, errors) = self.predictor.feed_learn(text);
        let sdr = encode_text(text);
        let mut scored: Vec<(&PhaseNode, u32)> = self.nodes.iter()
            .map(|n| {
                let nsdr = encode_text(&format!("{} {}", n.kind, n.name));
                (n, sdr.overlap(&nsdr))
            })
            .collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1));
        let top: Vec<&PhaseNode> = scored.into_iter().take(5).map(|(n, _)| n).collect();
        (tm_match, errors, top)
    }

    pub fn reflect(&self) -> String {
        let total_err: f64 = self.nodes.iter().map(|n| n.l0_err + n.l1_err).sum::<f64>()
            / (self.nodes.len().max(1) as f64 * 2.0);
        let impls = self.nodes.iter().filter(|n| n.kind == "impl").count();
        let fns = self.nodes.iter().filter(|n| n.kind == "fn").count();
        let structs = self.nodes.iter().filter(|n| n.kind == "struct").count();
        format!("mirror: {} nodes ({}fn {}struct {}impl) avg_err={:.4} cells={}",
            self.nodes.len(), fns, structs, impls, total_err,
            self.predictor.tm.cells.len())
    }

    fn extract_fn(line: &str) -> Option<String> {
        let trimmed = line.trim();
        if trimmed.starts_with("fn ") && trimmed.contains('(') {
            let raw = trimmed.strip_prefix("fn ")?;
            let name = raw.split('(').next()?.trim();
            if !name.is_empty() && !name.contains(' ') {
                return Some(name.to_string());
            }
        }
        None
    }

    fn extract_struct(line: &str) -> Option<String> {
        let trimmed = line.trim();
        if trimmed.starts_with("pub struct ") || trimmed.starts_with("struct ") {
            let raw = if trimmed.starts_with("pub struct ") {
                trimmed.strip_prefix("pub struct ")?
            } else {
                trimmed.strip_prefix("struct ")?
            };
            let name = raw.split(|c: char| c.is_whitespace() || c == '{' || c == ';').next()?;
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
        None
    }

    fn extract_c_fn(line: &str) -> Option<String> {
        let trimmed = line.trim();
        let skip = [
            "if ", "while ", "for ", "switch ", "else ", "return ",
            "#", "//", "/*", "*", "typedef", "enum", "union",
        ];
        for s in &skip {
            if trimmed.starts_with(s) { return None; }
        }
        if !trimmed.contains('(') || !trimmed.contains(')') { return None; }
        if !trimmed.contains('{') && !trimmed.ends_with(')') { return None; }
        let before_paren = trimmed.split('(').next()?;
        let tokens: Vec<&str> = before_paren.split_whitespace().collect();
        if tokens.len() < 2 { return None; }
        let name = tokens[tokens.len() - 1];
        if name.contains('*') || name == ")" { return None; }
        if name.chars().all(|c| c.is_uppercase() || c == '_') && name.chars().any(|c| c == '_') { return None; }
        Some(name.trim_matches('*').to_string())
    }

    fn extract_c_struct(line: &str) -> Option<String> {
        let trimmed = line.trim();
        if trimmed.starts_with("typedef struct ") || trimmed.starts_with("struct ") {
            let raw = if trimmed.starts_with("typedef struct ") {
                trimmed.strip_prefix("typedef struct ")?
            } else {
                trimmed.strip_prefix("struct ")?
            };
            let name = raw.split_whitespace().next().filter(|n| !n.is_empty() && *n != "{")?;
            if name != "{" { Some(name.to_string()) } else { None }
        } else {
            None
        }
    }

    fn extract_macro(line: &str) -> Option<(String, String)> {
        let trimmed = line.trim();
        if trimmed.starts_with("#define ") {
            let raw = trimmed.strip_prefix("#define ")?;
            let name = raw.split_whitespace().next()?;
            let body = raw.splitn(2, |c: char| c.is_whitespace()).nth(1).unwrap_or("");
            Some((name.to_string(), body.to_string()))
        } else {
            None
        }
    }

    fn extract_ifdef(line: &str) -> Option<String> {
        let trimmed = line.trim();
        if trimmed.starts_with("#ifdef ") {
            trimmed.strip_prefix("#ifdef ").map(|s| s.trim().to_string())
        } else if trimmed.starts_with("#ifndef ") {
            trimmed.strip_prefix("#ifndef ").map(|s| s.trim().to_string())
        } else if trimmed.starts_with("#endif") {
            Some("#endif".to_string())
        } else if trimmed.starts_with("#else") {
            Some("#else".to_string())
        } else {
            None
        }
    }

    fn extract_impl(line: &str) -> Option<String> {
        let trimmed = line.trim();
        if trimmed.starts_with("impl ") {
            let raw = trimmed.strip_prefix("impl ")?;
            let name = raw.split_whitespace().next()?;
            if !name.is_empty() && name != "{" {
                return Some(name.to_string());
            }
        }
        None
    }

    fn extract_java_class(line: &str) -> Option<String> {
        let trimmed = line.trim();
        let keywords = ["class ", "interface ", "enum ", "@interface ", "record "];
        for kw in &keywords {
            if trimmed.contains(kw) {
                if let Some(idx) = trimmed.find(kw) {
                    let after = &trimmed[idx + kw.len()..];
                    let name = after.split(|c: char| c.is_whitespace() || c == '{' || c == '<' || c == '(' || c == ';').next()?;
                    if !name.is_empty() && name != "{" && name.chars().next()?.is_uppercase() {
                        return Some(name.to_string());
                    }
                }
            }
        }
        None
    }

    fn extract_java_method(line: &str) -> Option<String> {
        let trimmed = line.trim();
        let skip = ["if ", "while ", "for ", "switch ", "else ", "return ",
                     "//", "/*", "*", "import ", "package ", "@interface ", "@Override"];
        for s in &skip { if trimmed.starts_with(s) { return None; } }
        if !trimmed.contains('(') || !trimmed.contains(')') { return None; }
        // Skip annotation lines, they precede the actual method
        if trimmed.starts_with('@') { return None; }
        // Find the last word before '(' — it should be the method name
        let before_paren = trimmed.split('(').next()?;
        let tokens: Vec<&str> = before_paren.split_whitespace().collect();
        if tokens.len() < 2 { return None; }
        let name = tokens[tokens.len() - 1];
        let excludes = [")", "(", "{", "}", ";", ""];
        if excludes.contains(&name) { return None; }
        // Skip constructors (same name as class would be OK, but skip if starts with uppercase)
        if name.contains('=') || name.contains(';') || !name.chars().any(|c| c.is_alphabetic()) {
            return None;
        }
        // Must have an access modifier or be a static/abstract method
        let has_modifier = trimmed.starts_with("public ") || trimmed.starts_with("private ")
            || trimmed.starts_with("protected ") || trimmed.starts_with("static ")
            || trimmed.starts_with("abstract ") || tokens.contains(&"public")
            || tokens.contains(&"private") || tokens.contains(&"protected");
        if !has_modifier { return None; }
        Some(name.to_string())
    }

    pub fn is_c_cpp(path: &str) -> bool {
        path.ends_with(".c") || path.ends_with(".cpp") || path.ends_with(".cxx")
            || path.ends_with(".h") || path.ends_with(".hpp") || path.ends_with(".hxx")
            || path.ends_with(".cu") || path.ends_with(".cuh") || path.ends_with(".cc")
    }

    pub fn index_file_c(&mut self, path: &str) -> Vec<PhaseNode> {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };
        let mut nodes = Vec::new();
        let lines: Vec<&str> = content.lines().collect();
        let mut i = 0;
        let mut file_counter = 0usize;
        let plang = file_language(path);
        while i < lines.len() {
            let line = lines[i];
            let trimmed = line.trim();

            if let Some((name, body)) = Self::extract_macro(trimmed) {
                let (tm_match, errors) = self.predictor.feed_learn(&format!("#define {} {}", name, body));
                file_counter += 1;
                nodes.push(PhaseNode {
                    path: path.to_string(), line: i + 1,
                    kind: "macro".into(), name,
                    tm_match, l0_err: errors.first().copied().unwrap_or(1.0),
                    l1_err: errors.get(1).copied().unwrap_or(1.0),
                    language: plang.clone(), file_order: file_counter,
                });
                i += 1;
                continue;
            }

            if let Some(tag) = Self::extract_ifdef(trimmed) {
                let (tm_match, errors) = self.predictor.feed_learn(&format!("#preprocessor {}", tag));
                file_counter += 1;
                nodes.push(PhaseNode {
                    path: path.to_string(), line: i + 1,
                    kind: "preproc".into(), name: tag,
                    tm_match, l0_err: errors.first().copied().unwrap_or(1.0),
                    l1_err: errors.get(1).copied().unwrap_or(1.0),
                    language: plang.clone(), file_order: file_counter,
                });
                i += 1;
                continue;
            }

            if let Some(name) = Self::extract_c_struct(trimmed) {
                let mut block = String::new();
                let start = i;
                let mut brace = 1usize;
                block.push_str(lines[i]); block.push('\n');
                i += 1;
                while i < lines.len() && brace > 0 {
                    let l = lines[i];
                    brace += l.matches('{').count();
                    brace -= l.matches('}').count();
                    block.push_str(l); block.push('\n');
                    i += 1;
                }
                let (tm_match, errors) = self.predictor.feed_learn(&block);
                file_counter += 1;
                nodes.push(PhaseNode {
                    path: path.to_string(), line: start + 1,
                    kind: "struct".into(), name,
                    tm_match, l0_err: errors.first().copied().unwrap_or(1.0),
                    l1_err: errors.get(1).copied().unwrap_or(1.0),
                    language: plang.clone(), file_order: file_counter,
                });
                continue;
            }

            if let Some(name) = Self::extract_c_fn(trimmed) {
                if trimmed.ends_with(';') { i += 1; continue; }
                let mut block = String::new();
                let start = i;
                let mut brace = 0usize;
                let mut found_open = false;
                while i < lines.len() {
                    let l = lines[i];
                    block.push_str(l); block.push('\n');
                    brace += l.matches('{').count();
                    brace += l.matches('}').count();
                    if l.contains('{') { found_open = true; }
                    if found_open && brace == 0 { break; }
                    i += 1;
                }
                let (tm_match, errors) = self.predictor.feed_learn(&block);
                file_counter += 1;
                nodes.push(PhaseNode {
                    path: path.to_string(), line: start + 1,
                    kind: "fn".into(), name,
                    tm_match, l0_err: errors.first().copied().unwrap_or(1.0),
                    l1_err: errors.get(1).copied().unwrap_or(1.0),
                    language: plang.clone(), file_order: file_counter,
                });
                i += 1;
                continue;
            }

            i += 1;
        }
        self.cache.entry(path.to_string()).or_default().extend(nodes.clone());
        self.nodes.extend(nodes.clone());
        nodes
    }
}

#[derive(Clone, Debug)]
pub struct InspectReport {
    pub path: String,
    pub lines: usize,
    pub bytes: usize,
    pub entropy: f64,
    pub struct_entropy: f64,
    pub unique_ratio: f64,
    pub avg_line_len: f64,
    pub sdr_density: f64,
    pub top_match: String,
    pub top_overlap: f64,
    pub anomaly_score: f64,
}

fn chunks_from_lines(lines: &[&str], path: &str, is_c: bool, is_java: bool) -> Vec<RawChunk> {
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        if is_java {
            if let Some(ref name) = SelfMirror::extract_java_class(trimmed) {
                let mut block = String::new();
                let start = i; let mut brace = 1usize;
                block.push_str(lines[i]); block.push('\n'); i += 1;
                while i < lines.len() && brace > 0 {
                    let l = lines[i]; brace += l.matches('{').count();
                    brace -= l.matches('}').count(); block.push_str(l); block.push('\n'); i += 1;
                }
                chunks.push(RawChunk { path: path.to_string(), line: start + 1,
                    kind: "class".into(), name: name.clone(), text: block });
                continue;
            }
            if let Some(ref name) = SelfMirror::extract_java_method(trimmed) {
                if trimmed.ends_with(';') { i += 1; continue; }
                let mut block = String::new();
                let start = i; let mut brace = 0usize; let mut found_open = false;
                while i < lines.len() {
                    let l = lines[i]; block.push_str(l); block.push('\n');
                    brace += l.matches('{').count(); brace += l.matches('}').count();
                    if l.contains('{') { found_open = true; }
                    if found_open && brace == 0 { break; }
                    i += 1;
                }
                chunks.push(RawChunk { path: path.to_string(), line: start + 1,
                    kind: "fn".into(), name: name.clone(), text: block });
                i += 1; continue;
            }
        } else if is_c {
            if let Some((ref mname, ref body)) = SelfMirror::extract_macro(trimmed) {
                chunks.push(RawChunk {
                    path: path.to_string(), line: i + 1,
                    kind: "macro".into(), name: mname.clone(),
                    text: format!("#define {} {}", mname, body),
                });
                i += 1; continue;
            }
            if let Some(ref tag) = SelfMirror::extract_ifdef(trimmed) {
                chunks.push(RawChunk {
                    path: path.to_string(), line: i + 1,
                    kind: "preproc".into(), name: tag.clone(),
                    text: format!("#preproc {}", tag),
                });
                i += 1; continue;
            }
            if let Some(ref name) = SelfMirror::extract_c_struct(trimmed) {
                let mut block = String::new();
                let start = i; let mut brace = 1usize;
                block.push_str(lines[i]); block.push('\n'); i += 1;
                while i < lines.len() && brace > 0 {
                    let l = lines[i]; brace += l.matches('{').count();
                    brace -= l.matches('}').count(); block.push_str(l); block.push('\n'); i += 1;
                }
                chunks.push(RawChunk { path: path.to_string(), line: start + 1,
                    kind: "struct".into(), name: name.clone(), text: block });
                continue;
            }
            if let Some(ref name) = SelfMirror::extract_c_fn(trimmed) {
                if trimmed.ends_with(';') { i += 1; continue; }
                let mut block = String::new();
                let start = i; let mut brace = 0usize; let mut found_open = false;
                while i < lines.len() {
                    let l = lines[i]; block.push_str(l); block.push('\n');
                    brace += l.matches('{').count(); brace += l.matches('}').count();
                    if l.contains('{') { found_open = true; }
                    if found_open && brace == 0 { break; }
                    i += 1;
                }
                chunks.push(RawChunk { path: path.to_string(), line: start + 1,
                    kind: "fn".into(), name: name.clone(), text: block });
                i += 1; continue;
            }
        } else {
            if let Some(ref name) = SelfMirror::extract_fn(trimmed) {
                let mut block = String::new(); let start = i;
                while i < lines.len() && !lines[i].trim().starts_with('}') {
                    block.push_str(lines[i]); block.push('\n'); i += 1;
                }
                if i < lines.len() { block.push_str(lines[i]); }
                chunks.push(RawChunk { path: path.to_string(), line: start + 1,
                    kind: "fn".into(), name: name.clone(), text: block });
                i += 1; continue;
            }
            if let Some(ref name) = SelfMirror::extract_struct(trimmed) {
                let mut block = String::new(); let start = i;
                while i < lines.len() && !lines[i].trim().starts_with('}') {
                    block.push_str(lines[i]); block.push('\n'); i += 1;
                }
                if i < lines.len() { block.push_str(lines[i]); }
                chunks.push(RawChunk { path: path.to_string(), line: start + 1,
                    kind: "struct".into(), name: name.clone(), text: block });
                i += 1; continue;
            }
            if let Some(ref name) = SelfMirror::extract_impl(trimmed) {
                let mut block = String::new(); let start = i; let mut brace = 1usize;
                block.push_str(lines[i]); block.push('\n'); i += 1;
                while i < lines.len() && brace > 0 {
                    let l = lines[i]; brace += l.matches('{').count();
                    brace -= l.matches('}').count(); block.push_str(l); block.push('\n'); i += 1;
                }
                chunks.push(RawChunk { path: path.to_string(), line: start + 1,
                    kind: "impl".into(), name: name.clone(), text: block });
                continue;
            }
        }
        i += 1;
    }
    chunks
}

impl SelfMirror {
    pub fn is_java(path: &str) -> bool {
        path.ends_with(".java")
    }

    pub fn scan_file(path: &str) -> Vec<RawChunk> {
        let content = std::fs::read_to_string(path).unwrap_or_default();
        let lines: Vec<&str> = content.lines().collect();
        let is_c = Self::is_c_cpp(path);
        let is_java = Self::is_java(path);
        chunks_from_lines(&lines, path, is_c, is_java)
    }

    pub fn scan_dir(dir: &str) -> Vec<RawChunk> {
        let mut all = Vec::new();
        let walk = match std::fs::read_dir(dir) {
            Ok(d) => d,
            Err(_) => return all,
        };
        let mut entries: Vec<_> = walk.filter_map(|e| e.ok()).collect();
        entries.sort_by_key(|e| e.path());
        for entry in entries {
            let p = entry.path();
            let ps = p.to_str().unwrap_or("");
            let ext = p.extension().map(|e| e.to_str().unwrap_or("")).unwrap_or("");
            if p.is_dir() {
                let dname = p.file_name().unwrap_or_default().to_str().unwrap_or("");
                if dname == ".venv" || dname == "target" || dname == "node_modules" || dname == ".git" { continue; }
                all.extend(Self::scan_dir(ps));
            } else if ext == "rs" || Self::is_c_cpp(ps) || Self::is_java(ps) {
                let chunks = Self::scan_file(ps);
                println!("  {} → {} chunks", p.display(), chunks.len());
                all.extend(chunks);
            }
        }
        all
    }

    pub fn train_predictor(&mut self, epochs: usize) -> f64 {
        self.train_predictor_chunked(epochs, 10)
    }

    pub fn train_predictor_chunked(&mut self, epochs: usize, chunk_size: usize) -> f64 {
        self.train_predictor_inner(epochs, chunk_size, false, false)
    }

    pub fn train_predictor_fast(&mut self, epochs: usize, chunk_size: usize) -> f64 {
        self.train_predictor_inner(epochs, chunk_size, false, true)
    }

    pub fn train_predictor_ff(&mut self, epochs: usize, chunk_size: usize) -> f64 {
        self.train_predictor_inner(epochs, chunk_size, true, false)
    }

    fn train_predictor_inner(&mut self, epochs: usize, chunk_size: usize, use_ff: bool, skip_tm: bool) -> f64 {
        let paths: BTreeSet<String> = self.nodes.iter().map(|n| n.path.clone()).collect();
        let count = paths.len();
        let cs = chunk_size.max(1);
        println!("  Re-training HJEPA on {} files ({} epochs, chunk_size={}, ff={})...", count, epochs, cs, use_ff);
        let mut files: Vec<String> = paths.into_iter().collect();
        files.sort();
        let mut est_chunks = 0usize;
        for fp in &files {
            if let Ok(c) = std::fs::read_to_string(fp) {
                let wc = c.split_whitespace().count().max(1);
                est_chunks += (wc + cs - 1) / cs;
            }
        }
        println!("  Estimated chunks per epoch: {} (total across {} epochs: {})", est_chunks, epochs, est_chunks * epochs);
        let mut total_loss = 0.0f64;
        let mut total_cnt = 0usize;
        let mut neg_pool: Vec<Hypervector> = Vec::new();

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0f64;
            let mut epoch_cnt = 0usize;

            if skip_tm {
                let pre_w0: f64 = self.predictor.hjepa.levels[0].weights.iter().sum();
                let pre_w1: f64 = self.predictor.hjepa.levels[1].weights.iter().sum();
                let pre_w2: f64 = self.predictor.hjepa.levels[2].weights.iter().sum();
                println!("  Pre-train weights: w0={:.2} w1={:.2} w2={:.2}", pre_w0, pre_w1, pre_w2);
                for (fi, fp) in files.iter().enumerate() {
                    let content = match std::fs::read_to_string(fp) {
                        Ok(c) => c,
                        Err(_) => continue,
                    };
                    let words: Vec<&str> = content.split_whitespace().collect();
                    for chunk in words.chunks(cs) {
                        let errors = self.predictor.feed_learn_hv_only(&chunk.join(" "));
                        for &e in &errors {
                            epoch_loss += e;
                        }
                        epoch_cnt += errors.len();
                    }
                    if (fi + 1) % 20 == 0 {
                        print!("\r  Fast: {}/{} files  chunks={}  loss={:.4}", fi + 1, 200, epoch_cnt, epoch_loss / epoch_cnt.max(1) as f64);
                        use std::io::{Write, stdout};
                        stdout().flush().ok();
                    }
                }
                let post_w0: f64 = self.predictor.hjepa.levels[0].weights.iter().sum();
                let post_w1: f64 = self.predictor.hjepa.levels[1].weights.iter().sum();
                let post_w2: f64 = self.predictor.hjepa.levels[2].weights.iter().sum();
                println!("\r  Trained: {} files, {} chunks  loss={:.4}  w=[{:.2} {:.2} {:.2}] → [{:.2} {:.2} {:.2}] Δ0={:.4}",
                    200, epoch_cnt, epoch_loss / epoch_cnt.max(1) as f64,
                    pre_w0, pre_w1, pre_w2, post_w0, post_w1, post_w2, post_w0 - pre_w0);
            } else {
                // Original sequential loop (non-fast mode)
                for (fi, fp) in files.iter().enumerate() {
                    let content = match std::fs::read_to_string(fp) {
                        Ok(c) => c,
                        Err(_) => continue,
                    };
                    let words: Vec<&str> = content.split_whitespace().collect();
                    for chunk in words.chunks(cs) {
                        let errors = if use_ff {
                            self.predictor.feed_learn_ff(&chunk.join(" "), &neg_pool)
                        } else {
                            self.predictor.feed_learn_no_tm(&chunk.join(" "))
                        };
                        for &e in &errors {
                            epoch_loss += e;
                        }
                        epoch_cnt += errors.len();
                        if use_ff && !self.predictor.buffer.is_empty() {
                            neg_pool.push(self.predictor.buffer.last().unwrap().clone());
                            if neg_pool.len() > 200 { neg_pool.remove(0); }
                        }
                    }
                    if (fi + 1) % 50 == 0 {
                        print!("\r  Epoch {:3}/{}  file {}/{}  chunks={}", epoch + 1, epochs, fi + 1, files.len(), epoch_cnt);
                        use std::io::{Write, stdout};
                        stdout().flush().ok();
                    }
                }
            }

            let avg = epoch_loss / epoch_cnt.max(1) as f64;
            total_loss += avg;
            total_cnt += 1;
            print!("\r  Epoch {:3}/{}  chunk_size={} avg_loss={:.4}", epoch + 1, epochs, cs, avg);
            use std::io::{Write, stdout};
            stdout().flush().ok();
            self.save();
            print!(" [saved]");
            stdout().flush().ok();
        }
        let final_avg = total_loss / total_cnt.max(1) as f64;
        println!("\n  Done. Final avg loss: {:.4}", final_avg);
        final_avg
    }

    pub fn checkpoint(&self, tag: &str) {
        let tm_path = format!("fuga_mirror_tm_{}.bin", tag);
        let jepa_path = format!("fuga_mirror_jepa_{}.bin", tag);
        let node_path = format!("fuga_mirror_nodes_{}.bin", tag);
        self.predictor.tm.save(&tm_path);
        let _ = self.predictor.hjepa.save(&jepa_path);
        use std::io::Write;
        if let Ok(mut f) = std::fs::File::create(&node_path) {
            let n = self.nodes.len() as u32;
            f.write_all(&n.to_le_bytes()).ok();
            for node in &self.nodes {
                let pb = node.path.as_bytes();
                f.write_all(&(pb.len() as u32).to_le_bytes()).ok();
                f.write_all(pb).ok();
                f.write_all(&(node.line as u32).to_le_bytes()).ok();
                let kb = node.kind.as_bytes();
                f.write_all(&(kb.len() as u32).to_le_bytes()).ok();
                f.write_all(kb).ok();
                let nb = node.name.as_bytes();
                f.write_all(&(nb.len() as u32).to_le_bytes()).ok();
                f.write_all(nb).ok();
                f.write_all(&node.tm_match.to_le_bytes()).ok();
                f.write_all(&node.l0_err.to_le_bytes()).ok();
                f.write_all(&node.l1_err.to_le_bytes()).ok();
                let lb = node.language.as_bytes();
                f.write_all(&(lb.len() as u32).to_le_bytes()).ok();
                f.write_all(lb).ok();
                f.write_all(&(node.file_order as u64).to_le_bytes()).ok();
            }
        }
    }

    pub fn train_from_chunks(&mut self, chunks: &[RawChunk]) -> usize {
        let total = chunks.len();
        let mut file_counters: HashMap<String, usize> = HashMap::new();
        for chunk in chunks {
            let name = &chunk.name;
            if !self.name_sdr_cache.contains_key(name) {
                self.name_sdr_cache.insert(name.clone(), encode_text(name));
            }
        }
        for (idx, chunk) in chunks.iter().enumerate() {
            let (tm_match, errors) = self.predictor.feed_learn(&chunk.text);
            let order = file_counters.entry(chunk.path.clone()).or_insert(0);
            *order += 1;
            let node = PhaseNode {
                path: chunk.path.clone(), line: chunk.line,
                kind: chunk.kind.clone(), name: chunk.name.clone(),
                tm_match, l0_err: errors.first().copied().unwrap_or(1.0),
                l1_err: errors.get(1).copied().unwrap_or(1.0),
                language: file_language(&chunk.path),
                file_order: *order,
            };
            self.cache.entry(chunk.path.clone()).or_default().push(node.clone());
            self.nodes.push(node);
            if (idx + 1) % 500 == 0 {
                let pct = (idx + 1) as f64 / total as f64 * 100.0;
                let cells = self.predictor.tm.cells.len();
                let segs: usize = self.predictor.tm.cells.iter().map(|c| c.segments.len()).sum();
                println!("  trained {}/{} ({:.0}%)  cells={} segments={}  (checkpoint)", idx + 1, total, pct, cells, segs);
                self.checkpoint(&format!("{}", idx + 1));
            } else if (idx + 1) % 100 == 0 || idx == total - 1 {
                let pct = (idx + 1) as f64 / total as f64 * 100.0;
                let cells = self.predictor.tm.cells.len();
                let segs: usize = self.predictor.tm.cells.iter().map(|c| c.segments.len()).sum();
                println!("  trained {}/{} ({:.0}%)  cells={} segments={}", idx + 1, total, pct, cells, segs);
            }
        }
        total
    }

    pub fn index_dir_fast(&mut self, dir: &str) -> usize {
        println!("  Phase 1: scanning files...");
        let chunks = Self::scan_dir(dir);
        println!("\n  {} raw chunks extracted", chunks.len());
        println!("  Phase 2: training TM + H-JEPA...\n");
        let total = self.train_from_chunks(&chunks);
        println!("\n  Done: {} phase nodes", total);
        total
    }

    pub fn inspect_file(&self, path: &str) -> InspectReport {
        let content = std::fs::read_to_string(path).unwrap_or_default();
        let lines = content.lines().count();
        let bytes = content.len();
        let profile = crate::ai::anomaly::StyloProfile::compute(&content);
        let sdr = encode_text(&content);
        let pop = sdr.popcount();
        let sdr_density = pop as f64 / 8192.0;

        let mut top_match = String::new();
        let mut top_overlap = 0.0f64;
        if !self.nodes.is_empty() {
            let mut best = 0u32;
            let mut best_name = String::new();
            for n in &self.nodes {
                let nsdr = encode_text(&format!("{} {}", n.kind, n.name));
                let o = sdr.overlap(&nsdr);
                if o > best {
                    best = o;
                    best_name = format!("{}::{}", n.kind, n.name);
                }
            }
            top_match = best_name;
            top_overlap = best as f64 / 8192.0;
        }

        let anomaly_score = 1.0 - (profile.token_entropy / 10.0).min(1.0) + (1.0 - sdr_density / 0.02).abs();

        InspectReport {
            path: path.to_string(), lines, bytes,
            entropy: profile.token_entropy,
            struct_entropy: profile.structural_entropy,
            unique_ratio: profile.unique_ratio,
            avg_line_len: profile.avg_line_len,
            sdr_density, top_match, top_overlap, anomaly_score,
        }
    }

    pub fn inspect_dir(&self, dir: &str) -> Vec<InspectReport> {
        let mut reports = Vec::new();
        let walk = match std::fs::read_dir(dir) {
            Ok(d) => d,
            Err(_) => return reports,
        };
        let mut entries: Vec<_> = walk.filter_map(|e| e.ok()).collect();
        entries.sort_by_key(|e| e.path());
        for entry in entries {
            let p = entry.path();
            let ps = p.to_str().unwrap_or("");
            let ext = p.extension().map(|e| e.to_str().unwrap_or("")).unwrap_or("");
            if ext == "rs" || Self::is_c_cpp(ps) || Self::is_java(ps) {
                let report = self.inspect_file(ps);
                println!("  {}  entropy={:.2} struct_entropy={:.2} density={:.3} anomaly={:.3}  → {}",
                    p.display(), report.entropy, report.struct_entropy,
                    report.sdr_density, report.anomaly_score, report.top_match);
                reports.push(report);
            } else if p.is_dir() {
                reports.extend(self.inspect_dir(ps));
            }
        }
        reports
    }

    pub fn evaluate_debug(&mut self) -> String {
        let mut sim_pred = 0.0f64;
        let mut sim_baseline = 0.0f64;
        let mut cnt = 0usize;
        let paths: Vec<String> = self.nodes.iter().map(|n| n.path.clone()).collect::<BTreeSet<_>>().into_iter().collect();
        let max_chunks = 500usize;
        for fp in &paths {
            if cnt >= max_chunks { break; }
            let content = match std::fs::read_to_string(fp) { Ok(c) => c, Err(_) => continue };
            for chunk in content.split_whitespace().collect::<Vec<_>>().chunks(10) {
                if cnt >= max_chunks { break; }
                let text = chunk.join(" ");
                let sdr = encode_text(&text);
                let (tm_pred, _) = self.predictor.tm.feed(&sdr);
                let hv = crate::ai::temporal_predictor::sdr_to_hypervector(&tm_pred, self.predictor.hjepa.dim);
                self.predictor.buffer.push(hv);
                if self.predictor.buffer.len() > self.predictor.buf_capacity { self.predictor.buffer.remove(0); }
                if self.predictor.buffer.len() <= self.predictor.hjepa.levels[0].context_len { continue; }
                let ctx: Vec<&Hypervector> = self.predictor.buffer[..self.predictor.buffer.len()-1].iter().collect();
                let actual = self.predictor.buffer.last().unwrap();
                let baseline = ctx[ctx.len()-1];
                let pred = self.predictor.hjepa.levels[0].predict(&ctx);
                sim_pred += pred.similarity(actual);
                sim_baseline += baseline.similarity(actual);
                cnt += 1;
            }
        }
        format!("DBG sim(pred,actual)={:.4} sim(baseline,actual)={:.4} n={}", sim_pred/cnt as f64, sim_baseline/cnt as f64, cnt)
    }
}

pub struct AutoCorrectSuggestion {
    pub node: PhaseNode,
    pub resonance: f64,
}

pub struct AutoCorrectEngine {
    pub mirror: SelfMirror,
    pub correction_history: Vec<(String, String, f64)>,
}

impl AutoCorrectEngine {
    pub fn new(mirror: SelfMirror) -> Self {
        AutoCorrectEngine {
            mirror,
            correction_history: Vec::new(),
        }
    }

    pub fn load() -> Option<Self> {
        SelfMirror::load().map(AutoCorrectEngine::new)
    }

    pub fn suggest_correction(&mut self, text: &str) -> Option<AutoCorrectSuggestion> {
        let (_tm_match, errors, top) = self.mirror.query(text);
        let l2_err = *errors.get(2).unwrap_or(&1.0);
        if l2_err < 0.57 {
            return None;
        }
        top.first().map(|&node| AutoCorrectSuggestion {
            node: node.clone(),
            resonance: 1.0 - l2_err,
        })
    }

    pub fn apply_correction(&mut self, text: &str) -> (String, Vec<String>) {
        let mut patches = Vec::new();
        let suggestion = self.suggest_correction(text);
        if let Some(sugg) = suggestion {
            patches.push(format!(
                "# L2 divergence → mirror match: {}::{} (line {})\n# resonance={:.3}",
                sugg.node.path, sugg.node.name, sugg.node.line, sugg.resonance
            ));
            patches.push(format!(
                "// suggested pattern from phase graph\n// original: {}\n// mirror: {} {}",
                text, sugg.node.kind, sugg.node.name
            ));
            self.correction_history.push((text.to_string(), sugg.node.name.clone(), sugg.resonance));
        }
        (text.to_string(), patches)
    }

    pub fn stats(&self) -> String {
        format!("corrections={} history={}", self.mirror.reflect(), self.correction_history.len())
    }
}

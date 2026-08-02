use super::memory_store::MemoryStore;
use super::moe::MoEStore;
use super::resonance_attention::{AttentionCell, ResonanceAttention};
use super::router::{DynamicRouter, ExpertConfig, TargetExpert};
use crate::core::hypervector::Hypervector;
use crate::core::wave_cube::WaveCube;
use crate::weaver::WeaverEngine;
use crate::weaver::pattern_matcher::TokenInfo;
use crate::weaver::super_token::{SuperToken, TokenRole};
use crate::weaver::vocabulary::TokenVocabulary;
use std::collections::{HashMap, HashSet};
use std::path::Path;

fn extract_code_signatures(tokens: &[TokenInfo]) -> Vec<String> {
    let mut sigs = Vec::new();
    let type_kws = [
        "fn", "func", "def", "pub", "impl", "trait", "struct", "enum", "int", "void", "char",
        "double", "float", "long", "short", "unsigned", "size_t", "static", "inline", "const",
        "extern", "bool", "union", "signed",
    ];

    let words: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();

    for (i, w) in words.iter().enumerate() {
        let kw = w.to_lowercase();
        let is_type = type_kws.contains(&kw.as_str());
        let self_has_paren = kw.contains('(');

        if !is_type && !self_has_paren {
            continue;
        }

        let has_prev_type = is_type
            || (i > 0 && type_kws.contains(&words[i - 1].to_lowercase().as_str()))
            || (i > 1 && type_kws.contains(&words[i - 2].to_lowercase().as_str()));

        if !has_prev_type {
            continue;
        }

        let start = if is_type { i } else { i.saturating_sub(2) };

        let mut sig = String::new();
        let mut found_paren = false;
        for j in start..words.len().min(start + 9) {
            let tok = words[j];
            if tok == "{" {
                break;
            }
            if sig.len() > 70 {
                break;
            }
            sig.push(' ');
            sig.push_str(tok);
            if tok.contains('(') {
                found_paren = true;
            }
        }
        let trimmed = sig.trim().to_string();
        let lower = trimmed.to_lowercase();

        if trimmed.is_empty() || !found_paren {
            continue;
        }
        if lower.starts_with("if ")
            || lower.starts_with("while ")
            || lower.starts_with("for ")
            || lower.starts_with("switch ")
            || lower.starts_with("return ")
        {
            continue;
        }
        if lower.contains("copyright") || lower.contains("license") {
            continue;
        }
        if trimmed.starts_with('(') || trimmed.starts_with('#') {
            continue;
        }
        sigs.push(trimmed);
    }
    let mut seen = std::collections::HashSet::new();
    sigs.retain(|s| {
        let name = s.split('(').next().unwrap_or("").trim().to_string();
        if name.is_empty() || name.len() > 30 {
            return true;
        }
        if !seen.insert(name) {
            return false;
        }
        true
    });
    sigs.truncate(20);
    sigs
}

pub struct FugaAI<const N: usize, const S: usize> {
    pub weaver: WeaverEngine,
    pub attention: ResonanceAttention,
    pub cube: WaveCube<N, S>,
    pub vocab: Option<TokenVocabulary>,
    pub memory: MemoryStore,
    pub moe: MoEStore,
    pub expert_config: ExpertConfig,
    pub dim: usize,
    pub df_map: HashMap<u32, u32>,
    pub total_docs: u32,
    pub idf_weights: HashMap<u32, f64>,
}

pub struct AIOutput {
    pub super_tokens: Vec<SuperToken>,
    pub attention_map: Vec<AttentionCell>,
    pub route: TargetExpert,
    pub response_tokens: Option<Vec<TokenInfo>>,
    pub cube_entropy: f64,
    pub cube_coherence: f64,
}

impl<const N: usize, const S: usize> FugaAI<N, S> {
    pub fn new(dim: usize, window: usize) -> Self {
        Self {
            weaver: WeaverEngine::new(dim, window),
            attention: ResonanceAttention::new(dim),
            cube: WaveCube::<N, S>::new(dim),
            vocab: None,
            memory: MemoryStore::new(),
            moe: MoEStore::new("fuga_code_cube"),
            expert_config: ExpertConfig::default(),
            dim,
            df_map: HashMap::new(),
            total_docs: 0,
            idf_weights: HashMap::new(),
        }
    }

    pub fn set_vocab(&mut self, vocab: TokenVocabulary) {
        self.vocab = Some(vocab);
    }

    pub fn think(&mut self, tokens: &[TokenInfo]) -> AIOutput {
        let idf = if self.idf_weights.is_empty() {
            None
        } else {
            Some(&self.idf_weights)
        };
        let result = self.weaver.compress_stream(tokens, idf);

        let mut all_attention = Vec::new();
        for st in &result.super_tokens {
            let map = self.attention.beam_attention(st, &self.cube, 8);
            all_attention.extend(map);

            let route = if result.super_tokens.len() == 1 {
                DynamicRouter::route(st)
            } else {
                DynamicRouter::route_by_peek(&tokens.first().map(|t| t.text.as_str()).unwrap_or(""))
            };

            let resonance = all_attention.first().map(|c| c.score).unwrap_or(0.0);
            if resonance > self.expert_config.activation_threshold {
                let top_cell = &all_attention[0];
                self.attention.write_attention(
                    &mut self.cube.clone(),
                    top_cell.x,
                    top_cell.y,
                    top_cell.z,
                    top_cell.w,
                    top_cell.v,
                    &st.vector,
                );
            }

            if route == TargetExpert::MemoryWrite {
                let mut coords = [0; N];
                coords[0] = st.raw_tokens.len() % S;
                coords[1] = st.token_count % S;
                if N > 2 {
                    coords[2] = (route as usize) % S;
                }
                let existing = self.cube.cell_at(&coords);
                let bound = existing.bind(&st.vector);
                self.cube.write_at(&coords, &bound);
            }
        }

        all_attention.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        all_attention.truncate(16);

        let primary_route =
            DynamicRouter::route_by_peek(&tokens.first().map(|t| t.text.as_str()).unwrap_or(""));

        let entropy = self.cube.global_entropy();
        let coherence = self.cube.coherence();

        AIOutput {
            super_tokens: result.super_tokens,
            attention_map: all_attention,
            route: primary_route,
            response_tokens: None,
            cube_entropy: entropy,
            cube_coherence: coherence,
        }
    }

    pub fn think_with_response(&mut self, tokens: &[TokenInfo]) -> AIOutput {
        let mut output = self.think(tokens);

        if let Some(vocab) = &self.vocab {
            let ids: HashSet<u32> = tokens.iter().map(|t| t.id).collect();
            let unweave = self
                .weaver
                .unweave_stream_filtered(&output.super_tokens, vocab, &ids);
            output.response_tokens = Some(unweave.recovered_tokens);
        }

        output
    }

    pub fn absorb_knowledge(&mut self, super_tokens: &[SuperToken]) {
        for (i, st) in super_tokens.iter().enumerate() {
            let mut coords = [0; N];
            let mut tmp = i;
            for d in 0..N {
                coords[d] = tmp % S;
                tmp /= S;
            }
            let existing = self.cube.cell_at(&coords);
            let bound = existing.bind(&st.vector);
            self.cube.write_at(&coords, &bound);

            let role = if st.role_flags.bits() & TokenRole::CODE_CHUNK.bits() != 0 {
                "code"
            } else if st.role_flags.bits() & TokenRole::MATH_EXPR.bits() != 0 {
                "math"
            } else {
                "general"
            };

            let ms = S / 2;
            let role_idx = match role {
                "code" => ms,
                "math" => ms + 1,
                _ => ms.saturating_sub(1),
            };
            if role_idx < S {
                let mut role_coords = [ms; N];
                role_coords[1] = role_idx;
                let existing_role = self.cube.cell_at(&role_coords);
                let bound_role = existing_role.bind(&st.vector);
                self.cube.write_at(&role_coords, &bound_role);
            }
        }

        for d in 0..N {
            self.cube.wave_flow(d, 1);
        }
    }

    pub fn batch_absorb_knowledge(&mut self, super_tokens: &[Vec<SuperToken>]) {
        for (batch_i, batch) in super_tokens.iter().enumerate() {
            for (i, st) in batch.iter().enumerate() {
                let mut coords = [0; N];
                let idx = batch_i + i;
                let mut tmp = idx;
                for d in 0..N {
                    coords[d] = tmp % S;
                    tmp /= S;
                }
                let existing = self.cube.cell_at(&coords);
                let bound = existing.bind(&st.vector);
                self.cube.write_at(&coords, &bound);

                let ms = S / 2;
                let role_idx = if st.role_flags.bits() & TokenRole::CODE_CHUNK.bits() != 0 {
                    ms
                } else if st.role_flags.bits() & TokenRole::MATH_EXPR.bits() != 0 {
                    ms + 1
                } else {
                    ms.saturating_sub(1)
                };
                if role_idx < S {
                    let mut role_coords = [ms; N];
                    role_coords[1] = role_idx;
                    let existing_role = self.cube.cell_at(&role_coords);
                    let bound_role = existing_role.bind(&st.vector);
                    self.cube.write_at(&role_coords, &bound_role);
                }
            }
        }

        for d in 0..N {
            self.cube.wave_flow(d, 1);
        }
    }

    pub fn query_memory(&self, st: &SuperToken) -> Vec<AttentionCell> {
        self.attention.beam_attention(st, &self.cube, 8)
    }

    pub fn generate_concept(&mut self, concept: &str) -> AIOutput {
        let token = TokenInfo {
            id: 0,
            text: concept.to_string(),
        };
        self.think(&[token])
    }

    pub fn absorb_with_source(&mut self, tokens: &[TokenInfo], source_doc: &str) {
        let output = self.think(tokens);
        let role_hint = output.route.name();
        for st in &output.super_tokens {
            let para_text: String = tokens
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            self.memory.store(st, &para_text, source_doc, role_hint);
        }
        if !output.super_tokens.is_empty() {
            self.cube_absorb(&output.super_tokens);
        }
    }

    pub fn absorb_with_quality(
        &mut self,
        tokens: &[TokenInfo],
        source_doc: &str,
        quality: &crate::quality_filter::QualityScore,
        source_text: &str,
    ) -> bool {
        if quality.weight <= 0.0 {
            return false;
        }
        let output = self.think(tokens);
        let role_hint = output.route.name();
        let absorb_count = ((output.super_tokens.len() as f64) * quality.weight).ceil() as usize;
        let sigs = extract_code_signatures(tokens);
        let _fname = Path::new(source_doc)
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();
        let code_text = if source_text.len() > 1600 {
            let cutoff = source_text.floor_char_boundary(1600);
            format!(
                "{}…\n// {} {}",
                &source_text[..cutoff],
                source_doc,
                sigs.join("; ")
            )
        } else {
            format!("{}\n// {} {}", source_text, source_doc, sigs.join("; "))
        };
        for (_i, st) in output.super_tokens.iter().enumerate().take(absorb_count) {
            self.memory.store(st, &code_text, source_doc, role_hint);
        }
        if !output.super_tokens.is_empty() && quality.weight > 0.3 {
            self.cube_absorb(&output.super_tokens);
        }
        true
    }

    pub fn cube_absorb(&mut self, super_tokens: &[SuperToken]) {
        for (i, st) in super_tokens.iter().enumerate() {
            let mut coords = [0; N];
            let mut tmp = i;
            for d in 0..N {
                coords[d] = tmp % S;
                tmp /= S;
            }
            let existing = self.cube.cell_at(&coords);
            let bound = existing.bind(&st.vector);
            self.cube.write_at(&coords, &bound);

            let ms = S / 2;
            let role_idx = if st.role_flags.bits() & TokenRole::CODE_CHUNK.bits() != 0 {
                ms
            } else if st.role_flags.bits() & TokenRole::MATH_EXPR.bits() != 0 {
                ms + 1
            } else {
                ms.saturating_sub(1)
            };
            if role_idx < S {
                let mut role_coords = [ms; N];
                role_coords[1] = role_idx;
                let existing_role = self.cube.cell_at(&role_coords);
                let bound_role = existing_role.bind(&st.vector);
                self.cube.write_at(&role_coords, &bound_role);
            }
        }
        for d in 0..N {
            self.cube.wave_flow(d, 1);
        }
    }

    pub fn accumulate_df(&mut self, tokens: &[TokenInfo]) {
        let mut seen = HashSet::new();
        self.total_docs += 1;
        for t in tokens {
            if seen.insert(t.id) {
                *self.df_map.entry(t.id).or_insert(0) += 1;
            }
        }
    }

    pub fn compute_idf(&mut self) {
        if self.total_docs == 0 {
            return;
        }
        self.idf_weights.clear();
        let n = self.total_docs as f64;
        for (&id, &df) in &self.df_map {
            let idf = 1.0 + (n / df as f64).ln();
            self.idf_weights.insert(id, idf);
        }
    }

    pub fn build_moe_from_memory(&mut self) {
        for entry in self.memory.all_entries() {
            let st = SuperToken::new(entry.vector.clone(), 0);
            self.moe
                .store(&st, &entry.text, &entry.source_doc, &entry.role_hint);
        }
    }

    pub fn answer(&mut self, query: &str) -> String {
        let tokens: Vec<TokenInfo> = query
            .split_whitespace()
            .enumerate()
            .map(|(_, w)| TokenInfo {
                id: crate::weaver::token_id(&w),
                text: w.to_string(),
            })
            .collect();

        let output = self.think(&tokens);

        if self.memory.size() == 0 {
            return format!(
                "No knowledge stored. Route: {}, entropy: {:.4}",
                output.route.name(),
                output.cube_entropy
            );
        }

        let mut result = String::new();
        result.push_str(&format!("Answer for: {:?}\n", query));
        result.push_str(&format!("Route: {}\n", output.route.name()));
        result.push('\n');

        let mut seen_texts = std::collections::HashSet::new();

        for st in &output.super_tokens {
            let results = self.memory.search(&st.vector, 3);
            if results.is_empty() {
                let ctx = self.memory.retrieve_context(&st.vector, 2);
                if !ctx.is_empty() && !seen_texts.contains(&ctx) {
                    seen_texts.insert(ctx.clone());
                    result.push_str(&ctx);
                    result.push_str("\n\n");
                }
            }
            for (_idx, sim, entry) in &results {
                if seen_texts.contains(&entry.text) {
                    continue;
                }
                seen_texts.insert(entry.text.clone());
                result.push_str(&format!("[{}] (sim={:.3})\n", entry.source_doc, sim));
                result.push_str(&entry.text);
                result.push('\n');
            }
        }

        if seen_texts.is_empty() {
            let text_results = self.memory.search_by_text(query, 3);
            for (_idx, _sim, entry) in &text_results {
                if seen_texts.contains(&entry.text) {
                    continue;
                }
                seen_texts.insert(entry.text.clone());
                result.push_str(&format!("{}(text): {}\n", entry.source_doc, entry.text));
            }
        }
        result
    }

    pub fn answer_from_output(&self, output: &AIOutput, query: &str) -> String {
        if self.memory.size() == 0 {
            return format!(
                "No knowledge stored. Route: {}, entropy: {:.4}",
                output.route.name(),
                output.cube_entropy
            );
        }

        let mut result = String::new();
        result.push_str(&format!("Answer for: {:?}\n", query));
        result.push_str(&format!("Route: {}\n", output.route.name()));
        result.push('\n');

        let mut seen_texts = std::collections::HashSet::new();

        for st in &output.super_tokens {
            let results = self.memory.search(&st.vector, 3);
            if results.is_empty() {
                let ctx = self.memory.retrieve_context(&st.vector, 2);
                if !ctx.is_empty() && !seen_texts.contains(&ctx) {
                    seen_texts.insert(ctx.clone());
                    result.push_str(&ctx);
                    result.push_str("\n\n");
                }
            }
            for (_idx, sim, entry) in &results {
                if seen_texts.contains(&entry.text) {
                    continue;
                }
                seen_texts.insert(entry.text.clone());
                result.push_str(&format!("[{}] (sim={:.3})\n", entry.source_doc, sim));
                result.push_str(&entry.text);
                result.push('\n');
            }
        }

        if seen_texts.is_empty() {
            let text_results = self.memory.search_by_text(query, 3);
            for (_idx, _sim, entry) in &text_results {
                if seen_texts.contains(&entry.text) {
                    continue;
                }
                seen_texts.insert(entry.text.clone());
                result.push_str(&format!("{}{}\n", entry.source_doc, entry.text));
            }
        }
        result
    }

    pub fn answer_from_moe(&self, query: &str) -> String {
        let domain = MoEStore::domain_for(query);
        let result = self.moe.search_by_text(domain, query, 3);
        let mut out = String::new();
        for (_idx, _sim, entry) in &result {
            out.push_str(&format!("{}: {}\n", entry.source_doc, entry.text));
        }
        if out.is_empty() {
            let all = self.moe.search_all_by_text(query, 3);
            for (_idx, _sim, entry, _dom) in &all {
                out.push_str(&format!("{}: {}\n", entry.source_doc, entry.text));
            }
        }
        out
    }

    pub fn solve(&mut self, problem: &str) -> String {
        let mut result = String::new();
        result.push_str(&format!("Problem: {:?}\n", problem));

        let sentences: Vec<&str> = problem
            .split(|c: char| c == '.' || c == '?' || c == '!')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        result.push_str(&format!("Sub-questions: {}\n", sentences.len()));
        result.push('\n');

        let mut all_context = String::new();
        for (i, sentence) in sentences.iter().enumerate() {
            result.push_str(&format!("Step {}: {:?}\n", i + 1, sentence));

            let tokens: Vec<TokenInfo> = sentence
                .split_whitespace()
                .enumerate()
                .map(|(_, w)| TokenInfo {
                    id: crate::weaver::token_id(&w),
                    text: w.to_string(),
                })
                .collect();

            let output = self.think(&tokens);

            for st in &output.super_tokens {
                let ctx = self.memory.retrieve_context(&st.vector, 1);
                if !ctx.is_empty() {
                    all_context.push_str(&ctx);
                    all_context.push('\n');
                }
            }

            let cube_matches = self.query_memory(
                &output
                    .super_tokens
                    .first()
                    .cloned()
                    .unwrap_or_else(|| SuperToken::new(Hypervector::random(self.dim), 0)),
            );
            if !cube_matches.is_empty() {
                result.push_str(&format!("  Cube resonance: {} cells\n", cube_matches.len()));
                for cell in cube_matches.iter().take(3) {
                    result.push_str(&format!(
                        "    cell ({},{},{},{},{}): score={:.4}\n",
                        cell.x, cell.y, cell.z, cell.w, cell.v, cell.score
                    ));
                }
            }
            result.push('\n');
        }

        if !all_context.is_empty() {
            result.push_str("Retrieved context:\n");
            let mut seen = std::collections::HashSet::new();
            for line in all_context.lines() {
                if seen.contains(line) {
                    continue;
                }
                seen.insert(line.to_string());
                result.push_str(line);
                result.push('\n');
            }
        }

        result
    }
}

impl AIOutput {
    pub fn display(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Fuga AI Output ===\n");
        out.push_str(&format!("SuperTokens:     {}\n", self.super_tokens.len()));
        out.push_str(&format!("Route:           {}\n", self.route.name()));
        out.push_str(&format!("Cube entropy:    {:.4}\n", self.cube_entropy));
        out.push_str(&format!("Cube coherence:  {:.4}\n", self.cube_coherence));
        out.push_str(&format!("Attention hits:  {}\n", self.attention_map.len()));
        out.push('\n');
        for cell in &self.attention_map {
            out.push_str(&format!(
                "  Cell ({},{},{},{},{}): score={:.4}\n",
                cell.x, cell.y, cell.z, cell.w, cell.v, cell.score
            ));
        }
        if let Some(tokens) = &self.response_tokens {
            out.push_str(&format!("\nResponse tokens: {}\n", tokens.len()));
            for (i, t) in tokens.iter().enumerate().take(10) {
                out.push_str(&format!("  [{}] id={} text={:?}\n", i, t.id, t.text));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::hypervector::Hypervector;

    fn make_test_ai() -> FugaAI<3, 4> {
        FugaAI::<3, 4>::new(1024, 4)
    }

    #[test]
    fn test_ai_creation() {
        let ai = make_test_ai();
        assert_eq!(ai.dim, 1024);
    }

    #[test]
    fn test_think_with_simple_tokens() {
        let mut ai = make_test_ai();
        let tokens = vec![
            TokenInfo {
                id: crate::weaver::token_id("Hello"),
                text: "Hello".into(),
            },
            TokenInfo {
                id: crate::weaver::token_id("world"),
                text: "world".into(),
            },
        ];
        let output = ai.think(&tokens);
        assert!(output.cube_entropy > 0.0);
        assert_eq!(output.route, TargetExpert::GeneralLanguage);
    }

    #[test]
    fn test_absorb_knowledge_changes_cube() {
        let mut ai = make_test_ai();
        let hv = Hypervector::random(1024);
        let mut st = SuperToken::new(hv, 0);
        st.raw_tokens = vec![1, 2, 3];
        let before = ai.cube.cell(0, 0, 0);
        ai.absorb_knowledge(&[st]);
        let after = ai.cube.cell(0, 0, 0);
        assert_ne!(before.similarity(&after), 1.0);
    }

    #[test]
    fn test_query_memory_returns_cells() {
        let mut ai = make_test_ai();
        let hv = Hypervector::random(1024);
        let st = SuperToken::new(hv, 0);
        let cells = ai.query_memory(&st);
        assert!(cells.len() <= 8);
    }

    #[test]
    fn test_think_code_routes_to_code_logic() {
        let mut ai = make_test_ai();
        let tokens = vec![TokenInfo {
            id: crate::weaver::token_id("def"),
            text: "def foo(x):".into(),
        }];
        let output = ai.think(&tokens);
        assert_eq!(output.route, TargetExpert::CodeLogic);
    }
}

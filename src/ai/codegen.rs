use super::core::FugaAI;
use super::resonance_attention::AttentionCell;
use crate::core::hypervector::Hypervector;
use crate::sandbox::{Canonicalizer, harness::SandboxHarness};
use crate::weaver::{pattern_matcher::TokenInfo, token_id};

pub struct CodegenResult {
    pub generated_text: String,
    pub resonance_cells: Vec<AttentionCell>,
    pub memory_hits: usize,
    pub temperature: f64,
}

fn sanitize(raw: &str) -> Option<String> {
    let words: Vec<&str> = raw.split_whitespace().collect();
    if words.is_empty() {
        return None;
    }

    let code_start = words.iter().position(|w| {
        !w.contains('/')
            && !w.ends_with(".rs")
            && !w.ends_with(".go")
            && !w.ends_with(".c")
            && !w.ends_with(".h")
            && *w != "mod.rs"
    });

    match code_start {
        Some(start) if start < words.len() => {
            let cleaned: Vec<&str> = words[start..].to_vec();
            let raw_code = cleaned.join(" ");
            let raw_code = raw_code.trim_start_matches("Code: ");
            if raw_code.is_empty() {
                None
            } else {
                Some(raw_code.to_string())
            }
        }
        _ => None,
    }
}

fn cell_coords<const N: usize>(cell: &AttentionCell) -> [usize; N] {
    let mut c = [0; N];
    c[0] = cell.x;
    if N > 1 {
        c[1] = cell.y;
    }
    if N > 2 {
        c[2] = cell.z;
    }
    if N > 3 {
        c[3] = cell.w;
    }
    if N > 4 {
        c[4] = cell.v;
    }
    c
}

fn format_code_block(fragments: &[String], max_tokens: usize) -> String {
    let fragments: Vec<&str> = fragments.iter().map(|s| s.as_str()).collect();
    let mut lines = Vec::new();
    let mut word_count = 0;

    for frag in &fragments {
        if word_count >= max_tokens {
            break;
        }
        for line in frag.lines() {
            if word_count >= max_tokens {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                lines.push(String::new());
                continue;
            }
            let line_words: Vec<&str> = trimmed.split_whitespace().collect();
            let available = max_tokens.saturating_sub(word_count);
            if line_words.len() > available {
                lines.push(line_words[..available].join(" "));
                word_count = max_tokens;
            } else {
                lines.push(trimmed.to_string());
                word_count += line_words.len();
            }
        }
    }

    if lines.is_empty() {
        return String::new();
    }

    let mut indent = 0usize;
    let mut out = Vec::new();
    for line in &lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            out.push(String::new());
            continue;
        }
        let closes =
            trimmed.starts_with('}') || trimmed.starts_with(']') || trimmed.starts_with(')');
        let actual_indent = if closes && indent > 0 {
            indent - 1
        } else {
            indent
        };
        out.push(format!("{}{}", "    ".repeat(actual_indent), trimmed));
        if trimmed.ends_with('{') || trimmed.ends_with('[') {
            indent += 1;
        }
        if trimmed.starts_with('}') && indent > 0 {
            indent -= 1;
        }
    }

    out.join("\n")
}

pub fn generate<const N: usize, const S: usize>(
    ai: &mut FugaAI<N, S>,
    seed: &str,
    max_tokens: usize,
    temperature: f64,
) -> CodegenResult {
    let seed_tokens: Vec<TokenInfo> = seed
        .split_whitespace()
        .enumerate()
        .map(|(_, w)| TokenInfo {
            id: token_id(&w),
            text: w.to_string(),
        })
        .collect();

    let output = ai.think(&seed_tokens);

    let seed_vec = output
        .super_tokens
        .first()
        .map(|st| st.vector.clone())
        .unwrap_or_else(|| Hypervector::random(ai.dim));

    let st = crate::weaver::super_token::SuperToken::new(seed_vec.clone(), 0);
    let cells: Vec<AttentionCell> = ai
        .attention
        .beam_attention(&st, &ai.cube, 16)
        .into_iter()
        .filter(|c| c.score > 0.2)
        .take(8)
        .collect();

    if cells.is_empty() {
        return CodegenResult {
            generated_text: "No resonant memory matches found for the given seed.".to_string(),
            resonance_cells: cells,
            memory_hits: 0,
            temperature,
        };
    }

    let mut seen = std::collections::HashSet::new();
    let mut candidates: Vec<(f64, String, Hypervector)> = Vec::new();

    let cell_hvs: Vec<_> = cells
        .iter()
        .map(|cell| {
            let coords: [usize; N] = cell_coords(cell);
            let mut arr = [0; N];
            for i in 0..N {
                arr[i] = coords[i];
            }
            (cell.score, ai.cube.cell_at(&arr))
        })
        .collect();

    let mut collect = |text: &str, sim: f64, vec: &Hypervector| {
        if let Some(clean) = sanitize(text) {
            if seen.insert(clean.clone()) {
                candidates.push((sim, clean, vec.clone()));
            }
        }
    };

    if ai.memory.size() > 0 {
        for (_idx, sim, entry) in ai.memory.search(&seed_vec, 5) {
            collect(&entry.text, sim, &entry.vector);
        }
    }

    for (_score, cell_hv) in &cell_hvs {
        if ai.memory.size() == 0 {
            break;
        }
        for (_idx, sim, entry) in ai.memory.search(cell_hv, 5) {
            collect(&entry.text, sim, &entry.vector);
        }
    }

    for i in 0..cell_hvs.len().min(4) {
        for j in (i + 1)..cell_hvs.len().min(4) {
            let bundle = cell_hvs[i].1.bundle(&[&cell_hvs[j].1]);
            let blend_score = (cell_hvs[i].0 + cell_hvs[j].0) / 2.0;
            for (_idx, sim, entry) in ai.memory.search(&bundle, 2) {
                collect(&entry.text, sim * blend_score, &entry.vector);
            }
        }
    }

    if candidates.is_empty() {
        return CodegenResult {
            generated_text: format!(
                "Resonance found ({} cells) but no memory matches.",
                cells.len()
            ),
            resonance_cells: cells,
            memory_hits: 0,
            temperature,
        };
    }

    candidates.sort_by(|a, b| {
        let sim_a = seed_vec.similarity(&a.2);
        let sim_b = seed_vec.similarity(&b.2);
        sim_b
            .partial_cmp(&sim_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let fragments: Vec<String> = candidates.into_iter().map(|(_, t, _)| t).collect();
    let fragments = Canonicalizer::dedup(&fragments);
    let fragments_clone = fragments.clone();

    // Sandbox validation: filter and score fragments
    let harness = SandboxHarness::new();
    let mut validated_fragments = Vec::new();
    let mut total_reward = 0.0;
    let mut total_weight = 0.0;

    for frag in fragments {
        let result = harness.evaluate(&frag, "fragment.rs");
        // Accept if compiles (even with warnings) and reward >= 0, or if no errors but warnings
        if result.compiles && result.reward >= 0.0 {
            validated_fragments.push(frag);
            total_reward += result.reward.max(0.0);
            total_weight += 1.0;
        }
    }

    // Fallback: if sandbox filtered everything, use canonicalized fragments
    let final_fragments = if validated_fragments.is_empty() {
        fragments_clone
    } else {
        validated_fragments
    };

    let generated_text = format_code_block(&final_fragments, max_tokens);

    CodegenResult {
        generated_text,
        resonance_cells: cells,
        memory_hits: seen.len(),
        temperature,
    }
}

impl CodegenResult {
    pub fn to_text(&self) -> String {
        self.generated_text.clone()
    }

    pub fn display(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Fuga CodeGen ===\n");
        out.push_str(&format!("Temperature: {:.2}\n", self.temperature));
        out.push_str(&format!(
            "Resonance cells: {}\n",
            self.resonance_cells.len()
        ));
        out.push_str(&format!("Memory hits: {}\n\n", self.memory_hits));

        let mut count = 0;
        for line in self.generated_text.lines() {
            if count >= 40 {
                break;
            }
            let display = if line.len() > 120 {
                format!("{}...", &line[..117])
            } else {
                line.to_string()
            };
            out.push_str(&display);
            out.push('\n');
            count += 1;
        }
        if self.generated_text.lines().count() > 40 {
            out.push_str("...\n");
        }
        out
    }
}

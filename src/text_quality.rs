use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum TextSourceType {
    Dialogue,
    Narrative,
    Forum,
    Poetry,
    Unknown,
}

impl TextSourceType {
    pub fn name(&self) -> &str {
        match self {
            TextSourceType::Dialogue => "dialogue",
            TextSourceType::Narrative => "narrative",
            TextSourceType::Forum => "forum",
            TextSourceType::Poetry => "poetry",
            TextSourceType::Unknown => "text",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextQualityScore {
    pub source_type: TextSourceType,
    pub safety: f64,
    pub coherence: f64,
    pub collage_risk: f64,
    pub semantic_reward: f64,
    pub violations: usize,
    pub weight: f64,
    pub summary: String,
    pub path: String,
}

pub struct TextQualityFilter {
    dim: usize,
}

impl TextQualityFilter {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn analyze(&self, text: &str, source_type: TextSourceType, path: &str) -> TextQualityScore {
        let sentences = split_sentences(text);
        let words: Vec<&str> = text.split_whitespace().collect();
        let _total_chars = text.len();

        let sentence_count = sentences.len();
        let empty_sentences = sentences.iter().filter(|s| s.trim().is_empty()).count();
        let short_sentences = sentences.iter().filter(|s| s.trim().len() < 2).count();

        let (pair_issues, _pair_ok) = check_punctuation_pairs(text);
        let violations = empty_sentences + short_sentences + pair_issues;
        let good_sentences = sentence_count.saturating_sub(empty_sentences + short_sentences);

        let collage_risk = compute_collage_risk(&words);

        let type_token_ratio = if words.is_empty() {
            1.0
        } else {
            let unique: std::collections::HashSet<&&str> = words.iter().collect();
            unique.len() as f64 / words.len() as f64
        };

        let _avg_word_len: f64 = if words.is_empty() {
            0.0
        } else {
            words.iter().map(|w| w.len() as f64).sum::<f64>() / words.len() as f64
        };

        let char_entropy = compute_char_entropy(text);

        let sentence_coherence = compute_sentence_coherence(&sentences);

        let has_cap_start = sentences.iter()
            .filter(|s| !s.trim().is_empty())
            .filter(|s| s.trim().chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
            .count();
        let cap_ratio = if good_sentences > 0 {
            has_cap_start as f64 / good_sentences as f64
        } else {
            0.0
        };

        let safety = calculate_text_safety(text, &words, &sentences);

        let noise_penalty = if char_entropy < 1.5 || char_entropy > 5.5 {
            0.3
        } else if char_entropy < 2.0 || char_entropy > 5.0 {
            0.15
        } else {
            0.0
        };

        let collage_penalty = if collage_risk > 0.6 {
            0.3
        } else if collage_risk > 0.4 {
            0.15
        } else {
            0.0
        };

        let violation_penalty = if violations > sentence_count / 2 {
            0.3
        } else if violations > sentence_count / 4 {
            0.15
        } else {
            0.0
        };

        let tt_penalty = if type_token_ratio < 0.15 {
            0.5
        } else if type_token_ratio < 0.25 {
            0.2
        } else {
            0.0
        };

        let cap_bonus = if cap_ratio > 0.6 { 0.1 } else if cap_ratio > 0.3 { 0.05 } else { 0.0 };
        let coherence_bonus = (sentence_coherence * 0.2).min(0.2);

        let semantic_reward = ((1.0 - collage_risk) * 0.4 + sentence_coherence * 0.3
            + cap_bonus + coherence_bonus).min(1.0);

        let raw_weight = (((1.0f64 - collage_penalty) * (1.0f64 - noise_penalty)
            * (1.0f64 - violation_penalty) * (1.0f64 - tt_penalty))
            .clamp(0.0, 1.0)) * safety;

        let weight = if raw_weight > 0.0 {
            match source_type {
                TextSourceType::Dialogue => raw_weight.max(0.3),
                TextSourceType::Narrative => raw_weight.max(0.3),
                TextSourceType::Poetry => (raw_weight * 0.8).max(0.2),
                _ => raw_weight,
            }
        } else {
            0.0
        };

        let summary = format!(
            "{}: sentences={} safe={:.2} collage={:.2} coherence={:.2} reward={:.2} v={} → w={:.2}",
            source_type.name(), sentence_count, safety, collage_risk,
            sentence_coherence, semantic_reward, violations, weight,
        );

        TextQualityScore {
            source_type,
            safety,
            coherence: sentence_coherence,
            collage_risk,
            semantic_reward,
            violations,
            weight,
            summary,
            path: path.to_string(),
        }
    }

    pub fn scan_directory(&mut self, dir: &str, recursive: bool) -> Result<Vec<(String, TextQualityScore)>, String> {
        use walkdir::WalkDir;
        let mut results = Vec::new();
        let walker = if recursive {
            WalkDir::new(dir).follow_links(true).into_iter()
        } else {
            WalkDir::new(dir).follow_links(true).max_depth(1).into_iter()
        };

        for entry in walker.filter_map(|e| e.ok()) {
            let path = entry.path();
            if !path.is_file() { continue; }
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
            let is_text = matches!(ext.as_str(), "txt" | "jsonl" | "csv" | "srt" | "md" | "html" | "xml" | "text");
            if !is_text { continue; }

            let source = match std::fs::read_to_string(path) {
                Ok(s) => s,
                Err(_) => continue,
            };
            if source.len() < 20 { continue; }

            let path_str = path.to_string_lossy().to_string();
            let source_type = detect_source_type(&path_str, &source);
            let score = self.analyze(&source, source_type, &path_str);
            results.push((path_str, score));
        }

        results.sort_by(|a, b| b.1.weight.partial_cmp(&a.1.weight).unwrap());
        Ok(results)
    }
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let text = text.replace('\r', "");
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?' | '\n') {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }
    sentences
}

fn check_punctuation_pairs(text: &str) -> (usize, bool) {
    let mut issues = 0;
    let pairs = [('(', ')'), ('[', ']'), ('{', '}'), ('"', '"'), ('\'', '\'')];
    for &(open, close) in &pairs {
        let opens = text.matches(open).count();
        let closes = text.matches(close).count();
        if opens != closes {
            issues += opens.abs_diff(closes);
        }
    }
    (issues, issues == 0)
}

fn compute_collage_risk(words: &[&str]) -> f64 {
    if words.len() < 5 {
        return 0.0;
    }

    let mut word_freq: HashMap<&str, usize> = HashMap::new();
    for w in words {
        let w = w.trim_matches(|c: char| c.is_ascii_punctuation()).to_lowercase();
        if !w.is_empty() {
            *word_freq.entry(Box::leak(w.into_boxed_str())).or_insert(0) += 1;
        }
    }

    if word_freq.is_empty() {
        return 0.0;
    }

    let total = words.len() as f64;
    let top_repeat: usize = word_freq.values().filter(|&&c| c > 1).count();
    let repeat_ratio = top_repeat as f64 / word_freq.len() as f64;

    let max_freq = *word_freq.values().max().unwrap_or(&1) as f64;
    let max_ratio = if words.len() > 1 { max_freq / total } else { 0.0 };

    let mut bigram_repeat = 0;
    let mut bigrams: HashMap<(String, String), usize> = HashMap::new();
    for pair in words.windows(2) {
        let a = pair[0].trim_matches(|c: char| c.is_ascii_punctuation()).to_lowercase();
        let b = pair[1].trim_matches(|c: char| c.is_ascii_punctuation()).to_lowercase();
        if !a.is_empty() && !b.is_empty() {
            *bigrams.entry((a, b)).or_insert(0) += 1;
        }
    }
    let bigram_total = bigrams.len();
    if bigram_total > 0 {
        bigram_repeat = bigrams.values().filter(|&&c| c > 1).count();
    }
    let bigram_risk = if bigram_total > 0 {
        bigram_repeat as f64 / bigram_total as f64
    } else {
        0.0
    };

    (repeat_ratio * 0.4 + max_ratio * 0.3 + bigram_risk * 0.3).min(1.0)
}

fn compute_char_entropy(text: &str) -> f64 {
    let text: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    if text.len() < 2 {
        return 0.0;
    }
    let mut freq: HashMap<char, usize> = HashMap::new();
    for ch in text.chars() {
        *freq.entry(ch).or_insert(0) += 1;
    }
    let total = text.len() as f64;
    let mut entropy = 0.0;
    for &count in freq.values() {
        let p = count as f64 / total;
        entropy -= p * p.log2();
    }
    entropy
}

fn compute_sentence_coherence(sentences: &[String]) -> f64 {
    let non_empty: Vec<&str> = sentences.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if non_empty.len() < 2 {
        return 0.5;
    }
    let mut overlap_scores = Vec::new();
    for pair in non_empty.windows(2) {
        let a_words: Vec<&str> = pair[0].split_whitespace().collect();
        let b_words: Vec<&str> = pair[1].split_whitespace().collect();
        if a_words.is_empty() || b_words.is_empty() {
            continue;
        }
        let a_set: std::collections::HashSet<&&str> = a_words.iter().collect();
        let b_set: std::collections::HashSet<&&str> = b_words.iter().collect();
        let common = a_set.intersection(&b_set).count();
        let union = a_set.union(&b_set).count();
        if union > 0 {
            overlap_scores.push(common as f64 / union as f64);
        }
    }
    if overlap_scores.is_empty() {
        0.5
    } else {
        overlap_scores.iter().sum::<f64>() / overlap_scores.len() as f64
    }
}

fn calculate_text_safety(text: &str, words: &[&str], _sentences: &[String]) -> f64 {
    let mut safety = 1.0;

    let toxic_patterns = [
        "http://", "https://", "www.",
        "<script", "<?php", "javascript:",
        "\x00", "\r\n\r\n",
    ];
    for pat in &toxic_patterns {
        if text.contains(pat) {
            safety *= 0.7;
        }
    }

    let gibberish_chars = words.iter().filter(|w| {
        let w = w.trim_matches(|c: char| c.is_ascii_punctuation());
        if w.len() > 15 && w.chars().all(|c| c.is_alphabetic()) {
            return true;
        }
        let alpha = w.chars().filter(|c| c.is_alphabetic()).count();
        if w.len() > 3 && alpha == 0 {
            return true;
        }
        false
    }).count();

    if gibberish_chars > words.len() / 3 {
        safety *= 0.3;
    } else if gibberish_chars > words.len() / 5 {
        safety *= 0.6;
    }

    let repeated_chars = words.iter().filter(|w| {
        let w = w.trim_matches(|c: char| c.is_ascii_punctuation());
        if w.len() < 3 { return false; }
        let chars: Vec<char> = w.chars().collect();
        chars.windows(3).any(|c| c[0] == c[1] && c[1] == c[2])
    }).count();
    if repeated_chars > words.len() / 4 {
        safety *= 0.5;
    }

    (safety as f64).clamp(0.0, 1.0)
}

fn detect_source_type(path: &str, content: &str) -> TextSourceType {
    if content.lines().count() < 3 {
        return TextSourceType::Unknown;
    }

    let dialogue_markers = content.matches("— ").count()
        + content.matches(" - ").count()
        + content.matches('»').count()
        + content.matches('«').count();
    let quote_lines = content.lines()
        .filter(|l| l.trim().starts_with('"') || l.trim().starts_with('—') || l.trim().starts_with('-'))
        .count();
    let _colons = content.matches(':').count();

    if (dialogue_markers > 5 || quote_lines > 3) && (dialogue_markers + quote_lines) as f64 > content.lines().count() as f64 * 0.3 {
        return TextSourceType::Dialogue;
    }

    let path_lower = path.to_lowercase();
    if path_lower.contains("subtitle") || path_lower.contains("srt") || path_lower.contains("dialog") {
        return TextSourceType::Dialogue;
    }
    if path_lower.contains("forum") || path_lower.contains("reddit") || path_lower.contains("habr") || path_lower.contains("stack") {
        return TextSourceType::Forum;
    }
    if path_lower.contains("poem") || path_lower.contains("poetry") || path_lower.contains("стих") {
        return TextSourceType::Poetry;
    }

    TextSourceType::Narrative
}

pub fn extract_dialogue_pairs(content: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let mut context = String::new();
    let lines: Vec<&str> = content.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let is_speech = line.starts_with('"') || line.starts_with('—') || line.starts_with('-')
            || line.starts_with('«') || line.starts_with('»')
            || line.contains(':') && line.len() < 200;

        if is_speech {
            if !context.is_empty() && context.len() < 500 {
                let response = line.trim_matches('"').trim()
                    .trim_start_matches('—').trim()
                    .trim_start_matches('-').trim()
                    .trim_start_matches('«').trim_end_matches('»').trim()
                    .to_string();
                if !response.is_empty() && response.len() > 3 {
                    pairs.push((context.clone(), response));
                }
            }
            let clean = line.trim_matches('"').trim()
                .trim_start_matches('—').trim()
                .trim_start_matches('-').trim()
                .trim_start_matches('«').trim_end_matches('»').trim()
                .to_string();
            if !clean.is_empty() {
                context = clean;
            }
        } else if line.ends_with('.') || line.ends_with('!') || line.ends_with('?') {
            context = line.to_string();
        }

        i += 1;
    }

    pairs
}

pub fn summarize_text_quality(results: &[(String, TextQualityScore)]) -> String {
    let total = results.len();
    let high = results.iter().filter(|(_, s)| s.weight >= 0.8).count();
    let medium = results.iter().filter(|(_, s)| s.weight >= 0.4 && s.weight < 0.8).count();
    let low = results.iter().filter(|(_, s)| s.weight > 0.0 && s.weight < 0.4).count();
    let blocked = results.iter().filter(|(_, s)| s.weight == 0.0).count();

    let avg_weight: f64 = results.iter().map(|(_, s)| s.weight).sum::<f64>() / total.max(1) as f64;
    let avg_coherence: f64 = results.iter().map(|(_, s)| s.coherence).sum::<f64>() / total.max(1) as f64;
    let avg_collage: f64 = results.iter().map(|(_, s)| s.collage_risk).sum::<f64>() / total.max(1) as f64;
    let avg_reward: f64 = results.iter().map(|(_, s)| s.semantic_reward).sum::<f64>() / total.max(1) as f64;

    let dialogue_count = results.iter().filter(|(_, s)| s.source_type == TextSourceType::Dialogue).count();
    let narrative_count = results.iter().filter(|(_, s)| s.source_type == TextSourceType::Narrative).count();
    let forum_count = results.iter().filter(|(_, s)| s.source_type == TextSourceType::Forum).count();

    let mut out = String::new();
    out.push_str(&format!("Text files:  {}\n", total));
    out.push_str(&format!("  Dialogue:  {}\n", dialogue_count));
    out.push_str(&format!("  Narrative: {}\n", narrative_count));
    out.push_str(&format!("  Forum:     {}\n", forum_count));
    out.push_str(&format!("  High (w≥0.8):   {}\n", high));
    out.push_str(&format!("  Med (0.4≤w<0.8): {}\n", medium));
    out.push_str(&format!("  Low (0<w<0.4):  {}\n", low));
    out.push_str(&format!("  Blocked (w=0):  {} ({:.0}%)\n", blocked, blocked as f64 / total.max(1) as f64 * 100.0));
    out.push_str(&format!("Avg weight:    {:.3}\n", avg_weight));
    out.push_str(&format!("Avg coherence: {:.3}\n", avg_coherence));
    out.push_str(&format!("Avg collage:   {:.3}\n", avg_collage));
    out.push_str(&format!("Avg reward:    {:.3}\n", avg_reward));
    out.push('\n');
    for (path, score) in results {
        out.push_str(&format!("  {}  {}\n", score.summary, path));
    }
    out
}

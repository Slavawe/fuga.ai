use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SpecialToken {
    pub id: u32,
    pub content: String,
    pub special: bool,
}

#[derive(Debug, Clone)]
pub struct TokenConfig {
    pub vocab_size: usize,
    pub special_tokens: HashMap<u32, SpecialToken>,
    pub pat_str: String,
    pub bos_token: String,
    pub eos_token: String,
    pub pad_token: String,
}

impl TokenConfig {
    pub fn from_json(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path, e))?;
        let json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse JSON: {}", e))?;

        let mut special_tokens = HashMap::new();
        if let Some(added) = json.get("added_tokens_decoder").and_then(|v| v.as_object()) {
            for (key, val) in added {
                let id: u32 = key.parse().unwrap_or(0);
                let content = val.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let special = val.get("special").and_then(|v| v.as_bool()).unwrap_or(false);
                special_tokens.insert(id, SpecialToken { id, content, special });
            }
        }

        let bos = json.get("bos_token").and_then(|v| v.as_str()).unwrap_or("[BOS]").to_string();
        let eos = json.get("eos_token").and_then(|v| v.as_str()).unwrap_or("[EOS]").to_string();
        let pad = json.get("pad_token").and_then(|v| v.as_str()).unwrap_or("").to_string();

        let mut extra_specials = Vec::new();
        if let Some(extra) = json.get("additional_special_tokens").and_then(|v| v.as_array()) {
            for v in extra {
                if let Some(s) = v.as_str() {
                    extra_specials.push(s.to_string());
                }
            }
        }

        Ok(TokenConfig {
            vocab_size: json.get("model_max_length").and_then(|v| v.as_u64()).unwrap_or(128000) as usize,
            special_tokens,
            pat_str: String::new(),
            bos_token: bos,
            eos_token: eos,
            pad_token: pad,
        })
    }
}

pub struct TokenBuilder {
    configs: Vec<TokenConfig>,
    merged_special: HashMap<u32, SpecialToken>,
    merged_vocab_size: usize,
}

impl TokenBuilder {
    pub fn new() -> Self {
        Self {
            configs: Vec::new(),
            merged_special: HashMap::new(),
            merged_vocab_size: 0,
        }
    }

    pub fn load_config(&mut self, path: &str) -> Result<(), String> {
        let config = TokenConfig::from_json(path)?;
        for (id, tok) in &config.special_tokens {
            self.merged_special.entry(*id).or_insert_with(|| tok.clone());
        }
        self.merged_vocab_size = self.merged_vocab_size.max(config.vocab_size);
        self.configs.push(config);
        Ok(())
    }

    pub fn load_configs_from_dir(&mut self, dir: &str) -> Result<usize, String> {
        let mut count = 0;
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "json" {
                        self.load_config(path.to_str().unwrap())?;
                        count += 1;
                    }
                }
            }
        }
        Ok(count)
    }

    pub fn build_flat_vocab(&self) -> Vec<(u32, String)> {
        let mut tokens: Vec<(u32, String)> = self.merged_special.iter()
            .map(|(id, t)| (*id, t.content.clone()))
            .collect();
        tokens.sort_by_key(|(id, _)| *id);

        let max_id = tokens.last().map(|(id, _)| *id).unwrap_or(0);
        let base_vocab = if max_id > 0 { max_id as usize + 1 } else { 0 };
        let total = base_vocab.max(self.merged_vocab_size);

        let mut full: Vec<(u32, String)> = (0..total as u32)
            .map(|i| (i, format!("<|token_{}|>", i)))
            .collect();
        for (id, content) in &tokens {
            if (*id as usize) < total {
                full[*id as usize] = (*id, content.clone());
            }
        }
        full
    }

    pub fn merged_special(&self) -> &HashMap<u32, SpecialToken> { &self.merged_special }
    pub fn vocab_size(&self) -> usize { self.merged_vocab_size }

    pub fn report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("Configs loaded: {}\n", self.configs.len()));
        out.push_str(&format!("Merged vocab size: {}\n", self.merged_vocab_size));
        out.push_str(&format!("Special tokens: {}\n", self.merged_special.len()));
        out.push_str("\nSpecial tokens:\n");
        let mut sorted: Vec<_> = self.merged_special.iter().collect();
        sorted.sort_by_key(|(id, _)| *id);
        for (id, tok) in &sorted {
            out.push_str(&format!("  {}: {:?} (special: {})\n", id, tok.content, tok.special));
        }
        out
    }
}

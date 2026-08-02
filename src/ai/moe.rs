use super::memory_store::{MemoryEntry, MemoryStore};
use crate::core::hypervector::Hypervector;
use crate::weaver::super_token::SuperToken;
use std::collections::HashMap;

pub const BUILTIN_DOMAINS: &[&str] = &[
    "dialogue",
    "narrative",
    "code",
    "general",
    "forum",
    "poetry",
    "dialogue_pair",
];

pub struct MoEStore {
    base_path: String,
    experts: HashMap<String, MemoryStore>,
    sizes: HashMap<String, usize>,
}

impl MoEStore {
    pub fn new(base_path: &str) -> Self {
        Self {
            base_path: base_path.to_string(),
            experts: HashMap::new(),
            sizes: HashMap::new(),
        }
    }

    pub fn mem_path(domain: &str) -> String {
        format!("fuga_moe_{}.bin", domain)
    }

    pub fn discover_domains() -> Vec<String> {
        let mut domains: Vec<String> = BUILTIN_DOMAINS.iter().map(|s| s.to_string()).collect();
        if let Ok(entries) = std::fs::read_dir(".") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if let Some(rest) = name
                    .strip_prefix("fuga_moe_")
                    .and_then(|s| s.strip_suffix(".bin"))
                {
                    let d = rest.to_string();
                    if !domains.contains(&d) {
                        domains.push(d);
                    }
                }
            }
        }
        domains.sort();
        domains
    }

    pub fn domain_for(query: &str) -> &'static str {
        let q = query.to_lowercase();
        if q.contains("fn ")
            || q.contains("impl ")
            || q.contains("struct ")
            || q.contains("python")
            || q.contains("rust")
            || q.contains("javascript")
            || q.contains("code")
            || q.contains("syntax")
            || q.contains("compiler")
            || q.contains("функция")
            || q.contains("код")
            || q.contains("программа")
            || q.contains("напиши")
            || q.contains("создай")
        {
            "code"
        } else if q.contains("sysadmin")
            || q.contains("deploy")
            || q.contains("server")
            || q.contains("docker")
            || q.contains("nginx")
            || q.contains("config")
            || q.contains("админ")
            || q.contains("сервер")
        {
            "sysadmin"
        } else if q.contains("рассказ")
            || q.contains("истори")
            || q.contains("книг")
            || q.contains("рома")
            || q.contains("novel")
            || q.contains("story")
            || q.contains("narrative")
            || q.contains("chapter")
            || q.contains("page")
        {
            "narrative"
        } else if q.contains("диалог")
            || q.contains("разговор")
            || q.contains("бесед")
            || q.contains("расскажи")
            || q.contains("hello")
            || q.contains("hi ")
            || q.contains("how are you")
            || q.contains("привет")
            || q.contains("как дела")
        {
            "dialogue"
        } else if q.contains("doc")
            || q.contains("manual")
            || q.contains("guide")
            || q.contains("документация")
            || q.contains("описание")
        {
            "docs"
        } else if q.contains("architecture")
            || q.contains("design")
            || q.contains("pattern")
            || q.contains("архитектура")
            || q.contains("проект")
        {
            "architecture"
        } else {
            "general"
        }
    }

    pub fn load_domain(&mut self, domain: &str) -> Result<(), String> {
        if self.experts.contains_key(domain) {
            return Ok(());
        }
        let path = format!("{}_{}", self.base_path.replace(".bin", ""), domain);
        let mem_path = format!("{}_mem.bin", path);
        if !std::path::Path::new(&mem_path).exists() {
            self.sizes.insert(domain.to_string(), 0);
            self.experts.insert(domain.to_string(), MemoryStore::new());
            return Ok(());
        }
        let mut mem = MemoryStore::load_bin(&mem_path)?;
        let size = mem.size();
        self.sizes.insert(domain.to_string(), size);
        if domain == "code" {
            let idx_path = format!("{}_mem.idx", path);
            if std::path::Path::new(&idx_path).exists() {
                mem.load_text_index(&idx_path).ok();
            } else {
                mem.build_text_index();
                mem.save_text_index(&idx_path).ok();
            }
            let hnsw_path = format!("{}_vsa.bin", path);
            if std::path::Path::new(&hnsw_path).exists() {
                if mem.load_vsa_index(&hnsw_path).is_err() {
                    eprintln!("  VSA index load failed (non-fatal, skipping)");
                }
            }
        }
        self.experts.insert(domain.to_string(), mem);
        Ok(())
    }

    pub fn add_domain(&mut self, domain: &str) -> Result<(), String> {
        if self.experts.contains_key(domain) {
            return Ok(());
        }
        let mem = MemoryStore::new();
        let mem_path = Self::mem_path(domain);
        mem.save_bin(&mem_path)?;
        self.experts.insert(domain.to_string(), mem);
        self.sizes.insert(domain.to_string(), 0);
        println!("  ✓ Created domain '{}' (saved to {})", domain, mem_path);
        Ok(())
    }

    pub fn unload(&mut self, domain: &str) {
        self.experts.remove(domain);
        self.sizes.remove(domain);
    }

    pub fn search(
        &self,
        domain: &str,
        query: &Hypervector,
        top_k: usize,
    ) -> Vec<(usize, f64, &MemoryEntry)> {
        match self.experts.get(domain) {
            Some(mem) => mem.search(query, top_k),
            None => Vec::new(),
        }
    }

    pub fn search_by_text(
        &self,
        domain: &str,
        query_text: &str,
        top_k: usize,
    ) -> Vec<(usize, f64, &MemoryEntry)> {
        match self.experts.get(domain) {
            Some(mem) => mem.search_by_text(query_text, top_k),
            None => Vec::new(),
        }
    }

    pub fn search_with_prompts(
        &self,
        domain: &str,
        query: &Hypervector,
        prompts: &[&Hypervector],
        top_k: usize,
    ) -> Vec<(usize, f64, &MemoryEntry)> {
        match self.experts.get(domain) {
            Some(mem) => mem.search_with_prompts(query, prompts, top_k),
            None => Vec::new(),
        }
    }

    pub fn search_all_by_text(
        &self,
        query_text: &str,
        top_k: usize,
    ) -> Vec<(usize, f64, &MemoryEntry, &str)> {
        let mut all = Vec::new();
        for (domain, mem) in &self.experts {
            for (idx, sim, entry) in mem.search_by_text(query_text, top_k) {
                all.push((idx, sim, entry, domain.as_str()));
            }
        }
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        all.truncate(top_k);
        all
    }

    pub fn store(&mut self, st: &SuperToken, text: &str, source_doc: &str, role_hint: &str) {
        let is_code_file = source_doc.ends_with(".rs")
            || source_doc.ends_with(".py")
            || source_doc.ends_with(".js")
            || source_doc.ends_with(".ts")
            || source_doc.ends_with(".jsx")
            || source_doc.ends_with(".html")
            || source_doc.ends_with(".css")
            || source_doc.ends_with(".scss")
            || source_doc.ends_with(".c")
            || source_doc.ends_with(".cpp")
            || source_doc.ends_with(".h")
            || source_doc.ends_with(".go")
            || source_doc.ends_with(".java")
            || source_doc.ends_with(".rs")
            || source_doc.ends_with(".toml")
            || source_doc.ends_with(".json")
            || source_doc.ends_with(".yaml")
            || source_doc.ends_with(".tsx")
            || source_doc.ends_with(".vue")
            || source_doc.ends_with(".svelte");
        let domain = match role_hint {
            "dialogue" | "Dialogue" => "dialogue",
            "narrative" | "Narrative" => "narrative",
            "forum" | "Forum" => "forum",
            "poetry" | "Poetry" => "poetry",
            "dialogue_pair" => "dialogue",
            "sysadmin" | "Sysadmin" => "sysadmin",
            "docs" | "Docs" => "docs",
            "architecture" | "Architecture" => "architecture",
            "code" => "code",
            _ if is_code_file => "code",
            _ => "general",
        };
        let key = domain.to_string();
        let mem = self
            .experts
            .entry(key.clone())
            .or_insert_with(MemoryStore::new);
        mem.store(st, text, source_doc, role_hint);
        *self.sizes.entry(key).or_insert(0) += 1;
    }

    pub fn domain_size(&self, domain: &str) -> usize {
        self.sizes.get(domain).copied().unwrap_or(0)
    }

    pub fn total_size(&self) -> usize {
        self.sizes.values().sum()
    }

    pub fn domain_sizes(&self) -> Vec<(&str, usize)> {
        let mut v: Vec<(&str, usize)> = self.sizes.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }

    pub fn available_domains(&self) -> Vec<String> {
        self.experts.keys().cloned().collect()
    }

    pub fn save_all(&self) -> Result<(), String> {
        for (domain, mem) in &self.experts {
            let path = format!("{}_{}", self.base_path.replace(".bin", ""), domain);
            let mem_path = format!("{}_mem.bin", path);
            mem.save_bin(&mem_path)?;
        }
        Ok(())
    }

    pub fn load_all(&mut self) -> Result<(), String> {
        let domains = Self::discover_domains();
        for domain in &domains {
            let _ = self.load_domain(domain);
        }
        Ok(())
    }
}

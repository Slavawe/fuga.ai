use crate::ai::hnsw::VsaIndex;
use crate::core::hypervector::Hypervector;
use crate::weaver::super_token::SuperToken;
use std::collections::{HashMap, HashSet};

pub const NUM_ATTRACTORS: usize = 64;

/// Guard-ридеры для бинарных load-путей (аудит 22.08): усечённый/
/// малформированный файл даёт Err, а не панику slice out of range и не
/// гигантскую with_capacity по счётчику из файла.
fn rd_u32(data: &[u8], pos: &mut usize) -> Result<u32, String> {
    if *pos + 4 > data.len() {
        return Err("corrupt bin: EOF reading u32".into());
    }
    let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(v)
}

fn rd_u64(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    if *pos + 8 > data.len() {
        return Err("corrupt bin: EOF reading u64".into());
    }
    let v = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(v)
}

fn rd_bytes<'a>(data: &'a [u8], pos: &mut usize, len: usize) -> Result<&'a [u8], String> {
    if len > data.len().saturating_sub(*pos) {
        return Err(format!("corrupt bin: {} bytes overrun file at {}", len, pos));
    }
    let s = &data[*pos..*pos + len];
    *pos += len;
    Ok(s)
}

#[derive(Clone)]
pub struct AttractorIndex {
    pub attractors: Vec<Hypervector>,
    pub clusters: Vec<Vec<u32>>,
}

impl AttractorIndex {
    pub fn build(entries: &[MemoryEntry], dim: usize) -> Self {
        let mut attractors: Vec<Hypervector> = (0..NUM_ATTRACTORS)
            .map(|_| Hypervector::random(dim))
            .collect();
        let mut clusters = vec![Vec::new(); NUM_ATTRACTORS];

        for (i, entry) in entries.iter().enumerate() {
            let best = attractors
                .iter()
                .enumerate()
                .map(|(ai, a)| (ai, entry.vector.similarity(a)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(ai, _)| ai)
                .unwrap_or(0);
            clusters[best].push(i as u32);
        }

        for ai in 0..NUM_ATTRACTORS {
            if clusters[ai].is_empty() {
                continue;
            }
            let ref_vecs: Vec<&Hypervector> = clusters[ai]
                .iter()
                .map(|&idx| &entries[idx as usize].vector)
                .collect();
            let centroid = ref_vecs[0].bundle(&ref_vecs[1..]);
            attractors[ai] = centroid;
        }

        let mut reassign = vec![Vec::new(); NUM_ATTRACTORS];
        for ci in 0..NUM_ATTRACTORS {
            for &ei in &clusters[ci] {
                let best = attractors
                    .iter()
                    .enumerate()
                    .map(|(ai, a)| (ai, entries[ei as usize].vector.similarity(a)))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(ai, _)| ai)
                    .unwrap_or(0);
                reassign[best].push(ei);
            }
        }

        Self {
            attractors,
            clusters: reassign,
        }
    }

    pub fn search(
        &self,
        query: &Hypervector,
        top_k: usize,
        entries: &[MemoryEntry],
        top_a: usize,
    ) -> Vec<(usize, f64)> {
        let mut attractor_scores: Vec<(usize, f64)> = self
            .attractors
            .iter()
            .enumerate()
            .map(|(i, a)| (i, query.similarity(a)))
            .collect();
        attractor_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut seen = vec![false; entries.len()];
        let mut candidates: Vec<(usize, f64)> = Vec::new();

        for &(ai, _) in attractor_scores.iter().take(top_a) {
            for &ei in &self.clusters[ai] {
                if seen[ei as usize] {
                    continue;
                }
                seen[ei as usize] = true;
                let sim = query.similarity(&entries[ei as usize].vector);
                candidates.push((ei as usize, sim));
            }
        }

        let total_searched = candidates.len();
        if total_searched < top_k && entries.len() > total_searched {
            let need = top_k
                .saturating_sub(total_searched)
                .min(entries.len() - total_searched);
            let mut pool: Vec<usize> = (0..entries.len()).filter(|i| !seen[*i]).collect();
            fastrand::shuffle(&mut pool);
            for &idx in pool.iter().take(need) {
                let sim = query.similarity(&entries[idx].vector);
                candidates.push((idx, sim));
            }
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(top_k);
        candidates
    }
}

#[derive(Clone)]
pub struct MemoryEntry {
    pub vector: Hypervector,
    pub text: String,
    pub source_doc: String,
    pub role_hint: String,
}

#[derive(Clone)]
pub struct MemoryStore {
    entries: Vec<MemoryEntry>,
    text_index: Option<HashMap<String, Vec<u32>>>,
    vsa_idx: Option<VsaIndex>,
    attractor_idx: Option<AttractorIndex>,
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            text_index: None,
            vsa_idx: None,
            attractor_idx: None,
        }
    }
}

impl MemoryStore {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            text_index: None,
            vsa_idx: None,
            attractor_idx: None,
        }
    }

    pub fn store(&mut self, st: &SuperToken, text: &str, source_doc: &str, role_hint: &str) {
        self.text_index = None;
        self.vsa_idx = None;
        self.attractor_idx = None;
        self.entries.push(MemoryEntry {
            vector: st.vector.clone(),
            text: text.to_string(),
            source_doc: source_doc.to_string(),
            role_hint: role_hint.to_string(),
        });
    }

    pub fn store_raw(
        &mut self,
        vector: &Hypervector,
        text: &str,
        source_doc: &str,
        role_hint: &str,
    ) {
        self.text_index = None;
        self.vsa_idx = None;
        self.attractor_idx = None;
        self.entries.push(MemoryEntry {
            vector: vector.clone(),
            text: text.to_string(),
            source_doc: source_doc.to_string(),
            role_hint: role_hint.to_string(),
        });
    }

    pub fn build_text_index(&mut self) {
        let mut index: HashMap<String, Vec<u32>> = HashMap::new();
        for (i, e) in self.entries.iter().enumerate() {
            let idx = i as u32;
            for w in e.text.split_whitespace() {
                if w.len() <= 2 {
                    continue;
                }
                let key = w.to_lowercase();
                index.entry(key).or_default().push(idx);
            }
        }
        self.text_index = Some(index);
    }

    pub fn search(&self, query: &Hypervector, top_k: usize) -> Vec<(usize, f64, &MemoryEntry)> {
        self.search_with_prompts(query, &[], top_k)
    }

    pub fn search_with_prompts(
        &self,
        query: &Hypervector,
        prompts: &[&Hypervector],
        top_k: usize,
    ) -> Vec<(usize, f64, &MemoryEntry)> {
        let modulated = if prompts.is_empty() {
            query.clone()
        } else {
            let mut result = query.clone();
            for p in prompts {
                result = result.bind(p);
            }
            result
        };
        if let Some(ref vsa) = self.vsa_idx {
            let results = vsa.search(&modulated, top_k);
            if !results.is_empty() {
                return results
                    .into_iter()
                    .map(|(i, s)| (i, s, &self.entries[i]))
                    .collect();
            }
        }
        let vectors: Vec<&Hypervector> = self.entries.iter().map(|e| &e.vector).collect();
        let gpu_results = crate::gpu::gpu_memory_search(&modulated, &vectors, top_k, 0.0);
        let results: Vec<(usize, f64)> = if let Some(gpu_hits) = gpu_results {
            gpu_hits.into_iter().take(top_k).collect()
        } else {
            let mut scores: Vec<(usize, f64)> = self
                .entries
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    let partial = modulated.partial_similarity(&e.vector, 8);
                    if partial >= 0.53 {
                        (i, modulated.similarity(&e.vector))
                    } else {
                        (i, partial * 0.3)
                    }
                })
                .collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scores.into_iter().take(top_k).collect()
        };
        results
            .into_iter()
            .map(|(i, s)| (i, s, &self.entries[i]))
            .collect()
    }

    pub fn search_by_text(
        &self,
        query_text: &str,
        top_k: usize,
    ) -> Vec<(usize, f64, &MemoryEntry)> {
        // Стоп-слова: частотные служебные тонут в лексике 604K-записей кода
        // (any/how/are/you — в каждом C-headere), расталкивая ключевые.
        // Оставляем только значимые слова запроса (08.08, диалог-диагноз).
        const LEX_STOP: &[&str] = &[
            "the", "and", "for", "that", "this", "with", "from", "was", "are", "not", "have",
            "has", "its", "but", "than", "them", "then", "they", "were", "will", "into", "more",
            "most", "some", "would", "their", "there", "which", "been", "what", "when", "where",
            "your", "is", "of", "to", "in", "if", "a", "or", "as", "at", "it", "we", "he", "she",
            "his", "her", "be", "how", "you", "does", "do", "can", "why", "who", "only", "very",
            "just", "also", "about", "between", "over", "under", "during", "before", "after",
            "а", "в", "и", "на", "с", "о", "не", "как", "что", "для", "по", "из", "у",
        ];
        let words: Vec<String> = query_text
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .map(|w| w.to_lowercase())
            .filter(|w| !LEX_STOP.contains(&w.as_str()))
            .collect();
        if words.is_empty() {
            return vec![];
        }

        if let Some(ref index) = self.text_index {
            let mut candidate_scores: HashMap<u32, f64> = HashMap::new();
            for w in &words {
                if let Some(entries) = index.get(w) {
                    for &eid in entries {
                        // +1 за совпадение слова в тексте, +2 за совпадение
                        // слова в имени файла-источника (сильный сигнал).
                        let e = &self.entries[eid as usize];
                        let fname = std::path::Path::new(&e.source_doc)
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("")
                            .to_lowercase();
                        let boost = if fname.contains(w.as_str()) { 2.0 } else { 0.0 };
                        *candidate_scores.entry(eid).or_insert(0.0) += 1.0 + boost;
                    }
                }
            }
            if candidate_scores.is_empty() {
                return vec![];
            }
            let max_score = 3.0 * words.len() as f64;
            let mut scores: Vec<(u32, f64)> = candidate_scores.into_iter().collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scores.truncate(top_k);
            return scores
                .into_iter()
                .map(|(i, s)| (i as usize, s / max_score, &self.entries[i as usize]))
                .collect();
        }

        let _query_lower = query_text.to_lowercase();
        let mut scores: Vec<(usize, f64)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let lower = e.text.to_lowercase();
                let matches = words.iter().filter(|w| lower.contains(w.as_str())).count();
                (i, matches as f64)
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let max_score = words.len() as f64;
        scores.truncate(top_k);
        scores
            .into_iter()
            .filter(|(_, s)| *s > 0.0)
            .map(|(i, s)| (i, s / max_score, &self.entries[i]))
            .collect()
    }

    pub fn retrieve_context(&self, query: &Hypervector, window: usize) -> String {
        let nearest = self.search(query, 3);
        if nearest.is_empty() {
            return String::new();
        }

        let mut seen = HashSet::new();
        let mut context = String::new();

        for &(idx, sim, entry) in &nearest {
            if seen.contains(&entry.text.as_str()) {
                continue;
            }
            seen.insert(entry.text.as_str());

            if !context.is_empty() {
                context.push('\n');
            }
            context.push_str(&format!(
                "[{} sim={:.3}] {}",
                entry.source_doc, sim, entry.text
            ));

            for delta in 1..=window {
                if idx >= delta {
                    let prev = &self.entries[idx - delta];
                    if !seen.contains(&prev.text.as_str()) {
                        seen.insert(prev.text.as_str());
                        context.push('\n');
                        context.push_str(&format!("[{} ctx] {}", prev.source_doc, prev.text));
                    }
                }
                if idx + delta < self.entries.len() {
                    let next = &self.entries[idx + delta];
                    if !seen.contains(&next.text.as_str()) {
                        seen.insert(next.text.as_str());
                        context.push('\n');
                        context.push_str(&format!("[{} ctx] {}", next.source_doc, next.text));
                    }
                }
            }
        }
        context
    }

    pub fn size(&self) -> usize {
        self.entries.len()
    }

    pub fn all_entries(&self) -> &[MemoryEntry] {
        &self.entries
    }

    pub fn save_bin(&self, path: &str) -> Result<(), String> {
        use std::io::Write;
        let mut f =
            std::fs::File::create(path).map_err(|e| format!("Failed to create {}: {}", path, e))?;
        let count = self.entries.len() as u32;
        f.write_all(&count.to_le_bytes())
            .map_err(|e| format!("Failed to write count: {}", e))?;
        for entry in &self.entries {
            let dim = entry.vector.dim as u32;
            f.write_all(&dim.to_le_bytes())
                .map_err(|e| format!("Failed to write dim: {}", e))?;
            let word_bytes: Vec<u8> = entry
                .vector
                .words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            f.write_all(&word_bytes)
                .map_err(|e| format!("Failed to write vector: {}", e))?;

            let text_bytes = entry.text.as_bytes();
            let text_len = text_bytes.len() as u32;
            f.write_all(&text_len.to_le_bytes())
                .map_err(|e| format!("Failed to write text len: {}", e))?;
            f.write_all(text_bytes)
                .map_err(|e| format!("Failed to write text: {}", e))?;

            let doc_bytes = entry.source_doc.as_bytes();
            let doc_len = doc_bytes.len() as u32;
            f.write_all(&doc_len.to_le_bytes())
                .map_err(|e| format!("Failed to write doc len: {}", e))?;
            f.write_all(doc_bytes)
                .map_err(|e| format!("Failed to write doc: {}", e))?;

            let role_bytes = entry.role_hint.as_bytes();
            let role_len = role_bytes.len() as u32;
            f.write_all(&role_len.to_le_bytes())
                .map_err(|e| format!("Failed to write role len: {}", e))?;
            f.write_all(role_bytes)
                .map_err(|e| format!("Failed to write role: {}", e))?;
        }
        Ok(())
    }

    pub fn save_text_index(&self, path: &str) -> Result<(), String> {
        use std::io::Write;
        let index = match self.text_index {
            Some(ref idx) => idx,
            None => return Err("No text index to save".to_string()),
        };
        let mut f =
            std::fs::File::create(path).map_err(|e| format!("Failed to create {}: {}", path, e))?;
        let count = index.len() as u32;
        f.write_all(&count.to_le_bytes())
            .map_err(|e| format!("Failed to write index count: {}", e))?;
        for (word, entries) in index {
            let word_bytes = word.as_bytes();
            f.write_all(&(word_bytes.len() as u32).to_le_bytes())
                .map_err(|e| format!("Failed to write word len: {}", e))?;
            f.write_all(word_bytes)
                .map_err(|e| format!("Failed to write word: {}", e))?;
            f.write_all(&(entries.len() as u32).to_le_bytes())
                .map_err(|e| format!("Failed to write entry count: {}", e))?;
            for &eid in entries {
                f.write_all(&eid.to_le_bytes())
                    .map_err(|e| format!("Failed to write entry id: {}", e))?;
            }
        }
        Ok(())
    }

    pub fn load_text_index(&mut self, path: &str) -> Result<(), String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open index {}: {}", path, e))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| format!("Failed to mmap index {}: {}", path, e))?;
        let data = &mmap[..];
        let mut pos = 0usize;

        // Guard-ридеры (аудит 22.08): усечённый/малформированный индекс → Err,
        // не паника; счётчики из файла ограничивают with_capacity размером файла.
        let count = rd_u32(data, &mut pos)?;
        let mut index =
            HashMap::with_capacity((count.min((data.len() / 8 + 1) as u32)) as usize);
        for _ in 0..count {
            let word_len = rd_u32(data, &mut pos)? as usize;
            let word = String::from_utf8(rd_bytes(data, &mut pos, word_len)?.to_vec())
                .map_err(|e| format!("UTF-8 error in index: {}", e))?;

            let entry_count = rd_u32(data, &mut pos)?;
            let mut entries = Vec::with_capacity((entry_count as usize).min(data.len() / 4 + 1));
            for _ in 0..entry_count {
                entries.push(rd_u32(data, &mut pos)?);
            }
            index.insert(word, entries);
        }
        self.text_index = Some(index);
        Ok(())
    }

    pub fn build_vsa_index(&mut self) {
        if self.entries.len() < 2 {
            return;
        }
        eprintln!(
            "  Building VSA-LSH index for {} vectors...",
            self.entries.len()
        );
        let start = std::time::Instant::now();
        let hv: Vec<Hypervector> = self.entries.iter().map(|e| e.vector.clone()).collect();
        let idx = VsaIndex::build(&hv);
        eprintln!(
            "  VSA-LSH built in {:.2}s ({} buckets x {} tables)",
            start.elapsed().as_secs_f64(),
            1 << crate::ai::hnsw::HASH_BITS,
            crate::ai::hnsw::NUM_TABLES
        );
        self.vsa_idx = Some(idx);
    }

    pub fn save_vsa_index(&self, path: &str) -> Result<(), String> {
        match self.vsa_idx {
            Some(ref idx) => idx.save(path),
            None => Err("No VSA index to save".to_string()),
        }
    }

    pub fn load_vsa_index(&mut self, path: &str) -> Result<(), String> {
        let dim = if self.entries.is_empty() {
            8192
        } else {
            self.entries[0].vector.dim
        };
        let idx = VsaIndex::load(path, dim)?;
        if idx.size() != self.entries.len() {
            return Err(format!(
                "VSA index size {} != entries {}",
                idx.size(),
                self.entries.len()
            ));
        }
        self.vsa_idx = Some(idx);
        Ok(())
    }

    pub fn load_bin(path: &str) -> Result<Self, String> {
        let file =
            std::fs::File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| format!("Failed to mmap {}: {}", path, e))?;
        let data = &mmap[..];
        let mut pos = 0usize;

        // Guard-ридеры (аудит 22.08): любая длина/счётчик из файла проверяется
        // против остатка ДО среза и ДО аллокации (0xFFFFFFFF больше не
        // заказывает сотни ГБ и не паникует на усеченном файле).
        let count = rd_u32(data, &mut pos)?;
        let mut entries = Vec::with_capacity((count as usize).min(data.len() / 16 + 1));

        for _ in 0..count {
            let dim = rd_u32(data, &mut pos)? as usize;
            let wc = (dim + 63) / 64;
            let vec_bytes = wc * 8;
            if vec_bytes > data.len().saturating_sub(pos) {
                return Err(format!(
                    "corrupt bin: vector {} bytes overruns file at {}",
                    vec_bytes, pos
                ));
            }
            let mut words = vec![0u64; wc];
            for (i, w) in words.iter_mut().enumerate() {
                *w = rd_u64(data, &mut pos)?;
            }
            let vector = Hypervector { dim, words };

            let text_len = rd_u32(data, &mut pos)? as usize;
            let text = String::from_utf8(rd_bytes(data, &mut pos, text_len)?.to_vec())
                .map_err(|e| format!("UTF-8 error: {}", e))?;

            let doc_len = rd_u32(data, &mut pos)? as usize;
            let source_doc = String::from_utf8(rd_bytes(data, &mut pos, doc_len)?.to_vec())
                .map_err(|e| format!("UTF-8 error: {}", e))?;

            let role_len = rd_u32(data, &mut pos)? as usize;
            let role_hint = String::from_utf8(rd_bytes(data, &mut pos, role_len)?.to_vec())
                .map_err(|e| format!("UTF-8 error: {}", e))?;

            entries.push(MemoryEntry {
                vector,
                text,
                source_doc,
                role_hint,
            });
        }
        Ok(Self {
            entries,
            text_index: None,
            vsa_idx: None,
            attractor_idx: None,
        })
    }

    pub fn build_attractor_index(&mut self) {
        if self.entries.len() < NUM_ATTRACTORS {
            return;
        }
        let dim = self.entries[0].vector.dim;
        eprintln!(
            "  Building Attractor index ({} clusters)...",
            NUM_ATTRACTORS
        );
        let start = std::time::Instant::now();
        let idx = AttractorIndex::build(&self.entries, dim);
        eprintln!(
            "  Attractor index built in {:.2}s",
            start.elapsed().as_secs_f64()
        );
        self.attractor_idx = Some(idx);
    }

    pub fn save_attractor_index(&self, path: &str) -> Result<(), String> {
        use std::io::Write;
        let idx = match self.attractor_idx {
            Some(ref idx) => idx,
            None => return Err("No attractor index to save".to_string()),
        };
        let mut f =
            std::fs::File::create(path).map_err(|e| format!("Failed to create {}: {}", path, e))?;
        for a in &idx.attractors {
            let dim = a.dim as u32;
            f.write_all(&dim.to_le_bytes()).map_err(|e| e.to_string())?;
            for w in &a.words {
                f.write_all(&w.to_le_bytes()).map_err(|e| e.to_string())?;
            }
        }
        for cluster in &idx.clusters {
            f.write_all(&(cluster.len() as u32).to_le_bytes())
                .map_err(|e| e.to_string())?;
            for &eid in cluster {
                f.write_all(&eid.to_le_bytes()).map_err(|e| e.to_string())?;
            }
        }
        Ok(())
    }

    pub fn load_attractor_index(&mut self, path: &str) -> Result<(), String> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open attractor index {}: {}", path, e))?;
        let dim = if self.entries.is_empty() {
            8192
        } else {
            self.entries[0].vector.dim
        };
        let _wc = (dim + 63) / 64;
        let mut attractors = Vec::with_capacity(NUM_ATTRACTORS);
        for _ in 0..NUM_ATTRACTORS {
            let mut dim_buf = [0u8; 4];
            f.read_exact(&mut dim_buf).map_err(|e| e.to_string())?;
            let adim = u32::from_le_bytes(dim_buf) as usize;
            let awc = (adim + 63) / 64;
            let mut words = vec![0u64; awc];
            let mut buf = vec![0u8; awc * 8];
            f.read_exact(&mut buf).map_err(|e| e.to_string())?;
            for i in 0..awc {
                let mut b = [0u8; 8];
                b.copy_from_slice(&buf[i * 8..(i + 1) * 8]);
                words[i] = u64::from_le_bytes(b);
            }
            attractors.push(Hypervector { dim: adim, words });
        }
        let mut clusters = Vec::with_capacity(NUM_ATTRACTORS);
        for _ in 0..NUM_ATTRACTORS {
            let mut len_buf = [0u8; 4];
            f.read_exact(&mut len_buf).map_err(|e| e.to_string())?;
            let count = u32::from_le_bytes(len_buf) as usize;
            let mut cluster = Vec::with_capacity(count);
            for _ in 0..count {
                let mut eid_buf = [0u8; 4];
                f.read_exact(&mut eid_buf).map_err(|e| e.to_string())?;
                cluster.push(u32::from_le_bytes(eid_buf));
            }
            clusters.push(cluster);
        }
        self.attractor_idx = Some(AttractorIndex {
            attractors,
            clusters,
        });
        Ok(())
    }
}

impl MemoryStore {
    /// Append entries to an existing memory file without loading existing entries.
    /// Opens file in read-write, seeks to end, appends entries, updates count at offset 0.
    pub fn append_entries(path: &str, new_entries: &[MemoryEntry]) -> Result<usize, String> {
        use std::fs::OpenOptions;
        use std::io::{Read, Seek, SeekFrom, Write};

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| format!("Failed to open {}: {}", path, e))?;

        // Read current count
        let mut count_buf = [0u8; 4];
        file.read_exact(&mut count_buf)
            .map_err(|e| format!("Failed to read count: {}", e))?;
        let old_count = u32::from_le_bytes(count_buf) as usize;

        // Seek to end
        file.seek(SeekFrom::End(0))
            .map_err(|e| format!("Seek failed: {}", e))?;

        // Write new entries
        for entry in new_entries {
            let dim = entry.vector.dim as u32;
            file.write_all(&dim.to_le_bytes())
                .map_err(|e| format!("Failed to write dim: {}", e))?;

            let word_bytes: Vec<u8> = entry
                .vector
                .words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            file.write_all(&word_bytes)
                .map_err(|e| format!("Failed to write vector: {}", e))?;

            let text_bytes = entry.text.as_bytes();
            let text_len = text_bytes.len() as u32;
            file.write_all(&text_len.to_le_bytes())
                .map_err(|e| format!("Failed to write text len: {}", e))?;
            file.write_all(text_bytes)
                .map_err(|e| format!("Failed to write text: {}", e))?;

            let doc_bytes = entry.source_doc.as_bytes();
            let doc_len = doc_bytes.len() as u32;
            file.write_all(&doc_len.to_le_bytes())
                .map_err(|e| format!("Failed to write doc len: {}", e))?;
            file.write_all(doc_bytes)
                .map_err(|e| format!("Failed to write doc: {}", e))?;

            let role_bytes = entry.role_hint.as_bytes();
            let role_len = role_bytes.len() as u32;
            file.write_all(&role_len.to_le_bytes())
                .map_err(|e| format!("Failed to write role len: {}", e))?;
            file.write_all(role_bytes)
                .map_err(|e| format!("Failed to write role: {}", e))?;
        }

        // Update count at beginning
        let new_count = (old_count + new_entries.len()) as u32;
        file.seek(SeekFrom::Start(0))
            .map_err(|e| format!("Seek to start failed: {}", e))?;
        file.write_all(&new_count.to_le_bytes())
            .map_err(|e| format!("Failed to write new count: {}", e))?;
        file.flush().map_err(|e| format!("Flush failed: {}", e))?;

        Ok(new_entries.len())
    }
}

use crate::ai::self_mirror::SelfMirror;
use crate::ai::sdr::SDR_DIM;
use crate::core::hypervector::Hypervector;
use crate::core::tokenizer_bridge::encode_bytes_nopos;

pub const CRYSTAL_MAGIC: &[u8] = b"FUGA_XL1";
pub const CRYSTAL_VERSION: u8 = 1;
pub const DEFAULT_DIM: usize = 8192;
pub const DEFAULT_RESONANCE_THRESHOLD: f64 = 0.35;
pub const KIND_L0: u8 = 0;
pub const KIND_L1: u8 = 1;
pub const KIND_L2: u8 = 2;

// FNV-1a 64: stable, deterministic binary key for the L0 hash index.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[derive(Clone, Debug)]
pub struct CrystalEntry {
    pub hv: Hypervector,
    pub key: u64,
    pub key_text: String,
    pub resonance: f32,
    pub route: u16,
    pub kind: u8,
    pub text: String,
}

pub struct PhaseCrystal {
    pub dim: usize,
    pub entries: Vec<CrystalEntry>,
    pub l0_index: std::collections::HashMap<u64, usize>,
    pub threshold: f64,
}

impl PhaseCrystal {
    pub fn new(dim: usize, threshold: f64) -> Self {
        PhaseCrystal { dim, entries: Vec::new(), l0_index: std::collections::HashMap::new(), threshold }
    }

    // --- Build: transpile trained mirror nodes into the phase crystal ---
    pub fn build_from_mirror(mirror: &SelfMirror, max_entries: usize, threshold: f64) -> Self {
        let dim = SDR_DIM;
        let mut crystal = PhaseCrystal::new(dim, threshold);
        let nodes = &mirror.nodes;
        let mut scored: Vec<(usize, f32)> = nodes.iter().enumerate()
            .map(|(i, n)| {
                let res = 1.0 - ((n.l0_err + n.l1_err) / 2.0) as f32;
                (i, res)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (ni, res) in scored.into_iter().take(max_entries) {
            let node = &nodes[ni];
            let text_key = format!("{} {}", node.kind, node.name);
            let hv: Hypervector = encode_bytes_nopos(text_key.as_bytes(), dim);
            let key = fnv1a(text_key.as_bytes());
            let route = ((key >> 8) & 0xFF) as u16;
            let snippet = mirror.source_snippet_for_path(&node.path, node.line, 6);
            let text = if snippet.is_empty() {
                format!("{} {} ({})", node.kind, node.name, node.path)
            } else {
                format!("{} {} ({})\n{}", node.kind, node.name, node.path, snippet)
            };
            let text = text.chars().take(200).collect();
            let entry = CrystalEntry { hv, key, key_text: text_key, resonance: res, route, kind: KIND_L1, text };
            let idx = crystal.entries.len();
            crystal.l0_index.insert(key, idx);
            crystal.entries.push(entry);
        }

        // L2 Concept Network: bundle per-kind phase profiles into concept vectors (majority vote)
        let mut kind_hvs: Vec<(String, Vec<Hypervector>)> = Vec::new();
        for e in &crystal.entries {
            let kind = kind_of_text(&e.text);
            match kind_hvs.iter_mut().find(|(k, _)| *k == kind) {
                Some((_, v)) => v.push(e.hv.clone()),
                None => kind_hvs.push((kind.to_string(), vec![e.hv.clone()])),
            }
        }
        for (k, hv_list) in kind_hvs {
            let hv = majority_bundle(&hv_list, dim);
            let text_key = format!("concept:{}", k);
            let key = fnv1a(text_key.as_bytes());
            let route = ((key >> 8) & 0xFF) as u16;
            let entry = CrystalEntry {
                hv,
                key,
                key_text: text_key.clone(),
                resonance: 0.0,
                route,
                kind: KIND_L2,
                text: text_key,
            };
            let idx = crystal.entries.len();
            crystal.l0_index.insert(key, idx);
            crystal.entries.push(entry);
        }
        crystal
    }

    // --- Query: Popcount(Query XOR Dump) resonance scan, O(1) exact via L0 hash index ---
    pub fn query(&self, text: &str) -> Option<CrystalHit> {
        self.query_threshold(text, self.threshold)
    }

    pub fn query_threshold(&self, text: &str, threshold: f64) -> Option<CrystalHit> {
        // Phase 1 (O(1)): exact binary key on the L0 hash index — pure hashmap hit.
        let key = fnv1a(text.as_bytes());
        if let Some(&idx) = self.l0_index.get(&key) {
            let e = &self.entries[idx];
            return Some(CrystalHit { entry: e.clone(), resonance: e.resonance.max(1.0), matched: true, exact: true });
        }
        // Phase 2: resonator matrix scan — popcount XOR over the dump.
        let qhv = encode_bytes_nopos(text.as_bytes(), self.dim);
        let mut best: Option<(f64, usize)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            let o = overlap_score(&qhv, &e.hv);
            if best.map_or(true, |(bs, _)| o > bs) {
                best = Some((o, i));
            }
        }
        let (best_o, best_i) = best?;
        if best_o >= threshold {
            let e = &self.entries[best_i];
            Some(CrystalHit { entry: e.clone(), resonance: best_o as f32, matched: true, exact: false })
        } else {
            None
        }
    }

    // Raw O(1) associative lookup timing: popcount of (Query XOR entry) via count_ones.
    pub fn popcount_scan(&self, text: &str) -> (usize, Vec<(usize, u64)>) {
        let qhv = encode_bytes_nopos(text.as_bytes(), self.dim);
        let mut scored = Vec::with_capacity(self.entries.len());
        for (i, e) in self.entries.iter().enumerate() {
            let mut xor_pop = 0u64;
            for w in 0..qhv.words.len() {
                xor_pop += (qhv.words[w] ^ e.hv.words[w]).count_ones() as u64;
            }
            scored.push((i, xor_pop));
        }
        scored.sort_by(|a, b| a.1.cmp(&b.1));
        (self.entries.len(), scored.into_iter().take(5).collect())
    }

    pub fn stats(&self) -> String {
        let bytes = self.dim / 8;
        let mem = self.entries.len() * bytes;
        let l1 = self.entries.iter().filter(|e| e.kind == KIND_L1).count();
        let l2 = self.entries.iter().filter(|e| e.kind == KIND_L2).count();
        format!(
            "crystal: {} entries (L1={} L2={}) dim={} core={:.2}MB l0_keys={} threshold={:.2}",
            self.entries.len(), l1, l2, self.dim,
            mem as f64 / 1_048_576.0, self.l0_index.len(), self.threshold,
        )
    }

    // --- Serialization: header + entries (original order) + L0 index + text ---
    // Entries are written in `self.entries` order so that l0_index offsets
    // (combined indices) stay valid after online learn() appends mixed kinds.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(CRYSTAL_MAGIC);
        buf.push(CRYSTAL_VERSION);
        buf.extend_from_slice(&(self.dim as u32).to_le_bytes());
        let n1 = self.entries.iter().filter(|e| e.kind == KIND_L1).count();
        let n2 = self.entries.len() - n1;
        buf.extend_from_slice(&(n1 as u32).to_le_bytes());
        buf.extend_from_slice(&(n2 as u32).to_le_bytes());
        buf.extend_from_slice(&(self.l0_index.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.threshold.to_le_bytes());
        for e in &self.entries { Self::write_entry(&mut buf, e); }
        // L0 hash index: (key, combined entry offset) pairs
        for (k, &off) in &self.l0_index {
            buf.extend_from_slice(&k.to_le_bytes());
            buf.extend_from_slice(&(off as u32).to_le_bytes());
        }
        std::fs::write(path, &buf).map_err(|e| format!("save {}: {}", path, e))
    }

    fn write_entry(buf: &mut Vec<u8>, e: &CrystalEntry) {
        for w in &e.hv.words { buf.extend_from_slice(&w.to_le_bytes()); }
        buf.extend_from_slice(&e.key.to_le_bytes());
        let ktb = e.key_text.as_bytes();
        buf.extend_from_slice(&(ktb.len() as u32).to_le_bytes());
        buf.extend_from_slice(ktb);
        buf.extend_from_slice(&e.resonance.to_le_bytes());
        buf.extend_from_slice(&e.route.to_le_bytes());
        buf.push(e.kind);
        let tb = e.text.as_bytes();
        buf.extend_from_slice(&(tb.len() as u32).to_le_bytes());
        buf.extend_from_slice(tb);
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read {}: {}", path, e))?;
        if data.len() < 25 { return Err("too short".into()); }
        if &data[..8] != CRYSTAL_MAGIC { return Err("bad magic".into()); }
        if data[8] != CRYSTAL_VERSION { return Err("bad version".into()); }
        let dim = u32::from_le_bytes(data[9..13].try_into().unwrap()) as usize;
        let n1 = u32::from_le_bytes(data[13..17].try_into().unwrap()) as usize;
        let n2 = u32::from_le_bytes(data[17..21].try_into().unwrap()) as usize;
        let n0 = u32::from_le_bytes(data[21..25].try_into().unwrap()) as usize;
        let threshold = f64::from_le_bytes(data[25..33].try_into().unwrap());
        let mut pos = 33usize;
        let mut entries: Vec<CrystalEntry> = Vec::new();
        for _ in 0..(n1 + n2) {
            let wc = (dim + 63) / 64;
            if pos + wc * 8 > data.len() { return Err("short hv".into()); }
            let mut words = vec![0u64; wc];
            for j in 0..wc {
                words[j] = u64::from_le_bytes(data[pos+j*8..pos+(j+1)*8].try_into().unwrap());
            }
            pos += wc * 8;
            if pos + 8 + 4 + 4 + 2 + 1 + 4 > data.len() { return Err("short entry head".into()); }
            let key = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap());
            pos += 8;
            let ktlen = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + ktlen > data.len() { return Err("short key_text".into()); }
            let key_text = String::from_utf8(data[pos..pos+ktlen].to_vec()).unwrap_or_default();
            pos += ktlen;
            let resonance = f32::from_le_bytes(data[pos..pos+4].try_into().unwrap());
            pos += 4;
            let route = u16::from_le_bytes(data[pos..pos+2].try_into().unwrap());
            pos += 2;
            let kind = data[pos];
            pos += 1;
            let tlen = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + tlen > data.len() { return Err("short text".into()); }
            let text = String::from_utf8(data[pos..pos+tlen].to_vec()).unwrap_or_default();
            pos += tlen;
            entries.push(CrystalEntry { hv: Hypervector::from_raw(dim, words), key, key_text, resonance, route, kind, text });
        }
        let mut l0_index = std::collections::HashMap::with_capacity(n0);
        for _ in 0..n0 {
            if pos + 12 > data.len() { break; }
            let k = u64::from_le_bytes(data[pos..pos+8].try_into().unwrap());
            let off = u32::from_le_bytes(data[pos+8..pos+12].try_into().unwrap()) as usize;
            pos += 12;
            l0_index.insert(k, off);
        }
        Ok(PhaseCrystal { dim, entries, l0_index, threshold })
    }

    /// Re-encode every entry's stored phase from its own text using the
    /// position-invariant n-gram encoder. Needed when the encoder changes
    /// (e.g. positional → nopos): entries written by an older encoder no
    /// longer resonate with queries. L2 concept vectors are rebuilt from the
    /// re-encoded L1 phases. Returns the number of entries touched.
    pub fn reencode_nopos(&mut self) -> usize {
        let mut touched = 0usize;
        for e in self.entries.iter_mut() {
            let content = if e.text.trim().is_empty() {
                e.key_text.clone()
            } else {
                e.text.clone()
            };
            let new_hv = encode_bytes_nopos(content.as_bytes(), self.dim);
            if new_hv.words != e.hv.words {
                e.hv = new_hv;
                touched += 1;
            }
        }
        // Rebuild L2 concept bundles from the re-encoded L1 phases.
        let mut kind_hvs: Vec<(String, Vec<Hypervector>)> = Vec::new();
        for e in self.entries.iter().filter(|e| e.kind == KIND_L1) {
            let kind = kind_of_text(&e.text);
            match kind_hvs.iter_mut().find(|(k, _)| *k == kind) {
                Some((_, v)) => v.push(e.hv.clone()),
                None => kind_hvs.push((kind.to_string(), vec![e.hv.clone()])),
            }
        }
        let dim = self.dim;
        for (k, hv_list) in kind_hvs {
            if let Some(e) = self.entries.iter_mut().find(|e| e.kind == KIND_L2 && e.key_text == format!("concept:{}", k)) {
                e.hv = majority_bundle(&hv_list, dim);
            }
        }
        touched
    }

    // --- Online phase adaptation (Hebbian plasticity) ---

    /// Weighted majority blend: `(1-alpha)` weight on `a`, `alpha` on `b`.
    /// Odd sample count (11) avoids tie-to-zero on 50/50 splits. This is the
    /// `v_new = v_old + alpha*(A⊗B)` update, applied bitwise.
    pub fn weighted_majority(a: &Hypervector, b: &Hypervector, alpha: f64) -> Hypervector {
        let n = 11usize;
        let a_w = ((1.0 - alpha).max(0.0) * n as f64).round() as usize;
        let b_w = n - a_w;
        let threshold = n / 2;
        let wc = a.words.len();
        let mut result = vec![0u64; wc];
        for w in 0..wc {
            let aw = if w < a.words.len() { a.words[w] } else { 0 };
            let bw = if w < b.words.len() { b.words[w] } else { 0 };
            let mut rw = 0u64;
            for bit in 0..64 {
                let mut count = 0u16;
                if ((aw >> bit) & 1) == 1 { count += a_w as u16; }
                if ((bw >> bit) & 1) == 1 { count += b_w as u16; }
                if count > threshold as u16 { rw |= 1u64 << bit; }
            }
            result[w] = rw;
        }
        Hypervector { dim: a.dim, words: result }
    }

    /// Learn a new fact (or Hebbian-blend into an existing key). Returns
    /// `(index, updated_existing)`. `concept:` keys become L2 phase vectors,
    /// everything else a keyed L1 phase profile.
    pub fn learn(&mut self, key_text: &str, text: &str, alpha: f64) -> (usize, bool) {
        // Encode the *content*, not the key. The key already routes exact
        // hits through the L0 hash index; stuffing the (often long, path-like)
        // key into the n-gram VSA would crowd out MAX_GRAMS=48 content grams
        // and destroy fuzzy resonance on the body. Stored HV = content only.
        let content = if text.trim().is_empty() {
            key_text.to_string()
        } else {
            text.to_string()
        };
        let hv: Hypervector = encode_bytes_nopos(content.as_bytes(), self.dim);
        let key = fnv1a(key_text.as_bytes());
        if let Some(&idx) = self.l0_index.get(&key) {
            let old = &self.entries[idx].hv;
            self.entries[idx].hv = Self::weighted_majority(old, &hv, alpha);
            self.entries[idx].resonance = 1.0 - ((1.0 - self.entries[idx].resonance) * 0.5);
            self.entries[idx].text = text.to_string();
            return (idx, true);
        }
        let route = ((key >> 8) & 0xFF) as u16;
        let kind = if key_text.starts_with("concept:") { KIND_L2 } else { KIND_L1 };
        let entry = CrystalEntry {
            hv,
            key,
            key_text: key_text.to_string(),
            resonance: 1.0,
            route,
            kind,
            text: text.to_string(),
        };
        let idx = self.entries.len();
        self.l0_index.insert(key, idx);
        self.entries.push(entry);
        (idx, false)
    }

    /// Remove a phase by exact key. Returns true if found and removed.
    pub fn forget(&mut self, key_text: &str) -> bool {
        let key = fnv1a(key_text.as_bytes());
        if !self.l0_index.contains_key(&key) {
            return false;
        }
        self.entries.retain(|e| e.key != key);
        self.l0_index.clear();
        for (i, e) in self.entries.iter().enumerate() {
            self.l0_index.insert(e.key, i);
        }
        true
    }
}

#[derive(Clone, Debug)]
pub struct CrystalHit {
    pub entry: CrystalEntry,
    pub resonance: f32,
    pub matched: bool,
    pub exact: bool,
}

fn overlap_score(q: &Hypervector, e: &Hypervector) -> f64 {
    // soft_overlap: intersection / max(popcount). Density-agnostic — both
    // query and stored phase come from the n-gram VSA encoder (tokenizer
    // bridge), so a bounded-gram query vs. a bounded-gram chunk scores
    // cleanly without cosine's sqrt-penalty on unbalanced densities.
    let mut overlap = 0u32;
    let n = q.words.len().min(e.words.len());
    for w in 0..n {
        overlap += (q.words[w] & e.words[w]).count_ones();
    }
    let qnz = q.words.iter().map(|w| w.count_ones()).sum::<u32>();
    let enz = e.words.iter().map(|w| w.count_ones()).sum::<u32>();
    let denom = qnz.max(enz) as f64;
    if denom <= 0.0 { 0.0 } else { overlap as f64 / denom }
}

fn majority_bundle(hvs: &[Hypervector], dim: usize) -> Hypervector {
    let wc = dim.div_ceil(64);
    let mut words = vec![0u64; wc];
    if hvs.is_empty() {
        return Hypervector { dim, words };
    }
    let n = hvs.len();
    let mid = n / 2;
    for w in 0..wc {
        let mut bits = 0u64;
        for bit in 0..64 {
            let ones = hvs.iter().filter(|h| w < h.words.len() && ((h.words[w] >> bit) & 1) == 1).count();
            if ones > mid {
                bits |= 1u64 << bit;
            }
        }
        words[w] = bits;
    }
    Hypervector { dim, words }
}

fn kind_of_text(text: &str) -> &str {
    let first = text.split_whitespace().next().unwrap_or("unknown");
    if first.starts_with("concept:") { "concept" } else { first }
}

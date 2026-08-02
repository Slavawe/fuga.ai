use crate::ai::self_mirror::SelfMirror;
use crate::core::hypervector::Hypervector;
use crate::core::tokenizer_bridge::{encode_bytes_nopos, encode_bytes_nopos_min3};

pub const CRYSTAL_MAGIC: &[u8] = b"FUGA_XL1";
pub const CRYSTAL_VERSION: u8 = 2;
pub const DEFAULT_DIM: usize = 8192;
pub const DEFAULT_RESONANCE_THRESHOLD: f64 = 0.35;
pub const KIND_L0: u8 = 0;
pub const KIND_L1: u8 = 1;
pub const KIND_L2: u8 = 2;

// --- Fractal hierarchy: each level owns its own VSA dimension ---
// L0 (tokens/syntax)   → 8192  — fast, cheap, exact byte-level fingerprinting
// L1 (functions/blocks) → 16384 — aggregates L0 phases, holds argument/type relations
// L2 (meta concepts)    → 32768 — concept bundles over whole knowledge, max noise immunity
pub const DIM_L0: usize = 8192;
pub const DIM_L1: usize = 16384;
pub const DIM_L2: usize = 32768;
// L2 lives in 4x the space of L0; random crosstalk drops ~sqrt(D), so its
// acceptance threshold can be much lower without false positives.
pub const L2_THRESHOLD_SCALE: f64 = 0.4;

pub fn dim_for_kind(kind: u8) -> usize {
    match kind {
        KIND_L0 => DIM_L0,
        KIND_L2 => DIM_L2,
        _ => DIM_L1,
    }
}

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
        PhaseCrystal {
            dim,
            entries: Vec::new(),
            l0_index: std::collections::HashMap::new(),
            threshold,
        }
    }

    // --- Build: transpile trained mirror nodes into the phase crystal ---
    pub fn build_from_mirror(mirror: &SelfMirror, max_entries: usize, threshold: f64) -> Self {
        // L0 syntax layer uses the smallest dim (fast, exact fingerprinting);
        // L1 phase profiles are encoded at DIM_L1, L2 concept bundles at DIM_L2.
        let dim = DIM_L1;
        let mut crystal = PhaseCrystal::new(dim, threshold);
        let nodes = &mirror.nodes;
        let mut scored: Vec<(usize, f32)> = nodes
            .iter()
            .enumerate()
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
            let entry = CrystalEntry {
                hv,
                key,
                key_text: text_key,
                resonance: res,
                route,
                kind: KIND_L1,
                text,
            };
            let idx = crystal.entries.len();
            crystal.l0_index.insert(key, idx);
            crystal.entries.push(entry);
        }

        // L2 Concept Network: bundle per-kind phase profiles into concept
        // vectors (majority vote) in the 32k L2 space — each member is
        // re-encoded at DIM_L2 so the concept inherits L2 noise immunity.
        let mut kind_contents: Vec<(String, Vec<String>)> = Vec::new();
        for e in &crystal.entries {
            let kind = kind_of_text(&e.text);
            let content = if e.text.trim().is_empty() {
                e.key_text.clone()
            } else {
                e.text.clone()
            };
            match kind_contents.iter_mut().find(|(k, _)| *k == kind) {
                Some((_, v)) => v.push(content),
                None => kind_contents.push((kind.to_string(), vec![content])),
            }
        }
        for (k, contents) in kind_contents {
            let hv_list: Vec<Hypervector> = contents
                .iter()
                .map(|c| encode_bytes_nopos(c.as_bytes(), DIM_L2))
                .collect();
            let hv = majority_bundle(&hv_list, DIM_L2);
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
        self.query_config(text, QueryConfig::default())
    }

    pub fn query_threshold(&self, text: &str, threshold: f64) -> Option<CrystalHit> {
        let cfg = QueryConfig {
            threshold,
            ..QueryConfig::default()
        };
        self.query_config(text, cfg)
    }

    pub fn query_config(&self, text: &str, cfg: QueryConfig) -> Option<CrystalHit> {
        let threshold = cfg.threshold;
        // Phase 1 (O(1)): exact binary key on the L0 hash index — pure hashmap hit.
        let key = fnv1a(text.as_bytes());
        if let Some(&idx) = self.l0_index.get(&key) {
            let e = &self.entries[idx];
            return Some(CrystalHit {
                entry: e.clone(),
                resonance: e.resonance.max(1.0),
                matched: true,
                exact: true,
            });
        }
        // Phase 2: hierarchical resonator scan. Entries are grouped by their
        // own dimension (L0=8k, L1=16k, L2=32k); the query is encoded once per
        // group in that group's space. Higher-dim groups get a proportionally
        // lower acceptance threshold (cfg.l2_scale, default L2_THRESHOLD_SCALE)
        // to catch the ultra-fine associative links the smaller spaces would
        // drown. With cfg.gate_l1 (strict Router-Gate mode) a candidate may
        // only be accepted if the query also resonates on the L1 (16k) level —
        // L2 never fires on its own, killing borderline 0.14–0.16 false hits
        // while keeping every real semantic association.
        let mut best: Option<(f64, f64, usize)> = None;
        let mut l1_ok = !cfg.gate_l1;

        let mut dims: Vec<usize> = self.entries.iter().map(|e| e.hv.dim).collect();
        dims.sort_unstable();
        dims.dedup();
        for d in dims {
            let is_l2 = d >= DIM_L2;
            let scale = if is_l2 { cfg.l2_scale } else { 1.0 };
            let group_thr = threshold * scale;
            let qhv = encode_bytes_nopos(text.as_bytes(), d);
            for (i, e) in self.entries.iter().enumerate() {
                if e.hv.dim != d {
                    continue;
                }
                let o = overlap_score(&qhv, &e.hv);
                if o >= group_thr && best.map_or(true, |(bs, _, _)| o > bs) {
                    if is_l2 && cfg.gate_l1 && !l1_ok {
                        continue;
                    }
                    best = Some((o, group_thr, i));
                }
            }
            if !is_l2 && !l1_ok && best.is_some() {
                l1_ok = true;
            }
        }

        let (best_o, best_thr, best_i) = best?;
        if best_o >= best_thr {
            let e = &self.entries[best_i];
            Some(CrystalHit {
                entry: e.clone(),
                resonance: best_o as f32,
                matched: true,
                exact: false,
            })
        } else {
            None
        }
    }

    // --- Cross-dimensional phase bridge (Hippocampal projection) ---
    /// Deterministic up-projection 8k → 32k (and any dim_a → dim_b).
    /// Each set bit of the source phase is scattered into the target space via
    /// a permute hash keyed by the source bit index — so the same source phase
    /// always lands on the same target phase (permutation, not duplication),
    /// and orthogonal sources stay near-orthogonal after projection. This lets
    /// a static 8k MoE crystal (long-term knowledge) bind into a dynamic 32k
    /// episodic crystal (working context) without re-encoding the knowledge.
    pub fn project(&self, from: &Hypervector, to_dim: usize) -> Hypervector {
        project_phase(from, to_dim)
    }

    // Raw O(1) associative lookup timing: popcount of (Query XOR entry) via count_ones.
    // Entries are scored in their own dim (L1 → 16k, L2 → 32k) with a query
    // vector encoded per-level, so mixed-dimension hierarchies compare cleanly.
    pub fn popcount_scan(&self, text: &str) -> (usize, Vec<(usize, u64)>) {
        let mut scored = Vec::with_capacity(self.entries.len());
        for (i, e) in self.entries.iter().enumerate() {
            let d = e.hv.dim;
            let qhv = encode_bytes_nopos(text.as_bytes(), d);
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
        let l1 = self.entries.iter().filter(|e| e.kind == KIND_L1).count();
        let l2 = self.entries.iter().filter(|e| e.kind == KIND_L2).count();
        let l0 = self.entries.iter().filter(|e| e.kind == KIND_L0).count();
        let mem: usize = self.entries.iter().map(|e| e.hv.dim / 8).sum();
        format!(
            "crystal: {} entries (L0={} L1={} L2={}) dims=8k/16k/32k core={:.2}MB l0_keys={} threshold={:.2} (L2×{:.1})",
            self.entries.len(),
            l0,
            l1,
            l2,
            mem as f64 / 1_048_576.0,
            self.l0_index.len(),
            self.threshold,
            L2_THRESHOLD_SCALE,
        )
    }

    // --- Serialization: header + entries (original order) + L0 index + text ---
    // v2: each entry stores its own word count (wc) so the hierarchical
    // L0(8k)/L1(16k)/L2(32k) dims coexist in one dump. v1 (single dim in
    // header) is still loadable for backward compatibility.
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
        for e in &self.entries {
            Self::write_entry(&mut buf, e);
        }
        // L0 hash index: (key, combined entry offset) pairs. Sort by key so
        // the dump is byte-deterministic across processes (HashMap order is
        // randomized per process and would break reproducible round-trips).
        let mut idx: Vec<(u64, usize)> = self.l0_index.iter().map(|(&k, &o)| (k, o)).collect();
        idx.sort_unstable();
        for (k, off) in idx {
            buf.extend_from_slice(&k.to_le_bytes());
            buf.extend_from_slice(&off.to_le_bytes());
        }
        std::fs::write(path, &buf).map_err(|e| format!("save {}: {}", path, e))
    }

    fn write_entry(buf: &mut Vec<u8>, e: &CrystalEntry) {
        let wc = e.hv.words.len() as u32;
        buf.extend_from_slice(&wc.to_le_bytes());
        for w in &e.hv.words {
            buf.extend_from_slice(&w.to_le_bytes());
        }
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
        if data.len() < 33 {
            return Err(format!("too short: {} bytes", data.len()));
        }
        if &data[..8] != CRYSTAL_MAGIC {
            return Err("bad magic".into());
        }
        let version = data[8];
        if version > CRYSTAL_VERSION {
            return Err(format!("bad version {}", version));
        }
        let dim = u32::from_le_bytes(data[9..13].try_into().unwrap()) as usize;
        let n1 = u32::from_le_bytes(data[13..17].try_into().unwrap()) as usize;
        let n2 = u32::from_le_bytes(data[17..21].try_into().unwrap()) as usize;
        let n0 = u32::from_le_bytes(data[21..25].try_into().unwrap()) as usize;
        let threshold = f64::from_le_bytes(data[25..33].try_into().unwrap());
        let mut pos = 33usize;
        let mut entries: Vec<CrystalEntry> = Vec::new();
        for _ in 0..(n1 + n2) {
            let wc = if version >= 2 {
                let wc32 = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
                pos += 4;
                wc32
            } else {
                (dim + 63) / 64
            };
            let e_dim = wc * 64;
            if pos + wc * 8 > data.len() {
                return Err("short hv".into());
            }
            let mut words = vec![0u64; wc];
            for j in 0..wc {
                words[j] =
                    u64::from_le_bytes(data[pos + j * 8..pos + (j + 1) * 8].try_into().unwrap());
            }
            pos += wc * 8;
            if pos + 8 + 4 + 4 + 2 + 1 + 4 > data.len() {
                return Err("short entry head".into());
            }
            let key = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            pos += 8;
            let ktlen = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + ktlen > data.len() {
                return Err("short key_text".into());
            }
            let key_text = String::from_utf8(data[pos..pos + ktlen].to_vec()).unwrap_or_default();
            pos += ktlen;
            let resonance = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let route = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap());
            pos += 2;
            let kind = data[pos];
            pos += 1;
            let tlen = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if pos + tlen > data.len() {
                return Err("short text".into());
            }
            let text = String::from_utf8(data[pos..pos + tlen].to_vec()).unwrap_or_default();
            pos += tlen;
            entries.push(CrystalEntry {
                hv: Hypervector::from_raw(e_dim, words),
                key,
                key_text,
                resonance,
                route,
                kind,
                text,
            });
        }
        let mut l0_index = std::collections::HashMap::with_capacity(n0);
        // v1 wrote the L0 offset as a u32 (12 B per pair); v2 uses usize (16 B).
        let idx_stride: usize = if version >= 2 { 16 } else { 12 };
        for _ in 0..n0 {
            if pos + idx_stride > data.len() {
                return Err(format!(
                    "truncated index: {} of {} pairs",
                    l0_index.len(),
                    n0
                ));
            }
            let k = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
            let off = if version >= 2 {
                u64::from_le_bytes(data[pos + 8..pos + 16].try_into().unwrap()) as usize
            } else {
                u32::from_le_bytes(data[pos + 8..pos + 12].try_into().unwrap()) as usize
            };
            pos += idx_stride;
            l0_index.insert(k, off);
        }
        Ok(PhaseCrystal {
            dim,
            entries,
            l0_index,
            threshold,
        })
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
            let new_hv = encode_bytes_nopos(content.as_bytes(), dim_for_kind(e.kind));
            if new_hv.words != e.hv.words || new_hv.dim != e.hv.dim {
                e.hv = new_hv;
                touched += 1;
            }
        }
        // Rebuild L2 concept bundles from the re-encoded L1 phases in L2 space.
        let mut kind_contents: Vec<(String, Vec<String>)> = Vec::new();
        for e in self.entries.iter().filter(|e| e.kind == KIND_L1) {
            let kind = kind_of_text(&e.text);
            let content = if e.text.trim().is_empty() {
                e.key_text.clone()
            } else {
                e.text.clone()
            };
            match kind_contents.iter_mut().find(|(k, _)| *k == kind) {
                Some((_, v)) => v.push(content),
                None => kind_contents.push((kind.to_string(), vec![content])),
            }
        }
        for (k, contents) in kind_contents {
            let hv_list: Vec<Hypervector> = contents
                .iter()
                .map(|c| encode_bytes_nopos(c.as_bytes(), DIM_L2))
                .collect();
            let new_hv = majority_bundle(&hv_list, DIM_L2);
            if let Some(e) = self
                .entries
                .iter_mut()
                .find(|e| e.kind == KIND_L2 && e.key_text == format!("concept:{}", k))
            {
                e.hv = new_hv;
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
                if ((aw >> bit) & 1) == 1 {
                    count += a_w as u16;
                }
                if ((bw >> bit) & 1) == 1 {
                    count += b_w as u16;
                }
                if count > threshold as u16 {
                    rw |= 1u64 << bit;
                }
            }
            result[w] = rw;
        }
        Hypervector {
            dim: a.dim,
            words: result,
        }
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
        let kind = if key_text.starts_with("concept:") {
            KIND_L2
        } else if self.dim >= DIM_L2 {
            // Episodic cortex crystal: everything lives in the 32k space.
            KIND_L2
        } else {
            KIND_L1
        };
        let hv: Hypervector = encode_bytes_nopos(content.as_bytes(), dim_for_kind(kind));
        let key = fnv1a(key_text.as_bytes());
        if let Some(&idx) = self.l0_index.get(&key) {
            let old = &self.entries[idx].hv;
            self.entries[idx].hv = Self::weighted_majority(old, &hv, alpha);
            self.entries[idx].resonance = 1.0 - ((1.0 - self.entries[idx].resonance) * 0.5);
            self.entries[idx].text = text.to_string();
            return (idx, true);
        }
        let route = ((key >> 8) & 0xFF) as u16;
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

/// Resonance tuning knobs for the hierarchical scan.
/// - `l2_scale`: L2 threshold multiplier. 0.4 = creative/associative (fires on
///   weak but real links), 0.5+ = stricter (cuts borderline 0.14–0.16 noise
///   while keeping all real semantic associations 0.31+).
/// - `gate_l1`: strict Router-Gate mode — L2 (32k) candidates are only
///   accepted if the query also resonates on L1 (16k). L2 never fires alone.
#[derive(Clone, Copy, Debug)]
pub struct QueryConfig {
    pub threshold: f64,
    pub l2_scale: f64,
    pub gate_l1: bool,
}

impl Default for QueryConfig {
    fn default() -> Self {
        QueryConfig {
            threshold: DEFAULT_RESONANCE_THRESHOLD,
            l2_scale: L2_THRESHOLD_SCALE,
            gate_l1: false,
        }
    }
}

/// Deterministic cross-dimensional phase projection (the Hippocampal bridge).
/// Scatters each set bit of `from` into `to_dim` using a permute hash keyed by
/// the source bit index. Linear probing resolves hash collisions so the set
/// popcount is preserved exactly (density → purity of the projection); the same
/// source phase always lands on the same target phase, and orthogonal sources
/// stay near-orthogonal. A projected phase can be bound (XOR) with native
/// phases of the target space.
pub fn project_phase(from: &Hypervector, to_dim: usize) -> Hypervector {
    let wc = to_dim.div_ceil(64);
    let mut words = vec![0u64; wc];
    let mut buf = [0u8; 8];
    let ratio = (to_dim as f64 / from.dim.max(1) as f64).max(1.0);
    for (wi, w) in from.words.iter().enumerate() {
        let mut bits = *w;
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let src_bit = wi * 64 + b;
            buf.copy_from_slice(&(src_bit as u64).to_le_bytes());
            let h = fnv1a(&buf);
            let base = (h % from.dim as u64) as usize;
            let mut pos = ((base as f64 * ratio) as usize) % to_dim;
            // Linear probing: keep the mapping a bijection on the set bits.
            while (words[pos / 64] >> (pos % 64)) & 1 == 1 {
                pos = (pos + 1) % to_dim;
            }
            words[pos / 64] |= 1u64 << (pos % 64);
        }
    }
    Hypervector { dim: to_dim, words }
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
    if denom <= 0.0 {
        0.0
    } else {
        overlap as f64 / denom
    }
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
            let ones = hvs
                .iter()
                .filter(|h| w < h.words.len() && ((h.words[w] >> bit) & 1) == 1)
                .count();
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
    if first.starts_with("concept:") {
        "concept"
    } else {
        first
    }
}

// ============================================================================
// Tri-Anchor Framework (Fuga 2.0): три базисных фазовых аттрактора, относительно
// которых проверяется каждая идея/трансплантированное предложение перед выдачей.
//
//   F1 Fact & Domain      — жёсткая связь со статическими L0/L1 знаниями (MoE-слепки)
//   F2 Causal & Structural — пермутационный сдвиг: вывод следует из условия (порядок)
//   F3 Intent & Memory     — динамический эпизодический контекст L2-Кортекса (32k)
//
// Total resonance = α·Res(F1) + β·Res(F2) + γ·Res(F3). Если хотя бы один
// фундамент не резонирует (порог ANCHOR_RESONANCE_MIN), фаза сбрасывается
// в детерминированное молчание — галлюцинация не проходит каскад.
// ============================================================================

pub const ANCHOR_RESONANCE_MIN: f64 = 0.30;
pub const ANCHOR_WEIGHT_FACT: f64 = 0.40;
pub const ANCHOR_WEIGHT_LOGIC: f64 = 0.30;
pub const ANCHOR_WEIGHT_INTENT: f64 = 0.30;
/// Each anchor must clear this floor to contribute. Sits just above the min3
/// residual crosstalk floor so no single anchor can rubber-stamp a claim on
/// generic tokens alone. Sweep over the 12-case bench: any floor in
/// [0.02, 0.12] keeps accuracy at 1.00, so 0.05 is a safe interior point.
pub const ANCHOR_FLOOR: f64 = 0.05;
/// Minimum weighted total for acceptance. Empirically grounded claims land at
/// 0.25–0.45, off-domain hallucinations at 0.10–0.22; 0.23 is the widest
/// separating cut (bench accuracy 1.00).
pub const ANCHOR_TOTAL_MIN: f64 = 0.23;

#[derive(Clone)]
pub struct ReasoningFoundations {
    pub domain_fact: Hypervector, // F1: 32k фаза доменного ядра (L0/L1 static)
    pub causal_logic: Hypervector, // F2: 32k пермутационный каркас порядка (Shift)
    pub intent_cortex: Hypervector, // F3: 32k динамический эпизодический контекст
}

impl ReasoningFoundations {
    /// Build F1/F2/F3 from the static crystal (domain core), the live
    /// episodic cortex, the current intent text and the candidate being vetted.
    ///
    /// F1 is anchored on the candidate itself: the top-K static entries most
    /// relevant to the *candidate* are re-encoded directly in 32k and merged
    /// sparsely. Anchoring on the intent instead would let a hallucination ride
    /// the generic documentation stop-words shared by every entry. Anchoring on
    /// the candidate, a grounded claim overlaps its own supporting facts while
    /// a hallucination overlaps none of them.
    pub fn build(
        domain: &PhaseCrystal,
        cortex: &PhaseCrystal,
        intent: &str,
        candidate: &str,
    ) -> Self {
        let dim = DIM_L2;
        // min3 encoder: drop 1–2 byte grams whose corpus-wide stopword
        // crosstalk (~20% of any text's bits) smears fact vs hallucination.
        let enc = |s: &str| encode_bytes_nopos_min3(s.as_bytes(), dim);

        // Rank static entries by relevance to the CANDIDATE (native-dim overlap).
        let q = encode_bytes_nopos(candidate.as_bytes(), DIM_L1);
        let mut ranked: Vec<(f64, &CrystalEntry)> = domain
            .entries
            .iter()
            .filter(|e| e.kind == KIND_L1 || e.kind == KIND_L0)
            .map(|e| (overlap_score(&q, &e.hv), e))
            .collect();
        ranked.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let k = ranked.len().min(16);
        // IMPORTANT: anchors are built by *re-encoding the fact text directly
        // in 32k* (min3 encoder), NOT by projecting the stored phase.
        // project_phase maps an 8k VSA into 32k for the hippocampal bridge, but
        // that layout never aligns with a direct 32k encoding of the same
        // words — so a candidate encoded natively in 32k would only ever see
        // random crosstalk. Encoding both sides in the same space makes the
        // coverage metric meaningful.
        let proj: Vec<Hypervector> = ranked[..k]
            .iter()
            .map(|(_, e)| {
                let content = if e.text.trim().is_empty() {
                    e.key_text.clone()
                } else {
                    e.text.clone()
                };
                enc(&content)
            })
            .collect();
        // Sparse domain core: bits set in >=2 of the relevant phases (keeps
        // shared topic vocabulary, drops per-chunk noise).
        let mut domain_fact = Hypervector::new(dim);
        let wc = dim.div_ceil(64);
        for w in 0..wc {
            let mut bits = 0u64;
            for bit in 0..64 {
                let ones = proj
                    .iter()
                    .filter(|h| ((h.words[w] >> bit) & 1) == 1)
                    .count();
                if ones >= 2 {
                    bits |= 1u64 << bit;
                }
            }
            domain_fact.words[w] = bits;
        }

        // F2: causal scaffold — permuted (Shift) domain phase bound with the
        // logical-connective phase. Binding pins the *order*: a claim that only
        // shares surface words but breaks the if→then chain shifts away.
        let causal_words = enc("if then because therefore implies");
        let domain_shifted = permute_phase(&domain_fact, 0x9E37_79B9_7F4A_7C15);
        let causal_logic = bind_phase(&domain_shifted, &causal_words);

        // F3: intent anchor = current goal phase blended into live cortex state.
        let intent_vec = enc(intent);
        let mut cortex_32k: Vec<Hypervector> = Vec::new();
        for e in cortex.entries.iter().filter(|e| e.kind == KIND_L2).take(64) {
            cortex_32k.push(e.hv.clone());
        }
        let mut cortex_state = Hypervector::new(dim);
        if cortex_32k.len() >= 2 {
            let wc = dim.div_ceil(64);
            for w in 0..wc {
                let mut bits = 0u64;
                for bit in 0..64 {
                    let ones = cortex_32k
                        .iter()
                        .filter(|h| ((h.words[w] >> bit) & 1) == 1)
                        .count();
                    if ones >= 2 {
                        bits |= 1u64 << bit;
                    }
                }
                cortex_state.words[w] = bits;
            }
        }
        let intent_cortex = PhaseCrystal::weighted_majority(&intent_vec, &cortex_state, 0.5);

        ReasoningFoundations {
            domain_fact,
            causal_logic,
            intent_cortex,
        }
    }

    /// Coverage resonance of a candidate against an anchor: the fraction of the
    /// candidate's own phase bits that lie inside the anchor. Dense random
    /// crosstalk contributes ~density(anchor) regardless of topic, so a sparse
    /// anchored fact scores high (most bits covered) and a hallucination low.
    fn coverage(cand: &Hypervector, anchor: &Hypervector) -> f64 {
        let mut overlap = 0u32;
        let n = cand.words.len().min(anchor.words.len());
        for w in 0..n {
            overlap += (cand.words[w] & anchor.words[w]).count_ones();
        }
        let cnz = cand.words.iter().map(|w| w.count_ones()).sum::<u32>();
        if cnz == 0 {
            0.0
        } else {
            overlap as f64 / cnz as f64
        }
    }

    /// Per-anchor resonances of a candidate against the triad.
    pub fn resonances(&self, candidate: &Hypervector) -> [f64; 3] {
        [
            Self::coverage(candidate, &self.domain_fact),
            Self::coverage(candidate, &self.causal_logic),
            Self::coverage(candidate, &self.intent_cortex),
        ]
    }

    /// Weighted total: α·Res(F1) + β·Res(F2) + γ·Res(F3).
    pub fn total_resonance(&self, candidate: &Hypervector) -> f64 {
        let r = self.resonances(candidate);
        ANCHOR_WEIGHT_FACT * r[0] + ANCHOR_WEIGHT_LOGIC * r[1] + ANCHOR_WEIGHT_INTENT * r[2]
    }

    /// A transplanted proposal passes if it clears the floor on every anchor
    /// (no anchor fires purely on stopword crosstalk) and the weighted total
    /// meets ANCHOR_TOTAL_MIN. A hallucination either collapses one anchor
    /// below the floor (duck → F3=0.02) or, if it rides shared topic words,
    /// still fails the total because F2/F3 stay near their floors.
    pub fn evaluate_transplant(&self, candidate: &Hypervector) -> bool {
        let r = self.resonances(candidate);
        let floor_ok = r[0] >= ANCHOR_FLOOR && r[1] >= ANCHOR_FLOOR && r[2] >= ANCHOR_FLOOR;
        floor_ok && self.total_resonance(candidate) >= ANCHOR_TOTAL_MIN
    }
}

/// VSA binding: phase ⊗ phase = elementwise XOR (dim preserved).
pub fn bind_phase(a: &Hypervector, b: &Hypervector) -> Hypervector {
    let wc = a.words.len().min(b.words.len());
    let mut words = vec![0u64; wc];
    for w in 0..wc {
        words[w] = a.words[w] ^ b.words[w];
    }
    Hypervector { dim: a.dim, words }
}

/// Deterministic permutation (Shift): scatters each set bit to a new position
/// via FNV-1a + linear probing, preserving density exactly. Seed salts the
/// permutation so F2's order-scaffold decorrelates from the raw domain phase.
pub fn permute_phase(hv: &Hypervector, seed: u64) -> Hypervector {
    let wc = hv.dim.div_ceil(64);
    let mut words = vec![0u64; wc];
    let mut buf = [0u8; 8];
    for (wi, w) in hv.words.iter().enumerate() {
        let mut bits = *w;
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let src = wi * 64 + b;
            buf.copy_from_slice(&(src as u64).to_le_bytes());
            let h = fnv1a(&buf) ^ seed;
            let mut pos = (h % hv.dim as u64) as usize;
            while (words[pos / 64] >> (pos % 64)) & 1 == 1 {
                pos = (pos + 1) % hv.dim;
            }
            words[pos / 64] |= 1u64 << (pos % 64);
        }
    }
    Hypervector { dim: hv.dim, words }
}

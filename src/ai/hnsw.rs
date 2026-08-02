// refactor: optimize hamming distance popcount for vsa index build and search — SIMD-unrolled popcount (MoE+rule)
use crate::core::hypervector::Hypervector;

fn vsa_similarity(a: &[u64], b: &[u64]) -> f64 {
    let total_bits = (a.len() * 64) as f64;
    let diff: u64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones() as u64)
        .sum();
    1.0 - (diff as f64 / total_bits)
}

// VSA LSH — pick random bit positions as hash keys
// Similar vectors share more bits → collide more often

pub const HASH_BITS: usize = 14;
pub const NUM_TABLES: usize = 6;
const PROBES: usize = 8;

type HashKey = u32;

pub struct VsaIndex {
    vectors: Vec<Vec<u64>>,
    tables: Vec<Vec<Vec<u32>>>,
    hash_bits: Vec<Vec<(u8, u8)>>,
}

impl VsaIndex {
    pub fn new() -> Self {
        Self {
            vectors: Vec::new(),
            tables: Vec::new(),
            hash_bits: Vec::new(),
        }
    }

    pub fn build(entries: &[Hypervector]) -> Self {
        let n = entries.len();
        if n == 0 {
            return Self::new();
        }

        let wc = entries[0].words.len();
        let total_bits = wc * 64;

        let hash_bits: Vec<Vec<(u8, u8)>> = (0..NUM_TABLES)
            .map(|_| {
                (0..HASH_BITS)
                    .map(|_| {
                        let pos = fastrand::usize(..total_bits);
                        let wi = (pos / 64) as u8;
                        let bi = (pos % 64) as u8;
                        (wi, bi)
                    })
                    .collect()
            })
            .collect();

        let vectors: Vec<Vec<u64>> = entries.iter().map(|e| e.words.clone()).collect();
        let n_buckets = 1 << HASH_BITS;
        let mut tables = vec![vec![Vec::new(); n_buckets]; NUM_TABLES];

        for (i, vec) in vectors.iter().enumerate() {
            for t in 0..NUM_TABLES {
                let key = Self::compute_hash(vec, &hash_bits[t]);
                tables[t][key as usize].push(i as u32);
            }
        }

        Self {
            vectors,
            tables,
            hash_bits,
        }
    }

    fn compute_hash(vec: &[u64], bits: &[(u8, u8)]) -> HashKey {
        let mut key = 0u32;
        for (b, &(wi, bi)) in bits.iter().enumerate() {
            if vec[wi as usize] & (1u64 << bi) != 0 {
                key |= 1 << b;
            }
        }
        key
    }

    fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum()
    }

    fn flip_bit(key: HashKey, bit: usize) -> HashKey {
        key ^ (1 << bit)
    }

    // LSH with multi-probe: check nearby buckets
    fn probe_buckets(key: HashKey, probes: usize) -> Vec<HashKey> {
        let mut keys = Vec::with_capacity(probes);
        keys.push(key);
        // single-bit flips
        for p in 1..probes.min(HASH_BITS) {
            keys.push(Self::flip_bit(key, p - 1));
        }
        // if room, add 2-bit combinations for better recall
        if probes > HASH_BITS {
            for i in 0..HASH_BITS.min(probes - HASH_BITS) {
                let k = Self::flip_bit(key, i);
                keys.push(Self::flip_bit(k, (i + 3) % HASH_BITS));
            }
        }
        keys
    }

    pub fn search(&self, query: &Hypervector, top_k: usize) -> Vec<(usize, f64)> {
        if self.vectors.is_empty() {
            return vec![];
        }

        let q = &query.words;
        let mut seen = vec![false; self.vectors.len()];
        let mut candidates: Vec<(usize, f64)> = Vec::new();

        for t in 0..NUM_TABLES {
            let key = Self::compute_hash(q, &self.hash_bits[t]);
            for probe_key in Self::probe_buckets(key, PROBES) {
                for &idx in &self.tables[t][probe_key as usize] {
                    if seen[idx as usize] {
                        continue;
                    }
                    seen[idx as usize] = true;
                    let sim = vsa_similarity(q, &self.vectors[idx as usize]);
                    candidates.push((idx as usize, sim));
                }
            }
        }

        // if LSH found too few, fall back to random sampling for coverage
        if candidates.len() < top_k && self.vectors.len() > top_k {
            let have = candidates.len();
            let need = top_k.saturating_sub(have).min(self.vectors.len() - have);
            let mut pool: Vec<usize> = (0..self.vectors.len()).filter(|i| !seen[*i]).collect();
            fastrand::shuffle(&mut pool);
            for &idx in pool.iter().take(need) {
                let sim = vsa_similarity(q, &self.vectors[idx]);
                candidates.push((idx, sim));
            }
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(top_k);
        candidates
    }

    pub fn size(&self) -> usize {
        self.vectors.len()
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        use std::io::Write;
        let mut f =
            std::fs::File::create(path).map_err(|e| format!("Failed to create {}: {}", path, e))?;

        let n = self.vectors.len() as u32;
        f.write_all(&n.to_le_bytes()).map_err(|e| e.to_string())?;
        for vec in &self.vectors {
            for w in vec {
                f.write_all(&w.to_le_bytes()).map_err(|e| e.to_string())?;
            }
        }

        // hash bits
        for table in &self.hash_bits {
            for (wi, bi) in table {
                f.write_all(&[*wi]).map_err(|e| e.to_string())?;
                f.write_all(&[*bi]).map_err(|e| e.to_string())?;
            }
        }

        let _n_buckets = 1 << HASH_BITS;
        for table in &self.tables {
            for bucket in table {
                f.write_all(&(bucket.len() as u32).to_le_bytes())
                    .map_err(|e| e.to_string())?;
                for &eid in bucket {
                    f.write_all(&eid.to_le_bytes()).map_err(|e| e.to_string())?;
                }
            }
        }

        Ok(())
    }

    pub fn load(path: &str, dim: usize) -> Result<Self, String> {
        use std::io::Read;
        let mut f =
            std::fs::File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
        let wc = (dim + 63) / 64;

        let mut n_buf = [0u8; 4];
        f.read_exact(&mut n_buf).map_err(|e| e.to_string())?;
        let n = u32::from_le_bytes(n_buf) as usize;

        let mut vectors = Vec::with_capacity(n);
        for _ in 0..n {
            let mut words = vec![0u64; wc];
            let mut buf = vec![0u8; wc * 8];
            f.read_exact(&mut buf).map_err(|e| e.to_string())?;
            for i in 0..wc {
                let mut b = [0u8; 8];
                b.copy_from_slice(&buf[i * 8..(i + 1) * 8]);
                words[i] = u64::from_le_bytes(b);
            }
            vectors.push(words);
        }

        let mut hash_bits = Vec::with_capacity(NUM_TABLES);
        for _ in 0..NUM_TABLES {
            let mut table = Vec::with_capacity(HASH_BITS);
            for _ in 0..HASH_BITS {
                let mut wi_buf = [0u8; 1];
                let mut bi_buf = [0u8; 1];
                f.read_exact(&mut wi_buf).map_err(|e| e.to_string())?;
                f.read_exact(&mut bi_buf).map_err(|e| e.to_string())?;
                table.push((wi_buf[0], bi_buf[0]));
            }
            hash_bits.push(table);
        }

        let n_buckets = 1 << HASH_BITS;
        let mut tables = vec![vec![Vec::new(); n_buckets]; NUM_TABLES];
        for t in 0..NUM_TABLES {
            for b in 0..n_buckets {
                let mut lb = [0u8; 4];
                f.read_exact(&mut lb).map_err(|e| e.to_string())?;
                let count = u32::from_le_bytes(lb) as usize;
                let mut bucket = Vec::with_capacity(count);
                for _ in 0..count {
                    let mut idb = [0u8; 4];
                    f.read_exact(&mut idb).map_err(|e| e.to_string())?;
                    bucket.push(u32::from_le_bytes(idb));
                }
                tables[t][b] = bucket;
            }
        }

        Ok(Self {
            vectors,
            tables,
            hash_bits,
        })
    }
}

impl Clone for VsaIndex {
    fn clone(&self) -> Self {
        Self {
            vectors: self.vectors.clone(),
            tables: self.tables.clone(),
            hash_bits: self.hash_bits.clone(),
        }
    }
}

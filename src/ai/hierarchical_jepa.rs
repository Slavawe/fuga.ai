use crate::core::hypervector::Hypervector;
use crate::safety::circuit_breaker::{FugaCircuitBreaker, SystemState};
use crate::vsa::topology::{ls_bind, phase_smooth};
use crate::weaver::token_id;
use rand::{Rng, RngCore, SeedableRng};
use std::hash::{Hash, Hasher};

pub const NUM_LEVELS: usize = 3;
pub const DEFAULT_L0_CTX: usize = 4;
pub const DEFAULT_L1_CTX: usize = 3;
pub const DEFAULT_L2_CTX: usize = 2;
pub const DEFAULT_L0_STRIDE: usize = 1;
pub const DEFAULT_L1_STRIDE: usize = 3;
pub const DEFAULT_L2_STRIDE: usize = 5;
pub const DEFAULT_DIM: usize = 8192;
pub const TRAIN_EPOCHS: usize = 100;
const LR: f64 = 0.05;
pub const PERM_EXPANSION: usize = 4; // multiple VSA projections per position
fn active_bits(words: &[u64], dim: usize) -> Vec<usize> {
    let mut bits = Vec::with_capacity(dim / 2);
    for (wi, &w) in words.iter().enumerate() {
        let base = wi * 64;
        let limit = base + 64.min(dim - base);
        for bi in 0..(limit - base) {
            if (w >> bi) & 1 == 1 {
                bits.push(base + bi);
            }
        }
    }
    bits
}

fn hv_to_threshold_bits(cont: &[f64]) -> Vec<i8> {
    cont.iter().map(|&v| if v >= 0.0 { 1 } else { 0 }).collect()
}

fn hv_to_topk_bits(cont: &[f64], k: usize) -> Vec<i8> {
    if cont.is_empty() { return Vec::new(); }
    let mut idx: Vec<usize> = (0..cont.len()).collect();
    idx.sort_by(|&a, &b| cont[b].partial_cmp(&cont[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut bits = vec![0i8; cont.len()];
    for &i in idx.iter().take(k.min(cont.len())) {
        bits[i] = 1;
    }
    bits
}

fn dot_bipolar(a: &Hypervector, b: &Hypervector) -> f64 {
    a.words.iter().zip(b.words.iter()).flat_map(|(&wa, &wb)| {
        (0..64).map(move |bi| {
            let ai = if (wa >> bi) & 1 == 1 { 1.0 } else { -1.0 };
            let bi = if (wb >> bi) & 1 == 1 { 1.0 } else { -1.0 };
            ai * bi
        })
    }).sum()
}

#[derive(Clone, Debug)]
pub struct JepaLevel {
    pub name: String,
    pub dim: usize,
    pub context_len: usize,
    pub stride: usize,
    pub perm_offsets: Vec<usize>,
    pub weights: Vec<f64>,
    pub bundling_weights: Vec<f64>,
    pub velocity: Vec<f64>,
    pub lr: f64,
    pub mode: u8,  // 0 = linear per-dim, 1 = VSA bundling (scalar per projection), 2 = phase (resonance)
    pub top_k: usize,  // sparse phase router: keep only top-k projections by resonance (0 = all)
    pub delta_ones: usize,  // EMA of active bits in observed delta hypervectors (~1% of dim)
    pub num_expert_group: usize,  // grouped top-k router: how many projection groups (0/1 = ungrouped)
    pub topk_group: usize,        // Kimi/DeepSeek-style: keep only these top groups before per-group top_k
}

impl JepaLevel {
    pub fn new(name: &str, dim: usize, context_len: usize, stride: usize) -> Self {
        let mut rng = rand::thread_rng();
        let perm_offsets: Vec<usize> = (0..context_len * PERM_EXPANSION)
            .map(|_| rng.gen_range(1..dim))
            .collect();
        let wlen = context_len * PERM_EXPANSION * dim;
        let mut weights = Vec::with_capacity(wlen);
        for _ in 0..wlen {
            weights.push(rng.gen_range(-0.005..0.005));
        }
        let bcount = context_len * PERM_EXPANSION;
        let bundling_weights = vec![1.0 / bcount as f64; bcount];
        let velocity = vec![0.0; wlen];
        let lr = LR;
        JepaLevel { name: name.to_string(), dim, context_len, stride, perm_offsets, weights, bundling_weights, velocity, lr, mode: 2, top_k: 0, delta_ones: 0, num_expert_group: 1, topk_group: 0 }
    }

    // Kimi/DeepSeek grouped top-k: scores[0..bcount] are the resonance magnitudes.
    // Projections are split into num_expert_group groups; each group scores as the sum
    // of its top-2 members; only topk_group groups survive, then top_k within them.
    pub fn grouped_topk_mask(&self, bcount: usize, scores: &[f64]) -> Vec<bool> {
        let mut keep = vec![false; bcount];
        if self.top_k == 0 || bcount == 0 { return vec![true; bcount]; }
        let ng = if self.num_expert_group > 1 && self.topk_group > 0
            && self.topk_group < self.num_expert_group && self.num_expert_group <= bcount {
            self.num_expert_group
        } else { 1 };
        if ng == 1 {
            let mut idx: Vec<usize> = (0..bcount).collect();
            idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
            for &i in idx.iter().take(self.top_k.min(bcount)) { keep[i] = true; }
            return keep;
        }
        let gsize = bcount / ng;
        let mut group_score = vec![0.0f64; ng];
        for g in 0..ng {
            let base = g * gsize;
            let end = if g == ng - 1 { bcount } else { base + gsize };
            let mut top2 = [0.0f64; 2];
            for i in base..end {
                let v = scores[i];
                if v > top2[0] { top2[1] = top2[0]; top2[0] = v; }
                else if v > top2[1] { top2[1] = v; }
            }
            group_score[g] = top2[0] + top2[1];
        }
        let mut gidx: Vec<usize> = (0..ng).collect();
        gidx.sort_by(|&a, &b| group_score[b].partial_cmp(&group_score[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut gkeep = vec![false; ng];
        for &g in gidx.iter().take(self.topk_group.min(ng)) { gkeep[g] = true; }
        // top_k within surviving groups
        let mut cand: Vec<usize> = Vec::new();
        for g in 0..ng {
            if !gkeep[g] { continue; }
            let base = g * gsize;
            let end = if g == ng - 1 { bcount } else { base + gsize };
            for i in base..end { cand.push(i); }
        }
        cand.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
        for &i in cand.iter().take(self.top_k.min(cand.len())) { keep[i] = true; }
        keep
    }

    pub fn predict_continuous(&self, context: &[&Hypervector]) -> Vec<f64> {
        if self.mode >= 1 {
            return self.predict_continuous_bundled(context);
        }
        let mut raw = vec![0.0f64; self.dim];
        let n = context.len().min(self.context_len);
        for (i, hv) in context.iter().take(n).enumerate() {
            for k in 0..PERM_EXPANSION {
                let base = (i * PERM_EXPANSION + k) * self.dim;
                let idx = i * PERM_EXPANSION + k;
                let offset = self.perm_offsets.get(idx).copied().unwrap_or(0);
                let permuted = hv.permute(offset);
                for bit in active_bits(&permuted.words, self.dim) {
                    raw[bit] += self.weights[base + bit];
                }
            }
        }
        raw
    }

    pub fn predict_continuous_bundled(&self, context: &[&Hypervector]) -> Vec<f64> {
        let mut raw = vec![0.0f64; self.dim];
        let n = context.len().min(self.context_len);
        let bcount = n * PERM_EXPANSION;
        let _total = self.context_len * PERM_EXPANSION;
        if self.top_k > 0 && self.top_k < bcount {
            // Sparse Phase Router: score every projection in the window by |resonance weight|,
            // keep only the top_k most salient (optionally grouped Kimi/DeepSeek-style), drop the rest.
            let scores: Vec<f64> = self.bundling_weights.iter()
                .take(bcount).map(|w| w.abs()).collect();
            let keep = self.grouped_topk_mask(bcount, &scores);
            for (i, hv) in context.iter().take(n).enumerate() {
                for k in 0..PERM_EXPANSION {
                    let idx = i * PERM_EXPANSION + k;
                    if !keep[idx] { continue; }
                    let offset = self.perm_offsets.get(idx).copied().unwrap_or(0);
                    let permuted = hv.permute(offset);
                    let w = self.bundling_weights[idx];
                    for bit in active_bits(&permuted.words, self.dim) {
                        raw[bit] += w;
                    }
                }
            }
            return raw;
        }
        for (i, hv) in context.iter().take(n).enumerate() {
            for k in 0..PERM_EXPANSION {
                let idx = i * PERM_EXPANSION + k;
                let offset = self.perm_offsets.get(idx).copied().unwrap_or(0);
                let permuted = hv.permute(offset);
                let w = self.bundling_weights[idx];
                for bit in active_bits(&permuted.words, self.dim) {
                    raw[bit] += w;
                }
            }
        }
        raw
    }

    pub fn predict(&self, context: &[&Hypervector]) -> Hypervector {
        self.predict_with_temp(context, 1.0)
    }

    pub fn predict_with_temp(&self, context: &[&Hypervector], temperature: f64) -> Hypervector {
        if context.is_empty() {
            return Hypervector::random(self.dim);
        }
        let baseline = context[context.len() - 1];
        let mut cont = self.predict_continuous(context);
        if temperature > 0.0 && (temperature - 1.0).abs() > 1e-6 {
            let inv_t = 1.0 / temperature;
            for v in &mut cont {
                *v *= inv_t;
            }
        }
        let pred_bits = if self.mode == 2 && self.delta_ones > 0 && self.delta_ones < self.dim {
            // Sparse Phase Router threshold: emit exactly the observed delta density (top-k raw bits)
            hv_to_topk_bits(&cont, self.delta_ones)
        } else {
            hv_to_threshold_bits(&cont)
        };
        let pred_delta = Hypervector::from_i8_bits(self.dim, &pred_bits);
        baseline.bind(&pred_delta)
    }

    pub fn similarity_to_expected(&self, context: &[&Hypervector], actual: &Hypervector) -> f64 {
        let pred = self.predict(context);
        dot_bipolar(&pred, actual) / self.dim as f64
    }

    pub fn train_step(&mut self, context: &[&Hypervector], actual: &Hypervector, margin: f64) -> f64 {
        if self.mode >= 1 {
            return self.train_step_bundled(context, actual, margin);
        }
        let dim = self.dim;
        let n_ctx = context.len().min(self.context_len);

        let baseline = context[context.len() - 1];
        let delta = baseline.bind(actual);

        let mut raw = vec![0.0f64; dim];
        for (i, hv) in context.iter().take(n_ctx).enumerate() {
            for k in 0..PERM_EXPANSION {
                let base = (i * PERM_EXPANSION + k) * dim;
                let idx = i * PERM_EXPANSION + k;
                let offset = self.perm_offsets.get(idx).copied().unwrap_or(0);
                let permuted = hv.permute(offset);
                for bit in active_bits(&permuted.words, dim) {
                    raw[bit] += self.weights[base + bit];
                }
            }
        }

        let lr = self.lr;
        let n = (n_ctx * PERM_EXPANSION) as f64;
        let decay = self.lr * 0.01 / n;
        for (i, hv) in context.iter().take(n_ctx).enumerate() {
            for k in 0..PERM_EXPANSION {
                let base = (i * PERM_EXPANSION + k) * dim;
                let p_idx = i * PERM_EXPANSION + k;
                let offset = self.perm_offsets.get(p_idx).copied().unwrap_or(0);
                let permuted = hv.permute(offset);
                for bit in active_bits(&permuted.words, dim) {
                    let w_idx = base + bit;
                    let d_bit = (delta.words[bit / 64] >> (bit % 64)) & 1;
                    let target = if d_bit == 1 { 1.0 } else { -1.0 };
                    let error = target - raw[bit] / n;
                    self.weights[w_idx] = self.weights[w_idx] * (1.0 - decay) + lr * error;
                }
            }
        }

        let raw_norm = raw.iter().map(|r| r * r).sum::<f64>().sqrt();
        if raw_norm < 1e-9 { return 1.0; }
        let mut dot = 0.0f64;
        for (wi, &aw) in actual.words.iter().enumerate() {
            for bi in 0..64 {
                let bit = wi * 64 + bi;
                if bit >= dim { break; }
                let target = if (aw >> bi) & 1 == 1 { 1.0 } else { -1.0 };
                dot += raw[bit] * target;
            }
        }
        1.0 - (dot / (raw_norm * (dim as f64).sqrt())).clamp(-1.0, 1.0)
    }

    pub fn train_step_bundled(&mut self, context: &[&Hypervector], actual: &Hypervector, _margin: f64) -> f64 {
        let dim = self.dim;
        let n_ctx = context.len().min(self.context_len);

        // VSA target: model predicts the transition baseline -> actual, so weights are fit
        // to the delta hypervector; predict_with_temp then recovers actual via baseline.bind(delta).
        let baseline = context[context.len() - 1];
        let delta = baseline.bind(actual);

        if self.mode == 2 {
            // Phase Overlap: weights = resonance scores against the delta, set in one shot
            let ones = delta.to_i8_bits().iter().filter(|&&b| b == 1).count();
            if self.delta_ones == 0 {
                self.delta_ones = ones;
            } else {
                self.delta_ones = (self.delta_ones * 99 + ones) / 100;
            }
            for (i, hv) in context.iter().take(n_ctx).enumerate() {
                for k in 0..PERM_EXPANSION {
                    let idx = i * PERM_EXPANSION + k;
                    let offset = self.perm_offsets.get(idx).copied().unwrap_or(0);
                    let permuted = hv.permute(offset);
                    let sim = dot_bipolar(&permuted, &delta) / dim as f64;
                    self.bundling_weights[idx] = sim;
                }
            }
            // Sparse Phase Router: keep only top_k projections by |resonance| (optionally
            // grouped Kimi/DeepSeek-style), zero the rest
            if self.top_k > 0 && self.top_k < n_ctx * PERM_EXPANSION {
                let bcount = n_ctx * PERM_EXPANSION;
                let scores: Vec<f64> = self.bundling_weights.iter()
                    .take(bcount).map(|w| w.abs()).collect();
                let keep = self.grouped_topk_mask(bcount, &scores);
                for (i, w) in self.bundling_weights.iter_mut().enumerate() {
                    if i < bcount && !keep[i] { *w = 0.0; }
                }
            }
        }

        let mut raw = vec![0.0f64; dim];
        for (i, hv) in context.iter().take(n_ctx).enumerate() {
            for k in 0..PERM_EXPANSION {
                let idx = i * PERM_EXPANSION + k;
                let offset = self.perm_offsets.get(idx).copied().unwrap_or(0);
                let permuted = hv.permute(offset);
                let w = self.bundling_weights[idx];
                for bit in active_bits(&permuted.words, dim) {
                    raw[bit] += w;
                }
            }
        }

        if self.mode != 2 {
            // SGD update for mode 0/1 — target is the delta bits (transition), not actual
            let bcount = n_ctx * PERM_EXPANSION;
            let nf = bcount as f64;
            let lr = self.lr * 10.0;
            for (i, hv) in context.iter().take(n_ctx).enumerate() {
                for k in 0..PERM_EXPANSION {
                    let idx = i * PERM_EXPANSION + k;
                    let offset = self.perm_offsets.get(idx).copied().unwrap_or(0);
                    let permuted = hv.permute(offset);
                    let mut grad = 0.0f64;
                    for bit in active_bits(&permuted.words, dim) {
                        let d_bit = (delta.words[bit / 64] >> (bit % 64)) & 1;
                        let target = if d_bit == 1 { 1.0 } else { -1.0 };
                        grad += target - raw[bit] / nf;
                    }
                    self.bundling_weights[idx] += lr * grad / dim as f64;
                }
            }
        }

        let raw_norm = raw.iter().map(|r| r * r).sum::<f64>().sqrt();
        if raw_norm < 1e-9 { return 1.0; }
        let mut dot = 0.0f64;
        for (wi, &aw) in actual.words.iter().enumerate() {
            for bi in 0..64 {
                let bit = wi * 64 + bi;
                if bit >= dim { break; }
                let target = if (aw >> bi) & 1 == 1 { 1.0 } else { -1.0 };
                dot += raw[bit] * target;
            }
        }
        1.0 - (dot / (raw_norm * (dim as f64).sqrt())).clamp(-1.0, 1.0)
    }

    pub fn train_step_ff(&mut self, context: &[&Hypervector], actual: &Hypervector, negative: &Hypervector, _margin: f64) -> f64 {
        let dim = self.dim;
        let n_ctx = context.len().min(self.context_len);

        let mut raw = vec![0.0f64; dim];
        for (i, hv) in context.iter().take(n_ctx).enumerate() {
            for k in 0..PERM_EXPANSION {
                let base = (i * PERM_EXPANSION + k) * dim;
                let p_idx = i * PERM_EXPANSION + k;
                let offset = self.perm_offsets.get(p_idx).copied().unwrap_or(0);
                let permuted = hv.permute(offset);
                for bit in active_bits(&permuted.words, dim) {
                    raw[bit] += self.weights[base + bit];
                }
            }
        }

        let lr = self.lr;
        let n = (n_ctx * PERM_EXPANSION) as f64;
        let decay = self.lr * 0.005 / n;
        let mut pos_goodness = 0.0f64;
        let mut neg_goodness = 0.0f64;
        let mut cnt = 0usize;

        for (i, hv) in context.iter().take(n_ctx).enumerate() {
            for k in 0..PERM_EXPANSION {
                let base = (i * PERM_EXPANSION + k) * dim;
                let p_idx = i * PERM_EXPANSION + k;
                let offset = self.perm_offsets.get(p_idx).copied().unwrap_or(0);
                let permuted = hv.permute(offset);
                for bit in active_bits(&permuted.words, dim) {
                    let w_idx = base + bit;
                    let a_bit = (actual.words[bit / 64] >> (bit % 64)) & 1;
                    let t_pos: f64 = if a_bit == 1 { 1.0 } else { -1.0 };

                    let n_bit = (negative.words[bit / 64] >> (bit % 64)) & 1;
                    let t_neg: f64 = if n_bit == 1 { 1.0 } else { -1.0 };

                    pos_goodness += raw[bit] * t_pos;
                    neg_goodness += raw[bit] * t_neg;
                    cnt += 1;

                    self.weights[w_idx] = self.weights[w_idx] * (1.0 - decay) + lr * (t_pos - t_neg);
                }
            }
        }

        if cnt == 0 { return 1.0; }
        let contrast = (pos_goodness - neg_goodness) / cnt as f64;
        (1.0 - contrast).clamp(0.0, 2.0)
    }

    fn save(&self, buf: &mut Vec<u8>) {
        let nb = self.name.as_bytes();
        buf.extend_from_slice(&(nb.len() as u32).to_le_bytes());
        buf.extend_from_slice(nb);
        buf.extend_from_slice(&(self.context_len as u32).to_le_bytes());
        buf.extend_from_slice(&(self.stride as u32).to_le_bytes());
        buf.push(6u8);  // level version 6 = perm_count + mode + bundling_weights + top_k + delta_ones + grouped_topk
        let perm_count = self.perm_offsets.len() as u32;
        buf.extend_from_slice(&perm_count.to_le_bytes());
        for &off in &self.perm_offsets {
            buf.extend_from_slice(&(off as u32).to_le_bytes());
        }
        for &w in &self.weights {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf.push(self.mode);
        for &w in &self.bundling_weights {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        buf.extend_from_slice(&(self.top_k as u32).to_le_bytes());
        buf.extend_from_slice(&(self.delta_ones as u32).to_le_bytes());
        buf.extend_from_slice(&(self.num_expert_group as u32).to_le_bytes());
        buf.extend_from_slice(&(self.topk_group as u32).to_le_bytes());
    }

    fn load(data: &[u8], offset: &mut usize) -> Result<Self, String> {
        if *offset + 4 > data.len() { return Err("short header".into()); }
        let nl = u32::from_le_bytes(data[*offset..*offset+4].try_into().unwrap()) as usize;
        *offset += 4;
        if *offset + nl > data.len() { return Err("short name".into()); }
        let name = String::from_utf8(data[*offset..*offset+nl].to_vec()).map_err(|e| format!("utf8: {}", e))?;
        *offset += nl;
        if *offset + 8 > data.len() { return Err("short after name".into()); }
        let ctx = u32::from_le_bytes(data[*offset..*offset+4].try_into().unwrap()) as usize;
        *offset += 4;
        let stride = u32::from_le_bytes(data[*offset..*offset+4].try_into().unwrap()) as usize;
        *offset += 4;
        // Check level version byte
        let lv = data.get(*offset).copied().unwrap_or(0);
        let is_new = lv >= 2;
        let is_bundled = lv >= 3;
        let has_topk = lv >= 4;
        let has_delta_ones = lv >= 5;
        let has_grouped_topk = lv >= 6;
        if is_new { *offset += 1; }
        let perm_count = if is_new {
            let pc = u32::from_le_bytes(data[*offset..*offset+4].try_into().unwrap()) as usize;
            *offset += 4;
            pc
        } else {
            ctx
        };
        let mut offsets = Vec::with_capacity(perm_count);
        for _ in 0..perm_count {
            offsets.push(u32::from_le_bytes(data[*offset..*offset+4].try_into().unwrap()) as usize);
            *offset += 4;
        }
        let dim = 8192usize;
        let wlen = perm_count * dim;
        let mut weights = Vec::with_capacity(wlen);
        for _ in 0..wlen {
            weights.push(f64::from_le_bytes(data[*offset..*offset+8].try_into().unwrap()));
            *offset += 8;
        }
        let (mode, bundling_weights) = if is_bundled {
            let m = data.get(*offset).copied().unwrap_or(0);
            *offset += 1;
            let mut bw = Vec::with_capacity(perm_count);
            for _ in 0..perm_count {
                bw.push(f64::from_le_bytes(data[*offset..*offset+8].try_into().unwrap()));
                *offset += 8;
            }
            (m, bw)
        } else {
            (0u8, vec![1.0 / perm_count as f64; perm_count])
        };
        let velocity = vec![0.0; wlen];
        let top_k = if has_topk {
            if *offset + 4 > data.len() { return Err("short top_k".into()); }
            let t = u32::from_le_bytes(data[*offset..*offset+4].try_into().unwrap()) as usize;
            *offset += 4;
            t
        } else { 0 };
        let delta_ones = if has_delta_ones {
            if *offset + 4 > data.len() { return Err("short delta_ones".into()); }
            let d = u32::from_le_bytes(data[*offset..*offset+4].try_into().unwrap()) as usize;
            *offset += 4;
            d
        } else { 0 };
        let (num_expert_group, topk_group) = if has_grouped_topk {
            if *offset + 8 > data.len() { return Err("short grouped_topk".into()); }
            let g = u32::from_le_bytes(data[*offset..*offset+4].try_into().unwrap()) as usize;
            let tg = u32::from_le_bytes(data[*offset+4..*offset+8].try_into().unwrap()) as usize;
            *offset += 8;
            (g, tg)
        } else { (1, 0) };
        Ok(JepaLevel { name, dim, context_len: ctx, stride, perm_offsets: offsets, weights, bundling_weights, velocity, lr: LR, mode, top_k, delta_ones, num_expert_group, topk_group })
    }
}

pub struct HierarchicalJEPA {
    pub dim: usize,
    pub levels: Vec<JepaLevel>,
}

impl HierarchicalJEPA {
    pub fn new(dim: usize) -> Self {
        HierarchicalJEPA {
            dim,
            levels: vec![
                JepaLevel::new("L0", dim, DEFAULT_L0_CTX, DEFAULT_L0_STRIDE),
                JepaLevel::new("L1", dim, DEFAULT_L1_CTX, DEFAULT_L1_STRIDE),
                JepaLevel::new("L2", dim, DEFAULT_L2_CTX, DEFAULT_L2_STRIDE),
            ],
        }
    }

    pub fn predict_sequence(&self, context: &[&Hypervector], steps: usize) -> Vec<Hypervector> {
        self.predict_sequence_beam(context, steps, 1, 1.0)
            .first().map(|s| s.clone()).unwrap_or_default()
    }

    pub fn predict_sequence_beam(&self, context: &[&Hypervector], steps: usize, beam_width: usize, temperature: f64) -> Vec<Vec<Hypervector>> {
        if context.is_empty() || beam_width == 0 {
            return Vec::new();
        }
        let ctx0 = self.levels[0].context_len;
        let mut beams: Vec<(Vec<Hypervector>, f64)> = vec![(
            context.iter().map(|h| (*h).clone()).collect(),
            0.0,
        )];
        for _ in 0..steps {
            let mut candidates: Vec<(Vec<Hypervector>, f64)> = Vec::new();
            for (seq, score) in &beams {
                for _ in 0..beam_width {
                    let start = seq.len().saturating_sub(ctx0);
                    let win: Vec<&Hypervector> = seq[start..].iter().collect();
                    let use_temp = if beam_width > 1 { temperature } else { 1.0 };
                    let l0_pred = self.levels[0].predict_with_temp(&win, use_temp);
                    let l1_win: Vec<&Hypervector> = seq[seq.len().saturating_sub(self.levels[1].context_len)..].iter().collect();
                    let l1_pred = if l1_win.len() >= self.levels[1].context_len {
                        self.levels[1].predict_with_temp(&l1_win, use_temp)
                    } else {
                        l0_pred.clone()
                    };
                    let corrected = dampen_correction(&l1_pred, &l0_pred);
                    let mut new_seq = seq.clone();
                    new_seq.push(corrected.clone());
                    let entropy = corrected.entropy();
                    let novelty = if entropy > 0.0 && entropy < 1.0 {
                        -(entropy * (entropy + 0.01).ln() + (1.0 - entropy) * (1.0 - entropy + 0.01).ln())
                    } else {
                        0.0
                    };
                    let new_score = score - novelty * 0.1;
                    candidates.push((new_seq, new_score));
                }
            }
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(beam_width);
            beams = candidates;
        }
        beams.into_iter().map(|(seq, _)| seq[seq.len() - steps..].to_vec()).collect()
    }

    pub fn predict(&self, context: &[&Hypervector]) -> Vec<Hypervector> {
        if context.len() < self.levels[0].context_len {
            return vec![Hypervector::random(self.dim); 3];
        }
        let ctx0 = self.levels[0].context_len;
        let ctx1 = self.levels[1].context_len;
        let ctx2 = self.levels[2].context_len;

        let l0_pred = self.levels[0].predict(context);

        let mut l0_traj: Vec<Hypervector> = context.iter().map(|h| (*h).clone()).collect();
        l0_traj.push(l0_pred.clone());
        let l0_needed = ctx0 + ctx1;
        while l0_traj.len() < l0_needed {
            let w: Vec<&Hypervector> = l0_traj[l0_traj.len()-ctx0..].iter().collect();
            l0_traj.push(self.levels[0].predict(&w));
        }

        let l1_win: Vec<&Hypervector> = l0_traj[l0_traj.len()-ctx1..].iter().collect();
        let l1_pred = self.levels[1].predict(&l1_win);

        let mut err_traj: Vec<Hypervector> = Vec::new();
        for i in 0..l0_traj.len() - ctx1 {
            let w: Vec<&Hypervector> = l0_traj[i..i+ctx1].iter().collect();
            let p = self.levels[1].predict(&w);
            err_traj.push(phase_smooth(&ls_bind(&p, &l0_traj[i + ctx1], 32), 2));
        }
        err_traj.push(phase_smooth(&ls_bind(&l1_pred, &l0_pred, 32), 2));

        while err_traj.len() < ctx2 {
            let fake = Hypervector::random(self.dim);
            err_traj.push(fake);
        }

        let l2_win: Vec<&Hypervector> = err_traj[err_traj.len()-ctx2..].iter().collect();
        let l2_correction = self.levels[2].predict(&l2_win);
        let corrected_l1 = dampen_correction(&l1_pred, &l2_correction);

        vec![l0_pred, l1_pred, corrected_l1]
    }

    pub fn predict_refined(&self, context: &[&Hypervector], temps: &[f64]) -> (Vec<Hypervector>, f64) {
        if context.len() < self.levels[0].context_len {
            let fallback = vec![Hypervector::random(self.dim); 3];
            return (fallback, 0.0);
        }
        let ctx0 = self.levels[0].context_len;
        let ctx1 = self.levels[1].context_len;
        let ctx2 = self.levels[2].context_len;

        let l0_pred = self.levels[0].predict(context);

        let mut l0_traj: Vec<Hypervector> = context.iter().map(|h| (*h).clone()).collect();
        l0_traj.push(l0_pred.clone());
        let l0_needed = ctx0 + ctx1;
        while l0_traj.len() < l0_needed {
            let w: Vec<&Hypervector> = l0_traj[l0_traj.len()-ctx0..].iter().collect();
            l0_traj.push(self.levels[0].predict(&w));
        }

        let l1_win: Vec<&Hypervector> = l0_traj[l0_traj.len()-ctx1..].iter().collect();
        let l1_pred = self.levels[1].predict(&l1_win);

        let mut err_traj: Vec<Hypervector> = Vec::new();
        for i in 0..l0_traj.len() - ctx1 {
            let w: Vec<&Hypervector> = l0_traj[i..i+ctx1].iter().collect();
            let p = self.levels[1].predict(&w);
            err_traj.push(phase_smooth(&ls_bind(&p, &l0_traj[i + ctx1], 32), 2));
        }
        err_traj.push(phase_smooth(&ls_bind(&l1_pred, &l0_pred, 32), 2));

        while err_traj.len() < ctx2 {
            let fake = Hypervector::random(self.dim);
            err_traj.push(fake);
        }

        let l2_win: Vec<&Hypervector> = err_traj[err_traj.len()-ctx2..].iter().collect();

        let mut corrected_list: Vec<Hypervector> = Vec::new();
        for &t in temps {
            let l2_correction = self.levels[2].predict_with_temp(&l2_win, t);
            corrected_list.push(dampen_correction(&l1_pred, &l2_correction));
        }

        let mut converge = 0.0;
        let mut pairs = 0usize;
        for i in 0..corrected_list.len() {
            for j in i+1..corrected_list.len() {
                let si = crate::ai::sdr::sparsify(&corrected_list[i]);
                let sj = crate::ai::sdr::sparsify(&corrected_list[j]);
                converge += si.soft_overlap(&sj);
                pairs += 1;
            }
        }
        converge /= if pairs > 0 { pairs as f64 } else { 1.0 };

        let chosen = if converge > 0.7 {
            let sdr_refs: Vec<_> = corrected_list.iter().map(|h| crate::ai::sdr::sparsify(h)).collect();
            let bundled = crate::ai::sdr::SdrVector::bundle_multi(&sdr_refs);
            bundled.to_hypervector(self.dim)
        } else {
            l0_pred.clone()
        };

        (vec![l0_pred, l1_pred, chosen], converge)
    }

    pub fn learn(&mut self, context: &[&Hypervector], actual: &[&Hypervector]) -> Vec<f64> {
        let mut errors = Vec::new();
        let cb = FugaCircuitBreaker::new(0.5700);

        for li in 0..self.levels.len() {
            let level = &mut self.levels[li];
            let ctx_len = level.context_len;
            let margin = 1.0;

            if context.len() < ctx_len {
                errors.push(1.0);
                continue;
            }
            let win_slice = &context[context.len()-ctx_len..];
            let win: Vec<&Hypervector> = win_slice.iter().copied().collect();

            if li == 0 {
                let loss = level.train_step(&win, actual[0], margin);
                errors.push(loss);
            } else if li == 1 && actual.len() > 1 {
                let loss = level.train_step(&win, actual[1], margin);
                errors.push(loss);
            } else if li == 2 && actual.len() > 2 {
                let loss = level.train_step(&win, actual[2], margin);
                let state = cb.inspect(loss as f32);
                match state {
                    SystemState::DivergingWarning(_l) => {
                        level.lr *= 0.5;
                        errors.push(loss);
                    }
                    SystemState::CriticalResetRequired => {
                        let mut rng = rand::thread_rng();
                        for w in &mut level.weights {
                            *w = rng.gen_range(-0.005..0.005);
                        }
                        level.velocity.fill(0.0);
                        errors.push(1.0);
                    }
                    _ => errors.push(loss),
                }
            }
        }
        errors
    }

    pub fn learn_ff(&mut self, context: &[&Hypervector], actual: &[&Hypervector], negative_pool: &[Hypervector]) -> Vec<f64> {
        let mut errors = Vec::new();
        for li in 0..self.levels.len() {
            let level = &mut self.levels[li];
            let ctx_len = level.context_len;
            if context.len() < ctx_len {
                errors.push(1.0);
                continue;
            }
            let win_slice = &context[context.len()-ctx_len..];
            let win: Vec<&Hypervector> = win_slice.iter().copied().collect();

            let neg = if !negative_pool.is_empty() {
                let ri = (context.len().wrapping_mul(li.wrapping_add(1))) % negative_pool.len();
                &negative_pool[ri]
            } else {
                actual[li.min(actual.len()-1)]
            };

            if li == 0 {
                let loss = level.train_step_ff(&win, actual[0], neg, 1.0);
                errors.push(loss);
            } else if li == 1 && actual.len() > 1 {
                let loss = level.train_step_ff(&win, actual[1], neg, 1.0);
                errors.push(loss);
            } else if li == 2 && actual.len() > 2 {
                let loss = level.train_step_ff(&win, actual[2], neg, 1.0);
                errors.push(loss);
            }
        }
        errors
    }

    pub fn train_cross_domain(&mut self, pairs_path: &str, epochs: usize) -> f64 {
        let data = std::fs::read_to_string(pairs_path).unwrap_or_default();
        let pairs: Vec<serde_json::Value> = data.lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        println!("  Loaded {} doc→code pairs from {}\n", pairs.len(), pairs_path);
        if pairs.is_empty() { return 1.0; }

        let mut rng = rand::thread_rng();
        let _ctx1 = self.levels[1].context_len;

        for epoch in 0..epochs {
            let mut loss_sum = 0.0f64;
            let mut cnt = 0usize;
            let n_sample = pairs.len().min(3000);
            for _ in 0..n_sample {
                let p = &pairs[rng.gen_range(0..pairs.len())];
                let doc = p["doc"].as_str().unwrap_or("");
                let code = p["code"].as_str().unwrap_or("");
                if doc.len() < 10 || code.len() < 10 { continue; }

                let doc_hv = doc_str_to_hv(doc, self.dim);
                let code_hv = code_str_to_hv(code, self.dim);

                let doc_pred = self.levels[0].predict(&[&doc_hv]);
                let code_pred = self.levels[0].predict(&[&code_hv]);

                let l1_pred = self.levels[1].predict(&[&doc_pred]);
                let margin = 1.0;
                let loss = self.levels[1].train_step(&[&doc_pred], &code_pred, margin);
                loss_sum += loss;
                cnt += 1;

                let neg = Hypervector::random(self.dim);
                let neg_pred = self.levels[0].predict(&[&neg]);
                let sim_pos = dot_bipolar(&l1_pred, &code_pred) / self.dim as f64;
                let sim_neg = dot_bipolar(&l1_pred, &neg_pred) / self.dim as f64;
                let contrastive = (1.0 - sim_pos).max(0.0) + sim_neg.max(0.0).max(0.0);
                loss_sum += 0.1 * contrastive;
                cnt += 1;
            }
            print!("\r    CD epoch {:3}/{}  loss={:.4}", epoch + 1, epochs, loss_sum / cnt as f64);
            use std::io::{Write, stdout};
            stdout().flush().ok();
        }
        println!();

        let mut final_sim = 0.0f64;
        let mut final_cnt = 0usize;
        for p in &pairs[..pairs.len().min(500)] {
            let doc = p["doc"].as_str().unwrap_or("");
            let code = p["code"].as_str().unwrap_or("");
            if doc.len() < 10 || code.len() < 10 { continue; }
            let doc_hv = doc_str_to_hv(doc, self.dim);
            let code_hv = code_str_to_hv(code, self.dim);
            let doc_pred = self.levels[0].predict(&[&doc_hv]);
            let code_pred = self.levels[0].predict(&[&code_hv]);
            let l1_pred = self.levels[1].predict(&[&doc_pred]);
            final_sim += dot_bipolar(&l1_pred, &code_pred) / self.dim as f64;
            final_cnt += 1;
        }
        let avg_sim = if final_cnt > 0 { final_sim / final_cnt as f64 } else { 0.0 };
        println!("  Cross-domain L1 similarity: {:.4} (1.0 = perfect alignment)\n", avg_sim);
        avg_sim
    }

    pub fn train_on_directory(&mut self, dir: &str, epochs: usize) -> f64 {
        let exts = &[".rs", ".py", ".js", ".ts", ".c", ".cpp", ".h", ".go", ".java", ".toml", ".json", ".yaml", ".txt", ".md"];
        let mut files = Vec::new();
        collect_files(dir, exts, &mut files);
        files.truncate(1000);
        println!("  Found {} files in {} (capped at 1000)\n", files.len(), dir);

        let mut raw_seqs: Vec<Vec<Hypervector>> = Vec::new();
        let mut total_chunks = 0usize;

        for (fi, fp) in files.iter().enumerate() {
            let src = match std::fs::read_to_string(fp) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let words: Vec<&str> = src.split_whitespace().collect();
            let n_chunks = (words.len() + 9) / 10;
            if n_chunks <= 6 { continue; }
            total_chunks += n_chunks;

            let hv: Vec<Hypervector> = words.chunks(10).map(|c| {
                let token_hvs: Vec<Hypervector> = c.iter().map(|w| {
                    let id = token_id(w);
                    deterministic_hv(self.dim, &format!("token_{}", id))
                }).collect();
                if token_hvs.is_empty() {
                    Hypervector::new(self.dim)
                } else {
                    let refs: Vec<&Hypervector> = token_hvs.iter().collect();
                    refs[0].bundle(&refs[1..]).balance_density()
                }
            }).collect();

            if hv.len() > self.levels[0].context_len + 1 {
                raw_seqs.push(hv);
            }

            if (fi + 1) % 500 == 0 {
                print!("\r  Encoding: {} files, {} chunks", fi + 1, total_chunks);
                use std::io::{Write, stdout};
                stdout().flush().ok();
            }
        }
        println!("\n  Encoded {} files, {} total chunks\n", files.len(), total_chunks);

        let mut rng = rand::thread_rng();

        // --- Train L0 on raw chunks ---
        println!("  Training L0 (raw chunks → next chunk)...");
        for epoch in 0..epochs {
            let margin = 1.0;
            let mut loss_sum = 0.0f64;
            let mut cnt = 0usize;
            let ctx = self.levels[0].context_len;
            let n_sample = raw_seqs.iter().map(|s| s.len() - ctx).sum::<usize>().min(3000);
            for _ in 0..n_sample {
                let si = rng.gen_range(0..raw_seqs.len());
                let seq = &raw_seqs[si];
                if seq.len() <= ctx + 1 { continue; }
                let wi = rng.gen_range(0..seq.len() - ctx);
                let window: Vec<&Hypervector> = seq[wi..wi+ctx].iter().collect();
                loss_sum += self.levels[0].train_step(&window, &seq[wi + ctx], margin);
                cnt += 1;
            }
            print!("\r    L0 epoch {:3}/{}  loss={:.4}", epoch + 1, epochs, loss_sum / cnt as f64);
            use std::io::{Write, stdout};
            stdout().flush().ok();
        }
        println!();

        // --- Build L1 sequences from L0 predictions ---
        println!("  Generating L1 sequences (L0 predictions)...");
        let ctx0 = self.levels[0].context_len;
        let ctx1 = self.levels[1].context_len;
        const MAX_PREDS: usize = 50000;
        let mut l1_seqs: Vec<Vec<Hypervector>> = Vec::new();
        let mut pred_count = 0usize;
        for (fi, seq) in raw_seqs.iter().enumerate() {
            if pred_count >= MAX_PREDS { break; }
            let limit = (seq.len() - ctx0).min(MAX_PREDS.saturating_sub(pred_count));
            let mut preds = Vec::with_capacity(limit);
            for i in 0..limit {
                let win: Vec<&Hypervector> = seq[i..i+ctx0].iter().collect();
                preds.push(self.levels[0].predict(&win));
            }
            pred_count += preds.len();
            if preds.len() > ctx1 + 1 {
                l1_seqs.push(preds);
            }
            if (fi + 1) % 200 == 0 {
                print!("\r    {} / {} sequences ({} predictions)", fi + 1, raw_seqs.len(), pred_count);
                use std::io::{Write, stdout};
                stdout().flush().ok();
            }
        }
        println!("\n    Generated {} L1 sequences ({} predictions)", l1_seqs.len(), pred_count);

        // --- Train L1 on L0 prediction sequences ---
        println!("  Training L1 (L0 predictions → next L0 prediction)...");
        for epoch in 0..epochs {
            let margin = 1.0;
            let mut loss_sum = 0.0f64;
            let mut cnt = 0usize;
            let n_sample = l1_seqs.iter().map(|s| s.len() - ctx1).sum::<usize>().min(3000);
            for _ in 0..n_sample {
                let si = rng.gen_range(0..l1_seqs.len());
                let seq = &l1_seqs[si];
                if seq.len() <= ctx1 + 1 { continue; }
                let wi = rng.gen_range(0..seq.len() - ctx1);
                let window: Vec<&Hypervector> = seq[wi..wi+ctx1].iter().collect();
                loss_sum += self.levels[1].train_step(&window, &seq[wi + ctx1], margin);
                cnt += 1;
            }
            print!("\r    L1 epoch {:3}/{}  loss={:.4}", epoch + 1, epochs, loss_sum / cnt as f64);
            use std::io::{Write, stdout};
            stdout().flush().ok();
        }
        println!();

        // --- Build L2 sequences: L1 error deltas (l1_pred XOR actual_l0_pred) ---
        println!("  Generating L2 sequences (L1 error deltas)...");
        let ctx2 = self.levels[2].context_len;
        const MAX_ERRORS: usize = 50000;
        let mut l2_seqs: Vec<Vec<Hypervector>> = Vec::new();
        let mut err_count = 0usize;
        for (fi, seq) in l1_seqs.iter().enumerate() {
            if err_count >= MAX_ERRORS { break; }
            let limit = (seq.len() - ctx1).min(MAX_ERRORS.saturating_sub(err_count));
            let mut errors = Vec::with_capacity(limit);
            for i in 0..limit {
                let win: Vec<&Hypervector> = seq[i..i+ctx1].iter().collect();
                let l1_pred = self.levels[1].predict(&win);
                let actual = &seq[i + ctx1];
                let delta = phase_smooth(&ls_bind(&l1_pred, actual, 32), 2);
                errors.push(dampen_correction(&l1_pred, &delta));
            }
            err_count += errors.len();
            if errors.len() > ctx2 + 1 {
                l2_seqs.push(errors);
            }
            if (fi + 1) % 200 == 0 {
                print!("\r    {} / {} sequences ({} errors)", fi + 1, l1_seqs.len(), err_count);
                use std::io::{Write, stdout};
                stdout().flush().ok();
            }
        }
        println!("\n    Generated {} L2 sequences ({} error deltas)", l2_seqs.len(), err_count);

        // --- Train L2 on L1 error deltas ---
        let cb = FugaCircuitBreaker::new(0.5700);
        println!("  Training L2 (error deltas → next error delta)...");
        for epoch in 0..epochs {
            let margin = 1.0;
            let mut loss_sum = 0.0f64;
            let mut cnt = 0usize;
            let n_sample = l2_seqs.iter().map(|s| s.len() - ctx2).sum::<usize>().min(3000);
            for _ in 0..n_sample {
                let si = rng.gen_range(0..l2_seqs.len());
                let seq = &l2_seqs[si];
                if seq.len() <= ctx2 + 1 { continue; }
                let wi = rng.gen_range(0..seq.len() - ctx2);
                let window: Vec<&Hypervector> = seq[wi..wi+ctx2].iter().collect();
                loss_sum += self.levels[2].train_step(&window, &seq[wi + ctx2], margin);
                cnt += 1;
            }
            let avg_loss = loss_sum / cnt as f64;
            let state = cb.inspect(avg_loss as f32);
            match state {
                SystemState::Nominal => {},
                SystemState::DivergingWarning(l) => {
                    print!(" ⚠ cb={:.4}", l);
                    self.levels[2].lr *= 0.5;
                },
                SystemState::CriticalResetRequired => {
                    println!("\n    🔴 CB critical at epoch {} (loss={:.4}) — resetting L2 weights", epoch + 1, avg_loss);
                    let mut rng = rand::thread_rng();
                    for w in &mut self.levels[2].weights {
                        *w = rng.gen_range(-0.005..0.005);
                    }
                    self.levels[2].velocity.fill(0.0);
                },
            }
            print!("\r    L2 epoch {:3}/{}  loss={:.4}", epoch + 1, epochs, avg_loss);
            use std::io::{Write, stdout};
            stdout().flush().ok();
        }
        println!("\n");

        // --- Final eval ---
        let mut final_loss = [0.0f64; 3];
        let mut final_cnt = [0usize; 3];

        for seq in &raw_seqs {
            let ctx = self.levels[0].context_len;
            for i in 0..seq.len() - ctx {
                let win: Vec<&Hypervector> = seq[i..i+ctx].iter().collect();
                final_loss[0] += 1.0 - self.levels[0].similarity_to_expected(&win, &seq[i+ctx]);
                final_cnt[0] += 1;
            }
        }
        for seq in &l1_seqs {
            for i in 0..seq.len() - ctx1 {
                let win: Vec<&Hypervector> = seq[i..i+ctx1].iter().collect();
                final_loss[1] += 1.0 - self.levels[1].similarity_to_expected(&win, &seq[i+ctx1]);
                final_cnt[1] += 1;
            }
        }
        // L2 eval: corrected L1 vs actual L0 prediction
        for seq in &l1_seqs {
            let ctx = ctx1;
            for i in 0..seq.len() - ctx {
                let win: Vec<&Hypervector> = seq[i..i+ctx].iter().collect();
                let l1_pred = self.levels[1].predict(&win);
                if i + 1 < ctx2 { continue; }
                let mut err_win = Vec::with_capacity(ctx2);
                for j in i + 1 - ctx2..=i {
                    let sub_win: Vec<&Hypervector> = seq[j..j+ctx].iter().collect();
                    let sub_pred = self.levels[1].predict(&sub_win);
                    let sub_delta = phase_smooth(&ls_bind(&sub_pred, &seq[j + ctx], 32), 2);
                    err_win.push(dampen_correction(&sub_pred, &sub_delta));
                }
                let err_refs: Vec<&Hypervector> = err_win.iter().collect();
                let l2_correction = self.levels[2].predict(&err_refs);
                let corrected = dampen_correction(&l1_pred, &l2_correction);
                let actual = &seq[i + ctx];
                let loss = 1.0 - (dot_bipolar(&corrected, actual) / self.dim as f64);
                final_loss[2] += loss;
                final_cnt[2] += 1;
            }
        }

        let avg: Vec<f64> = (0..3).map(|li| if final_cnt[li] > 0 { final_loss[li] / final_cnt[li] as f64 } else { 0.5 }).collect();
        let overall = avg.iter().sum::<f64>() / 3.0;
        println!("  Final eval: L0={:.4} L1={:.4} L2={:.4} avg={:.4}", avg[0], avg[1], avg[2], overall);
        overall
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"HJEPA");       // magic
        buf.push(1u8);                          // version
        buf.extend_from_slice(&(self.dim as u32).to_le_bytes());
        buf.extend_from_slice(&(self.levels.len() as u32).to_le_bytes());
        for l in &self.levels { l.save(&mut buf); }
        std::fs::write(path, &buf).map_err(|e| format!("save {}: {}", path, e))
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read {}: {}", path, e))?;
        if data.len() < 9 { return Err("too short".into()); }
        if &data[0..5] != b"HJEPA" { return Err("bad magic".into()); }
        let ver = data[5];
        if ver != 1 { return Err(format!("unknown version {}", ver)); }
        let dim = u32::from_le_bytes(data[6..10].try_into().unwrap()) as usize;
        let n = u32::from_le_bytes(data[10..14].try_into().unwrap()) as usize;
        let mut off = 14usize;
        let mut levels = Vec::with_capacity(n);
        for _ in 0..n { levels.push(JepaLevel::load(&data, &mut off)?); }
        Ok(HierarchicalJEPA { dim, levels })
    }
}

fn collect_files(dir: &str, exts: &[&str], out: &mut Vec<String>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                if let Some(n) = p.file_name().and_then(|n| n.to_str()) {
                    if n.starts_with('.') || n == "node_modules" || n == "target" || n == "thirdparty" { continue; }
                }
                collect_files(&p.to_string_lossy(), exts, out);
            } else if let Some(e) = p.extension().and_then(|e| e.to_str()) {
                if exts.iter().any(|x| *x == format!(".{}", e).as_str()) {
                    out.push(p.to_string_lossy().to_string());
                }
            }
        }
    }
}

fn dampen_correction(base: &Hypervector, correction: &Hypervector) -> Hypervector {
    let xored = base.bind(correction);
    base.bundle(&[&base, &base, &xored])
}

fn str_chunk_hv(text: &str, dim: usize) -> Hypervector {
    let words: Vec<&str> = text.split_whitespace().collect();
    let token_hvs: Vec<Hypervector> = words.iter().map(|w| {
        let id = token_id(w);
        deterministic_hv(dim, &format!("token_{}", id))
    }).collect();
    if token_hvs.is_empty() {
        return Hypervector::random(dim);
    }
    let refs: Vec<&Hypervector> = token_hvs.iter().collect();
    refs[0].bundle(&refs[1..]).balance_density()
}

fn doc_str_to_hv(text: &str, dim: usize) -> Hypervector {
    str_chunk_hv(text, dim)
}

fn code_str_to_hv(text: &str, dim: usize) -> Hypervector {
    str_chunk_hv(text, dim)
}

fn deterministic_hv(dim: usize, seed: &str) -> Hypervector {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut hasher);
    let h = hasher.finish();
    let mut s = [0u8; 32];
    let bytes = h.to_le_bytes();
    for i in 0..8 { s[i] = bytes[i]; }
    let mut rng: rand::rngs::StdRng = SeedableRng::from_seed(s);
    let wc = (dim + 63) / 64;
    let mut words = vec![0u64; wc];
    for word in &mut words {
        *word = rng.next_u64();
    }
    Hypervector { dim, words }
}

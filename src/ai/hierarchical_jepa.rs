use crate::core::hypervector::Hypervector;
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
const LR: f64 = 0.01;
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

fn dot_bipolar(a: &Hypervector, b: &Hypervector) -> f64 {
    a.words.iter().zip(b.words.iter()).flat_map(|(&wa, &wb)| {
        (0..64).map(move |bi| {
            let ai = if (wa >> bi) & 1 == 1 { 1.0 } else { -1.0 };
            let bi = if (wb >> bi) & 1 == 1 { 1.0 } else { -1.0 };
            ai * bi
        })
    }).sum()
}

fn ls_bind(a: &Hypervector, b: &Hypervector, block_bits: usize) -> Hypervector {
    let dim = a.dim;
    let n_words = (dim + 63) / 64;
    let n_blocks = dim / block_bits;
    let phase_bits = (block_bits as f64).log2() as usize;
    let mask = block_bits - 1;

    let mut words = vec![0u64; n_words];

    for blk in 0..n_blocks {
        let base = blk * block_bits;

        let mut phase = 0usize;
        let step = block_bits / phase_bits;
        for pi in 0..phase_bits {
            let src = base + pi * step;
            if src < dim && ((b.words[src / 64] >> (src % 64)) & 1) == 1 {
                phase |= 1 << pi;
            }
        }
        phase &= mask;

        for bi in 0..block_bits {
            let src_bit = base + bi;
            if src_bit >= dim { break; }
            if ((a.words[src_bit / 64] >> (src_bit % 64)) & 1) == 1 {
                let dst_bit = base + (bi + phase) % block_bits;
                words[dst_bit / 64] |= 1 << (dst_bit % 64);
            }
        }
    }

    Hypervector { dim, words }
}

fn phase_smooth(hv: &Hypervector, radius: usize) -> Hypervector {
    if radius == 0 { return hv.clone(); }
    let perms: Vec<Hypervector> = (1..=radius).map(|k| hv.permute(k)).collect();
    let refs: Vec<&Hypervector> = perms.iter().collect();
    hv.bundle(&refs).balance_density()
}

#[derive(Clone, Debug)]
pub struct JepaLevel {
    pub name: String,
    pub dim: usize,
    pub context_len: usize,
    pub stride: usize,
    pub perm_offsets: Vec<usize>,
    pub weights: Vec<f64>,
    pub velocity: Vec<f64>,
    pub lr: f64,
}

impl JepaLevel {
    pub fn new(name: &str, dim: usize, context_len: usize, stride: usize) -> Self {
        let mut rng = rand::thread_rng();
        let perm_offsets: Vec<usize> = (0..context_len)
            .map(|_| rng.gen_range(1..dim))
            .collect();
        let wlen = context_len * dim;
        let mut weights = Vec::with_capacity(wlen);
        for _ in 0..wlen {
            weights.push(rng.gen_range(-0.005..0.005));
        }
        let velocity = vec![0.0; wlen];
        let lr = LR;
        JepaLevel { name: name.to_string(), dim, context_len, stride, perm_offsets, weights, velocity, lr }
    }

    pub fn predict_continuous(&self, context: &[&Hypervector]) -> Vec<f64> {
        let mut raw = vec![0.0f64; self.dim];
        let n = context.len().min(self.context_len);
        for (i, hv) in context.iter().take(n).enumerate() {
            let base = i * self.dim;
            for bit in active_bits(&hv.words, self.dim) {
                raw[bit] += self.weights[base + bit];
            }
        }
        raw
    }

    pub fn predict(&self, context: &[&Hypervector]) -> Hypervector {
        if context.is_empty() {
            return Hypervector::random(self.dim);
        }
        let baseline = context[context.len() - 1];
        let cont = self.predict_continuous(context);
        let pred_bits = hv_to_threshold_bits(&cont);
        let pred_delta = Hypervector::from_i8_bits(self.dim, &pred_bits);
        baseline.bind(&pred_delta)
    }

    pub fn similarity_to_expected(&self, context: &[&Hypervector], actual: &Hypervector) -> f64 {
        let pred = self.predict(context);
        dot_bipolar(&pred, actual) / self.dim as f64
    }

    pub fn train_step(&mut self, context: &[&Hypervector], actual: &Hypervector, margin: f64) -> f64 {
        let dim = self.dim;
        let n_ctx = context.len().min(self.context_len);

        let mut raw = vec![0.0f64; dim];
        for (i, hv) in context.iter().take(n_ctx).enumerate() {
            let base = i * dim;
            for bit in active_bits(&hv.words, dim) {
                raw[bit] += self.weights[base + bit];
            }
        }

        let baseline = context[context.len() - 1];
        let lr = 0.01;
        let decay = 0.001;
        for (i, hv) in context.iter().take(n_ctx).enumerate() {
            let base = i * dim;
            for bit in active_bits(&hv.words, dim) {
                let idx = base + bit;
                let a_bit = (actual.words[bit / 64] >> (bit % 64)) & 1;
                let b_bit = (baseline.words[bit / 64] >> (bit % 64)) & 1;
                let residual_target = if a_bit == b_bit { -1.0 } else { 1.0 };
                let error = margin * residual_target - raw[bit];
                self.weights[idx] = self.weights[idx] * (1.0 - decay) + lr * error;
            }
        }

        let raw_norm = raw.iter().map(|r| r * r).sum::<f64>().sqrt();
        if raw_norm < 1e-9 { return 1.0; }
        let mut dot = 0.0f64;
        for (wi, (&aw, &bw)) in actual.words.iter().zip(baseline.words.iter()).enumerate() {
            for bi in 0..64 {
                let bit = wi * 64 + bi;
                if bit >= dim { break; }
                let a_bit = (aw >> bi) & 1;
                let b_bit = (bw >> bi) & 1;
                let res = if a_bit == b_bit { -1.0 } else { 1.0 };
                dot += raw[bit] * res;
            }
        }
        1.0 - (dot / (raw_norm * (dim as f64).sqrt())).clamp(-1.0, 1.0)
    }

    fn save(&self, buf: &mut Vec<u8>) {
        let nb = self.name.as_bytes();
        buf.extend_from_slice(&(nb.len() as u32).to_le_bytes());
        buf.extend_from_slice(nb);
        buf.extend_from_slice(&(self.context_len as u32).to_le_bytes());
        buf.extend_from_slice(&(self.stride as u32).to_le_bytes());
        for &off in &self.perm_offsets {
            buf.extend_from_slice(&(off as u32).to_le_bytes());
        }
        for &w in &self.weights {
            buf.extend_from_slice(&w.to_le_bytes());
        }
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
        let mut offsets = Vec::with_capacity(ctx);
        for _ in 0..ctx {
            offsets.push(u32::from_le_bytes(data[*offset..*offset+4].try_into().unwrap()) as usize);
            *offset += 4;
        }
        let dim = 8192usize;
        let wlen = ctx * dim;
        let mut weights = Vec::with_capacity(wlen);
        for _ in 0..wlen {
            weights.push(f64::from_le_bytes(data[*offset..*offset+8].try_into().unwrap()));
            *offset += 8;
        }
        let velocity = vec![0.0; wlen];
        Ok(JepaLevel { name, dim, context_len: ctx, stride, perm_offsets: offsets, weights, velocity, lr: LR })
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
        let corrected_l1 = l1_pred.bind(&l2_correction);

        vec![l0_pred, l1_pred, corrected_l1]
    }

    pub fn train_on_directory(&mut self, dir: &str, epochs: usize) -> f64 {
        let exts = &[".rs", ".py", ".js", ".ts", ".c", ".cpp", ".h", ".go", ".java", ".toml", ".json", ".yaml"];
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
        let mut l1_seqs: Vec<Vec<Hypervector>> = Vec::new();
        for (fi, seq) in raw_seqs.iter().enumerate() {
            let mut preds = Vec::with_capacity(seq.len() - ctx0);
            for i in 0..seq.len() - ctx0 {
                let win: Vec<&Hypervector> = seq[i..i+ctx0].iter().collect();
                preds.push(self.levels[0].predict(&win));
            }
            if preds.len() > ctx1 + 1 {
                l1_seqs.push(preds);
            }
            if (fi + 1) % 200 == 0 {
                print!("\r    {} / {} sequences", fi + 1, raw_seqs.len());
                use std::io::{Write, stdout};
                stdout().flush().ok();
            }
        }
        println!("\n    Generated {} L1 sequences", l1_seqs.len());

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
        let mut l2_seqs: Vec<Vec<Hypervector>> = Vec::new();
        for (fi, seq) in l1_seqs.iter().enumerate() {
            let mut errors = Vec::with_capacity(seq.len() - ctx1);
            for i in 0..seq.len() - ctx1 {
                let win: Vec<&Hypervector> = seq[i..i+ctx1].iter().collect();
                let l1_pred = self.levels[1].predict(&win);
                let actual = &seq[i + ctx1];
                errors.push(phase_smooth(&ls_bind(&l1_pred, actual, 32), 2));
            }
            if errors.len() > ctx2 + 1 {
                l2_seqs.push(errors);
            }
            if (fi + 1) % 200 == 0 {
                print!("\r    {} / {} sequences", fi + 1, l1_seqs.len());
                use std::io::{Write, stdout};
                stdout().flush().ok();
            }
        }
        println!("\n    Generated {} L2 sequences (L1 error deltas)", l2_seqs.len());

        // --- Train L2 on L1 error deltas ---
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
            print!("\r    L2 epoch {:3}/{}  loss={:.4}", epoch + 1, epochs, loss_sum / cnt as f64);
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
                    err_win.push(phase_smooth(&ls_bind(&sub_pred, &seq[j + ctx], 32), 2));
                }
                let err_refs: Vec<&Hypervector> = err_win.iter().collect();
                let l2_correction = self.levels[2].predict(&err_refs);
                let corrected = l1_pred.bind(&l2_correction);
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

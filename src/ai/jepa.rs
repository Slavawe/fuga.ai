use crate::core::hypervector::Hypervector;
use rand::Rng;

const DEFAULT_DIM: usize = 8192;
const MAX_CONTEXT: usize = 8;

pub struct JepaPredictor {
    pub dim: usize,
    pub context_len: usize,
    pub perm_offsets: Vec<usize>,
    pub weights: Vec<f64>,
}

impl JepaPredictor {
    pub fn new(dim: usize, context_len: usize) -> Self {
        let cl = context_len.max(1).min(MAX_CONTEXT);
        let mut rng = rand::thread_rng();
        let perm_offsets: Vec<usize> = (0..cl).map(|_| rng.gen_range(1..dim)).collect();
        let weights = vec![1.0 / cl as f64; cl];
        Self { dim, context_len: cl, perm_offsets, weights }
    }

    pub fn predict(&self, context: &[&Hypervector]) -> Hypervector {
        let n = context.len().min(self.context_len);
        if n == 0 {
            return Hypervector::random(self.dim);
        }
        let offset = self.context_len - n;
        let mut combined = context[0].permute(self.perm_offsets[offset]);
        for i in 1..n {
            let permuted = context[i].permute(self.perm_offsets[offset + i]);
            combined = combined.bundle(&[&permuted]);
        }
        combined
    }

    pub fn train_on_sequences(
        &mut self,
        sequences: &[Vec<Hypervector>],
        epochs: usize,
    ) -> f64 {
        let mut rng = rand::thread_rng();
        let mut best_loss = f64::MAX;
        let mut best_perm = self.perm_offsets.clone();
        let mut best_weights = self.weights.clone();

        for _epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut count = 0;

            for seq in sequences {
                if seq.len() < self.context_len + 1 {
                    continue;
                }
                for i in 0..seq.len() - self.context_len {
                    let context: Vec<&Hypervector> = seq[i..i + self.context_len].iter().collect();
                    let predicted = self.predict(&context);
                    let actual = &seq[i + self.context_len];
                    let sim = predicted.similarity(actual);
                    total_loss += 1.0 - sim;
                    count += 1;
                }
            }

            let avg_loss = if count > 0 { total_loss / count as f64 } else { 1.0 };

            if avg_loss < best_loss {
                best_loss = avg_loss;
                best_perm = self.perm_offsets.clone();
                best_weights = self.weights.clone();
            }

            for p in &mut self.perm_offsets {
                if rng.gen_bool(0.3) {
                    *p = rng.gen_range(1..self.dim);
                } else {
                    let delta = rng.gen_range(1..101);
                    *p = (*p + delta) % self.dim;
                    if *p == 0 { *p = 1; }
                }
            }
            for w in &mut self.weights {
                *w += rng.gen_range(-0.1..0.1);
                if *w < 0.01 { *w = 0.01; }
            }
            let sum: f64 = self.weights.iter().sum();
            for w in &mut self.weights {
                *w /= sum;
            }
        }

        self.perm_offsets = best_perm;
        self.weights = best_weights;
        best_loss
    }

    pub fn similarity_to_expected(&self, context: &[&Hypervector], actual: &Hypervector) -> f64 {
        let predicted = self.predict(context);
        predicted.similarity(actual)
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)
            .map_err(|e| format!("Failed to create {}: {}", path, e))?;
        let dim = self.dim as u32;
        f.write_all(&dim.to_le_bytes()).map_err(|e| e.to_string())?;
        let cl = self.context_len as u32;
        f.write_all(&cl.to_le_bytes()).map_err(|e| e.to_string())?;
        for &p in &self.perm_offsets {
            f.write_all(&(p as u32).to_le_bytes()).map_err(|e| e.to_string())?;
        }
        for &w in &self.weights {
            f.write_all(&w.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, String> {
        
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Failed to open {}: {}", path, e))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| format!("Failed to mmap {}: {}", path, e))?;
        let data = &mmap[..];
        let mut pos = 0usize;
        let dim = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
        pos += 4;
        let cl = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
        pos += 4;
        let mut perm_offsets = Vec::with_capacity(cl);
        for _ in 0..cl {
            let p = u32::from_le_bytes(data[pos..pos+4].try_into().unwrap()) as usize;
            pos += 4;
            perm_offsets.push(p);
        }
        let mut weights = Vec::with_capacity(cl);
        for _ in 0..cl {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&data[pos..pos+8]);
            pos += 8;
            weights.push(f64::from_le_bytes(buf));
        }
        Ok(Self { dim, context_len: cl, perm_offsets, weights })
    }
}

use crate::ai::hierarchical_jepa::HierarchicalJEPA;
use crate::ai::htm_temporal::TemporalMemory;
use crate::ai::sdr::{SDR_WORDS, SdrVector, encode_text};
use crate::core::hypervector::Hypervector;

pub fn sdr_to_hypervector(sdr: &SdrVector, dim: usize) -> Hypervector {
    let mut hv = Hypervector::new(dim);
    let copy_len = hv.words.len().min(SDR_WORDS);
    for i in 0..copy_len {
        hv.words[i] = sdr.bits[i];
    }
    hv
}

const BUFFER_PATH: &str = "fuga_buffer.bin";
const BUFFER_MAGIC: &[u8] = b"FUGA_BUF";

pub struct TemporalPredictor {
    pub tm: TemporalMemory,
    pub hjepa: HierarchicalJEPA,
    pub buffer: Vec<Hypervector>,
    pub buf_capacity: usize,
}

impl TemporalPredictor {
    pub fn save_buffer(&self) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::File::create(BUFFER_PATH) {
            f.write_all(BUFFER_MAGIC).ok();
            f.write_all(&(self.buffer.len() as u32).to_le_bytes()).ok();
            for hv in &self.buffer {
                f.write_all(&(hv.dim as u32).to_le_bytes()).ok();
                let bytes = hv.to_bytes();
                f.write_all(&(bytes.len() as u32).to_le_bytes()).ok();
                f.write_all(&bytes).ok();
            }
        }
    }

    pub fn load_buffer(&mut self) {
        let data = match std::fs::read(BUFFER_PATH) {
            Ok(d) => d,
            Err(_) => return,
        };
        if &data[..8] != BUFFER_MAGIC {
            return;
        }
        let mut off = 8usize;
        if off + 4 > data.len() {
            return;
        }
        let n = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let mut buf = Vec::with_capacity(n);
        for _ in 0..n {
            if off + 4 > data.len() {
                break;
            }
            let dim = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
            off += 4;
            if off + 4 > data.len() {
                break;
            }
            let blen = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
            off += 4;
            if off + blen > data.len() {
                break;
            }
            let words: Vec<u64> = data[off..off + blen]
                .chunks(8)
                .filter_map(|c| Some(u64::from_le_bytes(c.try_into().ok()?)))
                .collect();
            off += blen;
            buf.push(Hypervector::from_raw(dim, words));
        }
        self.buffer = buf;
    }
}

impl TemporalPredictor {
    pub fn new(tm: TemporalMemory, hjepa: HierarchicalJEPA) -> Self {
        let buf_capacity = hjepa.levels[0].context_len + hjepa.levels[1].context_len + 2;
        TemporalPredictor {
            tm,
            hjepa,
            buffer: Vec::new(),
            buf_capacity,
        }
    }

    pub fn dim(&self) -> usize {
        self.hjepa.dim
    }

    pub fn feed(&mut self, text: &str) -> (f64, f64, f64) {
        let sdr = encode_text(text);
        let (tm_pred, tm_match) = self.tm.feed(&sdr);

        let hv = sdr_to_hypervector(&tm_pred, self.hjepa.dim);
        self.buffer.push(hv);
        if self.buffer.len() > self.buf_capacity {
            self.buffer.remove(0);
        }

        if self.buffer.len() < self.hjepa.levels[0].context_len {
            return (tm_match, 1.0, 1.0);
        }

        let ctx: Vec<&Hypervector> = self.buffer.iter().collect();
        let preds = self.hjepa.predict(&ctx);
        let actual = self.buffer.last().unwrap();
        let l0_err = 1.0 - self.hjepa.levels[0].similarity_to_expected(&ctx, actual);
        let l1_err = if self.buffer.len()
            >= self.hjepa.levels[0].context_len + self.hjepa.levels[1].context_len
        {
            let l1_ctx_end = self.buffer.len() - self.hjepa.levels[0].context_len;
            let l1_start = l1_ctx_end.saturating_sub(self.hjepa.levels[1].context_len);
            let l1_ctx: Vec<&Hypervector> = self.buffer[l1_start..l1_ctx_end].iter().collect();
            let l0_pred = &preds[0];
            1.0 - self.hjepa.levels[1].similarity_to_expected(&l1_ctx, l0_pred)
        } else {
            1.0
        };

        (tm_match, l0_err, l1_err)
    }

    pub fn feed_learn(&mut self, text: &str) -> (f64, Vec<f64>) {
        let sdr = encode_text(text);
        let (tm_pred, tm_match) = self.tm.feed(&sdr);

        let hv = sdr_to_hypervector(&tm_pred, self.hjepa.dim);
        self.buffer.push(hv);
        if self.buffer.len() > self.buf_capacity {
            self.buffer.remove(0);
        }

        if self.buffer.len() < self.hjepa.levels[0].context_len + 1 {
            return (tm_match, vec![1.0; 3]);
        }

        let buf_minus_1 = self.buffer.len() - 1;
        let ctx: Vec<&Hypervector> = self.buffer[..buf_minus_1].iter().collect();
        let actual = &self.buffer[buf_minus_1];

        let l0_ctx_end = buf_minus_1;
        let l0_start = l0_ctx_end.saturating_sub(self.hjepa.levels[0].context_len);
        let l0_ctx: Vec<&Hypervector> = self.buffer[l0_start..l0_ctx_end].iter().collect();
        let l0_pred = self.hjepa.levels[0].predict(&l0_ctx);

        let l1_pred =
            if buf_minus_1 >= self.hjepa.levels[0].context_len + self.hjepa.levels[1].context_len {
                let l1_ctx_end = buf_minus_1 - self.hjepa.levels[0].context_len;
                let l1_start = l1_ctx_end.saturating_sub(self.hjepa.levels[1].context_len);
                let l1_ctx: Vec<&Hypervector> = self.buffer[l1_start..l1_ctx_end].iter().collect();
                Some(self.hjepa.levels[1].predict(&l1_ctx))
            } else {
                None
            };

        let mut actuals: Vec<Hypervector> = Vec::new();
        actuals.push(actual.clone());
        if let Some(ref l1) = l1_pred {
            actuals.push(l1.clone());
        } else {
            actuals.push(l0_pred.clone());
        }
        actuals.push(l0_pred.clone());

        let actual_refs: Vec<&Hypervector> = actuals.iter().collect();
        let errors = self.hjepa.learn(&ctx, &actual_refs);
        (tm_match, errors)
    }

    pub fn feed_learn_no_tm(&mut self, text: &str) -> Vec<f64> {
        let sdr = encode_text(text);
        let (tm_pred, _tm_match) = self.tm.feed_no_learn(&sdr);

        let hv = sdr_to_hypervector(&tm_pred, self.hjepa.dim);
        self.buffer.push(hv);
        if self.buffer.len() > self.buf_capacity {
            self.buffer.remove(0);
        }

        if self.buffer.len() < self.hjepa.levels[0].context_len + 1 {
            return vec![1.0; 3];
        }

        let buf_minus_1 = self.buffer.len() - 1;
        let ctx: Vec<&Hypervector> = self.buffer[..buf_minus_1].iter().collect();
        let actual = &self.buffer[buf_minus_1];

        let l0_ctx_end = buf_minus_1;
        let l0_start = l0_ctx_end.saturating_sub(self.hjepa.levels[0].context_len);
        let l0_ctx: Vec<&Hypervector> = self.buffer[l0_start..l0_ctx_end].iter().collect();
        let l0_pred = self.hjepa.levels[0].predict(&l0_ctx);

        let l1_pred =
            if buf_minus_1 >= self.hjepa.levels[0].context_len + self.hjepa.levels[1].context_len {
                let l1_ctx_end = buf_minus_1 - self.hjepa.levels[0].context_len;
                let l1_start = l1_ctx_end.saturating_sub(self.hjepa.levels[1].context_len);
                let l1_ctx: Vec<&Hypervector> = self.buffer[l1_start..l1_ctx_end].iter().collect();
                Some(self.hjepa.levels[1].predict(&l1_ctx))
            } else {
                None
            };

        let mut actuals: Vec<Hypervector> = Vec::new();
        actuals.push(actual.clone());
        if let Some(ref l1) = l1_pred {
            actuals.push(l1.clone());
        } else {
            actuals.push(l0_pred.clone());
        }
        actuals.push(l0_pred.clone());

        let actual_refs: Vec<&Hypervector> = actuals.iter().collect();
        self.hjepa.learn(&ctx, &actual_refs)
    }

    pub fn feed_learn_ff(&mut self, text: &str, neg_pool: &[Hypervector]) -> Vec<f64> {
        let sdr = encode_text(text);
        let (tm_pred, _tm_match) = self.tm.feed_no_learn(&sdr);

        let hv = sdr_to_hypervector(&tm_pred, self.hjepa.dim);
        self.buffer.push(hv);
        if self.buffer.len() > self.buf_capacity {
            self.buffer.remove(0);
        }

        if self.buffer.len() < self.hjepa.levels[0].context_len + 1 {
            return vec![1.0; 3];
        }

        let buf_minus_1 = self.buffer.len() - 1;
        let ctx: Vec<&Hypervector> = self.buffer[..buf_minus_1].iter().collect();
        let actual = &self.buffer[buf_minus_1];

        let l0_ctx_end = buf_minus_1;
        let l0_start = l0_ctx_end.saturating_sub(self.hjepa.levels[0].context_len);
        let l0_ctx: Vec<&Hypervector> = self.buffer[l0_start..l0_ctx_end].iter().collect();
        let l0_pred = self.hjepa.levels[0].predict(&l0_ctx);

        let l1_pred =
            if buf_minus_1 >= self.hjepa.levels[0].context_len + self.hjepa.levels[1].context_len {
                let l1_ctx_end = buf_minus_1 - self.hjepa.levels[0].context_len;
                let l1_start = l1_ctx_end.saturating_sub(self.hjepa.levels[1].context_len);
                let l1_ctx: Vec<&Hypervector> = self.buffer[l1_start..l1_ctx_end].iter().collect();
                Some(self.hjepa.levels[1].predict(&l1_ctx))
            } else {
                None
            };

        let mut actuals: Vec<Hypervector> = Vec::new();
        actuals.push(actual.clone());
        if let Some(ref l1) = l1_pred {
            actuals.push(l1.clone());
        } else {
            actuals.push(l0_pred.clone());
        }
        actuals.push(l0_pred.clone());

        let actual_refs: Vec<&Hypervector> = actuals.iter().collect();
        self.hjepa.learn_ff(&ctx, &actual_refs, neg_pool)
    }

    pub fn generate(&mut self, text: &str, steps: usize) -> Vec<Hypervector> {
        let sdr = encode_text(text);
        let (tm_pred, _) = self.tm.feed(&sdr);

        let hv = sdr_to_hypervector(&tm_pred, self.hjepa.dim);
        self.buffer.push(hv);
        if self.buffer.len() > self.buf_capacity {
            self.buffer.remove(0);
        }

        if self.buffer.len() < self.hjepa.levels[0].context_len {
            return Vec::new();
        }

        let ctx: Vec<&Hypervector> = self.buffer.iter().collect();
        self.hjepa.predict_sequence(&ctx, steps)
    }

    /// Build a decode vocabulary mapping words to their Hypervectors in the
    /// same SDR→HV space the buffer feeds into the H-JEPA (raw word SDRs are
    /// the space the levels learned over), so predicted latents can be decoded
    /// back to tokens by nearest-neighbour similarity.
    pub fn word_vocab(&self, words: &[String]) -> Vec<(String, Hypervector)> {
        let mut seen = std::collections::HashSet::new();
        words
            .iter()
            .map(|w| w.to_lowercase())
            .filter(|w| w.len() >= 2 && seen.insert(w.clone()))
            .map(|w| {
                let sdr = encode_text(&w);
                let hv = sdr_to_hypervector(&sdr, self.hjepa.dim);
                (w, hv)
            })
            .collect()
    }

    /// H-JEPA sequential generation for the mask: seed the buffer with the raw
    /// SDR Hypervector of each seed word (the same space [`Self::word_vocab`]
    /// and `feed_learn_hv_only` use), roll out `steps` latent Hypervectors with
    /// `predict_sequence`, then decode each latent to the nearest vocabulary
    /// word. Repeats are suppressed.
    pub fn generate_words(
        &mut self,
        seed: &str,
        steps: usize,
        vocab: &[(String, Hypervector)],
        min_sim: f64,
    ) -> Vec<String> {
        for w in seed.split_whitespace() {
            let sdr = encode_text(w);
            let hv = sdr_to_hypervector(&sdr, self.hjepa.dim);
            self.buffer.push(hv);
            if self.buffer.len() > self.buf_capacity {
                self.buffer.remove(0);
            }
        }
        // A short seed may leave the buffer shorter than the level-0 context
        // window; pad with the last encoded token so the roll-out still runs.
        let ctx0 = self.hjepa.levels[0].context_len;
        while self.buffer.len() < ctx0 {
            let pad = self
                .buffer
                .last()
                .cloned()
                .unwrap_or_else(|| sdr_to_hypervector(&SdrVector::zero(), self.hjepa.dim));
            self.buffer.push(pad);
        }
        if self.buffer.len() < ctx0 {
            return Vec::new();
        }
        let ctx: Vec<&Hypervector> = self.buffer.iter().collect();
        let seq = self.hjepa.predict_sequence(&ctx, steps);

        let mut out: Vec<String> = Vec::new();
        for hv in &seq {
            let mut best: Option<(usize, f64)> = None;
            for (i, (_, v)) in vocab.iter().enumerate() {
                let sim = hv.similarity(v);
                match best {
                    Some((_, bs)) if sim > bs => best = Some((i, sim)),
                    None => best = Some((i, sim)),
                    _ => {}
                }
            }
            if let Some((i, sim)) = best {
                if sim >= min_sim && out.last().map(|l: &String| l != &vocab[i].0).unwrap_or(true)
                {
                    out.push(vocab[i].0.clone());
                }
            }
        }
        out
    }

    pub fn feed_learn_hv_only(&mut self, text: &str) -> Vec<f64> {
        let sdr = encode_text(text);
        let hv = sdr_to_hypervector(&sdr, self.hjepa.dim);
        self.buffer.push(hv);
        if self.buffer.len() > self.buf_capacity {
            self.buffer.remove(0);
        }
        if self.buffer.len() < self.hjepa.levels[0].context_len + 1 {
            return vec![1.0; 3];
        }
        let buf_minus_1 = self.buffer.len() - 1;
        let ctx: Vec<&Hypervector> = self.buffer[..buf_minus_1].iter().collect();
        let actual = &self.buffer[buf_minus_1];
        let l0_ctx_end = buf_minus_1;
        let l0_start = l0_ctx_end.saturating_sub(self.hjepa.levels[0].context_len);
        let l0_ctx: Vec<&Hypervector> = self.buffer[l0_start..l0_ctx_end].iter().collect();
        let l0_pred = self.hjepa.levels[0].predict(&l0_ctx);
        let l1_pred =
            if buf_minus_1 >= self.hjepa.levels[0].context_len + self.hjepa.levels[1].context_len {
                let l1_ctx_end = buf_minus_1 - self.hjepa.levels[0].context_len;
                let l1_start = l1_ctx_end.saturating_sub(self.hjepa.levels[1].context_len);
                let l1_ctx: Vec<&Hypervector> = self.buffer[l1_start..l1_ctx_end].iter().collect();
                Some(self.hjepa.levels[1].predict(&l1_ctx))
            } else {
                None
            };
        let mut actuals: Vec<Hypervector> = Vec::new();
        actuals.push(actual.clone());
        if let Some(ref l1) = l1_pred {
            actuals.push(l1.clone());
        } else {
            actuals.push(l0_pred.clone());
        }
        actuals.push(l0_pred.clone());
        let actual_refs: Vec<&Hypervector> = actuals.iter().collect();
        self.hjepa.learn(&ctx, &actual_refs)
    }

    pub fn stats(&self) -> String {
        let total_segments: usize = self.tm.cells.iter().map(|c| c.segments.len()).sum();
        format!(
            "tm_cells={} tm_segments={} hjepa_lr=[{:.4},{:.4},{:.4}] buf={}",
            self.tm.cells.len(),
            total_segments,
            self.hjepa.levels[0].lr,
            self.hjepa.levels[1].lr,
            self.hjepa.levels[2].lr,
            self.buffer.len()
        )
    }
}

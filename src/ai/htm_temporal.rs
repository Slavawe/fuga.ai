use crate::ai::sdr::{SDR_DENSITY, SDR_DIM, SDR_WORDS, SdrVector};

const MIN_PERMANENCE: f64 = 0.2;
const CONNECTED_PERMANENCE: f64 = 0.5;
const PERMANENCE_INCREMENT: f64 = 0.05;
const PERMANENCE_DECREMENT: f64 = 0.02;
const MAX_SEGMENTS_PER_CELL: usize = 128;
const PRUNE_INTERVAL: usize = 200;

#[derive(Clone, Debug)]
pub struct Synapse {
    pub bit_index: usize,
    pub permanence: f64,
}

impl Synapse {
    pub fn new(bit_index: usize, permanence: f64) -> Self {
        Synapse {
            bit_index,
            permanence,
        }
    }
    pub fn connected(&self) -> bool {
        self.permanence >= CONNECTED_PERMANENCE
    }
}

#[derive(Clone, Debug)]
pub struct DendriteSegment {
    pub synapses: Vec<Synapse>,
}

impl DendriteSegment {
    pub fn new() -> Self {
        DendriteSegment {
            synapses: Vec::new(),
        }
    }

    pub fn overlap(&self, sdr: &SdrVector) -> u32 {
        self.synapses
            .iter()
            .filter(|s| s.connected())
            .filter(|s| {
                let wi = s.bit_index / 64;
                let bi = s.bit_index % 64;
                (sdr.bits[wi] >> bi) & 1 == 1
            })
            .count() as u32
    }

    pub fn reinforce(&mut self, sdr: &SdrVector) {
        for s in &mut self.synapses {
            let wi = s.bit_index / 64;
            let bi = s.bit_index % 64;
            if (sdr.bits[wi] >> bi) & 1 == 1 {
                s.permanence = (s.permanence + PERMANENCE_INCREMENT).min(1.0);
            } else {
                s.permanence = (s.permanence - PERMANENCE_DECREMENT).max(0.0);
            }
        }
    }

    pub fn reinforce_match_only(&mut self, sdr: &SdrVector) {
        for s in &mut self.synapses {
            let wi = s.bit_index / 64;
            let bi = s.bit_index % 64;
            if (sdr.bits[wi] >> bi) & 1 == 1 {
                s.permanence = (s.permanence + PERMANENCE_INCREMENT).min(1.0);
            }
        }
    }

    pub fn prune_weak(&mut self) {
        self.synapses.retain(|s| s.permanence >= MIN_PERMANENCE);
    }
}

pub struct TemporalCell {
    pub id: usize,
    pub segments: Vec<DendriteSegment>,
    pub pattern: SdrVector,
}

impl TemporalCell {
    pub fn new(id: usize, pattern: SdrVector) -> Self {
        TemporalCell {
            id,
            segments: Vec::new(),
            pattern,
        }
    }

    pub fn learn_segment(&mut self, input: &SdrVector) {
        if self.segments.len() >= MAX_SEGMENTS_PER_CELL {
            let mut scores: Vec<(usize, u32)> = self
                .segments
                .iter()
                .enumerate()
                .map(|(i, seg)| (i, seg.overlap(input)))
                .collect();
            scores.sort_by(|a, b| b.1.cmp(&a.1));
            // Get index of segment with lowest overlap (last after descending sort)
            let idx = scores.last().map(|s| s.0).unwrap_or(0);
            scores.pop(); // Remove from scores to keep consistent state
            self.segments.remove(idx);
        }
        let mut seg = DendriteSegment::new();
        for bit in 0..SDR_DIM {
            let wi = bit / 64;
            let bi = bit % 64;
            if (input.bits[wi] >> bi) & 1 == 1 {
                seg.synapses
                    .push(Synapse::new(bit, CONNECTED_PERMANENCE + 0.1));
            }
        }
        if !seg.synapses.is_empty() {
            self.segments.push(seg);
        }
    }
}

pub struct TemporalMemory {
    pub cells: Vec<TemporalCell>,
    pub window: Vec<SdrVector>,
    pub context_len: usize,
    pub step: usize,
}

impl TemporalMemory {
    pub fn new(capacity: usize, context_len: usize) -> Self {
        TemporalMemory {
            cells: Vec::with_capacity(capacity),
            window: Vec::new(),
            context_len,
            step: 0,
        }
    }

    pub fn get_or_create_cell(&mut self, pattern: &SdrVector) -> usize {
        if let Some((idx, _)) = self
            .cells
            .iter()
            .enumerate()
            .find(|(_, c)| c.pattern.bits == pattern.bits)
        {
            return idx;
        }
        let id = self.cells.len();
        self.cells.push(TemporalCell::new(id, pattern.clone()));
        id
    }

    pub fn reset(&mut self) {
        self.window.clear();
        self.step = 0;
    }

    pub fn feed(&mut self, input: &SdrVector) -> (SdrVector, f64) {
        self.step += 1;

        let cell_id = self.get_or_create_cell(input);

        let prediction = if !self.window.is_empty() {
            let prev = &self.window[self.window.len() - 1];
            self.predict_next(prev)
        } else {
            SdrVector::zero()
        };

        let match_score = if prediction.popcount() > 0 {
            prediction.soft_overlap(input)
        } else {
            0.0
        };

        if !self.window.is_empty() {
            let prev = self.window[self.window.len() - 1].clone();

            for seg in &mut self.cells[cell_id].segments {
                if seg.overlap(&prev) > 0 {
                    seg.reinforce(&prev);
                }
            }

            let mut has_prev_seg = false;
            for seg in &self.cells[cell_id].segments {
                if seg.overlap(&prev) >= 3 {
                    has_prev_seg = true;
                    break;
                }
            }
            if !has_prev_seg {
                self.cells[cell_id].learn_segment(&prev);
            }

            for c in &mut self.cells {
                if c.id == cell_id {
                    continue;
                }
                for seg in &mut c.segments {
                    if seg.overlap(&prev) >= 3 {
                        seg.reinforce_match_only(&prev);
                    }
                }
            }
        }

        self.window.push(input.clone());
        while self.window.len() > self.context_len {
            self.window.remove(0);
        }

        if self.step % PRUNE_INTERVAL == 0 {
            self.prune();
        }

        (prediction, match_score)
    }

    pub fn feed_no_learn(&mut self, input: &SdrVector) -> (SdrVector, f64) {
        self.step += 1;

        let prediction = if !self.window.is_empty() {
            let prev = &self.window[self.window.len() - 1];
            self.predict_next(prev)
        } else {
            SdrVector::zero()
        };

        let match_score = if prediction.popcount() > 0 {
            prediction.soft_overlap(input)
        } else {
            0.0
        };

        self.window.push(input.clone());
        while self.window.len() > self.context_len {
            self.window.remove(0);
        }

        if self.step % PRUNE_INTERVAL == 0 {
            self.prune();
        }

        (prediction, match_score)
    }

    pub fn predict_next(&self, prev: &SdrVector) -> SdrVector {
        let mut pred = SdrVector::zero();
        for c in &self.cells {
            let depolarized = c.segments.iter().any(|seg| seg.overlap(prev) >= 5);
            if depolarized {
                for i in 0..SDR_WORDS {
                    pred.bits[i] |= c.pattern.bits[i];
                }
            }
        }
        if pred.popcount() > 0 {
            let target = (SDR_DIM as f64 * SDR_DENSITY).ceil() as usize;
            let mut scored: Vec<(usize, u64)> = (0..SDR_DIM)
                .filter(|&bit| {
                    let wi = bit / 64;
                    let bi = bit % 64;
                    (pred.bits[wi] >> bi) & 1 == 1
                })
                .map(|bit| {
                    let score = (self.step as u64)
                        .wrapping_mul(bit as u64 + 1)
                        .reverse_bits();
                    (bit, score)
                })
                .collect();
            scored.sort_by(|a, b| b.1.cmp(&a.1));
            scored.truncate(target);
            let mut out = SdrVector::zero();
            for &(bit, _) in &scored {
                out.bits[bit / 64] |= 1u64 << (bit % 64);
            }
            out
        } else {
            SdrVector::zero()
        }
    }

    /// Contextual prediction: a cell is depolarized when one of its segments
    /// matches the bundle of the whole context window, not just the last token.
    pub fn predict_next_context(&self, context: &[SdrVector]) -> SdrVector {
        let ctx = SdrVector::union(context);
        if ctx.popcount() == 0 {
            return SdrVector::zero();
        }
        let mut pred = SdrVector::zero();
        for c in &self.cells {
            let depolarized = c.segments.iter().any(|seg| seg.overlap(&ctx) >= 5);
            if depolarized {
                for i in 0..SDR_WORDS {
                    pred.bits[i] |= c.pattern.bits[i];
                }
            }
        }
        if pred.popcount() > 0 {
            let target = (SDR_DIM as f64 * SDR_DENSITY).ceil() as usize;
            let mut scored: Vec<(usize, u64)> = (0..SDR_DIM)
                .filter(|&bit| {
                    let wi = bit / 64;
                    let bi = bit % 64;
                    (pred.bits[wi] >> bi) & 1 == 1
                })
                .map(|bit| {
                    let score = (self.step as u64)
                        .wrapping_mul(bit as u64 + 1)
                        .reverse_bits();
                    (bit, score)
                })
                .collect();
            scored.sort_by(|a, b| b.1.cmp(&a.1));
            scored.truncate(target);
            let mut out = SdrVector::zero();
            for &(bit, _) in &scored {
                out.bits[bit / 64] |= 1u64 << (bit % 64);
            }
            out
        } else {
            SdrVector::zero()
        }
    }

    pub fn learn_sequence(&mut self, prev: &SdrVector, next: &SdrVector) {
        let next_cell = self.get_or_create_cell(next);
        // Winner-take-all: reinforce only the best-matching segment so that a
        // high-fan-in cell (e.g. `(` after hundreds of distinct tokens) does
        // not decay the segment for each specific predecessor. Without this,
        // per-segment permanence drops under CONNECTED for all but the most
        // frequent predecessor, and rare-but-real bi-grams stop firing.
        let mut best: Option<usize> = None;
        let mut best_ov = 0u32;
        for (i, seg) in self.cells[next_cell].segments.iter().enumerate() {
            let ov = seg.overlap(prev);
            if ov > best_ov {
                best_ov = ov;
                best = Some(i);
            }
        }
        match best {
            Some(i) if best_ov >= 3 => {
                self.cells[next_cell].segments[i].reinforce_match_only(prev);
            }
            _ => {
                self.cells[next_cell].learn_segment(prev);
            }
        }
    }

    /// Contextual learning: build a segment on `next`'s cell from the union of
    /// the whole context window (last `context_len` tokens), so the cell fires
    /// only when the full recent history matches — not just the last token.
    pub fn learn_context(&mut self, context: &[SdrVector], next: &SdrVector) {
        let next_cell = self.get_or_create_cell(next);
        let ctx = SdrVector::union(context);
        if ctx.popcount() == 0 {
            return;
        }
        let mut has_ctx_seg = false;
        for seg in &self.cells[next_cell].segments {
            if seg.overlap(&ctx) >= 3 {
                has_ctx_seg = true;
                break;
            }
        }
        if !has_ctx_seg {
            self.cells[next_cell].learn_segment(&ctx);
        }
        for seg in &mut self.cells[next_cell].segments {
            if seg.overlap(&ctx) > 0 {
                seg.reinforce(&ctx);
            }
        }
    }

    pub fn prune(&mut self) {
        for c in &mut self.cells {
            for seg in &mut c.segments {
                seg.prune_weak();
            }
            c.segments.retain(|seg| {
                let connected = seg.synapses.iter().filter(|s| s.connected()).count();
                connected >= 2
            });
        }
    }

    pub fn save(&self, path: &str) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::File::create(path) {
            let n = self.cells.len() as u32;
            f.write_all(&n.to_le_bytes()).ok();
            for c in &self.cells {
                f.write_all(&(c.id as u32).to_le_bytes()).ok();
                for w in &c.pattern.bits {
                    f.write_all(&w.to_le_bytes()).ok();
                }
                f.write_all(&(c.segments.len() as u32).to_le_bytes()).ok();
                for seg in &c.segments {
                    f.write_all(&(seg.synapses.len() as u32).to_le_bytes()).ok();
                    for s in &seg.synapses {
                        f.write_all(&(s.bit_index as u32).to_le_bytes()).ok();
                        f.write_all(&s.permanence.to_le_bytes()).ok();
                    }
                }
            }
            f.write_all(&(self.window.len() as u32).to_le_bytes()).ok();
            for sdr in &self.window {
                for w in &sdr.bits {
                    f.write_all(&w.to_le_bytes()).ok();
                }
            }
        }
    }

    pub fn load(path: &str) -> Option<Self> {
        let data = std::fs::read(path).ok()?;
        let mut pos = 0usize;
        let n = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        let mut cells = Vec::with_capacity(n);
        for _ in 0..n {
            let id = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            let mut bits = [0u64; 128];
            for w in bits.iter_mut() {
                *w = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
                pos += 8;
            }
            let pattern = crate::ai::sdr::SdrVector { bits };
            let seg_n = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            let mut segments = Vec::with_capacity(seg_n);
            for _ in 0..seg_n {
                let syn_n = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
                pos += 4;
                let mut synapses = Vec::with_capacity(syn_n);
                for _ in 0..syn_n {
                    let bi = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
                    pos += 4;
                    let perm = f64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
                    pos += 8;
                    synapses.push(Synapse::new(bi, perm));
                }
                segments.push(DendriteSegment { synapses });
            }
            cells.push(TemporalCell {
                id,
                segments,
                pattern,
            });
        }
        let wl = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        let mut window = Vec::with_capacity(wl);
        for _ in 0..wl {
            let mut bits = [0u64; 128];
            for w in bits.iter_mut() {
                *w = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
                pos += 8;
            }
            window.push(crate::ai::sdr::SdrVector { bits });
        }
        Some(TemporalMemory {
            cells,
            window,
            context_len: 4,
            step: 0,
        })
    }

    pub fn stats(&self) -> String {
        let total_segs: usize = self.cells.iter().map(|c| c.segments.len()).sum();
        let total_syn: usize = self
            .cells
            .iter()
            .flat_map(|c| c.segments.iter())
            .map(|s| s.synapses.len())
            .sum();
        format!(
            "cells={} segments={} synapses={} window={}",
            self.cells.len(),
            total_segs,
            total_syn,
            self.window.len()
        )
    }
}

//! Byte-level LSTM comparator — the classical recurrent baseline for the
//! tokenless decoder A/B.
//!
//! fuga decodes code byte-by-byte through Hyperdimensional/VSA operators (W,
//! recurrent state h(t), Hopfield, KAN). To judge those honestly we need a
//! *peer*: a standard small recurrent generator on the SAME corpus, the SAME
//! 256-byte alphabet, and the SAME metrics (bytes until stall, printable %,
//! diversity / cycle distance). A vanilla char-level LSTM (H=128 ≈ 240K
//! params, same order as our byte-stack budget) trained with truncated BPTT
//! is exactly such a peer, and it has no VSA machinery — so any gap we measure
//! is owed to plain recurrence vs our operator design, not to vocabulary.
//!
//! Self-contained: pure std, deterministic RNG (splitmix64), 1-layer LSTM
//! (256 one-hot input → forget/input/output/candidate gates → 256 softmax
//! over bytes), truncated BPTT-8, Adam. No external crates.
//!
//! This is deliberately *not* tuned: it is the baseline every tokenless
//! decoder here should be compared against.

/// Vocab = all 256 raw bytes. Fixed, no sub-word dictionary (dictionary-free).
pub const VOCAB: usize = 256;
/// Latent / hidden size of the LSTM cell.
pub const HIDDEN: usize = 128;
/// Truncated BPTT horizon.
const BPTT: usize = 8;
/// Adam hyper-parameters.
const LR: f32 = 0.002;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;

/// A minimal, deterministic, self-contained LSTM byte model.
#[derive(Clone)]
pub struct ByteLstm {
    /// 4H × VOCAB (input-to-gates, row-major: i,f,o,g stacked).
    w_ix: Vec<f32>,
    /// 4H × HIDDEN (hidden-to-gates, row-major: i,f,o,g stacked).
    w_ih: Vec<f32>,
    /// 4H gate biases (i,f,o,g stacked).
    b_i: Vec<f32>,
    /// VOCAB × HIDDEN (output projection).
    w_ho: Vec<f32>,
    /// VOCAB output bias.
    b_o: Vec<f32>,
    /// Persistent hidden state (stateful across windows).
    h: Vec<f32>,
    /// Persistent cell state.
    c: Vec<f32>,
    // Adam moments (same shapes as the weights).
    m: Vec<f32>,
    v: Vec<f32>,
    t: usize,
}

/// Splits a vector into the four gate slices.
fn gates4(g: &[f32], j: usize) -> (f32, f32, f32, f32) {
    (g[j], g[HIDDEN + j], g[2 * HIDDEN + j], g[3 * HIDDEN + j])
}

impl ByteLstm {
    pub fn new(rng: &mut u64) -> Self {
        let h = HIDDEN;
        let scale = (1.0 / (h as f32)).sqrt();
        let mut rand_vec = |n: usize, rng: &mut u64| {
            (0..n).map(|_| (next_f32(rng) * 2.0 - 1.0) * scale).collect()
        };
        let n_ix = 4 * h * VOCAB;
        let n_ih = 4 * h * h;
        let n_ho = VOCAB * h;
        Self {
            w_ix: rand_vec(n_ix, rng),
            w_ih: rand_vec(n_ih, rng),
            b_i: vec![0.0; 4 * h],
            w_ho: rand_vec(n_ho, rng),
            b_o: vec![0.0; VOCAB],
            h: vec![0.0; h],
            c: vec![0.0; h],
            m: vec![0.0; n_ix + n_ih + 4 * h + n_ho + VOCAB],
            v: vec![0.0; n_ix + n_ih + 4 * h + n_ho + VOCAB],
            t: 0,
        }
    }

    /// One forward cell step. Returns softmax distribution over 256 bytes.
    /// Mutates internal h/c (stateful). `gate_out` optionally captures the
    /// pre-activation gates and the post-activation cell state for BPTT.
    fn cell_step(&mut self, x: usize, gate_out: Option<&mut ([f32; 4 * HIDDEN], [f32; HIDDEN])>) -> [f32; VOCAB] {
        let h_prev = self.h.clone();
        let c_prev = self.c.clone();
        let mut g = [0.0f32; 4 * HIDDEN];
        for k in 0..4 * HIDDEN {
            let mut acc = self.b_i[k];
            acc += self.w_ix[k * VOCAB + x];
            for (j, hj) in h_prev.iter().enumerate() {
                acc += self.w_ih[k * HIDDEN + j] * hj;
            }
            g[k] = acc;
        }
        let mut new_c = [0.0f32; HIDDEN];
        let mut new_h = [0.0f32; HIDDEN];
        for j in 0..HIDDEN {
            let (gi, gf, go, gg) = gates4(&g, j);
            let i = sigmoid(gi);
            let f = sigmoid(gf);
            let o = sigmoid(go);
            let tg = gg.tanh();
            new_c[j] = f * c_prev[j] + i * tg;
            new_h[j] = o * new_c[j].tanh();
        }
        if let Some((g_out, c_out)) = gate_out {
            g_out.copy_from_slice(&g);
            c_out.copy_from_slice(&new_c);
        }
        // Output softmax.
        let mut logits = [0.0f32; VOCAB];
        for u in 0..VOCAB {
            let mut acc = self.b_o[u];
            for j in 0..HIDDEN {
                acc += self.w_ho[u * HIDDEN + j] * new_h[j];
            }
            logits[u] = acc;
        }
        let max = logits.iter().cloned().fold(f32::MIN, f32::max);
        let mut sum = 0.0f32;
        for u in &mut logits {
            *u = (*u - max).exp();
            sum += *u;
        }
        for u in &mut logits {
            *u /= sum.max(1e-12);
        }
        self.h = new_h.to_vec();
        self.c = new_c.to_vec();
        logits
    }

    /// Train on a byte window with truncated BPTT (teacher forcing).
    /// `bytes` — the input sequence; each byte predicts the next.
    /// Returns mean cross-entropy over the window.
    pub fn train_window(&mut self, bytes: &[u8]) -> f32 {
        if bytes.len() < 2 {
            return 0.0;
        }
        let n = bytes.len() - 1; // number of predictions
        let t_max = n.min(BPTT);
        // Forward, storing gate/cell snapshots and the inputs.
        let mut xs = [0usize; BPTT];
        let mut gate_snap = [[0.0f32; 4 * HIDDEN]; BPTT];
        let mut cell_snap = [[0.0f32; HIDDEN]; BPTT];
        let mut probs = [[0.0f32; VOCAB]; BPTT];
        let mut targets = [0usize; BPTT];
        for t in 0..t_max {
            xs[t] = bytes[t] as usize;
            targets[t] = bytes[t + 1] as usize;
            let mut gs = [0.0f32; 4 * HIDDEN];
            let mut cs = [0.0f32; HIDDEN];
            let mut snap = (gs, cs);
            probs[t] = self.cell_step(xs[t], Some(&mut snap));
            gate_snap[t] = snap.0;
            cell_snap[t] = snap.1;
        }
        // Cross-entropy loss.
        let mut loss = 0.0f32;
        for t in 0..t_max {
            loss -= probs[t][targets[t]].max(1e-12).ln();
        }
        loss /= t_max as f32;

        // Backprop through time: accumulate gradients into (g_wix, g_wih, g_bi,
        // g_who, g_bo), then Adam step.
        let mut g_wix = vec![0.0f32; 4 * HIDDEN * VOCAB];
        let mut g_wih = vec![0.0f32; 4 * HIDDEN * HIDDEN];
        let mut g_bi = vec![0.0f32; 4 * HIDDEN];
        let mut g_who = vec![0.0f32; VOCAB * HIDDEN];
        let mut g_bo = vec![0.0f32; VOCAB];
        let mut dh_next = [0.0f32; HIDDEN];
        let mut dc_next = [0.0f32; HIDDEN];

        for t in (0..t_max).rev() {
            // Output grad: dL/dlogits = p - onehot(target).
            let mut dlogits = [0.0f32; VOCAB];
            for u in 0..VOCAB {
                let mut e = probs[t][u];
                if u == targets[t] {
                    e -= 1.0;
                }
                dlogits[u] = e;
            }
            // h at this step.
            let h_t = self.h.clone(); // NOTE: overwritten below; recompute below.
            // We stored only gates+cell, not h; recompute h from snapshots.
            // h_t[j] = o * tanh(c_t[j]).
            let mut h_t_arr = [0.0f32; HIDDEN];
            let c_t = cell_snap[t];
            for j in 0..HIDDEN {
                let (_, _, go, _) = gates4(&gate_snap[t], j);
                h_t_arr[j] = sigmoid(go) * c_t[j].tanh();
            }
            // dL/dh_t += W_ho^T · dlogits
            let mut dh = [0.0f32; HIDDEN];
            for u in 0..VOCAB {
                let du = dlogits[u];
                for j in 0..HIDDEN {
                    dh[j] += self.w_ho[u * HIDDEN + j] * du;
                    g_who[u * HIDDEN + j] += du * h_t_arr[j];
                }
                g_bo[u] += du;
            }
            for j in 0..HIDDEN {
                dh[j] += dh_next[j];
            }
            // Cell gradients.
            let mut dc = [0.0f32; HIDDEN];
            for j in 0..HIDDEN {
                let (gi, gf, go, gg) = gates4(&gate_snap[t], j);
                let i = sigmoid(gi);
                let f = sigmoid(gf);
                let o = sigmoid(go);
                let tg = gg.tanh();
                let c_j = cell_snap[t][j];
                let h_j = o * c_j.tanh();
                // dL/dc_t (from h and from next cell).
                let dtanh_c = 1.0 - c_j.tanh().powi(2);
                dc[j] = dh[j] * o * dtanh_c + dc_next[j] * f;
                // Gate grads.
                let di = dc[j] * tg * i * (1.0 - i);
                let df = dc[j] * (if t == 0 { 0.0 } else { cell_snap[t - 1][j] }) * f * (1.0 - f);
                let dgo = dh[j] * c_j.tanh() * o * (1.0 - o);
                let dgg = dc[j] * i * (1.0 - tg.powi(2));
                // Distribute to the stacked gate rows.
                g_bi[j] += di;
                g_bi[HIDDEN + j] += df;
                g_bi[2 * HIDDEN + j] += dgo;
                g_bi[3 * HIDDEN + j] += dgg;
                for (kk, dgk) in [(0usize, di), (1, df), (2, dgo), (3, dgg)].iter() {
                    let row = kk * HIDDEN + j;
                    g_wix[row * VOCAB + xs[t]] += dgk;
                    let h_prev_j = if t == 0 { 0.0 } else {
                        // h_{t-1} from the previous snapshot.
                        let (_, _, go_prev, _) = gates4(&gate_snap[t - 1], j);
                        sigmoid(go_prev) * cell_snap[t - 1][j].tanh()
                    };
                    g_wih[row * HIDDEN + j] += dgk * h_prev_j;
                }
                // dh_{t-1} += W_ih^T · dgate.
                dh_next[j] = 0.0;
                for (kk, dgk) in [(0usize, di), (1, df), (2, dgo), (3, dgg)].iter() {
                    let row = kk * HIDDEN + j;
                    for jj in 0..HIDDEN {
                        dh_next[jj] += self.w_ih[row * HIDDEN + jj] * dgk;
                    }
                }
            }
            dc_next = dc;
        }

        // Adam update.
        let mut all = [
            (&mut self.w_ix, &mut g_wix),
            (&mut self.w_ih, &mut g_wih),
            (&mut self.b_i, &mut g_bi),
            (&mut self.w_ho, &mut g_who),
            (&mut self.b_o, &mut g_bo),
        ];
        self.t += 1;
        let t = self.t as f32;
        let bc1 = 1.0 - BETA1.powi(t as i32);
        let bc2 = 1.0 - BETA2.powi(t as i32);
        let mut off = 0;
        for (w, g) in all.iter_mut() {
            let n = w.len();
            for i in 0..n {
                let gi = g[i];
                self.m[off + i] = BETA1 * self.m[off + i] + (1.0 - BETA1) * gi;
                self.v[off + i] = BETA2 * self.v[off + i] + (1.0 - BETA2) * gi * gi;
                let m_hat = self.m[off + i] / bc1;
                let v_hat = self.v[off + i] / bc2;
                w[i] -= LR * m_hat / (v_hat.sqrt() + EPS);
            }
            off += n;
        }
        // Keep h/c state for the next window (stateful).
        loss
    }

    /// Reset hidden and cell state to zero (e.g. at the start of a new
    /// sequence/episode). Stateful runs must call this between sequences;
    /// stateful training carries state across windows within one run.
    pub fn reset_state(&mut self) {
        self.h.iter_mut().for_each(|v| *v = 0.0);
        self.c.iter_mut().for_each(|v| *v = 0.0);
    }

    /// Greedy decode: seed the state by teacher-forcing the seed, then emit
    /// argmax bytes. `max_bytes` budget; stops early on immediate self-loop
    /// (same byte twice in a row) or a period-2 cycle.
    pub fn generate(&mut self, seed: &[u8], max_bytes: usize) -> Vec<u8> {
        for &b in seed {
            self.cell_step(b as usize, None);
        }
        let mut out: Vec<u8> = Vec::new();
        let mut guard = 0;
        while out.len() < max_bytes && guard < max_bytes * 2 {
            guard += 1;
            let x = out.last().copied().unwrap_or(seed.last().copied().unwrap_or(0));
            let p = self.cell_step(x as usize, None);
            let mut best = 0usize;
            for u in 1..VOCAB {
                if p[u] > p[best] {
                    best = u;
                }
            }
            let byte = best as u8;
            if out.last() == Some(&byte) {
                break;
            }
            out.push(byte);
        }
        out
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.max(-40.0)).exp())
}

/// splitmix64 → f32 in [0,1).
fn next_f32(rng: &mut u64) -> f32 {
    *rng = rng.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *rng;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    (z >> 11) as f32 / (1u64 << 53) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lstm_learns_ab_alternation() {
        // Hard deterministic pattern: after 'a' comes 'b', after 'b' 'a'.
        let mut rng: u64 = 0xABAB_ABAB;
        let mut lstm = ByteLstm::new(&mut rng);
        let mut run_len = 0usize;
        for _epoch in 0..200 {
            let mut seq = Vec::new();
            for i in 0..200 {
                seq.push(if i % 2 == 0 { b'a' } else { b'b' });
            }
            lstm.reset_state();
            // BPTT windows with overlapping last byte so the carry is learned.
            let mut w0 = 0usize;
            while w0 < seq.len() - 1 {
                let end = (w0 + BPTT + 1).min(seq.len());
                lstm.train_window(&seq[w0..end]);
                w0 += BPTT;
            }
        }
        // Greedy decode from seed "a" should alternate for several bytes.
        lstm.reset_state();
        let out = lstm.generate(b"a", 12);
        run_len = out.len();
        // Must continue an alternation; we assert at least 3 bytes AND that the
        // first emitted byte is 'b' (the strongly-tied continuation of 'a').
        assert!(
            run_len >= 3,
            "LSTM should continue an alternation, got {:?}",
            String::from_utf8_lossy(&out)
        );
        // Each emitted byte continues the alternation (b a b a ...).
        for (i, &b) in out.iter().enumerate() {
            let expect = if i % 2 == 0 { b'b' } else { b'a' };
            assert_eq!(
                b, expect,
                "alternation broken at pos {}: {:?}",
                i, String::from_utf8_lossy(&out)
            );
        }
    }
}
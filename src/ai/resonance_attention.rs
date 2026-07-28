use crate::core::hypervector::Hypervector;
use crate::core::wave_cube::WaveCube;
use crate::weaver::super_token::SuperToken;
use crate::gpu;

pub struct ResonanceAttention {
    dim: usize,
}

#[derive(Clone)]
pub struct AttentionCell {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub w: usize,
    pub v: usize,
    pub score: f64,
}

impl ResonanceAttention {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn resonance_score(&self, q: &Hypervector, k: &Hypervector) -> f64 {
        q.similarity(k)
    }

    pub fn attention_map<const N: usize, const S: usize>(
        &self, st: &SuperToken, cube: &WaveCube<N, S>, region: Option<&[(usize, usize); N]>
    ) -> Vec<AttentionCell> {
        let default_ranges = [(0, S); N];
        let ranges = region.unwrap_or(&default_ranges);

        let mut results = Vec::new();
        self.enumerate_cells(cube, ranges, &mut |coords: &[usize; N], cell_hv: &Hypervector| {
            let score = self.resonance_score(&st.vector, cell_hv);
            if score > 0.55 {
                results.push(AttentionCell {
                    x: coords.get(0).copied().unwrap_or(0),
                    y: coords.get(1).copied().unwrap_or(0),
                    z: coords.get(2).copied().unwrap_or(0),
                    w: coords.get(3).copied().unwrap_or(0),
                    v: coords.get(4).copied().unwrap_or(0),
                    score,
                });
            }
        });
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(16);
        results
    }

    fn enumerate_cells<const N: usize, const S: usize, F>(
        &self, cube: &WaveCube<N, S>, ranges: &[(usize, usize); N], mut f: F,
    )
    where
        F: FnMut(&[usize; N], &Hypervector),
    {
        for idx in 0..WaveCube::<N, S>::TOTAL_CELLS {
            let coords = cube.coords_from_idx(idx);
            let in_range = coords.iter().zip(ranges.iter()).all(|(&c, &(lo, hi))| c >= lo && c < hi);
            if !in_range { continue; }
            let cell_hv = cube.cell_at(&coords);
            f(&coords, &cell_hv);
        }
    }

    pub fn beam_attention<const N: usize, const S: usize>(
        &self, st: &SuperToken, cube: &WaveCube<N, S>, beam: usize
    ) -> Vec<AttentionCell> {
        let total = WaveCube::<N, S>::TOTAL_CELLS;
        let sigma = 0.5 / (cube.dim as f64).sqrt();
        let threshold = 0.5 + sigma.max(0.005) * 1.5;

        if gpu::is_gpu_available() {
            let gpu_scores = gpu::gpu_resonance_scan(&st.vector, &cube.cube);
            if let Some(scores) = gpu_scores {
                let mut cells: Vec<AttentionCell> = scores.iter().enumerate()
                    .filter(|&(_, s)| *s as f64 > threshold)
                    .map(|(idx, &s)| {
                        let coords = cube.coords_from_idx(idx);
                        AttentionCell {
                            x: coords[0],
                            y: coords.get(1).copied().unwrap_or(0),
                            z: coords.get(2).copied().unwrap_or(0),
                            w: coords.get(3).copied().unwrap_or(0),
                            v: coords.get(4).copied().unwrap_or(0),
                            score: s as f64,
                        }
                    })
                    .collect();
                cells.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                cells.truncate(beam);
                return cells;
            }
        }

        let step = (total + beam - 1) / beam;

        let chunk_results: Vec<AttentionCell> = (0..total).step_by(step)
            .flat_map(|start| {
                let end = (start + step).min(total);
                let mut best: Option<AttentionCell> = None;
                for idx in start..end {
                    let coords = cube.coords_from_idx(idx);
                    let cell_hv = cube.cell_at(&coords);
                    let score = st.vector.similarity(&cell_hv);
                    if score > threshold {
                        let cell = AttentionCell {
                            x: coords[0],
                            y: coords.get(1).copied().unwrap_or(0),
                            z: coords.get(2).copied().unwrap_or(0),
                            w: coords.get(3).copied().unwrap_or(0),
                            v: coords.get(4).copied().unwrap_or(0),
                            score,
                        };
                        match &best {
                            Some(b) if score > b.score => best = Some(cell),
                            None => best = Some(cell),
                            _ => {}
                        }
                    }
                }
                best
            })
            .collect();

        let mut sorted = chunk_results;
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        sorted.truncate(beam);
        sorted
    }

    pub fn write_attention<const N: usize, const S: usize>(
        &self, cube: &mut WaveCube<N, S>, x: usize, y: usize, z: usize, w: usize, v: usize, hv: &Hypervector,
    ) {
        let mut coords = [0; N];
        coords[0] = x;
        if N > 1 { coords[1] = y; }
        if N > 2 { coords[2] = z; }
        if N > 3 { coords[3] = w; }
        if N > 4 { coords[4] = v; }
        let existing = cube.cell_at(&coords);
        let bound = existing.bind(hv);
        cube.write_at(&coords, &bound);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resonance_score_identical() {
        let attn = ResonanceAttention::new(8192);
        let hv = Hypervector::random(8192);
        let score = attn.resonance_score(&hv, &hv);
        assert!((score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_resonance_score_random() {
        let attn = ResonanceAttention::new(8192);
        let a = Hypervector::random(8192);
        let b = Hypervector::random(8192);
        let score = attn.resonance_score(&a, &b);
        assert!(score > 0.3 && score < 0.7);
    }

    #[test]
    fn test_attention_map_empty_region() {
        let attn = ResonanceAttention::new(1024);
        let cube = WaveCube::<3, 4>::new(1024);
        let st = SuperToken::new(Hypervector::random(1024), 0);
        let map = attn.attention_map(&st, &cube, Some(&[(0, 0); 3]));
        assert!(map.is_empty());
    }

    #[test]
    fn test_write_attention_changes_cube() {
        let attn = ResonanceAttention::new(1024);
        let mut cube = WaveCube::<3, 4>::new(1024);
        let before = cube.cell(0, 0, 0);
        let hv = Hypervector::random(1024);
        attn.write_attention(&mut cube, 0, 0, 0, 0, 0, &hv);
        let after = cube.cell(0, 0, 0);
        assert_ne!(before.similarity(&after), 1.0);
    }
}

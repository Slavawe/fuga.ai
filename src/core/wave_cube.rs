use super::hypervector::Hypervector;
use rand::RngCore;
use std::io::Read;

#[derive(Clone)]
pub struct WaveCube<const N: usize, const S: usize> {
    pub dim: usize,
    pub cube: Vec<Hypervector>,
}

impl<const N: usize, const S: usize> WaveCube<N, S> {
    pub const TOTAL_CELLS: usize = S.pow(N as u32);

    pub fn new(dim: usize) -> Self {
        let total = Self::TOTAL_CELLS;
        let mut rng = rand::thread_rng();
        let mut cube = Vec::with_capacity(total);
        for _ in 0..total {
            let mut hv = Hypervector::new(dim);
            for w in &mut hv.words {
                *w = rng.next_u64();
            }
            let rem = dim % 64;
            if rem != 0 {
                let last = hv.words.len() - 1;
                hv.words[last] &= (1u64 << rem) - 1;
            }
            cube.push(hv);
        }
        WaveCube { dim, cube }
    }

    pub fn idx(&self, coords: &[usize; N]) -> usize {
        let mut idx = 0;
        let mut i = 0;
        while i < N {
            idx = idx * S + coords[i];
            i += 1;
        }
        idx
    }

    pub fn coords_from_idx(&self, mut idx: usize) -> [usize; N] {
        let mut coords = [0; N];
        let mut i = N;
        while i > 0 {
            i -= 1;
            coords[i] = idx % S;
            idx /= S;
        }
        coords
    }

    pub fn cell(&self, x: usize, y: usize, z: usize) -> Hypervector {
        let mut coords = [0; N];
        coords[0] = x;
        coords[1] = y;
        if N >= 3 {
            coords[2] = z;
        }
        self.cube[self.idx(&coords)].clone()
    }

    pub fn write_cell(&mut self, x: usize, y: usize, z: usize, hv: &Hypervector) {
        let mut coords = [0; N];
        coords[0] = x;
        coords[1] = y;
        if N >= 3 {
            coords[2] = z;
        }
        let i = self.idx(&coords);
        self.cube[i] = hv.clone();
    }

    pub fn cell_at(&self, coords: &[usize; N]) -> Hypervector {
        self.cube[self.idx(coords)].clone()
    }

    pub fn write_at(&mut self, coords: &[usize; N], hv: &Hypervector) {
        let i = self.idx(coords);
        self.cube[i] = hv.clone();
    }

    pub fn wave_flow(&mut self, axis: usize, shift: isize) {
        if shift == 0 {
            return;
        }
        let s = ((shift % S as isize) + S as isize) as usize % S;
        if s == 0 {
            return;
        }
        let total = Self::TOTAL_CELLS;
        let block = {
            let mut b = 1;
            let mut d = N;
            while d > axis + 1 {
                b *= S;
                d -= 1;
            }
            b
        };
        let stride = block * S;
        let shift_blocks = s * block;

        let mut new_cube = self.cube.clone();
        let mut base = 0;
        while base < total {
            for offset in 0..block {
                let src = base + ((offset + shift_blocks) % stride);
                let dst = base + offset;
                new_cube[dst] = self.cube[src].clone();
            }
            base += stride;
        }
        self.cube = new_cube;
    }

    pub fn wave_flow_x(&mut self, shift: isize) {
        self.wave_flow(0, shift)
    }
    pub fn wave_flow_y(&mut self, shift: isize) {
        self.wave_flow(1, shift)
    }
    pub fn wave_flow_z(&mut self, shift: isize) {
        self.wave_flow(2, shift)
    }
    pub fn wave_flow_w(&mut self, shift: isize) {
        self.wave_flow(3, shift)
    }

    pub fn absorb_from_triangle(
        &mut self,
        triangle_syntax: &Hypervector,
        triangle_semantics: &Hypervector,
        triangle_chaos: &Hypervector,
    ) {
        let ms = S / 2;
        self.write_cell(0, 0, 0, &self.cell(0, 0, 0).bind(triangle_syntax));
        self.write_cell(
            S - 1,
            ms,
            ms,
            &self.cell(S - 1, ms, ms).bind(triangle_semantics),
        );
        self.write_cell(
            ms,
            ms,
            S - 1,
            &self.cell(ms, ms, S - 1).bind(triangle_chaos),
        );
        self.wave_flow_x(1);
        self.wave_flow_y(1);
        self.wave_flow_z(1);
    }

    pub fn global_entropy(&self) -> f64 {
        let total_bits = Self::TOTAL_CELLS * self.dim;
        let ones: u64 = self
            .cube
            .iter()
            .map(|hv| hv.entropy() * hv.dim as f64)
            .sum::<f64>() as u64;
        ones as f64 / total_bits as f64
    }

    pub fn coherence(&self) -> f64 {
        if N < 3 {
            return 0.0;
        }
        let mut sum = 0.0;
        let mut count = 0;
        let mut i = 0;
        while i < S {
            let a = self.cell(i, i, i);
            let b = self.cell(S - 1 - i, S - 1 - i, S - 1 - i);
            sum += a.similarity(&b);
            count += 1;
            i += 1;
        }
        sum / count as f64
    }

    pub fn save_bin(&self, path: &str) -> Result<(), String> {
        use std::io::Write;
        let header: Vec<u32> = vec![1u32, S as u32, N as u32, self.dim as u32];
        let mut f =
            std::fs::File::create(path).map_err(|e| format!("Failed to create {}: {}", path, e))?;
        let header_bytes: Vec<u8> = header.iter().flat_map(|v| v.to_le_bytes()).collect();
        f.write_all(&header_bytes)
            .map_err(|e| format!("Failed to write header: {}", e))?;
        for hv in &self.cube {
            let word_bytes: Vec<u8> = hv.words.iter().flat_map(|w| w.to_le_bytes()).collect();
            f.write_all(&word_bytes)
                .map_err(|e| format!("Failed to write cell: {}", e))?;
        }
        Ok(())
    }

    pub fn load_bin(path: &str) -> Result<Self, String> {
        let file =
            std::fs::File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| format!("Failed to mmap {}: {}", path, e))?;
        let data = &mmap[..];

        let v0 = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let v1 = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;

        let (file_side, file_ndim, dim) = if v0 == 1 {
            let side = v1;
            let ndim = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
            let d = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
            (side, ndim, d)
        } else {
            let d = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
            (v0, 3, d)
        };

        if file_side != S {
            return Err(format!(
                "Cube side mismatch: file has S={}, expected S={}",
                file_side, S
            ));
        }
        if file_ndim != N {
            return Err(format!(
                "Cube ndim mismatch: file has N={}, expected N={}",
                file_ndim, N
            ));
        }

        let wc = (dim + 63) / 64;
        let data_len = data.len() - 16;
        let cell_count = data_len / (wc * 8);
        if cell_count != Self::TOTAL_CELLS {
            return Err(format!(
                "Cube data size mismatch: {} cells, expected {}",
                cell_count,
                Self::TOTAL_CELLS
            ));
        }

        let mut cube = Vec::with_capacity(cell_count);
        let mut off = 16;
        for _ in 0..cell_count {
            let mut words = vec![0u64; wc];
            for i in 0..wc {
                words[i] =
                    u64::from_le_bytes(data[off + i * 8..off + (i + 1) * 8].try_into().unwrap());
            }
            off += wc * 8;
            cube.push(Hypervector { dim, words });
        }

        Ok(WaveCube { dim, cube })
    }
}

impl<const S: usize> WaveCube<3, S> {
    pub fn idx_3d(&self, x: usize, y: usize, z: usize) -> usize {
        (x * S + y) * S + z
    }
}

pub fn peek_cube_header(path: &str) -> Result<(usize, usize, usize), String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
    let mut buf = [0u8; 16];
    f.read_exact(&mut buf)
        .map_err(|e| format!("Failed to read header: {}", e))?;
    let v0 = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let v1 = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
    if v0 == 1 {
        let side_len = v1;
        let ndim = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
        Ok((ndim, side_len, dim))
    } else {
        let dim = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;
        Ok((3, v0, dim))
    }
}
// SIMD-optimized popcount helpers — unrolled 4-wide for cache locality
fn popcount_chunks(words: &[u64]) -> u64 {
    let mut total: u64 = 0;
    let chunks = words.chunks_exact(4);
    let remainder = chunks.remainder();
    for chunk in chunks {
        total += chunk[0].count_ones() as u64
            + chunk[1].count_ones() as u64
            + chunk[2].count_ones() as u64
            + chunk[3].count_ones() as u64;
    }
    for &w in remainder {
        total += w.count_ones() as u64;
    }
    total
}

fn popcount_xor_pair(a: &[u64], b: &[u64], n: usize) -> u64 {
    let limit = a.len().min(b.len()).min(n);
    let (a_main, a_rem) = a[..limit].split_at(limit / 4 * 4);
    let (b_main, b_rem) = b[..limit].split_at(limit / 4 * 4);
    let mut total: u64 = 0;
    for i in (0..a_main.len()).step_by(4) {
        total += (a_main[i] ^ b_main[i]).count_ones() as u64
            + (a_main[i + 1] ^ b_main[i + 1]).count_ones() as u64
            + (a_main[i + 2] ^ b_main[i + 2]).count_ones() as u64
            + (a_main[i + 3] ^ b_main[i + 3]).count_ones() as u64;
    }
    for i in 0..a_rem.len() {
        total += (a_rem[i] ^ b_rem[i]).count_ones() as u64;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    type Cube34 = WaveCube<3, 4>;

    fn make_cube() -> Cube34 {
        Cube34::new(1024)
    }

    #[test]
    fn test_creation() {
        let cube = make_cube();
        assert_eq!(cube.cube.len(), 64);
        let ent = cube.global_entropy();
        assert!(ent > 0.45 && ent < 0.55);
    }

    #[test]
    fn test_4d_creation() {
        let cube = WaveCube::<4, 4>::new(1024);
        assert_eq!(cube.cube.len(), 256);
        let ent = cube.global_entropy();
        assert!(ent > 0.45 && ent < 0.55);
    }

    #[test]
    fn test_wave_flow_preserves_dimensions() {
        let mut cube: Cube34 = Cube34::new(512);
        cube.wave_flow_x(2);
        assert_eq!(cube.cube.len(), 64);
    }

    #[test]
    fn test_4d_wave_flow() {
        let mut cube = WaveCube::<4, 4>::new(512);
        cube.wave_flow(3, 1);
        assert_eq!(cube.cube.len(), 256);
    }

    #[test]
    fn test_absorb_from_triangle() {
        let mut cube: Cube34 = Cube34::new(1024);
        let t_syntax = Hypervector::random(1024);
        let t_semantics = Hypervector::random(1024);
        let t_chaos = Hypervector::random(1024);
        let entropy_before = cube.global_entropy();
        cube.absorb_from_triangle(&t_syntax, &t_semantics, &t_chaos);
        let entropy_after = cube.global_entropy();
        assert_ne!(entropy_before, entropy_after);
    }

    #[test]
    fn test_save_load_bin_roundtrip() {
        let cube = WaveCube::<4, 4>::new(256);
        let path = "/tmp/test_cube.bin";
        cube.save_bin(path).unwrap();
        let loaded = WaveCube::<4, 4>::load_bin(path).unwrap();
        assert_eq!(loaded.dim, cube.dim);
        assert_eq!(loaded.cube.len(), cube.cube.len());
        for i in 0..cube.cube.len() {
            let sim = cube.cube[i].similarity(&loaded.cube[i]);
            assert!(sim > 0.99, "Cell {} similarity: {}", i, sim);
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_save_load_3d_roundtrip() {
        let cube = WaveCube::<3, 4>::new(256);
        let path = "/tmp/test_cube_3d.bin";
        cube.save_bin(path).unwrap();
        let loaded = WaveCube::<3, 4>::load_bin(path).unwrap();
        assert_eq!(loaded.cube.len(), 64);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_cell_write_read() {
        let mut cube = make_cube();
        let hv = Hypervector::random(1024);
        cube.write_cell(1, 2, 3, &hv);
        let read = cube.cell(1, 2, 3);
        assert!(hv.similarity(&read) > 0.99);
    }

    #[test]
    fn test_4d_cell_write_read() {
        let mut cube = WaveCube::<4, 4>::new(1024);
        let hv = Hypervector::random(1024);
        cube.write_at(&[1, 2, 3, 1], &hv);
        let read = cube.cell_at(&[1, 2, 3, 1]);
        assert!(hv.similarity(&read) > 0.99);
    }

    #[test]
    fn test_coherence() {
        let cube = make_cube();
        let coh = cube.coherence();
        assert!(coh >= 0.0 && coh <= 1.0);
    }
}

use super::hypervector::Hypervector;
use super::wave_cube::WaveCube;
use memmap2::Mmap;
use std::fs::File;

pub struct MappedCube {
    pub side_len: usize,
    pub ndim: usize,
    pub dim: usize,
    word_count: usize,
    _mmap: Mmap,
}

impl MappedCube {
    pub fn open(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|e| format!("Failed to mmap {}: {}", path, e))?;

        if mmap.len() < 16 {
            return Err("File too small for header".into());
        }

        let read_u32 = |off: usize| -> u32 {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&mmap[off..off + 4]);
            u32::from_le_bytes(buf)
        };

        let v0 = read_u32(0) as usize;
        let v1 = read_u32(4) as usize;

        let (side_len, ndim, dim) = if v0 == 1 {
            let ndim = v1;
            let dim = read_u32(8) as usize;
            (v1, ndim, dim)
        } else {
            let dim = read_u32(12) as usize;
            (v0, 3, dim)
        };

        let wc = (dim + 63) / 64;
        let data_len = mmap.len() - 16;
        let cell_count = data_len / (wc * 8);
        let expected = side_len.pow(ndim as u32);
        if cell_count != expected {
            return Err(format!(
                "MappedCube: {} cells, expected {}",
                cell_count, expected
            ));
        }

        Ok(MappedCube {
            side_len,
            ndim,
            dim,
            word_count: wc,
            _mmap: mmap,
        })
    }

    fn cell_offset(&self, coords: &[usize]) -> usize {
        let mut idx = 0;
        for &c in coords.iter() {
            idx = idx * self.side_len + c;
        }
        16 + idx * self.word_count * 8
    }

    pub fn cell(&self, x: usize, y: usize, z: usize) -> Hypervector {
        let off = self.cell_offset(&[x, y, z]);
        let mut words = vec![0u64; self.word_count];
        for i in 0..self.word_count {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&self._mmap[off + i * 8..off + (i + 1) * 8]);
            words[i] = u64::from_le_bytes(buf);
        }
        Hypervector {
            dim: self.dim,
            words,
        }
    }

    pub fn cell_at(&self, coords: &[usize]) -> Hypervector {
        let off = self.cell_offset(coords);
        let mut words = vec![0u64; self.word_count];
        for i in 0..self.word_count {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&self._mmap[off + i * 8..off + (i + 1) * 8]);
            words[i] = u64::from_le_bytes(buf);
        }
        Hypervector {
            dim: self.dim,
            words,
        }
    }

    pub fn to_wave_cube<const N: usize, const S: usize>(&self) -> WaveCube<N, S> {
        assert_eq!(self.ndim, N, "MappedCube ndim {} != N {}", self.ndim, N);
        assert_eq!(
            self.side_len, S,
            "MappedCube side_len {} != S {}",
            self.side_len, S
        );
        let total = self.side_len.pow(self.ndim as u32);
        let mut cube = Vec::with_capacity(total);
        for idx in 0..total {
            let mut coords = vec![0; self.ndim];
            let mut tmp = idx;
            for i in (0..self.ndim).rev() {
                coords[i] = tmp % self.side_len;
                tmp /= self.side_len;
            }
            cube.push(self.cell_at(&coords));
        }
        WaveCube {
            dim: self.dim,
            cube,
        }
    }
}

pub struct MappedMemory {
    pub dim: usize,
    pub num_entries: usize,
    word_count: usize,
    _mmap: Mmap,
    offsets: Vec<usize>,
}

impl MappedMemory {
    pub fn open(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
        let mmap =
            unsafe { Mmap::map(&file) }.map_err(|e| format!("Failed to mmap {}: {}", path, e))?;

        if mmap.len() < 4 {
            return Err("File too small for header".into());
        }

        let mut count_buf = [0u8; 4];
        count_buf.copy_from_slice(&mmap[0..4]);
        let num_entries = u32::from_le_bytes(count_buf) as usize;

        let mut offsets = Vec::with_capacity(num_entries);
        let mut pos = 4;

        for _ in 0..num_entries {
            offsets.push(pos);
            if pos + 4 > mmap.len() {
                return Err("Truncated memory file: dim".into());
            }
            let mut dim_buf = [0u8; 4];
            dim_buf.copy_from_slice(&mmap[pos..pos + 4]);
            let dim = u32::from_le_bytes(dim_buf) as usize;
            let wc = (dim + 63) / 64;
            pos += 4 + wc * 8;

            for _ in 0..3 {
                if pos + 4 > mmap.len() {
                    return Err("Truncated memory file: string len".into());
                }
                let mut len_buf = [0u8; 4];
                len_buf.copy_from_slice(&mmap[pos..pos + 4]);
                let slen = u32::from_le_bytes(len_buf) as usize;
                pos += 4 + slen;
            }
        }

        let dim = {
            let mut dim_buf = [0u8; 4];
            dim_buf.copy_from_slice(&mmap[4..8]);
            u32::from_le_bytes(dim_buf) as usize
        };

        Ok(MappedMemory {
            dim,
            num_entries,
            word_count: (dim + 63) / 64,
            _mmap: mmap,
            offsets,
        })
    }

    pub fn entry_vector(&self, idx: usize) -> Option<Hypervector> {
        if idx >= self.num_entries {
            return None;
        }
        let pos = self.offsets[idx];
        let mmap = &self._mmap;

        let mut dim_buf = [0u8; 4];
        dim_buf.copy_from_slice(&mmap[pos..pos + 4]);
        let dim = u32::from_le_bytes(dim_buf) as usize;
        let wc = (dim + 63) / 64;

        let mut words = vec![0u64; wc];
        for i in 0..wc {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&mmap[pos + 4 + i * 8..pos + 4 + (i + 1) * 8]);
            words[i] = u64::from_le_bytes(buf);
        }
        Some(Hypervector { dim, words })
    }

    pub fn search(&self, query: &Hypervector, top_k: usize) -> Vec<(usize, f64)> {
        let mut scores: Vec<(usize, f64)> = (0..self.num_entries)
            .filter_map(|i| {
                let hv = self.entry_vector(i)?;
                let sim = query.similarity(&hv);
                if sim > 0.55 { Some((i, sim)) } else { None }
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores.truncate(top_k);
        scores
    }
}

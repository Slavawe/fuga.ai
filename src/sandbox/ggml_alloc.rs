use std::cmp::Ordering;
use std::mem;
use std::ptr::NonNull;

const MAX_FREE_BLOCKS: usize = 256;
const MAX_CHUNKS: usize = 16;

#[derive(Clone, Copy, Debug)]
pub struct FreeBlock {
    offset: usize,
    size: usize,
}

#[derive(Clone, Debug)]
pub struct Chunk {
    free_blocks: Vec<FreeBlock>,
    max_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferAddress {
    pub chunk: i32,
    pub offset: usize,
}

impl BufferAddress {
    pub const INVALID: Self = BufferAddress { chunk: -1, offset: usize::MAX };
}

pub struct DynTallocr {
    alignment: usize,
    max_chunk_size: usize,
    chunks: Vec<Chunk>,
}

impl Chunk {
    fn new(min_size: usize, max_chunk_size: usize, is_last: bool) -> Self {
        let block_size = if is_last { usize::MAX / 2 } else { min_size.max(max_chunk_size) };
        Chunk {
            free_blocks: vec![FreeBlock { offset: 0, size: block_size }],
            max_size: 0,
        }
    }

    fn insert_block(&mut self, offset: usize, size: usize) {
        assert!(self.free_blocks.len() < MAX_FREE_BLOCKS);
        let pos = self.free_blocks.partition_point(|b| b.offset < offset);
        self.free_blocks.insert(pos, FreeBlock { offset, size });
    }

    fn remove_block(&mut self, idx: usize) {
        self.free_blocks.remove(idx);
    }

    fn best_fit_position(&self, size: usize) -> Option<(usize, usize)> {
        let mut best_idx = None;
        let mut best_size = usize::MAX;
        for (i, block) in self.free_blocks.iter().enumerate().take(self.free_blocks.len().saturating_sub(1)) {
            if block.size >= size && block.size < best_size {
                best_idx = Some(i);
                best_size = block.size;
            }
        }
        best_idx.map(|i| (i, best_size))
    }
}

impl DynTallocr {
    pub fn new(alignment: usize, max_buffer_size: usize) -> Self {
        DynTallocr {
            alignment,
            max_chunk_size: max_buffer_size.min(usize::MAX / 2),
            chunks: Vec::new(),
        }
    }

    fn aligned_offset(&self, offset: usize) -> usize {
        let align = self.alignment;
        assert!(align.is_power_of_two());
        let mask = align - 1;
        (offset + mask) & !mask
    }

    pub fn new_chunk(&mut self, min_size: usize) -> Option<usize> {
        if self.chunks.len() >= MAX_CHUNKS {
            return None;
        }
        let is_last = self.chunks.len() == MAX_CHUNKS - 1;
        self.chunks.push(Chunk::new(min_size, self.max_chunk_size, is_last));
        Some(self.chunks.len() - 1)
    }

    pub fn alloc(&mut self, size: usize) -> Option<BufferAddress> {
        let size = self.aligned_offset(size);
        let mut best_chunk = None;
        let mut best_block = None;
        let mut best_size = usize::MAX;
        let mut max_avail = 0;

        for (c, chunk) in self.chunks.iter().enumerate() {
            max_avail = max_avail.max(
                chunk.free_blocks.iter().map(|b| b.size).max().unwrap_or(0)
            );
            if let Some((bi, bs)) = chunk.best_fit_position(size) {
                if bs < best_size {
                    best_chunk = Some(c);
                    best_block = Some(bi);
                    best_size = bs;
                }
            }
        }

        if best_block.is_none() {
            let mut best_reuse: i64 = i64::MIN;
            for (c, chunk) in self.chunks.iter().enumerate() {
                if let Some(block) = chunk.free_blocks.last() {
                    max_avail = max_avail.max(block.size);
                    if block.size >= size {
                        let reuse = chunk.max_size as i64 - block.offset as i64 - size as i64;
                        let better = if best_reuse < 0 { reuse > best_reuse } else { reuse >= 0 && reuse < best_reuse };
                        if better {
                            best_chunk = Some(c);
                            best_block = Some(chunk.free_blocks.len() - 1);
                            best_reuse = reuse;
                        }
                    }
                }
            }
        }

        let best_chunk = best_chunk.or_else(|| self.new_chunk(size));
        let best_chunk = best_chunk?;
        let best_block = best_block.unwrap_or(0);

        let chunk = &mut self.chunks[best_chunk];
        let block = &mut chunk.free_blocks[best_block];
        let addr = BufferAddress { chunk: best_chunk as i32, offset: block.offset };
        block.offset += size;
        block.size -= size;
        if block.size == 0 {
            chunk.remove_block(best_block);
        }
        chunk.max_size = chunk.max_size.max(addr.offset + size);
        Some(addr)
    }

    pub fn free_bytes(&mut self, addr: BufferAddress, mut size: usize) {
        size = self.aligned_offset(size);
        let chunk = &mut self.chunks[addr.chunk as usize];
        for i in 0..chunk.free_blocks.len() {
            let block = &chunk.free_blocks[i];
            if block.offset + block.size == addr.offset {
                chunk.free_blocks[i].size += size;
                if i + 1 < chunk.free_blocks.len() {
                    let next = chunk.free_blocks[i + 1];
                    if chunk.free_blocks[i].offset + chunk.free_blocks[i].size == next.offset {
                        chunk.free_blocks[i].size += next.size;
                        chunk.remove_block(i + 1);
                    }
                }
                return;
            }
            if addr.offset + size == block.offset {
                chunk.free_blocks[i].offset = addr.offset;
                chunk.free_blocks[i].size += size;
                if i > 0 {
                    let prev = chunk.free_blocks[i - 1];
                    if prev.offset + prev.size == chunk.free_blocks[i].offset {
                        chunk.free_blocks[i - 1].size += chunk.free_blocks[i].size;
                        chunk.remove_block(i);
                    }
                }
                return;
            }
        }
        chunk.insert_block(addr.offset, size);
    }

    pub fn reset(&mut self) {
        self.chunks.clear();
    }

    pub fn max_size(&self, chunk: usize) -> usize {
        self.chunks.get(chunk).map(|c| c.max_size).unwrap_or(0)
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_alloc() -> DynTallocr {
        DynTallocr::new(64, 1024 * 1024)
    }

    #[test]
    fn new_chunk_creates_single_free_block() {
        let mut a = make_alloc();
        assert_eq!(a.new_chunk(128), Some(0));
        assert_eq!(a.chunks[0].free_blocks.len(), 1);
    }

    #[test]
    fn alloc_returns_valid_address() {
        let mut a = make_alloc();
        a.new_chunk(1024);
        let addr = a.alloc(256);
        assert!(addr.is_some());
        let addr = addr.unwrap();
        assert_eq!(addr.chunk, 0);
        assert!(addr.offset < 1024);
    }

    #[test]
    fn alloc_respects_alignment() {
        let mut a = make_alloc();
        a.new_chunk(64);
        let a1 = a.alloc(1).unwrap();
        assert_eq!(a1.offset % 64, 0);
    }

    #[test]
    fn free_merges_adjacent_blocks() {
        let mut a = DynTallocr::new(64, 2048);
        a.new_chunk(2048);
        let a1 = a.alloc(128).unwrap();
        let a2 = a.alloc(128).unwrap();
        a.free_bytes(a1, 128);
        a.free_bytes(a2, 128);
        assert_eq!(a.chunks[0].free_blocks.len(), 1);
        assert_eq!(a.chunks[0].free_blocks[0].size, 2048);
    }

    #[test]
    fn alloc_after_free_reuses_space() {
        let mut a = DynTallocr::new(64, 4096);
        a.new_chunk(4096);
        let a1 = a.alloc(256).unwrap();
        let _a2 = a.alloc(256).unwrap();
        a.free_bytes(a1, 256);
        let a3 = a.alloc(256).unwrap();
        assert_eq!(a3.offset, a1.offset);
    }

    #[test]
    fn reset_clears_all() {
        let mut a = make_alloc();
        a.new_chunk(512);
        a.alloc(128);
        a.reset();
        assert_eq!(a.chunk_count(), 0);
    }

    #[test]
    fn max_chunks_limit() {
        let mut a = make_alloc();
        for _ in 0..MAX_CHUNKS {
            assert!(a.new_chunk(64).is_some());
        }
        assert!(a.new_chunk(64).is_none());
    }

    #[test]
    fn large_allocation_spans_chunks() {
        let mut a = DynTallocr::new(64, 256);
        for _ in 0..8 {
            assert!(a.alloc(256).is_some());
        }
    }

    #[test]
    fn max_size_tracks_high_water_mark() {
        let mut a = DynTallocr::new(64, 4096);
        a.new_chunk(4096);
        let a1 = a.alloc(1024).unwrap();
        let _a2 = a.alloc(512).unwrap();
        assert!(a.max_size(0) >= 1024 + 512);
        a.free_bytes(a1, 1024);
        assert!(a.max_size(0) >= 1536);
    }

    #[test]
    fn aligned_offset_powers_of_two() {
        let a = DynTallocr::new(128, 4096);
        assert_eq!(a.aligned_offset(0), 0);
        assert_eq!(a.aligned_offset(1), 128);
        assert_eq!(a.aligned_offset(127), 128);
        assert_eq!(a.aligned_offset(128), 128);
        assert_eq!(a.aligned_offset(129), 256);
    }
}

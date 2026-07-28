#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>

#ifndef GGML_ALLOCATOR_DEBUG
#define GGML_ALLOCATOR_DEBUG 0
#endif

static constexpr int   MAX_FREE_BLOCKS    = 256;
static constexpr int   MAX_CHUNKS         = 16;
static constexpr int   MAX_SRC            = 8;
static constexpr int   HASH_TENSORS       = 1024;

static inline size_t aligned_offset(const void * buffer, size_t offset, size_t alignment) {
    assert(alignment && !(alignment & (alignment - 1)));
    size_t align = (alignment - (reinterpret_cast<uintptr_t>(buffer) + offset) % alignment) % alignment;
    return offset + align;
}

#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define GGML_PAD(x, a) (((x) + (a) - 1) & ~((a) - 1))

enum ggml_op {
    GGML_OP_NONE = 0,
    GGML_OP_FILL, GGML_OP_SCALE, GGML_OP_ADD, GGML_OP_ADD_ID, GGML_OP_ADD1,
    GGML_OP_SUB, GGML_OP_MUL, GGML_OP_DIV, GGML_OP_SQR, GGML_OP_SQRT,
    GGML_OP_LOG, GGML_OP_UNARY, GGML_OP_ROPE, GGML_OP_ROPE_BACK,
    GGML_OP_SILU_BACK, GGML_OP_RMS_NORM, GGML_OP_RMS_NORM_BACK,
    GGML_OP_SOFT_MAX, GGML_OP_SOFT_MAX_BACK, GGML_OP_DIAG_MASK_ZERO,
    GGML_OP_DIAG_MASK_INF,
};
enum ggml_tensor_flags {
    GGML_TENSOR_FLAG_OUTPUT = 1,
    GGML_TENSOR_FLAG_INPUT  = 2,
};

struct buffer_address {
    int    chunk;
    size_t offset;
    bool operator<(const buffer_address & o) const {
        return chunk != o.chunk ? chunk < o.chunk : offset < o.offset;
    }
    bool operator==(const buffer_address & o) const {
        return chunk == o.chunk && offset == o.offset;
    }
};
static constexpr buffer_address BUFFER_ADDRESS_INVALID = { -1, SIZE_MAX };

struct free_block {
    size_t offset;
    size_t size;
};

struct Chunk {
    free_block free_blocks[MAX_FREE_BLOCKS];
    int        n_free_blocks;
    size_t     max_size;

    void insert_block(size_t offset, size_t size) {
        assert(n_free_blocks < MAX_FREE_BLOCKS);
        int pos = 0;
        while (pos < n_free_blocks && free_blocks[pos].offset < offset) pos++;
        memmove(&free_blocks[pos+1], &free_blocks[pos], (n_free_blocks - pos) * sizeof(free_block));
        free_blocks[pos] = {offset, size};
        n_free_blocks++;
    }

    void remove_block(int idx) {
        memmove(&free_blocks[idx], &free_blocks[idx+1], (n_free_blocks - idx - 1) * sizeof(free_block));
        n_free_blocks--;
    }

    int best_fit_position(size_t size) const {
        int best = -1;
        size_t best_size = SIZE_MAX;
        for (int i = 0; i < n_free_blocks - 1; i++) {
            if (free_blocks[i].size >= size && free_blocks[i].size < best_size) {
                best = i;
                best_size = free_blocks[i].size;
            }
        }
        return best;
    }
};

struct DynTallocr {
    size_t          alignment;
    size_t          max_chunk_size;
    Chunk *         chunks[MAX_CHUNKS];
    int             n_chunks;

    DynTallocr(size_t align, size_t max_buf_sz)
        : alignment(align)
        , max_chunk_size(std::min(max_buf_sz, SIZE_MAX/2))
        , n_chunks(0)
    {
        memset(chunks, 0, sizeof(chunks));
    }

    ~DynTallocr() {
        for (int i = 0; i < n_chunks; i++) free(chunks[i]);
    }

    int new_chunk(size_t min_size) {
        if (n_chunks >= MAX_CHUNKS) return -1;
        Chunk * ch = (Chunk *)calloc(1, sizeof(Chunk));
        ch->n_free_blocks = 1;
        ch->free_blocks[0] = {0, std::max(min_size, max_chunk_size)};
        if (n_chunks == MAX_CHUNKS - 1) ch->free_blocks[0].size = SIZE_MAX/2;
        chunks[n_chunks++] = ch;
        return n_chunks - 1;
    }

    buffer_address alloc(size_t size) {
        size = (size + alignment - 1) & ~(alignment - 1);
        int best_chunk = -1, best_block = -1;
        size_t best_size = SIZE_MAX, max_avail = 0;

        for (int c = 0; c < n_chunks; c++) {
            Chunk * ch = chunks[c];
            for (int i = 0; i < ch->n_free_blocks - 1; i++) {
                max_avail = MAX(max_avail, ch->free_blocks[i].size);
                if (ch->free_blocks[i].size >= size && ch->free_blocks[i].size < best_size) {
                    best_chunk = c; best_block = i; best_size = ch->free_blocks[i].size;
                }
            }
        }

        if (best_block < 0) {
            int64_t best_reuse = INT64_MIN;
            for (int c = 0; c < n_chunks; c++) {
                Chunk * ch = chunks[c];
                if (ch->n_free_blocks > 0) {
                    auto & blk = ch->free_blocks[ch->n_free_blocks - 1];
                    max_avail = MAX(max_avail, blk.size);
                    int64_t reuse = (int64_t)ch->max_size - (int64_t)blk.offset - (int64_t)size;
                    if (blk.size >= size) {
                        bool better = best_reuse < 0 ? reuse > best_reuse
                                                     : (reuse >= 0 && reuse < best_reuse);
                        if (better) { best_chunk = c; best_block = ch->n_free_blocks - 1; best_reuse = reuse; }
                    }
                }
            }
        }

        if (best_block < 0) {
            best_chunk = new_chunk(size);
            best_block = 0;
        }
        if (best_chunk < 0) {
            fprintf(stderr, "ggml_alloc: OOM (%zu bytes, max_avail %zu)\n", size, max_avail);
            abort();
        }

        Chunk * ch = chunks[best_chunk];
        auto & blk = ch->free_blocks[best_block];
        buffer_address addr = {best_chunk, blk.offset};
        blk.offset += size;
        blk.size -= size;
        if (blk.size == 0) ch->remove_block(best_block);
        ch->max_size = MAX(ch->max_size, addr.offset + size);
        return addr;
    }

    void free_bytes(buffer_address addr, size_t size) {
        size = (size + alignment - 1) & ~(alignment - 1);
        Chunk * ch = chunks[addr.chunk];
        for (int i = 0; i < ch->n_free_blocks; i++) {
            auto & blk = ch->free_blocks[i];
            if (blk.offset + blk.size == addr.offset) {
                blk.size += size;
                if (i + 1 < ch->n_free_blocks &&
                    blk.offset + blk.size == ch->free_blocks[i+1].offset) {
                    blk.size += ch->free_blocks[i+1].size;
                    ch->remove_block(i+1);
                }
                return;
            }
            if (addr.offset + size == blk.offset) {
                blk.offset = addr.offset;
                blk.size += size;
                if (i > 0) {
                    auto & prev = ch->free_blocks[i-1];
                    if (prev.offset + prev.size == blk.offset) {
                        prev.size += blk.size;
                        ch->remove_block(i);
                    }
                }
                return;
            }
        }
        ch->insert_block(addr.offset, size);
    }

    void reset() {
        for (int i = 0; i < n_chunks; i++) { free(chunks[i]); chunks[i] = nullptr; }
        n_chunks = 0;
    }

    size_t get_max_size(int chunk) const {
        return (chunk < n_chunks) ? chunks[chunk]->max_size : 0;
    }
};

static bool op_can_inplace(int op) {
    switch (op) {
        case GGML_OP_FILL: case GGML_OP_SCALE:
        case GGML_OP_DIAG_MASK_ZERO: case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_ADD: case GGML_OP_ADD_ID: case GGML_OP_ADD1:
        case GGML_OP_SUB: case GGML_OP_MUL: case GGML_OP_DIV:
        case GGML_OP_SQR: case GGML_OP_SQRT: case GGML_OP_LOG:
        case GGML_OP_UNARY: case GGML_OP_ROPE: case GGML_OP_ROPE_BACK:
        case GGML_OP_SILU_BACK: case GGML_OP_RMS_NORM:
        case GGML_OP_RMS_NORM_BACK: case GGML_OP_SOFT_MAX:
        case GGML_OP_SOFT_MAX_BACK:
            return true;
        default: return false;
    }
}

struct HashNode {
    int n_children, n_views, buffer_id;
    buffer_address addr;
    bool allocated;
};

struct TensorAlloc {
    int buffer_id;
    buffer_address addr;
    size_t size_max;
};

struct NodeAlloc {
    TensorAlloc dst;
    TensorAlloc src[MAX_SRC];
};

struct LeafAlloc {
    TensorAlloc leaf;
};

struct GraphAllocator {
    std::vector<DynTallocr *> tallocs;
    std::vector<size_t>       alignments;
    std::vector<size_t>       max_sizes;
    int                       n_buffers;

    HashNode    hash_values[HASH_TENSORS];
    bool        hash_used[HASH_TENSORS];
    NodeAlloc * node_allocs = nullptr;
    LeafAlloc * leaf_allocs = nullptr;
    int         n_nodes = 0, n_leafs = 0;

    GraphAllocator() : n_buffers(0) {
        memset(hash_values, 0, sizeof(hash_values));
        memset(hash_used, 0, sizeof(hash_used));
    }

    ~GraphAllocator() {
        free(node_allocs);
        free(leaf_allocs);
    }

    void add_buffer(size_t alignment, size_t max_size) {
        tallocs.push_back(new DynTallocr(alignment, max_size));
        alignments.push_back(alignment);
        this->max_sizes.push_back(max_size);
        n_buffers++;
    }

    uint64_t hash_ptr(const void * p) const {
        return (uint64_t)(uintptr_t)p;
    }

    int hash_insert(const void * key) {
        uint64_t h = hash_ptr(key);
        for (int i = 0; i < HASH_TENSORS; i++) {
            int idx = (int)((h + i) % HASH_TENSORS);
            if (!hash_used[idx]) { hash_used[idx] = true; return idx; }
        }
        return -1;
    }

    int hash_find(const void * key) const {
        uint64_t h = hash_ptr(key);
        for (int i = 0; i < HASH_TENSORS; i++) {
            int idx = (int)((h + i) % HASH_TENSORS);
            if (!hash_used[idx]) return -1;
        }
        return -1;
    }

    HashNode * get_or_create(const void * key) {
        uint64_t h = hash_ptr(key);
        for (int i = 0; i < HASH_TENSORS; i++) {
            int idx = (int)((h + i) % HASH_TENSORS);
            if (!hash_used[idx]) { hash_used[idx] = true; return &hash_values[idx]; }
            return &hash_values[idx];
        }
        return nullptr;
    }

    void reset_hash() {
        memset(hash_used, 0, sizeof(hash_used));
        memset(hash_values, 0, sizeof(hash_values));
    }
};

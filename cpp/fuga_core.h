// fuga_core.h — C++ ядро байтового стека (порт из src/ai/*).
//
// Точная математика из Rust-референса (bin-совместимо по форматам):
//   - byte_basis(b):   fnv1a(byte) → детерминированный гипервектор → sparsify
//                      (SDR 8192, плотность 0.02). Rust: sdr.rs:440.
//   - encode_bytes_sdr(bytes): позиционная свёртка со structure_shift (977)
//                      до STRUCTURE_DENSITY 0.06. Rust: sdr.rs:451.
//   - SdrEncoder:      8192 → 512-латент, splitmix64-хэш ±1, L2-норма.
//                      Rust: latent_jepa.rs:67 (splitmix64 эквивалент).
//   - Widrow-Hoff:     W += lr·err·(P·x), stride=4, cap=2.0/50.
//                      Rust: latent_jepa.rs:260.
//   - FBW1 sidecar:    "FBW1" + u32 len + f32[LATENT²] (le).
//                      Rust: htm_temporal.rs:1018 (save_byte_w).
//
// C++ НЕ использует внешние зависимости: std:: только. RNG — splitmix64
// (тот же family, что Rust StdRng-константы, детерминирован на платформах).
#ifndef FUGA_CORE_H
#define FUGA_CORE_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace fuga {

constexpr int SDR_DIM = 8192;
constexpr double SDR_DENSITY = 0.02;
constexpr double STRUCTURE_DENSITY = 0.06;
constexpr int STRUCTURE_STRIDE = 977;
constexpr int LATENT_DIM = 512;
constexpr float ROW_NORM_CAP = 2.0f;
constexpr uint64_t UPDATE_STRIDE = 4;
constexpr uint64_t CAP_EVERY = 50;

// ---------------------------------------------------------------------------
// splitmix64 — детерминированный RNG (match Rust StdRng-семейства по seed)
// ---------------------------------------------------------------------------
inline uint64_t splitmix64(uint64_t &state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// FNV-1a (match crate::ai::crystal::fnv1a)
inline uint64_t fnv1a(const uint8_t *bytes, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= bytes[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

// ---------------------------------------------------------------------------
// SDR: 8192 бит в u64[128]
// ---------------------------------------------------------------------------
struct Sdr {
    uint64_t bits[128]; // 8192 / 64
    Sdr() { std::memset(bits, 0, sizeof(bits)); }
    bool bit_at(int i) const { return (bits[i >> 6] >> (i & 63)) & 1; }
    void set_bit(int i) { bits[i >> 6] |= (1ULL << (i & 63)); }
    int popcount() const {
        int n = 0;
        for (int i = 0; i < 128; ++i) n += __builtin_popcountll(bits[i]);
        return n;
    }
    // soft overlap (доля общих бит от меньшего)
    float soft_overlap(const Sdr &o) const {
        int both = 0, mine = 0;
        for (int i = 0; i < 128; ++i) {
            both += __builtin_popcountll(bits[i] & o.bits[i]);
            mine += __builtin_popcountll(bits[i]);
        }
        if (mine == 0) return 0.0f;
        return static_cast<float>(both) / static_cast<float>(mine);
    }
    static Sdr zero() { return Sdr(); }
};

// sparsify: гипервектор (u64 words) → SDR фиксированной плотности.
// Rust: SdrVector::from_hypervector — top-`target` битов по хэшу
// splitmix64(hv_seed + bit), hv_seed = fold(31·w + acc). sdr.rs:69-114.
inline Sdr sparsify(const uint64_t *words, int word_count, int dim) {
    int target = static_cast<int>(std::ceil(dim * SDR_DENSITY));
    // hv_seed = fold(acc*31 + w)
    uint64_t hv_seed = 0;
    for (int i = 0; i < word_count; ++i) hv_seed = hv_seed * 31 + words[i];
    struct Cand { int bit; uint64_t score; };
    std::vector<Cand> cands;
    cands.reserve(dim);
    for (int i = 0; i < word_count; ++i) {
        uint64_t w = words[i];
        int base = i * 64;
        for (int b = 0; b < 64 && base + b < dim; ++b) {
            if ((w >> b) & 1) {
                int bit = base + b;
                uint64_t x = hv_seed + bit;
                x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
                x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
                x ^= x >> 31;
                cands.push_back({bit, x});
            }
        }
    }
    // топ-target по score (asc — меньший score раньше), tie: bit asc
    std::sort(cands.begin(), cands.end(), [](const Cand &a, const Cand &b) {
        if (a.score != b.score) return a.score < b.score;
        return a.bit < b.bit;
    });
    if (static_cast<int>(cands.size()) > target) cands.resize(target);
    Sdr out;
    for (const auto &c : cands) out.set_bit(c.bit);
    return out;
}

// byte_basis(b): детерминированный SDR для байта.
// Rust: fnv1a(&[b]) → deterministic_hv(h) → sparsify.
// deterministic_hv: StdRng(seed) 128×u64 (8192 бит). C++: splitmix64 stream.
inline Sdr byte_basis(uint8_t b) {
    uint64_t seed = fnv1a(&b, 1);
    uint64_t state = seed;
    uint64_t words[128];
    for (int i = 0; i < 128; ++i) words[i] = splitmix64(state);
    return sparsify(words, 128, SDR_DIM);
}

// structure_shift(pos) = (pos * STRUCTURE_STRIDE) % SDR_DIM
inline int structure_shift(int pos) {
    return static_cast<int>((static_cast<int64_t>(pos) * STRUCTURE_STRIDE) % SDR_DIM);
}

// encode_bytes_sdr(bytes): позиционная свёртка (Rust sdr.rs:451).
inline Sdr encode_bytes_sdr(const uint8_t *bytes, size_t len) {
    if (len == 0) return Sdr::zero();
    std::vector<uint32_t> counts(SDR_DIM, 0);
    for (size_t pos = 0; pos < len; ++pos) {
        Sdr base = byte_basis(bytes[pos]);
        int shift = structure_shift(static_cast<int>(pos));
        for (int wi = 0; wi < 128; ++wi) {
            uint64_t x = base.bits[wi];
            while (x != 0) {
                int bi = __builtin_ctzll(x);
                int bit = wi * 64 + bi;
                counts[(bit + shift) % SDR_DIM] += 1;
                x &= x - 1;
            }
        }
    }
    int target = static_cast<int>(std::ceil(SDR_DIM * STRUCTURE_DENSITY));
    struct Cand { uint32_t count; uint64_t h; int bit; };
    std::vector<Cand> scored;
    for (int bit = 0; bit < SDR_DIM; ++bit) {
        if (counts[bit] > 0) {
            uint64_t hv = fnv1a(reinterpret_cast<const uint8_t *>(&bit), sizeof(bit));
            scored.push_back({counts[bit], hv, bit});
        }
    }
    std::sort(scored.begin(), scored.end(), [](const Cand &a, const Cand &b) {
        if (a.count != b.count) return a.count > b.count;
        return a.h < b.h; // Rust: a.2.cmp(&b.2)
    });
    if (static_cast<int>(scored.size()) > target) scored.resize(target);
    Sdr out;
    for (const auto &c : scored) out.set_bit(c.bit);
    return out;
}

// structure_sdr_from_sdrs(context): свертка окна SDR (Rust sdr.rs:379).
inline Sdr structure_sdr_from_sdrs(const std::vector<Sdr> &tokens) {
    if (tokens.empty()) return Sdr::zero();
    std::vector<uint32_t> counts(SDR_DIM, 0);
    for (size_t pos = 0; pos < tokens.size(); ++pos) {
        int shift = structure_shift(static_cast<int>(pos));
        for (int wi = 0; wi < 128; ++wi) {
            uint64_t x = tokens[pos].bits[wi];
            while (x != 0) {
                int bi = __builtin_ctzll(x);
                int bit = wi * 64 + bi;
                counts[(bit + shift) % SDR_DIM] += 1;
                x &= x - 1;
            }
        }
    }
    int target = static_cast<int>(std::ceil(SDR_DIM * STRUCTURE_DENSITY));
    struct Cand { uint32_t count; uint64_t h; int bit; };
    std::vector<Cand> scored;
    for (int bit = 0; bit < SDR_DIM; ++bit) {
        if (counts[bit] > 0) {
            uint64_t hv = fnv1a(reinterpret_cast<const uint8_t *>(&bit), sizeof(bit));
            scored.push_back({counts[bit], hv, bit});
        }
    }
    std::sort(scored.begin(), scored.end(), [](const Cand &a, const Cand &b) {
        if (a.count != b.count) return a.count > b.count;
        return a.h < b.h;
    });
    if (static_cast<int>(scored.size()) > target) scored.resize(target);
    Sdr out;
    for (const auto &c : scored) out.set_bit(c.bit);
    return out;
}

// ---------------------------------------------------------------------------
// SdrEncoder: 8192 → 512 латент (Rust latent_jepa.rs:67, splitmix64-хэш)
// ---------------------------------------------------------------------------
struct SdrEncoder {
    uint64_t seed;
    explicit SdrEncoder(uint64_t s = 0x9E3779B97F4A7C15ULL) : seed(s) {}

    uint64_t hash(int latent, int bit) const {
        uint64_t x = seed + static_cast<uint64_t>(latent) * 0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(bit);
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    std::vector<float> encode(const Sdr &sdr) const {
        std::vector<float> values(LATENT_DIM, 0.0f);
        // Итерируем ТОЛЬКО установленные биты (164 шт.), не все 8192.
        // Rust: `if sdr.bit_at(bit) == 0 { continue; }` — то же самое.
        for (int wi = 0; wi < 128; ++wi) {
            uint64_t x = sdr.bits[wi];
            while (x != 0) {
                int bit = wi * 64 + __builtin_ctzll(x);
                for (int latent = 0; latent < LATENT_DIM; ++latent) {
                    uint64_t h = hash(latent, bit);
                    values[latent] += (h & 1) == 0 ? 1.0f : -1.0f;
                }
                x &= x - 1;
            }
        }
        float norm = 0.0f;
        for (float v : values) norm += v * v;
        norm = std::sqrt(norm > 0 ? norm : 1e-8f);
        for (float &v : values) v /= norm;
        return values;
    }
};

// ---------------------------------------------------------------------------
// LatentPredictor (C++): W (LATENT²), Widrow-Hoff с OWM-P
// ---------------------------------------------------------------------------
struct LatentPredictor {
    SdrEncoder encoder;
    std::vector<float> w;        // LATENT² row-major
    std::vector<float> p;        // OWM-проектор (identity по умолчанию)
    uint64_t updates = 0;
    uint64_t cap_firings = 0;

    LatentPredictor() : w(LATENT_DIM * LATENT_DIM, 0.0f), p(LATENT_DIM * LATENT_DIM, 0.0f) {
        for (int i = 0; i < LATENT_DIM; ++i) p[i * LATENT_DIM + i] = 1.0f; // identity
    }

    std::vector<float> apply_w(const std::vector<float> &x) const {
        std::vector<float> y(LATENT_DIM, 0.0f);
        for (int o = 0; o < LATENT_DIM; ++o) {
            const float *row = &w[o * LATENT_DIM];
            float acc = 0.0f;
            for (int i = 0; i < LATENT_DIM; ++i) acc += row[i] * x[i];
            y[o] = acc;
        }
        return y;
    }

    std::vector<float> apply_p_with(const std::vector<float> &x, const std::vector<float> &proj) const {
        std::vector<float> y(LATENT_DIM, 0.0f);
        for (int o = 0; o < LATENT_DIM; ++o) {
            const float *row = &proj[o * LATENT_DIM];
            float acc = 0.0f;
            for (int i = 0; i < LATENT_DIM; ++i) acc += row[i] * x[i];
            y[o] = acc;
        }
        return y;
    }

    // learn_transition(context, actual, lr) — Rust latent_jepa.rs:260
    float learn_transition(const std::vector<Sdr> &context, const Sdr &actual, float lr) {
        if (context.empty()) return 0.0f;
        updates += 1;
        bool apply_delta = (updates % UPDATE_STRIDE == 0);
        Sdr ctx_sdr = structure_sdr_from_sdrs(context);
        std::vector<float> x = encoder.encode(ctx_sdr);
        std::vector<float> target = encoder.encode(actual);
        std::vector<float> pred = apply_delta ? apply_w(x) : x;
        std::vector<float> px = apply_delta ? apply_p_with(x, p) : x;
        float err_norm = 0.0f;
        for (int o = 0; o < LATENT_DIM; ++o) {
            float error = target[o] - pred[o];
            err_norm += error * error;
            if (apply_delta) {
                float *row = &w[o * LATENT_DIM];
                for (int i = 0; i < LATENT_DIM; ++i) row[i] += lr * error * px[i];
            }
        }
        if (apply_delta && (updates % CAP_EVERY == 0)) {
            for (int o = 0; o < LATENT_DIM; ++o) {
                float *row = &w[o * LATENT_DIM];
                float sq = 0.0f;
                for (int i = 0; i < LATENT_DIM; ++i) sq += row[i] * row[i];
                if (sq > ROW_NORM_CAP) {
                    cap_firings += 1;
                    float scale = std::sqrt(ROW_NORM_CAP / sq);
                    for (int i = 0; i < LATENT_DIM; ++i) row[i] *= scale;
                }
            }
        }
        return std::sqrt(err_norm);
    }
};

// ---------------------------------------------------------------------------
// FBW1 sidecar IO (бит-в-бит совместимо с Rust save_byte_w/load_byte_w)
// ---------------------------------------------------------------------------
inline bool save_byte_w(const std::string &path, const std::vector<float> &w) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write("FBW1", 4);
    uint32_t len = static_cast<uint32_t>(w.size());
    f.write(reinterpret_cast<const char *>(&len), 4);
    for (float v : w) {
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        f.write(reinterpret_cast<const char *>(&bits), 4); // le bytes
    }
    return f.good();
}

inline std::vector<float> load_byte_w(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    std::streamsize size = f.tellg();
    f.seekg(0);
    if (size < 8) return {};
    std::vector<uint8_t> data(static_cast<size_t>(size));
    f.read(reinterpret_cast<char *>(data.data()), size);
    if (std::memcmp(data.data(), "FBW1", 4) != 0) return {};
    uint32_t n;
    std::memcpy(&n, data.data() + 4, 4);
    if (n != LATENT_DIM * LATENT_DIM || 8 + static_cast<size_t>(n) * 4 > data.size()) return {};
    std::vector<float> w(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, data.data() + 8 + i * 4, 4);
        std::memcpy(&w[i], &bits, 4);
    }
    return w;
}

} // namespace fuga

#endif // FUGA_CORE_H

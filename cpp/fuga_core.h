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
#include <optional>
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

    // Widrow-Hoff НАПРЯМУЮ в латентном пространстве (латент → латент):
    // для JEPA-уровней, которые предсказывают гипервекторы из гипервекторов.
    // x — латент окна (усреднение входа), target — латент e.g. следующего шага.
    // Тот же delta-rule и cap, что и learn_transition, но вход уже латент.
    float learn_latent(const std::vector<float> &x, const std::vector<float> &target, float lr) {
        updates += 1;
        bool apply_delta = (updates % UPDATE_STRIDE == 0);
        std::vector<float> pred = apply_delta ? apply_w(x) : x;
        float err_norm = 0.0f;
        for (int o = 0; o < LATENT_DIM; ++o) {
            float error = target[o] - pred[o];
            err_norm += error * error;
            if (apply_delta) {
                float *row = &w[o * LATENT_DIM];
                for (int i = 0; i < LATENT_DIM; ++i) row[i] += lr * error * x[i];
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
    // OWM-consolidate: P ← P − P·Aᵀ·(A·P·Aᵀ + α·I)⁻¹·A·P (Woodbury, K×K).
    // directions: латентные направления для защиты (эпоха-консолидация).
    // top_k: максимум консолидируемых направлений (Gram-Schmidt-редукция).
    // Возвращает число направлений; 0 если некорректно (как Rust).
    int consolidate_owm(const std::vector<std::vector<float>> &directions, int top_k, float alpha) {
        const int d = LATENT_DIM;
        int m = (int)directions.size();
        if (m == 0 || top_k <= 0) return 0;
        int k = std::min(top_k, m);
        // --- Gram-Schmidt-редукция: до k ортогональных единичных строк A ---
        std::vector<std::vector<float>> chosen;
        // сортировка индексов по убыванию нормы (как Rust idx.sort_by)
        std::vector<int> idx(m);
        for (int i = 0; i < m; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            float na = 0, nb = 0;
            for (float v : directions[a]) na += v * v;
            for (float v : directions[b]) nb += v * v;
            return na > nb;
        });
        for (int ii = 0; ii < m && (int)chosen.size() < k; ++ii) {
            int i = idx[ii];
            std::vector<float> res = directions[i];
            for (const auto &c : chosen) {
                float dot = 0;
                for (int j = 0; j < d; ++j) dot += c[j] * res[j];
                for (int j = 0; j < d; ++j) res[j] -= dot * c[j];
            }
            float norm = 0;
            for (float v : res) norm += v * v;
            norm = std::sqrt(norm);
            if (norm > 1e-6f) {
                for (float &v : res) v /= norm;
                chosen.push_back(res);
            }
        }
        k = (int)chosen.size();
        if (k == 0) return 0;
        // A: k×d row-major
        std::vector<float> a(k * d);
        for (int r = 0; r < k; ++r)
            for (int j = 0; j < d; ++j) a[r * d + j] = chosen[r][j];
        // PAᵀ = P·Aᵀ (d×k)
        std::vector<float> pat(d * k);
        for (int l = 0; l < d; ++l) {
            for (int j = 0; j < k; ++j) {
                float acc = 0;
                for (int c = 0; c < d; ++c) acc += p[l * d + c] * a[j * d + c];
                pat[l * k + j] = acc;
            }
        }
        // M = A·PAᵀ + αI (k×k)
        std::vector<float> mtx(k * k, 0.0f);
        for (int i = 0; i < k; ++i)
            for (int j = 0; j < k; ++j) {
                float acc = 0;
                for (int l = 0; l < d; ++l) acc += a[i * d + l] * pat[l * k + j];
                mtx[i * k + j] = acc;
            }
        for (int i = 0; i < k; ++i) mtx[i * k + i] += alpha;
        auto minv = invert_square(mtx, k);
        if (!minv) return 0;
        // TM = PAᵀ·M⁻¹ (d×k)
        std::vector<float> tm(d * k);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < k; ++j) {
                float acc = 0;
                for (int l = 0; l < k; ++l) acc += pat[i * k + l] * (*minv)[l * k + j];
                tm[i * k + j] = acc;
            }
        // XA = TM·A (d×d)
        std::vector<float> xa(d * d);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j) {
                float acc = 0;
                for (int l = 0; l < k; ++l) acc += tm[i * k + l] * a[l * d + j];
                xa[i * d + j] = acc;
            }
        // P_new = P − XA·P
        std::vector<float> pnew(d * d);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j) {
                float acc = 0;
                for (int l = 0; l < d; ++l) acc += xa[i * d + l] * p[l * d + j];
                pnew[i * d + j] = p[i * d + j] - acc;
            }
        p = std::move(pnew);
        return k;
    }

    // Обращение квадратной матрицы (Гаусс с частичным поворотом).
    // Возвращает nullopt если сингулярна (как Rust invert_square → None).
    std::optional<std::vector<float>> invert_square(const std::vector<float> &mat, int n) {
        std::vector<float> a = mat;
        std::vector<float> inv(n * n, 0.0f);
        for (int i = 0; i < n; ++i) inv[i * n + i] = 1.0f;
        for (int col = 0; col < n; ++col) {
            // partial pivoting
            int piv = col;
            float best = std::fabs(a[col * n + col]);
            for (int r = col + 1; r < n; ++r) {
                float v = std::fabs(a[r * n + col]);
                if (v > best) { best = v; piv = r; }
            }
            if (best < 1e-12f) return std::nullopt;
            if (piv != col) {
                for (int j = 0; j < n; ++j) {
                    std::swap(a[col * n + j], a[piv * n + j]);
                    std::swap(inv[col * n + j], inv[piv * n + j]);
                }
            }
            float diag = a[col * n + col];
            for (int j = 0; j < n; ++j) {
                a[col * n + j] /= diag;
                inv[col * n + j] /= diag;
            }
            for (int r = 0; r < n; ++r) {
                if (r == col) continue;
                float f = a[r * n + col];
                if (f == 0.0f) continue;
                for (int j = 0; j < n; ++j) {
                    a[r * n + j] -= f * a[col * n + j];
                    inv[r * n + j] -= f * inv[col * n + j];
                }
            }
        }
        return inv;
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

// ---------------------------------------------------------------------------
// ЕДИНЫЙ ФОРМАТ FUGA1 — один файл, разные обучения.
//
// Структура: MAGIC "FUGA1" + последовательность секций
//   [u32 tag][u32 len][len байт]
//   tag=1 LOCAL_W  f32[LATENT²]  локальный байтовый W (главный путь)
//   tag=2 PATCH_W  f32[LATENT²]  патчевый W_patch (two-speed)
//   tag=3 OWM_P    f32[LATENT²]  OWM-проектор
//   tag=4 META     u64 steps | u64 patch_steps | u32 ctx | u32 version
//   tag=5 HJEPA    f32[3*LATENT²] веса L0/L1/L2 (опционально)
//   tag=0 END
// C++ и Rust читают/пишут ОДИН формат (bin-совместим).
// ---------------------------------------------------------------------------
enum UnifiedTag : uint32_t {
    TAG_END = 0,
    TAG_LOCAL_W = 1,
    TAG_PATCH_W = 2,
    TAG_OWM_P = 3,
    TAG_META = 4,
    TAG_HJEPA = 5,
};

inline constexpr const char *UNIFIED_MAGIC = "FUGA1";

struct UnifiedMeta {
    uint64_t steps = 0;
    uint64_t patch_steps = 0;
    uint32_t ctx = 4;
    uint32_t version = 1;
};

// Записать секцию [tag][len][data] (маленькая-endian)
inline void write_u32(std::ofstream &f, uint32_t v) { f.write(reinterpret_cast<const char *>(&v), 4); }
inline void write_u64(std::ofstream &f, uint64_t v) { f.write(reinterpret_cast<const char *>(&v), 8); }
inline void write_section(std::ofstream &f, uint32_t tag, const std::vector<float> &vals) {
    write_u32(f, tag);
    write_u32(f, static_cast<uint32_t>(vals.size() * 4));
    for (float v : vals) {
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        f.write(reinterpret_cast<const char *>(&bits), 4);
    }
}

// Единый чекпоинт: локальный W + патчевый W + OWM-P (+ HJEPA-веса опц.).
// Возвращает false при ошибке записи.
inline bool save_unified(const std::string &path,
                         const std::vector<float> &local_w,
                         const std::vector<float> &patch_w,
                         const std::vector<float> &owm_p,
                         const UnifiedMeta &meta,
                         const std::vector<float> &hjepa_flat = {}) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(UNIFIED_MAGIC, 5);
    write_section(f, TAG_LOCAL_W, local_w);
    write_section(f, TAG_PATCH_W, patch_w);
    write_section(f, TAG_OWM_P, owm_p);
    // META: 16 байт + 8 байт = 24
    write_u32(f, TAG_META);
    write_u32(f, 24);
    write_u64(f, meta.steps);
    write_u64(f, meta.patch_steps);
    write_u32(f, meta.ctx);
    write_u32(f, meta.version);
    if (!hjepa_flat.empty()) {
        write_section(f, TAG_HJEPA, hjepa_flat);
    }
    write_u32(f, TAG_END);
    write_u32(f, 0);
    return f.good();
}

// Прочитать единый чекпоинт. Заполняет только найденные секции;
// возвращает false если magic не FUGA1.
inline bool load_unified(const std::string &path,
                         std::vector<float> *local_w,
                         std::vector<float> *patch_w,
                         std::vector<float> *owm_p,
                         UnifiedMeta *meta,
                         std::vector<float> *hjepa_flat = nullptr) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize size = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (!f.read(reinterpret_cast<char *>(data.data()), size)) return false;
    if (data.size() < 5 || std::memcmp(data.data(), UNIFIED_MAGIC, 5) != 0) return false;
    size_t pos = 5;
    auto rd32 = [&](size_t off) -> uint32_t {
        uint32_t v; std::memcpy(&v, data.data() + off, 4); return v;
    };
    auto rd64 = [&](size_t off) -> uint64_t {
        uint64_t v; std::memcpy(&v, data.data() + off, 8); return v;
    };
    while (pos + 8 <= data.size()) {
        uint32_t tag = rd32(pos);
        uint32_t len = rd32(pos + 4);
        pos += 8;
        if (pos + len > data.size()) break;
        std::vector<float> vals;
        if (tag == TAG_LOCAL_W || tag == TAG_PATCH_W || tag == TAG_OWM_P ||
            tag == TAG_HJEPA) {
            size_t n = len / 4;
            vals.reserve(n);
            for (size_t i = 0; i < n; ++i) {
                uint32_t bits; std::memcpy(&bits, data.data() + pos + i * 4, 4);
                float v; std::memcpy(&v, &bits, 4);
                vals.push_back(v);
            }
        }
        switch (tag) {
            case TAG_LOCAL_W: if (local_w) *local_w = vals; break;
            case TAG_PATCH_W: if (patch_w) *patch_w = vals; break;
            case TAG_OWM_P: if (owm_p) *owm_p = vals; break;
            case TAG_HJEPA: if (hjepa_flat) *hjepa_flat = vals; break;
            case TAG_META:
                if (meta && len >= 24) {
                    meta->steps = rd64(pos);
                    meta->patch_steps = rd64(pos + 8);
                    meta->ctx = rd32(pos + 16);
                    meta->version = rd32(pos + 20);
                }
                break;
            case TAG_END:
                return true;
            default:
                break;
        }
        pos += len;
    }
    return true;
}

} // namespace fuga

#endif // FUGA_CORE_H

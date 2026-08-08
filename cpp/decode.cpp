// decode.cpp — C++ байтовые декодеры (порт tm_generate.rs).
//
// Читает FBW1 sidecar (local W из Rust save_byte_w или C++ train) и
// декодирует с seed:
//   - naive    : top-1 байт по cosine, порог LATENT_MIN_COSINE=0.05,
//                стоп на повторе последнего байта (Rust: tm_generate_latent_bytes)
//   - recurrent: SSM-lite h(t), mix, gap-адаптивный φ (Rust: tm_generate_recurrent)
//   - entropy  : BLT two-speed: gap порог решает local vs global patch
//                (Rust: tm_generate_two_speed_entropy)
//
// Usage:
//   decode --w fuga_byte_w_800.bin --patch fuga_cpp_patch.bin \
//          --decoder naive|recurrent|entropy --seed "fn main() {" \
//          --max 200 --ctx 4 [--gap 0.60] [--mix 0.4] [--phi 0.9] [--psize 2]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "fuga_core.h"

using namespace fuga;

static constexpr float LATENT_MIN_COSINE = 0.05f;

static float cosine(const std::vector<float> &a, const std::vector<float> &b) {
    float dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// Патч-словарь из patch sidecar: не читается напрямую (sidecar хранит только W).
// Вместо этого строим фиксированный vocab из распространённых 2-байтов,
// чьи латенты предсказываются W_patch. (Честный порт: в Rust patch_vocab
// собирается из тренировочного корпуса; здесь даём --vocab файл, строки = патчи.)
static std::vector<std::vector<uint8_t>> load_vocab(const std::string &path) {
    std::vector<std::vector<uint8_t>> vocab;
    std::ifstream f(path);
    if (!f) return vocab;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<uint8_t> p;
        for (char c : line) {
            if (c == '\n') break;
            p.push_back(static_cast<uint8_t>(c));
        }
        if (p.size() >= 2) vocab.push_back(p);
    }
    return vocab;
}

int main(int argc, char **argv) {
    std::string w_path, patch_path, vocab_path, decoder = "recurrent", seed = "fn main() {";
    int max_bytes = 200, ctx = 4, psize = 2;
    float gap_thresh = 0.60f, mix = 0.4f, phi = 0.9f;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "--w")        w_path = next();
        else if (a == "--patch")    patch_path = next();
        else if (a == "--vocab")    vocab_path = next();
        else if (a == "--decoder")  decoder = next();
        else if (a == "--seed")     seed = next();
        else if (a == "--max")      max_bytes = std::stoi(next());
        else if (a == "--ctx")      ctx = std::stoi(next());
        else if (a == "--gap")      gap_thresh = std::stof(next());
        else if (a == "--mix")      mix = std::stof(next());
        else if (a == "--phi")      phi = std::stof(next());
        else if (a == "--psize")    psize = std::stoi(next());
    }

    auto w = load_byte_w(w_path);
    if (w.empty()) {
        std::cerr << "ERROR: cannot load W from " << w_path << " (bad magic/size?)\n";
        return 1;
    }
    LatentPredictor local;
    local.w = w;
    std::vector<float> wp;
    if (!patch_path.empty()) {
        wp = load_byte_w(patch_path);
        if (wp.empty()) std::cerr << "WARN: patch W not loaded (" << patch_path << ")\n";
    }
    // patch predictor W (используется только для entropy)
    LatentPredictor patch_p;
    if (!wp.empty()) patch_p.w = wp;

    // Pre-encode the fixed 256-byte alphabet latents (frozen encoder).
    SdrEncoder enc;
    std::vector<std::vector<float>> byte_lats;
    byte_lats.reserve(256);
    for (int b = 0; b < 256; ++b) {
        byte_lats.push_back(enc.encode(byte_basis(static_cast<uint8_t>(b))));
    }

    // vocabulary for entropy decoder
    auto vocab = load_vocab(vocab_path);
    std::vector<std::vector<float>> patch_lats;
    if (!vocab.empty()) {
        for (const auto &p : vocab) {
            patch_lats.push_back(enc.encode(encode_bytes_sdr(p.data(), p.size())));
        }
    }

    std::vector<uint8_t> state(seed.begin(), seed.end());
    std::vector<uint8_t> out;
    std::vector<float> h(LATENT_DIM, 0.0f);
    int guard = 0;

    auto t0 = std::chrono::steady_clock::now();

    if (decoder == "naive") {
        while ((int)out.size() < max_bytes && guard < max_bytes * 2) {
            ++guard;
            size_t win_lo = state.size() > (size_t)ctx ? state.size() - ctx : 0;
            std::vector<Sdr> window_sdrs;
            for (size_t j = win_lo; j < state.size(); ++j)
                            window_sdrs.push_back(byte_basis(state[j]));
                        std::vector<float> pred_lat = local.apply_w(enc.encode(structure_sdr_from_sdrs(window_sdrs)));
                        float best_score = -1.0f;
                        int best_byte = -1;
                        for (int b = 0; b < 256; ++b) {
                            float c = cosine(pred_lat, byte_lats[b]);
                if (c < LATENT_MIN_COSINE) continue;
                if (c > best_score) { best_score = c; best_byte = b; }
            }
            if (best_byte < 0) break;
            if (best_score < LATENT_MIN_COSINE) break;
            if (!out.empty() && out.back() == (uint8_t)best_byte) break;
            out.push_back((uint8_t)best_byte);
            state.push_back((uint8_t)best_byte);
        }
    } else if (decoder == "recurrent") {
        while ((int)out.size() < max_bytes && guard < max_bytes * 2) {
            ++guard;
            size_t win_lo = state.size() > (size_t)ctx ? state.size() - ctx : 0;
            std::vector<Sdr> window_sdrs;
            for (size_t j = win_lo; j < state.size(); ++j)
                window_sdrs.push_back(byte_basis(state[j]));
            // predict_next_rnn: input = local + mix·h, renormalized, then W
            std::vector<float> local_v = enc.encode(structure_sdr_from_sdrs(window_sdrs));
            std::vector<float> input(LATENT_DIM);
            for (int i = 0; i < LATENT_DIM; ++i) input[i] = local_v[i] + mix * h[i];
            float nrm = 0;
            for (float v : input) nrm += v * v;
            nrm = std::sqrt(nrm > 0 ? nrm : 1e-8f);
            for (float &v : input) v /= nrm;
            std::vector<float> pred = local.apply_w(input);
            // gap-adaptive phi: уверен → полное запоминание, нет → забыть
            float best_c = -1, second_c = -1;
            int best_byte = -1;
            for (int b = 0; b < 256; ++b) {
                float c = cosine(pred, byte_lats[b]);
                if (c > best_c) { second_c = best_c; best_c = c; best_byte = b; }
                else if (c > second_c) second_c = c;
            }
            float gap = best_c - second_c;
            float phi_eff;
            if (gap >= 0.30f) phi_eff = phi;
            else if (gap <= 0.10f) phi_eff = 0.05f;
            else phi_eff = 0.05f + (phi - 0.05f) * (gap - 0.10f) / 0.20f;
            if (!out.empty() && out.back() == (uint8_t)best_byte && out.size() > 2) break;
            out.push_back((uint8_t)best_byte);
            state.push_back((uint8_t)best_byte);
            // advance_h: h' = phi·h + (1-phi)·enc(byte)
            std::vector<float> enc_b = enc.encode(byte_basis((uint8_t)best_byte));
            for (int i = 0; i < LATENT_DIM; ++i)
                h[i] = phi_eff * h[i] + (1.0f - phi_eff) * enc_b[i];
        }
    } else if (decoder == "entropy") {
        if (vocab.empty()) {
            std::cerr << "ERROR: --decoder entropy требует --vocab (список патчей)\n";
            return 2;
        }
        int pad = 0;
        (void)pad;
        while ((int)out.size() < max_bytes && guard < max_bytes * 2) {
            ++guard;
            size_t win_lo = state.size() > (size_t)ctx ? state.size() - ctx : 0;
            std::vector<Sdr> ws;
            for (size_t j = win_lo; j < state.size(); ++j) ws.push_back(byte_basis(state[j]));
            std::vector<float> pred = local.apply_w(enc.encode(structure_sdr_from_sdrs(ws)));
            float top1 = std::numeric_limits<float>::lowest();
            float top2 = std::numeric_limits<float>::lowest();
            int top1b = 0;
            for (int b = 0; b < 256; ++b) {
                float c = cosine(pred, byte_lats[b]);
                if (c > top1) { top2 = top1; top1 = c; top1b = b; }
                else if (c > top2) top2 = c;
            }
            float gap = top1 - top2;
            if (gap >= gap_thresh) {
                uint8_t b = (uint8_t)top1b;
                if (!out.empty() && out.back() == b && out.size() > 2) break;
                out.push_back(b);
                state.push_back(b);
            } else {
                // global patch selection
                if (patch_lats.empty()) break; // no vocab → cannot choose patch
                size_t np = state.size() / psize;
                if (np == 0) break;
                // последние 4 патча окна
                std::vector<Sdr> pat_sdrs;
                size_t start = np > 4 ? np - 4 : 0;
                for (size_t k = start; k < np; ++k) {
                    const uint8_t *pp = &state[k * psize];
                    pat_sdrs.push_back(encode_bytes_sdr(pp, psize));
                }
                std::vector<float> pred_p = patch_p.apply_w(enc.encode(structure_sdr_from_sdrs(pat_sdrs)));
                float best = -1.0f; int best_idx = -1;
                for (size_t vi = 0; vi < vocab.size(); ++vi) {
                    float c = cosine(pred_p, patch_lats[vi]);
                    if (c < LATENT_MIN_COSINE) continue;
                    if (c > best) { best = c; best_idx = (int)vi; }
                }
                if (best_idx < 0 || best < LATENT_MIN_COSINE) break;
                const auto &patch = vocab[best_idx];
                out.insert(out.end(), patch.begin(), patch.end());
                state.insert(state.end(), patch.begin(), patch.end());
            }
        }
    } else {
        std::cerr << "unknown decoder: " << decoder << "\n";
        return 3;
    }

    auto el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::string text(out.begin(), out.end());
    std::string shown;
    for (size_t i = 0; i < text.size() && i < 80; ++i) shown += (text[i] >= 32 && text[i] < 127) ? text[i] : '.';
    std::cout << out.size() << " B :: " << shown << "\n";
    std::cout << "decoded in " << el << "s  (" << (out.size() / (el > 0 ? el : 1e-9)) << " B/s)\n";
    return 0;
}
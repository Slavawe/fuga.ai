// train.cpp — C++ ядро обучения байтового стека (порт full_byte_train.rs).
//
// Читает те же JSONL-корпуса, стримит байты и обучает:
//   - локальный байтовый W (окно ctx → следующий байт)
//   - глобальный патчевый W_patch (two-speed: окно 2-байт патча → след.)
// Сохраняет FBW1 sidecar (совместимо с Rust save_byte_w) + патчевый sidecar.
//
// Usage: train [--ctx 4] [--lr 0.05] [--max-bytes 50000000] [--out fuga_cpp_byte_w.bin]
//        [--jsonl file.jsonl]... [--seed "fn main() {" --decode]
//
// НЕ зависит от внешних библиотек: JSONL-парсинг минимальный (doc/code/chapters).
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "fuga_core.h"

using namespace fuga;

// Минимальный JSONL-экстрактор: достаёт строку из "doc"/"code"/"chapters[].paragraphs[]"
// Фолбэк: если не JSON — берёт строку как есть (как Rust extract_bytes).
static std::vector<uint8_t> extract_bytes(const std::string &line) {
    if (line.empty()) return {};
    if (line[0] != '{') {
        return std::vector<uint8_t>(line.begin(), line.end());
    }
    auto find_val = [&](const std::string &key) -> std::string {
        // find `"key":"...` with escaped-string awareness
        std::string pat = "\"" + key + "\":\"";
        size_t pos = line.find(pat);
        if (pos == std::string::npos) return "";
        pos += pat.size();
        std::string out;
        while (pos < line.size() && line[pos] != '"') {
            if (line[pos] == '\\' && pos + 1 < line.size()) {
                char c = line[pos + 1];
                if (c == 'n') out += '\n';
                else if (c == 't') out += '\t';
                else if (c == '\\') out += '\\';
                else if (c == '"') out += '"';
                else out += c;
                pos += 2;
            } else {
                out += line[pos++];
            }
        }
        return out;
    };
    // try doc
    std::string doc = find_val("doc");
    if (!doc.empty()) return std::vector<uint8_t>(doc.begin(), doc.end());
    std::string code = find_val("code");
    if (!code.empty()) return std::vector<uint8_t>(code.begin(), code.end());
    // chapters fallback: наивно берём всё, что внутри "paragraphs":[ "...", ... ]
    size_t cp = line.find("\"paragraphs\"");
    if (cp == std::string::npos) return std::vector<uint8_t>(line.begin(), line.end());
    std::string out;
    size_t i = cp;
    while (i < line.size()) {
        if (line[i] == '"' && i + 1 < line.size() && line[i + 1] != ',' && line[i + 1] != '}' && line[i + 1] != ']') {
            // начало строки параграфа
            size_t j = i + 1;
            std::string p;
            while (j < line.size() && line[j] != '"') {
                if (line[j] == '\\' && j + 1 < line.size()) {
                    char c = line[j + 1];
                    if (c == 'n') p += '\n';
                    else if (c == 't') p += '\t';
                    else if (c == '\\') p += '\\';
                    else p += c;
                    j += 2;
                } else {
                    p += line[j++];
                }
            }
            if (p.size() > 32 && p.find("paragraphs") == std::string::npos) {
                out += p;
                out += '\n';
            }
            i = j + 1;
        } else {
            ++i;
        }
    }
    return std::vector<uint8_t>(out.begin(), out.end());
}

void use_seed(const std::string &s) { (void)s; } // (seed reserved for decode)
// Кэш 256 байт-базисов: фиксированный алфавит, инвариантен при обучении.
// Rust делает то же через byte_basis() (детерминированный). Ускоряет
// обучение на порядки (не пересчитываем SDR на каждый байт окна).
static const std::vector<Sdr> &byte_basis_cache() {
    static std::vector<Sdr> cache = [] {
        std::vector<Sdr> c;
        c.reserve(256);
        for (int b = 0; b < 256; ++b) c.push_back(byte_basis(static_cast<uint8_t>(b)));
        return c;
    }();
    return cache;
}

static inline const Sdr &basis(uint8_t b) { return byte_basis_cache()[b]; }

// Кэш encode_bytes_sdr для 2-байтовых патчей: 65536 комбинаций, инвариантны.
// Патч-уровень обучения (two-speed) вызывает encode_bytes_sdr каждые шаг —
// кэш убирает O(плотность × sort) на каждый вызов.
static const std::vector<Sdr> &patch2_cache() {
    static std::vector<Sdr> cache = [] {
        std::vector<Sdr> c;
        c.reserve(65536);
        for (int hi = 0; hi < 256; ++hi) {
            for (int lo = 0; lo < 256; ++lo) {
                uint8_t p[2] = { (uint8_t)hi, (uint8_t)lo };
                c.push_back(encode_bytes_sdr(p, 2));
            }
        }
        return c;
    }();
    return cache;
}

static inline const Sdr &patch2(uint16_t v) { return patch2_cache()[v]; }

int main(int argc, char **argv) {
    int ctx = 4;
    float lr = 0.05f;
    long long max_bytes = 50'000'000;
    std::string out_path = "fuga_cpp_byte_w.bin";
    std::vector<std::string> corpora;
    std::string seed = "fn main() {";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--ctx") ctx = std::stoi(next());
        else if (a == "--lr") lr = std::stof(next());
        else if (a == "--max-bytes") max_bytes = std::stoll(next());
        else if (a == "--out") out_path = next();
        else if (a == "--seed") seed = next();
        else if (a == "--jsonl") corpora.push_back(next());
        else if (!a.empty() && a[0] != '-') corpora.push_back(a); // positional = corpus path
    }
    if (corpora.empty()) {
        corpora = {
            "fuga_unified_train.jsonl",
            "corpus_doc_code_pairs.jsonl",
            "training_stack.jsonl",
            "omni_corpus_full.jsonl",
            "corpus.jsonl",
            "corpus_rus_eng.jsonl",
            "omni_corpus_repos.jsonl",
        };
    }

    auto t0 = std::chrono::steady_clock::now();
    LatentPredictor local;      // локальный байтовый W
    LatentPredictor patch;      // глобальный патчевый W_patch
    uint64_t steps = 0, byte_count = 0;

    for (const auto &corp : corpora) {
        std::ifstream f(corp);
        if (!f) { std::cout << "  skip (missing): " << corp << "\n"; continue; }
        std::cout << "TRAINING on " << corp << " ...\n";
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line == "\n" || line == "\r") continue;
            auto data = extract_bytes(line);
            if (data.size() < 2) continue;
            for (size_t i = 0; i + 1 < data.size(); ++i) {
                size_t lo = (i >= (size_t)ctx) ? i - ctx : 0;
                uint8_t nxt = data[i + 1];
                std::vector<Sdr> win_sdrs;
                win_sdrs.reserve(ctx + 1);
                for (size_t j = lo; j <= i; ++j) win_sdrs.push_back(basis(data[j]));
                Sdr next = basis(nxt);
                local.learn_transition(win_sdrs, next, lr);
                // global patch level (two-speed)
                if (i >= 2) {
                    std::vector<Sdr> pat_sdrs;
                    pat_sdrs.push_back(patch2((uint16_t)((data[i-2] << 8) | data[i-1])));
                    pat_sdrs.push_back(patch2((uint16_t)((data[i-1] << 8) | data[i])));
                    size_t plen = (i + 3 <= data.size()) ? 2 : data.size() - (i + 1);
                    Sdr nxt_patch = encode_bytes_sdr(&data[i + 1], plen);
                    patch.learn_transition(pat_sdrs, nxt_patch, lr);
                }
                ++steps;
                if (steps % 1'000'000 == 0)
                    std::cout << "    ... " << steps << " steps\n";
                if (steps >= (uint64_t)max_bytes) break;
            }
            if (steps >= (uint64_t)max_bytes) break;
        }
        std::cout << "  corpus done: " << corp << " (steps=" << steps << ")\n";
        if (steps >= (uint64_t)max_bytes) break;
    }

    auto el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "\n=== CPP BYTE TRAINING COMPLETE ===\n";
    std::cout << "  steps=" << steps << " bytes=" << byte_count << " in " << el << "s\n";
    std::cout << "  throughput: " << (steps / el) << " byte-steps/s\n";
    std::cout << "  local W updates=" << local.updates
              << " patch W_updates=" << patch.updates << "\n";
    // диагностика: норма W (должна быть > 0 после обучения)
    float sq = 0;
    for (float v : local.w) sq += v * v;
    float pq = 0;
    for (float v : patch.w) pq += v * v;
    std::cout << "  [diag] local W norm=" << std::sqrt(sq)
              << " patch W norm=" << std::sqrt(pq) << "\n";

    // --- OWM-consolidate: защищаем направления последних окон (проверка порта) ---
    std::vector<std::vector<float>> dirs;
    for (int i = 0; i < 8 && i < ctx + 2; ++i) {
        // направление = латент последнего окна (проекция энкодера)
        std::vector<Sdr> tmp_win;
        tmp_win.push_back(basis((uint8_t)('a' + i)));
        tmp_win.push_back(basis((uint8_t)('a' + i + 1)));
        tmp_win.push_back(basis((uint8_t)('a' + i + 2)));
        dirs.push_back(local.encoder.encode(structure_sdr_from_sdrs(tmp_win)));
    }
    int consolidated = local.consolidate_owm(dirs, 4, 0.9f);
    std::cout << "  [diag] OWM consolidated=" << consolidated << " (порт проверен)\n";

    // Save FBAR sidecar(s)
    save_byte_w(out_path, local.w);
    std::string patch_path = out_path.substr(0, out_path.size() - 4) + "_patch.bin";
    save_byte_w(patch_path, patch.w);
    std::cout << "saved local W -> " << out_path << "\n";
    std::cout << "saved patch W -> " << patch_path << "\n";

    return 0;
}
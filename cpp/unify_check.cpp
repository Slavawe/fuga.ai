// unify_check.cpp — C++ читает FUGA1, написанный Rust (обратное направление).
// Проверяет числа: local[i]=0.5*i/N, patch[i]=-0.5*i/N, OWM-P diag=2.0.
#include <cstdio>
#include <string>
#include <vector>
#include "fuga_core.h"

using namespace fuga;

int main(int argc, char **argv) {
    std::string path = argc > 1 ? argv[1] : "/tmp/rust_written.fuga";
    std::vector<float> local_w, patch_w, owm_p, hjepa;
    UnifiedMeta meta;
    if (!load_unified(path, &local_w, &patch_w, &owm_p, &meta, &hjepa)) {
        std::printf("LOAD FAIL (magic не FUGA1?)\n");
        return 1;
    }
    std::printf("== C++ читает Rust-FUGA1: %s ==\n", path.c_str());
    std::printf("  LOCAL_W %zu | PATCH_W %zu | OWM_P %zu\n",
                local_w.size(), patch_w.size(), owm_p.size());
    std::printf("  META steps=%llu patch=%llu ctx=%u ver=%u\n",
                (unsigned long long)meta.steps, (unsigned long long)meta.patch_steps,
                meta.ctx, meta.version);
    if (local_w.size() >= 2 && patch_w.size() >= 2 && owm_p.size() >= 512) {
        std::printf("  local[0]=%.4f (ожид 0.0000)  local[1]=%.4f\n", local_w[0], local_w[1]);
        std::printf("  patch[0]=%.4f (ожид -0.0000) patch[last]=%.4f\n",
                    patch_w[0], patch_w[patch_w.size() - 1]);
        std::printf("  owm[0]=%.4f (ожид 2.0000)    owm[512+1]=%.4f\n",
                    owm_p[0], owm_p[513]);
    }
    return 0;
}
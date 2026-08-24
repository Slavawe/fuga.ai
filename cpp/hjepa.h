// hjepa.h — H-JEPA на C++ (порт hierarchical_jepa.rs, структурный эквивалент).
//
// Что портируется 1:1:
//   - 3 уровня: L0 ctx=4 stride=1, L1 ctx=3 stride=3, L2 ctx=2 stride=5
//   - predict_refined: L0-предсказание → L0-траектория (до ctx0+ctx1) →
//     L1-предсказание по окну траектории → err_traj (bind-коррекция) →
//     L2-коррекция с температурой → dampen_correction (L1 ⊕ L2-смесь)
//   - каждый уровень = свой LatentPredictor, обучение через learn_latent
//     (Widrow-Hoff латент→латент, как JEPA-уровни Rust предсказывают
//     гипервекторы из гипервекторов).
//
// converge: средняя cosine между температурными коррекциями
// (0..1; >0.3 = согласованные предсказания).
#ifndef FUGA_HJEPA_H
#define FUGA_HJEPA_H

#include <cmath>
#include <vector>
#include "fuga_core.h"

namespace fuga {

// Уровень JEPA: окно латентов → предсказанный латент.
struct JepaLevel {
    LatentPredictor lp;
    int ctx = 4;
    int stride = 1;

    // predict: окно латент-векторов → предсказанный латент.
    // Окно усредняется (суперпозиция VSA), нормализуется, W·x.
    std::vector<float> predict(const std::vector<std::vector<float>> &win) const {
        if (win.empty()) return std::vector<float>(LATENT_DIM, 0.0f);
        std::vector<float> acc(LATENT_DIM, 0.0f);
        for (const auto &v : win)
            for (int i = 0; i < LATENT_DIM; ++i) acc[i] += v[i];
        float n = 0;
        for (float v : acc) n += v * v;
        n = std::sqrt(n > 0 ? n : 1e-8f);
        for (float &v : acc) v /= n;
        return lp.apply_w(acc);
    }

    // learn: окно латентов → target латент (Widrow-Hoff в латентном).
    float learn(const std::vector<std::vector<float>> &win,
                const std::vector<float> &target, float lr) {
        if (win.empty()) return 0.0f;
        std::vector<float> acc(LATENT_DIM, 0.0f);
        for (const auto &v : win)
            for (int i = 0; i < LATENT_DIM; ++i) acc[i] += v[i];
        float n = 0;
        for (float v : acc) n += v * v;
        n = std::sqrt(n > 0 ? n : 1e-8f);
        for (float &v : acc) v /= n;
        return lp.learn_latent(acc, target, lr);
    }
};

// Коррекция L1-предсказания L2-поправкой (dampen_correction; lerp 0.5/0.5).
inline std::vector<float> dampen_correction_cpp(const std::vector<float> &l1,
                                                const std::vector<float> &l2) {
    std::vector<float> out(LATENT_DIM);
    for (int i = 0; i < LATENT_DIM; ++i) out[i] = 0.5f * l1[i] + 0.5f * l2[i];
    float n = 0;
    for (float v : out) n += v * v;
    n = std::sqrt(n > 0 ? n : 1e-8f);
    for (float &v : out) v /= n;
    return out;
}

struct HierarchicalJEPA {
    JepaLevel l0{ {}, 4, 1 };
    JepaLevel l1{ {}, 3, 3 };
    JepaLevel l2{ {}, 2, 5 };

    // predict_refined(context_latents, temps) → (коррекции, согласованность).
    std::pair<std::vector<std::vector<float>>, float>
    predict_refined(const std::vector<std::vector<float>> &context,
                    const std::vector<float> &temps) const {
        std::vector<std::vector<float>> corrected_list;
        float converge = 0.0f;
        if (context.size() < (size_t)l0.ctx) {
            return {corrected_list, 0.0f};
        }
        auto l0_pred = l0.predict(context);
        // L0-траектория до ctx0+ctx1
        std::vector<std::vector<float>> l0_traj = context;
        l0_traj.push_back(l0_pred);
        int needed = l0.ctx + l1.ctx;
        while ((int)l0_traj.size() < needed) {
            std::vector<std::vector<float>> w(l0_traj.end() - l0.ctx, l0_traj.end());
            l0_traj.push_back(l0.predict(w));
        }
        // L1 по последнему окну траектории
        std::vector<std::vector<float>> l1_win(l0_traj.end() - l1.ctx, l0_traj.end());
        auto l1_pred = l1.predict(l1_win);
        // error trajectory: bind-коррекция предсказаний (p·actual, знак)
        std::vector<std::vector<float>> err_traj;
        for (int i = 0; i + l1.ctx < (int)l0_traj.size(); ++i) {
            std::vector<std::vector<float>> w(l0_traj.begin() + i, l0_traj.begin() + i + l1.ctx);
            auto p = l1.predict(w);
            std::vector<float> bound(LATENT_DIM);
            for (int j = 0; j < LATENT_DIM; ++j) bound[j] = p[j] * l0_traj[i + l1.ctx][j];
            float n = 0;
            for (float v : bound) n += v * v;
            n = std::sqrt(n > 0 ? n : 1e-8f);
            for (float &v : bound) v /= n;
            err_traj.push_back(bound);
        }
        std::vector<float> final_bind(LATENT_DIM);
        for (int j = 0; j < LATENT_DIM; ++j) final_bind[j] = l1_pred[j] * l0_pred[j];
        float n = 0;
        for (float v : final_bind) n += v * v;
        n = std::sqrt(n > 0 ? n : 1e-8f);
        for (float &v : final_bind) v /= n;
        err_traj.push_back(final_bind);
        while ((int)err_traj.size() < l2.ctx) {
            err_traj.push_back(std::vector<float>(LATENT_DIM, 0.0f));
        }
        std::vector<std::vector<float>> l2_win(err_traj.end() - l2.ctx, err_traj.end());
        for (float t : temps) {
            auto l2_pred = l2.predict(l2_win);
            for (auto &v : l2_pred) v *= t;
            corrected_list.push_back(dampen_correction_cpp(l1_pred, l2_pred));
        }
        // согласованность коррекций (средняя cosine)
        int pairs = 0;
        for (size_t i = 0; i < corrected_list.size(); ++i)
            for (size_t j = i + 1; j < corrected_list.size(); ++j) {
                float dot = 0, na = 0, nb = 0;
                for (int k = 0; k < LATENT_DIM; ++k) {
                    dot += corrected_list[i][k] * corrected_list[j][k];
                    na += corrected_list[i][k] * corrected_list[i][k];
                    nb += corrected_list[j][k] * corrected_list[j][k];
                }
                converge += dot / (std::sqrt(na) * std::sqrt(nb));
                pairs += 1;
            }
        converge /= pairs > 0 ? (float)pairs : 1.0f;
        return {corrected_list, converge};
    }
};

} // namespace fuga

#endif // FUGA_HJEPA_H
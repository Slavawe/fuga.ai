
"""Кросс-модальное зацепление через Action-векторы (Этап 3 карты v2.0).

Гипотеза: текст заземляется не в пикселях (стена Этапа I), а в ВЕКТОРАХ
ТРАНСФОРМАЦИИ среды. Слово предсказывает действие; действие детерминированно
преобразует состояние. Верификация объективна: примени предсказанный
трансформ к S(t) — совпадёт ли с реальным S(t+1)?

Это проверка ПРИНЦИПА на синтетическом зацеплении; перенос на естественный
язык требует семантических энкодеров (см. Этап I).
"""

from __future__ import annotations

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from astral.astral_env import ScaledAstralEnvironment
from antitf.rust_bridge import packed_to_torch


def main():
    random.seed(0); torch.manual_seed(0)
    dim = 32768
    env = ScaledAstralEnvironment(vector_dim=dim)

    # Словарь концептов: 32 слова, каждое закреплено за действием (0..7)
    VOCAB = [f"концепт_{i}" for i in range(32)]
    word_to_action = {w: i % 8 for i, w in enumerate(VOCAB)}

    def hv_word(w):
        return packed_to_torch(np.asarray(env.binder.bind_batch([[w]])))[0].flatten()

    word_hvs = torch.stack([hv_word(w) for w in VOCAB])
    labels = torch.tensor([word_to_action[w] for w in VOCAB])

    class GroundingBridge(nn.Module):
        def __init__(self, dim, n_actions=8):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 1024), nn.LayerNorm(1024), nn.SiLU(),
                nn.Linear(1024, 512), nn.SiLU(),
                nn.Linear(512, n_actions))

        def forward(self, x):
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return self.net(x)

    bridge = GroundingBridge(dim)
    opt = torch.optim.Adam(bridge.parameters(), lr=1e-3)

    print("[phase 1] слово -> действие (зацепление через трансформацию)")
    for step in range(401):
        logits = bridge(word_hvs)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 400:
            acc = (logits.argmax(1) == labels).float().mean().item()
            print(f"  step {step}: ce={loss.item():.4f} action_acc={acc:.3f}")

    # ===== ВЕРИФИКАЦИЯ: язык управляет миром =====
    # Берём состояние среды, слово из словаря -> предсказанное действие ->
    # применяем ТРАНСФОРМАЦИЮ сами и сверяем с реальным шагом среды.
    print("\n[phase 2] language-steered verification")
    state = env.get_state()
    state["hv"] = state["hv"].flatten()
    exact = total = 0
    with torch.no_grad():
        for trial in range(200):
            w = random.choice(VOCAB)
            pred_a = int(bridge(hv_word(w).unsqueeze(0)).argmax())
            # применяем ПРЕДСКАЗАННУЮ трансформацию к текущему состоянию
            predicted_next = torch.sign(
                torch.roll(state["hv"], pred_a * 64) * 0.9 + 0.1)
            predicted_next[predicted_next == 0] = 1
            # реальный шаг среды с ИСТИННЫМ действием этого слова
            real = env.step_action(state["hv"], word_to_action[w])
            match = torch.equal(predicted_next.flatten(), real["hv"].flatten())
            exact += int(match); total += 1
            state = real
    print(f"  language->world exact match: {exact}/{total} = "
          f"{exact/max(total,1):.3f}")

    # контроль: случайные действия не должны совпадать так же часто
    random_match = 0
    state = env.get_state(); state["hv"] = state["hv"].flatten()
    with torch.no_grad():
        for _ in range(200):
            w = random.choice(VOCAB)
            wrong_a = random.randint(0, 7)
            if wrong_a == word_to_action[w]:
                wrong_a = (wrong_a + 1) % 8
            predicted_next = torch.sign(
                torch.roll(state["hv"], wrong_a * 64) * 0.9 + 0.1)
            predicted_next[predicted_next == 0] = 1
            real = env.step_action(state["hv"], word_to_action[w])
            random_match += int(torch.equal(
                predicted_next.flatten(), real["hv"].flatten()))
            state = real
    print(f"  control (wrong action): {random_match}/200 = "
          f"{random_match/200:.3f}")


if __name__ == "__main__":
    main()

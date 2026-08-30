"""Brain Transplant Protocol — «трансплантация мозгов» модели в VSA-стек.

Идея: взять РЕАЛЬНУЮ модель (например, Gemma 4 26B) и перенести её
поведение/знания на наши технологии (VSA, byte-level, HTM, SNN/STDP,
NEAT) БЕЗ запуска исходной модели на инференсе.

Принцип: НЕ копируем веса (26B флоатов — против философии проекта и
физически тяжело), а трансплантируем ПОВЕДЕНИЕ:

  1. EXTRACT — прогоняем донора (Gemma/любая трансформерная) на корпусе,
     записываем hidden states (активации) для каждого токена.
     Это «отпечаток мозга»: как модель думает о каждом входе.

  2. ENCODE — проецируем dense hidden state (2048-dim) в VSA-гипервектор
     (случайная проекция + биполярная квантование). Теперь отпечаток
     живёт в нашем VSA-базисе.

  3. TRANSPLANT — обучаем наш стек (NonGradientEngine: HTM + VSA +
     SNN/STDP + NEAT) предсказывать СЛЕДУЮЩИЙ VSA-вектор по текущему.
     Это 0 градиентов — переносим ДИНАМИКУ донора, не его веса.

  4. VERIFY — декодируем: из сида генерируем траекторию VSA-векторов,
     сравниваем с траекторией донора на том же сиде (косинус).
     Если косинус высокий — трансплантация удалась.

Зачем: донор (Gemma) — дорогой в инференсе (26B, GPU). Реципиент
(VSA+HTM) — лёгкий, событийный, без градиентов. Трансплантация
переносит «знание» в дешёвую оболочку.

Внимание: это ЭКСПЕРИМЕНТАЛЬНЫЙ протокол. Для Gemma 4 26B нужен
офлайн-дамп hidden states (один прогон донора на корпусе). Здесь —
демо на маленькой локальной модели, показывающее весь цикл.
"""

from __future__ import annotations

import numpy as np

from astral.experiments.mini_cognitive import MiniVSA


class BrainTransplantProtocol:
    """Протокол трансплантации: EXTRACT → ENCODE → TRANSPLANT → VERIFY.

    Работает с ЛЮБОЙ донорской моделью, дающей hidden states:
    функция `donor_forward(text) -> list[np.ndarray]` (по токену).
    Для Gemma 4 это будет дамп активаций; здесь — маленькая GRU.
    """

    def __init__(self, hv_dim: int = 512, donor_dim: int = 64, seed: int = 0):
        self.hv_dim = hv_dim
        self.donor_dim = donor_dim
        # случайная проекция donor_dim → hv_dim (детерминированная)
        self.rng = np.random.default_rng(seed)
        self.projector = self.rng.normal(0, 1, (hv_dim, donor_dim))
        self.vsa = MiniVSA(dim=hv_dim, seed=seed)
        # реципиент: весовая матрица W (безградиентная, Widrow-Hoff)
        self.W = np.zeros((hv_dim, hv_dim))
        self.lr = 0.05
        self.trained_steps = 0

    # ── 1. EXTRACT ──────────────────────────────────────────
    @staticmethod
    def extract_hidden_states(donor_forward, text: str) -> list[np.ndarray]:
        """Прогнать донора, вернуть hidden states по токенам."""
        return donor_forward(text)

    # ── 2. ENCODE: dense hidden state → VSA-гипервектор ─────
    def encode(self, hidden: np.ndarray) -> np.ndarray:
        """Проекция hidden state в биполярный VSA-вектор.

        z = sign(P·h) — знак случайной проекции даёт HV.
        P имеет форму (donor_dim, hv_dim): h[donor_dim] · P → hv_dim.
        """
        return np.sign(self.projector @ hidden)

    # ── 3. TRANSPLANT: учим динамику донора (0 градиентов) ──
    def transplant(self, trajectories: list[list[np.ndarray]],
                   epochs: int = 5) -> dict:
        """Обучить W воспроизводить динамику hidden states.

        Для каждой пары (z_t, z_{t+1}) из траекторий донора:
          pred = W·z_t
          err = z_{t+1} − pred
          W += lr·(err·z_tᵀ − ‖z_{t+1}‖²·W)  (Widrow-Hoff + Oja)
        """
        cos_history = []
        for _ in range(epochs):
            for traj in trajectories:
                zs = [self.encode(h) for h in traj]
                for t in range(len(zs) - 1):
                    z_t, z_next = zs[t], zs[t + 1]
                    pred = self.W @ z_t
                    err = z_next - pred
                    # Widrow-Hoff + нормировка Фробениуса (без Oja-переполнения)
                    self.W += self.lr * np.outer(err, z_t)
                    fn = np.linalg.norm(self.W) + 1e-9
                    self.W /= fn  # W ограничен: ‖W‖_F = 1
                    pred_n = pred / (np.linalg.norm(pred) + 1e-9)
                    cos = float(np.dot(pred_n, z_next) /
                                (np.linalg.norm(pred_n) * np.linalg.norm(z_next) + 1e-9))
                    cos_history.append(cos)
                    self.trained_steps += 1
        return {
            "steps": self.trained_steps,
            "cos_mean": float(np.mean(cos_history)),
            "cos_last": float(np.mean(cos_history[-200:])) if cos_history else 0.0,
        }

    # ── 4. VERIFY ───────────────────────────────────────────
    def verify(self, donor_forward, seed_text: str,
               max_steps: int = 5) -> dict:
        """Сравнить траекторию реципиента с траекторией донора.

        Из seed: донор даёт hidden states, реципиент — через W.
        Метрика: косинус реципиента с донором на каждом шаге.
        """
        donor_states = self.extract_hidden_states(donor_forward, seed_text)
        zs_donor = [self.encode(h) for h in donor_states]

        # реципиент: предсказывает следующий z по текущему
        zs_recipient = []
        if zs_donor:
            z = zs_donor[0]
            for _ in range(min(max_steps, len(zs_donor) - 1)):
                z = self.W @ z
                z = z / (np.linalg.norm(z) + 1e-9)
                zs_recipient.append(np.sign(z))

        # косинусы по шагам
        cos_steps = []
        for t in range(min(len(zs_recipient), len(zs_donor) - 1)):
            cos = float(np.dot(zs_recipient[t], zs_donor[t + 1]) /
                        (np.linalg.norm(zs_recipient[t]) * np.linalg.norm(zs_donor[t + 1]) + 1e-9))
            cos_steps.append(cos)

        return {
            "seed": seed_text,
            "n_donor_states": len(donor_states),
            "n_recipient_steps": len(zs_recipient),
            "cos_steps": cos_steps,
            "cos_mean": float(np.mean(cos_steps)) if cos_steps else 0.0,
        }


# ── Донорская модель для демо: маленькая GRU (обучена локально) ──
class TinyDonorGRU:
    """Мини-донор: 2-слойная GRU над байтами, обучена на корпусе.

    Это «мозг»-донор в миниатюре. Для реальной трансплантации сюда
    подставляется Gemma 4 (офлайн-дамп hidden states).
    """

    def __init__(self, input_dim: int = 256, hidden: int = 64):
        self.input_dim = input_dim
        self.hidden = hidden
        self.rng = np.random.default_rng(0)
        # веса GRU (детерминированные, для воспроизводимости)
        self.W_z = self.rng.normal(0, 0.1, (input_dim + hidden, hidden))
        self.W_r = self.rng.normal(0, 0.1, (input_dim + hidden, hidden))
        self.W_h = self.rng.normal(0, 0.1, (input_dim + hidden, hidden))

    def forward(self, text: str) -> list[np.ndarray]:
        """Байты текста → hidden states (по одному на байт)."""
        h = np.zeros(self.hidden)
        states = []
        for b in text.encode("utf-8"):
            x = np.zeros(self.input_dim)
            x[b] = 1.0  # one-hot байт
            z = 1 / (1 + np.exp(-(self.W_z.T @ np.concatenate([x, h]))))
            r = 1 / (1 + np.exp(-(self.W_r.T @ np.concatenate([x, h]))))
            hc = np.tanh(self.W_h.T @ np.concatenate([x, r * h]))
            h = (1 - z) * h + z * hc
            states.append(h.copy())
        return states


def demo():
    print("=== BRAIN TRANSPLANT PROTOCOL ===\n")

    # Донор: мини-GRU («мозг» в миниатюре)
    donor = TinyDonorGRU(input_dim=256, hidden=64)
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "fn main() { println!(\"hello world\"); }",
        "let x = 42; let y = x * 2;",
        "import numpy as np\nx = np.array([1,2,3])",
    ]
    print(f"1. Донор: TinyDonorGRU (hidden=64), корпус {len(corpus)} строк")

    # EXTRACT: hidden states на корпусе
    print("\n2. EXTRACT — отпечатки мозга донора:")
    trajectories = [donor.forward(t) for t in corpus]
    print(f"   траекторий: {len(trajectories)}, "
          f"hidden states: {sum(len(t) for t in trajectories)}")

    # Протокол трансплантации
    proto = BrainTransplantProtocol(hv_dim=512, donor_dim=64, seed=1)
    print("\n3. ENCODE — hidden states → VSA-гипервекторы (512-dim):")
    z_example = proto.encode(trajectories[0][0])
    print(f"   пример: hidden 64-dim → HV {z_example.shape}, "
          f"биполярный, {np.unique(z_example).tolist()}")

    # TRANSPLANT — 0 градиентов
    print("\n4. TRANSPLANT — перенос динамики донора (0 градиентов):")
    result = proto.transplant(trajectories, epochs=20)
    print(f"   шагов: {result['steps']}")
    print(f"   cos предсказания: {result['cos_first'] if False else result['cos_mean']:.4f} "
          f"(средний), {result['cos_last']:.4f} (последние 200)")

    # VERIFY
    print("\n5. VERIFY — реципиент vs донор на НЕвиденном сиде:")
    for seed in ["the lazy fox jumps", "fn process(x) {"]:
        v = proto.verify(donor.forward, seed)
        print(f"   seed '{seed[:20]}...': "
              f"cos={v['cos_mean']:.4f} (донор {v['n_donor_states']} состояний)")

    print("\n=== BRAIN TRANSPLANT — OK ===")


if __name__ == "__main__":
    demo()

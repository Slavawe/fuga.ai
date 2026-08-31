"""3D Language Agent — связка Point-JEPA (Rust) + CrossLingualVSA.

Единый агент, понимающий ПРОСТРАНСТВО (облака точек) и ЯЗЫК (RU/EN/ZH):

  Облако точек → Point-JEPA латент (Rust: point_cloud_encode)
  Словесное описание → CrossLingualVSA проекция → концепт-центроид

  Связка: концепт (шар/куб/плоскость/...) имеет ОБА представления:
    - латент формы (из точек)
    - концепт-центроид (из слов на 3 языках)

  Задачи агента:
    1. Узнай форму: облако точек → опиши словами (на любом языке)
    2. Найди форму: словесное описание → сгенерируй/найди облако
    3. Предскажи: следующая форма во времени → опиши словами
    4. Zero-shot: учитель знает форму на RU, ученик спрашивает на ZH

Всё на Widrow-Hoff/VSA — без градиентов (философия проекта).
"""

from __future__ import annotations

import numpy as np

from astral.experiments.cross_lingual_vsa import CrossLingualVSA
from astral.experiments.point_jepa import PointCloudEncoder, PointJEPA, make_shape


# Русское название формы → тройка переводов (для VSA-проекции)
SHAPE_TRANSLATIONS = {
    "шар": ("шар", "sphere", "球"),
    "куб": ("куб", "cube", "立方体"),
    "плоскость": ("плоскость", "plane", "平面"),
    "линия": ("линия", "line", "线"),
    "кластер": ("кластер", "cluster", "簇"),
}

# Соответствие формы и её ключа для make_shape
SHAPE_KEYS = {"шар": "sphere", "куб": "cube", "плоскость": "plane",
              "линия": "line", "кластер": "cluster"}


class ThreeDLanguageAgent:
    """Агент: пространство (Point-JEPA) + язык (CrossLingualVSA).

    Контракт:
      learn_shapes(n_points)        — собрать латенты форм
      describe_point_cloud(points)  — форма → словесное описание (lang)
      find_shape(description, lang) — описание → форма (латент/облако)
      predict_next(cloud)           — предсказание следующей формы
    """

    def __init__(self, dim: int = 512, seed: int = 42):
        self.dim = dim
        self.seed = seed
        # Rust-флаги (если доступны)
        try:
            import fuga_core
            self._rust = fuga_core if hasattr(fuga_core, "point_cloud_encode") else None
        except Exception:
            self._rust = None

        self.pj_encoder = PointCloudEncoder(dim=dim, seed=seed)
        self.pj = PointJEPA(dim=dim, lr=0.05)
        self.clvsa = CrossLingualVSA(dim=dim, seed=seed)

        # Латент формы (ключ: 'шар', 'куб', ...)
        self.shape_latents: dict[str, np.ndarray] = {}
        # Концепт-центроид формы (из CrossLingualVSA)
        self.shape_concepts: dict[str, np.ndarray] = {}

    # ---- обучение связки ----
    def learn_shapes(self, n_points: int = 64) -> None:
        """Собрать латенты форм (Point-JEPA) + концепт-центроиды (VSA)."""
        rng = np.random.default_rng(self.seed)

        # Point-JEPA: латенты форм
        for ru_name, key in SHAPE_KEYS.items():
            cloud = make_shape(key, n=n_points, rng=rng)
            self.shape_latents[ru_name] = self._encode_cloud(cloud)

        # CrossLingualVSA: проекции (обучение на переводах форм)
        pairs = [tuple(SHAPE_TRANSLATIONS[k]) for k in SHAPE_KEYS]
        self.clvsa.train_pairs(pairs, epochs=30, lr=0.1)
        for ru_name in SHAPE_KEYS:
            ru, en, zh = SHAPE_TRANSLATIONS[ru_name]
            # концепт-центроид = проекция RU (все три должны совпадать)
            self.shape_concepts[ru_name] = self.clvsa.project(ru, "RU")

    # ---- кодирование ----
    def _encode_cloud(self, cloud: np.ndarray) -> np.ndarray:
        """Облако точек → латент (Rust, если доступен, иначе Python)."""
        if self._rust is not None:
            omega = np.stack([self.pj_encoder.omega["x"],
                              self.pj_encoder.omega["y"],
                              self.pj_encoder.omega["z"]])
            return np.asarray(
                self._rust.point_cloud_encode(
                    np.ascontiguousarray(cloud, dtype=np.float32),
                    np.ascontiguousarray(omega, dtype=np.float32),
                )
            )
        return self.pj_encoder.encode(cloud)

    # ---- 1. Узнай форму: облако → словесное описание ----
    def describe_point_cloud(self, cloud: np.ndarray, lang: str = "RU") -> tuple[str, float]:
        """Облако точек → название формы (ближайший латент)."""
        lat = self._encode_cloud(cloud)
        best_name, best_cos = None, -1.0
        for name, shape_lat in self.shape_latents.items():
            c = float(np.dot(lat, shape_lat) / (
                np.linalg.norm(lat) * np.linalg.norm(shape_lat) + 1e-9))
            if c > best_cos:
                best_name, best_cos = name, c
        # Переводим на нужный язык
        ru, en, zh = SHAPE_TRANSLATIONS[best_name]
        word = {"RU": ru, "EN": en, "ZH": zh}[lang]
        return word, best_cos

    # ---- 2. Найди форму: словесное описание → облако ----
    def find_shape(self, description: str, lang: str) -> tuple[str, float]:
        """Словесное описание → название формы (через концепт-пространство)."""
        # Проекция описания в концепт-пространство
        desc_proj = self.clvsa.project(description, lang)
        best_name, best_cos = None, -1.0
        for name, concept in self.shape_concepts.items():
            c = float(np.dot(desc_proj, concept) / (
                np.linalg.norm(desc_proj) * np.linalg.norm(concept) + 1e-9))
            if c > best_cos:
                best_name, best_cos = name, c
        return best_name, best_cos

    # ---- 3. Предскажи следующую форму ----
    def predict_next(self, cloud: np.ndarray) -> tuple[str, np.ndarray]:
        """Предсказание следующего облака: латент → ближайшая форма."""
        lat = self._encode_cloud(cloud)
        pred = self.pj.predict_next_rust(self._rust, lat) if self._rust else self.pj.predict_next(cloud)
        best_name, best_cos = None, -1.0
        for name, shape_lat in self.shape_latents.items():
            c = float(np.dot(pred, shape_lat) / (
                np.linalg.norm(pred) * np.linalg.norm(shape_lat) + 1e-9))
            if c > best_cos:
                best_name, best_cos = name, c
        return best_name, pred

    # ---- обучение предиктора переходов ----
    def train_transitions(self, seq: list[np.ndarray], epochs: int = 30) -> float:
        """Обучение Point-JEPA предиктора на последовательности облаков."""
        return self.pj.train(seq, epochs=epochs)

    # ---- 4. Zero-shot учитель ----
    def teacher_verify(self, hypothesis: str, hyp_lang: str,
                       axiom: str, axiom_lang: str) -> tuple[bool, float]:
        """Учитель (знает форму на одном языке) верифицирует на другом."""
        return self.clvsa.teacher_verify(hypothesis, hyp_lang, axiom, axiom_lang)


def demo():
    print("═" * 64)
    print("3D LANGUAGE AGENT: Point-JEPA (Rust) + CrossLingualVSA")
    print("═" * 64)

    agent = ThreeDLanguageAgent(dim=512)
    agent.learn_shapes(n_points=64)

    # Rust-ядро?
    print(f"\n[0] Rust-ядро Point-JEPA: {'АКТИВНО ✓' if agent._rust is not None else 'нет (Python)'}")

    rng = np.random.default_rng(7)

    # 1. Узнай форму (облако → слово на 3 языках)
    print("\n[1] РАСПОЗНАВАНИЕ ФОРМ (облако → слово):")
    for key, ru_name in [("sphere", "шар"), ("cube", "куб"), ("plane", "плоскость"),
                         ("cluster", "кластер")]:
        cloud = make_shape(key, n=64, rng=rng)
        word_ru, c_ru = agent.describe_point_cloud(cloud, "RU")
        word_en, c_en = agent.describe_point_cloud(cloud, "EN")
        word_zh, c_zh = agent.describe_point_cloud(cloud, "ZH")
        ok = word_ru == ru_name
        print(f"  {'✓' if ok else '✗'} {ru_name:10s} → RU:{word_ru} EN:{word_en} ZH:{word_zh} "
              f"(cos={c_ru:.2f})")

    # 2. Найди форму (слово на 3 языках → форма)
    print("\n[2] ПОИСК ФОРМЫ ПО ОПИСАНИЮ (слово → форма):")
    for lang in ["RU", "EN", "ZH"]:
        for ru_name, (ru, en, zh) in SHAPE_TRANSLATIONS.items():
            word = {"RU": ru, "EN": en, "ZH": zh}[lang]
            found, c = agent.find_shape(word, lang)
            mark = "✓" if found == ru_name else "✗"
            print(f"  [{lang}] {word:8s} → {found} {mark} (cos={c:.2f})")

    # 3. Предсказание следующей формы
    print("\n[3] ПРЕДСКАЗАНИЕ ПЕРЕХОДОВ (время):")
    # Последовательность: шар→куб→плоскость→линия→кластер→шар→...
    shapes_seq = ["sphere", "cube", "plane", "line", "cluster"]
    clouds = [make_shape(s, n=64, rng=rng) for s in shapes_seq]
    seq = [clouds[i % len(shapes_seq)] for i in range(15)]
    loss = agent.train_transitions(seq, epochs=30)
    print(f"  loss предиктора: {loss:.3f}")
    for i in range(min(5, len(seq) - 1)):
        pred_name, _ = agent.predict_next(seq[i])
        actual_name = shapes_seq[(i + 1) % len(shapes_seq)]
        ru_actual = [k for k, v in SHAPE_KEYS.items() if v == actual_name][0]
        mark = "✓" if pred_name == ru_actual else "✗"
        print(f"  {shapes_seq[i]:8s} → предсказано: {pred_name:10s} "
              f"(факт: {ru_actual}) {mark}")

    # 4. Zero-shot учитель
    print("\n[4] ZERO-SHOT УЧИТЕЛЬ (RU → EN/ZH):")
    for ru_name, (ru, en, zh) in SHAPE_TRANSLATIONS.items():
        ok, c = agent.teacher_verify(en, "EN", ru, "RU")
        ok2, c2 = agent.teacher_verify(zh, "ZH", ru, "RU")
        print(f"  {ru_name:10s}: EN cos={c:.3f} {'✓' if ok else '✗'}, "
              f"ZH cos={c2:.3f} {'✓' if ok2 else '✗'}")

    print("\n=== 3D LANGUAGE AGENT OK ===")


if __name__ == "__main__":
    demo()
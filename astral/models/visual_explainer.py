"""VisualExplainer — наглядное объяснение результатов модели.

По заказу: «чтобы он мог объяснять как и показывать — не только
текст, но и изображения, чтобы было наглядно».

Возможности:
  1. VSA-пространство: матрица косинусов концептов (тепловая карта)
  2. BLT-патчи: границы патчей, энтропия, предсказания
  3. Кросс-языковая карта: RU/EN/ZH концепты clustered
  4. Геометрия: точки, кривые, геометрический медиан
  5. TTS: озвучивание объяснения (speak)

Архитектура (SKILL 2: APALL):
  VisualExplainer (Application) → matplotlib (Infrastructure)
                                   → tempfile (Infrastructure)
                                   → text_to_speech (optional, Hermes tool)
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # без GUI
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors


class VisualExplainer:
    """Визуальное объяснение результатов модели.

    Генерирует PNG-изображения в /tmp/fuga_vis/.
    """

    def __init__(self, out_dir: str = "/tmp/fuga_vis"):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        # Единая цветовая схема (тёмная тема)
        self.bg_color = "#1a1a2e"
        self.text_color = "#e0e0e0"
        self.accent = "#00d4ff"
        self.accent2 = "#ff6b6b"
        self.accent3 = "#ffd93d"

    def _path(self, name: str) -> str:
        return os.path.join(self.out_dir, name)

    # ================================================================
    # 1. VSA-ПРОСТРАНСТВО: тепловая карта косинусов концептов
    # ================================================================
    def vsa_concept_map(self, concepts: list[str],
                        sim_matrix: np.ndarray,
                        title: str = "VSA-пространство концептов") -> str:
        """Тепловая карта косинусного сходства между концептами.

        Args:
            concepts: названия концептов
            sim_matrix: N×N матрица косинусов
            title: заголовок

        Returns: путь к PNG
        """
        fig, ax = plt.subplots(figsize=(max(8, len(concepts) * 0.6),
                                        max(6, len(concepts) * 0.5)))
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.bg_color)

        n = len(concepts)
        im = ax.imshow(sim_matrix, cmap="RdYlBu_r", vmin=-1, vmax=1,
                       aspect="equal")

        # Цветовые метки
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(concepts, rotation=45, ha="right",
                           fontsize=9, color=self.text_color)
        ax.set_yticklabels(concepts, fontsize=9, color=self.text_color)

        # Значения на клетках
        for i in range(n):
            for j in range(n):
                val = sim_matrix[i, j]
                color = "white" if abs(val) < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

        ax.set_title(title, color=self.text_color, fontsize=12, pad=12)
        fig.colorbar(im, ax=ax, label="косинус", shrink=0.8)

        path = self._path(f"vsa_concept_{datetime.now().strftime('%H%M%S')}.png")
        plt.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight",
                    facecolor=self.bg_color)
        plt.close(fig)
        return path

    # ================================================================
    # 2. BLT-ПАТЧИ: визуализация границ и энтропии
    # ================================================================
    def blt_patches(self, text: str, offsets: list[int],
                    entropies: Optional[list[float]] = None,
                    title: str = "BLT-патчи: границы по энтропии") -> str:
        """Визуализация BLT-патчей — границы, энтропия, содержимое.

        Args:
            text: исходный текст
            offsets: границы патчей [0, 3, 7, 12, ...]
            entropies: энтропия на границе (опционально)
            title: заголовок

        Returns: путь к PNG
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.bg_color)

        # Текст как "дорожка"
        chars = list(text[:80])
        n = len(chars)
        ax.set_xlim(-0.5, n + 0.5)
        ax.set_ylim(-1, 4)

        # Патчи — цветные блоки
        colors = plt.cm.tab10(np.linspace(0, 1, len(offsets) - 1))
        for i in range(len(offsets) - 1):
            s, e = offsets[i], min(offsets[i + 1], n)
            ax.axvspan(s, e, alpha=0.2, color=colors[i % 10], zorder=1)
            if s < n:
                ax.annotate(f"P{i}", (s + (e - s) / 2, 3),
                           ha="center", fontsize=8, color=self.accent)

        # Символы
        for i, ch in enumerate(chars):
            ax.text(i, 1.5, ch, ha="center", va="center",
                   fontsize=11, color=self.text_color,
                   bbox=dict(boxstyle="round,pad=0.1",
                            facecolor="#333344", edgecolor="none"))

        # Границы патчей (вертикальные линии)
        for off in offsets:
            if off <= n:
                ax.axvline(x=off, color=self.accent2, linewidth=1.2,
                          linestyle="--", alpha=0.7)

        # Энтропия (если есть)
        if entropies and len(entropies) == len(offsets):
            ax2 = ax.twinx()
            ax2.plot(range(len(offsets)), entropies, color=self.accent3,
                    marker="o", linewidth=1.5, alpha=0.8, markersize=4)
            ax2.set_ylabel("энтропия", color=self.accent3, fontsize=9)
            ax2.tick_params(axis="y", labelcolor=self.accent3)

        ax.set_title(title, color=self.text_color, fontsize=11, pad=10)
        ax.set_xlabel("позиция (байты)", color=self.text_color)
        ax.tick_params(colors=self.text_color)
        ax.set_yticks([])

        path = self._path(f"blt_patches_{datetime.now().strftime('%H%M%S')}.png")
        plt.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight",
                   facecolor=self.bg_color)
        plt.close(fig)
        return path

    # ================================================================
    # 3. КРОСС-ЯЗЫКОВАЯ КАРТА: RU/EN/ZH концепты
    # ================================================================
    def cross_lingual_map(self, concepts: dict[str, dict[str, str]],
                          cos_matrix: np.ndarray,
                          title: str = "Кросс-языковая карта концептов") -> str:
        """Визуализация соответствия концептов в 3 языках.

        Args:
            concepts: {concept: {RU: str, EN: str, ZH: str}}
            cos_matrix: (3*N × 3*N) матрица косинусов между всеми парами
            title: заголовок

        Returns: путь к PNG
        """
        names = list(concepts.keys())
        langs = ["RU", "EN", "ZH"]
        n = len(names)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor(self.bg_color)

        for li, lang in enumerate(langs):
            ax = axes[li]
            ax.set_facecolor(self.bg_color)
            # Срез матрицы: n×n (родной vs родной)
            start = li * n
            block = cos_matrix[start:start + n, start:start + n]
            im = ax.imshow(block, cmap="RdYlBu_r", vmin=-1, vmax=1,
                          aspect="equal")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(names, rotation=45, ha="right",
                              fontsize=7, color=self.text_color)
            ax.set_yticklabels(names, fontsize=7, color=self.text_color)
            ax.set_title(f"{lang} → {lang}", color=self.text_color,
                        fontsize=10)

        fig.suptitle(title, color=self.text_color, fontsize=12, y=1.02)
        fig.colorbar(im, ax=axes, label="косинус", shrink=0.6)

        path = self._path(f"cross_lingual_{datetime.now().strftime('%H%M%S')}.png")
        plt.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight",
                   facecolor=self.bg_color)
        plt.close(fig)
        return path

    # ================================================================
    # 4. ГЕОМЕТРИЯ: точки, кривые, решения
    # ================================================================
    def geometry(self, points: list[tuple[float, float]],
                 title: str = "Геометрическая задача",
                 highlight: Optional[tuple[float, float]] = None,
                 curve_eq: Optional[str] = None,
                 curve_x: Optional[np.ndarray] = None,
                 curve_y: Optional[np.ndarray] = None) -> str:
        """Визуализация геометрической задачи.

        Args:
            points: точки (x, y)
            title: заголовок
            highlight: точка-оптимум (если есть)
            curve_eq: подпись кривой
            curve_x, curve_y: точки кривой

        Returns: путь к PNG
        """
        fig, ax = plt.subplots(figsize=(8, 7))
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor("#16213e")

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        # Точки
        ax.scatter(xs, ys, c=self.accent, s=120, zorder=5,
                  edgecolors="white", linewidth=1, label="вершины")

        # Подписи точек
        for i, (x, y) in enumerate(points):
            ax.annotate(f"  {chr(65 + i)}({x},{y})", (x, y),
                       fontsize=9, color=self.text_color,
                       ha="left", va="bottom")

        # Кривая (эллиптическая / функция)
        if curve_x is not None and curve_y is not None:
            ax.plot(curve_x, curve_y, color=self.accent2, linewidth=1.5,
                   alpha=0.7, label=curve_eq or "кривая")

        # Highlight (оптимум)
        if highlight:
            hx, hy = highlight
            ax.scatter([hx], [hy], c=self.accent3, s=200, zorder=6,
                      marker="*", edgecolors="white", linewidth=1.5,
                      label="оптимум")
            ax.annotate(f"P({hx:.2f}, {hy:.2f})", (hx, hy),
                       fontsize=10, color=self.accent3,
                       ha="left", va="bottom",
                       bbox=dict(facecolor="#1a1a2e", alpha=0.7,
                                edgecolor=self.accent3))

        ax.set_title(title, color=self.text_color, fontsize=12, pad=10)
        ax.set_xlabel("x", color=self.text_color)
        ax.set_ylabel("y", color=self.text_color)
        ax.tick_params(colors=self.text_color)
        ax.legend(loc="upper right", fontsize=8,
                 facecolor="#1a1a2e", edgecolor=self.text_color,
                 labelcolor=self.text_color)
        ax.grid(alpha=0.15, color=self.text_color)
        ax.set_aspect("equal")

        path = self._path(f"geometry_{datetime.now().strftime('%H%M%S')}.png")
        plt.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight",
                   facecolor=self.bg_color)
        plt.close(fig)
        return path

    # ================================================================
    # 5. TTS: озвучивание объяснения
    # ================================================================
    def speak(self, text: str, lang: str = "ru") -> None:
        """Озвучивание объяснения.

        Сохраняет .wav через espeak (если доступен), иначе — только
        печать. В Hermes-агенте: text_to_speech(text=text).
        """
        # Проверка: espeak доступен?
        wav = os.path.join(self.out_dir, "explanation.wav")
        try:
            import subprocess
            subprocess.run(
                ["espeak", "-v", lang, "-w", wav, text],
                capture_output=True, timeout=30,
            )
            if os.path.exists(wav) and os.path.getsize(wav) > 100:
                print(f"  🔊 Аудио: {wav}")
        except Exception:
            print(f"  🔊 TTS: «{text[:60]}...» (espeak не установлен)")

    # ================================================================
    # 6. УНИВЕРСАЛЬНЫЙ ВЫВОД: всё вместе
    # ================================================================
    def explain(self, analysis: dict, output_type: str = "text") -> str:
        """Универсальное объяснение: текст + изображение + опционально TTS.

        Args:
            analysis: словарь с результатами анализа
            output_type: "text" | "image" | "both" | "speak"

        Returns: путь к главному PNG (если есть)
        """
        result = ""
        # Текстовое объяснение
        if output_type in ("text", "both"):
            print(f"\n{'═' * 60}")
            print(f"  {analysis.get('title', 'Анализ')}")
            print(f"{'═' * 60}")
            for line in analysis.get("explanation", []):
                print(f"  {line}")
            print()

        # Изображение
        if output_type in ("image", "both"):
            img = analysis.get("image_path")
            if img and os.path.exists(img):
                result = img
                print(f"  📊 Изображение: {img}")
            else:
                print("  ⚠ Нет изображения для этого анализа")

        # TTS
        if output_type == "speak":
            text = " ".join(analysis.get("explanation", []))
            if text:
                self.speak(text)

        return result


def demo():
    """Демонстрация VisualExplainer на всех типах визуализации."""
    ve = VisualExplainer()
    print("═══ VisualExplainer: наглядное объяснение ═══\n")

    # 1. VSA-пространство
    concepts = ["данные", "матрица", "гипотеза", "граф", "память"]
    rng = np.random.RandomState(42)
    sim = np.clip(rng.randn(5, 5) * 0.3 + 0.5, -1, 1)
    np.fill_diagonal(sim, 1.0)
    p1 = ve.vsa_concept_map(concepts, sim)
    print(f"  [1] VSA-пространство → {p1}")

    # 2. BLT-патчи
    text = "connected directed graph is a fundamental concept"
    offsets = [0, 4, 9, 13, 18, 22, 27, 30, 34, 38, 41, 46, 49, 53]
    entropies = [3.2, 2.1, 1.8, 2.5, 1.2, 3.0, 0.8, 2.2, 1.5, 2.8, 1.1, 2.4, 1.9]
    p2 = ve.blt_patches(text, offsets, entropies)
    print(f"  [2] BLT-патчи → {p2}")

    # 3. Кросс-языковая карта
    concepts_dict = {
        "данные": {"RU": "данные", "EN": "data", "ZH": "数据"},
        "граф": {"RU": "граф", "EN": "graph", "ZH": "图"},
        "функция": {"RU": "функция", "EN": "function", "ZH": "函数"},
    }
    n = 3
    # имитация матрицы 9×9 (3 языка × 3 концепта)
    cos_mat = np.eye(9)
    for i in range(3):
        for j in range(3):
            if i == j:
                cos_mat[i, i] = 1.0
                cos_mat[i + 3, i + 3] = 1.0
                cos_mat[i + 6, i + 6] = 1.0
                # cross-lingual для одного концепта
                cos_mat[i, i + 3] = cos_mat[i + 3, i] = 0.95
                cos_mat[i, i + 6] = cos_mat[i + 6, i] = 0.95
                cos_mat[i + 3, i + 6] = cos_mat[i + 6, i + 3] = 0.95
    p3 = ve.cross_lingual_map(concepts_dict, cos_mat)
    print(f"  [3] Кросс-языковая карта → {p3}")

    # 4. Геометрия
    pts = [(0, 4), (4, 4), (4, 0), (1, 1)]
    # Эллиптическая кривая y² = x³ - 7
    x_vals = np.linspace(2, 35, 500)
    y_vals = np.sqrt(np.clip(x_vals ** 3 - 7, 0, None))
    p4 = ve.geometry(pts, title="Геометрический медиан + эллиптическая кривая",
                     highlight=(2.0, 2.0),
                     curve_eq="y² = x³ − 7",
                     curve_x=x_vals, curve_y=y_vals)
    print(f"  [4] Геометрия → {p4}")

    print(f"\n  Все изображения: {ve.out_dir}/")
    print("  ═══ VisualExplainer OK ═══\n")


if __name__ == "__main__":
    demo()
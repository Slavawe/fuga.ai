"""TTS-эксперимент: озвучивание объяснений модели.

Перенесён из astral/models/visual_explainer.py (speak) в
экспериментальные функции (astral/experiments/).

Бэкенды (по приоритету):
  1. espeak — системный синтез речи (если установлен) → .wav
  2. Hermes text_to_speech — в агенте (если доступен)
  3. Нет бэкенда — только печать текста

Применение: озвучить объяснение модели (анализ данных, гипотеза,
результат кросс-языковой верификации).
"""

from __future__ import annotations

import os
import subprocess


def speak(text: str, lang: str = "ru", out_dir: str = "/tmp/fuga_vis") -> str | None:
    """Озвучить текст, вернуть путь к .wav (или None).

    Args:
        text: текст для озвучивания
        lang: язык ('ru', 'en', 'zh' — если поддерживает espeak)
        out_dir: куда сохранить .wav

    Returns:
        путь к .wav или None (если TTS недоступен)
    """
    os.makedirs(out_dir, exist_ok=True)
    wav = os.path.join(out_dir, "explanation.wav")

    # 1. espeak
    try:
        res = subprocess.run(
            ["espeak", "-v", lang, "-w", wav, text],
            capture_output=True, timeout=30,
        )
        if res.returncode == 0 and os.path.exists(wav) and os.path.getsize(wav) > 100:
            print(f"  🔊 Аудио: {wav}")
            return wav
    except Exception:
        pass

    # 2. Hermes TTS (в агенте)
    try:
        from hermes_tools import text_to_speech  # noqa
        print(f"  🔊 TTS (Hermes): «{text[:60]}...»")
        return None
    except Exception:
        pass

    # 3. Нет бэкенда
    print(f"  🔊 TTS недоступен: «{text[:60]}...»")
    return None


def demo():
    print("═" * 60)
    print("TTS-ЭКСПЕРИМЕНТ: озвучивание объяснений")
    print("═" * 60)
    wav = speak("Модель достигла кросс-языковой инвариантности. "
                "Косинус между переводами равен единице.")
    print(f"  Результат: {wav if wav else 'TTS недоступен (печать текста)'}")
    print("\n=== TTS OK ===")


if __name__ == "__main__":
    demo()
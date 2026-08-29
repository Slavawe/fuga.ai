# ROADMAP — Единая дорога обучения Fuga (29.08.2026)

> Связывание всех лучших решений проекта в ОДНУ дорогу обучения.
> Каждый шаг = проверенный A/B-результат из AGENTS.md + новая технология сессии.

## Принцип: Трёхуровневая иерархия + концепт-ось

```
  КОНЦЕПТ (lang-jepa, EMA-таргет)      ← ЧТО сказать (смысл)
       ↓  z_{t+1} = Predictor(z_t)
  МАКРО (W_macro, Byte-H-JEPA)         ← СТРУКТУРА (ветвление графа)
       ↓  score += βm·cos(W_macro·x, lat)
  ПАТЧ (W_patch, two-speed)            ← КАРКАС (AST-синтаксис)
       ↓  score += βp·cos(W_patch·x, lat)
  БАЙТ (W_local, Widrow-Hoff)          ← СИМВОЛЫ (точные имена)
       ↓  argmax cos(W·x) + rep_word/rep_phrase
  ДЕКОД (V2/MB3 + резонаторы)          ← ЭМИССИЯ (текст/код)
```

## Шаг 1. Чистый корпус (сделано, 29.08)
- **Проблема**: смешанный RU/EN корпус размывал биграммы → мусор на выходе
- **Решение**: `/tmp/clean_corpus2.jsonl` (5000 строк: 3000 EN-текст + 2000 Rust-код)
- **Результат**: naive 120B связного английского (`related to the struct re`),
  код-сид → `let` — vs 1B мусора на смешанном
- **Следующий шаг**: собрать БОЛЬШОЙ чистый корпус (100K+ строк кода + текста)

## Шаг 2. Обучение (главный путь: unified_gpu_train)
```bash
systemd-inhibit --what=sleep ./target/release/unified_gpu_train \
  --jsonl "/tmp/clean_corpus2.jsonl" \
  --max-steps 1000000 \
  --ctx 8 \
  --lambda-patch 0.4 \
  --lambda-floor 0.10 \
  --lambda-tau 500000 \
  --out /tmp/clean_1M.fuga \
  --ckpt-every 250000
```
- Проверенные параметры: ctx=8, λ-curriculum 0.4→0.1 (τ=500K), early stop 1M-1.2M
- OWM защита: consolidate каждые N шагов (16+ направлений)
- GPU: GTX 1660 Ti ~1200-1800 pairs/s

## Шаг 3. Концепт-канал (lang-jepa + FUGA1 tag=8)
```python
# Обучить концепт-предиктор (EMA-таргет, smooth-L1)
python astral/train_langjepa.py  # → /tmp/langjepa_vault.pt

# Сериализация в FUGA1 (вставка перед END)
python astral/fuga1_concept.py --fuga /tmp/clean_1M.fuga \
  --pt /tmp/langjepa_vault.pt --out /tmp/clean_1M_concept.fuga
```
- Результат: концепт = смысл следующего предложения (не токен)
- Мост к декодеру: `astral/langjepa_mb3_bridge.py`

## Шаг 4. Декодирование (V2 + MB3 + концепт-приор)
```
score = cos(W·x)                                    # байтовый
      + βp·cos(W_patch·x_patch, lat_patch)          # патчевый
      + βm·cos(W_macro·x_ctx, lat_patch)            # макро
      + βc·cos(concept, lat_patch)                  # концепт (lang-jepa)
      − rep_word·count(word)  − rep_phrase·count(phrase)
```
- v6.2 калибровка: rep_word=0.20, rep_phrase=0.8, PHR_LEN=12, window=9
- MB3: top_k=8, β=0.3, ws_pen=0.30, conf_th=0.05
- Инструмент: `v6_validate <ckpt> <corpus>` — полная конфигурация

## Шаг 5. Резонаторы как память-услуга (HDC/FPE/PhaseCrystal)
```
S = X1⊗X2⊗...⊗XN (суперпозиция)
  → HDCResonator: N-факторное разложение (72% N=2, 63% N=3)
  → FPEVSA: фазовый резонанс (88% N=2) + дробные степени
  → PhaseCrystal: фазовые веса (смешение до схлопывания)
```
- Роль: НЕ замена VSA, а слой разложения/извлечения из памяти
- Применение: recall концептов, комбинаторное изобретение новых пар
- Интеграция: UnifiedEngine.recall / combine

## Шаг 6. Анти-коллапс (Barlow Twins)
- barlow_loss(z_a, z_b, λ=0.005) — кросс-корреляция → identity
- Применение: в JEPA-тренинге (z_pred vs z_EMA) — предотвращает коллапс
- В отличие от VICReg: 1 гиперпараметр вместо 3
- Интеграция: UnifiedEngine.train_anti_collapse(texts)

## Шаг 7. Единый движок (UnifiedEngine)
```python
from astral.models.unified_engine import UnifiedEngine
eng = UnifiedEngine(dim=2048)
eng.memorize("force", "gravity")      # память: слово → концепт
eng.speak("the force of gravity is")  # речь:  концепт-цепочка
eng.codegen("fn", length=5)           # код:   каркас через резонанс
eng.combine(text_seed, code_seed)     # комбо: слово → код
eng.train_anti_collapse(texts)        # анти-коллапс
```

## Шаг 8. Валидация (A/B)
```bash
# Декодеры: v2 vs MB3 vs MB3+concept на одних чекпоинтах
./target/release/v6_validate /tmp/clean_1M.fuga /tmp/clean_corpus2.jsonl
python astral/langjepa_mb3_bridge.py /tmp/clean_1M_concept.fuga

# Метрики
python3 src/bin/name_metric.py <ckpt>     # точность имён
# C-AST errors (tree-sitter)              # валидность кода
# Длина генерации (200B бюджет)
```

## Приоритеты (что даёт больше всего)

| # | Действие | Эффект | Риск |
|---|----------|--------|------|
| 1 | Большой чистый корпус (100K+ строк) | Связные слова+код | НИЗКИЙ |
| 2 | Прогон 1M-1.2M (ранняя остановка) | Длина+синтаксис | НИЗКИЙ |
| 3 | Концепт-канал в декодер (βc>0) | Смысловая связность | СРЕДНИЙ |
| 4 | Резонаторы в pipeline | Извлечение из памяти | СРЕДНИЙ |
| 5 | Feature-gating bin-стендов | Чистота сборки | НИЗКИЙ |

## Архитектурные уроки (не повторять)
1. Смешанный RU/EN корпус размывает биграммы — ТОЛЬКО одноязычный/кодовый
2. Окно декодера ДОЛЖНО совпадать с окном обучения (window=ctx+1)
3. Единый энкодер-базис для W и W_patch (seed 0xF03D_C0DE)
4. LMS (err = t − W·x), НЕ Hebb (err ≈ t) — иначе частотный шум
5. sync_channel везде (mpsc → OOM)
6. systemd-inhibit для длинных прогонов (suspend убивает процесс)

## Состояние (29.08.2026)
- [x] Шаг 1: чистый корпус + тест 100K/500K
- [x] Шаг 3: lang-jepa адаптер + FUGA1 tag=8 (обучен на vault)
- [x] Шаг 5: HDC/FPE/PhaseCrystal (пруф: 72/88/100%)
- [x] Шаг 6: Barlow Twins (анти-коллапс)
- [x] Шаг 7: UnifiedEngine (speak/codegen/combine)
- [ ] Шаг 2: большой прогон 1M на чистом корпусе (идёт: /tmp/code_500k.fuga)
- [ ] Шаг 4: концепт-приор βc в полном декодере
- [ ] Шаг 8: полный A/B v2/MB3/MB3+concept

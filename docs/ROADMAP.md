# ROADMAP — единая дорога обучения Fuga (обновлено 30.08)

## ✅ Неградиентная интеграция (Backprop-Free Engine) — ВЫПОЛНЕНО

**Цель**: полный отказ от loss.backward()/SGD в основном ядре в пользу
биологической пластичности (HTM + VSA + SNN/STDP + NEAT/HyperNEAT).

### Что сделано
1. **`astral/nongradient_engine.py`** — пром-версия безградиентного движка
   (перенесён из experiments/): HTM(SDR) + VSA(bind/bundle) + SNN(STDP) +
   NEAT/HyperNEAT(CPPN). Метод `train_facts()` — полное обучение на всех
   `fuga_memory_*/fuga_memory.facts.jsonl`.
2. **`astral/models/unified_engine.py`** — удалён `torch.optim.Adam` и
   `loss.backward()` из `train_anti_collapse()`. Заменено на безградиентный
   анти-коллапс: **Gram-Schmidt ортонормализация + Hebbian-обновление** головы.
3. **Экспериментальные модули** (`astral/experiments/`): NEAT/HyperNEAT,
   SNN/нейроморфные, HTM-мост, Mini Cognitive Stack, NonGradientEngine.

### Результаты обучения (0 градиентов)
- Корпус: 15 fuga_memory_* библиотек, **38,885 фактов**, 29,909 токенов
- **HTM top-1 точность предсказания следующего токена: 98.7%**
- VSA-fitness: -0.0025 → +0.0031 (улучшение)
- Веса: Oja-нормализация держит (соревновательное Hebbian)

### Сравнение парадигм
| Задача | Градиентный | FUGA Non-Grad |
|---|---|---|
| Оптимизация | SGD/Adam, backward | STDP/Hebb (Oja) |
| Топология | фикс. MLP/Transformer | NEAT-эволюция |
| Память | RNN/Attention | HTM-SDR (1 шаг) |
| Контекст | Dense embeddings | VSA-гипервекторы |

### Оставшееся
- [ ] HyperNEAT-адаптация порогов LIF/коэффициентов STDP прямо в прогоне
- [ ] Multi-Objective fitness (косинус + разреженность)
- [ ] Сравнение NonGrad vs Grad на одном корпусе (метрика)

## Прошлые шаги (закрыты)
1. Чистый корпус (EN + код) ✅
2. Прогон 1M (clean_1M.fuga, 185B V2) ✅
3. Концепт-канал lang-jepa (FUGA1 tag=8) ✅
4. Декодер V2/MB3+concept (v6_validate) ✅
5. Резонаторы HDC/FPE/PhaseCrystal ✅
6. Анти-коллапс Barlow ✅
7. Единый движок UnifiedEngine ✅
8. Масштабирование 512→768 (8.26M параметров) ✅
9. Мультиязычный корпус (4 репо, 4.4M фрагментов) ✅
10. MB4 концепт-канал в Rust ✅
11. Эксперименты NEAT/SNN/HTM/Mini-Stack ✅
12. **Backprop-Free интеграция** ✅

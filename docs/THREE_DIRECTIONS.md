# Три Направления Fuga (29.08.2026)

> Язык + Код + Пространство — единый фазовый базис FPE-VSA.

## 1. Языковое Направление (Lang-JEPA + Relational Δθ)

**Текущее:** Lang-JEPA адаптер (EMA-таргет, концепт следующего предложения).
**Новое:** реляционные фазовые сдвиги Δθ между объектами.

```
"куб слева от сферы" → HV_куб ⊗ e^{i·Δθ_x} → HV_сфера
Δθ = encode_relation("слева от", dim)  # непрерывный фазовый сдвиг
```

**Реализация:** `astral/models/relational_concept.py`
- RelationalEncoder: текст → фазовый сдвиг Δθ
- RelationalBinder: bind(субъект, Δθ, объект) → суперпозиция
- Извлечение: FPE-резонанс → "что/где/как связано"

## 2. Кодовое Направление (Byte-H-JEPA → AST/Spatial)

**Текущее:** Byte-H-JEPA + MB3 = 200B генерации, V2 = 185B.
**Новое:** пространственные связи → компиляция в AST-структуры.

```
"куб на сфере" → compile() → struct CubeOnSphere { pos: Vec3, rot: Quat }
```

**Реализация:** `astral/models/code_spatial_compiler.py`
- SpatialCompiler: реляционный граф → Rust/Python AST
- CodeSpatialDecoder: MB3 + пространственные приоры

## 3. Пространственное Самосознание (Spatial-JEPA) — НОВОЕ

**Ядро:** эгоцентрический SE(3) базис через FPE-VSA.

```
HV_Self = HV_Identity ⊗ e^{i(x·ωx + y·ωy + z·ωz)} ⊗ Rot(θ, φ, ψ)
```

**Реализация:** `astral/models/spatial_jepa.py`
- EgoFrame: SE(3) binding положения + ориентации
- OccupancyGrid: фазовый кристалл объектов вокруг Self
- WorldModelJEPA: предсказание следующего состояния при действии A

**Пруф:** синтетический — "куб на столе", "сфера слева"

## 4. VL-JEPA (Восстановление)

**Веса есть** (vljepa_projector.pt 14MB, vljepa_bridge_v1.pt 8.7MB), **кода нет**.
- projector: img(2048→768→256) + cap(2048→768→256)
- bridge: vision_proj(256→?) → VSA-пространство

**Реализация:** `astral/models/vljepa_restore.py`
- Загрузить сиротские веса
- VLJEPAEncoder: изображение → VSA-латент (FastVSA)
- VLCaptionBridge: caption → VSA-латент (второй projector)
- Связка: изображение ↔ текст через единый VSA-базис

## Единый базис: FPE-VSA (Complex-Valued Phase)

Все три направления используют ОДИН базис:
- **FPE-VSA** (astral/models/resonator_hdc.py) — комплексные фазы e^{iθ}
- **Дробные степени** x^p — иерархия/интерполяция
- **Фазовый кристалл** — карта occupancy (self ↔ объекты)
- **Связывание** = фазовое сложение (bind = θ₁ + θ₂)
- **Резонанс** = разложение суперпозиции на составляющие

## Порядок реализации

1. D3: Spatial-JEPA (SE(3) + occupancy grid + world model) — CORE
2. D1: Relational Δθ (язык → пространство) — на основе D3
3. V1: VL-JEPA (восстановить веса, bridge) — на основе D3
4. D2: Code Spatial Compiler (пространство → AST/структуры)
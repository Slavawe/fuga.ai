# Fuga 1.0 — Hierarchical JEPA & VSA Architecture

**English** · [Русский](#русский) · [中文](#中文) · [Español](#español)

---

## English

Experimental non-transformer hierarchical predictive memory built on Vector Symbolic Architectures (VSA) and Joint Embedding Predictive Architecture (JEPA). Replaces backpropagation with local Delta Rule updates.

### Results (Baseline 1.0)

8192-bit hypervectors, 1000 source files, 10 epochs:

| Level | Loss | Role |
|---|---|---|
| L0 | **0.566** | Static background absorption |
| L1 | **0.447** | Macro-pattern extraction |
| L2 | **0.507** | Self-correction (metacognition) |

### Usage

```bash
cargo run -- h-jepa-train <repo_dir> 8192 10
cargo test
```

### License

Apache-2.0

---

## Русский

Экспериментальная иерархическая предиктивная память без трансформеров, построенная на векторно-символических архитектурах (VSA) и JEPA. Вместо backpropagation — локальные обновления Delta Rule.

### Результаты (Baseline 1.0)

Гипервекторы 8192 бит, 1000 файлов, 10 эпох:

| Уровень | Loss | Роль |
|---|---|---|
| L0 | **0.566** | Поглощение статического фона |
| L1 | **0.447** | Извлечение макропаттернов |
| L2 | **0.507** | Самокоррекция (метапознание) |

### Использование

```bash
cargo run -- h-jepa-train <директория_репозиториев> 8192 10
cargo test
```

### Лицензия

Apache-2.0

---

## 中文

基于向量符号架构 (VSA) 和联合嵌入预测架构 (JEPA) 的实验性非 Transformer 层次预测记忆。用局部 Delta Rule 更新替代反向传播。

### 结果 (Baseline 1.0)

8192 位超向量，1000 个源文件，10 个周期：

| 层级 | Loss | 作用 |
|---|---|---|
| L0 | **0.566** | 静态背景吸收 |
| L1 | **0.447** | 宏观模式提取 |
| L2 | **0.507** | 自我修正（元认知） |

### 使用

```bash
cargo run -- h-jepa-train <仓库目录> 8192 10
cargo test
```

### 许可证

Apache-2.0

---

## Español

Memoria predictiva jerárquica experimental sin transformers, construida sobre Arquitecturas Vectoriales Simbólicas (VSA) y JEPA. Reemplaza la retropropagación con actualizaciones locales Delta Rule.

### Resultados (Baseline 1.0)

Hipervectores de 8192 bits, 1000 archivos, 10 épocas:

| Nivel | Loss | Rol |
|---|---|---|
| L0 | **0.566** | Absorción de fondo estático |
| L1 | **0.447** | Extracción de macropatrones |
| L2 | **0.507** | Autocorrección (metacognición) |

### Uso

```bash
cargo run -- h-jepa-train <directorio_repos> 8192 10
cargo test
```

### Licencia

Apache-2.0

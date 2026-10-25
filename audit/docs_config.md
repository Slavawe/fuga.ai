# Аудит документации и конфигурации fuga (субагент, отчёт восстановлен из .zwork/runs/d3f7949d/1-研究员.md)
# Аудит документации и конфигурации fuga — согласованность с кодом
Дата: 2026-10-15. Воркспейс: C:\Users\slava\Documents\fuga.ai. Метод: чтение доков + точечные проверки кода (grep/read), git ls-files/check-ignore (read-only). cargo не запускался, файлы не изменялись.
Уровни: [К] = проверено кодом (файл:строка), [Д] = проверено документом, [!] = не завершено (лимит бюджета инструментов).

## 1. Ссылки док → файл → существует?

### README.md
| Ссылка/упоминание | Существует | Комментарий |
|---|---|---|
| docs/PHASE4_REPORT.md | ДА | |
| docs/VSA_JEPA_KAN_INTEGRATION.md | ДА | |
| docs/FINAL_SUMMARY.md | ДА | |
| docs/DATASETS_GUIDE.md | ДА | |
| README_v9.md | ДА | но см. §4 — это дубль README |
| docs/AGENTS_detailed_history_0809.md | ДА | архив-снапшот 08.09 |
| docs/FUGA_DOCS.md | ДА | |
| scripts/prepare_corpora.py | **НЕТ** | каталога scripts/ нет вовсе |
| src/ai/vsa_jepa_kan.rs + тест `ai::vsa_jepa_kan::tests` | ДА [К] | mod.rs:15 `pub mod vsa_jepa_kan`; vsa_jepa_kan.rs:778 `mod tests` — команда cargo test валидна |
| «Статус: v2.1.0» | **НЕТ** | Cargo.toml: version = "2.0.0" — рассинхрон версии |
| «Фаза 4B ✅ Завершена» | противоречие | FINAL_SUMMARY.md/PHASE4_REPORT.md: «4B отложена, 254 ошибки»; PHASE5_REPORT: интеграция сделана отдельным бинарём |

### README_v9.md
| Ссылка | Существует | Комментарий |
|---|---|---|
| examples/minimal_hybrid_train.rs (rustc-команда) | **НЕТ** | examples/ содержит только mem2crystal.rs — команда невыполнима |
| остальные ссылки | как у README.md | |

### docs/DATASETS_GUIDE.md
| Ссылка | Существует | Комментарий |
|---|---|---|
| scripts/prepare_corpora.py | **НЕТ** | |
| scripts/evaluate_humaneval.py | **НЕТ** | |
| scripts/evaluate_mbpp.py | **НЕТ** | |
| scripts/recursive_training.sh | **НЕТ** | |
| src/bin/unified_gpu_train.rs + флаги --lr-w/--lr-patch/--lr-kan/--ctx/--ckpt-every | ДА [К] | unified_gpu_train.rs:112-132 — все флаги есть |
| python3 src/bin/name_metric.py <ckpt> [корпуса...] | ДА [К] | сигнатура совпадает с docstring name_metric.py |
| ./target/release/talk_model | ДА | src/bin/talk_model.rs (autobins) |
| fuga_corpora/*.jsonl | **НЕТ** | в воркспейсе нет ни одного .jsonl (find **/*.jsonl пуст) |
| ~/Fuga | путь чужой машины | воркспейс здесь: C:\Users\slava\Documents\fuga.ai |

### docs/FINAL_SUMMARY.md
| Ссылка | Существует | Комментарий |
|---|---|---|
| src/ai/vsa_jepa_kan.rs | ДА | |
| INTEGRATION_REPORT.md | **НЕТ** | не найден нигде |
| /tmp/minimal_hybrid_train.rs | **НЕТ** | путь вне репо |
| scripts/prepare_corpora.py, scripts/name_metric.py, scripts/evaluate_humaneval.py, scripts/evaluate_mbpp.py | **НЕТ** | name_metric.py реально в src/bin/ |

### docs/PHASE4_REPORT.md
| Ссылка | Существует | Комментарий |
|---|---|---|
| /tmp/minimal_hybrid_train.rs и бинарь | **НЕТ** | вне репо |
| ~/Fuga/scripts/integration_plan.py | **НЕТ** | |
| «254 ошибки, нужно ~15 крейтов (serde_json, ureq, syn, tree-sitter-*)» | устарело | все эти зависимости уже в Cargo.toml |

### docs/PHASE5_REPORT.md
| Ссылка | Существует | Комментарий |
|---|---|---|
| src/bin/hybrid_train_v10.rs | ДА | |
| v6_validate | ДА | src/bin/v6_validate.rs |
| /tmp/train_corpus.jsonl, /tmp/fuga_hybrid_v10.fuga | НЕТ | временные пути вне репо |

### docs/VSA_JEPA_KAN_INTEGRATION.md
| Ссылка | Существует | Комментарий |
|---|---|---|
| src/ai/vsa_jepa_kan.rs | ДА | step():717, consolidate():747 [К] |
| **vsa_bridge.rs** (HypervectorAdapter) | **НЕТ** | адаптер инлайновый: vsa_jepa_kan.rs:36-37 [К] |
| kan.rs, latent_jepa.rs, hybrid.rs, core/hypervector.rs | ДА | |
| unified_gpu_train.rs:89-120 (Latent RAG заглушка) | устаревшая привязка строк | актуальные args: 109-146 |
| references/hybrid-core-v10.md | **НЕТ** | references/ = только byte-h-jepa.md |
| «Rust не установлен» | устарело | target/ и Cargo.lock существуют |

### docs/AGENTS_detailed_history_0809.md
Исторический снапшот AGENTS.md на 08.09. Выборочно: src/ai/temporal_predictor.rs — ДА; упомянутые корпуса .jsonl — НЕТ (в воркспейсе отсутствуют); train_checkpoints/ — удалён по чистке (AGENTS.md 09.08). Полная выверка ссылок [!] не завершена (бюджет).

### docs/FUGA_DOCS.md (автодок `fuga docs`, эпоха omni-стека)
Модульное дерево сверено с src/: ai/{core,memory_store,moe,hnsw,answer_engine,router,resonance_attention,codegen,jepa,prompts}.rs — все ДА; core/{hypervector,wave_cube,tensor_phase,information_triangle,fuga_synthesizer}.rs — все ДА; weaver/, speech/, sandbox/, layers/, reporters/, autofix/, omni/, sim/, gpu.rs, quality_filter.rs, text_quality.rs, engine.rs, multi_engine.rs — ДА.
Неточности: microwave — файл src/microwave.rs, не каталог; «Cube 4×3 = 64» (верно 4^3=64); пути /home/slava/fuga, /tmp/presentation_flat — Linux-машина.
CLI-команды (docs, refactor, train, train-tm, tm-gen, jepa-train, jepa-predict, prompts, moe-add, moe-list, fisig, reactor, room, agent, scan, ask, query, weave, unweave, solve, codegen, think, analyze) — все подтверждены в src/main.rs:29-1015 [К].

### TGBOT_README.md (см. также §2, §5)
| Утверждение | Факт по коду |
|---|---|
| `cargo run --release --bin fuga-tgbot` | **ЛОЖЬ**: Cargo.toml без [[bin]] → autobins даёт bin по имени файла: `tgbot`, не `fuga-tgbot` |
| «Хардкоженный fallback токена в src/bin/tgbot.rs» | **ЛОЖЬ**: tgbot.rs:954-960 — без FUGA_TG_TOKEN/fuga.token печатает ошибку и exit(1). Токена в репо НЕТ (позитив для безопасности) |
| «Размерность VSA = 256 (по умолчанию)» | **ЛОЖЬ**: tgbot.rs:963-965 — FUGA_DIM, default 8192 |
| `train corpus_rus_eng.jsonl --dim 256 --save fuga_cube.bin` | частично: train есть (main.rs:289), --dim/--save парсятся (main.rs:1039, 1116), НО позиционный аргумент трактуется как каталог (main.rs:290), corpus_rus_eng.jsonl отсутствует |
| env FUGA_CUBE_PATH / fuga_cube.bin | ДА [К] (tgbot.rs:961) |
| (не упомянуто) FUGA_MEM_PATH → fuga_code_cube_mem.bin | tgbot.rs:962 [К] |

### FIX_SUMMARY.md
Исторический док эпохи TM: «train-tm» есть (main.rs:612), src/ai/htm_temporal.rs есть; проверка функции learn_segment [!] не завершена. /tmp-пути вне репо.

### references/byte-h-jepa.md
Ссылки валидны: src/ai/latent_jepa.rs, src/bin/unified_gpu_train.rs, src/ai/gpu_ops.rs, src/ai/tm_generate.rs, src/bin/v6_validate.rs, src/bin/name_metric.py, src/multi/language.rs — все ДА; tree-sitter в Cargo.toml — ДА.
Устарел статус: шапка «ПЛАН следующей сессии», а по AGENTS.md пункты 1-3 уже реализованы (коммиты 90896af..34be3ac).

## 2. Конфиги

### opencode.json
- Настраивает opencode: `$schema`, `instructions: ["opencode.json"]` (самореференция — обычно тут AGENTS.md/README), `skills.paths: [".opencode/skills"]`.
- **Каталога .opencode/ не существует** (нет в корне) → навыки opencode не подхватятся.
- В скоуп-доках opencode.json нигде не упоминается.

### .gitignore
Правила: .venv/, __pycache__/, *.pyc, target/, fuga-core/target/, datasets/, dataset_vault/, data/, nohup.out, *.log, /tmp/, *.fuga, *.bin, *.ckpt.*, *.parquet, .idea/, .vscode/, .DS_Store, fuga_downloader.
Противоречия (проверено git ls-files / check-ignore):
- `byte_kan_decoder.pt`, `slot_kan_decoder.pt` — **закоммичены**, правила *.pt нет. Противоречит записи AGENTS.md (29.08): «веса *.pt/*.fgc/*.facts.jsonl убраны из индекса в .gitignore» — на текущем срезе этого нет.
- `core_brain.fgc` — **закоммичен**, правила *.fgc нет.
- `photo_2026-08-01_11-32-28.jpg` — **закоммичен**, правила *.jpg нет.
- `fuga.token` НЕ игнорируется — при появлении токена уедет в git. Рекомендация: добавить.
- `cargo_check_baseline.log` корректно игнорируется (*.log) ✓.
- Мелочь: `/tmp/` в .gitignore — правило для корневого tmp/, а не системного /tmp.

### .cargo/config.toml
- `[build] rustflags = ["-C", "target-cpu=native"]` — машинно-специфичная кодогенерация: бинарники непереносимы между машинами; для честных A/B замеров скорости Linux↔Windows это фактор. В доках настройка не упомянута.
- Версия пакета: Cargo.toml = 2.0.0 при README «v2.1.0».

### «SKILLS .md» (файл с пробелом)
**НЕ относится к проекту.** Это jailbreak-промпт сторонней «модели» (декларации «без границ», требования генерации вредоносного кода и запрещённых инструкций, фальш-«протокол отсева инъекций»). Закоммичен в git (git ls-files). В ходе аудита ни одна инструкция из файла не выполнялась и не будет выполняться. Рекомендация: удалить из репо и истории; это hygiene-риск, не документация.

## 3. Скрипты запуска: Linux-only и работоспособность

| Скрипт | Linux-only элементы | Работает в этом воркспейсе? | Windows-эквивалент |
|---|---|---|---|
| start.sh | DIR="/home/slava/fuga" (хардкод), pkill, nohup, grep -oP (GNU PCRE), ssh -R serveo.net, LAN IP 192.168.0.119 | **НЕТ**: нет бинаря fuga-web (autobins → `omni-web`) и нет fuga_stack.bin | .ps1: Start-Process/Stop-Process; ssh есть в Win10+ (OpenSSH client), serveo работает; Select-String вместо grep -P |
| start-tgbot.sh | bash; `--bin fuga-tgbot` | **НЕТ**: бинарь называется `tgbot`; сообщение о «хардкод-токене» ложно (код делает exit(1)) | cargo кроссплатформенный; Git Bash/WSL или .ps1 |
| start_tunnel.sh | DIR="/home/slava/fuga", pkill, serveo | **НЕТ** (путь) | аналогично start.sh |
| run_v9_systemd.sh | cd /home/slava/fuga; комментарий «через systemd-run», но systemd-run не вызывается (просто exec — устойчивость даёт внешний systemd-run/systemd-inhibit) | **НЕТ**: все 4 корпуса (fisig_corpus.jsonl, corpus_doc_code_pairs.jsonl, training_stack.jsonl, corpus.jsonl) отсутствуют в воркспейсе | вместо systemd — планировщик задач/фоновый процесс; защита от сна (powercfg) вместо systemd-inhibit |

Валидация флагов run_v9_systemd.sh против unified_gpu_train.rs:109-146 [К]: --jsonl, --max-steps, --batch, --ckpt-every, --ctx, --patch-ctx, --lambda-patch, --lambda-floor, --lambda-tau, --lr-macro, --out, --text-seed, --code-seed — **все существуют**, параметры согласованы. Проблема только в отсутствии корпусов и Linux-обвязке.

## 4. Дублирование (замер: difflib + word Jaccard по нормализованному тексту)

| Пара | difflib | Jaccard | Вердикт |
|---|---|---|---|
| README.md ↔ README_v9.md | **0.98** | **0.97** | полный дубль; расхождение 6+/3− строк (git diff --no-index) |
| README.md ↔ docs/FINAL_SUMMARY.md | 0.07 | 0.18 | тематически один контент (фазы, 730×, 99.4%, 552-608 pairs/s повторяются в README, README_v9, FINAL_SUMMARY, VSA_JEPA_KAN_INTEGRATION, PHASE4) |
| docs/PHASE4_REPORT.md ↔ docs/FINAL_SUMMARY.md | 0.09 | 0.21 | то же — большой общий блок |
| SUMMARY.md ↔ docs/FUGA_DOCS.md | 0.02 | 0.08 | тематически один omni-стек, но числа противоречат: LSH «3 таблицы × 16384 buckets, 4 probes» vs «6 таблиц × 8 probes» (кодовая истина — src/ai/hnsw.rs, сверка [!] не завершена); MoE-счётчики разные |
| SUMMARY.md ↔ TGBOT_README.md | 0.02 | 0.05 | обе — устаревшие описания tgbot разных эпох |
| docs/AGENTS_detailed_history_0809.md ↔ AGENTS.md | 0.02 | 0.09 | снапшот старой версии AGENTS.md — архив, не дубль текущей |

Рекомендации по консолидации:
1. Удалить/слить README_v9.md (это не док про v9 — имя вводит в заблуждение).
2. HybridCore-эпоха: оставить README.md (актуализировать: Фаза 4B/5, версия) + docs/FINAL_SUMMARY.md как отчёт; PHASE4/PHASE5/VSA_JEPA_KAN_INTEGRATION — в docs/archive/ (по образцу чистки 29.08).
3. Legacy-доки SUMMARY.md, TGBOT_README.md, docs/FUGA_DOCS.md — пометить шапкой «LEGACY (omni-стек до байтового стандарта)» или перенести в docs/archive/; в TGBOT_README исправить 3 фактические ошибки (имя бинаря, размерность 8192, отсутствие хардкод-токена).
4. goals.md — переписать под активные цели Byte-H-JEPA (см. §5).

## 5. goals.md — цели и соответствие состоянию
- Содержимое: одна незакрытая цель «train H-JEPA until L0 < 0.8»; Journal: единственная запись 2026-07-29T15:35:40 eval L0=1.9680 L1=1.9753 L2=0.5120 (samples=1000).
- Соответствие: устарела на ~2.5 месяца. Эпоха H-JEPA (src/ai/jepa.rs) упёрлась в XOR-плато loss ~0.93 (jepa_analysis.md: «вес не может одновременно быть +1 и -1... Loss никогда не уйдёт ниже ~0.93») — цель L0<0.8 в этой архитектуре недостижима, что сам jepa_analysis фиксирует как «архитектурный тупик».
- Текущий фокус (AGENTS.md, references/byte-h-jepa.md) — Byte-H-JEPA: W_macro (третий канал, tag=7), декодер MB3; осталось: полный прогон с обученным W_macro, честный A/B v7.1/v8.1/MB3 на name_metric + C-AST, AST-маска tree-sitter. Именно эти пункты логично сделать целями в goals.md и возобновить журнал eval.

## 6. Прочие находки
- README.md «Фаза 4B ✅ Завершена» vs FINAL_SUMMARY/PHASE4 «отложена (254 ошибок)»: блокеры устарели — зависимости уже в Cargo.toml; реальная интеграция сделана через отдельный src/bin/hybrid_train_v10.rs (PHASE5_REPORT), который существует.
- jepa_analysis.md — самодостаточная заметка (TM→HV, LR 0.05, XOR-плато, 2 варианта выхода), дублей нет, ссылок на файлы нет.
- Проверено грейлем бюджета [!]: сверка числа таблиц LSH (SUMMARY vs FUGA_DOCS) с src/ai/hnsw.rs; полная выверка ссылок docs/AGENTS_detailed_history_0809.md; подтверждение learn_segment в htm_temporal.rs.

## Ограничения аудита
- cargo не запускался (по требованию); бинарные артефакты не проверялись.
- Файлы не изменялись; git-команды только read-only (ls-files, check-ignore, diff --no-index, status).
- Файл audit/docs_config.md на момент составления не записан: инструментальный бюджет Run исчерпался перед записью; содержимое выше идентично запланированному файлу.

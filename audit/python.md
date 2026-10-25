# Аудит Python-экосистемы fuga (субагент «研究员», отчёт восстановлен из .zwork/runs/d3f7949d/0-研究员.md)
Дата: 2026-10-15. Метод: статический разбор (файлы не запускались). Охват: 25 корневых .py + antitf (15) + necli (~180) + py/ + tools/ + tests/.

## Топ-5 выводов
1. **`import fuga_core` сломан**: это PyO3-расширение из Rust-крейта fuga-core (cdylib, abi3-py310), собранного .pyd/.so нет нигде — падают 14 корневых скриптов; штатно деградирует только antitf/rust_bridge.py:9-13 (HAS_RUST=False).
2. **Все данные отсутствуют**: datasets/, dataset_vault/, text_corpus/, temp_repos/ не существуют; для 5 файлов (kaikki_ru.jsonl, lexicon_ru_50k.txt, GSM8K parquet, 2 conceptnet-файла) генераторов в репо вообще нет — только ручная загрузка; Tatoeba/RuCoLA antitf скачивает сам.
3. **Linux-пути в 5 файлах**: gen_synthetic.py:3, gen_comments.py:3,7, gen_wiki.py:3 (/home/slava), py/train_cpp.py:91,98 и tools/to_jsonl.py:40,45 (/tmp), плюс agent_loop.py:14 (necli/.venv/bin/python3, есть fallback) — на Windows-копии нерабочие или пишут не туда.
4. **Все три бинарных артефакта в корне — сироты**: core_brain.fgc (2.2MB) не упоминается ни одним файлом репо (0 вхождений, писатель неизвестен), byte_kan_decoder.pt и slot_kan_decoder.pt — только выходы своих скриптов (torch.save), читателей нет — кандидаты в архив.
5. **necli (~180 файлов, v0.19.0) — самый живой код**, Windows-ready (msvcrt/win32-ветки; риск: top-level termios в commands/helpers.py:10 не подтверждён), но без внешних сервисов работает лишь как оболочка: чат требует LLM-провайдера, fuga_query — опциональна с фолбэком; импортных связей корневые .py ↔ necli нет (только процессные через agent_loop.py).

## Рекомендации
- Починить или явно задокументировать fuga_core-мост (либо собрать PyO3-расширение, либо пометить скрипты как требующие сборки).
- Linux-пути в gen_*.py, py/train_cpp.py, tools/to_jsonl.py → параметризовать/заменить на относительные.
- Осиротевшие артефакты (core_brain.fgc, *.pt) — в архив/удаление по решению пользователя.
- Данные (datasets/, corpus) — вне репо; генераторы отсутствуют для 5 файлов — пометить в DATASETS_GUIDE.

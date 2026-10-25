#!/usr/bin/env python3
# make_corpus.py — подготовка корпуса «ОДИН Python-файл» для unified_gpu_train.
# Формат: одна JSONL-строка {"code": <первые 85% файла>} = ОДИН документ,
# окна обучения скользят по всему файлу без разрывов строк.
# Валидация: хвост 15% сырыми байтами (.bin, BIN-режим тренера).
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "agent_loop.py"
OUT_DIR = Path(__file__).resolve().parent
VAL_SHARE = 0.15
REPEAT = 22  # тренер делает ОДИН проход по корпусу без циклизации; повторяю
             # документ (методология синтетического пруфа: один шаблон ×40000)
             # чтобы набрать 400K пар

data = SRC.read_bytes()
cut = int(len(data) * (1.0 - VAL_SHARE))
train_bytes, val_bytes = data[:cut], data[cut:]

jsonl = OUT_DIR / "agent_loop_train.jsonl"
with jsonl.open("w", encoding="utf-8") as f:
    line = json.dumps({"code": train_bytes.decode("utf-8", errors="strict")}, ensure_ascii=False)
    for _ in range(REPEAT):
        f.write(line + "\n")

val = OUT_DIR / "agent_loop_val.bin"
val.write_bytes(val_bytes)

print(f"src: {SRC.name} = {len(data)} bytes")
print(f"train: {jsonl.name} = {cut} bytes x {REPEAT} docs")
print(f"val:   {val.name} = {len(val_bytes)} bytes")

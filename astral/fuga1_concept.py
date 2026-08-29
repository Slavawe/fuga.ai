#!/usr/bin/env python3
"""Запись/чтение секции CONCEPT_W (tag=8) в FUGA1-файл.

Формат FUGA1: magic "FUGA1" + секции [u32 tag][u32 len][bytes], END=0.
Секция CONCEPT_W — плоский f32-blob весов lang-jepa концепт-предиктора
(структура интерпретируется astral/models/lang_jepa_adapter.py; Rust/C++
хранят как непрозрачный blob, bin-совместимы).

Правило: секции идут ДО TAG_END — при добавлении в существующий файл
концепт-секция вставляется перед END, сохраняя порядок секций.
"""
from __future__ import annotations

import os
import struct


MAGIC = b"FUGA1"
TAG_END = 0
TAG_CONCEPT_W = 8


def _write_section(buf: bytearray, tag: int, data: bytes) -> None:
    buf += struct.pack("<I", tag)
    buf += struct.pack("<I", len(data))
    buf += data


def append_concept_section(fuga_path: str, concept_flat: bytes, out_path: str) -> int:
    """Вставляет секцию CONCEPT_W перед END в FUGA1-файл.

    Возвращает число f32 в записанной секции.
    """
    data = bytearray(open(fuga_path, "rb").read())
    if data[:5] != MAGIC:
        raise ValueError(f"{fuga_path}: не FUGA1 (magic != FUGA1)")

    pos = 5
    # Ищем позицию TAG_END
    end_pos = None
    while pos + 8 <= len(data):
        tag = struct.unpack_from("<I", data, pos)[0]
        ln = struct.unpack_from("<I", data, pos + 4)[0]
        if tag == TAG_END:
            end_pos = pos
            break
        pos += 8 + ln
    if end_pos is None:
        raise ValueError(f"{fuga_path}: нет TAG_END — файл повреждён")

    # Проверяем, нет ли уже CONCEPT_W (не дублировать)
    p2 = 5
    while p2 + 8 <= end_pos:
        tag = struct.unpack_from("<I", data, p2)[0]
        ln = struct.unpack_from("<I", data, p2 + 4)[0]
        if tag == TAG_CONCEPT_W:
            raise ValueError(f"{fuga_path}: секция CONCEPT_W уже есть")
        p2 += 8 + ln

    # Собираем: [до END] + CONCEPT_W + [END-секция]
    new_buf = bytearray()
    new_buf += data[:end_pos]
    _write_section(new_buf, TAG_CONCEPT_W, concept_flat)
    new_buf += data[end_pos:]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(new_buf)
    return len(concept_flat) // 4


def read_concept_section(fuga_path: str) -> bytes | None:
    """Читает секцию CONCEPT_W из FUGA1-файла (или None)."""
    data = open(fuga_path, "rb").read()
    if data[:5] != MAGIC:
        return None
    pos = 5
    while pos + 8 <= len(data):
        tag = struct.unpack_from("<I", data, pos)[0]
        ln = struct.unpack_from("<I", data, pos + 4)[0]
        if tag == TAG_END:
            return None
        if pos + 8 + ln > len(data):
            return None
        if tag == TAG_CONCEPT_W:
            return bytes(data[pos + 8 : pos + 8 + ln])
        pos += 8 + ln
    return None


def torch_state_to_flat(state_dict: dict) -> bytes:
    """Конвертирует state_dict концепт-предиктора в плоский f32-blob.

    Формат: [n_params:u32][n_elems:u32][shape:u32*n][f32 данных...]
    по каждому тензору — структура восстанавливаемая на Python.
    """
    import torch

    out = bytearray()
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().float()
        shape = list(t.shape)
        flat = t.reshape(-1)
        out += struct.pack("<I", len(name.encode()))
        out += name.encode()
        out += struct.pack("<I", len(shape))
        out += struct.pack(f"<{len(shape)}I", *shape)
        out += struct.pack(f"<{len(flat)}f", *flat.tolist())
    return bytes(out)


def predictor_flat(sd_predictor: dict) -> bytes:
    """Чистый f32-blob концепт-предиктора БЕЗ метаданных.

    7 тензоров в ФИКСИРОВАННОМ порядке (как Rust ConceptPredictor):
      query [512], in_proj_w [1536*512], in_proj_b [1536],
      out_proj_w [512*512], out_proj_b [512], ln_w [512], ln_b [512]
    Итого: 1,052,160 f32. Rust читает напрямую (from_flat).
    """
    import struct
    import torch

    order = [
        "query",
        "context_attention.in_proj_weight",
        "context_attention.in_proj_bias",
        "context_attention.out_proj.weight",
        "context_attention.out_proj.bias",
        "projection.weight",
        "projection.bias",
    ]
    out = bytearray()
    for key in order:
        t = sd_predictor[key].detach().cpu().float().reshape(-1)
        out += struct.pack(f"<{len(t)}f", *t.tolist())
    return bytes(out)


def predictor_flat_to_dict(blob: bytes) -> dict:
    """Обратно: чистый flat f32 → dict из 7 тензоров (фиксированные формы)."""
    import struct
    import torch

    shapes = [
        (1, 1, 512),        # query
        (1536, 512),        # in_proj_w
        (1536,),            # in_proj_b
        (512, 512),         # out_proj_w
        (512,),             # out_proj_b
        (512,),             # ln_w
        (512,),             # ln_b
    ]
    keys = [
        "query",
        "context_attention.in_proj_weight",
        "context_attention.in_proj_bias",
        "context_attention.out_proj.weight",
        "context_attention.out_proj.bias",
        "projection.weight",
        "projection.bias",
    ]
    out = {}
    pos = 0
    for key, shape in zip(keys, shapes):
        n = int(__import__("math").prod(shape))
        vals = struct.unpack_from(f"<{n}f", blob, pos)
        pos += 4 * n
        out[key] = torch.tensor(list(vals)).reshape(shape)
    return out


def flat_to_torch_state(blob: bytes) -> dict:
    """Восстанавливает state_dict из плоского f32-blob (см. torch_state_to_flat)."""
    import torch

    state: dict = {}
    pos = 0
    while pos < len(blob):
        n_name = struct.unpack_from("<I", blob, pos)[0]
        pos += 4
        name = blob[pos : pos + n_name].decode()
        pos += n_name
        n_shape = struct.unpack_from("<I", blob, pos)[0]
        pos += 4
        shape = list(struct.unpack_from(f"<{n_shape}I", blob, pos))
        pos += 4 * n_shape
        n_elems = int(__import__("math").prod(shape))
        vals = struct.unpack_from(f"<{n_elems}f", blob, pos)
        pos += 4 * n_elems
        state[name] = torch.tensor(list(vals)).reshape(shape)
    return state


if __name__ == "__main__":
    import sys

    # Тест: записать пустую концепт-секцию в существующий FUGA1
    if len(sys.argv) < 2:
        print("использование: fuga1_concept.py <existing.fuga> [--roundtrip]")
        sys.exit(1)
    src = sys.argv[1]
    n = append_concept_section(src, torch_state_to_flat({"probe": __import__("torch").zeros(4)}), src + ".with_concept")
    print(f"записано f32: {n} -> {src}.with_concept")
    blob = read_concept_section(src + ".with_concept")
    print(f"обратно: {len(blob)//4} f32, state: {list(flat_to_torch_state(blob).keys())}")

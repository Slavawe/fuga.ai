
"""CodeQueryEngine: AST-графы -> VSA-факты в PersistentVSAMemory.

Каждый узел (функция/структура) сохраняется как факт:
  code:имя_функции -- isa -- function_definition
  code:имя_структуры -- isa -- struct_specifier
Запрос: поиск по префиксу имени -> возврат метаданных.
"""

from __future__ import annotations

from __future__ import annotations

import re
import sys

import numpy as np

sys.path.insert(0, ".")

from fuga_memory import PersistentVSAMemory


class CodeQueryEngine:
    def __init__(self, binder, memory_dir="fuga_memory_code"):
        self.binder = binder
        self.mem = PersistentVSAMemory(binder, directory=memory_dir)
        self._index: dict[str, list[tuple[str, str]]] = {}  # name -> [(type, text)]

    def add_ast_meta(self, file_path, items):
        name = None
        bname = file_path.split("/")[-1] if "/" in file_path else file_path
        for ntype, text in items:
            if ntype in ("function_definition", "function_declaration",
                         "method_definition", "function_item"):
                nm = text.split("(")[0].strip().split()[-1]
                if nm and nm.isidentifier() and len(nm) > 1:
                    self._index.setdefault(nm, []).append(
                        (ntype, f"{bname}: {text[:200]}"))
                    self.mem.add_fact("en", f"code:{nm}", ntype, text[:80],
                                      dedupe_key=("code", ntype, nm))
            if ntype in ("struct_specifier", "class_declaration",
                         "type_definition"):
                nm = text.split()[-1].strip() if text else ""
                if nm and len(nm) > 1:
                    self._index.setdefault(nm, []).append(
                        (ntype, f"{bname}: {text[:200]}"))
                    self.mem.add_fact("en", f"code:{nm}", ntype, text[:80],
                                      dedupe_key=("code", ntype, nm))

    def query(self, name: str, limit=3) -> list[tuple[str, str, str]]:
        """Поиск по префиксу имени -> [(type, text, file)]."""
        matches = []
        for key, entries in self._index.items():
            if name.lower() in key.lower():
                for ntype, text in entries[:limit]:
                    matches.append((ntype, text, key))
        return matches[:limit]

    def query_fact(self, name: str, limit=3) -> list[str]:
        matches = self.query(name, limit)
        if not matches:
            return []
        return [f"{t} ({n})" for t, n, _ in matches]

    def load_index_from_disk(self) -> int:
        """Пересобрать self._index из сохранённых фактов-файла."""
        import json
        import os
        path = os.path.join(self.mem.dir, "fuga_memory.facts.jsonl")
        n = 0
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    nm = d["subject"].replace("code:", "", 1)
                    self._index.setdefault(nm, []).append(
                        (d["relation"], f"{d['object'][:120]}"))
                    n += 1
        return n
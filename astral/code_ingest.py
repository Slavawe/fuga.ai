
"""AST Code Ingest: Tree-Sitter (6 языков) -> VSA-кристаллы.

На файл извлекаем:
  - функции/методы (сигнатура),
  - типы/структуры/классы,
  - рёбра вызовов (caller -> callee) как связывания.
Каждый элемент кодируется HybridBinder'ом; рёбра = bind(caller, callee).
Метрика: K-lines/sec и число узлов/рёбер.
"""

from __future__ import annotations

from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fuga_core
from antitf.rust_bridge import packed_to_torch
from tree_sitter import Language, Parser

LANGS = {
    ".c": "tree_sitter_c", ".h": "tree_sitter_c",
    ".cpp": "tree_sitter_cpp", ".hpp": "tree_sitter_cpp", ".cc": "tree_sitter_cpp",
    ".rs": "tree_sitter_rust",
    ".py": "tree_sitter_python",
    ".go": "tree_sitter_go",
    ".java": "tree_sitter_java",
}

PARSERS = {}
for ext, mod_name in LANGS.items():
    if mod_name not in PARSERS:
        mod = __import__(mod_name)
        PARSERS[mod_name] = Parser(Language(mod.language()))


def walk(node, names):
    if node.type in ("function_definition", "function_declaration",
                     "method_definition", "function_item", "method_declaration",
                     "class_declaration", "struct_specifier", "type_definition",
                     "interface_declaration"):
        text = node.text.decode("utf-8", "ignore") if node.text else ""
        name = text.split("(")[0].split()[0] if node.type in (
            "function_definition", "function_declaration",
            "method_definition", "function_item", "method_declaration") \
            else None
        names.append((node.type, text[:160]))
    for c in node.children:
        walk(c, names)


def extract_file(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    if ext not in LANGS:
        return None
    try:
        src = open(path, "rb").read()
    except OSError:
        return None
    if len(src) > 1_500_000:
        return None
    parser = PARSERS[LANGS[ext]]
    tree = parser.parse(src)
    items = []
    walk(tree.root_node, items)
    calls = []
    for ntype, text in items:
        if ntype in ("function_definition", "function_declaration",
                     "method_definition", "function_item"):
            name = text.split("(")[0].strip().split()[-1]
            if name and name.isidentifier():
                calls.append(name)
    return {"path": path, "lines": src.count(b"\n"), "items": items,
            "call_names": calls}


def encode_file(binder, meta: dict, dim=2048):
    """Каждый элемент -> HV; вызов-связь = bind(caller, callee)."""
    hvs = []
    for ntype, text in meta["items"]:
        pk = np.asarray(binder.bind_batch(
            [[f"AST:{ntype}:{t}" for t in (text[:60],)]]))
        hvs.append(packed_to_torch(pk)[0])
    edges = []
    names = meta["call_names"]
    for i in range(len(names) - 1):
        if names[i] != names[i + 1]:
            a = np.asarray(binder.bind_batch([[f"FN:{names[i]}"]]))[0]
            b = np.asarray(binder.bind_batch([[f"FN:{names[i+1]}"]]))[0]
            edges.append((names[i], names[i + 1],
                          float(np.count_nonzero(a ^ b))))
    return hvs, edges


def main(root_dirs: list[str]):
    binder = fuga_core.HybridBinder(2048)
    files = []
    for root in root_dirs:
        for ext in LANGS:
            files += glob.glob(os.path.join(root, "**", "*" + ext),
                               recursive=True)
    files = [f for f in files if "/target/" not in f
             and ".venv" not in f and "node_modules" not in f]
    print(f"[code-ingest] найдено файлов: {len(files)}")

    t0 = time.time()
    total_lines = total_items = total_edges = 0
    samples = []
    for i, f in enumerate(files[:3000]):
        meta = extract_file(f)
        if not meta:
            continue
        hvs, edges = encode_file(binder, meta)
        total_lines += meta["lines"]
        total_items += len(meta["items"])
        total_edges += len(edges)
        if len(samples) < 3:
            samples.append((f, [n for n, _ in meta["items"][:4]]))
    dt = time.time() - t0
    klps = total_lines / 1024 / max(dt, 1e-9)
    print(f"[code-ingest] {dt:.1f}s | {total_lines} строк "
          f"({klps:.2f} K-lines/s) | узлов AST: {total_items} | "
          f"рёбер вызовов: {total_edges}")
    print("примеры узлов:")
    for path, items in samples:
        print(f"  {os.path.basename(path)}: {items[:3]}")


if __name__ == "__main__":
    roots = sys.argv[1:] or ["antitf", "astral", "fuga-core/src"]
    main(roots)

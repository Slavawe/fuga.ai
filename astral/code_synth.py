
"""Code Synth: AST-фреймы из якорей спутника + Tree-Sitter валидация.

Синтез кода не «побуквенно», а скелетами (FunctionDef -> params -> body),
параметризованными языком и именами VQ-якорей. Каждая генерация проходит
Tree-Sitter-проверку: ERROR-узлов должно быть 0.
"""

from __future__ import annotations

from __future__ import annotations

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tree_sitter import Language, Parser

# --- AST-фреймы по языкам: {язык: (расширение, имя парсера, функция-шаблон)} ---
FRAMES = {}


def _c_frame(name, anchors):
    a = ", ".join(f"struct page *{x}" for x in anchors[:2])
    body = "    unsigned long nr = 0;\n" + "".join(
        f"    if (p == {x}) nr++;\n" for x in anchors[:2])
    return (f"#include <linux/mm.h>\n\n"
            f"static inline unsigned long {name}(struct page *p)\n{{\n"
            f"{body}    return nr;\n}}\n")


def _rs_frame(name, anchors):
    body = "\n".join(f"    let {x} = {x}_meta();" for x in anchors[:2])
    return (f"pub fn {name}(p: &[u8]) -> u64 {{\n{body}\n"
            f"    p.len() as u64\n}}\n")


def _py_frame(name, anchors):
    body = "\n".join(f"    result = {x}(data)" for x in anchors[:2])
    return (f"def {name}(data):\n{body}\n    return result\n")


def _go_frame(name, anchors):
    body = "\n".join(f"\t{x}()" for x in anchors[:2])
    return (f"func {name}(data []byte) int {{\n{body}\n\treturn len(data)\n}}\n")


PARSER_LANG = {}


def _get_parser(module_name: str) -> Parser:
    if module_name not in PARSER_LANG:
        mod = __import__(module_name)
        PARSER_LANG[module_name] = Parser(Language(mod.language()))
    return PARSER_LANG[module_name]


def synthesize(language: str, name: str, anchors: list[str]) -> tuple[str, dict]:
    frames = {"c": _c_frame, "rust": _rs_frame, "python": _py_frame,
              "go": _go_frame}
    parser_mods = {"c": "tree_sitter_c", "rust": "tree_sitter_rust",
                   "python": "tree_sitter_python", "go": "tree_sitter_go"}
    if language not in frames:
        return "", {"error": f"unknown lang {language}"}
    code = frames[language](name, anchors)
    parser = _get_parser(parser_mods[language])
    tree = parser.parse(code.encode())
    # подсчёт ERROR-узлов
    errors = 0

    def walk(node):
        nonlocal errors
        if node.type == "ERROR" or node.is_missing:
            errors += 1
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return code, {"errors": errors, "root": tree.root_node.type}


def demo():
    tests = [
        ("c", "vmalloc_monitor", ["vmalloc_init", "schedule"]),
        ("rust", "json_parser", ["from_json", "parse"]),
        ("python", "parse_docs", ["parse", "add"]),
        ("go", "kernel_probe", ["readRequestJSON", "add"]),
    ]
    for lang, name, anchors in tests:
        code, info = synthesize(lang, name, anchors)
        status = "VALID" if info.get("errors") == 0 else \
            f"INVALID({info.get('errors')} ERROR)"
        print(f"[{lang}] {name}: {status}")
        if lang == "c":
            print(code)


if __name__ == "__main__":
    demo()

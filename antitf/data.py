from __future__ import annotations

import torch

CORPUS_TASK_0 = [
    "int add(int a, int b) { return a + b; }",
    "for (int i = 0; i < n; i++) { sum += arr[i]; }",
    "if (ptr == NULL) { return -1; }",
    "void swap(int *a, int *b) { int t = *a; *a = *b; *b = t; }",
    "while (*s++) { count++; }",
    "struct Point { int x; int y; };",
    "double dist(struct Point p) { return sqrt(p.x * p.x + p.y * p.y); }",
    "static const char *names[] = { \"alpha\", \"beta\" };",
]

CORPUS_TASK_1 = [
    "switch (cmd) { case 1: start(); break; default: stop(); }",
    "typedef unsigned long ulong_t;",
    "char buf[256]; memcpy(buf, src, len);",
    "if (x > 0 && y < 10) { printf(\"%d\\n\", x); continue; }",
    "size_t n = sizeof(struct Node);",
    "qsort(items, n, sizeof(item_t), cmp_fn);",
    "float f = (float)atoi(argv[1]) / 2.0f;",
    "do { step(); } while (!done);",
]


def make_windows(corpus: list[str], window: int = 64) -> tuple[torch.Tensor, list[bytes]]:
    data = "\n".join(corpus).encode("utf-8")
    starts = range(0, max(len(data) - window, 1), window // 2)
    windows = []
    for s in starts:
        chunk = data[s : s + window]
        chunk = chunk + b"\x00" * (window - len(chunk))
        windows.append(torch.frombuffer(bytearray(chunk), dtype=torch.uint8).clone())
    return torch.stack(windows), [bytes(w.tolist()) for w in windows]

#!/usr/bin/env python3
"""Сгенерировать корпус «hello world» на разных языках для демо-обучения.

Результат: /tmp/hello_world.jsonl (совместим с unified_gpu_train).
Использование: python3 tools/make_hello_world.py
"""
import json
import random

HELLO_WORLD = [
    'fn main() { println!("hello world"); }',
    'int main() { printf("hello world\\n"); return 0; }',
    'print("hello world")',
    'package main; import "fmt"; func main() { fmt.Println("hello world") }',
    '<?php echo "hello world"; ?>',
    'fn main() { println!("hello world"); }',
    'fn greet() { println!("hello world"); }',
    'fn main() { let s = "hello world"; println!("{}", s); }',
    'int main() { puts("hello world"); return 0; }',
    '#include <stdio.h>; int main() { printf("hello world"); return 0; }',
    'fn main() { for _ in 0..3 { println!("hello world"); } }',
    'fn main() { let x = 4; println!("hello world"); }',
    'fn main() { println!("hello world"); println!("hello world"); }',
    '#include <stdio.h>; int main() { fprintf(stdout, "hello world"); }',
    'fn hello() { println!("hello world"); } fn main() { hello(); }',
]


def main():
    rows = [{"code": h} for h in HELLO_WORLD * 20]
    random.shuffle(rows)
    with open("/tmp/hello_world.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"корпус: {len(rows)} строк (hello world на {len(set(HELLO_WORLD))} языках) → /tmp/hello_world.jsonl")


if __name__ == "__main__":
    main()

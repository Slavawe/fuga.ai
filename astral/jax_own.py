#!/usr/bin/env python3
"""OwnJAX: собственная версия JAX в духе бестокенового стека.

Минимальный автодиф-движок: reverse-mode grad + vmap + jit.
Примитивы с vjp-правилами, регистрируемые как VSA-факты (bind op->роль).
Цель: показать, что ИИ может написать собственный чистый аналог JAX
без внешних фреймворков, и связать его с VSA-памятью.
"""

from __future__ import annotations

import functools
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Array(np.ndarray):
    pass


# ---------- примитивы + vjp-правила ----------
VJPS = {}


def _register(name, forward, vjp):
    VJPS[name] = (forward, vjp)


def _add(a, b):
    return a + b

def _add_vjp(a, b):
    return lambda g: (g, g)


def _mul(a, b):
    return a * b

def _mul_vjp(a, b):
    return lambda g: (g * b, g * a)


def _dot(a, b):
    return a @ b

def _dot_vjp(a, b):
    return lambda g: (g @ b.T, a.T @ g)


def _tanh(x):
    return np.tanh(x)

def _tanh_vjp(x):
    t = np.tanh(x)
    return lambda g: g * (1 - t * t)


def _reduce_sum(x):
    return np.sum(x)

def _reduce_sum_vjp(x):
    return lambda g: np.full_like(x, g)


def _exp(x):
    return np.exp(x)

def _exp_vjp(x):
    return lambda g: g * np.exp(x)


for name, fwd, vjp in [
    ("add", _add, _add_vjp), ("mul", _mul, _mul_vjp),
    ("dot", _dot, _dot_vjp), ("tanh", _tanh, _tanh_vjp),
    ("reduce_sum", _reduce_sum, _reduce_sum_vjp), ("exp", _exp, _exp_vjp),
]:
    _register(name, fwd, vjp)


# ---------- прямая запись (tape) ----------
class Tape:
    def __init__(self):
        self.nodes = []

    def record(self, op, inputs, out):
        self.nodes.append((op, inputs, out))
        return out


def _wrap(tape):
    """Возвращает функции-примитивы, записывающие на ленту."""

    def make(op):
        def f(*args):
            fwd, _ = VJPS[op]
            out = fwd(*args)
            tape.record(op, args, out)
            return out
        return f
    return make


# ---------- grad: reverse-mode по ленте ----------
def _unbind(tape, out, grads):
    """Прямой reverse pass: возвращает {node_id: cotangent}."""
    cot = {id(out): grads}
    for op, inputs, out in reversed(tape.nodes):
        if id(out) not in cot:
            continue
        g = cot[id(out)]
        _, vjp = VJPS[op]
        gs = vjp(*inputs)(g)
        gs = gs if isinstance(gs, tuple) else (gs,)
        for inp, gi in zip(inputs, gs):
            cot[id(inp)] = cot.get(id(inp), np.zeros_like(inp)) + gi
    return cot


def grad(f):
    """Возвращает функцию, дающую градиент f по её первому аргументу."""
    def grad_fn(x):
        tape = Tape()
        wrap = _wrap(tape)
        out = f(x, wrap)
        cot = _unbind(tape, out, np.ones_like(out))
        return cot[id(x)] if id(x) in cot else np.zeros_like(x)
    return grad_fn


# ---------- vmap ----------
def vmap(f, in_axes=(0,)):
    """Батчевая версия f по ведущей оси."""
    def batched(*args):
        batch = args[0].shape[0]
        outs = [f(*[a[i] if a.ndim > 0 else a for a in args]) for i in range(batch)]
        return np.stack(outs)
    return batched


# ---------- jit (кэш по форме входов) ----------
def jit(f):
    cache = {}

    @functools.wraps(f)
    def wrapped(*args):
        key = tuple(getattr(a, "shape", None) for a in args)
        if key not in cache:
            cache[key] = f(*args)
        return cache[key]
    return wrapped


# ---------- VSA-регистр операций ----------
def register_ops_vsa(binder, mem):
    """Примитивы OwnJAX -> VSA-факты (op -> роль) в общей памяти."""
    n = 0
    for op in VJPS:
        mem.add_fact("en", f"ownjax:{op}", "primitive", "vjp_rule",
                     dedupe_key=("ownjax", op, "primitive"))
        n += 1
    return n


# ---------- демо/тесты ----------
def main():
    # 1. grad квадрата: d/dx x^2 = 2x
    g = grad(lambda x, f: f("mul")(x, x))
    print(f"[grad] d(x^2)/dx at 3 = {g(np.array(3.0))} (ожидание 6.0)")

    W = np.array([[1.0, 2.0], [3.0, 4.0]])

    # модель, определённая ОДИН раз через именованные примитивы
    def model(x, dot, tanh, reduce_sum):
        h = tanh(dot(x, W))
        return reduce_sum(h)

    # аналитический градиент через ленту
    def grad_analytic(x0):
        tape = Tape()
        wrap = _wrap(tape)
        out = model(x0, wrap("dot"), wrap("tanh"), wrap("reduce_sum"))
        cot = _unbind(tape, out, np.ones_like(out))
        return cot[id(x0)]

    # численный градиент (конечные разности) на тех же примитивах
    def num_fn(x):
        return model(x, _dot, _tanh, _reduce_sum)

    x0 = np.array([0.5, -0.5])
    g_analytic = grad_analytic(x0)
    eps = 1e-6
    g_num = np.array([
        (num_fn(x0 + eps * np.eye(2)[i]) - num_fn(x0 - eps * np.eye(2)[i])) / (2 * eps)
        for i in range(2)
    ])
    print(f"[grad] аналитический {g_analytic} vs численный {g_num} | "
          f"сходятся: {np.allclose(g_analytic, g_num, atol=1e-4)}")

    # 3. vmap
    vm = vmap(lambda x: x * 2)
    print(f"[vmap] {vm(np.array([1.0, 2.0, 3.0]))} (ожидание [2,4,6])")

    # 4. jit
    @jit
    def double(x):
        return x * 2
    print(f"[jit] {double(np.array(4.0))} (ожидание 8.0)")

    # 5. VSA-регистр примитивов
    import fuga_core
    from fuga_memory import PersistentVSAMemory
    binder = fuga_core.HybridBinder(2048)
    mem = PersistentVSAMemory(binder, directory="fuga_memory_ownjax")
    n = register_ops_vsa(binder, mem)
    print(f"[vsa] OwnJAX примитивов в VSA-памяти: {n}")


if __name__ == "__main__":
    main()

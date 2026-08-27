#!/usr/bin/env python3
"""Autonomous satellite: sat_легкий_агент_мониторинга_ядр (self-spawned by mother model).
Connects to Rust core fuga-core (FastVSA bit ops).
Params: ~84.2M, Anchors: 4, Ops: Pi^[0, 1, 2, 3]
Shared VSA memory NOT copied - anchors exported from parent.
"""
import sys
import numpy as np
sys.path.insert(0, '/home/slava/Anti-Tronsformers')

from fuga_core import FastVSA

DIM = 32768
WORDS = DIM // 64
OPS = [0, 1, 2, 3]

_vsa = FastVSA(DIM)

def run(input_bytes: bytes) -> bytes:
    # демаршалинг: байты -> u64 слова -> Rust-ротация -> байты
    arr = np.frombuffer(input_bytes[:DIM // 8], dtype=np.uint64).copy()
    if arr.shape[0] < WORDS:
        pad = np.zeros(WORDS - arr.shape[0], dtype=np.uint64)
        arr = np.concatenate([arr, pad])
    arr = arr[:WORDS]
    for k in OPS:
        arr = np.asarray(_vsa.rotate(arr, k * 64))
    return arr.tobytes()

if __name__ == "__main__":
    data = sys.stdin.buffer.read()
    sys.stdout.buffer.write(run(data))

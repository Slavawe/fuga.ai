# Architecture — Fuga Cognitive Engine

## Layers (data flows left to right)

```
Data (bytes/text/code)
  → Tree-Sitter / Byte Stream
    → fuga_core (PyO3, Rust)
      → FastVSA (Phase Crystal)
        → W/W_patch/KAN (GPU Widrow-Hoff)
          → Decoders (v2/MB/entropy)
```

## Canonical Source of Truth

| Concept | Canonical | Bridge |
|---------|-----------|--------|
| VSA | fuga-core/src/lib.rs (Rust) | FastVSA via PyO3 |
| KAN | src/ai/kan.rs (Rust) | antitf/kan.py (fallback) |
| Memory | astral/core/memory.py | PersistentVSAMemory |
| Code Index | fuga-core/src/code_index.rs | CodeQueryEngine (Py) |

## Directory Map

```
src/           Rust core (55K LOC): TM, H-JEPA, OWM, GPU, decoders
src/bin/       Binaries: production/ tools/ experiments/
astral/        Python self-improvement layer (7K LOC)
  core/        Production: binder, memory, code_memory, file_agent
  models/      Canonical models: mamba_jepa_hybrid, kan
  ingest/      Data pipelines: code_ingest, multilingual_stream
  agents/      Autonomous: satellite, self_improve
  experiments/ Research sandbox (safe to delete)
antitf/        Python↔Rust bridge (PyO3 bindings)
fuga-core/     Rust VSA core + CodeIndexer (PyO3)
cpp/           C++ port (fuga_core.h, decode.cpp)
```

## Build Commands

```bash
# Rust core
cargo build --release

# Python layer (editable)
pip install -e ./astral

# PyO3 bridge
cd fuga-core && maturin develop --release

# Tests
cargo test --release --lib
python -m pytest astral/tests/
```

## Known Issues

1. Flaky: `encoder_is_deterministic_and_512_dimensional` (race in TOKEN_SDR_CACHE)
2. 122 cargo warnings (deprecated API, unused imports)
3. Legacy deps: rapier3d, minifb, hound (unused by main path)

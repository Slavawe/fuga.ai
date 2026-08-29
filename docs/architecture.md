# Architecture — Fuga Cognitive Engine (актуально на 29.08.2026)

## Layers (data flows left to right)

```
Data (bytes/text/code)
  → Tree-Sitter / Byte Stream
    → fuga_core (PyO3, Rust)
      → FastVSA / Phase Crystal (VSA-ядро)
        → Unified Pipeline (Python) → BIM
          → W/W_patch/W_macro Lang-JEPA (GPU Widrow-Hoff)
            → Decoders (v2/MB/entropy/Concept)
              → Generation (text/code/combine)
```

## Canonical Source of Truth

| Concept | Canonical | Bridge |
|---------|-----------|--------|
| VSA (HV) | fuga-core/src/ (Rust) | FastVSA via PyO3 |
| KAN splines | src/ai/kan.rs (Rust + GPU) | antitf/ (fallback) |
| Memory | astral/core/memory.py | PersistentVSAMemory |
| Barlow Twins | astral/models/barlow_twins.py | barlow_loss() |
| lang-jepa concept | astral/models/lang_jepa_adapter.py | FUGA1 tag=8 (CONCEPT_W) |
| HDC/FPE/PhaseCrystal | astral/models/resonator_hdc.py | VSA-ядро |
| Unified Engine | astral/models/unified_engine.py | speak/codegen/combine |
| FugaTokenizer | astral/fuga_tokenizer.py | VSA-якоря 2048-d |

## Current Architecture (29.08.2026)

### Rust Core (58,741 LOC, 174 files, 42 bin-stends)
- `src/ai/` — TM, JEPA, H-JEPA, OWM, KAN, GPU ops, decoders
- `src/bin/` — 42 стенда (production/tools/experiments)
- `src/cli/` — 13 модулей CLI (main.rs 1,085 строк)
- `fuga-core/` — PyO3 bridge (FastVSA, HybridBinder, CodeIndexer)

### Python Astral (9,629 LOC, 82 files)
- `astral/core/` — Pipeline, Memory, Binder, FileAgent
- `astral/models/` — KAN, lang-jepa, Barlow, Resonators, UnifiedEngine
- `astral/agents/` — Satellite, Self-Improve, Evolution
- `astral/ingest/` — Code ingest, Multilingual stream
- `astral/experiments/` — Sandbox

### Format: FUGA1 (magic "FUGA1", binary-compat C++/Rust)
| Tag | Section | Size |
|-----|---------|------|
| 1 | LOCAL_W | 512² f32 |
| 2 | PATCH_W | 512² f32 |
| 3 | OWM_P | 512² f32 |
| 4 | META | steps/ctx/version |
| 5 | HJEPA | optional |
| 6 | KAN_C | 512²×6 f32 |
| 7 | MACRO_W | W_macro f32 |
| 8 | CONCEPT_W | lang-jepa weights f32 |

## Key Improvements (session 28-29.08.2026)

1. **lang-jepa adapter** — EMA-target concept encoder, concept→text decoder
2. **Barlow Twins** — anti-collapse via cross-correlation
3. **HDC Resonator** — N-factor factorization (Frady 2020)
4. **FPE-VSA** — Complex-valued phase, fractional power encodings
5. **PhaseCrystal Resonator** — phase weights for creative mixing
6. **Unified Engine** — single think/speak/codegen/combine cycle
7. **FUGA1 tag=8** — CONCEPT_W serialization

## Best Known Decoder Config (from AGENTS.md v6.2)
```
V2: α=0, τ=0.01, corridor=0, min_cos=0.001, β=0,
    rep_word=0.20, rep_phrase=0.8, window=9 (ctx=8), PHR_LEN=12
MB3: top_k=8, β=0.3, ws_pen=0.30, conf_th=0.05,
     + βm·cos(W_macro·x, lat) + βc·cos(concept, lat)
```

## Build Commands
```bash
# Rust core
cargo build --release

# Python layer (editable)
cd astral && pip install -e .

# PyO3 bridge
cd fuga-core && PATH="$HOME/.cargo/bin:$PATH" ../.venv/bin/maturin develop --release

# Tests
cargo test --release --lib        # 137/137
cd astral && python -m pytest     # 1 test
```

## Known Issues
1. Flaky: `encoder_is_deterministic_and_512_dimensional` (race in TOKEN_SDR_CACHE)
2. ~122 cargo warnings (deprecated API, unused imports)
3. Legacy deps: rapier3d, minifb, hound (unused by main path)
4. 42 bin-stends need feature-gating (production/tools/experiments)
5. Only 1 Python test (need more)
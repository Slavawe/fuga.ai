# ADR-003: Byte-Level Two-Speed as Main Path

## Status: Accepted

## Context
Multiple decoders evaluated on 800 .rs snippets:
- naive byte: 2-3 bytes
- two-speed: 52 bytes (×26)
- entropy/BLT: 200 bytes (full budget, 891 B/s)
- recurrent h(t): 17 bytes
- beam-3: 13 bytes
- KAN: 1 byte
- LSTM peer: 3 bytes (full corpus)

## Decision
Byte-level two-speed latent training (W_local + W_patch + KAN + OWM)
is the canonical main path.

## Rationale
Entropy/BLT decoder achieves 200 bytes with recognizable morphemes.
The generation ceiling is NOT the decoder but the quality of the local
byte-level W (linear landscape doesn't separate structural attractors).
KAN is the only operator that separated linearly inseparable attractors
in synthetic proof — requires calibration for production.

## Consequences
- All training pipelines (CPU/GPU) follow this path
- FUGA1 format stores LOCAL_W + PATCH_W + OWM_P + KAN_C
- H-JEPA/TM cells are auxiliary (stored as optional sections)

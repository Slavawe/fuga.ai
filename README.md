# Fuga 2.0 — VSA Hierarchical Predictive Memory & Code Generation

**Non-transformer AGI core** built on Vector Symbolic Architectures (VSA), Hierarchical JEPA, Temporal Memory, and locality-sensitive binding. Replaces backpropagation with local Delta Rule updates and generates Rust code via pure VSA/TM autoregression — no LLM dependency.

## Current Capabilities

### 1. VSA Token-Level Code Generation
```
cargo run --release -- generate-code "fn new" --tokens
```
- Char-level tokenizer splits on non-alphanumeric chars, recognizes multi-char operators (`->`, `::`, `=>`, `!=`, `==`, `>=`, `<=`, `+=`, `-=`, `&&`, `||`)
- Vocabulary: top-4000 frequent Rust tokens from corpus
- TM-based prediction with **Winner-Take-All (WTA)** — finds best-matching cell instead of bundling all predictions
- **Inhibition of Return (Phase Fatigue)** — penalizes repeatedly-won cells, forces exploration
- **reinforce_match_only** — cross-cell reinforcement doesn't decrement synapses, preserving learned transitions
- Anti-repetition window (16 tokens exact match)
- Output: real Rust syntax — `pub struct AnomalyEvent { #[derive(Clone, Debug)] ... }`

### 2. PhaseNode Autoregressive Generation
```
cargo run --release -- generate-code "fn new"
```
- 27,206 PhaseNodes indexed from Rust stdlib, tokio, smoltcp, fuga source
- `ls_bind` (phase-shift binding) replaces scalar goal_bonus for semantic conditioning
- Language filter: excludes C/C++ nodes when seed contains Rust patterns
- System 2 multi-step reflection: 4 temperatures + convergence check + bundled consensus

### 3. Self-Mirror & Autonomous Indexing
- Indexes source files into PhaseNodes with hierarchical JEPA encoding
- TM trained on token-level bigrams (20,820+ steps) + synthetic Rust patterns
- `index_generated_snippets()` — trains TM+HJEPA on generated output, saves back to mirror

### 4. Anomaly Detection Module
```
cargo test --test test_anomaly_detection
```
- `AnomalyEvent` — detects phase overload (pred_count > 100, power_mw > 500)
- `is_critical()` — triggers when overshoot + power_mw > 1000 (e.g., Morris Worm replication loop)

### 5. MoE Memory & Answer Engine
- 683,972 memory entries across 5 domains (code, narrative, dialogue, forum, general)
- Multi-expert search by text or VSA vector
- Answer engine with resonance attention

## Architecture

| Component | Description |
|---|---|
| Hypervector | 8192-bit, ~2% density (~164 active bits), XOR bind / sum bundle / permute |
| Hierarchical JEPA | L0 (static), L1 (macro), L2 (metacognition) with ls_bind phase-shift |
| Temporal Memory | Cells with DendriteSegments, learn_segment/reinforce, prune, predict_next |
| SDR | encode_text() → deterministic hash-based sparse binary vector |
| Tokenizer | Char-level: splits identifiers from operators, multi-char operator recognition |

## Usage

```bash
# Build
cargo build --release

# Index source files into mirror
cargo run --release -- mirror-index

# Token-level generation
cargo run --release -- generate-code "fn new" --tokens

# PhaseNode generation
cargo run --release -- generate-code "struct Foo"

# Train predictor
cargo run --release -- train-predictor 5 100

# Self-query
cargo run --release -- self-query "async fn handle"

# Run tests
cargo test
cargo test --test test_anomaly_detection
```

## License

Apache-2.0

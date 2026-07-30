# Fuga 2.0 — VSA Hierarchical Predictive Memory & Code Generation

**Non-transformer AGI core** built on Vector Symbolic Architectures (VSA), Hierarchical JEPA, Temporal Memory, and locality-sensitive binding. Replaces backpropagation with local Delta Rule updates and generates Rust code via pure VSA/TM autoregression — no LLM dependency.

---

## Training Pipeline

### 1. Index source files into phase graph

```bash
# Index a directory — reads .rs files, creates PhaseNodes with SDR encoding
cargo run --release -- mirror-index src/ai

# Load existing mirror and index another directory
cargo run --release -- mirror-index src/core
```

Creates `fuga_mirror_nodes.bin` (phase nodes), `fuga_mirror_tm.bin` (TM), `fuga_mirror_jepa.bin` (HJEPA).

### 2. Train the predictor (HJEPA + TM)

```bash
# Train on existing mirror nodes (5 epochs, chunk=1)
cargo run --release -- train-predictor 5

# With larger chunks for sequence patterns
cargo run --release -- train-predictor 10 --chunk 3
```

### 3. Train token vocabulary (embedded in generation)

```bash
# Builds top-4000 char-level token vocab from indexed .rs files
# then trains TM on 20000+ token bigram steps
# then generates tokens
cargo run --release -- generate-code "fn new" --tokens
```

The token trainer:
- Char-level tokenizer: splits identifiers from operators, recognizes `->` `::` `=>` `!=` `==` `>=` `<=` `+=` `-=` `&&` `||`
- Syntactic pattern injection: 14 hardcoded Rust patterns × 5 repeats
- WTA (Winner-Take-All) prediction with Inhibition of Return
- Anti-repetition window (16 tokens)

---

## Generation

### Token-level (syntactic)

```bash
cargo run --release -- generate-code "fn new" --tokens
```

Outputs real Rust tokens: `( ) { } [ ] , :: . ' -> \` + identifiers, numbers

### PhaseNode-level (semantic)

```bash
# Beam search over PhaseNode graph
cargo run --release -- generate-code "struct Foo"

# Autoregressive mode (generates full snippets)
cargo run --release -- generate-code "fn new" --gen

# With beam width and temperature
cargo run --release -- generate-code "async fn" --beam 3 --temp 1.2
```

---

## Query & Evaluation

```bash
# Self-query — find matching phase nodes
cargo run --release -- self-query "async fn handle"

# Evaluate mirror quality
cargo run --release -- eval

# Inspect text or file
cargo run --release -- inspect "fn new() -> Self"
cargo run --release -- inspect src/main.rs
```

---

## Tests

```bash
# All library tests
cargo test --lib

# Anomaly detection (Inhibition of Return, overshoot)
cargo test --test test_anomaly_detection -- --nocapture

# JEPA / TM / MoE tests
cargo test --test jepa_test
cargo test --test hierarchical_jepa_test
cargo test --test moe_routing_test
```

---

## Architecture

| Component | Description |
|---|---|
| Hypervector | 8192-bit, ~2% density (~164 active bits), XOR bind / sum bundle / permute |
| Hierarchical JEPA | L0 (static), L1 (macro), L2 (metacognition) with ls_bind phase-shift |
| Temporal Memory | Cells with DendriteSegments, learn_segment / reinforce / prune / predict_next |
| SDR (Sparse Distributed Representation) | `encode_text()` → deterministic hash-based sparse binary vector |
| Tokenizer | Char-level: splits identifiers from operators, multi-char operator recognition |
| WTA | Winner-Take-All with Inhibition of Return (fatigue = wins × 10, decay every 10 steps) |
| AnomalyEvent | Detects phase overload — `pred_count > 100` or `power_mw > 500` triggers overshoot |

## License

Apache-2.0

# Fuga Omni — Project Summary

## Objective
- Transformer-less RSVP/VSA omni-assistant with MoE memory, GPU-accelerated (CUDA), Web UI, Telegram bot, generative UI, and autonomous zx agent loop.

## Constraints & Environment
- **GPU**: GTX 1660 Ti (6GB VRAM, sm_75), CUDA 12.x
- **VSA cube**: 4³=64, S=4, N=3 → HDV dim 8192 (`fuga_code_cube.bin`)
- **RAM**: 7.5 GB + 7.5 GB swap (OOM in release builds; debug only)
- **Storage**: 25 GB (BTRFS on overlay, 98% full at 16 GB; logs get rotated)
- **Binary serialization**: BSON (`serialize_web`/`deserialize_web`, ~120 MB per domain `.bin`)
- **ZX**: v8.8.8 at `/usr/bin/zx`
- **Telegram**: bot token from `FUGA_TG_TOKEN` env var or `fuga.token` file
- **Serveo tunnel**: dynamic URL, resets on restart (no autoconnect)

## Architecture

### VSA Engine (`cuda_vsa.cu` / `src/ai/`)
- `bind(⊕)`, `bundle(⊗)`, permute(ρ), conjugate — all in CUDA kernels
- Similarity: cosine (bundles) vs. threshold 0.45
- MemoryItem: `{hdv: Vec<f32>, text: String, sig: String, domain: u8, ph_n: u8, ph_total: u8, timestamp: i64}`
- MoE experts: narrative (346K), dialogue (233K), code (103K), forum (1.3K), general (1) = **683,972 total**

### Memory Store (`memory_store.rs`)
- `train()`: bundles 1..N phrases → stores each with domain+phase
- `search_by_text(query)`: first checks HashMap text-index (code domain only, <0.01s), fallback O(N) scan
- `search(query, threshold)`: VSA-LSH first → if no results, text-fallback
- `absorb_with_quality()`: stores code body + signature

### Indexing (code domain only)
- **Text index**: `HashMap<String, Vec<u32>>`, serialized as `*_mem.idx` (42 MB), loads in ~3s
- **VSA-LSH index**: `VsaIndex` in `hnsw.rs` — 3 hash tables × 16384 buckets, 14-bit hashes from random bit positions. Builds in **0.23s** for 103K vectors. Saved as `*_vsa.bin` (103 MB). Multi-probe: checks neighbor buckets (4 probes).

### Web Server (`omni-web.rs`, port 8080)
- `GET /` → embedded SITE_HTML (styled chat UI)
- `POST /api/chat` → `answer_from_moe()` → returns HTML + thinking trace
- `GET /api/stats` → domain sizes + CUDA info
- Starts with pre-cached `.idx` + `.vsa.bin` for code domain

### Telegram Bot (`tgbot.rs`)
- `load_all()` → loads code + narrative + dialogue + forum experts on startup
- Non-command messages → `answer_from_moe()` (all domains via text search)
- `/stats` → domain counts

### CLI (`main.rs`)
Commands:
- `fuga scan` — unified: train + evaluate + serialize MoE
- `fuga ui [file]` — generate UI code from MoE
- `fuga agent <task> [--force]` — full autonomous zx cycle:
  1. Search 103K code patterns
  2. Generate `.mjs` with execSync + learned context
  3. Security AST gate (`child_process`, `exec(`, `__proto__`, etc.)
  4. Execute with `zx <tmpfile>`, capture stdout/stderr
  5. Save experience to `agent_results/agent_<ts>.txt`
  - `--force` overrides security gate
- `fuga generate` — various generation modes
- `fuga sim` — batch simulation

## Current State

### Working
- **Agent loop**: search → generate `.mjs` → security gate → zx exec → save experience (full cycle ~5s)
- **`--force` flag**: bypasses security gate for trusted tasks
- **execSync** in generated scripts (instead of `$` tagged templates that broke with dynamic args)
- **Text index**: code domain, cached on disk, <0.01s queries
- **VSA-LSH**: 0.23s build on 103K vectors, multi-probe search
- **VSA ops**: bind/bundle/permute on GPU; text `search_by_text` fallback always returns something

### Issues
- **VSA vector search**: cosine threshold 0.45 → 0 matches on 103K code vectors (entropy ~0.5). All results come from text-fallback
- **OOM in release mode**: can only compile/run debug
- **Disk 98% full**: BTRFS on overlay, need log rotation
- **Serveo tunnel**: manual restart required

## Next Steps
1. `fuga train --absorb agent_results/*.txt` — absorb successful agent results into MoE
2. Train more dialogue (80 OpenSubtitles chunks) and JS/HTML (freeCodeCamp, mdn/content)
3. Fix VSA vector similarity (lower threshold? different cube params?)
4. Autoconnect Serveo on boot

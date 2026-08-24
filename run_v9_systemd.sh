#!/bin/bash
# v9 smoke/full через systemd-run: переживает перезапуски Hermes-шлюза.
# Аргументы: $1 = max_steps (напр. 30000 для smoke, 1500000 для полного)
set -e
cd /home/slava/fuga
STEPS="${1:-30000}"
OUT="${2:-/tmp/v9_smoke.fuga}"
LOG="${OUT%.fuga}.log"
exec ./target/release/unified_gpu_train \
  --jsonl "fisig_corpus.jsonl,corpus_doc_code_pairs.jsonl,training_stack.jsonl,corpus.jsonl" \
  --max-steps "$STEPS" --batch 256 --ckpt-every 500000 --ctx 8 --patch-ctx 8 \
  --lambda-patch 0.4 --lambda-floor 0.10 --lambda-tau 500000 --lr-macro 0.05 \
  --out "$OUT" \
  --text-seed "the force of gravity is" \
  --code-seed "fn main() {" > "$LOG" 2>&1
echo "DONE:$STEPS $OUT" >> "$LOG"
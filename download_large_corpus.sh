#!/bin/bash
# Massive text corpus download for Fuga conversational training
set -e
DIR="/home/slava/fuga/text_corpus"
mkdir -p "$DIR/raw" "$DIR/dialogue" "$DIR/literature" "$DIR/forum"

echo "╔══════════════════════════════════════════╗"
echo "║  Fuga Large Text Corpus Download        ║"
echo "╚══════════════════════════════════════════╝"

# 1. OpenSubtitles Russian (dialogues)
echo ""
echo "[1/5] OpenSubtitles Russian..."
SUBS="$DIR/dialogue/opensubtitles"
mkdir -p "$SUBS"

for attempt in 1 2; do
  if [ ! -f "$SUBS/ru_large.txt" ]; then
    echo "  Attempt $attempt..."
    wget -q --timeout=30 "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/ru.txt.gz" -O /tmp/ru_full.gz 2>/dev/null && {
      gunzip -c /tmp/ru_full.gz | head -500000 > "$SUBS/ru_large.txt"
      rm /tmp/ru_full.gz
      echo "  ✓ $(wc -l < "$SUBS/ru_large.txt") lines"
      break
    } || echo "  ✗ failed"
  fi
done

# 2. Gutenberg
echo ""
echo "[2/5] Project Gutenberg..."
GUTEN="$DIR/literature/gutenberg"
mkdir -p "$GUTEN"
for id in $(seq 1 50); do
  [ -f "$GUTEN/pg$id.txt" ] && continue
  wget -q --timeout=5 "https://www.gutenberg.org/cache/epub/$id/pg$id.txt" -O "$GUTEN/pg$id.txt" 2>/dev/null && printf "." || { rm -f "$GUTEN/pg$id.txt"; }
done
echo ""
echo "  ✓ $(find "$GUTEN" -name '*.txt' -size +10c | wc -l) books"

# 3. Wikipedia RU
echo ""
echo "[3/5] Wikipedia RU..."
python3 /home/slava/fuga/gen_wiki.py 2>/dev/null || echo "  ✗ failed"

# 4. Synthetic dialogues + prose
echo ""
echo "[4/5] Generating synthetic data..."
python3 /home/slava/fuga/gen_synthetic.py 2>/dev/null && echo "  ✓ done"

# 5. Code comments as forum data
echo ""
echo "[5/5] Code comments..."
python3 /home/slava/fuga/gen_comments.py 2>/dev/null && echo "  ✓ done"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Complete                               ║"
echo "╚══════════════════════════════════════════╝"
echo ""
find "$DIR" -name '*.txt' -size +10c | wc -l
du -sh "$DIR"
echo ""
echo "Run: fuga train-text text_corpus/"

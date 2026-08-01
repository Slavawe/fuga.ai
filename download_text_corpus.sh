#!/bin/bash
# Download text corpora for Fuga Conversational-Literary Cube
set -e
DIR="/home/slava/fuga/text_corpus"
mkdir -p "$DIR/raw" "$DIR/dialogue" "$DIR/literature" "$DIR/forum"

echo "Downloading text corpora for Fuga..."

# 1. Russian literature from lib.ru (public domain classics)
echo "[1/4] Russian literature (lib.ru samples)..."
LIBRU="$DIR/literature"
pushd "$LIBRU" > /dev/null
# Short stories by Chekhov, Pushkin, Gogol (public domain)
for url in \
  "https://lib.ru/LITRA/CHEHOW/chehov_rasskazy.txt" \
  "https://lib.ru/LITRA/PUSHKIN/pushkin_stihi.txt" \
  "https://lib.ru/LITRA/GOGOL/gogol_nos.txt" \
  "https://lib.ru/LITRA/TOLSTOY/tolstoy_rasskazy.txt"; do
  wget -q --timeout=10 "$url" 2>/dev/null && echo "  Downloaded $(basename $url)" || echo "  Skip $(basename $url)"
done
popd > /dev/null

# 2. Project Gutenberg Russian texts
echo "[2/4] Project Gutenberg (Russian)..."
GUTEN="$DIR/literature/gutenberg"
mkdir -p "$GUTEN"
pushd "$GUTEN" > /dev/null
# Russian literature in public domain on PG
for book_id in 2554 20010 132 2555 23907 2556 2197 600 1513; do
  wget -q --timeout=10 "https://www.gutenberg.org/cache/epub/$book_id/pg$book_id.txt" -O "pg$book_id.txt" 2>/dev/null && echo "  Book $book_id OK" || echo "  Book $book_id skip"
done
popd > /dev/null

# 3. OpenSubtitles Russian sample (small subset)
echo "[3/4] OpenSubtitles Russian sample..."
SUBS="$DIR/dialogue/opensubtitles"
mkdir -p "$SUBS"
pushd "$SUBS" > /dev/null
wget -q --timeout=15 "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/ru.txt.gz" -O "ru.txt.gz" 2>/dev/null && {
  gunzip -f "ru.txt.gz"
  # Take first 50000 lines as sample
  head -50000 "ru.txt" > "ru_sample.txt"
  rm "ru.txt"
  echo "  OpenSubtitles sample: $(wc -l < ru_sample.txt) lines"
} || echo "  OpenSubtitles skip (server may be slow)"
popd > /dev/null

# 4. Wikipedia Russian sample
echo "[4/4] Wikipedia Russian sample..."
WIKI="$DIR/forum/wiki"
mkdir -p "$WIKI"
pushd "$WIKI" > /dev/null
wget -q --timeout=15 "https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles1.xml.bz2" -O "ruwiki.xml.bz2" 2>/dev/null && {
  bzip2 -d "ruwiki.xml.bz2"
  # Extract first 5000 lines that look like text
  grep -v "^<" "ruwiki.xml" | head -20000 > "ru_wiki_sample.txt"
  rm "ruwiki.xml"
  echo "  Wikipedia sample: $(wc -l < ru_wiki_sample.txt) lines"
} || echo "  Wikipedia skip (large file)"
popd > /dev/null

# Summary
echo ""
echo "=== Download Summary ==="
echo "Literature: $(find $DIR/literature -name '*.txt' 2>/dev/null | wc -l) files"
echo "Dialogue:   $(find $DIR/dialogue -name '*.txt' 2>/dev/null | wc -l) files"
echo "Total size: $(du -sh $DIR 2>/dev/null | cut -f1)"
echo ""
echo "Next: run 'fuga train-text text_corpus/' to absorb into cube"

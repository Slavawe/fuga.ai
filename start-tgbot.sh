#!/bin/bash
set -e

echo "🧠 Fuga AI Telegram Bot Launcher"

CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-target}
BIN="$CARGO_TARGET_DIR/release/fuga-tgbot"

if [ ! -f "$BIN" ] || [ "$1" = "--rebuild" ]; then
    echo "⚙️  Building fuga-tgbot (release)..."
    cargo build --release --bin fuga-tgbot --manifest-path "$(dirname "$0")/Cargo.toml"
fi

if [ -n "$FUGA_TG_TOKEN" ]; then
    echo "Using FUGA_TG_TOKEN env var"
elif [ -f "fuga.token" ]; then
    echo "Using token from ./fuga.token"
else
    echo "Using hardcoded fallback token (from src/bin/tgbot.rs)"
fi

if [ ! -f "fuga_cube.bin" ]; then
    echo "ℹ️  No fuga_cube.bin found — bot will start with fresh random cube."
    echo "   Train first: cargo run -- release -- train <corpus.jsonl>"
fi

echo ""
echo "Starting bot..."
exec "$BIN"

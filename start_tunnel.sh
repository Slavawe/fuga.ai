#!/bin/bash
PORT=${1:-8080}
DIR="/home/slava/fuga"

# Kill old tunnels
pkill -f "serveo.net" 2>/dev/null

# Start tunnel, capture output
ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
    -R 80:localhost:"$PORT" serveo.net > "$DIR/tunnel.log" 2>&1 &
PID=$!

# Wait for URL
for i in $(seq 1 15); do
    sleep 2
    URL=$(grep -oP 'https://[a-zA-Z0-9.-]+\.serveousercontent\.com' "$DIR/tunnel.log" 2>/dev/null | tail -1)
    if [ -n "$URL" ]; then
        echo "$URL" > "$DIR/tunnel_url.txt"
        echo ""
        echo "  Public URL: $URL"
        echo "  PID: $PID"
        echo ""
        exit 0
    fi
done

echo "Failed to get tunnel URL"
exit 1

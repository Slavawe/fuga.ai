#!/bin/bash
DIR="/home/slava/fuga"
cd "$DIR"

echo "Starting Fuga Web + Public Tunnel..."
echo ""

# Kill old
pkill -f "fuga-web" 2>/dev/null
pkill -f "serveo.net" 2>/dev/null
sleep 1

# Start web server
FUGA_CUBE_PATH="$DIR/fuga_code_cube.bin" FUGA_WEB_PORT=8080 \
  nohup "$DIR/target/release/fuga-web" > "$DIR/web_output.log" 2>&1 &
echo "  Web server: PID $!"

# Wait for server
sleep 4

# Start tunnel
ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
    -R 80:localhost:8080 serveo.net > "$DIR/tunnel.log" 2>&1 &
TPID=$!

for i in $(seq 1 15); do
    sleep 2
    URL=$(grep -oP 'https://[a-zA-Z0-9.-]+\.serveousercontent\.com' "$DIR/tunnel.log" 2>/dev/null | tail -1)
    if [ -n "$URL" ]; then
        echo "$URL" > "$DIR/tunnel_url.txt"
        echo ""
        echo "  Local:  http://localhost:8080"
        echo "  LAN:    http://192.168.0.119:8080"
        echo "  Public: $URL"
        echo ""
        echo "  Web PID: $(pgrep -f 'fuga-web' | head -1)"
        echo "  Tunnel PID: $TPID"
        echo ""
        exit 0
    fi
done

echo "Tunnel failed - check $DIR/tunnel.log"
exit 1

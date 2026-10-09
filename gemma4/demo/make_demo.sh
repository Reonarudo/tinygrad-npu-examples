#!/usr/bin/env bash
# Makes gemma4/serve-demo.gif with nobody at the keyboard: starts gemma4_serve.py on the board over ssh and waits until it answers,
# records the Ollama CLI on this host asking it (VHS, serve-demo.tape: typed, then streamed at the board's real speed), scales the
# recording to a 480 x 270 GIF at 12.5 frames a second (bonsai2-27b/serve-demo.gif's size and rate), and stops the server.
#
#   DEMO_SSH=<board> gemma4/demo/make_demo.sh
#
# On this host: vhs (`brew install vhs`, which brings ttyd and ffmpeg), the `ollama` CLI (the client only), ssh to the board.
# On the board: the gemma4 example packed (gemma4_pack.py, gemma4_mtp.py) and its requirements; nothing else is changed there.
# Env:
#   DEMO_SSH     ssh destination of the board (required)
#   DEMO_HOST    the board's address for HTTP from this host (default: DEMO_SSH's host part)
#   DEMO_SERVE   the board's command that runs the server in the foreground on port 8000
#                (default: cd ~/example-tinygrad-npu-uses/gemma4 && python3 gemma4_serve.py --host 0.0.0.0 --port 8000)
#   DEMO_MODEL   the model name the server reports (default gemma4-e2b; gemma4-e4b with an E4B cache)
#   DEMO_PROMPT  the question (no backticks, '|' or '@')
#   DEMO_OUT     the GIF (default gemma4/serve-demo.gif)
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
HERE=$(cd "$(dirname "$0")" && pwd)
: "${DEMO_SSH:?set DEMO_SSH to the ssh destination of the board}"
DEMO_HOST=${DEMO_HOST:-${DEMO_SSH#*@}}
DEMO_SERVE=${DEMO_SERVE:-"cd ~/example-tinygrad-npu-uses/gemma4 && python3 gemma4_serve.py --host 0.0.0.0 --port 8000"}
DEMO_MODEL=${DEMO_MODEL:-gemma4-e2b}
DEMO_PROMPT=${DEMO_PROMPT:-"Write a short Python script that prints the first 10 prime numbers, then explain it in one sentence."}
DEMO_OUT=${DEMO_OUT:-$HERE/../serve-demo.gif}
for t in vhs ffmpeg ollama curl; do command -v $t > /dev/null || { echo "make_demo: $t not found" >&2; exit 1; }; done
URL=http://$DEMO_HOST:8000
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT

# the server in a session of its own on the board, so that stopping it stops everything it started
if curl -sf -m 5 "$URL/api/version" > /dev/null; then echo "make_demo: something already answers at $URL" >&2; exit 1; fi
START="setsid nohup bash -c '$DEMO_SERVE' > /tmp/gemma4-demo-serve.log 2>&1 < /dev/null & echo \$!"
PGID=$(ssh "$DEMO_SSH" "$START")
stop() { ssh "$DEMO_SSH" "kill -TERM -- -$PGID 2> /dev/null; true"; rm -rf "$WORK"; }
trap stop EXIT
echo "make_demo: server started on $DEMO_SSH (process group $PGID); waiting for $URL"
for i in $(seq 600); do
  curl -sf -m 5 "$URL/api/version" > /dev/null && break
  ssh "$DEMO_SSH" "kill -0 $PGID 2> /dev/null" || { ssh "$DEMO_SSH" "tail -20 /tmp/gemma4-demo-serve.log"; echo "make_demo: the server exited" >&2; exit 1; }
  sleep 2
done
curl -sf -m 120 "$URL/api/generate" -d "{\"model\": \"$DEMO_MODEL\", \"prompt\": \"Hi\", \"stream\": false, \"options\": {\"num_predict\": 4}}" > /dev/null   # (a first request)

sed -e "s|@OUT@|\"$WORK/demo.mp4\"|" -e "s|@HOST@|$DEMO_HOST|" -e "s|@MODEL@|$DEMO_MODEL|" -e "s|@PROMPT@|$DEMO_PROMPT|" "$HERE/serve-demo.tape" > "$WORK/demo.tape"
(cd "$WORK" && vhs demo.tape > vhs.log 2>&1) || { tail -20 "$WORK/vhs.log"; exit 1; }
ffmpeg -v error -y -i "$WORK/demo.mp4" -vf "fps=12.5,scale=480:270:flags=lanczos,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" "$DEMO_OUT"
echo "make_demo: $DEMO_OUT ($(du -h "$DEMO_OUT" | cut -f1))"

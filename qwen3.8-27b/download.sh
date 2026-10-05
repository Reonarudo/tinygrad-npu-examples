#!/bin/bash
# Fetch Qwen/Qwen3.8-27B-FP8 (27 GB, Apache-2.0, not gated) into $QWEN_DIR (default /mnt/ssd/qwen3.8-27b-fp8). Resumable.
set -e
D=${QWEN_DIR:-/mnt/ssd/qwen3.8-27b-fp8}; mkdir -p "$D"; cd "$D"
REPO=Qwen/Qwen3.8-27B-FP8; B=https://huggingface.co/$REPO/resolve/main
AUTH=(); [ -f ~/.hf_token ] && AUTH=(-H "Authorization: Bearer $(cat ~/.hf_token)")
files=$(curl -sfL "${AUTH[@]}" "https://huggingface.co/api/models/$REPO/tree/main" \
        | python3 -c 'import sys, json; print(" ".join(e["path"] for e in json.load(sys.stdin) if e["type"] == "file"))')
for f in $files; do
  echo "== $f $(date +%T)"
  curl -sfL -C - "${AUTH[@]}" "$B/$f" -o "$f" || { echo "FAILED $f"; exit 1; }
done
echo "== done $(date +%T): $(du -sh "$D" | cut -f1) in $D"

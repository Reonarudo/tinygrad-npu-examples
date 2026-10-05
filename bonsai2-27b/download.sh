#!/bin/bash
# Fetch what the Bonsai 2 27B example needs into $BONSAI_DIR (default /mnt/ssd/bonsai2; set it to your path). Resumable: a file
# already complete (its size equals the repository's) is skipped, a partial one is continued.
#   prism-ml/Ternary-Bonsai-2-27B-gguf (Apache-2.0): Ternary-Bonsai-2-27B-PTQ1_0.gguf (5.9 GB) and its LICENSE / NOTICE / README
#   Qwen/Qwen3.8-27B-FP8 (Apache-2.0), into $BONSAI_DIR/qwen3.8-27b-mtp: only mtp.safetensors (0.48 GB, the speculative-decoding
#     drafter), tokenizer.json, and the small config / license files -- not the 27 GB checkpoint
#   bash download.sh            download (about 6.4 GB)
#   bash download.sh --dry-run  list the files, their sizes and what is already present; download nothing
set -e
D=${BONSAI_DIR:-/mnt/ssd/bonsai2}; Q=$D/qwen3.8-27b-mtp
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
AUTH=(); [ -f ~/.hf_token ] && AUTH=(-H "Authorization: Bearer $(cat ~/.hf_token)")
fetch() {   # fetch <repo> <dest dir> <file...>
  local repo=$1 dest=$2; shift 2; [ $DRY = 1 ] || mkdir -p "$dest"
  local sizes; sizes=$(curl -sfL "${AUTH[@]}" "https://huggingface.co/api/models/$repo/tree/main" \
    | python3 -c 'import sys, json; [print(e["path"], e["size"]) for e in json.load(sys.stdin) if e["type"] == "file"]') \
    || { echo "FAILED to list $repo"; exit 1; }
  for f in "$@"; do
    local want; want=$(awk -v f="$f" '$1 == f { print $2 }' <<< "$sizes")
    [ -n "$want" ] || { echo "FAILED: $f is not in $repo"; exit 1; }
    local have=0; [ -e "$dest/$f" ] && have=$(stat -L -c %s "$dest/$f" 2>/dev/null || stat -L -f %z "$dest/$f")
    if [ "$have" = "$want" ]; then echo "== $repo/$f: complete ($want bytes)"; continue; fi
    echo "== $repo/$f: $want bytes, $have present $(date +%T)"; [ $DRY = 1 ] && continue
    curl -sfL -C - "${AUTH[@]}" "https://huggingface.co/$repo/resolve/main/$f" -o "$dest/$f" || { echo "FAILED $f"; exit 1; }
  done
}
fetch prism-ml/Ternary-Bonsai-2-27B-gguf "$D" Ternary-Bonsai-2-27B-PTQ1_0.gguf LICENSE NOTICE.txt README.md
fetch Qwen/Qwen3.8-27B-FP8 "$Q" mtp.safetensors tokenizer.json tokenizer_config.json config.json LICENSE
[ $DRY = 1 ] && { echo "== dry run: nothing downloaded"; exit 0; }
echo "== done $(date +%T): $(du -shL "$D" | cut -f1) in $D"
echo "   BONSAI_GGUF=$D/Ternary-Bonsai-2-27B-PTQ1_0.gguf; the drafter's files (bonsai2_pack.py --mtp): $Q"

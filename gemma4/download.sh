#!/bin/bash
# Fetch the Gemma 4 GGUFs this example runs into $GEMMA_DIR (default /mnt/ssd/models/gemma-4, where the scripts look; set it to
# your path). Resumable: a file already complete (its size equals the repository's) is skipped, a partial one is continued; every
# file downloaded is checked against the sha256 the repository lists for it.
#   ggml-org/gemma-4-E2B-it-GGUF (Apache-2.0): gemma-4-E2B-it-Q8_0.gguf (5.0 GB) and its MTP draft head mtp-gemma-4-E2B-it-Q8_0.gguf
#     (0.1 GB, for speculative decoding: the generator finds it beside the model's file)
#   ggml-org/gemma-4-E4B-it-GGUF (Apache-2.0): gemma-4-E4B-it-Q8_0.gguf (8.0 GB) and mtp-gemma-4-E4B-it-Q8_0.gguf (0.1 GB)
#   bash download.sh [e2b | e4b | both]   download (default e2b: about 5.1 GB; e4b about 8.1 GB)
#   bash download.sh ... --dry-run        list the files, their sizes and what is already present; download nothing
#   bash download.sh ... --verify         also check the sha256 of the files already complete
set -e
D=${GEMMA_DIR:-/mnt/ssd/models/gemma-4}
DRY=0; VERIFY=0; WHICH=e2b
for a in "$@"; do
  case $a in
    --dry-run) DRY=1;; --verify) VERIFY=1;;
    e2b|e4b|both) WHICH=$a;;
    *) echo "usage: bash download.sh [e2b | e4b | both] [--dry-run] [--verify]"; exit 1;;
  esac
done
AUTH=(); [ -f ~/.hf_token ] && AUTH=(-H "Authorization: Bearer $(cat ~/.hf_token)")
sha256() { if command -v sha256sum > /dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
check() {   # check <file> <sha256>
  echo "== $(basename "$1"): sha256 $(date +%T)"
  local got; got=$(sha256 "$1")
  [ "$got" = "$2" ] || { echo "FAILED: $1 has sha256 $got, the repository lists $2 (delete it and run again)"; exit 1; }
}
fetch() {   # fetch <repo> <file...>
  local repo=$1; shift; [ $DRY = 1 ] || mkdir -p "$D"
  local list; list=$(curl -sfL "${AUTH[@]}" "https://huggingface.co/api/models/$repo/tree/main" \
    | python3 -c 'import sys, json; [print(e["path"], e["size"], (e.get("lfs") or {}).get("oid", "-")) for e in json.load(sys.stdin) if e["type"] == "file"]') \
    || { echo "FAILED to list $repo"; exit 1; }
  for f in "$@"; do
    local want sum; read -r want sum <<< "$(awk -v f="$f" '$1 == f { print $2, $3 }' <<< "$list")"
    [ -n "$want" ] || { echo "FAILED: $f is not in $repo"; exit 1; }
    local have=0; [ -e "$D/$f" ] && have=$(stat -L -c %s "$D/$f" 2>/dev/null || stat -L -f %z "$D/$f")
    if [ "$have" = "$want" ]; then
      echo "== $repo/$f: complete ($want bytes)"
      [ $DRY = 0 ] && [ $VERIFY = 1 ] && check "$D/$f" "$sum"
      continue
    fi
    echo "== $repo/$f: $want bytes, $have present $(date +%T)"; [ $DRY = 1 ] && continue
    curl -sfL -C - "${AUTH[@]}" "https://huggingface.co/$repo/resolve/main/$f" -o "$D/$f" || { echo "FAILED $f"; exit 1; }
    check "$D/$f" "$sum"
  done
}
case $WHICH in e2b|both) fetch ggml-org/gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q8_0.gguf mtp-gemma-4-E2B-it-Q8_0.gguf;; esac
case $WHICH in e4b|both) fetch ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q8_0.gguf mtp-gemma-4-E4B-it-Q8_0.gguf;; esac
[ $DRY = 1 ] && { echo "== dry run: nothing downloaded"; exit 0; }
echo "== done $(date +%T): $(du -shL "$D" | cut -f1) in $D"
echo "   GEMMA_GGUF=$D/gemma-4-E2B-it-Q8_0.gguf (E4B: $D/gemma-4-E4B-it-Q8_0.gguf); the draft heads are beside them as mtp-<file name>"

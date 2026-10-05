#!/bin/bash
# Fetch ideogram-ai/ideogram-4-fp8 into ~/ideogram4 (gated: a read token in ~/ideogram4/.token), then pack the two
# transformers' weights into the caches the sampler reads. Resumable: curl continues partial files, packing skips a
# finished cache.
set -e
D=${IDEOGRAM4_DIR:-~/ideogram4}; D=$(eval echo "$D"); mkdir -p "$D"; cd "$D"
[ -f .token ] || { echo "put a Hugging Face read token (with access to ideogram-ai/ideogram-4-fp8) in $D/.token"; exit 1; }
[ -n "$TG" ] || { echo "set TG to the tinygrad checkout (the repository's tinygrad/ submodule)"; exit 1; }
B=https://huggingface.co/ideogram-ai/ideogram-4-fp8/resolve/main
HERE=$(cd "$(dirname "$0")" && pwd)

fetch() {  # $1 = path in the repo
  mkdir -p "$(dirname "$1")"
  echo "== $1 $(date +%T)"
  curl -sfL -C - -H "Authorization: Bearer $(cat .token)" "$B/$1" -o "$1" || { echo "FAILED $1"; exit 1; }
}
for f in model_index.json scheduler/scheduler_config.json \
         transformer/config.json transformer/diffusion_pytorch_model.safetensors.index.json transformer/diffusion_pytorch_model.safetensors \
         unconditional_transformer/config.json unconditional_transformer/diffusion_pytorch_model.safetensors.index.json unconditional_transformer/diffusion_pytorch_model.safetensors \
         vae/config.json vae/diffusion_pytorch_model.safetensors; do
  fetch "$f"
done
# the text encoder (Qwen3-VL-8B) and its tokenizer: every file of the two folders
for dir in text_encoder tokenizer; do
  for f in $(curl -sfL -H "Authorization: Bearer $(cat .token)" "https://huggingface.co/api/models/ideogram-ai/ideogram-4-fp8/tree/main/$dir" \
            | python3 -c 'import sys, json; print(" ".join(e["path"] for e in json.load(sys.stdin) if e["type"] == "file"))'); do
    fetch "$f"
  done
done

# the packed weight caches (E4M3 panels in the GEMM layout): ~8.4 GB each, a few minutes each
[ -f fpcache/done ]      || { python3 "$HERE/ideogram4_fp_pack.py" --weights unconditional_transformer --out fpcache --layers 0-33 && touch fpcache/done; }
[ -f fpcache_cond/done ] || { python3 "$HERE/ideogram4_fp_pack.py" --weights transformer --out fpcache_cond --layers 0-33 && touch fpcache_cond/done; }
echo "== done $(date +%T): $D"

#!/bin/bash
# One prompt to one image on the NPU:  ./run.sh "prompt" out/name [preset] [seed] [size]
#   -> out/name_text.npz (the conditioning), out/name_latents.npy (the sampled latents), out/name.png
set -e
[ $# -ge 2 ] || { echo "usage: $0 \"prompt\" out/name [V4_TURBO_12|V4_DEFAULT_20|V4_QUALITY_48] [seed] [size]"; exit 1; }
[ -n "$TG" ] || { echo "set TG to the tinygrad checkout (the repository's tinygrad/ submodule)"; exit 1; }
PROMPT=$1; OUT=$2; PRESET=${3:-V4_TURBO_12}; SEED=${4:-123}; SIZE=${5:-1024}
D=${IDEOGRAM4_DIR:-~/ideogram4}; D=$(eval echo "$D")
HERE=$(cd "$(dirname "$0")" && pwd); mkdir -p "$(dirname "$OUT")"

echo "== 1/3 text conditioning (host CPU, ~2.5 min)"
python3 -u "$HERE/ideogram4_text.py" --prompt "$PROMPT" --out "${OUT}_text.npz" \
  --te "$D/text_encoder" --tok "$D/tokenizer" --weights "$D/transformer"

echo "== 2/3 sampling on the NPU ($PRESET, ${SIZE}px, seed $SEED)"
python3 -u "$HERE/ideogram4_fp_1024.py" --text "${OUT}_text.npz" --preset "$PRESET" --size "$SIZE" --seed "$SEED" \
  --cond "$D/fpcache_cond" --uncond "$D/fpcache" --wcond "$D/transformer" --wuncond "$D/unconditional_transformer" \
  --out "${OUT}_latents.npy"

echo "== 3/3 VAE decode on the NPU"
python3 -u "$HERE/vae_npu.py" "${OUT}_latents.npy" --out "${OUT}.png"
echo "== ${OUT}.png"

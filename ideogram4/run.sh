#!/bin/bash
# One prompt to one image on the NPU:  ./run.sh "prompt" out/name [preset] [seed] [size]
#   -> out/name_text.npz (the conditioning), out/name_latents.npy (the sampled latents), out/name.png
# Q8=1: the W8A8 linears on the AIFF (the packs in $D/q8pack, the descriptor templates in $D/q8desc; see README).
# NPU_TEXT=0: the text encoder in host numpy instead of on the NPU; TEXT_PY then picks its Python (a numpy with OpenBLAS is ~30x
# faster than the reference BLAS).
set -e
[ $# -ge 2 ] || { echo "usage: $0 \"prompt\" out/name [V4_TURBO_12|V4_DEFAULT_20|V4_QUALITY_48] [seed] [size]"; exit 1; }
[ -n "$TG" ] || { echo "set TG to the tinygrad checkout (the repository's tinygrad/ submodule)"; exit 1; }
PROMPT=$1; OUT=$2; PRESET=${3:-V4_TURBO_12}; SEED=${4:-123}; SIZE=${5:-1024}
D=${IDEOGRAM4_DIR:-~/ideogram4}; D=$(eval echo "$D")
HERE=$(cd "$(dirname "$0")" && pwd); mkdir -p "$(dirname "$OUT")"

if [ "${NPU_TEXT:-1}" = 1 ]; then   # the text encoder on the NPU (the cache $D/te_p48 from ideogram4_te_pack.py); NPU_TEXT=0: host numpy
  echo "== 1/3 text conditioning (NPU)"
  python3 -u "$HERE/ideogram4_text.py" --npu --cache "$D/te_p48" --prompt "$PROMPT" --out "${OUT}_text.npz" \
    --te "$D/text_encoder" --tok "$D/tokenizer" --weights "$D/transformer"
else
  echo "== 1/3 text conditioning (host CPU)"
  ${TEXT_PY:-python3} -u "$HERE/ideogram4_text.py" --prompt "$PROMPT" --out "${OUT}_text.npz" \
    --te "$D/text_encoder" --tok "$D/tokenizer" --weights "$D/transformer"
fi

if [ -n "$Q8" ]; then   # the AIFF descriptor templates for this prompt's row count (once per row count; existing ones are skipped)
  [ -n "$AIFF_TG" ] || { echo "Q8=1 needs AIFF_TG (the tinygrad tree with the AIFF register generator, see README)"; exit 1; }
  python3 -u "$HERE/ideogram4_aiff_desc.py" --text "${OUT}_text.npz" --out "$D/q8desc" --pipeline
fi

echo "== 2/3 sampling on the NPU ($PRESET, ${SIZE}px, seed $SEED)"
python3 -u "$HERE/ideogram4_fp_1024.py" --text "${OUT}_text.npz" --preset "$PRESET" --size "$SIZE" --seed "$SEED" \
  --cond "$D/fpcache_cond" --uncond "$D/fpcache" --wcond "$D/transformer" --wuncond "$D/unconditional_transformer" \
  ${Q8:+--q8 "$D/q8pack" --q8desc "$D/q8desc"} --out "${OUT}_latents.npy"

echo "== 3/3 VAE decode on the NPU"
python3 -u "$HERE/vae_npu.py" "${OUT}_latents.npy" --out "${OUT}.png"
echo "== ${OUT}.png"

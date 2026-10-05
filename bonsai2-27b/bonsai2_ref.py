#!/usr/bin/env python3
"""Ternary Bonsai 2 27B through qwen3.8-27b's numpy fp32 reference (qwen38_ref.py, profile bonsai2-27b), on the host CPU.

One teacher-forced pass over a token sequence: the embedding, the 64 layers one after another (each layer's weights are
dequantised as its matmuls stream them, 4096 rows at a time, so the resident set stays at a few GB), the final norm and
the head. The logits at every position are written, so a single pass checks both a reference implementation's prompt logits
and its greedy continuation (feed the prompt plus the generated tokens but the last: position i must predict token i + 1).

    python3 bonsai2_ref.py --tokens out/run.tokens -o out/ref [--layers]   # the prompt's token ids (e.g. from tokenizer.json)
    python3 bonsai2_ref.py --ids 9707,11,847 -o out/ref

Writes <o>.logits (float32 [n, vocab]) and with --layers <o>.l_out-<l> (float32 [n, 5120], each layer's output) and
<o>.model.input_embed, the names the reference dump gives the same tensors. With --qwen38 FOLDER the same pass runs Qwen3.8-27B
from its FP8 checkpoint instead (profile qwen3.8-27b), for a comparison of the two models' residual bases."""
import argparse, os, sys, time
_Q38 = next((sys.argv[i + 1] for i, v in enumerate(sys.argv[:-1]) if v == "--qwen38"), None)
os.environ["QWEN_MODEL"] = "qwen3.8-27b" if _Q38 else "bonsai2-27b"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qwen3.8-27b"))
import numpy as np
import qwen38_ref as R                                                      # noqa: E402

def read_tokens(path):
  """A reference dump's .tokens: the prompt ("p id") and the greedy tokens ("g id"); the sequence fed is prompt + gen[:-1]."""
  p, g = [], []
  for line in open(path):
    k, v = line.split(); (p if k == "p" else g).append(int(v))
  return p, g

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--gguf", default="/mnt/ssd/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf")
  ap.add_argument("--tokens"); ap.add_argument("--ids"); ap.add_argument("-o", required=True)
  ap.add_argument("--layers", action="store_true", help="also write every layer's output")
  ap.add_argument("--nl", type=int, default=R.NL, help="run only the first NL layers (debugging)")
  ap.add_argument("--qwen38", help="run Qwen3.8-27B (this FP8 checkpoint folder) instead of Bonsai 2")
  a = ap.parse_args()
  if a.tokens: p, g = read_tokens(a.tokens); ids = p + g[:-1]
  else: ids = [int(t) for t in a.ids.split(",")]
  W = R.Weights(a.qwen38 or a.gguf); t0 = time.time()
  x = W.embed(ids).astype(np.float32)
  if a.layers: x.tofile(a.o + ".model.input_embed")
  print(f"{len(ids)} tokens: {ids}", flush=True)
  for l in range(a.nl):
    t = time.time(); x, _ = R.layer(W, l, x)
    if a.layers: x.astype(np.float32).tofile(f"{a.o}.l_out-{l}")
    print(f"layer {l:2d} {R.LAYER_TYPES[l]:6s} {time.time() - t:5.1f} s  |x| {np.abs(x).max():.3g}", flush=True)
  h = W.final_norm(x); h.astype(np.float32).tofile(a.o + ".result_norm")
  t = time.time(); logits = W.lm_head(h); logits.astype(np.float32).tofile(a.o + ".logits")
  print(f"head {time.time() - t:.1f} s; total {time.time() - t0:.0f} s; argmax {logits.argmax(-1).tolist()}")

if __name__ == "__main__": main()

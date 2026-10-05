"""The NPU's greedy tokens vs a float32 numpy forward of the whole model (qwen3.8-27b's reference layers on Ornith's dequantised
weights, fp32 activations): at every generated position, does fp32 pick the NPU's token, and how close is its runner-up?
Also where another implementation (e.g. one that quantises the activations to Q8_0 too) diverged.
  python3 ref_compare.py prompt.txt npu_out.npz [other_text.txt] [n_positions]"""
import os, sys, time
os.environ.setdefault("QWEN_MODEL", "ornith-9b")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "..", "qwen3.8-27b"))
import numpy as np
import qwen38_ref as R
from qwen38_tokenize import Tok

prompt, npz = sys.argv[1], sys.argv[2]; other = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
K = int(sys.argv[4]) if len(sys.argv) > 4 else 24
tok = Tok(os.environ.get("QWEN_TOK", "/mnt/ssd/ornith-9b-npu")); W = R.Weights(os.environ.get("ORNITH_GGUF", "/mnt/ssd/models/ornith-1.0-9b-Q8_0.gguf"))
ids = tok.encode(open(prompt).read()); out = np.load(npz)["out_ids"].tolist()[:K]; n = len(ids)
seq = ids + out[:-1]                                                     # every generated token's context
t0 = time.perf_counter(); x = W.embed(seq)
for l in range(R.NL): x, _ = R.layer(W, l, x)
lg = W.lm_head(W.final_norm(x[n - 1:]))                                  # [len(out), vocab]: the predictions of out[0], out[1], ...
print(f"fp32 forward over {len(seq)} tokens in {time.perf_counter() - t0:.0f} s")
top2 = np.argsort(-lg, 1)[:, :2]; agree = [int(top2[i, 0]) == out[i] for i in range(len(out))]
marg = lg[np.arange(len(out)), top2[:, 0]] - lg[np.arange(len(out)), top2[:, 1]]
print(f"the NPU's token = fp32's argmax at {sum(agree)} of {len(out)} positions; fp32 top-1 margin median {np.median(marg):.2f}")
for i in range(len(out)):
  if not agree[i]: print(f"   position {i}: NPU {tok.decode([out[i]])!r} (fp32 logit {lg[i, out[i]]:.3f}) vs fp32 {tok.decode([int(top2[i, 0])])!r} ({lg[i, top2[i, 0]]:.3f})")
if other:
  o = open(other).read(); npu_text = ""
  for i, t in enumerate(out):
    nxt = npu_text + tok.decode([t])
    if not o.startswith(nxt):
      cand = tok.encode(o[len(npu_text):len(npu_text) + 24])[:1]
      print(f"   the other implementation diverges at position {i}: it took {tok.decode(cand)!r}, the NPU {tok.decode([t])!r}; "
            f"fp32 logits: NPU's {lg[i, t]:.3f}, the other's {lg[i, cand[0]]:.3f} (fp32 argmax {tok.decode([int(top2[i, 0])])!r}, margin {marg[i]:.3f})")
      break
    npu_text = nxt

import sys, numpy as np
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SHARDS = os.environ.get("QWEN_SHARDS", "/tmp/qwen38")   # a folder with config.json and layers-0 / layers-3 .safetensors
import qwen38_ref as R
t = np.load("" + SHARDS + "/truth.npz"); W = R.Weights(SHARDS); x = t["x"]
for l in (0, 3):
  y, st = R.layer(W, l, x); want = t[f"y{l}"]
  d = y - want; dd = want - x
  print(f"layer {l} ({R.LAYER_TYPES[l]}): max|d| {np.abs(d).max():.3g}  of max|y| {np.abs(want).max():.3g}  rel err of the layer's delta {np.linalg.norm(d)/np.linalg.norm(dd):.3g}  cosine {float((y*want).sum()/np.linalg.norm(y)/np.linalg.norm(want)):.9f}")
# decode consistency of the DeltaNet: the 12 tokens in one call vs 8 + 4 with the carried state
y_all, _ = R.gdn_layer(W, 0, x); y8, (S, cs) = R.gdn_layer(W, 0, x[:8]); y4, _ = R.gdn_layer(W, 0, x[8:], state=S, conv_state=cs)
print("GDN prefix+carry vs one call: max|d|", np.abs(np.concatenate([y8, y4]) - y_all).max())

"""How much the int8 activation costs on the model's own hidden states: q / k / v / o / gate / down of a layer from the prompt's last
row, float vs the per-32 int8 activation (qact_ref), its two-level form (a second int8 of the residual), and llama.cpp's Q8_K
(one scale a 256). python3 act_quant.py [layers]"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import qwen3moe_ref as QR, qwen3moe_qgemv as Q
from tokenizers import Tokenizer
M = QR.Model(); c = M.c
ids = Tokenizer.from_file("/mnt/ssd/models/qwen3-30b-a3b-tokenizer/tokenizer.json").encode(QR.chat_text("What is the capital of Portugal? Answer in one sentence."), add_special_tokens=False).ids
layers = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 6, 24, 47]
def q32(x):
  q, sx, _ = Q.qact_ref(x); return (q.reshape(-1, 32).astype(np.float32) * sx[:, None]).reshape(-1)
def q32x2(x):
  a = q32(x); return a + q32(x - a)
def q8k(x):
  x = x.reshape(-1, 256); s = np.abs(x).max(1, keepdims=True) / 127; s[s == 0] = 1; return (np.round(x / s) * s).reshape(-1)
pos = np.arange(len(ids)); x = M.embed(ids); kv = {}
for l in range(max(layers) + 1):
  if l in layers:
    p = f"blk.{l}."; h = QR.rms(x[-1:], M.mat(p + "attn_norm.weight"), c.EPS)[0]
    out = []
    for nm in ("attn_q", "attn_k", "attn_v"):
      W = M.mat(p + nm + ".weight").astype(np.float64); want = W @ h.astype(np.float64)
      out.append(f"{nm[5:]} " + " ".join(f"{fn.__name__} {np.linalg.norm(W @ fn(h).astype(np.float64) - want) / np.linalg.norm(want):.1e}" for fn in (q32, q32x2, q8k)))
    amax = np.abs(h).reshape(-1, 32).max(1); rmsb = np.sqrt((h.reshape(-1, 32) ** 2).mean(1))
    print(f"layer {l:2d}: h max/rms a 32 (median, max) {np.median(amax / rmsb):.1f} {np.max(amax / rmsb):.1f} | " + " | ".join(out), flush=True)
  x = M.layer(l, x, pos, kv)

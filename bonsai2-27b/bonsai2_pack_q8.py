"""A Bonsai 2 GGUF requantised to Q8_0 or Q4_K (`llama-quantize --allow-requantize --pure`) -> an NPU cache in that format's GEMM
(Q8_0: gemm_gs q8=1, ks 8; Q4_K and Q4_0: q8=5 "q4m", ks 32, a `q4m` marker -- Q4_0 as a = d, b = -8 d, exact), for a
like-for-like comparison with llama.cpp on the same weights. The linears and the head
as gemm_fp16.pack_b_group_q8 streams (int8 codes, a scale per 32 weights, ks 8: ornith-9b's format); the same folding as bonsai2_pack:
the HF row order (W.OUT_ROWS), the down input's signs in the up rows' codes (q[k] x s_INTER[k], exact; the Q8_0 kernel takes d >= 0), the 1 / sqrt(block) in the
scales when the block folds it (fold_norm). Everything else -- L{l}_small.npz (norms, DeltaNet parameters, the linears' meta: the same
shapes), hadamard.npz, outside_small.npz, the drafter's `mtp` -- comes from the model's ternary cache, unchanged. Run it with
QWEN_BONSAI_Q8=1 (qwen38_ref: Q8 with HAD), QWEN_NPU = this cache, BONSAI_GGUF = the PTQ1_0 GGUF (the host's embedding).
  QWEN_MODEL=bonsai2-2b python3 bonsai2_pack_q8.py --gguf 2b-Q8_0.gguf --src /mnt/ssd/bonsai2-2b-npu --out /mnt/ssd/bonsai2-2b-npu-q8"""
import argparse, os, shutil, sys
os.environ.setdefault("QWEN_MODEL", "bonsai2-2b"); os.environ["QWEN_BONSAI_Q8"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "qwen3.8-27b"))
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "tinygrad"))))
import numpy as np
from zy import gemm_fp16 as G                                             # noqa: E402
import qwen38_ref as R                                                    # noqa: E402
from qwen38_pack import LINEARS, fuse_all                                 # noqa: E402
from bonsai2_weights import Weights                                       # noqa: E402

NS = 3
def tensor(W, tname, rows=None):
  """A GGUF tensor -> ("q8", q int8 [N, K], d [N, K/32]) or ("q4", u uint8 [N, K], a, b [N, K/32]: w = a u + b), rows in `rows` order."""
  kind, dims, raw = W.g.raw(tname); K, N = dims[0], int(np.prod(dims[1:]))
  if kind == "Q8_0": q, d = W.g.q8(tname); parts = ["q8", q, d.astype(np.float32)]
  elif kind == "Q4_K": u, a, b = G.q4k_split(np.asarray(raw).reshape(N, K // 256, 144)); parts = ["q4", u, a.astype(np.float32), b.astype(np.float32)]
  elif kind == "Q4_0":   # w = d (q - 8) per 32: exactly q4m's a u + b with a = d, b = -8 d (no Q4_0 kernel needed)
    blk = np.asarray(raw).reshape(N, K // 32, 18); d = blk[:, :, :2].copy().view(np.float16)[..., 0].astype(np.float32); qs = blk[:, :, 2:]
    u = np.concatenate([qs & 15, qs >> 4], axis=2).reshape(N, K)          # byte j: value j (low nibble), value j + 16 (high)
    parts = ["q4", u.astype(np.uint8), d, (-8.0 * d).astype(np.float32)]
  else: raise ValueError((tname, kind))
  if rows is not None: parts = [parts[0]] + [x[rows] for x in parts[1:]]
  return parts

def fold_t(t, fold, sgn=None):
  """The scale fold (1 / sqrt(block) when the block folds it) and, for the up projection, the down input's signs per row: Q8_0 in the
  codes (the Q8_0 kernel widens d x 2^10 as fp16 bits and takes d >= 0, as llama.cpp's Q8_0 always is: a negative scale came out
  as ~1e10; q in [-127, 127], so the negation is exact), Q4_K in a and b (w = a u + b: -w = (-a) u + (-b); q4m multiplies by them)."""
  if t[0] == "q8":
    q, d = t[1], t[2] * fold
    if sgn is not None: q = (q.astype(np.int16) * sgn.astype(np.int16)[:, None]).astype(np.int8)
    return ["q8", q, d]
  u, a, b = t[1], t[2] * fold, t[3] * fold
  if sgn is not None: a, b = a * sgn[:, None], b * sgn[:, None]
  return ["q4", u, a.astype(np.float32), b.astype(np.float32)]

def cat(ts): return [ts[0][0]] + [np.concatenate([t[i] for t in ts]) for i in range(1, len(ts[0]))]
def pack(t, g): return G.pack_b_group_q8(t[1], t[2], g, NS, 8) if t[0] == "q8" else G.pack_b_group_q4m(t[1], t[2], t[3], g, NS, 32)
def stream(t):
  ngroups = -(-t[1].shape[0] // (16 * NS))
  return np.concatenate([pack(t, g) for g in range(ngroups)]), 16 * NS * ngroups

def lin(W, hf, sg, fold):
  t_, rest = W.gguf_name(hf)
  return fold_t(tensor(W, t_, W.OUT_ROWS.get(rest)), fold, sg[R.INTER] if hf.endswith("mlp.up_proj") else None)

def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--gguf", required=True); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
  a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); W = Weights(a.gguf)
  hz = np.load(os.path.join(a.src, "hadamard.npz")); sg = {int(k[1:]): hz[k].astype(np.float32) for k in hz.files if k.startswith("s")}
  fold = np.float32(1.0 / np.sqrt(int(hz["block"]))) if int(hz["fold_norm"]) else np.float32(1.0)
  big = 0
  for l in range(R.NL):
    kind, p = R.LAYER_TYPES[l], f"model.language_model.layers.{l}."
    meta = dict(zip(*(lambda z: (list(z["meta_names"]), [tuple(int(v) for v in m) for m in z["meta"]]))(np.load(os.path.join(a.src, f"L{l}_small.npz")))))
    for name, parts in LINEARS[kind]:
      t = cat([lin(W, p + x, sg, fold) for x in parts]); fmt = t[0]
      st, npad = stream(t); assert meta[name] == (t[1].shape[0], npad, t[1].shape[1]), (l, name, meta[name], t[1].shape, npad)
      f = os.path.join(a.out, f"L{l}_{name}.bin"); st.tofile(f + ".tmp"); os.replace(f + ".tmp", f); big = max(big, st.nbytes)
    shutil.copyfile(os.path.join(a.src, f"L{l}_small.npz"), os.path.join(a.out, f"L{l}_small.npz"))
    print(f"   layer {l:2d} ({kind})", flush=True)
  # the head: parts of whole groups, each no bigger than the biggest layer stream (a weight slot holds it)
  t = fold_t(tensor(W, W.head), fold); N, K = t[1].shape; gs = 16 * NS; ngroups = -(-N // gs)
  one = pack([t[0]] + [x[:gs] for x in t[1:]], 0).nbytes; per = max(8, (big // one) // 8 * 8); parts = []
  for p_, g0 in enumerate(range(0, ngroups, per)):
    g1 = min(ngroups, g0 + per); f = os.path.join(a.out, f"{R.HEAD}_{p_}.bin")
    with open(f + ".tmp", "wb") as fh:
      for c0 in range(g0, g1, 64):
        c1 = min(g1, c0 + 64); r0, r1 = c0 * gs, min(c1 * gs, N)
        tc = [t[0]] + [x[r0:r1] for x in t[1:]]; fh.write(np.concatenate([pack(tc, g) for g in range(c1 - c0)]).tobytes())
    os.replace(f + ".tmp", f); parts.append((g0, g1))
  np.savez(os.path.join(a.out, f"{R.HEAD}.npz"), parts=np.array(parts, np.int64), n=np.array([N, K]))
  print(f"   head: {len(parts)} parts of <= {per} groups", flush=True)
  for f in ("hadamard.npz", "outside_small.npz", "tokenizer.json"):
    if os.path.exists(os.path.join(a.src, f)): shutil.copyfile(os.path.join(a.src, f), os.path.join(a.out, f))
  if os.path.isdir(os.path.join(a.src, "mtp")) and not os.path.exists(os.path.join(a.out, "mtp")): os.symlink(os.path.join(a.src, "mtp"), os.path.join(a.out, "mtp"))
  open(os.path.join(a.out, "q8_from"), "w").write(f"{a.gguf}\nbonsai2_pack_q8.py: {'Q8_0 (q8 1, ks 8)' if fmt == 'q8' else 'Q4_K (q8 5 q4m, ks 32)'} streams of the requantised GGUF; small tensors from {a.src}\n")
  if fmt == "q4": open(os.path.join(a.out, "q4m"), "w").write("gemm_gs(q8=5): Q4_K as pack_b_group_q4m (ks 32); bonsai2_pack_q8.py\n")
  if os.path.exists(os.path.join(a.src, "fused")): fuse_all(a.out, range(R.NL))

if __name__ == "__main__": main()

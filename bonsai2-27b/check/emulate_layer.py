#!/usr/bin/env python3
"""A host emulation of the NPU's bonsai2-27b layers and head from the packed cache, against bonsai2_ref (qwen3.8-27b/qwen38_ref.py
on the GGUF): qwen38_ref's own layer math, with every linear replaced by what the device does with the cache --
  A = fp16(had(x * s) [x 2^-5 if not folded]): the unnormalised 1024-block Walsh-Hadamard of the input times its signs, rounded
      to fp16 as the A producers do (qwen38_kernels.had_ref; the signs of down's input are in the up rows' scales instead);
  C = sum over K-slices, in fp32 and in order, of fp32(A_slice @ w_slice) x the stored scale (k_gemm_gs(tern)'s accumulation),
      w and the scales unpacked from the cache's streams (bonsai2_pack.unpack_stream);
so the pack (codes, scales, permutations, folded signs and normalisation), the transform conventions and the fp16 A rounding
are checked end to end against the GGUF reference; the kernels themselves were gated on the vendor simulator (had_a32, the ternary
GEMM; those harnesses are not part of this repository). Also reported: max |A| per linear (the fp16 range the folded
2^-5 leaves: A is 32x the normalised transform).

    python3 check/emulate_layer.py --cache /mnt/ssd/bonsai2-npu --layers 0,3 [--x out/ref] [--head out/ref.result_norm]
`--x PREFIX`: real inputs -- layer l is fed PREFIX.l_out-<l-1> (PREFIX.model.input_embed for l = 0), bonsai2_ref.py --layers'
dumps of the 21-token P0 sequence; default random rows. `--head FILE`: the final-normed rows -> the head's logits vs the GGUF's."""
import argparse, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
import bonsai2_pack as P                                                 # (sets QWEN_MODEL=bonsai2-27b; the paths)
import qwen38_ref as R
from qwen38_pack import LINEARS
from qwen38_kernels import had_ref

STATS = {}

class CacheWeights:
  """qwen38_ref's Weights interface over the packed cache: `mm` emulates the device's linear, `f32` reads L{l}_small.npz."""
  def __init__(self, cache):
    self.cache = cache; z = np.load(os.path.join(cache, "hadamard.npz"))
    self.signs = {int(k[1:]): z[k].astype(np.float32) for k in z.files if k.startswith("s")}
    self.post = 1.0 if int(z["fold_norm"]) else float(1.0 / np.sqrt(int(z["block"]))); self._w = {}
  def shard(self, l): return self
  @staticmethod
  def _split(name):
    p = "model.language_model.layers."; l, rest = name[len(p):].split(".", 1); return int(l), rest
  def weights(self, l, rest):
    """(w int8 [N, K], scales fp32 [N, K / 128]) of an HF linear from its stream file (a part of gu / kv: its rows)."""
    if (l, rest) not in self._w:
      sm = np.load(os.path.join(self.cache, f"L{l}_small.npz")); meta = {str(k): tuple(int(v) for v in m) for k, m in zip(sm["meta_names"], sm["meta"])}
      for name, parts in LINEARS[R.LAYER_TYPES[l]]:
        if rest not in parts: continue
        N, npad, K = meta[name]; c, s = P.unpack_stream(open(os.path.join(self.cache, f"L{l}_{name}.bin"), "rb").read(), npad, K)
        sizes = {"gu": [R.INTER, R.INTER], "kv": [R.NKV * R.HD, R.NKV * R.HD]}.get(name, [N]); assert sum(sizes) == N
        i = parts.index(rest); r0 = sum(sizes[:i]); r1 = r0 + sizes[i]
        self._w = {(l, rest): ((c[r0:r1].astype(np.int8) - 1).astype(np.int8), s[r0:r1])}
    return self._w[(l, rest)]
  def mm(self, sf, x, name):
    l, rest = self._split(name); w, sc = self.weights(l, rest); K = x.shape[1]; n = x.shape[0]
    s = np.ones(K, np.float32) if rest == "mlp.down_proj" else self.signs[K]          # down: its signs are in the up rows' scales
    v = had_ref("plain", np.asarray(x, np.float32), s, post=self.post)              # the producers' transform (fp32), then fp16
    STATS[rest] = max(STATS.get(rest, 0.0), float(np.abs(v).max()))
    a = v.astype(np.float16).astype(np.float32); acc = None
    for j in range(K // 128):                                                        # k_gemm_gs(tern): per slice, scaled, summed in order
      d = (a[:, 128 * j:128 * (j + 1)].astype(np.float64) @ w[:, 128 * j:128 * (j + 1)].T.astype(np.float64)).astype(np.float32)
      v_ = (d * sc[None, :, j]).astype(np.float32); acc = v_ if acc is None else (v_ + acc).astype(np.float32)
    return acc
  def f32(self, name):
    l, rest = self._split(name); return np.load(os.path.join(self.cache, f"L{l}_small.npz"))[rest.replace(".", "_")]

def head_check(cache, W, rows):
  """The head parts (emulated GEMM, A = fp16(had(rms(x) (1 + w) s))) vs the GGUF reference's logits on final-normed rows."""
  h = np.load(os.path.join(cache, f"{R.HEAD}.npz")); N, K = (int(v) for v in h["n"]); cw = CacheWeights(cache)
  a = had_ref("plain", rows, cw.signs[K], post=cw.post).astype(np.float16).astype(np.float32); out = []
  for i, (g0, g1) in enumerate(h["parts"]):
    c, s = P.unpack_stream(open(os.path.join(cache, f"{R.HEAD}_{i}.bin"), "rb").read(), (int(g1) - int(g0)) * 48, K); w = c.astype(np.int8) - 1; acc = None
    for j in range(K // 128):
      d = (a[:, 128 * j:128 * (j + 1)].astype(np.float64) @ w[:, 128 * j:128 * (j + 1)].T.astype(np.float64)).astype(np.float32)
      v_ = (d * s[None, :, j]).astype(np.float32); acc = v_ if acc is None else (v_ + acc).astype(np.float32)
    out.append(acc); del c, w
  got = np.concatenate(out, 1)[:, :N]; want = W.lm_head(rows)
  print(f"head: {rows.shape[0]} rows: top-1 {int((got.argmax(1) == want.argmax(1)).sum())}/{rows.shape[0]}; max |d| {np.abs(got - want).max():.4f} "
        f"(max |logit| {np.abs(want).max():.2f}); max |A| {np.abs(a).max():.1f}", flush=True)
  return bool((got.argmax(1) == want.argmax(1)).all())

def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True)
  ap.add_argument("--gguf", default=os.environ.get("BONSAI_GGUF", "/mnt/ssd/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf"))
  ap.add_argument("--layers", default="0,3"); ap.add_argument("--x"); ap.add_argument("--n", type=int, default=12); ap.add_argument("--head")
  ap.add_argument("--tol", type=float, default=1e-2, help="max relative error of the layer's delta (fp16 A against the fp32 reference)")
  a = ap.parse_args(); W = P.Weights(a.gguf); CW = CacheWeights(a.cache); ok = True
  for l in (int(v) for v in a.layers.split(",")):
    if a.x: x = np.fromfile(f"{a.x}.l_out-{l - 1}" if l else f"{a.x}.model.input_embed", np.float32).reshape(-1, R.H)
    else: x = np.random.default_rng(l).standard_normal((a.n, R.H)).astype(np.float32)
    t0 = time.perf_counter(); want, _ = R.layer(W, l, x); t1 = time.perf_counter(); STATS.clear(); got, _ = R.layer(CW, l, x); t2 = time.perf_counter()
    d = got - want; dd = want - x; rel = float(np.linalg.norm(d) / np.linalg.norm(dd))
    g64, w64 = got.astype(np.float64), want.astype(np.float64); cos = float((g64 * w64).sum() / np.linalg.norm(g64) / np.linalg.norm(w64)); good = rel < a.tol and bool(np.isfinite(got).all()); ok &= good
    print(f"layer {l:2d} ({R.LAYER_TYPES[l]}, {x.shape[0]} rows{' real' if a.x else ' random'}): rel err of the layer's delta {rel:.2e}, max |d| {np.abs(d).max():.3g} "
          f"(max |y| {np.abs(want).max():.3g}), cosine {cos:.7f} -> {'OK' if good else 'FAIL'}  [ref {t1 - t0:.0f} s, emulation {t2 - t1:.0f} s]", flush=True)
    print("   max |A| (fp16 operand, x32 normalised): " + ", ".join(f"{k} {v:.1f}" for k, v in STATS.items()), flush=True)
  if a.head: ok &= head_check(a.cache, W, np.fromfile(a.head, np.float32).reshape(-1, R.H))   # result_norm: already final-normed
  print("EMULATION:", "PASS" if ok else "FAIL", flush=True); sys.exit(0 if ok else 1)

if __name__ == "__main__": main()

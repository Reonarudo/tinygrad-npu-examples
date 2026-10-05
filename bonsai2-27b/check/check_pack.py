#!/usr/bin/env python3
"""The pack round trip: every stream of a bonsai2_pack.py cache unpacked (bonsai2_pack.unpack_stream) must give back the GGUF's
decoded weights after the documented folding -- codes - 1 == the PTQ1_0 trits (the DeltaNet's V rows in HF order), the stored
scales == d x 2^-5 exactly (x s17408[row] on the up rows), the padding rows code 1 / scale 0 -- and the fused files must be their
parts back to back; L{l}_small.npz must equal bonsai2_weights' tensors, hadamard.npz the GGUF's signs, the head parts the output
weight's rows. No device.

    python3 check/check_pack.py --cache /mnt/ssd/bonsai2-npu [--gguf ...] [--layers 0-3] [--head]
"""
import argparse, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
import bonsai2_pack as P                                                 # (sets QWEN_MODEL=bonsai2-27b)
import qwen38_ref as R
from qwen38_pack import LINEARS, SMALL, FUSED

def check_stream(path, npad, K, t, sc):
  c, s = P.unpack_stream(open(path, "rb").read(), npad, K); N = t.shape[0]
  ok = dict(codes=bool(np.array_equal(c[:N].astype(np.int16) - 1, t.astype(np.int16))), scales=bool(np.array_equal(s[:N], sc)),
            pad=bool((c[N:] == 1).all() and (s[N:] == 0).all()), size=os.path.getsize(path) == (npad // 48) * (K // 128) * (P.NS * P.KS * 16 + P.NS * 64))
  return ok

def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--cache", required=True)
  ap.add_argument("--gguf", default=os.environ.get("BONSAI_GGUF", "/mnt/ssd/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf"))
  ap.add_argument("--layers", default="0-3"); ap.add_argument("--head", action="store_true"); a = ap.parse_args()
  W = P.Weights(a.gguf); sg = P.signs(W); allok = True
  def report(what, ok):
    nonlocal allok; good = all(ok.values()); allok &= good
    print(f"  {what}: " + ", ".join(f"{k} {v}" for k, v in ok.items()) + f" -> {'OK' if good else 'FAIL'}", flush=True)
  z = np.load(os.path.join(a.cache, "hadamard.npz"))
  report("hadamard.npz", {f"s{w}": bool(np.array_equal(z[f"s{w}"], v)) for w, v in sg.items()} | dict(block=int(z["block"]) == 1024, fold=int(z["fold_norm"]) == 1,
         markers=os.path.exists(os.path.join(a.cache, "tern")) and open(os.path.join(a.cache, "scales")).read().split()[0] == "single"))
  lo, hi = (int(v) for v in a.layers.split("-"))
  for l in range(lo, hi + 1):
    kind, p = R.LAYER_TYPES[l], f"model.language_model.layers.{l}."
    sm = np.load(os.path.join(a.cache, f"L{l}_small.npz")); meta = {str(k): tuple(int(v) for v in m) for k, m in zip(sm["meta_names"], sm["meta"])}
    for name, parts in LINEARS[kind]:
      ts = [P.ternary(W, p + q) for q in parts]; t = np.concatenate([x for x, _ in ts])
      d = np.concatenate([y for _, y in ts]) * np.float32(2.0 ** -5)
      if parts[-1] == "mlp.up_proj": d[R.INTER:] *= sg[R.INTER][:, None]
      N, npad, K = meta[name]; assert N == t.shape[0] and K == t.shape[1]
      report(f"L{l} {name} ({' | '.join(parts)}) [{N} x {K}]", check_stream(os.path.join(a.cache, f"L{l}_{name}.bin"), npad, K, t, d.astype(np.float32)))
    report(f"L{l} small", {n: bool(np.array_equal(sm[n.replace(".", "_")], W.f32(p + n))) for n in SMALL[kind]})
    fz = os.path.join(a.cache, f"L{l}_fused.npz")
    if os.path.exists(fz):
      f = np.load(fz); name, subs = str(f["name"]), [str(s) for s in f["subs"]]
      cat = b"".join(open(os.path.join(a.cache, f"L{l}_{s}.bin"), "rb").read() for s in subs)
      report(f"L{l} fused {name}", dict(bytes=open(os.path.join(a.cache, f"L{l}_{name}.bin"), "rb").read() == cat, subs=tuple(subs) == FUSED[kind][1]))
  if a.head:
    h = np.load(os.path.join(a.cache, f"{R.HEAD}.npz")); N, K = (int(v) for v in h["n"]); ok = {}
    for i, (g0, g1) in enumerate(h["parts"]):
      r0, r1 = int(g0) * 48, min(int(g1) * 48, N); t, d = W.g.ptq("output.weight", slice(r0, r1))
      c, s = P.unpack_stream(open(os.path.join(a.cache, f"{R.HEAD}_{i}.bin"), "rb").read(), (int(g1) - int(g0)) * 48, K)
      n = r1 - r0; ok[f"part {i}"] = bool(np.array_equal(c[:n].astype(np.int16) - 1, t) and np.array_equal(s[:n], (d.astype(np.float32) * np.float32(2.0 ** -5))) and (c[n:] == 1).all() and (s[n:] == 0).all())
    o = np.load(os.path.join(a.cache, "outside_small.npz"))
    ok["final norm"] = bool(np.array_equal(o["norm"], W.f32("model.language_model.norm.weight"))); ok["n"] = tuple(o["lm_head_n"]) == (N, -(-N // 48) * 48, K)
    report("head", ok)
  print("PACK ROUND TRIP:", "PASS" if allok else "FAIL", flush=True); sys.exit(0 if allok else 1)

if __name__ == "__main__": main()

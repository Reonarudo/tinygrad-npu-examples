#!/usr/bin/env python3
"""A trained MTP drafter (a training run's output: fc [H, 2H] and the merged linears
q k v o g u d, fp32) into an MTP cache directory holding the TERNARY stream the trunk uses (gemm_fp16.pack_b_group_tern, ks 32,
ns 3, fp16 scale tables): every matrix ternarised per (row, 128-wide K block) by TWN -- threshold 0.7 mean|w|, scale = mean |w| of
the kept weights, rounded to the fp16 table's exact value -- the formula the exact replay scores (replay.py fake_quant("tern")),
so the board's drafter is the replay's bit for bit. Norms and meta copied from the base cache's L64_small.npz; markers tern /
scales / tscale as bonsai2_pack.markers writes them (no hadamard: the MTP layer is unrotated). QWEN_MTP_CACHE=<out>.

    python3 pack_mtp_tern.py ~/align/pilot2/qat_tern.npz ~/align/weights/L64_small.npz ~/align/pilot2/mtp_tern
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen38_pack as P   # noqa: E402
G, NS, KS = P.G, P.NS, P.KS

def ternarise(w):
  """fp32 [N, K] -> (t int8 {-1, 0, 1} [N, K], scale fp32 [N, K / 128]): TWN per 128-block, scales exact in the fp16 table (s x 2^15)."""
  N, K = w.shape; assert K % 128 == 0, w.shape; b = w.reshape(N, K // 128, 128); ab = np.abs(b)
  keep = ab > 0.7 * ab.mean(-1, keepdims=True)
  s = (ab * keep).sum(-1) / np.maximum(keep.sum(-1), 1)
  s = (s.astype(np.float32) * np.float32(2.0 ** 15)).astype(np.float16).astype(np.float32) * np.float32(2.0 ** -15)
  return (np.sign(b) * keep).reshape(N, K).astype(np.int8), s.astype(np.float32)

def stream(t, s):
  ngroups = -(-t.shape[0] // (16 * NS))
  return np.concatenate([G.pack_b_group_tern(t, s, g, NS, KS, tscale="f16") for g in range(ngroups)]), 16 * NS * ngroups

def main():
  src, small_src, out = (os.path.expanduser(v) for v in sys.argv[1:4]); os.makedirs(out, exist_ok=True)
  assert not [f for f in os.listdir(out) if f.endswith(".bin")] or os.path.exists(os.path.join(out, "tern")), f"{out} holds another format"
  z = np.load(src); W = {k: z[k].astype(np.float32) for k in z.files}; small = dict(np.load(small_src))
  meta = {str(k): tuple(int(v) for v in m) for k, m in zip(small["meta_names"], small["meta"])}
  parts = dict(q=("q",), kv=("k", "v"), o=("o",), gu=("g", "u"), dn=("d",))
  for name, ps in parts.items():
    t0 = time.perf_counter(); w = np.concatenate([W[p] for p in ps]); t, s = ternarise(w)
    st, npad = stream(t, s); assert (t.shape[0], npad, t.shape[1]) == meta[name], (name, t.shape, npad, meta[name])
    f = os.path.join(out, f"L{P.MTP_L}_{name}.bin"); st.tofile(f + ".tmp"); os.replace(f + ".tmp", f)
    print(f"  {name}: {w.shape} -> {os.path.getsize(f) / 1e6:.1f} MB, zeros {float((t == 0).mean()):.3f} [{time.perf_counter() - t0:.0f} s]", flush=True)
  w = W["fc"]; t, s = ternarise(w); st, npad = stream(t, s); f = os.path.join(out, f"L{P.MTP_L}_fc.bin"); st.tofile(f + ".tmp"); os.replace(f + ".tmp", f)
  np.savez(os.path.join(out, f"L{P.MTP_L}_fc.npz"), meta=np.array([w.shape[0], npad, w.shape[1]], np.int64)); print(f"  fc: {w.shape} -> {os.path.getsize(f) / 1e6:.1f} MB")
  np.savez(os.path.join(out, f"L{P.MTP_L}_small.npz"), **small)
  open(os.path.join(out, "tern"), "w").write("gemm_fp16.pack_b_group_tern (ks 32, ns 3), gemm_gs(tern=True); pack_mtp_tern.py (unrotated MTP layer)\n")
  open(os.path.join(out, "scales"), "w").write("single\nthe ternary stream's single-layout scale table (gemm_gs scales='single')\n")
  open(os.path.join(out, "tscale"), "w").write("f16\nthe ternary scale tables as fp16 s x 2^15 (gemm_fp16.pack_b_group_tern(tscale='f16'), gemm_gs(tscale='f16'))\n")
  open(os.path.join(out, "trained"), "w").write(f"pack_mtp_tern.py {src}\n"); print(f"wrote {out}")

if __name__ == "__main__": main()

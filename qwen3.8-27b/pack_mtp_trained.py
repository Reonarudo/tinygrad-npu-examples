#!/usr/bin/env python3
"""A trained MTP drafter (a training run's output: fc [H, 2H] and the merged linears q k v o g u d,
fp32) into an MTP cache directory in qwen38_pack.pack_mtp's layout -- every matrix E4M3 with 128 x 128 block scales max|w| / 448,
the formula the exact replay scores (replay.py fc_as_packed), so the board's drafter is the replay's bit for bit; the norms and the
meta copied from the base cache's L64_small.npz. Point the runtime at it with QWEN_MTP_CACHE=<out>.

    python3 pack_mtp_trained.py ~/align/pilot2/eal_r64_e1.npz ~/align/weights/L64_small.npz ~/align/pilot2/mtp_eal
"""
import os, shutil, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen38_pack as P   # noqa: E402

def quant(w):
  """fp32 [N, K] -> (E4M3 codes, scale_inv [N/128, K/128]) with max|w| / 448 a block (pack_mtp's fc rule, the replay's fc_as_packed)."""
  N, K = w.shape; assert N % 128 == 0 and K % 128 == 0, w.shape; nb = N // 128
  sinv = (np.maximum(np.abs(w.reshape(nb, 128, K // 128, 128)).max((1, 3)), 1e-12) / 448.0).astype(np.float32)
  return P.e4m3_encode(w / np.repeat(np.repeat(sinv, 128, 0), 128, 1)), sinv

def main():
  src, small_src, out = (os.path.expanduser(v) for v in sys.argv[1:4]); os.makedirs(out, exist_ok=True); P.SCALES = "single"
  z = np.load(src); W = {k: z[k].astype(np.float32) for k in z.files}; small = dict(np.load(small_src))
  meta = {str(k): tuple(int(v) for v in m) for k, m in zip(small["meta_names"], small["meta"])}
  parts = dict(q=("q",), kv=("k", "v"), o=("o",), gu=("g", "u"), dn=("d",))
  for name, ps in parts.items():
    t0 = time.perf_counter(); w = np.concatenate([W[p] for p in ps]); codes, sinv = quant(w)
    st, npad = P.stream(codes, sinv); assert (codes.shape[0], npad, codes.shape[1]) == meta[name], (name, codes.shape, npad, meta[name])
    f = os.path.join(out, f"L{P.MTP_L}_{name}.bin"); st.tofile(f + ".tmp"); os.replace(f + ".tmp", f)
    print(f"  {name}: {w.shape} -> {os.path.getsize(f) / 1e6:.1f} MB [{time.perf_counter() - t0:.0f} s]", flush=True)
  w = W["fc"]; codes, sinv = quant(w); st, npad = P.stream(codes, sinv); f = os.path.join(out, f"L{P.MTP_L}_fc.bin"); st.tofile(f + ".tmp"); os.replace(f + ".tmp", f)
  np.savez(os.path.join(out, f"L{P.MTP_L}_fc.npz"), meta=np.array([w.shape[0], npad, w.shape[1]], np.int64)); print(f"  fc: {w.shape} -> {os.path.getsize(f) / 1e6:.1f} MB")
  np.savez(os.path.join(out, f"L{P.MTP_L}_small.npz"), **small)
  open(os.path.join(out, "scales"), "w").write("single\nqwen38_pack.py --scales single: gemm_fp16.pack_b_group_bscale(scales='single'), gemm_gs(scales='single')\n")
  open(os.path.join(out, "trained"), "w").write(f"pack_mtp_trained.py {src}\n"); print(f"wrote {out}")

if __name__ == "__main__": main()

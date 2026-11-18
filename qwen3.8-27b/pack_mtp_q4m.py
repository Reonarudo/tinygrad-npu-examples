#!/usr/bin/env python3
"""A trained MTP drafter (pilot.py --quant int4 --out: fc [H, 2H] and the merged linears q k v o g u d, fp32) into an MTP cache
holding the q4m stream (gemm_fp16.pack_b_group_q4m, the Q4_K kernel, gemm_gs(q8=5), ks 32, ns 3): every matrix int4 per (row,
128-wide K block) by symmetric absmax -- q = round(w / s) in -7..7, s = max|w| / 7 rounded to fp16 -- stored as u = q + 8 with
a = s, b = -8 s per 32 (the Q4_0 identity bonsai2_pack_q8 uses), the formula the exact replay scores (replay.py fake_quant("int4")).
Norms and meta copied from the base cache's L64_small.npz; marker `q4m`, no `scales` marker (q4m's table is its own, "dup" to
gemm_gs). The runtime (qwen38_npu.Layers(fp8=True)) reads the marker. QWEN_MTP_CACHE=<out>.

    python3 pack_mtp_q4m.py ~/align/pilot2/qat_int4.npz ~/align/weights/L64_small.npz ~/align/pilot2/mtp_q4m
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen38_pack as P   # noqa: E402
G, NS, KS = P.G, P.NS, P.KS

def int4(w):
  """fp32 [N, K] -> (u uint8 [N, K] = q + 8, a fp16 [N, K / 32], b = -8 a): symmetric absmax per 128 block, s exact in fp16."""
  N, K = w.shape; assert K % 128 == 0, w.shape; b = w.reshape(N, K // 128, 128)
  s = (np.abs(b).max(-1) / 7.0 + 1e-12).astype(np.float16).astype(np.float32)
  q = np.clip(np.round(b / s[:, :, None]), -7, 7).astype(np.int8).reshape(N, K)
  a = np.repeat(s, 4, 1).astype(np.float16)
  return (q.astype(np.int16) + 8).astype(np.uint8), a, (-8.0 * a).astype(np.float16)

def stream(u, a, b):
  ngroups = -(-u.shape[0] // (16 * NS))
  return np.concatenate([G.pack_b_group_q4m(u, a, b, g, NS, KS) for g in range(ngroups)]), 16 * NS * ngroups

def main():
  src, small_src, out = (os.path.expanduser(v) for v in sys.argv[1:4]); os.makedirs(out, exist_ok=True)
  assert not [f for f in os.listdir(out) if f.endswith(".bin")] or os.path.exists(os.path.join(out, "q4m")), f"{out} holds another format"
  z = np.load(src); W = {k: z[k].astype(np.float32) for k in z.files}; small = dict(np.load(small_src))
  meta = {str(k): tuple(int(v) for v in m) for k, m in zip(small["meta_names"], small["meta"])}
  parts = dict(q=("q",), kv=("k", "v"), o=("o",), gu=("g", "u"), dn=("d",))
  for name, ps in parts.items():
    t0 = time.perf_counter(); w = np.concatenate([W[p] for p in ps]); u, a, b = int4(w)
    st, npad = stream(u, a, b); assert (u.shape[0], npad, u.shape[1]) == meta[name], (name, u.shape, npad, meta[name])
    f = os.path.join(out, f"L{P.MTP_L}_{name}.bin"); st.tofile(f + ".tmp"); os.replace(f + ".tmp", f)
    print(f"  {name}: {w.shape} -> {os.path.getsize(f) / 1e6:.1f} MB [{time.perf_counter() - t0:.0f} s]", flush=True)
  w = W["fc"]; u, a, b = int4(w); st, npad = stream(u, a, b); f = os.path.join(out, f"L{P.MTP_L}_fc.bin"); st.tofile(f + ".tmp"); os.replace(f + ".tmp", f)
  np.savez(os.path.join(out, f"L{P.MTP_L}_fc.npz"), meta=np.array([w.shape[0], npad, w.shape[1]], np.int64)); print(f"  fc: {w.shape} -> {os.path.getsize(f) / 1e6:.1f} MB")
  np.savez(os.path.join(out, f"L{P.MTP_L}_small.npz"), **small)
  open(os.path.join(out, "q4m"), "w").write("gemm_fp16.pack_b_group_q4m (ks 32, ns 3), gemm_gs(q8=5); pack_mtp_q4m.py (int4 symmetric, u = q + 8, a = s, b = -8 s)\n")
  open(os.path.join(out, "trained"), "w").write(f"pack_mtp_q4m.py {src}\n"); print(f"wrote {out}")

if __name__ == "__main__": main()

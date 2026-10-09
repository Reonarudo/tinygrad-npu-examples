#!/usr/bin/env python3
"""Gemma 4's GGUF (Q8_0) -> the per-layer NPU cache gemma4_generate.py maps zero-copy, in the Qwen family's layout:

  L{l}_{linear}.bin   the gemm_gs(q8=1) streams (int8 codes + a scale per 32 weights, gemm_fp16.pack_b_group_q8, 3 strips a group):
                      qkv (the layer's own q | k|v, k|v from group goff_qkv[1]) or q (a KV-shared layer), o, gu (gate | up), dn,
                      and on the E-series pg (the PLE gate inp_gate) and pp (its projection)
  L{l}_small.npz      the norms (as gemma4_npu.layer_small gives them), meta_names / meta (npad, K) per linear, goff_<linear>, vsame
  head_q8.bin / .npz  the tied head (token_embd) as one stream over the whole vocabulary (n, npad, K)
  ple_proj_q8.bin     (E-series) per_layer_model_proj, BF16 in the GGUF, quantised to Q8_0 here (d = amax / 127 per 32, as ggml)
  outside_small.npz   output_norm, per_layer_proj_norm, ple_meta (n, npad, K)
  gemma4.json         the GGUF it came from and the shapes (written last: its presence marks a complete cache)
The embedding rows and the PLE table rows stay in the GGUF (looked up on the host per token).

    python3 gemma4_pack.py [--gguf /mnt/ssd/models/gemma-4/gemma-4-E2B-it-Q8_0.gguf] [--out /mnt/ssd/gemma4-e2b-npu]
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gemma4_npu as GN                                                  # noqa: E402
import gemma4_ref as GR                                                  # noqa: E402

def save(path, arr):
  arr.tofile(path + ".tmp"); os.replace(path + ".tmp", path)

def pack_layer(M, l, out):
  meta, goff = {}, {}
  for k, subs in GN.layer_linears(M, l).items():
    st, npad, K_, g = GN.pack_linear(M, l, subs); save(os.path.join(out, f"L{l}_{k}.bin"), st); meta[k] = (npad, K_); goff[k] = g
  sm = GN.layer_small(M, l); sm["meta_names"] = np.array(list(meta)); sm["meta"] = np.array([meta[k] for k in meta], np.int64)
  for k, g in goff.items(): sm[f"goff_{k}"] = np.array(g, np.int64)
  sm["vsame"] = np.array(int(not M.has(f"blk.{l}.attn_v.weight")))
  np.savez(os.path.join(out, f"L{l}_small.npz"), **sm)

def pack_stream(path, q, d, chunk=64):
  """A Q8_0 matrix as one stream, written group-chunk by group-chunk (the head: 5462 groups) -> npad."""
  N = q.shape[0]; ng = -(-N // 48)
  with open(path + ".tmp", "wb") as fh:
    for g0 in range(0, ng, chunk):
      g1 = min(ng, g0 + chunk); r0, r1 = g0 * 48, min(g1 * 48, N)
      fh.write(np.concatenate([GN.G.pack_b_group_q8(q[r0:r1], d[r0:r1], g, GN.NS, GN.KS) for g in range(g1 - g0)]).tobytes())
  os.replace(path + ".tmp", path); return 48 * ng

def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--gguf", default=GR.DEFAULT); ap.add_argument("--out", default="/mnt/ssd/gemma4-e2b-npu")
  a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.perf_counter()
  M = GR.Model(a.gguf, cache_layers=False); c = M.c
  for l in range(c.NL):
    if os.path.exists(os.path.join(a.out, f"L{l}_small.npz")): continue
    t1 = time.perf_counter(); pack_layer(M, l, a.out); print(f"   layer {l:2d} packed in {time.perf_counter() - t1:.1f} s", flush=True)
  q, d = M.g.q8("token_embd.weight"); t1 = time.perf_counter()
  npad = pack_stream(os.path.join(a.out, "head_q8.bin"), q, d); N, K = q.shape
  np.savez(os.path.join(a.out, "head_q8.npz"), meta=np.array([N, npad, K], np.int64)); print(f"   head packed in {time.perf_counter() - t1:.0f} s", flush=True)
  out = {"output_norm": M.t("output_norm.weight").astype(np.float32)}
  if c.PLE:
    w = M.g.f32("per_layer_model_proj.weight"); qp, dp = GN.q8_quantise(w)
    pn = pack_stream(os.path.join(a.out, "ple_proj_q8.bin"), qp, dp)
    out.update(per_layer_proj_norm=M.t("per_layer_proj_norm.weight").astype(np.float32), ple_meta=np.array([w.shape[0], pn, w.shape[1]], np.int64))
  np.savez(os.path.join(a.out, "outside_small.npz"), **out)
  json.dump(dict(gguf=os.path.abspath(a.gguf), NL=c.NL, H=c.H, PLE=c.PLE, vocab=int(N), packer="gemma4_pack.py q8=1 ks 8 ns 3"),
            open(os.path.join(a.out, "gemma4.json"), "w"), indent=1)
  tot = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
  print(f"   cache {a.out}: {tot / 1e9:.2f} GB in {time.perf_counter() - t0:.0f} s", flush=True)

if __name__ == "__main__": main()

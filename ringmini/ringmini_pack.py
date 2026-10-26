#!/usr/bin/env python3
"""Ring-mini-2.0's GGUF (Q4_K_M) -> the per-layer NPU cache `ringmini_npu.py` streams: every linear a decode-only qdot stream in its
checkpoint's format (nothing requantised), read by `qwen3moe_qgemv.qgemvk_src`'s kinds:

  Q5_K  attn_qkv                                     -> q5n (code 7, 5.75 bits)
  Q4_K  attn_output, the experts' and the shared expert's gate / up, the dense layer's gate / up -> q4m (code 5: the gemm_gs stream the kind "q4" reads)
  Q6_K  the experts' / shared / dense down            -> q6n (code 6, 6.6 bits)
  Q6_K  output head                                  -> q8f qb16 parts (code 4: kind "q6"), as qwen3moe_pack
  L{l}_small.npz: the norms (w_attn, w_ffn, qn, kn), the router with w_ffn folded in, the selection bias; `meta`: per stream (code, N,
  npad, K, S) with S the per-expert stream bytes of a stacked tensor.

    python3 ringmini_pack.py [--gguf ...] [--out /mnt/ssd/ringmini-npu] [--layers 0-19] [--jobs 2] [--head]
"""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "tinygrad")))); sys.path[:0] = [HERE, os.path.join(HERE, "..", "qwen3-30b-a3b"), os.path.join(HERE, "..", "qwen3.8-27b"), os.path.join(HERE, "..", "ornith-9b")]
from zy import gemm_fp16 as G                                             # noqa: E402
import qwen3moe_qgemv as Q                                                # noqa: E402
import ringmini_ref as RR                                                 # noqa: E402

NS, KS = 3, 32
OUT = os.environ.get("RING_NPU", "/mnt/ssd/ringmini-npu")
HEAD_PARTS = 4


def write(path, arr):
  tmp = path + ".tmp"
  with open(tmp, "wb") as f: f.write(np.ascontiguousarray(arr).tobytes())
  os.replace(tmp, path)


def stream(kind, blk):
  """K-quant super-blocks [N, K / 256, bytes] -> (the stream of all groups, its code, npad)."""
  N = blk.shape[0]; ng = -(-N // 48)
  if kind == "Q4_K": u, a, b = G.q4k_split(blk); return np.concatenate([G.pack_b_group_q4m(u, a, b, g, NS, KS) for g in range(ng)]), 5, 48 * ng
  if kind == "Q5_K": u, sc, m, d, dmin = Q.q5k_split_raw(blk); return np.concatenate([Q.pack_b_group_q5n(u, sc, m, d, dmin, g) for g in range(ng)]), 7, 48 * ng
  if kind == "Q6_K": q, sc, d = Q.q6k_split_raw(blk); return np.concatenate([Q.pack_b_group_q6n(q, sc, d, g) for g in range(ng)]), 6, 48 * ng
  raise ValueError(kind)


def pack_layer(M, l, out):
  c, p = M.c, f"blk.{l}."; meta = {}
  def lin(name, tensor):
    kind, b = M.blocks(p + tensor); st, code, npad = stream(kind, b); write(os.path.join(out, f"L{l}_{name}.bin"), st); meta[name] = (code, b.shape[0], npad, b.shape[1] * 256, 0)
  def exps(name, tensor):
    kind, b = M.blocks(p + tensor); sts = []
    for e in range(b.shape[0]): st, code, npad = stream(kind, b[e]); sts.append(st)
    S = sts[0].size; assert all(s.size == S for s in sts); write(os.path.join(out, f"L{l}_{name}.bin"), np.concatenate(sts)); meta[name] = (code, b.shape[1], npad, b.shape[2] * 256, S)
  lin("qkv", "attn_qkv.weight"); lin("o", "attn_output.weight")
  if l < c.NDENSE: lin("dgate", "ffn_gate.weight"); lin("dup", "ffn_up.weight"); lin("ddown", "ffn_down.weight")
  else:
    exps("gate", "ffn_gate_exps.weight"); exps("up", "ffn_up_exps.weight"); exps("down", "ffn_down_exps.weight")
    lin("sgate", "ffn_gate_shexp.weight"); lin("sup", "ffn_up_shexp.weight"); lin("sdn", "ffn_down_shexp.weight")
  w = lambda n: M.g.f32(p + n).astype(np.float32)
  small = dict(w_attn=w("attn_norm.weight"), w_ffn=w("ffn_norm.weight"), qn=w("attn_q_norm.weight"), kn=w("attn_k_norm.weight"))
  if l >= c.NDENSE: small["router"] = np.ascontiguousarray(w("ffn_gate_inp.weight") * small["w_ffn"][None]); small["bias"] = w("exp_probs_b.bias")
  names = sorted(meta)
  np.savez(os.path.join(out, f"L{l}_small.npz"), meta_names=np.array(names), meta=np.array([list(meta[k]) for k in names], np.int64), **small)


def pack_head(M, out):
  kd, b = M.blocks("output.weight"); assert kd == "Q6_K", kd; N = b.shape[0]; ng = -(-N // 48); per = -(-ng // HEAD_PARTS)
  q, d = G.q6k_split(b); parts = []
  for j in range(HEAD_PARTS):
    g0, g1 = j * per, min(ng, (j + 1) * per); r0 = g0 * 48
    write(os.path.join(out, f"head{j}.bin"), np.concatenate([G.pack_b_group_q8f(q, d, gg, NS, KS, qb=16) for gg in range(g0, g1)])); parts.append((4, min(N, g1 * 48) - r0, (g1 - g0) * 48, q.shape[1], r0))
  np.savez(os.path.join(out, "head.npz"), parts=np.array(parts, np.int64), w_out=M.g.f32("output_norm.weight").astype(np.float32))


def _one(args):
  gguf, l, out = args; M = RR.Model(gguf); t0 = time.perf_counter(); pack_layer(M, l, out); return l, time.perf_counter() - t0


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--gguf", default=RR.DEFAULT); ap.add_argument("--out", default=OUT); ap.add_argument("--layers", default="0-19")
  ap.add_argument("--jobs", type=int, default=2); ap.add_argument("--head", action="store_true"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
  if a.head: M = RR.Model(a.gguf); t0 = time.perf_counter(); pack_head(M, a.out); print(f"head: {time.perf_counter() - t0:.0f} s", flush=True); return
  lo, hi = (int(v) for v in a.layers.split("-")) if "-" in a.layers else (int(a.layers), int(a.layers))
  from multiprocessing import Pool
  with Pool(a.jobs) as pool:
    for l, dt in pool.imap_unordered(_one, [(a.gguf, l, a.out) for l in range(lo, hi + 1)]): print(f"layer {l}: {dt:.0f} s", flush=True)


if __name__ == "__main__": main()

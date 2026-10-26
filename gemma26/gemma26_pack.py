#!/usr/bin/env python3
"""Gemma 4 26B-A4B's GGUF (unsloth UD-Q4_K_M) -> the per-layer NPU cache gemma26_npu.py streams, nothing requantised:
  Q8_0  attention q / k / (v) / o, the dense gate / up / down, the tied head  -> the "q8" qdot stream (code 2; K padded to 128s: the dense
        down's 2112 -> 2176, the padding columns zero)
  Q4_K  the fused gate|up experts [E][1408][2816]                             -> q4m (code 5, the kind "q4"), expert e at e x S
  Q5_1  the down experts [E][2816][704]                                       -> q51 (code 9, 6 bits; K 704 -> 768); the one Q8_0 layer -> q8
  L{l}_small.npz: the norms, layer_output_scale, the router (ffn_gate_inp.weight x scale / sqrt(H) folded per input), the experts' scale
    python3 gemma26_pack.py [--gguf ...] [--out ~/npu-packs/gemma26-npu] [--layers 0-29] [--jobs 2] [--head]
"""
import argparse, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "tinygrad")))); sys.path[:0] = [HERE, os.path.join(HERE, "..", "qwen3-30b-a3b"), os.path.join(HERE, "..", "qwen3.8-27b"), os.path.join(HERE, "..", "ornith-9b"), os.path.join(HERE, "..", "gemma4")]
from zy import gemm_fp16 as G                                             # noqa: E402
import qwen3moe_qgemv as Q                                                # noqa: E402
import gemma26_ref as GR                                                  # noqa: E402

NS, KS = 3, 32
OUT = os.environ.get("G26_NPU", os.path.expanduser("~/npu-packs/gemma26-npu"))
HEAD_PARTS = 4
KP = 128


def write(path, arr):
  tmp = path + ".tmp"
  with open(tmp, "wb") as f: f.write(np.ascontiguousarray(arr).tobytes())
  os.replace(tmp, path)


def padk(a, k2, axis=1):
  if a.shape[axis] == k2: return a
  pad = [(0, 0)] * a.ndim; pad[axis] = (0, k2 - a.shape[axis]); return np.pad(a, pad)


def stream_q8(q, d):
  K2 = -(-q.shape[1] // KP) * KP; q, d = padk(q, K2), padk(d, K2 // 32); ng = -(-q.shape[0] // 48)
  return np.concatenate([G.pack_b_group_q8f(q, d, g, NS, KS, qb=32) for g in range(ng)]), 2, 48 * ng, K2


def stream_q8_blocks(b):
  """Q8_0 blocks [N, K / 32, 34] -> the q8 stream (an expert's)."""
  q = b[:, :, 2:].reshape(b.shape[0], -1).view(np.int8); d = b[:, :, :2].copy().view(np.float16)[..., 0].astype(np.float32); return stream_q8(q, d)


def stream_q4m(b):
  """Q4_K blocks [N, K / 256, 144] -> the q4m stream (code 5)."""
  u, a, bb = G.q4k_split(b); ng = -(-u.shape[0] // 48); return np.concatenate([G.pack_b_group_q4m(u, a, bb, g, NS, KS) for g in range(ng)]), 5, 48 * ng, u.shape[1]


def stream_q51(b):
  """Q5_1 blocks [N, K / 32, 24] -> the q51 stream (code 9), K padded to 128s (zero weights: u 0, d 0, m 0)."""
  u, d, m = Q.q51_split_raw(b); K2 = -(-u.shape[1] // KP) * KP; u, d, m = padk(u, K2), padk(d, K2 // 32), padk(m, K2 // 32); ng = -(-u.shape[0] // 48)
  return np.concatenate([Q.pack_b_group_q51(u, d, m, g) for g in range(ng)]), 9, 48 * ng, K2


def pack_layer(M, l, out):
  c, p, g = M.c, f"blk.{l}.", M.g; meta = {}
  def lin(name, t):
    q, d = g.q8(p + t + ".weight"); st, code, npad, K2 = stream_q8(q, d); write(os.path.join(out, f"L{l}_{name}.bin"), st); meta[name] = (code, q.shape[0], npad, K2, 0)
  lin("q", "attn_q"); lin("k", "attn_k"); lin("dgate", "ffn_gate"); lin("dup", "ffn_up"); lin("ddown", "ffn_down")
  qo, do = g.q8(p + "attn_output.weight")
  if qo.shape[1] <= 4096: st, code, npad, K2 = stream_q8(qo, do); write(os.path.join(out, f"L{l}_o.bin"), st); meta["o"] = (code, qo.shape[0], npad, K2, 0)
  else:                                                                   # the global layers' K 8192: two halves (the qdot package holds K <= 4096)
    h = qo.shape[1] // 2
    for j in range(2): st, code, npad, K2 = stream_q8(qo[:, j * h:(j + 1) * h], do[:, j * h // 32:(j + 1) * h // 32]); write(os.path.join(out, f"L{l}_o{j}.bin"), st); meta[f"o{j}"] = (code, qo.shape[0], npad, K2, 0)
  if M.has(p + "attn_v.weight"): lin("v", "attn_v")
  for name, t, fn in (("gu", "ffn_gate_up_exps", None), ("dn", "ffn_down_exps", None)):
    kind, b = M.blocks(p + t + ".weight"); fn = {"Q4_K": stream_q4m, "Q5_1": stream_q51, "Q8_0": stream_q8_blocks}[kind]; sts = []
    for e in range(b.shape[0]): st, code, npad, K2 = fn(b[e]); sts.append(st)
    S = sts[0].size; assert all(s.size == S for s in sts); write(os.path.join(out, f"L{l}_{name}.bin"), np.concatenate(sts)); meta[name] = (code, b.shape[1], npad, K2, S)
  w = lambda n: M.t(p + n).astype(np.float32)
  small = dict(w_attn=w("attn_norm.weight"), qnw=w("attn_q_norm.weight"), knw=w("attn_k_norm.weight"), w_post_attn=w("post_attention_norm.weight"), w_ffn=w("ffn_norm.weight"),
               w_post_ffw=w("post_ffw_norm.weight"), w_post1=w("post_ffw_norm_1.weight"), w_post2=w("post_ffw_norm_2.weight"), w_pre2=w("pre_ffw_norm_2.weight"),
               lscale=np.array([M.t(p + "layer_output_scale.weight")[0] if M.has(p + "layer_output_scale.weight") else 1.0], np.float32),
               router=np.ascontiguousarray(w("ffn_gate_inp.weight") * (w("ffn_gate_inp.scale") * np.float32(1.0 / math.sqrt(c.H)))[None]), escale=w("ffn_down_exps.scale"))
  names = sorted(meta); np.savez(os.path.join(out, f"L{l}_small.npz"), meta_names=np.array(names), meta=np.array([list(meta[k]) for k in names], np.int64), **small)


def pack_head(M, out):
  q, d = M.g.q8("token_embd.weight"); K2 = -(-q.shape[1] // KP) * KP; q, d = padk(q, K2), padk(d, K2 // 32); N = q.shape[0]; ng = -(-N // 48); per = -(-ng // HEAD_PARTS); parts = []
  for j in range(HEAD_PARTS):
    g0, g1 = j * per, min(ng, (j + 1) * per); r0 = g0 * 48
    write(os.path.join(out, f"head{j}.bin"), np.concatenate([G.pack_b_group_q8f(q, d, gg, NS, KS, qb=32) for gg in range(g0, g1)])); parts.append((2, min(N, g1 * 48) - r0, (g1 - g0) * 48, K2, r0))
  np.savez(os.path.join(out, "head.npz"), parts=np.array(parts, np.int64), w_out=M.t("output_norm.weight").astype(np.float32))


def _one(args):
  gguf, l, out = args; M = GR.Model(gguf, cache_layers=False); t0 = time.perf_counter(); pack_layer(M, l, out); return l, time.perf_counter() - t0


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--gguf", default=GR.DEFAULT); ap.add_argument("--out", default=OUT); ap.add_argument("--layers", default="0-29")
  ap.add_argument("--jobs", type=int, default=2); ap.add_argument("--head", action="store_true"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
  if a.head: M = GR.Model(a.gguf); t0 = time.perf_counter(); pack_head(M, a.out); print(f"head: {time.perf_counter() - t0:.0f} s", flush=True); return
  lo, hi = (int(v) for v in a.layers.split("-")) if "-" in a.layers else (int(a.layers), int(a.layers))
  from multiprocessing import Pool
  with Pool(a.jobs) as pool:
    for l, dt in pool.imap_unordered(_one, [(a.gguf, l, a.out) for l in range(lo, hi + 1)]): print(f"layer {l}: {dt:.0f} s", flush=True)


if __name__ == "__main__": main()

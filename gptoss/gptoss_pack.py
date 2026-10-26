#!/usr/bin/env python3
"""gpt-oss-20b's GGUF (MXFP4) -> the per-layer NPU cache `gptoss_npu.py` streams: nothing requantised, K padded from 2880 to 2944
(the kernels' 128-wide slices; the padding columns' weights are zero and the activation rows are zero there):
  Q8_0  attention q / k / v / o, the head -> the "q8" qdot stream (pack_b_group_q8f qb=32, code 2)
  MXFP4 experts gate / up / down         -> "q5n" (code 7, 5.75 bits; the e2m1 values are 5-bit integers x a power of two, so the Q5_K
        stream holds them: per 256-super-block d = 2^(e_max - 5), a block's scale sc = 2^(e - e_max + 5), u = 2 w / 2^e + 12, m = sc,
        dmin = 12 d -- exact unless a block's exponent is more than 5 octaves below its super-block's max (0.0002% of gpt-oss-20b's
        blocks, then rounded to d sc units); K padded to 3072 for the 256-blocks), expert e at e x S. The 4.5-bit "mx4" stream (code 8,
        --experts mx4) is bandwidth-exact but its nibble decoder runs at 5 GB/s against q5n's 20 on the board.
  L{l}_small.npz: w_attn, w_post, the q / k / v / o biases, sinks, the router + bias, the experts' gate / up / down biases
    python3 gptoss_pack.py [--gguf ...] [--out /mnt/ssd/gptoss-npu] [--layers 0-23] [--jobs 2] [--experts q5n|mx4] [--head]
"""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "tinygrad")))); sys.path[:0] = [HERE, os.path.join(HERE, "..", "qwen3-30b-a3b"), os.path.join(HERE, "..", "qwen3.8-27b"), os.path.join(HERE, "..", "ornith-9b")]
from zy import gemm_fp16 as G                                             # noqa: E402
import qwen3moe_qgemv as Q                                                # noqa: E402
import gptoss_ref as OR                                                   # noqa: E402

NS, KS = 3, 32
OUT = os.environ.get("OSS_NPU", "/mnt/ssd/gptoss-npu")
HEAD_PARTS = 4
KP = 128                                                                  # K padded to a multiple of the kernels' slice


def write(path, arr):
  tmp = path + ".tmp"
  with open(tmp, "wb") as f: f.write(np.ascontiguousarray(arr).tobytes())
  os.replace(tmp, path)


def padk(a, k2, axis=1):
  if a.shape[axis] == k2: return a
  pad = [(0, 0)] * a.ndim; pad[axis] = (0, k2 - a.shape[axis]); return np.pad(a, pad)


def stream_q8(q, d):
  """Q8_0 (codes [N, K], scales [N, K / 32]) -> (the q8 qdot stream, code 2, npad, K padded)."""
  K2 = -(-q.shape[1] // KP) * KP; q, d = padk(q, K2), padk(d, K2 // 32); ng = -(-q.shape[0] // 48)
  return np.concatenate([G.pack_b_group_q8f(q, d, g, NS, KS, qb=32) for g in range(ng)]), 2, 48 * ng, K2


def stream_mx4(blk):
  """MXFP4 blocks [N, K / 32, 17] -> (the mx4 stream, code 8, npad, K padded)."""
  code, dh = Q.mxfp4_split_raw(blk); K2 = -(-code.shape[1] // KP) * KP; code, dh = padk(code, K2), padk(dh, K2 // 32); ng = -(-code.shape[0] // 48)
  return np.concatenate([Q.pack_b_group_mx4(code, dh, g) for g in range(ng)]), 8, 48 * ng, K2


def mx4_to_q5(code, dh):
  """MXFP4 (code [N, K] e2m1 nibbles, dh [N, K / 32] the halved scale 2^(e - 128)) -> the Q5_K fields (u, sc, m, d, dmin) at K padded
  to 256s: w = dh kv[code] = d sc (u - 12), d = max dh of the super-block / 32, sc = 32 dh / d_max (an integer for spreads <= 5)."""
  N, K = code.shape; K2 = -(-K // 256) * 256; code, dh = padk(code, K2), padk(dh, K2 // 32); v = Q.KV_FP4[code].astype(np.float64)   # 2 w / 2^e
  dh8 = dh.reshape(N, K2 // 256, 8).astype(np.float64); dmax = dh8.max(-1, keepdims=True); d = dmax / 32
  sc = np.where(dmax > 0, dh8 / np.where(dmax > 0, dmax, 1) * 32, 0); exact = np.isclose(sc, np.round(sc)) & (sc >= 1)
  sci = np.where(exact, np.round(sc), np.where(sc > 0, 1, 0))                         # a block more than 5 octaves down keeps sc 1, its values rounded
  f = np.where(sci > 0, dh8 / np.where(sci > 0, d * sci, 1), 0)                        # the value multiplier (1 when exact)
  u = np.clip(np.round(v.reshape(N, K2 // 256, 8, 32) * f[..., None]) + 12, 0, 31).astype(np.uint8).reshape(N, K2)
  sci = sci.astype(np.uint8).reshape(N, K2 // 32); return u, sci, sci.copy(), d.reshape(N, K2 // 256).astype(np.float32), (12 * d).reshape(N, K2 // 256).astype(np.float32)


def stream_q5n(blk):
  """MXFP4 blocks [N, K / 32, 17] -> (the q5n qdot stream, code 7, npad, K padded to 256s)."""
  u, sc, m, d, dmin = mx4_to_q5(*Q.mxfp4_split_raw(blk)); ng = -(-u.shape[0] // 48)
  return np.concatenate([Q.pack_b_group_q5n(u, sc, m, d, dmin, g) for g in range(ng)]), 7, 48 * ng, u.shape[1]


EXPERTS = {"mx4": stream_mx4, "q5n": stream_q5n}


def pack_layer(M, l, out, experts="q5n"):
  c, p = M.c, f"blk.{l}."; meta = {}; g = M.g
  for name, t in (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_output")):
    q, d = g.q8(p + t + ".weight"); st, code, npad, K2 = stream_q8(q, d); write(os.path.join(out, f"L{l}_{name}.bin"), st); meta[name] = (code, q.shape[0], npad, K2, 0)
  for name in ("gate", "up", "down"):
    kind, b = M.blocks(p + f"ffn_{name}_exps.weight"); assert kind == "MXFP4"; sts = []
    for e in range(b.shape[0]): st, code, npad, K2 = EXPERTS[experts](b[e]); sts.append(st)
    S = sts[0].size; assert all(s.size == S for s in sts); write(os.path.join(out, f"L{l}_{name}.bin"), np.concatenate(sts)); meta[name] = (code, b.shape[1], npad, K2, S)
  w = lambda n: g.f32(p + n).astype(np.float32)
  small = dict(w_attn=w("attn_norm.weight"), w_post=w("post_attention_norm.weight"), bq=w("attn_q.bias"), bk=w("attn_k.bias"), bv=w("attn_v.bias"), bo=w("attn_output.bias"), sinks=w("attn_sinks.weight"),
               router=w("ffn_gate_inp.weight"), rbias=w("ffn_gate_inp.bias"), bg=w("ffn_gate_exps.bias"), bu=w("ffn_up_exps.bias"), bd=w("ffn_down_exps.bias"))
  names = sorted(meta); np.savez(os.path.join(out, f"L{l}_small.npz"), meta_names=np.array(names), meta=np.array([list(meta[k]) for k in names], np.int64), **small)


def pack_head(M, out):
  q, d = M.g.q8("output.weight"); K2 = -(-q.shape[1] // KP) * KP; q, d = padk(q, K2), padk(d, K2 // 32); N = q.shape[0]; ng = -(-N // 48); per = -(-ng // HEAD_PARTS); parts = []
  for j in range(HEAD_PARTS):
    g0, g1 = j * per, min(ng, (j + 1) * per); r0 = g0 * 48
    write(os.path.join(out, f"head{j}.bin"), np.concatenate([G.pack_b_group_q8f(q, d, gg, NS, KS, qb=32) for gg in range(g0, g1)])); parts.append((2, min(N, g1 * 48) - r0, (g1 - g0) * 48, K2, r0))
  np.savez(os.path.join(out, "head.npz"), parts=np.array(parts, np.int64), w_out=M.g.f32("output_norm.weight").astype(np.float32))


def _one(args):
  gguf, l, out, experts = args; M = OR.Model(gguf); t0 = time.perf_counter(); pack_layer(M, l, out, experts); return l, time.perf_counter() - t0


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--gguf", default=OR.DEFAULT); ap.add_argument("--out", default=OUT); ap.add_argument("--layers", default="0-23")
  ap.add_argument("--jobs", type=int, default=2); ap.add_argument("--head", action="store_true"); ap.add_argument("--experts", default="q5n", choices=sorted(EXPERTS)); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
  if a.head: M = OR.Model(a.gguf); t0 = time.perf_counter(); pack_head(M, a.out); print(f"head: {time.perf_counter() - t0:.0f} s", flush=True); return
  lo, hi = (int(v) for v in a.layers.split("-")) if "-" in a.layers else (int(a.layers), int(a.layers))
  from multiprocessing import Pool
  with Pool(a.jobs) as pool:
    for l, dt in pool.imap_unordered(_one, [(a.gguf, l, a.out, a.experts) for l in range(lo, hi + 1)]): print(f"layer {l}: {dt:.0f} s", flush=True)


if __name__ == "__main__": main()

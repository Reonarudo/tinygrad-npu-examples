#!/usr/bin/env python3
"""Qwen3-30B-A3B's GGUF Q4_K_M -> the per-layer NPU cache the layers stream (`qwen3moe_npu.py`): every linear as a gemm_gs stream in
its own format -- Q4_K as `pack_b_group_q4f` (gemm_gs(q8=3): 4-bit codes, w = a u + b per 32 weights, 5.0 bits a weight), Q6_K as
`pack_b_group_q8f(qb=16)` (q8=4: 8-bit codes, a scale per 16, 9 bits) -- nothing requantised.

  L{l}_attn.bin   q | k|v (one GEMM: q's 86 groups, then k|v's 22 from group 86) when v is Q4_K; q | k (k from group 86) when v is Q6_K
  L{l}_v.bin      v alone (the 24 layers whose attn_v is Q6_K)
  L{l}_o.bin      o
  L{l}_gate.bin   the 128 experts' gate projections, expert e at e x S_gate (equal streams: gemm_gs(bsel)'s slots are e x S)
  L{l}_up.bin     the same for up
  L{l}_down.bin   the same for down (Q4_K or Q6_K by layer)
  L{l}_small.npz  the norms, the router with the post-attention norm's weight folded in, and `meta`: per stream (q8, N, npad, K, S)
  head{j}.bin     the output head (Q6_K) in parts of whole groups; head.npz their (q8, N, npad, K, row0)

    python3 qwen3moe_pack.py [--gguf ...] [--out /mnt/ssd/qwen3-30b-a3b-npu] [--layers 0-47] [--jobs 4]
"""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "tinygrad")))); sys.path[:0] = [HERE, os.path.join(HERE, "..", "qwen3.8-27b")]
from zy import gemm_fp16 as G                                           # noqa: E402
import qwen3moe_ref as QR                                               # noqa: E402

NS, KS = 3, 32                                                          # 3 strips (48 columns) a group; 128-wide K-slices
Q8 = {"Q4_K": 3, "Q6_K": 4}
Q4M = os.environ.get("QWEN3MOE_Q4", "q4m") == "q4m"                      # Q4_K as gemm_gs(q8=5) (q4m: a u by multiplies, b by an mma k-step; default) or q8=3 (q4f)
HEAD_PARTS = 4


def stream(kind, blk):
  """Super-blocks [N, K / 256, bytes] of one kind -> (the gemm_gs stream of all groups, its q8 mode, npad)."""
  N = blk.shape[0]; ng = -(-N // (16 * NS))
  if kind == "Q4_K":
    u, a, b = G.q4k_split(blk); st = [(G.pack_b_group_q4m if Q4M else G.pack_b_group_q4f)(u, a, b, g, NS, KS) for g in range(ng)]
    return np.concatenate(st), 5 if Q4M else 3, 16 * NS * ng
  elif kind == "Q6_K":
    q, d = G.q6k_split(blk); st = [G.pack_b_group_q8f(q, d, g, NS, KS, qb=16) for g in range(ng)]
  else: raise ValueError(kind)
  return np.concatenate(st), Q8[kind], 16 * NS * ng


def write(path, arr):
  tmp = path + ".tmp"
  with open(tmp, "wb") as f: f.write(np.ascontiguousarray(arr).tobytes())
  os.replace(tmp, path)


def pack_layer(M, l, out):
  c, p = M.c, f"blk.{l}."; meta = {}
  def lin(name, parts):
    """parts: [[tensor, ...], ...] -- each sub-part's tensors' rows stacked, sub-parts starting at whole groups; one kind."""
    sts, goffs, g, kinds = [], [], 0, set()
    for sub in parts:
      kb = [M.blocks(p + t + ".weight") for t in sub]; kinds |= {k for k, _ in kb}
      st, q8, npad = stream(kb[0][0], np.concatenate([b for _, b in kb])); sts.append(st); goffs.append(g); g += npad // 48
    assert len(kinds) == 1, (l, name, kinds); K = kb[0][1].shape[1] * 256
    write(os.path.join(out, f"L{l}_{name}.bin"), np.concatenate(sts)); meta[name] = (q8, 0, 48 * g, K, 0) + tuple(goffs)
  vq6 = M.kind(p + "attn_v.weight") == "Q6_K"
  if vq6: lin("attn", [["attn_q"], ["attn_k"]]); lin("v", [["attn_v"]])
  else: lin("attn", [["attn_q"], ["attn_k", "attn_v"]])
  lin("o", [["attn_output"]])
  for t in ("gate", "up", "down"):
    kind, b = M.blocks(f"{p}ffn_{t}_exps.weight"); E = b.shape[0]; sts = []
    for e in range(E): st, q8, npad = stream(kind, b[e]); sts.append(st)
    S = sts[0].size; assert all(s.size == S for s in sts)
    write(os.path.join(out, f"L{l}_{t}.bin"), np.concatenate(sts)); meta[t] = (q8, b.shape[1], npad, b.shape[2] * 256, S)
  w = lambda n: M.t(p + n).astype(np.float32)
  small = dict(w_in=w("attn_norm.weight"), w_ffn=w("ffn_norm.weight"), knw=w("attn_k_norm.weight"),
               qnw=(w("attn_q_norm.weight") * np.float32(1.0 / np.sqrt(c.HD))).astype(np.float32),      # the scores' 1 / sqrt(HD) folded in
               router=np.ascontiguousarray(M.t(p + "ffn_gate_inp.weight").astype(np.float32) * w("ffn_norm.weight")[None]))
  names = sorted(meta); mx = max(len(meta[k]) for k in names)
  np.savez(os.path.join(out, f"L{l}_small.npz"), meta_names=np.array(names), meta=np.array([list(meta[k]) + [-1] * (mx - len(meta[k])) for k in names], np.int64), **small)


def pack_head(M, out):
  kind, b = M.blocks("output.weight"); N = b.shape[0]; ng = -(-N // 48); per = -(-ng // HEAD_PARTS); rows = []
  for j in range(HEAD_PARTS):
    r0, r1 = 48 * per * j, min(N, 48 * per * (j + 1)); st, q8, npad = stream(kind, b[r0:r1])
    write(os.path.join(out, f"head{j}.bin"), st); rows.append((q8, r1 - r0, npad, b.shape[1] * 256, r0))
  np.savez(os.path.join(out, "head.npz"), parts=np.array(rows, np.int64), w_out=M.t("output_norm.weight").astype(np.float32))


def pack_q6n(M, l, out):
  """Layer l's decode-only q6n copies of its Q6_K linears (down_q6n, v_q6n: qwen3moe_qgemv.pack_b_group_q6n, 6.6 bits a weight
  against the prefill's 9), appended to its meta as (6, N, npad, K, S). The decode layer reads them when present (QWEN3MOE_Q6N)."""
  import qwen3moe_qgemv as Q
  c, p = M.c, f"blk.{l}."; z = np.load(os.path.join(out, f"L{l}_small.npz")); meta = {str(k): [int(v) for v in m] for k, m in zip(z["meta_names"], z["meta"])}
  if "down_q6n" in meta: return False
  def st6(blk):
    N = blk.shape[0]; q, sc, d = Q.q6k_split_raw(blk); ng = -(-N // 48)
    return np.concatenate([Q.pack_b_group_q6n(q, sc, d, g) for g in range(ng)]), 48 * ng
  todo = []
  if M.kind(p + "ffn_down_exps.weight") == "Q6_K":
    kind, b = M.blocks(p + "ffn_down_exps.weight"); sts = []
    for e in range(b.shape[0]): st, npad = st6(b[e]); sts.append(st)
    write(os.path.join(out, f"L{l}_down_q6n.bin"), np.concatenate(sts)); todo.append(("down_q6n", (6, b.shape[1], npad, b.shape[2] * 256, sts[0].size)))
  if M.kind(p + "attn_v.weight") == "Q6_K":
    kind, b = M.blocks(p + "attn_v.weight"); st, npad = st6(b)
    write(os.path.join(out, f"L{l}_v_q6n.bin"), st); todo.append(("v_q6n", (6, 0, npad, b.shape[1] * 256, 0, 0)))
  if not todo: return False
  for k, m in todo: meta[k] = list(m)
  names = sorted(meta); mx = max(len(meta[k]) for k in names); small = {k: z[k] for k in z.files if not k.startswith("meta")}
  np.savez(os.path.join(out, f"L{l}_small.npz.tmp.npz"), meta_names=np.array(names), meta=np.array([[v for v in meta[k] if v != -1] + [-1] * (mx - len([v for v in meta[k] if v != -1])) for k in names], np.int64), **small)
  os.replace(os.path.join(out, f"L{l}_small.npz.tmp.npz"), os.path.join(out, f"L{l}_small.npz")); return True


def pack_q4n(M, l, out):
  """Layer l's decode-only q4n copies of its Q4_K attention linears (attn_q4n: q | k (| v) stacked as attn, o_q4n), 4.75 bits a weight
  against the shared stream's 5.0, strip-major (16-column units); appended to its meta as (7, 0, npad, K, 0, goffs..)."""
  import qwen3moe_qgemv as Q
  c, p = M.c, f"blk.{l}."; z = np.load(os.path.join(out, f"L{l}_small.npz")); meta = {str(k): [int(v) for v in m if v != -1] for k, m in zip(z["meta_names"], z["meta"])}
  if "attn_q4n" in meta: return False
  def st4(blk):
    N = blk.shape[0]; u, sc, m, d, dmin = Q.q4k_split_raw(blk); ng = -(-N // 48)
    return np.concatenate([Q.pack_b_group_q4n(u, sc, m, d, dmin, st) for st in range(3 * ng)]), 48 * ng
  def lin(name, parts):
    sts, goffs, g = [], [], 0
    for sub in parts:
      kb = [M.blocks(p + t + ".weight") for t in sub]; assert all(k == "Q4_K" for k, _ in kb), (l, name)
      st, npad = st4(np.concatenate([b for _, b in kb])); sts.append(st); goffs.append(g); g += npad // 48
    write(os.path.join(out, f"L{l}_{name}.bin"), np.concatenate(sts)); meta[name] = [7, 0, 48 * g, kb[0][1].shape[1] * 256, 0] + goffs
  vq6 = M.kind(p + "attn_v.weight") == "Q6_K"
  lin("attn_q4n", [["attn_q"], ["attn_k"]] if vq6 else [["attn_q"], ["attn_k", "attn_v"]]); lin("o_q4n", [["attn_output"]])
  names = sorted(meta); mx = max(len(meta[k]) for k in names); small = {k: z[k] for k in z.files if not k.startswith("meta")}
  np.savez(os.path.join(out, f"L{l}_small.npz.tmp.npz"), meta_names=np.array(names), meta=np.array([meta[k] + [-1] * (mx - len(meta[k])) for k in names], np.int64), **small)
  os.replace(os.path.join(out, f"L{l}_small.npz.tmp.npz"), os.path.join(out, f"L{l}_small.npz")); return True


def _pack_q4n_one(args):
  path, l, out = args; return l, pack_q4n(QR.Model(path), l, out)


def _pack_q6n_one(args):
  path, l, out = args; return l, pack_q6n(QR.Model(path), l, out)


def pack_head_q6n(M, out):
  """The head's decode-only stream (qwen3moe_qgemv.pack_b_group_q6n, 6.6 bits a weight; the qdot head reads it when present)."""
  import qwen3moe_qgemv as Q
  kind, b = M.blocks("output.weight"); assert kind == "Q6_K"; N = b.shape[0]; q, sc, d = Q.q6k_split_raw(b)
  write(os.path.join(out, "head_q6n.bin"), np.concatenate([Q.pack_b_group_q6n(q, sc, d, g) for g in range(-(-N // 48))]))


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--gguf", default=QR.DEFAULT); ap.add_argument("--out", default="/mnt/ssd/qwen3-30b-a3b-npu")
  ap.add_argument("--layers", default="all"); ap.add_argument("--head", action="store_true", help="the head too (default: when --layers all)")
  ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--head-q6n", action="store_true", help="only the head's decode-only q6n stream")
  ap.add_argument("--q6n", action="store_true", help="only the layers' decode-only q6n copies of their Q6_K linears (down, v)")
  ap.add_argument("--q4n", action="store_true", help="only the layers' decode-only q4n copies of their Q4_K attention linears (attn, o)"); a = ap.parse_args()
  os.makedirs(a.out, exist_ok=True); M = QR.Model(a.gguf)
  layers = list(range(M.c.NL)) if a.layers == "all" else [int(v) for v in a.layers.split(",")] if "-" not in a.layers else list(range(int(a.layers.split("-")[0]), int(a.layers.split("-")[1]) + 1))
  t0 = time.perf_counter()
  if a.head_q6n: pack_head_q6n(M, a.out); print(f"head q6n: {time.perf_counter() - t0:.0f} s"); return
  if a.q6n or a.q4n:
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.jobs) as ex:
      for l, done in ex.map(_pack_q6n_one if a.q6n else _pack_q4n_one, [(a.gguf, l, a.out) for l in layers]): print(f"layer {l}: {'packed' if done else 'nothing to do'} ({time.perf_counter() - t0:.0f} s)", flush=True)
    return
  if a.head or a.layers == "all": pack_head(M, a.out); print(f"head: {time.perf_counter() - t0:.0f} s", flush=True)
  todo = [l for l in layers if not os.path.exists(os.path.join(a.out, f"L{l}_small.npz"))]   # (resumable: small.npz is written last)
  if a.jobs > 1 and len(todo) > 1:
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.jobs) as ex:
      for l in ex.map(_pack_one, [(a.gguf, l, a.out) for l in todo]): print(f"layer {l}: {time.perf_counter() - t0:.0f} s", flush=True)
  else:
    for l in todo: pack_layer(M, l, a.out); print(f"layer {l}: {time.perf_counter() - t0:.0f} s", flush=True)
  print(f"done: {len(todo)} layers in {time.perf_counter() - t0:.0f} s -> {a.out}")


def _pack_one(args):
  path, l, out = args; pack_layer(QR.Model(path), l, out); return l


if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Qwen3.5-35B-A3B's GGUF (Q4_K_M) -> the per-layer NPU cache `qwen35moe_npu.py` streams: one K-slice convention (ks 32, 3 strips a
group) for every linear, each in its checkpoint's format:

  Q8_0 linears (the DeltaNet in_proj_qkv / in_proj_z / out_proj, the attention q / k|v / o, the shared expert's gate / up / down):
                gemm_gs(q8=2) streams -- `pack_b_group_q8f` (8-bit codes, an fp16 scale per 32; the kernel builds the fp16 weights)
                under the HF names and head order (ornith_weights undoes the GGUF converter's V-head tiling)
  Q4_K experts (gate / up): gemm_gs(q8=5) `pack_b_group_q4m` streams, expert e at e x S (gemm_gs(bsel)'s slots), as qwen3moe_pack
  Q5_K / Q6_K experts (down): the decode-only qdot streams q5n (5.75 bits, qwen3moe_qgemv.pack_b_group_q5n; meta code 7) / q6n (6.6
                bits, code 6) -- nothing requantised; read by the qdot GEMV only (the prefill of these goes token by token until a
                matrix-unit expand exists). `q5k_requant_q4` (the 4-bit requantisation, 8-9 % a weight) stays for reference.
  L{l}_small.npz: the norms (1 + w folded where the kernels take it), the DeltaNet parameters in gdn_tok3's conventions (as
                qwen38_npu reads them: in_proj_a|b rows with the input norm's weight folded, conv taps transposed, A_log, dt_bias,
                the gated norm), the router with the post-attention norm folded in, the shared expert's gate row; `meta`: per stream
                (q8, N, npad, K, S)
  head{j}.bin / head.npz: the Q6_K output head in parts (as qwen3moe_pack), outside_small.npz the final norm

    python3 qwen35moe_pack.py [--gguf ...] [--out /mnt/ssd/qwen35moe-npu] [--layers 0-39] [--jobs 4]
"""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["QWEN_MODEL"] = "qwen3.5-35b-a3b"
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "tinygrad")))); sys.path[:0] = [HERE, os.path.join(HERE, "..", "qwen3.8-27b"), os.path.join(HERE, "..", "qwen3-30b-a3b"), os.path.join(HERE, "..", "ornith-9b")]
from zy import gemm_fp16 as G                                             # noqa: E402
import qwen38_ref as R                                                    # noqa: E402
from qwen35moe_weights import Weights, DEQ                                # noqa: E402

NS, KS = 3, 32
DEFAULT = os.environ.get("QWEN35MOE_GGUF", "/mnt/ssd/models/Qwen3.5-35B-A3B-Q4_K_M.gguf")
OUT = os.environ.get("QWEN35MOE_NPU", "/mnt/ssd/qwen35moe-npu")
HEAD_PARTS = 4
# the Q8_0 linears of a layer (HF names, as qwen38_pack.LINEARS; the attention's q | k|v fused along N as there)
LIN8 = {"linear": [("qkv", ("linear_attn.in_proj_qkv",)), ("z", ("linear_attn.in_proj_z",)), ("o", ("linear_attn.out_proj",))],
        "full": [("q", ("self_attn.q_proj",)), ("kv", ("self_attn.k_proj", "self_attn.v_proj")), ("o", ("self_attn.o_proj",))]}


def write(path, arr):
  tmp = path + ".tmp"
  with open(tmp, "wb") as f: f.write(np.ascontiguousarray(arr).tobytes())
  os.replace(tmp, path)


def stream_q8f(q, d):
  """Q8_0 (codes int8 [N, K], scales [N, K / 32]) -> (the q8f stream of all groups, q8 mode 2, npad)."""
  ng = -(-q.shape[0] // (16 * NS)); return np.concatenate([G.pack_b_group_q8f(q, d, g, NS, KS) for g in range(ng)]), 2, 16 * NS * ng


def q5k_requant_q4(blk):
  """Q5_K super-blocks [N, K / 256, 176] -> (u uint8 [N, K] in 0..15, a, b float32 [N, K / 32]) with w ~ a u + b: ggml's values
  requantised per 32-weight block (min / max)."""
  N, nsb = blk.shape[:2]; w = DEQ["Q5_K"](blk.reshape(-1, 176)).reshape(N, nsb * 8, 32)
  lo, hi = w.min(-1), w.max(-1); a = (hi - lo) / 15.0; a = np.where(a == 0, 1.0, a).astype(np.float32); b = lo.astype(np.float32)
  u = np.clip(np.rint((w - b[..., None]) / a[..., None]), 0, 15).astype(np.uint8)
  return u.reshape(N, nsb * 256), a.reshape(N, nsb * 8), b.reshape(N, nsb * 8)


def stream_kquant(kind, blk):
  """K-quant super-blocks [N, K / 256, bytes] of an expert -> (the stream of all groups, its code, npad): Q4_K as q4m (gemm_gs q8=5 and the
  qdot kind "q4"), Q5_K as q5n (7), Q6_K as q6n (6)."""
  import qwen3moe_qgemv as Q
  N = blk.shape[0]; ng = -(-N // (16 * NS))
  if kind == "Q4_K": u, a, b = G.q4k_split(blk); return np.concatenate([G.pack_b_group_q4m(u, a, b, g, NS, KS) for g in range(ng)]), 5, 16 * NS * ng
  if kind == "Q5_K": u, sc, m, d, dmin = Q.q5k_split_raw(blk); return np.concatenate([Q.pack_b_group_q5n(u, sc, m, d, dmin, g) for g in range(ng)]), 7, 16 * NS * ng
  if kind == "Q6_K": q, sc, d = Q.q6k_split_raw(blk); return np.concatenate([Q.pack_b_group_q6n(q, sc, d, g) for g in range(ng)]), 6, 16 * NS * ng
  raise ValueError(kind)


def pack_layer(W, l, out):
  kind, p = R.LAYER_TYPES[l], f"model.language_model.layers.{l}."; meta = {}; g = W.g; q = f"blk.{l}."
  for name, parts in LIN8[kind]:                                           # the Q8_0 linears in the HF head order
    qd = [W.q8(p + x) for x in parts]; cq, cd = np.concatenate([a for a, _ in qd]), np.concatenate([b for _, b in qd])
    st, q8, npad = stream_q8f(cq, cd); write(os.path.join(out, f"L{l}_{name}.bin"), st); meta[name] = (q8, cq.shape[0], npad, cq.shape[1], 0)
  for t in ("gate", "up", "down"):                                          # the experts: one stream file, expert e at e x S
    kd, b = W.blocks(f"{q}ffn_{t}_exps.weight"); sts = []
    for e in range(b.shape[0]): st, q8, npad = stream_kquant(kd, b[e]); sts.append(st)
    S = sts[0].size; assert all(s.size == S for s in sts)
    write(os.path.join(out, f"L{l}_{t}.bin"), np.concatenate(sts)); meta[t] = (q8, b.shape[1], npad, b.shape[2] * 256, S)
  for name, t in (("sgate", "gate_shexp"), ("sup", "up_shexp"), ("sdn", "down_shexp")):   # the shared expert (Q8_0): a 9th head of the experts' GEMMs
    cq, cd = g.q8(f"{q}ffn_{t}.weight"); st, q8, npad = stream_q8f(cq, cd); write(os.path.join(out, f"L{l}_{name}.bin"), st); meta[name] = (q8, cq.shape[0], npad, cq.shape[1], 0)
  small = {}
  small["input_layernorm_weight"] = W.f32(p + "input_layernorm.weight"); small["post_attention_layernorm_weight"] = W.f32(p + "post_attention_layernorm.weight")
  if kind == "linear":
    for n in ("linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.conv1d.weight", "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight", "linear_attn.norm.weight"):
      small[n.replace(".", "_")] = W.f32(p + n)
  else:
    for n in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"): small[n.replace(".", "_")] = W.f32(p + n)
  wpost = 1.0 + small["post_attention_layernorm_weight"]
  small["router"] = np.ascontiguousarray(g.f32(f"{q}ffn_gate_inp.weight").astype(np.float32) * wpost[None])     # on rms(x) (the norm folded)
  small["shexp_gate"] = np.ascontiguousarray(g.f32(f"{q}ffn_gate_inp_shexp.weight").astype(np.float32).reshape(-1) * wpost)
  names = sorted(meta)
  np.savez(os.path.join(out, f"L{l}_small.npz"), meta_names=np.array(names), meta=np.array([list(meta[k]) for k in names], np.int64), **small)


def pack_head(W, out):
  kd, b = W.blocks("output.weight"); assert kd == "Q6_K", kd; N = b.shape[0]; ng = -(-N // (16 * NS)); per = -(-ng // HEAD_PARTS) * 1
  q, d = G.q6k_split(b); parts = []
  for j in range(HEAD_PARTS):
    g0, g1 = j * per, min(ng, (j + 1) * per); r0 = g0 * 16 * NS
    st = np.concatenate([G.pack_b_group_q8f(q, d, gg, NS, KS, qb=16) for gg in range(g0, g1)])
    write(os.path.join(out, f"head{j}.bin"), st); parts.append((4, min(N, g1 * 16 * NS) - r0, (g1 - g0) * 16 * NS, q.shape[1], r0))
  np.savez(os.path.join(out, "head.npz"), parts=np.array(parts, np.int64), w_out=(1.0 + W.f32("model.language_model.norm.weight")).astype(np.float32))


def _one(args):
  gguf, l, out = args; W = Weights(gguf); t0 = time.perf_counter(); pack_layer(W, l, out); return l, time.perf_counter() - t0


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--gguf", default=DEFAULT); ap.add_argument("--out", default=OUT)
  ap.add_argument("--layers", default="0-39"); ap.add_argument("--jobs", type=int, default=2); ap.add_argument("--head", action="store_true")
  a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
  lo, hi = (int(v) for v in a.layers.split("-")) if "-" in a.layers else (int(a.layers), int(a.layers))
  if a.head: W = Weights(a.gguf); t0 = time.perf_counter(); pack_head(W, a.out); print(f"head: {time.perf_counter() - t0:.0f} s", flush=True); return
  from multiprocessing import Pool
  with Pool(a.jobs) as pool:
    for l, dt in pool.imap_unordered(_one, [(a.gguf, l, a.out) for l in range(lo, hi + 1)]): print(f"layer {l}: {dt:.0f} s", flush=True)


if __name__ == "__main__": main()

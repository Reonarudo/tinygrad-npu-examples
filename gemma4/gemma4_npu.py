#!/usr/bin/env python3
"""One Gemma 4 decoder layer on the NPU, in the GEMM's rows mode (n <= 12 rows: a decode step or a verify pass), against the numpy
reference (gemma4_ref.py). The linears run on the TEC matrix unit as the exact Q8_0 GEMM (`gemm_gs(q8=1)`: int8 codes, a scale
per 32 weights, packed by gemm_fp16.pack_b_group_q8 -- from the GGUF here, or from gemma4_pack.py's cache); everything around
them is hand-written kernels: qwen38_kernels' rms_a (the A producer with the norm: Gemma's w passed as the multiplier), unpack and
swiglu_a(act="gelu"), and gemma4_kernels' gattn_kv / gattn_q (attention), pnresid32 (post-norm + residual + layer scalar) and
gmul_a32 (the PLE gate).

`layer_body` issues one layer's kernels on fixed buffers -- the input rows `x`, the output rows `out`, the caches, the position
buffer, the stack of every layer's PLE rows (and the layer's offset into it) -- so the same code runs eagerly here (the gate) and
under TinyJit in gemma4_generate.py (a frozen graph per layer block, replayed every token with no host copies).
The layer's own q | k|v run as one GEMM (one stream: q's groups, then k|v's from group `goff`).

    python3 gemma4_npu.py --gate 14 --n 4      (layer 14 on a prompt's last 4 tokens: their hidden states, the cache prefix and a
                                                 KV-shared layer's source cache from the numpy reference's layers before it)
"""
import argparse, ctypes, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "tinygrad")))); sys.path.insert(0, os.path.join(HERE, "..", "qwen3.8-27b")); sys.path.insert(0, HERE)
from tinygrad import Tensor, dtypes                                      # noqa: E402
from zy import OA, gemm_fp16 as G                                        # noqa: E402
from qwen38_kernels import Kernels                                       # noqa: E402
import gemma4_kernels as GK                                              # noqa: E402
import gemma4_ref as GR                                                  # noqa: E402

DEV, NS, KS = "ZHOUYI", 3, 8                                              # 3 strips a group; K-slices of one Q8_0 block (q8=1)
PLE_ROWS = 12                                                            # the PLE stack's rows a layer ([NL][12][PLE]: any n <= 12)
# rows mode, n <= 12 (decode, verify, prefill chunks): GEMMA_PN_DMA (default 1) the post-norm residual as pnss + pnap and the PLE gate
# as gmul_a32d (DMA; else pnresid32's / gmul_a32's plain loads), GEMMA_ATT_DMA (default 1) the attention as gattn_kvd + gattn_part +
# gattn_comb (kv-head units -- and row groups past 4 rows --, DMA, online softmax; else gattn_kv + gattn_q). With the DMA attention
# the local layers' caches are rings of RING rows (gemma4_generate)
PN_DMA = os.environ.get("GEMMA_PN_DMA", "1") == "1"
ATT_DMA = os.environ.get("GEMMA_ATT_DMA", "1") == "1"
def ring_rows(c): return c.W + 16                                         # >= W + M - 1 for any pass of M <= 12 rows

def dev(a, dt=None): return Tensor(np.ascontiguousarray(a), device=DEV, **({} if dt is None else {"dtype": dt})).realize()
def rup(x, m): return -(-x // m) * m
def poke(t, a):
  """The host array `a` into the start of the realised device tensor `t`'s buffer (a memmove into its host mapping), after a job
  left in flight (the async graph tail) has finished: qwen38_npu.poke's rule."""
  from tinygrad import Device
  Device[DEV].synchronize(); a = np.ascontiguousarray(a); b = t.uop.base.buffer; b.ensure_allocated(); raw = b._buf
  assert a.nbytes <= b.nbytes and raw.va, (a.nbytes, b.nbytes)
  ctypes.memmove(raw.va, a.ctypes.data, a.nbytes)

def q8_stream(q, d):
  """Q8_0 codes [N, K] + scales [N, K / 32] -> the gemm_gs(q8=1) stream (all groups) and N padded to the group (48)."""
  ng = -(-q.shape[0] // (16 * NS)); return np.concatenate([G.pack_b_group_q8(q, d, g, NS, KS) for g in range(ng)]), 16 * NS * ng

def q8_quantise(w):
  """A float matrix [N, K] -> GGUF Q8_0 (codes int8 [N, K], scales fp16-rounded float32 [N, K / 32]): d = amax / 127 per 32, as ggml."""
  b = w.reshape(w.shape[0], -1, 32).astype(np.float32); d = (np.abs(b).max(-1) / np.float32(127.0)).astype(np.float16).astype(np.float32)
  q = np.clip(np.round(b / np.where(d == 0, 1, d)[..., None]), -127, 127).astype(np.int8)
  return q.reshape(w.shape), d

def layer_linears(M, l):
  """Layer l's linears: name -> [sub-parts], each sub-part a list of GGUF tensors whose rows are stacked; a fused linear's sub-parts
  start at whole groups (their goff). The layer's own q | k|v fused ("qkv"); a KV-shared layer has q alone."""
  c = M.c; p = f"blk.{l}."; own = c.KV_SRC is None or c.KV_SRC[l] == l; vsame = not M.has(p + "attn_v.weight")
  lin = {"qkv": [["attn_q"], ["attn_k"] if vsame else ["attn_k", "attn_v"]]} if own else {"q": [["attn_q"]]}
  lin.update(o=[["attn_output"]], gu=[["ffn_gate", "ffn_up"]], dn=[["ffn_down"]])
  if c.PLE: lin.update(pg=[["inp_gate"]], pp=[["proj"]])
  return lin

def pack_linear(M, l, subs):
  """-> (stream uint8, npad, K, [goff of each sub-part])."""
  p = f"blk.{l}."; sts, goffs, g, K_ = [], [], 0, None
  for parts in subs:
    qd = [M.g.q8(f"{p}{t}.weight") for t in parts]; q, d = np.concatenate([a for a, _ in qd]), np.concatenate([b for _, b in qd])
    st, npad = q8_stream(q, d); sts.append(st); goffs.append(g); g += npad // 48; K_ = q.shape[1]
  return np.concatenate(sts), 48 * g, K_, goffs

def layer_small(M, l):
  """Layer l's small fp32 vectors, as the kernels take them (wv_*: the post-norm weight, then the scalar the residual add ends with)."""
  c = M.c; p = f"blk.{l}."; own = c.KV_SRC is None or c.KV_SRC[l] == l
  w = lambda t: M.t(p + t).astype(np.float32)
  s = float(M.t(p + "layer_output_scale.weight")[0]) if M.has(p + "layer_output_scale.weight") else 1.0
  sm = {"w_in": w("attn_norm.weight"), "w_ffn": w("ffn_norm.weight"), "qnw": w("attn_q_norm.weight"),
        "wv_attn": np.append(w("post_attention_norm.weight"), np.float32(1.0)),
        "wv_ffw": np.append(w("post_ffw_norm.weight"), np.float32(1.0 if c.PLE else s))}
  if own: sm["knw"] = w("attn_k_norm.weight")
  if c.PLE: sm["wv_ple"] = np.append(w("post_norm.weight"), np.float32(s))
  return sm

class LayerW:
  """Layer l's weights on the device: `lin` name -> (B stream tensor, npad, K), `goff` name -> its sub-parts' group offsets,
  `sm` name -> fp32 vector tensor; `own` (the layer writes a cache), `vsame` (V = K)."""
  def __init__(self, c, l, lin, goff, sm, vsame=False):
    self.c, self.l, self.lin, self.goff, self.sm, self.vsame = c, l, lin, goff, sm, vsame
    self.own = c.KV_SRC is None or c.KV_SRC[l] == l
  @classmethod
  def from_gguf(cls, M, l):
    lin, goff = {}, {}
    for k, subs in layer_linears(M, l).items():
      st, npad, K_, g = pack_linear(M, l, subs); lin[k] = (dev(st), npad, K_); goff[k] = g
    return cls(M.c, l, lin, goff, {k: dev(v) for k, v in layer_small(M, l).items()}, vsame=not M.has(f"blk.{l}.attn_v.weight"))
  def gemm(self, K, a, k, n):
    b, npad, K_ = self.lin[k]; ng = npad // (16 * NS)
    out = K.buf(f"ct_{k}_{ng}", ng * K.nrb * NS * 192, dtypes.float32)
    return OA.gemm_gs(a, b, ks=KS, ns=NS, nrb=K.nrb, nslices=K_ // (4 * KS), ngroups=ng, b8=True, bscale=True, q8=1, scales="dup",
                      piece=K.nrb, rows=n, out=out).realize()

CT_SHARED = {}
class SharedK(Kernels):
  """Kernels whose GEMM output tiles (tags ct_*) are one buffer per (tag, size) for every geometry and the drafter: their sizes do not
  depend on the rows (rows mode: nrb 2), and the geometries never run at once. E4B's head tiles alone are 25 MB a geometry; with a
  set per geometry the device's buffer memory ran out at 5 geometries + the drafter."""
  def buf(self, tag, n, dt):
    if tag.startswith("ct_") and tag not in self.bufs:
      key = (tag, n, dt)
      if key not in CT_SHARED: CT_SHARED[key] = Tensor.zeros(n, device=DEV, dtype=dt).contiguous().realize()
      self.bufs[tag] = CT_SHARED[key]
    return super().buf(tag, n, dt)

def _desc(K, key, fn):
  if key not in K.bufs: K.bufs[key] = dev(fn())
  return K.bufs[key]

def pnres(K, c_, n, eps, out, x, ct, wv):
  """out = (x + rms(C) * w) * s (wv = w | s): pnss + pnap by DMA (PN_DMA, rows mode, n <= 4), else pnresid32."""
  if PN_DMA and K.compact and n <= 12:
    d = _desc(K, f"pnres_desc|{c_}|{n}", lambda: GK.pnres_desc(c_, n)); ss = K.buf("pn_ss", 12 * 16, dtypes.float32)
    K.call(f"pnss|{c_}|{n}", GK.pnss_src(c_, K.nrb, n), ss, ct, d)
    return K.call(f"pnap|{c_}|{n}", GK.pnap_src(c_, K.nrb, eps, n), out, x, ct, wv, ss, d)
  return K.call(f"pnresid32|{c_}|{n}", GK.pnresid32_src(c_, K.nrb, eps, real=n), out, x, ct, wv)

def layer_body(K, W, n, x, out, Kc, Vc, cs, posb, tmax, pl=None, pidx=None, ring=0):
    """Layer W.l on n rows: x fp32 rows [R, H] (a persistent buffer) -> `out` (another); Kc / Vc: the layer's (or its KV source's)
    caches [TMAX * NKV * HD] (`ring`: a local layer's ring of that many rows, position t at row t % ring -- the DMA attention only);
    cs: the layer kind's RoPE table [TMAX][HD]; posb int32 [16]: the first row's position; pl / pidx: the stack of every layer's PLE
    rows [NL][PLE_ROWS][PLE] and this layer's offset into it (int32 [1]). Every kernel writes a persistent buffer of K (scratch shared
    by the layers: they run in order). Returns `out`."""
    c, l = W.c, W.l; H, hd, nkv, sm = c.H, c.HD[l], c.NKV[l], W.sm; R_ = K.nrb
    a_h = K.rms_a(x, sm["w_in"], H, "a_h")
    ct = W.gemm(K, a_h, "qkv" if W.own else "q", n)
    q_rows = K.unpack(ct, c.NH * hd, f"q_rows{hd}")
    att_dma = ATT_DMA and K.compact and n <= 12; win = c.W if c.SWA[l] else 0
    assert att_dma or not ring, "the ring caches need the DMA attention (GEMMA_ATT_DMA=1, n <= 12)"
    if W.own:
      KR = (1 if W.vsame else 2) * nkv * hd
      kv_rows = K.unpack(ct, KR, f"kv_rows{KR}", goff=W.goff["qkv"][1])
      if att_dma: K.call(f"gattn_kvd|{nkv}|{hd}|{n}|{KR}|{W.vsame}|{ring}", GK.gattn_kvd_src(nkv, hd, c.EPS, n, KR, W.vsame, ring), Kc, Vc, kv_rows, sm["knw"], cs, posb,
                         _desc(K, f"gattn_kvd_desc|{KR}|{hd}", lambda: GK.gattn_kvd_desc(KR, hd)))
      else: K.call(f"gattn_kv|{nkv}|{hd}|{n}|{KR}|{W.vsame}", GK.gattn_kv_src(nkv, hd, tmax, c.EPS, n, KR, W.vsame), Kc, Vc, kv_rows, sm["knw"], cs, posb)
    o_rows = K.rows_buf(f"o_rows{hd}", c.NH * hd)
    if att_dma:
      HG, BT, P, MR = GK.gattn_cfgr(c.NH, nkv, hd, n); part = K.buf(f"ga_part|{hd}|{n}", GK.gattn_part_size(c.NH, nkv, hd, n), dtypes.float32)
      K.call(f"gattn_part|{nkv}|{hd}|{n}|{win}|{ring}", GK.gattn_part_src(c.NH, nkv, hd, c.EPS, n, c.NH * hd, win, ring), part, q_rows, Kc, Vc, sm["qnw"], cs, posb,
             _desc(K, f"gattn_part_desc|{nkv}|{hd}|{n}", lambda: GK.gattn_part_desc(nkv, hd, MR, HG, BT)))
      K.call(f"gattn_comb|{nkv}|{hd}|{n}", GK.gattn_comb_src(c.NH, nkv, hd, n), o_rows, part, _desc(K, f"gattn_comb_desc|{nkv}|{hd}|{n}", lambda: GK.gattn_comb_desc(c.NH, nkv, hd, n)))
    else: K.call(f"gattn_q|{nkv}|{hd}|{n}|{tmax}|{win}", GK.gattn_q_src(c.NH, nkv, hd, tmax, c.EPS, n, c.NH * hd, win), o_rows, q_rows, Kc, Vc, sm["qnw"], cs, posb)
    x1 = pnres(K, H, n, c.EPS, K.rows_buf("x1", H), x, W.gemm(K, K.rms_a(o_rows, None, c.NH * hd, f"a_o{hd}", norm=False), "o", n), sm["wv_attn"])
    ff = c.FF[l]
    a_g = K.swiglu_a(W.gemm(K, K.rms_a(x1, sm["w_ffn"], H, "a_f"), "gu", n), ff, f"a_sw{ff}", act="gelu")
    x2 = pnres(K, H, n, c.EPS, out if not c.PLE else K.rows_buf("x2", H), x1, W.gemm(K, a_g, "dn", n), sm["wv_ffw"])
    if not c.PLE: return x2
    ct_pg = W.gemm(K, K.rms_a(x2, None, H, "a_p", norm=False), "pg", n)
    a_pm = K.buf("a_pm", K.a_size(c.PLE), dtypes.uint16)
    if PN_DMA and K.compact and n <= 12:
      K.call(f"gmul_a32d|{c.PLE}|{n}|poff", GK.gmul_a32d_src(c.PLE, R_, n, poff=True), a_pm, ct_pg, pl, pidx, _desc(K, f"gmul_a32d_desc|{c.PLE}|{n}", lambda: GK.gmul_a32d_desc(c.PLE, R_, n)))
    else: K.call(f"gmul_a32|{c.PLE}|{n}|{K.compact}|poff", GK.gmul_a32_src(c.PLE, R_, real=n, compact=K.compact, poff=True), a_pm, ct_pg, pl, pidx)
    return pnres(K, H, n, c.EPS, out, x2, W.gemm(K, a_pm, "pp", n), sm["wv_ple"])

def rope_cs(M, l, tmax):
  """Layer l's kind's RoPE table [TMAX][HD]: cos then sin of the HD / 2 frequencies (the reference's own: its halves repeat)."""
  cs = M.rope_cs(l, np.arange(tmax)); h = M.c.HD[l] // 2
  return np.ascontiguousarray(np.concatenate([cs[0][:, :h], cs[1][:, :h]], -1))

class Layer:
  """Layer l of the model `M` (a gemma4_ref.Model) for n rows, alone (the gate): its weights packed from the GGUF, its own
  persistent buffers; L(x, Kc, Vc, pos0, ple) -> the output rows."""
  def __init__(self, M, l, n, tmax):
    c = M.c; self.M, self.l, self.n, self.tmax, self.c = M, l, n, tmax, c
    assert 1 <= n <= 12, "rows mode only (n <= 12)"
    self.K = Kernels(2, 24, c.EPS, real=n); self.R = 24
    self.W = LayerW.from_gguf(M, l); self.cs = dev(rope_cs(M, l, tmax))
    self.posb = self.K.buf("posb", 16, dtypes.int32); self.pidx = self.K.buf("pidx", 1, dtypes.int32)
    self.xin, self.xout = self.K.rows_buf("xio0", c.H), self.K.rows_buf("xio1", c.H)
    self.pl = self.K.buf("pl", PLE_ROWS * max(c.PLE, 1), dtypes.float32)
  def __call__(self, x, Kc, Vc, pos0, ple=None):
    """x: fp32 rows [R, H] (host); Kc / Vc: device caches; ple: fp32 rows [n, PLE] (host) -> the output rows (device [R, H])."""
    poke(self.posb, np.array([pos0] + [0] * 15, np.int32)); poke(self.xin, np.asarray(x, np.float32).reshape(self.R, self.c.H))
    if ple is not None: pz = np.zeros((PLE_ROWS, self.c.PLE), np.float32); pz[:len(ple)] = ple; poke(self.pl, pz)
    return layer_body(self.K, self.W, self.n, self.xin, self.xout, Kc, Vc, self.cs, self.posb, self.tmax, self.pl, self.pidx)

PROMPT = [2, 105, 2364, 107, 3689, 563, 506, 5279, 529, 23613, 236881, 25685, 528, 886, 13315, 236761, 106, 107, 105, 4368, 107]

def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--gguf", default=GR.DEFAULT)
  ap.add_argument("--gate", default="4", help="layers: a list (0,4,14), a range (0-34) or all")
  ap.add_argument("--n", type=int, default=4, help="rows: the prompt's last n tokens (the cache prefix holds the ones before)")
  ap.add_argument("--tmax", type=int, default=64); ap.add_argument("--reps", type=int, default=2)
  ap.add_argument("--tol", type=float, default=1e-3, help="PASS: the relative error of the layer's delta below this")
  ap.add_argument("ids", nargs="*", type=int); a = ap.parse_args()
  M = GR.Model(a.gguf); c = M.c; n = a.n; ids = a.ids or PROMPT; p0 = len(ids) - n
  if a.gate == "all": layers = list(range(c.NL))
  elif "-" in a.gate: lo, hi = map(int, a.gate.split("-")); layers = list(range(lo, hi + 1))
  else: layers = [int(v) for v in a.gate.split(",")]
  # the reference over the whole prompt: every layer's input / output rows and every cache (causal: the first p0 rows of a cache
  # are what the layer had written before these n rows)
  t0 = time.perf_counter(); kv = {}; pos = np.arange(len(ids)); x = M.embed(ids)
  pl = M.ple_inputs(ids, x) if c.PLE else None; xs = [x]
  for j in range(max(layers) + 1): x = M.layer(j, x, pos, kv, None if pl is None else pl[:, j]); xs.append(x)
  print(f"numpy reference, {len(ids)} tokens through layer {max(layers)}: {time.perf_counter() - t0:.1f} s", flush=True)
  worst = 0.0
  for l in layers:
    src = c.KV_SRC[l]; L = Layer(M, l, n, a.tmax); hd, nkv = c.HD[l], c.NKV[l]
    Kc0 = np.zeros((a.tmax, nkv, hd), np.float32); Vc0 = np.zeros_like(Kc0); m = p0 if src == l else len(ids)
    Kc0[:m], Vc0[:m] = kv[src][0][:m], kv[src][1][:m]
    xin = np.zeros((L.R, c.H), np.float32); xin[:n] = xs[l][p0:]; want = xs[l + 1][p0:]
    pin = None if not c.PLE else pl[p0:, l]
    for rep in range(a.reps):
      Kc, Vc = dev(Kc0.reshape(-1)), dev(Vc0.reshape(-1)); t0 = time.perf_counter()
      y = L(xin, Kc, Vc, p0, pin); got = OA.host_invalidate(y).numpy().reshape(L.R, c.H)[:n]; dt = time.perf_counter() - t0
      d = got - want; r = float(np.linalg.norm(d) / np.linalg.norm(want - xs[l][p0:]))
      msg = (f"layer {l:2d} ({'local' if c.SWA[l] else 'global'}, HD {hd}, kv of {src:2d}) {n} rows at {p0}: {dt:.2f} s | max|d| {np.abs(d).max():.3g} "
             f"(max|y| {np.abs(want).max():.3g}) | rel err of the layer's delta {r:.3g} | finite {bool(np.isfinite(got).all())}")
      if src == l:
        kg = OA.host_invalidate(Kc).numpy().reshape(a.tmax, nkv, hd)[p0:p0 + n]; kw = kv[l][0][p0:p0 + n]
        msg += f" | new K rows rel {np.linalg.norm(kg - kw) / np.linalg.norm(kw):.3g}"
      if rep == a.reps - 1: worst = max(worst, r if np.isfinite(r) else np.inf); print(msg, flush=True)
    del L
  print(f"GATE {'PASS' if worst < a.tol else 'FAIL'}: worst rel err of a layer's delta {worst:.3g} over layers {layers[0]}..{layers[-1]} ({len(layers)})", flush=True)

if __name__ == "__main__": main()

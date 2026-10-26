#!/usr/bin/env python3
"""One Qwen3-30B-A3B decoder layer on the NPU in the GEMM's rows mode (n <= 12 rows), from qwen3moe_pack.py's cache, against the
numpy reference (qwen3moe_ref.py).

    attention:  rms_a (x w_attn) -> q | k|v GEMM (Q4_K; q | k + a Q6_K v GEMM and kv_rows in the 24 layers whose v is Q6_K)
                -> gattn_kvd (k RMSNorm x w_kn + RoPE, v raw, into the caches) -> gattn_part / gattn_comb (q RMSNorm x w_qn / sqrt(HD)
                + RoPE, causal, online softmax) -> o GEMM -> resid (x1 = x + o)
    experts:    gemv (the router on rms(x1), w_ffn folded into it) -> the host's softmax / top 8 / renormalisation -> the slots and
                weights poked -> gate and up as 8-head gemm_gs(bsel) GEMMs into one tile buffer [gate x 8 | up x 8] -> swiglu_a
                (6144 columns: the 8 experts' A, 768 each) -> down as an 8-head gemm_gs(bsel) GEMM -> moe_comb (x1 + sum_h w_h C_h)

The kernels are qwen3.8-27b's (rms_a, unpack, resid, swiglu_a, gemv) and gemma4's (gattn_*, with v left unnormed), plus moe_comb /
kv_rows (qwen3moe_kernels.py). The routing is the host's: one sync a layer (`route`); the rest is fixed buffers, as gemma4's.

    python3 qwen3moe_npu.py --gate 0,6 --n 1      (the prompt's last n tokens through those layers; the cache prefix from the reference)
"""
import argparse, ctypes, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "tinygrad"))))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "qwen3.8-27b"), os.path.join(HERE, "..", "gemma4")]
from tinygrad import Tensor, Device, dtypes                              # noqa: E402
from zy import OA                                                         # noqa: E402
from qwen38_kernels import Kernels                                        # noqa: E402
import gemma4_kernels as GK                                               # noqa: E402
import qwen3moe_kernels as MK                                             # noqa: E402
import qwen3moe_ref as QR                                                 # noqa: E402

DEV, NS, KS = "ZHOUYI", 3, 32
CACHE = os.environ.get("QWEN3MOE_NPU", "/mnt/ssd/qwen3-30b-a3b-npu")
CS = lambda K: K.nrb * NS * 192                                           # floats of one group's C tiles

def dev(a, dt=None): return Tensor(np.ascontiguousarray(a), device=DEV, **({} if dt is None else {"dtype": dt})).realize()
def poke(t, a):
  """The host array `a` into the start of the realised device tensor `t` (after any job in flight has finished)."""
  Device[DEV].synchronize(); a = np.ascontiguousarray(a); b = t.uop.base.buffer; b.ensure_allocated(); raw = b._buf
  assert a.nbytes <= b.nbytes and raw.va, (a.nbytes, b.nbytes); ctypes.memmove(raw.va, a.ctypes.data, a.nbytes)

def read_meta(l, cache=CACHE):
  z = np.load(os.path.join(cache, f"L{l}_small.npz"))
  meta = {str(k): tuple(int(v) for v in m if v != -1) for k, m in zip(z["meta_names"], z["meta"])}
  return meta, {k: z[k] for k in z.files if not k.startswith("meta")}

class LayerW:
  """Layer l's weights on the device: `st` name -> the stream tensor, `meta` name -> (q8, N, npad, K, S, goff...), `sm` the small
  fp32 tensors."""
  def __init__(self, l, cache=CACHE):
    self.l = l; self.meta, small = read_meta(l, cache); self.vq6 = "v" in self.meta
    self.st = {k: dev(np.fromfile(os.path.join(cache, f"L{l}_{k}.bin"), np.uint8)) for k in self.meta}
    self.sm = {k: dev(v.astype(np.float32)) for k, v in small.items()}

def gemm(K, W, k, a, n, out, heads=1, a_stride=0, c_stride=0, c_off=0, bsel=None):
  q8, _, npad, Kd, _ = W.meta[k][:5]; ng = npad // (16 * NS)
  return OA.gemm_gs(a, W.st[k], ks=KS, ns=NS, nrb=K.nrb, nslices=Kd // (4 * KS), ngroups=ng, heads=heads, a_stride=a_stride, c_stride=c_stride,
                    c_off=c_off, b8=True, bscale=True, q8=q8, piece=K.nrb if K.compact or K.nrb <= 20 else 0, rows=n if K.compact else 0, bsel=bsel, out=out).realize()   # (full layout: the row blocks in pieces whose C fits the 64 KiB window)

class Bufs:
  """The layer's scratch beyond Kernels': the tile buffers of the fused / expert GEMMs, the slots and the routing weights."""
  def __init__(self, K, c):
    cs = CS(K); self.ng_gu, self.ng_dn = c.FF // 48, -(-c.H // 48)
    self.ct_attn = K.buf("ct_attn", (-(-c.NH * c.HD // 48) + 2 * -(-c.NKV * c.HD // 48)) * cs, dtypes.float32)
    self.ct_v = K.buf("ct_v", -(-c.NKV * c.HD // 48) * cs, dtypes.float32)
    self.ct_o = K.buf("ct_o", self.ng_dn * cs, dtypes.float32)
    self.ct_gu = K.buf("ct_gu", 2 * c.TOPK * self.ng_gu * cs, dtypes.float32)
    self.ct_dn = K.buf("ct_dn", c.TOPK * self.ng_dn * cs, dtypes.float32)
    self.sel_gu, self.sel_dn = K.buf("sel_gu", 16, dtypes.int32), K.buf("sel_dn", 16, dtypes.int32)
    self.wt = K.buf("moe_w", 12 * c.TOPK, dtypes.float32)

def desc(K, key, fn):
  if key not in K.bufs: K.bufs[key] = dev(fn())
  return K.bufs[key]

def attn_body(K, B, W, c, n, x, Kc, Vc, cs, posb, core=None):
  """x fp32 rows [R, H] -> x1 = x + attention (Kernels' "x1"). `core(q_rows) -> o_rows`: another attention core (the batched
  prefill's matrix-unit one) in place of gattn_part / gattn_comb."""
  sm, H = W.sm, c.H; QN, KVN = c.NH * c.HD, c.NKV * c.HD; gq = -(-QN // 48)
  a_h = K.rms_a(x, sm["w_in"], H, "a_h")
  gemm(K, W, "attn", a_h, n, B.ct_attn)
  q_rows = K.unpack(B.ct_attn, QN, "q_rows")
  if W.vq6:
    gemm(K, W, "v", a_h, n, B.ct_v)
    kv_rows = K.run(K.key(f"kv_rows|{KVN}|{n}|{gq}", MK.kv_rows_src(KVN, KVN, K.nrb, n, goff1=gq)), "kv_rows", K.rows * 2 * KVN, dtypes.float32, B.ct_attn, B.ct_v)
  else: kv_rows = K.unpack(B.ct_attn, 2 * KVN, "kv_rows", goff=gq)
  K.call(f"gattn_kvd|{c.NKV}|{c.HD}|{n}|{2 * KVN}|nov", GK.gattn_kvd_src(c.NKV, c.HD, c.EPS, n, 2 * KVN, vnorm=False), Kc, Vc, kv_rows, sm["knw"], cs, posb,
         desc(K, f"gattn_kvd_desc|{2 * KVN}|{c.HD}", lambda: GK.gattn_kvd_desc(2 * KVN, c.HD)))
  if core is not None: o_rows = core(q_rows)
  else:
    HG, BT, P, MR = GK.gattn_cfgr(c.NH, c.NKV, c.HD, n); part = K.buf(f"ga_part|{n}", GK.gattn_part_size(c.NH, c.NKV, c.HD, n), dtypes.float32)
    o_rows = K.rows_buf("o_rows", QN)
    K.call(f"gattn_part|{c.NKV}|{c.HD}|{n}|0|0", GK.gattn_part_src(c.NH, c.NKV, c.HD, c.EPS, n, QN), part, q_rows, Kc, Vc, sm["qnw"], cs, posb,
           desc(K, f"gattn_part_desc|{c.NKV}|{c.HD}|{n}", lambda: GK.gattn_part_desc(c.NKV, c.HD, MR, HG, BT)))
    K.call(f"gattn_comb|{c.NKV}|{c.HD}|{n}", GK.gattn_comb_src(c.NH, c.NKV, c.HD, n), o_rows, part, desc(K, f"gattn_comb_desc|{c.NKV}|{c.HD}|{n}", lambda: GK.gattn_comb_desc(c.NH, c.NKV, c.HD, n)))
  gemm(K, W, "o", K.rms_a(o_rows, None, QN, "a_o", norm=False), n, B.ct_o)
  return K.resid(x, B.ct_o, H, "x1")

ROUTER = os.environ.get("QWEN3MOE_ROUTER", "dma")                         # the router's gemv: gemv32d (weights by DMA) or "plain" (gemv32)

def router(K, W, c, x1):
  """The router's logits on the device [R, NE] (rms(x1) @ (W_r w_ffn)^T): gemv32d, the same bits as gemv32 (ROUTER=plain)."""
  if ROUTER == "plain": return K.gemv(x1, W.sm["router"], c.H, c.NE, "rlog", norm=True)
  return K.run(K.key(f"gemv32d|{c.H}|{c.NE}|{K.real}", MK.gemv32d_src(K.real, c.H, c.NE, norm=True, eps=c.EPS)), "rlog", K.rows * c.NE, dtypes.float32,
               x1, W.sm["router"], desc(K, f"gemv32d_desc{c.H}", lambda: MK.gemv32d_desc(c.H))).reshape(K.rows, c.NE)

def route(c, logits, n):
  """Host: logits [n, NE] -> (top [n, TOPK], weights [n, TOPK]) as the reference (softmax, top 8, renormalised)."""
  p = np.exp(logits - logits.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
  top = np.argsort(-p, -1, kind="stable")[:, :c.TOPK]; w = np.take_along_axis(p, top, -1)
  return top, w / w.sum(-1, keepdims=True)

def moe_body(K, B, W, c, n, x1, out):
  """The experts of the slots / weights already poked -> out = x1 + sum_h w_h expert_h (one row: n = 1)."""
  cs, H = CS(K), c.H
  a_f = K.rms_a(x1, W.sm["w_ffn"], H, "a_f")
  gemm(K, W, "gate", a_f, n, B.ct_gu, heads=c.TOPK, c_stride=B.ng_gu * cs, bsel=B.sel_gu)
  gemm(K, W, "up", a_f, n, B.ct_gu, heads=c.TOPK, c_stride=B.ng_gu * cs, c_off=c.TOPK * B.ng_gu * cs, bsel=B.sel_gu)
  a_sw = K.swiglu_a(B.ct_gu, c.TOPK * c.FF, "a_sw")
  gemm(K, W, "down", a_sw, n, B.ct_dn, heads=c.TOPK, a_stride=K.a_size(c.FF), c_stride=B.ng_dn * cs, bsel=B.sel_dn)
  return K.call(f"moe_comb|{H}|{n}|{c.TOPK}", MK.moe_comb_src(H, K.nrb, n, c.TOPK, B.ng_dn * cs), out, x1, B.ct_dn, B.wt)

DEVROUTE = os.environ.get("QWEN3MOE_DEVROUTE", "1") == "1"                # the routing on the device (moe_route), else the host's

def dev_route(K, B, W, c, logits):
  """moe_route: the router's logits -> the slots and weights in B (one row), on the device."""
  S_gu, S_dn = W.meta["gate"][4], W.meta["down"][4]
  return K.call(f"moe_route|{c.NE}|{S_gu}|{S_dn}", MK.moe_route_src(c.NE, c.TOPK, S_gu, S_dn), B.sel_gu, B.sel_dn, B.wt, logits,
                desc(K, f"moe_route_desc|{c.NE}", lambda: MK.moe_route_desc(c.NE, c.TOPK)))

def set_route(B, W, c, top, wt):
  """The slots (expert x its stream's bytes) and the weights of one row's experts into their device buffers."""
  assert top.shape[0] == 1, "one row (decode) for now: a multi-row pass needs the union of its rows' experts"
  poke(B.sel_gu, np.array(list(top[0] * W.meta["gate"][4]) + [0] * (16 - c.TOPK), np.int32))
  poke(B.sel_dn, np.array(list(top[0] * W.meta["down"][4]) + [0] * (16 - c.TOPK), np.int32))
  w = np.zeros(12 * c.TOPK, np.float32); w[:c.TOPK] = wt[0]; poke(B.wt, w)

def rope_cs(M, tmax):
  cs = M.rope_cs(np.arange(tmax)); return np.ascontiguousarray(np.concatenate([cs[0], cs[1]], -1))

QDOT = os.environ.get("QWEN3MOE_QDOT", "1") == "1"                       # decode on the qdot GEMVs (qwen3moe_qdec); 0: gemm_gs rows mode

class Layer:
  """Layer l alone (the gate): its weights from the cache, its own buffers; L(x, Kc, Vc, pos0) -> the output rows (device)."""
  def __init__(self, M, l, n, tmax, cache=CACHE):
    c = M.c; self.c, self.l, self.n = c, l, n; assert n == 1, "the gate runs one row (the routing is per row)"
    self.K = Kernels(2, 24, c.EPS, real=n); self.R = 24; self.B = Bufs(self.K, c)
    if QDOT:
      import qwen3moe_qdec as QD
      self.QD = QD; self.W = QD.QLayerW(l, cache); self.QB = QD.QBufs(self.K, c)
    else: self.W = LayerW(l, cache)
    self.cs = dev(rope_cs(M, tmax)); self.posb = self.K.buf("posb", 16, dtypes.int32)
    self.xin, self.xout = self.K.rows_buf("xio0", c.H), self.K.rows_buf("xio1", c.H)
  def __call__(self, x, Kc, Vc, pos0):
    K, B, W, c, n = self.K, self.B, self.W, self.c, self.n
    poke(self.posb, np.array([pos0] + [0] * 15, np.int32)); poke(self.xin, np.asarray(x, np.float32).reshape(self.R, c.H))
    if QDOT:
      y = self.QD.qlayer(K, self.QB, B.sel_gu, B.sel_dn, B.wt, W.st, W.meta, W.sm, c, self.xin, Kc, Vc, self.cs, self.posb, self.xout, W.d_up)
      lg = OA.host_invalidate(K.bufs["q_rlog"]).numpy()[None, :c.NE]
      dk = "down_q6n" if self.QD.Q6N and "down_q6n" in W.st else "down"    # (sel_dn holds the slots of the down stream the layer reads)
      top = OA.host_invalidate(B.sel_dn).numpy()[None, :c.TOPK] // W.meta[dk][4]; wt = OA.host_invalidate(B.wt).numpy()[None, :c.TOPK]
      self.last_route = (top, wt, lg); return y
    x1 = attn_body(K, B, W, c, n, self.xin, Kc, Vc, self.cs, self.posb)
    rl = router(K, W, c, x1)
    if DEVROUTE:
      dev_route(K, B, W, c, rl); lg = OA.host_invalidate(rl).numpy().reshape(self.R, c.NE)[:n]
      top = OA.host_invalidate(B.sel_gu).numpy()[None, :c.TOPK] // W.meta["gate"][4]; wt = OA.host_invalidate(B.wt).numpy()[None, :c.TOPK]
      assert (OA.host_invalidate(B.sel_dn).numpy()[:c.TOPK] == top[0] * W.meta["down"][4]).all(), "sel_dn disagrees with sel_gu"
    else:
      lg = OA.host_invalidate(rl).numpy().reshape(self.R, c.NE)[:n]; top, wt = route(c, lg, n); set_route(B, W, c, top, wt)
    self.last_route = (top, wt, lg)
    return moe_body(K, B, W, c, n, x1, self.xout)

PROMPT_TEXT = "What is the capital of Portugal? Answer in one sentence."

def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--gguf", default=QR.DEFAULT); ap.add_argument("--cache", default=CACHE)
  ap.add_argument("--gate", default="0", help="layers: a list (0,6) or a range (0-47)"); ap.add_argument("--n", type=int, default=1)
  ap.add_argument("--tmax", type=int, default=64); ap.add_argument("--reps", type=int, default=2)
  ap.add_argument("--tol", type=float, default=5e-3, help="PASS: the relative error of the layer's delta below this")
  ap.add_argument("--tok", default=os.environ.get("QWEN3MOE_TOK", "/mnt/ssd/models/qwen3-30b-a3b-tokenizer")); a = ap.parse_args()
  from tokenizers import Tokenizer
  M = QR.Model(a.gguf); c = M.c; n = a.n
  ids = Tokenizer.from_file(os.path.join(a.tok, "tokenizer.json")).encode(QR.chat_text(PROMPT_TEXT), add_special_tokens=False).ids; p0 = len(ids) - n
  layers = list(range(int(a.gate.split("-")[0]), int(a.gate.split("-")[1]) + 1)) if "-" in a.gate else [int(v) for v in a.gate.split(",")]
  t0 = time.perf_counter(); kv, hid = {}, {}; pos = np.arange(len(ids)); x = M.embed(ids); xs = [x]
  for j in range(max(layers) + 1): x = M.layer(j, x, pos, kv, hid); xs.append(x)
  print(f"numpy reference, {len(ids)} tokens through layer {max(layers)}: {time.perf_counter() - t0:.1f} s", flush=True)
  worst = 0.0
  for l in layers:
    t0 = time.perf_counter(); L = Layer(M, l, n, a.tmax, a.cache); tl = time.perf_counter() - t0
    Kc0 = np.zeros((a.tmax, c.NKV, c.HD), np.float32); Vc0 = np.zeros_like(Kc0); Kc0[:p0], Vc0[:p0] = kv[l][0][:p0], kv[l][1][:p0]
    xin = np.zeros((L.R, c.H), np.float32); xin[:n] = xs[l][p0:]; want = xs[l + 1][p0:]
    rt, rw = hid["route"][l][0][p0:], hid["route"][l][1][p0:]
    for rep in range(a.reps):
      Kc, Vc = dev(Kc0.reshape(-1)), dev(Vc0.reshape(-1)); t0 = time.perf_counter()
      y = L(xin, Kc, Vc, p0); got = OA.host_invalidate(y).numpy().reshape(L.R, c.H)[:n]; dt = time.perf_counter() - t0
      d = got - want; r = float(np.linalg.norm(d) / np.linalg.norm(want - xs[l][p0:]))
      top, wt, _ = L.last_route; same = bool((np.sort(top, -1) == np.sort(rt, -1)).all())
      kg = OA.host_invalidate(Kc).numpy().reshape(a.tmax, c.NKV, c.HD)[p0:p0 + n]; kw = kv[l][0][p0:p0 + n]
      vg = OA.host_invalidate(Vc).numpy().reshape(a.tmax, c.NKV, c.HD)[p0:p0 + n]; vw = kv[l][1][p0:p0 + n]
      msg = (f"layer {l:2d} ({'v Q6_K' if L.W.vq6 else 'v Q4_K'}, down {'Q6_K' if L.W.meta['down'][0] == 4 else 'Q4_K'}) {n} row at {p0}: {dt * 1e3:.0f} ms "
             f"(load {tl:.1f} s) | rel err of the layer's delta {r:.3g} | experts {'= the reference' if same else f'{top[0].tolist()} vs {rt[0].tolist()}'} "
             f"(weights max|d| {np.abs(np.sort(wt, -1) - np.sort(rw, -1)).max():.2g}) | new K rel {np.linalg.norm(kg - kw) / np.linalg.norm(kw):.2g}, V rel {np.linalg.norm(vg - vw) / np.linalg.norm(vw):.2g} | finite {bool(np.isfinite(got).all())}")
      if rep == a.reps - 1: worst = max(worst, r if np.isfinite(r) else np.inf); print(msg, flush=True)
    del L
  print(f"GATE {'PASS' if worst < a.tol else 'FAIL'}: worst rel err of a layer's delta {worst:.3g} over layers {layers} ({len(layers)})", flush=True)

if __name__ == "__main__": main()

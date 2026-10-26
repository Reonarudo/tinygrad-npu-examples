#!/usr/bin/env python3
"""LFM2.5-24B-A2B on the NPU: a layer is one TinyJit graph of csrc kernels on persistent buffers -- the qdot GEMVs over the checkpoint's
own streams (q4m, q6n), the short-conv block's taps and state shift as one row kernel (lfm25_kernels.py; its two-row state per layer on
the device), the attention layers by qwen3.8-27b's fused decode kernel (q / k RMSNorm, full RoPE on 64 dims; its sigmoid gate fed +30),
the routing on the device (seqmix/route.py "sigmoid_bias": sigmoid, the selection bias, top 4, normalised), the two dense layers'
11776-wide FFN on the rows-mode GEMM (beyond the qdot package). Weights in KMD weight buffers, two slots a layer; the positions poked
before the graphs; nothing syncs mid-token.

    python3 lfm25_npu.py --gate 0,2,3           # layers against the numpy reference
    python3 lfm25_npu.py "prompt" --max-new 24  # generation (the prompt through the decode step a token)
"""
import argparse, ctypes, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "tinygrad"))))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "qwen3.8-27b"), os.path.join(HERE, "..", "qwen3-30b-a3b"), os.path.join(HERE, "..", "gemma4"), os.path.join(HERE, "..", "ornith-9b")]
from tinygrad import Tensor, Device, dtypes                              # noqa: E402
from zy import OA                                                         # noqa: E402
from qwen38_kernels import Kernels, ATT_BT                                # noqa: E402
import qwen3moe_qgemv as QG                                               # noqa: E402
import lfm25_ref as LR                                                    # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "seqmix")); sys.path.insert(0, os.path.join(HERE, "..", "gptoss"))
import route as RT, rows as RW                                            # noqa: E402
import lfm25_kernels as LK                                                # noqa: E402

DEV = "ZHOUYI"; CHUNK = 2 << 20; NS, KS = 3, 32
CACHE = os.environ.get("LFM_NPU", "/mnt/ssd/lfm25-npu")
KIND = {5: "q4", 6: "q6n", 7: "q5n", 4: "q6"}
JIT = os.environ.get("LFM_JIT", "1") != "0"


def dev(a, dt=None): return Tensor(np.ascontiguousarray(a), device=DEV, **({} if dt is None else {"dtype": dt})).realize()
def poke(t, a):
  Device[DEV].synchronize(); a = np.ascontiguousarray(a); b = t.uop.base.buffer; b.ensure_allocated(); raw = b._buf
  assert a.nbytes <= b.nbytes and raw.va, (a.nbytes, b.nbytes); ctypes.memmove(raw.va, a.ctypes.data, a.nbytes)
def read_into(path, mv, size, nt=8):
  import threading
  def part(lo, hi):
    with open(path, "rb", buffering=0) as f:
      while lo < hi: lo += os.preadv(f.fileno(), [mv[lo:min(hi, lo + (64 << 20))]], lo)
  bounds = [size * i // nt for i in range(nt + 1)]; th = [threading.Thread(target=part, args=(bounds[i], bounds[i + 1])) for i in range(nt)]
  for t in th: t.start()
  for t in th: t.join()


def gemm(K, st, meta, a, out):
  """A rows-mode gemm_gs over a packed stream (the dense layers' FFN)."""
  code, _, npad, Kd, _ = meta; ng = npad // 48
  return OA.gemm_gs(a, st, ks=KS, ns=NS, nrb=K.nrb, nslices=Kd // (4 * KS), ngroups=ng, b8=True, bscale=True, q8=code, piece=K.nrb, rows=1, out=out).realize()


class Model:
  def __init__(self, gguf=LR.DEFAULT, cache=CACHE, tmax=512, log=print):
    t0 = time.perf_counter(); self.M = LR.Model(gguf); c = self.c = self.M.c; self.tmax = tmax; NL = c.NL
    K = self.K = Kernels(2, 24, c.EPS, real=1); self.Rr = 24; self.NPH, self.NPF = QG.qgemvk_np(c.H), QG.qgemvk_np(c.FF)
    self.metas, self.sms, self.kind = [], [], []
    for l in range(NL):
      z = np.load(os.path.join(cache, f"L{l}_small.npz")); self.metas.append({str(k): tuple(int(v) for v in m) for k, m in zip(z["meta_names"], z["meta"])})
      self.sms.append({k: z[k] for k in z.files if not k.startswith("meta")})
      self.kind.append(("attn" if l in c.ATTN else "conv", "dense" if l < c.NDENSE else "moe") + tuple(self.metas[l][k][0] for k in sorted(self.metas[l])))
    raw = self.raw = Device[DEV].raw; self.packs = {}
    for l in range(NL):
      sizes = {k: os.path.getsize(os.path.join(cache, f"L{l}_{k}.bin")) for k in sorted(self.metas[l])}; offs, o = {}, 0
      for k, sz in sizes.items(): offs[k] = o; o += -(-(sz + 16) // CHUNK) * CHUNK
      self.packs[l] = (sizes, offs, o)
    for l in range(NL): assert self.packs[l][1] == self.packs[self.kind.index(self.kind[l])][1], l
    big = max(p[2] for p in self.packs.values()); self.NSLOT = 2; self.slots = [raw.slot_alloc(big) for _ in range(self.NSLOT)]; self.wbufs = {}
    for l in range(NL):
      sizes, offs, n = self.packs[l]; wid, mm = raw.wbuf_alloc(n)
      for k, sz in sizes.items(): read_into(os.path.join(cache, f"L{l}_{k}.bin"), memoryview(mm)[offs[k]:offs[k] + sz], sz)
      self.wbufs[l] = (wid, mm)
    self.views = {}
    for par in range(self.NSLOT):
      for kd in set(self.kind):
        sizes, offs, _ = self.packs[self.kind.index(kd)]
        self.views[(par, kd)] = {k: Tensor.from_blob(self.slots[par][1] + offs[k], (sz,), dtype=dtypes.uint8, device=DEV) for k, sz in sizes.items()}
    log(f"weights: {NL} layers in weight buffers, {sum(p[2] for p in self.packs.values()) / 1e9:.1f} GB, two {big / 1e6:.0f} MB slots ({time.perf_counter() - t0:.0f} s)", flush=True)
    hold = lambda tag, a: K.hold(tag, dev(np.ascontiguousarray(a, np.float32)))
    self.w_attn = [hold(f"w_attn{l}", self.sms[l]["w_attn"]) for l in range(NL)]; self.w_ffn = [hold(f"w_ffn{l}", self.sms[l]["w_ffn"]) for l in range(NL)]
    self.aidx = {l: i for i, l in enumerate(c.ATTN)}; NA = len(c.ATTN)
    self.QN = hold("qn_all", np.stack([self.sms[l]["qn"] for l in c.ATTN])); self.KN = hold("kn_all", np.stack([self.sms[l]["kn"] for l in c.ATTN]))
    self.conv_w = {l: dev(np.ascontiguousarray(self.sms[l]["conv"].T)) for l in range(NL) if l not in c.ATTN}; self.conv_st = {l: Tensor.zeros(c.LC - 1, c.H, device=DEV).contiguous().realize() for l in range(NL) if l not in c.ATTN}
    self.router = {l: hold(f"router{l}", self.sms[l]["router"]) for l in range(NL) if "router" in self.sms[l]}; self.bias = {l: dev(self.sms[l]["bias"].astype(np.float32)) for l in range(NL) if "bias" in self.sms[l]}
    inv = 1.0 / (c.THETA ** (np.arange(0, c.HD, 2, dtype=np.float64) / c.HD)); f = np.arange(tmax, dtype=np.float64)[:, None] * inv[None]; emb = np.concatenate([f, f], -1)
    self.ROPE = hold("rope_tab", np.concatenate([np.cos(emb), np.sin(emb)], -1).astype(np.float32).ravel())
    ksz = (NA * tmax + ATT_BT) * c.NKV * c.HD; self.Kall, self.Vall = K.buf("K_all", ksz, dtypes.float32), K.buf("V_all", ksz, dtypes.float32)
    self.aix = [K.buf(f"aidx{i}", 2, dtypes.int32) for i in range(NA)]
    z = np.load(os.path.join(cache, "head.npz")); parts = [tuple(int(v) for v in r) for r in z["parts"]]; self.Vn = sum(p[1] for p in parts)
    self.head_all = dev(np.concatenate([np.fromfile(os.path.join(cache, f"head{j}.bin"), np.uint8) for j in range(len(parts))])); self.w_out = dev(z["w_out"].astype(np.float32))
    self.sel0 = dev(np.array([0] + [-1] * 15, np.int32)); self.tops = K.buf("top_head0", 48, dtypes.float32)
    self.pk = Tensor.zeros(c.H // 128 * QG.SL_REC // 4, device=DEV).contiguous().realize()
    self.xio = [K.rows_buf("xio0", c.H), K.rows_buf("xio1", c.H)]; self.x1 = K.rows_buf("x1r", c.H); self.yrow = K.rows_buf("yrow", c.H)
    self.q_rows, self.kv_rows = K.rows_buf("q_rows", c.NH * 2 * c.HD), K.rows_buf("kv_rows", 2 * c.NKV * c.HD)
    self.y_q, self.y_k, self.y_v, self.y_o = (K.buf(t, QG.qgemvk_np(n) + 64, dtypes.float32) for t, n in (("y_q", c.H), ("y_k", c.NKV * c.HD), ("y_v", c.NKV * c.HD), ("y_o", c.H)))
    self.y_bcx = K.buf("y_bcx", QG.qgemvk_np(3 * c.H) + 64, dtypes.float32); self.y_kv = K.buf("y_kv", 2 * QG.qgemvk_np(c.NKV * c.HD) + 64, dtypes.float32); self.y_gu = K.buf("y_gu", 2 * c.TOPK * self.NPF, dtypes.float32); self.y_dn = K.buf("y_dn", c.TOPK * self.NPH, dtypes.float32)
    self.wt = K.buf("moe_w", 16, dtypes.float32); self.sel_gu, self.sel_dn = K.buf("sel_gu", 16, dtypes.int32), K.buf("sel_dn", 16, dtypes.int32)
    self.jits = {}; self.jd = Tensor.zeros(16, device=DEV).contiguous().realize()
    cs = K.nrb * NS * 192; self.ct_dg, self.ct_du = K.buf("ct_dg", -(-c.FFD // 48) * cs, dtypes.float32), K.buf("ct_du", -(-c.FFD // 48) * cs, dtypes.float32); self.ct_dd = K.buf("ct_dd", -(-c.H // 48) * cs, dtypes.float32)
    self.dh = K.rows_buf("dh", c.FFD)
    log(f"ready in {time.perf_counter() - t0:.0f} s", flush=True)

  def qgemv(self, kd, N, Kd, y, w, sel, xs, nw, heads=1, act="norm", gu_np=0, yoff=0, top=False, nbuf=3):
    K = self.K; key = f"qgemv|{kd}|{N}|{Kd}|{heads}|{act}|{gu_np}|{yoff}|{top}|{nbuf}"
    src = QG.qgemvk_src(kd, N, Kd, heads=heads, act=act, gu_np=gu_np, yoff=yoff, top=top, eps=self.c.EPS, nbuf=nbuf)
    return K.call(key, src, y, self.pk, w, sel, K.desc(f"q_desc|{kd}|{Kd}", lambda: QG.qgemvk_desc(kd, Kd)), y, xs, nw)

  def whole(self, l, _):
    """The layer as kernels on persistent buffers (one TinyJit graph; a dummy input): xio[l % 2] -> xio[(l + 1) % 2]."""
    c, K = self.c, self.K; W = self.views[(l % self.NSLOT, self.kind[l])]; meta = self.metas[l]; out = self.xio[(l + 1) % 2]; H = c.H; x = self.xio[l % 2]
    if l in c.ATTN:
      i = self.aidx[l]; KVN = c.NKV * c.HD
      self.qgemv(KIND[meta["q"][0]], H, H, self.y_q, W["q"], self.sel0, x, self.w_attn[l])
      self.qgemv(KIND[meta["k"][0]], KVN, H, self.y_kv, W["k"], self.sel0, x, self.w_attn[l]); self.qgemv(KIND[meta["v"][0]], KVN, H, self.y_kv, W["v"], self.sel0, x, self.w_attn[l], yoff=KVN)
      K.call(f"qrows_gate|{c.NH}|{c.HD}", RW.qrows_gate_src(c.NH, c.HD), self.q_rows, self.y_q)                   # [q | gate = 30] rows (row 0)
      oc = K.attn_dec2(self.q_rows, self.y_kv, self.QN, self.KN, self.ROPE, self.Kall, self.Vall, self.aix[i], c.NH, c.NKV, c.HD, self.tmax, c.HD)
      self.qgemv(KIND[meta["o"][0]], H, H, self.y_o, W["o"], self.sel0, oc, oc, act="plain")
    else:                                                                 # the short conv: b | c | g rows, u = b g, the taps over (state, u), y = c v
      self.qgemv(KIND[meta["cin"][0]], 3 * H, H, self.y_bcx, W["cin"], self.sel0, x, self.w_attn[l])
      yr = K.call(f"lfm_conv|{H}|{c.LC}", LK.lfm_conv_src(H, c.LC), self.yrow, self.y_bcx, self.conv_st[l], self.conv_w[l])
      self.qgemv(KIND[meta["cout"][0]], H, H, self.y_o, W["cout"], self.sel0, yr, yr, act="plain")
    x1 = K.call(f"rowadd|{H}", RW.rowadd_src(H), self.x1, x, self.y_o)
    if l < c.NDENSE:                                                      # the dense FFN on the rows-mode GEMM (K 11776)
      a_f = K.rms_a(x1, self.w_ffn[l], H, "a_f"); gemm(K, W["dgate"], meta["dgate"], a_f, self.ct_dg); gemm(K, W["dup"], meta["dup"], a_f, self.ct_du)
      g_, u_ = K.unpack(self.ct_dg, c.FFD, "dg_rows"), K.unpack(self.ct_du, c.FFD, "du_rows"); dh = K.call(f"swiglu_rows|{c.FFD}", LK.swiglu_rows_src(c.FFD), self.dh, g_, u_)
      gemm(K, W["ddown"], meta["ddown"], K.rms_a(dh, None, c.FFD, "a_dd", norm=False), self.ct_dd)
      return K.resid(x1, self.ct_dd, H, f"xio{(l + 1) % 2}")
    rl = K.gemv(x1, self.router[l], H, c.NE, "rlog", norm=True)
    K.call(f"route|sigmoid_bias|{c.NE}|{c.TOPK}|{meta['gate'][4]}|{meta['down'][4]}", RT.route_src("sigmoid_bias", c.NE, c.TOPK, meta["gate"][4], meta["down"][4]), self.sel_gu, self.sel_dn, self.wt, rl, self.bias[l], rl)
    self.qgemv(KIND[meta["gate"][0]], c.FF, H, self.y_gu, W["gate"], self.sel_gu, x1, self.w_ffn[l], heads=c.TOPK)
    self.qgemv(KIND[meta["up"][0]], c.FF, H, self.y_gu, W["up"], self.sel_gu, x1, self.w_ffn[l], heads=c.TOPK, yoff=c.TOPK * self.NPF)
    self.qgemv(KIND[meta["down"][0]], H, c.FF, self.y_dn, W["down"], self.sel_dn, self.y_gu, x1, heads=c.TOPK, act="swiglu", gu_np=self.NPF, nbuf=2)   # K 1536 + the swiglu package: 2 buffers fit
    return K.call(f"moe_comb2|{H}|{self.NPH}|{c.TOPK}", QG.moe_comb2_src(H, self.NPH, c.TOPK, nt=4), out, x1, self.y_dn, self.wt, K.desc(f"moe_comb2_desc|{H}|{self.NPH}|{c.TOPK}", lambda: QG.moe_comb2_desc(H, self.NPH, c.TOPK)))

  def layer(self, l, x, pos):
    try: self.raw.slot_map(self.slots[l % self.NSLOT][0], self.wbufs[l][0], 0, self.packs[l][2], drain=False)
    except TypeError: self.raw.slot_map(self.slots[l % self.NSLOT][0], self.wbufs[l][0], 0, self.packs[l][2])
    if l in self.c.ATTN and not getattr(self, "_pos_set", False): i = self.aidx[l]; poke(self.aix[i], np.array([i, pos], np.int32))
    if JIT:
      if l not in self.jits:
        import functools
        from tinygrad.engine.jit import TinyJit
        self.jits[l] = TinyJit(functools.partial(self.whole, l))
      return self.jits[l](self.jd)
    return self.whole(l, None)

  def step(self, tok, pos):
    c, K = self.c, self.K; x0 = np.zeros((self.Rr, c.H), np.float32); x0[0] = self.M.embed([tok])[0]; poke(self.xio[0], x0); x = self.xio[0]
    for i in range(len(c.ATTN)): poke(self.aix[i], np.array([i, pos], np.int32))
    self._pos_set = True
    for l in range(c.NL): x = self.layer(l, x, pos)
    self._pos_set = False
    self.qgemv("q6", self.Vn, c.H, self.tops, self.head_all, self.sel0, x, self.w_out, top=True, nbuf=2)
    return int(OA.host_invalidate(K.head_reduce((0,))).numpy()[0])

  def generate(self, ids, n_new, stop=(LR.IM_END,), on_token=None):
    t0 = time.perf_counter(); nxt = None
    for p, t in enumerate(ids): nxt = self.step(t, p)
    t_pre = time.perf_counter() - t0; out = [nxt]; pos = len(ids); t1 = time.perf_counter()
    if on_token: on_token(nxt)
    while len(out) < n_new and out[-1] not in stop:
      out.append(self.step(out[-1], pos)); pos += 1
      if on_token: on_token(out[-1])
    return out, t_pre, time.perf_counter() - t1


def gate(m, layers, ids, tol):
  M, c, K = m.M, m.c, m.K; p0 = len(ids) - 1; kv, st, hid = {}, {}, {}; pos = np.arange(len(ids)); x = M.embed(ids); xs = [x]
  for j in range(max(layers) + 1): x = M.layer(j, x, pos, kv, st, hid); xs.append(x)
  worst = 0.0
  for l in layers:
    if l in c.ATTN:
      i = m.aidx[l]; kk = np.zeros((len(c.ATTN) * m.tmax + ATT_BT, c.NKV, c.HD), np.float32); vv = np.zeros_like(kk)
      kk[i * m.tmax:i * m.tmax + p0], vv[i * m.tmax:i * m.tmax + p0] = kv[l][0][:p0], kv[l][1][:p0]; poke(m.Kall, kk.reshape(-1)); poke(m.Vall, vv.reshape(-1))
    else:                                                                 # the conv state before the last token: the reference's state minus the last u
      m.conv_st[l].assign(dev(M_conv_state(M, l, xs[l][:p0], st))).realize()
    xin = np.zeros((m.Rr, c.H), np.float32); xin[0] = xs[l][p0]; poke(m.xio[l % 2], xin); t0 = time.perf_counter()
    y = m.layer(l, m.xio[l % 2], p0); got = OA.host_invalidate(y).numpy().reshape(m.Rr, c.H)[0]; dt = time.perf_counter() - t0; want = xs[l + 1][p0]
    if l not in c.ATTN and os.environ.get("LFM_DEBUG"):                   # the conv path piece by piece against the reference
      p_ = f"blk.{l}."; h = LR.rms(xs[l][p0:p0 + 1], M.mat(p_ + "attn_norm.weight"), c.EPS); bcx = (h @ M.mat(p_ + "shortconv.in_proj.weight").T)[0]
      dbcx = OA.host_invalidate(m.y_bcx).numpy()[:3 * c.H]; stc = {}; yref = M.conv(l, h, {l: M_conv_state(M, l, xs[l][:p0], stc)})[0]
      x1d = OA.host_invalidate(m.x1).numpy().reshape(m.Rr, c.H)[0]; x1r = xs[l][p0] + yref
      rl_ = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
      print(f"    in_proj row rel err {rl_(dbcx, bcx):.3g} (b {rl_(dbcx[:c.H], bcx[:c.H]):.2g}, c {rl_(dbcx[c.H:2 * c.H], bcx[c.H:2 * c.H]):.2g}, g {rl_(dbcx[2 * c.H:], bcx[2 * c.H:]):.2g}) | x1 (after the conv block) rel err of its delta {rl_(x1d - xs[l][p0], yref):.3g}", flush=True)
    r = float(np.linalg.norm(got - want) / np.linalg.norm(want - xs[l][p0])); worst = max(worst, r); msg = ""
    if l >= c.NDENSE:
      top = OA.host_invalidate(m.sel_gu).numpy()[:c.TOPK] // m.metas[l]["gate"][4]; w = OA.host_invalidate(m.wt).numpy()[:c.TOPK]; rt, rw = hid["route"][l]; same = bool((np.sort(top) == np.sort(rt[p0])).all())
      msg = f" | experts {'= the reference' if same else f'{sorted(top.tolist())} vs {sorted(rt[p0].tolist())}'} (weights max|d| {np.abs(np.sort(w) - np.sort(rw[p0])).max():.2g})"
    print(f"layer {l:2d} ({m.kind[l][0]}, {m.kind[l][1]}) row at {p0}: {dt * 1e3:.0f} ms | rel err of the layer's delta {r:.3g}{msg} | finite {bool(np.isfinite(got).all())}", flush=True)
  print(f"GATE {'PASS' if worst < tol else 'FAIL'}: worst {worst:.3g} over {layers}", flush=True)


def M_conv_state(M, l, x_prefix, st):
  """The conv state (the last 2 u rows) after the prefix rows x_prefix of layer l's input: recomputed from the reference's weights."""
  c, p = M.c, f"blk.{l}."; h = LR.rms(x_prefix, M.mat(p + "attn_norm.weight"), c.EPS); bcx = h @ M.mat(p + "shortconv.in_proj.weight").T; H = c.H
  u = bcx[:, :H] * bcx[:, 2 * H:]; ext = np.concatenate([np.zeros((c.LC - 1, H), np.float32), u]); return ext[-(c.LC - 1):].astype(np.float32)


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("prompt", nargs="?", default="What is the capital of Portugal? Answer in one sentence.")
  ap.add_argument("--gate", default=""); ap.add_argument("--max-new", type=int, default=24); ap.add_argument("--tmax", type=int, default=512); ap.add_argument("--cache", default=CACHE); ap.add_argument("--tol", type=float, default=5e-3); a = ap.parse_args()
  from tokenizers import Tokenizer
  tk = Tokenizer.from_file(os.path.join(LR.TOK, "tokenizer.json")); ids = tk.encode(LR.chat_text(a.prompt), add_special_tokens=False).ids
  m = Model(cache=a.cache, tmax=a.tmax); print(f"{len(ids)} prompt tokens", flush=True)
  if a.gate: gate(m, [int(v) for v in a.gate.split(",")], ids, a.tol); return
  out, t_pre, t_dec = m.generate(ids, a.max_new, on_token=lambda t: print(tk.decode([t]), end="", flush=True))
  print(f"\n\nids: {out}\nprefill {len(ids)} tokens in {t_pre:.1f} s ({len(ids) / t_pre:.2f} tok/s, one at a time); decode {len(out) - 1} tokens in {t_dec:.1f} s ({(len(out) - 1) / max(t_dec, 1e-9):.2f} tok/s)")


if __name__ == "__main__": main()

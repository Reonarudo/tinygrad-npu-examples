#!/usr/bin/env python3
"""Ring-mini-2.0 (16B-A1.4B) on the NPU: decode on the qdot GEMVs over the checkpoint's own streams (q5n qkv, q4m o / gate / up, q6n down),
the attention over the cache by qwen3.8-27b's fused decode kernel (q / k RMSNorm, RoPE on 64 of 128 dims; its per-head sigmoid gate fed
+30 = a gate of 1), the routing on the host (sigmoid scores, the selection bias, 8 groups -> 4, top 8, normalised x 2.5), the shared
expert as head 0 of the down rows, moe_comb2 summing them. Weights resident in KMD weight buffers, two window slots mapped a layer.

    python3 ringmini_npu.py --gate 0,1,5            # layers against the numpy reference (one row at the prompt's last position)
    python3 ringmini_npu.py "prompt" --max-new 32   # generation (the prompt through the decode step a token)
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
import ringmini_ref as RR                                                 # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "seqmix")); import route as RT; import rows as RW   # noqa: E402  (the device routing kernel, the row kernels)

DEV = "ZHOUYI"; CHUNK = 2 << 20
CACHE = os.environ.get("RING_NPU", "/mnt/ssd/ringmini-npu")
JIT = os.environ.get("RING_JIT", "1") != "0"
DEVROUTE = os.environ.get("RING_DEVROUTE", "1") != "0"                    # the routing by the device (seqmix/route.py): a layer is one graph, no host sync                             # the two halves of a layer as TinyJit graphs (per slot parity and layer kind)
KIND = {5: "q4", 6: "q6n", 7: "q5n", 4: "q6"}


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


class Model:
  def __init__(self, gguf=RR.DEFAULT, cache=CACHE, tmax=512, log=print):
    t0 = time.perf_counter(); self.M = RR.Model(gguf); c = self.c = self.M.c; self.tmax = tmax; NL = c.NL
    K = self.K = Kernels(2, 24, c.EPS, real=1); self.Rr = 24; self.NPH, self.NPF, self.NPD = QG.qgemvk_np(c.H), QG.qgemvk_np(c.FF), QG.qgemvk_np(c.FFD)
    self.metas, self.sms, self.kind = [], [], []
    for l in range(NL):
      z = np.load(os.path.join(cache, f"L{l}_small.npz")); self.metas.append({str(k): tuple(int(v) for v in m) for k, m in zip(z["meta_names"], z["meta"])})
      self.sms.append({k: z[k] for k in z.files if not k.startswith("meta")})
      self.kind.append(("dense" if l < c.NDENSE else "moe",) + tuple(self.metas[l][k][0] for k in sorted(self.metas[l])))   # the stream codes vary by layer (Q4_K_M's mix)
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
    # the small weights: per-layer device vectors, the attention's q / k norms as stacks for attn_dec2, the RoPE table
    hold = lambda tag, a: K.hold(tag, dev(np.ascontiguousarray(a, np.float32)))
    self.w_attn = [hold(f"w_attn{l}", self.sms[l]["w_attn"]) for l in range(NL)]; self.w_ffn = [hold(f"w_ffn{l}", self.sms[l]["w_ffn"]) for l in range(NL)]
    self.QN = hold("qn_all", np.stack([self.sms[l]["qn"] for l in range(NL)])); self.KN = hold("kn_all", np.stack([self.sms[l]["kn"] for l in range(NL)]))
    self.router = [hold(f"router{l}", self.sms[l]["router"]) if "router" in self.sms[l] else None for l in range(NL)]; self.bias = [self.sms[l].get("bias") for l in range(NL)]
    self.bias_d = [hold(f"bias{l}", self.sms[l]["bias"]) if "bias" in self.sms[l] else None for l in range(NL)]
    inv = 1.0 / (c.THETA ** (np.arange(0, c.ROT, 2, dtype=np.float64) / c.ROT)); f = np.arange(tmax, dtype=np.float64)[:, None] * inv[None]; emb = np.concatenate([f, f], -1)
    self.ROPE = hold("rope_tab", np.concatenate([np.cos(emb), np.sin(emb)], -1).astype(np.float32).ravel())
    ksz = (NL * tmax + ATT_BT) * c.NKV * c.HD; self.Kall, self.Vall = K.buf("K_all", ksz, dtypes.float32), K.buf("V_all", ksz, dtypes.float32)
    self.aix = [K.buf(f"aidx{l}", 2, dtypes.int32) for l in range(NL)]
    # the head: the Q6_K parts as one q6 stream, the top-1 on the device
    z = np.load(os.path.join(cache, "head.npz")); parts = [tuple(int(v) for v in r) for r in z["parts"]]; self.Vn = sum(p[1] for p in parts)
    self.head_all = dev(np.concatenate([np.fromfile(os.path.join(cache, f"head{j}.bin"), np.uint8) for j in range(len(parts))])); self.w_out = dev(z["w_out"].astype(np.float32))
    self.sel0 = dev(np.array([0] + [-1] * 15, np.int32)); self.d_head = dev(QG.qgemvk_desc("q6", c.H)); self.tops = K.buf("top_head0", 48, dtypes.float32)
    self.pk = Tensor.zeros(c.H // 128 * QG.SL_REC // 4, device=DEV).contiguous().realize()
    # the row buffers
    self.xio = [K.rows_buf("xio0", c.H), K.rows_buf("xio1", c.H)]; self.x1 = K.rows_buf("x1r", c.H)
    self.q_rows, self.kv_rows = K.rows_buf("q_rows", c.NH * 2 * c.HD), K.rows_buf("kv_rows", 2 * c.NKV * c.HD)
    self.y_q, self.y_kv, self.y_o = K.buf("y_q", self.NPH + 64, dtypes.float32), K.buf("y_kv", QG.qgemvk_np(2 * c.NKV * c.HD) + 64, dtypes.float32), K.buf("y_o", self.NPH + 64, dtypes.float32)
    self.y_gu, self.y_sgu, self.y_dgu = K.buf("y_gu", 2 * c.TOPK * self.NPF, dtypes.float32), K.buf("y_sgu", 2 * self.NPF, dtypes.float32), K.buf("y_dgu", 2 * self.NPD, dtypes.float32)
    self.y_dn = K.buf("y_dn", (c.TOPK + 1) * self.NPH, dtypes.float32); self.wt = K.buf("moe_w", 16, dtypes.float32)
    self.sel_gu, self.sel_dn = K.buf("sel_gu", 16, dtypes.int32), K.buf("sel_dn", 16, dtypes.int32)
    self.gate30 = Tensor.full((self.Rr, c.NH, c.HD), 30.0, device=DEV).contiguous().realize()
    self.dh = K.rows_buf("dh", c.FFD); self.pk_d = Tensor.zeros(c.FFD // 128 * QG.SL_REC // 4, device=DEV).contiguous().realize(); self.jits = {}; self.jd = Tensor.zeros(16, device=DEV).contiguous().realize()
    log(f"ready in {time.perf_counter() - t0:.0f} s", flush=True)

  def qgemv(self, kd, N, Kd, y, w, sel, xs, nw, heads=1, act="norm", gu_np=0, yoff=0, split=0, y2=None, top=False, nbuf=3, pkg=None):
    K = self.K; key = f"qgemv|{kd}|{N}|{Kd}|{heads}|{act}|{gu_np}|{yoff}|{split}|{top}|{nbuf}"
    src = QG.qgemvk_src(kd, N, Kd, heads=heads, act=act, gu_np=gu_np, yoff=yoff, split=split, top=top, eps=self.c.EPS, nbuf=nbuf)
    return K.call(key, src, y, pkg if pkg is not None else self.pk, w, sel, K.desc(f"q_desc|{kd}|{Kd}", lambda: QG.qgemvk_desc(kd, Kd)), y2 if y2 is not None else y, xs, nw)

  def route_lg(self, l, lg):
    """The host's routing from the router's logits row (as the reference): (top, weights)."""
    c = self.c; lg = lg.reshape(1, -1)
    s = 1.0 / (1.0 + np.exp(-lg)); sel = s + self.bias[l][None]; gsz = c.NE // c.NG
    grp = np.sort(sel.reshape(1, c.NG, gsz), -1)[..., -2:].sum(-1); keep = np.argsort(-grp, -1, kind="stable")[:, :c.TG]
    mask = np.full((1, c.NG), -np.inf); np.put_along_axis(mask, keep, 0.0, -1); sel = (sel.reshape(1, c.NG, gsz) + mask[..., None]).reshape(1, c.NE)
    top = np.argsort(-sel, -1, kind="stable")[0, :c.TOPK]; w = s[0, top]
    if c.NORMW: w = w / max(w.sum(), 6.103515625e-5)
    return top, w * c.SCALE

  def attn_half(self, l, _):
    """xio[l % 2] (persistent rows) -> x1 (persistent) and the router's logits (a persistent buffer; dense layers: x1 only). The graph's
    only input is a dummy: a hand-written kernel must not read a JIT input buffer (its replay would read the capture-time one)."""
    c, K = self.c, self.K; W = self.views[(l % self.NSLOT, self.kind[l])]; meta = self.metas[l]; QN_, KVN = c.NH * c.HD, 2 * c.NKV * c.HD; x = self.xio[l % 2]
    self.qgemv(KIND[meta["qkv"][0]], QN_ + KVN, c.H, self.y_q, W["qkv"], self.sel0, x, self.w_attn[l], split=QN_, y2=self.y_kv)
    K.call(f"qrows_gate|{c.NH}|{c.HD}", RW.qrows_gate_src(c.NH, c.HD), self.q_rows, self.y_q)                   # [q | gate = 30] rows (row 0)
    oc = K.attn_dec2(self.q_rows, self.y_kv, self.QN, self.KN, self.ROPE, self.Kall, self.Vall, self.aix[l], c.NH, c.NKV, c.HD, self.tmax, c.ROT)   # the kv row straight from the GEMV's y2
    self.qgemv(KIND[meta["o"][0]], c.H, QN_, self.y_o, W["o"], self.sel0, oc, oc, act="plain")
    x1 = K.call(f"rowadd|{c.H}", RW.rowadd_src(c.H), self.x1, x, self.y_o)
    return K.gemv(x1, self.router[l], c.H, c.NE, "rlog", norm=True) if self.kind[l][0] == "moe" else x1

  def ffn_half(self, l, _):
    """The slots / weights poked -> xio[(l + 1) % 2] = x1 + the experts (+ the shared expert) or the dense FFN."""
    c, K = self.c, self.K; W = self.views[(l % self.NSLOT, self.kind[l])]; meta = self.metas[l]; x1 = self.x1; out = self.xio[(l + 1) % 2]
    if self.kind[l][0] == "dense":
      self.qgemv(KIND[meta["dgate"][0]], c.FFD, c.H, self.y_dgu, W["dgate"], self.sel0, x1, self.w_ffn[l]); self.qgemv(KIND[meta["dup"][0]], c.FFD, c.H, self.y_dgu, W["dup"], self.sel0, x1, self.w_ffn[l], yoff=self.NPD)
      K.call(f"qact4m|{c.FFD}|swiglu|{self.NPD}", QG.qact4m_src(c.FFD, heads=1, mode="swiglu", xs=self.NPD), self.pk_d, self.y_dgu, self.y_dgu)   # silu(g) u from the GEMV's rows
      self.qgemv(KIND[meta["ddown"][0]], c.H, c.FFD, self.y_dn, W["ddown"], self.sel0, self.y_dgu, self.y_dgu, act="pkg", pkg=self.pk_d, nbuf=2); nh = 1
    else:
      self.qgemv(KIND[meta["gate"][0]], c.FF, c.H, self.y_gu, W["gate"], self.sel_gu, x1, self.w_ffn[l], heads=c.TOPK)
      self.qgemv(KIND[meta["up"][0]], c.FF, c.H, self.y_gu, W["up"], self.sel_gu, x1, self.w_ffn[l], heads=c.TOPK, yoff=c.TOPK * self.NPF)
      self.qgemv(KIND[meta["sgate"][0]], c.FFS, c.H, self.y_sgu, W["sgate"], self.sel0, x1, self.w_ffn[l]); self.qgemv(KIND[meta["sup"][0]], c.FFS, c.H, self.y_sgu, W["sup"], self.sel0, x1, self.w_ffn[l], yoff=self.NPF)
      self.qgemv(KIND[meta["down"][0]], c.H, c.FF, self.y_dn, W["down"], self.sel_dn, self.y_gu, x1, heads=c.TOPK, act="swiglu", gu_np=self.NPF, yoff=self.NPH)
      self.qgemv(KIND[meta["sdn"][0]], c.H, c.FFS, self.y_dn, W["sdn"], self.sel0, self.y_sgu, x1, act="swiglu", gu_np=self.NPF); nh = c.TOPK + 1
    return K.call(f"moe_comb2|{c.H}|{self.NPH}|{nh}", QG.moe_comb2_src(c.H, self.NPH, nh, nt=4), out, x1, self.y_dn, self.wt, K.desc(f"moe_comb2_desc|{c.H}|{self.NPH}|{nh}", lambda: QG.moe_comb2_desc(c.H, self.NPH, nh)))

  def whole(self, l, _):
    """The layer as one graph: the attention half, the routing on the device (the slots / weights into sel_gu / sel_dn / wt), the FFN half."""
    c, K = self.c, self.K; meta = self.metas[l]; rl = self.attn_half(l, None)
    if self.kind[l][0] == "moe":
      K.call(f"route|group|{c.NE}|{c.TOPK}|{meta['gate'][4]}|{meta['down'][4]}", RT.route_src("group", c.NE, c.TOPK, meta["gate"][4], meta["down"][4], c.NG, c.TG, c.SCALE, woff=1, w0=1.0),
             self.sel_gu, self.sel_dn, self.wt, rl, self.bias_d[l], rl)
    return self.ffn_half(l, None)

  def layer(self, l, x, pos):
    """x (persistent rows, the token in row 0) at `pos` -> the layer's output into xio[(l + 1) % 2]. RING_JIT=1: TinyJit graphs per layer,
    captured on their second run and replayed after; RING_DEVROUTE=1: the layer as one graph (the routing on the device), else two halves
    around the host's routing."""
    c = self.c; meta = self.metas[l]; out = self.xio[(l + 1) % 2]
    try: self.raw.slot_map(self.slots[l % self.NSLOT][0], self.wbufs[l][0], 0, self.packs[l][2], drain=False)   # no wait: the job in flight reads the other slot
    except TypeError: self.raw.slot_map(self.slots[l % self.NSLOT][0], self.wbufs[l][0], 0, self.packs[l][2])
    if not getattr(self, "_pos_set", False): poke(self.aix[l], np.array([l, pos], np.int32))                      # (the step pokes every layer's up front: no mid-step sync)
    key = l                                                                # a graph per layer: its own small weights, position buffer and slot parity
    if DEVROUTE:
      if self.kind[l][0] == "dense" and not getattr(self, "_pos_set", False): poke(self.wt, np.array([1.0] + [0.0] * 15, np.float32))
      if JIT:
        if key not in self.jits:
          import functools
          from tinygrad.engine.jit import TinyJit
          self.jits[key] = TinyJit(functools.partial(self.whole, l))
        y = self.jits[key](self.jd)
      else: y = self.whole(l, None)
      self.last_route = None; return y
    if JIT:
      if key not in self.jits:
        import functools
        from tinygrad.engine.jit import TinyJit
        self.jits[key] = (TinyJit(functools.partial(self.attn_half, l)), TinyJit(functools.partial(self.ffn_half, l)))
      ja, jf = self.jits[key]; rl = ja(self.jd)
    else: rl = self.attn_half(l, None)
    if self.kind[l][0] == "dense": poke(self.wt, np.array([1.0] + [0.0] * 15, np.float32))
    else:
      lg = OA.host_invalidate(rl).numpy().reshape(self.Rr, c.NE)[0].astype(np.float64); top, w = self.route_lg(l, lg); self.last_route = (top, w)
      poke(self.sel_gu, np.array(list(top * meta["gate"][4]) + [-1] * (16 - c.TOPK), np.int32)); poke(self.sel_dn, np.array(list(top * meta["down"][4]) + [-1] * (16 - c.TOPK), np.int32))
      wt = np.zeros(16, np.float32); wt[0] = 1.0; wt[1:c.TOPK + 1] = w; poke(self.wt, wt)
    return jf(self.jd) if JIT else self.ffn_half(l, None)

  def step(self, tok, pos):
    c, K = self.c, self.K; x0 = np.zeros((self.Rr, c.H), np.float32); x0[0] = self.M.embed([tok])[0]; poke(self.xio[0], x0); x = self.xio[0]
    for l in range(c.NL): poke(self.aix[l], np.array([l, pos], np.int32))                                          # every layer's (index, position) before the graphs run
    if DEVROUTE: poke(self.wt, np.array([1.0] + [0.0] * 15, np.float32))
    self._pos_set = True
    for l in range(c.NL): x = self.layer(l, x, pos)
    self._pos_set = False
    self.qgemv("q6", self.Vn, c.H, self.tops, self.head_all, self.sel0, x, self.w_out, top=True, nbuf=2)
    return int(OA.host_invalidate(K.head_reduce((0,))).numpy()[0])

  def generate(self, ids, n_new, stop=(RR.EOS,), on_token=None):
    t0 = time.perf_counter(); nxt = None
    for p, t in enumerate(ids): nxt = self.step(t, p)
    t_pre = time.perf_counter() - t0; out = [nxt]; pos = len(ids); t1 = time.perf_counter()
    if on_token: on_token(nxt)
    while len(out) < n_new and out[-1] not in stop:
      out.append(self.step(out[-1], pos)); pos += 1
      if on_token: on_token(out[-1])
    return out, t_pre, time.perf_counter() - t1


def gate(m, layers, ids, tol):
  """The prompt's last token through `layers` one at a time, the prefix's caches from the reference; the layer's delta vs the reference."""
  M, c, K = m.M, m.c, m.K; p0 = len(ids) - 1; kv, hid = {}, {}; pos = np.arange(len(ids)); x = M.embed(ids); xs = [x]
  for j in range(max(layers) + 1): x = M.layer(j, x, pos, kv, hid); xs.append(x)
  worst = 0.0
  for l in layers:
    kk = np.zeros((c.NL * m.tmax + ATT_BT, c.NKV, c.HD), np.float32); vv = np.zeros_like(kk); kk[l * m.tmax:l * m.tmax + p0], vv[l * m.tmax:l * m.tmax + p0] = kv[l][0][:p0], kv[l][1][:p0]
    poke(m.Kall, kk.reshape(-1)); poke(m.Vall, vv.reshape(-1))
    xin = np.zeros((m.Rr, c.H), np.float32); xin[0] = xs[l][p0]; poke(m.xio[l % 2], xin); t0 = time.perf_counter()   # layer l reads xio[l % 2]
    y = m.layer(l, m.xio[l % 2], p0); got = OA.host_invalidate(y).numpy().reshape(m.Rr, c.H)[0]; dt = time.perf_counter() - t0; want = xs[l + 1][p0]
    r = float(np.linalg.norm(got - want) / np.linalg.norm(want - xs[l][p0])); worst = max(worst, r)
    msg = ""
    if m.kind[l][0] == "moe":
      if m.last_route is None:
        sg = OA.host_invalidate(m.sel_gu).numpy()[:c.TOPK] // m.metas[l]["gate"][4]; wv = OA.host_invalidate(m.wt).numpy()[1:c.TOPK + 1]; m.last_route = (sg, wv)
      top, w = m.last_route; rt, rw = hid["route"][l]; same = bool((np.sort(top) == np.sort(rt[p0])).all())
      msg = f" | experts {'= the reference' if same else f'{sorted(top.tolist())} vs {sorted(rt[p0].tolist())}'} (weights max|d| {np.abs(np.sort(w) - np.sort(rw[p0])).max():.2g})"
    print(f"layer {l:2d} ({m.kind[l][0]}) row at {p0}: {dt * 1e3:.0f} ms | rel err of the layer's delta {r:.3g}{msg} | finite {bool(np.isfinite(got).all())}", flush=True)
  print(f"GATE {'PASS' if worst < tol else 'FAIL'}: worst {worst:.3g} over {layers}", flush=True)


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("prompt", nargs="?", default="What is the capital of Portugal? Answer in one sentence.")
  ap.add_argument("--gate", default=""); ap.add_argument("--max-new", type=int, default=24); ap.add_argument("--no-think", action="store_true"); ap.add_argument("--tmax", type=int, default=512)
  ap.add_argument("--cache", default=CACHE); ap.add_argument("--tol", type=float, default=5e-3); a = ap.parse_args()
  from tokenizers import Tokenizer
  tk = Tokenizer.from_file(os.path.join(RR.TOK, "tokenizer.json")); ids = tk.encode(RR.chat_text(a.prompt, not a.no_think), add_special_tokens=False).ids
  m = Model(cache=a.cache, tmax=a.tmax); print(f"{len(ids)} prompt tokens", flush=True)
  if a.gate: gate(m, [int(v) for v in a.gate.split(",")], ids, a.tol); return
  out, t_pre, t_dec = m.generate(ids, a.max_new, on_token=lambda t: print(tk.decode([t]), end="", flush=True))
  print(f"\n\nids: {out}\nprefill {len(ids)} tokens in {t_pre:.1f} s ({len(ids) / t_pre:.2f} tok/s, one at a time); decode {len(out) - 1} tokens in {t_dec:.1f} s ({(len(out) - 1) / max(t_dec, 1e-9):.2f} tok/s)")


if __name__ == "__main__": main()

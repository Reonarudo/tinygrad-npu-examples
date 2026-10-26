#!/usr/bin/env python3
"""Batched prefill for Qwen3-30B-A3B on the NPU: a chunk of up to M = 128 prompt tokens through every layer at once.

  attention   the decode layer's kernels in the GEMM's full layout (12 row blocks, 144 rows: the chunk's tokens, zeros after)
  routing     the host's: the chunk's x1 rows read back, the router in float32 (the reference's), softmax / top 8 / renormalised
  experts     each expert's tokens in slots of 12 rows (SMAX slots, the unused ones empty): rms_gather builds every slot's A from its
              tokens' rows (fp16(x inv w)), gate / up / down run as gemm_gs(bsel) heads, one a slot (rows mode, 12 rows), and
              moe_scatter sums each token's 8 (slot, row) results with its weights onto x1
  the cache   the chunk's K / V rows land at their positions (rows past the chunk's tokens write rows the causal attention never reads)

An expert's weights are read once per slot (~1.7 times a chunk), not once per token. The prompt's last token goes through the decode
step (which also gives the first new token). `qwen3moe_generate.py --prefill batch` (the default) uses it.

    python3 qwen3moe_prefill.py --gate 0-3     (a ~100-token prompt: every layer's rows vs the reference's)
"""
import argparse, functools, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import qwen3moe_npu as QN                                                  # noqa: E402
import qwen3moe_ref as QR                                                  # noqa: E402
from qwen3moe_npu import Kernels, Tensor, Device, OA, dtypes, dev, poke, MK   # noqa: E402

M_CHUNK = int(os.environ.get("QWEN3MOE_CHUNK", "512"))                  # the long prompts' chunk (a prompt up to M_SHORT tokens: chunks of M_SHORT)
M_SHORT = int(os.environ.get("QWEN3MOE_CHUNK_SHORT", "128"))
RB = int(os.environ.get("QWEN3MOE_SLOT_RB", "2" if M_CHUNK >= 256 else "1"))   # an expert slot's row blocks (12 rows each) at M_CHUNK: 1 = rows mode
PROF = os.environ.get("QWEN3MOE_PROF", "0") == "1"
P_CHUNK_TMAX = M_CHUNK                                                     # (the bench: positions up to 3 chunks)
JIT = os.environ.get("QWEN3MOE_JIT", "1") != "0"
DEVROUTE = os.environ.get("QWEN3MOE_PREFILL_DEVROUTE", "1") == "1"         # the routing and the slots on the device: one graph a layer, no read-back                         # each layer's attention + router and its experts (per slot count) as graphs
MMA_ATTN = os.environ.get("QWEN3MOE_MMA_ATTN", "1") == "1"                # the prefill's attention on the matrix unit (chunks from position 0)
S_STEP = 32                                                              # the experts' slot count rounded up to this (a graph per count)
GU_G, H_G = 16, 43                                                         # groups of 48: the 768 expert rows; the 2048 hidden


class Prefill:
  def __init__(self, m, M=M_CHUNK, rb=None):
    c = self.c = m.c; self.m, self.M = m, M
    self.NRB = -(-(M + 1) // 24) * 2; self.ROWS = 12 * self.NRB              # the chunk's rows in whole row-block pairs (+1: a zero row)
    self.RB = rb or (RB if M == M_CHUNK else 2 if M >= 256 else 1)           # an expert slot's row blocks (12 rows each): 1 = rows mode
    self.ER = 12 * self.RB; self.SMAX = c.NE + -(-c.TOPK * M // self.ER)        # slots of ER rows: every expert's last one partial
    self.KP = Kernels(self.NRB, self.ROWS, c.EPS, real=M); self.KE = Kernels(2, 24, c.EPS, real=12) if self.RB == 1 else Kernels(self.RB, self.ER, c.EPS, real=self.ER)
    self.BP = QN.Bufs(self.KP, c); self.posb = self.KP.buf("posb", 16, dtypes.int32)
    self.xb = [self.KP.rows_buf("pxio0", c.H), self.KP.rows_buf("pxio1", c.H)]
    CS = QN.CS(self.KE); S = self.SMAX; self.CSE = CS
    z = lambda n_, dt=dtypes.float32: Tensor.zeros(n_, device=QN.DEV, dtype=dt).contiguous().realize()
    self.inv = z(16 * self.ROWS); self.a_sl = z(S * self.KE.a_size(c.H), dtypes.uint16)
    self.ct_gu = z(2 * S * GU_G * CS); self.ct_dn = z(S * H_G * CS)
    self.sel_gu, self.sel_dn, self.tok = z(S, dtypes.int32), z(S, dtypes.int32), z(self.ER * S, dtypes.int32)
    self.ent, self.wt = z(self.ROWS * c.TOPK, dtypes.int32), z(self.ROWS * c.TOPK)
    self.wrow, self.y = z(self.ER * S), z(self.ER * S * c.H)                 # the slot rows' weights; their weighted down rows
    self.d_sr, self.d_rs = dev(MK.slot_rows_desc(self.RB, c.H)), dev(MK.rows_sum_desc(c.H, 512)); self.d_sw = dev(MK.swiglu_d_desc(self.RB))
    self.router = [QN.read_meta(l, m.cache)[1]["router"].astype(np.float32) for l in range(c.NL)]
    self.wffn = [QN.read_meta(l, m.cache)[1]["w_ffn"].astype(np.float32) for l in range(c.NL)]
    self.rlog = z(self.ROWS * c.NE); self.d_gm = dev(MK.gemv32m_desc(c.H))
    self.jits, self.jd = {}, Tensor.zeros(16, device=QN.DEV).contiguous().realize()
    self.tbs = {}; self.topw, self.nreal, self.ids = z(self.M * 16, dtypes.int32), z(16, dtypes.int32), z(self.M, dtypes.int32)
    self.d_rt, self.d_rsl = dev(MK.route_top_desc(c.NE)), dev(MK.route_slots_desc())
    self.d_inv = dev(MK.inv_rms_desc(c.H)); self.d_gat = dev(MK.rms_gather_desc(c.H)); self.slots_used = []

  def route(self, l, logits, n):
    """Host: the chunk's n rows' router logits -> the slots (expert, up to 12 tokens) -> tok / sel / ent / wt (poked)."""
    c = self.c; top, wt = QN.route(c, logits[:n], n)
    S, ER = self.SMAX, self.ER; tok = np.full(ER * S, -1, np.int32); sg = np.zeros(S, np.int32); sd = np.zeros(S, np.int32)
    ent = np.zeros(self.ROWS * c.TOPK, np.int32); w = np.zeros(self.ROWS * c.TOPK, np.float32); wr = np.zeros(ER * S, np.float32); s = 0
    meta = self.m.metas[l]; Sg, Sd = meta["gate"][4], meta["down"][4]
    for e in np.unique(top):
      ts, ks = np.nonzero(top == e)
      for i0 in range(0, len(ts), ER):
        tt, kk = ts[i0:i0 + ER], ks[i0:i0 + ER]
        tok[ER * s:ER * s + len(tt)] = tt; sg[s], sd[s] = e * Sg, e * Sd
        ent[tt * c.TOPK + kk] = ER * s + np.arange(len(tt)); w[tt * c.TOPK + kk] = wt[tt, kk]; wr[ER * s:ER * s + len(tt)] = wt[tt, kk]; s += 1
    assert s <= S, (s, S)
    st = S_STEP if self.RB == 1 else 16
    self.slots_used.append(s); self.s_run = min(S, -(-s // st) * st)
    for b_, a_ in ((self.tok, tok), (self.sel_gu, sg), (self.sel_dn, sd), (self.ent, ent), (self.wt, w), (self.wrow, wr)): poke(b_, a_)
    return top, wt

  def tick(self, k):
    if PROF:
      Device[QN.DEV].synchronize(); t = time.perf_counter(); self.prof[k] = self.prof.get(k, 0.0) + t - self._t; self._t = t
  def tables(self, l):
    """Layer l's slot tables on the device (route_slots writes them; one set a layer: no buffer a kernel reads through its cache is
    rewritten by another layer's job)."""
    if l not in self.tbs:
      c, S, ER = self.c, self.SMAX, self.ER; z = lambda n_, dt=dtypes.float32: Tensor.zeros(n_, device=QN.DEV, dtype=dt).contiguous().realize()
      t = type("Tables", (), {})(); t.tok, t.wrow, t.sel_gu, t.sel_dn = z(ER * S, dtypes.int32), z(ER * S), z(S, dtypes.int32), z(S, dtypes.int32)
      t.ent, t.wt, t.nused = z(self.ROWS * c.TOPK, dtypes.int32), z(self.ROWS * c.TOPK), z(16, dtypes.int32); self.tbs[l] = t
    return self.tbs[l]
  def experts(self, l, Wl, x1, out, S=None, tb=None):
    """The slots' experts (the first S slots; tb: the device's tables, which skip the unused ones -- else the host's, poked):
    out = x1 + sum over each token's 8 (slot, row) results."""
    c, KP, KE, CS = self.c, self.KP, self.KE, self.CSE; H = c.H; S = S or self.SMAX
    nu = tb.nused if tb is not None else None; tb = tb or self; nx = (nu,) if nu is not None else ()
    KP.call(f"inv_rms|{H}|{self.M}", MK.inv_rms_src(H, self.M, c.EPS), self.inv, x1, self.d_inv)
    KP.call(f"rms_gather|{H}|{S}|{self.RB}|{nu is not None}", MK.rms_gather_src(H, S, self.RB, nused=nu is not None), self.a_sl, x1, self.inv, Wl.sm["w_ffn"], tb.tok, self.d_gat, *nx); self.tick("gather")
    def g(k, a, out_, c_off=0, a_stride=0, sel=None):
      q8, _, npad, Kd, _ = Wl.meta[k][:5]
      OA.gemm_gs(a, Wl.st[k], ks=QN.KS, ns=QN.NS, nrb=KE.nrb, nslices=Kd // (4 * QN.KS), ngroups=npad // 48, heads=S, a_stride=a_stride,
                 c_stride=(npad // 48) * CS, c_off=c_off, b8=True, bscale=True, q8=q8, piece=KE.nrb, rows=12 if self.RB == 1 else 0, bsel=sel, out=out_).realize()
    g("gate", self.a_sl, self.ct_gu, a_stride=KE.a_size(H), sel=tb.sel_gu); self.tick("gate")
    g("up", self.a_sl, self.ct_gu, c_off=S * GU_G * CS, a_stride=KE.a_size(H), sel=tb.sel_gu); self.tick("up")
    if self.RB == 1: a_sw = KE.swiglu_a(self.ct_gu, S * c.FF, f"a_sw{S}")
    else:
      a_sw = KE.buf(f"a_swd{S}", S * KE.a_size(c.FF), dtypes.uint16)
      KP.call(f"swiglu_d|{S}|{self.RB}|{nu is not None}", MK.swiglu_d_src(S, self.RB, c.FF, nused=nu is not None), a_sw, self.ct_gu, self.d_sw, *nx)
    self.tick("swiglu")
    g("down", a_sw, self.ct_dn, a_stride=KE.a_size(c.FF), sel=tb.sel_dn); self.tick("down")
    if self.RB == 1: return KP.call(f"moe_scatter|{H}|{self.M}|{self.RB}", MK.moe_scatter_src(H, self.M, KE.nrb, H_G * CS, c.TOPK, self.ER), out, x1, self.ct_dn, tb.wt, tb.ent)
    KP.call(f"slot_rows|{S}|{self.RB}|{nu is not None}", MK.slot_rows_src(S, self.RB, H_G, H, nused=nu is not None), self.y, self.ct_dn, tb.wrow, self.d_sr, *nx); self.tick("slot_rows")
    return KP.call(f"rows_sum|{self.M}", MK.rows_sum_src(self.M, c.TOPK, H), out, x1, self.y, tb.ent, self.d_rs)

  def chunk(self, ids, p0, hook=None, feed=None):
    """Tokens `ids` (n <= M) at positions p0 .. p0 + n - 1 through the 48 layers (the caches get their rows). hook(l, x_out rows)."""
    c, m, n = self.c, self.m, len(ids); assert n <= self.M and p0 + self.M <= m.tmax, (n, p0, m.tmax)
    poke(self.posb, np.array([p0] + [0] * 15, np.int32))
    if getattr(m, "emb", None) is not None and feed is None:                # the rows from the token ids on the device (embed_rows)
      ii = np.zeros(self.M, np.int32); ii[:n] = ids; poke(self.ids, ii); poke(self.nreal, np.array([n] + [0] * 15, np.int32))
      self.KP.call(f"embed_rows|{self.M}", MK.embed_rows_src(c.H, self.M), self.xb[0], self.ids, self.nreal, m.emb, m.d_emb)
    else: x0 = np.zeros((self.ROWS, c.H), np.float32); x0[:n] = m.M.embed(list(ids)); poke(self.xb[0], x0)
    self.prof = getattr(self, "prof", {}); self._t = time.perf_counter()
    if DEVROUTE and getattr(m, "emb", None) is None: poke(self.nreal, np.array([n] + [0] * 15, np.int32))
    for l in range(c.NL):
      if feed is not None and l in feed:                                     # (the gate: layer l on the reference's input rows)
        xf = np.zeros((self.ROWS, c.H), np.float32); xf[:n] = feed[l]; poke(self.xb[l % 2], xf)
      if DEVROUTE:                 # one graph: attention, router, route_top, route_slots, experts -- nothing read back
        m.layer_w(l, drain=PROF or hook is not None); self.tick("map")
        self.run(("pl", l, p0), functools.partial(self.f_layer, l, p0 + self.M)); self.tick("layer")
        if hook:
          self.slots_used.append(int(OA.host_invalidate(self.tables(l).nused).numpy()[0]))
          hook(l, OA.host_invalidate(self.KP.bufs["x1"]).numpy().reshape(self.ROWS, c.H)[:n], "x1")
          hook(l, OA.host_invalidate(self.topw).numpy().reshape(-1, 16)[:n, :8], "route")
          hook(l, OA.host_invalidate(self.xb[(l + 1) % 2]).numpy().reshape(self.ROWS, c.H)[:n], "out")
        continue
      m.layer_w(l); self.tick("map")
      self.run(("pa", l, p0), functools.partial(self.f_attn, l, p0 + self.M)); self.tick("attention")
      if hook: hook(l, OA.host_invalidate(self.KP.bufs["x1"]).numpy().reshape(self.ROWS, c.H)[:n], "x1")
      lg = OA.host_invalidate(self.rlog).numpy().reshape(self.ROWS, c.NE)
      top, _ = self.route(l, lg, n); self.tick("route")
      if hook: hook(l, top, "route")
      S = self.s_run; self.run(("pe", l, S), functools.partial(self.f_exp, l, S)); self.tick("experts")
      if hook: hook(l, OA.host_invalidate(self.xb[(l + 1) % 2]).numpy().reshape(self.ROWS, c.H)[:n], "out")
    Device[QN.DEV].synchronize()

  def lw(self, l):
    m = self.m; kd = m.kind[l]
    import qwen3moe_generate as QG
    return QG.W(m.views[(l % m.NSLOT, kd)], m.metas[l], m.sms[l], kd)
  def mma_core(self, l, T, q_rows):
    """The matrix-unit attention over keys 0 .. T - 1 (the chunk's T = p0 + M; softmax_a holds a row block's scores of T <= 528):
    q rows -> o rows (Kernels' "o_rows")."""
    c, m, KP = self.c, self.m, self.KP; NH, NKV, HD = c.NH, c.NKV, c.HD; QR = NH * HD; nrb = KP.nrb
    ngq = -(-T // 48); nstrip = 3 * ngq; nsl = -(-T // 128); bf = lambda tag, n_, dt=dtypes.uint16: KP.buf(tag, n_, dt)
    qa, kb, vb = bf("m_qa", NH * nrb * 1536), bf(f"m_kb{T}", NKV * nstrip * 2048), bf(f"m_vb{T}", NKV * 9 * nsl * 2048)
    pa, cts, cto = bf(f"m_pa{T}", NH * nsl * nrb * 1536), bf(f"m_cs{T}", NH * ngq * nrb * 576, dtypes.float32), bf("m_co", NH * 3 * nrb * 576, dtypes.float32)
    o = KP.rows_buf("o_rows", QR); d = lambda k, f: QN.desc(KP, k, f)
    KP.call(f"qrope_a|{nrb}", MK.qrope_a_src(NH, HD, nrb, c.EPS, QR), qa, q_rows, self.lw(l).sm["qnw"], m.cs, self.posb, d("d_qrope", lambda: MK.qrope_a_desc(QR, HD)))
    KP.call(f"k_panels|{nstrip}", MK.k_panels_src(NKV, HD, nstrip), kb, m.Kc[l], d("d_kp", lambda: MK.k_panels_desc(NKV, HD)))
    KP.call(f"v_panels|{nsl}", MK.v_panels_src(NKV, HD, nsl), vb, m.Vc[l], d("d_vp", lambda: MK.v_panels_desc(NKV, HD)))
    OA.gemm_gs(qa, kb, ks=32, ns=3, nrb=nrb, nslices=1, ngroups=ngq, heads=NH, a_stride=nrb * 1536, b_stride=nstrip * 2048, b_head_div=NH // NKV, c_stride=ngq * nrb * 576, out=cts).realize()
    KP.call(f"softmax_a|{nrb}|{ngq}|{nsl}|{T}", MK.softmax_a_src(NH, nrb, ngq, nsl, T), pa, cts, self.posb, d(f"d_sm{ngq}", lambda: MK.softmax_a_desc(ngq, nrb)))
    OA.gemm_gs(pa, vb, ks=32, ns=3, nrb=nrb, nslices=nsl, ngroups=3, heads=NH, a_stride=nsl * nrb * 1536, b_stride=9 * nsl * 2048, b_head_div=NH // NKV, c_stride=3 * nrb * 576, out=cto).realize()
    return KP.call(f"o_rows|{nrb}", MK.o_rows_src(NH, HD, nrb, QR), o, cto, d("d_or", lambda: MK.o_rows_desc(nrb, QR, HD)))
  def attn_dma(self, l, T, x):
    """The chunk's attention, every layout step by DMA (the matrix-unit core): inv_rms + rms_ad -> q|k|v GEMM -> qrope_c (Q from its
    C tiles) and the kv rows (unpack / kv_rows) -> gattn_kvd (the cache) -> k_panels / v_panels -> QK^T -> softmax_a -> PV -> o_a
    (the o projection's A) -> o GEMM -> resid_d. -> x1 (Kernels' "x1")."""
    c, m, KP, B = self.c, self.m, self.KP, self.BP; NH, NKV, HD, H = c.NH, c.NKV, c.HD, c.H; QR, KVN = NH * HD, NKV * HD; nrb = KP.nrb; gq = -(-QR // 48)
    Wl = self.lw(l); sm = Wl.sm; d = lambda k, f: QN.desc(KP, k, f); bf = lambda tag, n_, dt=dtypes.uint16: KP.buf(tag, n_, dt)
    ngq = -(-T // 48); nstrip = 3 * ngq; nsl = -(-T // 128)
    KP.call(f"inv_rms|{H}|{KP.rows}", MK.inv_rms_src(H, KP.rows, c.EPS), self.inv, x, self.d_inv)
    a_h = bf("p_ah", KP.a_size(H)); KP.call(f"rms_ad|{H}|{nrb}", MK.rms_ad_src(H, nrb), a_h, x, self.inv, sm["w_in"], d("d_rmsad", lambda: MK.rms_ad_desc(H)))
    QN.gemm(KP, Wl, "attn", a_h, self.M, B.ct_attn)
    qa = bf("m_qa", NH * nrb * 1536)
    KP.call(f"qrope_c|{nrb}", MK.qrope_c_src(NH, HD, nrb, c.EPS), qa, B.ct_attn, sm["qnw"], m.cs, self.posb, d("d_qrc", lambda: MK.qrope_c_desc(nrb, HD)))
    if Wl.vq6:
      QN.gemm(KP, Wl, "v", a_h, self.M, B.ct_v)
      kv_rows = KP.run(KP.key(f"kv_rows|{KVN}|{self.M}|{gq}", MK.kv_rows_src(KVN, KVN, nrb, self.M, goff1=gq)), "kv_rows", KP.rows * 2 * KVN, dtypes.float32, B.ct_attn, B.ct_v)
    else: kv_rows = KP.unpack(B.ct_attn, 2 * KVN, "kv_rows", goff=gq)
    KP.call(f"gattn_kvd|{NKV}|{HD}|{self.M}|{2 * KVN}|nov", QN.GK.gattn_kvd_src(NKV, HD, c.EPS, self.M, 2 * KVN, vnorm=False), m.Kc[l], m.Vc[l], kv_rows, sm["knw"], m.cs, self.posb,
            d(f"gattn_kvd_desc|{2 * KVN}|{HD}", lambda: QN.GK.gattn_kvd_desc(2 * KVN, HD)))
    kb, vb = bf(f"m_kb{T}", NKV * nstrip * 2048), bf(f"m_vb{T}", NKV * 9 * nsl * 2048)
    pa, cts, cto = bf(f"m_pa{T}", NH * nsl * nrb * 1536), bf(f"m_cs{T}", NH * ngq * nrb * 576, dtypes.float32), bf("m_co", NH * 3 * nrb * 576, dtypes.float32)
    KP.call(f"k_panels|{nstrip}", MK.k_panels_src(NKV, HD, nstrip), kb, m.Kc[l], d("d_kp", lambda: MK.k_panels_desc(NKV, HD)))
    KP.call(f"v_panels|{nsl}", MK.v_panels_src(NKV, HD, nsl), vb, m.Vc[l], d("d_vp", lambda: MK.v_panels_desc(NKV, HD)))
    OA.gemm_gs(qa, kb, ks=32, ns=3, nrb=nrb, nslices=1, ngroups=ngq, heads=NH, a_stride=nrb * 1536, b_stride=nstrip * 2048, b_head_div=NH // NKV, c_stride=ngq * nrb * 576, out=cts).realize()
    KP.call(f"softmax_a|{nrb}|{ngq}|{nsl}|{T}", MK.softmax_a_src(NH, nrb, ngq, nsl, T), pa, cts, self.posb, d(f"d_sm{ngq}", lambda: MK.softmax_a_desc(ngq, nrb)))
    OA.gemm_gs(pa, vb, ks=32, ns=3, nrb=nrb, nslices=nsl, ngroups=3, heads=NH, a_stride=nsl * nrb * 1536, b_stride=9 * nsl * 2048, b_head_div=NH // NKV, c_stride=3 * nrb * 576, out=cto).realize()
    a_o = bf("p_ao", KP.a_size(QR)); KP.call(f"o_a|{nrb}", MK.o_a_src(NH, HD, nrb), a_o, cto, d("d_oa", lambda: MK.o_a_desc(nrb)))
    QN.gemm(KP, Wl, "o", a_o, self.M, B.ct_o)
    x1 = KP.buf("x1", KP.rows * H, dtypes.float32)
    return KP.call(f"resid_d|{H}|{nrb}", MK.resid_d_src(H, nrb), x1, x, B.ct_o, d("d_rsd", lambda: MK.resid_d_desc(H, nrb)))
  def f_attn(self, l, T, _):
    """Layer l's attention on the chunk's rows and the router's logits (rlog: the M rows, padding included); T: the chunk's keys
    (the matrix-unit core when its scores fit, else gattn_part)."""
    c, m = self.c, self.m; Wl = self.lw(l)
    if MMA_ATTN and -(-T // 48) <= 11: x1 = self.attn_dma(l, T, self.xb[l % 2]).reshape(self.KP.rows, c.H)
    else: x1 = QN.attn_body(self.KP, self.BP, Wl, c, self.M, self.xb[l % 2], m.Kc[l], m.Vc[l], m.cs, self.posb)
    return self.KP.call(f"gemv32m|{self.M}|{c.H}|{c.NE}", MK.gemv32m_src(self.M, c.H, c.NE, c.EPS), self.rlog, x1, Wl.sm["router"], self.d_gm)
  def f_layer(self, l, T, _):
    """Layer l whole on the device: attention, the router's logits (gemv32m), route_top, route_slots (layer l's tables), the
    experts over all SMAX slots (the unused ones skipped by their -1 slot)."""
    c = self.c; Wl = self.lw(l); tb = self.tables(l); meta = self.m.metas[l]
    self.f_attn(l, T, None)
    x1 = self.KP.bufs["x1"].reshape(self.ROWS, c.H)
    self.KP.call(f"route_top|{self.M}", MK.route_top_src(self.M, c.NE, c.TOPK), self.topw, self.rlog, self.d_rt)
    self.KP.call(f"route_slots|{self.M}|{meta['gate'][4]}|{meta['down'][4]}", MK.route_slots_src(self.M, c.NE, self.SMAX, self.ER, meta["gate"][4], meta["down"][4], c.TOPK),
                 tb.tok, tb.wrow, tb.sel_gu, tb.sel_dn, tb.ent, tb.wt, tb.nused, self.topw, self.nreal, self.d_rsl)
    return self.experts(l, Wl, x1, self.xb[(l + 1) % 2], self.SMAX, tb)
  def f_exp(self, l, S, _):
    return self.experts(l, self.lw(l), self.KP.bufs["x1"].reshape(self.ROWS, self.c.H), self.xb[(l + 1) % 2], S)
  def run_(self, key, fn):
    if not JIT: return fn(None)
    import qwen3moe_generate as QG
    if key not in self.jits:
      from tinygrad.engine.jit import TinyJit
      self.jits[key] = TinyJit(fn)
    j = self.jits[key]
    if j.cnt < 2:                    # (capture: the prefill's many large graphs share their launch state -- their own clones would not fit
      import tinygrad.runtime.graph.zhouyi as ZG   #  the NPU's window beside the weights; the decode graphs keep theirs)
      own = ZG.OWN; ZG.OWN = os.environ.get("QWEN3MOE_PREFILL_OWN", "0") == "1"
      try:
        with QG.one_job(): return j(self.jd)
      finally: ZG.OWN = own
    return j(self.jd)
  def run(self, *a):
    if len(a) == 2 and isinstance(a[0], tuple): return self.run_(*a)
    ids, p0 = (a + (0,))[:2] if len(a) < 2 else a[:2]
    return self.prefill(ids, p0)
  def prefill(self, ids, p0=0):
    """The prompt `ids` in chunks of M from position p0 (the caller's decode step takes the token after)."""
    for i in range(0, len(ids), self.M): self.chunk(ids[i:i + self.M], p0 + i)


TEXT = ("Explain, in a few paragraphs, how a lighthouse works: the light source, the lens that concentrates it into a beam, the "
        "rotation that makes it flash, and how sailors use the pattern of flashes to tell one lighthouse from another at night.")

def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--gate", default="0-3"); ap.add_argument("--tol", type=float, default=5e-3)
  ap.add_argument("--bench", type=int, default=0, help="time this many 128-token chunks (random ids) instead of the gate"); a = ap.parse_args()
  if a.bench:
    import qwen3moe_generate as QG
    print(f"host: CPUs {QG.pin_cpus()}, CPU-latency request {QG.hold_cpu_latency()} us", flush=True)
    m = QG.Model(tmax=4 * P_CHUNK_TMAX); P = Prefill(m); rng = np.random.default_rng(0)
    for _ in range(3): P.chunk(list(rng.integers(0, 150000, P.M)), 0)       # (the graphs: the second call captures)
    P.prof = {}; P.slots_used = []
    t0 = time.perf_counter()
    for i in range(a.bench): P.chunk(list(rng.integers(0, 150000, P.M)), 0)          # pp-style: every chunk from position 0
    dt = time.perf_counter() - t0; print(f"{a.bench} chunks of {P.M}: {dt / a.bench:.2f} s a chunk -> {P.M * a.bench / dt:.1f} tok/s; slots used {np.mean([int(OA.host_invalidate(t.nused).numpy()[0]) for t in P.tbs.values()]) if P.tbs else np.mean(P.slots_used):.0f} of {P.SMAX}")
    if P.prof: print("per layer (ms): " + ", ".join(f"{k} {v / a.bench / 48 * 1e3:.2f}" for k, v in P.prof.items()))
    return
  from tokenizers import Tokenizer
  import qwen3moe_generate as QG
  ids = Tokenizer.from_file(os.path.join(QG.TOK, "tokenizer.json")).encode(QR.chat_text(TEXT), add_special_tokens=False).ids
  lo, hi = (int(v) for v in a.gate.split("-"))
  m = QG.Model(tmax=1024); P = Prefill(m); print(f"{len(ids)} prompt tokens; {P.SMAX} slots", flush=True)
  t0 = time.perf_counter(); R = m.M; kv, hid = {}, {}; x = R.embed(ids); xs = [x]; pos = np.arange(len(ids))
  x1s = []
  for l in range(hi + 1): x1 = x + R.attn(l, x, pos, kv); x1s.append(x1); x = x1 + R.moe(l, x1, hid); xs.append(x)
  print(f"reference through layer {hi}: {time.perf_counter() - t0:.0f} s", flush=True)
  worst = [0.0]
  def hook(l, y, what):
    if lo <= l <= hi and what == "x1":
      d = y - x1s[l]; print(f"layer {l:2d} attention: rel err of x1 - x {np.linalg.norm(d) / np.linalg.norm(x1s[l] - xs[l]):.3g}, worst row {max(np.linalg.norm(d, axis=1) / np.linalg.norm(x1s[l] - xs[l], axis=1)):.3g} (row {int(np.argmax(np.linalg.norm(d, axis=1)))})", flush=True)
    if lo <= l <= hi and what == "route":
      top = y; ref = hid["route"][l][0]; nd = int((np.sort(top, -1) != np.sort(ref, -1)).any(-1).sum()); print(f"layer {l:2d} routing: {nd} of {len(top)} rows with other experts", flush=True)
    if lo <= l <= hi and what == "out":
      d = y - xs[l + 1]; r = float(np.linalg.norm(d) / np.linalg.norm(xs[l + 1] - xs[l])); worst[0] = max(worst[0], r)
      print(f"layer {l:2d}: rel err of the layer's update over {len(ids)} rows {r:.3g} (worst row {max(np.linalg.norm(d, axis=1) / np.linalg.norm(xs[l + 1] - xs[l], axis=1)):.3g}), slots {P.slots_used[-1]}, finite {bool(np.isfinite(y).all())}", flush=True)
  t0 = time.perf_counter(); P.chunk(ids, 0, hook, feed={l: xs[l] for l in range(lo, hi + 1)}); print(f"chunk (with the per-layer read-backs): {time.perf_counter() - t0:.1f} s", flush=True)
  print(f"GATE {'PASS' if worst[0] < a.tol else 'FAIL'}: worst {worst[0]:.3g} over layers {lo}..{hi}")

if __name__ == "__main__": main()

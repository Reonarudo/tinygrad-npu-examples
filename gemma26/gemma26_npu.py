#!/usr/bin/env python3
"""Gemma 4 26B-A4B on the NPU: a layer is one TinyJit graph of csrc kernels on persistent buffers -- the qdot GEMVs over the checkpoint's
own streams (q8 attention / dense FFN / head, q4m fused gate|up experts, q51 down experts; gemma26_pack.py), the attention by the gemma4
example's DMA kernels (gattn_kvd: k norm + RoPE + the cache, v normed; gattn_part / gattn_comb: q norm + RoPE, the window on local
layers), the post-norm residuals / gelu_tanh gating / the final two-branch post-norm as gemma26_kernels.py row kernels, the routing on
the device (seqmix/route.py "softmax"; the per-expert scale applied in the combine). Global layers: V = K's pre-RoPE norm, HD 512, the
o projection in two K halves (the qdot package holds K <= 4096). The position is poked before the graphs; nothing syncs mid-token.

    python3 gemma26_npu.py --gate 0,5,29         # layers against the numpy reference (gemma26_ref.py)
    python3 gemma26_npu.py "prompt" --max-new 24  # generation (the prompt through the decode step a token)
"""
import argparse, ctypes, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "tinygrad"))))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "qwen3.8-27b"), os.path.join(HERE, "..", "qwen3-30b-a3b"), os.path.join(HERE, "..", "gemma4"), os.path.join(HERE, "..", "ornith-9b"), os.path.join(HERE, "..", "seqmix"), os.path.join(HERE, "..", "gptoss")]
from tinygrad import Tensor, Device, dtypes                              # noqa: E402
from zy import OA                                                         # noqa: E402
from qwen38_kernels import Kernels                                        # noqa: E402
import qwen3moe_qgemv as QG                                               # noqa: E402
import gemma4_kernels as GK, gemma4_npu as GN                             # noqa: E402  (the attention kernels, the RoPE table)
import route as RT, rows as RW                                            # noqa: E402
import gptoss_kernels as GKN                                              # noqa: E402  (rms_pad)
import gemma26_kernels as G26                                             # noqa: E402
import gemma26_ref as GR                                                  # noqa: E402

DEV = "ZHOUYI"; CHUNK = 2 << 20
CACHE = os.environ.get("G26_NPU", os.path.expanduser("~/npu-packs/gemma26-npu"))
KIND = {2: "q8", 5: "q4", 9: "q51"}
JIT = os.environ.get("G26_JIT", "1") != "0"
KMAX = 4096                                                               # the qdot package's largest K (LSRAM)


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
  def __init__(self, gguf=GR.DEFAULT, cache=CACHE, tmax=512, log=print):
    t0 = time.perf_counter(); self.M = GR.Model(gguf, cache_layers=False); c = self.c = self.M.c; self.tmax = tmax; NL = c.NL; H = c.H; NH = c.NH
    K = self.K = Kernels(2, 24, c.EPS, real=1); self.Rr = 24
    self.metas, self.sms, self.kind = [], [], []
    for l in range(NL):
      z = np.load(os.path.join(cache, f"L{l}_small.npz")); self.metas.append({str(k): tuple(int(v) for v in m) for k, m in zip(z["meta_names"], z["meta"])})
      self.sms.append({k: z[k] for k in z.files if not k.startswith("meta")}); self.kind.append((bool(c.SWA[l]),) + tuple((k, self.metas[l][k][0]) for k in sorted(self.metas[l])))
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
    d32 = lambda a: dev(np.ascontiguousarray(a, np.float32))
    self.sm = {k: [d32(self.sms[l][k]) for l in range(NL)] for k in ("w_attn", "qnw", "knw", "w_post_attn", "w_ffn", "w_post_ffw", "w_post1", "w_post2", "w_pre2", "lscale", "escale")}
    self.KPR = -(-H // 512) * 512; self.router_p = [d32(np.pad(self.sms[l]["router"], ((0, 0), (0, self.KPR - H)))) for l in range(NL)]
    self.ones_h = d32(np.ones(H)); self.one = d32(np.ones(1))
    self.Kc = {l: Tensor.zeros(tmax * c.NKV[l] * c.HD[l], device=DEV).contiguous().realize() for l in range(NL)}; self.Vc = {l: Tensor.zeros(tmax * c.NKV[l] * c.HD[l], device=DEV).contiguous().realize() for l in range(NL)}
    self.cs = {sw: dev(GN.rope_cs(self.M, [l for l in range(NL) if bool(c.SWA[l]) == sw][0], tmax)) for sw in (True, False)}
    self.posb = K.buf("posb", 16, dtypes.int32)
    self.NPH = QG.qgemvk_np(H); self.FFD = self.metas[0]["dgate"][1]; self.KPD = self.metas[0]["ddown"][3]; self.FFE = self.metas[0]["dn"][1] and self.metas[0]["gu"][1] // 2; self.KPE = self.metas[0]["dn"][3]; self.NPE = QG.qgemvk_np(self.metas[0]["gu"][1])
    z = np.load(os.path.join(cache, "head.npz")); parts = [tuple(int(v) for v in r) for r in z["parts"]]; self.Vn = sum(p[1] for p in parts)
    self.head_all = dev(np.concatenate([np.fromfile(os.path.join(cache, f"head{j}.bin"), np.uint8) for j in range(len(parts))])); self.w_out = d32(z["w_out"])
    self.sel0 = dev(np.array([0] + [-1] * 15, np.int32)); self.tops = K.buf("top_head0", 48, dtypes.float32)
    mkpk = lambda Kd, heads=1: Tensor.zeros(heads * Kd // 128 * QG.SL_REC // 4, device=DEV).contiguous().realize()
    self.pk, self.pk_o, self.pk_d, self.pk_e = mkpk(H), mkpk(KMAX), mkpk(self.KPD), mkpk(self.KPE, c.NEU)
    self.xio = [K.rows_buf("xio0", H), K.rows_buf("xio1", H)]; self.x1 = K.rows_buf("x1r", H); self.xn = K.rows_buf("xn", H); self.xr = K.rows_buf("xr", self.KPR)
    HDM = max(c.HD); self.y_q = K.buf("y_q", QG.qgemvk_np(NH * HDM) + 64, dtypes.float32); self.y_kv = K.buf("y_kv", 2 * QG.qgemvk_np(max(c.NKV[l] * c.HD[l] for l in range(NL))) + 64, dtypes.float32)
    self.y_o, self.y_o2, self.y_df = (K.buf(t, self.NPH + 64, dtypes.float32) for t in ("y_o", "y_o2", "y_df"))
    self.y_dgu = K.buf("y_dgu", 2 * QG.qgemvk_np(self.FFD) + 64, dtypes.float32); self.hrows_d = K.buf("hrows_d", self.KPD, dtypes.float32)
    self.y_gu = K.buf("y_gu", c.NEU * self.NPE, dtypes.float32); self.hrows_e = K.buf("hrows_e", c.NEU * self.KPE, dtypes.float32); self.y_dn = K.buf("y_dn", c.NEU * self.NPH, dtypes.float32)
    self.wt = K.buf("wt", 16, dtypes.float32); self.sel_gu, self.sel_dn = K.buf("sel_gu", 16, dtypes.int32), K.buf("sel_dn", 16, dtypes.int32)
    self.part = {}
    for sw in (True, False):
      l = [l for l in range(NL) if bool(c.SWA[l]) == sw][0]; self.part[sw] = K.buf(f"ga_part{c.HD[l]}", GK.gattn_part_size(NH, c.NKV[l], c.HD[l], 1), dtypes.float32)
    self.o_rows = {hd: K.rows_buf(f"o_rows{hd}", NH * hd) for hd in set(c.HD)}
    self.jits = {}; self.jd = Tensor.zeros(16, device=DEV).contiguous().realize()
    log(f"ready in {time.perf_counter() - t0:.0f} s", flush=True)

  def qgemv(self, kd, N, Kd, y, w, sel, pkg, heads=1, top=False, nbuf=3, act_heads=1, yoff=0):
    K = self.K; nbuf = min(nbuf, max(1, (32768 - Kd // 128 * QG.SL_REC - 256) // QG.SLB[kd]))
    key = f"qgemv|{kd}|{N}|{Kd}|{heads}|pkg|{top}|{nbuf}|{act_heads}|{yoff}"
    src = QG.qgemvk_src(kd, N, Kd, heads=heads, act="pkg", top=top, eps=self.c.EPS, nbuf=nbuf, act_heads=act_heads, yoff=yoff)
    return K.call(key, src, y, pkg, w, sel, K.desc(f"q_desc|{kd}|{Kd}", lambda: QG.qgemvk_desc(kd, Kd)), y, pkg, pkg)
  def package(self, rows, pkg, Kd, heads=1): return self.K.call(f"qact4m|{Kd}|{heads}", QG.qact4m_src(Kd, heads=heads), pkg, rows, rows)
  def normed(self, x, w, out=None, KP=None):
    """rms(x) w into `out` (xn: width H; xr: zero-padded to KPR for the router's gemv32)."""
    out = self.xn if out is None else out; KP = self.c.H if KP is None else KP
    return self.K.call(f"rms_pad|{self.c.H}|{KP}", GKN.rms_pad_src(self.c.H, KP, self.c.EPS), out, x, w)

  def whole(self, l, _):
    c, K = self.c, self.K; W = self.views[(l % self.NSLOT, self.kind[l])]; meta = self.metas[l]; out = self.xio[(l + 1) % 2]; x = self.xio[l % 2]
    H, NH, hd, nkv, local = c.H, c.NH, c.HD[l], c.NKV[l], bool(c.SWA[l]); sm = {k: v[l] for k, v in self.sm.items()}; cs = self.cs[local]; eps = c.EPS
    # attention: q, k (, v) GEMVs; the cache write with k's norm + RoPE (v normed); the softmax over the cache; o
    xn = self.normed(x, sm["w_attn"]); self.package(xn, self.pk, H)
    self.qgemv("q8", NH * hd, H, self.y_q, W["q"], self.sel0, self.pk); self.qgemv("q8", nkv * hd, H, self.y_kv, W["k"], self.sel0, self.pk)
    if local: self.qgemv("q8", nkv * hd, H, self.y_kv, W["v"], self.sel0, self.pk, yoff=nkv * hd)
    KR = (2 if local else 1) * nkv * hd
    K.call(f"gattn_kvd|{nkv}|{hd}|1|{KR}|{not local}|0", GK.gattn_kvd_src(nkv, hd, eps, 1, KR, vsame=not local, ring=0), self.Kc[l], self.Vc[l], self.y_kv, sm["knw"], cs, self.posb, K.desc(f"gattn_kvd_desc|{KR}|{hd}", lambda: GK.gattn_kvd_desc(KR, hd)))
    HG, BT, P, MR = GK.gattn_cfgr(NH, nkv, hd, 1); win = c.W if local else 0; o_rows = self.o_rows[hd]
    K.call(f"gattn_part|{nkv}|{hd}|1|{win}|0", GK.gattn_part_src(NH, nkv, hd, eps, 1, NH * hd, win, 0), self.part[local], self.y_q, self.Kc[l], self.Vc[l], sm["qnw"], cs, self.posb, K.desc(f"gattn_part_desc|{nkv}|{hd}|1", lambda: GK.gattn_part_desc(nkv, hd, MR, HG, BT)))
    K.call(f"gattn_comb|{nkv}|{hd}|1", GK.gattn_comb_src(NH, nkv, hd, 1), o_rows, self.part[local], K.desc(f"gattn_comb_desc|{nkv}|{hd}|1", lambda: GK.gattn_comb_desc(NH, nkv, hd, 1)))
    if NH * hd <= KMAX:
      self.package(o_rows, self.pk_o, NH * hd); self.qgemv("q8", H, NH * hd, self.y_o, W["o"], self.sel0, self.pk_o)
    else:                                                                 # the o projection in two K halves (streams o0 | o1), summed
      half = NH * hd // 2
      for j, y in ((0, self.y_o), (1, self.y_o2)):
        K.call(f"rowcopy|{half}|{j}", RW.rowslice_src(half, j * half), self.xr, o_rows); self.package(self.xr, self.pk_o, half); self.qgemv("q8", H, half, y, W[f"o{j}"], self.sel0, self.pk_o)
      K.call(f"rowadd|{H}", RW.rowadd_src(H), self.y_o, self.y_o, self.y_o2)
    x1 = K.call(f"g_pnres|{H}", G26.g_pnres_src(H, eps), self.x1, x, self.y_o, sm["w_post_attn"], self.one)
    # the dense FFN (gate | up in one row, gelu_tanh gated, down over K padded)
    xn = self.normed(x1, sm["w_ffn"]); self.package(xn, self.pk, H); NPD = QG.qgemvk_np(self.FFD)
    self.qgemv("q8", self.FFD, H, self.y_dgu, W["dgate"], self.sel0, self.pk); self.qgemv("q8", self.FFD, H, self.y_dgu, W["dup"], self.sel0, self.pk, yoff=NPD)
    K.call(f"geglu_rows|1|{self.FFD}|{NPD}|{self.KPD}", G26.geglu_rows_src(1, self.FFD, NPD, self.KPD), self.hrows_d, self.y_dgu)   # g = y[0:FFD], u = y[NPD:NPD + FFD]: NPD == FFD here
    self.package(self.hrows_d, self.pk_d, self.KPD); self.qgemv("q8", H, self.KPD, self.y_df, W["ddown"], self.sel0, self.pk_d)
    # the experts: the router on rms(x1) (its scale folded), softmax top 8 on the device, gate|up as 8 heads, gelu gating, down, the combine
    xr = self.normed(x1, self.ones_h, self.xr, self.KPR); rl = K.gemv(xr, self.router_p[l], self.KPR, c.NE, "rlog")
    S_gu, S_dn = meta["gu"][4], meta["dn"][4]
    K.call(f"route|softmax|{c.NE}|{c.NEU}|{S_gu}|{S_dn}", RT.route_src("softmax", c.NE, c.NEU, S_gu, S_dn), self.sel_gu, self.sel_dn, self.wt, rl, rl, rl)
    xn = self.normed(x1, sm["w_pre2"]); self.package(xn, self.pk, H)
    self.qgemv(KIND[meta["gu"][0]], meta["gu"][1], H, self.y_gu, W["gu"], self.sel_gu, self.pk, heads=c.NEU)
    K.call(f"geglu_rows|{c.NEU}|{self.FFE}|{self.NPE}|{self.KPE}", G26.geglu_rows_src(c.NEU, self.FFE, self.NPE, self.KPE), self.hrows_e, self.y_gu)
    self.package(self.hrows_e, self.pk_e, self.KPE, heads=c.NEU)
    self.qgemv(KIND[meta["dn"][0]], H, self.KPE, self.y_dn, W["dn"], self.sel_dn, self.pk_e, heads=c.NEU, act_heads=c.NEU)
    return K.call(f"g26_fin|{H}|{c.NEU}|{self.NPH}|{S_dn}", G26.g26_fin_src(H, c.NEU, self.NPH, S_dn, eps), out, x1, self.y_df, self.y_dn, self.wt, self.sel_dn, sm["escale"], sm["w_post1"], sm["w_post2"], sm["w_post_ffw"], sm["lscale"])

  def layer(self, l, x, pos):
    try: self.raw.slot_map(self.slots[l % self.NSLOT][0], self.wbufs[l][0], 0, self.packs[l][2], drain=False)
    except TypeError: self.raw.slot_map(self.slots[l % self.NSLOT][0], self.wbufs[l][0], 0, self.packs[l][2])
    if JIT:
      if l not in self.jits:
        import functools
        from tinygrad.engine.jit import TinyJit
        self.jits[l] = TinyJit(functools.partial(self.whole, l))
      return self.jits[l](self.jd)
    return self.whole(l, None)

  def step(self, tok, pos):
    c, K = self.c, self.K; x0 = np.zeros((self.Rr, c.H), np.float32); x0[0] = self.M.embed([tok])[0]; poke(self.xio[0], x0); poke(self.posb, np.array([pos] + [0] * 15, np.int32)); x = self.xio[0]
    for l in range(c.NL): x = self.layer(l, x, pos)
    xn = self.normed(x, self.w_out); self.package(xn, self.pk, c.H); self.qgemv("q8", self.Vn, c.H, self.tops, self.head_all, self.sel0, self.pk, top=True, nbuf=2)
    return int(OA.host_invalidate(K.head_reduce((0,))).numpy()[0])

  def generate(self, ids, n_new, stop=(1, 106), on_token=None):
    t0 = time.perf_counter(); nxt = None
    for p, t in enumerate(ids): nxt = self.step(t, p)
    t_pre = time.perf_counter() - t0; out = [nxt]; pos = len(ids); t1 = time.perf_counter()
    if on_token: on_token(nxt)
    while len(out) < n_new and out[-1] not in stop:
      out.append(self.step(out[-1], pos)); pos += 1
      if on_token: on_token(out[-1])
    return out, t_pre, time.perf_counter() - t1


def gate(m, layers, ids, tol):
  M, c, K = m.M, m.c, m.K; p0 = len(ids) - 1; kv, hid = {}, {}; M.hid = hid; pos = np.arange(len(ids)); x = M.embed(ids); xs = [x]
  for j in range(max(layers) + 1): x = M.layer(j, x, pos, kv); xs.append(x)
  worst = 0.0
  for l in layers:
    nkv, hd = c.NKV[l], c.HD[l]; kk = np.zeros((m.tmax, nkv, hd), np.float32); vv = np.zeros_like(kk); Kr, Vr, P = kv[l]; kk[:p0], vv[:p0] = Kr[:p0], Vr[:p0]
    m.Kc[l].assign(dev(kk.reshape(-1))).realize(); m.Vc[l].assign(dev(vv.reshape(-1))).realize()
    xin = np.zeros((m.Rr, c.H), np.float32); xin[0] = xs[l][p0]; poke(m.xio[l % 2], xin); poke(m.posb, np.array([p0] + [0] * 15, np.int32)); t0 = time.perf_counter()
    y = m.layer(l, m.xio[l % 2], p0); got = OA.host_invalidate(y).numpy().reshape(m.Rr, c.H)[0]; dt = time.perf_counter() - t0; want = xs[l + 1][p0]
    r = float(np.linalg.norm(got - want) / np.linalg.norm(want - xs[l][p0])); worst = max(worst, r)
    top = OA.host_invalidate(m.sel_dn).numpy()[:c.NEU] // m.metas[l]["dn"][4]; w = OA.host_invalidate(m.wt).numpy()[:c.NEU] * m.sms[l]["escale"][top]
    rt, rw = hid["route"][l]; same = bool((np.sort(top) == np.sort(rt[p0])).all())
    x1d = OA.host_invalidate(m.x1).numpy().reshape(m.Rr, c.H)[0]; kv2 = {l: (Kr[:p0], Vr[:p0], P[:p0])}; ra = "n/a"
    print(f"layer {l:2d} ({'local' if c.SWA[l] else 'global'}) row at {p0}: {dt * 1e3:.0f} ms | rel err of the layer's delta {r:.3g} | experts {'= the reference' if same else f'{sorted(top.tolist())} vs {sorted(rt[p0].tolist())}'} (weights max|d| {np.abs(np.sort(w) - np.sort(rw[p0])).max():.2g}) | finite {bool(np.isfinite(got).all())}", flush=True)
  print(f"GATE {'PASS' if worst < tol else 'FAIL'}: worst {worst:.3g} over {layers}", flush=True)


def main():
  import gemma4_tokenize as GT
  ap = argparse.ArgumentParser(); ap.add_argument("prompt", nargs="?", default="What is the capital of Portugal? Answer in one sentence.")
  ap.add_argument("--gate", default=""); ap.add_argument("--max-new", type=int, default=24); ap.add_argument("--tmax", type=int, default=512); ap.add_argument("--cache", default=CACHE); ap.add_argument("--tol", type=float, default=5e-3); a = ap.parse_args()
  m = Model(cache=a.cache, tmax=a.tmax); tk = GT.Tok(m.M.g.meta); ids = tk.encode(GT.chat_messages([{"role": "user", "content": a.prompt}])); print(f"{len(ids)} prompt tokens", flush=True)
  if a.gate: gate(m, [int(v) for v in a.gate.split(",")], ids, a.tol); return
  out, t_pre, t_dec = m.generate(ids, a.max_new, on_token=lambda t: print(tk.decode([t]), end="", flush=True))
  print(f"\n\nids: {out}\nprefill {len(ids)} tokens in {t_pre:.1f} s ({len(ids) / t_pre:.2f} tok/s, one at a time); decode {len(out) - 1} tokens in {t_dec:.1f} s ({(len(out) - 1) / max(t_dec, 1e-9):.2f} tok/s)")


if __name__ == "__main__": main()

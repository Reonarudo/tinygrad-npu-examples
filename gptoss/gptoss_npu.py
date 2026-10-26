#!/usr/bin/env python3
"""gpt-oss-20b on the NPU: a layer is one TinyJit graph of csrc kernels on persistent buffers -- the qdot GEMVs (the "q8" kind for the
Q8_0 attention linears at K 2944; the MXFP4 experts repacked losslessly into the "q5n" stream at K 3072, see gptoss_pack.py), the
attention over the cache by the library's sinks + sliding-window kernel (seqmix/attn.py: even layers see 128 positions, every head's
sink logit in the softmax denominator), the biases / YaRN RoPE / clamped SwiGLU / weighted combine as gptoss_kernels.py row kernels,
the routing on the device (seqmix/route.py "topk_softmax"). The positions are poked before the graphs run; nothing syncs mid-token.

    python3 gptoss_npu.py --gate 0,1,23         # layers against the numpy reference
    python3 gptoss_npu.py "prompt" --max-new 24  # generation (the prompt through the decode step a token)
"""
import argparse, ctypes, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "tinygrad"))))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "qwen3.8-27b"), os.path.join(HERE, "..", "qwen3-30b-a3b"), os.path.join(HERE, "..", "gemma4"), os.path.join(HERE, "..", "ornith-9b"), os.path.join(HERE, "..", "seqmix")]
from tinygrad import Tensor, Device, dtypes                              # noqa: E402
from zy import OA                                                         # noqa: E402
from qwen38_kernels import Kernels                                        # noqa: E402
import qwen3moe_qgemv as QG                                               # noqa: E402
import attn as SA, route as RT, rows as RW                                 # noqa: E402  (seqmix: the sinks / window attention, the device routing, row kernels)
import gptoss_kernels as GKN                                              # noqa: E402
import gptoss_ref as OR                                                   # noqa: E402

DEV = "ZHOUYI"; CHUNK = 2 << 20
CACHE = os.environ.get("OSS_NPU", "/mnt/ssd/gptoss-npu")
KIND = {2: "q8", 7: "q5n", 8: "mx4"}
JIT = os.environ.get("OSS_JIT", "1") != "0"                               # a layer as one TinyJit graph (the routing on the device)


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
  def __init__(self, gguf=OR.DEFAULT, cache=CACHE, tmax=512, log=print):
    t0 = time.perf_counter(); self.M = OR.Model(gguf); c = self.c = self.M.c; self.tmax = tmax; NL = c.NL; H = c.H
    K = self.K = Kernels(2, 24, c.EPS, real=1); self.Rr = 24
    self.metas, self.sms, self.kind = [], [], []
    for l in range(NL):
      z = np.load(os.path.join(cache, f"L{l}_small.npz")); self.metas.append({str(k): tuple(int(v) for v in m) for k, m in zip(z["meta_names"], z["meta"])})
      self.sms.append({k: z[k] for k in z.files if not k.startswith("meta")}); self.kind.append(tuple(self.metas[l][k][0] for k in sorted(self.metas[l])))
    self.KP = self.metas[0]["q"][3]; self.KPE = self.metas[0]["gate"][3]; self.KPR = max(-(-self.KP // 512) * 512, self.KPE); self.NPH, self.NPF = QG.qgemvk_np(H), QG.qgemvk_np(c.FF)           # K padded (2944); the output pitches
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
    self.w_attn = [d32(self.sms[l]["w_attn"]) for l in range(NL)]; self.w_post = [d32(self.sms[l]["w_post"]) for l in range(NL)]
    self.bq, self.bk, self.bv, self.bo = ([d32(self.sms[l][k]) for l in range(NL)] for k in ("bq", "bk", "bv", "bo"))
    self.sink = d32(np.stack([self.sms[l]["sinks"] for l in range(NL)]))                        # [L][NH]
    cs, sn = self.M.rope_cs(np.arange(tmax)); self.cstab = d32(np.concatenate([cs, sn], 1))       # the YaRN table [tmax][HD]: cos | sin
    self.bg, self.bu, self.bd = ([d32(self.sms[l][k]) for l in range(NL)] for k in ("bg", "bu", "bd"))
    self.router_p = [d32(np.pad(self.sms[l]["router"], ((0, 0), (0, self.KPR - H)))) for l in range(NL)]; self.rbias = [d32(self.sms[l]["rbias"]) for l in range(NL)]
    NKV, HD = c.NKV, c.HD; self.Kl, self.Vl = (Tensor.zeros(NL * tmax * NKV * HD, device=DEV).contiguous().realize() for _ in range(2))
    self.aix = [K.buf(f"aidx{l}", 2, dtypes.int32) for l in range(NL)]
    HG, BT, P = SA.cfg(c.NH, NKV, HD); self.part = K.buf("sa_part", NKV * (c.NH // NKV // HG) * P * SA.block_floats(HG, HD), dtypes.float32); self.d_attn = dev(SA.attn_part_desc(c.NH, NKV, HD))
    z = np.load(os.path.join(cache, "head.npz")); parts = [tuple(int(v) for v in r) for r in z["parts"]]; self.Vn = sum(p[1] for p in parts)
    self.head_all = dev(np.concatenate([np.fromfile(os.path.join(cache, f"head{j}.bin"), np.uint8) for j in range(len(parts))])); self.w_out = d32(z["w_out"])
    self.sel0 = dev(np.array([0] + [-1] * 15, np.int32)); self.tops = K.buf("top_head0", 48, dtypes.float32)
    self.pk = Tensor.zeros(self.KP // 128 * QG.SL_REC // 4, device=DEV).contiguous().realize(); self.pk_o = Tensor.zeros(c.NH * HD // 128 * QG.SL_REC // 4, device=DEV).contiguous().realize()
    self.xio = [K.rows_buf("xio0", H), K.rows_buf("xio1", H)]; self.x1 = K.rows_buf("x1r", H); self.xn = K.rows_buf("xn", self.KPR); self.orow = K.rows_buf("orow", c.NH * HD)
    self.q_f, self.k_f, self.v_f = K.buf("q_f", c.NH * HD, dtypes.float32), K.buf("k_f", NKV * HD, dtypes.float32), K.buf("v_f", NKV * HD, dtypes.float32); self.o_f = K.buf("o_f", c.NH * HD, dtypes.float32)
    self.y_q, self.y_k, self.y_v, self.y_o = (K.buf(t, QG.qgemvk_np(n) + 64, dtypes.float32) for t, n in (("y_q", c.NH * HD), ("y_k", NKV * HD), ("y_v", NKV * HD), ("y_o", H)))
    self.y_g, self.y_u, self.y_dn = (K.buf(t, c.TOPK * n, dtypes.float32) for t, n in (("y_g", self.NPF), ("y_u", self.NPF), ("y_dn", self.NPH)))
    self.hrows = K.buf("hrows", c.TOPK * self.KPE, dtypes.float32); self.pk_e = Tensor.zeros(self.KPE // 128 * QG.SL_REC // 4, device=DEV).contiguous().realize()
    self.pk4 = Tensor.zeros(c.TOPK * self.KPE // 128 * QG.SL_REC // 4, device=DEV).contiguous().realize()
    self.bgu = K.buf("bgu", 2 * c.TOPK * c.FF, dtypes.float32); self.bdw = K.buf("bdw", H, dtypes.float32); self.wt4 = K.buf("wt4", 16, dtypes.float32)
    self.sel_e = K.buf("sel_e", 16, dtypes.int32); self.sel_d = K.buf("sel_d", 16, dtypes.int32); self.jits = {}; self.jd = Tensor.zeros(16, device=DEV).contiguous().realize()
    log(f"ready in {time.perf_counter() - t0:.0f} s", flush=True)

  def qgemv(self, kd, N, Kd, y, w, sel, pkg, heads=1, top=False, nbuf=3, act_heads=1):
    """A GEMV over a prebuilt package (`package`): at K 2944 the in-kernel package producers do not fit LSRAM beside the weight buffers."""
    K = self.K; nbuf = min(nbuf, max(1, (32768 - Kd // 128 * QG.SL_REC - 256) // QG.SLB[kd]))   # the package + nbuf weight chunks must fit LSRAM (o: K 4096 -> 2)
    key = f"qgemv|{kd}|{N}|{Kd}|{heads}|pkg|{top}|{nbuf}|{act_heads}"
    src = QG.qgemvk_src(kd, N, Kd, heads=heads, act="pkg", top=top, eps=self.c.EPS, nbuf=nbuf, act_heads=act_heads)
    return K.call(key, src, y, pkg, w, sel, K.desc(f"q_desc|{kd}|{Kd}", lambda: QG.qgemvk_desc(kd, Kd)), y, pkg, pkg)
  def package(self, rows, pkg, Kd, heads=1):
    """The activation package of `heads` rows (fp32 [heads][Kd] contiguous) into `pkg` (qact4m)."""
    return self.K.call(f"qact4m|{Kd}|{heads}", QG.qact4m_src(Kd, heads=heads), pkg, rows, rows)

  def normed(self, x, w):
    """rms(x) w as the padded row buffer xn (the kernels' K 2944 and the router GEMV's 3072; the padding columns zero)."""
    return self.K.call(f"rms_pad|{self.c.H}|{self.KPR}", GKN.rms_pad_src(self.c.H, self.KPR, self.c.EPS), self.xn, x, w)

  def whole(self, l, _):
    """The layer as kernels on persistent buffers (one TinyJit graph): the position / layer index in aix[l], poked before."""
    c, K = self.c, self.K; W = self.views[(l % self.NSLOT, self.kind[l])]; meta = self.metas[l]; out = self.xio[(l + 1) % 2]; x = self.xio[l % 2]
    H, HD, NKV, NH = c.H, c.HD, c.NKV, c.NH
    xn = self.normed(x, self.w_attn[l]); self.package(xn, self.pk, self.KP)
    for k_, y_ in (("q", self.y_q), ("k", self.y_k), ("v", self.y_v)): self.qgemv(KIND[meta[k_][0]], meta[k_][1], self.KP, y_, W[k_], self.sel0, self.pk)
    K.call(f"qkv_rope|{NH}|{NKV}|{HD}", GKN.qkv_rope_src(NH, NKV, HD), self.q_f, self.k_f, self.v_f, self.y_q, self.y_k, self.y_v, self.bq[l], self.bk[l], self.bv[l], self.cstab, self.aix[l])
    win = c.SWA if l in c.SWA_LAYERS else 0
    K.call(f"attn_part|{NH}|{NKV}|{HD}|{self.tmax}|{win}|s", SA.attn_part_src(NH, NKV, HD, 1.0 / np.sqrt(HD), self.tmax, window=win, sinks=True), self.part, self.q_f, self.Kl, self.Vl, self.k_f, self.v_f, self.sink, self.aix[l], self.d_attn)
    K.call(f"attn_comb|{NH}|{NKV}|{HD}", SA.attn_comb_src(NH, NKV, HD), self.o_f, self.part)
    self.package(self.o_f, self.pk_o, NH * HD); self.qgemv(KIND[meta["o"][0]], H, meta["o"][3], self.y_o, W["o"], self.sel0, self.pk_o)
    x1 = K.call(f"rowadd3|{H}", RW.rowadd3_src(H), self.x1, x, self.y_o, self.bo[l])
    mn = self.normed(x1, self.w_post[l]); self.package(mn, self.pk_e, self.KPE)                 # the experts' K (3072 for q5n's 256-blocks)
    rl = K.gemv(mn, self.router_p[l], self.KPR, c.NE, "rlog")                                    # gemv32: K in 512s
    K.call(f"route|topk_softmax|{c.NE}|{c.TOPK}|{meta['gate'][4]}|{meta['down'][4]}", RT.route_src("topk_softmax", c.NE, c.TOPK, meta["gate"][4], meta["down"][4]), self.sel_e, self.sel_d, self.wt4, rl, self.rbias[l], rl)
    self.qgemv(KIND[meta["gate"][0]], c.FF, self.KPE, self.y_g, W["gate"], self.sel_e, self.pk_e, heads=c.TOPK); self.qgemv(KIND[meta["up"][0]], c.FF, self.KPE, self.y_u, W["up"], self.sel_e, self.pk_e, heads=c.TOPK)
    K.call(f"oai_glu|{c.TOPK}|{c.FF}|{self.NPF}|{self.KPE}|{meta['gate'][4]}", GKN.oai_glu_src(c.TOPK, c.FF, self.NPF, self.KPE, meta["gate"][4]), self.hrows, self.y_g, self.y_u, self.bg[l], self.bu[l], self.sel_e)
    self.package(self.hrows, self.pk4, self.KPE, heads=c.TOPK)
    self.qgemv(KIND[meta["down"][0]], H, self.KPE, self.y_dn, W["down"], self.sel_d, self.pk4, heads=c.TOPK, act_heads=c.TOPK)
    return K.call(f"oss_comb|{c.TOPK}|{H}|{self.NPH}|{meta['down'][4]}", GKN.oss_comb_src(c.TOPK, H, self.NPH, meta["down"][4]), out, x1, self.y_dn, self.wt4, self.bd[l], self.sel_d)

  def layer(self, l, x, pos):
    try: self.raw.slot_map(self.slots[l % self.NSLOT][0], self.wbufs[l][0], 0, self.packs[l][2], drain=False)
    except TypeError: self.raw.slot_map(self.slots[l % self.NSLOT][0], self.wbufs[l][0], 0, self.packs[l][2])
    if not getattr(self, "_pos_set", False): poke(self.aix[l], np.array([l, pos], np.int32))
    if JIT:
      if l not in self.jits:
        import functools
        from tinygrad.engine.jit import TinyJit
        self.jits[l] = TinyJit(functools.partial(self.whole, l))
      return self.jits[l](self.jd)
    return self.whole(l, None)

  def step(self, tok, pos):
    c, K = self.c, self.K; x0 = np.zeros((self.Rr, c.H), np.float32); x0[0] = self.M.embed([tok])[0]; poke(self.xio[0], x0); x = self.xio[0]
    for l in range(c.NL): poke(self.aix[l], np.array([l, pos], np.int32))
    self._pos_set = True
    for l in range(c.NL): x = self.layer(l, x, pos)
    self._pos_set = False
    xn = self.normed(x, self.w_out); self.package(xn, self.pk, self.KP); self.qgemv("q8", self.Vn, self.KP, self.tops, self.head_all, self.sel0, self.pk, top=True, nbuf=2)
    return int(OA.host_invalidate(K.head_reduce((0,))).numpy()[0])

  def generate(self, ids, n_new, stop=(OR.RETURN, OR.END), on_token=None):
    t0 = time.perf_counter(); nxt = None
    for p, t in enumerate(ids): nxt = self.step(t, p)
    t_pre = time.perf_counter() - t0; out = [nxt]; pos = len(ids); t1 = time.perf_counter()
    if on_token: on_token(nxt)
    while len(out) < n_new and out[-1] not in stop:
      out.append(self.step(out[-1], pos)); pos += 1
      if on_token: on_token(out[-1])
    return out, t_pre, time.perf_counter() - t1


def gate(m, layers, ids, tol):
  M, c, K = m.M, m.c, m.K; p0 = len(ids) - 1; kv, hid = {}, {}; pos = np.arange(len(ids)); x = M.embed(ids); xs = [x]
  for j in range(max(layers) + 1): x = M.layer(j, x, pos, kv, hid); xs.append(x)
  worst = 0.0
  for l in layers:
    kk = np.zeros((c.NL, m.tmax, c.NKV, c.HD), np.float32); vv = np.zeros_like(kk); kk[l, :p0], vv[l, :p0] = kv[l][0][:p0], kv[l][1][:p0]
    m.Kl.assign(dev(kk.reshape(-1))).realize(); m.Vl.assign(dev(vv.reshape(-1))).realize()
    xin = np.zeros((m.Rr, c.H), np.float32); xin[0] = xs[l][p0]; poke(m.xio[l % 2], xin); t0 = time.perf_counter()
    y = m.layer(l, m.xio[l % 2], p0); got = OA.host_invalidate(y).numpy().reshape(m.Rr, c.H)[0]; dt = time.perf_counter() - t0; want = xs[l + 1][p0]
    r = float(np.linalg.norm(got - want) / np.linalg.norm(want - xs[l][p0])); worst = max(worst, r)
    top = OA.host_invalidate(m.sel_e).numpy()[:c.TOPK] // m.metas[l]["gate"][4]; w = OA.host_invalidate(m.wt4).numpy()[:c.TOPK]
    rt, rw = hid["route"][l]; same = bool((np.sort(top) == np.sort(rt[p0])).all())
    x1d = OA.host_invalidate(m.x1).numpy().reshape(m.Rr, c.H)[0]; kv2 = {l: (kv[l][0][:p0], kv[l][1][:p0])}; x1r = xs[l][p0] + M.attn(l, xs[l][p0:p0 + 1], pos[p0:p0 + 1], kv2)[0]
    ra = float(np.linalg.norm(x1d - x1r) / np.linalg.norm(x1r - xs[l][p0]))
    print(f"layer {l:2d} ({'swa' if l in c.SWA_LAYERS else 'full'}) row at {p0}: {dt * 1e3:.0f} ms | rel err of the layer's delta {r:.3g} (attention {ra:.3g}) | experts {'= the reference' if same else f'{sorted(top.tolist())} vs {sorted(rt[p0].tolist())}'} (weights max|d| {np.abs(np.sort(w) - np.sort(rw[p0])).max():.2g}) | finite {bool(np.isfinite(got).all())}", flush=True)
  print(f"GATE {'PASS' if worst < tol else 'FAIL'}: worst {worst:.3g} over {layers}", flush=True)


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("prompt", nargs="?", default="What is the capital of Portugal? Answer in one sentence.")
  ap.add_argument("--gate", default=""); ap.add_argument("--max-new", type=int, default=24); ap.add_argument("--tmax", type=int, default=512); ap.add_argument("--cache", default=CACHE); ap.add_argument("--tol", type=float, default=5e-3); a = ap.parse_args()
  from tokenizers import Tokenizer
  tk = Tokenizer.from_file(os.path.join(OR.TOK, "tokenizer.json")); ids = tk.encode(OR.chat_text(a.prompt), add_special_tokens=False).ids
  m = Model(cache=a.cache, tmax=a.tmax); print(f"{len(ids)} prompt tokens", flush=True)
  if a.gate: gate(m, [int(v) for v in a.gate.split(",")], ids, a.tol); return
  out, t_pre, t_dec = m.generate(ids, a.max_new, on_token=lambda t: print(tk.decode([t]), end="", flush=True))
  print(f"\n\nids: {out}\nprefill {len(ids)} tokens in {t_pre:.1f} s ({len(ids) / t_pre:.2f} tok/s, one at a time); decode {len(out) - 1} tokens in {t_dec:.1f} s ({(len(out) - 1) / max(t_dec, 1e-9):.2f} tok/s)")


if __name__ == "__main__": main()

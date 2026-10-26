#!/usr/bin/env python3
"""Gemma 4's MTP drafter (gemma4_mtp.py) on qdot GEMVs: the matrix-unit path spent 2.7 of a draft's 5.7 ms in 68 small kernels around
the 3 ms its 71 MB head costs at the DDR port. Here every linear is a `qwen3moe_qgemv.qgemvk` integer GEMV over a decode-only
Q8_0 stream (`pack_b_group_q8f(qb=32)`: the GGUF's own int8 codes and fp16 scales, nothing requantised) with int8 activations (two
levels), the activation packages built by small producer kernels (`dpkg`: plain, GeGLU, the sandwich norm + residual), and the
attention by gemma4's gattn_part / gattn_comb against the target's caches, unchanged. A draft is 41 kernels (was 69), all C.
A draft only has to agree with the target's greedy choice often; its exactness costs nothing in the output (the verify pass decides).

    python3 gemma4_draft.py [--mtp ...] [--out /mnt/ssd/gemma4-e2b-npu/mtp]       (pack the q8 streams into <out>/q8/)
"""
import argparse, functools, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "qwen3-30b-a3b"))
import gemma4_npu as GN                                                   # noqa: E402
from gemma4_npu import dev, poke, OA, dtypes                              # noqa: E402
import gemma4_kernels as GK                                              # noqa: E402
import gemma4_ref as GR                                                  # noqa: E402
import gemma4_mtp as GM                                                  # noqa: E402
import qwen3moe_qgemv as Q                                               # noqa: E402
from zy import gemm_fp16 as G                                            # noqa: E402

def q8_stream(q, d):
  """Q8_0 codes [N, K] + scales [N, K / 32] -> the q8 qdot stream (all groups; N padded to 48)."""
  ng = -(-q.shape[0] // 48); return np.concatenate([G.pack_b_group_q8f(q, d, g, 3, 32, qb=32) for g in range(ng)]), 48 * ng

def pack(mtp_gguf, out):
  os.makedirs(out, exist_ok=True); D = GR.Model(mtp_gguf, cache_layers=False); c = D.c; t0 = time.perf_counter(); meta = {}
  def lin(name, tensors):
    qd = [D.g.q8(t) for t in tensors]; q, d = np.concatenate([a for a, _ in qd]), np.concatenate([b for _, b in qd])
    st, npad = q8_stream(q, d); GM.save(os.path.join(out, f"{name}.bin"), st); meta[name] = (int(npad), int(q.shape[1]), int(q.shape[0]))
  for l in range(c.NL):
    p = f"blk.{l}."
    lin(f"D{l}_q", [p + "attn_q.weight"]); lin(f"D{l}_o", [p + "attn_output.weight"]); lin(f"D{l}_gu", [p + "ffn_gate.weight", p + "ffn_up.weight"]); lin(f"D{l}_dn", [p + "ffn_down.weight"])
  lin("pre", ["nextn.pre_projection.weight"]); lin("post", ["nextn.post_projection.weight"]); lin("head", ["token_embd.weight"])
  names = sorted(meta); np.savez(os.path.join(out, "meta.npz"), names=np.array(names), meta=np.array([meta[k] for k in names], np.int64))
  print(f"   drafter q8 streams {out}: {sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1e6:.0f} MB in {time.perf_counter() - t0:.0f} s", flush=True)

# ---- the package producers (12 tasks over the K / 128 records). Every input arrives by DMA and every output leaves by DMA: within
# one job the cores' write-back L2 keeps no coherence, so a plain load of a buffer another core rewrote (y_o, x1: every layer) can be stale.
def dpkg_desc(K, np_=0): return Q.V._desc_slots((K * 4,), ((np_ or K) * 4 * (2 if np_ else 1),), ((K + 1) * 4,), (Q.SL_REC,))

def dpkg_src(K, mode, eps=1e-6, np_=0, nt=12):
  """pkg = the activation package (qwen3moe_qgemv's layout) of
       plain      x[K]
       geglu      gelu_tanh(x[k]) x[np_ + k]                   (the gate | up GEMV's row, 2 np_ floats)
       sandwich   rms(x1) wn, with x1 = (x + rms(y) wv[0:K]) wv[K] written to x1 (pnap's formula: the layer scalar on the sum) (every task computes the two norms, task 0 writes x1)
  args: pkg, x, y, wv, wn, x1, desc (dpkg_desc(K, np_)). LSRAM: x (and y, wv, wn) staged at X.., the record built at RB."""
  nsl = K // 128; V8 = lambda e: f"*(__global float8*)({e})"; XB = 0; XN = (2 * np_ if np_ else K) * 4; YB = XN; WV = YB + K * 4; WN = (WV + (K + 1) * 4 + 63) & ~63; RB = (WN + K * 4) if mode == "sandwich" else XN
  assert RB + Q.SL_REC <= 32768 - 64, (mode, K, RB)
  ss = lambda src: f"float8 s8 = BC(0.0f); for (int i = 0; i < {K}; i += 8) {{ float8 v = {V8(f'{src} + i')}; s8 += v * v; }} float8 iv = BC(1.0f / __builtin_sqrtf(hsum8(s8) * {1.0 / K!r}f + {eps!r}f));"
  if mode == "plain": fill, ld = f"DMA_FILL(0, DESC(desc, 0), {XB}, (int)x); DMA_WAIT(0);", lambda i: V8(f"X + k + {8 * i}")
  elif mode == "geglu": fill, ld = f"DMA_FILL(0, DESC(desc, 1), {XB}, (int)x); DMA_WAIT(0);", lambda i: f"gelu8({V8(f'X + k + {8 * i}')}) * {V8(f'X + {np_} + k + {8 * i}')}"
  else:
    fill = f"""DMA_FILL(0, DESC(desc, 0), {XB}, (int)x); DMA_FILL(1, DESC(desc, 0), {YB}, (int)y); DMA_FILL(2, DESC(desc, 2), {WV}, (int)wv); DMA_FILL(3, DESC(desc, 0), {WN}, (int)wn);
  DMA_WAIT(0); DMA_WAIT(1); DMA_WAIT(2); DMA_WAIT(3); __global float* Y = LSF({YB}); __global float* WVp = LSF({WV}); __global float* WNp = LSF({WN});
  {{ {ss("Y")} float8 sc = BC(WVp[{K}]);
    for (int i = 0; i < {K}; i += 8) *(__global float8*)(X + i) = (*(__global float8*)(X + i) + {V8("Y + i")} * iv * {V8("WVp + i")}) * sc; }}
  if (core_id == 0) {{ DMA_DRAIN(0, DESC(desc, 0), {XB}, (int)x1); DMA_WAIT(0); }}
  {ss("X")}"""
    ld = lambda i: f"{V8(f'X + k + {8 * i}')} * {V8(f'WNp + k + {8 * i}')} * iv"
  return Q.V.FULL_H + Q.QH + GEL + f"""
__kernel void dpkg(__global uchar* restrict pkg, __global float* restrict x, __global float* restrict y, __global float* restrict wv,
                   __global float* restrict wn, __global float* restrict x1, __global int* restrict desc, const int core_id) {{
  __global float* X = LSF({XB}); {fill}
  for (int sl = core_id * {nsl} / {nt}; sl < (core_id + 1) * {nsl} / {nt}; sl++) {{
    __global uchar* o = LSB({RB});
    for (int sb = 0; sb < 4; sb++) {{ int k = sl * 128 + sb * 32; qsub({ld(0)}, {ld(1)}, {ld(2)}, {ld(3)}, o, sb); }}
    qfin(o); DMA_DRAIN(1, DESC(desc, 3), {RB}, (int)(pkg + sl * {Q.SL_REC})); DMA_WAIT(1);
  }}
}}"""

QD_FUSE = os.environ.get("GEMMA_QDRAFT_FUSE", "1") == "1"   # the draft as ONE launch: its kernels inline phases meeting at device barriers (qwen3moe_qdec.fuse_src)
GEL = "static inline float8 gelu8(float8 u) { float8 z = BC(1.5957691216057308f) * (u + BC(0.044715f) * u * u * u); return u * VRCP(BC(1.0f) + exp2_f8(z * BC(-1.4426950408889634f))); }\n"
NOUNROLL_DEF = '#define NOUNROLL _Pragma("clang loop unroll(disable)")\n'

class QDrafter:
  """gemma4_mtp.Drafter's interface (step / chain / set_pos) on the qdot kernels, next to a gemma4_generate.Model `T`."""
  def __init__(self, T, mtp_gguf, mcache):
    t0 = time.perf_counter(); self.T = T; tc = T.c; qc = os.path.join(mcache, "q8")
    self.D = GR.Model(mtp_gguf, cache_layers=False); c = self.c = self.D.c; assert c.arch == "gemma4-assistant" and c.H_OUT == tc.H
    if not os.path.exists(os.path.join(qc, "meta.npz")): pack(mtp_gguf, qc)
    z = np.load(os.path.join(qc, "meta.npz")); self.meta = {str(k): tuple(int(v) for v in m) for k, m in zip(z["names"], z["meta"])}
    self.st = T.Wt.blob({k: os.path.join(qc, f"{k}.bin") for k in self.meta})
    self.sm = [{k: dev(v) for k, v in np.load(os.path.join(mcache, f"D{l}_small.npz")).items()} for l in range(c.NL)]
    self.w_out = dev(np.load(os.path.join(mcache, "meta.npz"))["output_norm"]); self.vocab = self.meta["head"][2]
    self.src = {True: tc.KV_SRC[tc.NL - 2] if tc.SWA[tc.NL - 2] else tc.KV_SRC[tc.NL - 1], False: tc.KV_SRC[tc.NL - 1] if not tc.SWA[tc.NL - 1] else tc.KV_SRC[tc.NL - 2]}
    self.K = GN.SharedK(2, 24, c.EPS, real=1); K = self.K; H = c.H
    nb = lambda tag, n: K.buf(tag, n, dtypes.float32); npd = Q.qgemvk_np
    self.din = K.rows_buf("qd_in", 2 * tc.H); self.x = [nb("qd_x0", npd(H) + 64), nb("qd_x1", npd(H) + 64)]; self.x1 = nb("qd_xm", npd(H) + 64)
    self.pk = {k: K.buf(f"qd_pkg{k}", k // 128 * Q.SL_REC // 4, dtypes.float32) for k in {2 * tc.H, H, 2048} | set(c.NH * hd for hd in c.HD)}
    self.q_rows = {hd: nb(f"qd_q{hd}", npd(c.NH * hd) + 64) for hd in set(c.HD)}
    self.o_rows = {hd: K.rows_buf(f"qd_o{hd}", c.NH * hd) for hd in set(c.HD)}
    self.y_o, self.y_gu, self.y_dn = nb("qd_yo", npd(H) + 64), nb("qd_ygu", npd(4096) + 64), nb("qd_ydn", npd(H) + 64)
    self.tops = nb("top_head0", 48); self.h_next = nb("h_next", npd(tc.H) + 64); self.sel0 = K.buf("qd_sel0", 16, dtypes.int32); poke(self.sel0, np.array([0] + [-1] * 15, np.int32))
    self.posd = GN.dev(np.zeros(16, np.int32)); l_loc, l_glo = c.SWA.index(True), c.SWA.index(False)
    self.cs = {s: dev(GN.rope_cs(self.D, l, T.tmax + 1)[1:]) for s, l in ((True, l_loc), (False, l_glo))}
    self.jit = None; self.sqh = np.float32(math.sqrt(tc.H)); self.desc = {}
    print(f"   qdot drafter: {qc} ({sum(os.path.getsize(os.path.join(qc, f + '.bin')) for f in self.meta) / 1e6:.0f} MB of Q8_0 streams), target caches of layers {self.src[True]} / {self.src[False]}, set up in {time.perf_counter() - t0:.1f} s", flush=True)

  def d(self, key, fn):
    if key not in self.desc: self.desc[key] = dev(fn())
    return self.desc[key]
  def gemv(self, name, N, K_, y, pkg, act="pkg", xs=None, nw=None, top=False, nbuf=3):
    if act != "pkg" and K_ >= 2048: nbuf = 2                      # (LSRAM: the producer stages the K floats beside the weight buffers)
    K = self.K; src = Q.qgemvk_src("q8", N, K_, act=act, top=top, eps=self.c.EPS, nbuf=nbuf)
    return K.call(f"qgemv|qd|{name}|{N}|{K_}|{act}|{int(top)}", src, y, pkg, self.st[name], self.sel0, self.d(f"qd_desc|{K_}", lambda: Q.qgemvk_desc("q8", K_)), y, xs if xs is not None else y, nw if nw is not None else y)
  def pkg(self, mode, K_, pkg, x, y=None, wv=None, wn=None, x1=None, np_=0):
    K = self.K; a = lambda t: t if t is not None else x
    return K.call(f"dpkg|{mode}|{K_}|{np_}", dpkg_src(K_, mode, self.c.EPS, np_), pkg, x, a(y), a(wv), a(wn), a(x1), self.d(f"dpkg_desc|{K_}|{np_}", lambda: dpkg_desc(K_, np_)))

  def f_draft(self, _):
    K, c, T = self.K, self.c, self.T; tc = T.c; H = c.H
    self.pkg("plain", 2 * tc.H, self.pk[2 * tc.H], self.din); x = self.x[0]
    self.gemv("pre", H, 2 * tc.H, x, self.pk[2 * tc.H])
    pk_in = None
    for l in range(c.NL):
      sm, hd, nkv = self.sm[l], c.HD[l], c.NKV[l]; s = self.src[c.SWA[l]]; Kc, Vc = T.cache[s]; ring = T.ring[s]; QN_ = c.NH * hd
      q = self.q_rows[hd]
      if pk_in is None: self.gemv(f"D{l}_q", QN_, H, q, self.pk[H], act="norm", xs=x, nw=sm["w_in"])
      else: self.gemv(f"D{l}_q", QN_, H, q, self.pk[H])
      HG, BT, P, MR = GK.gattn_cfgr(c.NH, nkv, hd, 1); win = c.W - 1 if c.SWA[l] else 0
      part = K.buf(f"ga_part|{hd}", GK.gattn_part_size(c.NH, nkv, hd, 1), dtypes.float32)
      K.call(f"gattn_part|d|{c.NH}|{nkv}|{hd}|{win}|{ring}", GK.gattn_part_src(c.NH, nkv, hd, c.EPS, 1, QN_, win, ring), part, q, Kc, Vc, sm["qnw"], self.cs[c.SWA[l]], self.posd,
             GN._desc(K, f"gattn_part_desc|d|{nkv}|{hd}", lambda: GK.gattn_part_desc(nkv, hd, MR, HG, BT)))
      o = self.o_rows[hd]
      K.call(f"gattn_comb|d|{c.NH}|{nkv}|{hd}", GK.gattn_comb_src(c.NH, nkv, hd, 1), o, part, GN._desc(K, f"gattn_comb_desc|d|{nkv}|{hd}", lambda: GK.gattn_comb_desc(c.NH, nkv, hd, 1)))
      self.gemv(f"D{l}_o", H, QN_, self.y_o, self.pk[QN_], act="plain", xs=o)
      self.pkg("sandwich", H, self.pk[H], x, self.y_o, sm["wv_attn"], sm["w_ffn"], self.x1)
      self.gemv(f"D{l}_gu", 2 * c.FF[l], H, self.y_gu, self.pk[H])
      self.pkg("geglu", c.FF[l], self.pk[c.FF[l]], self.y_gu, np_=c.FF[l])
      self.gemv(f"D{l}_dn", H, c.FF[l], self.y_dn, self.pk[c.FF[l]])
      xn = self.x[(l + 1) % 2]; wn = self.sm[l + 1]["w_in"] if l + 1 < c.NL else self.w_out
      self.pkg("sandwich", H, self.pk[H], self.x1, self.y_dn, sm["wv_ffw"], wn, xn); x = xn; pk_in = self.pk[H]
    self.gemv("head", self.vocab, H, self.tops, pk_in, top=True)
    import qwen38_kernels as QK
    ids = K.buf("ids_head", 16, dtypes.int32); hdesc = self.d("head_reduce_desc1", lambda: QK.head_reduce_desc(1))
    K.call("head_reduce|1|(0,)", QK.head_reduce_src(1, (0,)), ids, self.tops, hdesc)
    self.gemv("post", tc.H, H, self.h_next, pk_in)
    return ids

  def f_draft_fused(self, _):
    """f_draft's kernels as one 12-task launch (qwen3moe_qdec.fuse_src: inline phases, a device barrier between them), plus a one-task
    no-op so the graph is a chain job, not a lone program's."""
    import qwen3moe_qdec as QD, qwen38_kernels as QK
    K = self.K
    if not hasattr(self, "fprog"):
      rec = []; orig = K.call
      K.call = lambda name, src, out, *ins: (rec.append((name, src, (out,) + ins)), out)[1]
      try: ids = self.f_draft(None)
      finally: K.call = orig
      eps = self.c.EPS; nr = {hd: GK._norm_rope(hd, eps) for hd in set(self.c.HD)}; hd0 = min(nr); phases = []
      for name, src, args in rec:                       # one norm_rope a head size (the global layer's differs): renamed in its phases
        for hd, piece in nr.items():
          if hd != hd0 and piece in src: src = src.replace(piece, "").replace("norm_rope(", f"norm_rope{hd}(")
        phases.append((name, src, args))
      known = [Q.V.FULL_H, QK.TILE, GK.EXP, Q.QH, GEL, NOUNROLL_DEF, nr[hd0]] + [nr[hd].replace("norm_rope(", f"norm_rope{hd}(") for hd in nr if hd != hd0]
      src, uniq = QD.fuse_src(phases, name="qdraft_f", known=known)
      self.fprog = OA.register_csrc("qdraft_f", src, ntasks=QD.BAR_NT); self.fargs = uniq; self.fids = ids
      self.fbar = K.buf("qd_bar", 16 * (QD.BAR_NT + 1), dtypes.int32); self.fbstat = K.buf("qd_bstat", 16 * QD.BAR_NT, dtypes.int32); self.fbdesc = dev(QD.bar_desc())
      self.ftail = OA.register_csrc("qtail", "__kernel void qtail(__global int* restrict s, const int core_id) { }", ntasks=1)
      print(f"   qdot drafter fused: {len(rec)} phases, {len(uniq)} buffers", flush=True)
    OA.csrc_call(self.fprog, self.fargs[0], *self.fargs[1:], self.fbar, self.fbdesc, self.fbstat).realize()
    OA.csrc_call(self.ftail, self.fbstat).realize()
    return self.fids

  def step(self, tok, h):
    T = self.T; poke(self.din, np.concatenate([T.M.embed([tok])[0], np.asarray(h, np.float32).reshape(-1)])[None])
    from tinygrad.engine.jit import TinyJit
    if T.jit:
      if self.jit is None: self.jit = TinyJit(self.f_draft_fused if QD_FUSE else self.f_draft)
      if self.jit.cnt < 2:
        from gemma4_generate import one_job
        with one_job(): self.jit(T.jd)
      else: self.jit(T.jd)
    else: (self.f_draft_fused if QD_FUSE else self.f_draft)(None)
    o = OA.host_invalidate(self.K.bufs["ids_head"]).numpy(); hn = OA.host_invalidate(self.h_next).numpy()[:T.c.H].copy()
    return int(o[0]), float(o[1:2].view(np.float32)[0]), hn
  def set_pos(self, P): poke(self.posd, np.array([P - 1] + [0] * 15, np.int32))
  chain = GM.Drafter.chain

def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--mtp", default=GM.default_mtp(GR.DEFAULT)); ap.add_argument("--out", default="/mnt/ssd/gemma4-e2b-npu/mtp")
  a = ap.parse_args(); pack(a.mtp, os.path.join(a.out, "q8"))

if __name__ == "__main__": main()

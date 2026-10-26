"""Unit checks of the qdot drafter's pieces vs numpy: the q8 GEMV kernel on the drafter's own tensors (act pkg / norm / plain), and the
dpkg producers (plain, geglu, sandwich) against qwen3moe_qgemv.qact4m_ref. python3 qdraft_unit.py"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import gemma4_npu as GN, gemma4_generate as GG, gemma4_mtp as GM, gemma4_draft as GD, gemma4_ref as GR
from gemma4_npu import dev, poke, OA, dtypes
import qwen3moe_qgemv as Q
from tinygrad import Tensor
cache = GG.DEFAULT_CACHE; meta = json.load(open(os.path.join(cache, "gemma4.json"))); D = GR.Model(GM.default_mtp(meta["gguf"]), cache_layers=False); c = D.c
K_ = GN.SharedK(2, 24, c.EPS, real=1); rng = np.random.default_rng(0)
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
def q8ref(levels, q, d):   # Q8_0 with the kernel's operands: d fp16-exact, q int8, per 32
  N, K = q.shape; y = np.zeros(N)
  for qa, sx in levels:
    P = (q.reshape(N, K // 32, 32).astype(np.int64) * qa.reshape(1, K // 32, 32).astype(np.int64)).sum(-1); y = y + (d.astype(np.float64) * sx[None].astype(np.float64) * P).sum(1)
  return y
for tname in ("blk.0.attn_q.weight", "blk.0.ffn_down.weight", "nextn.pre_projection.weight"):
  q, d = D.g.q8(tname); N, K = q.shape; st, npad = GD.q8_stream(q, d); W8 = dev(st); x = rng.standard_normal(K).astype(np.float32)
  levels, sm = Q.qact_ref(x); ref = q8ref(levels, q, d); want = (q.astype(np.float64) * np.repeat(d, 32, 1)) @ x.astype(np.float64)
  xd = dev(x); pkg = Tensor.zeros(K // 128 * Q.SL_REC // 4, device="ZHOUYI").contiguous().realize(); K_.call(f"qact4m|{K}|1|plain|{Q.LEV}", Q.qact4m_src(K), pkg, xd, xd)
  yd = Tensor.zeros(npad + 64, device="ZHOUYI").contiguous().realize(); sel = dev(np.array([0] + [-1] * 15, np.int32)); dsc = dev(Q.qgemvk_desc("q8", K))
  K_.call(f"qgemv|u|q8|{N}|{K}|pkg", Q.qgemvk_src("q8", N, K), yd, pkg, W8, sel, dsc, yd, xd, xd); got = OA.host_invalidate(yd).numpy()[:N]
  nb = 2 if K >= 2048 else 3
  if K > 2048: print(f"{tname} [{N} x {K}]: pkg vs int ref {np.abs(got - ref).max() / np.abs(ref).max():.1e} (vs float64 rel {rel(got, want):.1e})", flush=True); continue
  K_.call(f"qgemv|u|q8|{N}|{K}|plain", Q.qgemvk_src("q8", N, K, act="plain", nbuf=nb), yd, pkg, W8, sel, dsc, yd, xd, xd); gotp = OA.host_invalidate(yd).numpy()[:N]
  w = rng.uniform(0.5, 1.5, K).astype(np.float32); xn = x / np.sqrt((x * x).mean() + c.EPS) * w; lv, _ = Q.qact_ref(xn); refn = q8ref(lv, q, d)
  K_.call(f"qgemv|u|q8|{N}|{K}|norm", Q.qgemvk_src("q8", N, K, act="norm", eps=c.EPS, nbuf=nb), yd, pkg, W8, sel, dsc, yd, xd, dev(w)); gotn = OA.host_invalidate(yd).numpy()[:N]
  print(f"{tname} [{N} x {K}]: pkg vs int ref {np.abs(got - ref).max() / np.abs(ref).max():.1e} (vs float64 rel {rel(got, want):.1e}); plain {np.abs(gotp - ref).max() / np.abs(ref).max():.1e}; norm vs ref {np.abs(gotn - refn).max() / np.abs(refn).max():.1e}", flush=True)
# dpkg modes
for mode, K in (("plain", 3072), ("geglu", 2048), ("sandwich", 256)):
  x = rng.standard_normal(2 * K if mode == "geglu" else K).astype(np.float32); y = rng.standard_normal(K).astype(np.float32)
  wv = rng.uniform(0.5, 1.5, K + 1).astype(np.float32); wn = rng.uniform(0.5, 1.5, K).astype(np.float32)
  if mode == "plain": want_x = x
  elif mode == "geglu": g = x[:K].astype(np.float64); want_x = (g / (1 + np.exp(-2 * np.sqrt(2 / np.pi) * (g + 0.044715 * g ** 3))) * x[K:]).astype(np.float32)
  else:
    x1 = (x + y / np.sqrt((y * y).mean() + c.EPS) * wv[:K]) * wv[K]; want_x = (x1 / np.sqrt((x1 * x1).mean() + c.EPS) * wn).astype(np.float32)
  pkg = Tensor.zeros(K // 128 * Q.SL_REC // 4, device="ZHOUYI").contiguous().realize(); x1d = Tensor.zeros(K + 64, device="ZHOUYI").contiguous().realize()
  np_ = K if mode == "geglu" else 0
  K_.call(f"dpkg|u|{mode}|{K}", GD.dpkg_src(K, mode, c.EPS, np_), pkg, dev(x), dev(y), dev(wv), dev(wn), x1d, dev(GD.dpkg_desc(K, np_)))
  pk = np.frombuffer(OA.host_invalidate(pkg).numpy().tobytes(), np.uint8).reshape(-1, Q.SL_REC); pr = Q.qact4m_ref(want_x).reshape(-1, Q.SL_REC)
  f32 = lambda a, o, n: a[:, o:o + 4 * n].copy().view(np.float32)
  msg = f"dpkg {mode} K {K}: q1 codes differ at {int((pk[:, :128] != pr[:, :128]).sum())} of {pk.shape[0] * 128}, sx max rel {np.abs(f32(pk, Q.O_SX, 8) - f32(pr, Q.O_SX, 8)).max() / np.abs(f32(pr, Q.O_SX, 8)).max():.1e}"
  if mode == "sandwich": msg += f"; x1 rel {rel(OA.host_invalidate(x1d).numpy()[:K], x1.astype(np.float32)):.1e}"
  print(msg, flush=True)

"""Where a row costs on the 2B: the GEMMs alone (gu: 12288 x 2048, dn: 2048 x 6144, qkv|z, o) and the producers around them (had_a32,
swiglu_a, resid, unpack) at 1 / 4 / 8 / 12 rows, each as a JIT replay of 10 launches."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.environ.setdefault("QWEN_MODEL", "bonsai2-2b"); os.environ["QWEN_DRAFT"] = "none"; os.environ["QWEN_SPEC_GEOS"] = "2,3,4"
import bonsai2_generate, numpy as np
import qwen38_generate as QG, qwen38_ref as R
from tinygrad import Device
from tinygrad.engine.jit import TinyJit
M = QG.Model(1, os.environ["QWEN_NPU"], os.environ["QWEN_DIR"], 512, 12); dev = Device["ZHOUYI"]
def timed(fn, x, reps=10):
  def f(_):
    for _ in range(reps): r = fn()
    return r
  j = TinyJit(f)
  for _ in range(2): j(x)
  dev.synchronize(); ts = []
  for _ in range(5):
    t0 = time.perf_counter(); j(x); dev.synchronize(); ts.append((time.perf_counter() - t0) / reps)
  return min(ts) * 1e3
print(f"{'rows':>4s} {'gu GEMM':>9s} {'dn GEMM':>9s} {'qkvz GEMM':>10s} {'o GEMM':>8s} {'had_a32':>8s} {'swiglu_a':>9s} {'resid':>7s} {'unpack':>7s}  (ms a launch)")
for n in (1, 4, 8, 12):
  L = M._ver(n); K = L.K; X = M.xbvs[n][0]; L.prefetch(0); slot, meta, sm = L.load(0); i = M.gidx[0]
  a_h = K.rms_a(X, (M.Win_g, M.sidx[i]), R.H, "a_h"); ct_gu = L.lin_ct(a_h, slot, meta, "gu"); a_sw = K.swiglu_a(ct_gu, R.INTER, "a_sw"); ct_dn = L.lin_ct(a_sw, slot, meta, "dn")
  o_rows = K.rows_buf("o_rows", R.NV * R.DV); a_o = K.rms_a(o_rows, None, R.NV * R.DV, "a_o", norm=False)
  qk = "qkvz" if "qkvz" in meta else "qkv"
  r = dict(gu=timed(lambda: L.lin_ct(a_h, slot, meta, "gu"), X), dn=timed(lambda: L.lin_ct(a_sw, slot, meta, "dn"), X), qkvz=timed(lambda: L.lin_ct(a_h, slot, meta, qk), X), o=timed(lambda: L.lin_ct(a_o, slot, meta, "o"), X),
           had=timed(lambda: K.rms_a(X, (M.Win_g, M.sidx[i]), R.H, "a_h"), X), sw=timed(lambda: K.swiglu_a(ct_gu, R.INTER, "a_sw"), X), resid=timed(lambda: K.resid(X, ct_dn, R.H, "x1"), X), unp=timed(lambda: K.unpack(ct_dn, R.H, "u_rows"), X))
  print(f"{n:4d} {r['gu']:9.3f} {r['dn']:9.3f} {r['qkvz']:10.3f} {r['o']:8.3f} {r['had']:8.3f} {r['sw']:9.3f} {r['resid']:7.3f} {r['unp']:7.3f}", flush=True)

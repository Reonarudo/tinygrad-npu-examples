"""The 2B's linears alone at 1 / 4 / 8 / 12 rows (the GEMM's rows mode): the cost of a row tile, decisive for batching + speculation."""
import os, sys, time, functools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.environ["QWEN_MODEL"] = "bonsai2-2b"; os.environ["QWEN_DRAFT"] = "none"; os.environ["QWEN_SPEC_GEOS"] = "2,3,4"
import bonsai2_generate, numpy as np
import qwen38_generate as QG, qwen38_ref as R
from tinygrad import Device
from tinygrad.engine.jit import TinyJit
M = QG.Model(1, os.environ["QWEN_NPU"], os.environ["QWEN_DIR"], 512, 12); dev = Device["ZHOUYI"]
for n in (1, 4, 8, 12):
  L = M._ver(n); K = L.K; X = M.xbvs[n][0]; L.prefetch(0); slot, meta, sm = L.load(0); i = M.gidx[0]
  def body(_):
    a_h = K.rms_a(X, (M.Win_g, M.sidx[i]), R.H, "a_h"); ct = L.lin_cts(a_h, slot, meta, ("qkv", "z"))
    a_o = K.rms_a(X, None, R.NV * R.DV, "a_o", norm=False); x1 = K.resid(X, L.lin_ct(a_o, slot, meta, "o"), R.H, "x1")
    return L.mlp(x1, slot, meta, (M.Wpost_g, M.sidx[i]), "xiov1").realize()
  j = TinyJit(body)
  for _ in range(3): j(M.jd if hasattr(M, "jd") else X)
  dev.synchronize(); ts = []
  for _ in range(7):
    t0 = time.perf_counter(); j(X); dev.synchronize(); ts.append(time.perf_counter() - t0)
  print(f"  {n:2d} rows: a DeltaNet layer's linears (qkv|z, o, gate|up, down + norms / residuals) {np.median(ts) * 1e3:.2f} ms  -> x24 layers {np.median(ts) * 24 * 1e3:.0f} ms", flush=True)

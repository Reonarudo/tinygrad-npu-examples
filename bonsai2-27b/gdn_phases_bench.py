"""gdn_tokb's phases a task at ROWS rows on the 2B (gdn_batch's "phases" diag: cycle stamps into a dg buffer; a JIT replay of 10
launches, the last launch's stamps read back). Cycles at the TEC clock (1.2034 GHz)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.environ.setdefault("QWEN_MODEL", "bonsai2-2b"); os.environ["QWEN_DRAFT"] = "none"; os.environ["QWEN_SPEC_GEOS"] = "2,3,4"
import bonsai2_generate, numpy as np
import qwen38_generate as QG, qwen38_ref as R, qwen38_kernels as QK, gdn_batch as GB
from bonsai2_batch import Batch
from tinygrad import Device, dtypes
from tinygrad.engine.jit import TinyJit
from zy import OA
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
n = int(os.environ.get("ROWS", "1")); B = Batch(M, n, 512); L = B.L; K = L.K; X = B.X[0]; L.prefetch(0); slot, meta, sm = L.load(0); i = M.gidx[0]
a_h = K.rms_a(X, (M.Win_g, M.sidx[i]), R.H, "a_h"); ct = L.lin_cts(a_h, slot, meta, ("qkv", "z")); xoff, zoff = ct["qkv"][1], ct["z"][1]
O = K.rows_buf("o_gather", max(R.NV * R.DV, R.NH * R.HD)); d = K.desc(f"gdn_tokb_desc|{GB.GDN_KB}", lambda: GB.gdn_tokb_desc(R.DK, R.DV, R.H, QK.gdn_hpt(R.NV), B.C, R.CONV))
dg = K.buf("gdn_dg", 12 * 16, dtypes.int32)
src = QK.ct_goff((GB.gdn_tokb2_src if GB.TOKB2 else GB.gdn_tokb_src)(R.NV, R.NK, R.DK, R.DV, B.C, R.CONV, R.H, K.nrb, R.EPS, n, diag=("phases",)), K.nrb, (xoff, zoff), ("x", "z"))
t = timed(lambda: K.call(f"gdn_tokb|{n}|{xoff}|{zoff}|{GB.GDN_KB}|{GB.TOKB2}|{GB.SPREAD}|phases", src, O, B.Sall, B.Call, B.gtab[0], B.ptab, ct["qkv"][0], ct["z"][0], X, B.Wab, B.Adt, B.Cwt, B.Nw, d, dg), X)
v = OA.host_invalidate(dg).numpy().reshape(12, 16)
print(f"  {os.environ['QWEN_MODEL']} {n} rows KB {GB.GDN_KB} tokb2 {GB.TOKB2} spread {GB.SPREAD}: {t:.3f} ms a launch (phases build); cycles a task, tasks 0..11:", flush=True)
for k, name in ((0, "total"), (1, "a|b"), (2, "prep"), (3, "fill wait"), (4, "drain wait"), (5, "compute"), (6, "out"), (7, "units"), (8, "blocks")):
  print(f"    {name:11s} " + " ".join(f"{int(x):8d}" for x in v[:, k]), flush=True)

"""The batched mixers alone at 1 / 4 / 8 rows on the 2B (a JIT replay of 10 launches): gdn_tokb (a DeltaNet layer) and attn_decb."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.environ["QWEN_MODEL"] = "bonsai2-2b"; os.environ["QWEN_DRAFT"] = "none"; os.environ["QWEN_SPEC_GEOS"] = "2,3,4"
import bonsai2_generate, numpy as np
import qwen38_generate as QG, qwen38_ref as R, qwen38_kernels as QK, gdn_batch as GB, attn_batch as AB
from bonsai2_batch import Batch
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
for n in (1, 4, 8):
  B = Batch(M, n, 512); L = B.L; K = L.K; X = B.X[0]; L.prefetch(0); slot, meta, sm = L.load(0); i = M.gidx[0]
  a_h = K.rms_a(X, (M.Win_g, M.sidx[i]), R.H, "a_h"); ct = L.lin_cts(a_h, slot, meta, ("qkv", "z")); xoff, zoff = ct["qkv"][1], ct["z"][1]
  src = QK.ct_goff(GB.gdn_tokb_src(R.NV, R.NK, R.DK, R.DV, B.C, R.CONV, R.H, K.nrb, R.EPS, n), K.nrb, (xoff, zoff), ("x", "z")); O = K.rows_buf("o_gather", max(R.NV * R.DV, R.NH * R.HD))
  d = K.desc("gdn_tokb_desc", lambda: GB.gdn_tokb_desc(R.DK, R.DV, R.H, QK.gdn_hpt(R.NV), B.C, R.CONV))
  t_g = timed(lambda: K.call(f"gdn_tokb|{n}|{xoff}|{zoff}", src, O, B.Sall, B.Call, B.gtab[0], B.ptab, ct["qkv"][0], ct["z"][0], X, B.Wab, B.Adt, B.Cwt, B.Nw, d), X)
  slot2, meta2, _ = L.load(M.aidx and [l for l in range(R.NL) if R.LAYER_TYPES[l] == "full"][0]); a2 = K.rms_a(X, (M.Win_a, B.aix0[0]), R.H, "a_h"); ct2 = L.lin_cts(a2, slot2, meta2, ("q", "kv"))
  q_rows = K.unpack(ct2["q"][0], R.NH * 2 * R.HD, "q_rows", goff=ct2["q"][1]); kv_rows = K.unpack(ct2["kv"][0], 2 * R.NKV * R.HD, "kv_rows", goff=ct2["kv"][1])
  import attn_gqa; d2 = K.desc("attn_dec3_desc", lambda: attn_gqa.attn_dec3_desc(R.NH, R.NKV, R.HD))
  t_a = timed(lambda: K.call(f"attn_decb|{n}|512", AB.attn_decb_src(R.NH, R.NKV, R.HD, 512, R.ROT, R.EPS, n), O, q_rows, kv_rows, B.QN, B.KN, M.ROPE, B.Kall, B.Vall, B.atab[0], d2), X)
  print(f"  {n:2d} rows: gdn_tokb {t_g:.3f} ms (x18 layers {t_g * 18:.1f})  attn_decb {t_a:.3f} ms (x6 {t_a * 6:.1f})", flush=True)

"""The Hadamard A producers of the 27B alone: rms (the 5120-wide norm inputs: qkv|z, q|k|v, gate|up, the head), plain (the 6144-wide
o_proj / out_proj inputs) and swiglu (down_proj's 17408-wide input from the gate|up tiles), each with the rotation (the model's
default producers) and without it (K.had = None: the unrotated producer Qwen3.8-27B runs at the same width, the same A layout), at
1 / 2 / 3 / 4 / 8 rows, each a JIT replay of 10 launches, ms a launch (the min of 5 replays). The difference is the transform's cost.

  python3 had_bench.py [rows ...]"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("QWEN_MODEL", "bonsai2-27b"); os.environ["QWEN_DRAFT"] = "none"; os.environ["QWEN_NPU_IDS"] = "0"; os.environ["QWEN_SPEC_TREE"] = "off"; os.environ.setdefault("QWEN_SPEC_GEOS", "2,3,4")
import bonsai2_generate  # noqa: F401  (the profile's defaults and paths)
import qwen38_generate as QG, qwen38_ref as R
from tinygrad import Device
from tinygrad.engine.jit import TinyJit
ROWS = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 8]
M = QG.Model(1, os.environ["QWEN_NPU"], os.environ["QWEN_DIR"], 512, max(ROWS + [4])); dev = Device["ZHOUYI"]
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
def kname(K, fn):
  """The kernel keys a producer call registers (its first, untimed call)."""
  seen, orig = [], K.key
  def rec(name, *a, **k): seen.append(name.split("|")[0] + ("|fast" if "|fast" in name else "") + ("|rec" if "|rec" in name else "")); return orig(name, *a, **k)
  K.key = rec
  try: fn()
  finally: K.key = orig
  return "+".join(dict.fromkeys(seen))
print(f"model {os.environ['QWEN_MODEL']}  widths: rms {R.H}, plain {R.NV * R.DV}, swiglu {R.INTER}   (ms a launch)", flush=True)
FAST512 = os.environ.get("HAD_BENCH_FAST512") == "1"      # Bonsai 2 2B / 4B: also had_fast(hb=512) against had_rows, and their bytes compared
if FAST512: os.environ["QWEN_HAD_FAST512"] = "0"
print(f"{'rows':>4s} | {'rms had':>8s} {'rms no':>7s} | {'plain had':>9s} {'plain no':>8s} | {'swiglu had':>10s} {'swiglu no':>9s} | kernels", flush=True)
for n in ROWS:
  L = M._ver(n); K = L.K; X = M.xbvs[n][0]; L.prefetch(0); slot, meta, sm = L.load(0); i = M.gidx[0]
  a_h = K.rms_a(X, (M.Win_g, M.sidx[i]), R.H, "a_h"); ct_gu = L.lin_ct(a_h, slot, meta, "gu")
  o_rows = K.rows_buf("o_rows", R.NV * R.DV)
  calls = dict(rms=lambda: K.rms_a(X, (M.Win_g, M.sidx[i]), R.H, "a_h"), plain=lambda: K.rms_a(o_rows, None, R.NV * R.DV, "a_o", norm=False),
               swiglu=lambda: K.swiglu_a(ct_gu, R.INTER, "a_sw"))
  r, names = {}, {}
  for nm, fn in calls.items():
    names[nm + "_had"] = kname(K, fn); r[nm + "_had"] = timed(fn, X)
    saved = K.had; K.had = None
    try: names[nm + "_no"] = kname(K, fn); r[nm + "_no"] = timed(fn, X)
    finally: K.had = saved
  print(f"{n:4d} | {r['rms_had']:8.3f} {r['rms_no']:7.3f} | {r['plain_had']:9.3f} {r['plain_no']:8.3f} | {r['swiglu_had']:10.3f} {r['swiglu_no']:9.3f} | "
        + " ".join(f"{k}={v}" for k, v in names.items()), flush=True)
  if FAST512:
    from zy import OA
    line = []
    for nm, fn in calls.items():
      os.environ["QWEN_HAD_FAST512"] = "0"; old = OA.host_invalidate(fn()).numpy().copy(); t_old = r[nm + "_had"]
      os.environ["QWEN_HAD_FAST512"] = "1"; kn = kname(K, fn); new = OA.host_invalidate(fn()).numpy().copy(); t_new = timed(fn, X)
      os.environ["QWEN_HAD_FAST512"] = "0"
      same = old.shape == new.shape and bool((old.view("u1") == new.view("u1")).all())
      line.append(f"{nm}: had_rows {t_old:.3f} -> fast512 {t_new:.3f} ms, bytes identical {same} ({kn})")
    print("      " + " | ".join(line), flush=True)

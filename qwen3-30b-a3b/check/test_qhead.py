"""The output head on the qdot GEMV (one launch over the 4 parts' concatenated q8=4 streams, the norm's package built in-kernel, the
top-1 kept per task) + head_reduce, vs float32 logits from the GGUF and vs the gemm_gs head (gemm_gs + head_top + head_reduce),
both timed.  python3 test_qhead.py [reps]"""
import os, sys, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import qwen3moe_npu as QN                     # noqa: E402
import qwen3moe_qgemv as Q                    # noqa: E402
from qwen3moe_npu import Kernels, Tensor, Device, OA, dtypes, dev  # noqa: E402
from tinygrad.engine.jit import TinyJit       # noqa: E402
from tinygrad.helpers import Context, ContextVar  # noqa: E402
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
M = QN.QR.Model(); c = M.c; cache = QN.CACHE
z = np.load(os.path.join(cache, "head.npz")); parts = [tuple(int(v) for v in r) for r in z["parts"]]; w_out = z["w_out"].astype(np.float32)
bins = [np.fromfile(os.path.join(cache, f"head{j}.bin"), np.uint8) for j in range(len(parts))]
N = sum(p[1] for p in parts); H = c.H
K = Kernels(2, 24, c.EPS, real=1)
x = np.random.default_rng(0).standard_normal(H).astype(np.float32) * 2; xr = np.zeros((24, H), np.float32); xr[0] = x; xd = dev(xr); wd = dev(w_out)
# float32 reference logits
h = x / np.sqrt((x * x).mean() + c.EPS) * w_out; kind, b = M.blocks("output.weight"); logits = np.empty(N, np.float32)
for r0 in range(0, N, 8192):
  logits[r0:r0 + 8192] = QN.QR.dequant(kind, b[r0:r0 + 8192]).reshape(-1, H) @ h
want = int(np.argmax(logits)); srt = np.sort(logits)
print(f"float32 reference: top-1 {want} (logit {logits[want]:.4f}, runner-up gap {srt[-1] - srt[-2]:.4f})", flush=True)
# qdot head
Wall = dev(np.concatenate(bins)); sel0 = dev(np.array([0] + [-1] * 15, np.int32)); tops = K.buf("top_head0", 48, dtypes.float32)
src = Q.qgemvk_src("q6", N, H, act="norm", top=True, eps=c.EPS, nbuf=2); d = dev(Q.qgemvk_desc("q6", H)); pk = Tensor.zeros(H // 128 * Q.SL_REC // 4, device="ZHOUYI").contiguous().realize()
def qhead():
  K.call(f"qgemv|head|{N}", src, tops, pk, Wall, sel0, d, tops, xd, wd); return K.head_reduce((0,), tag="ids_qhead")
ids = qhead(); got = OA.host_invalidate(ids).numpy(); tp = OA.host_invalidate(tops).numpy()[:48].reshape(12, 4)
pr = np.exp(logits - logits.max()); pr = 1 / pr.sum()
print(f"qdot head: top-1 {got[0]} ({'== the reference' if got[0] == want else 'DIFFERS'}), probability {got[1:2].view(np.float32)[0]:.4g} (reference {pr:.4g}); "
      f"task maxima vs reference max {np.abs(tp[:, 0].max() - logits.max()):.2e}", flush=True)
# the head over the decode-only q6n stream (6.6 bits a weight), when packed
p6 = os.path.join(cache, "head_q6n.bin"); q6n = os.path.exists(p6)
if q6n:
  W6 = dev(np.fromfile(p6, np.uint8)); src6 = Q.qgemvk_src("q6n", N, H, act="norm", top=True, eps=c.EPS, nbuf=2); d6 = dev(Q.qgemvk_desc("q6n", H))
  def qhead6():
    K.call(f"qgemv|head6n|{N}", src6, tops, pk, W6, sel0, d6, tops, xd, wd); return K.head_reduce((0,), tag="ids_qhead6")
  ids6 = qhead6(); got6 = OA.host_invalidate(ids6).numpy(); tp6 = OA.host_invalidate(tops).numpy()[:48].reshape(12, 4)
  print(f"q6n head: top-1 {got6[0]} ({'== the reference' if got6[0] == want else 'DIFFERS'}), probability {got6[1:2].view(np.float32)[0]:.4g}; "
        f"task maxima vs reference max {np.abs(tp6[:, 0].max() - logits.max()):.2e}", flush=True)
# the gemm_gs head (as qwen3moe_generate.f_head)
heads = [dev(bb) for bb in bins]
def ghead():
  a = K.rms_a(xd, wd, H, "a_head")
  for j, (q8, Nj, npad, Kd, r0) in enumerate(parts):
    ng = npad // 48; ct = K.buf(f"ct_head{ng}", ng * QN.CS(K), dtypes.float32)
    OA.gemm_gs(a, heads[j], ks=QN.KS, ns=QN.NS, nrb=K.nrb, nslices=Kd // (4 * QN.KS), ngroups=ng, b8=True, bscale=True, q8=q8, piece=K.nrb, rows=1, out=ct).realize()
    K.head_top(ct, Nj, f"top_head{j}")
  return K.head_reduce(tuple(p[4] for p in parts))
gid = OA.host_invalidate(ghead()).numpy()[0]; print(f"gemm_gs head: top-1 {gid} ({'== the reference' if gid == want else 'DIFFERS'})", flush=True)
def timeit(fn):
  def f(_):
    r = None
    for _ in range(reps): r = fn()
    return r
  j = TinyJit(f); jd = Tensor.zeros(16, device="ZHOUYI").contiguous().realize()
  for _ in range(2):
    with Context(**{k: v for k, v in (("JIT_BATCH_SIZE", 0), ("ZHOUYI_CHAIN_CAP", 4096)) if k in ContextVar._cache}): j(jd)
  Device["ZHOUYI"].synchronize(); ts = []
  for _ in range(3):
    t0 = time.perf_counter(); j(jd); Device["ZHOUYI"].synchronize(); ts.append((time.perf_counter() - t0) / reps)
  return min(ts)
tq = timeit(qhead); tg = timeit(ghead); nb = sum(bb.nbytes for bb in bins)
print(f"head: qdot {tq * 1e3:.2f} ms ({nb / tq / 1e9:.1f} GB/s), gemm_gs {tg * 1e3:.2f} ms", flush=True)
if q6n: t6 = timeit(qhead6); print(f"head on the q6n stream: {t6 * 1e3:.2f} ms ({W6.nbytes() / t6 / 1e9:.1f} GB/s of {W6.nbytes() / 1e6:.0f} MB)", flush=True)

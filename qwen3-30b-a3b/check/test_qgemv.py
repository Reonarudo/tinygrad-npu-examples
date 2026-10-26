"""The qdot decode GEMV (`qwen3moe_qgemv.qgemvk`) on Qwen3-30B-A3B's own tensors, over their gemm_gs streams (Q4_K: q8=5, Q6_K: q8=4),
against float64: the activation package (qact4m) vs its numpy reference, 1 head vs the integer reference, the split output, all the
heads of a stacked tensor vs the 1-head run, the launch timed.  python3 test_qgemv.py [tensor[:expert]] [reps]   (QWEN3MOE_QLEV 1 / 2)"""
import os, sys, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import qwen3moe_npu as QN                     # noqa: E402  (the tinygrad / zy paths)
import qwen3moe_qgemv as Q                    # noqa: E402
from tinygrad import Tensor, Device           # noqa: E402
from tinygrad.engine.jit import TinyJit       # noqa: E402
from tinygrad.helpers import Context, ContextVar  # noqa: E402
from zy import OA, gemm_fp16 as G             # noqa: E402
from gguf_read import GGUF, GGML, dequant_q4_k, dequant_q5_k, dequant_q6_k  # noqa: E402

g = GGUF(os.environ.get("QWEN3MOE_GGUF", "/mnt/ssd/models/Qwen3-30B-A3B-Q4_K_M.gguf"))
def blocks(spec):
  name, _, e = spec.partition(":"); kind, dims, raw = g.raw(name); bb = GGML[g.tensors[name][1]][2]; K, N = dims[0], dims[1]
  return kind, np.asarray(raw.reshape(-1, N, K // 256, bb)[int(e or 0)])
dev = lambda arr: Tensor(np.ascontiguousarray(arr), device="ZHOUYI").realize()
rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)


def timed(call, reps, xd):
  def f(_):
    r = None
    for _ in range(reps): r = call()
    return r
  j = TinyJit(f)
  for _ in range(2):
    with Context(**{k: v for k, v in (("JIT_BATCH_SIZE", 0), ("ZHOUYI_CHAIN_CAP", 4096)) if k in ContextVar._cache}): j(xd)
  Device["ZHOUYI"].synchronize(); ts = []
  for _ in range(5):
    t0 = time.perf_counter(); j(xd); Device["ZHOUYI"].synchronize(); ts.append((time.perf_counter() - t0) / reps)
  return min(ts)


def main():
  spec = sys.argv[1] if len(sys.argv) > 1 else "blk.0.ffn_gate_exps.weight:5"; reps = int(sys.argv[2]) if len(sys.argv) > 2 else 10
  name, _, e0 = spec.partition(":"); e0 = int(e0 or 0); nh = 8 if "exps" in name else 1
  streams = []
  for e in range(nh):
    kind, bl = blocks(f"{name}:{e}")
    if kind == "Q4_K" and os.environ.get("QG_KIND", "q4") == "q4n":      # the decode-only 4.75-bit strip stream
      ue, sce, me, de, dme = Q.q4k_split_raw(bl); N, K = ue.shape; kd = "q4n"
      streams.append(np.concatenate([Q.pack_b_group_q4n(ue, sce, me, de, dme, st) for st in range(3 * -(-N // 48))]))
      if e == e0: op, w = (ue, sce, me, de, dme), dequant_q4_k(bl.reshape(-1, 144)).reshape(N, K).astype(np.float64)
    elif kind == "Q4_K":
      ue, ae, be = G.q4k_split(bl); N, K = ue.shape; kd = "q4"
      streams.append(np.concatenate([G.pack_b_group_q4m(ue, ae, be, gi, 3, 32) for gi in range(-(-N // 48))]))
      if e == e0: op, w = (ue, ae, be), dequant_q4_k(bl.reshape(-1, 144)).reshape(N, K).astype(np.float64)
    elif kind == "Q5_K":                                       # the decode-only 5.75-bit stream (Qwen3.5 / 3.6-35B-A3B's down experts)
      ue, sce, me, de, dme = Q.q5k_split_raw(bl); N, K = ue.shape; kd = "q5n"
      streams.append(np.concatenate([Q.pack_b_group_q5n(ue, sce, me, de, dme, gi) for gi in range(-(-N // 48))]))
      if e == e0: op, w = (ue, sce, me, de, dme), dequant_q5_k(bl.reshape(-1, 176)).reshape(N, K).astype(np.float64)
    elif os.environ.get("QG_KIND", "q6") == "q6n":          # the decode-only 6.6-bit stream
      qe, sce, de = Q.q6k_split_raw(bl); N, K = qe.shape; kd = "q6n"
      streams.append(np.concatenate([Q.pack_b_group_q6n(qe, sce, de, gi) for gi in range(-(-N // 48))]))
      if e == e0: op, w = (qe, sce, de), dequant_q6_k(bl.reshape(-1, 210)).reshape(N, K).astype(np.float64)
    else:
      qe, de = G.q6k_split(bl); N, K = qe.shape; kd = "q6"
      streams.append(np.concatenate([G.pack_b_group_q8f(qe, de, gi, 3, 32, qb=16) for gi in range(-(-N // 48))]))
      if e == e0: op, w = (qe, de), dequant_q6_k(bl.reshape(-1, 210)).reshape(N, K).astype(np.float64)
  NP = Q.qgemvk_np(N); x = np.random.default_rng(0).standard_normal(K).astype(np.float32)
  REF = {"q4": lambda lv, s_: Q.qgemv_ref(lv, s_, *op), "q6": lambda lv, s_: Q.qgemv6_ref(lv, *op), "q6n": lambda lv, s_: Q.qgemv6n_ref(lv, *op), "q4n": lambda lv, s_: Q.qgemv4n_ref(lv, s_, *op), "q5n": lambda lv, s_: Q.qgemv5n_ref(lv, s_, *op)}[kd]
  CS0 = 4 if kd == "q4n" else 1
  levels, sm = Q.qact_ref(x); ref = REF(levels, sm); want = x.astype(np.float64) @ w.T
  print(f"{spec} {kind} [{N} x {K}] stream {streams[0].nbytes / (N * K) * 8:.2f} bits a weight; {Q.LEV} int8 level(s) vs float64: rel {rel(ref, want):.2e}")
  K_ = QN.Kernels(12, 24, 1e-6, real=1); xd = dev(x)
  pkg = Tensor.zeros(K // 128 * Q.SL_REC // 4, device="ZHOUYI").contiguous().realize()
  K_.call(f"qact4m|{K}|1|plain|{Q.LEV}", Q.qact4m_src(K), pkg, xd, xd)
  pk = np.frombuffer(OA.host_invalidate(pkg).numpy().tobytes(), np.uint8).reshape(-1, Q.SL_REC); pr = Q.qact4m_ref(x).reshape(-1, Q.SL_REC)
  f32 = lambda a, o, n: a[:, o:o + 4 * n].copy().view(np.float32)
  print(f"  qact4m: q1 codes differ at {int((pk[:, :128] != pr[:, :128]).sum())}, q2 at {int((pk[:, 128:256] != pr[:, 128:256]).sum())} of {pk.shape[0] * 128}; "
        f"sx / sums max rel {max(np.abs(f32(pk, Q.O_SX, 8) - f32(pr, Q.O_SX, 8)).max() / np.abs(f32(pr, Q.O_SX, 8)).max(), np.abs(f32(pk, Q.O_SME, 16) - f32(pr, Q.O_SME, 16)).max() / np.abs(f32(pr, Q.O_SME, 16)).max()):.1e}")
  W8 = dev(np.concatenate(streams)); d8 = dev(Q.qgemvk_desc(kd, K, cs=CS0)); yd = Tensor.zeros(nh * NP + 64, device="ZHOUYI").contiguous().realize()
  sel1 = dev(np.array([e0 * streams[0].nbytes] + [-1] * 15, np.int32))
  K_.call(f"qgemv|t|{kd}|{N}|{K}|1|{Q.LEV}", Q.qgemvk_src(kd, N, K, cs=CS0), yd, pkg, W8, sel1, d8, yd, xd, xd)
  got = OA.host_invalidate(yd).numpy()[:N]
  print(f"  qgemv {kd} 1 head vs the integer reference: max|d|/max {np.abs(got - ref).max() / np.abs(ref).max():.1e}; vs float64 rel {rel(got, want):.2e}")
  if nh == 1 and N > 1024:
    sp = (N // 2) // 16 * 16; y2 = Tensor.zeros(NP, device="ZHOUYI").contiguous().realize(); ys = Tensor.zeros(NP, device="ZHOUYI").contiguous().realize()
    K_.call(f"qgemv|t|{kd}|{N}|{K}|s{sp}|{Q.LEV}", Q.qgemvk_src(kd, N, K, split=sp, cs=CS0), ys, pkg, W8, sel1, d8, y2, xd, xd)
    a_, b_ = OA.host_invalidate(ys).numpy(), OA.host_invalidate(y2).numpy()
    print(f"  split at {sp}: equal {bool((a_[:sp] == got[:sp]).all() and (b_[:N - sp] == got[sp:]).all())}")
    nw = np.random.default_rng(1).uniform(0.5, 1.5, K).astype(np.float32); xn = x / np.sqrt((x * x).mean() + 1e-6) * nw
    lv, s_ = Q.qact_ref(xn); refn = REF(lv, s_); pkx = Tensor.zeros(K // 128 * Q.SL_REC // 4, device="ZHOUYI").contiguous().realize()
    K_.call(f"qgemv|t|{kd}|{N}|{K}|norm|{Q.LEV}", Q.qgemvk_src(kd, N, K, act="norm", export=True, cs=CS0), ys, pkx, W8, sel1, d8, ys, xd, dev(nw))
    gn = OA.host_invalidate(ys).numpy()[:N]; px = np.frombuffer(OA.host_invalidate(pkx).numpy().tobytes(), np.uint8).reshape(-1, Q.SL_REC)
    print(f"  act=norm (in-kernel package) vs reference: max|d|/max {np.abs(gn - refn).max() / np.abs(refn).max():.1e}; exported q1 codes differ at "
          f"{int((px[:, :128] != Q.qact4m_ref(xn).reshape(-1, Q.SL_REC)[:, :128]).sum())}")
  CS, NB = int(os.environ.get("QG_CS", str(CS0))), int(os.environ.get("QG_NBUF", "3"))         # slices a weight DMA, buffers
  sel8 = dev(np.array([e * streams[0].nbytes for e in range(nh)] + [-1] * (16 - nh), np.int32)); src8 = Q.qgemvk_src(kd, N, K, heads=nh, cs=CS, nbuf=NB)
  key8 = f"qgemv|t|{kd}|{N}|{K}|{nh}|{Q.LEV}|{CS}|{NB}"; d8 = dev(Q.qgemvk_desc(kd, K, cs=CS))
  K_.call(key8, src8, yd, pkg, W8, sel8, d8, yd, xd, xd); got8 = OA.host_invalidate(yd).numpy()[:nh * NP].reshape(nh, NP)
  print(f"  {nh} heads: head {e0} == the 1-head run: {bool((got8[e0, :N] == got).all())}; padding zero: {bool((got8[:, N:] == 0).all())}")
  if nh == 8 and K == 768:          # swiglu: the 8 heads' packages from gate | up rows
    gu = np.random.default_rng(2).standard_normal((16, K)).astype(np.float32); gud = dev(gu)
    K_.call(f"qgemv|t|{kd}|{N}|{K}|sw|{Q.LEV}", Q.qgemvk_src(kd, N, K, heads=8, act="swiglu", gu_np=K, cs=CS0), yd, pkg, W8, sel8, d8, yd, gud, xd)
    gs = OA.host_invalidate(yd).numpy()[:8 * NP].reshape(8, NP); errs = []
    for h in range(8):
      v = (gu[h] / (1 + np.exp(-gu[h])) * gu[8 + h]).astype(np.float32); lv, s_ = Q.qact_ref(v); _, bl = blocks(f"{name}:{h}")
      r_ = Q.qgemv_ref(lv, s_, *G.q4k_split(bl)) if kd == "q4" else Q.qgemv6_ref(lv, *G.q6k_split(bl)) if kd == "q6" else Q.qgemv6n_ref(lv, *Q.q6k_split_raw(bl)) if kd == "q6n" else Q.qgemv5n_ref(lv, s_, *Q.q5k_split_raw(bl)) if kd == "q5n" else Q.qgemv4n_ref(lv, s_, *Q.q4k_split_raw(bl)); errs.append(np.abs(gs[h, :N] - r_).max() / np.abs(r_).max())
    print(f"  act=swiglu, 8 heads vs reference: worst max|d|/max {max(errs):.1e}")
  t = timed(lambda: K_.call(key8, src8, yd, pkg, W8, sel8, d8, yd, xd, xd), reps, xd); nb = nh * streams[0].nbytes
  print(f"  qgemv {kd} {nh} heads x [{N} x {K}] (chunks of {CS} slices, {NB} buffers): {t * 1e6:.1f} us a launch, {nb / t / 1e9:.1f} GB/s of weights")


if __name__ == "__main__":
  main()

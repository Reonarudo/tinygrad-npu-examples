"""The batched prefill's matrix-unit attention (qrope_a, k_panels, v_panels, the QK^T / PV GEMMs, softmax_a, o_rows) on random
q rows and caches vs numpy: causal attention from position 0, GQA 32 / 4, head 128."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import qwen3moe_npu as QN
from qwen3moe_npu import Tensor, OA, dtypes, dev, poke, MK
NH, NKV, HD, nrb, M = 32, 4, 128, 44, 512; R = 12 * nrb; QR = NH * HD; TMAX = 1024; T = M; eps = 1e-6; rng = np.random.default_rng(0)
ngq = -(-T // 48); nstrip = 3 * ngq; nsl = -(-T // 128)
q = (rng.standard_normal((R, QR)) * 0.5).astype(np.float32); qnw = (rng.uniform(0.5, 1.5, HD) / np.sqrt(HD)).astype(np.float32)
Kc = (rng.standard_normal((TMAX, NKV, HD)) * 0.5).astype(np.float32); Vc = rng.standard_normal((TMAX, NKV, HD)).astype(np.float32)
ang = np.arange(TMAX)[:, None] * (1e6 ** (-np.arange(0, HD, 2) / HD))[None]; cs = np.concatenate([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
z = lambda n_, dt=dtypes.float32: Tensor.zeros(n_, device=QN.DEV, dtype=dt).contiguous().realize()
qd, qnwd, Kd, Vd, csd = dev(q), dev(qnw), dev(Kc.reshape(-1)), dev(Vc.reshape(-1)), dev(cs); idx = z(16, dtypes.int32)
qa, kb, vb = z(NH * nrb * 1536, dtypes.uint16), z(NKV * nstrip * 2048, dtypes.uint16), z(NKV * 9 * nsl * 2048, dtypes.uint16)
pa = z(NH * nsl * nrb * 1536, dtypes.uint16); cts = z(NH * ngq * nrb * 576); cto = z(NH * 3 * nrb * 576); o = z(R * QR)
call = lambda name, src, *a: OA.csrc_call(OA.register_csrc(name, src, ntasks=12), *a).realize()
call("qrope_a", MK.qrope_a_src(NH, HD, nrb, eps, QR), qa, qd, qnwd, csd, idx, dev(MK.qrope_a_desc(QR, HD)))
call("k_panels", MK.k_panels_src(NKV, HD, nstrip), kb, Kd, dev(MK.k_panels_desc(NKV, HD)))
call("v_panels", MK.v_panels_src(NKV, HD, nsl), vb, Vd, dev(MK.v_panels_desc(NKV, HD)))
OA.gemm_gs(qa, kb, ks=32, ns=3, nrb=nrb, nslices=1, ngroups=ngq, heads=NH, a_stride=nrb * 1536, b_stride=nstrip * 2048, b_head_div=NH // NKV, c_stride=ngq * nrb * 576, out=cts).realize()
call("softmax_a", MK.softmax_a_src(NH, nrb, ngq, nsl, T), pa, cts, idx, dev(MK.softmax_a_desc(ngq, nrb)))
OA.gemm_gs(pa, vb, ks=32, ns=3, nrb=nrb, nslices=nsl, ngroups=3, heads=NH, a_stride=nsl * nrb * 1536, b_stride=9 * nsl * 2048, b_head_div=NH // NKV, c_stride=3 * nrb * 576, out=cto).realize()
call("o_rows", MK.o_rows_src(NH, HD, nrb, QR), o, cto, dev(MK.o_rows_desc(nrb, QR, HD)))
got = OA.host_invalidate(o).numpy().reshape(R, NH, HD)[:M]
# numpy: q-norm (x qnw) + rope, scores vs K (already the cache's), causal softmax, x V
qq = q[:M].reshape(M, NH, HD); qq = qq / np.sqrt((qq ** 2).mean(-1, keepdims=True) + eps) * qnw
c_, s_ = cs[:M, None, :HD // 2], cs[:M, None, HD // 2:]; x1, x2 = qq[..., :HD // 2], qq[..., HD // 2:]; qq = np.concatenate([x1 * c_ - x2 * s_, x2 * c_ + x1 * s_], -1)
qg = qq.reshape(M, NKV, NH // NKV, HD); sc = np.einsum("mkgd,tkd->mkgt", qg.astype(np.float64), Kc[:T].astype(np.float64))
sc = np.where((np.arange(T)[None] <= np.arange(M)[:, None])[:, None, None], sc, -np.inf); p = np.exp(sc - sc.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
want = np.einsum("mkgt,tkd->mkgd", p, Vc[:T].astype(np.float64)).reshape(M, NH, HD)
print(f"mma attention ({M} rows, T {T}): rel Frobenius {np.linalg.norm(got - want) / np.linalg.norm(want):.3e}, max |d| {np.abs(got - want).max():.3e} (max |want| {np.abs(want).max():.3f}), finite {bool(np.isfinite(got).all())}")
import time
from tinygrad import Device
steps = {
  "qrope_a": lambda: call("qrope_a", MK.qrope_a_src(NH, HD, nrb, eps, QR), qa, qd, qnwd, csd, idx, dev_d["q"]),
  "k_panels": lambda: call("k_panels", MK.k_panels_src(NKV, HD, nstrip), kb, Kd, dev_d["k"]),
  "v_panels": lambda: call("v_panels", MK.v_panels_src(NKV, HD, nsl), vb, Vd, dev_d["v"]),
  "gemm QK": lambda: OA.gemm_gs(qa, kb, ks=32, ns=3, nrb=nrb, nslices=1, ngroups=ngq, heads=NH, a_stride=nrb * 1536, b_stride=nstrip * 2048, b_head_div=NH // NKV, c_stride=ngq * nrb * 576, out=cts).realize(),
  "softmax_a": lambda: call("softmax_a", MK.softmax_a_src(NH, nrb, ngq, nsl, T), pa, cts, idx, dev_d["s"]),
  "gemm PV": lambda: OA.gemm_gs(pa, vb, ks=32, ns=3, nrb=nrb, nslices=nsl, ngroups=3, heads=NH, a_stride=nsl * nrb * 1536, b_stride=9 * nsl * 2048, b_head_div=NH // NKV, c_stride=3 * nrb * 576, out=cto).realize(),
  "o_rows": lambda: call("o_rows", MK.o_rows_src(NH, HD, nrb, QR), o, cto, dev_d["o"]),
}
dev_d = {"q": dev(MK.qrope_a_desc(QR, HD)), "k": dev(MK.k_panels_desc(NKV, HD)), "v": dev(MK.v_panels_desc(NKV, HD)), "s": dev(MK.softmax_a_desc(ngq, nrb)), "o": dev(MK.o_rows_desc(nrb, QR, HD))}
for name, f in steps.items():
  f(); Device[QN.DEV].synchronize(); t0 = time.perf_counter()
  for _ in range(3): f()
  Device[QN.DEV].synchronize(); print(f"{name:10s} {(time.perf_counter() - t0) / 3 * 1e3:8.2f} ms", flush=True)

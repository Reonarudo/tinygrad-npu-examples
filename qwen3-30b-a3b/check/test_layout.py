"""The prefill's DMA layout kernels (o_a, qrope_c, resid_d, rms_ad + inv_rms, swiglu_d) on random tiles / rows vs numpy."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import qwen3moe_npu as QN
from qwen3moe_npu import Tensor, OA, dtypes, dev, poke, MK
from zy import gemm_fp16 as G
rng = np.random.default_rng(0); call = lambda name, src, *a: OA.csrc_call(OA.register_csrc(name, src, ntasks=12), *a).realize()
def pack_c(rows, nrb, ng):                 # rows [12 nrb, 48 ng] -> the C tiles [g][rb][s][i][j][r][c]
  return np.ascontiguousarray(rows.reshape(nrb, 3, 4, ng, 3, 4, 4).transpose(3, 0, 4, 1, 5, 2, 6)).ravel()
def unpack_a(a, nrb, K):                   # the A layout [slice][rb][kk][i][r][k] fp16 -> rows [12 nrb, K]
  return a.view(np.float16).reshape(K // 128, nrb, 32, 3, 4, 4).transpose(1, 3, 4, 0, 2, 5).reshape(12 * nrb, K).astype(np.float32)
z = lambda n_, dt=dtypes.float32: Tensor.zeros(n_, device=QN.DEV, dtype=dt).contiguous().realize()
rel = lambda g, w: float(np.abs(g - w).max() / np.abs(w).max())
nrb, NH, HD = 44, 32, 128
# o_a: PV C tiles per head (3 groups, 144 cols) -> A of K = 4096
pv = rng.standard_normal((NH, 12 * nrb, 144)).astype(np.float32); ct = dev(np.concatenate([pack_c(pv[h], nrb, 3) for h in range(NH)])); a = z(NH * nrb * 1536, dtypes.uint16)
call("o_a", MK.o_a_src(NH, HD, nrb), a, ct, dev(MK.o_a_desc(nrb)))
got = unpack_a(OA.host_invalidate(a).numpy(), nrb, NH * HD); want = np.concatenate([pv[h][:, :HD] for h in range(NH)], 1).astype(np.float16).astype(np.float32)
print("o_a max rel", rel(got, want))
# qrope_c: the q|k|v C tiles (108 groups) -> Q A per head (norm x qnw, rope)
ngq = 108; qkv = rng.standard_normal((12 * nrb, 48 * ngq)).astype(np.float32); ctq = dev(pack_c(qkv, nrb, ngq)); qnw = rng.uniform(0.5, 1.5, HD).astype(np.float32)
ang = np.arange(1024)[:, None] * (1e6 ** (-np.arange(0, HD, 2) / HD))[None]; cs = np.concatenate([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
idx = z(16, dtypes.int32); poke(idx, np.array([0] * 16, np.int32)); qa = z(NH * nrb * 1536, dtypes.uint16)
call("qrope_c", MK.qrope_c_src(NH, HD, nrb, 1e-6), qa, ctq, dev(qnw), dev(cs), idx, dev(MK.qrope_c_desc(nrb, HD)))
qg = OA.host_invalidate(qa).numpy().view(np.float16).reshape(NH, nrb, 32, 3, 4, 4).transpose(0, 1, 3, 4, 2, 5).reshape(NH, 12 * nrb, HD).astype(np.float32)
q = qkv[:, :NH * HD].reshape(-1, NH, HD); q = q / np.sqrt((q ** 2).mean(-1, keepdims=True) + 1e-6) * qnw
c_, s_ = cs[:12 * nrb, None, :64], cs[:12 * nrb, None, 64:]; q = np.concatenate([q[..., :64] * c_ - q[..., 64:] * s_, q[..., 64:] * c_ + q[..., :64] * s_], -1).transpose(1, 0, 2)
print("qrope_c max rel", rel(qg, q))
# resid_d: x + C (2048 columns, 43 groups)
x = rng.standard_normal((12 * nrb, 2048)).astype(np.float32); co = rng.standard_normal((12 * nrb, 43 * 48)).astype(np.float32); out = z(12 * nrb * 2048)
call("resid_d", MK.resid_d_src(2048, nrb), out, dev(x), dev(pack_c(co, nrb, 43)), dev(MK.resid_d_desc(2048, nrb)))
print("resid_d max rel", rel(OA.host_invalidate(out).numpy().reshape(-1, 2048), x + co[:, :2048]))
# rms_ad (+ inv_rms): fp16(x inv w) as the full layout's A
inv = z(16 * 12 * nrb); w = rng.uniform(0.5, 1.5, 2048).astype(np.float32); aa = z((2048 // 128) * nrb * 1536, dtypes.uint16)
call("inv_rms", MK.inv_rms_src(2048, 12 * nrb, 1e-6), inv, dev(x), dev(MK.inv_rms_desc(2048)))
call("rms_ad", MK.rms_ad_src(2048, nrb), aa, dev(x), inv, dev(w), dev(MK.rms_ad_desc(2048)))
want = (x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6) * w).astype(np.float16).astype(np.float32)
print("rms_ad max rel", rel(unpack_a(OA.host_invalidate(aa).numpy(), nrb, 2048), want))
# swiglu_d: S heads of 768 (16 groups) x 2 row blocks, gate heads then up heads
S, rb = 20, 2; gu = rng.standard_normal((2, S, 12 * rb, 768)).astype(np.float32)
ctg = dev(np.concatenate([pack_c(gu[k, h], rb, 16) for k in range(2) for h in range(S)])); asw = z(S * 6 * rb * 1536, dtypes.uint16)
call("swiglu_d", MK.swiglu_d_src(S, rb, 768), asw, ctg, dev(MK.swiglu_d_desc(rb)))
got = OA.host_invalidate(asw).numpy().view(np.float16).reshape(S, 6, rb, 32, 3, 4, 4).transpose(0, 2, 4, 5, 1, 3, 6).reshape(S, 12 * rb, 768).astype(np.float32)
g = gu[0]; want = g / (1 + np.exp(-g)) * gu[1]
print("swiglu_d max rel", rel(got, want))

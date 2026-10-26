"""The prefill's small kernels on random inputs vs numpy: inv_rms, rms_gather, moe_scatter."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import qwen3moe_npu as QN
from qwen3moe_npu import Kernels, Tensor, OA, dtypes, dev, MK
H, M, R = 2048, 128, 144; rng = np.random.default_rng(0)
K = Kernels(12, R, 1e-6, real=M)
x = rng.standard_normal((R, H)).astype(np.float32); xd = dev(x); inv = Tensor.zeros(16 * R, device=QN.DEV).contiguous().realize()
K.call(f"inv_rms|{H}|{M}", MK.inv_rms_src(H, M, 1e-6), inv, xd, dev(MK.inv_rms_desc(H)))
iv = OA.host_invalidate(inv).numpy()[::16][:M]; want = 1 / np.sqrt((x[:M] ** 2).mean(-1) + 1e-6)
print("inv_rms (host-written x) max rel", np.abs(iv - want).max() / want.max(), iv[:4], want[:4])
# x written by a plain-store kernel (resid32 in the full layout) first
ct = Tensor.zeros(43 * 12 * 3 * 192, device=QN.DEV).contiguous().realize(); x1 = K.resid(xd.reshape(R, H), ct, H, "x1t")
K.call(f"inv_rms|{H}|{M}", MK.inv_rms_src(H, M, 1e-6), inv, x1, dev(MK.inv_rms_desc(H)))
iv = OA.host_invalidate(inv).numpy()[::16][:M]; print("inv_rms (x from resid32) max rel", np.abs(iv - want).max() / want.max())

w = rng.standard_normal((128, H)).astype(np.float32) * 0.02; out = Tensor.zeros(R * 128, device=QN.DEV).contiguous().realize()
K.call(f"gemv32m|{M}|{H}|128", MK.gemv32m_src(M, H, 128, 1e-6), out, xd, dev(w), dev(MK.gemv32m_desc(H)))
got = OA.host_invalidate(out).numpy().reshape(R, 128)[:M]; want = (x[:M] / np.sqrt((x[:M] ** 2).mean(-1, keepdims=True) + 1e-6)) @ w.T
print("gemv32m max rel", np.abs(got - want).max() / np.abs(want).max())

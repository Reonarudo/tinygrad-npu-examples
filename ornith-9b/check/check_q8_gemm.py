"""gemm_gs(q8) -- int8 codes, a scale per 32 weights, the codes expanded to fp16 SUBNORMALS -- on a real Ornith linear vs numpy:
the board test that the matrix unit honours subnormal fp16 inputs. python3 check_q8_gemm.py [tensor] [rows]"""
import os, sys, numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tinygrad")))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tinygrad import Tensor, dtypes
from zy import OA, gemm_fp16 as G
from gguf_read import GGUF

path = os.environ.get("ORNITH_GGUF", "/mnt/ssd/models/ornith-1.0-9b-Q8_0.gguf")
name = sys.argv[1] if len(sys.argv) > 1 else "blk.0.attn_qkv.weight"; rows = int(sys.argv[2]) if len(sys.argv) > 2 else 0
g = GGUF(path); q, d = g.q8(name); N, K = q.shape
DBG = os.environ.get("Q8_DBG", "")                  # debug arms: unit_d (every d = 2^-10: the fp16 scale 1.0), zero_q (want 0), ones_a
if "unit_d" in DBG: d = np.full_like(d, 2.0 ** -10)
if "zero_q" in DBG: q = np.zeros_like(q)
Q8F = os.environ.get("Q8F") == "1"                 # gemm_gs(q8=2): the weights built in fp16 by the expand, 32-step slices
ks, ns = (32 if Q8F else 8), 3; ng = -(-N // 48); nsl = K // (4 * ks)
nrb = 2; MP = 24; rng = np.random.default_rng(0)
a = np.zeros((MP, K), np.float32); a[:rows or MP] = rng.standard_normal((rows or MP, K)); a16 = a.astype(np.float16)
if "ones_a" in DBG: a[:rows or MP] = 1; a16 = a.astype(np.float16)
pa = G.pack_a_slices(a16, ks)
rt = -(-rows // 4) if rows else 0
if rows: pa = np.ascontiguousarray(np.frombuffer(pa.tobytes(), np.uint8).reshape(nsl, nrb, ks, 3, 32)[:, 0, :, :rt])
stream = np.concatenate([(G.pack_b_group_q8f if Q8F else G.pack_b_group_q8)(q, d, gi, ns, ks) for gi in range(ng)])
A = Tensor(np.frombuffer(pa.tobytes(), np.uint16).copy(), device="ZHOUYI").realize(); B = Tensor(stream, device="ZHOUYI").realize()
c = OA.gemm_gs(A, B, ks=ks, ns=ns, nrb=nrb, nslices=nsl, ngroups=ng, b8=True, bscale=True, q8=2 if Q8F else 1, piece=nrb, rows=rows).realize()
ct = OA.host_invalidate(c).numpy().reshape(ng, nrb, ns, 192)
got = np.concatenate([G.unpack_c_gs(ct[gi].tobytes(), nrb, ns) for gi in range(ng)], 1)[:rows or MP, :N]
w = q.astype(np.float64) * np.repeat(d.astype(np.float64), 32, 1)
want = a16[:rows or MP].astype(np.float64) @ w.T
small = np.abs(q) < 8; tiny = (small * np.abs(w)).sum() / np.abs(w).sum()
print(f"{name} [{N} x {K}] rows {rows or MP}: max|d| / max|want| {np.abs(got - want).max() / np.abs(want).max():.2e} | rel Frobenius "
      f"{np.linalg.norm(got - want) / np.linalg.norm(want):.2e} | finite {bool(np.isfinite(got).all())} | weights with |q| < 8: {small.mean():.1%} ({tiny:.1%} of |w|)")
# if the unit flushed subnormal inputs, every code would read 0 (all are subnormal as fp16): the output would be ~0
if DBG:
  np.set_printoptions(precision=4, linewidth=160); print("   got[0,:8]", got[0, :8], "\n   want[0,:8]", want[0, :8]); print("   got/want[0,:16]", (got[0, :16] / np.where(want[0, :16] == 0, 1, want[0, :16])))
  print("   got[:4,0]", got[:4, 0], "want[:4,0]", want[:4, 0]); np.savez("/tmp/q8dbg.npz", got=got, want=want)
print("   output magnitude vs reference:", f"{np.abs(got).mean() / np.abs(want).mean():.4f}")

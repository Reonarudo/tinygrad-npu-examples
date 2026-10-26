"""gemm_gs on Qwen3-30B-A3B's own Q4_K / Q6_K tensors (q8=3: `pack_b_group_q4f`, q8=4: `pack_b_group_q8f(qb=16)`) vs float64 of
ggml's dequantisation. python3 check_qk_gemm.py [tensor[:expert]] [rows]   (default: a Q4_K and a Q6_K tensor, rows 1 / 4 / 12)"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "..", "tinygrad")))); sys.path.insert(0, os.path.join(HERE, "..", "..", "ornith-9b"))
from tinygrad import Tensor
from zy import OA, gemm_fp16 as G
from gguf_read import GGUF, GGML, dequant_q4_k, dequant_q6_k

g = GGUF(os.environ.get("QWEN3MOE_GGUF", "/mnt/ssd/models/Qwen3-30B-A3B-Q4_K_M.gguf"))
def blocks(spec):                     # "name[:expert]" -> (kind, super-blocks [N, K/256, bytes])
  name, _, e = spec.partition(":"); kind, dims, raw = g.raw(name); bb = GGML[g.tensors[name][1]][2]; K, N = dims[0], dims[1]
  blk = raw.reshape(-1, N, K // 256, bb)[int(e or 0)]
  return kind, np.asarray(blk)

def run(spec, rows):
  kind, blk = blocks(spec); N, K = blk.shape[0], blk.shape[1] * 256; ks, ns = 32, 3; ng = -(-N // 48); nsl = K // (4 * ks); nrb = 2; MP = 24
  if kind == "Q4_K":
    q4m = os.environ.get("QWEN3MOE_Q4", "q4m") == "q4m"
    u, a, b = G.q4k_split(blk); stream = np.concatenate([(G.pack_b_group_q4m if q4m else G.pack_b_group_q4f)(u, a, b, gi, ns, ks) for gi in range(ng)]); q8 = 5 if q4m else 3
    w = dequant_q4_k(blk.reshape(-1, 144)).reshape(N, K)
  else:
    q, d = G.q6k_split(blk); stream = np.concatenate([G.pack_b_group_q8f(q, d, gi, ns, ks, qb=16) for gi in range(ng)]); q8 = 4
    w = dequant_q6_k(blk.reshape(-1, 210)).reshape(N, K)
  rng = np.random.default_rng(0); a_ = np.zeros((MP, K), np.float32); a_[:rows or MP] = rng.standard_normal((rows or MP, K)); a16 = a_.astype(np.float16)
  pa = G.pack_a_slices(a16, ks); rt = -(-rows // 4) if rows else 0
  if rows: pa = np.ascontiguousarray(np.frombuffer(pa.tobytes(), np.uint8).reshape(nsl, nrb, ks, 3, 32)[:, 0, :, :rt])
  A = Tensor(np.frombuffer(pa.tobytes(), np.uint16).copy(), device="ZHOUYI").realize(); B = Tensor(stream, device="ZHOUYI").realize()
  c = OA.gemm_gs(A, B, ks=ks, ns=ns, nrb=nrb, nslices=nsl, ngroups=ng, b8=True, bscale=True, q8=q8, piece=nrb, rows=rows).realize()
  ct = OA.host_invalidate(c).numpy().reshape(ng, nrb, ns, 192)
  got = np.concatenate([G.unpack_c_gs(ct[gi].tobytes(), nrb, ns) for gi in range(ng)], 1)[:rows or MP, :N]
  want = a16[:rows or MP].astype(np.float64) @ w.astype(np.float64).T
  fro = np.linalg.norm(got - want) / np.linalg.norm(want)
  print(f"{spec} {kind} [{N} x {K}] rows {rows or MP}: max|d| / max|want| {np.abs(got - want).max() / np.abs(want).max():.2e} | rel Frobenius {fro:.2e} | "
        f"finite {bool(np.isfinite(got).all())} | {stream.nbytes / (N * K) * 8:.2f} bits a weight -> {'PASS' if fro < 2e-3 and np.isfinite(got).all() else 'FAIL'}")
  return fro < 2e-3

specs = [sys.argv[1]] if len(sys.argv) > 1 else ["blk.0.attn_q.weight", "blk.0.ffn_gate_exps.weight:5", "blk.0.ffn_down_exps.weight:7", "output.weight"]
rowss = [int(sys.argv[2])] if len(sys.argv) > 2 else [1, 4, 12]
ok = True
for s in specs:
  for r in rowss:
    if s == "output.weight" and r != 1: continue
    ok &= run(s, r)
sys.exit(0 if ok else 1)

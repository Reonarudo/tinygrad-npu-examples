"""gemm_gs(bsel) on Qwen3-30B-A3B's experts: E experts of a layer's stacked tensor packed as equal streams, `heads` = 8 GEMMs in one
launch, head h on the expert slot h names; the slots rewritten by the host between launches (the routing's per-token pick) and
every head checked against float64 -- a stale slot (a TEC d-cache line from the previous launch) shows as a wrong expert.
python3 check_bsel.py [tensor] [E] [reps]   (default blk.0.ffn_gate_exps.weight (Q4_K), 16 experts, 4 launches)"""
import ctypes, os, sys, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "..", "tinygrad")))); sys.path.insert(0, os.path.join(HERE, "..", "..", "ornith-9b"))
from tinygrad import Tensor, Device, dtypes
from zy import OA, gemm_fp16 as G
from gguf_read import GGUF, GGML, dequant_q4_k, dequant_q6_k

name = sys.argv[1] if len(sys.argv) > 1 else "blk.0.ffn_gate_exps.weight"; E = int(sys.argv[2]) if len(sys.argv) > 2 else 16
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 4; HEADS, ks, ns, nrb, rows = 8, 32, 3, 2, 1
g = GGUF(os.environ.get("QWEN3MOE_GGUF", "/mnt/ssd/models/Qwen3-30B-A3B-Q4_K_M.gguf"))
kind, dims, raw = g.raw(name); bb = GGML[g.tensors[name][1]][2]; K, N = dims[0], dims[1]; blk = np.asarray(raw).reshape(-1, N, K // 256, bb)[:E]
ng = -(-N // 48); nsl = K // (4 * ks); t0 = time.perf_counter(); streams, ws = [], []
for e in range(E):
  if kind == "Q4_K":
    u, a, b = G.q4k_split(blk[e]); streams.append(np.concatenate([G.pack_b_group_q4f(u, a, b, gi, ns, ks) for gi in range(ng)])); q8 = 3
    ws.append(dequant_q4_k(blk[e].reshape(-1, 144)).reshape(N, K))
  else:
    q, d = G.q6k_split(blk[e]); streams.append(np.concatenate([G.pack_b_group_q8f(q, d, gi, ns, ks, qb=16) for gi in range(ng)])); q8 = 4
    ws.append(dequant_q6_k(blk[e].reshape(-1, 210)).reshape(N, K))
S = streams[0].size; print(f"{name} {kind} [{E} x {N} x {K}]: packed in {time.perf_counter() - t0:.1f} s, {S} B an expert", flush=True)
B = Tensor(np.concatenate(streams), device="ZHOUYI").realize()
rng = np.random.default_rng(0); x = rng.standard_normal((1, K)).astype(np.float16)
pa = np.ascontiguousarray(np.frombuffer(G.pack_a_slices(np.pad(x, ((0, 23), (0, 0))), ks).tobytes(), np.uint8).reshape(nsl, nrb, ks, 3, 32)[:, 0, :, :1])
A = Tensor(np.frombuffer(pa.tobytes(), np.uint16).copy(), device="ZHOUYI").realize()
sel = Tensor.zeros(16, dtype=dtypes.int32, device="ZHOUYI").contiguous().realize()
cs = ng * nrb * ns * 192; out = Tensor.zeros(HEADS * cs, dtype=dtypes.float32, device="ZHOUYI").contiguous().realize()
def poke(t, arr):
  Device["ZHOUYI"].synchronize(); arr = np.ascontiguousarray(arr); b_ = t.uop.base.buffer; b_.ensure_allocated(); ctypes.memmove(b_._buf.va, arr.ctypes.data, arr.nbytes)
ok = True
for rep in range(reps):
  pick = rng.choice(E, HEADS, replace=False); poke(sel, np.array(list(pick * S) + [0] * 8, np.int32))
  t0 = time.perf_counter()
  c = OA.gemm_gs(A, B, ks=ks, ns=ns, nrb=nrb, nslices=nsl, ngroups=ng, heads=HEADS, c_stride=cs, b8=True, bscale=True, q8=q8, piece=nrb, rows=rows, bsel=sel, out=out).realize()
  ct = OA.host_invalidate(c).numpy().reshape(HEADS, ng, nrb, ns, 192); dt = time.perf_counter() - t0
  errs = []
  for h in range(HEADS):
    got = np.concatenate([G.unpack_c_gs(ct[h, gi].tobytes(), nrb, ns) for gi in range(ng)], 1)[:1, :N]
    want = x.astype(np.float64) @ ws[pick[h]].astype(np.float64).T; errs.append(np.linalg.norm(got - want) / np.linalg.norm(want))
  good = max(errs) < 2e-3; ok &= good
  print(f"launch {rep}: experts {list(pick)} -> rel Frobenius per head max {max(errs):.2e} (min {min(errs):.2e}), {dt * 1e3:.1f} ms -> {'PASS' if good else 'FAIL'}", flush=True)
sys.exit(0 if ok else 1)

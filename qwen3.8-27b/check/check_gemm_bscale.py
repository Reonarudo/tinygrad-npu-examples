"""gemm_gs(b8, bscale) on a layer-0 linear (128x128 block scales) vs numpy: M=24 python3 check_gemm_bscale.py [linear_attn.in_proj_qkv|mlp.down_proj|...] [--scales dup|single]
ROWS=4,8,9,12: the rows mode instead (the verify pass's GEMM: compact A, one row piece, nrb 2), each row count checked vs numpy and
timed (the median of 5 warm calls: row tiles 1 / 2 use the prefetching schedule, 3 (rows 9..12) the plain one).
--scales single: the stream packed with each block scale once (gemm_fp16.pack_b_group_bscale(scales="single")) and the
kernel built for it (the same numbers: bit-identical C to the duplicated layout); default dup."""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tinygrad")))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tinygrad import Tensor, dtypes
from zy import OA, gemm_fp16 as G
import qwen38_ref as R
DEV = "ZHOUYI"; ks, ns = 32, 3; SCALES = "dup"
if "--scales" in sys.argv: i = sys.argv.index("--scales"); SCALES = sys.argv[i + 1]; del sys.argv[i:i + 2]   # the layout packed and run (gemm_gs(scales=))
assert SCALES in ("dup", "single"), f"--scales {SCALES!r}: dup | single"
W = R.Weights(os.environ.get("QWEN_DIR", "/mnt/ssd/qwen3.8-27b-fp8")); sf = W.shard(0)
name = "model.language_model.layers.0." + (sys.argv[1] if len(sys.argv) > 1 else "linear_attn.in_proj_qkv")
codes, sinv = sf.codes(name + ".weight"), sf.f32(name + ".weight_scale_inv"); N, K = codes.shape
ngroups, nslices = -(-N // (16 * ns)), K // 128
M = int(os.environ.get("M", "24")); nrb = -(-M // 12); nrb += nrb % 2; MP = 12 * nrb
rng = np.random.default_rng(0); a = np.zeros((MP, K), np.float32); a[:M] = rng.standard_normal((M, K)).astype(np.float32)
a16 = a.astype(np.float16)
t0 = time.perf_counter(); stream = np.concatenate([G.pack_b_group_bscale(codes, sinv, g, ns, ks, scales=SCALES) for g in range(ngroups)]); tp = time.perf_counter() - t0
print(f"{name.split('.')[-1]}: N {N} -> {16*ns*ngroups} ({ngroups} groups), K {K} ({nslices} slices), M {M} -> {MP}; stream {stream.nbytes/1e6:.1f} MB ({SCALES} scales) packed in {tp:.1f} s", flush=True)
A = Tensor(G.pack_a_slices(a16, ks).ravel().view(np.uint16), device=DEV).realize(); B = Tensor(stream, device=DEV).realize()
c = OA.gemm_gs(A, B, ks=ks, ns=ns, nrb=nrb, nslices=nslices, ngroups=ngroups, b8=True, bscale=True, scales=SCALES, piece=nrb if nrb < 4 else 0).realize()
t0 = time.perf_counter(); c = OA.gemm_gs(A, B, ks=ks, ns=ns, nrb=nrb, nslices=nslices, ngroups=ngroups, b8=True, bscale=True, scales=SCALES, piece=nrb if nrb < 4 else 0).realize(); tg = time.perf_counter() - t0
ct = OA.host_invalidate(c).numpy().reshape(ngroups, nrb, ns, 192) if hasattr(OA, "host_invalidate") else c.numpy().reshape(ngroups, nrb, ns, 192)
got = np.concatenate([G.unpack_c_gs(ct[g].tobytes(), nrb, ns) for g in range(ngroups)], 1)[:M, :N]
want = a16[:M].astype(np.float32) @ (R._E4M3[codes] * np.repeat(np.repeat(sinv, 128, 0), 128, 1)[:N, :K]).T
d = np.abs(got - want); print(f"gemm {tg*1e3:.0f} ms | max |d| {d.max():.3g} of max |want| {np.abs(want).max():.3g} | rel err {np.linalg.norm(got-want)/np.linalg.norm(want):.3g} | finite {bool(np.isfinite(got).all())}")
if d.max() > 1e-2 * np.abs(want).max():
  bad = np.argwhere(d > 1e-2 * np.abs(want).max()); print("first bad (row, col):", bad[:8].tolist(), "got", got[tuple(bad[:4].T)].round(3).tolist(), "want", want[tuple(bad[:4].T)].round(3).tolist())

for rows in [int(r) for r in os.environ.get("ROWS", "").split(",") if r]:   # the rows mode: nrb 2, compact A of ceil(rows / 4) row tiles
  rt = -(-rows // 4); a = np.zeros((24, K), np.float32); a[:rows] = rng.standard_normal((rows, K)).astype(np.float32); a16 = a.astype(np.float16)
  pa = np.ascontiguousarray(np.frombuffer(G.pack_a_slices(a16, ks).tobytes(), np.uint8).reshape(nslices, 2, ks, 3, 32)[:, 0, :, :rt])
  A = Tensor(np.frombuffer(pa.tobytes(), np.uint16).copy(), device=DEV).realize()
  run = lambda: OA.gemm_gs(A, B, ks=ks, ns=ns, nrb=2, nslices=nslices, ngroups=ngroups, b8=True, bscale=True, scales=SCALES, piece=2, rows=rows).realize()
  c = run(); ts = []
  for _ in range(5): t0 = time.perf_counter(); c = run(); ts.append(time.perf_counter() - t0)
  ct = OA.host_invalidate(c).numpy().reshape(ngroups, 2, ns, 192)
  got = np.concatenate([G.unpack_c_gs(ct[g].tobytes(), 2, ns) for g in range(ngroups)], 1)[:24, :N]
  want = a16[:rows].astype(np.float32) @ (R._E4M3[codes] * np.repeat(np.repeat(sinv, 128, 0), 128, 1)[:N, :K]).T
  print(f"rows {rows:2d} (row tiles {rt}): gemm {1e3 * np.median(ts):.2f} ms (median of 5) | rel err {np.linalg.norm(got[:rows] - want) / np.linalg.norm(want):.3g} | "
        f"rows {rows}.. of the first tile(s) zero {bool(not got[rows:4 * rt].any())} | finite {bool(np.isfinite(got).all())}", flush=True)

"""The CPU GEMV with llama.cpp's kernels (check/host_gemv.c over libggml-cpu) on the model's own rows: correctness vs float32 and GB/s --
three experts' gate / up / down (Q4_K, as the CPU's share of a layer) and the head (Q6_K). python3 test_host_gemv.py [threads] [cpus] [llama.cpp dir]"""
import os, sys, time, ctypes, subprocess, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import qwen3moe_ref as QR
T = int(sys.argv[1]) if len(sys.argv) > 1 else 4; cpus = [int(c) for c in sys.argv[2].split(",")] if len(sys.argv) > 2 and sys.argv[2] else None
LL = sys.argv[3] if len(sys.argv) > 3 else "/mnt/ssd/Developer/llama.cpp"; so = os.path.join(HERE, "libhostgemv.so")
subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", so, os.path.join(HERE, "host_gemv.c"), f"-L{LL}/build/bin", f"-Wl,-rpath,{LL}/build/bin", "-lggml-cpu", "-lggml-base", "-lpthread"], check=True)
lib = ctypes.CDLL(so); P = ctypes.c_void_p
lib.q8k_quantize.argtypes = [P, P, ctypes.c_int]; lib.gemv_kq.argtypes = [ctypes.c_int, P, ctypes.c_int, P, ctypes.c_int, P, ctypes.c_int, P]
cp = ctypes.cast((ctypes.c_int * T)(*cpus), P) if cpus else None
M = QR.Model(); c = M.c; rng = np.random.default_rng(0)
def q8k(x):
  K = x.size; y = np.zeros(K // 256 * 292, np.uint8); lib.q8k_quantize(x.ctypes.data, y.ctypes.data, K); return y
def bench(name, kind, rows, K, x, want, reps=5):
  N = rows.shape[0]; out = np.zeros(N, np.float32); yq = q8k(x)
  run = lambda: lib.gemv_kq(kind, rows.ctypes.data, N, yq.ctypes.data, K, out.ctypes.data, T, cp)
  run(); err = np.linalg.norm(out - want) / np.linalg.norm(want); ts = []
  for _ in range(reps):
    t0 = time.perf_counter(); run(); ts.append(time.perf_counter() - t0)
  print(f"  {name}: [{N} x {K}] {rows.nbytes / 1e6:.1f} MB: {min(ts) * 1e3:.2f} ms, {rows.nbytes / min(ts) / 1e9:.1f} GB/s; rel err vs float32 {err:.2e}", flush=True)
  return min(ts), rows.nbytes
print(f"{T} threads on {cpus}", flush=True)
# three experts of layer 6: gate | up (K = 2048), then down (K = 768); the rows of the three stacked (one GEMV each)
tt, bb = 0.0, 0
for t, Kd in (("gate", c.H), ("up", c.H), ("down", c.FF)):
  kind, b = M.blocks(f"blk.6.ffn_{t}_exps.weight"); rows = np.ascontiguousarray(b[:3].reshape(-1, b.shape[-1] * b.shape[-2]))   # 3 experts' rows
  x = rng.standard_normal(Kd).astype(np.float32); want = QR.dequant(kind, b[:3]).reshape(rows.shape[0], Kd) @ x
  t_, n_ = bench(f"3 experts' {t} ({kind})", 4 if kind == "Q4_K" else 6, rows, Kd, x, want); tt += t_; bb += n_
print(f"  three experts a layer: {tt * 1e3:.2f} ms, {bb / 1e6:.1f} MB ({bb / tt / 1e9:.1f} GB/s); x 48 layers = {tt * 48 * 1e3:.0f} ms a token", flush=True)
kind, b = M.blocks("output.weight"); N = b.shape[0]; rows = np.ascontiguousarray(b.reshape(N, -1)); x = rng.standard_normal(c.H).astype(np.float32)
want = np.concatenate([QR.dequant(kind, b[r0:r0 + 8192]).reshape(-1, c.H) @ x for r0 in range(0, N, 8192)])
bench("the head (Q6_K)", 6, rows, c.H, x, want)

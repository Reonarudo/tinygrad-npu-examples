"""The CPU's Q6_K head GEMV (check/host_head.c, NEON sdot, T threads on the big cores) on the GGUF's own output.weight rows: top-1 and
logit error vs float32, and its GB/s -- the CPU's share of a split head. python3 test_host_head.py [threads] [cpus] [rows]"""
import os, sys, time, ctypes, subprocess, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import qwen3moe_ref as QR
T = int(sys.argv[1]) if len(sys.argv) > 1 else 4; cpus = [int(c) for c in sys.argv[2].split(",")] if len(sys.argv) > 2 else None
so = os.path.join(HERE, "libhosthead.so"); subprocess.run(["gcc", "-O3", "-march=armv8.2-a+dotprod", "-shared", "-fPIC", "-o", so, os.path.join(HERE, "host_head.c"), "-lpthread"], check=True)
lib = ctypes.CDLL(so); lib.head_q6k.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
print('loading', flush=True); M = QR.Model(); c = M.c; H = c.H
kind, b = M.blocks("output.weight"); assert kind == "Q6_K"; N = b.shape[0]; rows = np.ascontiguousarray(b.reshape(N, -1)); nrows = int(sys.argv[3]) if len(sys.argv) > 3 else N
x = np.random.default_rng(0).standard_normal(H).astype(np.float32) * 2; w_out = M.t("output_norm.weight").astype(np.float32); h = (x / np.sqrt((x * x).mean() + c.EPS) * w_out).astype(np.float32)
hb = h.reshape(-1, 256); amax = np.abs(hb).max(1); xs = (amax / 127).astype(np.float32); xq = np.clip(np.rint(hb / xs[:, None]), -127, 127).astype(np.int8).reshape(-1)
logits = np.zeros(nrows, np.float32); cp = (ctypes.c_int * T)(*cpus) if cpus else None
def run(): lib.head_q6k(rows.ctypes.data, nrows, xq.ctypes.data, xs.ctypes.data, H, logits.ctypes.data, T, ctypes.cast(cp, ctypes.c_void_p) if cp else None)
print('rows ready', flush=True)
run(); ref = np.empty(nrows, np.float32)
for r0 in range(0, nrows, 8192): ref[r0:r0 + 8192] = QR.dequant(kind, b[r0:min(r0 + 8192, nrows)]).reshape(-1, H) @ h
print(f"{nrows} rows: top-1 {int(np.argmax(logits))} vs float32 {int(np.argmax(ref))}; max |d| {np.abs(logits - ref).max():.4f} of max {np.abs(ref).max():.2f} (int8 activation, one scale a 256)", flush=True)
ts = []
for _ in range(5):
  t0 = time.perf_counter(); run(); ts.append(time.perf_counter() - t0)
nb = rows.nbytes * nrows / N; print(f"host head: {T} threads{' on ' + str(cpus) if cpus else ''}: {min(ts) * 1e3:.2f} ms, {nb / min(ts) / 1e9:.1f} GB/s of {nb / 1e6:.0f} MB", flush=True)

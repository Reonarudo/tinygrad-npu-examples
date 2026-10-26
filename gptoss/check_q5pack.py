#!/usr/bin/env python3
"""mx4_to_q5 against the MXFP4 dequant on synthetic blocks: exact for spreads <= 5, rounded (reported) beyond."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "..", "qwen3-30b-a3b"), os.path.join(HERE, "..", "qwen3.8-27b"), os.path.join(HERE, "..", "ornith-9b")]
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "tinygrad"))))
import qwen3moe_qgemv as Q, gptoss_pack as P
rng = np.random.default_rng(1); N, K = 96, 2880
code = rng.integers(0, 16, (N, K), dtype=np.uint8); e = rng.integers(118, 124, (N, K // 32)); e[0, :8] = [122, 117, 116, 115, 114, 122, 122, 122]   # a spread-7 block in row 0
dh = np.ldexp(1.0, e - 128).astype(np.float32); want = (Q.KV_FP4[code] * dh.repeat(32, 1)).astype(np.float64)
u, sc, m, d, dmin = P.mx4_to_q5(code, dh); K2 = u.shape[1]
got = (d.repeat(256, 1) * sc.repeat(32, 1) * u.astype(np.float64) - dmin.repeat(256, 1) * m.repeat(32, 1))[:, :K]
err = np.abs(got - want); print(f"K {K} -> {K2}; max |err| rows 1.. {err[1:].max():.3g} (exact: {bool((err[1:] == 0).all())}); row 0 (spread 7 blocks) max |err| {err[0].max():.3g} of max |w| {np.abs(want[0]).max():.3g}")
print("u range", u.min(), u.max(), "sc range", sc.min(), sc.max(), "VERDICT:", "PASS" if (err[1:] == 0).all() and err[0].max() <= 2 ** -12 else "FAIL")

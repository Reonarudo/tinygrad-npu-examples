#!/usr/bin/env python3
"""The calibration statistics kernels against numpy (exact: a max): colmax_a16 on a random fp16 A operand (folded into a
non-zero running max) and colmax_i16 on random int16 rows."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinygrad import dtypes                     # noqa: E402
from zy import OA, gemm_fp16 as G               # noqa: E402
import ideogram4_q8 as Q                        # noqa: E402
import ideogram4_fp_backend as FB               # noqa: E402

rng = np.random.RandomState(3); rows, ok = 4704, True
for K in (4608, 12288):
    a = (rng.randn(rows, K) * rng.choice([0.1, 1, 30], K)).astype(np.float16); a[4621:] = 0
    prev = np.abs(rng.randn(K)).astype(np.float32)
    A = FB.dev(G.pack_a_slices(a, 24).ravel().view(np.uint16)); O = FB.dev(np.concatenate([prev, np.zeros(-(-K // 24576) * 24576 - K, np.float32)]))
    k = OA.register_csrc("colmax_a16", Q.colmax_a16_src(rows // 12, K), ntasks=12)
    OA.csrc_call(k, O, A, FB.dev(Q.colmax_a16_descs())).realize()
    got = FB.host_read(O).view(np.float32)[:K]; want = np.maximum(prev, np.abs(a.astype(np.float32)).max(0))
    n = int((got != want).sum()); ok &= n == 0; print("colmax_a16 K %5d: %d / %d columns differ" % (K, n, K))
for W in (2 * 13824, 2 * 24576, 3 * 4608):
    p = rng.randint(-32768, 32768, (rows, W)).astype(np.int16); p[:, rng.choice(W, 50)] //= 1000
    P = FB.dev(np.concatenate([p.ravel(), np.zeros(rows * 8, np.int16)])); O = FB.zeros(-(-W // 24576) * 24576, dtypes.float32)
    k = OA.register_csrc("colmax_i16", Q.colmax_i16_src(rows, W), ntasks=12)
    OA.csrc_call(k, O, P, FB.dev(Q.colmax_i16_descs(rows, W))).realize()
    got = FB.host_read(O).view(np.float32)[:W]; want = np.abs(p.astype(np.float32)).max(0)
    n = int((got != want).sum()); ok &= n == 0; print("colmax_i16 W %5d: %d / %d columns differ" % (W, n, W))
print("ALL EXACT" if ok else "MISMATCH")

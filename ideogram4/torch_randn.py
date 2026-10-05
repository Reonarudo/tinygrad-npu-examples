#!/usr/bin/env python3
"""`torch.manual_seed(s); torch.randn(n)` (CPU, float32) in numpy, bit-for-bit: torch's CPU generator is an
MT19937 seeded exactly like numpy's legacy `RandomState(s)`; a float32 uniform takes 24 bits of one
32-bit draw; `normal_fill` fills the buffer with uniforms, then Box-Muller over blocks of 16 (element j
with j + 8), recomputing the last 16 from fresh uniforms when n % 16 != 0. Only n >= 16 (the vectorized
path the pipeline's latents take). Checked against the documented `manual_seed(0); randn(4, 4)`.
"""
import numpy as np


def randn(seed: int, n: int) -> np.ndarray:
    assert n >= 16
    rs = np.random.RandomState(seed)
    extra = 16 if n % 16 else 0
    raw = rs.randint(0, 2 ** 32, size=n + extra, dtype=np.uint64).astype(np.uint32)   # one 32-bit draw each
    u = ((raw & ((1 << 24) - 1)).astype(np.float32) * np.float32(2.0 ** -24)).astype(np.float32)
    data = u[:n].copy()

    def fill16(d):
        u1 = np.float32(1) - d[:8]; u2 = d[8:16]
        r = np.sqrt(np.float32(-2) * np.log(u1)).astype(np.float32); th = (np.float32(2.0 * np.pi) * u2).astype(np.float32)
        return np.concatenate([r * np.cos(th), r * np.sin(th)]).astype(np.float32)

    for i in range(0, n - 15, 16): data[i:i + 16] = fill16(data[i:i + 16])
    if extra: data[n - 16:] = fill16(u[n:n + 16])
    return data


if __name__ == "__main__":
    want = np.array([[-1.1258, -1.1524, -0.2506, -0.4339], [0.8487, 0.6920, -0.3160, -2.1152],
                     [0.3223, -1.2633, 0.3500, 0.3081], [0.1198, 1.2377, 1.1168, -0.2473]], np.float32)
    got = randn(0, 16).reshape(4, 4)
    print(got.round(4)); print("matches torch.manual_seed(0); torch.randn(4, 4):", np.allclose(got, want, atol=1e-4))

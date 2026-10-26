"""LFM2.5's row kernels between the GEMVs (one token, row 0; 4 tasks):
  lfm_conv     the short conv on the in_proj row y = b | c | g (each H): u = b g; v = sum_t w[t] ext[t] over ext = (state rows, u);
               yrow[0:H] = c v; then the state shifts (state[t] = state[t + 1], state[LC - 2] = u) -- in place, a task its own columns
  swiglu_rows  dh[0:N] = silu(g) u over two rows (the dense layers' unpacked gate / up rows)."""
import os, sys
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
from zy import vec_f16 as V                                              # noqa: E402
from gptoss_kernels import EXP, NT                                       # noqa: E402  (exp2_d4 / sig8)


def lfm_conv_src(H, LC):
  assert H % 32 == 0 and LC >= 2
  taps = " + ".join(f"*(__global float8*)(w + {t} * {H} + i) * *(__global float8*)(st + {t} * {H} + i)" for t in range(LC - 1))
  shift = "".join(f" *(__global float8*)(st + {t} * {H} + i) = *(__global float8*)(st + {t + 1} * {H} + i);" for t in range(LC - 2))
  return V.FULL_H + f"""
__kernel void lfm_conv(__global float* restrict yrow, __global float* restrict y, __global float* restrict st, __global float* restrict w, const int core_id) {{
  for (int i = core_id * 8; i < {H}; i += {8 * NT}) {{
    float8 u = *(__global float8*)(y + i) * *(__global float8*)(y + {2 * H} + i);
    float8 v = {taps} + *(__global float8*)(w + {(LC - 1) * H} + i) * u;
    *(__global float8*)(yrow + i) = *(__global float8*)(y + {H} + i) * v;{shift} *(__global float8*)(st + {(LC - 2) * H} + i) = u;
  }}
}}"""


def swiglu_rows_src(N):
  assert N % 32 == 0
  return V.FULL_H + EXP + f"""
__kernel void swiglu_rows(__global float* restrict dh, __global float* restrict g, __global float* restrict u, const int core_id) {{
  for (int i = core_id * 8; i < {N}; i += {8 * NT}) {{ float8 a = *(__global float8*)(g + i); *(__global float8*)(dh + i) = a * sig8(a) * *(__global float8*)(u + i); }}
}}"""

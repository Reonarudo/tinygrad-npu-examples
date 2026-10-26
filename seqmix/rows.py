"""Small row kernels around the qdot GEMVs (the generic tinygrad kernels for the same copies cost milliseconds each on the NPU):
  rowadd      out[0:H] = a[0:H] + b[0:H]            (row 0 of persistent row buffers; the residual after a GEMV's rows)
  qrows_gate  q_rows[h][0:HD] = y[h HD + .], q_rows[h][HD:2 HD] = gate   (the Qwen3.5-style [q | gate] rows for attn_dec2 from plain q rows)
4 tasks, plain loads: a few thousand floats each."""
import os, sys
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
from zy import vec_f16 as V                                              # noqa: E402

NT = 12                                                                   # the Kernels class launches 12 tasks (3 cores x 4 TECs): a stride of 96 floats; a non-idempotent kernel (lfm_conv) needs every column owned by one task


def rowadd_src(H):
  assert H % 32 == 0
  return V.FULL_H + f"""
__kernel void rowadd(__global float* restrict out, __global float* restrict a, __global float* restrict b, const int core_id) {{
  for (int i = core_id * 8; i < {H}; i += {8 * NT}) *(__global float8*)(out + i) = *(__global float8*)(a + i) + *(__global float8*)(b + i);
}}"""


def qrows_gate_src(NH, HD, gate=30.0):
  assert HD % 8 == 0
  return V.FULL_H + f"""
__kernel void qrows_gate(__global float* restrict q_rows, __global float* restrict y, const int core_id) {{
  for (int h = core_id; h < {NH}; h += {NT}) for (int d = 0; d < {HD}; d += 8) {{
    *(__global float8*)(q_rows + h * {2 * HD} + d) = *(__global float8*)(y + h * {HD} + d);
    *(__global float8*)(q_rows + h * {2 * HD} + {HD} + d) = BC({gate!r}f);
  }}
}}"""


def rowadd3_src(H):
  assert H % 32 == 0
  return V.FULL_H + f"""
__kernel void rowadd3(__global float* restrict out, __global float* restrict a, __global float* restrict b, __global float* restrict c, const int core_id) {{
  for (int i = core_id * 8; i < {H}; i += {8 * NT}) *(__global float8*)(out + i) = *(__global float8*)(a + i) + *(__global float8*)(b + i) + *(__global float8*)(c + i);
}}"""


def rowcat_src(n1, n2):
  """out[0:n1] = a[0:n1], out[n1:n1 + n2] = b[0:n2]."""
  assert n1 % 8 == 0 and n2 % 8 == 0
  return V.FULL_H + f"""
__kernel void rowcat(__global float* restrict out, __global float* restrict a, __global float* restrict b, const int core_id) {{
  for (int i = core_id * 8; i < {n1}; i += {8 * NT}) *(__global float8*)(out + i) = *(__global float8*)(a + i);
  for (int i = core_id * 8; i < {n2}; i += {8 * NT}) *(__global float8*)(out + {n1} + i) = *(__global float8*)(b + i);
}}"""


def rowslice_src(n, off):
  """out[0:n] = a[off:off + n] (a K-half of a long row for a GEMV whose package holds K <= 4096)."""
  assert n % 8 == 0 and off % 8 == 0
  return V.FULL_H + f"""
__kernel void rowcopy(__global float* restrict out, __global float* restrict a, const int core_id) {{
  for (int i = core_id * 8; i < {n}; i += {8 * NT}) *(__global float8*)(out + i) = *(__global float8*)(a + {off} + i);
}}"""

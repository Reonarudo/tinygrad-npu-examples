"""Gemma 4 26B-A4B's row kernels between the qdot GEMVs (one token, row 0; 12 tasks, every column owned by one task):
  g_pnres     out[0:H] = (x + rms(y) w) s                         (the post-norm residual; s a device float: 1 after the attention,
              layer_output_scale after the FFN)
  geglu_rows  h[k][0:F] = gelu_tanh(y[k][0:F]) y[k][F:2F], h[k][F:KP] = 0   (gate | up in one GEMV row of pitch NP; the experts' 8 heads
              or the dense FFN's one)
  g26_fin     out = x1 + rms(f) w3 s, f = rms(yf) w1 + rms(m) w2, m[i] = sum_k wt[k] es[sel[k] / S] y[k][i]   (the dense FFN's output
              yf and the experts' weighted sum, each post-normed, the sum post-normed again; every task redoes the three row sums)"""
import os, sys
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gptoss"))
from zy import vec_f16 as V                                              # noqa: E402
from gptoss_kernels import EXP, NT                                       # noqa: E402  (exp2_d4 / sig8; 12 tasks)

SQ2PI = 0.7978845608028654


def g_pnres_src(H, eps):
  assert H % 8 == 0
  return V.FULL_H + f"""
__kernel void g_pnres(__global float* restrict out, __global float* restrict x, __global float* restrict y, __global float* restrict w, __global float* restrict sv, const int core_id) {{
  float8 s = BC(0.0f); for (int i = 0; i < {H}; i += 8) {{ float8 v = *(__global float8*)(y + i); s += v * v; }}
  float8 inv = BC(1.0f / __builtin_sqrtf(hsum8(s) * {1.0 / H!r}f + {eps!r}f)), sc = BC(sv[0]);
  for (int i = core_id * 8; i < {H}; i += {8 * NT}) *(__global float8*)(out + i) = (*(__global float8*)(x + i) + *(__global float8*)(y + i) * inv * *(__global float8*)(w + i)) * sc;
}}"""


def geglu_rows_src(heads, F, NP, KP):
  assert F % 8 == 0 and KP % 8 == 0 and KP >= F
  return V.FULL_H + EXP + f"""
static inline float8 gelu8(float8 x) {{ float8 z = x * (BC(1.0f) + BC(0.044715f) * x * x); return x * sig8(BC({2 * SQ2PI!r}f) * z); }}
__kernel void geglu_rows(__global float* restrict h, __global float* restrict y, const int core_id) {{
  for (int k = 0; k < {heads}; k++) {{
    __global float* o = h + k * {KP}; __global float* g = y + k * {NP}; __global float* u = g + {F};
    for (int i = core_id * 8; i < {F}; i += {8 * NT}) *(__global float8*)(o + i) = gelu8(*(__global float8*)(g + i)) * *(__global float8*)(u + i);
    for (int i = {F} + core_id * 8; i < {KP}; i += {8 * NT}) *(__global float8*)(o + i) = BC(0.0f);
  }}
}}"""


def g26_fin_src(H, TOPK, NPH, S_dn, eps):
  assert H % 8 == 0
  msum = " + ".join(f"BC(wk[{k}]) * *(__global float8*)(y + {k} * {NPH} + i)" for k in range(TOPK))
  return V.FULL_H + f"""
__kernel void g26_fin(__global float* restrict out, __global float* restrict x1, __global float* restrict yf, __global float* restrict y, __global float* restrict wt,
                      __global int* restrict sel, __global float* restrict es, __global float* restrict w1, __global float* restrict w2, __global float* restrict w3,
                      __global float* restrict sv, const int core_id) {{
  float wk[{TOPK}]; for (int k = 0; k < {TOPK}; k++) wk[k] = wt[k] * es[sel[k] / {S_dn}];
  float8 sf = BC(0.0f), sm = BC(0.0f);
  for (int i = 0; i < {H}; i += 8) {{ float8 a = *(__global float8*)(yf + i); sf += a * a; float8 m = {msum}; sm += m * m; }}
  float8 invf = BC(1.0f / __builtin_sqrtf(hsum8(sf) * {1.0 / H!r}f + {eps!r}f)), invm = BC(1.0f / __builtin_sqrtf(hsum8(sm) * {1.0 / H!r}f + {eps!r}f));
  float8 s2 = BC(0.0f);
  for (int i = 0; i < {H}; i += 8) {{ float8 m = {msum}; float8 f = *(__global float8*)(yf + i) * invf * *(__global float8*)(w1 + i) + m * invm * *(__global float8*)(w2 + i); s2 += f * f; }}
  float8 inv2 = BC(1.0f / __builtin_sqrtf(hsum8(s2) * {1.0 / H!r}f + {eps!r}f)), sc = BC(sv[0]);
  for (int i = core_id * 8; i < {H}; i += {8 * NT}) {{
    float8 m = {msum}; float8 f = *(__global float8*)(yf + i) * invf * *(__global float8*)(w1 + i) + m * invm * *(__global float8*)(w2 + i);
    *(__global float8*)(out + i) = (*(__global float8*)(x1 + i) + f * inv2 * *(__global float8*)(w3 + i)) * sc;
  }}
}}"""

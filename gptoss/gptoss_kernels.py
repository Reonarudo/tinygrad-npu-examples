"""gpt-oss's row kernels between the qdot GEMVs (the generic tinygrad kernels for the same cost milliseconds each): one token, row 0.
  rms_pad    xn[0:H] = rms(x) w, xn[H:KP] = 0                                        (the GEMVs' padded input row)
  qkv_rope   q_f = rope(y_q + bq) [NH][HD], k_f = rope(y_k + bk) [NKV][HD], v_f = y_v + bv; the YaRN table's row `pos` (cs [TMAX][HD]: cos | sin)
  oai_glu    h[k][0:FF] = act(g[k] + bg[e_k], u[k] + bu[e_k]), h[k][FF:KP] = 0; act(a, u) = min(a, 7) sigmoid(1.702 min(a, 7)) (clamp(u, -7, 7) + 1);
             e_k = sel[k] / S (the expert of head k), bg / bu [E][FF]
  oss_comb   out[0:H] = x1 + bo-free: sum_k wt[k] (y[k][0:H] + bd[e_k]) with e_k = sel_dn[k] / S_dn            (the down biases weighted)
12 tasks (the Kernels class launch)."""
import os, sys
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
from zy import vec_f16 as V                                              # noqa: E402

NT = 12                                                                   # the Kernels class launches 12 tasks (3 cores x 4 TECs): a stride of 96 floats; a non-idempotent kernel (lfm_conv) needs every column owned by one task
EXP = """
static inline float8 exp2_d4(float8 t) {
  t = VMAX(t, BC(-126.0f)); t = VMIN(t, BC(126.0f));
  float8 k = VRINT(t); float8 f = t - k;
  float8 p = BC(9.618129e-3f);
  p = p * f + BC(5.550411e-2f); p = p * f + BC(2.402265e-1f); p = p * f + BC(6.931472e-1f); p = p * f + BC(1.0f);
  return VSCAL2(p, VI(k));
}
static inline float8 sig8(float8 t) { return VRCP(BC(1.0f) + exp2_d4(BC(-1.4426950408889634f) * t)); }
"""


def rms_pad_src(H, KP, eps):
  assert H % 8 == 0 and KP % 8 == 0
  return V.FULL_H + f"""
__kernel void rms_pad(__global float* restrict xn, __global float* restrict x, __global float* restrict w, const int core_id) {{
  float8 s = BC(0.0f); for (int i = 0; i < {H}; i += 8) {{ float8 v = *(__global float8*)(x + i); s += v * v; }}
  float8 inv = BC(1.0f / __builtin_sqrtf(hsum8(s) * {1.0 / H!r}f + {eps!r}f));
  for (int i = core_id * 8; i < {H}; i += {8 * NT}) *(__global float8*)(xn + i) = *(__global float8*)(x + i) * inv * *(__global float8*)(w + i);
  for (int i = {H} + core_id * 8; i < {KP}; i += {8 * NT}) *(__global float8*)(xn + i) = BC(0.0f);
}}"""


def qkv_rope_src(NH, NKV, HD):
  HR = HD // 2; assert HR % 8 == 0
  rope = lambda dst, src, b, h: f"""
    for (int i = 0; i < {HR}; i += 8) {{
      float8 x1 = *(__global float8*)({src} + {h} * {HD} + i) + *(__global float8*)({b} + {h} * {HD} + i), x2 = *(__global float8*)({src} + {h} * {HD} + {HR} + i) + *(__global float8*)({b} + {h} * {HD} + {HR} + i);
      float8 c = *(__global float8*)(cs + i), sn = *(__global float8*)(cs + {HR} + i);
      *(__global float8*)({dst} + {h} * {HD} + i) = x1 * c - x2 * sn; *(__global float8*)({dst} + {h} * {HD} + {HR} + i) = x2 * c + x1 * sn;
    }}"""
  return V.FULL_H + f"""
__kernel void qkv_rope(__global float* restrict q_f, __global float* restrict k_f, __global float* restrict v_f, __global float* restrict y_q, __global float* restrict y_k,
                       __global float* restrict y_v, __global float* restrict bq, __global float* restrict bk, __global float* restrict bv, __global float* restrict tab,
                       __global int* restrict idx, const int core_id) {{
  __global float* cs = tab + idx[1] * {HD};
  for (int h = core_id; h < {NH}; h += {NT}) {{{rope("q_f", "y_q", "bq", "h")}
  }}
  for (int h = core_id; h < {NKV}; h += {NT}) {{{rope("k_f", "y_k", "bk", "h")}
    for (int i = 0; i < {HD}; i += 8) *(__global float8*)(v_f + h * {HD} + i) = *(__global float8*)(y_v + h * {HD} + i) + *(__global float8*)(bv + h * {HD} + i);
  }}
}}"""


def oai_glu_src(TOPK, FF, NPF, KP, S, alpha=1.702, limit=7.0):
  assert FF % 8 == 0 and KP % 8 == 0
  return V.FULL_H + EXP + f"""
__kernel void oai_glu(__global float* restrict h, __global float* restrict yg, __global float* restrict yu, __global float* restrict bg, __global float* restrict bu,
                      __global int* restrict sel, const int core_id) {{
  for (int k = 0; k < {TOPK}; k++) {{
    int e = sel[k] / {S}; __global float* o = h + k * {KP};
    for (int i = core_id * 8; i < {FF}; i += {8 * NT}) {{
      float8 a = VMIN(*(__global float8*)(yg + k * {NPF} + i) + *(__global float8*)(bg + e * {FF} + i), BC({limit!r}f));
      float8 u = VMIN(VMAX(*(__global float8*)(yu + k * {NPF} + i) + *(__global float8*)(bu + e * {FF} + i), BC({-limit!r}f)), BC({limit!r}f));
      *(__global float8*)(o + i) = a * sig8(BC({alpha!r}f) * a) * (u + BC(1.0f));
    }}
    for (int i = {FF} + core_id * 8; i < {KP}; i += {8 * NT}) *(__global float8*)(o + i) = BC(0.0f);
  }}
}}"""


def oss_comb_src(TOPK, H, NPH, S_dn):
  return V.FULL_H + f"""
__kernel void oss_comb(__global float* restrict out, __global float* restrict x1, __global float* restrict y, __global float* restrict wt, __global float* restrict bd,
                       __global int* restrict sel, const int core_id) {{
  for (int i = core_id * 8; i < {H}; i += {8 * NT}) {{
    float8 acc = *(__global float8*)(x1 + i);
    for (int k = 0; k < {TOPK}; k++) {{ int e = sel[k] / {S_dn}; acc += BC(wt[k]) * (*(__global float8*)(y + k * {NPH} + i) + *(__global float8*)(bd + e * {H} + i)); }}
    *(__global float8*)(out + i) = acc;
  }}
}}"""

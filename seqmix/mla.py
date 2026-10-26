"""Multi-head latent attention (DeepSeek-V2 / GLM-4.7-Flash / Kimi Linear MLA layers) over the compressed cache, absorbed decode form.

Per layer the cache holds one row a position: r_t = [c_t (KVL floats: kv_a_layernorm(W_kv_a x_t)) | k_pe,t (RD floats, RoPE'd)].
Query head h brings q_lat,h = W_UK,h^T q_nope,h (KVL) and q_pe,h (RD, RoPE'd); its score at t is (q_lat,h . c_t + q_pe,h . k_pe,t) * scale
(scale = 1 / sqrt(NOPE + RD)), the softmax over t <= pos, and its latent output o_h = sum_t p_t c_t (KVL); out_h = W_UV,h o_h is a
GEMV outside (the absorptions are slot-selected GEMVs the backend has: one weight slot a head). So this is attention with ONE kv
head of width KVL + RD whose value is the row's first KVL floats, shared by all NH query heads.

  mla_part   decode (one query row). A unit = (a group of HG query heads, a slice of the positions 0..pos): the cache rows stream
             through LSRAM in BT-row blocks (double-buffered DMA), the softmax online (a block's max rescales the accumulators). The
             unit of the last slice also takes the new row `kv_new` (position pos): it scores it, and the head group 0's unit writes it
             into the cache. Record block of a unit: o [HG][KVL] then (max, sum) [HG][2], padded to 64-byte lines; part[(hg + c NHG)].
  mla_comb   the slices merged a head: o_lat [NH][KVL] = sum_c o_c 2^(m_c - m) / sum_c s_c 2^(m_c - m) (maxima in log2 units).
Contract: `mla_ref` (float64).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
from zy import vec_f16 as V                                              # noqa: E402
from mamba2 import HELPERS                                               # noqa: E402  (exp2_d4, F8)

NT = 12
LSRAM = 32768 - 64
LOG2E = 1.4426950408889634


def cfg(NH, KVL, RD, BTMAX=8):
  """-> (HG, BT, P): query heads a unit, cache rows a DMA block, position slices. The largest HG dividing NH whose q [HG][KVL + RD], o
  [HG][KVL] and two blocks of BT rows fit LSRAM with BT >= 2, then P so the units fill the 12 tasks."""
  RW = KVL + RD
  for HG in [h for h in range(min(NH, 4), 0, -1) if NH % h == 0]:
    for BT in range(BTMAX, 1, -1):
      if HG * RW * 4 + HG * KVL * 4 + 2 * BT * RW * 4 + RW * 4 + 256 <= LSRAM:
        nu = NH // HG; return HG, BT, (max(1, NT // nu) if nu < NT else 1)
  raise AssertionError((NH, KVL, RD))


def block_floats(HG, KVL): return HG * KVL + -(-2 * HG // 16) * 16


def mla_part_desc(NH, KVL, RD):
  """slots 0 .. BT-1: b + 1 contiguous cache rows (RW floats each); BT: the unit's HG q rows (contiguous, RW each); BT + 1: one row;
  BT + 2: a unit's record block."""
  HG, BT, P = cfg(NH, KVL, RD); RW = KVL + RD
  return V._desc_slots(*[((b + 1) * RW * 4,) for b in range(BT)], (HG * RW * 4,), (RW * 4,), (block_floats(HG, KVL) * 4,))


def mla_part_src(NH, KVL, RD, scale, TMAX):
  """args: part [NHG P][block], q [NH][KVL + RD] (q_lat | q_pe, both final), Cl [L][TMAX][RW] (the caches), kv_new [RW] (the row of
  position pos), idx (int32 [2]: layer, pos), desc (mla_part_desc)."""
  HG, BT, P = cfg(NH, KVL, RD); RW = KVL + RD; NHG = NH // HG; NU = NHG * P; BLK = block_floats(HG, KVL); KV8 = KVL // 8
  QS = 0; OS = QS + HG * RW * 4; ML = OS + HG * KVL * 4; KB = OS + BLK * 4; NR = KB + 2 * BT * RW * 4; END = NR + RW * 4
  assert END <= LSRAM and ML + 2 * HG * 4 <= KB, (END, ML, KB)
  SC = float(scale) * LOG2E
  block = lambda kb, bl: f"""
      for (int h = 0; h < {HG}; h++) {{
        __global float* q = qs + h * {RW}; float8 s8 = BC(-3.0e38f);
        for (int j = 0; j < {bl}; j++) {{
          float8 a = BC(0.0f); __global float* k = {kb} + j * {RW};
          for (int e = 0; e < {RW}; e += 8) a += *(__global float8*)(q + e) * *(__global float8*)(k + e);
          s8[j] = __builtin_aipu_rpadd_tfp32_pw(a, a, ALL)[0] * {SC!r}f;
        }}
        float bm = __builtin_aipu_rpmax_tfp32_pw(s8, s8, ALL)[0];
        __global float* o = os + h * {KVL};
        if (bm > mx[h]) {{
          float8 f = exp2_d4(BC(mx[h] - bm)); sm[h] *= f[0];
          for (int e = 0; e < {KVL}; e += 8) *(__global float8*)(o + e) *= f;
          mx[h] = bm;
        }}
        float8 pr = exp2_d4(s8 - BC(mx[h])); sm[h] += __builtin_aipu_rpadd_tfp32_pw(pr, pr, ALL)[0];
        for (int j = 0; j < {bl}; j++) {{
          float8 pb = BC(pr[j]); __global float* v = {kb} + j * {RW};
          for (int e = 0; e < {KVL}; e += 8) *(__global float8*)(o + e) += pb * *(__global float8*)(v + e);
        }}
      }}"""
  return V.FULL_H + HELPERS + f"""
__kernel void mla_part(__global float* restrict part, __global float* restrict q, __global float* restrict Cl, __global float* restrict kv_new,
                       __global int* restrict idx, __global int* restrict desc, const int core_id) {{
  int L = idx[0], pos = idx[1]; __global float* C = Cl + L * {TMAX * RW};
  __global float* qs = LSF({QS}); __global float* os = LSF({OS}); __global float* ml = LSF({ML}); __global float* nr = LSF({NR});
  for (int u = core_id; u < {NU}; u += {NT}) {{
    int hg = u / {P}, c = u % {P}; int last = c == {P - 1};
    int t0 = c * pos / {P}, t1 = (c + 1) * pos / {P};                          /* the cache rows 0 .. pos - 1; the last slice adds pos */
    DMA_FILL(2, DESC(desc, {BT}), {QS}, (int)(q + hg * {HG * RW}));
    if (last) DMA_FILL(3, DESC(desc, {BT + 1}), {NR}, (int)kv_new);
    DMA_WAIT(2); if (last) DMA_WAIT(3);
    float mx[{HG}], sm[{HG}];
    for (int i = 0; i < {HG}; i++) {{ mx[i] = -3.0e38f; sm[i] = 0.0f; }}
    for (int i = 0; i < {HG * KVL}; i += 8) *(__global float8*)(os + i) = BC(0.0f);
    #define BLEN(t) ({BT} < t1 - (t) ? {BT} : t1 - (t))
    #define ISSUE(t, buf) {{ int bl_ = BLEN(t); DMA_FILL(buf, DESC(desc, bl_ - 1), {KB} + (buf) * {BT * RW * 4}, (int)(C + (t) * {RW})); }}
    int t = t0, p = 0;
    if (t < t1) ISSUE(t, 0);
    while (t < t1) {{
      int bl = BLEN(t), tn = t + bl;
      if (tn < t1) {{ if (p == 0) ISSUE(tn, 1) else ISSUE(tn, 0) }}
      if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
      __global float* kb = LSF({KB} + p * {BT * RW * 4});{block("kb", "bl")}
      t = tn; p ^= 1;
    }}
    if (last) {{
      {block("nr", 1)}
      if (hg == 0) {{ DMA_DRAIN(3, DESC(desc, {BT + 1}), {NR}, (int)(C + pos * {RW})); DMA_WAIT(3); }}
    }}
    for (int i = 0; i < {HG}; i++) {{ ml[2 * i] = mx[i]; ml[2 * i + 1] = sm[i]; }}
    DMA_DRAIN(2, DESC(desc, {BT + 2}), {OS}, (int)(part + (hg + c * {NHG}) * {BLK})); DMA_WAIT(2);
    #undef ISSUE
    #undef BLEN
  }}
  DMA_WAIT_ALL();
}}"""


def mla_comb_src(NH, KVL, RD):
  """args: o_lat [NH][KVL], part (mla_part's records). A unit = a head (its P slices merged, the sum divided out)."""
  HG, BT, P = cfg(NH, KVL, RD); NHG = NH // HG; BLK = block_floats(HG, KVL)
  return V.FULL_H + HELPERS + f"""
__kernel void mla_comb(__global float* restrict o_lat, __global float* restrict part, const int core_id) {{
  for (int h = core_id; h < {NH}; h += {NT}) {{
    int hg = h / {HG}, hh = h % {HG}; float m = -3.0e38f;
    for (int c = 0; c < {P}; c++) {{ float mc = part[(hg + c * {NHG}) * {BLK} + {HG * KVL} + 2 * hh]; m = mc > m ? mc : m; }}
    float s = 0.0f; float8 f[{P}];
    for (int c = 0; c < {P}; c++) {{
      __global float* rec = part + (hg + c * {NHG}) * {BLK};
      f[c] = exp2_d4(BC(rec[{HG * KVL} + 2 * hh] - m)); s += rec[{HG * KVL} + 2 * hh + 1] * f[c][0];
    }}
    float8 inv = BC(1.0f / s);
    for (int e = 0; e < {KVL}; e += 8) {{
      float8 acc = BC(0.0f);
      for (int c = 0; c < {P}; c++) acc += *(__global float8*)(part + (hg + c * {NHG}) * {BLK} + hh * {KVL} + e) * f[c];
      *(__global float8*)(o_lat + h * {KVL} + e) = acc * inv;
    }}
  }}
}}"""


# ---- the Contract (float64)
def mla_ref(q, C, kv_new, pos, scale, KVL):
  """q [NH][KVL + RD], C [T][KVL + RD] (rows 0 .. pos - 1 used), kv_new [KVL + RD] (position pos). -> (o_lat [NH][KVL], C' with row pos)."""
  q = q.astype(np.float64); C = C.astype(np.float64).copy(); C[pos] = kv_new.astype(np.float64)
  rows = C[:pos + 1]; s = (q @ rows.T) * scale; s = s - s.max(1, keepdims=True); p = np.exp(s); p /= p.sum(1, keepdims=True)
  return p @ rows[:, :KVL], C

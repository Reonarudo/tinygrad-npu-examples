"""Decode GQA attention over fp32 caches with attention sinks and a sliding window (gpt-oss), the general kernel CCA's core also uses.

Cache layout a layer: K [TMAX][NKV][HD], V [TMAX][NKV][HD] (position-major: the DMA fetches b + 1 rows of one kv head with a pitch
of NKV HD floats). The query row q [NH][HD] is final (normed, RoPE'd); scale is applied to the scores in log2 units.
Sinks (gpt-oss): the softmax denominator carries exp(s_h) for a per-head learned logit s_h: slice 0 starts its online softmax at
max = s_h log2e, sum = 1, o = 0 (the other slices at -inf / 0), so the merge counts it once. Window W > 0 (gpt-oss 128): only the
positions pos - W + 1 .. pos are visible.

  attn_part  a unit = (kv head, group of HG query heads, slice of the visible positions); the unit of the last slice also scores
             the new row k_new / v_new [NKV][HD] (position pos) and, for head group 0, writes it into the caches.
  attn_comb  the slices merged a head -> o [NH][HD].
Contract: `attn_ref` (float64).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
from zy import vec_f16 as V                                              # noqa: E402
from mamba2 import HELPERS                                               # noqa: E402

NT = 12
LSRAM = 32768 - 64
LOG2E = 1.4426950408889634


def cfg(NH, NKV, HD, BTMAX=8):
  """-> (HG, BT, P): the largest HG (a divisor of the group) whose q / o and two K + two V blocks of BT rows fit, then P slices."""
  g = NH // NKV
  for HG in [h for h in range(min(g, 8), 0, -1) if g % h == 0]:
    for BT in range(BTMAX, 1, -1):
      if 2 * HG * HD * 4 + 4 * BT * HD * 4 + 2 * HD * 4 + 256 <= LSRAM:
        nu = NKV * (g // HG); return HG, BT, (max(1, NT // nu) if nu < NT else 1)
  raise AssertionError((NH, NKV, HD))


def block_floats(HG, HD): return HG * HD + -(-2 * HG // 16) * 16


def attn_part_desc(NH, NKV, HD):
  """slots 0 .. BT-1: b + 1 rows of one kv head (pitch NKV HD); BT: HG q rows (contiguous); BT + 1: one HD row; BT + 2: a record block."""
  HG, BT, P = cfg(NH, NKV, HD)
  return V._desc_slots(*[((b + 1) * HD * 4, HD * 4, NKV * HD * 4, HD * 4) for b in range(BT)], (HG * HD * 4,), (HD * 4,), (block_floats(HG, HD) * 4,))


def attn_part_src(NH, NKV, HD, scale, TMAX, window=0, sinks=False):
  """args: part [NKV NHG P][block], q [NH][HD], Kl [L][TMAX][NKV][HD], Vl (same), k_new [NKV][HD], v_new [NKV][HD], sink [L][NH] (raw logits;
  any buffer when sinks=False), idx (int32 [2]: layer, pos), desc (attn_part_desc)."""
  HG, BT, P = cfg(NH, NKV, HD); g = NH // NKV; NHG = g // HG; NU = NKV * NHG * P; BLK = block_floats(HG, HD)
  QS = 0; OS = QS + HG * HD * 4; ML = OS + HG * HD * 4; KB = OS + BLK * 4; VB = KB + BT * HD * 4 * 2; KN = VB + BT * HD * 4 * 2; VN = KN + HD * 4; END = VN + HD * 4
  assert END <= LSRAM and ML + 2 * HG * 4 <= KB, (END,)
  SC = float(scale) * LOG2E
  block = lambda kb, vb, bl: f"""
      for (int h = 0; h < {HG}; h++) {{
        __global float* q = qs + h * {HD}; float8 s8 = BC(-3.0e38f);
        for (int j = 0; j < {bl}; j++) {{
          float8 a = BC(0.0f); __global float* k = {kb} + j * {HD};
          for (int e = 0; e < {HD}; e += 8) a += *(__global float8*)(q + e) * *(__global float8*)(k + e);
          s8[j] = __builtin_aipu_rpadd_tfp32_pw(a, a, ALL)[0] * {SC!r}f;
        }}
        float bm = __builtin_aipu_rpmax_tfp32_pw(s8, s8, ALL)[0];
        __global float* o = os + h * {HD};
        if (bm > mx[h]) {{
          float8 f = exp2_d4(BC(mx[h] - bm)); sm[h] *= f[0];
          for (int e = 0; e < {HD}; e += 8) *(__global float8*)(o + e) *= f;
          mx[h] = bm;
        }}
        float8 pr = exp2_d4(s8 - BC(mx[h])); sm[h] += __builtin_aipu_rpadd_tfp32_pw(pr, pr, ALL)[0];
        for (int j = 0; j < {bl}; j++) {{
          float8 pb = BC(pr[j]); __global float* v = {vb} + j * {HD};
          for (int e = 0; e < {HD}; e += 8) *(__global float8*)(o + e) += pb * *(__global float8*)(v + e);
        }}
      }}"""
  init = (f"if (c == 0) {{ mx[i] = sink[L * {NH} + kh * {g} + hg * {HG} + i] * {LOG2E!r}f; sm[i] = 1.0f; }} else {{ mx[i] = -3.0e38f; sm[i] = 0.0f; }}"
          if sinks else "mx[i] = -3.0e38f; sm[i] = 0.0f;")
  lo = f"int lo = pos - {window - 1}; if (lo < 0) lo = 0;" if window else "int lo = 0;"
  return V.FULL_H + HELPERS + f"""
__kernel void attn_part(__global float* restrict part, __global float* restrict q, __global float* restrict Kl, __global float* restrict Vl,
                        __global float* restrict k_new, __global float* restrict v_new, __global float* restrict sink, __global int* restrict idx,
                        __global int* restrict desc, const int core_id) {{
  int L = idx[0], pos = idx[1]; __global float* Kc = Kl + L * {TMAX * NKV * HD}; __global float* Vc = Vl + L * {TMAX * NKV * HD};
  __global float* qs = LSF({QS}); __global float* os = LSF({OS}); __global float* ml = LSF({ML});
  {lo}
  for (int u = core_id; u < {NU}; u += {NT}) {{
    int kh = u / {NHG * P}, hg = (u / {P}) % {NHG}, c = u % {P}; int last = c == {P - 1};
    int n = pos - lo, t0 = lo + c * n / {P}, t1 = lo + (c + 1) * n / {P};          /* cached rows lo .. pos - 1; the last slice adds pos */
    DMA_FILL(2, DESC(desc, {BT}), {QS}, (int)(q + (kh * {g} + hg * {HG}) * {HD}));
    if (last) {{ DMA_FILL(0, DESC(desc, {BT + 1}), {KN}, (int)(k_new + kh * {HD})); DMA_FILL(1, DESC(desc, {BT + 1}), {VN}, (int)(v_new + kh * {HD})); }}
    DMA_WAIT(2); if (last) {{ DMA_WAIT(0); DMA_WAIT(1); }}
    float mx[{HG}], sm[{HG}];
    for (int i = 0; i < {HG}; i++) {{ {init} }}
    for (int i = 0; i < {HG * HD}; i += 8) *(__global float8*)(os + i) = BC(0.0f);
    #define BLEN(t) ({BT} < t1 - (t) ? {BT} : t1 - (t))
    #define ISSUE(t, buf) {{ int bl_ = BLEN(t); DMA_FILL(buf, DESC(desc, bl_ - 1), {KB} + (buf) * {BT * HD * 4}, (int)(Kc + ((t) * {NKV} + kh) * {HD})); \\
                            DMA_FILL((buf) + 2, DESC(desc, bl_ - 1), {VB} + (buf) * {BT * HD * 4}, (int)(Vc + ((t) * {NKV} + kh) * {HD})); }}
    int t = t0, p = 0;
    if (t < t1) ISSUE(t, 0);
    while (t < t1) {{
      int bl = BLEN(t), tn = t + bl;
      if (tn < t1) {{ if (p == 0) ISSUE(tn, 1) else ISSUE(tn, 0) }}
      if (p == 0) {{ DMA_WAIT(0); DMA_WAIT(2); }} else {{ DMA_WAIT(1); DMA_WAIT(3); }}
      __global float* kb = LSF({KB} + p * {BT * HD * 4}); __global float* vb = LSF({VB} + p * {BT * HD * 4});{block("kb", "vb", "bl")}
      t = tn; p ^= 1;
    }}
    if (last) {{
      {block(f"LSF({KN})", f"LSF({VN})", 1)}
      if (hg == 0) {{
        DMA_DRAIN(0, DESC(desc, {BT + 1}), {KN}, (int)(Kc + (pos * {NKV} + kh) * {HD})); DMA_DRAIN(1, DESC(desc, {BT + 1}), {VN}, (int)(Vc + (pos * {NKV} + kh) * {HD}));
        DMA_WAIT(0); DMA_WAIT(1);
      }}
    }}
    for (int i = 0; i < {HG}; i++) {{ ml[2 * i] = mx[i]; ml[2 * i + 1] = sm[i]; }}
    DMA_DRAIN(2, DESC(desc, {BT + 2}), {OS}, (int)(part + ((kh * {NHG} + hg) + c * {NKV * NHG}) * {BLK})); DMA_WAIT(2);
    #undef ISSUE
    #undef BLEN
  }}
  DMA_WAIT_ALL();
}}"""


def attn_comb_src(NH, NKV, HD):
  """args: o [NH][HD], part. A unit = a head."""
  HG, BT, P = cfg(NH, NKV, HD); g = NH // NKV; NHG = g // HG; BLK = block_floats(HG, HD)
  return V.FULL_H + HELPERS + f"""
__kernel void attn_comb(__global float* restrict o, __global float* restrict part, const int core_id) {{
  for (int h = core_id; h < {NH}; h += {NT}) {{
    int kh = h / {g}, hg = (h % {g}) / {HG}, hh = h % {HG}; int rb = kh * {NHG} + hg; float m = -3.0e38f;
    for (int c = 0; c < {P}; c++) {{ float mc = part[(rb + c * {NKV * NHG}) * {BLK} + {HG * HD} + 2 * hh]; m = mc > m ? mc : m; }}
    float s = 0.0f; float8 f[{P}];
    for (int c = 0; c < {P}; c++) {{
      __global float* rec = part + (rb + c * {NKV * NHG}) * {BLK};
      f[c] = exp2_d4(BC(rec[{HG * HD} + 2 * hh] - m)); s += rec[{HG * HD} + 2 * hh + 1] * f[c][0];
    }}
    float8 inv = BC(1.0f / s);
    for (int e = 0; e < {HD}; e += 8) {{
      float8 acc = BC(0.0f);
      for (int c = 0; c < {P}; c++) acc += *(__global float8*)(part + (rb + c * {NKV * NHG}) * {BLK} + hh * {HD} + e) * f[c];
      *(__global float8*)(o + h * {HD} + e) = acc * inv;
    }}
  }}
}}"""


# ---- the Contract (float64)
def attn_ref(q, K, Vc, k_new, v_new, pos, scale, NKV, window=0, sink=None):
  """q [NH][HD], K / Vc [T][NKV][HD] (rows < pos used), k_new / v_new [NKV][HD] (position pos), sink [NH] or None. -> (o [NH][HD], K', V')."""
  NH, HD = q.shape; g = NH // NKV; K = K.astype(np.float64).copy(); Vc = Vc.astype(np.float64).copy(); K[pos] = k_new; Vc[pos] = v_new
  lo = max(0, pos - window + 1) if window else 0; o = np.zeros((NH, HD))
  for h in range(NH):
    kh = h // g; ks = K[lo:pos + 1, kh]; vs = Vc[lo:pos + 1, kh]; s = ks @ q[h].astype(np.float64) * scale
    m = s.max() if sink is None else max(s.max(), float(sink[h])); p = np.exp(s - m); den = p.sum() + (np.exp(float(sink[h]) - m) if sink is not None else 0.0)
    o[h] = (p @ vs) / den
  return o, K, Vc

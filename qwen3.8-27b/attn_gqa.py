"""The verify / draft attention in kv-head units: `attn_gqa_src` builds a drop-in for attn_part / attn_partt (the same
arguments, the same part records [NH][M][P][HD + 16], byte for byte) whose units share each K / V block across the g = NH / NKV
query heads and the M rows of a kv head, and `attn_comb2_src` a DMA-staged attn_comb (the same o_rows, byte for byte).

Why. attn_part's unit is (query head, position slice): each kv head's K / V rows cross the DDR port once per query head (6x on the
27B), its dots are single-accumulator fma chains (4-cycle latency: a quarter rate) with a scalar 8-lane sum per score, and its q
rows, the new rows' k / v and the cache rows it writes go through the data cache (a serial miss each). Here:
  - a unit is (kv head h_kv, position slice c, a run of the slice's records j0..j1), a record = (query head, row) of that kv head
    (g M of them a slice); the records of all units are spread evenly over NTE tasks (a task takes a contiguous range of the
    NKV P g M record-slices, cut at slice boundaries and at what LSRAM holds), so a slice's K / V rows cross the port once per
    task that holds records of it (~2x on the 27B at 4 rows, 1x on Ornith) instead of g times;
  - pass 1 takes two positions x eight records a step: sixteen independent fma chains, each record's dot accumulated in
    attn_part's order (one float8 accumulator over the HD / 8 chunks, i ascending); the eight lanes of eight accumulators are then
    transposed and summed lane 0 + 1 + ... + 7, left to right -- hsum8's order -- so every score is attn_part's bit for bit;
  - the scores are kept transposed (a float8 per position = eight records), so each record's max is a lane-wise max (exact in any
    order) and its sum is attn_part's: lane class l (positions t0 + l + 8k) accumulated over k ascending, then the classes summed
    0..7. attn_part's chunks past the slice (its CL positions, -3e38 each) add exp2_d4((-3e38 - max) log2 e) per lane; they are
    added here as that same value, the same number of times in the same order -- no exp for them;
  - pass 2 keeps 2 records x 64 dims of o in registers over a K / V block (the fma order per element is attn_part's, t ascending);
  - the q rows (per head run, one 2-D request), the new rows' k / v and the records (one 2-D drain per head run) move by DMA;
    the M new rows go into the cache by DMA (units (row, kv head) on the tasks with the least attention work).
`tree`: attn_part_tree_src / attn_part_rg_src(tree=True)'s rule (row r at position pos + depth[r], attending the cache and the new
rows set in anc[r]); the 12th argument `tree` (spec_tree.table). Any M <= 12.
"""
import numpy as np


def _K():
  import qwen38_kernels as K                                              # (lazy: qwen38_kernels imports this module for its wiring)
  return K


NEG = "(-3.0e38f)"


def _norm_rope_c(HD, ROT, eps):
  """attn_part's norm_rope, character for character (the generated code, hence the rounding, must be the same)."""
  HR = ROT // 2
  return f"""
static inline __attribute__((always_inline)) void norm_rope(__global float* restrict dst, __global float* restrict src, __global float* restrict w1, __global float* restrict cs) {{
  float8 acc = BC(0.0f);
  for (int i = 0; i < {HD}; i += 8) {{ float8 v = *(__global float8*)(src + i); acc += v * v; }}
  float8 inv = BC(1.0f / __builtin_sqrtf(hsum8(acc) * {1.0 / HD!r}f + {eps!r}f));
  for (int i = 0; i < {HD}; i += 8) *(__global float8*)(dst + i) = *(__global float8*)(src + i) * inv * *(__global float8*)(w1 + i);
  float8 x[{ROT // 8}];
  for (int v = 0; v < {ROT // 8}; v++) x[v] = *(__global float8*)(dst + 8 * v);
  for (int v = 0; v < {HR // 8}; v++) {{
    *(__global float8*)(dst + 8 * v) = x[v] * *(__global float8*)(cs + 8 * v) - x[v + {HR // 8}] * *(__global float8*)(cs + {ROT} + 8 * v);
    *(__global float8*)(dst + {HR} + 8 * v) = x[v + {HR // 8}] * *(__global float8*)(cs + {HR} + 8 * v) + x[v] * *(__global float8*)(cs + {ROT} + {HR} + 8 * v);
  }}
}}
"""


# the 8 x 8 transpose of eight accumulators a0..a7 (three stages of zipl / ziph, which the compiler emits as single instructions) and
# the left-to-right sum of the transposed rows: lane l = a_l[0] + a_l[1] + ... + a_l[7], hsum8's order
_HSUMT = """
#define ZL(a, b) __builtin_shufflevector((a), (b), 0, 8, 1, 9, 2, 10, 3, 11)
#define ZH(a, b) __builtin_shufflevector((a), (b), 4, 12, 5, 13, 6, 14, 7, 15)
#define LANE(v, l) __builtin_shufflevector((v), (v), l, l, l, l, l, l, l, l)
static inline __attribute__((always_inline)) float8 hsumT(float8 a0, float8 a1, float8 a2, float8 a3, float8 a4, float8 a5, float8 a6, float8 a7) {
  float8 b0 = ZL(a0, a4), b1 = ZH(a0, a4), b2 = ZL(a1, a5), b3 = ZH(a1, a5), b4 = ZL(a2, a6), b5 = ZH(a2, a6), b6 = ZL(a3, a7), b7 = ZH(a3, a7);
  float8 c0 = ZL(b0, b4), c1 = ZH(b0, b4), c2 = ZL(b1, b5), c3 = ZH(b1, b5), c4 = ZL(b2, b6), c5 = ZH(b2, b6), c6 = ZL(b3, b7), c7 = ZH(b3, b7);
  float8 t0 = ZL(c0, c4), t1 = ZH(c0, c4), t2 = ZL(c1, c5), t3 = ZH(c1, c5), t4 = ZL(c2, c6), t5 = ZH(c2, c6), t6 = ZL(c3, c7), t7 = ZH(c3, c7);
  return ((((((t0 + t1) + t2) + t3) + t4) + t5) + t6) + t7;
}
"""

L8 = range(8)


def _p1(HD, two):
  """C: the dots of the octet's eight records (q chunk-major at `qb`: chunk i of lane l at qb + (8 i + l) 8) with the k row(s) at
  kr0 (and kr1 when `two`), each accumulated over the chunks in order -> x0..x7 (y0..y7); i unrolled by two (immediate offsets)."""
  d = "; ".join(f"float8 x{l} = BC(0.0f)" + (f", y{l} = BC(0.0f)" if two else "") for l in L8) + ";"
  st = []
  for h in (0, 1):
    st.append(f"float8 k{2 * h} = *(__global float8*)(ka + {8 * h})" + (f", k{2 * h + 1} = *(__global float8*)(kc + {8 * h});" if two else ";"))
    st += [f"{{ float8 q = *(__global float8*)(qq + {64 * h + 8 * l}); x{l} += q * k{2 * h};" + (f" y{l} += q * k{2 * h + 1};" if two else "") + " }" for l in L8]
  return (f"{d} {{ __global float* qq = qb; __global float* ka = kr0;" + (" __global float* kc = kr1;" if two else "") +
          f" for (int i = 0; i < {HD // 8}; i += 2) {{ " + " ".join(st) + " qq += 128; ka += 16;" + (" kc += 16;" if two else "") + " } }")


def _p2_pair(HD, la, lb):
  """C: o (records-major at oa / ob) += e(t) v(t) for the block's nt rows at vb (pitch HD), the e of lanes la / lb of the score
  octet rows at pe (pitch 8): two records x 64 dims in registers, t in order."""
  ld = " ".join(f"float8 a{k} = *(__global float8*)(oa + d + {8 * k}), b{k} = *(__global float8*)(ob + d + {8 * k});" for k in L8)
  fm = " ".join(f"{{ float8 vv = *(__global float8*)(vr + {8 * k}); a{k} += sa * vv; b{k} += sb * vv; }}" for k in L8)
  st = " ".join(f"*(__global float8*)(ob + d + {8 * k}) = b{k}; *(__global float8*)(oa + d + {8 * k}) = a{k};" for k in L8)
  return (f"for (int d = 0; d < {HD}; d += 64) {{ {ld} for (int j = 0; j < nt; j++) {{ float8 ev = *(__global float8*)(pe + j * 8); "
          f"float8 sa = LANE(ev, {la}), sb = LANE(ev, {lb}); __global float* vr = vb + j * {HD} + d; {fm} }} {st} }}")


def _p2_pair_full(HD, la, lb, BT):
  """_p2_pair for a whole block (nt == BT), the BT positions unrolled, one base pointer per position advanced 64 dims a step (every
  load at an immediate offset < 512 B)."""
  ld = " ".join(f"float8 a{k} = *(__global float8*)(pa_ + {8 * k}), b{k} = *(__global float8*)(pb_ + {8 * k});" for k in L8)
  ev = " ".join(f"float8 e{j} = *(__global float8*)(pe + {8 * j}); float8 sa{j} = LANE(e{j}, {la}), sb{j} = LANE(e{j}, {lb});" for j in range(BT))
  body = " ".join(f"{{ float8 vv = *(__global float8*)(w{j} + {8 * k}); a{k} += sa{j} * vv; b{k} += sb{j} * vv; }}" for j in range(BT) for k in L8)
  st = " ".join(f"*(__global float8*)(pb_ + {8 * k}) = b{k}; *(__global float8*)(pa_ + {8 * k}) = a{k};" for k in L8)
  ptr = " ".join(f"__global float* w{j} = vb + {j * HD}; __asm__ volatile(\"\" : \"+r\"(w{j}));" for j in range(BT))
  adv = " ".join(f"w{j} += 64;" for j in range(BT))
  return (f"{{ {ev} {ptr} __global float* pa_ = oa; __global float* pb_ = ob; "
          f"__asm__ volatile(\"\" : \"+r\"(pa_), \"+r\"(pb_)); for (int d = 0; d < {HD // 64}; d++) {{ {ld} {body} {st} pa_ += 64; pb_ += 64; {adv} }} }}")


def _p2_one(HD):
  """C: o (at oa) += sa v (at vr) over HD, 64 dims a step."""
  return (f"for (int d = 0; d < {HD}; d += 64) {{ " + " ".join(f"float8 a{k} = *(__global float8*)(oa + d + {8 * k});" for k in L8) + " " +
          " ".join(f"a{k} += sa * *(__global float8*)(vr + d + {8 * k});" for k in L8) + " " + " ".join(f"*(__global float8*)(oa + d + {8 * k}) = a{k};" for k in L8) + " }")


def gqa_layout(NH, NKV, HD, TMAX, M, BT):
  """LSRAM byte offsets: K / V blocks (double-buffered; also the q staging), the normed new k, a raw row, then the records."""
  KB0 = 0; KN = KB0 + 2 * BT * HD * 4; KS = KN + HD * 4; REC = KS + HD * 4; AV = 32768 - 64 - REC
  return dict(KB0=KB0, KN=KN, KS=KS, REC=REC, AV=AV, QST=2 * BT)


def attn_gqa_desc(NH, NKV, HD, TMAX, M, BT=None):
  K = _K(); BT = K.ATT_BT if BT is None else BT; P = K.attn_parts(NH, HD, TMAX); PW = HD + 16; QR = NH * 2 * HD
  lay = gqa_layout(NH, NKV, HD, TMAX, M, BT)
  sl = [(BT * HD * 4, HD * 4, NKV * HD * 4, HD * 4), (HD * 4,)]
  sl += [(n * HD * 4, HD * 4, QR * 4, HD * 4) for n in range(1, lay["QST"] + 1)]           # q rows of one head (pitch QR)
  sl += [(n * PW * 4, PW * 4, P * PW * 4, PW * 4) for n in range(1, M + 1)]                 # records of one head (pitch P PW)
  return K.V._desc_slots(*sl)


def attn_gqa_src(NH, NKV, HD, TMAX, ROT, eps, M, tree=False, BT=None, NTE=None, name=None):
  """See the module docstring. args: part, q_rows, kv_rows, qnw_all, knw_all, rope, Kall, Vall, idx (layer, pos), desc
  (attn_gqa_desc)[, tree]. NTE: the tasks that take attention records (default 12; the rest only write cache rows)."""
  K = _K(); BT = K.ATT_BT if BT is None else BT; NT = K.NT; NTE = NT if NTE is None else NTE
  g = NH // NKV; P = K.attn_parts(NH, HD, TMAX); CL = K._attn_cl(TMAX, P); NCH = CL // 8; G = g * M
  scale = 1.0 / HD ** 0.5; LOG2E = 1.4426950408889634; QR, KR = NH * 2 * HD, 2 * NKV * HD; PW = HD + 16
  assert 1 <= M <= 12 and HD % 64 == 0 and BT % 2 == 0 and CL % 8 == 0 and 1 <= NTE <= NT
  lay = gqa_layout(NH, NKV, HD, TMAX, M, BT); KB0, KN, KS, REC, AV, QST = (lay[k] for k in ("KB0", "KN", "KS", "REC", "AV", "QST"))
  QD, RD = 2, 2 + QST                                                         # desc slots: q runs (n rows: QD + n - 1), record drains (RD + n - 1)
  RP = (lambda r: f"(pos + tree[{r}])") if tree else (lambda r: f"(pos + ({r}))")                    # row r's position (its RoPE)
  RPN = (lambda t: f"(pos + tree[({t}) - pos])") if tree else (lambda t: f"({t})")                  # new row t's RoPE position
  VIS = (lambda r, t: f"((tree[{M} + ({r})] >> (({t}) - pos)) & 1)") if tree else (lambda r, t: f"(({t}) <= pos + ({r}))")
  kname = name or ("attn_partt" if tree else "attn_part")
  xs = ", ".join(f"x{l}" for l in L8); ys = ", ".join(f"y{l}" for l in L8)
  sm_decl = "; ".join(f"float8 S{l} = BC(0.0f)" for l in L8) + ";"
  ex = lambda l: f"float8 e = exp2_d4((*(__global float8*)(so + (8 * k + {l}) * 8) - mx) * BC({LOG2E!r}f)); *(__global float8*)(so + (8 * k + {l}) * 8) = e; S{l} += e;"
  sm_full = " ".join(f"{{ {ex(l)} }}" for l in L8)
  sm_part = " ".join(f"if (8 * k + {l} < n) {{ {ex(l)} }} else S{l} += ep;" for l in L8)
  sm_pad = " ".join(f"S{l} += ep;" for l in L8)
  p2 = " ".join(f"if (o * 8 + {2 * p} < R) {{ __global float* oa = QO + (o * 8 + {2 * p}) * {PW}; __global float* ob = o * 8 + {2 * p + 1} < R ? oa + {PW} : LSF({KN}); "
                + f"if (nt == {BT}) {{ {_p2_pair_full(HD, 2 * p, 2 * p + 1, BT)} }} else {{ {_p2_pair(HD, 2 * p, 2 * p + 1)} }} }}" for p in range(4))
  return K.V.FULL_H + K.EXP + _norm_rope_c(HD, ROT, eps) + _HSUMT + f"""
__kernel void {kname}(__global float* restrict part, __global float* restrict q_rows, __global float* restrict kv_rows,
                        __global float* restrict qnw_all, __global float* restrict knw_all, __global float* restrict rope,
                        __global float* restrict Kall, __global float* restrict Vall, __global int* restrict idx, __global int* restrict desc, {"__global int* restrict tree, " if tree else ""}const int core_id) {{
  int L = idx[0], pos = idx[1], T = pos + {M};
  __global float* Kc = Kall + (L * {TMAX}) * {NKV * HD}; __global float* Vc = Vall + (L * {TMAX}) * {NKV * HD};
  __global float* qnw = qnw_all + L * {HD}; __global float* knw = knw_all + L * {HD};
  /* phase 0: the M new k / v rows into the cache by DMA, a unit = (row, kv head), from the last task down (they hold the fewest records) */
  for (int u = {NT - 1} - core_id; u < {NKV * M}; u += {NT}) {{
    int r = u / {NKV}, kh = u % {NKV};
    DMA_FILL(2, DESC(desc, 1), {KS}, (int)(kv_rows + r * {KR} + kh * {HD}));
    DMA_FILL(3, DESC(desc, 1), {KB0}, (int)(kv_rows + r * {KR} + {NKV * HD} + kh * {HD}));
    DMA_WAIT(2); norm_rope(LSF({KN}), LSF({KS}), knw, rope + {RP("r")} * {2 * ROT});
    DMA_DRAIN(2, DESC(desc, 1), {KN}, (int)(Kc + ((pos + r) * {NKV} + kh) * {HD}));
    DMA_WAIT(3); DMA_DRAIN(3, DESC(desc, 1), {KB0}, (int)(Vc + ((pos + r) * {NKV} + kh) * {HD}));
    DMA_WAIT(2); DMA_WAIT(3);
  }}
  {"int lid = core_id;" if NTE == NT else f"int lid = (core_id & 3) * 3 + (core_id >> 2);                  /* tasks spread over the 3 cores (each core's DMA engine serialises its TECs' requests) */"}
  if (lid >= {NTE}) return;
  int smax = (T + {P - 1}) / {P}, CLd = (smax + 7) & ~7;                 /* a slice's positions (at most), the score octets' length */
  int Rf = {G}; while (Rf > 1 && ((Rf + 7) >> 3) * ({8 * PW * 4} + CLd * 32) > {AV}) Rf--;
  if (Rf > 8) Rf &= ~7;                                                   /* whole octets when there are several */
  int w = lid * {NKV * P * G} / {NTE}, w1 = (lid + 1) * {NKV * P * G} / {NTE};
  __global float* QO = LSF({REC});
  while (w < w1) {{
    int pr = w / {G}, j0 = w % {G}, j1 = j0 + (w1 - w); if (j1 > {G}) j1 = {G}; if (j1 - j0 > Rf) j1 = j0 + Rf;
    w += j1 - j0;
    int kvh = pr / {P}, c = pr % {P}, R = j1 - j0, NO = (R + 7) >> 3;
    int t0 = c * T / {P}, t1 = (c + 1) * T / {P}, tc = t1 < pos ? t1 : pos, n = t1 - t0;
    __global float* SC = QO + NO * {8 * PW};                              /* octet o: SC + o * CLd * 8; position t: + (t - t0) * 8 */
    DMA_WAIT(3);                                                          /* the previous unit's record drains (they read QO) */
    /* the q rows, normed and rotated, into the octets chunk-major (chunk i of lane l at + (8 i + l) 8); one request per run of a head's rows */
    if (R & 7) for (int i = 0; i < {HD // 8}; i++) for (int l = R & 7; l < 8; l++) *(__global float8*)(QO + (NO - 1) * {8 * PW} + (8 * i + l) * 8) = BC(0.0f);
    for (int jj = 0; jj < R; ) {{
      int j = j0 + jj, h = kvh * {g} + j / {M}, r = j % {M}, nr = {M} - r;
      if (nr > R - jj) nr = R - jj; if (nr > {QST}) nr = {QST};
      DMA_FILL(2, DESC(desc, {QD} - 1 + nr), {KB0}, (int)(q_rows + r * {QR} + h * {2 * HD})); DMA_WAIT(2);
      for (int ii = 0; ii < nr; ii++) {{
        norm_rope(LSF({KN}), LSF({KB0}) + ii * {HD}, qnw, rope + {RP("r + ii")} * {2 * ROT});
        __global float* qd = QO + ((jj + ii) >> 3) * {8 * PW} + ((jj + ii) & 7) * 8;
        for (int i = 0; i < {HD // 8}; i++) *(__global float8*)(qd + i * 64) = *(__global float8*)(LSF({KN}) + i * 8);
      }}
      jj += nr;
    }}
    for (int o = 0; o < NO; o++) for (int t = 0; t < n; t++) *(__global float8*)(SC + o * CLd * 8 + t * 8) = BC{NEG};
    /* pass 1: the cache rows [t0, tc) in blocks of {BT} (double-buffered), two positions x eight records a step */
    int nb = tc > t0 ? (tc - t0 + {BT - 1}) / {BT} : 0;
    if (nb > 0) DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Kc + (t0 * {NKV} + kvh) * {HD}));
    for (int b = 0; b < nb; b++) {{
      int p = b & 1;
      if (b + 1 < nb) {{ if (p == 0) DMA_FILL(1, DESC(desc, 0), {KB0 + BT * HD * 4}, (int)(Kc + ((t0 + (b + 1) * {BT}) * {NKV} + kvh) * {HD}));
                        else        DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Kc + ((t0 + (b + 1) * {BT}) * {NKV} + kvh) * {HD})); }}
      if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
      __global float* kb = LSF({KB0} + p * {BT * HD * 4});
      for (int jt = 0; jt < {BT}; jt += 2) {{
        int t = t0 + b * {BT} + jt; if (t >= tc) break;
        __global float* kr0 = kb + jt * {HD}; __global float* kr1 = kr0 + {HD};
        for (int o = 0; o < NO; o++) {{
          __global float* qb = QO + o * {8 * PW};
          {_p1(HD, True)}
          __global float* so = SC + o * CLd * 8 + (t - t0) * 8;
          *(__global float8*)so = hsumT({xs}) * BC({scale!r}f);
          if (t + 1 < tc) *(__global float8*)(so + 8) = hsumT({ys}) * BC({scale!r}f);
        }}
      }}
    }}
    /* the new rows (t >= pos): k normed and rotated locally, scored where the record's row sees it */
    for (int t = tc > t0 ? tc : t0; t < t1; t++) {{
      DMA_FILL(2, DESC(desc, 1), {KS}, (int)(kv_rows + (t - pos) * {KR} + kvh * {HD})); DMA_WAIT(2);
      norm_rope(LSF({KN}), LSF({KS}), knw, rope + {RPN("t")} * {2 * ROT});
      __global float* kr0 = LSF({KN});
      for (int o = 0; o < NO; o++) {{
        __global float* qb = QO + o * {8 * PW};
        {_p1(HD, False)}
        __global float* so = SC + o * CLd * 8 + (t - t0) * 8;
        *(__global float8*)so = hsumT({xs}) * BC({scale!r}f);
        for (int l = 0; l < 8 && o * 8 + l < R; l++) {{ int r = (j0 + o * 8 + l) % {M}; if (!{VIS("r", "t")}) so[l] = {NEG}; }}
      }}
    }}
    /* each record's max and sum (attn_part's: lane classes over its CL / 8 chunks, the chunks past the slice as their constant) */
    for (int o = 0; o < NO; o++) {{
      __global float* so = SC + o * CLd * 8;
      float8 mx = BC{NEG}; for (int t = 0; t < n; t++) mx = VMAX(mx, *(__global float8*)(so + t * 8));
      float8 ep = exp2_d4((BC{NEG} - mx) * BC({LOG2E!r}f));
      {sm_decl}
      int k = 0;
      for (; 8 * k + 8 <= n; k++) {{ {sm_full} }}
      if (8 * k < n) {{ {sm_part} k++; }}
      for (; k < {NCH}; k++) {{ {sm_pad} }}
      float8 sm = ((((((S0 + S1) + S2) + S3) + S4) + S5) + S6) + S7;
      for (int l = 0; l < 8; l++) {{
        __global float* rec = QO + (o * 8 + l) * {PW};
        for (int i = 0; i < {PW}; i += 8) *(__global float8*)(rec + i) = BC(0.0f);
        rec[{HD}] = mx[l]; rec[{HD + 1}] = n > 0 ? sm[l] : 0.0f;
      }}
    }}
    /* pass 2: o += p V, two records x 64 dims in registers over a block */
    if (nb > 0) DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Vc + (t0 * {NKV} + kvh) * {HD}));
    for (int b = 0; b < nb; b++) {{
      int p = b & 1;
      if (b + 1 < nb) {{ if (p == 0) DMA_FILL(1, DESC(desc, 0), {KB0 + BT * HD * 4}, (int)(Vc + ((t0 + (b + 1) * {BT}) * {NKV} + kvh) * {HD}));
                        else        DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Vc + ((t0 + (b + 1) * {BT}) * {NKV} + kvh) * {HD})); }}
      if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
      __global float* vb = LSF({KB0} + p * {BT * HD * 4});
      int tb = t0 + b * {BT}, nt = tc - tb < {BT} ? tc - tb : {BT};
      for (int o = 0; o < NO; o++) {{
        __global float* pe = SC + o * CLd * 8 + (tb - t0) * 8;
        {p2}
      }}
    }}
    for (int t = tc > t0 ? tc : t0; t < t1; t++) {{                     /* the new rows' v, where the record's row sees it */
      DMA_FILL(2, DESC(desc, 1), {KS}, (int)(kv_rows + (t - pos) * {KR} + {NKV * HD} + kvh * {HD})); DMA_WAIT(2);
      __global float* vr = LSF({KS});
      for (int jj = 0; jj < R; jj++) {{
        int r = (j0 + jj) % {M}; if (!{VIS("r", "t")}) continue;
        float8 sa = BC(SC[(jj >> 3) * CLd * 8 + (jj & 7) + (t - t0) * 8]); __global float* oa = QO + jj * {PW};
        {_p2_one(HD)}
      }}
    }}
    /* the records out: one 2-D drain per run of rows of one head */
    for (int jj = 0; jj < R; ) {{
      int j = j0 + jj, h = kvh * {g} + j / {M}, r = j % {M}, nr = {M} - r; if (nr > R - jj) nr = R - jj;
      DMA_WAIT(3); DMA_DRAIN(3, DESC(desc, {RD} - 1 + nr), {REC} + jj * {PW * 4}, (int)(part + ((h * {M} + r) * {P} + c) * {PW}));
      jj += nr;
    }}
  }}
  DMA_WAIT(3);
}}"""


def attn_comb2_desc(NH, HD, TMAX, M):
  K = _K(); P = K.attn_parts(NH, HD, TMAX); PW = HD + 16; QR, OR = NH * 2 * HD, NH * HD
  return K.V._desc_slots(*([(n * P * PW * 4,) for n in range(1, M + 1)] + [(n * HD * 4, HD * 4, QR * 4, HD * 4) for n in range(1, M + 1)]
                           + [(n * HD * 4, HD * 4, OR * 4, HD * 4) for n in range(1, M + 1)]))


def attn_comb2_src(NH, HD, TMAX, M, name="attn_comb"):
  """attn_comb with every row by DMA: a unit = (head h, a run of rows): its rows' P slice records (contiguous), their gates (pitch
  QR) in by two requests, attn_comb's arithmetic per (row, head), the outputs drained by one 2-D request (pitch OR). The units
  (NH x ceil(M / RU) of them) fill the 12 tasks. args: o_rows, part, q_rows, desc (attn_comb2_desc)."""
  K = _K(); NT = K.NT; P = K.attn_parts(NH, HD, TMAX); PW = HD + 16; LOG2E = 1.4426950408889634; QR, OR = NH * 2 * HD, NH * HD
  RU = M
  while RU > 1 and NH * -(-M // RU) < NT: RU -= 1                           # rows a unit: the most that still give >= NT units
  while RU * (P * PW + 2 * HD) * 4 > 32768 - 64: RU -= 1
  NU = -(-M // RU); RU = -(-M // NU)                                         # equal runs (8 rows: 4 + 4, not 7 + 1)
  NU = -(-M // RU); GT = RU * P * PW * 4; OB = GT + RU * HD * 4
  return K.V.FULL_H + K.EXP + f"""
__kernel void {name}(__global float* restrict o_rows, __global float* restrict part, __global float* restrict q_rows, __global int* restrict desc, const int core_id) {{
  __global float* pp0 = LSF(0); __global float* gt0 = LSF({GT}); __global float* ob0 = LSF({OB});
  for (int uu = core_id; uu < {NH * NU}; uu += {NT}) {{
    int h = uu / {NU}, r0 = (uu % {NU}) * {RU}, nr = {M} - r0 < {RU} ? {M} - r0 : {RU};
    DMA_WAIT(2);
    DMA_FILL(0, DESC(desc, nr - 1), 0, (int)(part + (h * {M} + r0) * {P * PW}));
    DMA_FILL(1, DESC(desc, {M} - 1 + nr), {GT}, (int)(q_rows + r0 * {QR} + h * {2 * HD} + {HD}));
    DMA_WAIT(0); DMA_WAIT(1);
    for (int rr = 0; rr < nr; rr++) {{
      __global float* pp = pp0 + rr * {P * PW}; __global float* gt = gt0 + rr * {HD}; __global float* ob = ob0 + rr * {HD};
      float m = -3.0e38f; for (int c = 0; c < {P}; c++) if (pp[c * {PW} + {HD} + 1] > 0.0f && pp[c * {PW} + {HD}] > m) m = pp[c * {PW} + {HD}];
      float w[{P}], l = 0.0f;
      for (int c = 0; c < {P}; c++) {{ w[c] = pp[c * {PW} + {HD} + 1] > 0.0f ? exp2_d4(BC((pp[c * {PW} + {HD}] - m) * {LOG2E!r}f))[0] : 0.0f; l += w[c] * pp[c * {PW} + {HD} + 1]; }}
      float8 inv = BC(1.0f / l);
      for (int j = 0; j < {HD}; j += 8) {{
        float8 o = BC(0.0f); for (int c = 0; c < {P}; c++) o += BC(w[c]) * *(__global float8*)(pp + c * {PW} + j);
        *(__global float8*)(ob + j) = o * inv * VRCP(BC(1.0f) + exp2_d4(BC({-LOG2E!r}f) * *(__global float8*)(gt + j)));
      }}
    }}
    DMA_DRAIN(2, DESC(desc, {2 * M} - 1 + nr), {OB}, (int)(o_rows + r0 * {OR} + h * {HD}));
  }}
  DMA_WAIT(2);
}}"""


def bench_arm(kern, NH, NKV, HD, TMAX, ROT, eps, M, opts, I, desc, part):
  """abench.py's hook: (src, kernel name, output floats, input tensors) for kern gqa / comb2 with opts (tree, BT<n>, NTE<n>)."""
  K = _K(); from tinygrad import Tensor; DEV = "ZHOUYI"
  o = {x.rstrip("0123456789"): (int(x[len(x.rstrip("0123456789")):]) if x[-1].isdigit() else True) for x in opts}
  P = K.attn_parts(NH, HD, TMAX); PW = HD + 16
  if kern == "gqa":
    BT = o.get("BT", K.ATT_BT); tree = bool(o.get("tree")); NTE = o.get("NTE")
    src = attn_gqa_src(NH, NKV, HD, TMAX, ROT, eps, M, tree=tree, BT=BT, NTE=NTE)
    ins = [I["q"], I["kv"], I["qnw"], I["knw"], I["rope"], I["K"], I["V"], I["idx"], desc(f"gqa{M}|{BT}", lambda: attn_gqa_desc(NH, NKV, HD, TMAX, M, BT))] + ([I["tree"]] if tree else [])
    return src, ("attn_partt" if tree else "attn_part"), NH * M * P * PW, ins
  if kern == "comb2":
    src = attn_comb2_src(NH, HD, TMAX, M)
    pt = part if part is not None else Tensor(np.random.default_rng(1).standard_normal(NH * M * P * PW).astype(np.float32), device=DEV).realize()
    return src, "attn_comb", 24 * NH * HD, [pt, I["q"], desc(f"comb2|{M}", lambda: attn_comb2_desc(NH, HD, TMAX, M))]
  raise SystemExit(f"kern {kern}?")


def attn_dec3_desc(NH, NKV, HD, BT=2):
  K = _K(); g = NH // NKV
  return K.V._desc_slots(*([(BT * HD * 4, HD * 4, NKV * HD * 4, HD * 4), (HD * 4,)] + [(n * HD * 4, HD * 4, 2 * HD * 4, HD * 4) for n in range(1, g + 1)]
                           + [(n * HD * 4,) for n in range(1, g + 1)]))


def attn_dec3_src(NH, NKV, HD, TMAX, ROT, eps, U=1, BT=2):
  """attn_dec2 (the plain decode step's attention, one row) in kv-head units: a unit = (kv head, one of U runs of its g query heads);
  the unit streams the head's K / V rows [0, pos) once by DMA (blocks of BT rows, double-buffered), scores two positions x its
  heads a step (attn_gqa's transposed sum = dot256's order), and keeps attn_dec2's arithmetic per head: the max from -1e30, the
  exps summed over t = 0..pos in order (lanes = heads), o += p V in t order, 1 / sum, the gate. The new token's k / v rows go into
  the cache by DMA on the tasks without a unit. Same arguments as attn_dec2 plus desc (attn_dec3_desc). Scores: TMAX x 8 floats."""
  K = _K(); NT = K.NT; g = NH // NKV; scale = 1.0 / HD ** 0.5; LOG2E = 1.4426950408889634
  assert 1 <= U <= g and NKV * U < NT and BT % 2 == 0 and g <= 8
  NHU = -(-g // U)                                                            # heads a unit (the last may hold fewer)
  KB0 = 0; KN = KB0 + 2 * BT * HD * 4; KS = KN + HD * 4; QO = KS + HD * 4; SC = QO + 8 * HD * 4; END = SC + TMAX * 32
  assert END <= 32768 - 64 and TMAX * 32 >= NHU * HD * 4 and 2 * BT * HD * 4 >= HD * 4, (END, NHU)
  xs = ", ".join(f"x{l}" for l in L8); ys = ", ".join(f"y{l}" for l in L8)
  p2 = " ".join(f"if ({2 * p} < R) {{ __global float* oa = QOp + {2 * p * HD}; __global float* ob = {2 * p + 1} < R ? oa + {HD} : LSF({KS}); "
                + _p2_pair(HD, 2 * p, 2 * p + 1) + " }" for p in range(4))
  return K.V.FULL_H + K.EXP + _norm_rope_c(HD, ROT, eps) + _HSUMT + f"""
__kernel void attn_dec2(__global float* restrict o_rows, __global float* restrict q_rows, __global float* restrict kv_rows,
                        __global float* restrict qnw_all, __global float* restrict knw_all, __global float* restrict rope,
                        __global float* restrict Kall, __global float* restrict Vall, __global int* restrict idx, __global int* restrict desc, const int core_id) {{
  int L = idx[0], pos = idx[1];
  __global float* Kc = Kall + (L * {TMAX}) * {NKV * HD}; __global float* Vc = Vall + (L * {TMAX}) * {NKV * HD};
  __global float* qnw = qnw_all + L * {HD}; __global float* knw = knw_all + L * {HD}; __global float* cs = rope + pos * {2 * ROT};
  if (core_id >= {NKV * U}) {{                                       /* the new token's k (normed, rotated) and v into the cache by DMA */
    for (int kh = core_id - {NKV * U}; kh < {NKV}; kh += {NT - NKV * U}) {{
      DMA_FILL(2, DESC(desc, 1), {KS}, (int)(kv_rows + kh * {HD})); DMA_FILL(3, DESC(desc, 1), {KB0}, (int)(kv_rows + {NKV * HD} + kh * {HD}));
      DMA_WAIT(2); norm_rope(LSF({KN}), LSF({KS}), knw, cs); DMA_DRAIN(2, DESC(desc, 1), {KN}, (int)(Kc + (pos * {NKV} + kh) * {HD}));
      DMA_WAIT(3); DMA_DRAIN(3, DESC(desc, 1), {KB0}, (int)(Vc + (pos * {NKV} + kh) * {HD})); DMA_WAIT(2); DMA_WAIT(3);
    }}
    return;
  }}
  int kvh = core_id / {U}, part = core_id % {U}, h0 = kvh * {g} + part * {NHU}, R = {g} - part * {NHU}; if (R > {NHU}) R = {NHU};
  __global float* QOp = LSF({QO}); __global float* S = LSF({SC});
  DMA_FILL(2, DESC(desc, 1 + R), {SC}, (int)(q_rows + h0 * {2 * HD})); DMA_FILL(3, DESC(desc, 1), {KS}, (int)(kv_rows + kvh * {HD}));
  for (int i = 0; i < {HD // 8}; i++) for (int l = R; l < 8; l++) *(__global float8*)(QOp + (8 * i + l) * 8) = BC(0.0f);
  DMA_WAIT(2);
  for (int ii = 0; ii < R; ii++) {{
    norm_rope(LSF({KN}), S + ii * {HD}, qnw, cs);
    for (int i = 0; i < {HD // 8}; i++) *(__global float8*)(QOp + (8 * i + ii) * 8) = *(__global float8*)(LSF({KN}) + i * 8);
  }}
  DMA_WAIT(3); norm_rope(LSF({KN}), LSF({KS}), knw, cs);
  __global float* qb = QOp;
  int nb = (pos + {BT - 1}) / {BT};
  if (nb > 0) DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Kc + kvh * {HD}));
  for (int b = 0; b < nb; b++) {{
    int p = b & 1;
    if (b + 1 < nb) {{ if (p == 0) DMA_FILL(1, DESC(desc, 0), {KB0 + BT * HD * 4}, (int)(Kc + ((b + 1) * {BT} * {NKV} + kvh) * {HD}));
                      else        DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Kc + ((b + 1) * {BT} * {NKV} + kvh) * {HD})); }}
    if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
    __global float* kb = LSF({KB0} + p * {BT * HD * 4});
    for (int jt = 0; jt < {BT}; jt += 2) {{
      int t = b * {BT} + jt; if (t >= pos) break;
      __global float* kr0 = kb + jt * {HD}; __global float* kr1 = kr0 + {HD};
      {_p1(HD, True)}
      *(__global float8*)(S + t * 8) = hsumT({xs}) * BC({scale!r}f);
      if (t + 1 < pos) *(__global float8*)(S + t * 8 + 8) = hsumT({ys}) * BC({scale!r}f);
    }}
  }}
  {{ __global float* kr0 = LSF({KN}); {_p1(HD, False)} *(__global float8*)(S + pos * 8) = hsumT({xs}) * BC({scale!r}f); }}
  float8 mx = BC(-1e30f); for (int t = 0; t <= pos; t++) mx = VMAX(mx, *(__global float8*)(S + t * 8));
  float8 sum = BC(0.0f);
  for (int t = 0; t <= pos; t++) {{ float8 e = exp2_d4((*(__global float8*)(S + t * 8) - mx) * BC({LOG2E!r}f)); *(__global float8*)(S + t * 8) = e; sum += e; }}
  for (int i = 0; i < R * {HD}; i += 8) *(__global float8*)(QOp + i) = BC(0.0f);
  if (nb > 0) DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Vc + kvh * {HD}));
  for (int b = 0; b < nb; b++) {{
    int p = b & 1;
    if (b + 1 < nb) {{ if (p == 0) DMA_FILL(1, DESC(desc, 0), {KB0 + BT * HD * 4}, (int)(Vc + ((b + 1) * {BT} * {NKV} + kvh) * {HD}));
                      else        DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Vc + ((b + 1) * {BT} * {NKV} + kvh) * {HD})); }}
    if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
    __global float* vb = LSF({KB0} + p * {BT * HD * 4});
    int tb = b * {BT}, nt = pos - tb < {BT} ? pos - tb : {BT};
    __global float* pe = S + tb * 8;
    {p2}
  }}
  DMA_FILL(2, DESC(desc, 1), {KS}, (int)(kv_rows + {NKV * HD} + kvh * {HD})); DMA_WAIT(2);
  for (int j0 = 0; j0 < R; j0 += {2 * BT}) {{
    int ng = R - j0 < {2 * BT} ? R - j0 : {2 * BT};
    DMA_FILL(3, DESC(desc, 1 + ng), {KB0}, (int)(q_rows + (h0 + j0) * {2 * HD} + {HD})); DMA_WAIT(3);
    for (int jj = j0; jj < j0 + ng; jj++) {{
      float8 pb = BC(S[pos * 8 + jj]); __global float* oa = QOp + jj * {HD}; __global float* vr = LSF({KS}); __global float* gp = LSF({KB0}) + (jj - j0) * {HD};
      float8 inv = BC(1.0f / sum[jj]);
      for (int j = 0; j < {HD}; j += 8) {{
        float8 o = *(__global float8*)(oa + j) + pb * *(__global float8*)(vr + j);
        float8 sg = VRCP(BC(1.0f) + exp2_d4(BC({-LOG2E!r}f) * *(__global float8*)(gp + j)));
        *(__global float8*)(oa + j) = o * inv * sg;
      }}
    }}
  }}
  DMA_DRAIN(2, DESC(desc, {1 + g} + R), {QO}, (int)(o_rows + h0 * {HD})); DMA_WAIT(2);
}}"""


def gqa_nte(NH, NKV, HD, TMAX, M, BT=None, T_est=None, port_gbs=21.6, core_gbs=7.2, cyc_oct=380.0, ghz=1.2):
  """The task count (attn_gqa_src's NTE) a cost model picks: for n = 12..1 tasks, at T_est positions (default TMAX / 4), the K / V
  bytes (each task's runs of records cut at slice boundaries and at what LSRAM holds, a run streaming its slice's K and V once)
  over the port (at most core_gbs a core: one DMA engine a core, the tasks spread over the cores), against the busiest task's
  octet-positions (a run of r records costs ceil(r / 8) octets a position, ~cyc_oct cycles: simulator bundle counts of v3);
  the n with the least max(), fewer tasks only when 2 % better. A model, not a measurement (QWEN_ATTN_NTE overrides)."""
  K = _K(); BT = K.ATT_BT if BT is None else BT; g = NH // NKV; P = K.attn_parts(NH, HD, TMAX); G = g * M; PW = HD + 16
  T = T_est or max(M + 1, TMAX // 4); AV = gqa_layout(NH, NKV, HD, TMAX, M, BT)["AV"]; CLd = (-(-T // P) + 7) & ~7
  Rf = G
  while Rf > 1 and -(-Rf // 8) * (8 * PW * 4 + CLd * 32) > AV: Rf -= 1
  if Rf > 8: Rf &= ~7
  W = NKV * P * G; best = None
  for n in range(K.NT, 0, -1):
    runs = 0; busiest = 0
    for k in range(n):
      w, w1 = k * W // n, (k + 1) * W // n; octs = 0
      while w < w1:
        j0 = w % G; j1 = min(G, j0 + (w1 - w), j0 + Rf); runs += 1; octs += -(-(j1 - j0) // 8); w += j1 - j0
      busiest = max(busiest, octs)
    bw = min(port_gbs, core_gbs * min(3, n))
    t = max(runs * (T / P) * 2 * HD * 4 / (bw * 1e3), busiest * (T / P) * cyc_oct / (ghz * 1e3))
    if best is None or t < best[0] * 0.98: best = (t, n)
  return best[1]

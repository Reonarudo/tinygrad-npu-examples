"""The vision tower's hand-written kernels (12 tasks), beside the GEMM (gemm_gs on Q8_0 panels) and the A-layout producer
(Kernels.rms_a(norm=False)):
  ln32        LayerNorm (mean, variance, (1 + 0) w, b) -> fp32 rows                                  out[r] = (x[r] - mean) / sqrt(var + eps) w + b
  geluc32     C tiles (a GEMM's) + bias -> GELU (tanh form, as ggml) -> fp32 rows
  residb32    x + C tiles + bias -> fp32 rows (a linear's epilogue with its residual)
  rope_rows   the 2-D rope of the vision attention on q and k in place (pairs (i, i + 32) of the 64 head dims; cos | sin a row)
  vit_attn    bidirectional attention over N rows: units = (head, a run of up to 8 NO query rows), the head's K and V streamed by
              DMA in blocks with the long-context kernels' online softmax (attn_gqa.py's _p1 / _p2_pair over 64-dim heads)
Row layouts: qkv [N][3 NH HD] fp32 (q | k | v), o [N][NH HD]."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "qwen3.8-27b"))
import numpy as np
from qwen38_kernels import NT, TILE, EXP, GELU, V        # noqa: E402
import attn_gqa as AG                                     # noqa: E402

L8 = range(8)


def tile_expr(col, nrb, rb="rb", i="i"):
  """C: the 16 floats (row quad i, columns col .. col + 3) of row block rb in a GEMM's C tiles."""
  return f"ct + (((({col}) / 48) * {nrb} + {rb}) * 3 + (({col}) % 48) / 16) * 192 + ({i} * 4 + (({col}) % 16) / 4) * 16"


def ln32_src(c, eps, n):
  """LayerNorm of n rows of c (c % 8 == 0): out[r] = (x - mean) * rsqrt(var + eps) * w + b; a task takes rows core_id, + 12, ..."""
  assert c % 8 == 0
  return V.FULL_H + f"""
__kernel void ln32(__global float* restrict out, __global float* restrict x, __global float* restrict w, __global float* restrict b, const int core_id) {{
  for (int r = core_id; r < {n}; r += {NT}) {{
    __global float* p = x + r * {c}; float8 s = BC(0.0f);
    for (int i = 0; i < {c}; i += 8) s += *(__global float8*)(p + i);
    float mean = hsum8(s) * {1.0 / c!r}f; float8 m8 = BC(mean), v = BC(0.0f);
    for (int i = 0; i < {c}; i += 8) {{ float8 d = *(__global float8*)(p + i) - m8; v += d * d; }}
    float8 inv = BC(1.0f / __builtin_sqrtf(hsum8(v) * {1.0 / c!r}f + {eps!r}f));
    for (int i = 0; i < {c}; i += 8) *(__global float8*)(out + r * {c} + i) = (*(__global float8*)(p + i) - m8) * inv * *(__global float8*)(w + i) + *(__global float8*)(b + i);
  }}
}}"""


def geluc32_src(c, nrb, n, bias=True):
  """GELU(C + bias) of a GEMM's C tiles (c columns = the groups' 48-column span or less) -> fp32 rows [n][c]; a unit = (row block,
  a range of column quads)."""
  assert c % 4 == 0; nq = c // 4; rbr = -(-n // 12); P = max(1, min(nq // 4, NT // rbr)); chunk = -(-nq // P); chunk += (-chunk) % 4
  return V.FULL_H + TILE + EXP + GELU + f"""
__kernel void geluc32(__global float* restrict out, __global float* restrict ct{", __global float* restrict bias" if bias else ""}, const int core_id) {{
  for (int u = core_id; u < {rbr * P}; u += {NT}) {{
    int rb = u / {P}, part = u % {P}; int q1 = (part + 1) * {chunk} < {nq} ? (part + 1) * {chunk} : {nq};
    for (int i = 0; i < 3; i++) for (int q = part * {chunk}; q < q1; q++) {{
      int col = 4 * q; __global float* t = {tile_expr("col", nrb)};
      {"float4 b4 = *(__global float4*)(bias + col); float8 b8 = F8(b4, b4);" if bias else "float8 b8 = BC(0.0f);"}
      float8 t01 = gl(F8(*(__global float4*)(t), *(__global float4*)(t + 4)) + b8, BC(1.0f)), t23 = gl(F8(*(__global float4*)(t + 8), *(__global float4*)(t + 12)) + b8, BC(1.0f));
      int row = rb * 12 + 4 * i;
      if (row < {n}) *(__global float4*)(out + row * {c} + col) = __builtin_shufflevector(t01, t01, 0, 1, 2, 3);
      if (row + 1 < {n}) *(__global float4*)(out + (row + 1) * {c} + col) = __builtin_shufflevector(t01, t01, 4, 5, 6, 7);
      if (row + 2 < {n}) *(__global float4*)(out + (row + 2) * {c} + col) = __builtin_shufflevector(t23, t23, 0, 1, 2, 3);
      if (row + 3 < {n}) *(__global float4*)(out + (row + 3) * {c} + col) = __builtin_shufflevector(t23, t23, 4, 5, 6, 7);
    }}
  }}
}}"""


def residb32_src(c, nrb, n):
  """out = x + C + bias (fp32 rows [n][c]); a unit = (row block, a range of column quads)."""
  assert c % 4 == 0; nq = c // 4; rbr = -(-n // 12); P = max(1, min(nq // 4, NT // rbr)); chunk = -(-nq // P); chunk += (-chunk) % 4
  return V.FULL_H + TILE + f"""
__kernel void residb32(__global float* restrict out, __global float* restrict x, __global float* restrict ct, __global float* restrict bias, const int core_id) {{
  for (int u = core_id; u < {rbr * P}; u += {NT}) {{
    int rb = u / {P}, part = u % {P}; int q1 = (part + 1) * {chunk} < {nq} ? (part + 1) * {chunk} : {nq};
    for (int i = 0; i < 3; i++) for (int q = part * {chunk}; q < q1; q++) {{
      int col = 4 * q; __global float* t = {tile_expr("col", nrb)}; float4 b4 = *(__global float4*)(bias + col);
      for (int r = 0; r < 4; r++) {{ int row = rb * 12 + 4 * i + r; if (row < {n}) *(__global float4*)(out + row * {c} + col) = *(__global float4*)(x + row * {c} + col) + *(__global float4*)(t + 4 * r) + b4; }}
    }}
  }}
}}"""


def rope_rows_src(n, NH, HD, pitch):
  """q and k of n rows (the qkv rows, pitch floats a row: q at 0, k at NH HD) rotated in place by the vision rope: for every head,
  pairs (i, i + HD/2), i < HD/2, by the row's cos | sin table (tab [n][HD]: cos[HD/2] | sin[HD/2])."""
  HR = HD // 2; assert HR % 8 == 0
  return V.FULL_H + f"""
__kernel void rope_rows(__global float* restrict out, __global float* restrict qkv, __global float* restrict tab, const int core_id) {{
  for (int r = core_id; r < {n}; r += {NT}) {{
    __global float* cs = tab + r * {HD};
    for (int h = 0; h < {2 * NH}; h++) {{
      __global float* v = qkv + r * {pitch} + h * {HD};
      for (int i = 0; i < {HR}; i += 8) {{
        float8 a = *(__global float8*)(v + i), b = *(__global float8*)(v + {HR} + i), c = *(__global float8*)(cs + i), s = *(__global float8*)(cs + {HR} + i);
        *(__global float8*)(v + i) = a * c - b * s; *(__global float8*)(v + {HR} + i) = b * c + a * s;
      }}
    }}
  }}
  if (core_id == 0) out[0] = 1.0f;
}}"""


def vit_attn_desc(HD, pitch, opitch, BT, NO):
  """slot 0: BT K / V rows of one head (pitch `pitch` floats); 1 .. NO: 8 n q rows (pitch); NO + 1 .. 2 NO: 8 n o rows out (pitch opitch)."""
  sl = [(BT * HD * 4, HD * 4, pitch * 4, HD * 4)] + [(8 * k * HD * 4, HD * 4, pitch * 4, HD * 4) for k in range(1, NO + 1)] + [(8 * k * HD * 4, HD * 4, opitch * 4, HD * 4) for k in range(1, NO + 1)]
  return V._desc_slots(*sl)


def _p2_pair_v(HD, la, lb):
  """attn_gqa's _p2_pair for a head of HD dims (a multiple of 16): 64-dim blocks, then the tail block of HD % 64 dims (the 9B's
  heads: 72 dims padded to 80 = 64 + 16)."""
  def blk(d0, nd):
    ks = range(nd // 8)
    ld = " ".join(f"float8 a{k} = *(__global float8*)(oa + {d0 + 8 * k}), b{k} = *(__global float8*)(ob + {d0 + 8 * k});" for k in ks)
    fm = " ".join(f"{{ float8 vv = *(__global float8*)(vr + {8 * k}); a{k} += sa * vv; b{k} += sb * vv; }}" for k in ks)
    st = " ".join(f"*(__global float8*)(ob + {d0 + 8 * k}) = b{k}; *(__global float8*)(oa + {d0 + 8 * k}) = a{k};" for k in ks)
    return (f"{{ {ld} for (int j = 0; j < nt; j++) {{ float8 ev = *(__global float8*)(pe + j * 8); float8 sa = LANE(ev, {la}), sb = LANE(ev, {lb}); "
            f"__global float* vr = vb + j * {HD} + {d0}; {fm} }} {st} }}")
  return " ".join(blk(d0, min(64, HD - d0)) for d0 in range(0, HD, 64))


def vit_attn_clc(HD, BT=8, NO=3):
  """The largest position chunk (a multiple of BT) whose score rows fit LSRAM beside the K / V blocks, the q octets and the o records."""
  fixed = 2 * BT * HD * 4 + 2 * 8 * NO * HD * 4 + NO * 64; return min(128, (32768 - 64 - fixed) // (NO * 32) // BT * BT)


def vit_attn_src(NH, HD, n, pitch, BT=8, NO=3, CLC=None, hd_real=None):
  """Bidirectional attention over n rows: qkv [n][pitch] (q | k | v, NH heads of HD), o -> out [n][NH HD]. A unit = (head h, the
  query rows 8 NO u .. + 8 NO - 1): the q octets staged chunk-major, the head's K then V rows streamed in blocks of BT by DMA
  (double-buffered) in position chunks of CLC with the online softmax (running max / sum a record, o rescaled), the dots and the
  P V pass as attn_gqa's _p1 / _p2_pair_v; o scaled by 1 / sum and drained. HD a multiple of 16 (the dots take 16-dim chunks, the P V
  pass 64-dim blocks and a tail); `hd_real`: the head's unpadded dims, the softmax scale's (the 9B's 72 in 80)."""
  if CLC is None: CLC = vit_attn_clc(HD, BT, NO)
  assert HD % 16 == 0 and BT % 2 == 0 and CLC % BT == 0; R8 = 8 * NO; NU = -(-n // R8); scale = 1.0 / (hd_real or HD) ** 0.5; LOG2E = 1.4426950408889634
  KB0 = 0; QO = KB0 + 2 * BT * HD * 4; OO = QO + R8 * HD * 4; MS = OO + R8 * HD * 4; SC = MS + NO * 64; END = SC + NO * CLC * 32
  assert END <= 32768 - 64, END
  xs = ", ".join(f"x{l}" for l in L8); ys = ", ".join(f"y{l}" for l in L8)
  p2 = " ".join(f"{{ __global float* oa = OB + (o * 8 + {2 * p}) * {HD}; __global float* ob = oa + {HD}; {_p2_pair_v(HD, 2 * p, 2 * p + 1)} }}" for p in range(4))
  return V.FULL_H + EXP + AG._HSUMT + f"""
__kernel void vit_attn(__global float* restrict out, __global float* restrict qkv, __global int* restrict desc, const int core_id) {{
  for (int un = core_id; un < {NH * NU}; un += {NT}) {{
    int h = un / {NU}, u = un % {NU}, r0 = u * {R8}, nr = {n} - r0 < {R8} ? {n} - r0 : {R8}, NOu = (nr + 7) / 8;
    __global float* K0 = qkv + {NH * HD} + h * {HD}; __global float* V0 = qkv + {2 * NH * HD} + h * {HD};
    __global float* QB = LSF({QO}); __global float* OB = LSF({OO}); __global float* MSb = LSF({MS}); __global float* S = LSF({SC});
    /* the q rows into LSRAM (one 2-D request), then chunk-major octets: chunk i of lane l at QB + o * 8 HD + (8 i + l) 8 */
    DMA_FILL(2, DESC(desc, NOu), {OO}, (int)(qkv + r0 * {pitch} + h * {HD})); DMA_WAIT(2);
    for (int o = 0; o < NOu; o++) for (int l = 0; l < 8; l++) {{
      int jj = o * 8 + l;
      for (int i = 0; i < {HD // 8}; i++) *(__global float8*)(QB + o * {8 * HD} + (8 * i + l) * 8) = jj < nr ? *(__global float8*)(OB + jj * {HD} + i * 8) : BC(0.0f);
    }}
    for (int i = 0; i < {R8 * HD}; i += 8) *(__global float8*)(OB + i) = BC(0.0f);
    for (int o = 0; o < NOu; o++) {{ *(__global float8*)(MSb + o * 16) = BC(-1e30f); *(__global float8*)(MSb + o * 16 + 8) = BC(0.0f); }}
    for (int ca = 0; ca < {n}; ca += {CLC}) {{
      int cb = ca + {CLC} < {n} ? ca + {CLC} : {n}, nc = cb - ca, nb = (nc + {BT - 1}) / {BT};
      DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(K0 + ca * {pitch}));
      for (int b = 0; b < nb; b++) {{
        int p = b & 1;
        if (b + 1 < nb) {{ if (p == 0) DMA_FILL(1, DESC(desc, 0), {KB0 + BT * HD * 4}, (int)(K0 + (ca + (b + 1) * {BT}) * {pitch}));
                          else        DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(K0 + (ca + (b + 1) * {BT}) * {pitch})); }}
        if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
        __global float* kb = LSF({KB0} + p * {BT * HD * 4});
        for (int jt = 0; jt < {BT}; jt += 2) {{
          int t = ca + b * {BT} + jt; if (t >= cb) break;
          __global float* kr0 = kb + jt * {HD}; __global float* kr1 = kr0 + {HD};
          for (int o = 0; o < NOu; o++) {{
            __global float* qb = QB + o * {8 * HD};
            {AG._p1(HD, True)}
            __global float* so = S + o * {CLC * 8} + (t - ca) * 8;
            *(__global float8*)so = hsumT({xs}) * BC({scale!r}f);
            if (t + 1 < cb) *(__global float8*)(so + 8) = hsumT({ys}) * BC({scale!r}f);
          }}
        }}
      }}
      for (int o = 0; o < NOu; o++) {{                                   /* the online softmax of this chunk */
        __global float* so = S + o * {CLC * 8}; __global float* ms = MSb + o * 16;
        float8 mo = *(__global float8*)ms, lo = *(__global float8*)(ms + 8);
        float8 mc = BC(-1e30f); for (int t = 0; t < nc; t++) mc = VMAX(mc, *(__global float8*)(so + t * 8));
        float8 mn = VMAX(mo, mc), al = exp2_d4((mo - mn) * BC({LOG2E!r}f)), sm = lo * al;
        for (int t = 0; t < nc; t++) {{ float8 e = exp2_d4((*(__global float8*)(so + t * 8) - mn) * BC({LOG2E!r}f)); *(__global float8*)(so + t * 8) = e; sm += e; }}
        *(__global float8*)ms = mn; *(__global float8*)(ms + 8) = sm;
        if (ca > 0) for (int l = 0; l < 8; l++) {{ __global float* rec = OB + (o * 8 + l) * {HD}; float8 a = BC(al[l]); for (int i = 0; i < {HD}; i += 8) *(__global float8*)(rec + i) *= a; }}
      }}
      DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(V0 + ca * {pitch}));
      for (int b = 0; b < nb; b++) {{
        int p = b & 1;
        if (b + 1 < nb) {{ if (p == 0) DMA_FILL(1, DESC(desc, 0), {KB0 + BT * HD * 4}, (int)(V0 + (ca + (b + 1) * {BT}) * {pitch}));
                          else        DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(V0 + (ca + (b + 1) * {BT}) * {pitch})); }}
        if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
        __global float* vb = LSF({KB0} + p * {BT * HD * 4});
        int tb = ca + b * {BT}, nt = cb - tb < {BT} ? cb - tb : {BT};
        for (int o = 0; o < NOu; o++) {{
          __global float* pe = S + o * {CLC * 8} + (tb - ca) * 8;
          {p2}
        }}
      }}
    }}
    for (int o = 0; o < NOu; o++) {{ float8 sm = *(__global float8*)(MSb + o * 16 + 8); for (int l = 0; l < 8; l++) {{ __global float* rec = OB + (o * 8 + l) * {HD}; float8 inv = BC(1.0f / sm[l]); for (int i = 0; i < {HD}; i += 8) *(__global float8*)(rec + i) *= inv; }} }}
    DMA_DRAIN(3, DESC(desc, {NO} + NOu), {OO}, (int)(out + r0 * {NH * HD} + h * {HD})); DMA_WAIT(3);
  }}
}}"""

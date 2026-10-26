"""Qwen3-30B-A3B's own TEC kernel (the backend's `register_csrc`), beside the qwen3.8-27b / gemma4 ones it reuses:

  moe_comb: out = x + sum_h w[r, h] C_h   -- the expert block's weighted sum and the residual add, from the down projection's
            C tiles of `heads` experts (one gemm_gs(bsel) launch: head h's tiles `hstride` floats apart)
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qwen3.8-27b"))
from qwen38_kernels import NT, TILE, EXP, _real, V                         # noqa: E402


def moe_comb_src(c, nrb, real, heads, hstride):
  """out [rows, c] = x + sum_h w[r * heads + h] C_h for c columns (C_h's groups of 48 cover >= c columns), fp32 rows; w fp32
  [real][heads]. A unit = (row block, a contiguous range of column quads aligned to 64-byte lines), as resid32: tasks on different
  cores never store into the same line. The heads are summed in order, then added to x (h = 0 first)."""
  assert c % 4 == 0; nq = c // 4; rbr, iqr = _real(real, nrb); P = max(1, min(nq // 4, NT // rbr)); chunk = -(-nq // P); chunk += (-chunk) % 4
  return V.FULL_H + TILE + f"""
__kernel void moe_comb(__global float* restrict out, __global float* restrict x, __global float* restrict ct, __global float* restrict w, const int core_id) {{
  for (int u = core_id; u < {rbr * P}; u += {NT}) {{
    int rb = u / {P}, part = u % {P}; int q1 = (part + 1) * {chunk} < {nq} ? (part + 1) * {chunk} : {nq};
    for (int i = 0; i < {iqr}; i++) for (int q = part * {chunk}; q < q1; q++) {{
      int col = 4 * q; __global float* t = ct + ((col / 48 * {nrb} + rb) * 3 + (col % 48) / 16) * 192 + (i * 4 + (col % 16) / 4) * 16;
      __global float* xr = x + (rb * 12 + 4 * i) * {c} + col; __global float* orow = out + (rb * 12 + 4 * i) * {c} + col;
      for (int r = 0; r < 4; r++) {{
        int row = rb * 12 + 4 * i + r;
        if (row < {real}) {{
          float4 s = (float4){{0.0f, 0.0f, 0.0f, 0.0f}};
          for (int h = 0; h < {heads}; h++) {{ float wh = w[row * {heads} + h]; s += (float4){{wh, wh, wh, wh}} * *(__global float4*)(t + h * {hstride} + 4 * r); }}
          *(__global float4*)(orow + r * {c}) = *(__global float4*)(xr + r * {c}) + s;
        }}
      }}
    }}
  }}
}}"""


def kv_rows_src(c1, c2, nrb, real, goff1=0, goff2=0):
  """out [rows, c1 + c2] = (C1's columns 0 .. c1 - 1 from its group goff1 | C2's 0 .. c2 - 1 from goff2), fp32 rows: k and v from
  two GEMMs (the layers whose v is Q6_K and k Q4_K) as the attention's kv rows. A unit = (row block, a column quad range), as
  resid32."""
  assert c1 % 4 == 0 and c2 % 4 == 0; c = c1 + c2; nq = c // 4; rbr, iqr = _real(real, nrb); P = max(1, min(nq // 4, NT // rbr))
  chunk = -(-nq // P); chunk += (-chunk) % 4
  return V.FULL_H + TILE + f"""
__kernel void kv_rows(__global float* restrict out, __global float* restrict ct1, __global float* restrict ct2, const int core_id) {{
  for (int u = core_id; u < {rbr * P}; u += {NT}) {{
    int rb = u / {P}, part = u % {P}; int q1 = (part + 1) * {chunk} < {nq} ? (part + 1) * {chunk} : {nq};
    for (int i = 0; i < {iqr}; i++) for (int q = part * {chunk}; q < q1; q++) {{
      int col = 4 * q; int second = col >= {c1}; int cc = second ? col - {c1} : col;
      __global float* t = (second ? ct2 + {goff2 * nrb * 3 * 192} : ct1 + {goff1 * nrb * 3 * 192}) + ((cc / 48 * {nrb} + rb) * 3 + (cc % 48) / 16) * 192 + (i * 4 + (cc % 16) / 4) * 16;
      __global float* orow = out + (rb * 12 + 4 * i) * {c} + col;
      for (int r = 0; r < 4; r++) if (rb * 12 + 4 * i + r < {real}) *(__global float4*)(orow + r * {c}) = *(__global float4*)(t + 4 * r);
    }}
  }}
}}"""


GV_JC, GV_KB = 4, 512                                                     # gemv32d: output columns a unit, K a block


def gemv32d_desc(k): return V._desc_slots((GV_JC * GV_KB * 4, GV_KB * 4, k * 4, GV_KB * 4))   # a unit's JC weight rows' K block


def gemv32d_src(rows, k, n, norm=False, eps=1e-6):
  """qwen38_kernels.gemv32 (out fp32 [rows, n] = x [rows, k] @ w [n, k]^T, `norm`: the rows RMS-normalised, the norm's weight
  folded into w) with the weight blocks brought into LSRAM by double-buffered 2D DMA instead of plain loads, and JC = 4 columns a
  unit (the router's 128 columns: 32 units over the 12 tasks). Each column's sums are gemv32's (the same order): the same bits.
  args: out, x, w, desc (gemv32d_desc(k))."""
  JC, KB = GV_JC, GV_KB; assert k % KB == 0 and n % JC == 0; P, NB, BLK = n // JC, k // KB, JC * KB * 4
  ss_decl = f" float ss[{rows}]; for (int m = 0; m < {rows}; m++) ss[m] = 0.0f;" if norm else ""
  ss_acc = " sq += xv * xv;" if norm else ""; ss_sum = " ss[m] += hsum8(sq);" if norm else ""
  inv_mul = f" * (1.0f / __builtin_sqrtf(ss[m] * {1.0 / k!r}f + {eps!r}f))" if norm else ""
  acc8 = "".join(f" a{j} += xv * *(__global float8*)(WB + {j * KB} + i);" for j in range(JC))
  return V.FULL_H + f"""
__kernel void gemv32d(__global float* restrict out, __global float* restrict x, __global float* restrict w, __global int* restrict desc, const int core_id) {{
  for (int u = core_id; u < {P}; u += {NT}) {{
    float acc[{rows}][{JC}];{ss_decl}
    for (int m = 0; m < {rows}; m++) for (int j = 0; j < {JC}; j++) acc[m][j] = 0.0f;
    __global float* wu = w + u * {JC * k};
    DMA_FILL(0, DESC(desc, 0), 0, (int)wu);
    for (int b = 0; b < {NB}; b++) {{
      int s = b & 1; int kb = b * {KB};
      if (b + 1 < {NB}) DMA_FILL(s ^ 1, DESC(desc, 0), (s ^ 1) * {BLK}, (int)(wu + kb + {KB}));
      DMA_WAIT(s);
      __global float* WB = LSF(s * {BLK});
      for (int m = 0; m < {rows}; m++) {{
        __global float* xr = x + m * {k} + kb;
        float8 {", ".join(f"a{j} = BC(0.0f)" for j in range(JC))}, sq = BC(0.0f);
        for (int i = 0; i < {KB}; i += 8) {{ float8 xv = *(__global float8*)(xr + i);{ss_acc}{acc8} }}
        {" ".join(f"acc[m][{j}] += hsum8(a{j});" for j in range(JC))}{ss_sum}
      }}
    }}
    for (int m = 0; m < {rows}; m++) for (int j = 0; j < {JC}; j++) out[m * {n} + u * {JC} + j] = acc[m][j]{inv_mul};
  }}
  DMA_WAIT_ALL();
}}"""


def moe_route_desc(NE, TOPK): return V._desc_slots((NE * 4,), (64,), (TOPK * 4,))   # a row's logits; the 16 slots; a row's weights


def moe_route_src(NE, TOPK, S_gu, S_dn):
  """The routing on the device, one row (decode): the row's NE logits (by DMA) -> the TOPK largest (the first index on ties: the
  host's stable argsort of -p), their softmax (= the renormalised top-k of the softmax over all NE: the denominator cancels) ->
  sel_gu[h] = e_h S_gu, sel_dn[h] = e_h S_dn (the experts' byte offsets in their stacks; slots TOPK.. 0), wt[h] = w_h, all by
  DMA. One task. args: sel_gu, sel_dn, wt, logits, desc (moe_route_desc)."""
  assert TOPK == 8 and NE % 32 == 0; LOG2E = 1.4426950408889634
  return V.FULL_H + EXP + f"""
__kernel void moe_route(__global int* restrict selg, __global int* restrict seld, __global float* restrict wt, __global float* restrict lg, __global int* restrict desc, const int core_id) {{
  if (core_id == 0) {{
    DMA_FILL(0, DESC(desc, 0), 0, (int)lg); DMA_WAIT(0);
    __global float* L = LSF(0); __global int* SG = (__global int*)LSF({NE * 4}); __global int* SD = SG + 16; __global float* WO = LSF({NE * 4 + 128});
    unsigned taken[{NE // 32}]; for (int i = 0; i < {NE // 32}; i++) taken[i] = 0u;
    float v[8]; int id[8];
    for (int k = 0; k < 8; k++) {{
      float b = -3.0e38f; int bi = 0;
      for (int e = 0; e < {NE}; e++) if (!((taken[e >> 5] >> (e & 31)) & 1u) && L[e] > b) {{ b = L[e]; bi = e; }}
      taken[bi >> 5] |= 1u << (bi & 31); v[k] = b; id[k] = bi;
    }}
    float8 t = (float8){{v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]}};
    float8 ex = exp2_d4((t - BC(v[0])) * BC({LOG2E!r}f)); float s = hsum8(ex); float8 w = ex * BC(1.0f / s);
    for (int k = 0; k < 8; k++) {{ SG[k] = id[k] * {S_gu}; SD[k] = id[k] * {S_dn}; SG[k + 8] = 0; SD[k + 8] = 0; }}
    *(__global float8*)WO = w;
    DMA_DRAIN(1, DESC(desc, 1), {NE * 4}, (int)selg); DMA_DRAIN(2, DESC(desc, 1), {NE * 4 + 64}, (int)seld); DMA_DRAIN(3, DESC(desc, 2), {NE * 4 + 128}, (int)wt);
    DMA_WAIT_ALL();
  }}
}}"""


# ---- batched prefill: the tokens' experts in slots of 12 rows (one gemm_gs head a slot, rows mode) ----

def inv_rms_desc(c): return V._desc_slots((c * 4,))

def inv_rms_src(c, m, eps):
  """inv[16 t] = 1 / sqrt(mean(x[t]^2) + eps) for the m rows of x fp32 [*, c] (a row by DMA; the squares summed as rms_a32d's phase 1:
  float8 lanes, K ascending). A task a row (rows core_id, core_id + 12, ...); each value on a 64-byte line of its own (tasks on
  different cores storing into one line lose data). args: inv (fp32 [16 m]), x, desc (inv_rms_desc)."""
  return V.FULL_H + f"""
__kernel void inv_rms(__global float* restrict inv, __global float* restrict x, __global int* restrict desc, const int core_id) {{
  for (int t = core_id; t < {m}; t += {NT}) {{
    DMA_FILL(0, DESC(desc, 0), 0, (int)(x + t * {c})); DMA_WAIT(0);
    __global float* X = LSF(0); float8 a = BC(0.0f);
    for (int k = 0; k < {c}; k += 8) {{ float8 v = *(__global float8*)(X + k); a += v * v; }}
    inv[16 * t] = 1.0f / __builtin_sqrtf(hsum8(a) * {1.0 / c!r}f + {eps!r}f);
  }}
}}"""

def rms_gather_desc(c): return V._desc_slots((512,), (3072,), (c * 4,))   # a row's 128-column slice; a slot's slice of A; the norm weight

def rms_gather_src(c, nslot, rb=1, nused=False):
  """The slots' A (rows mode, rt 3: per slot [K / 128 slices][32 k-quads][3 row tiles][4 rows][4 k] fp16, a gemm_gs head's A) from
  the token rows: slot s row r = fp16(x[tok[12 rb s + r]] inv[tok] w) (tok -1: a zero row); `rb` > 1: a slot is rb row blocks of
  the GEMM's full layout ([slice][rb][32 k-quads][3 row tiles][4][4], a head of 12 rb rows). A unit = (slot, slice, row block): its 12 rows' 128
  columns arrive by DMA, are scaled and converted (CVT16 of row pairs) into the slice's 3 KB in LSRAM, which leaves by one DMA.
  args: a (fp16 [nslot][24576 c / 2048]), x, inv (inv_rms's: [16 t]), w, tok (int32 [12 nslot]), desc (rms_gather_desc)."""
  nsl = c // 128; WO = 12 * 512; AO = WO + c * 4; assert AO + 3072 <= 32768 - 64
  return V.FULL_H + TILE + f"""
__kernel void rms_gather(__global half* restrict a, __global float* restrict x, __global float* restrict inv, __global float* restrict w,
                         __global int* restrict tok, __global int* restrict desc, {"__global int* restrict nused, " if nused else ""}const int core_id) {{
  int ns_ = {"nused[0]" if nused else nslot};
  DMA_FILL(2, DESC(desc, 2), {WO}, (int)w); DMA_WAIT(2);
  __global float* WL = LSF({WO});
  for (int u = core_id; u < {nslot * nsl * rb}; u += {NT}) {{
    int s = u / {nsl * rb}, sl = (u / {rb}) % {nsl}, b = u % {rb}; float sc[12];
    if (s >= ns_) continue;
    for (int r = 0; r < 12; r++) {{
      int t = tok[s * {12 * rb} + b * 12 + r]; __global float* R = LSF(r * 512);
      if (t >= 0) {{ DMA_FILL(0, DESC(desc, 0), r * 512, (int)(x + t * {c} + sl * 128)); sc[r] = inv[16 * t]; }}
      else {{ for (int k = 0; k < 128; k += 8) *(__global float8*)(R + k) = BC(0.0f); sc[r] = 0.0f; }}
    }}
    DMA_WAIT(0);
    __global float* Wl = WL + sl * 128; __global half* A = LSH({AO});
    for (int kk = 0; kk < 32; kk++) {{
      float4 wq = *(__global float4*)(Wl + 4 * kk);
      for (int i = 0; i < 3; i++) {{
        float4 q[4];
        for (int r4 = 0; r4 < 4; r4++) {{ int r = 4 * i + r4; float s4 = sc[r]; q[r4] = *(__global float4*)(LSF(r * 512) + 4 * kk) * wq * (float4){{s4, s4, s4, s4}}; }}
        *(__global half16*)(A + (kk * 3 + i) * 16) = CVT16(F8(q[0], q[1]), F8(q[2], q[3]));
      }}
    }}
    DMA_DRAIN(1, DESC(desc, 1), {AO}, (int)(a + ((s * {nsl} + sl) * {rb} + b) * 1536)); DMA_WAIT(1);
  }}
  DMA_WAIT_ALL();
}}"""

def moe_scatter_src(c, m, nrb, hstride, topk=8, er=12):
  """out [t] = x [t] + sum_k w[t, k] C[ent[t, k]] for the m token rows: ent = slot x er + row (the slot's GEMM head's C tiles,
  `hstride` floats apart, its er rows in its nrb row blocks), the entries summed in order (k ascending), as moe_comb. A unit = (row, a range of
  column quads aligned to 64-byte lines). args: out, x, ct, w (fp32 [m][topk]), ent (int32 [m][topk])."""
  assert c % 4 == 0; nq = c // 4; P = max(1, min(nq // 16, NT)); chunk = -(-nq // P); chunk += (-chunk) % 4
  return V.FULL_H + TILE + f"""
__kernel void moe_scatter(__global float* restrict out, __global float* restrict x, __global float* restrict ct, __global float* restrict w,
                          __global int* restrict ent, const int core_id) {{
  for (int u = core_id; u < {m * P}; u += {NT}) {{
    int t = u / {P}, part = u % {P}; int q1 = (part + 1) * {chunk} < {nq} ? (part + 1) * {chunk} : {nq};
    __global float* hb[{topk}]; float wk[{topk}];
    for (int k = 0; k < {topk}; k++) {{ int e = ent[t * {topk} + k]; int h = e / {er}, r = e % {er}; int rr = r % 12;
      hb[k] = ct + h * {hstride} + (r / 12) * 576 + ((rr / 4) * 4) * 16 + (rr % 4) * 4; wk[k] = w[t * {topk} + k]; }}
    for (int q = part * {chunk}; q < q1; q++) {{
      int col = 4 * q; int co = ((col / 48 * {nrb}) * 3 + (col % 48) / 16) * 192 + ((col % 16) / 4) * 16;
      float4 s = (float4){{0.0f, 0.0f, 0.0f, 0.0f}};
      for (int k = 0; k < {topk}; k++) s += (float4){{wk[k], wk[k], wk[k], wk[k]}} * *(__global float4*)(hb[k] + co);
      *(__global float4*)(out + t * {c} + col) = *(__global float4*)(x + t * {c} + col) + s;
    }}
  }}
}}"""


GM_JC, GM_KB, GM_RG = 16, 128, 8                                          # gemv32m: columns a unit (a 64-byte line), K a block, rows a unit

def gemv32m_desc(k): return V._desc_slots((GM_JC * GM_KB * 4, GM_KB * 4, k * 4, GM_KB * 4), (GM_RG * GM_KB * 4, GM_KB * 4, k * 4, GM_KB * 4))

def gemv32m_src(m, k, n, eps=1e-6):
  """The router for many rows (the batched prefill): out [m, n] = rms(x [m, k]) @ w [n, k]^T (the norm's weight folded into w),
  fp32. A unit = (8 rows, 16 columns): the columns' weight block and the rows' x block (128 of K) by double-buffered DMA, 16 x 8
  accumulators; the rows' squares summed on the way. Each unit stores whole 64-byte lines (16 floats of a row).
  args: out, x, w, desc (gemv32m_desc(k))."""
  JC, KB, RG = GM_JC, GM_KB, GM_RG; assert k % KB == 0 and n % JC == 0 and m % RG == 0
  NCU, NB = n // JC, k // KB; WB, XB = JC * KB * 4, RG * KB * 4; XO = 2 * WB; assert XO + 2 * XB <= 32768 - 64
  return V.FULL_H + f"""
__kernel void gemv32m(__global float* restrict out, __global float* restrict x, __global float* restrict w, __global int* restrict desc, const int core_id) {{
  for (int u = core_id; u < {(m // RG) * NCU}; u += {NT}) {{
    int rg = u / {NCU}, cu = u % {NCU};
    __global float* wu = w + cu * {JC * k}; __global float* xu = x + rg * {RG * k};
    float acc[{RG}][{JC}]; float ss[{RG}];
    for (int r = 0; r < {RG}; r++) {{ ss[r] = 0.0f; for (int j = 0; j < {JC}; j++) acc[r][j] = 0.0f; }}
    DMA_FILL(0, DESC(desc, 0), 0, (int)wu); DMA_FILL(2, DESC(desc, 1), {XO}, (int)xu);
    for (int b = 0; b < {NB}; b++) {{
      int s = b & 1;
      if (b + 1 < {NB}) {{ DMA_FILL(s ^ 1, DESC(desc, 0), (s ^ 1) * {WB}, (int)(wu + (b + 1) * {KB})); DMA_FILL(2 + (s ^ 1), DESC(desc, 1), {XO} + (s ^ 1) * {XB}, (int)(xu + (b + 1) * {KB})); }}
      DMA_WAIT(s); DMA_WAIT(2 + s);
      __global float* WL = LSF(s * {WB}); __global float* XL = LSF({XO} + s * {XB});
      for (int r = 0; r < {RG}; r++) {{
        float8 sq = BC(0.0f); float8 a[{JC}];
        for (int j = 0; j < {JC}; j++) a[j] = BC(0.0f);
        for (int i = 0; i < {KB}; i += 8) {{
          float8 xv = *(__global float8*)(XL + r * {KB} + i); sq += xv * xv;
          for (int j = 0; j < {JC}; j++) a[j] += xv * *(__global float8*)(WL + j * {KB} + i);
        }}
        ss[r] += hsum8(sq); for (int j = 0; j < {JC}; j++) acc[r][j] += hsum8(a[j]);
      }}
    }}
    for (int r = 0; r < {RG}; r++) {{
      float iv = 1.0f / __builtin_sqrtf(ss[r] * {1.0 / k!r}f + {eps!r}f);
      for (int j = 0; j < {JC}; j++) out[(rg * {RG} + r) * {n} + cu * {JC} + j] = acc[r][j] * iv;
    }}
  }}
  DMA_WAIT_ALL();
}}"""


def slot_rows_desc(rb, c):
  """slot 0: one group's C block of a slot (rb row blocks x 2304 B, contiguous); 1: a group's 48 columns of the 12 rb rows out (192 B
  a row, pitch 4 c); 2: the last group's (c % 48 columns: the rest is past the row)."""
  tail = c - (c // 48) * 48 or 48
  return V._desc_slots((rb * 2304,), (12 * rb * 192, 192, c * 4, 192), (12 * rb * tail * 4, tail * 4, c * 4, 192))

def slot_rows_src(nslot, rb, ngr, c, nused=False):
  """The expert slots' down-projection C tiles -> fp32 rows Y [nslot * 12 rb][c], each row times its weight (wrow [nslot * 12 rb],
  0 for an empty row): a unit = (slot, group) -- the group's C block (its rb row blocks) arrives by one DMA, the 12 rb rows of its
  48 columns are built (x the weight) in LSRAM and leave by one 2D DMA. args: y, ct, wrow, desc (slot_rows_desc)."""
  ER = 12 * rb; CB = rb * 2304; YO = CB; assert YO + ER * 192 <= 32768 - 64
  return V.FULL_H + f"""
__kernel void slot_rows(__global float* restrict y, __global float* restrict ct, __global float* restrict wrow, __global int* restrict desc, {"__global int* restrict nused, " if nused else ""}const int core_id) {{
  int ns_ = {"nused[0]" if nused else nslot};
  for (int u = core_id; u < {nslot * ngr}; u += {NT}) {{
    int s = u / {ngr}, g = u % {ngr}; if (s >= ns_) continue;
    DMA_FILL(0, DESC(desc, 0), 0, (int)(ct + (s * {ngr} + g) * {rb * 576})); DMA_WAIT(0);
    __global float* C = LSF(0); __global float* Y = LSF({YO});
    for (int r = 0; r < {ER}; r++) {{
      float w = wrow[s * {ER} + r]; float4 wv = (float4){{w, w, w, w}}; int b = r / 12, rr = r % 12;
      for (int q = 0; q < 12; q++) {{
        int col = 4 * q; *(__global float4*)(Y + r * 48 + col) = wv * *(__global float4*)(C + b * 576 + (col / 16) * 192 + ((rr / 4) * 4 + (col % 16) / 4) * 16 + (rr % 4) * 4);
      }}
    }}
    if (g == {ngr - 1}) DMA_DRAIN(1, DESC(desc, 2), {YO}, (int)(y + s * {ER * c} + g * 48)); else DMA_DRAIN(1, DESC(desc, 1), {YO}, (int)(y + s * {ER * c} + g * 48));
    DMA_WAIT(1);
  }}
  DMA_WAIT_ALL();
}}"""

def rows_sum_desc(c, ch): return V._desc_slots((ch * 4,))

def rows_sum_src(m, topk, c, ch=512):
  """out[t] = x[t] + sum_k Y[ent[t, k]] (the weighted rows of slot_rows; the entries in order, k ascending) for the m token rows,
  every row piece by DMA: a unit = (token, a `ch`-column chunk). args: out, x, y, ent (int32 [m][topk]), desc (rows_sum_desc)."""
  assert c % ch == 0; P = c // ch; assert (topk + 1) * ch * 4 <= 32768 - 64
  return V.FULL_H + f"""
__kernel void rows_sum(__global float* restrict out, __global float* restrict x, __global float* restrict y, __global int* restrict ent, __global int* restrict desc, const int core_id) {{
  for (int u = core_id; u < {m * P}; u += {NT}) {{
    int t = u / {P}, p = u % {P};
    DMA_FILL(2, DESC(desc, 0), 0, (int)(x + t * {c} + p * {ch}));
    for (int k = 0; k < {topk}; k++) {{                                       /* one request in flight a flag: entries on flags 0 / 1 */
      if (k >= 2) DMA_WAIT(k & 1);
      DMA_FILL(k & 1, DESC(desc, 0), (k + 1) * {ch * 4}, (int)(y + ent[t * {topk} + k] * {c} + p * {ch}));
    }}
    DMA_WAIT(0); DMA_WAIT(1); DMA_WAIT(2);
    __global float* X = LSF(0);
    for (int i = 0; i < {ch}; i += 8) {{
      float8 s = BC(0.0f);
      for (int k = 0; k < {topk}; k++) s += *(__global float8*)(LSF((k + 1) * {ch * 4}) + i);
      *(__global float8*)(X + i) += s;
    }}
    DMA_DRAIN(3, DESC(desc, 0), 0, (int)(out + t * {c} + p * {ch})); DMA_WAIT(3);
  }}
  DMA_WAIT_ALL();
}}"""


# ---- the batched prefill's attention on the matrix unit: S = Q K^T and O = P V as fp16 gemm_gs calls (32 heads, b_head_div 8) ----
# Q per head as an A layout of K = HD (one 128-wide slice): [rb][32 k-quads][3 row tiles][4 rows][4 dims] fp16 (1536 halves a row block);
# K per kv head as B panels (N = keys, K = dims): [group][3 strips][32 k-quads][4 tiles][4 keys][4 dims]; P per head as an A layout
# of K = keys: [key slice][rb][32][3][4][4]; V^T per kv head as B panels (N = dims, 3 groups / 144 with the last 16 zero, K = keys):
# [group][key slice][3][32][4][4 dims][4 keys]. The scores' 1 / sqrt(HD) is in the q norm's weight.

def qrope_a_desc(QR, HD): return V._desc_slots((12 * HD * 4, HD * 4, QR * 4, HD * 4), (3072,), (HD * 4,), (12 * HD * 4,))

def qrope_a_src(NH, HD, nrb, eps, QR):
  """The prefill's Q in the A layout: head h, row r = fp16(rope(rms(q[r, h]) w, position idx[0] + r)) (NeoX halves). A unit =
  (head, row block): its 12 rows' head slice by one 2D DMA, normed and rotated in LSRAM, the 3 KB A block out by DMA.
  args: qa (fp16 [NH][nrb][1536]), q_rows, qnw, cs (the RoPE table [tmax][HD]: cos | sin), idx, desc (qrope_a_desc)."""
  H2 = HD // 2; XO, AO, WO = 0, 12 * HD * 4, 12 * HD * 4 + 3072; CO = WO + HD * 4; assert CO + 12 * HD * 4 <= 32768 - 64
  return V.FULL_H + TILE + f"""
__kernel void qrope_a(__global half* restrict qa, __global float* restrict q, __global float* restrict w, __global float* restrict cs,
                      __global int* restrict idx, __global int* restrict desc, const int core_id) {{
  int pos0 = idx[0];
  DMA_FILL(2, DESC(desc, 2), {WO}, (int)w); DMA_WAIT(2);
  __global float* W = LSF({WO});
  for (int u = core_id; u < {NH * nrb}; u += {NT}) {{
    int h = u / {nrb}, rb = u % {nrb};
    DMA_FILL(0, DESC(desc, 0), {XO}, (int)(q + rb * {12 * QR} + h * {HD}));
    DMA_FILL(1, DESC(desc, 3), {CO}, (int)(cs + (pos0 + rb * 12) * {HD})); DMA_WAIT(0); DMA_WAIT(1);
    __global float* X = LSF({XO}); __global float* CS = LSF({CO});
    for (int r = 0; r < 12; r++) {{
      __global float* x = X + r * {HD}; __global float* c = CS + r * {HD}; float8 a = BC(0.0f);
      for (int d = 0; d < {HD}; d += 8) {{ float8 v = *(__global float8*)(x + d); a += v * v; }}
      float iv = 1.0f / __builtin_sqrtf(hsum8(a) * {1.0 / HD!r}f + {eps!r}f); float8 ivv = BC(iv);
      for (int d = 0; d < {H2}; d += 8) {{
        float8 x1 = *(__global float8*)(x + d) * ivv * *(__global float8*)(W + d), x2 = *(__global float8*)(x + d + {H2}) * ivv * *(__global float8*)(W + d + {H2});
        float8 co = *(__global float8*)(c + d), si = *(__global float8*)(c + d + {H2});
        *(__global float8*)(x + d) = x1 * co - x2 * si; *(__global float8*)(x + d + {H2}) = x2 * co + x1 * si;
      }}
    }}
    __global half* A = LSH({AO});
    for (int kk = 0; kk < {HD // 4}; kk++) for (int i = 0; i < 3; i++) {{
      __global float* x0 = X + (4 * i) * {HD} + 4 * kk;
      *(__global half16*)(A + (kk * 3 + i) * 16) = CVT16(F8(*(__global float4*)x0, *(__global float4*)(x0 + {HD})), F8(*(__global float4*)(x0 + {2 * HD}), *(__global float4*)(x0 + {3 * HD})));
    }}
    DMA_DRAIN(3, DESC(desc, 1), {AO}, (int)(qa + u * 1536)); DMA_WAIT(3);
  }}
  DMA_WAIT_ALL();
}}"""

def k_panels_desc(NKV, HD): return V._desc_slots((16 * HD * 4, HD * 4, NKV * HD * 4, HD * 4), (16 * HD * 2,))

def k_panels_src(NKV, HD, nstrip):
  """K (the cache, fp32 [t][NKV][HD]) as fp16 B panels a kv head (`nstrip` strips of 16 keys = nstrip / 3 groups): a unit = (kv head,
  strip): its 16 keys' rows by one 2D DMA, the 4 KB strip out. args: kb (fp16 [NKV][nstrip][2048]), Kc, desc (k_panels_desc)."""
  assert HD == 128
  return V.FULL_H + TILE + f"""
__kernel void k_panels(__global half* restrict kb, __global float* restrict Kc, __global int* restrict desc, const int core_id) {{
  for (int u = core_id; u < {NKV * nstrip}; u += {NT}) {{
    int h = u / {nstrip}, q = u % {nstrip};
    DMA_FILL(0, DESC(desc, 0), 0, (int)(Kc + (q * 16 * {NKV} + h) * {HD})); DMA_WAIT(0);
    __global float* X = LSF(0); __global half* B = LSH({16 * HD * 4});
    for (int kk = 0; kk < 32; kk++) for (int j = 0; j < 4; j++) {{
      __global float* x0 = X + (4 * j) * {HD} + 4 * kk;
      *(__global half16*)(B + (kk * 4 + j) * 16) = CVT16(F8(*(__global float4*)x0, *(__global float4*)(x0 + {HD})), F8(*(__global float4*)(x0 + {2 * HD}), *(__global float4*)(x0 + {3 * HD})));
    }}
    DMA_DRAIN(1, DESC(desc, 1), {16 * HD * 4}, (int)(kb + u * 2048)); DMA_WAIT(1);
  }}
  DMA_WAIT_ALL();
}}"""

def v_panels_desc(NKV, HD): return V._desc_slots((128 * 64, 64, NKV * HD * 4, 64), (4096,))

def v_panels_src(NKV, HD, nsl):
  """V^T (the cache, fp32 [t][NKV][HD]) as fp16 B panels a kv head (N = the HD dims in 3 groups, the 16 past HD zero; K = nsl slices of
  128 keys): a unit = (kv head, group, key slice, strip): 128 keys x 16 dims by one 2D DMA, transposed into the 4 KB strip.
  args: vb (fp16 [NKV][3][nsl][3][2048]), Vc, desc (v_panels_desc)."""
  assert HD == 128
  return V.FULL_H + TILE + f"""
__kernel void v_panels(__global half* restrict vb, __global float* restrict Vc, __global int* restrict desc, const int core_id) {{
  for (int u = core_id; u < {NKV * 3 * nsl * 3}; u += {NT}) {{
    int h = u / {9 * nsl}, g = (u / {3 * nsl}) % 3, c = (u / 3) % {nsl}, s = u % 3; int d0 = 48 * g + 16 * s;
    __global half* B = LSH(8192);
    if (d0 >= {HD}) {{ for (int i = 0; i < 2048; i += 16) *(__global half16*)(B + i) = CVT16(BC(0.0f), BC(0.0f)); }}
    else {{
      DMA_FILL(0, DESC(desc, 0), 0, (int)(Vc + (c * 128 * {NKV} + h) * {HD} + d0)); DMA_WAIT(0);
      __global float* X = LSF(0);                                            /* [128 keys][16 dims] */
      for (int kk = 0; kk < 32; kk++) for (int j = 0; j < 4; j++) {{
        float4 q[4];
        for (int n = 0; n < 4; n++) {{ int d = 4 * j + n; q[n] = (float4){{X[(4 * kk) * 16 + d], X[(4 * kk + 1) * 16 + d], X[(4 * kk + 2) * 16 + d], X[(4 * kk + 3) * 16 + d]}}; }}
        *(__global half16*)(B + (kk * 4 + j) * 16) = CVT16(F8(q[0], q[1]), F8(q[2], q[3]));
      }}
    }}
    DMA_DRAIN(1, DESC(desc, 1), 8192, (int)(vb + u * 2048)); DMA_WAIT(1);
  }}
  DMA_WAIT_ALL();
}}"""

def softmax_a_desc(ngq, nrb): return V._desc_slots((ngq * 2304, 2304, nrb * 2304, 2304), (3072,))

def softmax_a_src(NH, nrb, ngq, nsl, T):
  """The scores' C tiles (a head: ngq groups of 48 keys, nrb row blocks) -> P = softmax over the visible keys (t <= idx[0] + r, t < T)
  as the A layout of K = nsl x 128 keys (fp16, zero past the visible). A unit = (head, row block): the row block's C pieces of every
  group by one 2D DMA. A C tile (4 rows x 4 keys) is a P tile, so everything runs on tile pairs (two float8: rows 4i, 4i + 1 / 4i + 2,
  4i + 3): the masked lanes get min(0, (visible - key - 1/2) 1e30) added before the max and the exp (exp2 polynomial: 0 in fp16),
  the key tiles past the row block's last visible key are zeros. args: pa (fp16 [NH][nsl][nrb][1536]), ct, idx, desc (softmax_a_desc)."""
  CB = ngq * 2304; PO = CB; assert PO + 3072 <= 32768 - 64 and nsl * 128 <= ngq * 48 + 128
  L2E = 1.4426950408889634
  return V.FULL_H + TILE + EXP + f"""
#define CT(t, i) (C + ((t) / 48) * 192 * 3 + (((t) % 48) / 16) * 192 + ((i) * 4 + ((t) % 16) / 4) * 16)
static inline float8 pen(float8 vis, int t) {{ float8 k = BC((float)t) + (float8){{0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 2.0f, 3.0f}};
  return VMIN(BC(0.0f), (vis - k - BC(0.5f)) * BC(1.0e30f)); }}
static inline float8 rows2(float a, float b) {{ return (float8){{a, a, a, a, b, b, b, b}}; }}
static inline float max4(float8 v, int h) {{ float m = v[4 * h]; for (int k = 1; k < 4; k++) m = v[4 * h + k] > m ? v[4 * h + k] : m; return m; }}
static inline float sum4(float8 v, int h) {{ return v[4 * h] + v[4 * h + 1] + v[4 * h + 2] + v[4 * h + 3]; }}
__kernel void softmax_a(__global half* restrict pa, __global float* restrict ct, __global int* restrict idx, __global int* restrict desc, const int core_id) {{
  int pos0 = idx[0];
  for (int u = core_id; u < {NH * nrb}; u += {NT}) {{
    int h = u / {nrb}, rb = u % {nrb}; int base = pos0 + rb * 12; int tend = base + 12 < {T} ? base + 12 : {T};
    DMA_FILL(0, DESC(desc, 0), 0, (int)(ct + (h * {ngq * nrb} + rb) * 576)); DMA_WAIT(0);
    __global float* C = LSF(0); float8 V0[3], V1[3], M0[3], M1[3], I0[3], I1[3];
    for (int i = 0; i < 3; i++) {{
      float v[4]; for (int r4 = 0; r4 < 4; r4++) {{ int x = base + 4 * i + r4 + 1; v[r4] = (float)(x < {T} ? x : {T}); }}
      V0[i] = rows2(v[0], v[1]); V1[i] = rows2(v[2], v[3]);
      float8 m0 = BC(-3.0e38f), m1 = BC(-3.0e38f);
      for (int t = 0; t < tend; t += 4) {{ __global float* p = CT(t, i); m0 = VMAX(m0, *(__global float8*)p + pen(V0[i], t)); m1 = VMAX(m1, *(__global float8*)(p + 8) + pen(V1[i], t)); }}
      M0[i] = rows2(max4(m0, 0), max4(m0, 1)); M1[i] = rows2(max4(m1, 0), max4(m1, 1));
      float8 s0 = BC(0.0f), s1 = BC(0.0f);
      for (int t = 0; t < tend; t += 4) {{ __global float* p = CT(t, i);
        s0 += exp2_d4((*(__global float8*)p + pen(V0[i], t) - M0[i]) * BC({L2E!r}f)); s1 += exp2_d4((*(__global float8*)(p + 8) + pen(V1[i], t) - M1[i]) * BC({L2E!r}f)); }}
      I0[i] = rows2(1.0f / sum4(s0, 0), 1.0f / sum4(s0, 1)); I1[i] = rows2(1.0f / sum4(s1, 0), 1.0f / sum4(s1, 1));
    }}
    __global half* P = LSH({PO});
    for (int c = 0; c < {nsl}; c++) {{
      for (int kk = 0; kk < 32; kk++) for (int i = 0; i < 3; i++) {{
        int t = c * 128 + 4 * kk; float8 e0 = BC(0.0f), e1 = BC(0.0f);
        if (t < tend) {{ __global float* p = CT(t, i);
          e0 = exp2_d4((*(__global float8*)p + pen(V0[i], t) - M0[i]) * BC({L2E!r}f)) * I0[i]; e1 = exp2_d4((*(__global float8*)(p + 8) + pen(V1[i], t) - M1[i]) * BC({L2E!r}f)) * I1[i]; }}
        *(__global half16*)(P + (kk * 3 + i) * 16) = CVT16(e0, e1);
      }}
      DMA_DRAIN(1, DESC(desc, 1), {PO}, (int)(pa + ((h * {nsl} + c) * {nrb} + rb) * 1536)); DMA_WAIT(1);
    }}
  }}
  DMA_WAIT_ALL();
}}"""

def o_rows_desc(nrb, QR, HD): return V._desc_slots((3 * 2304, 2304, nrb * 2304, 2304), (12 * HD * 4, HD * 4, QR * 4, HD * 4))

def o_rows_src(NH, HD, nrb, QR):
  """O = P V's C tiles (a head: 3 groups of 48 dims, nrb row blocks) -> fp32 rows o [r][h HD + d]. A unit = (head, row block): the
  3 groups' pieces by one 2D DMA, the 12 rows' HD floats out by one. args: o, ct, desc (o_rows_desc)."""
  YO = 3 * 2304; assert YO + 12 * HD * 4 <= 32768 - 64
  return V.FULL_H + f"""
__kernel void o_rows(__global float* restrict o, __global float* restrict ct, __global int* restrict desc, const int core_id) {{
  for (int u = core_id; u < {NH * nrb}; u += {NT}) {{
    int h = u / {nrb}, rb = u % {nrb};
    DMA_FILL(0, DESC(desc, 0), 0, (int)(ct + (h * {3 * nrb} + rb) * 576)); DMA_WAIT(0);
    __global float* C = LSF(0); __global float* Y = LSF({YO});
    for (int r = 0; r < 12; r++) for (int d = 0; d < {HD}; d += 4)
      *(__global float4*)(Y + r * {HD} + d) = *(__global float4*)(C + (d / 48) * 576 + ((d % 48) / 16) * 192 + ((r / 4) * 4 + (d % 16) / 4) * 16 + (r % 4) * 4);
    DMA_DRAIN(1, DESC(desc, 1), {YO}, (int)(o + rb * {12 * QR} + h * {HD})); DMA_WAIT(1);
  }}
  DMA_WAIT_ALL();
}}"""


# ---- the batched prefill's layout kernels by DMA: C tiles ([group][rb][3 strips][3 row tiles][4 col tiles][4][4] fp32) <-> rows <-> A
# layouts ([slice][rb][32 k-quads][3 row tiles][4][4] fp16) without the plain-load passes. A C tile (4 rows x 4 columns) is an A
# tile's 4 rows x 4 k when the columns are K: the conversions are CVT16 of a tile's two float8 halves.
CTP = "#define CTILE(C, g0, col, i) ((C) + ((col) / 48 - (g0)) * 576 + (((col) % 48) / 16) * 192 + ((i) * 4 + ((col) % 16) / 4) * 16)\n"

def _gdesc(ng, nrb): return (ng * 2304, 2304, nrb * 2304, 2304)       # ng groups' pieces of one row block (a C tile buffer of nrb row blocks)

def o_a_desc(nrb): return V._desc_slots(_gdesc(3, nrb), (3072,))

def o_a_src(NH, HD, nrb):
  """O = P V's C tiles (a head: 3 groups of 48 dims, nrb row blocks) -> the o projection's A (K = NH HD: head h is K-slice h).
  A unit = (head, row block): the 3 groups' pieces in, the 3 KB A block out. args: a (fp16 [NH][nrb][1536]), ct, desc (o_a_desc)."""
  assert HD == 128
  return V.FULL_H + TILE + CTP + f"""
__kernel void o_a(__global half* restrict a, __global float* restrict ct, __global int* restrict desc, const int core_id) {{
  for (int u = core_id; u < {NH * nrb}; u += {NT}) {{
    int h = u / {nrb}, rb = u % {nrb};
    DMA_FILL(0, DESC(desc, 0), 0, (int)(ct + (h * {3 * nrb} + rb) * 576)); DMA_WAIT(0);
    __global float* C = LSF(0); __global half* A = LSH({3 * 2304});
    for (int kk = 0; kk < 32; kk++) for (int i = 0; i < 3; i++) {{ __global float* p = CTILE(C, 0, 4 * kk, i); *(__global half16*)(A + (kk * 3 + i) * 16) = CVT16(*(__global float8*)p, *(__global float8*)(p + 8)); }}
    DMA_DRAIN(1, DESC(desc, 1), {3 * 2304}, (int)(a + u * 1536)); DMA_WAIT(1);
  }}
  DMA_WAIT_ALL();
}}"""

def qrope_c_desc(nrb, HD): return V._desc_slots(_gdesc(3, nrb), _gdesc(4, nrb), (3072,), (HD * 4,), (12 * HD * 4,))

def qrope_c_src(NH, HD, nrb, eps):
  """qrope_a straight from the q | k|v GEMM's C tiles (q: columns 0 .. NH HD): head h's 128 columns span 3 or 4 groups of 48, read by
  one 2D DMA; the rows are built in LSRAM, normed, rotated and converted as qrope_a. args: qa, ct, qnw, cs, idx, desc (qrope_c_desc)."""
  assert HD == 128; H2 = HD // 2; CB = 4 * 2304; XO, AO = CB, CB + 12 * HD * 4; WO = AO + 3072; CO = WO + HD * 4; assert CO + 12 * HD * 4 <= 32768 - 64
  return V.FULL_H + TILE + CTP + f"""
__kernel void qrope_c(__global half* restrict qa, __global float* restrict ct, __global float* restrict w, __global float* restrict cs,
                      __global int* restrict idx, __global int* restrict desc, const int core_id) {{
  int pos0 = idx[0];
  DMA_FILL(2, DESC(desc, 3), {WO}, (int)w); DMA_WAIT(2);
  __global float* W = LSF({WO});
  for (int u = core_id; u < {NH * nrb}; u += {NT}) {{
    int h = u / {nrb}, rb = u % {nrb}; int g0 = (h * {HD}) / 48, g1 = (h * {HD} + {HD - 1}) / 48;
    if (g1 - g0 == 3) DMA_FILL(0, DESC(desc, 1), 0, (int)(ct + (g0 * {nrb} + rb) * 576)); else DMA_FILL(0, DESC(desc, 0), 0, (int)(ct + (g0 * {nrb} + rb) * 576));
    DMA_FILL(1, DESC(desc, 4), {CO}, (int)(cs + (pos0 + rb * 12) * {HD})); DMA_WAIT(0); DMA_WAIT(1);
    __global float* C = LSF(0); __global float* X = LSF({XO}); __global float* CS = LSF({CO});
    for (int r = 0; r < 12; r++) for (int d = 0; d < {HD}; d += 4) {{ int col = h * {HD} + d; *(__global float4*)(X + r * {HD} + d) = *(__global float4*)(CTILE(C, g0, col, r / 4) + (r % 4) * 4); }}
    for (int r = 0; r < 12; r++) {{
      __global float* x = X + r * {HD}; __global float* c = CS + r * {HD}; float8 a = BC(0.0f);
      for (int d = 0; d < {HD}; d += 8) {{ float8 v = *(__global float8*)(x + d); a += v * v; }}
      float8 ivv = BC(1.0f / __builtin_sqrtf(hsum8(a) * {1.0 / HD!r}f + {eps!r}f));
      for (int d = 0; d < {H2}; d += 8) {{
        float8 x1 = *(__global float8*)(x + d) * ivv * *(__global float8*)(W + d), x2 = *(__global float8*)(x + d + {H2}) * ivv * *(__global float8*)(W + d + {H2});
        float8 co = *(__global float8*)(c + d), si = *(__global float8*)(c + d + {H2});
        *(__global float8*)(x + d) = x1 * co - x2 * si; *(__global float8*)(x + d + {H2}) = x2 * co + x1 * si;
      }}
    }}
    __global half* A = LSH({AO});
    for (int kk = 0; kk < {HD // 4}; kk++) for (int i = 0; i < 3; i++) {{
      __global float* x0 = X + (4 * i) * {HD} + 4 * kk;
      *(__global half16*)(A + (kk * 3 + i) * 16) = CVT16(F8(*(__global float4*)x0, *(__global float4*)(x0 + {HD})), F8(*(__global float4*)(x0 + {2 * HD}), *(__global float4*)(x0 + {3 * HD})));
    }}
    DMA_DRAIN(3, DESC(desc, 2), {AO}, (int)(qa + u * 1536)); DMA_WAIT(3);
  }}
  DMA_WAIT_ALL();
}}"""

RD_Q = 4                                                                  # resid_d: groups a unit

def resid_d_desc(c, nrb):
  ngr = -(-c // 48); tail = c - (ngr // RD_Q) * RD_Q * 48 if ngr % RD_Q else RD_Q * 48; tg = -(-tail // 48)
  return V._desc_slots(_gdesc(RD_Q, nrb), _gdesc(tg, nrb), (12 * RD_Q * 192, RD_Q * 192, c * 4, RD_Q * 192), (12 * tail * 4, tail * 4, c * 4, RD_Q * 192))

def resid_d_src(c, nrb):
  """out = x + C (the o projection's tiles, c columns) for every row of the full layout by DMA: a unit = (row block, RD_Q groups): the
  groups' C pieces and the 12 rows' columns in, the sum out (the last unit: the columns left). args: out, x, ct, desc (resid_d_desc)."""
  ngr = -(-c // 48); nq = -(-ngr // RD_Q); W = RD_Q * 48; tail = c - (nq - 1) * W
  CB = RD_Q * 2304; XO = CB; assert XO + 12 * W * 4 <= 32768 - 64
  return V.FULL_H + CTP + f"""
__kernel void resid_d(__global float* restrict out, __global float* restrict x, __global float* restrict ct, __global int* restrict desc, const int core_id) {{
  for (int u = core_id; u < {nrb * nq}; u += {NT}) {{
    int rb = u / {nq}, q = u % {nq}; int last = q == {nq - 1}; int w = last ? {tail} : {W};
    __global float* xs = x + rb * {12 * c} + q * {W}; __global float* cs = ct + (q * {RD_Q * nrb} + rb) * 576;
    if (last) {{ DMA_FILL(0, DESC(desc, 1), 0, (int)cs); DMA_FILL(1, DESC(desc, 3), {XO}, (int)xs); }}
    else {{ DMA_FILL(0, DESC(desc, 0), 0, (int)cs); DMA_FILL(1, DESC(desc, 2), {XO}, (int)xs); }}
    DMA_WAIT(0); DMA_WAIT(1);
    __global float* C = LSF(0); __global float* X = LSF({XO});
    for (int r = 0; r < 12; r++) for (int k = 0; k < w; k += 4) *(__global float4*)(X + r * {W} + k) += *(__global float4*)(CTILE(C, 0, k, r / 4) + (r % 4) * 4);
    __global float* os = out + rb * {12 * c} + q * {W};
    if (last) DMA_DRAIN(2, DESC(desc, 3), {XO}, (int)os); else DMA_DRAIN(2, DESC(desc, 2), {XO}, (int)os);
    DMA_WAIT(2);
  }}
  DMA_WAIT_ALL();
}}"""

def rms_ad_desc(c): return V._desc_slots((12 * 512, 512, c * 4, 512), (3072,), (512,))

def rms_ad_src(c, nrb):
  """The full layout's A = fp16(x inv w) by DMA (inv from inv_rms: [16 t]): a unit = (row block, K-slice): the 12 rows' 128 columns in,
  the 3 KB A block out. args: a, x, inv, w, desc (rms_ad_desc)."""
  nsl = c // 128
  return V.FULL_H + TILE + f"""
__kernel void rms_ad(__global half* restrict a, __global float* restrict x, __global float* restrict inv, __global float* restrict w, __global int* restrict desc, const int core_id) {{
  for (int u = core_id; u < {nrb * nsl}; u += {NT}) {{
    int rb = u / {nsl}, sl = u % {nsl};
    DMA_FILL(0, DESC(desc, 0), 0, (int)(x + rb * {12 * c} + sl * 128)); DMA_FILL(1, DESC(desc, 2), {12 * 512 + 3072}, (int)(w + sl * 128)); DMA_WAIT(0); DMA_WAIT(1);
    __global float* X = LSF(0); __global float* Wl = LSF({12 * 512 + 3072}); __global half* A = LSH({12 * 512}); float sc[12];
    for (int r = 0; r < 12; r++) sc[r] = inv[16 * (rb * 12 + r)];
    for (int kk = 0; kk < 32; kk++) {{
      float4 wq = *(__global float4*)(Wl + 4 * kk);
      for (int i = 0; i < 3; i++) {{
        float4 q[4];
        for (int r4 = 0; r4 < 4; r4++) {{ int r = 4 * i + r4; float s4 = sc[r]; q[r4] = *(__global float4*)(X + r * 128 + 4 * kk) * wq * (float4){{s4, s4, s4, s4}}; }}
        *(__global half16*)(A + (kk * 3 + i) * 16) = CVT16(F8(q[0], q[1]), F8(q[2], q[3]));
      }}
    }}
    DMA_DRAIN(2, DESC(desc, 1), {12 * 512}, (int)(a + (sl * {nrb} + rb) * 1536)); DMA_WAIT(2);
  }}
  DMA_WAIT_ALL();
}}"""

def swiglu_d_desc(rb): return V._desc_slots(_gdesc(3, rb), _gdesc(4, rb), (3072,))

def swiglu_d_src(S, rb, ff, nused=False):
  """The experts' down-projection A from the gate | up tiles (S heads of ff columns each, `rb` row blocks; gate heads first, then up):
  fp16(silu(g) u) per head as [K-slice][rb][...] A. A unit = (head, K-slice, row block): the slice's 3 or 4 groups of gate and of up
  in by DMA, the 3 KB A block out. args: a (fp16 [S][ff / 128][rb][1536]), ct, desc (swiglu_d_desc)."""
  ngh = ff // 48; nsl = ff // 128; GO = 4 * 2304; AO = 2 * GO; assert AO + 3072 <= 32768 - 64
  return V.FULL_H + TILE + EXP + CTP + f"""
__kernel void swiglu_d(__global half* restrict a, __global float* restrict ct, __global int* restrict desc, {"__global int* restrict nused, " if nused else ""}const int core_id) {{
  int ns_ = {"nused[0]" if nused else S};
  for (int u = core_id; u < {S * nsl * rb}; u += {NT}) {{
    int h = u / {nsl * rb}, sl = (u / {rb}) % {nsl}, b = u % {rb}; if (h >= ns_) continue; int g0 = (sl * 128) / 48, g1 = (sl * 128 + 127) / 48; int d = g1 - g0 == 3 ? 1 : 0;
    DMA_FILL(0, DESC(desc, d), 0, (int)(ct + ((h * {ngh} + g0) * {rb} + b) * 576));
    DMA_FILL(1, DESC(desc, d), {GO}, (int)(ct + (({S * ngh} + h * {ngh} + g0) * {rb} + b) * 576)); DMA_WAIT(0); DMA_WAIT(1);
    __global float* G = LSF(0); __global float* U = LSF({GO}); __global half* A = LSH({AO});
    for (int kk = 0; kk < 32; kk++) for (int i = 0; i < 3; i++) {{
      int col = sl * 128 + 4 * kk; __global float* pg = CTILE(G, g0, col, i); __global float* pu = CTILE(U, g0, col, i);
      *(__global half16*)(A + (kk * 3 + i) * 16) = CVT16(sw(*(__global float8*)pg, *(__global float8*)pu), sw(*(__global float8*)(pg + 8), *(__global float8*)(pu + 8)));
    }}
    DMA_DRAIN(2, DESC(desc, 2), {AO}, (int)(a + ((h * {nsl} + sl) * {rb} + b) * 1536)); DMA_WAIT(2);
  }}
  DMA_WAIT_ALL();
}}"""


# ---- the batched prefill's routing on the device: route_top (a token's top 8 and their softmax) -> route_slots (the tokens grouped
# by expert into slots of er rows: the tables the gather, the expert GEMMs and the scatter read). The same slots as the host's grouping:
# the experts ascending, each expert's tokens ascending.

def route_top_desc(NE): return V._desc_slots((NE * 4,), (64,))

def route_top_src(m, NE, TOPK=8):
  """For each of the m rows of the router's logits (fp32 [m][NE]): the TOPK largest (the first index on ties) and their softmax (=
  the renormalised top-k of the softmax over all NE), as one 64-byte line a row: [0..7] the ids (int32), [8..15] the weights.
  A task a row by DMA. args: topw (int32 [m][16]), logits, desc (route_top_desc)."""
  assert TOPK == 8 and NE % 32 == 0
  return V.FULL_H + EXP + f"""
__kernel void route_top(__global int* restrict topw, __global float* restrict lg, __global int* restrict desc, const int core_id) {{
  for (int t = core_id; t < {m}; t += {NT}) {{
    DMA_FILL(0, DESC(desc, 0), 0, (int)(lg + t * {NE})); DMA_WAIT(0);
    __global float* L = LSF(0); __global int* O = (__global int*)LSF({NE * 4});
    unsigned taken[{NE // 32}]; for (int i = 0; i < {NE // 32}; i++) taken[i] = 0u;
    float v[8]; int id[8];
    for (int k = 0; k < 8; k++) {{
      float b = -3.0e38f; int bi = 0;
      for (int e = 0; e < {NE}; e++) if (!((taken[e >> 5] >> (e & 31)) & 1u) && L[e] > b) {{ b = L[e]; bi = e; }}
      taken[bi >> 5] |= 1u << (bi & 31); v[k] = b; id[k] = bi;
    }}
    float8 tv = (float8){{v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]}};
    float8 ex = exp2_d4((tv - BC(v[0])) * BC({1.4426950408889634!r}f)); float8 w = ex * BC(1.0f / hsum8(ex));
    for (int k = 0; k < 8; k++) O[k] = id[k];
    *(__global float8*)((__global float*)O + 8) = w;
    DMA_DRAIN(1, DESC(desc, 1), {NE * 4}, (int)(topw + t * 16)); DMA_WAIT(1);
  }}
  DMA_WAIT_ALL();
}}"""

RS_CH = 64                                                                # route_slots: tokens a DMA chunk

def route_slots_desc(): return V._desc_slots((RS_CH * 64,))

def route_slots_src(m, NE, S, er, s_gu, s_dn, TOPK=8):
  """One task: route_top's lines of the first nreal[0] rows (by DMA, RS_CH rows a chunk) -> the slots of er rows: tok [S er] (-1:
  an empty row), wrow [S er] (the row's weight, 0 empty), sel_gu / sel_dn [S] (the expert's byte offset in its stack; -1: an unused
  slot, which the expert GEMMs skip), ent / wt [m][TOPK] (a token's k-th expert's slot row and weight; rows past nreal: 0), nused [1].
  args: tok, wrow, sel_gu, sel_dn, ent, wt, nused, topw, nreal, desc (route_slots_desc)."""
  assert TOPK == 8
  return V.FULL_H + f"""
__kernel void route_slots(__global int* restrict tok, __global float* restrict wrow, __global int* restrict sgu, __global int* restrict sdn,
                          __global int* restrict ent, __global float* restrict wt, __global int* restrict nused, __global int* restrict topw,
                          __global int* restrict nreal, __global int* restrict desc, const int core_id) {{
  if (core_id != 0) return;
  int n = nreal[0]; int cnt[{NE}], start[{NE}], pos[{NE}];
  for (int e = 0; e < {NE}; e++) {{ cnt[e] = 0; pos[e] = 0; }}
  __global int* T = (__global int*)LSF(0);
  for (int c0 = 0; c0 < n; c0 += {RS_CH}) {{
    DMA_FILL(0, DESC(desc, 0), 0, (int)(topw + c0 * 16)); DMA_WAIT(0);
    for (int t = c0; t < n && t < c0 + {RS_CH}; t++) for (int k = 0; k < 8; k++) cnt[T[(t - c0) * 16 + k]]++;
  }}
  int s = 0;
  for (int e = 0; e < {NE}; e++) {{ start[e] = s; s += (cnt[e] + {er - 1}) / {er}; }}
  for (int i = 0; i < {S}; i++) {{ sgu[i] = -1; sdn[i] = -1; }}
  for (int i = 0; i < {S * er}; i++) {{ tok[i] = -1; wrow[i] = 0.0f; }}
  for (int e = 0; e < {NE}; e++) for (int j = start[e]; j < (e + 1 < {NE} ? start[e + 1] : s); j++) {{ sgu[j] = e * {s_gu}; sdn[j] = e * {s_dn}; }}
  for (int i = 0; i < {m * TOPK}; i++) {{ ent[i] = 0; wt[i] = 0.0f; }}
  for (int c0 = 0; c0 < n; c0 += {RS_CH}) {{
    DMA_FILL(0, DESC(desc, 0), 0, (int)(topw + c0 * 16)); DMA_WAIT(0);
    for (int t = c0; t < n && t < c0 + {RS_CH}; t++) for (int k = 0; k < 8; k++) {{
      int e = T[(t - c0) * 16 + k]; float w = ((__global float*)T)[(t - c0) * 16 + 8 + k];
      int p = pos[e]++; int r = (start[e] + p / {er}) * {er} + p % {er};
      tok[r] = t; wrow[r] = w; ent[t * 8 + k] = r; wt[t * 8 + k] = w;
    }}
  }}
  nused[0] = s;
}}"""


# ---- the decode loop on the device: the head's token (head_reduce) -> embed_next -> the next step's input row

def embed_next_desc(c): return V._desc_slots((c // 256 * 144,), (c * 4,))

def embed_next_src(c):
  """The next token on the device: id = ids[0] (head_reduce's), p = idx[0] + 1 -> idx[0] = p, ring[p] = id, and x row 0 = token_embd
  row id dequantised (Q4_K: per 256-weight block d sc q - dmin m, the low nibbles of each 32 bytes then the high ones; ggml's order,
  as gguf_read). One task; the row in and out by DMA. args: x (fp32 rows), idx, ring (int32 [tmax]), ids, table (Q4_K rows: c / 256
  x 144 B), desc (embed_next_desc)."""
  nb = c // 256; RO = nb * 144; RO += (-RO) % 64; assert RO + c * 4 <= 32768 - 64
  return V.FULL_H + f"""
static inline float h2f(unsigned int h) {{                                   /* fp16 bits -> float (normal, subnormal, zero) */
  unsigned int sg = (h >> 15) & 1u, ex = (h >> 10) & 31u, mt = h & 1023u; float v;
  if (ex == 0u) v = (float)mt * 5.9604644775390625e-08f;
  else {{ unsigned int b = ((ex + 112u) << 23) | (mt << 13); v = *(float*)&b; }}
  return sg ? -v : v;
}}
__kernel void embed_next(__global float* restrict x, __global int* restrict idx, __global int* restrict ring, __global int* restrict ids,
                         __global unsigned char* restrict table, __global int* restrict desc, const int core_id) {{
  if (core_id != 0) return;
  int id = ids[0]; int p = idx[0] + 1; idx[0] = p; ring[p] = id;
  DMA_FILL(0, DESC(desc, 0), 0, (int)(table + id * {nb * 144})); DMA_WAIT(0);
  __global unsigned char* R = (__global unsigned char*)LSF(0); __global float* X = LSF({RO});
  for (int b = 0; b < {nb}; b++) {{
    __global unsigned char* B = R + b * 144; float d = h2f(B[0] | (B[1] << 8)), dm = h2f(B[2] | (B[3] << 8));
    __global unsigned char* sc = B + 4; __global unsigned char* qs = B + 16;
    for (int j = 0; j < 8; j++) {{
      int sv, mv;
      if (j < 4) {{ sv = sc[j] & 63; mv = sc[j + 4] & 63; }} else {{ sv = (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4); mv = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }}
      float ds = d * (float)sv, mm = dm * (float)mv; int i = j / 2, hi = j % 2;
      for (int k = 0; k < 32; k++) {{ int q = hi ? (qs[32 * i + k] >> 4) : (qs[32 * i + k] & 15); float t = ds * (float)q; X[b * 256 + j * 32 + k] = t - mm; }}
    }}
  }}
  DMA_DRAIN(1, DESC(desc, 1), {RO}, (int)x); DMA_WAIT(1);
}}"""


def embed_rows_src(c, m):
  """The prompt chunk's rows on the device: x[t] = token_embd row ids[t] dequantised (embed_next's Q4_K decode), t < m (rows past
  nreal[0]: zeros). A task a row by DMA. args: x (fp32 [m'][c]), ids (int32 [m]), nreal, table, desc (embed_next_desc)."""
  nb = c // 256; RO = nb * 144; RO += (-RO) % 64
  dec = embed_next_src(c)
  helper = dec[dec.index("static inline float h2f"):dec.index("__kernel void embed_next")]
  return V.FULL_H + helper + f"""
__kernel void embed_rows(__global float* restrict x, __global int* restrict ids, __global int* restrict nreal, __global unsigned char* restrict table,
                         __global int* restrict desc, const int core_id) {{
  int n = nreal[0];
  for (int t = core_id; t < {m}; t += {NT}) {{
    __global unsigned char* R = (__global unsigned char*)LSF(0); __global float* X = LSF({RO});
    if (t >= n) {{ for (int k = 0; k < {c}; k += 8) *(__global float8*)(X + k) = BC(0.0f); }}
    else {{
      DMA_FILL(0, DESC(desc, 0), 0, (int)(table + ids[t] * {nb * 144})); DMA_WAIT(0);
      for (int b = 0; b < {nb}; b++) {{
        __global unsigned char* B = R + b * 144; float d = h2f(B[0] | (B[1] << 8)), dm = h2f(B[2] | (B[3] << 8));
        __global unsigned char* sc = B + 4; __global unsigned char* qs = B + 16;
        for (int j = 0; j < 8; j++) {{
          int sv, mv;
          if (j < 4) {{ sv = sc[j] & 63; mv = sc[j + 4] & 63; }} else {{ sv = (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4); mv = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }}
          float ds = d * (float)sv, mm = dm * (float)mv; int i = j / 2, hi = j % 2;
          for (int k = 0; k < 32; k++) {{ int q = hi ? (qs[32 * i + k] >> 4) : (qs[32 * i + k] & 15); float tq = ds * (float)q; X[b * 256 + j * 32 + k] = tq - mm; }}
        }}
      }}
    }}
    DMA_DRAIN(1, DESC(desc, 1), {RO}, (int)(x + t * {c})); DMA_WAIT(1);
  }}
  DMA_WAIT_ALL();
}}"""


def moe_comb_ref(x, cts, w):
  """x [n, c], cts [heads, n, c] (unpacked), w [n, heads] -> x + sum_h w[:, h] C_h (the heads summed in order, as the kernel)."""
  s = np.zeros_like(x)
  for h in range(cts.shape[0]): s = s + w[:, h:h + 1] * cts[h]
  return x + s

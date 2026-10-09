"""Hand-written TEC kernels (the backend's `register_csrc`) for Gemma 4's layer pieces that the Qwen3.8 modules lack
(see gemma4_ref.py for the layer they compute). They read and write the same layouts as ../qwen3.8-27b/qwen38_kernels.py
(the GEMM's C tiles [group][rb][strip][3 row quads][4 col tiles][4 rows][4 cols], the A layout, fp32 rows) and reuse its helpers;
the GeGLU A producer is qwen38_kernels.swiglu_a32[d]_src(act="gelu").

  pnresid32:  out = (x + rms(C) * w) * s       -- the sandwich post-norm and the residual add (s: layer_output_scale, else 1)
  gmul_a32:   A = fp16(gelu_tanh(C) * P)        -- the PLE gate times the token's per-layer input (P fp32 rows), K = m_
  gattn_kv:   the new rows' k (RMSNorm x w, RoPE) and v (RMSNorm, no weight) into the layer's K / V cache at pos0 + t
  gattn_q:    per (row, q head): q RMSNorm x w and RoPE, scores against the cache (scale 1; causal, a window W on local layers),
              softmax, the V sum -> o rows. KV-shared layers call it with their source layer's caches.

RoPE is rotate-half over the whole head (dims i and i + HD / 2) from a per-layer-kind table cs [TMAX][HD]: cos of the HD / 2
frequencies then their sin (the global layers' proportional RoPE: cos 1 / sin 0 past the 64 rotated frequencies).
"""
import math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "qwen3.8-27b"))
import qwen38_kernels as QK                                              # noqa: E402
from qwen38_kernels import NT, TILE, EXP, _real, _units                  # noqa: E402
V = QK.V

LS_FLOATS = 8176                                                         # a task's LSRAM in floats (32 KiB - 64 B)

def _ctile(col, nrb):
  """The C tile row-quad base for column expression `col` of row block rb, row quad i (the 4 x 4 block holding it)."""
  return f"ct + ((({col}) / 48 * {nrb} + rb) * 3 + (({col}) % 48) / 16) * 192 + (i * 4 + (({col}) % 16) / 4) * 16"

def pnresid32_src(c, nrb, eps, real=None):
  """out [rows, c] = (x + C * inv_r * w) * s, inv_r = 1 / sqrt(mean_j C[r, j]^2 + eps) over the c columns, w = wv[0:c], s = wv[c].
  A unit = (row block, a line-aligned column range) as resid32; each unit first sums its rows' squares over all c columns (the
  real rows only: padding rows get inv 0 and are written as x * s -- never read). args: out, x, ct, wv."""
  assert c % 16 == 0; nq = c // 4; rbr, iqr = _real(real, nrb); rows = 12 * rbr if real is None else real
  P = max(1, min(nq // 4, NT // rbr)); chunk = -(-nq // P); chunk += (-chunk) % 4
  return V.FULL_H + TILE + f"""
__kernel void pnresid32(__global float* restrict out, __global float* restrict x, __global float* restrict ct, __global float* restrict wv, const int core_id) {{
  float s = wv[{c}];
  for (int u = core_id; u < {rbr * P}; u += {NT}) {{
    int rb = u / {P}, part = u % {P}; int q1 = (part + 1) * {chunk} < {nq} ? (part + 1) * {chunk} : {nq};
    for (int i = 0; i < {iqr}; i++) {{
      float inv[4];
      for (int r = 0; r < 4; r++) {{
        int m = rb * 12 + 4 * i + r; float4 acc = (float4){{0.0f, 0.0f, 0.0f, 0.0f}};
        if (m < {rows}) for (int col = 0; col < {c}; col += 4) {{ float4 v = *(__global float4*)({_ctile("col", nrb)} + 4 * r); acc += v * v; }}
        inv[r] = (m < {rows}) ? 1.0f / __builtin_sqrtf((acc[0] + acc[1] + acc[2] + acc[3]) * {1.0 / c!r}f + {eps!r}f) : 0.0f;
      }}
      for (int q = part * {chunk}; q < q1; q++) {{
        int col = 4 * q; __global float* t = {_ctile("col", nrb)};
        float4 w4 = *(__global float4*)(wv + col);
        __global float* xr = x + (rb * 12 + 4 * i) * {c} + col; __global float* orow = out + (rb * 12 + 4 * i) * {c} + col;
        for (int r = 0; r < 4; r++) *(__global float4*)(orow + r * {c}) = (*(__global float4*)(xr + r * {c}) + *(__global float4*)(t + 4 * r) * (float4){{inv[r], inv[r], inv[r], inv[r]}} * w4) * (float4){{s, s, s, s}};
      }}
    }}
  }}
}}"""

# ---- pnresid32 by DMA (rows mode, real <= 12 rows, any c % 16 == 0): two kernels, because the norm needs the whole row of C before
# any column can be written and C's row is scattered over every group's tiles. pnss: task t sums the squares of its groups' columns
# (the iqr = ceil(real / 4) row quads of row block 0 by one strided DMA) -> ss[t][16]; pnap: a unit (a task, or more units than tasks
# when a task's groups would not fit LSRAM: c 2560 at 12 rows) reads every task's partials (768 B), its groups' C pieces, its columns
# of the x rows and of w, and writes its columns of the out rows -- all by DMA. The 12 partials are added in task order. c not a
# multiple of 48 (E4B's 2560, the MTP drafter's 256): the last group's padding strips are skipped, the last unit's columns end at c.
def _pn_groups(c): ng = -(-c // 48); return ng, -(-ng // NT)

def _pn_cfg(c, real, nrb=2):
  """-> (ng, gm, iqr, PB, tl, GU, NU): groups, a task's most groups (pnss), row quads, a C piece's bytes, the padding columns of the
  last group, pnap's most groups a unit, pnap's units."""
  ng, gm = _pn_groups(c); iqr = min(3, -(-real // 4)); PB = 256 * iqr; tl = 48 * ng - c
  fit = lambda g: ((g - 1) * 3 * nrb + 3) * PB + real * g * 48 * 4 + NT * 64 + g * 48 * 4 <= 32768 - 64
  GU = max(g for g in range(1, gm + 1) if fit(g)); NU = max(NT, -(-ng // GU))
  return ng, gm, iqr, PB, tl, GU, NU

def pnres_desc(c, real):
  """slots 0 .. gm-1: g groups' row-quad pieces of row block 0 (g = slot + 1; iqr row quads a strip; the other row block's ride
  along, as head_topd's); then for g = 1 .. GU: the real rows of g groups' columns (pitch c); with padding columns (tl > 0) the same
  for the last unit's g groups less tl columns; the 12 tasks' partials (768 B); g groups' columns of w (and the last unit's)."""
  ng, gm, iqr, PB, tl, GU, NU = _pn_cfg(c, real); nrb = 2
  xs = [(real * g * 48 * 4, g * 48 * 4, c * 4, g * 48 * 4) for g in range(1, GU + 1)]
  if tl: xs += [(real * (g * 48 - tl) * 4, (g * 48 - tl) * 4, c * 4, (g * 48 - tl) * 4) for g in range(1, GU + 1)]
  ws = [(g * 48 * 4,) for g in range(1, GU + 1)] + ([((g * 48 - tl) * 4,) for g in range(1, GU + 1)] if tl else [])
  return QK.V._desc_slots(*[(((g - 1) * 3 * nrb + 3) * PB, PB, 768, PB) for g in range(1, max(gm, GU) + 1)], *xs, (NT * 64,), *ws)

def _pn_slots(c, real):
  """slot bases: (C, X, X tail, SS, W, W tail)."""
  ng, gm, iqr, PB, tl, GU, NU = _pn_cfg(c, real); C0 = 0; X0 = max(gm, GU); XT = X0 + GU; SS = XT + (GU if tl else 0); W0 = SS + 1; WT = W0 + GU
  return C0, X0, XT, SS, W0, WT

def pnss_src(c, nrb, real):
  """Task t's partial sums of squares of C's real rows over its groups [t ng / NT, (t + 1) ng / NT) -> ss[t * 16 + r] (r < real;
  0 for r >= real). args: ss, ct, desc (pnres_desc)."""
  ng, gm, iqr, PB, tl, GU, NU = _pn_cfg(c, real); assert c % 16 == 0 and 1 <= real <= 12 and nrb == 2
  F = PB // 4; strip_ok = f"if ((g0 + gg) * 48 + s * 16 < {c}) " if tl else ""
  sums = ", ".join(f"acc[{r}][0] + acc[{r}][1] + acc[{r}][2] + acc[{r}][3]" if r < real else "0.0f" for r in range(16))
  return V.FULL_H + TILE + f"""
__kernel void pnss(__global float* restrict ss, __global float* restrict ct, __global int* restrict desc, const int core_id) {{
  int g0 = core_id * {ng} / {NT}, g1 = (core_id + 1) * {ng} / {NT}, gn = g1 - g0;
  float4 acc[{real}]; for (int r = 0; r < {real}; r++) acc[r] = (float4){{0.0f, 0.0f, 0.0f, 0.0f}};
  if (gn > 0) {{
    DMA_FILL(0, DESC(desc, gn - 1), 0, (int)(ct + g0 * {3 * nrb * 192})); DMA_WAIT(0);
    for (int gg = 0; gg < gn; gg++) for (int s = 0; s < 3; s++) {strip_ok}{{
      __global float* q = LSF(0) + (gg * {3 * nrb} + s) * {F};
      for (int jt = 0; jt < 4; jt++) for (int r = 0; r < {real}; r++) {{ float4 v = *(__global float4*)(q + (r / 4) * 64 + jt * 16 + (r % 4) * 4); acc[r] += v * v; }}
    }}
  }}
  float o[16] = {{{sums}}};
  for (int r = 0; r < 16; r += 4) *(__global float4*)(ss + core_id * 16 + r) = (float4){{o[r], o[r + 1], o[r + 2], o[r + 3]}};
  DMA_WAIT_ALL();
}}"""

def pnap_src(c, nrb, eps, real):
  """out [rows, c] = (x + C * inv_r * w) * s on the real rows (inv_r from the 12 partials of pnss: 1 / sqrt(sum / c + eps)), each
  unit its groups' columns; w = wv[0:c], s = wv[c]. Rows >= real are not written. args: out, x, ct, wv, ss, desc (pnres_desc)."""
  ng, gm, iqr, PB, tl, GU, NU = _pn_cfg(c, real); assert c % 16 == 0 and 1 <= real <= 12 and nrb == 2
  C0, X0, XT, SS, W0, WT = _pn_slots(c, real); F = PB // 4
  CB = ((GU - 1) * 3 * nrb + 3) * PB; XO = CB; SO = XO + real * GU * 48 * 4; WO = SO + NT * 64; assert WO + GU * 48 * 4 <= 32768 - 64
  tail = (lambda full, t: f"(tail ? {t} : {full})") if tl else (lambda full, t: full)
  return V.FULL_H + TILE + f"""
__kernel void pnap(__global float* restrict out, __global float* restrict x, __global float* restrict ct, __global float* restrict wv,
                   __global float* restrict ss, __global int* restrict desc, const int core_id) {{
  float inv[{real}]; int have = 0;
  for (int u = core_id; u < {NU}; u += {NT}) {{
  int g0 = u * {ng} / {NU}, g1 = (u + 1) * {ng} / {NU}, gn = g1 - g0; int tail = g1 == {ng}; int W = gn * 48{" - (tail ? %d : 0)" % tl if tl else ""};
  if (gn > 0) {{
    DMA_FILL(0, DESC(desc, gn - 1), 0, (int)(ct + g0 * {3 * nrb * 192}));
    DMA_FILL(1, DESC(desc, {tail(f"{X0} + gn - 1", f"{XT} + gn - 1")}), {XO}, (int)(x + g0 * 48));
    if (!have) DMA_FILL(2, DESC(desc, {SS}), {SO}, (int)ss);
    DMA_FILL(3, DESC(desc, {tail(f"{W0} + gn - 1", f"{WT} + gn - 1")}), {WO}, (int)(wv + g0 * 48));
    if (!have) {{
      DMA_WAIT(2); __global float* S = LSF({SO});
      for (int r = 0; r < {real}; r++) {{ float t = 0.0f; for (int k = 0; k < {NT}; k++) t += S[k * 16 + r]; inv[r] = 1.0f / __builtin_sqrtf(t * {1.0 / c!r}f + {eps!r}f); }}
      have = 1;
    }}
    float s = wv[{c}];
    DMA_WAIT(0); DMA_WAIT(1); DMA_WAIT(3);
    __global float* X = LSF({XO}); __global float* Wl = LSF({WO}); float4 s4 = (float4){{s, s, s, s}};
    for (int k = 0; k < W; k += 4) {{
      __global float* q = LSF(0) + ((k / 48) * {3 * nrb} + (k % 48) / 16) * {F} + ((k % 16) / 4) * 16;
      float4 w4 = *(__global float4*)(Wl + k);
      for (int r = 0; r < {real}; r++) {{
        float4 iv = (float4){{inv[r], inv[r], inv[r], inv[r]}};
        *(__global float4*)(X + r * W + k) = (*(__global float4*)(X + r * W + k) + *(__global float4*)(q + (r / 4) * 64 + (r % 4) * 4) * iv * w4) * s4;
      }}
    }}
    DMA_DRAIN(3, DESC(desc, {tail(f"{X0} + gn - 1", f"{XT} + gn - 1")}), {XO}, (int)(out + g0 * 48)); DMA_WAIT(3);
  }}
  }}
  DMA_WAIT_ALL();
}}"""

def gmul_a32_src(m_, nrb, real=None, compact=False, poff=False):
  """C tiles (N >= m_: the PLE gate, groups of 48) and fp32 rows P [12 nrb, m_] (the token's per-layer input) -> the A layout of
  K = m_: fp16(gelu_tanh(C) * P) (`compact`: rows mode, as rms_a32). The GELU is swiglu_a32(act="gelu")'s `gl`.
  `poff`: a fourth argument, int32 pidx [1]: P starts pidx[0] floats into the buffer (one stack of every layer's rows, the layer's
  offset written once: the same kernel for every layer, no copy). poff=False leaves the source byte-identical."""
  assert m_ % 128 == 0; nsl = m_ // 128; rbr, iqr = _real(real, nrb); P = _units(rbr, nsl)
  assert not compact or rbr == 1
  obase, oel = ((f"sl * {32 * 16 * iqr}", f"(kq * {iqr} + i) * 16") if compact else (f"((sl * {nrb} + rb) * 32) * 48", "(kq * 3 + i) * 16"))
  pidx, padv = ("__global int* restrict pidx, ", "  pl += pidx[0];\n") if poff else ("", "")
  return V.FULL_H + TILE + EXP + QK.GELU + f"""
__kernel void gmul_a32(__global half* restrict out, __global float* restrict ct, __global float* restrict pl, {pidx}const int core_id) {{
{padv}  for (int u = core_id; u < {rbr * P}; u += {NT}) {{
    int rb = u / {P}, part = u % {P};
    for (int sl = part; sl < {nsl}; sl += {P}) {{
      __global half* o = out + {obase};
      for (int i = 0; i < {iqr}; i++) for (int kq = 0; kq < 32; kq++) {{
        int col = sl * 128 + 4 * kq; __global float* g = {_ctile("col", nrb)};
        __global float* p = pl + (rb * 12 + 4 * i) * {m_} + col;
        float8 p01 = F8(*(__global float4*)(p), *(__global float4*)(p + {m_})), p23 = F8(*(__global float4*)(p + {2 * m_}), *(__global float4*)(p + {3 * m_}));
        float8 t01 = gl(*(__global float8*)(g), p01), t23 = gl(*(__global float8*)(g + 8), p23);
        *(__global half16*)(o + {oel}) = CVT16(t01, t23);
      }}
    }}
  }}
}}"""

def gmul_a32d_desc(m_, nrb=2, real=4):
  """slots 0 .. 3: g + 1 groups' row-quad pieces of row block 0 (iqr = ceil(real / 4) row quads a strip; as head_topd's); 4: rows
  0 .. 4 iqr - 1 of one 128-column slice of P (pitch m_); 5: one slice's compact A (iqr tiles)."""
  iqr = min(3, -(-real // 4)); PB = 256 * iqr
  return QK.V._desc_slots(*[((g * 3 * nrb + 3) * PB, PB, 768, PB) for g in range(4)], (4 * iqr * 128 * 4, 128 * 4, m_ * 4, 128 * 4), (32 * 16 * 2 * iqr,))

def gmul_a32d_src(m_, nrb, real, poff=True):
  """gmul_a32 (compact, real <= 12) by DMA: a unit = one 128-column slice -- the C groups covering it, rows 0 .. 4 iqr - 1 of P (at
  pidx[0] floats into the stack when `poff`) arrive by DMA, the slice's A (32 x 16 iqr halves) leaves by one. The same arithmetic in
  the same order as gmul_a32: the same bits. args: out, ct, pl[, pidx], desc (gmul_a32d_desc(m_, nrb, real))."""
  assert m_ % 128 == 0 and 1 <= real <= 12 and nrb == 2; nsl = m_ // 128; iqr = min(3, -(-real // 4)); PB = 256 * iqr; F = PB // 4
  CB = (3 * 3 * nrb + 3) * PB; PO = CB; AO = PO + 4 * iqr * 128 * 4; assert AO + 1024 * iqr <= 32768 - 64
  return V.FULL_H + TILE + EXP + QK.GELU + f"""
__kernel void gmul_a32d(__global half* restrict out, __global float* restrict ct, __global float* restrict pl, {"__global int* restrict pidx, " if poff else ""}__global int* restrict desc, const int core_id) {{
{"  pl += pidx[0];" if poff else ""}
  for (int sl = core_id; sl < {nsl}; sl += {NT}) {{
    int ga = sl * 128 / 48, gb = (sl * 128 + 127) / 48;
    DMA_FILL(0, DESC(desc, gb - ga), 0, (int)(ct + ga * {3 * nrb * 192}));
    DMA_FILL(1, DESC(desc, 4), {PO}, (int)(pl + sl * 128)); DMA_WAIT(0); DMA_WAIT(1);
    __global float* P = LSF({PO}); __global half* A = (__global half*)LSF({AO});
    for (int i = 0; i < {iqr}; i++) for (int kq = 0; kq < 32; kq++) {{
      int col = sl * 128 + 4 * kq; __global float* g = LSF(0) + ((col / 48 - ga) * {3 * nrb} + (col % 48) / 16) * {F} + i * 64 + ((col % 16) / 4) * 16;
      __global float* Pi = P + i * 512;
      float8 p01 = F8(*(__global float4*)(Pi + 4 * kq), *(__global float4*)(Pi + 128 + 4 * kq)), p23 = F8(*(__global float4*)(Pi + 256 + 4 * kq), *(__global float4*)(Pi + 384 + 4 * kq));
      float8 t01 = gl(*(__global float8*)(g), p01), t23 = gl(*(__global float8*)(g + 8), p23);
      *(__global half16*)(A + (kq * {iqr} + i) * 16) = CVT16(t01, t23);
    }}
    DMA_DRAIN(2, DESC(desc, 5), {AO}, (int)(out + sl * {512 * iqr})); DMA_WAIT(2);
  }}
  DMA_WAIT_ALL();
}}"""

def ple_mix32_src(NL, P, H, nrb, eps, real, prow=12):
  """The per-layer inputs (E-series, SPEC 2e) on the device: out [NL][prow][P] fp32 rows (layer l's rows at l prow P, the stack
  gmul_a32(poff) reads) = (rms(C[r, l P : (l + 1) P] / sqrt(H)) * w + T[r, l P : (l + 1) P]) / sqrt(2), C the C tiles of the
  per_layer_model_proj GEMM (N = NL P), T fp32 rows [prow][NL P] (the token's table row x sqrt(P), looked up on the host), w the
  per_layer_proj_norm [P]. Only the `real` rows. A unit = (row, layer). args: out, ct, tok, w."""
  assert P % 4 == 0 and real <= prow; N = NL * P; sq = 1.0 / math.sqrt(H)
  return V.FULL_H + TILE + f"""
__kernel void ple_mix32(__global float* restrict out, __global float* restrict ct, __global float* restrict tok, __global float* restrict w, const int core_id) {{
  for (int u = core_id; u < {real * NL}; u += {NT}) {{
    int m = u / {NL}, l = u % {NL}; int rb = m / 12, i = (m % 12) / 4, r = m % 4;
    float4 acc = (float4){{0.0f, 0.0f, 0.0f, 0.0f}};
    for (int j = 0; j < {P}; j += 4) {{ int col = l * {P} + j; float4 v = *(__global float4*)({_ctile("col", nrb)} + 4 * r); acc += v * v; }}
    float inv = {sq!r}f / __builtin_sqrtf((acc[0] + acc[1] + acc[2] + acc[3]) * {1.0 / (P * H)!r}f + {eps!r}f);
    __global float* o = out + (l * {prow} + m) * {P}; __global float* t = tok + m * {N} + l * {P};
    float4 iv = (float4){{inv, inv, inv, inv}}, h = (float4){{{1 / math.sqrt(2.0)!r}f, {1 / math.sqrt(2.0)!r}f, {1 / math.sqrt(2.0)!r}f, {1 / math.sqrt(2.0)!r}f}};
    for (int j = 0; j < {P}; j += 4) {{
      int col = l * {P} + j; float4 v = *(__global float4*)({_ctile("col", nrb)} + 4 * r);
      *(__global float4*)(o + j) = (v * iv * *(__global float4*)(w + j) + *(__global float4*)(t + j)) * h;
    }}
  }}
}}"""

def _norm_rope(HD, eps):
  """norm_rope(dst, src, w, cs): dst (LSRAM) = rotate-half RoPE of src * rsqrt(mean(src^2) + eps) * w (w NULL: no weight;
  cs NULL: no rotation) -- the table row cs = [cos HD/2 | sin HD/2]."""
  h = HD // 2
  return f"""
static inline __attribute__((always_inline)) void norm_rope(__global float* restrict dst, __global float* restrict src, __global float* restrict w, __global float* restrict cs) {{
  float8 acc = BC(0.0f);
  for (int i = 0; i < {HD}; i += 8) {{ float8 v = *(__global float8*)(src + i); acc += v * v; }}
  float8 inv = BC(1.0f / __builtin_sqrtf(hsum8(acc) * {1.0 / HD!r}f + {eps!r}f));
  if (w) {{ for (int i = 0; i < {HD}; i += 8) *(__global float8*)(dst + i) = *(__global float8*)(src + i) * inv * *(__global float8*)(w + i); }}
  else {{ for (int i = 0; i < {HD}; i += 8) *(__global float8*)(dst + i) = *(__global float8*)(src + i) * inv; }}
  if (cs) for (int i = 0; i < {h}; i += 8) {{
    float8 a = *(__global float8*)(dst + i), b = *(__global float8*)(dst + {h} + i), c = *(__global float8*)(cs + i), s = *(__global float8*)(cs + {h} + i);
    *(__global float8*)(dst + i) = a * c - b * s; *(__global float8*)(dst + {h} + i) = b * c + a * s;
  }}
}}
"""

def gattn_kv_src(NKV, HD, TMAX, eps, M, KR, vsame=False):
  """The M new rows (positions idx[0] + t) of a cache-owning layer into its caches: Kc[pos][h] = rope(rms(k) * knw),
  Vc[pos][h] = rms(v) (no weight). kv_rows: fp32 rows of pitch KR floats, k heads at 0, v heads at NKV HD (`vsame`: V = K, the
  raw k -- Gemma 4 12B / 26B global layers). A unit = (t, kv head); the normed row is built in LSRAM, then stored.
  args: Kc (out) [TMAX][NKV][HD], Vc, kv_rows, knw [HD], cs [TMAX][HD], idx (int32 [1]: pos0)."""
  assert HD % 16 == 0
  return V.FULL_H + EXP + _norm_rope(HD, eps) + f"""
__kernel void gattn_kv(__global float* restrict Kc, __global float* restrict Vc, __global float* restrict kv_rows, __global float* restrict knw,
                       __global float* restrict cs, __global int* restrict idx, const int core_id) {{
  int pos0 = idx[0]; __global float* tmp = LSF(0);
  for (int u = core_id; u < {M * NKV}; u += {NT}) {{
    int t = u / {NKV}, h = u % {NKV}; int pos = pos0 + t;
    __global float* kr = kv_rows + t * {KR} + h * {HD}; __global float* vr = kv_rows + t * {KR} + {0 if vsame else NKV * HD} + h * {HD};
    norm_rope(tmp, kr, knw, cs + pos * {HD});
    for (int i = 0; i < {HD}; i += 8) *(__global float8*)(Kc + (pos * {NKV} + h) * {HD} + i) = *(__global float8*)(tmp + i);
    norm_rope(tmp, vr, (__global float*)0, (__global float*)0);
    for (int i = 0; i < {HD}; i += 8) *(__global float8*)(Vc + (pos * {NKV} + h) * {HD} + i) = *(__global float8*)(tmp + i);
  }}
}}"""

def gattn_q_src(NH, NKV, HD, TMAX, eps, M, QR, W=0):
  """Attention of M query rows (positions idx[0] + t) over a K / V cache holding positions 0 .. idx[0] + M - 1: per unit (t, q
  head h): q = rope(rms(q) * qnw) in LSRAM, scores q . K[t'] (scale 1) for t' in [lo, pos] (lo = max(0, pos - W + 1) when W,
  else 0), softmax (exp2 polynomial), o = sum p V / sum p -> o_rows [t][h * HD] (pitch NH HD). q_rows pitch QR floats.
  LSRAM: q, the output accumulator and the scores: 2 HD + TMAX floats.
  args: o_rows (out), q_rows, Kc, Vc, qnw [HD], cs [TMAX][HD], idx (int32 [1]: pos0)."""
  g = NH // NKV; LOG2E = 1.4426950408889634; assert HD % 16 == 0 and 2 * HD + TMAX <= LS_FLOATS, (HD, TMAX)
  lo = f"(pos - {W - 1} > 0 ? pos - {W - 1} : 0)" if W else "0"
  return V.FULL_H + EXP + _norm_rope(HD, eps) + f"""
__kernel void gattn_q(__global float* restrict o_rows, __global float* restrict q_rows, __global float* restrict Kc, __global float* restrict Vc,
                      __global float* restrict qnw, __global float* restrict cs, __global int* restrict idx, const int core_id) {{
  int pos0 = idx[0]; __global float* qs = LSF(0); __global float* oa = qs + {HD}; __global float* sc = oa + {HD};
  for (int u = core_id; u < {M * NH}; u += {NT}) {{
    int t = u / {NH}, h = u % {NH}; int kvh = h / {g}; int pos = pos0 + t; int lo = {lo};
    norm_rope(qs, q_rows + t * {QR} + h * {HD}, qnw, cs + pos * {HD});
    float m = -1e30f;
    for (int tt = lo; tt <= pos; tt++) {{
      __global float* kr = Kc + (tt * {NKV} + kvh) * {HD}; float8 acc = BC(0.0f);
      for (int i = 0; i < {HD}; i += 8) acc += *(__global float8*)(qs + i) * *(__global float8*)(kr + i);
      float sv = hsum8(acc); sc[tt - lo] = sv; if (sv > m) m = sv;
    }}
    float sum = 0.0f;
    for (int tt = lo; tt <= pos; tt++) {{ float8 e = exp2_d4(BC((sc[tt - lo] - m) * {LOG2E!r}f)); sc[tt - lo] = e[0]; sum += e[0]; }}
    for (int i = 0; i < {HD}; i += 8) *(__global float8*)(oa + i) = BC(0.0f);
    for (int tt = lo; tt <= pos; tt++) {{
      __global float* vr = Vc + (tt * {NKV} + kvh) * {HD}; float8 pb = BC(sc[tt - lo]);
      for (int i = 0; i < {HD}; i += 8) *(__global float8*)(oa + i) += pb * *(__global float8*)(vr + i);
    }}
    float8 inv = BC(1.0f / sum); __global float* o = o_rows + t * {NH * HD} + h * {HD};
    for (int i = 0; i < {HD}; i += 8) *(__global float8*)(o + i) = *(__global float8*)(oa + i) * inv;
  }}
}}"""



# ---- attention by DMA, in kv-head units (decode / short verify: M <= 4 rows). gattn_q's units were (row, q head): each walked every
# cached K / V row with plain loads -- Gemma's 8 q heads share one kv head, so every row was read 8 times at load latency. Here a unit
# = (a group of HG q heads of one kv head, a slice of the visible positions); its K / V rows stream through LSRAM in blocks of BT rows
# (double-buffered DMA) and each block serves all HG x M (head, row) pairs, with the softmax online (a block's max rescales the
# accumulators once): no score array, so no limit on the positions. gattn_comb merges the slices. `ring` (local layers): the cache
# holds `ring` rows, position t at row t % ring (ring >= W + M - 1: a pass's new rows never overwrite a row one of its rows sees).
GA_SLACK = 1024                                                          # LSRAM bytes kept for the m / l pairs and the descriptors' slack

def gattn_cfgr(NH, NKV, HD, M):
  """-> (HG, BT, P, MR): q heads a unit, cache rows a DMA block, position slices, query rows a unit -- MR the largest divisor of M,
  then the largest HG (a divisor of the group) whose q and o accumulators (2 MR HG HD floats), the RoPE row and q norm weight (2 HD)
  and two K + two V blocks fit LSRAM with BT >= 1 (BT up to 4), then P slices so the units fill the 12 tasks. M <= 4: MR = M (one
  row group: the decode / short-verify kernels as before); longer passes (prefill chunks, 5..12-row verify) split the rows."""
  g = NH // NKV
  for MR in [r for r in range(M, 0, -1) if M % r == 0]:
    for HG in [h for h in range(g, 0, -1) if g % h == 0]:
      for BT in (4, 3, 2, 1):
        if 2 * MR * HG * HD * 4 + 2 * HD * 4 + 4 * BT * HD * 4 + GA_SLACK <= 32768 - 64:
          nu = NKV * (g // HG) * (M // MR); return HG, BT, (max(1, NT // nu) if nu < NT else 1), MR
  raise AssertionError((NH, NKV, HD, M))

def gattn_cfg(NH, NKV, HD, M): return gattn_cfgr(NH, NKV, HD, M)[:3]

def _ga_block(M, HG, HD):
  """A unit's record block in floats: its o [M][HG][HD] then its (max, sum) pairs [M][HG][2], padded to whole 64-byte lines."""
  return M * HG * HD + -(-2 * M * HG // 16) * 16

def gattn_part_desc(NKV, HD, M, HG, BT):
  """slots 0 .. BT-1: b + 1 cache rows of one kv head (pitch NKV HD); BT: one row's HG heads of q (contiguous); BT + 1: one HD row
  (the RoPE row, the q norm weight); BT + 2: a unit's record block (M: the unit's rows, MR)."""
  return QK.V._desc_slots(*[((b + 1) * HD * 4, HD * 4, NKV * HD * 4, HD * 4) for b in range(BT)], (HG * HD * 4,), (HD * 4,), (_ga_block(M, HG, HD) * 4,))

def gattn_part_src(NH, NKV, HD, eps, M, QR, W=0, ring=0):
  """Unit u = (kv head kh, head group hg, row group rg, slice c): the MR query rows r0 + r (r0 = rg MR; positions idx[0] + r0 + r,
  pitch QR floats in q_rows) of heads h0 .. h0 + HG - 1 are RMS-normed (x qnw) and rotated in LSRAM, then every visible cached
  position t in the slice (the slices split [lo, pos + r0 + MR - 1], lo = max(0, pos + r0 - W + 1) with a window W, else 0; row r
  sees lo_r <= t <= pos + r) is scored (scale 1) against them and accumulated online. Every operand moves by DMA (q, the RoPE row,
  the norm weight, the K / V blocks, the result). Out: the unit's record block (_ga_block(MR) floats) at part + (u / P + (u % P)
  NKV NHG NRG) * block -- [slice][unit of heads and rows]: o [MR][HG][HD] (unnormalised), then (max, sum) [MR][HG] (sum 0: nothing
  seen). args: part, q_rows, Kc, Vc, qnw, cs, idx (pos0), desc (gattn_part_desc(NKV, HD, MR, HG, BT))."""
  HG, BT, P, MR = gattn_cfgr(NH, NKV, HD, M); g = NH // NKV; NHG = g // HG; NRG = M // MR; NU = NKV * NHG * NRG * P; LOG2E = 1.4426950408889634; BLK = _ga_block(MR, HG, HD)
  QS, OS = 0, MR * HG * HD * 4; ML = OS + MR * HG * HD * 4; CSO = OS + BLK * 4; QNO = CSO + HD * 4; KB = QNO + HD * 4; VB = KB + 2 * BT * HD * 4
  assert VB + 2 * BT * HD * 4 <= 32768 - 64 and BT <= 8, (VB, BT)
  row = (lambda t: f"(({t}) % {ring})") if ring else (lambda t: f"({t})")
  lo_r = (lambda r: f"(pos + {r} - {W - 1} > 0 ? pos + {r} - {W - 1} : 0)") if W else (lambda r: "0")
  blen = f"({BT} < t1 - (t) ? {BT} : t1 - (t))"
  if ring: blen = f"({blen} < {ring} - (t) % {ring} ? {blen} : {ring} - (t) % {ring})"
  return V.FULL_H + EXP + _norm_rope(HD, eps) + f"""
__kernel void gattn_part(__global float* restrict part, __global float* restrict q_rows, __global float* restrict Kc, __global float* restrict Vc,
                         __global float* restrict qnw, __global float* restrict cs, __global int* restrict idx, __global int* restrict desc, const int core_id) {{
  int pos = idx[0];
  __global float* qs = LSF({QS}); __global float* os = LSF({OS}); __global float* ml = LSF({ML}); __global float* qw = LSF({QNO});
  if (core_id < {NU}) {{ DMA_FILL(3, DESC(desc, {BT + 1}), {QNO}, (int)qnw); DMA_WAIT(3); }}
  for (int u = core_id; u < {NU}; u += {NT}) {{
    int kh = u / {NHG * NRG * P}, hg = (u / {NRG * P}) % {NHG}, rg = (u / {P}) % {NRG}, c = u % {P}; int h0 = kh * {g} + hg * {HG}, r0 = rg * {MR};
    int lo = {lo_r("r0")}, T = pos + r0 + {MR} - lo;
    int t0 = lo + c * T / {P}, t1 = lo + (c + 1) * T / {P};
    for (int r = 0; r < {MR}; r++) {{
      DMA_FILL(2, DESC(desc, {BT}), {OS} + r * {HG * HD * 4}, (int)(q_rows + (r0 + r) * {QR} + h0 * {HD}));   /* raw q staged in os */
      DMA_FILL(3, DESC(desc, {BT + 1}), {CSO}, (int)(cs + (pos + r0 + r) * {HD})); DMA_WAIT(2); DMA_WAIT(3);
      for (int h = 0; h < {HG}; h++) norm_rope(qs + (r * {HG} + h) * {HD}, os + (r * {HG} + h) * {HD}, qw, LSF({CSO}));
    }}
    float mx[{MR * HG}], sm[{MR * HG}];
    for (int i = 0; i < {MR * HG}; i++) {{ mx[i] = -3.0e38f; sm[i] = 0.0f; }}
    for (int i = 0; i < {MR * HG * HD}; i += 8) *(__global float8*)(os + i) = BC(0.0f);
    #define BLEN(t) {blen}
    #define ISSUE(t, buf) {{ int bl_ = BLEN(t); DMA_FILL(buf, DESC(desc, bl_ - 1), {KB} + (buf) * {BT * HD * 4}, (int)(Kc + {row("t")} * {NKV * HD} + kh * {HD})); \
                            DMA_FILL((buf) + 2, DESC(desc, bl_ - 1), {VB} + (buf) * {BT * HD * 4}, (int)(Vc + {row("t")} * {NKV * HD} + kh * {HD})); }}
    int t = t0, p = 0;
    if (t < t1) ISSUE(t, 0);
    while (t < t1) {{
      int bl = BLEN(t), tn = t + bl;
      if (tn < t1) {{ if (p == 0) ISSUE(tn, 1) else ISSUE(tn, 0) }}
      if (p == 0) {{ DMA_WAIT(0); DMA_WAIT(2); }} else {{ DMA_WAIT(1); DMA_WAIT(3); }}
      __global float* kb = LSF({KB} + p * {BT * HD * 4}); __global float* vb = LSF({VB} + p * {BT * HD * 4});
      for (int r = 0; r < {MR}; r++) {{
        int lr = {lo_r("r0 + r")}, hr = pos + r0 + r;
        for (int h = 0; h < {HG}; h++) {{
          int i = r * {HG} + h; __global float* q = qs + i * {HD}; float8 s8 = BC(-3.0e38f);
          for (int j = 0; j < bl; j++) {{
            int tt = t + j; if (tt < lr || tt > hr) continue;
            float8 a = BC(0.0f); __global float* k = kb + j * {HD};
            for (int e = 0; e < {HD}; e += 8) a += *(__global float8*)(q + e) * *(__global float8*)(k + e);
            s8[j] = hsum8(a);
          }}
          float bm = hmax8(s8); if (bm < -1.0e38f) continue;
          __global float* o = os + i * {HD};
          if (bm > mx[i]) {{
            float8 f = exp2_d4(BC((mx[i] - bm) * {LOG2E!r}f)); sm[i] *= f[0];
            for (int e = 0; e < {HD}; e += 8) *(__global float8*)(o + e) *= f;
            mx[i] = bm;
          }}
          float8 pr = exp2_d4((s8 - BC(mx[i])) * BC({LOG2E!r}f));
          for (int j = 0; j < bl; j++) {{
            if (s8[j] < -1.0e38f) continue;
            float8 pb = BC(pr[j]); sm[i] += pr[j]; __global float* v = vb + j * {HD};
            for (int e = 0; e < {HD}; e += 8) *(__global float8*)(o + e) += pb * *(__global float8*)(v + e);
          }}
        }}
      }}
      t = tn; p ^= 1;
    }}
    for (int i = 0; i < {MR * HG}; i++) {{ ml[2 * i] = mx[i]; ml[2 * i + 1] = sm[i]; }}
    DMA_DRAIN(2, DESC(desc, {BT + 2}), {OS}, (int)(part + (u / {P} + c * {NKV * NHG * NRG}) * {BLK})); DMA_WAIT(2);
    #undef ISSUE
    #undef BLEN
  }}
  DMA_WAIT_ALL();
}}"""

def gattn_comb_desc(NH, NKV, HD, M):
  """slot 0: one (row, head)'s o of every slice (P pieces of HD floats at the slices' pitch); 1: its (max, sum) pairs (P pieces of
  2 floats); 2: one o row's head (HD floats)."""
  HG, BT, P, MR = gattn_cfgr(NH, NKV, HD, M); SP = NKV * (NH // NKV // HG) * (M // MR) * _ga_block(MR, HG, HD) * 4
  return QK.V._desc_slots((P * HD * 4, HD * 4, SP, HD * 4), (P * 8, 8, SP, 8) if P > 1 else (8,), (HD * 4,))

def gattn_comb_src(NH, NKV, HD, M):
  """gattn_part's P slices merged per (row r, q head h): o = sum_c e^(m_c - m) o_c / sum_c e^(m_c - m) l_c -> o_rows [r][h HD]
  (pitch NH HD); the slices' o and (max, sum) arrive by two strided DMAs, the row leaves by one. args: o_rows, part, desc."""
  HG, BT, P, MR = gattn_cfgr(NH, NKV, HD, M); g = NH // NKV; NHG = g // HG; NRG = M // MR; BLK = _ga_block(MR, HG, HD); LOG2E = 1.4426950408889634
  OB = P * HD * 4; MB = OB + HD * 4; assert MB + P * 8 <= 32768 - 64
  return V.FULL_H + EXP + f"""
__kernel void gattn_comb(__global float* restrict o_rows, __global float* restrict part, __global int* restrict desc, const int core_id) {{
  for (int u = core_id; u < {M * NH}; u += {NT}) {{
    int r = u / {NH}, h = u % {NH}; int un = ((h / {g}) * {NHG} + (h % {g}) / {HG}) * {NRG} + r / {MR}, i = (r % {MR}) * {HG} + h % {HG};
    __global float* base = part + un * {BLK};
    DMA_FILL(0, DESC(desc, 0), 0, (int)(base + i * {HD})); DMA_FILL(1, DESC(desc, 1), {MB}, (int)(base + {MR * HG * HD} + 2 * i)); DMA_WAIT(0); DMA_WAIT(1);
    __global float* pp = LSF(0); __global float* ob = LSF({OB}); __global float* mm = LSF({MB});
    float m = -3.0e38f; for (int c = 0; c < {P}; c++) if (mm[2 * c + 1] > 0.0f && mm[2 * c] > m) m = mm[2 * c];
    float w[{P}], l = 0.0f;
    for (int c = 0; c < {P}; c++) {{ w[c] = mm[2 * c + 1] > 0.0f ? exp2_d4(BC((mm[2 * c] - m) * {LOG2E!r}f))[0] : 0.0f; l += w[c] * mm[2 * c + 1]; }}
    float8 inv = BC(1.0f / l);
    for (int e = 0; e < {HD}; e += 8) {{ float8 o = BC(0.0f); for (int c = 0; c < {P}; c++) o += BC(w[c]) * *(__global float8*)(pp + c * {HD} + e); *(__global float8*)(ob + e) = o * inv; }}
    DMA_DRAIN(2, DESC(desc, 2), {OB}, (int)(o_rows + r * {NH * HD} + h * {HD})); DMA_WAIT(2);
  }}
  DMA_WAIT_ALL();
}}"""

def gattn_part_size(NH, NKV, HD, M):
  """floats of gattn_part's record buffer."""
  HG, BT, P, MR = gattn_cfgr(NH, NKV, HD, M); return P * NKV * (NH // NKV // HG) * (M // MR) * _ga_block(MR, HG, HD)

def gattn_kvd_desc(KR, HD): return QK.V._desc_slots((KR * 4,), (HD * 4,))

def gattn_kvd_src(NKV, HD, eps, M, KR, vsame=False, ring=0):
  """gattn_kv by DMA: unit = (row t, kv head h, k or v): the row's projection arrives by DMA (the whole kv row, KR floats), is normed
  (and for k rotated, x knw) in LSRAM and leaves by DMA into the cache row (pos + t, or (pos + t) % ring). args: as gattn_kv + desc."""
  assert HD % 16 == 0
  row = (lambda t: f"(({t}) % {ring})") if ring else (lambda t: f"({t})")
  OUT = KR * 4; assert OUT + HD * 4 <= 32768 - 64
  return V.FULL_H + EXP + _norm_rope(HD, eps) + f"""
__kernel void gattn_kvd(__global float* restrict Kc, __global float* restrict Vc, __global float* restrict kv_rows, __global float* restrict knw,
                        __global float* restrict cs, __global int* restrict idx, __global int* restrict desc, const int core_id) {{
  int pos0 = idx[0];
  for (int u = core_id; u < {M * NKV * 2}; u += {NT}) {{
    int t = u / {NKV * 2}, h = (u / 2) % {NKV}, isv = u % 2; int pos = pos0 + t;
    DMA_FILL(0, DESC(desc, 0), 0, (int)(kv_rows + t * {KR})); DMA_WAIT(0);
    __global float* src = LSF(0) + (isv ? {0 if vsame else NKV * HD} : 0) + h * {HD}; __global float* dst = LSF({OUT});
    if (isv) norm_rope(dst, src, (__global float*)0, (__global float*)0); else norm_rope(dst, src, knw, cs + pos * {HD});
    DMA_DRAIN(1, DESC(desc, 1), {OUT}, (int)((isv ? Vc : Kc) + ({row("pos")} * {NKV} + h) * {HD})); DMA_WAIT(1);
  }}
  DMA_WAIT_ALL();
}}"""

# ---- numpy references (what each kernel computes, in fp32)
import numpy as np                                                       # noqa: E402

def ct_rows(ct, nrb, N, ns=3):
  """The C tiles -> fp32 rows [12 nrb, N] (the groups covering N)."""
  ng = -(-N // (16 * ns)); t = ct[:ng * nrb * ns * 192].reshape(ng, nrb, ns, 3, 4, 4, 4)
  return np.ascontiguousarray(t.transpose(1, 3, 5, 0, 2, 4, 6)).reshape(12 * nrb, 16 * ns * ng)[:, :N]

def ple_mix_ref(C, T, w, NL, P, H, eps=1e-6):
  """C [n, NL P] (the projection), T [n, NL P] (table rows x sqrt(P)) -> [NL, n, P]: gemma4_ref.Model.ple_inputs' tail."""
  y = (C * np.float32(1.0 / math.sqrt(H))).reshape(len(C), NL, P); y = y * (1.0 / np.sqrt((y * y).mean(-1, keepdims=True) + eps)) * w
  return ((y + T.reshape(len(C), NL, P)) * np.float32(1.0 / math.sqrt(2.0))).transpose(1, 0, 2)

def gelu_sig(x):
  """The kernels' GELU: x sigmoid(2 sqrt(2/pi) (x + 0.044715 x^3)) (= gelu_tanh exactly in real arithmetic)."""
  z = x + np.float32(0.044715) * x * x * x
  return x / (1.0 + np.exp(np.float32(-1.5957691216057308) * z))

def pnresid_ref(x, C, w, s, eps=1e-6):
  return (x + C * (1.0 / np.sqrt((C * C).mean(-1, keepdims=True) + eps)) * w) * np.float32(s)

def rope_table(theta, HD, TMAX, freq_factors=None):
  """cs [TMAX][HD]: cos then sin of the HD / 2 frequencies theta^(-2i / HD) (/ the factor: 1e30 -> 0, the proportional RoPE)."""
  inv = 1.0 / (theta ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
  if freq_factors is not None: inv = inv / np.asarray(freq_factors, np.float64)[:HD // 2]
  f = np.arange(TMAX, dtype=np.float64)[:, None] * inv[None]
  return np.concatenate([np.cos(f), np.sin(f)], -1).astype(np.float32)

def norm_rope_ref(x, w, cs, eps=1e-6):
  """x [..., HD]; cs [..., HD] (or None) as the table rows."""
  y = x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps))
  if w is not None: y = y * w
  if cs is None: return y
  h = y.shape[-1] // 2; c, s = cs[..., :h], cs[..., h:]; a, b = y[..., :h], y[..., h:]
  return np.concatenate([a * c - b * s, b * c + a * s], -1)

"""Hand-written TEC kernels (the backend's `register_csrc`) for the layouts around the block-scaled GEMM (ks 32: 128-wide K
slices of 32 k-quads; C tiles [group][rb][strip][3 row quads][4 col tiles][4 rows][4 cols]): the generic tinygrad kernels for the
same permutes cost ~10 ms each at decode shapes. A unit is (row block, part of the slices); direct DDR loads, no DMA.

  rms_a32:    A = fp16(x * rsqrt(mean(x^2) + eps) * w1) in the A layout (w1 = 1 + weight; `norm=False`: A = fp16(x))
  swiglu_a32: A = fp16(silu(g) * u) from the gate|up GEMM's tiles (N = 2 m_, gate columns first)
  resid32:    out = x + C (the tiles of a linear with N >= c)
"""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
from tinygrad import Tensor, dtypes                                      # noqa: E402
from zy import OA, vec_f16 as V                                          # noqa: E402
import gdn_asm                                                           # noqa: E402  (gdn_fast_src's sweeps)

NT, DEV = 12, "ZHOUYI"
TILE = "#define F8(a, b) __builtin_shufflevector((a), (b), 0, 1, 2, 3, 4, 5, 6, 7)\n"
EXP = """
static inline float8 exp2_d4(float8 t) {
  t = VMAX(t, BC(-126.0f)); t = VMIN(t, BC(126.0f));
  float8 k = VRINT(t); float8 f = t - k;
  float8 p = BC(9.618129e-3f);
  p = p * f + BC(5.550411e-2f); p = p * f + BC(2.402265e-1f); p = p * f + BC(6.931472e-1f); p = p * f + BC(1.0f);
  return VSCAL2(p, VI(k));
}
static inline float8 sw(float8 u, float8 q) { float8 e = exp2_d4(BC(-1.4426950408889634f) * u); return u * VRCP(BC(1.0f) + e) * q; }
"""
# act="gelu" (Gemma 4's gelu_pytorch_tanh gate, ../gemma4): 0.5 u (1 + tanh(z)) = u sigmoid(2 z), z = sqrt(2 / pi) (u + 0.044715 u^3),
# with swiglu's sigmoid (exp2 by EXP's polynomial, the reciprocal unit). Appended to EXP only for that act (SiLU sources unchanged).
GELU = """static inline float8 gl(float8 u, float8 q) { float8 z = u + BC(0.044715f) * u * u * u; float8 e = exp2_d4(BC(-2.302208198546288f) * z); return u * VRCP(BC(1.0f) + e) * q; }
"""
ACTS = {"silu": ("sw", "swiglu"), "gelu": ("gl", "geglu")}                  # act -> (the helper, the kernel name's stem)

def _units(nrb, nsl):
  """(parts per row block, slices per part): 12 tasks over nrb row blocks x P slice parts."""
  P = max(1, min(nsl, NT // nrb)); return P

def _real(real, nrb):
  """(row blocks, row quads per block) holding the `real` rows (the rest is padding: never read or written, so the
  persistent buffers keep their zeros there). Decode (1 row): one block, one quad."""
  real = 12 * nrb if real is None else real; rbr = -(-real // 12)
  return rbr, (min(3, -(-real // 4)) if rbr == 1 else 3)

def rms_a32_src(rows, c, nrb, eps, norm=True, real=None, compact=False, stacked=False):
  """`compact` (the GEMM's rows mode): the A layout of row block 0's used tiles only, [slice][k][iqr tiles][16 halves].
  `stacked`: w1 is a stack [L, c] and a 4th argument idx (int32, idx[0] = the layer) picks the row. Only the real rows' squares
  are summed (a padding row's inv is 0 whatever its content)."""
  assert c % 128 == 0; nsl = c // 128; rbr, iqr = _real(real, nrb); P = _units(rbr, nsl); rows = min(rows, 12 * rbr if real is None else real)
  assert not compact or rbr == 1
  obase, oel = ((f"sl * {32 * 16 * iqr}", f"(kq * {iqr} + i) * 16") if compact else (f"((sl * {nrb} + rb) * 32) * 48", "(kq * 3 + i) * 16"))
  inv_expr = f"(m < {rows}) ? 1.0f / __builtin_sqrtf(hsum8(acc) * {1.0 / c!r}f + {eps!r}f) : 0.0f" if norm else "1.0f"
  return V.FULL_H + TILE + f"""
__kernel void rms_a32(__global half* restrict out, __global float* restrict x, __global float* restrict w1{", __global int* restrict idx" if stacked else ""}, const int core_id) {{
{"  w1 += idx[0] * %d;" % c if stacked else ""}
  for (int u = core_id; u < {rbr * P}; u += {NT}) {{
    int rb = u / {P}, part = u % {P}; float inv[12];
    for (int r = 0; r < {4 * iqr}; r++) {{
      int m = rb * 12 + r; __global float* p = x + m * {c}; float8 acc = BC(0.0f);
      {"if (m < %d) for (int i = 0; i < %d; i += 8) { float8 v = *(__global float8*)(p + i); acc += v * v; }" % (rows, c) if norm else ""}
      inv[r] = {inv_expr};
    }}
    for (int sl = part; sl < {nsl}; sl += {P}) {{
      __global half* o = out + {obase};
      for (int i = 0; i < {iqr}; i++) {{
        __global float* x0 = x + (rb * 12 + 4 * i) * {c} + sl * 128;
        float8 i01 = ROWS2(inv[4 * i], inv[4 * i + 1]), i23 = ROWS2(inv[4 * i + 2], inv[4 * i + 3]);
        for (int kq = 0; kq < 32; kq++) {{
          float4 w4 = {"*(__global float4*)(w1 + sl * 128 + 4 * kq)" if norm else "(float4){1.0f, 1.0f, 1.0f, 1.0f}"}; float8 w8 = F8(w4, w4);
          float8 t01 = F8(*(__global float4*)(x0 + 4 * kq), *(__global float4*)(x0 + {c} + 4 * kq)) * i01 * w8;
          float8 t23 = F8(*(__global float4*)(x0 + 2 * {c} + 4 * kq), *(__global float4*)(x0 + 3 * {c} + 4 * kq)) * i23 * w8;
          *(__global half16*)(o + {oel}) = CVT16(t01, t23);
        }}
      }}
    }}
  }}
}}"""

RMS_CH = 256                                                                # rms_a32d: columns a phase-1 DMA chunk

def rms_a32d_desc(c, real):
  """slot 0: the real rows' next RMS_CH columns (pitch c); slot 1: one 128-column slice of the tiles' rows (4 iqr rows)."""
  iqr = min(3, -(-real // 4))
  return V._desc_slots((real * RMS_CH * 4, RMS_CH * 4, c * 4, RMS_CH * 4), (4 * iqr * 128 * 4, 128 * 4, c * 4, 128 * 4))

def rms_a32d_src(c, nrb, eps, real, norm=True, stacked=False):
  """rms_a32 in rows mode (compact A, `real` <= 12 rows) with the rows streamed through LSRAM by DMA instead of read by plain
  loads: phase 1 (norm) every task sums the squares of the whole real rows, RMS_CH columns a chunk, double-buffered (flags
  0 / 1); phase 2 each task's 128-column slices arrive by one DMA each (flag 2) and become the A layout as in rms_a32. The sums
  run in rms_a32's order: the same bits. args: out, x, w1 (or x when not `norm`), [idx], desc (rms_a32d_desc)."""
  assert c % 128 == 0 and c % RMS_CH == 0 and 1 <= real <= 12; nsl = c // 128; iqr = min(3, -(-real // 4)); NCH = c // RMS_CH
  CHB = real * RMS_CH * 4; SB = 2 * CHB; assert SB + 4 * iqr * 512 <= 32768 - 64
  inv_expr = f"(r < {real}) ? 1.0f / __builtin_sqrtf(hsum8(acc[r]) * {1.0 / c!r}f + {eps!r}f) : 0.0f" if norm else "1.0f"
  phase1 = f"""
  float8 acc[{real}]; for (int r = 0; r < {real}; r++) acc[r] = BC(0.0f);
  DMA_FILL(0, DESC(desc, 0), 0, (int)x);
  for (int ch = 0; ch < {NCH}; ch++) {{
    int p = ch & 1;
    if (ch + 1 < {NCH}) {{ if (p == 0) DMA_FILL(1, DESC(desc, 0), {CHB}, (int)(x + (ch + 1) * {RMS_CH})); else DMA_FILL(0, DESC(desc, 0), 0, (int)(x + (ch + 1) * {RMS_CH})); }}
    if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
    __global float* b = LSF(p * {CHB});
    for (int r = 0; r < {real}; r++) for (int i = 0; i < {RMS_CH}; i += 8) {{ float8 v = *(__global float8*)(b + r * {RMS_CH} + i); acc[r] += v * v; }}
  }}""" if norm else ""
  return V.FULL_H + TILE + f"""
__kernel void rms_a32d(__global half* restrict out, __global float* restrict x, __global float* restrict w1{", __global int* restrict idx" if stacked else ""}, __global int* restrict desc, const int core_id) {{
{"  w1 += idx[0] * %d;" % c if stacked else ""}
  if (core_id >= {nsl}) return;
  {phase1}
  float inv[12];
  for (int r = 0; r < {4 * iqr}; r++) inv[r] = {inv_expr};
  for (int sl = core_id; sl < {nsl}; sl += {NT}) {{
    DMA_FILL(2, DESC(desc, 1), {SB}, (int)(x + sl * 128)); DMA_WAIT(2);
    __global half* o = out + sl * {32 * 16 * iqr};
    for (int i = 0; i < {iqr}; i++) {{
      __global float* x0 = LSF({SB}) + (4 * i) * 128;
      float8 i01 = ROWS2(inv[4 * i], inv[4 * i + 1]), i23 = ROWS2(inv[4 * i + 2], inv[4 * i + 3]);
      for (int kq = 0; kq < 32; kq++) {{
        float4 w4 = {"*(__global float4*)(w1 + sl * 128 + 4 * kq)" if norm else "(float4){1.0f, 1.0f, 1.0f, 1.0f}"}; float8 w8 = F8(w4, w4);
        float8 t01 = F8(*(__global float4*)(x0 + 4 * kq), *(__global float4*)(x0 + 128 + 4 * kq)) * i01 * w8;
        float8 t23 = F8(*(__global float4*)(x0 + 256 + 4 * kq), *(__global float4*)(x0 + 384 + 4 * kq)) * i23 * w8;
        *(__global half16*)(o + (kq * {iqr} + i) * 16) = CVT16(t01, t23);
      }}
    }}
  }}
  DMA_WAIT_ALL();
}}"""

# ---- QWEN_SMALL2: the RMSNorm's row norms as a RECORD, and the A producer that reads it ----------------------------------------
# rms_a32d (norm) had every task sum the squares of every real row (12 tasks x the rows: 12-13x its floor at 3-8 rows). Here one
# launch computes each real row's 1 / rms ONCE (rms_rec: a task a row, or resid32q: the residual's own row-quad tasks) into a
# record -- float32 [12][16], row r's 1 / rms at [16 r] (one 64-byte line a row: no two tasks store into one line) -- and the A
# producer (rms_a32f) reads it. The sums run in rms_a32d's order (per row, lane by lane, K ascending; hsum8; the same expression):
# the same bits. QWEN_RMS2=0: rms_a32d as before.
NOUNROLL_H = '#define NOUNROLL _Pragma("clang loop unroll(disable)")\n'
REC_N = 12 * 16                                                             # the record's floats

def rms_rec_desc(c):
  """slot 0: half a row (c / 2 floats, contiguous); slot 1: one record line (64 B)."""
  return V._desc_slots((c * 2,), (64,))

def rms_rec_src(c, real, eps):
  """rec[16 r] = 1 / sqrt(mean(x[r]^2) + eps) for r < real (rms_a32d's phase 1 for one row: acc += v * v over float8 lanes in K
  order, hsum8, the same expression). A task a row: the row arrives by two DMAs (halves), the sum runs as each lands, the line
  leaves by DMA. args: rec (float32 [REC_N]), x (fp32 rows, pitch c), desc (rms_rec_desc)."""
  assert c % 16 == 0 and 1 <= real <= 12 and c * 4 + 64 <= 32768 - 64; H2 = c // 2
  return V.FULL_H + NOUNROLL_H + f"""
__kernel void rms_rec(__global float* restrict rec, __global float* restrict x, __global int* restrict desc, const int core_id) {{
  if (core_id < {real}) {{
    __global float* xr = x + core_id * {c};
    DMA_FILL(0, DESC(desc, 0), 0, (int)xr); DMA_FILL(1, DESC(desc, 0), {H2 * 4}, (int)(xr + {H2}));
    __global float* b = LSF(0); float8 acc = BC(0.0f);
    DMA_WAIT(0);
    NOUNROLL for (int k = 0; k < {H2}; k += 8) {{ float8 v = *(__global float8*)(b + k); acc += v * v; }}
    DMA_WAIT(1);
    NOUNROLL for (int k = {H2}; k < {c}; k += 8) {{ float8 v = *(__global float8*)(b + k); acc += v * v; }}
    __global float* o = LSF({c * 4}); o[0] = 1.0f / __builtin_sqrtf(hsum8(acc) * {1.0 / c!r}f + {eps!r}f);
    DMA_DRAIN(2, DESC(desc, 1), {c * 4}, (int)(rec + core_id * 16));
  }}
  DMA_WAIT_ALL();
}}"""

def rms_a32f_desc(c, real):
  """slot 0: one 128-column slice of the tiles' rows (4 iqr rows, pitch c); 1: the slice's 128 weights; 2: the slice's A block
  (32 iqr 32-byte pieces, contiguous); 3: the record's real lines."""
  iqr = min(3, -(-real // 4))
  return V._desc_slots((4 * iqr * 128 * 4, 128 * 4, c * 4, 128 * 4), (512,), (1024 * iqr,), (real * 64,), (real * RMS_CH * 4, RMS_CH * 4, c * 4, RMS_CH * 4))

def rms_a32f_src(c, eps, real, norm=True, stacked=False, guard=False):
  """rms_a32d's phase 2 with the row norms from a record (`norm`: rms_rec / resid32q wrote it) instead of every task's phase 1,
  and every operand by DMA: a task's slices (sl = core_id, core_id + NT, ..) double-buffered -- the slice's rows and weights in
  (flag p), its A block built in LSRAM with rms_a32d's expression ((x * inv) * w, CVT16) and out by one DMA (flag 2 + p; the
  1024 iqr bytes of a slice are one task's, whole lines). Padding rows (r >= real in the last quad) get inv 0, as in rms_a32d.
  The same bits as rms_a32d. args: out, x, w1 (or x when not `norm`), [idx], [rec (norm)], desc (rms_a32f_desc)."""
  assert c % 128 == 0 and 1 <= real <= 12; nsl = c // 128; iqr = min(3, -(-real // 4))
  XB = 4 * iqr * 512; WB = 512 if norm else 0; XS = XB + WB; OB = 1024 * iqr; X0 = 1024; O0 = X0 + 2 * XS
  assert O0 + 2 * OB <= 32768 - 64
  def fill(k, p):
    return (f"{{ int s_ = core_id + ({k}) * {NT}; DMA_FILL({p}, DESC(desc, 0), {X0 + p * XS}, (int)(x + s_ * 128));"
            + (f" DMA_FILL({p}, DESC(desc, 1), {X0 + p * XS + XB}, (int)(w1 + s_ * 128));" if norm else "") + " }")
  assert not guard or (norm and stacked)
  inv = (f"DMA_FILL(2, DESC(desc, 3), 0, (int)rec); DMA_WAIT(2);\n    __global float* R_ = LSF(0);\n"
         f"    NOUNROLL for (int r = 0; r < {4 * iqr}; r++) inv[r] = r < {real} ? R_[16 * r] : 0.0f;") if norm else \
        f"NOUNROLL for (int r = 0; r < {4 * iqr}; r++) inv[r] = 1.0f;"
  if guard:   # idx[0] == 0 (the stack's first layer: its input rows were written outside the kernels, e.g. the embedding, so the
              # record is stale): rms_a32d's phase 1 here (every task sums every real row; the same order: the same bits)
    CHB = real * RMS_CH * 4; NCH = c // RMS_CH; assert c % RMS_CH == 0 and 2 * CHB <= 32768 - 64
    inv = f"""if (idx[0] != 0) {{ {inv} }} else {{
    float8 acc[{real}]; for (int r = 0; r < {real}; r++) acc[r] = BC(0.0f);
    DMA_FILL(0, DESC(desc, 4), 0, (int)x);
    NOUNROLL for (int ch = 0; ch < {NCH}; ch++) {{
      int p = ch & 1;
      if (ch + 1 < {NCH}) {{ if (p == 0) DMA_FILL(1, DESC(desc, 4), {CHB}, (int)(x + (ch + 1) * {RMS_CH})); else DMA_FILL(0, DESC(desc, 4), 0, (int)(x + (ch + 1) * {RMS_CH})); }}
      if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
      __global float* b = LSF(p * {CHB});
      NOUNROLL for (int r = 0; r < {real}; r++) NOUNROLL for (int i = 0; i < {RMS_CH}; i += 8) {{ float8 v = *(__global float8*)(b + r * {RMS_CH} + i); acc[r] += v * v; }}
    }}
    NOUNROLL for (int r = 0; r < {4 * iqr}; r++) inv[r] = (r < {real}) ? 1.0f / __builtin_sqrtf(hsum8(acc[r]) * {1.0 / c!r}f + {eps!r}f) : 0.0f;
    }}"""
  return V.FULL_H + TILE + NOUNROLL_H + f"""
__kernel void rms_a32f(__global half* restrict out, __global float* restrict x, __global float* restrict w1{", __global int* restrict idx" if stacked else ""}{", __global float* restrict rec" if norm else ""}, __global int* restrict desc, const int core_id) {{
{"  w1 += idx[0] * %d;" % c if stacked else ""}
  if (core_id < {nsl}) {{                                  /* one exit for every task (an early `return` beside DMA: had_a32d_src) */
    int nk = ({nsl} - core_id + {NT - 1}) / {NT};
    float inv[12];
    {inv}
    {fill(0, 0)}
    NOUNROLL for (int k = 0; k < nk; k++) {{
      int p = k & 1;
      if (k + 1 < nk) {{ if (p == 0) {fill("k + 1", 1)} else {fill("k + 1", 0)} }}
      if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
      if (k >= 2) {{ if (p == 0) DMA_WAIT(2); else DMA_WAIT(3); }}      /* this buffer's previous A block gone */
      __global float* X = LSF({X0}) + p * {XS // 4}; __global half* O = LSH({O0}) + p * {OB // 2};
      NOUNROLL for (int i = 0; i < {iqr}; i++) {{
        __global float* x0 = X + (4 * i) * 128;
        float8 i01 = ROWS2(inv[4 * i], inv[4 * i + 1]), i23 = ROWS2(inv[4 * i + 2], inv[4 * i + 3]);
        NOUNROLL for (int kq = 0; kq < 32; kq++) {{
          float4 w4 = {f"*(__global float4*)(X + {XB // 4} + 4 * kq)" if norm else "(float4){1.0f, 1.0f, 1.0f, 1.0f}"}; float8 w8 = F8(w4, w4);
          float8 t01 = F8(*(__global float4*)(x0 + 4 * kq), *(__global float4*)(x0 + 128 + 4 * kq)) * i01 * w8;
          float8 t23 = F8(*(__global float4*)(x0 + 256 + 4 * kq), *(__global float4*)(x0 + 384 + 4 * kq)) * i23 * w8;
          *(__global half16*)(O + (kq * {iqr} + i) * 16) = CVT16(t01, t23);
        }}
      }}
      int sl = core_id + k * {NT};
      if (p == 0) DMA_DRAIN(2, DESC(desc, 2), {O0}, (int)(out + sl * {512 * iqr})); else DMA_DRAIN(3, DESC(desc, 2), {O0 + OB}, (int)(out + sl * {512 * iqr}));
    }}
  }}
  DMA_WAIT_ALL();
}}"""

def swiglu_a32_src(m_, nrb, real=None, compact=False, act="silu"):
  """gate | up tiles (N = 2 m_, groups of 48 columns, `nrb` row blocks) -> the A layout of K = m_ (`compact`: as rms_a32).
  `act`: "silu" (SwiGLU) or "gelu" (GeGLU with the tanh GELU: kernel geglu_a32)."""
  assert m_ % 128 == 0; nsl = m_ // 128; rbr, iqr = _real(real, nrb); P = _units(rbr, nsl); fn, stem = ACTS[act]
  assert not compact or rbr == 1
  obase, oel = ((f"sl * {32 * 16 * iqr}", f"(kq * {iqr} + i) * 16") if compact else (f"((sl * {nrb} + rb) * 32) * 48", "(kq * 3 + i) * 16"))
  def tile(col):   # the 16 floats of (row quad i, the 4 columns col..col+3) for row block rb: [4 rows][4 cols]
    return f"ct + ((({col}) / 48 * {nrb} + rb) * 3 + (({col}) % 48) / 16) * 192 + (i * 4 + (({col}) % 16) / 4) * 16"
  return V.FULL_H + TILE + EXP + (GELU if act == "gelu" else "") + f"""
__kernel void {stem}_a32(__global half* restrict out, __global float* restrict ct, const int core_id) {{
  for (int u = core_id; u < {rbr * P}; u += {NT}) {{
    int rb = u / {P}, part = u % {P};
    for (int sl = part; sl < {nsl}; sl += {P}) {{
      __global half* o = out + {obase};
      for (int i = 0; i < {iqr}; i++) for (int kq = 0; kq < 32; kq++) {{
        int col = sl * 128 + 4 * kq;
        __global float* g = {tile("col")}; __global float* p = {tile("col + %d" % m_)};
        float8 t01 = {fn}(*(__global float8*)(g), *(__global float8*)(p)), t23 = {fn}(*(__global float8*)(g + 8), *(__global float8*)(p + 8));
        *(__global half16*)(o + {oel}) = CVT16(t01, t23);
      }}
    }}
  }}
}}"""

def swiglu_a32d_desc(nrb, real):
  """slot 0: a chunk's gate (or up) pieces -- the iqr row quads (256 iqr bytes) of 3 nrb + 3 consecutive C sub-tiles from a group's
  first (pitch 768: the other row blocks' ride along; the last group's stop at row block 0); slot 1: a chunk's A (512 iqr bytes)."""
  iqr = min(3, -(-real // 4)); PB = 256 * iqr
  return V._desc_slots(((3 * nrb + 3) * PB, PB, 768, PB), (512 * iqr,))

def swiglu_a32d_src(m_, nrb, real, act="silu"):
  """swiglu_a32 in rows mode (compact A, `real` <= 12 rows) by DMA: a unit is a chunk of 64 columns (4 C sub-tiles; tasks take
  chunks core_id, core_id + NT, ...): its gate and up pieces (row quads 0..iqr-1 of 3 nrb + 3 sub-tiles from the chunk's first
  group) arrive in LSRAM double-buffered (flags 0 / 2, 1 / 3), the A of its 16 k-quads (all iqr quads: 512 iqr contiguous bytes,
  whole 64-byte lines, so one task writes every byte of them) is built there with swiglu_a32's expression and leaves by one DMA
  (unit k's on flag 2 + parity of k + 1, issued once unit k + 1's inputs are in; waited before that flag's next fill). The same
  operations on the same values: the same bits. args: out, ct, desc (swiglu_a32d_desc). `act`: as swiglu_a32_src (geglu_a32d)."""
  assert m_ % 128 == 0 and 1 <= real <= 12; iqr = min(3, -(-real // 4)); nch = m_ // 64; assert nch >= NT; fn, stem = ACTS[act]
  NPR = 3 * nrb; PB = 256 * iqr; GB = (NPR + 3) * PB; OB = 512 * iqr; OUT0 = 4 * GB; assert OUT0 + 2 * OB <= 32768 - 64
  def fill(k, p):
    return (f"{{ int c_ = core_id + ({k}) * {NT}; int j0_ = 4 * c_; int ju_ = j0_ + {m_ // 16};\n"
            f"        DMA_FILL({p}, DESC(desc, 0), {p * 2 * GB}, (int)(ct + (j0_ / 3) * {NPR * 192}));\n"
            f"        DMA_FILL({p + 2}, DESC(desc, 0), {p * 2 * GB + GB}, (int)(ct + (ju_ / 3) * {NPR * 192})); }}")
  def drain(flag, buf, k): return f"DMA_DRAIN({flag}, DESC(desc, 1), {OUT0 + buf * OB}, (int)(out + (core_id + ({k}) * {NT}) * {256 * iqr}));"
  return V.FULL_H + TILE + EXP + (GELU if act == "gelu" else "") + f"""
__kernel void {stem}_a32d(__global half* restrict out, __global float* restrict ct, __global int* restrict desc, const int core_id) {{
  if (core_id < {nch}) {{                                  /* one exit for every task (an early `return` beside DMA: had_a32d_src) */
  int nk = ({nch} - core_id + {NT - 1}) / {NT};
  {fill(0, 0)}
  for (int k = 0; k < nk; k++) {{
    int p = k & 1;
    if (k >= 2) {{ if (p == 0) DMA_WAIT(3); else DMA_WAIT(2); }}     /* unit k - 2's A gone: its flag and its LSRAM (this unit's) free */
    if (k + 1 < nk) {{ if (p == 0) {fill("k + 1", 1)} else {fill("k + 1", 0)} }}
    if (p == 0) {{ DMA_WAIT(0); DMA_WAIT(2); }} else {{ DMA_WAIT(1); DMA_WAIT(3); }}
    if (k >= 1) {{ if (p == 0) {drain(2, 1, "k - 1")} else {drain(3, 0, "k - 1")} }}
    __global float* G = LSF(p * {2 * GB}); __global float* U = G + {GB // 4}; __global half* O = LSH({OUT0} + p * {OB});
    int j0 = 4 * (core_id + k * {NT}), ju0 = j0 + {m_ // 16};
    for (int jj = 0; jj < 4; jj++) {{
      int j = j0 + jj, ju = ju0 + jj;
      __global float* g = G + ((j / 3 - j0 / 3) * {NPR} + j % 3) * {64 * iqr}; __global float* q = U + ((ju / 3 - ju0 / 3) * {NPR} + ju % 3) * {64 * iqr};
      for (int i = 0; i < {iqr}; i++) for (int cq = 0; cq < 4; cq++) {{
        __global float* gg = g + i * 64 + cq * 16; __global float* qq = q + i * 64 + cq * 16;
        float8 t01 = {fn}(*(__global float8*)(gg), *(__global float8*)(qq)), t23 = {fn}(*(__global float8*)(gg + 8), *(__global float8*)(qq + 8));
        *(__global half16*)(O + ((4 * jj + cq) * {iqr} + i) * 16) = CVT16(t01, t23);
      }}
    }}
  }}
  if ((nk - 1) & 1) {{ {drain(1, 1, "nk - 1")} }} else {{ {drain(0, 0, "nk - 1")} }}
  }}
  DMA_WAIT_ALL();
}}"""

HAD_B = 1024                                                                # had_a32: the Walsh-Hadamard block (along K)

def had_a32_src(mode, c, nrb, eps, real=None, compact=False, stacked=False, post=1.0):
  """The A producers of a model whose linears take their input rotated (QWEN_MODEL=bonsai2-27b: W' = W diag(s) Hb, fed Hb (s * x)):
  the row values as rms_a32 / swiglu_a32 make them, then the unnormalised Walsh-Hadamard transform over each HAD_B-wide block of
  K (natural Sylvester order: y[r] = sum_c (-1)^popcount(r & c) x[c]), times `post`, in the A layout (`compact` as rms_a32).
    mode "rms":    v = x * rsqrt(mean(x^2) + eps) * w          (w = (1 + weight) * s: the signs folded into the norm weight)
    mode "plain":  v = x * w                                   (w = s, the signs of K = c)
    mode "swiglu": v = silu(g) * u from the gate | up tiles     (N = 2 c; the signs folded into the up rows' scales)
  The normalisation 1 / sqrt(HAD_B) is `post` or the GEMM's scale table's (1.0 here). A unit = (row block, block of K); for each of
  the row block's quads in turn the quad's rows of the block go into LSRAM ([4 rows][HAD_B] fp32, 16 KiB), the 10 butterfly stages
  run there (the 3 in-vector stages by lane shuffles, then radix-4 passes), and the block's 8 K-slices are written. Rows past `real` are written as zeros.
  args: out (half), x (float rows, pitch c; "swiglu": the C tiles), w (float [c]; stacked: [L, c]) [, idx], core_id."""
  assert mode in ("rms", "plain", "swiglu") and c % HAD_B == 0; nb = c // HAD_B; rbr, iqr = _real(real, nrb)
  rows = 12 * rbr if real is None else real; assert not compact or rbr == 1
  # a unit = (row block, block of K) and runs the block's row quads in turn (iqr compact, else 3): the A layout interleaves the
  # quads' 32-byte pieces inside 64-byte lines, and two TECs storing into one line lose one's bytes on the board (resid32_src's
  # rule; the simulator has no caches, so a (quad, block) unit passed there and failed on silicon at 6+ rows)
  nub, qpu = (1, iqr) if compact else (rbr, 3)
  obase, oel = ((f"sl * {32 * 16 * iqr}", f"(kq * {iqr} + i) * 16") if compact else (f"((sl * {nrb} + rb) * 32) * 48", "(kq * 3 + i) * 16"))
  def tile(col):   # swiglu: the 16 floats of (row quad i of row block rb, the 4 columns col..col+3): [4 rows][4 cols]
    return f"x + ((({col}) / 48 * {nrb} + rb) * 3 + (({col}) % 48) / 16) * 192 + (i * 4 + (({col}) % 16) / 4) * 16"
  if mode == "swiglu":
    load = f"""
    for (int k = 0; k < {HAD_B}; k += 4) {{
      int col = b * {HAD_B} + k; __global float* g = {tile("col")}; __global float* p = {tile("col + %d" % c)};
      float8 t01 = sw(*(__global float8*)(g), *(__global float8*)(p)), t23 = sw(*(__global float8*)(g + 8), *(__global float8*)(p + 8));
      *(__global float4*)(L + k) = __builtin_shufflevector(t01, t01, 0, 1, 2, 3); *(__global float4*)(L + {HAD_B} + k) = __builtin_shufflevector(t01, t01, 4, 5, 6, 7);
      *(__global float4*)(L + {2 * HAD_B} + k) = __builtin_shufflevector(t23, t23, 0, 1, 2, 3); *(__global float4*)(L + {3 * HAD_B} + k) = __builtin_shufflevector(t23, t23, 4, 5, 6, 7);
    }}
    for (int r = nr; r < 4; r++) for (int k = 0; k < {HAD_B}; k += 8) *(__global float8*)(L + r * {HAD_B} + k) = BC(0.0f);"""
  else:
    inv = (f"""float8 acc = BC(0.0f); for (int k = 0; k < {c}; k += 8) {{ float8 v = *(__global float8*)(xr + k); acc += v * v; }}
        float inv = 1.0f / __builtin_sqrtf(hsum8(acc) * {1.0 / c!r}f + {eps!r}f);""" if mode == "rms" else "float inv = 1.0f;")
    load = f"""
    for (int r = 0; r < 4; r++) {{
      __global float* Lr = L + r * {HAD_B};
      if (r < nr) {{
        __global float* xr = x + (4 * q + r) * {c};
        {inv}
        for (int k = 0; k < {HAD_B}; k += 8) *(__global float8*)(Lr + k) = *(__global float8*)(xr + b * {HAD_B} + k) * BC(inv) * *(__global float8*)(w + b * {HAD_B} + k);
      }} else for (int k = 0; k < {HAD_B}; k += 8) *(__global float8*)(Lr + k) = BC(0.0f);
    }}"""
  scale = "" if post == 1.0 else f" * BC({post!r}f)"
  return V.FULL_H + TILE + (EXP if mode == "swiglu" else "") + f"""
static inline __attribute__((always_inline)) float8 h8(float8 v) {{               /* the butterfly stages 1, 2, 4 inside a float8 */
  float8 a = __builtin_shufflevector(v, v, 0, 0, 2, 2, 4, 4, 6, 6), b = __builtin_shufflevector(v, v, 1, 1, 3, 3, 5, 5, 7, 7);
  v = a + b * ((float8){{1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f}});
  a = __builtin_shufflevector(v, v, 0, 1, 0, 1, 4, 5, 4, 5); b = __builtin_shufflevector(v, v, 2, 3, 2, 3, 6, 7, 6, 7);
  v = a + b * ((float8){{1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f}});
  a = __builtin_shufflevector(v, v, 0, 1, 2, 3, 0, 1, 2, 3); b = __builtin_shufflevector(v, v, 4, 5, 6, 7, 4, 5, 6, 7);
  return a + b * ((float8){{1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f}});
}}
__kernel void had_a32(__global half* restrict out, __global float* restrict x{"" if mode == "swiglu" else ", __global float* restrict w"}{", __global int* restrict idx" if stacked else ""}, const int core_id) {{
{"  w += idx[0] * %d;" % c if stacked else ""}
  __global float* L = LSF(0);
  for (int u = core_id; u < {nub * nb}; u += {NT}) {{
    int rb = u / {nb}, b = u % {nb};
    for (int i = 0; i < {qpu}; i++) {{                 /* the unit's row quads in turn: one TEC writes all of a block's A lines */
    int q = rb * 3 + i; int nr = {rows} - 4 * q; nr = nr < 0 ? 0 : nr > 4 ? 4 : nr;
    {load}
    for (int r = 0; r < nr; r++) {{
      __global float* Lr = L + r * {HAD_B};
      for (int k = 0; k < {HAD_B}; k += 8) *(__global float8*)(Lr + k) = h8(*(__global float8*)(Lr + k));
      for (int h = 8; h < {HAD_B // 2}; h *= 4) for (int j0 = 0; j0 < {HAD_B}; j0 += 4 * h) for (int k = j0; k < j0 + h; k += 8) {{   /* stages h, 2h */
        float8 a = *(__global float8*)(Lr + k), bb = *(__global float8*)(Lr + k + h), cc = *(__global float8*)(Lr + k + 2 * h), d = *(__global float8*)(Lr + k + 3 * h);
        float8 s0 = a + bb, s1 = a - bb, s2 = cc + d, s3 = cc - d;
        *(__global float8*)(Lr + k) = s0 + s2; *(__global float8*)(Lr + k + h) = s1 + s3; *(__global float8*)(Lr + k + 2 * h) = s0 - s2; *(__global float8*)(Lr + k + 3 * h) = s1 - s3;
      }}
      for (int k = 0; k < {HAD_B // 2}; k += 8) {{                                     /* stage HAD_B / 2 */
        float8 a = *(__global float8*)(Lr + k), bb = *(__global float8*)(Lr + k + {HAD_B // 2});
        *(__global float8*)(Lr + k) = a + bb; *(__global float8*)(Lr + k + {HAD_B // 2}) = a - bb;
      }}
    }}
    for (int s = 0; s < {HAD_B // 128}; s++) {{
      int sl = b * {HAD_B // 128} + s; __global half* o = out + {obase};
      for (int kq = 0; kq < 32; kq++) {{
        int k = s * 128 + 4 * kq;
        float8 t01 = F8(*(__global float4*)(L + k), *(__global float4*)(L + {HAD_B} + k)){scale};
        float8 t23 = F8(*(__global float4*)(L + {2 * HAD_B} + k), *(__global float4*)(L + {3 * HAD_B} + k)){scale};
        *(__global half16*)(o + {oel}) = CVT16(t01, t23);
      }}
    }}
    }}
  }}
}}"""

# had_a32d's butterfly stages 1, 2, 4 on two float8 (16 values) by even / odd lane extracts instead of h8's shuffles (which the
# compiler lowers lane by lane: 96 bundles a float8 on the model). Each stage is E = even lanes, O = odd lanes of the pair, then
# E + O, E - O: the same sums as h8 (v[k] + v[k ^ h] for bit h clear, v[k ^ h] subtracted for bit h set), the order permuted.
H8X2 = r"""
#define EXTE(a, b) __builtin_aipu_exte_tfp32_tfp32((a), (b))
#define EXTO(a, b) __builtin_aipu_exto_tfp32_tfp32((a), (b))
#define EXTL(a, b) __builtin_aipu_extl_tfp32_tfp32((a), (b))
#define EXTH(a, b) __builtin_aipu_exth_tfp32_tfp32((a), (b))
static inline __attribute__((always_inline)) void h8x2(__global float8* p) {
  float8 v = p[0], w = p[1];
  float8 e = EXTE(v, w), o = EXTO(v, w); float8 s = e + o, d = e - o;      /* s: k = 0,2,4,6 of v, w; d: k = 1,3,5,7 */
  e = EXTE(s, d); o = EXTO(s, d); s = e + o; d = e - o;                    /* s: k = 0,4 | 1,5 (v, w); d: k = 2,6 | 3,7 */
  e = EXTE(s, d); o = EXTO(s, d); s = e + o; d = e - o;                    /* s: k = 0,1,2,3 (v, w interleaved); d: k = 4..7 */
  p[0] = EXTE(s, d); p[1] = EXTO(s, d);
}
"""

def had_a32d_desc(mode, c, nrb, real):
  """slot 0: the real rows' next RMS_CH columns (pitch c); 1 / 2: a full / the last row quad's rows of one HAD_B block (pitch c);
  3: the block's HAD_B weights; 4: the block's A pieces of one row quad (32-byte pieces at pitch 32 iqr: contiguous when iqr = 1);
  5 ("swiglu"): one row quad's 256 bytes of 3 nrb + 3 consecutive C sub-tiles (pitch 768: two groups' row block 0)."""
  iqr = min(3, -(-real // 4)); nl = real - 4 * (iqr - 1); RB = HAD_B * 4
  return V._desc_slots((real * RMS_CH * 4, RMS_CH * 4, c * 4, RMS_CH * 4), (4 * RB, RB, c * 4, RB), (nl * RB, RB, c * 4, RB), (RB,),
                       (8192,) if iqr == 1 else (8192, 32, 32 * iqr, 32), ((3 * nrb + 3) * 256, 256, 768, 256))

def had_a32d_src(mode, c, nrb, eps, real, stacked=False, post=1.0):
  """had_a32 in rows mode (compact A, `real` <= 12 rows) with every operand moved by DMA instead of cached loads / stores: the
  rows ("rms": phase 1 sums the whole real rows' squares as rms_a32d does, the same order; then each row quad's HAD_B block, its
  weights) or the quad's 256-byte pieces of the gate | up C sub-tiles ("swiglu", 64 columns a chunk, double-buffered) arrive in
  LSRAM, the transform runs there as in had_a32, and the block's A pieces of the quad are built in LSRAM and leave by one DMA.
  The same operations in the same order as had_a32: the same bits. A unit is still one block of K (all its quads), so one task
  writes every byte of the block's A lines. args: out, x, [w (stacked: [L, c]), [idx]], desc (had_a32d_desc), core_id."""
  assert mode in ("rms", "plain", "swiglu") and c % HAD_B == 0 and 1 <= real <= 12; nb = c // HAD_B; iqr = min(3, -(-real // 4))
  RB = HAD_B * 4; W0, OUT0, IN0 = 4 * RB, 5 * RB, 4 * RB; SWB = (3 * nrb + 3) * 256; CHB = real * RMS_CH * 4
  assert 2 * CHB <= 32768 - 64 and OUT0 + 8192 <= 32768 - 64 and IN0 + 4 * SWB <= 32768 - 64 and c % RMS_CH == 0
  if mode == "swiglu":   # chunk ch: sub-tile columns j0..j0+3 of the gate (j = col / 16), the up's at j0 + c / 16; flags: gate p, up p + 2
    def issue(ch, p):
      return (f"{{ int j0 = 64 * b + 4 * ({ch}); int ju = j0 + {c // 16};\n"
              f"        DMA_FILL({p}, DESC(desc, 5), {IN0} + {2 * SWB} * {p}, (int)(x + (j0 / 3) * {3 * nrb * 192} + i * 64));\n"
              f"        DMA_FILL({p} + 2, DESC(desc, 5), {IN0} + {2 * SWB} * {p} + {SWB}, (int)(x + (ju / 3) * {3 * nrb * 192} + i * 64)); }}")
    load = f"""
    {issue(0, 0)}
    for (int ch = 0; ch < {HAD_B // 64}; ch++) {{
      int p = ch & 1;
      if (ch + 1 < {HAD_B // 64}) {{ if (p == 0) {issue("ch + 1", 1)} else {issue("ch + 1", 0)} }}
      if (p == 0) {{ DMA_WAIT(0); DMA_WAIT(2); }} else {{ DMA_WAIT(1); DMA_WAIT(3); }}
      __global float* G = LSF({IN0} + p * {2 * SWB}); __global float* U = G + {SWB // 4};
      int j0 = 64 * b + 4 * ch, ju0 = j0 + {c // 16};
      for (int jj = 0; jj < 4; jj++) for (int cq = 0; cq < 4; cq += 2) {{   /* column quads cq, cq + 1: rows' 8 columns by half extracts */
        int j = j0 + jj, ju = ju0 + jj; int k = ch * 64 + jj * 16 + cq * 4;
        __global float* g = G + ((j / 3 - j0 / 3) * {3 * nrb} + j % 3) * 64 + cq * 16; __global float* q = U + ((ju / 3 - ju0 / 3) * {3 * nrb} + ju % 3) * 64 + cq * 16;
        float8 a01 = sw(*(__global float8*)(g), *(__global float8*)(q)), a23 = sw(*(__global float8*)(g + 8), *(__global float8*)(q + 8));
        float8 b01 = sw(*(__global float8*)(g + 16), *(__global float8*)(q + 16)), b23 = sw(*(__global float8*)(g + 24), *(__global float8*)(q + 24));
        *(__global float8*)(L + k) = EXTL(a01, b01); *(__global float8*)(L + {HAD_B} + k) = EXTH(a01, b01);
        *(__global float8*)(L + {2 * HAD_B} + k) = EXTL(a23, b23); *(__global float8*)(L + {3 * HAD_B} + k) = EXTH(a23, b23);
      }}
    }}
    for (int r = nr; r < 4; r++) for (int k = 0; k < {HAD_B}; k += 8) *(__global float8*)(L + r * {HAD_B} + k) = BC(0.0f);"""
    phase1 = ""
  else:
    phase1 = f"""
  float8 acc[{real}]; for (int r = 0; r < {real}; r++) acc[r] = BC(0.0f);
  DMA_FILL(0, DESC(desc, 0), 0, (int)x);
  for (int ch = 0; ch < {c // RMS_CH}; ch++) {{
    int p = ch & 1;
    if (ch + 1 < {c // RMS_CH}) {{ if (p == 0) DMA_FILL(1, DESC(desc, 0), {CHB}, (int)(x + (ch + 1) * {RMS_CH})); else DMA_FILL(0, DESC(desc, 0), 0, (int)(x + (ch + 1) * {RMS_CH})); }}
    if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
    __global float* bb = LSF(p * {CHB});
    for (int r = 0; r < {real}; r++) for (int k = 0; k < {RMS_CH}; k += 8) {{ float8 v = *(__global float8*)(bb + r * {RMS_CH} + k); acc[r] += v * v; }}
  }}
  float inv[{real}];
  for (int r = 0; r < {real}; r++) inv[r] = 1.0f / __builtin_sqrtf(hsum8(acc[r]) * {1.0 / c!r}f + {eps!r}f);""" if mode == "rms" else ""
    iv = "inv[4 * i + r]" if mode == "rms" else "1.0f"
    load = f"""
    if (i == 0) DMA_FILL(0, DESC(desc, 3), {W0}, (int)(w + b * {HAD_B}));
    if (nr == 4) DMA_FILL(2, DESC(desc, 1), 0, (int)(x + 4 * i * {c} + b * {HAD_B})); else DMA_FILL(2, DESC(desc, 2), 0, (int)(x + 4 * i * {c} + b * {HAD_B}));
    if (i == 0) DMA_WAIT(0);
    DMA_WAIT(2);
    __global float* W = LSF({W0});
    for (int r = 0; r < 4; r++) {{
      __global float* Lr = L + r * {HAD_B};
      if (r < nr) {{
        float iv = {iv};
        for (int k = 0; k < {HAD_B}; k += 8) *(__global float8*)(Lr + k) = *(__global float8*)(Lr + k) * BC(iv) * *(__global float8*)(W + k);
      }} else for (int k = 0; k < {HAD_B}; k += 8) *(__global float8*)(Lr + k) = BC(0.0f);
    }}"""
  scale = "" if post == 1.0 else f" * BC({post!r}f)"
  src = had_a32_src(mode, c, nrb, eps, real=real, compact=True, stacked=stacked, post=post)
  pre = src[:src.index("__kernel void had_a32(")] + H8X2                                        # the headers and h8, + h8x2
  return pre + f"""__kernel void had_a32d(__global half* restrict out, __global float* restrict x{"" if mode == "swiglu" else ", __global float* restrict w"}{", __global int* restrict idx" if stacked else ""}, __global int* restrict desc, const int core_id) {{
{"  w += idx[0] * %d;" % c if stacked else ""}
  if (core_id < {nb}) {{                                  /* one exit for every task: an early `return` beside DMA hung the simulator's job */
  {phase1}
  __global float* L = LSF(0); __global half* O = LSH({OUT0}); int pend = 0;
  for (int b = core_id; b < {nb}; b += {NT}) {{
    for (int i = 0; i < {iqr}; i++) {{                 /* the unit's row quads in turn: one TEC writes all of a block's A lines */
    int nr = {real} - 4 * i; nr = nr > 4 ? 4 : nr;
    if (pend) {{ DMA_WAIT(3); pend = 0; }}                 /* the last quad's A pieces gone (their LSRAM, and swiglu's inputs, alias) */
    {load}
    for (int r = 0; r < nr; r++) {{
      __global float* Lr = L + r * {HAD_B};
      for (int k = 0; k < {HAD_B}; k += 16) h8x2((__global float8*)(Lr + k));
      for (int h = 8; h < {HAD_B // 2}; h *= 4) for (int j0 = 0; j0 < {HAD_B}; j0 += 4 * h) for (int k = j0; k < j0 + h; k += 8) {{   /* stages h, 2h */
        float8 a = *(__global float8*)(Lr + k), bb = *(__global float8*)(Lr + k + h), cc = *(__global float8*)(Lr + k + 2 * h), d = *(__global float8*)(Lr + k + 3 * h);
        float8 s0 = a + bb, s1 = a - bb, s2 = cc + d, s3 = cc - d;
        *(__global float8*)(Lr + k) = s0 + s2; *(__global float8*)(Lr + k + h) = s1 + s3; *(__global float8*)(Lr + k + 2 * h) = s0 - s2; *(__global float8*)(Lr + k + 3 * h) = s1 - s3;
      }}
      for (int k = 0; k < {HAD_B // 2}; k += 8) {{                                     /* stage HAD_B / 2 */
        float8 a = *(__global float8*)(Lr + k), bb = *(__global float8*)(Lr + k + {HAD_B // 2});
        *(__global float8*)(Lr + k) = a + bb; *(__global float8*)(Lr + k + {HAD_B // 2}) = a - bb;
      }}
    }}
    for (int pc = 0; pc < {HAD_B // 4}; pc += 2) {{       /* pieces pc, pc + 1 = (slice pc / 32, kq pc % 32): columns 4 pc .. 4 pc + 7 */
      int k = 4 * pc;
      float8 r0 = *(__global float8*)(L + k), r1 = *(__global float8*)(L + {HAD_B} + k), r2 = *(__global float8*)(L + {2 * HAD_B} + k), r3 = *(__global float8*)(L + {3 * HAD_B} + k);
      *(__global half16*)(O + pc * 16) = CVT16((EXTL(r0, r1){scale}), (EXTL(r2, r3){scale}));
      *(__global half16*)(O + pc * 16 + 16) = CVT16((EXTH(r0, r1){scale}), (EXTH(r2, r3){scale}));
    }}
    DMA_DRAIN(3, DESC(desc, 4), {OUT0}, (int)(out + b * {HAD_B // 128 * 512 * iqr} + i * 16)); pend = 1;
    }}
  }}
  }}
  DMA_WAIT_ALL();
}}"""

def had_ref(mode, x, w=None, eps=1e-6, post=1.0, real=None):
  """numpy model of had_a32 before the fp16 rounding: fp32 rows [rows, c] (mode "swiglu": x = (g, u), [rows, c] each) -> the
  transformed rows. The transform is the kernel's fp32 operations in its order (exact for "plain"); the RMS sum and the SiLU
  follow the kernel's order but not its sqrt / exp2 approximations (a tolerance)."""
  f = np.float32
  if mode == "swiglu":
    g, u = (np.asarray(t, f) for t in x); v = (g / (f(1.0) + np.exp(-g)) * u).astype(f)
  else:
    x = np.asarray(x, f); v = (x * np.asarray(w, f)[None]).astype(f)
    if mode == "rms":
      acc = np.zeros((x.shape[0], 8), f)
      for k in range(0, x.shape[1], 8): acc = (acc + x[:, k:k + 8] * x[:, k:k + 8]).astype(f)   # float8 lanes, in K order
      hs = acc[:, 0]
      for j in range(1, 8): hs = (hs + acc[:, j]).astype(f)                                      # hsum8
      inv = (f(1.0) / np.sqrt(hs * f(1.0 / x.shape[1]) + f(eps))).astype(f)
      v = ((x * inv[:, None]).astype(f) * np.asarray(w, f)[None]).astype(f)
  n = v.shape[0]; t = v.astype(f).copy(); h = 1
  while h < HAD_B:                                                          # stage h: pairs (k, k + h), bit h of k clear
    t4 = t.reshape(n, -1, 2, h); a, b = t4[:, :, 0].copy(), t4[:, :, 1].copy(); t4[:, :, 0] = a + b; t4[:, :, 1] = a - b; h *= 2
  if real is not None: t[real:] = 0
  return (t * f(post)).astype(f)

def resid32_src(c, nrb, real=None):
  """out = x + C for c columns (C's groups of 48 cover >= c columns), fp32 rows. A unit = (row block, a contiguous range of
  column quads aligned to 64-byte lines): tasks on different cores must never store into the same line."""
  assert c % 4 == 0; nq = c // 4; rbr, iqr = _real(real, nrb); P = max(1, min(nq // 4, NT // rbr)); chunk = -(-nq // P); chunk += (-chunk) % 4
  return V.FULL_H + TILE + f"""
__kernel void resid32(__global float* restrict out, __global float* restrict x, __global float* restrict ct, const int core_id) {{
  for (int u = core_id; u < {rbr * P}; u += {NT}) {{
    int rb = u / {P}, part = u % {P}; int q1 = (part + 1) * {chunk} < {nq} ? (part + 1) * {chunk} : {nq};
    for (int i = 0; i < {iqr}; i++) for (int q = part * {chunk}; q < q1; q++) {{
      int col = 4 * q; __global float* t = ct + ((col / 48 * {nrb} + rb) * 3 + (col % 48) / 16) * 192 + (i * 4 + (col % 16) / 4) * 16;
      __global float* xr = x + (rb * 12 + 4 * i) * {c} + col; __global float* orow = out + (rb * 12 + 4 * i) * {c} + col;
      for (int r = 0; r < 4; r++) *(__global float4*)(orow + r * {c}) = *(__global float4*)(xr + r * {c}) + *(__global float4*)(t + 4 * r);
    }}
  }}
}}"""

RD_G = 3                                                                    # rows32d / resid32d: C groups (48 columns) a unit

def rows32d_desc(c, nrb, real):
  """slot 0 / 1: a unit's C groups (row block 0's 2304-byte slabs, pitch nrb slabs) full / the last unit's; 2 / 3: the real rows
  of a unit's columns (pitch c), full / the last unit's."""
  ngr = -(-c // 48); nu = -(-ngr // RD_G); gt = ngr - (nu - 1) * RD_G; W = 48 * RD_G; tail = c - (nu - 1) * W
  return V._desc_slots((RD_G * 2304, 2304, 2304 * nrb, 2304), (gt * 2304, 2304, 2304 * nrb, 2304),
                       (real * W * 4, W * 4, c * 4, W * 4), (real * tail * 4, tail * 4, c * 4, tail * 4))

def rows32d_src(c, nrb, real, resid, goff=0):
  """rows32 (resid: resid32, out = x + C) in rows mode (row block 0, `real` <= 12 rows) by DMA: a unit = RD_G groups of C (their
  row block 0 slabs) and the real rows of their columns arrive in LSRAM, the rows are built there and leave by one 2D DMA (the
  units' column ranges are 576-byte aligned: no two tasks share a line). The same additions as resid32: the same bits.
  args: out, [x,] ct, desc (rows32d_desc)."""
  assert c % 4 == 0 and 1 <= real <= 12; ngr = -(-c // 48); nu = -(-ngr // RD_G); W = 48 * RD_G; tail = c - (nu - 1) * W
  CB = RD_G * 2304; XO = CB; assert XO + real * W * 4 <= 32768 - 64
  calc = ("X[r * w + k] + " if resid else "") + "*(__global float4*)(C + (k / 48) * 576 + ((k % 48) / 16) * 192 + (r / 4) * 64 + ((k % 16) / 4) * 16 + (r % 4) * 4)"
  xin = (f"if (u == {nu - 1}) DMA_FILL(1, DESC(desc, 3), {XO}, (int)(x + u * {W})); else DMA_FILL(1, DESC(desc, 2), {XO}, (int)(x + u * {W})); DMA_WAIT(1);" if resid else "")
  goff_adv = f"  ct += {goff * nrb * 3 * 192};\n" if goff else ""
  return V.FULL_H + TILE + f"""
__kernel void {"resid32d" if resid else "rows32d"}(__global float* restrict out, {"__global float* restrict x, " if resid else ""}__global float* restrict ct, __global int* restrict desc, const int core_id) {{
{goff_adv}  __global float* C = LSF(0); __global float* X = LSF({XO});
  for (int u = core_id; u < {nu}; u += {NT}) {{
    int w = u == {nu - 1} ? {tail} : {W};
    if (u == {nu - 1}) DMA_FILL(0, DESC(desc, 1), 0, (int)(ct + u * {RD_G * nrb * 3 * 192})); else DMA_FILL(0, DESC(desc, 0), 0, (int)(ct + u * {RD_G * nrb * 3 * 192}));
    {xin}
    DMA_WAIT(0);
    for (int r = 0; r < {real}; r++) for (int k = 0; k < w; k += 4) *(__global float4*)(X + r * w + k) = {calc.replace("X[r * w + k]", "*(__global float4*)(X + r * w + k)")};
    if (u == {nu - 1}) DMA_DRAIN(2, DESC(desc, 3), {XO}, (int)(out + u * {W})); else DMA_DRAIN(2, DESC(desc, 2), {XO}, (int)(out + u * {W}));
    DMA_WAIT(2);
  }}
  DMA_WAIT_ALL();
}}"""

RQ_G = 8                                                                    # resid32q: C groups (48 columns) a chunk

def resid32q_desc(c, nrb, real):
  """slot g - 1 (g = 1 .. RQ_G): g groups' pieces of one sub-tile residue of one row quad (256 B, a group apart in DDR, 768 B apart
  in LSRAM); RQ_G + 0 / 1: a full chunk's / the last chunk's 4 rows (pitch c, LSRAM pitch 48 RQ_G); + 2 / 3: the same for the
  last quad's rows (nl); + 4 / 5: the record lines of 4 / nl rows."""
  iqr = min(3, -(-real // 4)); nl = real - 4 * (iqr - 1); W = 48 * RQ_G; ngr = -(-c // 48); nch = -(-ngr // RQ_G); WL = c - (nch - 1) * W
  xs = lambda n, w: (n * w * 4, w * 4, c * 4, W * 4)
  return V._desc_slots(*[(g * 256, 256, nrb * 2304, 768) for g in range(1, RQ_G + 1)], xs(4, W), xs(4, WL), xs(nl, W), xs(nl, WL), (4 * 64,), (nl * 64,))

def resid32q_src(c, nrb, real, eps):
  """resid32d (out = x + C, the real rows) that also writes the next RMSNorm's record (rms_rec's: rec[16 r] = 1 / rms of out's row
  r): a task per row quad -- the quad's rows and its C pieces (row block 0, three requests a chunk, one per sub-tile residue)
  stream through LSRAM by chunks of RQ_G groups, double-buffered; out = x + C is built in place (the same additions as resid32 /
  resid32d), accumulated into the rows' sums of squares as rms_a32d's phase 1 does (lane by lane, K ascending: the same bits),
  and leaves by one 2D DMA a chunk; the quad's record lines leave at the end. Rows >= real and columns >= c are not written.
  args: out, x, ct, rec (float32 [REC_N]), desc (resid32q_desc)."""
  assert c % 16 == 0 and 1 <= real <= 12; iqr = min(3, -(-real // 4)); nl = real - 4 * (iqr - 1); W = 48 * RQ_G
  ngr = -(-c // 48); nch = -(-ngr // RQ_G); WL = c - (nch - 1) * W; XB = 4 * W * 4; CB = RQ_G * 768; BUF = XB + CB; REC = 2 * BUF
  assert REC + 256 <= 32768 - 64 and WL % 16 == 0
  def body(R):      # one chunk's sub-tiles j (16 columns: units h = 0, 1 of 8), the R rows interleaved (their sums independent chains)
    lines = []
    for h in (0, 1):
      for r in range(R):
        lines.append(f"float8 v{r}{h} = *(__global float8*)(xk + {r * W + 8 * h}) + F8(*(__global float4*)(q + {32 * h + 4 * r}), *(__global float4*)(q + {32 * h + 16 + 4 * r}));")
      for r in range(R):
        lines.append(f"*(__global float8*)(xk + {r * W + 8 * h}) = v{r}{h}; acc{r} += v{r}{h} * v{r}{h};")
    return f"""NOUNROLL for (int j = 0; j < nj; j++) {{
        __global float* q = Cb + j * 64; __global float* xk = Xb + 16 * j;
        {" ".join(lines)}
      }}"""
  def quad(R, sx, sr):
    fill = lambda k, p: (f"{{ int ga_ = ({k}) * {RQ_G}; int gn_ = {ngr} - ga_ < {RQ_G} ? {ngr} - ga_ : {RQ_G};"
                         f" NOUNROLL for (int t_ = 0; t_ < 3; t_++) DMA_FILL({p}, DESC(desc, gn_ - 1), {p * BUF + XB} + t_ * 256, (int)(ct + (ga_ * {nrb * 3} + t_) * 192 + i * 64));"
                         f" DMA_FILL({p}, DESC(desc, ({k}) == {nch - 1} ? {sx + 1} : {sx}), {p * BUF}, (int)(xq + ({k}) * {W})); }}")
    accs = " ".join(f"float8 acc{r} = BC(0.0f);" for r in range(R))
    return f"""{{
    {accs}
    {fill(0, 0)}
    NOUNROLL for (int k = 0; k < {nch}; k++) {{
      int p = k & 1;
      if (k + 1 < {nch}) {{ if (p == 0) {{ if (k >= 1) DMA_WAIT(3); {fill("k + 1", 1)} }} else {{ DMA_WAIT(2); {fill("k + 1", 0)} }} }}
      if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
      __global float* Xb = LSF(p * {BUF}); __global float* Cb = Xb + {XB // 4}; int nj = k == {nch - 1} ? {WL // 16} : {W // 16};
      {body(R)}
      if (p == 0) DMA_DRAIN(2, DESC(desc, k == {nch - 1} ? {sx + 1} : {sx}), 0, (int)(oq + k * {W})); else DMA_DRAIN(3, DESC(desc, k == {nch - 1} ? {sx + 1} : {sx}), {BUF}, (int)(oq + k * {W}));
    }}
    __global float* Rl = LSF({REC});
    {" ".join(f"Rl[{16 * r}] = 1.0f / __builtin_sqrtf(hsum8(acc{r}) * {1.0 / c!r}f + {eps!r}f);" for r in range(R))}
    DMA_DRAIN(0, DESC(desc, {sr}), {REC}, (int)(rec + i * 64));
  }}"""
  S0 = RQ_G
  return V.FULL_H + TILE + NOUNROLL_H + f"""
__kernel void resid32q(__global float* restrict out, __global float* restrict x, __global float* restrict ct, __global float* restrict rec, __global int* restrict desc, const int core_id) {{
  if (core_id < {iqr}) {{
    const int i = core_id; __global float* xq = x + 4 * i * {c}; __global float* oq = out + 4 * i * {c};
    {quad(nl, S0 + 2, S0 + 5) if iqr == 1 else f"if (i == {iqr - 1}) {quad(nl, S0 + 2, S0 + 5)} else {quad(4, S0, S0 + 4)}"}
  }}
  DMA_WAIT_ALL();
}}"""

def ct_goff(src, nrb, goff, names=("ct",)):
  """A fused GEMM's C tiles: the kernel reads linear `name`'s tiles from group `goff` on -- its pointer advanced by goff whole
  groups ([group][nrb][3][192] floats) at the top of the body (the only edit; goff 0 leaves the source byte-identical)."""
  adv = "".join(f" {n} += {g * nrb * 3 * 192};" for n, g in zip(names, goff if isinstance(goff, tuple) else (goff,)) if g)
  if not adv: return src
  mark = "const int core_id) {\n"; assert src.count(mark) == 1, "one kernel body"
  return src.replace(mark, mark + " " + adv + "\n", 1)

def rows32_src(c, nrb, real=None, goff=0):
  """out fp32 [rows, c] = the C tiles (groups of 48 columns covering >= c). A unit = (row block, a line-aligned column range).
  `goff`: the tiles start at group goff of the buffer (a fused linear's slice, ct_goff)."""
  assert c % 4 == 0; nq = c // 4; rbr, iqr = _real(real, nrb); P = max(1, min(nq // 4, NT // rbr)); chunk = -(-nq // P); chunk += (-chunk) % 4
  return ct_goff(V.FULL_H + TILE + f"""
__kernel void rows32(__global float* restrict out, __global float* restrict ct, const int core_id) {{
  for (int u = core_id; u < {rbr * P}; u += {NT}) {{
    int rb = u / {P}, part = u % {P}; int q1 = (part + 1) * {chunk} < {nq} ? (part + 1) * {chunk} : {nq};
    for (int i = 0; i < {iqr}; i++) for (int q = part * {chunk}; q < q1; q++) {{
      int col = 4 * q; __global float* t = ct + ((col / 48 * {nrb} + rb) * 3 + (col % 48) / 16) * 192 + (i * 4 + (col % 16) / 4) * 16;
      __global float* orow = out + (rb * 12 + 4 * i) * {c} + col;
      for (int r = 0; r < 4; r++) *(__global float4*)(orow + r * {c}) = *(__global float4*)(t + 4 * r);
    }}
  }}
}}""", nrb, goff)

def head_top_src(nc, nrb, m, top3=False):
  """Top-1 of a head part's logits, from its GEMM's C tiles (groups of 48 columns, the first nc valid; rows 0..m-1 of row block 0):
  out fp32 [NT][m][4] = each task's (max, its column, sum of exp(logit - max), 0) over its share of the columns, the first column
  on ties. The host combines the tasks (and the parts): argmax = the first task with the largest max, probability = 1 / sum.
  `top3` (the kernel `head_top3`): out fp32 [NT][m][8] = each task's three largest (logit, column) pairs in order, then the same
  sum of exp (the same lanes in the same order: bit-identical to the top-1 kernel's) and 0; among equal logits the first column
  ranks first. The top-1 source is unchanged by the flag (byte-identical with top3=False)."""
  n8 = -(-nc // 8); tail = nc % 8
  lanes = lambda v: [f"{v}[{j}]" for j in range(8)]
  find = " else ".join(f"if ({e} == vm) am = c + {j};" for j, e in enumerate(lanes("v")))
  mask = "".join(f" v[{j}] = -3.0e38f;" for j in range(tail, 8)) if tail else ""       # the last unit: lanes tail.. are padding
  load = f"int c = 8 * u; __global float* p = ct + CTA(c) + ro; float8 v = F8(*(__global float4*)p, *(__global float4*)(p + 16));" + (f" if (u == {n8 - 1}) {{{mask} }}" if tail else "")
  if top3:
    ins = " ".join(f"x = v[{j}]; if (x > m1) {{ m3 = m2; c3 = c2; m2 = m1; c2 = c1; m1 = x; c1 = c + {j}; }} else if (x > m2) {{ m3 = m2; c3 = c2; m2 = x; c2 = c + {j}; }} else if (x > m3) {{ m3 = x; c3 = c + {j}; }}" for j in range(8))
    return V.FULL_H + TILE + EXP + f"""
#define CTA(c) ((((c) / 48) * {nrb * 3} + ((c) % 48) / 16) * 192 + (((c) % 16) / 4) * 16)
__kernel void head_top3(__global float* restrict out, __global float* restrict ct, const int core_id) {{
  int u0 = core_id * {n8} / {NT}, u1 = (core_id + 1) * {n8} / {NT};
  for (int m = 0; m < {m}; m++) {{
    int ro = (m / 4) * 64 + (m % 4) * 4; float m1 = -3.0e38f, m2 = -3.0e38f, m3 = -3.0e38f, x; int c1 = 0, c2 = 0, c3 = 0;
    for (int u = u0; u < u1; u++) {{ {load} if (hmax8(v) > m3) {{ {ins} }} }}
    float8 s = BC(0.0f);
    for (int u = u0; u < u1; u++) {{ {load} s += exp2_d4((v - BC(m1)) * BC(1.4426950408889634f)); }}
    __global float* o = out + (core_id * {m} + m) * 8; o[0] = m1; o[1] = (float)c1; o[2] = m2; o[3] = (float)c2; o[4] = m3; o[5] = (float)c3; o[6] = hsum8(s); o[7] = 0.0f;
  }}
}}"""
  return V.FULL_H + TILE + EXP + f"""
#define CTA(c) ((((c) / 48) * {nrb * 3} + ((c) % 48) / 16) * 192 + (((c) % 16) / 4) * 16)
__kernel void head_top(__global float* restrict out, __global float* restrict ct, const int core_id) {{
  int u0 = core_id * {n8} / {NT}, u1 = (core_id + 1) * {n8} / {NT};
  for (int m = 0; m < {m}; m++) {{
    int ro = (m / 4) * 64 + (m % 4) * 4; float mx = -3.0e38f; int am = 0;
    for (int u = u0; u < u1; u++) {{ {load} float vm = hmax8(v); if (vm > mx) {{ {find} mx = vm; }} }}
    float8 s = BC(0.0f);
    for (int u = u0; u < u1; u++) {{ {load} s += exp2_d4((v - BC(mx)) * BC(1.4426950408889634f)); }}
    __global float* o = out + (core_id * {m} + m) * 4; o[0] = mx; o[1] = (float)am; o[2] = hsum8(s); o[3] = 0.0f;
  }}
}}"""

def head_top3_row_src(nc, nrb, row):
  """head_top3 for ONE row (`row` of the C tiles' row block 0) and without the sum of exp (lane 6 = 0): the leaf tree's draft
  candidates (the top-1 and its probability come from head_topd / head_reduce in the same job). out fp32 [NT][1][8]."""
  src = head_top_src(nc, nrb, 1, top3=True)
  sm = "    float8 s = BC(0.0f);\n"; a = src.index(sm); b = src.index("\n", src.index("s += exp2_d4", a)) + 1
  src = src[:a] + src[b:]
  edits = [("__kernel void head_top3(", "__kernel void head_top3r("), ("int ro = (m / 4) * 64 + (m % 4) * 4;", f"int ro = {(row // 4) * 64 + (row % 4) * 4};"),
           ("o[6] = hsum8(s);", "o[6] = 0.0f;")]
  for x, y in edits:
    assert src.count(x) == 1, x
    src = src.replace(x, y)
  return src

def embed_tern_desc(K):
  """slot 0: one HAD_B block of a PTQ1_0 row (8 blocks of 28 bytes); 1: the block's signs / an output block (HAD_B fp32); 2: the ids (64 B)."""
  return V._desc_slots((HAD_B // 128 * 28,), (HAD_B * 4,), (64,))

def embed_tern_src(m, K, vocab, pitch, post=1.0 / 32):
  """The token embedding on the device (a PrismML PTQ1_0 table stored latent, z = Hb (s * e)): out fp32 rows r < m (pitch `pitch`
  floats) = s * (Hb z) of row ids[r] of `tab` (the GGUF tensor's bytes, K / 128 blocks of 28 a row: qs[24] five trits a byte, qh[2]
  four, d fp16 at byte 26; bonsai2_gguf's element order). A unit = (row, HAD_B block): its 8 PTQ1_0 blocks arrive by DMA, are
  decoded to t * d in LSRAM, transformed as had_a32d does (h8x2, then the radix-4 stages), scaled by `post` (1 / sqrt(HAD_B)) and
  the signs, and leave by one DMA (one task writes every byte of its 4 KiB). The ids are read by DMA (head_reduce / accept wrote
  them on other cores). Ids outside [0, vocab) are clamped. args: out, ids (int32, >= 16), tab (uint8), s (fp32 [K]), desc."""
  assert K % HAD_B == 0 and 1 <= m <= 16; nb = K // HAD_B; RB = K // 128 * 28; BB = HAD_B // 128 * 28
  L0, S0, Q0, I0 = 0, HAD_B * 4, 2 * HAD_B * 4, 2 * HAD_B * 4 + 256
  src = had_a32_src("plain", HAD_B, 1, 1e-6, real=4, compact=True)
  pre = src[:src.index("__kernel void had_a32(")] + H8X2 + r"""
static inline float h2f(int h) {                     /* fp16 bits -> float (no fp16 loads here: cl_khr_fp16 is off) */
  int e = (h >> 10) & 31, f = h & 1023; union { int i; float x; } u;
  if (e == 0) u.x = (float)f * (1.0f / 16777216.0f);
  else { u.i = ((e + 112) << 23) | (f << 13); }
  return (h & 0x8000) ? -u.x : u.x;
}
"""
  return pre + f"""
__kernel void embed_tern(__global float* restrict out, __global int* restrict ids, __global unsigned char* restrict tab, __global float* restrict s, __global int* restrict desc, const int core_id) {{
  if (core_id < {m * nb}) {{
  DMA_FILL(2, DESC(desc, 2), {I0}, (int)ids); DMA_WAIT(2);
  __global int* I = (__global int*)LSF({I0}); __global float* L = LSF({L0}); __global float* S = LSF({S0}); __global unsigned char* Q = (__global unsigned char*)LSF({Q0});
  for (int u = core_id; u < {m * nb}; u += {NT}) {{
    int r = u / {nb}, b = u % {nb}; int id = I[r]; id = id < 0 ? 0 : id >= {vocab} ? {vocab - 1} : id;
    DMA_FILL(0, DESC(desc, 0), {Q0}, (int)(tab + id * {RB} + b * {BB})); DMA_FILL(1, DESC(desc, 1), {S0}, (int)(s + b * {HAD_B}));
    DMA_WAIT(0); DMA_WAIT(1);
    for (int k = 0; k < {HAD_B // 128}; k++) {{
      __global unsigned char* q = Q + k * 28; __global float* Lb = L + k * 128;
      float d = h2f(q[26] | (q[27] << 8));
      for (int j = 0; j < 16; j++) {{ int v = q[j]; for (int n = 0; n < 5; n++) {{ Lb[n * 16 + j] = (float)((((v & 255) * 3) >> 8) - 1) * d; v *= 3; }} }}
      for (int j = 0; j < 8; j++) {{ int v = q[16 + j]; for (int n = 0; n < 5; n++) {{ Lb[80 + n * 8 + j] = (float)((((v & 255) * 3) >> 8) - 1) * d; v *= 3; }} }}
      for (int j = 0; j < 2; j++) {{ int v = q[24 + j]; for (int n = 0; n < 4; n++) {{ Lb[120 + n * 2 + j] = (float)((((v & 255) * 3) >> 8) - 1) * d; v *= 3; }} }}
    }}
    for (int k = 0; k < {HAD_B}; k += 16) h8x2((__global float8*)(L + k));
    for (int h = 8; h < {HAD_B // 2}; h *= 4) for (int j0 = 0; j0 < {HAD_B}; j0 += 4 * h) for (int k = j0; k < j0 + h; k += 8) {{
      float8 a = *(__global float8*)(L + k), bb = *(__global float8*)(L + k + h), cc = *(__global float8*)(L + k + 2 * h), dd = *(__global float8*)(L + k + 3 * h);
      float8 s0 = a + bb, s1 = a - bb, s2 = cc + dd, s3 = cc - dd;
      *(__global float8*)(L + k) = s0 + s2; *(__global float8*)(L + k + h) = s1 + s3; *(__global float8*)(L + k + 2 * h) = s0 - s2; *(__global float8*)(L + k + 3 * h) = s1 - s3;
    }}
    for (int k = 0; k < {HAD_B // 2}; k += 8) {{
      float8 a = *(__global float8*)(L + k), bb = *(__global float8*)(L + k + {HAD_B // 2});
      *(__global float8*)(L + k) = a + bb; *(__global float8*)(L + k + {HAD_B // 2}) = a - bb;
    }}
    for (int k = 0; k < {HAD_B}; k += 8) *(__global float8*)(L + k) = *(__global float8*)(L + k) * BC({post!r}f) * *(__global float8*)(S + k);
    DMA_DRAIN(3, DESC(desc, 1), {L0}, (int)(out + r * {pitch} + b * {HAD_B})); DMA_WAIT(3);
  }}
  }}
  DMA_WAIT_ALL();
}}"""

def accept_src(m):
  """Speculative decoding's acceptance on the device (one task): vt = the verify pass's input ids [cur, d1 .. d(m-1)], g = its head's
  ids (head_reduce) -> a = 1 + the drafts the model agrees with (the first a - 1 drafts equal g[0 .. a - 2]), the next token
  cur = g[a - 1]; writes out [a, cur, the a new tokens (d1 .. d(a-1), cur), cur ...] (16), accb[0] = a (gdn_commit), nxt = the new
  tokens (the drafter's catch-up ids; cur pads it) and vt[0] = cur. args: out, vt, g, accb, nxt."""
  return V.FULL_H + f"""
__kernel void accept(__global int* restrict out, __global int* restrict vt, __global int* restrict g, __global int* restrict accb, __global int* restrict nxt, const int core_id) {{
  if (core_id == 0) {{
    int a = 1; while (a < {m} && vt[a] == g[a - 1]) a++;
    int cur = g[a - 1];
    for (int j = 0; j < 16; j++) {{ int t = j < a - 1 ? vt[1 + j] : cur; nxt[j] = t; if (j < 14) out[2 + j] = t; }}
    out[0] = a; out[1] = cur; accb[0] = a; vt[0] = cur;
  }}
}}"""

def pick_src(j, row):
  """A draft's id into the next verify pass's input (one task): id = ids[row] (the draft head's head_reduce output) -> vt[1 + j ..
  15] (the later slots padded with it, as the host padded) and dt[0] (the chained draft's embedding id). args: vt, ids, dt."""
  return V.FULL_H + f"""
__kernel void pick_id(__global int* restrict vt, __global int* restrict ids, __global int* restrict dt, const int core_id) {{
  if (core_id == 0) {{ int id = ids[{row}]; for (int k = {1 + j}; k < 16; k++) vt[k] = id; dt[0] = id; }}
}}"""

def mtpin_src(r, mp, H, eps, hoff, catch):
  """The MTP layer's input rows on the device (qwen38_generate.mtp_rows): row i < r = [rms(E[i]) * ne1 | rms(h_i) * nh1] with
  h_i = rms(X[hoff + i]) * fn1 (`catch`: the 27B's final norm of its hidden rows) or X[hoff + i] (chained: the MTP layer's own
  output); rows r .. mp - 1 repeat row r - 1 (mtp_pass' padding). One task a row. args: out [mp, 2H], E [>= r, H], X [.., H], ne1,
  nh1, fn1 (the (1 + w) weights, fp32 [H])."""
  def ssq(v): return f"float8 a8 = BC(0.0f); for (int k = 0; k < {H}; k += 8) {{ float8 v = *(__global float8*)({v} + k); a8 += v * v; }} float inv = 1.0f / __builtin_sqrtf(hsum8(a8) * {1.0 / H!r}f + {eps!r}f);"
  hrow = f"""
    __global float* x = X + ({hoff} + i) * {H}; __global float* oh = o + {H};
    {{ {ssq("x")}
      {"for (int k = 0; k < %d; k += 8) *(__global float8*)(oh + k) = *(__global float8*)(x + k) * BC(inv) * *(__global float8*)(fn1 + k);" % H if catch else "for (int k = 0; k < %d; k += 8) *(__global float8*)(oh + k) = *(__global float8*)(x + k);" % H} }}
    {{ {ssq("oh")}
      for (int k = 0; k < {H}; k += 8) *(__global float8*)(oh + k) = *(__global float8*)(oh + k) * BC(inv) * *(__global float8*)(nh1 + k); }}"""
  return V.FULL_H + f"""
__kernel void mtp_in(__global float* restrict out, __global float* restrict E, __global float* restrict X, __global float* restrict ne1, __global float* restrict nh1, __global float* restrict fn1, const int core_id) {{
  for (int i = core_id; i < {r}; i += {NT}) {{
    __global float* o = out + i * {2 * H}; __global float* e = E + i * {H};
    {{ {ssq("e")}
      for (int k = 0; k < {H}; k += 8) *(__global float8*)(o + k) = *(__global float8*)(e + k) * BC(inv) * *(__global float8*)(ne1 + k); }}
    {hrow}
    for (int p = (i == {r - 1} ? {r} : {mp}); p < {mp}; p++) for (int k = 0; k < {2 * H}; k += 8) *(__global float8*)(out + p * {2 * H} + k) = *(__global float8*)(o + k);
  }}
}}"""

MI_PC = 256                                                                 # mtpin_d: columns an output piece (one drain)

def mtpin_d_desc(H):
  """slot 0: one whole H-float row (fp32) into LSRAM; slot 1: an MI_PC-float output piece."""
  return V._desc_slots((H * 4,), (MI_PC * 4,))

def mtpin_d_src(r, mp, H, eps, hoff, catch):
  """mtpin_src over the 12 TECs by DMA: a unit = (row i, half: the embedding's or the hidden row's, part of the half's columns);
  W = 2 r halves, P = max(1, NT // W) parts each (tasks W P.. idle). Each unit takes its half's whole source row into LSRAM by one
  DMA (E[i] or X[hoff + i]; rows another core or job wrote are never read by cached loads), sums its squares there in mtpin_src's
  order -- per lane over k = 0, 8, .., then hsum8 -- (the hidden half with `catch`: x * inv * fn1 rounded, then its squares, as
  mtpin_src's second pass reads the stored row), scales its part's columns in place with mtpin_src's expressions and drains them
  (MI_PC-float pieces; the last real row's pieces also to rows r .. mp - 1, mtp_pass' padding). The same operations on the same
  values: the same bits. args: out [mp, 2H], E, X, ne1, nh1, fn1, desc (mtpin_d_desc)."""
  assert H % MI_PC == 0 and H * 4 + 64 <= 32768 - 64 and 1 <= r <= mp; W = 2 * r; P = max(1, NT // W); NSL = H // MI_PC
  def ssq(body): return f"float8 a8 = BC(0.0f); for (int k = 0; k < {H}; k += 8) {{ {body} a8 += v * v; }} float inv = 1.0f / __builtin_sqrtf(hsum8(a8) * {1.0 / H!r}f + {eps!r}f);"
  return V.FULL_H + f"""
__kernel void mtp_in_d(__global float* restrict out, __global float* restrict E, __global float* restrict X, __global float* restrict ne1, __global float* restrict nh1, __global float* restrict fn1, __global int* restrict desc, const int core_id) {{
  if (core_id < {W * P}) {{                                /* one exit for every task (an early `return` beside DMA: had_a32d_src) */
  __global float* B = LSF(0);
  for (int u = core_id; u < {W * P}; u += {NT}) {{
    int wi = u / {P}, part = u % {P}, i = wi / 2, hf = wi % 2;
    __global float* src = hf ? X + ({hoff} + i) * {H} : E + i * {H};
    DMA_FILL(0, DESC(desc, 0), 0, (int)src); DMA_WAIT(0);
    float s1, s2 = 1.0f;
    {{ {ssq("float8 v = *(__global float8*)(B + k);")} s1 = inv; }}
    {"if (hf) { " + ssq("float8 v = *(__global float8*)(B + k) * BC(s1) * *(__global float8*)(fn1 + k);") + " s2 = inv; }" if catch else ""}
    int p0 = part * {NSL} / {P}, p1 = (part + 1) * {NSL} / {P};
    for (int p = p0; p < p1; p++) {{
      __global float* b = B + p * {MI_PC};
      if (hf) for (int k = 0; k < {MI_PC}; k += 8) {{
        {"float8 oh = *(__global float8*)(b + k) * BC(s1) * *(__global float8*)(fn1 + p * %d + k); *(__global float8*)(b + k) = oh * BC(s2) * *(__global float8*)(nh1 + p * %d + k);" % (MI_PC, MI_PC) if catch else "*(__global float8*)(b + k) = *(__global float8*)(b + k) * BC(s1) * *(__global float8*)(nh1 + p * %d + k);" % MI_PC}
      }} else for (int k = 0; k < {MI_PC}; k += 8) *(__global float8*)(b + k) = *(__global float8*)(b + k) * BC(s1) * *(__global float8*)(ne1 + p * {MI_PC} + k);
      for (int q = i; q < (i == {r - 1} ? {mp} : i + 1); q++) {{
        DMA_DRAIN(3, DESC(desc, 1), p * {MI_PC * 4}, (int)(out + q * {2 * H} + hf * {H} + p * {MI_PC})); DMA_WAIT(3);
      }}
    }}
  }}
  }}
  DMA_WAIT_ALL();
}}"""

def head_reduce_desc(m): return V._desc_slots((NT * m * 16,))

def head_reduce_src(m, c0s):
  """The head's token on the device: head_top's partials of every part ([NT][m][4] each, part p's columns from c0s[p]) -> out int32
  [ids (m) | probabilities (m, fp32 bits)], combine_tops' rule: the id is the column of the first (part, task) holding the largest
  max, the probability 1 / sum over (part, task) of sum_exp * exp(max - the largest). One task; the partials (stored by the head_top
  tasks of all three cores) are read by DMA, never by cached loads. args: out, the parts' partials, desc (head_reduce_desc)."""
  P = len(c0s); B = NT * m * 16; assert P * B <= 32768 - 64
  fill = " ".join(f"DMA_FILL(0, DESC(desc, 0), {p * B}, (int)t{p}); DMA_WAIT(0);" for p in range(P))
  scan = lambda body: " ".join(f"for (int t = 0; t < {NT}; t++) {{ __global float* q = T + {p * B // 4} + (t * {m} + r) * 4; {body.replace('C0', str(c0s[p]))} }}" for p in range(P))
  return V.FULL_H + TILE + EXP + f"""
__kernel void head_reduce(__global int* restrict out, {", ".join(f"__global float* restrict t{p}" for p in range(P))}, __global int* restrict desc, const int core_id) {{
  if (core_id == 0) {{
    {fill}
    __global float* T = LSF(0); __global float* pr = (__global float*)(out + {m});
    for (int r = 0; r < {m}; r++) {{
      float mx = -3.0e38f; int id = 0; float s = 0.0f;
      {scan("if (q[0] > mx) { mx = q[0]; id = C0 + (int)q[1]; }")}
      {scan("float8 e = exp2_d4(BC((q[0] - mx) * 1.4426950408889634f)); s += q[2] * e[0];")}
      out[r] = id; pr[r] = 1.0f / s;
    }}
  }}
}}"""

HT_CG = 6                                                                  # head_topd: C groups a DMA chunk (m <= 4: row quad 0)
HT_CG2 = 5                                                                 # head_topd, m 5..8: row quads 0 and 1 (2 x 13824 B at nrb 2)

def head_topd_desc(nc, nrb, nq=1):
  """slot g - 1: a chunk of g groups' (g = 1 .. HT_CG) row-quad-0 pieces of row block 0 (256 B each, pitch 768: the other row
  block's pieces ride along; the last group's stop at row block 0), the chunk size the slot index. `nq` = 2 (m 5..8): row quads 0
  and 1 (512 B pieces at pitch 768, packed 512 apart in LSRAM), g = 1 .. HT_CG2."""
  if nq == 1:
    n = lambda g: ((g - 1) * 3 * nrb + 3) * 256
    return V._desc_slots(*[(n(g), 256, 768, 256) for g in range(1, HT_CG + 1)])
  assert nq == 2
  return V._desc_slots(*[(((g - 1) * 3 * nrb + 3) * 512, 512, 768, 512) for g in range(1, HT_CG2 + 1)])

def head_topd_src(nc, nrb, m):
  """head_top (top-1, m <= 8 rows) with the C tiles streamed through LSRAM by DMA: a task owns whole groups [g0, g1) (columns in
  order, so the first column on ties is still its own first and the tasks' order is the columns'); per chunk of HT_CG groups the
  rows' maxima and sums of exp are folded in online (the sum rescaled when the max rises). out as head_top's: [NT][m][4] =
  (max, column, sum of exp(v - max), 0). m <= 4: row quad 0 of each strip (HT_CG groups a chunk); m 5..8: row quads 0 and 1
  (HT_CG2, desc head_topd_desc(nc, nrb, 2)). args: out, ct, desc (head_topd_desc)."""
  assert 1 <= m <= 8; ng = -(-nc // 48); SB = 3 * nrb
  cg, PF, RO = (HT_CG, 64, "r * 4") if m <= 4 else (HT_CG2, 128, "(r / 4) * 64 + (r % 4) * 4")   # m > 4: rows 4.. in row quad 1
  CH = ((cg - 1) * SB + 3) * PF * 4; assert 2 * CH <= 32768 - 64
  lanes = " else ".join(f"if (v[{j}] == vm) am[r] = c + {j};" for j in range(8))
  return V.FULL_H + TILE + EXP + f"""
__kernel void head_topd(__global float* restrict out, __global float* restrict ct, __global int* restrict desc, const int core_id) {{
  int g0 = core_id * {ng} / {NT}, g1 = (core_id + 1) * {ng} / {NT}; int nch = (g1 - g0 + {cg - 1}) / {cg};
  float mx[{m}], ssum[{m}]; int am[{m}]; for (int r = 0; r < {m}; r++) {{ mx[r] = -3.0e38f; ssum[r] = 0.0f; am[r] = 0; }}
  #define ISSUE(k, buf) {{ int ga = g0 + (k) * {cg}; int gn = g1 - ga < {cg} ? g1 - ga : {cg}; DMA_FILL(buf, DESC(desc, gn - 1), (buf) * {CH}, (int)(ct + ga * {SB * 192})); }}
  if (nch > 0) ISSUE(0, 0);
  for (int k = 0; k < nch; k++) {{
    int p = k & 1; if (k + 1 < nch) {{ if (p == 0) ISSUE(k + 1, 1) else ISSUE(k + 1, 0) }}
    if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
    __global float* B = LSF(p * {CH}); int ga = g0 + k * {cg}; int gn = g1 - ga < {cg} ? g1 - ga : {cg};
    for (int r = 0; r < {m}; r++) {{
      float cm = -3.0e38f; int ca = 0;
      for (int gg = 0; gg < gn; gg++) for (int sq = 0; sq < 6; sq++) {{      /* 8 columns: sub-tile sq / 2, column quads 2 (sq % 2), +1 */
        int c = (ga + gg) * 48 + sq * 8; __global float* q = B + (gg * {SB} + sq / 2) * {PF} + (sq % 2) * 32 + {RO};
        float8 v = F8(*(__global float4*)q, *(__global float4*)(q + 16));
        if (c + 8 > {nc}) {{ for (int j = 0; j < 8; j++) if (c + j >= {nc}) v[j] = -3.0e38f; }}
        float vm = hmax8(v); if (vm > cm) {{ cm = vm; ca = c; }}
      }}
      if (cm > mx[r]) {{                                                  /* the chunk's max beats the row's: its column, the rescale */
        for (int gg = 0; gg < gn; gg++) for (int sq = 0; sq < 6; sq++) {{
          int c = (ga + gg) * 48 + sq * 8; if (c != ca) continue;
          __global float* q = B + (gg * {SB} + sq / 2) * {PF} + (sq % 2) * 32 + {RO}; float8 v = F8(*(__global float4*)q, *(__global float4*)(q + 16));
          if (c + 8 > {nc}) {{ for (int j = 0; j < 8; j++) if (c + j >= {nc}) v[j] = -3.0e38f; }}
          float vm = cm; {lanes}
        }}
        if (ssum[r] != 0.0f) {{ float8 e = exp2_d4(BC((mx[r] - cm) * 1.4426950408889634f)); ssum[r] *= e[0]; }} mx[r] = cm;
      }}
      float8 s8 = BC(0.0f);
      for (int gg = 0; gg < gn; gg++) for (int sq = 0; sq < 6; sq++) {{
        int c = (ga + gg) * 48 + sq * 8; __global float* q = B + (gg * {SB} + sq / 2) * {PF} + (sq % 2) * 32 + {RO};
        float8 v = F8(*(__global float4*)q, *(__global float4*)(q + 16));
        if (c + 8 > {nc}) {{ for (int j = 0; j < 8; j++) if (c + j >= {nc}) v[j] = -3.0e38f; }}
        s8 += exp2_d4((v - BC(mx[r])) * BC(1.4426950408889634f));
      }}
      ssum[r] += hsum8(s8);
    }}
  }}
  for (int r = 0; r < {m}; r++) {{ __global float* o = out + (core_id * {m} + r) * 4; o[0] = mx[r]; o[1] = (float)am[r]; o[2] = ssum[r]; o[3] = 0.0f; }}
  DMA_WAIT_ALL();
  #undef ISSUE
}}"""

# ---- QWEN_SMALL2: head_top3 / head_top3r by DMA, bit-identical ---------------------------------------------------------------
# head_top3 read its C tiles by cached loads (Ornith's draft head ~485 us a part, head_top3r ~395). These keep its task split (8-
# column units, task t: units t n8 / NT .. (t + 1) n8 / NT), its per-row loops in unit order and its expressions (hmax8, the
# insertion, exp2_d4, hsum8): the same partials, bit for bit. The task's whole C groups stream through LSRAM: per chunk of <= T3_CG
# groups three requests (one per sub-tile residue s = 0, 1, 2: the row quads' pieces of the groups' row block 0, a group apart in
# DDR, packed [group][s][quads] in LSRAM), double-buffered (flags 0 / 1). head_top3d makes two passes over the range (the top-3, then
# the sum of exp against the final maximum), head_top3rd one (no sum).
T3_CG = 8                                                                    # head_top3d: C groups a chunk

def head_top3d_desc(nrb, nq):
  """slot g - 1 (g = 1 .. T3_CG): g groups' pieces of one sub-tile residue -- `nq` row quads (256 nq contiguous bytes) of row block
  0, a group apart (nrb * 2304 B) in DDR, 3 * 256 nq apart in LSRAM."""
  PB = 256 * nq
  return V._desc_slots(*[(g * PB, PB, nrb * 2304, 3 * PB) for g in range(1, T3_CG + 1)])

def head_top3d_src(nc, nrb, m, row=None):
  """head_top3 (m <= 8 rows: row quads 0 .. nq - 1) or, `row` given, head_top3r (that one row, no sum: one pass, its row quad only)
  with the C tiles by DMA (see T3_CG). out as theirs: fp32 [NT][m (1)][8]. args: out, ct, desc (head_top3d_desc(nrb, nq))."""
  one = row is not None; mm = 1 if one else m
  assert (1 <= m <= 8) if not one else 0 <= row < 12
  n8 = -(-nc // 8); tail = nc % 8; nq = 1 if one else -(-m // 4); PF = 64 * nq; CHB = T3_CG * 3 * PF * 4; assert 2 * CHB <= 32768 - 64
  qoff = (row // 4) * 64 if one else 0                                          # head_top3r: its row quad's pieces only
  ro = (lambda r: f"{(row % 4) * 4}") if one else (lambda r: f"(({r}) / 4) * 64 + (({r}) % 4) * 4")
  mask = "".join(f" v[{j}] = -3.0e38f;" for j in range(tail, 8)) if tail else ""
  load = (f"int c = 8 * u; __global float* p = B + ((u - 6 * ga) >> 1) * {PF} + (u & 1) * 32 + RO; "   # (group, sub-tile) = (u - 6 ga) / 2, jt = 2 (u & 1)
          
          f"float8 v = F8(*(__global float4*)p, *(__global float4*)(p + 16));" + (f" if (u == {n8 - 1}) {{{mask} }}" if tail else ""))
  ins = ("NOUNROLL for (int j = 0; j < 8; j++) { x = v[j]; if (x > m1) { m3 = m2; c3 = c2; m2 = m1; c2 = c1; m1 = x; c1 = c + j; } "
         "else if (x > m2) { m3 = m2; c3 = c2; m2 = x; c2 = c + j; } else if (x > m3) { m3 = x; c3 = c + j; } }")   # the lanes in order (a small loop body)
  def issue(k, buf):
    return (f"{{ int ga_ = g0 + ({k}) * {T3_CG}; int gn_ = g1 - ga_ < {T3_CG} ? g1 - ga_ : {T3_CG};"
            f" NOUNROLL for (int t_ = 0; t_ < 3; t_++) DMA_FILL({buf}, DESC(desc, gn_ - 1), {buf} * {CHB} + t_ * {PF * 4}, (int)(ct + (ga_ * {nrb * 3} + t_) * 192 + {qoff})); }}")
  def sweep(body):     # every chunk of the task's groups, double-buffered; body sees B (the chunk), ga / gn, and runs its units
    return f"""
  if (nch > 0) {issue(0, 0)}
  NOUNROLL for (int k = 0; k < nch; k++) {{
    int p = k & 1; if (k + 1 < nch) {{ if (p == 0) {issue("k + 1", 1)} else {issue("k + 1", 0)} }}
    if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
    __global float* B = LSF(p * {CHB}); int ga = g0 + k * {T3_CG}, gn = g1 - ga < {T3_CG} ? g1 - ga : {T3_CG};
    int ua = 6 * ga > u0 ? 6 * ga : u0, ub = 6 * (ga + gn) < u1 ? 6 * (ga + gn) : u1;   /* the chunk's units of the task (6 a group) */
    {body}
  }}"""
  p1 = f"""NOUNROLL for (int r = 0; r < {mm}; r++) {{
      int RO = {ro("r")}; float m1 = M1[r], m2 = M2[r], m3 = M3[r], x; int c1 = C1[r], c2 = C2[r], c3 = C3[r];
      NOUNROLL for (int u = ua; u < ub; u++) {{ {load} if (hmax8(v) > m3) {{ {ins} }} }}
      M1[r] = m1; M2[r] = m2; M3[r] = m3; C1[r] = c1; C2[r] = c2; C3[r] = c3;
    }}"""
  p2 = f"""NOUNROLL for (int r = 0; r < {mm}; r++) {{
      int RO = {ro("r")}; float8 s = S8[r]; float m1 = M1[r];
      NOUNROLL for (int u = ua; u < ub; u++) {{ {load} s += exp2_d4((v - BC(m1)) * BC(1.4426950408889634f)); }}
      S8[r] = s;
    }}"""
  name = "head_top3r" if one else "head_top3"
  return V.FULL_H + TILE + EXP + NOUNROLL_H + f"""
__kernel void {name}(__global float* restrict out, __global float* restrict ct, __global int* restrict desc, const int core_id) {{
  int u0 = core_id * {n8} / {NT}, u1 = (core_id + 1) * {n8} / {NT};
  int g0 = u0 / 6, g1 = u1 > u0 ? (u1 - 1) / 6 + 1 : g0; int nch = (g1 - g0 + {T3_CG - 1}) / {T3_CG};
  float M1[{mm}], M2[{mm}], M3[{mm}]; int C1[{mm}], C2[{mm}], C3[{mm}];{"" if one else f" float8 S8[{mm}];"}
  for (int r = 0; r < {mm}; r++) {{ M1[r] = -3.0e38f; M2[r] = -3.0e38f; M3[r] = -3.0e38f; C1[r] = 0; C2[r] = 0; C3[r] = 0;{"" if one else " S8[r] = BC(0.0f);"} }}
  {sweep(p1)}{"" if one else sweep(p2)}
  for (int r = 0; r < {mm}; r++) {{
    __global float* o = out + (core_id * {mm} + r) * 8; o[0] = M1[r]; o[1] = (float)C1[r]; o[2] = M2[r]; o[3] = (float)C2[r]; o[4] = M3[r]; o[5] = (float)C3[r];
    o[6] = {"0.0f" if one else "hsum8(S8[r])"}; o[7] = 0.0f;
  }}
  DMA_WAIT_ALL();
}}"""

def gemv32_src(rows, k, n, nt=NT, norm=False, eps=1e-6, stacked=False):
  """out fp32 [rows, n] = x [rows, k] @ w [n, k]^T (small n, fp32 weights). A unit = a group of JC output columns; K in blocks
  of KB with the JC weight rows' block staged in LSRAM (plain loads), every input row streamed once per block. `norm`: the rows
  are RMS-normalised on the way (out[m] *= 1 / sqrt(mean(x[m]^2) + eps), the squares summed in the main loop; the norm's
  weight is folded into w). `stacked`: w is a stack [L, n, k] and a 4th argument idx (int32 [1]) picks the layer."""
  JC, KB = 8, 512
  assert k % KB == 0 and n % JC == 0 and JC * KB * 4 <= 32768; P = n // JC
  ss_decl = f" float ss[{rows}]; for (int m = 0; m < {rows}; m++) ss[m] = 0.0f;" if norm else ""
  ss_acc = " sq += xv * xv;" if norm else ""; ss_sum = " ss[m] += hsum8(sq);" if norm else ""
  inv_mul = f" * (1.0f / __builtin_sqrtf(ss[m] * {1.0 / k!r}f + {eps!r}f))" if norm else ""
  sig_idx = ", __global int* restrict idx" if stacked else ""; w_sel = f"  w += idx[0] * {n * k};\n" if stacked else ""
  return V.FULL_H + f"""
__kernel void gemv32(__global float* restrict out, __global float* restrict x, __global float* restrict w{sig_idx}, const int core_id) {{
{w_sel}  for (int u = core_id; u < {P}; u += {nt}) {{
    float acc[{rows}][{JC}];{ss_decl}
    for (int m = 0; m < {rows}; m++) for (int j = 0; j < {JC}; j++) acc[m][j] = 0.0f;
    for (int kb = 0; kb < {k}; kb += {KB}) {{
      __global float* WB = LSF(0);
      for (int j = 0; j < {JC}; j++) for (int i = 0; i < {KB}; i += 8) *(__global float8*)(WB + j * {KB} + i) = *(__global float8*)(w + (u * {JC} + j) * {k} + kb + i);
      for (int m = 0; m < {rows}; m++) {{
        __global float* xr = x + m * {k} + kb;
        float8 a0 = BC(0.0f), a1 = BC(0.0f), a2 = BC(0.0f), a3 = BC(0.0f), a4 = BC(0.0f), a5 = BC(0.0f), a6 = BC(0.0f), a7 = BC(0.0f), sq = BC(0.0f);
        for (int i = 0; i < {KB}; i += 8) {{
          float8 xv = *(__global float8*)(xr + i);{ss_acc}
          a0 += xv * *(__global float8*)(WB + 0 * {KB} + i); a1 += xv * *(__global float8*)(WB + 1 * {KB} + i);
          a2 += xv * *(__global float8*)(WB + 2 * {KB} + i); a3 += xv * *(__global float8*)(WB + 3 * {KB} + i);
          a4 += xv * *(__global float8*)(WB + 4 * {KB} + i); a5 += xv * *(__global float8*)(WB + 5 * {KB} + i);
          a6 += xv * *(__global float8*)(WB + 6 * {KB} + i); a7 += xv * *(__global float8*)(WB + 7 * {KB} + i);
        }}
        acc[m][0] += hsum8(a0); acc[m][1] += hsum8(a1); acc[m][2] += hsum8(a2); acc[m][3] += hsum8(a3);
        acc[m][4] += hsum8(a4); acc[m][5] += hsum8(a5); acc[m][6] += hsum8(a6); acc[m][7] += hsum8(a7);{ss_sum}
      }}
    }}
    for (int m = 0; m < {rows}; m++) for (int j = 0; j < {JC}; j++) out[m * {n} + u * {JC} + j] = acc[m][j]{inv_mul};
  }}
}}"""

def ring_pitch(C):
  """The row pitch (floats) of the DeltaNet buffers of C-float qkv rows: the conv rings [L][CONV][CP], the conv taps [L][CONV][CP],
  the verify's raw rows [NG][M][CP]. One 64-byte line of padding a row: the L2 (8-way, 128 sets of 64 B, set = address bits
  [12:6]) put every row of a C * 4 = 32 KiB pitch (Ornith: a multiple of 8 KiB) in the same set, and with some placements of the
  buffers gdn_tokm's fast preparation hung the TEC; C + 16 spreads consecutive rows over consecutive sets for any base."""
  return C + 16

def gdn_tok_src(NV, NK, DK, DV, C, CONV, rows, eps):
  """The whole Gated DeltaNet token step after the projections, for all NV heads: the causal conv over the layer's ring of the
  last CONV-1 raw qkv rows + the new row, SiLU, the l2 norms of q / k, the gated delta rule on the layer's state, the output
  RMS norm x weight x SiLU(z), written as row 0 of `o_rows` (rows 1.. zeroed); the new raw row goes into the ring at pos % CONV.
  args: o_rows [rows, NV*DV], Sall [L, NV, DK, DV], Call [L, CONV, CP] (the rings, CP = ring_pitch(C)), idx (int32 [1]: the layer), posb (int32 [1]:
  the token's position), qkv_rows [rows, C] (row 0 read), z_rows [rows, NV*DV] (row 0), bd [2, NV] (beta, decay),
  cwt [L, CONV, CP] (the conv taps, tap t for the token pos-CONV+1+t), nw [L, DV] (per-layer stacks). A unit = one value head (its q / k head's conv and
  norm redone by the 3 units sharing it; only the first of the 3 writes their ring channels)."""
  REP = NV // NK; QO, KO, VO = 0, NK * DK, 2 * NK * DK; CP = ring_pitch(C); assert VO + NV * DV == C and DK == DV
  return V.FULL_H + EXP + f"""
static inline void conv_silu(__global float* restrict x, __global float* restrict Cr, __global float* restrict cwt, int pos, int off, __global float* restrict out) {{
  for (int i = 0; i < {DK}; i += 8) {{
    float8 acc = *(__global float8*)(x + off + i) * *(__global float8*)(cwt + {(CONV - 1) * CP} + off + i);
    for (int t = 0; t < {CONV - 1}; t++) {{
      int tok = pos - {CONV - 1} + t;
      if (tok >= 0) acc += *(__global float8*)(Cr + (tok % {CONV}) * {CP} + off + i) * *(__global float8*)(cwt + t * {CP} + off + i);
    }}
    *(__global float8*)(out + i) = sw(acc, BC(1.0f));
  }}
}}
static inline void l2n(__global float* restrict v, float scale) {{
  float8 s = BC(0.0f); for (int i = 0; i < {DK}; i += 8) {{ float8 a = *(__global float8*)(v + i); s += a * a; }}
  float8 inv = BC(scale / __builtin_sqrtf(hsum8(s) + 1e-6f)); for (int i = 0; i < {DK}; i += 8) *(__global float8*)(v + i) = *(__global float8*)(v + i) * inv;
}}
__kernel void gdn_tok(__global float* restrict o_rows, __global float* restrict Sall, __global float* restrict Call, __global int* restrict idx,
                      __global int* restrict posb, __global float* restrict x, __global float* restrict z, __global float* restrict bd,
                      __global float* restrict cwt, __global float* restrict nw, const int core_id) {{
  int L = idx[0], pos = posb[0]; __global float* S = Sall + L * {NV * DK * DV}; __global float* Cr = Call + L * {CONV * CP}; int slot = pos % {CONV};
  cwt += L * {CONV * CP}; nw += L * {DV};
  __global float* qs = LSF(0); __global float* ks = qs + {DK}; __global float* vs = ks + {DK};
  for (int h = core_id; h < {NV}; h += {NT}) {{
    int j = h / {REP}; __global float* Sh = S + h * {DK * DV}; float b = bd[h], d = bd[{NV} + h];
    conv_silu(x, Cr, cwt, pos, {QO} + j * {DK}, qs); l2n(qs, {1.0 / DK ** 0.5!r}f);
    conv_silu(x, Cr, cwt, pos, {KO} + j * {DK}, ks); l2n(ks, 1.0f);
    conv_silu(x, Cr, cwt, pos, {VO} + h * {DV}, vs);
    float8 kv[{DV // 8}]; for (int c = 0; c < {DV // 8}; c++) kv[c] = BC(0.0f);
    for (int i = 0; i < {DK}; i++) {{                          /* S *= d; kv += k_i * S_i */
      float8 kb = BC(ks[i]), db = BC(d); __global float* Sr = Sh + i * {DV};
      for (int c = 0; c < {DV // 8}; c++) {{ float8 s = *(__global float8*)(Sr + 8 * c) * db; *(__global float8*)(Sr + 8 * c) = s; kv[c] += kb * s; }}
    }}
    float8 delta[{DV // 8}]; for (int c = 0; c < {DV // 8}; c++) delta[c] = (*(__global float8*)(vs + 8 * c) - kv[c]) * BC(b);
    float8 ob[{DV // 8}]; for (int c = 0; c < {DV // 8}; c++) ob[c] = BC(0.0f);
    for (int i = 0; i < {DK}; i++) {{                          /* S_i += k_i delta; o += q_i S_i */
      float8 kb = BC(ks[i]), qb = BC(qs[i]); __global float* Sr = Sh + i * {DV};
      for (int c = 0; c < {DV // 8}; c++) {{ float8 s = *(__global float8*)(Sr + 8 * c) + kb * delta[c]; *(__global float8*)(Sr + 8 * c) = s; ob[c] += qb * s; }}
    }}
    float8 ss = BC(0.0f); for (int c = 0; c < {DV // 8}; c++) ss += ob[c] * ob[c];
    float8 inv = BC(1.0f / __builtin_sqrtf(hsum8(ss) * {1.0 / DV!r}f + {eps!r}f));
    for (int c = 0; c < {DV // 8}; c++) {{                     /* o = rmsnorm(o) * nw * silu(z) -> row 0; rows 1.. zero */
      float8 zv = *(__global float8*)(z + h * {DV} + 8 * c);
      *(__global float8*)(o_rows + h * {DV} + 8 * c) = ob[c] * inv * *(__global float8*)(nw + 8 * c) * sw(zv, BC(1.0f));
      for (int m = 1; m < {rows}; m++) *(__global float8*)(o_rows + m * {NV * DV} + h * {DV} + 8 * c) = BC(0.0f);
    }}
    for (int i = 0; i < {DV}; i += 8) *(__global float8*)(Cr + slot * {CP} + {VO} + h * {DV} + i) = *(__global float8*)(x + {VO} + h * {DV} + i);
    if (h % {REP} == 0) for (int i = 0; i < {DK}; i += 8) {{
      *(__global float8*)(Cr + slot * {CP} + {QO} + j * {DK} + i) = *(__global float8*)(x + {QO} + j * {DK} + i);
      *(__global float8*)(Cr + slot * {CP} + {KO} + j * {DK} + i) = *(__global float8*)(x + {KO} + j * {DK} + i);
    }}
  }}
}}"""

def gdn_step_src(NV, DK, DV):
  """One token of the gated delta rule for all NV heads, the state in place: S *= decay[h]; kv = k^T S; delta = (v - kv) beta;
  S += k (x) delta; o = q^T S. args: o [NV, DV], Sall [L, NV, DK, DV] (in / out: the layer's state at `idx[0]`), idx (int32 [1]),
  q, k [NV, DK], v [NV, DV], beta, decay [NV]. A unit = one head (its 128 x 128 fp32 state read once and written once)."""
  return V.FULL_H + f"""
__kernel void gdn_step(__global float* restrict o, __global float* restrict Sall, __global int* restrict idx, __global float* restrict q,
                       __global float* restrict k, __global float* restrict v, __global float* restrict beta, __global float* restrict decay, const int core_id) {{
  __global float* S = Sall + idx[0] * {NV * DK * DV};
  for (int h = core_id; h < {NV}; h += {NT}) {{
    __global float* Sh = S + h * {DK * DV}; float d = decay[h], b = beta[h];
    float8 kv[{DV // 8}]; for (int j = 0; j < {DV // 8}; j++) kv[j] = BC(0.0f);
    for (int i = 0; i < {DK}; i++) {{                          /* S *= d; kv += k_i * S_i (row i of the decayed state) */
      float8 kb = BC(k[h * {DK} + i]), db = BC(d); __global float* Sr = Sh + i * {DV};
      for (int j = 0; j < {DV // 8}; j++) {{ float8 s = *(__global float8*)(Sr + 8 * j) * db; *(__global float8*)(Sr + 8 * j) = s; kv[j] += kb * s; }}
    }}
    float8 delta[{DV // 8}]; for (int j = 0; j < {DV // 8}; j++) delta[j] = (*(__global float8*)(v + h * {DV} + 8 * j) - kv[j]) * BC(b);
    float8 ob[{DV // 8}]; for (int j = 0; j < {DV // 8}; j++) ob[j] = BC(0.0f);
    for (int i = 0; i < {DK}; i++) {{                          /* S_i += k_i delta; o += q_i S_i */
      float8 kb = BC(k[h * {DK} + i]), qb = BC(q[h * {DK} + i]); __global float* Sr = Sh + i * {DV};
      for (int j = 0; j < {DV // 8}; j++) {{ float8 s = *(__global float8*)(Sr + 8 * j) + kb * delta[j]; *(__global float8*)(Sr + 8 * j) = s; ob[j] += qb * s; }}
    }}
    for (int j = 0; j < {DV // 8}; j++) *(__global float8*)(o + h * {DV} + 8 * j) = ob[j];
  }}
}}"""

def gdn_prefill_src(NV, DK, DV, n):
  """The gated delta rule over n tokens for all NV heads, the state in place (from the given state): per token t and head h:
  S *= decay[t,h]; kv = k_t^T S; delta = (v_t - kv) beta[t,h]; S += k_t (x) delta; o_t = q_t^T S.
  args: o [n, NV, DV], S [NV, DK, DV] (in / out), q, k [n, NV, DK], v [n, NV, DV], beta, decay [n, NV]. A unit = one head."""
  return V.FULL_H + f"""
__kernel void gdn_prefill(__global float* restrict o, __global float* restrict S, __global float* restrict q, __global float* restrict k,
                          __global float* restrict v, __global float* restrict beta, __global float* restrict decay, const int core_id) {{
  for (int h = core_id; h < {NV}; h += {NT}) {{
    __global float* Sh = S + h * {DK * DV};
    for (int t = 0; t < {n}; t++) {{
      __global float* qt = q + (t * {NV} + h) * {DK}; __global float* kt = k + (t * {NV} + h) * {DK}; __global float* vt = v + (t * {NV} + h) * {DV};
      float d = decay[t * {NV} + h], b = beta[t * {NV} + h];
      float8 kv[{DV // 8}]; for (int j = 0; j < {DV // 8}; j++) kv[j] = BC(0.0f);
      for (int i = 0; i < {DK}; i++) {{
        float8 kb = BC(kt[i]), db = BC(d); __global float* Sr = Sh + i * {DV};
        for (int j = 0; j < {DV // 8}; j++) {{ float8 s = *(__global float8*)(Sr + 8 * j) * db; *(__global float8*)(Sr + 8 * j) = s; kv[j] += kb * s; }}
      }}
      float8 delta[{DV // 8}]; for (int j = 0; j < {DV // 8}; j++) delta[j] = (*(__global float8*)(vt + 8 * j) - kv[j]) * BC(b);
      float8 ob[{DV // 8}]; for (int j = 0; j < {DV // 8}; j++) ob[j] = BC(0.0f);
      for (int i = 0; i < {DK}; i++) {{
        float8 kb = BC(kt[i]), qb = BC(qt[i]); __global float* Sr = Sh + i * {DV};
        for (int j = 0; j < {DV // 8}; j++) {{ float8 s = *(__global float8*)(Sr + 8 * j) + kb * delta[j]; *(__global float8*)(Sr + 8 * j) = s; ob[j] += qb * s; }}
      }}
      for (int j = 0; j < {DV // 8}; j++) *(__global float8*)(o + (t * {NV} + h) * {DV} + 8 * j) = ob[j];
    }}
  }}
}}"""

def attn_decode_src(NH, NKV, HD, TMAX):
  """One query row per head against the layer's K / V cache: the new token's k / v rows are written at `pos` (by the tasks whose
  index is a kv head), scores over t <= pos, softmax, the weighted V sum. args: o [NH, HD], q [NH, HD], Kall / Vall
  [L, TMAX, NKV, HD], knew / vnew [NKV, HD], idx (int32 [2]: the layer, pos). A unit = one head; the scores live in LSRAM."""
  g = NH // NKV; scale = 1.0 / HD ** 0.5; LOG2E = 1.4426950408889634
  return V.FULL_H + EXP + f"""
__kernel void attn_decode(__global float* restrict o, __global float* restrict q, __global float* restrict Kall, __global float* restrict Vall,
                          __global float* restrict knew, __global float* restrict vnew, __global int* restrict idx, const int core_id) {{
  int L = idx[0], pos = idx[1];
  __global float* Kc = Kall + (L * {TMAX}) * {NKV * HD}; __global float* Vc = Vall + (L * {TMAX}) * {NKV * HD};
  if (core_id < {NKV}) {{                                          /* the new token's rows into the cache (this task's kv head) */
    for (int i = 0; i < {HD}; i += 8) {{
      *(__global float8*)(Kc + (pos * {NKV} + core_id) * {HD} + i) = *(__global float8*)(knew + core_id * {HD} + i);
      *(__global float8*)(Vc + (pos * {NKV} + core_id) * {HD} + i) = *(__global float8*)(vnew + core_id * {HD} + i);
    }}
  }}
  __global float* sc = LSF(0);
  for (int h = core_id; h < {NH}; h += {NT}) {{
    int kvh = h / {g}; __global float* qh = q + h * {HD}; float m = -1e30f;
    for (int t = 0; t <= pos; t++) {{
      __global float* kr = (t == pos) ? knew + kvh * {HD} : Kc + (t * {NKV} + kvh) * {HD};
      float8 acc = BC(0.0f);
      for (int i = 0; i < {HD}; i += 8) acc += *(__global float8*)(qh + i) * *(__global float8*)(kr + i);
      float sv = hsum8(acc) * {scale!r}f; sc[t] = sv; if (sv > m) m = sv;
    }}
    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {{ float8 e = exp2_d4(BC((sc[t] - m) * {LOG2E!r}f)); float ev = e[0]; sc[t] = ev; sum += ev; }}
    float8 ob[{HD // 8}]; for (int j = 0; j < {HD // 8}; j++) ob[j] = BC(0.0f);
    for (int t = 0; t <= pos; t++) {{
      __global float* vr = (t == pos) ? vnew + kvh * {HD} : Vc + (t * {NKV} + kvh) * {HD}; float8 pb = BC(sc[t]);
      for (int j = 0; j < {HD // 8}; j++) ob[j] += pb * *(__global float8*)(vr + 8 * j);
    }}
    float inv = 1.0f / sum;
    for (int j = 0; j < {HD // 8}; j++) *(__global float8*)(o + h * {HD} + 8 * j) = ob[j] * BC(inv);
  }}
}}"""

def attn_dec2_src(NH, NKV, HD, TMAX, ROT, eps):
  """Decode attention for one token, from the projections' rows to the gated output: q / k RMS-normalised ((1 + w), the layer's
  weights from per-layer stacks) and rotated on the first ROT dims (cos | sin from a per-position table), the new k / v written
  into the layer's cache at pos (tasks < NKV), scores over t <= pos, softmax, the weighted V sum, x sigmoid(gate) -> row 0 of
  o_rows. args: o_rows [rows, NH*HD], q_rows [rows, NH, 2*HD] (q | gate per head, row 0 read), kv_rows [rows, 2*NKV*HD]
  (k | v, row 0), qnw / knw [L, HD] (1 + w), rope [TMAX, 2*ROT] (cos | sin), Kall / Vall [L, TMAX, NKV, HD], idx (int32 [2]:
  the layer, pos). A unit = one head; q, k and the scores live in LSRAM."""
  g = NH // NKV; scale = 1.0 / HD ** 0.5; LOG2E = 1.4426950408889634; HR = ROT // 2
  assert HD % 8 == 0 and ROT % 8 == 0 and TMAX + 2 * HD <= 8000
  return V.FULL_H + EXP + f"""
static inline __attribute__((always_inline)) void norm_rope(__global float* restrict dst, __global float* restrict src, __global float* restrict w1, __global float* restrict cs) {{
  float8 acc = BC(0.0f);
  for (int i = 0; i < {HD}; i += 8) {{ float8 v = *(__global float8*)(src + i); acc += v * v; }}
  float8 inv = BC(1.0f / __builtin_sqrtf(hsum8(acc) * {1.0 / HD!r}f + {eps!r}f));
  for (int i = 0; i < {HD}; i += 8) *(__global float8*)(dst + i) = *(__global float8*)(src + i) * inv * *(__global float8*)(w1 + i);
  /* RoPE on the first ROT dims, in registers (no stack array: this compiler has scheduled a stack reload before its store):
     y[j] = x[j] cos[j] - x[j + HR] sin[j] (j < HR); y[j] = x[j] cos[j] + x[j - HR] sin[j] (j >= HR) */
  float8 x[{ROT // 8}];
  for (int v = 0; v < {ROT // 8}; v++) x[v] = *(__global float8*)(dst + 8 * v);
  for (int v = 0; v < {HR // 8}; v++) {{
    *(__global float8*)(dst + 8 * v) = x[v] * *(__global float8*)(cs + 8 * v) - x[v + {HR // 8}] * *(__global float8*)(cs + {ROT} + 8 * v);
    *(__global float8*)(dst + {HR} + 8 * v) = x[v + {HR // 8}] * *(__global float8*)(cs + {HR} + 8 * v) + x[v] * *(__global float8*)(cs + {ROT} + {HR} + 8 * v);
  }}
}}
static inline __attribute__((always_inline)) float dot256(__global float* restrict a, __global float* restrict b) {{
  float8 acc = BC(0.0f);
  for (int i = 0; i < {HD}; i += 8) acc += *(__global float8*)(a + i) * *(__global float8*)(b + i);
  return hsum8(acc);
}}
__kernel void attn_dec2(__global float* restrict o_rows, __global float* restrict q_rows, __global float* restrict kv_rows,
                        __global float* restrict qnw_all, __global float* restrict knw_all, __global float* restrict rope,
                        __global float* restrict Kall, __global float* restrict Vall, __global int* restrict idx, const int core_id) {{
  int L = idx[0], pos = idx[1];
  __global float* Kc = Kall + (L * {TMAX}) * {NKV * HD}; __global float* Vc = Vall + (L * {TMAX}) * {NKV * HD};
  __global float* qnw = qnw_all + L * {HD}; __global float* knw = knw_all + L * {HD}; __global float* cs = rope + pos * {2 * ROT};
  __global float* qs = LSF(0); __global float* ks = qs + {HD}; __global float* sc = ks + {HD};
  if (core_id < {NKV}) {{                                          /* the new token's k (normed, rotated) and v into the cache */
    norm_rope(ks, kv_rows + core_id * {HD}, knw, cs);
    for (int i = 0; i < {HD}; i += 8) {{
      *(__global float8*)(Kc + (pos * {NKV} + core_id) * {HD} + i) = *(__global float8*)(ks + i);
      *(__global float8*)(Vc + (pos * {NKV} + core_id) * {HD} + i) = *(__global float8*)(kv_rows + {NKV * HD} + core_id * {HD} + i);
    }}
  }}
  for (int h = core_id; h < {NH}; h += {NT}) {{
    int kvh = h / {g};
    norm_rope(qs, q_rows + h * {2 * HD}, qnw, cs);
    norm_rope(ks, kv_rows + kvh * {HD}, knw, cs);                   /* the new k, locally (another task may be writing it) */
    float m = -1e30f;
    for (int t = 0; t <= pos; t++) {{
      float sv = dot256(qs, t == pos ? ks : Kc + (t * {NKV} + kvh) * {HD}) * {scale!r}f; sc[t] = sv; if (sv > m) m = sv;
    }}
    float sum = 0.0f;
    for (int t = 0; t <= pos; t++) {{ float8 e = exp2_d4(BC((sc[t] - m) * {LOG2E!r}f)); sc[t] = e[0]; sum += e[0]; }}
    float8 ob[{HD // 8}]; for (int j = 0; j < {HD // 8}; j++) ob[j] = BC(0.0f);
    for (int t = 0; t <= pos; t++) {{
      __global float* vr = t == pos ? kv_rows + {NKV * HD} + kvh * {HD} : Vc + (t * {NKV} + kvh) * {HD}; float8 pb = BC(sc[t]);
      for (int j = 0; j < {HD // 8}; j++) ob[j] += pb * *(__global float8*)(vr + 8 * j);
    }}
    float8 inv = BC(1.0f / sum);
    for (int j = 0; j < {HD // 8}; j++) {{
      float8 gt = *(__global float8*)(q_rows + h * {2 * HD} + {HD} + 8 * j);
      float8 sg = VRCP(BC(1.0f) + exp2_d4(BC({-LOG2E!r}f) * gt));
      *(__global float8*)(o_rows + h * {HD} + 8 * j) = ob[j] * inv * sg;
    }}
  }}
}}"""

def gdn_lsr_src(NV, DK, DV, n, W=16, diag=None):
  """The gated delta rule over n tokens with the state in LSRAM. Each value column of a head's state is
  independent, so the state moves in blocks of W columns (DK x W fp32, one strided DMA of 64 B rows) that stay in LSRAM for
  all n tokens: S *= decay; kv = k^T S; delta = (v - kv) beta; S += k (x) delta; o = q^T S. A task owns NV / NT whole heads
  (their q / k / v stay in its cache); blocks are double-buffered (fill flags 0 / 1, drain flags 2 / 3).
  args: o [n, NV, DV], Sall [L, NV, DK, DV] (in / out: the layer at idx[0]), idx (int32 [1]), q, k [n, NV, DK], v [n, NV, DV],
  beta, decay [n, NV], desc (slot 0: a block: DK rows of W floats, DDR pitch DV floats)."""
  assert NV % NT == 0 and DV % W == 0 and W == 16 and 2 * DK * W * 4 <= 32768 and diag in (None, "dma", "compute")
  HPT, NB, BLK = NV // NT, DV // W, DK * W * 4
  src = V.FULL_H + f"""
__kernel void gdn_lsr(__global float* restrict o, __global float* restrict Sall, __global int* restrict idx, __global float* restrict q,
                      __global float* restrict k, __global float* restrict v, __global float* restrict beta, __global float* restrict decay,
                      __global int* restrict desc, const int core_id) {{
  __global float* S = Sall + idx[0] * {NV * DK * DV};
  int u0 = core_id * {HPT * NB}, nu = {HPT * NB};                  /* this task's units: heads [core_id * HPT, +HPT) x NB blocks */
  #define UBASE(u) (S + ((u) / {NB}) * {DK * DV} + ((u) % {NB}) * {W})
  DMA_FILL(0, DESC(desc, 0), 0, (int)UBASE(u0));
  for (int e = 0; e < nu; e++) {{
    int u = u0 + e, p = e & 1, h = u / {NB}, j0 = (u % {NB}) * {W};
    if (e + 1 < nu) {{                                               /* the next block into the other buffer, once its drain is done */
      if (p == 0) {{ DMA_WAIT(3); DMA_FILL(1, DESC(desc, 0), {BLK}, (int)UBASE(u + 1)); }}
      else        {{ DMA_WAIT(2); DMA_FILL(0, DESC(desc, 0), 0, (int)UBASE(u + 1)); }}
    }}
    if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
    __global float* B = LSF(p * {BLK});
    for (int t = 0; t < {n}; t++) {{
      __global float* kt = k + (t * {NV} + h) * {DK}; __global float* qt = q + (t * {NV} + h) * {DK};
      float8 d = BC(decay[t * {NV} + h]), b = BC(beta[t * {NV} + h]);
      float8 kv0 = BC(0.0f), kv1 = BC(0.0f);
      for (int i = 0; i < {DK}; i++) {{                            /* S *= d; kv += k_i S_i */
        float8 kb = BC(kt[i]); float8 s0 = *(__global float8*)(B + i * {W}) * d, s1 = *(__global float8*)(B + i * {W} + 8) * d;
        *(__global float8*)(B + i * {W}) = s0; *(__global float8*)(B + i * {W} + 8) = s1; kv0 += kb * s0; kv1 += kb * s1;
      }}
      __global float* vt = v + (t * {NV} + h) * {DV} + j0;
      float8 dl0 = (*(__global float8*)(vt) - kv0) * b, dl1 = (*(__global float8*)(vt + 8) - kv1) * b;
      float8 o0 = BC(0.0f), o1 = BC(0.0f);
      for (int i = 0; i < {DK}; i++) {{                            /* S_i += k_i delta; o += q_i S_i */
        float8 kb = BC(kt[i]), qb = BC(qt[i]);
        float8 s0 = *(__global float8*)(B + i * {W}) + kb * dl0, s1 = *(__global float8*)(B + i * {W} + 8) + kb * dl1;
        *(__global float8*)(B + i * {W}) = s0; *(__global float8*)(B + i * {W} + 8) = s1; o0 += qb * s0; o1 += qb * s1;
      }}
      __global float* ot = o + (t * {NV} + h) * {DV} + j0;
      *(__global float8*)(ot) = o0; *(__global float8*)(ot + 8) = o1;
    }}
    if (p == 0) DMA_DRAIN(2, DESC(desc, 0), 0, (int)UBASE(u)); else DMA_DRAIN(3, DESC(desc, 0), {BLK}, (int)UBASE(u));
  }}
  DMA_WAIT_ALL();
  #undef UBASE
}}"""
  head, body = src[:len(V.FULL_H)], src[len(V.FULL_H):]                      # diagnostics rewrite the kernel, never the header's macros
  if diag == "compute": body = body.replace("DMA_FILL(", "if (0) DMA_FILL(").replace("DMA_DRAIN(", "if (0) DMA_DRAIN(")
  if diag == "dma": body = body.replace("for (int t = 0; t < " + str(n) + "; t++)", "for (int t = 0; t < 0; t++)")
  return head + body

def gdn_lsr_desc(DK, DV, W=16):
  """slot 0: one state block, DK rows of W floats (64 B), DDR pitch DV floats, LSRAM contiguous."""
  return V._desc_slots((DK * W * 4, W * 4, DV * 4, W * 4))

def gdn_tok2_src(NV, NK, DK, DV, C, CONV, eps, W=16):
  """gdn_tok with the state streamed through LSRAM: a task owns NV / NT contiguous value heads; for each, the conv ring /
  SiLU / l2 norms put q, k, v in LSRAM (and the new raw row in the ring), the state goes by in blocks of W columns (double-
  buffered strided DMA, as gdn_lsr), and the gated RMSNorm x silu(z) of the head's DV outputs lands in row 0 of o_rows.
  args: as gdn_tok, then desc (gdn_lsr_desc). Rows 1.. of o_rows are not written (zero from allocation)."""
  REP = NV // NK; QO, KO, VO = 0, NK * DK, 2 * NK * DK; CP = ring_pitch(C); assert VO + NV * DV == C and DK == DV and NV % NT == 0 and W == 16
  HPT, NB, BLK = NV // NT, DV // W, DK * W * 4; QS = 2 * BLK
  return V.FULL_H + EXP + f"""
static inline __attribute__((always_inline)) void conv_silu(__global float* restrict x, __global float* restrict Cr, __global float* restrict cwt, int pos, int off, __global float* restrict out) {{
  for (int i = 0; i < {DK}; i += 8) {{
    float8 acc = *(__global float8*)(x + off + i) * *(__global float8*)(cwt + {(CONV - 1) * CP} + off + i);
    for (int t = 0; t < {CONV - 1}; t++) {{
      int tok = pos - {CONV - 1} + t;
      if (tok >= 0) acc += *(__global float8*)(Cr + (tok % {CONV}) * {CP} + off + i) * *(__global float8*)(cwt + t * {CP} + off + i);
    }}
    *(__global float8*)(out + i) = sw(acc, BC(1.0f));
  }}
}}
static inline __attribute__((always_inline)) void l2n(__global float* restrict v, float scale) {{
  float8 s = BC(0.0f); for (int i = 0; i < {DK}; i += 8) {{ float8 a = *(__global float8*)(v + i); s += a * a; }}
  float8 inv = BC(scale / __builtin_sqrtf(hsum8(s) + 1e-6f)); for (int i = 0; i < {DK}; i += 8) *(__global float8*)(v + i) = *(__global float8*)(v + i) * inv;
}}
__kernel void gdn_tok2(__global float* restrict o_rows, __global float* restrict Sall, __global float* restrict Call, __global int* restrict idx,
                       __global int* restrict posb, __global float* restrict x, __global float* restrict z, __global float* restrict bd,
                       __global float* restrict cwt, __global float* restrict nw, __global int* restrict desc, const int core_id) {{
  int L = idx[0], pos = posb[0]; __global float* S = Sall + L * {NV * DK * DV}; __global float* Cr = Call + L * {CONV * CP}; int slot = pos % {CONV};
  cwt += L * {CONV * CP}; nw += L * {DV};
  __global float* qs = LSF({QS}); __global float* ks = qs + {DK}; __global float* vs = ks + {DK}; __global float* oh = vs + {DV};
  int u0 = core_id * {HPT * NB}, nu = {HPT * NB};
  #define UBASE(u) (S + ((u) / {NB}) * {DK * DV} + ((u) % {NB}) * {W})
  DMA_FILL(0, DESC(desc, 0), 0, (int)UBASE(u0));
  float d = 0.0f, b = 0.0f;
  for (int e = 0; e < nu; e++) {{
    int u = u0 + e, p = e & 1, h = u / {NB}, jb = u % {NB}, j0 = jb * {W}, j = h / {REP};
    if (jb == 0) {{                                                 /* the head's q, k, v (conv, SiLU, l2 norms) and its ring rows */
      b = bd[h]; d = bd[{NV} + h];
      conv_silu(x, Cr, cwt, pos, {QO} + j * {DK}, qs); l2n(qs, {1.0 / DK ** 0.5!r}f);
      conv_silu(x, Cr, cwt, pos, {KO} + j * {DK}, ks); l2n(ks, 1.0f);
      conv_silu(x, Cr, cwt, pos, {VO} + h * {DV}, vs);
      for (int i = 0; i < {DV}; i += 8) *(__global float8*)(Cr + slot * {CP} + {VO} + h * {DV} + i) = *(__global float8*)(x + {VO} + h * {DV} + i);
      if (h % {REP} == 0) for (int i = 0; i < {DK}; i += 8) {{
        *(__global float8*)(Cr + slot * {CP} + {QO} + j * {DK} + i) = *(__global float8*)(x + {QO} + j * {DK} + i);
        *(__global float8*)(Cr + slot * {CP} + {KO} + j * {DK} + i) = *(__global float8*)(x + {KO} + j * {DK} + i);
      }}
    }}
    if (e + 1 < nu) {{
      if (p == 0) {{ DMA_WAIT(3); DMA_FILL(1, DESC(desc, 0), {BLK}, (int)UBASE(u + 1)); }}
      else        {{ DMA_WAIT(2); DMA_FILL(0, DESC(desc, 0), 0, (int)UBASE(u + 1)); }}
    }}
    if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
    __global float* B = LSF(p * {BLK});
    float8 db = BC(d), kv0 = BC(0.0f), kv1 = BC(0.0f);
    for (int i = 0; i < {DK}; i++) {{                              /* S *= d; kv += k_i S_i */
      float8 kb = BC(ks[i]); float8 s0 = *(__global float8*)(B + i * {W}) * db, s1 = *(__global float8*)(B + i * {W} + 8) * db;
      *(__global float8*)(B + i * {W}) = s0; *(__global float8*)(B + i * {W} + 8) = s1; kv0 += kb * s0; kv1 += kb * s1;
    }}
    float8 dl0 = (*(__global float8*)(vs + j0) - kv0) * BC(b), dl1 = (*(__global float8*)(vs + j0 + 8) - kv1) * BC(b);
    float8 o0 = BC(0.0f), o1 = BC(0.0f);
    for (int i = 0; i < {DK}; i++) {{                              /* S_i += k_i delta; o += q_i S_i */
      float8 kb = BC(ks[i]), qb = BC(qs[i]);
      float8 s0 = *(__global float8*)(B + i * {W}) + kb * dl0, s1 = *(__global float8*)(B + i * {W} + 8) + kb * dl1;
      *(__global float8*)(B + i * {W}) = s0; *(__global float8*)(B + i * {W} + 8) = s1; o0 += qb * s0; o1 += qb * s1;
    }}
    *(__global float8*)(oh + j0) = o0; *(__global float8*)(oh + j0 + 8) = o1;
    if (p == 0) DMA_DRAIN(2, DESC(desc, 0), 0, (int)UBASE(u)); else DMA_DRAIN(3, DESC(desc, 0), {BLK}, (int)UBASE(u));
    if (jb == {NB - 1}) {{                                          /* o = rmsnorm(o) * nw * silu(z) -> row 0 */
      float8 ss = BC(0.0f); for (int c = 0; c < {DV}; c += 8) {{ float8 ov = *(__global float8*)(oh + c); ss += ov * ov; }}
      float8 inv = BC(1.0f / __builtin_sqrtf(hsum8(ss) * {1.0 / DV!r}f + {eps!r}f));
      for (int c = 0; c < {DV}; c += 8)
        *(__global float8*)(o_rows + h * {DV} + c) = *(__global float8*)(oh + c) * inv * *(__global float8*)(nw + c) * sw(*(__global float8*)(z + h * {DV} + c), BC(1.0f));
    }}
  }}
  DMA_WAIT_ALL();
  #undef UBASE
}}"""

def gdn_hpt(NV):
  """Value heads a task in the DeltaNet token kernels: ceil(NV / NT), so every TEC that can have a head has one (Ornith's 32: 3 a
  task on 11 tasks, the last 2; the 27B's 48: 4 on all 12). The a|b rows are packed per task, padded to HPT heads (Layers.ab3)."""
  return -(-NV // NT)

def gdn_tok3_desc(DK, DV, H, HPT, KB=128, W=16):
  """slot 0: a state block (as gdn_lsr_desc); 1: the input row (H floats); 2: a chunk of a task's a|b rows (2 HPT rows of KB floats,
  DDR pitch H floats, LSRAM contiguous)."""
  return V._desc_slots((DK * W * 4, W * 4, DV * 4, W * 4), (H * 4,), (2 * HPT * KB * 4, KB * 4, H * 4, KB * 4))

def gdn_tok3_src(NV, NK, DK, DV, C, CONV, H, nrb, eps, W=16, KB=128, rows_out=False):
  """gdn_tok2 with its neighbours folded in: q | k | v and z are read from the GEMMs' C tiles (row 0) instead of unpacked
  rows; the a | b projection (the input row RMS-normalised, the norm weight folded into the rows), beta = sigmoid(b) and decay =
  exp(-exp(A_log) softplus(a + dt_bias)) are computed per task for its own heads (the input row and the task's 2 HPT a|b rows
  DMA'd through LSRAM in chunks of KB columns, double-buffered); the gated-norm output goes straight into o_proj's compact A layout
  (row 0 of each 4-row tile, rows 1-3 zero). args: out (half, compact A of K = NV DV), Sall, Call, idx, posb, qkv C tiles, z C tiles,
  the input rows, wab [L][NT][2 HPT][H], adt [L][2][NV] (-exp(A_log), dt_bias), cwt, nw, desc (gdn_tok3_desc).
  `rows_out` (a rotated-input model: o_proj's A is had_a32's, which needs whole 1024-blocks of K across heads): out is fp32 rows
  [rows, NV DV] and the gated-norm output goes to row 0 as fp32; the source is otherwise unchanged (False: byte-identical)."""
  src = _gdn_tok3_src(NV, NK, DK, DV, C, CONV, H, nrb, eps, W, KB)
  if not rows_out: return src
  a = "      __global half* oa = out + h * %d;" % (DV // 128 * 512); b = "      }\n    }\n  }\n  DMA_WAIT_ALL();"
  i, j = src.index(a), src.index(b); assert src.count(a) == 1 and src.count("__global half* restrict out") == 1
  store = (f"      for (int c = 0; c < {DV}; c += 8) *(__global float8*)(out + h * {DV} + c) = *(__global float8*)(oh + c) * inv8 * "
           f"*(__global float8*)(nw + c) * sw(LD8(z, h * {DV} + c), BC(1.0f));\n")
  return (src[:i] + store + src[j + len("      }\n"):]).replace("__global half* restrict out", "__global float* restrict out")

def _gdn_tok3_src(NV, NK, DK, DV, C, CONV, H, nrb, eps, W=16, KB=128):
  REP = NV // NK; QO, KO, VO = 0, NK * DK, 2 * NK * DK; CP = ring_pitch(C); assert VO + NV * DV == C and DK == DV == 128 and W == 16
  HPT, NB, BLK = gdn_hpt(NV), DV // W, DK * W * 4; NTU = -(-NV // HPT); LH = NV - (NTU - 1) * HPT   # HPT heads a task, the last LH
  QS = 2 * BLK; assert H % KB == 0
  XB, WCH, NCH = H * 4, 2 * HPT * KB * 4, H // KB
  assert XB + 2 * WCH <= 32768 - 64 and QS + 4 * DK * 4 <= 32768 - 64   # the a|b phase's buffers, then the state phase's (never live together)
  wacc = "".join(f" a{r} += xv * *(__global float8*)(WB + {r * KB} + i);" for r in range(2 * HPT))
  return V.FULL_H + TILE + EXP + f"""
#define CTA(c) ((((c) / 48) * {nrb * 3} + ((c) % 48) / 16) * 192 + (((c) % 16) / 4) * 16)
#define LD8(ct, c) F8(*(__global float4*)((ct) + CTA(c)), *(__global float4*)((ct) + CTA(c) + 16))
static inline __attribute__((always_inline)) float ex1(float t) {{ return hsum8(exp2_d4(BC(t * 1.4426950408889634f))) * 0.125f; }}
static inline __attribute__((always_inline)) float softplus1(float t) {{          /* max(t, 0) + log1p(exp(-|t|)), log1p = 2 atanh */
  float at = t < 0.0f ? -t : t; float y = ex1(-at); float s = y / (2.0f + y), s2 = s * s;
  float l = 2.0f * s * (1.0f + s2 * ({1/3!r}f + s2 * (0.2f + s2 * ({1/7!r}f + s2 * ({1/9!r}f + s2 * ({1/11!r}f + s2 * {1/13!r}f))))));
  return (t > 0.0f ? t : 0.0f) + l;
}}
static inline __attribute__((always_inline)) void conv_silu(__global float* restrict ct, __global float* restrict Cr, __global float* restrict cwt, int pos, int off, __global float* restrict out) {{
  for (int i = 0; i < {DK}; i += 8) {{
    float8 acc = LD8(ct, off + i) * *(__global float8*)(cwt + {(CONV - 1) * CP} + off + i);
    for (int t = 0; t < {CONV - 1}; t++) {{
      int tok = pos - {CONV - 1} + t;
      if (tok >= 0) acc += *(__global float8*)(Cr + (tok % {CONV}) * {CP} + off + i) * *(__global float8*)(cwt + t * {CP} + off + i);
    }}
    *(__global float8*)(out + i) = sw(acc, BC(1.0f));
  }}
}}
static inline __attribute__((always_inline)) void l2n(__global float* restrict v, float scale) {{
  float8 s = BC(0.0f); for (int i = 0; i < {DK}; i += 8) {{ float8 a = *(__global float8*)(v + i); s += a * a; }}
  float8 inv = BC(scale / __builtin_sqrtf(hsum8(s) + 1e-6f)); for (int i = 0; i < {DK}; i += 8) *(__global float8*)(v + i) = *(__global float8*)(v + i) * inv;
}}
__kernel void gdn_tok3(__global half* restrict out, __global float* restrict Sall, __global float* restrict Call, __global int* restrict idx,
                       __global int* restrict posb, __global float* restrict x, __global float* restrict z, __global float* restrict xin,
                       __global float* restrict wab, __global float* restrict adt, __global float* restrict cwt, __global float* restrict nw,
                       __global int* restrict desc, const int core_id) {{
{"  if (core_id >= %d) return;                                       /* NTU tasks own the heads */\n" % NTU if NTU < NT else ""}  int L = idx[0], pos = posb[0]; __global float* S = Sall + L * {NV * DK * DV}; __global float* Cr = Call + L * {CONV * CP}; int slot = pos % {CONV};
  cwt += L * {CONV * CP}; nw += L * {DV}; adt += L * {2 * NV}; int h0 = core_id * {HPT}, nh = core_id == {NTU - 1} ? {LH} : {HPT};
  /* ---- a | b of the task's heads: the input row and the a|b rows through LSRAM */
  __global float* wt = wab + L * {NTU * 2 * HPT * H} + core_id * {2 * HPT * H};
  DMA_FILL(2, DESC(desc, 1), 0, (int)xin);
  DMA_FILL(0, DESC(desc, 2), {XB}, (int)wt);
  float8 sq = BC(0.0f), {", ".join(f"a{r} = BC(0.0f)" for r in range(2 * HPT))};
  DMA_WAIT(2);
  for (int c = 0; c < {NCH}; c++) {{
    int p = c & 1;
    if (c + 1 < {NCH}) {{ if (p == 0) DMA_FILL(1, DESC(desc, 2), {XB + WCH}, (int)(wt + (c + 1) * {KB})); else DMA_FILL(0, DESC(desc, 2), {XB}, (int)(wt + (c + 1) * {KB})); }}
    if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
    __global float* WB = LSF({XB} + p * {WCH}); __global float* xc = LSF(0) + c * {KB};
    for (int i = 0; i < {KB}; i += 8) {{ float8 xv = *(__global float8*)(xc + i); sq += xv * xv;{wacc} }}
  }}
  float inv = 1.0f / __builtin_sqrtf(hsum8(sq) * {1.0 / H!r}f + {eps!r}f);
  float av[{HPT}] = {{{", ".join(f"hsum8(a{r}) * inv" for r in range(HPT))}}};
  float bv[{HPT}] = {{{", ".join(f"hsum8(a{HPT + r}) * inv" for r in range(HPT))}}};
  float bb[{HPT}], dd[{HPT}];
  for (int r = 0; r < nh; r++) {{
    bb[r] = 1.0f / (1.0f + ex1(-bv[r]));
    dd[r] = ex1(adt[h0 + r] * softplus1(av[r] + adt[{NV} + h0 + r]));
  }}
  /* ---- the delta rule, as gdn_tok2 */
  __global float* qs = LSF({QS}); __global float* ks = qs + {DK}; __global float* vs = ks + {DK}; __global float* oh = vs + {DV};
  int u0 = core_id * {HPT * NB}, nu = nh * {NB};
  #define UBASE(u) (S + ((u) / {NB}) * {DK * DV} + ((u) % {NB}) * {W})
  DMA_FILL(0, DESC(desc, 0), 0, (int)UBASE(u0));
  float d = 0.0f, b = 0.0f;
  for (int e = 0; e < nu; e++) {{
    int u = u0 + e, p = e & 1, h = u / {NB}, jb = u % {NB}, j0 = jb * {W}, j = h / {REP};
    if (jb == 0) {{                                                 /* the head's q, k, v (conv, SiLU, l2 norms) and its ring rows */
      b = bb[h - h0]; d = dd[h - h0];
      conv_silu(x, Cr, cwt, pos, {QO} + j * {DK}, qs); l2n(qs, {1.0 / DK ** 0.5!r}f);
      conv_silu(x, Cr, cwt, pos, {KO} + j * {DK}, ks); l2n(ks, 1.0f);
      conv_silu(x, Cr, cwt, pos, {VO} + h * {DV}, vs);
      for (int i = 0; i < {DV}; i += 8) *(__global float8*)(Cr + slot * {CP} + {VO} + h * {DV} + i) = LD8(x, {VO} + h * {DV} + i);
      if (h % {REP} == 0) for (int i = 0; i < {DK}; i += 8) {{
        *(__global float8*)(Cr + slot * {CP} + {QO} + j * {DK} + i) = LD8(x, {QO} + j * {DK} + i);
        *(__global float8*)(Cr + slot * {CP} + {KO} + j * {DK} + i) = LD8(x, {KO} + j * {DK} + i);
      }}
    }}
    if (e + 1 < nu) {{
      if (p == 0) {{ DMA_WAIT(3); DMA_FILL(1, DESC(desc, 0), {BLK}, (int)UBASE(u + 1)); }}
      else        {{ DMA_WAIT(2); DMA_FILL(0, DESC(desc, 0), 0, (int)UBASE(u + 1)); }}
    }}
    if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
    __global float* B = LSF(p * {BLK});
    float8 db = BC(d), kv0 = BC(0.0f), kv1 = BC(0.0f);
    for (int i = 0; i < {DK}; i++) {{                              /* S *= d; kv += k_i S_i */
      float8 kb = BC(ks[i]); float8 s0 = *(__global float8*)(B + i * {W}) * db, s1 = *(__global float8*)(B + i * {W} + 8) * db;
      *(__global float8*)(B + i * {W}) = s0; *(__global float8*)(B + i * {W} + 8) = s1; kv0 += kb * s0; kv1 += kb * s1;
    }}
    float8 dl0 = (*(__global float8*)(vs + j0) - kv0) * BC(b), dl1 = (*(__global float8*)(vs + j0 + 8) - kv1) * BC(b);
    float8 o0 = BC(0.0f), o1 = BC(0.0f);
    for (int i = 0; i < {DK}; i++) {{                              /* S_i += k_i delta; o += q_i S_i */
      float8 kb = BC(ks[i]), qb = BC(qs[i]);
      float8 s0 = *(__global float8*)(B + i * {W}) + kb * dl0, s1 = *(__global float8*)(B + i * {W} + 8) + kb * dl1;
      *(__global float8*)(B + i * {W}) = s0; *(__global float8*)(B + i * {W} + 8) = s1; o0 += qb * s0; o1 += qb * s1;
    }}
    *(__global float8*)(oh + j0) = o0; *(__global float8*)(oh + j0 + 8) = o1;
    if (p == 0) DMA_DRAIN(2, DESC(desc, 0), 0, (int)UBASE(u)); else DMA_DRAIN(3, DESC(desc, 0), {BLK}, (int)UBASE(u));
    if (jb == {NB - 1}) {{                                          /* o = rmsnorm(o) * nw * silu(z) -> o_proj's compact A, row 0 */
      float8 ss = BC(0.0f); for (int c = 0; c < {DV}; c += 8) {{ float8 ov = *(__global float8*)(oh + c); ss += ov * ov; }}
      float8 inv8 = BC(1.0f / __builtin_sqrtf(hsum8(ss) * {1.0 / DV!r}f + {eps!r}f)), Z = BC(0.0f);
      __global half* oa = out + h * {DV // 128 * 512};                  /* head h = K-slice h (DV = 128): 32 k-quads of 16 halves */
      for (int c = 0; c < {DV}; c += 8) {{
        float8 v = *(__global float8*)(oh + c) * inv8 * *(__global float8*)(nw + c) * sw(LD8(z, h * {DV} + c), BC(1.0f));
        *(__global half16*)(oa + (c / 4) * 16) = CVT16(__builtin_shufflevector(v, Z, 0, 1, 2, 3, 8, 9, 10, 11), Z);
        *(__global half16*)(oa + (c / 4 + 1) * 16) = CVT16(__builtin_shufflevector(v, Z, 4, 5, 6, 7, 8, 9, 10, 11), Z);
      }}
    }}
  }}
  DMA_WAIT_ALL();
  #undef UBASE
}}"""

GDN_WAVE = 6                                                                # gdn_tokm: tokens a wave (M > GDN_WAVE: gdn_tokm_waves_src)
def gdn_prep():
  """QWEN_GDN_PREP (default "dma"): gdn_tokm_src's preparation -- "dma" / "dma-lsarr" the DMA-staged ones,
  "fast" the cached-access preparation from registers (all M tokens a chunk), "cached" the per-token cached-access preparation
  ("none" or the empty string: "fast", the spelling before the two were one switch; kept for one release)."""
  v = os.environ.get("QWEN_GDN_PREP", "dma").strip(); v = "fast" if v in ("", "none") else v
  assert v in ("dma", "dma-lsarr", "fast", "cached"), f"QWEN_GDN_PREP={v!r}: dma | dma-lsarr | fast | cached"
  return v

def gdn_prep_key(v):
  """The kernel-key fields of preparation `v`: (fast, dma) as the keys spelled them when fast / cached was a switch of its own
  ("1|dma", "1|dma-lsarr", "1|" fast, "0|" cached), so a cached kernel keeps its key and no two preparations share one."""
  return f"{'0' if v == 'cached' else '1'}|{v if v.startswith('dma') else ''}"

def gdn_tokm_desc(DK, DV, H, HPT, M, KB=128, W=16, C=None, CONV=4, tree=False):
  """slot 0: a state block (as gdn_lsr_desc); 1: a chunk of the M input rows (M rows of KB floats, DDR pitch H floats); 2: a
  chunk of a task's a|b rows (as gdn_tok3_desc). With C (QWEN_GDN_PREP=dma; always for M > GDN_WAVE) also 3: the CONV rows of
  one 128-channel segment of the conv taps or the ring (DDR pitch ring_pitch(C), LSRAM contiguous); 4: the M raw rows of a
  segment (the same pitches; M > GDN_WAVE: a full wave's GDN_WAVE rows); 5: a head's block of o_proj's compact A (contiguous
  halves); M > GDN_WAVE: 6: the last wave's raw rows of a segment."""
  sl = [(DK * W * 4, W * 4, DV * 4, W * 4), (M * KB * 4, KB * 4, H * 4, KB * 4), (2 * HPT * KB * 4, KB * 4, H * 4, KB * 4)]
  assert C is not None or M <= GDN_WAVE, "gdn_tokm in waves: the DMA-staged preparation's slots"
  if C is not None: sl += [(CONV * DK * 4, DK * 4, ring_pitch(C) * 4, DK * 4), (min(M, GDN_WAVE) * DK * 4, DK * 4, ring_pitch(C) * 4, DK * 4), (DV // 128 * 512 * -(-M // 4) * 2,)]
  if M > GDN_WAVE: sl += [((1 if tree else M - (-(-M // GDN_WAVE) - 1) * GDN_WAVE) * DK * 4, DK * 4, ring_pitch(C) * 4, DK * 4)]   # (tree: a rescue wave's one row)
  return V._desc_slots(*sl)

GD_SLOT = 272                                                               # gdn_tokm(defer): floats a (head, token) slot: k 128 | v 128 | beta, decay, pad

def gdn_defer_desc(M, MS):
  """gdn_tokm(defer)'s slots 6 / 7, after gdn_tokm_desc's 0..5: a head's MS token slots in (contiguous); the M new k (or v) rows
  out (512 B each, DDR pitch one slot)."""
  return V._desc_slots((MS * GD_SLOT * 4,), (M * 512, 512, GD_SLOT * 4, 512))

def gdn_defer_src(src, NV, M, MS, DK=128, W=16):
  """gdn_tokm (rows_out, the DMA preparation, the chain kernel) with the commit deferred to the next verify pass: no per-token
  banks and no state write here; per (head, token) the update's inputs (the normed k, v, beta, decay: GD_SLOT floats) go to `banks`
  ([NG][NV][MS] slots, the head's task its only reader and writer), and each state block first takes the PREVIOUS pass's ap =
  accp[0] accepted updates (the same loops, the same order: the state after them == the old bank ap - 1 bit for bit) and is written
  back once -- the committed state; the M new tokens then run in LSRAM only (their outputs). gdn_commit then moves only the ring rows."""
  assert M <= MS <= 4 and DK == 128 and W == 16
  QS = 2 * DK * W * 4; KT0, VT0 = QS + M * DK * 4, QS + 2 * M * DK * 4; PV0 = QS + 4 * M * DK * 4; assert PV0 + MS * GD_SLOT * 4 <= 32768 - 64
  BLK = DK * W * 4; SL = MS * GD_SLOT
  edits = [
    ("__global float* restrict banks, __global float* restrict rawm, const int core_id) {\n  int L = idx[0], pos = posb[0];",
     "__global float* restrict banks, __global float* restrict rawm, __global int* restrict accp, const int core_id) {\n  int L = idx[0], pos = posb[0], ap = accp[0];"),
    ("      DMA_WAIT(fc); DMA_WAIT(fa);                                     /* the staging buffer's drain (the previous block); the previous head's output */\n",
     "      DMA_WAIT(fc); DMA_WAIT(fa);                                     /* the staging buffer's drain (the previous block); the previous head's output */\n"
     f"      DMA_FILL(fa, DESC(desc, 6), {PV0}, (int)(banks + (L * {NV} + h) * {SL})); DMA_WAIT(fa);   /* defer: the previous pass's update inputs of head h */\n"),
    (f"      if (qk) for (int t = 0; t < {M}; t++) {{ l2n(QT(t), {1.0 / DK ** 0.5!r}f); l2n(KT(t), 1.0f); }}\n      DMA_WAIT(fc);\n    }}\n",
     f"      if (qk) for (int t = 0; t < {M}; t++) {{ l2n(QT(t), {1.0 / DK ** 0.5!r}f); l2n(KT(t), 1.0f); }}\n      DMA_WAIT(fc);\n"
     f"      {{ __global float* kh = banks + (L * {NV} + h) * {SL};                 /* defer: this pass's k, v, beta, decay of head h */\n"
     f"        DMA_DRAIN(fc, DESC(desc, 7), {KT0}, (int)kh); DMA_WAIT(fc); DMA_DRAIN(fc, DESC(desc, 7), {VT0}, (int)(kh + 128)); DMA_WAIT(fc);\n"
     f"        for (int t = 0; t < {M}; t++) {{ kh[t * {GD_SLOT} + 256] = bb[t][h - h0]; kh[t * {GD_SLOT} + 257] = dd[t][h - h0]; }} }}\n    }}\n"),
    (f"    for (int t = 0; t < {M}; t++) {{\n      __global float* ks = KT(t); __global float* qs = QT(t);\n",
     f"""    if (ap > 0) {{                                                    /* defer: the previous pass's accepted updates, then the state back once */
      __global float* PV = LSF({PV0});
      for (int t = 0; t < ap; t++) {{
        __global float* ks = PV + t * {GD_SLOT};
        float8 db = BC(PV[t * {GD_SLOT} + 257]), kv0 = BC(0.0f), kv1 = BC(0.0f);
        for (int i = 0; i < {DK}; i++) {{
          float8 kb = BC(ks[i]); float8 s0 = *(__global float8*)(B + i * {W}) * db, s1 = *(__global float8*)(B + i * {W} + 8) * db;
          *(__global float8*)(B + i * {W}) = s0; *(__global float8*)(B + i * {W} + 8) = s1; kv0 += kb * s0; kv1 += kb * s1;
        }}
        float8 bt = BC(PV[t * {GD_SLOT} + 256]);
        float8 dl0 = (*(__global float8*)(PV + t * {GD_SLOT} + 128 + j0) - kv0) * bt, dl1 = (*(__global float8*)(PV + t * {GD_SLOT} + 128 + j0 + 8) - kv1) * bt;
        for (int i = 0; i < {DK}; i++) {{
          float8 kb = BC(ks[i]);
          float8 s0 = *(__global float8*)(B + i * {W}) + kb * dl0, s1 = *(__global float8*)(B + i * {W} + 8) + kb * dl1;
          *(__global float8*)(B + i * {W}) = s0; *(__global float8*)(B + i * {W} + 8) = s1;
        }}
      }}
      if (p == 0) {{ DMA_DRAIN(2, DESC(desc, 0), 0, (int)UBASE(u)); DMA_WAIT(2); }} else {{ DMA_DRAIN(3, DESC(desc, 0), {BLK}, (int)UBASE(u)); DMA_WAIT(3); }}
    }}
    for (int t = 0; t < {M}; t++) {{
      __global float* ks = KT(t); __global float* qs = QT(t);
"""),
    (f"""      if (t < {M - 1}) {{                                             /* the rollback point: the state after t + 1 tokens */
        if (p == 0) {{ DMA_DRAIN(2, DESC(desc, 0), 0, (int)BANK(t, u)); DMA_WAIT(2); }}
        else        {{ DMA_DRAIN(3, DESC(desc, 0), {BLK}, (int)BANK(t, u)); DMA_WAIT(3); }}
      }}
""", ""),
    (f"    if (p == 0) DMA_DRAIN(2, DESC(desc, 0), 0, (int)UBASE(u)); else DMA_DRAIN(3, DESC(desc, 0), {BLK}, (int)UBASE(u));\n", ""),
  ]
  for a_, b_ in edits:
    assert src.count(a_) == 1, a_
    src = src.replace(a_, b_)
  return src

def gdn_tokm_src(NV, NK, DK, DV, C, CONV, H, nrb, eps, M, NG, W=16, KB=128, diag=(), rows_out=False):
  """gdn_tokm (_gdn_tokm_src; M > GDN_WAVE: gdn_tokm_waves_src). `rows_out` (a rotated-input model: o_proj's A is had_a32's, which
  needs whole 1024-blocks of K across heads -- as gdn_tok3_src's): out is fp32 rows [rows, NV DV] and token t's gated-norm output
  goes to row t as fp32 (the expression the A store rounds to fp16, unchanged: fp16 of these rows == the default kernel's A),
  stored by the task owning the head (512 B a row, whole lines); the A block's LSRAM staging and its drain are dropped. The rest
  of the source is unchanged (False: byte-identical)."""
  src = _gdn_tokm_src(NV, NK, DK, DV, C, CONV, H, nrb, eps, M, NG, W, KB, diag)
  if not rows_out: return src
  assert src.count("__global half* restrict out") == 1 and src.count("__global half* oa = ") == 1
  i = src.index("      __global half* oa = ")
  if M > GDN_WAVE:              # the waves' kernel: wave w's rows t0 .. t0 + mw - 1; the stage's drain (the last wave) goes too
    a = "DMA_DRAIN(p, DESC(desc, 5), "; j = src.index("\n", src.index(a, i)) + 1
    row, rz, vt, n = "(t0 + t)", "t0 + t", "t", "mw"
  else:                         # the chain kernel, any QWEN_GDN_PREP: the block up to the end of `if (jb == NB - 1)`
    j = src.index("    }\n  }\n  DMA_WAIT", i); row, rz, vt, n = "t", "t", "t", str(M)
  store = (f"      for (int t = 0; t < {n}; t++) for (int c = 0; c < {DV}; c += 8)\n"
           f"        *(__global float8*)(out + {row} * {NV * DV} + h * {DV} + c) = *(__global float8*)(OT({vt}) + c) * BC(invt[{vt}]) * "
           f"*(__global float8*)(nw + c) * sw(LD8R(z, h * {DV} + c, {rz}), BC(1.0f));\n")
  return (src[:i] + store + src[j:]).replace("__global half* restrict out", "__global float* restrict out")

def _gdn_tokm_src(NV, NK, DK, DV, C, CONV, H, nrb, eps, M, NG, W=16, KB=128, diag=()):
  """gdn_tok3 for M consecutive tokens (speculative decoding's verify pass): rows 0..M-1 of the qkv / z C tiles and of the input
  rows are tokens at positions pos .. pos + M - 1. Each state block is streamed through LSRAM once and takes the M updates in
  order; after token t < M - 1 the block is also drained to bank t (the state after t + 1 tokens: the rollback point), after the
  last to the state itself. The conv ring is READ only (tasks share its rows; a write of the new rows would race the readers):
  token t's conv takes its earlier rows from the C tiles and the ring for positions < pos; the M raw rows go to `rawm` and
  gdn_commit writes the accepted ones. The gated-norm outputs of the M tokens go to rows 0..M-1 of o_proj's compact A.
  args: as gdn_tok3, then banks [M-1][NG][NV*DK*DV], rawm [NG][M][CP] (CP = ring_pitch(C), as the ring).
  `diag` (timing only, wrong results): "noab" skips the a|b phase, "noprep" the heads' q / k / v preparation, "nobank" the
  per-token bank drains, "nobankwait" their waits, "nostate" the state loops.
  M > GDN_WAVE: gdn_tokm_waves_src (the tokens in waves; its own source, the DMA-staged preparation whatever QWEN_GDN_PREP)."""
  if M > GDN_WAVE: return gdn_tokm_waves_src(NV, NK, DK, DV, C, CONV, H, nrb, eps, M, NG, W, KB, diag)
  D = set(diag)
  REP = NV // NK; QO, KO, VO = 0, NK * DK, 2 * NK * DK; CP = ring_pitch(C); assert VO + NV * DV == C and DK == DV == 128 and W == 16
  HPT, NB, BLK = gdn_hpt(NV), DV // W, DK * W * 4; NTU = -(-NV // HPT); LH = NV - (NTU - 1) * HPT   # HPT heads a task, the last LH
  assert H % KB == 0 and 1 <= M <= 7
  RT = -(-M // 4)                                                         # the compact A's row tiles (rows 4.. in the second)
  XSZ, WSZ, NCH = M * KB * 4, 2 * HPT * KB * 4, H // KB; XB, WB0 = 0, 2 * M * KB * 4
  QS = 2 * BLK; assert WB0 + 2 * WSZ <= QS and QS + 4 * M * DK * 4 <= 32768 - 64
  SZ = NV * DK * DV
  wacc = "".join(f" a{r} += xv * *(__global float8*)(WB + {r * KB} + i);" for r in range(2 * HPT))
  # the heads' q / k / v preparation from registers (all M tokens a chunk; 0.70 -> 0.30 ms, bit-identical); QWEN_GDN_PREP=cached:
  # the per-token preparation. Its burst of strided loads and write-back stores HUNG a TEC (a job reported as EXCEPTION after the
  # KMD's 30 s deadline) while the ring, conv-tap and raw-row buffers had row pitches of a multiple of 8 KiB (all rows of a buffer
  # in one L2 set) and one particular placement mod 8 KiB; the row pitch ring_pitch(C) = C + 16 removed it.
  PREPV = gdn_prep(); FAST = PREPV != "cached"
  PREP = (f"""    if ({"0" if "noprep" in D else "jb == 0"}) {{                      /* q / k once per key head on this task (consecutive heads share it) */
      if (h == h0 || h % {REP} == 0) {{
        conv_silu_all(x, Cr, cwt, pos, {QO} + j * {DK}, QT(0), raw, h % {REP} == 0);
        conv_silu_all(x, Cr, cwt, pos, {KO} + j * {DK}, KT(0), raw, h % {REP} == 0);
        for (int t = 0; t < {M}; t++) {{ l2n(QT(t), {1.0 / DK ** 0.5!r}f); l2n(KT(t), 1.0f); }}
      }}
      conv_silu_all(x, Cr, cwt, pos, {VO} + h * {DV}, VT(0), raw, 1);
    }}
""") if FAST else (f"""    if (jb == 0) {{
      for (int t = 0; t < {M}; t++) {{
        conv_silu_m(x, Cr, cwt, pos, {QO} + j * {DK}, t, QT(t)); l2n(QT(t), {1.0 / DK ** 0.5!r}f);
        conv_silu_m(x, Cr, cwt, pos, {KO} + j * {DK}, t, KT(t)); l2n(KT(t), 1.0f);
        conv_silu_m(x, Cr, cwt, pos, {VO} + h * {DV}, t, VT(t));
        for (int i = 0; i < {DV}; i += 8) *(__global float8*)(raw + t * {CP} + {VO} + h * {DV} + i) = LD8R(x, {VO} + h * {DV} + i, t);
        if (h % {REP} == 0) for (int i = 0; i < {DK}; i += 8) {{
          *(__global float8*)(raw + t * {CP} + {QO} + j * {DK} + i) = LD8R(x, {QO} + j * {DK} + i, t);
          *(__global float8*)(raw + t * {CP} + {KO} + j * {DK} + i) = LD8R(x, {KO} + j * {DK} + i, t);
        }}
      }}
    }}
""")
  # QWEN_GDN_PREP=dma (the default; fast / cached: the cached-access preparations above): the fast preparation with the conv taps and the ring rows DMA'd into LSRAM and the raw rows drained
  # from it, so the burst makes no cached access of those three buffers (only the C tiles' rows are still loaded through the
  # cache), and the small per-row arrays (the a|b sums, beta, decay) in LSRAM instead of the stack; the kernel ends in per-flag
  # waits instead of DMA_WAIT_ALL. Staging: the idle state buffer (1 - p; its drain, flag 3 - p, waited first) as two areas of
  # 2 CONV rows of DK floats, a segment's CONV tap rows then its CONV ring slots; the raw rows (the C tiles' rows, copied once)
  # in the OT rows (dead between a head's output and its next block). The next segment of the head is filled under the current
  # one's math (flags 1 - p and 2 + p: the state fill in flight holds flag p), the raw rows drained on flag 3 - p. The conv is one
  # small loop a token over LSRAM rows (the fast path's all-tokens-from-registers form spills ~85 vector registers a chunk to the
  # stack at M = 7, and the stack frames of a core's four TECs alias in L2 at the 8 KiB stack pitch). Same arithmetic in the same
  # order: bit-identical.
  # QWEN_GDN_PREP=dma-lsarr: also the a|b sums / sq, beta / decay and the output norms' factors (stack arrays in the default, some
  # loop-indexed) in LSRAM, every store at a compile-time offset (the loops writing them unrolled here); the hot loop's reads of
  # beta / decay are LSRAM loads at a (token, head) offset. Plain `dma` keeps those arrays exactly as the default declares them.
  DMAP, LSARR = PREPV.startswith("dma"), PREPV == "dma-lsarr"
  if DMAP:
    SEG = CONV * DK * 4; AREA = 2 * SEG; TAIL = QS + 4 * M * DK * 4; RAW0 = QS + 3 * M * DK * 4   # LSRAM bytes (RAW0: OT(0))
    ACC0 = TAIL; SQ0 = ACC0 + M * 2 * HPT * 4; BB0 = SQ0 + M * 4; DD0 = BB0 + M * HPT * 4; INV0 = DD0 + M * HPT * 4   # dma-lsarr's arrays
    LEND = INV0 + M * 4 if LSARR else TAIL
    VT0, OSZ = QS + 2 * M * DK * 4, DV // 128 * 512 * RT * 2                       # the head's output (OSZ bytes of halves) staged in VT
    assert 2 * AREA <= BLK and LEND <= 32768 - 64 and OSZ <= M * DK * 4
    PREP = f"""    if ({"0" if "noprep" in D else "jb == 0"}) {{                      /* QWEN_GDN_PREP=dma: the segments' taps / ring rows in by DMA, raw rows out */
      int fa = 1 - p, fb = 2 + p, fc = 3 - p, sb = (1 - p) * {BLK}, qk = h == h0 || h % {REP} == 0, ns = qk ? 3 : 1;
      int on = qk ? {QO} + j * {DK} : {VO} + h * {DV};
      DMA_WAIT(fc); DMA_WAIT(fa);                                     /* the staging buffer's drain (the previous block); the previous head's output */
      DMA_FILL(fa, DESC(desc, 3), sb, (int)(cwt + on)); DMA_FILL(fb, DESC(desc, 3), sb + {SEG}, (int)(Cr + on));
      for (int s = 0; s < ns; s++) {{                                 /* segments: q, k, v (qk) or v */
        int a = sb + (s & 1) * {AREA}, off = on, v = s == ns - 1;
        DMA_WAIT(fa); DMA_WAIT(fb); DMA_WAIT(fc);                     /* this segment's rows landed; segment s - 1's raw rows drained */
        if (s + 1 < ns) {{                                            /* the next segment's rows under this one's math */
          int an = sb + ((s + 1) & 1) * {AREA}; on = s == 0 ? {KO} + j * {DK} : {VO} + h * {DV};
          DMA_FILL(fa, DESC(desc, 3), an, (int)(cwt + on)); DMA_FILL(fb, DESC(desc, 3), an + {SEG}, (int)(Cr + on));
        }}
        conv_seg(x, LSF(a), OT(0), pos, off, QKV + (v ? {2 * M} : s * {M}) * {DK});
        if (v || h % {REP} == 0) DMA_DRAIN(fc, DESC(desc, 4), {RAW0}, (int)(raw + off));
      }}
      if (qk) for (int t = 0; t < {M}; t++) {{ l2n(QT(t), {1.0 / DK ** 0.5!r}f); l2n(KT(t), 1.0f); }}
      DMA_WAIT(fc);
    }}
"""
    X = lambda r: f"R{r}" if r < CONV - 1 else f"(Rw + {(r - CONV + 1) * DK})"          # the conv window's row r (rows < CONV - 1: the ring)
    CSEG = (f"""/* QWEN_GDN_PREP=dma: conv_silu_all's sums from LSRAM rows, one small loop a token (no vector spills): A = a segment's tap rows
   0..{CONV - 1} then its ring slots {CONV}..{2 * CONV - 1}, {DK} floats a row; Rw = the M raw rows, the C tiles' rows copied first */
static inline __attribute__((always_inline)) void conv_seg(__global float* restrict ct, __global float* restrict A, __global float* restrict Rw, int pos, int off,
                                                           __global float* restrict out) {{
  for (int i = 0; i < {DK}; i += 8) {{""" + "".join(f" *(__global float8*)(Rw + {t * DK} + i) = LD8R(ct, off + i, {t});" for t in range(M)) + " }\n"
      + "".join(f"  int p{r} = pos - {CONV - 1 - r}; __global float* R{r} = A + ({CONV} + (p{r} >= 0 ? p{r} : 0) % {CONV}) * {DK};\n" for r in range(CONV - 1))
      + "".join(f"  for (int i = 0; i < {DK}; i += 8) {{ float8 acc = *(__global float8*)({X(t + CONV - 1)} + i) * *(__global float8*)(A + {(CONV - 1) * DK} + i);"
                + "".join((f" acc += *(__global float8*)({X(t + tt)} + i) * *(__global float8*)(A + {tt * DK} + i);") if t + tt >= CONV - 1 else
                          (f" if (pos + {t - (CONV - 1) + tt} >= 0) acc += *(__global float8*)({X(t + tt)} + i) * *(__global float8*)(A + {tt * DK} + i);") for tt in range(CONV - 1))
                + f" *(__global float8*)(out + {t * DK} + i) = sw(acc, BC(1.0f)); }}\n" for t in range(M))
      + "}\n")
  src = V.FULL_H + TILE + EXP + f"""
#define CTA(c) ((((c) / 48) * {nrb * 3} + ((c) % 48) / 16) * 192 + (((c) % 16) / 4) * 16)
#define LD8R(ct, c, r) F8(*(__global float4*)((ct) + CTA(c) + ((r) / 4) * 64 + 4 * ((r) % 4)), *(__global float4*)((ct) + CTA(c) + ((r) / 4) * 64 + 16 + 4 * ((r) % 4)))
#define LO4(v) __builtin_shufflevector((v), (v), 0, 1, 2, 3)
#define HI4(v) __builtin_shufflevector((v), (v), 4, 5, 6, 7)
static inline __attribute__((always_inline)) float ex1(float t) {{ return hsum8(exp2_d4(BC(t * 1.4426950408889634f))) * 0.125f; }}
static inline __attribute__((always_inline)) float softplus1(float t) {{
  float at = t < 0.0f ? -t : t; float y = ex1(-at); float s = y / (2.0f + y), s2 = s * s;
  float l = 2.0f * s * (1.0f + s2 * ({1/3!r}f + s2 * (0.2f + s2 * ({1/7!r}f + s2 * ({1/9!r}f + s2 * ({1/11!r}f + s2 * {1/13!r}f))))));
  return (t > 0.0f ? t : 0.0f) + l;
}}
/* the causal conv + SiLU of 128 channels at `off` for all M tokens (token t -> out + t * DK), an 8-channel chunk at a time: the
   CONV taps and the window's CONV - 1 + M rows (the ring for positions < pos, the C tiles' rows 0..M-1) loaded ONCE a chunk and
   every token computed from registers (the sums in conv_silu_m's order: tap CONV - 1 first); `raw`: also the M raw rows -> rawm */
static inline __attribute__((always_inline)) void conv_silu_all(__global float* restrict ct, __global float* restrict Cr, __global float* restrict cwt, int pos, int off,
                                                                __global float* restrict out, __global float* restrict raw, int wr) {{
  /* every register named, every index a constant (no stack arrays); the ring row of a position < 0 is read from a valid row and
     dropped (a select must not be the only guard of an address: the compiler may load it anyway) */
  for (int i = 0; i < {DK}; i += 8) {{
{"".join(f"    float8 w{tt} = *(__global float8*)(cwt + {tt * CP} + off + i);" + chr(10) for tt in range(CONV))}{"".join(f"    float8 x{r} = pos >= {CONV - 1 - r} ? *(__global float8*)(Cr + ((pos - {CONV - 1 - r}) % {CONV}) * {CP} + off + i) : BC(0.0f);" + chr(10) if False else f"    int p{r} = pos - {CONV - 1 - r}; float8 x{r} = *(__global float8*)(Cr + ((p{r} >= 0 ? p{r} : 0) % {CONV}) * {CP} + off + i); if (p{r} < 0) x{r} = BC(0.0f);" + chr(10) for r in range(CONV - 1))}{"".join(f"    float8 x{CONV - 1 + t} = LD8R(ct, off + i, {t});" + chr(10) for t in range(M))}{"".join(("    { float8 acc = x%d * w%d;" % (t + CONV - 1, CONV - 1)) + "".join((" acc += x%d * w%d;" % (t + tt, tt)) if t + tt >= CONV - 1 else (" if (pos + %d >= 0) acc += x%d * w%d;" % (t - (CONV - 1) + tt, t + tt, tt)) for tt in range(CONV - 1)) + (" *(__global float8*)(out + %d + i) = sw(acc, BC(1.0f)); if (wr) *(__global float8*)(raw + %d + off + i) = x%d; }" % (t * DK, t * CP, t + CONV - 1)) + chr(10) for t in range(M))}  }}
}}
static inline __attribute__((always_inline)) void conv_silu_m(__global float* restrict ct, __global float* restrict Cr, __global float* restrict cwt, int pos, int off, int t, __global float* restrict out) {{
  for (int i = 0; i < {DK}; i += 8) {{
    float8 acc = LD8R(ct, off + i, t) * *(__global float8*)(cwt + {(CONV - 1) * CP} + off + i);
    for (int tt = 0; tt < {CONV - 1}; tt++) {{
      int rel = t - {CONV - 1} + tt;
      if (rel >= 0) acc += LD8R(ct, off + i, rel) * *(__global float8*)(cwt + tt * {CP} + off + i);
      else if (pos + rel >= 0) acc += *(__global float8*)(Cr + ((pos + rel) % {CONV}) * {CP} + off + i) * *(__global float8*)(cwt + tt * {CP} + off + i);
    }}
    *(__global float8*)(out + i) = sw(acc, BC(1.0f));
  }}
}}
static inline __attribute__((always_inline)) void l2n(__global float* restrict v, float scale) {{
  float8 s = BC(0.0f); for (int i = 0; i < {DK}; i += 8) {{ float8 a = *(__global float8*)(v + i); s += a * a; }}
  float8 inv = BC(scale / __builtin_sqrtf(hsum8(s) + 1e-6f)); for (int i = 0; i < {DK}; i += 8) *(__global float8*)(v + i) = *(__global float8*)(v + i) * inv;
}}
__kernel void gdn_tokm(__global half* restrict out, __global float* restrict Sall, __global float* restrict Call, __global int* restrict idx,
                       __global int* restrict posb, __global float* restrict x, __global float* restrict z, __global float* restrict xin,
                       __global float* restrict wab, __global float* restrict adt, __global float* restrict cwt, __global float* restrict nw,
                       __global int* restrict desc, __global float* restrict banks, __global float* restrict rawm, const int core_id) {{
{"  if (core_id >= %d) return;                                       /* NTU tasks own the heads */\n" % NTU if NTU < NT else ""}  int L = idx[0], pos = posb[0]; __global float* S = Sall + L * {SZ}; __global float* Cr = Call + L * {CONV * CP};
  cwt += L * {CONV * CP}; nw += L * {DV}; adt += L * {2 * NV}; int h0 = core_id * {HPT}, nh = core_id == {NTU - 1} ? {LH} : {HPT};
  __global float* raw = rawm + L * {M * CP};
  /* ---- a | b of the task's heads for the M rows: chunks of the rows and of the task's a|b rows through LSRAM (x: flags 2/3, w: 0/1) */
  __global float* wt = wab + L * {NTU * 2 * HPT * H} + core_id * {2 * HPT * H};
  float acc[{M}][{2 * HPT}], sq[{M}];
  for (int r = 0; r < {M}; r++) {{ sq[r] = 0.0f; for (int q = 0; q < {2 * HPT}; q++) acc[r][q] = 0.0f; }}
  {"" if "noab" in D else "DMA_FILL(2, DESC(desc, 1), %d, (int)xin); DMA_FILL(0, DESC(desc, 2), %d, (int)wt);" % (XB, WB0)}
  for (int c = 0; c < {0 if "noab" in D else NCH}; c++) {{
    int p = c & 1;
    if (c + 1 < {NCH}) {{
      if (p == 0) {{ DMA_FILL(3, DESC(desc, 1), {XB + XSZ}, (int)(xin + (c + 1) * {KB})); DMA_FILL(1, DESC(desc, 2), {WB0 + WSZ}, (int)(wt + (c + 1) * {KB})); }}
      else        {{ DMA_FILL(2, DESC(desc, 1), {XB}, (int)(xin + (c + 1) * {KB})); DMA_FILL(0, DESC(desc, 2), {WB0}, (int)(wt + (c + 1) * {KB})); }}
    }}
    if (p == 0) {{ DMA_WAIT(0); DMA_WAIT(2); }} else {{ DMA_WAIT(1); DMA_WAIT(3); }}
    __global float* WB = LSF({WB0} + p * {WSZ});
    for (int r = 0; r < {M}; r++) {{
      __global float* xc = LSF({XB} + p * {XSZ}) + r * {KB};
      float8 sq8 = BC(0.0f), {", ".join(f"a{r_} = BC(0.0f)" for r_ in range(2 * HPT))};
      for (int i = 0; i < {KB}; i += 8) {{ float8 xv = *(__global float8*)(xc + i); sq8 += xv * xv;{wacc} }}
      sq[r] += hsum8(sq8); {" ".join(f"acc[r][{q}] += hsum8(a{q});" for q in range(2 * HPT))}
    }}
  }}
  float bb[{M}][{HPT}], dd[{M}][{HPT}];
  for (int r = 0; r < {M}; r++) {{
    float inv = 1.0f / __builtin_sqrtf(sq[r] * {1.0 / H!r}f + {eps!r}f);
    for (int q = 0; q < nh; q++) {{
      bb[r][q] = 1.0f / (1.0f + ex1(-acc[r][{HPT} + q] * inv));
      dd[r][q] = ex1(adt[h0 + q] * softplus1(acc[r][q] * inv + adt[{NV} + h0 + q]));
    }}
  }}
  /* ---- the delta rule: per head, the M tokens' q / k / v, then each state block takes the M updates */
  __global float* QKV = LSF({QS});
  #define QT(t) (QKV + (t) * {DK})
  #define KT(t) (QKV + ({M} + (t)) * {DK})
  #define VT(t) (QKV + ({2 * M} + (t)) * {DK})
  #define OT(t) (QKV + ({3 * M} + (t)) * {DK})
  int u0 = core_id * {HPT * NB}, nu = nh * {NB};
  #define UBASE(u) (S + ((u) / {NB}) * {DK * DV} + ((u) % {NB}) * {W})
  #define BANK(t, u) (banks + ((t) * {NG} + L) * {SZ} + ((u) / {NB}) * {DK * DV} + ((u) % {NB}) * {W})
  DMA_FILL(0, DESC(desc, 0), 0, (int)UBASE(u0));
  for (int e = 0; e < nu; e++) {{
    int u = u0 + e, p = e & 1, h = u / {NB}, jb = u % {NB}, j0 = jb * {W}, j = h / {REP};
{PREP}    if (e + 1 < nu) {{
      if (p == 0) {{ DMA_WAIT(3); DMA_FILL(1, DESC(desc, 0), {BLK}, (int)UBASE(u + 1)); }}
      else        {{ DMA_WAIT(2); DMA_FILL(0, DESC(desc, 0), 0, (int)UBASE(u + 1)); }}
    }}
    if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
    __global float* B = LSF(p * {BLK});
    for (int t = 0; t < {M}; t++) {{
      __global float* ks = KT(t); __global float* qs = QT(t);
      float8 db = BC(dd[t][h - h0]), kv0 = BC(0.0f), kv1 = BC(0.0f);
      for (int i = 0; i < {0 if "nostate" in D else DK}; i++) {{
        float8 kb = BC(ks[i]); float8 s0 = *(__global float8*)(B + i * {W}) * db, s1 = *(__global float8*)(B + i * {W} + 8) * db;
        *(__global float8*)(B + i * {W}) = s0; *(__global float8*)(B + i * {W} + 8) = s1; kv0 += kb * s0; kv1 += kb * s1;
      }}
      float8 bt = BC(bb[t][h - h0]);
      float8 dl0 = (*(__global float8*)(VT(t) + j0) - kv0) * bt, dl1 = (*(__global float8*)(VT(t) + j0 + 8) - kv1) * bt;
      float8 o0 = BC(0.0f), o1 = BC(0.0f);
      for (int i = 0; i < {0 if "nostate" in D else DK}; i++) {{
        float8 kb = BC(ks[i]), qb = BC(qs[i]);
        float8 s0 = *(__global float8*)(B + i * {W}) + kb * dl0, s1 = *(__global float8*)(B + i * {W} + 8) + kb * dl1;
        *(__global float8*)(B + i * {W}) = s0; *(__global float8*)(B + i * {W} + 8) = s1; o0 += qb * s0; o1 += qb * s1;
      }}
      *(__global float8*)(OT(t) + j0) = o0; *(__global float8*)(OT(t) + j0 + 8) = o1;
      if (t < {0 if "nobank" in D else M - 1}) {{                                             /* the rollback point: the state after t + 1 tokens */
        if (p == 0) {{ DMA_DRAIN(2, DESC(desc, 0), 0, (int)BANK(t, u)); {"" if "nobankwait" in D else "DMA_WAIT(2);"} }}
        else        {{ DMA_DRAIN(3, DESC(desc, 0), {BLK}, (int)BANK(t, u)); {"" if "nobankwait" in D else "DMA_WAIT(3);"} }}
      }}
    }}
    if (p == 0) DMA_DRAIN(2, DESC(desc, 0), 0, (int)UBASE(u)); else DMA_DRAIN(3, DESC(desc, 0), {BLK}, (int)UBASE(u));
    if (jb == {NB - 1}) {{                                          /* the M tokens' o = rmsnorm(o) * nw * silu(z) -> rows 0..M-1 */
      float invt[{M}];
      for (int t = 0; t < {M}; t++) {{
        float8 ss = BC(0.0f); for (int c = 0; c < {DV}; c += 8) {{ float8 ov = *(__global float8*)(OT(t) + c); ss += ov * ov; }}
        invt[t] = 1.0f / __builtin_sqrtf(hsum8(ss) * {1.0 / DV!r}f + {eps!r}f);
      }}
      __global half* oa = out + h * {DV // 128 * 512 * RT};              /* head h = K-slice h: [kq][RT row tiles][16 halves] */
      float8 Z = BC(0.0f);
      for (int c = 0; c < {DV}; c += 8) {{
        {" ".join(f"float8 v{t} = " + (f"*(__global float8*)(OT({t}) + c) * BC(invt[{t}]) * *(__global float8*)(nw + c) * sw(LD8R(z, h * {DV} + c, {t}), BC(1.0f));" if t < M else "Z;") for t in range(4 * RT))}
        {" ".join(f"*(__global half16*)(oa + ((c / 4) * {RT} + {i}) * 16) = CVT16(F8(LO4(v{4*i}), LO4(v{4*i+1})), F8(LO4(v{4*i+2}), LO4(v{4*i+3}))); "
                  f"*(__global half16*)(oa + ((c / 4 + 1) * {RT} + {i}) * 16) = CVT16(F8(HI4(v{4*i}), HI4(v{4*i+1})), F8(HI4(v{4*i+2}), HI4(v{4*i+3})));" for i in range(RT))}
      }}
    }}
  }}
  DMA_WAIT_ALL();
  #undef UBASE
  #undef BANK
}}"""
  if not FAST:                    # the per-token preparation only: leave the fast path's helper out (the source of the parent commit)
    a = src.index("/* the causal conv + SiLU of 128 channels"); b = src.index("static inline __attribute__((always_inline)) void conv_silu_m(")
    src = src[:a] + src[b:]
  if DMAP:                        # the default source is untouched; the variant's edits, each asserted to land exactly once
    VDEF = lambda t: f"float8 v{t} = " + (f"*(__global float8*)(OT({t}) + c) * BC(invt[{t}]) * *(__global float8*)(nw + c) * sw(LD8R(z, h * {DV} + c, {t}), BC(1.0f));" if t < M else "Z;")
    VST = lambda i: (f"*(__global half16*)(oa + ((c / 4) * {RT} + {i}) * 16) = CVT16(F8(LO4(v{4*i}), LO4(v{4*i+1})), F8(LO4(v{4*i+2}), LO4(v{4*i+3}))); "
                     f"*(__global half16*)(oa + ((c / 4 + 1) * {RT} + {i}) * 16) = CVT16(F8(HI4(v{4*i}), HI4(v{4*i+1})), F8(HI4(v{4*i+2}), HI4(v{4*i+3})));")
    OPROJ = (f"""      __global half* oa = out + h * {DV // 128 * 512 * RT};              /* head h = K-slice h: [kq][RT row tiles][16 halves] */
      float8 Z = BC(0.0f);
      for (int c = 0; c < {DV}; c += 8) {{
        {" ".join(VDEF(t) for t in range(4 * RT))}
        {" ".join(VST(i) for i in range(RT))}
      }}
""")
    OPROJ_DMA = (f"""      __global half* oa = LSH({VT0});                                /* head h's block of o_proj's A, staged in VT (dead now), drained below */
      float8 Z = BC(0.0f);
""" + "".join(f"""      for (int c = 0; c < {DV}; c += 8) {{
        {" ".join(VDEF(t) for t in range(4 * i, 4 * i + 4))}
        {VST(i)}
      }}
""" for i in range(RT)) + f"""      DMA_DRAIN(p, DESC(desc, 5), {VT0}, (int)(out + h * {DV // 128 * 512 * RT}));   /* flag p: free here; waited at the next head */
""")
    edits = [("  DMA_WAIT_ALL();\n  #undef UBASE", "  DMA_WAIT(0); DMA_WAIT(1); DMA_WAIT(2); DMA_WAIT(3);       /* per flag, not DMA_WAIT_ALL (DCache.md §8b) */\n  #undef UBASE"),
             ("static inline __attribute__((always_inline)) void l2n(", CSEG + "static inline __attribute__((always_inline)) void l2n("),
             (OPROJ, OPROJ_DMA)]
    if LSARR:                     # LSRAM floats: acc [M][2 HPT] at ACC0, sq [M] at SQ0, bb / dd [M][HPT] at BB0 / DD0, invt [M] at INV0
      A_ = lambda r, q: f"LSF({ACC0})[{r * 2 * HPT + q}]"; S_ = lambda r: f"LSF({SQ0})[{r}]"; B_ = lambda r, q: f"LSF({BB0})[{r * HPT + q}]"; D_ = lambda r, q: f"LSF({DD0})[{r * HPT + q}]"
      edits += [
        (f"""  float acc[{M}][{2 * HPT}], sq[{M}];
  for (int r = 0; r < {M}; r++) {{ sq[r] = 0.0f; for (int q = 0; q < {2 * HPT}; q++) acc[r][q] = 0.0f; }}
""", "  /* the a|b sums: LSRAM, compile-time offsets */" + "".join(f" {S_(r)} = 0.0f;" + "".join(f" {A_(r, q)} = 0.0f;" for q in range(2 * HPT)) for r in range(M)) + "\n"),
        (f"""    for (int r = 0; r < {M}; r++) {{
      __global float* xc = LSF({XB} + p * {XSZ}) + r * {KB};
      float8 sq8 = BC(0.0f), {", ".join(f"a{r_} = BC(0.0f)" for r_ in range(2 * HPT))};
      for (int i = 0; i < {KB}; i += 8) {{ float8 xv = *(__global float8*)(xc + i); sq8 += xv * xv;{wacc} }}
      sq[r] += hsum8(sq8); {" ".join(f"acc[r][{q}] += hsum8(a{q});" for q in range(2 * HPT))}
    }}
""", "".join(f"""    {{
      __global float* xc = LSF({XB} + p * {XSZ}) + {r * KB};
      float8 sq8 = BC(0.0f), {", ".join(f"a{r_} = BC(0.0f)" for r_ in range(2 * HPT))};
      for (int i = 0; i < {KB}; i += 8) {{ float8 xv = *(__global float8*)(xc + i); sq8 += xv * xv;{wacc} }}
      {S_(r)} += hsum8(sq8); {" ".join(f"{A_(r, q)} += hsum8(a{q});" for q in range(2 * HPT))}
    }}
""" for r in range(M))),
        (f"""  float bb[{M}][{HPT}], dd[{M}][{HPT}];
  for (int r = 0; r < {M}; r++) {{
    float inv = 1.0f / __builtin_sqrtf(sq[r] * {1.0 / H!r}f + {eps!r}f);
    for (int q = 0; q < nh; q++) {{
      bb[r][q] = 1.0f / (1.0f + ex1(-acc[r][{HPT} + q] * inv));
      dd[r][q] = ex1(adt[h0 + q] * softplus1(acc[r][q] * inv + adt[{NV} + h0 + q]));
    }}
  }}
""", "".join(f"""  {{ float inv = 1.0f / __builtin_sqrtf({S_(r)} * {1.0 / H!r}f + {eps!r}f);""" + "".join(f"""
    if ({q} < nh) {{ {B_(r, q)} = 1.0f / (1.0f + ex1(-{A_(r, HPT + q)} * inv)); {D_(r, q)} = ex1(adt[h0 + {q}] * softplus1({A_(r, q)} * inv + adt[{NV} + h0 + {q}])); }}""" for q in range(HPT)) + " }\n" for r in range(M))),
        ("float8 db = BC(dd[t][h - h0])", f"float8 db = BC(LSF({DD0})[t * {HPT} + h - h0])"),
        ("float8 bt = BC(bb[t][h - h0]);", f"float8 bt = BC(LSF({BB0})[t * {HPT} + h - h0]);"),
        (f"""      float invt[{M}];
      for (int t = 0; t < {M}; t++) {{
        float8 ss = BC(0.0f); for (int c = 0; c < {DV}; c += 8) {{ float8 ov = *(__global float8*)(OT(t) + c); ss += ov * ov; }}
        invt[t] = 1.0f / __builtin_sqrtf(hsum8(ss) * {1.0 / DV!r}f + {eps!r}f);
      }}
""", f"      __global float* invt = LSF({INV0});                                /* LSRAM, compile-time offsets */\n" + "".join(f"""      {{ float8 ss = BC(0.0f); for (int c = 0; c < {DV}; c += 8) {{ float8 ov = *(__global float8*)(OT({t}) + c); ss += ov * ov; }}
        invt[{t}] = 1.0f / __builtin_sqrtf(hsum8(ss) * {1.0 / DV!r}f + {eps!r}f); }}
""" for t in range(M)))]
    for a_, b_ in edits:
      assert src.count(a_) == 1, a_
      src = src.replace(a_, b_)
  return src

def gdn_tokm_waves_src(NV, NK, DK, DV, C, CONV, H, nrb, eps, M, NG, W=16, KB=128, diag=(), tree=False):
  """gdn_tokm for GDN_WAVE < M <= 12 tokens (the verify pass's wide geometries), in waves of GDN_WAVE tokens within one call
  (tokens 0..5, then 6..M-1): past 7 tokens the per-token q / k / v / o rows (4 M DK floats) and the two state buffers no longer
  fit LSRAM together. A task's units are (head, wave, state block), in that order. Per (head, wave) the wave's q / k / v are
  prepared -- always the DMA-staged preparation of QWEN_GDN_PREP=dma (conv taps / ring rows in by DMA, raw rows drained by DMA:
  no cached access of the ring, the taps or the raw rows; the C tiles' rows are the only cached loads) --, then each state block
  takes the wave's updates. Wave w > 0 starts a block from bank w GDN_WAVE - 1 (the state after the previous wave's last token:
  drained there as its rollback point and waited for, then DMA'd back in as the block's fill); only the last wave drains the
  block to the state itself. q / k are prepared at every (head, wave) (the next value head's wave 0 needs its key head's tokens
  0..5 again); a key head's raw rows are drained by the task owning head j REP only, as in gdn_tokm. Wave w > 0's window rows
  before its first token are the C tiles' rows (copied into the staging area's ring slots; the ring is not read). The head's
  block of o_proj's compact A is assembled in an LSRAM stage that lives across the waves (the row tile split between two waves
  merged there) and drained after the last wave (flag p, waited by the next head's preparation). The small per-row arrays as
  in QWEN_GDN_PREP=dma (acc / sq / bb / dd / invt); per-flag waits at the end, no DMA_WAIT_ALL. The arithmetic of each token is
  gdn_tokm's in the same order: bit-identical to two consecutive gdn_tokm calls (GDN_WAVE tokens, then the rest at pos +
  GDN_WAVE, the ring committed in between). LSRAM: the a|b phase [0, 8 M KB + 16 HPT KB); then two state buffers | q, k, v, o of
  a wave | the A stage (ends at 31 744 B at M = 12). args and `diag` ("noab", "noprep", "nobank", "nobankwait", "nostate"): as
  gdn_tokm_src.
  `tree` (QWEN_SPEC_TREE, spec_tree.py; the kernel `gdn_tokt`): rows 0..GDN_WAVE-1 are the chain (wave 0, as above) and each
  row r >= GDN_WAVE a RESCUE row -- a one-token wave started from the state bank of its parent, bank depth[r] - 1 (a 16th
  argument `tree`, spec_tree.table: int32 [2 M], depth first; depth[r] = its chain position j, 1 <= j <= GDN_WAVE - 1), whose
  conv window is the parent's: chain rows j-3..j-1 (the C tiles' rows), or the ring slots for positions < pos when j < 3 (the
  ring segment is DMA'd in for every wave). The rescue row's own raw row and bank are its own (row r): bank r for r < M - 1,
  and the last row's state goes to Sall as the last wave's (gdn_commit_tree copies the committed path's bank unless the path
  ends there). Under spec_tree.chain_table (depth[r] = r) the sums are the chain kernel's in the same order: bit-identical
  (the simulator gate). Without `tree` the source is the chain kernel's, unchanged."""
  D = set(diag)
  REP = NV // NK; QO, KO, VO = 0, NK * DK, 2 * NK * DK; CP = ring_pitch(C); assert VO + NV * DV == C and DK == DV == 128 and W == 16
  HPT, NB, BLK = gdn_hpt(NV), DV // W, DK * W * 4; NTU = -(-NV // HPT); LH = NV - (NTU - 1) * HPT   # HPT heads a task, the last LH
  MW = GDN_WAVE; NW = -(-M // MW); ML = M - (NW - 1) * MW                     # waves of MW tokens, the last ML
  if tree: NW, ML = 1 + (M - MW), 1                                            # the tree: wave 0 the chain, then a one-token wave a rescue row
  assert H % KB == 0 and MW < M <= 12 and CONV - 1 <= MW
  RT, SZ, LIM = -(-M // 4), NV * DK * DV, 32768 - 64
  XSZ, WSZ, NCH = M * KB * 4, 2 * HPT * KB * 4, H // KB; XB, WB0 = 0, 2 * M * KB * 4
  QS = 2 * BLK; SEG = CONV * DK * 4; AREA = 2 * SEG                            # LSRAM bytes
  RAW0, OST, OSZ = QS + 3 * MW * DK * 4, QS + 4 * MW * DK * 4, DV // 128 * 512 * RT * 2   # OT(0) (the raw rows' staging); the A stage
  assert WB0 + 2 * WSZ <= LIM and 2 * AREA <= BLK and OST + OSZ <= LIM, (WB0 + 2 * WSZ, OST + OSZ)
  wacc = "".join(f" a{r} += xv * *(__global float8*)(WB + {r * KB} + i);" for r in range(2 * HPT))
  def wave(w):                                                                 # (first token, tokens)
    if tree: return (0, MW) if w == 0 else (MW + w - 1, 1)
    return w * MW, (ML if w == NW - 1 else MW)
  def conv_fn(w):
    """the causal conv + SiLU of a 128-channel segment for wave w's tokens, from LSRAM rows (gdn_tokm's conv_seg): A = the tap
    rows 0..CONV-1, then CONV-1 window rows before the wave (w = 0: the ring slots, by position; w > 0: the C tiles' rows
    t0-CONV+1.. copied into A's slots CONV..); Rw = the wave's raw rows (the C tiles' rows t0.., copied first).
    A rescue wave (tree, w > 0): the window rows by the row's depth `dep` (a runtime argument): depth q = dep - CONV + 1 + r is
    the C tiles' row q when q >= 0 (copied into the raw-row staging's free rows Rw + (1 + r) DK: a one-token wave stages one raw
    row; A's ring slots stay the ring's, which the window reads for q < 0), else the ring slot of position pos + q (the term
    dropped when that position is < 0, wave 0's guard). Under the chain table (dep = t0) the sums are wave w's above, in the same order."""
    t0, mw = wave(w); X = lambda r: f"R{r}" if r < CONV - 1 else f"(Rw + {(r - CONV + 1) * DK})"; resc = bool(w and tree)
    cp = "".join(f" *(__global float8*)(Rw + {t * DK} + i) = LD8R(ct, off + i, {t0 + t});" for t in range(mw))
    if resc: cp += "".join(f" if (q{r} >= 0) *(__global float8*)(Rw + {(1 + r) * DK} + i) = LD8R(ct, off + i, q{r});" for r in range(CONV - 1))
    elif w: cp += "".join(f" *(__global float8*)(A + {(CONV + r) * DK} + i) = LD8R(ct, off + i, {t0 - CONV + 1 + r});" for r in range(CONV - 1))
    s = (f"static inline __attribute__((always_inline)) void conv_w{w}(__global float* restrict ct, __global float* restrict A, __global float* restrict Rw, int pos, int off,\n"
         f"                                                          __global float* restrict out{', int dep' if resc else ''}) {{\n")
    if resc: s += "".join(f"  int q{r} = dep - {CONV - 1 - r}; int p{r} = pos + q{r};\n" for r in range(CONV - 1))
    s += f"  for (int i = 0; i < {DK}; i += 8) {{{cp} }}\n"
    s += "".join((f"  int p{r} = pos - {CONV - 1 - r}; __global float* R{r} = A + ({CONV} + (p{r} >= 0 ? p{r} : 0) % {CONV}) * {DK};\n" if w == 0 else
                  f"  __global float* R{r} = q{r} >= 0 ? Rw + {(1 + r) * DK} : A + ({CONV} + (p{r} >= 0 ? p{r} : 0) % {CONV}) * {DK};\n" if resc else
                  f"  __global float* R{r} = A + {(CONV + r) * DK};\n") for r in range(CONV - 1))
    for t in range(mw):
      s += f"  for (int i = 0; i < {DK}; i += 8) {{ float8 acc = *(__global float8*)({X(t + CONV - 1)} + i) * *(__global float8*)(A + {(CONV - 1) * DK} + i);"
      for tt in range(CONV - 1):
        term = f"acc += *(__global float8*)({X(t + tt)} + i) * *(__global float8*)(A + {tt * DK} + i);"
        if resc: s += f" if (p{tt} >= 0) {term}"
        else: s += f" {term}" if (t + tt >= CONV - 1 or w) else f" if (pos + {t0 + t - (CONV - 1) + tt} >= 0) {term}"
      s += f" *(__global float8*)(out + {t * DK} + i) = sw(acc, BC(1.0f)); }}\n"
    return s + "}\n"
  def oproj(w):
    """wave w's rows of head h's A block into the stage: per row tile holding them, the 4 rows' gated-norm outputs (rows of
    other waves zero; rows of an earlier wave kept: the tile merged with the stage's)."""
    t0, mw = wave(w); s = ""
    for i in range(t0 // 4, (t0 + mw - 1) // 4 + 1):
      vd = " ".join(f"float8 v{q} = " + (f"*(__global float8*)(OT({4 * i + q - t0}) + c) * BC(invt[{4 * i + q - t0}]) * *(__global float8*)(nw + c) * sw(LD8R(z, h * {DV} + c, {4 * i + q}), BC(1.0f));"
                                        if t0 <= 4 * i + q < t0 + mw else "Z;") for q in range(4))
      k = max(0, t0 - 4 * i)                                               # the tile's first k rows are an earlier wave's
      def st(adr, part):
        new = f"CVT16(F8({part}(v0), {part}(v1)), F8({part}(v2), {part}(v3)))"
        if not k: return f"*(__global half16*)({adr}) = {new};"
        sel = ", ".join([str(x) for x in range(4 * k)] + [str(16 + x) for x in range(4 * k, 16)])
        return f"{{ __global half16* pt = (__global half16*)({adr}); *pt = __builtin_shufflevector(*pt, {new}, {sel}); }}"
      s += (f"        for (int c = 0; c < {DV}; c += 8) {{\n          {vd}\n          {st(f'oa + ((c / 4) * {RT} + {i}) * 16', 'LO4')} "
            f"{st(f'oa + ((c / 4 + 1) * {RT} + {i}) * 16', 'HI4')}\n        }}\n")
    return s
  per_wave = lambda f: " else ".join(f"if (w == {w}) {{ {f(w)} }}" for w in range(NW))
  return V.FULL_H + TILE + EXP + f"""
#define CTA(c) ((((c) / 48) * {nrb * 3} + ((c) % 48) / 16) * 192 + (((c) % 16) / 4) * 16)
#define LD8R(ct, c, r) F8(*(__global float4*)((ct) + CTA(c) + ((r) / 4) * 64 + 4 * ((r) % 4)), *(__global float4*)((ct) + CTA(c) + ((r) / 4) * 64 + 16 + 4 * ((r) % 4)))
#define LO4(v) __builtin_shufflevector((v), (v), 0, 1, 2, 3)
#define HI4(v) __builtin_shufflevector((v), (v), 4, 5, 6, 7)
static inline __attribute__((always_inline)) float ex1(float t) {{ return hsum8(exp2_d4(BC(t * 1.4426950408889634f))) * 0.125f; }}
static inline __attribute__((always_inline)) float softplus1(float t) {{
  float at = t < 0.0f ? -t : t; float y = ex1(-at); float s = y / (2.0f + y), s2 = s * s;
  float l = 2.0f * s * (1.0f + s2 * ({1/3!r}f + s2 * (0.2f + s2 * ({1/7!r}f + s2 * ({1/9!r}f + s2 * ({1/11!r}f + s2 * {1/13!r}f))))));
  return (t > 0.0f ? t : 0.0f) + l;
}}
{"".join(conv_fn(w) for w in range(NW))}static inline __attribute__((always_inline)) void l2n(__global float* restrict v, float scale) {{
  float8 s = BC(0.0f); for (int i = 0; i < {DK}; i += 8) {{ float8 a = *(__global float8*)(v + i); s += a * a; }}
  float8 inv = BC(scale / __builtin_sqrtf(hsum8(s) + 1e-6f)); for (int i = 0; i < {DK}; i += 8) *(__global float8*)(v + i) = *(__global float8*)(v + i) * inv;
}}
__kernel void {"gdn_tokt" if tree else "gdn_tokm"}(__global half* restrict out, __global float* restrict Sall, __global float* restrict Call, __global int* restrict idx,
                       __global int* restrict posb, __global float* restrict x, __global float* restrict z, __global float* restrict xin,
                       __global float* restrict wab, __global float* restrict adt, __global float* restrict cwt, __global float* restrict nw,
                       __global int* restrict desc, __global float* restrict banks, __global float* restrict rawm, {"__global int* restrict tree, " if tree else ""}const int core_id) {{
{"  if (core_id >= %d) return;                                       /* NTU tasks own the heads */\n" % NTU if NTU < NT else ""}  int L = idx[0], pos = posb[0]; __global float* S = Sall + L * {SZ}; __global float* Cr = Call + L * {CONV * CP};
  cwt += L * {CONV * CP}; nw += L * {DV}; adt += L * {2 * NV}; int h0 = core_id * {HPT}, nh = core_id == {NTU - 1} ? {LH} : {HPT};
  __global float* raw = rawm + L * {M * CP};
{"".join(f"  int d{w} = tree[{MW + w - 1}];" for w in range(1, NW)) + "                     /* the rescue rows' depths (chain positions): wave w = row MW + w - 1 */" + chr(10) if tree else ""}{"  #define WDEP(w) (" + "".join(f"(w) == {w} ? d{w} : " for w in range(1, NW - 1)) + f"d{NW - 1})" + chr(10) if tree else ""}  /* ---- a | b of the task's heads for the M rows (all waves at once): chunks of the rows and of the task's a|b rows through LSRAM */
  __global float* wt = wab + L * {NTU * 2 * HPT * H} + core_id * {2 * HPT * H};
  float acc[{M}][{2 * HPT}], sq[{M}];
  for (int r = 0; r < {M}; r++) {{ sq[r] = 0.0f; for (int q = 0; q < {2 * HPT}; q++) acc[r][q] = 0.0f; }}
  {"" if "noab" in D else "DMA_FILL(2, DESC(desc, 1), %d, (int)xin); DMA_FILL(0, DESC(desc, 2), %d, (int)wt);" % (XB, WB0)}
  for (int c = 0; c < {0 if "noab" in D else NCH}; c++) {{
    int p = c & 1;
    if (c + 1 < {NCH}) {{
      if (p == 0) {{ DMA_FILL(3, DESC(desc, 1), {XB + XSZ}, (int)(xin + (c + 1) * {KB})); DMA_FILL(1, DESC(desc, 2), {WB0 + WSZ}, (int)(wt + (c + 1) * {KB})); }}
      else        {{ DMA_FILL(2, DESC(desc, 1), {XB}, (int)(xin + (c + 1) * {KB})); DMA_FILL(0, DESC(desc, 2), {WB0}, (int)(wt + (c + 1) * {KB})); }}
    }}
    if (p == 0) {{ DMA_WAIT(0); DMA_WAIT(2); }} else {{ DMA_WAIT(1); DMA_WAIT(3); }}
    __global float* WB = LSF({WB0} + p * {WSZ});
    for (int r = 0; r < {M}; r++) {{
      __global float* xc = LSF({XB} + p * {XSZ}) + r * {KB};
      float8 sq8 = BC(0.0f), {", ".join(f"a{r_} = BC(0.0f)" for r_ in range(2 * HPT))};
      for (int i = 0; i < {KB}; i += 8) {{ float8 xv = *(__global float8*)(xc + i); sq8 += xv * xv;{wacc} }}
      sq[r] += hsum8(sq8); {" ".join(f"acc[r][{q}] += hsum8(a{q});" for q in range(2 * HPT))}
    }}
  }}
  float bb[{M}][{HPT}], dd[{M}][{HPT}];
  for (int r = 0; r < {M}; r++) {{
    float inv = 1.0f / __builtin_sqrtf(sq[r] * {1.0 / H!r}f + {eps!r}f);
    for (int q = 0; q < nh; q++) {{
      bb[r][q] = 1.0f / (1.0f + ex1(-acc[r][{HPT} + q] * inv));
      dd[r][q] = ex1(adt[h0 + q] * softplus1(acc[r][q] * inv + adt[{NV} + h0 + q]));
    }}
  }}
  /* ---- the delta rule: units (head, wave, state block); per (head, wave) the wave's q / k / v, then each block takes its updates */
  __global float* QKV = LSF({QS});
  #define QT(t) (QKV + (t) * {DK})
  #define KT(t) (QKV + ({MW} + (t)) * {DK})
  #define VT(t) (QKV + ({2 * MW} + (t)) * {DK})
  #define OT(t) (QKV + ({3 * MW} + (t)) * {DK})
  int nu = nh * {NW * NB};
  #define UOF(e) ((h0 + (e) / {NW * NB}) * {NB} + (e) % {NB})                 /* unit e's state block */
  #define WOF(e) (((e) / {NB}) % {NW})                                       /* unit e's wave */
  #define UBASE(u) (S + ((u) / {NB}) * {DK * DV} + ((u) % {NB}) * {W})
  #define BANK(t, u) (banks + ((t) * {NG} + L) * {SZ} + ((u) / {NB}) * {DK * DV} + ((u) % {NB}) * {W})
  #define SRC(e) (WOF(e) == 0 ? UBASE(UOF(e)) : BANK({"WDEP(WOF(e))" if tree else f"WOF(e) * {MW}"} - 1, UOF(e)))   /* a block's state at its wave's start{": the parent's bank" if tree else ""} */
  DMA_FILL(0, DESC(desc, 0), 0, (int)SRC(0));
  for (int e = 0; e < nu; e++) {{
    int u = UOF(e), w = WOF(e), p = e & 1, h = u / {NB}, jb = u % {NB}, j0 = jb * {W}, j = h / {REP};
    int t0 = {f"w == 0 ? 0 : {MW} + w - 1, mw = w == 0 ? {MW} : 1" if tree else f"w * {MW}, mw = w == {NW - 1} ? {ML} : {MW}"};
    if ({"0" if "noprep" in D else "jb == 0"}) {{                      /* the wave's q / k / v: taps (and, wave 0, ring rows) in by DMA, raw rows out */
      int fa = 1 - p, fb = 2 + p, fc = 3 - p, sb = (1 - p) * {BLK}, on = {QO} + j * {DK};
      DMA_WAIT(fc); DMA_WAIT(fa);                                     /* the staging buffer's drain (the previous block); the previous head's output */
      DMA_FILL(fa, DESC(desc, 3), sb, (int)(cwt + on)); if ({"1" if tree else "w == 0"}) DMA_FILL(fb, DESC(desc, 3), sb + {SEG}, (int)(Cr + on));
      for (int s = 0; s < 3; s++) {{                                 /* segments: q, k, v */
        int a = sb + (s & 1) * {AREA}, off = on, v = s == 2;
        DMA_WAIT(fa); DMA_WAIT(fb); DMA_WAIT(fc);                     /* this segment's rows landed; segment s - 1's raw rows drained */
        if (s + 1 < 3) {{                                             /* the next segment's rows under this one's math */
          int an = sb + ((s + 1) & 1) * {AREA}; on = s == 0 ? {KO} + j * {DK} : {VO} + h * {DV};
          DMA_FILL(fa, DESC(desc, 3), an, (int)(cwt + on)); if ({"1" if tree else "w == 0"}) DMA_FILL(fb, DESC(desc, 3), an + {SEG}, (int)(Cr + on));
        }}
        __global float* o3 = QKV + (v ? {2 * MW} : s * {MW}) * {DK};
        {per_wave(lambda w: f"conv_w{w}(x, LSF(a), OT(0), pos, off, o3{f', d{w}' if (w and tree) else ''});")}
        if (v || h % {REP} == 0) {{ if ({"w != 0" if tree else f"w == {NW - 1}"}) DMA_DRAIN(fc, DESC(desc, 6), {RAW0}, (int)(raw + t0 * {CP} + off)); else DMA_DRAIN(fc, DESC(desc, 4), {RAW0}, (int)(raw + t0 * {CP} + off)); }}
      }}
      for (int t = 0; t < mw; t++) {{ l2n(QT(t), {1.0 / DK ** 0.5!r}f); l2n(KT(t), 1.0f); }}
      DMA_WAIT(fc);
    }}
    if (e + 1 < nu) {{
      if (p == 0) {{ DMA_WAIT(3); DMA_FILL(1, DESC(desc, 0), {BLK}, (int)SRC(e + 1)); }}
      else        {{ DMA_WAIT(2); DMA_FILL(0, DESC(desc, 0), 0, (int)SRC(e + 1)); }}
    }}
    if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
    __global float* B = LSF(p * {BLK});
    for (int t = 0; t < mw; t++) {{
      int tg = t0 + t; __global float* ks = KT(t); __global float* qs = QT(t);
      float8 db = BC(dd[tg][h - h0]), kv0 = BC(0.0f), kv1 = BC(0.0f);
      for (int i = 0; i < {0 if "nostate" in D else DK}; i++) {{
        float8 kb = BC(ks[i]); float8 s0 = *(__global float8*)(B + i * {W}) * db, s1 = *(__global float8*)(B + i * {W} + 8) * db;
        *(__global float8*)(B + i * {W}) = s0; *(__global float8*)(B + i * {W} + 8) = s1; kv0 += kb * s0; kv1 += kb * s1;
      }}
      float8 bt = BC(bb[tg][h - h0]);
      float8 dl0 = (*(__global float8*)(VT(t) + j0) - kv0) * bt, dl1 = (*(__global float8*)(VT(t) + j0 + 8) - kv1) * bt;
      float8 o0 = BC(0.0f), o1 = BC(0.0f);
      for (int i = 0; i < {0 if "nostate" in D else DK}; i++) {{
        float8 kb = BC(ks[i]), qb = BC(qs[i]);
        float8 s0 = *(__global float8*)(B + i * {W}) + kb * dl0, s1 = *(__global float8*)(B + i * {W} + 8) + kb * dl1;
        *(__global float8*)(B + i * {W}) = s0; *(__global float8*)(B + i * {W} + 8) = s1; o0 += qb * s0; o1 += qb * s1;
      }}
      *(__global float8*)(OT(t) + j0) = o0; *(__global float8*)(OT(t) + j0 + 8) = o1;
      if (tg < {0 if "nobank" in D else M - 1}) {{                                            /* the rollback point: the state after tg + 1 tokens */
        if (p == 0) {{ DMA_DRAIN(2, DESC(desc, 0), 0, (int)BANK(tg, u)); {"" if "nobankwait" in D else "DMA_WAIT(2);"} }}
        else        {{ DMA_DRAIN(3, DESC(desc, 0), {BLK}, (int)BANK(tg, u)); {"" if "nobankwait" in D else "DMA_WAIT(3);"} }}
      }}
    }}
    if (w == {NW - 1}) {{ if (p == 0) DMA_DRAIN(2, DESC(desc, 0), 0, (int)UBASE(u)); else DMA_DRAIN(3, DESC(desc, 0), {BLK}, (int)UBASE(u)); }}   /* the last wave: the state */
    if (jb == {NB - 1}) {{                                          /* the wave's o = rmsnorm(o) * nw * silu(z) -> its rows of the head's A block */
      float invt[{MW}];
      for (int t = 0; t < mw; t++) {{
        float8 ss = BC(0.0f); for (int c = 0; c < {DV}; c += 8) {{ float8 ov = *(__global float8*)(OT(t) + c); ss += ov * ov; }}
        invt[t] = 1.0f / __builtin_sqrtf(hsum8(ss) * {1.0 / DV!r}f + {eps!r}f);
      }}
      __global half* oa = LSH({OST});                                /* head h's block of o_proj's A: [kq][{RT} row tiles][16 halves], across the waves */
      float8 Z = BC(0.0f);
      {per_wave(lambda w: chr(10) + oproj(w) + "      ")}
      if (w == {NW - 1}) DMA_DRAIN(p, DESC(desc, 5), {OST}, (int)(out + h * {DV // 128 * 512 * RT}));   /* flag p: free here; waited at the next head */
    }}
  }}
  DMA_WAIT(0); DMA_WAIT(1); DMA_WAIT(2); DMA_WAIT(3);       /* per flag, not DMA_WAIT_ALL (DCache.md §8b) */
  #undef UBASE
  #undef BANK
  #undef SRC
  #undef UOF
  #undef WOF{chr(10) + "  #undef WDEP" if tree else ""}
}}"""

def _defer_src5(src, NV, M, MS):
  """gdn_defer_src with its row limit at LSRAM's (M <= MS <= 5; the function's own text, its one `<= 4` assert widened -- the
  slots fit: PV0 + 5 GD_SLOT floats = 32 064 B at M = 5), so the leaf tree can verify 5 rows on the deferred commit."""
  import inspect
  s = inspect.getsource(gdn_defer_src); a = "assert M <= MS <= 4 and"; assert s.count(a) == 1, "gdn_defer_src's row limit moved"
  ns = {}; exec(compile(s.replace(a, "assert M <= MS <= 5 and"), __file__, "exec"), globals(), ns)
  return ns["gdn_defer_src"](src, NV, M, MS)

def gdn_tokl_src(NV, NK, DK, DV, C, CONV, H, nrb, eps, M, MS, NG, W=16):
  """The LEAF TREE's DeltaNet (QWEN_SPEC_TREE=leaf, spec_tree.build_leaf; the kernel `gdn_tokl`): gdn_tokm(rows_out) with the
  deferred commit (gdn_defer_src), for a verify pass whose rows 0..nc-1 are the chain (the pass's token and its drafts) and rows
  nc..M-1 LEAVES -- another candidate for a chain position j (1 <= j <= nc - 1: a sibling of chain row j, parent chain row j - 1),
  never a parent. Two more runtime inputs: `tree` (spec_tree.table: int32 [2 M], depth first; the chain rows have depth[r] = r,
  nc is the first row that does not) and, in place of the deferred kernel's accp, `accp` = spec_tree.path_words of the PREVIOUS
  pass ([a, last, kv src, kv dst, its committed rows ...]): its a pending updates are the update slots of the committed rows in
  order, so a path ending on a leaf needs no copy -- the leaf's slot is applied in its place. Edits to the deferred source:
  - the conv window of every row by its depth (depth d: the raw rows d-3..d-1 of its ancestors, the ring for positions < pos):
    the same terms in the same order as the chain's window for a chain row;
  - the chain loop runs nc rows; after chain row t's update, each leaf of depth t + 1 takes its output from the block's state
    (the state after rows 0..t), read only: kq = k^T S, qs = q^T S, o = d qs + (q.k) (v - d kq) beta -- the chain's
    q^T (d S + k (v - d k^T S) beta) without writing S (no copy, no LSRAM); its update inputs (k, v, beta, decay) go to its slot
    as every row's;
  - the pending updates from slot path[i] instead of slot i.
  Under spec_tree.chain_table and a chain path ([a, .., 0..a-1]) the sums are the deferred kernel's in the same order:
  bit-identical (the simulator gate). M <= 5 (the deferred kernel's LSRAM, MS slots of GD_SLOT floats); the a leaf's output is
  the chain formula's in another order (fp32 rounding, ~1e-7 relative)."""
  assert 2 <= M <= MS <= 5 and W == 16 and gdn_prep() == "dma", "the leaf tree: 2..5 rows on the deferred, DMA-prepared chain kernel"
  base = gdn_tokm_src(NV, NK, DK, DV, C, CONV, H, nrb, eps, M, NG, rows_out=True)
  src = gdn_defer_src(base, NV, M, MS) if MS <= 4 else _defer_src5(base, NV, M, MS)
  GD = GD_SLOT; dps = ", ".join(f"dp{r}" for r in range(M))
  # the conv: every row's window by its depth
  a_ = src.index("static inline __attribute__((always_inline)) void conv_seg("); b_ = src.index("static inline __attribute__((always_inline)) void l2n(")
  conv = (f"static inline __attribute__((always_inline)) void conv_seg(__global float* restrict ct, __global float* restrict A, __global float* restrict Rw, int pos, int off,\n"
          f"                                                           __global float* restrict out, {', '.join(f'int d{r}' for r in range(M))}) {{\n"
          f"  for (int i = 0; i < {DK}; i += 8) {{" + "".join(f" *(__global float8*)(Rw + {t * DK} + i) = LD8R(ct, off + i, {t});" for t in range(M)) + " }\n")
  for t in range(M):
    conv += "  {" + "".join(f" int q{tt} = d{t} - {CONV - 1 - tt}, p{tt} = pos + q{tt}; __global float* W{tt} = q{tt} >= 0 ? Rw + q{tt} * {DK} : A + ({CONV} + (p{tt} >= 0 ? p{tt} : 0) % {CONV}) * {DK};"
                            for tt in range(CONV - 1)) + "\n"
    conv += (f"    for (int i = 0; i < {DK}; i += 8) {{ float8 acc = *(__global float8*)((Rw + {t * DK}) + i) * *(__global float8*)(A + {(CONV - 1) * DK} + i);"
             + "".join(f" if (p{tt} >= 0) acc += *(__global float8*)(W{tt} + i) * *(__global float8*)(A + {tt * DK} + i);" for tt in range(CONV - 1))
             + f" *(__global float8*)(out + {t * DK} + i) = sw(acc, BC(1.0f)); }} }}\n")
  src = src[:a_] + conv + "}\n" + src[b_:]
  call = f"conv_seg(x, LSF(a), OT(0), pos, off, QKV + (v ? {2 * M} : s * {M}) * {DK});"
  pend0 = "    if (ap > 0) {"; pend1 = "      if (p == 0) { DMA_DRAIN(2, DESC(desc, 0), 0, (int)UBASE(u)); DMA_WAIT(2); }"
  i0 = src.index(pend0); i1 = src.index(pend1, i0); blk = src[i0:i1]
  lp = "      for (int t = 0; t < ap; t++) {\n"; assert blk.count(lp) == 1 and blk.count(f"t * {GD}") == 5, blk
  blk = blk.replace(lp, lp + f"        int sl = accp[4 + t] * {GD};                              /* tree: the committed path's t-th row's slot */\n").replace(f"t * {GD}", "sl")
  src = src[:i0] + blk + src[i1:]
  leaf = "".join(f"""      if ({r} >= nc && dp{r} == t + 1) {{                              /* tree: leaf row {r} (parent chain row t), from the state after row t, read only */
        __global float* kl = KT({r}); __global float* ql = QT({r});
        float8 kq0 = BC(0.0f), kq1 = BC(0.0f), qq0 = BC(0.0f), qq1 = BC(0.0f), qk8 = BC(0.0f);
        for (int i = 0; i < {DK}; i++) {{
          float8 kb = BC(kl[i]), qb = BC(ql[i]); float8 s0 = *(__global float8*)(B + i * {W}), s1 = *(__global float8*)(B + i * {W} + 8);
          kq0 += kb * s0; kq1 += kb * s1; qq0 += qb * s0; qq1 += qb * s1;
        }}
        for (int i = 0; i < {DK}; i += 8) qk8 += *(__global float8*)(kl + i) * *(__global float8*)(ql + i);
        float8 db = BC(dd[{r}][h - h0]), bt = BC(bb[{r}][h - h0]), qk = BC(hsum8(qk8));
        float8 dl0 = (*(__global float8*)(VT({r}) + j0) - kq0 * db) * bt, dl1 = (*(__global float8*)(VT({r}) + j0 + 8) - kq1 * db) * bt;
        *(__global float8*)(OT({r}) + j0) = qq0 * db + qk * dl0; *(__global float8*)(OT({r}) + j0 + 8) = qq1 * db + qk * dl1;
      }}
""" for r in range(1, M))
  ost = "      *(__global float8*)(OT(t) + j0) = o0; *(__global float8*)(OT(t) + j0 + 8) = o1;\n"
  edits = [("__kernel void gdn_tokm(", "__kernel void gdn_tokl("),
           ("__global int* restrict accp, const int core_id) {\n  int L = idx[0], pos = posb[0], ap = accp[0];",
            "__global int* restrict accp, __global int* restrict tree, const int core_id) {\n  int L = idx[0], pos = posb[0], ap = accp[0];\n"
            f"  int {', '.join(f'dp{r} = tree[{r}]' for r in range(M))}, nc = {M};" + "".join(f" if (dp{r} != {r}) nc = {r};" for r in reversed(range(1, M)))
            + "   /* tree: the rows' depths; nc = the chain rows */"),
           (call, call[:-2] + f", {dps});"),
           (f"    for (int t = 0; t < {M}; t++) {{\n      __global float* ks = KT(t); __global float* qs = QT(t);\n",
            "    for (int t = 0; t < nc; t++) {                                  /* tree: the chain rows */\n      __global float* ks = KT(t); __global float* qs = QT(t);\n"),
           (ost, ost + leaf)]
  for x, y in edits:
    assert src.count(x) == 1, x
    src = src.replace(x, y)
  return src

def attn_part_tree_src(NH, NKV, HD, TMAX, ROT, eps, M, BT=None):
  """attn_part (M <= 7 rows) for a tree verify pass (the kernel `attn_partt`; M > 7: attn_part_rg_src(tree=True)): a 12th argument
  `tree` (spec_tree.table: int32 [2 M] = depth[r], anc[r]): row r's query and new k are rotated at position pos + depth[r]
  (its k / v still go to the cache at pos + r: gdn_commit_tree moves a committed leaf's to pos + depth), and row r attends the
  new row t only when bit t - pos of anc[r] is set. Under spec_tree.chain_table it is attn_part's arithmetic: bit-identical."""
  assert M <= 7
  src = attn_part_src(NH, NKV, HD, TMAX, ROT, eps, M, ATT_BT if BT is None else BT)
  edits = [("__kernel void attn_part(", "__kernel void attn_partt("),
           ("__global int* restrict desc, const int core_id) {", "__global int* restrict desc, __global int* restrict tree, const int core_id) {"),
           (f"rope + (pos + r) * {2 * ROT}", f"rope + (pos + tree[r]) * {2 * ROT}", 2),
           (f"rope + t * {2 * ROT}", f"rope + (pos + tree[t - pos]) * {2 * ROT}", 1),
           ("if (t <= pos + r)", f"if ((tree[{M} + r] >> (t - pos)) & 1)", 2)]
  for x, y, *n in edits:
    assert src.count(x) == (n[0] if n else 1), x
    src = src.replace(x, y)
  return src

GF_MS = 8                                                                   # gdn_fast_src: the most pending (deferred) updates a kernel takes

def gdn_fast():
  """QWEN_GDN_FAST: the deferred-commit verify kernel -- "asm" (default: gdn_fast_src, the hand-scheduled sweeps), "c" (gdn_fast_src
  with the C sweeps), "off" (gdn_defer_src, M <= 4 only; the bank path above)."""
  v = os.environ.get("QWEN_GDN_FAST", "asm").strip(); assert v in ("asm", "c", "off"), f"QWEN_GDN_FAST={v!r}: asm | c | off"
  return v

def gdn_fast_cfg(M, MS):
  """gdn_fast_src's LSRAM plan for M new tokens and MS pending-update slots -> dict(DB, VG, offsets): DB 1 = two state buffers
  (the next block's fill under this block's sweeps), 0 = one; VG = v column groups (1: every token's v row resident for the whole
  head; 2: half a row, the halves loaded per group of NB / 2 blocks -- the new tokens' v rows parked in a DDR scratch). The first
  of (DB 1 VG 1), (1, 2), (0, 1), (0, 2) that fits 32 KiB - 64 (rows of 512 B: k (MS + M), q (M), v (MS + M) / VG, o (M); the
  per-token beta / decay pairs at 64 B a token)."""
  T = MS + M
  for DB, VG in ((1, 1), (1, 2), (0, 1), (0, 2)):
    o = {"KA": (1 + DB) * 8192}; o["QA"] = o["KA"] + T * 512; o["VA"] = o["QA"] + M * 512; o["OA"] = o["VA"] + T * 512 // VG
    o["BD"] = o["OA"] + M * 512; o["END"] = o["BD"] + T * 64
    if o["END"] <= 32768 - 64 and (VG == 1 or MS >= M): return dict(DB=DB, VG=VG, **o)
  raise AssertionError(f"gdn_fast_src: M {M} MS {MS} does not fit LSRAM")

def gdn_fast_kbd(M, HPT, KB=128, H=5120):
  """gdn_fast_src's a|b DMA chunk (columns): 2 KB when both double-buffered chunks fit LSRAM (half the requests of KB), else KB."""
  return 2 * KB if 2 * (M + 2 * HPT) * 2 * KB * 4 <= 32768 - 64 and H % (2 * KB) == 0 else KB

def gdn_fast_desc(DK, DV, H, HPT, M, MS, C, CONV=4, KB=128, W=16, NV=48, a_out=False):
  """gdn_fast_src's DMA slots: 0 a state block; 1 a chunk of the M input rows; 2 a chunk of a task's a|b rows (chunks of
  gdn_fast_kbd columns); 3 the CONV rows of a
  128-channel segment (taps or ring); 4 the M raw rows of a segment; 5 / 6 / 7 the MS pending slots' k / v / beta-decay rows in
  (DDR pitch GD_SLOT floats); 8 / 9 / 10 the M new k / v / beta-decay rows out (the same pitch); VG 2: 11 the M v rows to / from
  the scratch (contiguous), 12 half the pending v rows (one column group), 13 half the new v rows from the scratch; 14 / 15 the MS
  pending / M new slots whole (one request each, when they fit the staging buffer); 16 a head's M output rows out (DDR pitch NV DV); 17 / 18 a 128-channel segment's C tiles, row quad 0 (14 / 17 virtual
  strips of 768 B, the first 256 B of each: gdn_fast_src's CTD); a_out: 19 a head's block of o_proj's compact A out (contiguous,
  DV / 128 x 32 k-quads x RT row tiles x 16 halves)."""
  G = GD_SLOT * 4; KBD = gdn_fast_kbd(M, HPT, KB, H)
  sl = [(DK * W * 4, W * 4, DV * 4, W * 4), (M * KBD * 4, KBD * 4, H * 4, KBD * 4), (2 * HPT * KBD * 4, KBD * 4, H * 4, KBD * 4),
        (CONV * DK * 4, DK * 4, ring_pitch(C) * 4, DK * 4), (M * DK * 4, DK * 4, ring_pitch(C) * 4, DK * 4),
        (MS * 512, 512, G, 512), (MS * 512, 512, G, 512), (MS * 64, 64, G, 64), (M * 512, 512, G, 512), (M * 512, 512, G, 512), (M * 64, 64, G, 64),
        (M * 512,), (MS * 256, 256, G, 256), (M * 256, 256, 512, 256), (MS * G,), (M * G,), (M * DV * 4, DV * 4, NV * DV * 4, DV * 4), (14 * 256, 256, 768, 256), (17 * 256, 256, 768, 256)]
  if a_out: sl.append((DV // 128 * 512 * -(-M // 4) * 2,))
  return V._desc_slots(*sl)

def gdn_fast_scratch(M):
  """floats of the VG 2 scratch after the [NG][NV][MS] slots in `banks`: a task's M new v rows."""
  return NT * M * 128

def gdn_sweeps_c(DK=128, W=16, tree=False):
  """gdn_fast_src's sweeps in C (sw_k0 / sw_p / sw_n / sw_nl: plain k / q rows, the compiler's schedule): the reference the asm
  sweeps (gdn_asm.sweep_c) are checked against, and the kernel's sweeps with QWEN_GDN_ASM=0. `tree` (gdn_fast_src(tree=True)):
  also sw_nb (sw_n storing s: a leaves' parent) and sw_lf (a leaf's read-only sums; st: the state decayed and stored too)."""
  # the sweeps (C; `B` the block, k / q rows by pointer, two float8 halves a row). ROW opens a row, END closes it.
  ROW = f"  for (int i = 0; i < {DK}; i++) {{ float8 b0 = *(__global float8*)(B + i * {W}), b1 = *(__global float8*)(B + i * {W} + 8);"
  ST2 = lambda a, b: f" *(__global float8*)(B + i * {W}) = {a}; *(__global float8*)(B + i * {W} + 8) = {b};"
  return f"""#define OPQ(v) __asm__("" : "+t"(v))                                     /* an opaque value: no fma contraction across it */
/* K0: d = S dn (stored); kv += kn d */
static inline __attribute__((always_inline)) void sw_k0(__global float* restrict B, __global float* restrict kn, float8 dn, float8* kv) {{
  float8 kv0 = BC(0.0f), kv1 = BC(0.0f);
{ROW} float8 nb = BC(kn[i]); float8 d0 = b0 * dn, d1 = b1 * dn;{ST2("d0", "d1")} kv0 += nb * d0; kv1 += nb * d1; }}
  kv[0] = kv0; kv[1] = kv1;
}}
/* P (pb = 0): s = S + k dl; d = s dn (stored); kv += kn d.  PB (pb = 1): s stored instead (the committed state) */
static inline __attribute__((always_inline)) void sw_p(__global float* restrict B, __global float* restrict kc, __global float* restrict kn, float8 dl0, float8 dl1, float8 dn, float8* kv, int pb) {{
  float8 kv0 = BC(0.0f), kv1 = BC(0.0f);
  if (pb) {{
{ROW} float8 kb = BC(kc[i]), nb = BC(kn[i]); float8 s0 = b0 + kb * dl0, s1 = b1 + kb * dl1;{ST2("s0", "s1")} float8 d0 = s0 * dn, d1 = s1 * dn; kv0 += nb * d0; kv1 += nb * d1; }}
  }} else {{
{ROW} float8 kb = BC(kc[i]), nb = BC(kn[i]); float8 s0 = b0 + kb * dl0, s1 = b1 + kb * dl1; float8 d0 = s0 * dn, d1 = s1 * dn;{ST2("d0", "d1")} kv0 += nb * d0; kv1 += nb * d1; }}
  }}
  kv[0] = kv0; kv[1] = kv1;
}}
/* N (n1 = 0): s = S + k dl; o += q s; d = s dn (stored); kv += kn d.  N1 (n1 = 1): S stored undecayed: s = (S dc) + k dl */
static inline __attribute__((always_inline)) void sw_n(__global float* restrict B, __global float* restrict kc, __global float* restrict qc, __global float* restrict kn,
                                                       float8 dl0, float8 dl1, float8 dc, float8 dn, float8* kv, float8* o, int n1) {{
  float8 kv0 = BC(0.0f), kv1 = BC(0.0f), o0 = BC(0.0f), o1 = BC(0.0f);
  if (n1) {{
{ROW} float8 kb = BC(kc[i]), qb = BC(qc[i]), nb = BC(kn[i]); float8 c0 = b0 * dc, c1 = b1 * dc; OPQ(c0); OPQ(c1);
      float8 s0 = c0 + kb * dl0, s1 = c1 + kb * dl1; o0 += qb * s0; o1 += qb * s1; float8 d0 = s0 * dn, d1 = s1 * dn;{ST2("d0", "d1")} kv0 += nb * d0; kv1 += nb * d1; }}
  }} else {{
{ROW} float8 kb = BC(kc[i]), qb = BC(qc[i]), nb = BC(kn[i]); float8 s0 = b0 + kb * dl0, s1 = b1 + kb * dl1; o0 += qb * s0; o1 += qb * s1;
      float8 d0 = s0 * dn, d1 = s1 * dn;{ST2("d0", "d1")} kv0 += nb * d0; kv1 += nb * d1; }}
  }}
  kv[0] = kv0; kv[1] = kv1; o[0] = o0; o[1] = o1;
}}
/* NL: s = S + k dl; o += q s (no store) */
static inline __attribute__((always_inline)) void sw_nl(__global float* restrict B, __global float* restrict kc, __global float* restrict qc, float8 dl0, float8 dl1, float8* o) {{
  float8 o0 = BC(0.0f), o1 = BC(0.0f);
{ROW} float8 kb = BC(kc[i]), qb = BC(qc[i]); float8 s0 = b0 + kb * dl0, s1 = b1 + kb * dl1; o0 += qb * s0; o1 += qb * s1; }}
  o[0] = o0; o[1] = o1;
}}
""" + ("" if not tree else f"""/* the leaf tree. NB (n1 = 0) / N1B (n1 = 1): sw_n with s stored instead of d (a leaves' parent: they read the state after it) */
static inline __attribute__((always_inline)) void sw_nb(__global float* restrict B, __global float* restrict kc, __global float* restrict qc, __global float* restrict kn,
                                                        float8 dl0, float8 dl1, float8 dc, float8 dn, float8* kv, float8* o, int n1) {{
  float8 kv0 = BC(0.0f), kv1 = BC(0.0f), o0 = BC(0.0f), o1 = BC(0.0f);
  if (n1) {{
{ROW} float8 kb = BC(kc[i]), qb = BC(qc[i]), nb = BC(kn[i]); float8 c0 = b0 * dc, c1 = b1 * dc; OPQ(c0); OPQ(c1);
      float8 s0 = c0 + kb * dl0, s1 = c1 + kb * dl1;{ST2("s0", "s1")} o0 += qb * s0; o1 += qb * s1; float8 d0 = s0 * dn, d1 = s1 * dn; kv0 += nb * d0; kv1 += nb * d1; }}
  }} else {{
{ROW} float8 kb = BC(kc[i]), qb = BC(qc[i]), nb = BC(kn[i]); float8 s0 = b0 + kb * dl0, s1 = b1 + kb * dl1;{ST2("s0", "s1")} o0 += qb * s0; o1 += qb * s1;
      float8 d0 = s0 * dn, d1 = s1 * dn; kv0 += nb * d0; kv1 += nb * d1; }}
  }}
  kv[0] = kv0; kv[1] = kv1; o[0] = o0; o[1] = o1;
}}
/* LF (st = 0): a leaf's kq += kl S, qq += ql S (read only; kq -> kv, qq -> o).  LFD (st = 1): also d = S dn -> stored */
static inline __attribute__((always_inline)) void sw_lf(__global float* restrict B, __global float* restrict kl, __global float* restrict ql, float8 dn, float8* kv, float8* o, int st) {{
  float8 kq0 = BC(0.0f), kq1 = BC(0.0f), qq0 = BC(0.0f), qq1 = BC(0.0f);
  if (st) {{
{ROW} float8 kb = BC(kl[i]), qb = BC(ql[i]); kq0 += kb * b0; kq1 += kb * b1; qq0 += qb * b0; qq1 += qb * b1; float8 d0 = b0 * dn, d1 = b1 * dn;{ST2("d0", "d1")} }}
  }} else {{
{ROW} float8 kb = BC(kl[i]), qb = BC(ql[i]); kq0 += kb * b0; kq1 += kb * b1; qq0 += qb * b0; qq1 += qb * b1; }}
  }}
  kv[0] = kq0; kv[1] = kq1; o[0] = qq0; o[1] = qq1;
}}
""")

def _gf_tree_chain(asm, M, MS, drain):
  """gdn_fast_src(tree=True)'s chain over a state block: the previous path's pending updates (records PV(0..ap-1); asm: the KP
  columns 0..ap-1, copied in path order), then the chain rows 0..nc-1. A chain row t with leaves (rows r >= nc of depth t + 1)
  stores its state undecayed (NB; N1B when it also takes the decay first, as N1) and each leaf reads it: LF (kq = kl.S, qq = ql.S),
  the last one LFD (the same, and S * decay(t + 1) stored: what N would have stored), its output then gdn_tokl_src's
  o = qq d + (q.k) (v - kq d) beta. The chain rows' arithmetic is the chain kernel's (bit-identical under the chain table); a leaf
  of the last chain row does not occur (spec_tree.build_leaf: a leaf is a sibling of a chain row >= 1)."""
  P = (lambda f, *a: f"sa_{f}(B, " + ", ".join(a) + ", kv, o)") if asm else None
  LEAF = f"""          {{ float8 db = BC(BDR({MS} + r)[1]), bt = BC(BDR({MS} + r)[0]), qk = BC(qkt[r]);   /* gdn_tokl_src's leaf output */
            float8 dl0 = (*(__global float8*)(VR({MS} + r) + jv) - kv[0] * db) * bt, dl1 = (*(__global float8*)(VR({MS} + r) + jv + 8) - kv[1] * db) * bt;
            *(__global float8*)(OR(r) + j0) = o[0] * db + qk * dl0; *(__global float8*)(OR(r) + j0 + 8) = o[1] * db + qk * dl1; }}
"""
  if asm:
    pend = f"""      sa_K0p(B, KPV(0), BC(BDR(PV(0))[1]), kv, o);
      for (int t = 0; t + 1 < ap; t++) {{ float8 d0 = DL(PV(t), 0), d1 = DL(PV(t), 1); sa_P(B, KPV(t), d0, d1, BC(BDR(PV(t + 1))[1]), kv, o); NOHWL(t); }}
      {{ float8 d0 = DL(PV(ap - 1), 0), d1 = DL(PV(ap - 1), 1); sa_PB(B, KPV(ap - 1), KQ(0), d0, d1, BC(BDR({MS})[1]), kv, o); }}
"""
    k0n = f"sa_K0n(B, KQ(0), BC(BDR({MS})[1]), kv, o);"
    row = f"""      if (lst < 0) {{ if (dk_) sa_N1(B, KQ(t), d0, d1, BC(BDR({MS} + t)[1]), dn, kv, o); else sa_N(B, KQ(t), d0, d1, dn, kv, o); }}
      else         {{ if (dk_) sa_N1B(B, KQ(t), d0, d1, BC(BDR({MS} + t)[1]), dn, kv, o); else sa_NB(B, KQ(t), d0, d1, dn, kv, o); }}
"""
    leaf = "if (r == lst) sa_LFD(B, KQ(r), dn, kv, o); else sa_LF(B, KQ(r), kv, o);"
    last = f"sa_NL(B, KQ(nc - 1), d0, d1, kv, o);"
  else:
    pend = f"""      sw_k0(B, KR(PV(0)), BC(BDR(PV(0))[1]), kv);
      for (int t = 0; t + 1 < ap; t++) {{ float8 d0 = DL(PV(t), 0), d1 = DL(PV(t), 1); sw_p(B, KR(PV(t)), KR(PV(t + 1)), d0, d1, BC(BDR(PV(t + 1))[1]), kv, 0); }}
      {{ float8 d0 = DL(PV(ap - 1), 0), d1 = DL(PV(ap - 1), 1); sw_p(B, KR(PV(ap - 1)), KR({MS}), d0, d1, BC(BDR({MS})[1]), kv, 1); }}
"""
    k0n = f"sw_k0(B, KR({MS}), BC(BDR({MS})[1]), kv);"
    row = f"""      float8 dc = dk_ ? BC(BDR({MS} + t)[1]) : BC(0.0f);
      if (lst < 0) sw_n(B, KR({MS} + t), QR(t), KR({MS} + t + 1), d0, d1, dc, dn, kv, o, dk_);
      else         sw_nb(B, KR({MS} + t), QR(t), KR({MS} + t + 1), d0, d1, dc, dn, kv, o, dk_);
"""
    leaf = f"sw_lf(B, KR({MS} + r), QR(r), dn, kv, o, r == lst);"
    last = f"sw_nl(B, KR({MS} + nc - 1), QR(nc - 1), d0, d1, o);"
  return f"""    /* the leaf tree's chain ({"asm" if asm else "C"}): the previous path's pending updates, then chain rows 0..nc-1, each chain row's leaves after it */
    int dk_ = 0;                                                      /* the block holds a state not yet decayed (N1 / N1B next) */
    if (ap > 0) {{
{pend}     {drain}
      dk_ = 1;
    }} else {k0n}
    for (int t = 0; t + 1 < nc; t++) {{
      int lst = -1; for (int r = nc; r < {M}; r++) if (dpa[r] == t + 1) lst = r;   /* the last leaf of chain row t (-1: none) */
      float8 d0 = DL({MS} + t, 0), d1 = DL({MS} + t, 1), dn = BC(BDR({MS} + t + 1)[1]);
{row}      *(__global float8*)(OR(t) + j0) = o[0]; *(__global float8*)(OR(t) + j0 + 8) = o[1]; dk_ = 0;
      if (lst >= 0) {{
        float8 kn0 = kv[0], kn1 = kv[1];                                /* (chain row t + 1's k.S, under the leaves' sums) */
        for (int r = nc; r <= lst; r++) {{
          NOHWL(r);
          if (dpa[r] != t + 1) continue;
          {leaf}
{LEAF}        }}
        kv[0] = kn0; kv[1] = kn1;
      }}
      NOHWL(t);
    }}
    {{ float8 d0 = DL({MS} + nc - 1, 0), d1 = DL({MS} + nc - 1, 1); {last}
      *(__global float8*)(OR(nc - 1) + j0) = o[0]; *(__global float8*)(OR(nc - 1) + j0 + 8) = o[1]; }}
"""

def _gf_a_out(src, DV, M, cf):
  """gdn_fast_src(a_out=True): the output as o_proj's compact A (gdn_tokm_src's default layout: per head h = K-slice h a block of
  [32 k-quads][RT row tiles][16 halves], a tile's 16 halves = 4 rows x 4 columns, rows >= M zero) instead of fp32 rows. The fp32
  rows are computed and staged exactly as before (the head's dead v rows, VA); they are then converted in LSRAM into the dead o
  rows (OA: the o rows' last reader was the staging) and drained as one contiguous block (slot 19). fp16 of the staged values ==
  gdn_tokm_src's A (the same expression, rounded by the same conversion); everything else is the rows kernel's text."""
  RT, OA0, VA0 = -(-M // 4), cf["OA"], cf["VA"]                           # (cf: gdn_fast_cfg, the kernel's LSRAM plan)
  assert M * 512 >= RT * 1024 and "dbg[0] = ph0" not in src, "a_out: the A block in the o rows; no phases diag"
  V8 = lambda t: f"float8 v{t} = " + (f"*(__global float8*)(sv + {t * DV} + c);" if t < M else "Z;")
  ST = lambda i: (f"*(__global half16*)(oa + ((c / 4) * {RT} + {i}) * 16) = CVT16(F8(LO4(v{4*i}), LO4(v{4*i+1})), F8(LO4(v{4*i+2}), LO4(v{4*i+3}))); "
                  f"*(__global half16*)(oa + ((c / 4 + 1) * {RT} + {i}) * 16) = CVT16(F8(HI4(v{4*i}), HI4(v{4*i+1})), F8(HI4(v{4*i+2}), HI4(v{4*i+3})));")
  a = f"      {{ int fo = 3 - p; DMA_WAIT(fo); DMA_DRAIN(fo, DESC(desc, 16), {VA0}, (int)(out + h * {DV})); }}"
  b = (f"      {{ __global half* oa = LSH({OA0}); __global float* sv = LSF({VA0}); float8 Z = BC(0.0f);   /* a_out: the staged rows -> o_proj's compact A (the dead o rows) */\n"
       + "".join(f"        NOUNROLL for (int c = 0; c < {DV}; c += 8) {{ {' '.join(V8(t) for t in range(4 * i, 4 * i + 4))} {ST(i)} }}\n" for i in range(RT))
       + f"        int fo = 3 - p; DMA_WAIT(fo); DMA_DRAIN(fo, DESC(desc, 19), {OA0}, (int)(out + h * {DV // 128 * 512 * RT})); }}")
  edits = [(a, b), ("(__global float* restrict out, __global float* restrict Sall,", "(__global half* restrict out, __global float* restrict Sall,"),
           ("#define NOHWL(i) ", "#define LO4(v) __builtin_shufflevector((v), (v), 0, 1, 2, 3)\n#define HI4(v) __builtin_shufflevector((v), (v), 4, 5, 6, 7)\n#define NOHWL(i) ")]
  for x, y in edits:
    assert src.count(x) == 1, x
    src = src.replace(x, y)
  return src

def gdn_fast_src(NV, NK, DK, DV, C, CONV, H, nrb, eps, M, NG, MS, W=16, KB=128, sweeps="asm", diag=(), tree=False, a_out=False):
  """The DeltaNet verify kernel with the commit deferred (gdn_defer_src's contract), restructured around FUSED SWEEPS: a state
  block (DK x W fp32, in LSRAM) takes a token's update and the next token's decay + k.S in ONE pass over its rows, so T tokens
  cost T + 1 passes instead of 2 T, and the previous pass's ap = accp[0] accepted updates and the M new tokens run as one chain:
    K0  first token: d = S * decay -> stored; kv += k d
    P   pending token u (u < ap - 1): s = S + k delta (S stored decayed); d = s * decay(u + 1) -> stored; kv(u + 1) += k(u + 1) d
    PB  the last pending update: s -> stored (the committed state: drained to Sall, waited); kv(new 0) += k(new 0) (s * decay)
    N1  new token 0 after PB: s = (S * decay) + k delta; o += q s; d = s * decay(1) -> stored; kv(1) += k(1) d
    N   new token t: s = S + k delta; o += q s; d -> stored; kv(t + 1) += ...
    NL  the last new token: s = S + k delta; o += q s (nothing stored: the new tokens' states are never written back)
  Every element sees the old kernel's operations in its order (S * decay a rounded multiply, then fma(k, delta, .); the k.S and q.S
  sums sequential over the rows from +0), so the committed state and every output are the old kernel's bits. M <= 8, MS <= GF_MS.
  Per head the update inputs of the M new tokens (k, v, beta, decay: GD_SLOT floats a (head, token) slot, as gdn_defer_src) go
  to `banks` [NG][NV][MS] for the next pass; the previous pass's are read first. args: gdn_defer_src's.
  `sweeps` "asm" (default): the sweeps as hand-scheduled bundles (gdn_asm.sweep_c; the k / q rows in its KQN / KP layouts, built
  at each head's start); "c": the C sweeps (gdn_sweeps_c, plain rows) -- the same arithmetic, the compiler's schedule.
  `diag` (timing only, wrong results; GDN_TOKM_DIAG): "noab" no a|b loop, "noconv" no conv / norms, "nosweep" no sweeps,
  "noout" no output rows, "nocopy" no C-tile rows into the conv's staging, "nosilu" the conv's SiLU left out, "conv2" each segment's conv twice, "dmaquiet" the conv with no DMA in flight, "notouch" no C-tile line touches, "outnoz" / "outnosw" the output rows without z / without the SiLU; "phases": the cycle counter (ctrl0 209) summed per phase, int32 [16] a task into o_rows' last row
  (a padding row; its reads cost ~450 cycles each, so the totals carry them).
  `tree` (QWEN_SPEC_TREE=leaf): the LEAF TREE's kernel `gdn_tokl` -- gdn_tokl_src's contract, arguments and results (+ `tree` =
  spec_tree.table; accp = the previous pass's spec_tree.path_words): rows 0..nc-1 the chain, rows nc..M-1 leaves (depth d: a
  sibling of chain row d, 1 <= d <= nc - 1). Every row's conv window by its depth; the pending updates are the records of the
  previous path's rows (asm: copied into the KP / v / beta-decay rows in path order; C: indexed through it); the chain rows run as
  above (_gf_tree_chain) except that a chain row with leaves stores its state undecayed (NB / N1B) and each leaf reads it (LF, the
  last LFD storing the decayed state the next chain row expects), its output gdn_tokl_src's read-only formula. Under
  spec_tree.chain_table and chain paths every result is gdn_fast_src's bit for bit; under trees gdn_tokl_src's (outputs, slots,
  raw rows, the state written back): checked on the vendor simulator. tree=False: the source
  is byte-identical to the chain kernel's.
  `a_out` (the models without a rotated input: Ornith, Qwen3.8-27B): `out` is o_proj's compact A layout (half; RT = ceil(M / 4)
  row tiles, rows >= M zero), as gdn_tokm_src's default output -- see _gf_a_out. False: the fp32 rows (gdn_tokm_src(rows_out)'s).
  """
  if a_out: return _gf_a_out(gdn_fast_src(NV, NK, DK, DV, C, CONV, H, nrb, eps, M, NG, MS, W, KB, sweeps, diag, tree), DV, M, gdn_fast_cfg(M, MS))
  D = set(diag); PH = "phases" in D
  # tree (the leaf tree, QWEN_SPEC_TREE=leaf): the kernel `gdn_tokl` (gdn_tokl_src's contract and arguments) -- see the docstring
  TR = bool(tree); DPX = (lambda e: f"dpt[{e}]") if TR else (lambda e: e); DPA = ", dpa" if TR else ""
  QKT = lambda k, q: "" if not TR else (f"        for (int t = 0; t < {M}; t++) {{ float8 qk8 = BC(0.0f); for (int i = 0; i < {DK}; i += 8) qk8 += *(__global float8*)({k} + i) * "
                                        f"*(__global float8*)({q} + i); qkt[t] = hsum8(qk8); }}   /* tree: q.k (a leaf's output), as gdn_tokl_src's */\n")
  TREE0 = "" if not TR else f"""  /* tree: the rows' depths (dpa; nc = the chain rows: the first row r >= 1 whose depth is not r), the previous path's rows (pth: the
     pending updates' records), q.k of the rows (qkt) */
  int dpa[{M}], pth[{MS}], nc = {M}; float qkt[{M}];
  for (int r = 0; r < {M}; r++) dpa[r] = tree[r];
  for (int r = {M - 1}; r >= 1; r--) if (dpa[r] != r) nc = r;
  for (int t = 0; t < {MS}; t++) pth[t] = t < ap ? accp[4 + t] : t;
"""
  # integer index math without `div` / `mod` (the scalar unit's divide is not a fast op): non-negative operands, powers of two by
  # shift / mask, 3 by an exact multiply-shift (x < 2^15)
  def IDIV(x, d):
    if d & (d - 1) == 0: return f"((int)((unsigned)({x}) >> {d.bit_length() - 1}))"
    assert d == 3; return f"((int)(((unsigned)({x}) * 43691u) >> 17))"
  IMOD = lambda x, d: f"((int)((unsigned)({x}) & {d - 1}))" if d & (d - 1) == 0 else f"(({x}) - 3 * {IDIV(x, 3)})"
  T = lambda k, t0="t_": f" ph{k} += CYC() - {t0};" if PH else ""          # phase k += now - t0
  T0 = lambda v="t_": f" int {v} = CYC();" if PH else ""
  REP = NV // NK; QO, KO, VO = 0, NK * DK, 2 * NK * DK; CP = ring_pitch(C); assert VO + NV * DV == C and DK == DV == 128 and W == 16
  HPT, NB, BLK = gdn_hpt(NV), DV // W, DK * W * 4; NTU = -(-NV // HPT); LH = NV - (NTU - 1) * HPT
  assert H % KB == 0 and 2 <= M <= 8 and M <= MS <= GF_MS
  assert all(d & (d - 1) == 0 or d == 3 for d in (NB, REP, CONV))           # (IDIV / IMOD)
  cf = gdn_fast_cfg(M, MS); DB, VG = cf["DB"], cf["VG"]; KA0, QA0, VA0, OA0, BD0 = cf["KA"], cf["QA"], cf["VA"], cf["OA"], cf["BD"]
  VW = DV // VG; NBG = NB // VG                                             # v row floats resident; blocks a column group
  SZ, SL = NV * DK * DV, MS * GD_SLOT
  KBD = gdn_fast_kbd(M, HPT, KB, H); NSB = KBD // KB                      # a|b: DMA chunks of KBD columns, the sums in KB-column sub-chunks (as before)
  XSZ, WSZ, NCH = M * KBD * 4, 2 * HPT * KBD * 4, H // KBD; XB, WB0 = 0, 2 * M * KBD * 4
  assert WB0 + 2 * WSZ <= 32768 - 64
  SEG = CONV * DK * 4; AREA = 2 * SEG; assert 2 * AREA <= BLK and M * 512 <= cf["BD"] - OA0
  wacc = "".join(f" a{r} += xv * *(__global float8*)(WB + {r * KBD} + i);" for r in range(2 * HPT))
  X = lambda r: f"R{r}" if r < CONV - 1 else f"(Rw + {(r - CONV + 1) * DK})"
  # the C tiles' rows by 32-byte loads: 8 channels x a row quad are two 4 x 4 tiles, 128 contiguous bytes (LD8R's two 16-byte loads a
  # token missed the cache one at a time); token t = lanes of u(4q + 0..3), q its row quad: the same values (data movement only)
  QL = lambda q: f" float8 u{4 * q} = *(__global float8*)(tp + {64 * q}), u{4 * q + 1} = *(__global float8*)(tp + {64 * q + 8}), u{4 * q + 2} = *(__global float8*)(tp + {64 * q + 16}), u{4 * q + 3} = *(__global float8*)(tp + {64 * q + 24});"
  TS = lambda t: (f"__builtin_shufflevector(u{4 * (t // 4) + (t % 4) // 2}, u{4 * (t // 4) + 2 + (t % 4) // 2}, " +
                  ("0, 1, 2, 3, 8, 9, 10, 11)" if t % 2 == 0 else "4, 5, 6, 7, 12, 13, 14, 15)"))
  TQ = lambda t: (QL(t // 4) if t % 4 == 0 else "") + f" *(__global float8*)(Rw + {t * DK} + i) = {TS(t)};"
  CSEG = (f"""static inline __attribute__((always_inline)) void conv_seg(__global float* restrict ct, __global float* restrict A, __global float* restrict Rw, int pos, int off,
                                                           __global float* restrict out{", int* restrict dpt" if TR else ""}{", int* restrict cs" if PH else ""}) {{
{"  int c0_ = CYC();" if PH else ""}
  {"" if "nocopy" in D or "notouch" in D else "NOUNROLL for (int i = 0; i < %d; i += 32) { __global float* tp = ct + CTA(off + i); __global float* tq = ct + CTA(off + i + 16);%s }" % (DK, "".join(f" TOUCH8(tp + {64 * q}, tq + {64 * q});" for q in range(-(-M // 4))))}
{"  int c1_ = CYC(); cs[0] += c1_ - c0_;" if PH else ""}
  NOUNROLL for (int i = 0; i < {0 if "nocopy" in D else DK}; i += 8) {{ __global float* tp = ct + CTA(off + i);""" + "".join(TQ(t) for t in range(M)) + " }\n" + ("  int c2_ = CYC(); cs[1] += c2_ - c1_;\n" if PH else "")
      # one token's window and SiLU a loop (a body under the 32-bundle loop buffer: a body of M tokens is fetch-bound at 4 TECs);
      # the window rows and the sums as gdn_tokm_src's (tap CONV - 1 first; a ring row of a position < 0 left out)
      + "".join(f"  int p{r} = pos - {CONV - 1 - r}; __global float* R{r} = A + ({CONV} + {IMOD(f'(p{r} >= 0 ? p{r} : 0)', CONV)}) * {DK};\n" for r in range(CONV - 1))
      # the conv proper: tokens in pairs (two independent SiLU chains a loop body: the chain is ~20 dependent ops) while every window
      # term is present (pos >= CONV - 1), else one token a loop with the terms' guards; the sums as gdn_tokm_src's either way
      + "".join(f"  #define XW{tt}(t) ((t) + {tt} < {CONV - 1} ? (" + " : ".join(f"(t) + {tt} == {r} ? R{r}" for r in range(CONV - 2)) + f" : R{CONV - 2}) : Rw + ((t) + {tt - (CONV - 1)}) * {DK})\n" for tt in range(CONV - 1))
      + f"""  int t0 = 0;
  if (pos >= {CONV - 1}) {{
    for (; t0 + 1 < {M}; t0 += 2) {{
      NOHWL(t0);
""" + "".join(f"      __global float* x{tt}a = XW{tt}({DPX('t0')}); __global float* x{tt}b = XW{tt}({DPX('t0 + 1')});\n" for tt in range(CONV - 1))
      + f"""      __global float* xra = Rw + t0 * {DK}; __global float* xrb = xra + {DK}; __global float* oa = out + t0 * {DK}; __global float* ob = oa + {DK};
      NOUNROLL for (int i = 0; i < {DK}; i += 8) {{
        float8 acc = *(__global float8*)(xra + i) * *(__global float8*)(A + {(CONV - 1) * DK} + i);""" + "".join(f" acc += *(__global float8*)(x{tt}a + i) * *(__global float8*)(A + {tt * DK} + i);" for tt in range(CONV - 1)) + f"""
        float8 acb = *(__global float8*)(xrb + i) * *(__global float8*)(A + {(CONV - 1) * DK} + i);""" + "".join(f" acb += *(__global float8*)(x{tt}b + i) * *(__global float8*)(A + {tt * DK} + i);" for tt in range(CONV - 1)) + """
        *(__global float8*)(oa + i) = """ + ("acc" if "nosilu" in D else "sw(acc, BC(1.0f))") + "; *(__global float8*)(ob + i) = " + ("acb" if "nosilu" in D else "sw(acb, BC(1.0f))") + """;
      }
    }
  }
""" + f"""  for (int t = t0; t < {M}; t++) {{
    NOHWL(t);
""" + "".join(f"    __global float* x{tt} = XW{tt}({DPX('t')});\n" for tt in range(CONV - 1))
      + "".join(f"    int c{tt} = {DPX('t')} + {tt} >= {CONV - 1} || pos + {DPX('t')} + {tt - (CONV - 1)} >= 0;\n" for tt in range(CONV - 1))
      + f"""    __global float* xr = Rw + t * {DK}; __global float* o = out + t * {DK};
    if ({" && ".join(f"c{tt}" for tt in range(CONV - 1))})
      NOUNROLL for (int i = 0; i < {DK}; i += 8) {{ float8 acc = *(__global float8*)(xr + i) * *(__global float8*)(A + {(CONV - 1) * DK} + i);""" + "".join(f" acc += *(__global float8*)(x{tt} + i) * *(__global float8*)(A + {tt * DK} + i);" for tt in range(CONV - 1)) + " *(__global float8*)(o + i) = " + ("acc" if "nosilu" in D else "sw(acc, BC(1.0f))") + """; }
    else
      NOUNROLL for (int i = 0; i < """ + str(DK) + """; i += 8) { float8 acc = *(__global float8*)(xr + i) * *(__global float8*)(A + """ + str((CONV - 1) * DK) + """ + i);""" + "".join(f" if (c{tt}) acc += *(__global float8*)(x{tt} + i) * *(__global float8*)(A + {tt * DK} + i);" for tt in range(CONV - 1)) + " *(__global float8*)(o + i) = " + ("acc" if "nosilu" in D else "sw(acc, BC(1.0f))") + """; }
  }
""" + "".join(f"  #undef XW{tt}\n" for tt in range(CONV - 1)) + ("  cs[2] += CYC() - c2_;\n" if PH else "") + """}
""")
  ASM = sweeps == "asm"; assert sweeps in ("asm", "c")
  ci = CSEG.index("  int p0 = pos - "); CBODY = CSEG[ci:CSEG.rindex("}")]                       # the conv's compute part (after the copy)
  CSEG2 = "" if not (ASM and DB and VG == 1 and M <= 4) else (
    f"""static inline __attribute__((always_inline)) void conv_cm(__global float* restrict A, __global float* restrict Rw, int pos, __global float* restrict out{", int* restrict dpt" if TR else ""}{", int* restrict cs" if PH else ""}) {{
{"  int c2_ = CYC();" if PH else ""}
""" + CBODY + f"""}}
/* CTD: a segment's raw rows (row quad 0) from its virtual strips in LSRAM (SB: virtual strip v0 of channel `off` first) */
static inline __attribute__((always_inline)) void conv_cpl(__global float* restrict SB, __global float* restrict Rw, int off) {{
  int s0_ = off >> 4, v0_ = 6 * {IDIV("s0_", 3)} + (s0_ - 3 * {IDIV("s0_", 3)});
  NOUNROLL for (int i = 0; i < {DK}; i += 8) {{
    int sg_ = s0_ + (i >> 4); __global float* tp = SB + (6 * {IDIV("sg_", 3)} + (sg_ - 3 * {IDIV("sg_", 3)}) - v0_) * 64 + ((i >> 3) & 1) * 32;""" + "".join(TQ(t) for t in range(M)) + """
  }
}
""")
  # CTD (asm, two state buffers, whole v rows, M <= 4): the C tiles' row quad 0 of a 128-channel segment by ONE DMA -- the 8 strips
  # read as uniformly spaced "virtual strips" (both row blocks' strips, 768 B apart: the other block's come along) into the current
  # state buffer, free during a head's start (its block-0 fill is issued after the start) and at the head's end (z)
  CTD = ASM and DB and VG == 1 and M <= 4
  SWEEPS = "".join(gdn_asm.sweep_c(k, M, MS) for k in ("K0p", "K0n", "P", "PB", "N1", "N", "NL") + (("NB", "N1B", "LF", "LFD") if TR else ())) if ASM else gdn_sweeps_c(DK, W, tree=TR)
  if "nosweep" in D: SWEEPS += "".join(f"#define {n}(...) ((void)0)\n" for n in ("sa_K0p", "sa_K0n", "sa_P", "sa_PB", "sa_N1", "sa_N", "sa_NL", "sw_k0", "sw_p", "sw_n", "sw_nl"))
  KQN0, KP0 = KA0, KA0 + M * 1024                                          # asm: [4-row group][new token] (k | q), [8-row group][pending slot] k
  CHAIN = (f"""    /* the chain (asm): pending 0..ap-1 (KP slots), then new 0..M-1 (KQN); one call site a sweep (text) */
    int t1 = 0;
    if (ap > 0) {{
      sa_K0p(B, KPV(0), BC(BDR(0)[1]), kv, o);
      for (int t = 0; t + 1 < ap; t++) {{ float8 d0 = DL(t, 0), d1 = DL(t, 1); sa_P(B, KPV(t), d0, d1, BC(BDR(t + 1)[1]), kv, o); NOHWL(t); }}
      {{ float8 d0 = DL(ap - 1, 0), d1 = DL(ap - 1, 1); sa_PB(B, KPV(ap - 1), KQ(0), d0, d1, BC(BDR({MS})[1]), kv, o); }}
     {T0("td_")} if (p == 0) {{ DMA_DRAIN(2, DESC(desc, 0), 0, (int)UBASE(u)); DMA_WAIT(2); }} else {{ DMA_DRAIN(3, DESC(desc, 0), {BLK}, (int)UBASE(u)); DMA_WAIT(3); }}{T(4, "td_")}
      {{ float8 d0 = DL({MS}, 0), d1 = DL({MS}, 1); sa_N1(B, KQ(0), d0, d1, BC(BDR({MS})[1]), BC(BDR({MS} + 1)[1]), kv, o); }}
      *(__global float8*)(OR(0) + j0) = o[0]; *(__global float8*)(OR(0) + j0 + 8) = o[1]; t1 = 1;
    }} else sa_K0n(B, KQ(0), BC(BDR({MS})[1]), kv, o);
    for (int t = t1; t + 1 < {M}; t++) {{
      float8 d0 = DL({MS} + t, 0), d1 = DL({MS} + t, 1); sa_N(B, KQ(t), d0, d1, BC(BDR({MS} + t + 1)[1]), kv, o);
      *(__global float8*)(OR(t) + j0) = o[0]; *(__global float8*)(OR(t) + j0 + 8) = o[1]; NOHWL(t);
    }}
    {{ float8 d0 = DL({MS + M - 1}, 0), d1 = DL({MS + M - 1}, 1); sa_NL(B, KQ({M - 1}), d0, d1, kv, o);
      *(__global float8*)(OR({M - 1}) + j0) = o[0]; *(__global float8*)(OR({M - 1}) + j0 + 8) = o[1]; }}
""" if ASM else f"""    /* the chain: pending 0..ap-1 (rows KR(0..ap-1)), then new 0..M-1 (KR(MS + t), QR(t)) */
    if (ap > 0) {{
      sw_k0(B, KR(0), BC(BDR(0)[1]), kv);
      for (int t = 0; t + 1 < ap; t++) {{ float8 d0 = DL(t, 0), d1 = DL(t, 1); sw_p(B, KR(t), KR(t + 1), d0, d1, BC(BDR(t + 1)[1]), kv, 0); }}
      {{ float8 d0 = DL(ap - 1, 0), d1 = DL(ap - 1, 1); sw_p(B, KR(ap - 1), KR({MS}), d0, d1, BC(BDR({MS})[1]), kv, 1); }}
      if (p == 0) {{ DMA_DRAIN(2, DESC(desc, 0), 0, (int)UBASE(u)); DMA_WAIT(2); }} else {{ DMA_DRAIN(3, DESC(desc, 0), {BLK}, (int)UBASE(u)); DMA_WAIT(3); }}
      {{ float8 d0 = DL({MS}, 0), d1 = DL({MS}, 1); sw_n(B, KR({MS}), QR(0), KR({MS} + 1), d0, d1, BC(BDR({MS})[1]), BC(BDR({MS} + 1)[1]), kv, o, 1); }}
    }} else {{
      sw_k0(B, KR({MS}), BC(BDR({MS})[1]), kv);
      {{ float8 d0 = DL({MS}, 0), d1 = DL({MS}, 1); sw_n(B, KR({MS}), QR(0), KR({MS} + 1), d0, d1, BC(0.0f), BC(BDR({MS} + 1)[1]), kv, o, 0); }}
    }}
    *(__global float8*)(OR(0) + j0) = o[0]; *(__global float8*)(OR(0) + j0 + 8) = o[1];
    for (int t = 1; t + 1 < {M}; t++) {{
      float8 d0 = DL({MS} + t, 0), d1 = DL({MS} + t, 1); sw_n(B, KR({MS} + t), QR(t), KR({MS} + t + 1), d0, d1, BC(0.0f), BC(BDR({MS} + t + 1)[1]), kv, o, 0);
      *(__global float8*)(OR(t) + j0) = o[0]; *(__global float8*)(OR(t) + j0 + 8) = o[1];
    }}
    {{ float8 d0 = DL({MS + M - 1}, 0), d1 = DL({MS + M - 1}, 1); sw_nl(B, KR({MS + M - 1}), QR({M - 1}), d0, d1, o);
      *(__global float8*)(OR({M - 1}) + j0) = o[0]; *(__global float8*)(OR({M - 1}) + j0 + 8) = o[1]; }}
""")
  if TR: CHAIN = _gf_tree_chain(ASM, M, MS, f"{T0('td_')} if (p == 0) {{ DMA_DRAIN(2, DESC(desc, 0), 0, (int)UBASE(u)); DMA_WAIT(2); }} else {{ DMA_DRAIN(3, DESC(desc, 0), {BLK}, (int)UBASE(u)); DMA_WAIT(3); }}{T(4, 'td_')}")
  STG = "(1 - p) * %d" % BLK if DB else "0"                                 # the prep's staging: the idle state buffer (DB) / the one buffer
  VNEW = f"LSF({VA0 + MS * VW * 4})" if VG == 1 else f"LSF({VA0})"          # where the v segment's conv lands (VG 2: a temp, then the scratch)
  PREP_C = f"""    if (jb == 0) {{                                                    /* head start: the pending updates' inputs in, q / k / v of the M new tokens */
      int fa = 1 - p, fb = 2 + p, fc = 3 - p, sb = {STG}, qk = h == h0 || {IMOD("h", REP)} == 0, ns = qk ? 3 : 1;
      DMA_WAIT(fa); DMA_WAIT(fb); DMA_WAIT(fc);
      DMA_FILL(fa, DESC(desc, 5), {KA0}, (int)sl); {"DMA_FILL(fb, DESC(desc, 6), %d, (int)(sl + 128));" % VA0 if VG == 1 else ""} DMA_FILL(fc, DESC(desc, 7), {BD0}, (int)(sl + 256));
      DMA_WAIT(fa); DMA_WAIT(fb); DMA_WAIT(fc);
      int on = qk ? {QO} + j * {DK} : {VO} + h * {DV};
      DMA_FILL(fa, DESC(desc, 3), sb, (int)(cwt + on)); DMA_FILL(fb, DESC(desc, 3), sb + {SEG}, (int)(Cr + on));
      for (int s = 0; s < ns; s++) {{                                 /* segments: q, k, v (qk) or v */
        int a = sb + (s & 1) * {AREA}, off = on, v = s == ns - 1;
        DMA_WAIT(fa); DMA_WAIT(fb); DMA_WAIT(fc);
        if (s + 1 < ns) {{
          int an = sb + ((s + 1) & 1) * {AREA}; on = s == 0 ? {KO} + j * {DK} : {VO} + h * {DV};
          DMA_FILL(fa, DESC(desc, 3), an, (int)(cwt + on)); DMA_FILL(fb, DESC(desc, 3), an + {SEG}, (int)(Cr + on));
        }}
        conv_seg(x, LSF(a), LSF({OA0}), pos, off, v ? {VNEW} : s == 0 ? QR(0) : KR({MS}){DPA}{", cs_" if PH else ""});
        if (v || {IMOD("h", REP)} == 0) DMA_DRAIN(fc, DESC(desc, 4), {OA0}, (int)(raw + off));
      }}
      if (qk) for (int t = 0; t < {M}; t++) {{ l2n(QR(t), {1.0 / DK ** 0.5!r}f); l2n(KR({MS} + t), 1.0f); }}
{"      if (qk) " + QKT("KR(%d + t)" % MS, "QR(t)").lstrip() if TR else ""}      for (int t = 0; t < {M}; t++) {{ BDR({MS} + t)[0] = bb[t][h - h0]; BDR({MS} + t)[1] = dd[t][h - h0]; }}
      DMA_WAIT(fc);                                                   /* (the raw rows' last drain: the o rows are free from here) */
      /* this pass's update inputs of head h out ({"v rows to the scratch; the slots' v at the head's end" if VG == 2 else "k, v, beta / decay"}) */
      DMA_DRAIN(fa, DESC(desc, 8), {KA0 + MS * 512}, (int)sl); DMA_DRAIN(fc, DESC(desc, 10), {BD0 + MS * 64}, (int)(sl + 256));
      {"DMA_DRAIN(fb, DESC(desc, 9), %d, (int)(sl + 128));" % (VA0 + MS * 512) if VG == 1 else "DMA_DRAIN(fb, DESC(desc, 11), %d, (int)(banks + %d + core_id * %d));" % (VA0, NG * NV * SL, M * 128)}
      DMA_WAIT(fa); DMA_WAIT(fb); DMA_WAIT(fc);
    }}
"""
  MERGE = VG == 1 and MS * GD_SLOT * 4 <= BLK and M * GD_SLOT * 4 <= BLK   # the pending / new records in one request each (they fit the staging)
  PREP_ASM = f"""    if (jb == 0) {{                                                    /* head start (asm layouts): q / k / v of the M new tokens, the pending updates' inputs */
      int fa = 1 - p, fb = 2 + p, fc = 3 - p, sb = {STG}, qk = h == h0 || {IMOD("h", REP)} == 0, ns = qk ? 3 : 1;
     {T0("t8_")} DMA_WAIT(fa); DMA_WAIT(fb); DMA_WAIT(fc);{T(8, "t8_")}
{"" if MERGE else f"""      {"DMA_FILL(fb, DESC(desc, 6), %d, (int)(sl + 128));" % VA0 if VG == 1 else ""} DMA_FILL(fc, DESC(desc, 7), {BD0}, (int)(sl + 256));   /* pending v (VG 1), beta / decay */
      DMA_WAIT(fb); DMA_WAIT(fc);
"""}      int on = {VO} + h * {DV};
      DMA_FILL(fa, DESC(desc, 3), sb, (int)(cwt + on)); DMA_FILL(fb, DESC(desc, 3), sb + {SEG}, (int)(Cr + on));{" CTS(p, p * %d, x, on);   /* CTD: the segment's C tiles into this block's (not yet filled) buffer */" % BLK if CTD else ""}
      for (int s = 0; s < ns; s++) {{                                 /* segments: v, then (qk) q into the KP rows, k into the staging's free area */
        int a = sb + (s & 1) * {AREA}, off = on;
       {T0("t9_")} DMA_WAIT(fa); DMA_WAIT(fb); DMA_WAIT(fc);{" DMA_WAIT(p);" if CTD else ""}{T(9, "t9_")}{chr(10) + "        conv_cpl(LSF(p * %d), LSF(%d), off); if (s == ns - 1) DMA_FILL(p, DESC(desc, 0), p * %d, (int)UBASE(u));   /* the buffer is free: the head's block 0 */" % (BLK, OA0, BLK) if CTD else ""}
        if (s + 1 < ns) {{
          int an = sb + ((s + 1) & 1) * {AREA}; on = s == 0 ? {QO} + j * {DK} : {KO} + j * {DK};
          DMA_FILL(fa, DESC(desc, 3), an, (int)(cwt + on)); DMA_FILL(fb, DESC(desc, 3), an + {SEG}, (int)(Cr + on));{" CTS(p, p * %d, x, on);" % BLK if CTD else ""}
        }}{f"{chr(10)}       {T0('tq_')} DMA_WAIT(fa); DMA_WAIT(fb); DMA_WAIT(fc);{T(9, 'tq_')}   /* diag dmaquiet: no DMA in flight under the conv */" if "dmaquiet" in D else ""}
       {T0("ta_")} {"" if "noconv" in D else (" __asm__ volatile(\"\" ::: \"memory\"); " if "conv2" in D else "").join([("conv_cm(LSF(a), LSF(%d), pos, s == 0 ? %s : s == 1 ? LSF(%d) : LSF(sb + %d)%s%s);" % (OA0, VNEW, KP0, AREA, DPA, ", cs_" if PH else "")) if CTD else ("conv_seg(x, LSF(a), LSF(%d), pos, off, s == 0 ? %s : s == 1 ? LSF(%d) : LSF(sb + %d)%s%s);" % (OA0, VNEW, KP0, AREA, DPA, ", cs_" if PH else ""))] * (2 if "conv2" in D else 1))}
{T(10, "ta_")}
        if (s == 0 || {IMOD("h", REP)} == 0) DMA_DRAIN(fc, DESC(desc, 4), {OA0}, (int)(raw + off));
      }}
     {T0("tb_")}
      if ({"0" if "noconv" in D else "qk"}) {{                                                        /* q / k normed as gdn_tokm_src's (one loop, literal scales), then KQN */
        __global float* qt = LSF({KP0}); __global float* kt = LSF(sb + {AREA});
        NOUNROLL for (int t = 0; t < {M}; t++) {{ l2n(qt + t * {DK}, {1.0 / DK ** 0.5!r}f); l2n(kt + t * {DK}, 1.0f); }}
{QKT("kt + t * %d" % DK, "qt + t * %d" % DK)}        for (int t = 0; t < {M}; t++) NOUNROLL for (int c = 0; c < {DK // 8}; c++) {{
          float8 k8 = *(__global float8*)(kt + t * {DK} + 8 * c), q8 = *(__global float8*)(qt + t * {DK} + 8 * c); __global float* v0 = KQ(t) + 2 * c * {M * 8};
          *(__global float8*)v0 = __builtin_shufflevector(k8, q8, 0, 1, 2, 3, 8, 9, 10, 11); *(__global float8*)(v0 + {M * 8}) = __builtin_shufflevector(k8, q8, 4, 5, 6, 7, 12, 13, 14, 15);
        }}
      }}
{T(11, "tb_")}
      for (int t = 0; t < {M}; t++) {{ BDR({MS} + t)[0] = bb[t][h - h0]; BDR({MS} + t)[1] = dd[t][h - h0]; }}
     {T0("tc2_")}
{f"""      DMA_FILL(fa, DESC(desc, 14), sb, (int)sl);                    /* the MS pending records (k | v | beta, decay) in one request (the staging is free now) */
      DMA_WAIT(fc);                                                   /* (the raw rows' last drain) */
      DMA_WAIT(fa);
      for (int u = 0; u < {MS}; u++) {{                                /* -> KP [8-row group][slot], the v rows, beta / decay */
        __global float* rc = LSF(sb) + {"pth[u]" if TR else "u"} * {GD_SLOT};
        NOUNROLL for (int g = 0; g < {DK // 8}; g++) {{ *(__global float8*)(KPV(u) + g * {MS * 8}) = *(__global float8*)(rc + 8 * g); *(__global float8*)(VR(u) + 8 * g) = *(__global float8*)(rc + {DK} + 8 * g); }}
        *(__global float8*)BDR(u) = *(__global float8*)(rc + {2 * DK}); *(__global float8*)(BDR(u) + 8) = *(__global float8*)(rc + {2 * DK} + 8);
      }}
      for (int t = 0; t < {M}; t++) {{                                  /* this pass's M records (k out of KQN | v | beta, decay), drained in one request */
        __global float* rc = LSF(sb) + t * {GD_SLOT};
        NOUNROLL for (int c = 0; c < {DK // 8}; c++) {{
          __global float* v0 = KQ(t) + 2 * c * {M * 8};
          *(__global float8*)(rc + 8 * c) = __builtin_shufflevector(*(__global float8*)v0, *(__global float8*)(v0 + {M * 8}), 0, 1, 2, 3, 8, 9, 10, 11);
          *(__global float8*)(rc + {DK} + 8 * c) = *(__global float8*)(VR({MS} + t) + 8 * c);
        }}
        *(__global float8*)(rc + {2 * DK}) = *(__global float8*)BDR({MS} + t); *(__global float8*)(rc + {2 * DK} + 8) = *(__global float8*)(BDR({MS} + t) + 8);
      }}
      DMA_DRAIN(fa, DESC(desc, 15), sb, (int)sl);
""" if MERGE else f"""      DMA_FILL(fa, DESC(desc, 5), sb, (int)sl);                     /* the pending k rows (the staging is free now) */
      DMA_WAIT(fc);                                                   /* (the raw rows' last drain: the o rows are free from here) */
      for (int t = 0; t < {M}; t++) for (int c = 0; c < {DK // 8}; c++) {{   /* the new k rows back out of KQN, for the slots */
        __global float* v0 = KQ(t) + 2 * c * {M * 8};
        *(__global float8*)(OR(t) + 8 * c) = __builtin_shufflevector(*(__global float8*)v0, *(__global float8*)(v0 + {M * 8}), 0, 1, 2, 3, 8, 9, 10, 11);
      }}
      DMA_WAIT(fa);
      for (int u = 0; u < {MS}; u++) for (int g = 0; g < {DK // 8}; g++)   /* the pending k rows -> KP [8-row group][slot] */
        *(__global float8*)(KPV(u) + g * {MS * 8}) = *(__global float8*)(LSF(sb) + {"pth[u]" if TR else "u"} * {DK} + 8 * g);
      /* this pass's update inputs of head h out ({"v rows to the scratch; the slots' v at the head's end" if VG == 2 else "k, v, beta / decay"}) */
      DMA_DRAIN(fa, DESC(desc, 8), {OA0}, (int)sl); DMA_DRAIN(fc, DESC(desc, 10), {BD0 + MS * 64}, (int)(sl + 256));
      {"DMA_DRAIN(fb, DESC(desc, 9), %d, (int)(sl + 128));" % (VA0 + MS * 512) if VG == 1 else "DMA_DRAIN(fb, DESC(desc, 11), %d, (int)(banks + %d + core_id * %d));" % (VA0, NG * NV * SL, M * 128)}
"""}      DMA_WAIT(fa); DMA_WAIT(fb); DMA_WAIT(fc);{T(12, "tc2_")}
    }}
"""
  src = V.FULL_H + TILE + EXP + f"""
#define CTX(c) ((int)((unsigned)(c) >> 4))                                /* (c a channel >= 0) */
#define CTG(c) ((int)(((unsigned)CTX(c) * 43691u) >> 17))                  /* c / 48 = (c >> 4) / 3, exact for c >> 4 < 2^15 */
#define CTA(c) ((CTG(c) * {nrb * 3} + (CTX(c) - 3 * CTG(c))) * 192 + ((int)((unsigned)(c) >> 2) & 3) * 16)   /* no div / mod */
#define LD8R(ct, c, r) F8(*(__global float4*)((ct) + CTA(c) + ((int)((unsigned)(r) >> 2)) * 64 + 4 * ((r) & 3)), *(__global float4*)((ct) + CTA(c) + ((int)((unsigned)(r) >> 2)) * 64 + 16 + 4 * ((r) & 3)))
static inline __attribute__((always_inline)) float ex1(float t) {{ return hsum8(exp2_d4(BC(t * 1.4426950408889634f))) * 0.125f; }}
static inline __attribute__((always_inline)) float softplus1(float t) {{
  float at = t < 0.0f ? -t : t; float y = ex1(-at); float s = y / (2.0f + y), s2 = s * s;
  float l = 2.0f * s * (1.0f + s2 * ({1/3!r}f + s2 * (0.2f + s2 * ({1/7!r}f + s2 * ({1/9!r}f + s2 * ({1/11!r}f + s2 * {1/13!r}f))))));
  return (t > 0.0f ? t : 0.0f) + l;
}}
#define NOHWL(i) __asm__ volatile("" : "+r"(i))                        /* an opaque loop counter: the compiler makes no hardware loop of it */
#define CYC() ({{ int c_; __asm__ volatile("mfctrl0 %0, 209" : "=r"(c_)); c_; }})   /* the TEC's cycle counter (diag "phases") */
#define NOUNROLL _Pragma("clang loop unroll(disable)")                 /* small text: code run once a head is fetched cold (~90 cycles a bundle) */
/* touch a DDR line (a vector load kept by `volatile`, its value unused): a burst of these keeps ~4 line misses in flight, where a
   load consumed at once waits for its line alone (~270 ns) -- the C tiles' rows are read cold once a head. TOUCH4: a 16-channel strip's row quad, 4 lines, 4 registers (a load into a register
   whose previous load is still outstanding waits for it) */
#define TOUCH4(p) __asm__ volatile("{{\\n ld t28, [%0+0]\\n ld t29, [%0+64]\\n}}\\n{{\\n ld t30, [%0+128]\\n ld t31, [%0+192]\\n}}\\n" :: "r"(p) : "t28", "t29", "t30", "t31")
#define TOUCH8(p, q) __asm__ volatile("{{\\n ld t24, [%0+0]\\n ld t25, [%0+64]\\n}}\\n{{\\n ld t26, [%0+128]\\n ld t27, [%0+192]\\n}}\\n{{\\n ld t28, [%1+0]\\n ld t29, [%1+64]\\n}}\\n{{\\n ld t30, [%1+128]\\n ld t31, [%1+192]\\n}}\\n" :: "r"(p), "r"(q) : "t24", "t25", "t26", "t27", "t28", "t29", "t30", "t31")
{CSEG}{CSEG2}static inline __attribute__((always_inline)) void l2n(__global float* restrict v, float scale) {{
  float8 s = BC(0.0f); NOUNROLL for (int i = 0; i < {DK}; i += 8) {{ float8 a = *(__global float8*)(v + i); s += a * a; }}
  float8 inv = BC(scale / __builtin_sqrtf(hsum8(s) + 1e-6f)); NOUNROLL for (int i = 0; i < {DK}; i += 8) *(__global float8*)(v + i) = *(__global float8*)(v + i) * inv;
}}

{SWEEPS}__kernel void {"gdn_tokl" if TR else "gdn_tokm"}(__global float* restrict out, __global float* restrict Sall, __global float* restrict Call, __global int* restrict idx,
                       __global int* restrict posb, __global float* restrict x, __global float* restrict z, __global float* restrict xin,
                       __global float* restrict wab, __global float* restrict adt, __global float* restrict cwt, __global float* restrict nw,
                       __global int* restrict desc, __global float* restrict banks, __global float* restrict rawm, __global int* restrict accp, {"__global int* restrict tree, " if TR else ""}const int core_id) {{
{"  if (core_id >= %d) return;                                       /* NTU tasks own the heads */\n" % NTU if NTU < NT else ""}  int L = idx[0], pos = posb[0], ap = accp[0]; __global float* S = Sall + L * {SZ}; __global float* Cr = Call + L * {CONV * CP};
{TREE0}{"  int ph0 = 0, ph1 = 0, ph2 = 0, ph3 = 0, ph4 = 0, ph5 = 0, ph6 = 0, ph7 = 0, ph8 = 0, ph9 = 0, ph10 = 0, ph11 = 0, ph12 = 0; int cs_[3] = {0, 0, 0}; int tk0 = CYC();" + chr(10) if PH else ""}
  cwt += L * {CONV * CP}; nw += L * {DV}; adt += L * {2 * NV}; int h0 = core_id * {HPT}, nh = core_id == {NTU - 1} ? {LH} : {HPT};
  __global float* raw = rawm + L * {M * CP};
  /* ---- a | b of the task's heads for the M rows (gdn_tokm_src's, unchanged) */
  __global float* wt = wab + L * {NTU * 2 * HPT * H} + core_id * {2 * HPT * H};
  float acc[{M}][{2 * HPT}], sq[{M}];
  for (int r = 0; r < {M}; r++) {{ sq[r] = 0.0f; for (int q = 0; q < {2 * HPT}; q++) acc[r][q] = 0.0f; }}
  {"" if "noab" in D else "DMA_FILL(2, DESC(desc, 1), %d, (int)xin); DMA_FILL(0, DESC(desc, 2), %d, (int)wt);" % (XB, WB0)}
  for (int c = 0; c < {0 if "noab" in D else NCH}; c++) {{
    int p = c & 1;
    if (c + 1 < {NCH}) {{
      if (p == 0) {{ DMA_FILL(3, DESC(desc, 1), {XB + XSZ}, (int)(xin + (c + 1) * {KBD})); DMA_FILL(1, DESC(desc, 2), {WB0 + WSZ}, (int)(wt + (c + 1) * {KBD})); }}
      else        {{ DMA_FILL(2, DESC(desc, 1), {XB}, (int)(xin + (c + 1) * {KBD})); DMA_FILL(0, DESC(desc, 2), {WB0}, (int)(wt + (c + 1) * {KBD})); }}
    }}
    if (p == 0) {{ DMA_WAIT(0); DMA_WAIT(2); }} else {{ DMA_WAIT(1); DMA_WAIT(3); }}
    for (int cc = 0; cc < {NSB}; cc++)                                /* the KB-column sub-chunks, in order */
    for (int r = 0; r < {M}; r++) {{
      __global float* WB = LSF({WB0} + p * {WSZ}) + cc * {KB};
      __global float* xc = LSF({XB} + p * {XSZ}) + r * {KBD} + cc * {KB};
      float8 sq8 = BC(0.0f), {", ".join(f"a{r_} = BC(0.0f)" for r_ in range(2 * HPT))};
      NOUNROLL for (int i = 0; i < {KB}; i += 8) {{ float8 xv = *(__global float8*)(xc + i); sq8 += xv * xv;{wacc} }}
      sq[r] += hsum8(sq8); {" ".join(f"acc[r][{q}] += hsum8(a{q});" for q in range(2 * HPT))}
    }}
  }}
  float bb[{M}][{HPT}], dd[{M}][{HPT}];
  for (int r = 0; r < {M}; r++) {{
    float inv = 1.0f / __builtin_sqrtf(sq[r] * {1.0 / H!r}f + {eps!r}f);
    for (int q = 0; q < nh; q++) {{
      bb[r][q] = 1.0f / (1.0f + ex1(-acc[r][{HPT} + q] * inv));
      dd[r][q] = ex1(adt[h0 + q] * softplus1(acc[r][q] * inv + adt[{NV} + h0 + q]));
    }}
  }}
{"  ph0 += CYC() - tk0;                                               /* phase 0: the a|b rows */" + chr(10) if PH else ""}  /* ---- the delta rule. LSRAM: state buffer(s) | k rows [MS pending | M new] | q rows [M] | v rows [MS | M] ({VW} floats) | o rows [M] | beta, decay [MS | M] (16 floats a token) */
  #define KR(t) (LSF({KA0}) + (t) * {DK})
  #define QR(t) (LSF({QA0}) + (t) * {DK})
  #define VR(t) (LSF({VA0}) + (t) * {VW})
  #define OR(t) (LSF({OA0}) + (t) * {DK})
  #define BDR(t) (LSF({BD0}) + (t) * 16)
  #define KQ(t) (LSF({KQN0}) + (t) * 8)
  #define KPV(u) (LSF({KP0}) + (u) * 8)
{"  #define PV(t) " + ("(t)" if ASM and MERGE else "pth[t]") + "                                            /* tree: pending update t's record (the previous path's t-th row) */" + chr(10) if TR else ""}  int u0 = core_id * {HPT * NB}, nu = nh * {NB};
  #define UBASE(u) (S + {IDIV("(u)", NB)} * {DK * DV} + {IMOD("(u)", NB)} * {W})
  #define ZR(c, t) ({{ int zs_ = ((int)((unsigned)(h * {DV} + (c)) >> 4)), zv_ = 6 * {IDIV("zs_", 3)} + (zs_ - 3 * {IDIV("zs_", 3)}), z0s_ = ((int)((unsigned)(h * {DV}) >> 4)), z0_ = 6 * {IDIV("z0s_", 3)} + (z0s_ - 3 * {IDIV("z0s_", 3)}); __global float* zp_ = LSF(p * {BLK}) + (zv_ - z0_) * 64 + (((c) >> 3) & 1) * 32 + 4 * (t); F8(*(__global float4*)zp_, *(__global float4*)(zp_ + 16)); }})
  #define CTS(flag, ls, base, c) {{ int s0_ = ((int)((unsigned)(c) >> 4)), g0_ = {IDIV("s0_", 3)}, r0_ = s0_ - 3 * g0_; DMA_FILL(flag, DESC(desc, r0_ == 2 ? 18 : 17), ls, (int)((base) + (6 * g0_ + r0_) * 192)); }}
  {"DMA_FILL(0, DESC(desc, 0), 0, (int)UBASE(u0));" if DB and not CTD else ""}
  for (int e = 0; e < nu; e++) {{
    NOHWL(e);                                                         /* (no hardware loop around the sweeps' own) */
    int u = u0 + e, p = {"e & 1" if DB else "0"}, h = {IDIV("u", NB)}, jb = {IMOD("u", NB)}, j0 = jb * {W}, j = {IDIV("h", REP)};
    __global float* sl = banks + (L * {NV} + h) * {SL};                 /* head h's update slots */
{"    int tp_ = jb == 0 ? CYC() : 0;" + chr(10) if PH else ""}{PREP_ASM if ASM else PREP_C}{"    if (jb == 0) ph1 += CYC() - tp_;                                 /* phase 1: the heads' starts */" + chr(10) if PH else ""}{"" if VG == 1 else f'''    if ({IMOD("jb", NBG)} == 0) {{                                              /* VG 2: column group jb / {NBG}'s halves of the v rows */
      int g = {IDIV("jb", NBG)}, fa = 1 - p, fb = 2 + p;
      DMA_WAIT(fa); DMA_WAIT(fb);
      DMA_FILL(fa, DESC(desc, 12), {VA0}, (int)(sl + 128 + g * {VW})); DMA_FILL(fb, DESC(desc, 13), {VA0 + MS * VW * 4}, (int)(banks + {NG * NV * SL} + core_id * {M * 128} + g * {VW}));
      DMA_WAIT(fa); DMA_WAIT(fb);
    }}
'''}    {f'''    if (e + 1 < nu{" && %s != 0" % IMOD("u + 1", NB) if CTD else ""}) {{
      if (p == 0) {{ DMA_WAIT(3); DMA_FILL(1, DESC(desc, 0), {BLK}, (int)UBASE(u + 1)); }}
      else        {{ DMA_WAIT(2); DMA_FILL(0, DESC(desc, 0), 0, (int)UBASE(u + 1)); }}
    }}
    {T0("tw_")} if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);{T(2, "tw_")}''' if DB else T0("tw_") + " DMA_WAIT(2); DMA_FILL(0, DESC(desc, 0), 0, (int)UBASE(u)); DMA_WAIT(0);" + T(2, "tw_")}
    __global float* B = LSF(p * {BLK});
    int jv = {IMOD("j0", VW)};                                                 /* the block's columns within the resident v rows */
    float8 kv[2], o[2];
    #define DL(t, i) ((*(__global float8*)(VR(t) + jv + 8 * (i)) - kv[i]) * BC(BDR(t)[0]))
{T0("tc_")}
{CHAIN}{T(3, "tc_") + chr(10) if PH else ""}    #undef DL
    if (jb == {NB - 1}) {{                                          /* the M tokens' o = rmsnorm(o) * nw * silu(z) -> rows 0..M-1 (fp32) */
     {T0("to_")}
      float invt[{M}];
      for (int t = 0; t < {M}; t++) {{
        float8 ss = BC(0.0f); NOUNROLL for (int c = 0; c < {DV}; c += 8) {{ float8 ov = *(__global float8*)(OR(t) + c); ss += ov * ov; }}
        invt[t] = 1.0f / __builtin_sqrtf(hsum8(ss) * {1.0 / DV!r}f + {eps!r}f);
      }}
      {"CTS(p, p * %d, z, h * %d); DMA_WAIT(p);   /* CTD: the head's z tiles into the done block's buffer */" % (BLK, DV) if CTD and "noout" not in D else ""}
      {"" if "noout" in D or "notouch" in D or CTD else "NOUNROLL for (int c = 0; c < %d; c += 32) { __global float* tp = z + CTA(h * %d + c); __global float* tq = z + CTA(h * %d + c + 16);%s }" % (DV, DV, DV, "".join(f" TOUCH8(tp + {64 * q}, tq + {64 * q});" for q in range(-(-M // 4))))}
      int tq = 0;
      for (; tq + 1 < {0 if "noout" in D else M}; tq += 2) {{          /* token pairs: two SiLU chains a body; the expression of gdn_tokm_src's */
        NOHWL(tq); __global float* oa = OR(tq); __global float* ob = OR(tq + 1); __global float* sa = LSF({VA0}) + tq * {DV}; __global float* sbr = sa + {DV};
        NOUNROLL for (int c = 0; c < {DV}; c += 8) {{
          *(__global float8*)(sa + c) = *(__global float8*)(oa + c) * BC(invt[tq]) * *(__global float8*)(nw + c) * {"BC(1.0f)" if "outnosw" in D else ("sw(BC(0.5f), BC(1.0f))" if "outnoz" in D else ("sw(ZR(c, tq), BC(1.0f))" if CTD else "sw(LD8R(z, h * %d + c, tq), BC(1.0f))" % DV))};
          *(__global float8*)(sbr + c) = *(__global float8*)(ob + c) * BC(invt[tq + 1]) * *(__global float8*)(nw + c) * {"BC(1.0f)" if "outnosw" in D else ("sw(BC(0.5f), BC(1.0f))" if "outnoz" in D else ("sw(ZR(c, tq + 1), BC(1.0f))" if CTD else "sw(LD8R(z, h * %d + c, tq + 1), BC(1.0f))" % DV))};
        }}
      }}
      for (int t = tq; t < {0 if "noout" in D else M}; t++) {{          /* (an odd last token) */
        NOHWL(t); __global float* orow = OR(t); __global float* srow = LSF({VA0}) + t * {DV};
        NOUNROLL for (int c = 0; c < {DV}; c += 8) *(__global float8*)(srow + c) = *(__global float8*)(orow + c) * BC(invt[t]) * *(__global float8*)(nw + c) * {"BC(1.0f)" if "outnosw" in D else ("sw(BC(0.5f), BC(1.0f))" if "outnoz" in D else ("sw(ZR(c, t), BC(1.0f))" if CTD else "sw(LD8R(z, h * %d + c, t), BC(1.0f))" % DV))};
      }}
      {{ int fo = 3 - p; DMA_WAIT(fo); DMA_DRAIN(fo, DESC(desc, 16), {VA0}, (int)(out + h * {DV})); }}{T(5, "to_")}   /* the M rows (staged in the head's dead v rows) by one request; waited by the next head's start */
{"" if VG == 1 else f'''      {{                                                              /* VG 2: the new v rows from the scratch to the slots (through the done block's buffer) */
        int fa = 1 - p;
        DMA_WAIT(fa); DMA_FILL(fa, DESC(desc, 11), p * {BLK}, (int)(banks + {NG * NV * SL} + core_id * {M * 128})); DMA_WAIT(fa);
        DMA_DRAIN(fa, DESC(desc, 9), p * {BLK}, (int)(sl + 128)); DMA_WAIT(fa);
      }}
'''}    }}
  }}
  DMA_WAIT(0); DMA_WAIT(1); DMA_WAIT(2); DMA_WAIT(3);       /* per flag, not DMA_WAIT_ALL (DCache.md §8b) */
{f"""  ph6 = CYC() - tk0; ph7 = nu;
  {{ __global int* dbg = (__global int*)(out + {(nrb * 12 - 1) * NV * DV}) + core_id * 16;   /* phases: o_rows' last (padding) row */
    dbg[0] = ph0; dbg[1] = ph1; dbg[2] = ph2; dbg[3] = ph3; dbg[4] = ph4; dbg[5] = ph5; dbg[6] = ph6; dbg[7] = ph7;
    dbg[8] = ph8; dbg[9] = ph9; dbg[10] = ph10; dbg[11] = ph11; dbg[12] = ph12; dbg[13] = cs_[0]; dbg[14] = cs_[1]; dbg[15] = cs_[2]; }}
""" if PH else ""}  #undef UBASE
}}"""
  return src

def gdn_flush_desc(DK, DV, MS, W=16):
  """gdn_flush_src: 0 a state block (as gdn_lsr_desc); 1 a head's MS update slots (contiguous)."""
  return V._desc_slots((DK * W * 4, W * 4, DV * 4, W * 4), (MS * GD_SLOT * 4,))

def gdn_flush_src(NV, DK, DV, NG, MS, W=16, tree=False):
  """The deferred commit's pending updates applied to every DeltaNet layer's state NOW (gdn_fast_src / gdn_defer_src leave the
  last verify pass's accepted updates in `banks` for the next verify pass to apply): what a plain decode step (gdn_tok3, which
  reads Sall) needs after a deferred verify pass -- the Q8_0 models prefill through the verify path. Per (layer, head, state block):
  the block in, the ap = accp[0] pending updates (slot t; `tree`: accp = spec_tree.path_words, slot accp[4 + t]) applied with
  gdn_defer_src's loops (the old kernel's per-token arithmetic and order: the state after them == the old bank / the fast
  kernel's committed state, bit for bit), the block back. The caller then clears the pending count (accp[0] = 0).
  args: Sall (in / out), banks, accp, desc (gdn_flush_desc). Tasks: blocks round-robin."""
  assert DK == 128 and W == 16 and MS <= GF_MS
  NB = DV // W; BLK = DK * W * 4; SL = MS * GD_SLOT; NBLK = NG * NV * NB
  return V.FULL_H + f"""
__kernel void gdn_flush(__global float* restrict Sall, __global float* restrict banks, __global int* restrict accp, __global int* restrict desc, const int core_id) {{
  int ap = accp[0];
  if (ap <= 0) return;
  __global float* B = LSF(0); __global float* PV = LSF({BLK});
  for (int u = core_id; u < {NBLK}; u += {NT}) {{
    int L = u / {NV * NB}, h = (u / {NB}) % {NV}, jb = u % {NB}, j0 = jb * {W};
    __global float* blk = Sall + (L * {NV} + h) * {DK * DV} + j0;
    DMA_FILL(0, DESC(desc, 0), 0, (int)blk); DMA_FILL(1, DESC(desc, 1), {BLK}, (int)(banks + (L * {NV} + h) * {SL})); DMA_WAIT(0); DMA_WAIT(1);
    for (int t = 0; t < ap; t++) {{
      int r = {"accp[4 + t]" if tree else "t"};
      __global float* ks = PV + r * {GD_SLOT};
      float8 db = BC(PV[r * {GD_SLOT} + 257]), kv0 = BC(0.0f), kv1 = BC(0.0f);
      for (int i = 0; i < {DK}; i++) {{
        float8 kb = BC(ks[i]); float8 s0 = *(__global float8*)(B + i * {W}) * db, s1 = *(__global float8*)(B + i * {W} + 8) * db;
        *(__global float8*)(B + i * {W}) = s0; *(__global float8*)(B + i * {W} + 8) = s1; kv0 += kb * s0; kv1 += kb * s1;
      }}
      float8 bt = BC(PV[r * {GD_SLOT} + 256]);
      float8 dl0 = (*(__global float8*)(PV + r * {GD_SLOT} + 128 + j0) - kv0) * bt, dl1 = (*(__global float8*)(PV + r * {GD_SLOT} + 128 + j0 + 8) - kv1) * bt;
      for (int i = 0; i < {DK}; i++) {{
        float8 kb = BC(ks[i]);
        float8 s0 = *(__global float8*)(B + i * {W}) + kb * dl0, s1 = *(__global float8*)(B + i * {W} + 8) + kb * dl1;
        *(__global float8*)(B + i * {W}) = s0; *(__global float8*)(B + i * {W} + 8) = s1;
      }}
    }}
    DMA_DRAIN(2, DESC(desc, 0), 0, (int)blk); DMA_WAIT(2);
  }}
}}"""

def gdn_commit_desc(CH=8192): return V._desc_slots((CH,))

def gdn_commit_tree_desc(KVB, CH=8192): return V._desc_slots((CH,), (KVB,))   # slot 1: one K (or V) cache row of a layer

def gdn_commit_tree_src(NV, DK, DV, C, CONV, M, NG, NATT, TMAX, NKV, HD, CH=8192, defer=False):
  """gdn_commit for a tree verify pass (QWEN_SPEC_TREE, spec_tree.py): the committed path's rows in `pathb` (spec_tree.path_words:
  [a, last row, kv source row, kv destination row, the path's rows 0..a-1, padding]). Every DeltaNet layer's state becomes the
  state after the path (bank `last`, unless last == M - 1: gdn_tokt left that row's state in Sall) and the ring takes the
  path's raw rows at positions pos .. pos + a - 1 (row path[i] at slot (pos + i) % CONV). A path ending on a rescue row also
  moves that row's K / V cache rows (written by attn_part at pos + row) to the position they belong to, pos + kv destination,
  in every attention layer's cache (not the MTP layer's: index NATT); DMA through LSRAM like the state. Tasks: layers.
  Under a chain path ([0..a-1], kv source -1) it does what gdn_commit does. args: Sall (out), banks, Call, rawm, Kall, Vall,
  pathb (int32 [4 + M]), posb, desc (gdn_commit_tree_desc).
  `defer` (the leaf tree, gdn_tokl): no state copy -- the next verify pass applies the path's updates from their slots (pathb)."""
  SZ, CP, KVR = NV * DK * DV, ring_pitch(C), NKV * HD; assert (SZ * 4) % CH == 0 and (C * 4) % CH == 0 and 4 * KVR <= CH
  return V.FULL_H + f"""
static inline __attribute__((always_inline)) void copy_ch(__global float* restrict dst, __global float* restrict src, int n, __global int* restrict desc) {{
  for (int c = 0; c < n; c++) {{                           /* CH bytes a request: fill, then drain; one request in flight per flag */
    DMA_FILL(0, DESC(desc, 0), 0, (int)(src + c * {CH // 4})); DMA_WAIT(0);
    DMA_DRAIN(2, DESC(desc, 0), 0, (int)(dst + c * {CH // 4})); DMA_WAIT(2);
  }}
}}
static inline __attribute__((always_inline)) void copy_row(__global float* restrict dst, __global float* restrict src, __global int* restrict desc) {{
  DMA_FILL(0, DESC(desc, 1), 0, (int)src); DMA_WAIT(0); DMA_DRAIN(2, DESC(desc, 1), 0, (int)dst); DMA_WAIT(2);
}}
__kernel void gdn_commit_tree(__global float* restrict Sall, __global float* restrict banks, __global float* restrict Call, __global float* restrict rawm,
                              __global float* restrict Kall, __global float* restrict Vall, __global int* restrict pathb, __global int* restrict posb,
                              __global int* restrict desc, const int core_id) {{
  int a = pathb[0], last = pathb[1], ksrc = pathb[2], kdst = pathb[3], pos = posb[0];
  for (int L = core_id; L < {NG}; L += {NT}) {{
{"" if defer else f"    if (last < {M - 1}) copy_ch(Sall + L * {SZ}, banks + (last * {NG} + L) * {SZ}, {SZ * 4 // CH}, desc);" + chr(10)}    for (int i = 0; i < a; i++) copy_ch(Call + (L * {CONV} + (pos + i) % {CONV}) * {CP}, rawm + (L * {M} + pathb[4 + i]) * {CP}, {C * 4 // CH}, desc);
  }}
  if (ksrc >= 0) for (int L = core_id; L < {NATT}; L += {NT}) {{      /* the rescue row's K / V rows to the path's position */
    copy_row(Kall + (L * {TMAX} + pos + kdst) * {KVR}, Kall + (L * {TMAX} + pos + ksrc) * {KVR}, desc);
    copy_row(Vall + (L * {TMAX} + pos + kdst) * {KVR}, Vall + (L * {TMAX} + pos + ksrc) * {KVR}, desc);
  }}
}}"""

RC_W = 5                                                                    # ring_commit: the words before the committed rows

def ring_commit_words(a, m, pos, path=None, NC=None, MS=12):
  """ring_commit's int32 [RC_W + MS]: a (rows committed), m (the verify pass's rows: rawm's row stride), pos (its first position),
  the K / V copy (source row, destination row; -1: none -- a chain path, or a path ending on a chain row), then the committed rows
  (the chain: 0..a-1; a tree: spec_tree.accept's path), padded with the last."""
  path = list(range(a)) if path is None else list(path); last = path[-1]
  src, dst = (last, a - 1) if NC is not None and last >= NC else (-1, -1)
  return np.array([a, m, pos, src, dst] + path + [last] * (MS - len(path)), np.int32)

def ring_commit_src(C, CONV, NG, NATT, TMAX, NKV, HD, CH=8192):
  """The deferred commit's copies (gdn_commit_tree(defer) / gdn_commit(ring_only)) with the pass's row count read from the words, so
  one kernel serves every verify geometry (and can sit inside a draft's JIT, QWEN_COMMIT_FOLD): each DeltaNet layer's ring takes the
  committed rows' raw rows at positions pos .. pos + a - 1 (row w[RC_W + i] at slot (pos + i) % CONV, rawm [NG][m][CP]); a path
  ending on a leaf moves that row's K / V cache rows (written at pos + source) to pos + destination in every trunk attention layer
  (not the MTP layer's, index NATT). The same DMA copies through LSRAM as those kernels: the same bytes. No state copy (the state
  commit is deferred to the next verify pass). Tasks: layers. args: Call (out), rawm, Kall, Vall, words (ring_commit_words), desc
  (gdn_commit_tree_desc)."""
  CP, KVR = ring_pitch(C), NKV * HD; assert (C * 4) % CH == 0 and 4 * KVR <= CH
  return V.FULL_H + f"""
static inline __attribute__((always_inline)) void copy_ch(__global float* restrict dst, __global float* restrict src, int n, __global int* restrict desc) {{
  for (int c = 0; c < n; c++) {{                           /* CH bytes a request: fill, then drain; one request in flight per flag */
    DMA_FILL(0, DESC(desc, 0), 0, (int)(src + c * {CH // 4})); DMA_WAIT(0);
    DMA_DRAIN(2, DESC(desc, 0), 0, (int)(dst + c * {CH // 4})); DMA_WAIT(2);
  }}
}}
static inline __attribute__((always_inline)) void copy_row(__global float* restrict dst, __global float* restrict src, __global int* restrict desc) {{
  DMA_FILL(0, DESC(desc, 1), 0, (int)src); DMA_WAIT(0); DMA_DRAIN(2, DESC(desc, 1), 0, (int)dst); DMA_WAIT(2);
}}
__kernel void ring_commit(__global float* restrict Call, __global float* restrict rawm, __global float* restrict Kall, __global float* restrict Vall,
                          __global int* restrict w, __global int* restrict desc, const int core_id) {{
  int a = w[0], m = w[1], pos = w[2], ksrc = w[3], kdst = w[4];
  for (int L = core_id; L < {NG}; L += {NT})
    for (int i = 0; i < a; i++) copy_ch(Call + (L * {CONV} + (pos + i) % {CONV}) * {CP}, rawm + (L * m + w[{RC_W} + i]) * {CP}, {C * 4 // CH}, desc);
  if (ksrc >= 0) for (int L = core_id; L < {NATT}; L += {NT}) {{      /* the leaf row's K / V rows to the path's position */
    copy_row(Kall + (L * {TMAX} + pos + kdst) * {KVR}, Kall + (L * {TMAX} + pos + ksrc) * {KVR}, desc);
    copy_row(Vall + (L * {TMAX} + pos + kdst) * {KVR}, Vall + (L * {TMAX} + pos + ksrc) * {KVR}, desc);
  }}
}}"""

def gdn_commit_src(NV, DK, DV, C, CONV, M, NG, CH=8192, ring_only=False):
  """After a verify pass accepted `a` of its M tokens (acc[0], 1..M): every DeltaNet layer's state becomes the state after a
  tokens (bank a - 1, unless a == M: gdn_tokm left it in place) and the ring takes the a accepted raw rows at positions pos ..
  pos + a - 1 (slot position % CONV). DDR -> LSRAM -> DDR in CH-byte DMA chunks (a row: its C floats, both at pitch CP =
  ring_pitch(C); the padding is not copied). A task owns whole layers.
  args: Sall (out), banks, Call (out), rawm, acc (int32 [1]), posb (the verify's first position), desc (gdn_commit_desc)."""
  SZ, CP = NV * DK * DV, ring_pitch(C); assert (SZ * 4) % CH == 0 and (C * 4) % CH == 0
  return V.FULL_H + f"""
static inline __attribute__((always_inline)) void copy_ch(__global float* restrict dst, __global float* restrict src, int n, __global int* restrict desc) {{
  for (int c = 0; c < n; c++) {{                           /* CH bytes a request: fill, then drain; one request in flight per flag */
    DMA_FILL(0, DESC(desc, 0), 0, (int)(src + c * {CH // 4})); DMA_WAIT(0);
    DMA_DRAIN(2, DESC(desc, 0), 0, (int)(dst + c * {CH // 4})); DMA_WAIT(2);
  }}
}}
__kernel void gdn_commit(__global float* restrict Sall, __global float* restrict banks, __global float* restrict Call, __global float* restrict rawm,
                         __global int* restrict accb, __global int* restrict posb, __global int* restrict desc, const int core_id) {{
  int a = accb[0], pos = posb[0];
  for (int L = core_id; L < {NG}; L += {NT}) {{
{"" if ring_only else "    if (a < %d) copy_ch(Sall + L * %d, banks + ((a - 1) * %d + L) * %d, %d, desc);" % (M, SZ, NG, SZ, SZ * 4 // CH) + chr(10)}    for (int i = 0; i < a; i++) copy_ch(Call + (L * {CONV} + (pos + i) % {CONV}) * {CP}, rawm + (L * {M} + i) * {CP}, {C * 4 // CH}, desc);
  }}
}}"""

def attn_decm_src(NH, NKV, HD, TMAX, ROT, eps, M):
  """attn_dec2 for M consecutive query rows (speculative decoding's verify pass) at positions pos .. pos + M - 1: the M new k / v
  rows go into the cache (tasks < NKV); row r attends t <= pos + r, the cache for t < pos and the new rows (normed and rotated
  locally: another task may be writing them) for t >= pos. A unit = (row, head). args: as attn_dec2; rows 0..M-1 read / written."""
  g = NH // NKV; scale = 1.0 / HD ** 0.5; LOG2E = 1.4426950408889634; HR = ROT // 2; QR, KR, OR = NH * 2 * HD, 2 * NKV * HD, NH * HD
  assert HD % 8 == 0 and ROT % 8 == 0 and TMAX + 2 * HD <= 8000
  return V.FULL_H + EXP + f"""
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
static inline __attribute__((always_inline)) float dot256(__global float* restrict a, __global float* restrict b) {{
  float8 acc = BC(0.0f);
  for (int i = 0; i < {HD}; i += 8) acc += *(__global float8*)(a + i) * *(__global float8*)(b + i);
  return hsum8(acc);
}}
__kernel void attn_decm(__global float* restrict o_rows, __global float* restrict q_rows, __global float* restrict kv_rows,
                        __global float* restrict qnw_all, __global float* restrict knw_all, __global float* restrict rope,
                        __global float* restrict Kall, __global float* restrict Vall, __global int* restrict idx, const int core_id) {{
  int L = idx[0], pos = idx[1];
  __global float* Kc = Kall + (L * {TMAX}) * {NKV * HD}; __global float* Vc = Vall + (L * {TMAX}) * {NKV * HD};
  __global float* qnw = qnw_all + L * {HD}; __global float* knw = knw_all + L * {HD};
  __global float* qs = LSF(0); __global float* ks = qs + {HD}; __global float* sc = ks + {HD};
  if (core_id < {NKV}) {{
    for (int r = 0; r < {M}; r++) {{
      norm_rope(ks, kv_rows + r * {KR} + core_id * {HD}, knw, rope + (pos + r) * {2 * ROT});
      for (int i = 0; i < {HD}; i += 8) {{
        *(__global float8*)(Kc + ((pos + r) * {NKV} + core_id) * {HD} + i) = *(__global float8*)(ks + i);
        *(__global float8*)(Vc + ((pos + r) * {NKV} + core_id) * {HD} + i) = *(__global float8*)(kv_rows + r * {KR} + {NKV * HD} + core_id * {HD} + i);
      }}
    }}
  }}
  for (int uu = core_id; uu < {NH * M}; uu += {NT}) {{
    int r = uu / {NH}, h = uu % {NH}, kvh = h / {g}, pr = pos + r;
    norm_rope(qs, q_rows + r * {QR} + h * {2 * HD}, qnw, rope + pr * {2 * ROT});
    float m = -1e30f;
    for (int t = 0; t <= pr; t++) {{
      __global float* kr;
      if (t < pos) kr = Kc + (t * {NKV} + kvh) * {HD};
      else {{ norm_rope(ks, kv_rows + (t - pos) * {KR} + kvh * {HD}, knw, rope + t * {2 * ROT}); kr = ks; }}
      float sv = dot256(qs, kr) * {scale!r}f; sc[t] = sv; if (sv > m) m = sv;
    }}
    float sum = 0.0f;
    for (int t = 0; t <= pr; t++) {{ float8 e = exp2_d4(BC((sc[t] - m) * {LOG2E!r}f)); sc[t] = e[0]; sum += e[0]; }}
    float8 ob[{HD // 8}]; for (int j = 0; j < {HD // 8}; j++) ob[j] = BC(0.0f);
    for (int t = 0; t <= pr; t++) {{
      __global float* vr = t < pos ? Vc + (t * {NKV} + kvh) * {HD} : kv_rows + (t - pos) * {KR} + {NKV * HD} + kvh * {HD}; float8 pb = BC(sc[t]);
      for (int j = 0; j < {HD // 8}; j++) ob[j] += pb * *(__global float8*)(vr + 8 * j);
    }}
    float8 inv = BC(1.0f / sum);
    for (int j = 0; j < {HD // 8}; j++) {{
      float8 gt = *(__global float8*)(q_rows + r * {QR} + h * {2 * HD} + {HD} + 8 * j);
      float8 sg = VRCP(BC(1.0f) + exp2_d4(BC({-LOG2E!r}f) * gt));
      *(__global float8*)(o_rows + r * {OR} + h * {HD} + 8 * j) = ob[j] * inv * sg;
    }}
  }}
}}"""

ATT_BT = 4                                                                   # attn_part: cache rows a DMA block (Kall / Vall keep this many rows of margin)

def attn_gqa_on():
  """QWEN_ATTN_GQA=1 (default): the verify / draft attention as attn_gqa.py's kv-head kernels (byte-identical records and o_rows);
  0: attn_part / attn_partt + attn_comb as before."""
  return os.environ.get("QWEN_ATTN_GQA", "1") != "0"

def attn_gqa_nte(M, NH=None, NKV=None, HD=None, TMAX=None):
  """The tasks taking attention records (attn_gqa_src's NTE): QWEN_ATTN_NTE=<n>, or `auto` (default: attn_gqa.gqa_nte's cost model --
  8 for the 27B's one-row draft passes, 12 otherwise)."""
  v = os.environ.get("QWEN_ATTN_NTE", "auto")
  if v == "auto":
    import attn_gqa as AG; n = AG.gqa_nte(NH, NKV, HD, TMAX, M); return None if n == NT else n
  return int(v) if v else None

def _attn_lsram(M, HD, CL, BT=ATT_BT): return 2 * M * HD * 4 + M * CL * 4 + 2 * BT * HD * 4 + HD * 4   # q, o, scores, K / V blocks, a new k
def _attn_cl(TMAX, P): return -(-(-(-TMAX // P) + 1) // 8) * 8                 # a slice's positions (+1 for the rounding), to 8

def attn_parts(NH, HD, TMAX):
  """Position slices a query head in attn_part: the fewest that make NH x P units fill the NT tasks evenly (a multiple of
  NT / gcd(NH, NT)) and leave a 7-row unit's buffers within LSRAM (Ornith's 16 heads: 3; the 27B's 24 at TMAX 512: 2)."""
  base = NT // math.gcd(NH, NT); P = base
  while _attn_lsram(7, HD, _attn_cl(TMAX, P)) > 32768 - 64: P += base
  return P

def attn_part_desc(NKV, HD, BT=ATT_BT): return V._desc_slots((BT * HD * 4, HD * 4, NKV * HD * 4, HD * 4))   # BT cache rows of one kv head

def attn_part_src(NH, NKV, HD, TMAX, ROT, eps, M, BT=ATT_BT):
  """attn_decm split for the TECs: a unit = (query head h, a slice c of the T = pos + M positions); the unit's M query rows take
  each K / V row once, streamed from the cache through LSRAM in blocks of BT rows (double-buffered DMA; the old kernel loaded a
  row per (row, head) with plain global loads: 16x the traffic at latency-bound speed). Pass 1 the scores (into LSRAM), then
  each row's max and exp, pass 2 the V rows. Rows t >= pos are the M new ones, normed and rotated locally from kv_rows (tasks
  < NKV write them into the cache meanwhile: no unit reads them there). Out: per (h, row, c) the unnormalised o [HD], max, sum
  -> part [NH][M][P][HD + 16]; attn_comb merges the P slices. args: part, q_rows, kv_rows, qnw_all, knw_all, rope, Kall, Vall,
  idx (layer, pos), desc (attn_part_desc). M > 7 (a unit's M rows no longer fit LSRAM): attn_part_rg_src."""
  if M > 7: return attn_part_rg_src(NH, NKV, HD, TMAX, ROT, eps, M, BT)
  g = NH // NKV; P = attn_parts(NH, HD, TMAX); scale = 1.0 / HD ** 0.5; LOG2E = 1.4426950408889634; HR = ROT // 2; QR, KR = NH * 2 * HD, 2 * NKV * HD
  CL = _attn_cl(TMAX, P)                                                      # a slice's positions, rounded to 8 (the scores' row pitch)
  QS, OS, SCO, KB0 = 0, M * HD * 4, 2 * M * HD * 4, 2 * M * HD * 4 + M * CL * 4
  KNO = KB0 + 2 * BT * HD * 4; assert KNO + HD * 4 <= 32768 - 64, KNO
  PW = HD + 16                                                                # a record: o, max, sum, padded to whole 64-byte lines (tasks must never share a line)
  return V.FULL_H + EXP + f"""
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
static inline __attribute__((always_inline)) float dot_hd(__global float* restrict a, __global float* restrict b) {{
  float8 acc = BC(0.0f);
  for (int i = 0; i < {HD}; i += 8) acc += *(__global float8*)(a + i) * *(__global float8*)(b + i);
  return hsum8(acc);
}}
__kernel void attn_part(__global float* restrict part, __global float* restrict q_rows, __global float* restrict kv_rows,
                        __global float* restrict qnw_all, __global float* restrict knw_all, __global float* restrict rope,
                        __global float* restrict Kall, __global float* restrict Vall, __global int* restrict idx, __global int* restrict desc, const int core_id) {{
  int L = idx[0], pos = idx[1], T = pos + {M};
  __global float* Kc = Kall + (L * {TMAX}) * {NKV * HD}; __global float* Vc = Vall + (L * {TMAX}) * {NKV * HD};
  __global float* qnw = qnw_all + L * {HD}; __global float* knw = knw_all + L * {HD};
  __global float* qs = LSF({QS}); __global float* os = LSF({OS}); __global float* sc = LSF({SCO}); __global float* kn = LSF({KNO});
  if (core_id < {NKV}) {{                                          /* the M new rows into the cache (read there only by later passes) */
    for (int r = 0; r < {M}; r++) {{
      norm_rope(kn, kv_rows + r * {KR} + core_id * {HD}, knw, rope + (pos + r) * {2 * ROT});
      for (int i = 0; i < {HD}; i += 8) {{
        *(__global float8*)(Kc + ((pos + r) * {NKV} + core_id) * {HD} + i) = *(__global float8*)(kn + i);
        *(__global float8*)(Vc + ((pos + r) * {NKV} + core_id) * {HD} + i) = *(__global float8*)(kv_rows + r * {KR} + {NKV * HD} + core_id * {HD} + i);
      }}
    }}
  }}
  for (int uu = core_id; uu < {NH * P}; uu += {NT}) {{
    int h = uu / {P}, c = uu % {P}, kvh = h / {g}; int t0 = c * T / {P}, t1 = (c + 1) * T / {P}, tc = t1 < pos ? t1 : pos;   /* [t0, tc) from the cache */
    for (int r = 0; r < {M}; r++) norm_rope(qs + r * {HD}, q_rows + r * {QR} + h * {2 * HD}, qnw, rope + (pos + r) * {2 * ROT});
    for (int i = 0; i < {M * CL}; i += 8) *(__global float8*)(sc + i) = BC(-3.0e38f);
    /* pass 1: scores (row r sees t <= pos + r) */
    int nb = tc > t0 ? (tc - t0 + {BT - 1}) / {BT} : 0;
    if (nb > 0) DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Kc + (t0 * {NKV} + kvh) * {HD}));
    for (int b = 0; b < nb; b++) {{
      int p = b & 1;
      if (b + 1 < nb) {{ if (p == 0) DMA_FILL(1, DESC(desc, 0), {KB0 + BT * HD * 4}, (int)(Kc + ((t0 + (b + 1) * {BT}) * {NKV} + kvh) * {HD}));
                        else        DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Kc + ((t0 + (b + 1) * {BT}) * {NKV} + kvh) * {HD})); }}
      if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
      __global float* kb = LSF({KB0} + p * {BT * HD * 4});
      for (int j = 0; j < {BT}; j++) {{
        int t = t0 + b * {BT} + j; if (t >= tc) break;
        for (int r = 0; r < {M}; r++) sc[r * {CL} + t - t0] = dot_hd(qs + r * {HD}, kb + j * {HD}) * {scale!r}f;
      }}
    }}
    for (int t = tc > t0 ? tc : t0; t < t1; t++) {{                 /* the new rows */
      norm_rope(kn, kv_rows + (t - pos) * {KR} + kvh * {HD}, knw, rope + t * {2 * ROT});
      for (int r = 0; r < {M}; r++) if (t <= pos + r) sc[r * {CL} + t - t0] = dot_hd(qs + r * {HD}, kn) * {scale!r}f;
    }}
    /* each row's max, exp, sum */
    float mx[{M}], sm[{M}];
    for (int r = 0; r < {M}; r++) {{
      float8 m8 = BC(-3.0e38f); for (int i = 0; i < {CL}; i += 8) m8 = VMAX(m8, *(__global float8*)(sc + r * {CL} + i)); mx[r] = hmax8(m8);
      float8 s8 = BC(0.0f);
      for (int i = 0; i < {CL}; i += 8) {{ float8 e = exp2_d4((*(__global float8*)(sc + r * {CL} + i) - BC(mx[r])) * BC({LOG2E!r}f)); *(__global float8*)(sc + r * {CL} + i) = e; s8 += e; }}
      sm[r] = t1 > t0 ? hsum8(s8) : 0.0f;
      for (int i = 0; i < {HD}; i += 8) *(__global float8*)(os + r * {HD} + i) = BC(0.0f);
    }}
    /* pass 2: o += p V (masked scores are exp(-huge) = 0 to float precision) */
    if (nb > 0) DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Vc + (t0 * {NKV} + kvh) * {HD}));
    for (int b = 0; b < nb; b++) {{
      int p = b & 1;
      if (b + 1 < nb) {{ if (p == 0) DMA_FILL(1, DESC(desc, 0), {KB0 + BT * HD * 4}, (int)(Vc + ((t0 + (b + 1) * {BT}) * {NKV} + kvh) * {HD}));
                        else        DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Vc + ((t0 + (b + 1) * {BT}) * {NKV} + kvh) * {HD})); }}
      if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
      __global float* vb = LSF({KB0} + p * {BT * HD * 4});
      for (int j = 0; j < {BT}; j++) {{
        int t = t0 + b * {BT} + j; if (t >= tc) break;
        for (int r = 0; r < {M}; r++) {{ float8 pb = BC(sc[r * {CL} + t - t0]); for (int i = 0; i < {HD}; i += 8) *(__global float8*)(os + r * {HD} + i) += pb * *(__global float8*)(vb + j * {HD} + i); }}
      }}
    }}
    for (int t = tc > t0 ? tc : t0; t < t1; t++) {{
      __global float* vr = kv_rows + (t - pos) * {KR} + {NKV * HD} + kvh * {HD};
      for (int r = 0; r < {M}; r++) if (t <= pos + r) {{ float8 pb = BC(sc[r * {CL} + t - t0]); for (int i = 0; i < {HD}; i += 8) *(__global float8*)(os + r * {HD} + i) += pb * *(__global float8*)(vr + i); }}
    }}
    for (int r = 0; r < {M}; r++) {{
      __global float* pp = part + ((h * {M} + r) * {P} + c) * {PW};
      for (int i = 0; i < {HD}; i += 8) *(__global float8*)(pp + i) = *(__global float8*)(os + r * {HD} + i);
      pp[{HD}] = mx[r]; pp[{HD} + 1] = sm[r];
    }}
  }}
  DMA_WAIT_ALL();
}}"""

def attn_part_rg_src(NH, NKV, HD, TMAX, ROT, eps, M, BT=ATT_BT, tree=False):
  """attn_part for 7 < M <= 12 rows: a unit = (query head h, position slice c, row group rg) -- the M rows in NRG = ceil(M / 6)
  groups of RG = ceil(M / NRG) (the last RL), since q, o and the scores of more than 7 rows do not fit LSRAM (HD = 256). Each
  group streams its slice's K / V rows once (the slice is read NRG times; the positions per slice, P, are attn_parts' for 7 rows,
  so attn_comb and the part buffer [NH][M][P][HD + 16] are unchanged). Per row the arithmetic is attn_part's in the same order.
  Ends in per-flag waits (every unit waits for its own blocks). args: as attn_part_src.
  `tree` (QWEN_SPEC_TREE, spec_tree.py; the kernel `attn_partt`): an 11th argument `tree` (spec_tree.table: int32 [2 M] =
  depth[r], anc[r]) replaces the causal rule over the pass's rows: row r is at position pos + depth[r] (its RoPE, and its k row's)
  and attends the cache t < pos plus the new rows whose bit is set in anc[r] (its ancestors and itself); the new k / v rows still
  go to the cache at pos + r (a rescue row's at its own slot: gdn_commit_tree moves a committed one to pos + depth). Under
  spec_tree.chain_table it is attn_part's arithmetic in the same order: bit-identical (the simulator gate)."""
  g = NH // NKV; P = attn_parts(NH, HD, TMAX); scale = 1.0 / HD ** 0.5; LOG2E = 1.4426950408889634; HR = ROT // 2; QR, KR = NH * 2 * HD, 2 * NKV * HD
  RP = (lambda r: f"(pos + tree[{r}])") if tree else (lambda r: f"(pos + {r})")   # row r's position
  VIS = (lambda r, t: f"((tree[{M} + {r}] >> ({t} - pos)) & 1)") if tree else (lambda r, t: f"{t} <= pos + {r}")   # row r attends new row t?
  NRG = -(-M // 6); RG = -(-M // NRG); RL = M - (NRG - 1) * RG; assert 7 < M <= 12 and 1 <= RL <= RG
  CL = _attn_cl(TMAX, P)
  QS, OS, SCO, KB0 = 0, RG * HD * 4, 2 * RG * HD * 4, 2 * RG * HD * 4 + RG * CL * 4
  KNO = KB0 + 2 * BT * HD * 4; assert KNO + HD * 4 <= 32768 - 64 and _attn_lsram(RG, HD, CL, BT) <= 32768 - 64, KNO
  PW = HD + 16
  return V.FULL_H + EXP + f"""
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
static inline __attribute__((always_inline)) float dot_hd(__global float* restrict a, __global float* restrict b) {{
  float8 acc = BC(0.0f);
  for (int i = 0; i < {HD}; i += 8) acc += *(__global float8*)(a + i) * *(__global float8*)(b + i);
  return hsum8(acc);
}}
__kernel void {"attn_partt" if tree else "attn_part"}(__global float* restrict part, __global float* restrict q_rows, __global float* restrict kv_rows,
                        __global float* restrict qnw_all, __global float* restrict knw_all, __global float* restrict rope,
                        __global float* restrict Kall, __global float* restrict Vall, __global int* restrict idx, __global int* restrict desc, {"__global int* restrict tree, " if tree else ""}const int core_id) {{
  int L = idx[0], pos = idx[1], T = pos + {M};
  __global float* Kc = Kall + (L * {TMAX}) * {NKV * HD}; __global float* Vc = Vall + (L * {TMAX}) * {NKV * HD};
  __global float* qnw = qnw_all + L * {HD}; __global float* knw = knw_all + L * {HD};
  __global float* qs = LSF({QS}); __global float* os = LSF({OS}); __global float* sc = LSF({SCO}); __global float* kn = LSF({KNO});
  if (core_id < {NKV}) {{                                          /* the M new rows into the cache (read there only by later passes) */
    for (int r = 0; r < {M}; r++) {{
      norm_rope(kn, kv_rows + r * {KR} + core_id * {HD}, knw, rope + {RP("r")} * {2 * ROT});
      for (int i = 0; i < {HD}; i += 8) {{
        *(__global float8*)(Kc + ((pos + r) * {NKV} + core_id) * {HD} + i) = *(__global float8*)(kn + i);
        *(__global float8*)(Vc + ((pos + r) * {NKV} + core_id) * {HD} + i) = *(__global float8*)(kv_rows + r * {KR} + {NKV * HD} + core_id * {HD} + i);
      }}
    }}
  }}
  for (int uu = core_id; uu < {NH * P * NRG}; uu += {NT}) {{
    int h = uu / {P * NRG}, c = (uu / {NRG}) % {P}, rg = uu % {NRG}, kvh = h / {g}; int t0 = c * T / {P}, t1 = (c + 1) * T / {P}, tc = t1 < pos ? t1 : pos;
    int r0 = rg * {RG}, nr = rg == {NRG - 1} ? {RL} : {RG};          /* the unit's rows r0 .. r0 + nr - 1 */
    for (int r = 0; r < nr; r++) norm_rope(qs + r * {HD}, q_rows + (r0 + r) * {QR} + h * {2 * HD}, qnw, rope + {RP("r0 + r")} * {2 * ROT});
    for (int i = 0; i < {RG * CL}; i += 8) *(__global float8*)(sc + i) = BC(-3.0e38f);
    /* pass 1: scores (row r sees t <= pos + r) */
    int nb = tc > t0 ? (tc - t0 + {BT - 1}) / {BT} : 0;
    if (nb > 0) DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Kc + (t0 * {NKV} + kvh) * {HD}));
    for (int b = 0; b < nb; b++) {{
      int p = b & 1;
      if (b + 1 < nb) {{ if (p == 0) DMA_FILL(1, DESC(desc, 0), {KB0 + BT * HD * 4}, (int)(Kc + ((t0 + (b + 1) * {BT}) * {NKV} + kvh) * {HD}));
                        else        DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Kc + ((t0 + (b + 1) * {BT}) * {NKV} + kvh) * {HD})); }}
      if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
      __global float* kb = LSF({KB0} + p * {BT * HD * 4});
      for (int j = 0; j < {BT}; j++) {{
        int t = t0 + b * {BT} + j; if (t >= tc) break;
        for (int r = 0; r < nr; r++) sc[r * {CL} + t - t0] = dot_hd(qs + r * {HD}, kb + j * {HD}) * {scale!r}f;
      }}
    }}
    for (int t = tc > t0 ? tc : t0; t < t1; t++) {{                 /* the new rows */
      norm_rope(kn, kv_rows + (t - pos) * {KR} + kvh * {HD}, knw, rope + {RP("t - pos") if tree else "t"} * {2 * ROT});
      for (int r = 0; r < nr; r++) if ({VIS("r0 + r", "t")}) sc[r * {CL} + t - t0] = dot_hd(qs + r * {HD}, kn) * {scale!r}f;
    }}
    /* each row's max, exp, sum */
    float mx[{RG}], sm[{RG}];
    for (int r = 0; r < nr; r++) {{
      float8 m8 = BC(-3.0e38f); for (int i = 0; i < {CL}; i += 8) m8 = VMAX(m8, *(__global float8*)(sc + r * {CL} + i)); mx[r] = hmax8(m8);
      float8 s8 = BC(0.0f);
      for (int i = 0; i < {CL}; i += 8) {{ float8 e = exp2_d4((*(__global float8*)(sc + r * {CL} + i) - BC(mx[r])) * BC({LOG2E!r}f)); *(__global float8*)(sc + r * {CL} + i) = e; s8 += e; }}
      sm[r] = t1 > t0 ? hsum8(s8) : 0.0f;
      for (int i = 0; i < {HD}; i += 8) *(__global float8*)(os + r * {HD} + i) = BC(0.0f);
    }}
    /* pass 2: o += p V (masked scores are exp(-huge) = 0 to float precision) */
    if (nb > 0) DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Vc + (t0 * {NKV} + kvh) * {HD}));
    for (int b = 0; b < nb; b++) {{
      int p = b & 1;
      if (b + 1 < nb) {{ if (p == 0) DMA_FILL(1, DESC(desc, 0), {KB0 + BT * HD * 4}, (int)(Vc + ((t0 + (b + 1) * {BT}) * {NKV} + kvh) * {HD}));
                        else        DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Vc + ((t0 + (b + 1) * {BT}) * {NKV} + kvh) * {HD})); }}
      if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
      __global float* vb = LSF({KB0} + p * {BT * HD * 4});
      for (int j = 0; j < {BT}; j++) {{
        int t = t0 + b * {BT} + j; if (t >= tc) break;
        for (int r = 0; r < nr; r++) {{ float8 pb = BC(sc[r * {CL} + t - t0]); for (int i = 0; i < {HD}; i += 8) *(__global float8*)(os + r * {HD} + i) += pb * *(__global float8*)(vb + j * {HD} + i); }}
      }}
    }}
    for (int t = tc > t0 ? tc : t0; t < t1; t++) {{
      __global float* vr = kv_rows + (t - pos) * {KR} + {NKV * HD} + kvh * {HD};
      for (int r = 0; r < nr; r++) if ({VIS("r0 + r", "t")}) {{ float8 pb = BC(sc[r * {CL} + t - t0]); for (int i = 0; i < {HD}; i += 8) *(__global float8*)(os + r * {HD} + i) += pb * *(__global float8*)(vr + i); }}
    }}
    for (int r = 0; r < nr; r++) {{
      __global float* pp = part + ((h * {M} + r0 + r) * {P} + c) * {PW};
      for (int i = 0; i < {HD}; i += 8) *(__global float8*)(pp + i) = *(__global float8*)(os + r * {HD} + i);
      pp[{HD}] = mx[r]; pp[{HD} + 1] = sm[r];
    }}
  }}
  DMA_WAIT(0); DMA_WAIT(1);                                        /* per flag (every unit waited for its own blocks already) */
}}"""

def attn_comb_src(NH, HD, M, TMAX):
  """attn_part's P slices merged per (row, head): o = sum_c e^(m_c - m) o_c / sum_c e^(m_c - m) l_c, times the sigmoid gate (the
  second half of the head's q_rows) -> o_rows. args: o_rows, part, q_rows."""
  P = attn_parts(NH, HD, TMAX); PW = HD + 16; LOG2E = 1.4426950408889634; QR, OR = NH * 2 * HD, NH * HD
  return V.FULL_H + EXP + f"""
__kernel void attn_comb(__global float* restrict o_rows, __global float* restrict part, __global float* restrict q_rows, const int core_id) {{
  for (int uu = core_id; uu < {NH * M}; uu += {NT}) {{
    int r = uu / {NH}, h = uu % {NH}; __global float* pp = part + (h * {M} + r) * {P * PW};
    float m = -3.0e38f; for (int c = 0; c < {P}; c++) if (pp[c * {PW} + {HD} + 1] > 0.0f && pp[c * {PW} + {HD}] > m) m = pp[c * {PW} + {HD}];
    float w[{P}], l = 0.0f;
    for (int c = 0; c < {P}; c++) {{ w[c] = pp[c * {PW} + {HD} + 1] > 0.0f ? exp2_d4(BC((pp[c * {PW} + {HD}] - m) * {LOG2E!r}f))[0] : 0.0f; l += w[c] * pp[c * {PW} + {HD} + 1]; }}
    float8 inv = BC(1.0f / l);
    for (int j = 0; j < {HD}; j += 8) {{
      float8 o = BC(0.0f); for (int c = 0; c < {P}; c++) o += BC(w[c]) * *(__global float8*)(pp + c * {PW} + j);
      float8 gt = *(__global float8*)(q_rows + r * {QR} + h * {2 * HD} + {HD} + j);
      *(__global float8*)(o_rows + r * {OR} + h * {HD} + j) = o * inv * VRCP(BC(1.0f) + exp2_d4(BC({-LOG2E!r}f) * gt));
    }}
  }}
}}"""

# ---- the full-attention PREFILL without cached row traffic (QWEN_ATTN_PREFILL=csrc; generate.py). The tinygrad prefill attention
# (the score kernel r_2_8_8_4_3_3_3_64_4 / r_2_16_16_4_3_3_3_64_4 and o = softmax(s) V * gate, r_2_24_64_4_4_3_24) runs as 4 tasks
# on one core and walks rows at pitches that are multiples of 8 KiB (q 24 / 48 KiB, k 4 KiB, v 8 KiB, o 24 KiB) with stack spills
# beside them: the 4-TEC same-set load + write-back burst of gdn_tokm's hang (a data-cache set-conflict hazard). Here
# every row moves by DMA through LSRAM; the only cached global reads are the small read-only tables (rope, q / k norm weights).

_NORM_ROPE = r"""
static inline __attribute__((always_inline)) void norm_rope(__global float* restrict dst, __global float* restrict src, __global float* restrict w1, __global float* restrict cs) {{
  float8 acc = BC(0.0f);
  for (int i = 0; i < {HD}; i += 8) {{ float8 v = *(__global float8*)(src + i); acc += v * v; }}
  float8 inv = BC(1.0f / __builtin_sqrtf(hsum8(acc) * {inv_hd!r}f + {eps!r}f));
  for (int i = 0; i < {HD}; i += 8) *(__global float8*)(dst + i) = *(__global float8*)(src + i) * inv * *(__global float8*)(w1 + i);
  float8 x[{R8}];
  for (int v = 0; v < {R8}; v++) x[v] = *(__global float8*)(dst + 8 * v);
  for (int v = 0; v < {H8}; v++) {{
    *(__global float8*)(dst + 8 * v) = x[v] * *(__global float8*)(cs + 8 * v) - x[v + {H8}] * *(__global float8*)(cs + {ROT} + 8 * v);
    *(__global float8*)(dst + {HR} + 8 * v) = x[v + {H8}] * *(__global float8*)(cs + {HR} + 8 * v) + x[v] * *(__global float8*)(cs + {ROT} + {HR} + 8 * v);
  }}
}}
"""
def _norm_rope(HD, ROT, eps): return _NORM_ROPE.format(HD=HD, ROT=ROT, HR=ROT // 2, R8=ROT // 8, H8=ROT // 16, inv_hd=1.0 / HD, eps=eps)

def attn_pkv_desc(HD): return V._desc_slots((HD * 4,))
def attn_pkv_src(NKV, HD, TMAX, ROT, eps, n, KR):
  """The prefill's n k / v rows into the cache of layer idx[0] at positions 0..n-1: k RMS-normed (knw_all row) and rotated, v as
  is. A unit = (position t, kv head): its k and v rows in by DMA, k normed and rotated in LSRAM, both drained by DMA (per-flag
  waits; no cached access of kv_rows / Kall / Vall). args: Kall, Vall, kv_rows (row pitch KR floats), knw_all, rope, idx, desc."""
  assert HD % 8 == 0 and ROT % 16 == 0
  return V.FULL_H + _norm_rope(HD, ROT, eps) + f"""
__kernel void attn_pkv(__global float* restrict Kall, __global float* restrict Vall, __global float* restrict kv_rows, __global float* restrict knw_all,
                       __global float* restrict rope, __global int* restrict idx, __global int* restrict desc, const int core_id) {{
  int L = idx[0];
  __global float* Kc = Kall + (L * {TMAX}) * {NKV * HD}; __global float* Vc = Vall + (L * {TMAX}) * {NKV * HD}; __global float* knw = knw_all + L * {HD};
  __global float* kin = LSF(0); __global float* kout = LSF({HD * 4});
  for (int uu = core_id; uu < {n * NKV}; uu += {NT}) {{
    int t = uu / {NKV}, h = uu % {NKV};
    DMA_FILL(0, DESC(desc, 0), 0, (int)(kv_rows + t * {KR} + h * {HD}));
    DMA_FILL(1, DESC(desc, 0), {2 * HD * 4}, (int)(kv_rows + t * {KR} + {NKV * HD} + h * {HD}));
    DMA_WAIT(0);
    norm_rope(kout, kin, knw, rope + t * {2 * ROT});
    DMA_DRAIN(2, DESC(desc, 0), {HD * 4}, (int)(Kc + (t * {NKV} + h) * {HD}));
    DMA_WAIT(1);
    DMA_DRAIN(3, DESC(desc, 0), {2 * HD * 4}, (int)(Vc + (t * {NKV} + h) * {HD}));
    DMA_WAIT(2); DMA_WAIT(3);
  }}
}}"""

def attn_ppart_desc(NKV, HD, BT=ATT_BT):
  return V._desc_slots((BT * HD * 4, HD * 4, NKV * HD * 4, HD * 4), (HD * 4,), ((HD + 16) * 4,))   # K / V blocks, a q row, a record
def attn_ppart_src(NH, NKV, HD, TMAX, ROT, eps, n, QR, BT=ATT_BT):
  """attn_part_rg for the prefill: n query rows at positions 0..n-1 against the cache rows 0..n-1 (attn_pkv wrote them), causal.
  A unit = (query head h, position slice c of [0, n), row group rg of <= 6 rows); the slice's K / V rows that the group can see
  stream through LSRAM in blocks of BT (double-buffered DMA, as attn_part); the q rows arrive by DMA and are normed / rotated in
  LSRAM; each row's record (o unnormalised [HD], max, sum; padded to HD + 16) is drained by DMA -> part [NH][n][P][HD + 16] for
  attn_pcomb. A row that sees nothing of the slice gets sum 0 (attn_comb's skip). args: part, q_rows (pitch QR), qnw_all, rope,
  Kall, Vall, idx, desc (attn_ppart_desc)."""
  g = NH // NKV; P = attn_parts(NH, HD, TMAX); scale = 1.0 / HD ** 0.5; LOG2E = 1.4426950408889634
  NRG = -(-n // 6); RG = -(-n // NRG); RL = n - (NRG - 1) * RG; assert 1 <= RL <= RG <= 6 and n <= TMAX
  CL = _attn_cl(TMAX, P); PW = HD + 16
  QS, OS = 0, RG * HD * 4; SCO = OS + RG * PW * 4; KB0 = SCO + RG * CL * 4; QST = KB0 + 2 * BT * HD * 4
  assert QST + HD * 4 <= 32768 - 64, QST
  return V.FULL_H + EXP + _norm_rope(HD, ROT, eps) + f"""
static inline __attribute__((always_inline)) float dot_hd(__global float* restrict a, __global float* restrict b) {{
  float8 acc = BC(0.0f);
  for (int i = 0; i < {HD}; i += 8) acc += *(__global float8*)(a + i) * *(__global float8*)(b + i);
  return hsum8(acc);
}}
__kernel void attn_ppart(__global float* restrict part, __global float* restrict q_rows, __global float* restrict qnw_all, __global float* restrict rope,
                         __global float* restrict Kall, __global float* restrict Vall, __global int* restrict idx, __global int* restrict desc, const int core_id) {{
  int L = idx[0];
  __global float* Kc = Kall + (L * {TMAX}) * {NKV * HD}; __global float* Vc = Vall + (L * {TMAX}) * {NKV * HD}; __global float* qnw = qnw_all + L * {HD};
  __global float* qs = LSF({QS}); __global float* os = LSF({OS}); __global float* sc = LSF({SCO}); __global float* qst = LSF({QST});
  for (int uu = core_id; uu < {NH * P * NRG}; uu += {NT}) {{
    int h = uu / {P * NRG}, c = (uu / {NRG}) % {P}, rg = uu % {NRG}, kvh = h / {g};
    int t0 = c * {n} / {P}, t1 = (c + 1) * {n} / {P};
    int r0 = rg * {RG}, nr = rg == {NRG - 1} ? {RL} : {RG};          /* the unit's rows r0 .. r0 + nr - 1 */
    int tc = t1 < r0 + nr ? t1 : r0 + nr;                           /* the slice's positions any of them sees: [t0, tc) */
    for (int r = 0; r < nr; r++) {{
      DMA_FILL(2, DESC(desc, 1), {QST}, (int)(q_rows + (r0 + r) * {QR} + h * {2 * HD})); DMA_WAIT(2);
      norm_rope(qs + r * {HD}, qst, qnw, rope + (r0 + r) * {2 * ROT});
    }}
    for (int i = 0; i < {RG * CL}; i += 8) *(__global float8*)(sc + i) = BC(-3.0e38f);
    /* pass 1: scores (row r sees t <= r0 + r) */
    int nb = tc > t0 ? (tc - t0 + {BT - 1}) / {BT} : 0;
    if (nb > 0) DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Kc + (t0 * {NKV} + kvh) * {HD}));
    for (int b = 0; b < nb; b++) {{
      int p = b & 1;
      if (b + 1 < nb) {{ if (p == 0) DMA_FILL(1, DESC(desc, 0), {KB0 + BT * HD * 4}, (int)(Kc + ((t0 + (b + 1) * {BT}) * {NKV} + kvh) * {HD}));
                        else        DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Kc + ((t0 + (b + 1) * {BT}) * {NKV} + kvh) * {HD})); }}
      if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
      __global float* kb = LSF({KB0} + p * {BT * HD * 4});
      for (int j = 0; j < {BT}; j++) {{
        int t = t0 + b * {BT} + j; if (t >= tc) break;
        for (int r = 0; r < nr; r++) if (t <= r0 + r) sc[r * {CL} + t - t0] = dot_hd(qs + r * {HD}, kb + j * {HD}) * {scale!r}f;
      }}
    }}
    /* each row's max, exp, sum -> the record's tail; o zeroed */
    for (int r = 0; r < nr; r++) {{
      float8 m8 = BC(-3.0e38f); for (int i = 0; i < {CL}; i += 8) m8 = VMAX(m8, *(__global float8*)(sc + r * {CL} + i)); float mx = hmax8(m8);
      float8 s8 = BC(0.0f);
      for (int i = 0; i < {CL}; i += 8) {{ float8 e = exp2_d4((*(__global float8*)(sc + r * {CL} + i) - BC(mx)) * BC({LOG2E!r}f)); *(__global float8*)(sc + r * {CL} + i) = e; s8 += e; }}
      for (int i = 0; i < {PW}; i += 8) *(__global float8*)(os + r * {PW} + i) = BC(0.0f);
      os[r * {PW} + {HD}] = mx; os[r * {PW} + {HD} + 1] = t0 <= r0 + r ? hsum8(s8) : 0.0f;
    }}
    /* pass 2: o += p V (masked scores are exp(-huge) = 0) */
    if (nb > 0) DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Vc + (t0 * {NKV} + kvh) * {HD}));
    for (int b = 0; b < nb; b++) {{
      int p = b & 1;
      if (b + 1 < nb) {{ if (p == 0) DMA_FILL(1, DESC(desc, 0), {KB0 + BT * HD * 4}, (int)(Vc + ((t0 + (b + 1) * {BT}) * {NKV} + kvh) * {HD}));
                        else        DMA_FILL(0, DESC(desc, 0), {KB0}, (int)(Vc + ((t0 + (b + 1) * {BT}) * {NKV} + kvh) * {HD})); }}
      if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
      __global float* vb = LSF({KB0} + p * {BT * HD * 4});
      for (int j = 0; j < {BT}; j++) {{
        int t = t0 + b * {BT} + j; if (t >= tc) break;
        for (int r = 0; r < nr; r++) {{ float8 pb = BC(sc[r * {CL} + t - t0]); for (int i = 0; i < {HD}; i += 8) *(__global float8*)(os + r * {PW} + i) += pb * *(__global float8*)(vb + j * {HD} + i); }}
      }}
    }}
    for (int r = 0; r < nr; r++) {{
      DMA_DRAIN(3, DESC(desc, 2), {OS} + r * {PW * 4}, (int)(part + ((h * {n} + r0 + r) * {P} + c) * {PW})); DMA_WAIT(3);
    }}
  }}
}}"""

def attn_pcomb_desc(NH, HD, TMAX): return V._desc_slots((attn_parts(NH, HD, TMAX) * (HD + 16) * 4,), (HD * 4,))
def attn_pcomb_src(NH, HD, TMAX, n, QR, OR):
  """attn_comb for the prefill's n rows with every row by DMA: a unit = (row r, head h): its P slice records and the gate (the
  second half of the head's q_rows) in, o = sum_c e^(m_c - m) o_c / sum_c e^(m_c - m) l_c * sigmoid(gate) drained to o_rows (row
  pitch OR). The arithmetic is attn_comb's. args: o_rows, part, q_rows (pitch QR), desc (attn_pcomb_desc)."""
  P = attn_parts(NH, HD, TMAX); PW = HD + 16; LOG2E = 1.4426950408889634; GT = P * PW * 4; OB = GT + HD * 4
  return V.FULL_H + EXP + f"""
__kernel void attn_pcomb(__global float* restrict o_rows, __global float* restrict part, __global float* restrict q_rows, __global int* restrict desc, const int core_id) {{
  __global float* pp = LSF(0); __global float* gt = LSF({GT}); __global float* ob = LSF({OB});
  for (int uu = core_id; uu < {NH * n}; uu += {NT}) {{
    int r = uu / {NH}, h = uu % {NH};
    DMA_FILL(0, DESC(desc, 0), 0, (int)(part + (h * {n} + r) * {P * PW}));
    DMA_FILL(1, DESC(desc, 1), {GT}, (int)(q_rows + r * {QR} + h * {2 * HD} + {HD}));
    DMA_WAIT(0);
    float m = -3.0e38f; for (int c = 0; c < {P}; c++) if (pp[c * {PW} + {HD} + 1] > 0.0f && pp[c * {PW} + {HD}] > m) m = pp[c * {PW} + {HD}];
    float w[{P}], l = 0.0f;
    for (int c = 0; c < {P}; c++) {{ w[c] = pp[c * {PW} + {HD} + 1] > 0.0f ? exp2_d4(BC((pp[c * {PW} + {HD}] - m) * {LOG2E!r}f))[0] : 0.0f; l += w[c] * pp[c * {PW} + {HD} + 1]; }}
    float8 inv = BC(1.0f / l);
    DMA_WAIT(1);
    for (int j = 0; j < {HD}; j += 8) {{
      float8 o = BC(0.0f); for (int c = 0; c < {P}; c++) o += BC(w[c]) * *(__global float8*)(pp + c * {PW} + j);
      *(__global float8*)(ob + j) = o * inv * VRCP(BC(1.0f) + exp2_d4(BC({-LOG2E!r}f) * *(__global float8*)(gt + j)));
    }}
    DMA_DRAIN(2, DESC(desc, 1), {OB}, (int)(o_rows + r * {OR} + h * {HD})); DMA_WAIT(2);
  }}
}}"""

class Kernels:
  """The kernels for one row geometry (nrb row blocks), registered once per (kind, shape). Every output is a persistent,
  preallocated device buffer selected by `tag`, so a tag must not be reused while its value is still needed. A hand-written kernel must not read a JIT input buffer directly (`hold` copies it)."""
  def __init__(self, nrb, rows, eps, real=None):
    self.nrb, self.rows, self.eps, self.reg, self.bufs = nrb, rows, eps, {}, {}
    self.real = rows if real is None else real        # rows holding data; the kernels skip the padding rows
    self.compact = self.real <= 12 and real is not None   # the GEMM's rows mode: A in the compact layout (row block 0's used tiles)
    self.rt = -(-self.real // 4) if self.compact else 3
    # a rotated-input model (R.HAD, bonsai2-27b): `had` = {K: the signs of a linear input of width K (device fp32)} -- the A producers
    # become had_a32 (rms / plain / swiglu), a plain one multiplying by these signs; `had_post` the factor after the transform
    # (1.0: the 1 / sqrt(HAD_B) is in the GEMM's scale tables). None: every producer as before (the other models)
    self.had, self.had_post = None, 1.0
    # small2 (Layers sets it from QWEN_SMALL2, default on; other users of Kernels, gemma4, keep the previous kernels): the RMSNorm
    # producers read a record of the rows' norms (rms_rec + rms_a32f, or the residual's own resid32q), head_top3[r] by DMA
    self.small2 = False
    self.tag_of = {}       # id(a tensor handed out: rows_buf / resid's output) -> (tag, tensor)
    self.ext = set()       # tags handed to callers by rows_buf: written outside the kernels too (pokes, the embedding, row moves)
    self.writers = {}      # tag -> the kernel kinds that wrote it (in program order, at capture)
    self.last_out = {}     # tag -> id of resid's latest returned view
  def s2(self, k="QWEN_SMALL2"):
    """The QWEN_SMALL2 kernel family `k` on this geometry (rows mode): QWEN_SMALL2=0 none; QWEN_RQ=0 / QWEN_RMS2=0 / QWEN_T3D=0 one off."""
    return self.small2 and self.compact and os.environ.get("QWEN_SMALL2", "1") == "1" and os.environ.get(k, "1") == "1"
  def _tag(self, x):
    t = self.tag_of.get(id(x)); return t[0] if t is not None and t[1] is x else None
  def _rec_src(self, x):
    """(the record's tag, guarded) when x's rows came from resid32q (the record beside them), else None. A tag also written outside
    the kernels (rows_buf: the residual stream's ping-pong rows, which the embedding fills before layer 0) needs the guard: the
    record is used when the layer index idx[0] != 0 (the stack's first layer -- layer 0 of the DeltaNet stack -- recomputes)."""
    t = self._tag(x)
    if t is None or self.writers.get(t) != {"rq"} or f"rec|{t}" not in self.bufs: return None
    return f"rec|{t}", t in self.ext
  def key(self, name, src):
    if name not in self.reg: self.reg[name] = OA.register_csrc(name.split("|")[0], src, ntasks=NT)
    return self.reg[name]
  # QWEN_AGM: a rows-mode GEMM's A operand (the tags "a_*": every one written by its producer right before the GEMMs that read
  # it) goes to the backend's A staging buffer in GM (OA.gemm_a_gm: one region every geometry shares; the GEMM's A requests then read
  # GM, off the DDR port). QWEN_AGM=0 (or ZHOUYI_GEMM_AGM=0): in DDR as before. The full-layout (non-rows) A stays in DDR.
  # Board, per call: -2 to -7 % at rows 1-4 (ternary, Q8_0, E4M3), -2 to -4 % at rows 5-8 (Q8_0, E4M3), -14 / -32 %
  # at rows 9-12 (Q8_0 / E4M3), C bit-identical; but the ternary rows 5-8 k-loop (rt 2) runs 2-10 % SLOWER with its A from GM (with the
  # text in DDR too: not instruction-fetch contention), so a rotated-input (ternary) geometry at rt 2 keeps A in DDR (QWEN_AGM=all: GM)
  def a_gm(self, tag, dt):
    v = os.environ.get("QWEN_AGM", "1")
    return tag.startswith("a_") and self.compact and dt == dtypes.uint16 and (v == "all" or (v == "1" and not (self.had is not None and self.rt == 2)))
  def buf(self, tag, n, dt):
    if tag not in self.bufs:
      g = OA.gemm_a_gm(n, dt, DEV) if self.a_gm(tag, dt) and hasattr(OA, "gemm_a_gm") else None
      self.bufs[tag] = g if g is not None else Tensor.zeros(n, device=DEV, dtype=dt).contiguous().realize()
      if os.environ.get("QWEN_PA"): print(f"[buf] {tag} n={n} pa={self.bufs[tag].uop.buffer._buf.pa:#x}", flush=True)
    assert self.bufs[tag].shape[0] == n, (tag, n, self.bufs[tag].shape)
    return self.bufs[tag]
  def rows_buf(self, tag, c):
    """A persistent fp32 [rows, c] buffer for a generic kernel's output that a hand-written kernel reads (`buf.assign(expr)`)."""
    t = self.buf(tag, self.rows * c, dtypes.float32).reshape(self.rows, c); self.ext.add(tag); self.tag_of[id(t)] = (tag, t); return t
  def hold(self, tag, t):
    """`t` (a JIT input or any tinygrad tensor) copied into the persistent buffer `tag`: a hand-written kernel must not read a JIT
    input buffer directly (its replay hangs) nor a recycled allocation."""
    n = int(np.prod(t.shape)); return self.buf(tag, n, t.dtype).assign(t.reshape(n)).realize().reshape(t.shape)
  def run(self, k, tag, n, dt, *ins, kind="k"):
    """The kernel into the persistent buffer `tag`, realised; the buffer itself is returned (a plain realised buffer, so
    a consumer's `.contiguous()` copies nothing: the copy tinygrad made of an `after` view hung the next kernel)."""
    buf = self.buf(tag, n, dt); OA.csrc_call(k, buf, *ins).realize(); self.writers.setdefault(tag, set()).add(kind); return buf
  def desc(self, dk, mk):
    if dk not in self.bufs: self.bufs[dk] = Tensor(mk(), device=DEV).realize()
    return self.bufs[dk]
  def rms_a(self, x, w1, c, tag, norm=True):
    """x fp32 [R, c] (w1 = 1 + weight fp32 [c], or (stack [L * c], idx) -- the layer's row picked on the device) -> the A layout
    halves (uint16 [nsl * nrb * 32 * 48])."""
    st = isinstance(w1, tuple)
    if self.had is not None:                     # rotated inputs: had_a32 (norm: w1 carries the signs; plain: the signs of width c)
      assert norm or c in self.had, f"rms_a(norm=False) of width {c}: no rotated linear takes it (no signs)"
      mode = "rms" if norm else "plain"; ws = (w1 if st else (w1,)) if norm else (self.had[c],)
      rs = self._rec_src(x) if norm and st and self.s2("QWEN_RQ") and self.had_dma() and self.had_fast() else None
      if rs is not None:                         # QWEN_SMALL2: the norms from the residual's record (had_fast v2 "rec", guarded)
        import had_fast as HF
        op = self.HAD_FAST_OPTS + ("rec",)
        k = self.key(f"had_a32d|fast|rec|{mode}|{c}|{self.real}|{self.had_post}", HF.had_fast_src(mode, c, self.nrb, self.eps, self.real, stacked=True, post=self.had_post, opts=op))
        d = self.desc(f"had_a32d_desc|fast|rec|{mode}|{c}", lambda: HF.had_fast_desc(mode, c, self.nrb, self.real, opts=op))
        return self.run(k, tag, self.a_size(c), dtypes.uint16, x, *ws, self.bufs[rs[0]], d)
      if self.had_dma(): return self.run(self.had_d(mode, c, st and norm), tag, self.a_size(c), dtypes.uint16, x, *ws, self.had_desc(mode, c))
      k = self.key(f"had_a32|{mode}|{c}|{self.real}|{self.compact}|{st}|{self.had_post}", had_a32_src(mode, c, self.nrb, self.eps, real=self.real, compact=self.compact, stacked=st and norm, post=self.had_post))
      return self.run(k, tag, self.a_size(c), dtypes.uint16, x, *ws)
    # rows mode with the norm: the rows through LSRAM by DMA (rms_a32d: -0.26 ms a call at 4 rows, bit-identical; never implicated
    # in the hang of gdn_tokm_src's note). QWEN_RMS_DMA=0: plain. Without the norm (the o / fc inputs): rms_a32d's phase 2 alone
    # under small_dma() (on the board, qwen3.8-27b: o 6144 at 4 rows 69.5 -> 24.4 us, the MTP fc's 10240 at 1 row 107 -> 43; the same bits)
    if self.s2("QWEN_RMS2") and c % 128 == 0 and (norm or self.small_dma()):
      # QWEN_SMALL2: rms_a32f -- the norms from a record (the residual's, resid32q; else rms_rec's, one launch: a task a row), the
      # slices by double-buffered DMA; without the norm the same phase 2 (the o / fc inputs). The same bits as rms_a32d
      d = self.desc(f"rms_a32f_desc{c}", lambda: rms_a32f_desc(c, self.real)); ws = (w1 if st else (w1 if norm else x,))
      if not norm:
        return self.run(self.key(f"rms_a32f|{c}|False|{self.real}", rms_a32f_src(c, self.eps, self.real, False)), tag, self.a_size(c), dtypes.uint16, x, x, d)
      rs = self._rec_src(x) if self.s2("QWEN_RQ") else None
      if rs is not None and rs[1] and not st: rs = None   # a guarded record needs the layer index
      if rs is None:
        rk = self.key(f"rms_rec|{c}|{self.real}", rms_rec_src(c, self.real, self.eps))
        rec = self.run(rk, "rec|rms", REC_N, dtypes.float32, x, self.desc(f"rms_rec_desc{c}", lambda: rms_rec_desc(c)))
        g = False
      else: rec, g = self.bufs[rs[0]], rs[1]
      k = self.key(f"rms_a32f|{c}|True|{self.real}|{st}|{g}", rms_a32f_src(c, self.eps, self.real, True, stacked=st, guard=g))
      return self.run(k, tag, self.a_size(c), dtypes.uint16, x, *ws, rec, d)
    if self.compact and ((norm and os.environ.get("QWEN_RMS_DMA", "1") == "1") or (not norm and self.small_dma())):
      dk = f"rms_a32d_desc{c}"
      if dk not in self.bufs: self.bufs[dk] = Tensor(rms_a32d_desc(c, self.real), device=DEV).realize()
      k = self.key(f"rms_a32d|{c}|{norm}|{self.real}|{st}", rms_a32d_src(c, self.nrb, self.eps, self.real, norm, stacked=st))
      return self.run(k, tag, self.a_size(c), dtypes.uint16, x, *(w1 if st else (w1 if norm else x,)), self.bufs[dk])
    k = self.key(f"rms_a32|{c}|{norm}|{self.real}|{self.compact}|{st}", rms_a32_src(self.rows, c, self.nrb, self.eps, norm, real=self.real, compact=self.compact, stacked=st))
    return self.run(k, tag, self.a_size(c), dtypes.uint16, x, *(w1 if st else (w1 if norm else x,)))
  # rows mode: had_a32d (every operand by DMA, lane-extract butterflies; at 4 rows rms 0.58 -> 0.075 ms a call, plain 0.29 -> 0.039,
  # swiglu 0.65 -> 0.106 on the board; the same bits as had_a32 on the simulator). QWEN_HAD_DMA=0: had_a32
  def had_dma(self): return self.compact and os.environ.get("QWEN_HAD_DMA", "1") == "1"
  # QWEN_HAD_FAST (default 1): had_fast.py's kernel (compact asm loops for the instruction fetch, no row re-fetch in rms's sums,
  # swiglu's sub-tiles by exact bytes; 4 rows on the board: rms 75 -> 46 us a call, plain 38 -> 25, swiglu 111 -> 78) with the
  # same bits as had_a32d (simulator and board gates). 0: had_a32d. "v2": the same
  # kernel with less executed text (rms 46 -> 41 us, plain 25 -> 22, swiglu 76 -> 69 at 4 rows); QWEN_HAD_V2=0: the "asm"+"swasm" one
  HAD_FAST_OPTS = ("v2",) if os.environ.get("QWEN_HAD_V2", "1") == "1" else ("asm", "swasm")
  def had_fast(self): return os.environ.get("QWEN_HAD_FAST", "1") == "1"
  def had_d(self, mode, c, st):
    if self.had_fast():
      import had_fast as HF
      return self.key(f"had_a32d|fast|{mode}|{c}|{self.real}|{st}|{self.had_post}", HF.had_fast_src(mode, c, self.nrb, self.eps, self.real, stacked=st, post=self.had_post, opts=self.HAD_FAST_OPTS))
    return self.key(f"had_a32d|{mode}|{c}|{self.real}|{st}|{self.had_post}", had_a32d_src(mode, c, self.nrb, self.eps, self.real, stacked=st, post=self.had_post))
  def had_desc(self, mode, c):
    fast = self.had_fast(); dk = f"had_a32d_desc|{'fast|' if fast else ''}{mode}|{c}"
    if dk not in self.bufs:
      if fast:
        import had_fast as HF
        self.bufs[dk] = Tensor(HF.had_fast_desc(mode, c, self.nrb, self.real, opts=self.HAD_FAST_OPTS), device=DEV).realize()
      else: self.bufs[dk] = Tensor(had_a32d_desc(mode, c, self.nrb, self.real), device=DEV).realize()
    return self.bufs[dk]
  def a_size(self, c):
    """halves of an A layout of K = c: compact (row block 0's rt tiles) in rows mode, else all row blocks' three tiles."""
    return (c // 128) * 32 * 16 * self.rt if self.compact else (c // 128) * self.nrb * 32 * 48
  def swiglu_a(self, ct, m_, tag, act="silu"):
    """The down projection's A from the gate|up tiles: fp16(silu(g) u), or `act="gelu"` fp16(gelu_tanh(g) u) (Gemma 4: geglu_a32[d])."""
    if act != "silu":
      assert self.had is None, "GeGLU on a rotated-input model"
      if self.small_dma():
        dk = f"swiglu_a32d_desc{m_}"
        if dk not in self.bufs: self.bufs[dk] = Tensor(swiglu_a32d_desc(self.nrb, self.real), device=DEV).realize()
        return self.run(self.key(f"{ACTS[act][1]}_a32d|{m_}|{self.real}", swiglu_a32d_src(m_, self.nrb, self.real, act)), tag, self.a_size(m_), dtypes.uint16, ct, self.bufs[dk])
      k = self.key(f"{ACTS[act][1]}_a32|{m_}|{self.real}|{self.compact}", swiglu_a32_src(m_, self.nrb, real=self.real, compact=self.compact, act=act))
      return self.run(k, tag, self.a_size(m_), dtypes.uint16, ct)
    if self.had is not None and self.had_dma(): return self.run(self.had_d("swiglu", m_, False), tag, self.a_size(m_), dtypes.uint16, ct, self.had_desc("swiglu", m_))
    if self.had is not None:                     # rotated inputs: the down projection's signs are in the up rows' scales
      k = self.key(f"had_a32|swiglu|{m_}|{self.real}|{self.compact}|{self.had_post}", had_a32_src("swiglu", m_, self.nrb, self.eps, real=self.real, compact=self.compact, post=self.had_post))
      return self.run(k, tag, self.a_size(m_), dtypes.uint16, ct)
    if self.small_dma():                         # rows mode: by DMA (swiglu_a32d)
      dk = f"swiglu_a32d_desc{m_}"
      if dk not in self.bufs: self.bufs[dk] = Tensor(swiglu_a32d_desc(self.nrb, self.real), device=DEV).realize()
      return self.run(self.key(f"swiglu_a32d|{m_}|{self.real}", swiglu_a32d_src(m_, self.nrb, self.real)), tag, self.a_size(m_), dtypes.uint16, ct, self.bufs[dk])
    k = self.key(f"swiglu_a32|{m_}|{self.real}|{self.compact}", swiglu_a32_src(m_, self.nrb, real=self.real, compact=self.compact))
    return self.run(k, tag, self.a_size(m_), dtypes.uint16, ct)
  # rows mode (every model's verify / draft / decode geometries): resid32d / rows32d / head_topd / swiglu_a32d / rms_a32d without
  # the norm by DMA -- 4 rows on the board (qwen3.8-27b): resid32 122 -> 27 us, rows32 (q) 347 -> 48, swiglu
  # 274 -> 61, head_top 3000 -> 280 a part; the same bits (resid32d / rows32d / swiglu_a32d / rms_a32d: simulator and board) and
  # the same ids (head_topd). QWEN_SMALL_DMA: 1 (default) every
  # rows-mode geometry, the non-rotated models' and bonsai2-27b's E4M3 drafter's included; had: the rotated-input
  # geometries only (the previous default); 0: the plain-load kernels everywhere
  def small_dma(self):
    v = os.environ.get("QWEN_SMALL_DMA", "1"); return self.compact and (v == "1" or (v == "had" and self.had is not None))
  def rd_desc(self, c):
    dk = f"rows32d_desc{c}"
    if dk not in self.bufs: self.bufs[dk] = Tensor(rows32d_desc(c, self.nrb, self.real), device=DEV).realize()
    return self.bufs[dk]
  def resid(self, x, ct, c, tag):
    if self.s2("QWEN_RQ") and c % 16 == 0:       # QWEN_SMALL2: resid32q -- the residual and its rows' norms (the record "rec|<tag>")
      k = self.key(f"resid32q|{c}|{self.real}", resid32q_src(c, self.nrb, self.real, self.eps))
      rec = self.buf(f"rec|{tag}", REC_N, dtypes.float32); d = self.desc(f"resid32q_desc{c}", lambda: resid32q_desc(c, self.nrb, self.real))
      out = self.run(k, tag, self.rows * c, dtypes.float32, x, ct, rec, d, kind="rq").reshape(self.rows, c)
      self.tag_of.pop(self.last_out.get(tag), None); self.tag_of[id(out)] = (tag, out); self.last_out[tag] = id(out)   # (the latest view only)
      return out
    if self.small_dma():                         # rows mode: by DMA (resid32d)
      k = self.key(f"resid32d|{c}|{self.real}", rows32d_src(c, self.nrb, self.real, True))
      return self.run(k, tag, self.rows * c, dtypes.float32, x, ct, self.rd_desc(c)).reshape(self.rows, c)
    k = self.key(f"resid32|{c}|{self.real}", resid32_src(c, self.nrb, real=self.real))
    return self.run(k, tag, self.rows * c, dtypes.float32, x, ct).reshape(self.rows, c)
  def unpack(self, ct, c, tag, goff=0):
    """The C tiles -> fp32 rows [rows, c]; `goff`: the linear's tiles start at group goff of `ct` (a fused GEMM's slice)."""
    if self.small_dma():
      k = self.key(f"rows32d|{c}|{self.real}|{goff}", rows32d_src(c, self.nrb, self.real, False, goff))
      return self.run(k, tag, self.rows * c, dtypes.float32, ct, self.rd_desc(c)).reshape(self.rows, c)
    k = self.key(f"rows32|{c}|{self.real}|{goff}", rows32_src(c, self.nrb, real=self.real, goff=goff))
    return self.run(k, tag, self.rows * c, dtypes.float32, ct).reshape(self.rows, c)
  def head_top(self, ct, nc, tag, top3=False):
    """A head part's top-1 per real row from its C tiles (head_top_src) -> fp32 [NT, real, 4] task partials; `top3`: the top-3
    kernel -> [NT, real, 8] (three (logit, column) pairs, the sum of exp, 0)."""
    if top3 and self.s2("QWEN_T3D") and self.real <= 8:   # QWEN_SMALL2: by DMA, the same partials
      nq = -(-self.real // 4); d = self.desc(f"head_top3d_desc{nq}", lambda: head_top3d_desc(self.nrb, nq))
      return self.run(self.key(f"head_top3d|{nc}|{self.real}", head_top3d_src(nc, self.nrb, self.real)), tag, NT * self.real * 8, dtypes.float32, ct, d)
    if top3:
      k = self.key(f"head_top3|{nc}|{self.real}", head_top_src(nc, self.nrb, self.real, top3=True))
      return self.run(k, tag, NT * self.real * 8, dtypes.float32, ct)
    if self.small_dma() and self.real <= 8:          # rows mode: the C tiles by DMA (head_topd; m > 4: row quads 0 and 1)
      dk = f"head_topd_desc{self.nrb}" if self.real <= 4 else f"head_topd_desc{self.nrb}q2"
      if dk not in self.bufs: self.bufs[dk] = Tensor(head_topd_desc(nc, self.nrb) if self.real <= 4 else head_topd_desc(nc, self.nrb, 2), device=DEV).realize()
      k = self.key(f"head_topd|{nc}|{self.real}", head_topd_src(nc, self.nrb, self.real))
      return self.run(k, tag, NT * self.real * 4, dtypes.float32, ct, self.bufs[dk])
    k = self.key(f"head_top|{nc}|{self.real}", head_top_src(nc, self.nrb, self.real))
    return self.run(k, tag, NT * self.real * 4, dtypes.float32, ct)
  def head_top3_row(self, ct, nc, tag, row):
    """A head part's top-3 (logit, column) on one row (head_top3_row_src) -> fp32 [NT, 1, 8] task partials (lane 6 = 0)."""
    if self.s2("QWEN_T3D"):                       # QWEN_SMALL2: by DMA (its row quad only), the same partials
      d = self.desc("head_top3d_desc1", lambda: head_top3d_desc(self.nrb, 1))
      return self.run(self.key(f"head_top3rd|{nc}|{row}", head_top3d_src(nc, self.nrb, 1, row=row)), tag, NT * 8, dtypes.float32, ct, d)
    k = self.key(f"head_top3r|{nc}|{row}", head_top3_row_src(nc, self.nrb, row))
    return self.run(k, tag, NT * 8, dtypes.float32, ct)
  def call(self, name, src, out, *ins):
    """A registered csrc kernel into an existing buffer `out` (persistent, realised), returned."""
    OA.csrc_call(self.key(name, src), out, *ins).realize(); return out
  def head_reduce(self, c0s, tag="ids_head"):
    """The head's ids and probabilities on the device (head_reduce_src) from the parts' partials top_head0.. -> int32 [16 ceil(2 m / 16)]."""
    m = self.real; dk = f"head_reduce_desc{m}"
    if dk not in self.bufs: self.bufs[dk] = Tensor(head_reduce_desc(m), device=DEV).realize()
    k = self.key(f"head_reduce|{m}|{tuple(c0s)}", head_reduce_src(m, tuple(c0s)))
    return self.run(k, tag, 16 * -(-2 * m // 16), dtypes.int32, *[self.bufs[f"top_head{j}"] for j in range(len(c0s))], self.bufs[dk])
  def gemv(self, x, w, kdim, n, tag, norm=False, idx=None):
    """x fp32 [rows, kdim] @ w fp32 [n, kdim]^T -> [rows, n] (small n); `norm`: the rows RMS-normalised first (weight folded into w);
    `idx` (int32 [1] buffer): w is a stack [L, n, kdim] and idx picks the layer."""
    k = self.key(f"gemv32|{kdim}|{n}|{norm}|{idx is not None}|{self.real}", gemv32_src(self.real, kdim, n, norm=norm, eps=self.eps, stacked=idx is not None))
    return self.run(k, tag, self.rows * n, dtypes.float32, x, w, *([idx] if idx is not None else [])).reshape(self.rows, n)
  def gdn_tok(self, Sall, Call, idx, posb, qkv_rows, z_rows, bd, cwt, nw, NV, NK, DK, DV, C, CONV):
    """The DeltaNet token step (conv, norms, the delta rule on the state `idx` of `Sall` / the ring of `Call`, the gated output
    norm) -> the persistent `o_rows` [rows, NV*DV] with the token in row 0. `cwt` [L, CONV, ring_pitch(C)] and `nw` [L, DV] are per-layer stacks."""
    kk = self.key("gdn_tok", gdn_tok_src(NV, NK, DK, DV, C, CONV, self.rows, self.eps))
    return self.run(kk, "o_rows", self.rows * NV * DV, dtypes.float32, Sall, Call, idx, posb, qkv_rows, z_rows, bd, cwt, nw).reshape(self.rows, NV * DV)
  def gdn_step(self, Sall, idx, q, k, v, beta, decay, NV, DK, DV):
    """One token of the gated delta rule on the state `idx` of the persistent stack `Sall` [L, NV, DK, DV], in place; returns o [NV, DV]."""
    kk = self.key("gdn_step", gdn_step_src(NV, DK, DV))
    o = self.buf("gdn_o", NV * DV, dtypes.float32)
    OA.csrc_call(kk, o, Sall, idx, q, k, v, beta, decay).realize(); return o.reshape(NV, DV)
  def gdn_prefill(self, S, q, k, v, beta, decay, NV, DK, DV, n):
    """The delta rule over n tokens on the persistent state `S` [NV, DK, DV] in place; returns o [n, NV, DV]."""
    kk = self.key(f"gdn_prefill|{n}", gdn_prefill_src(NV, DK, DV, n))
    o = self.buf(f"gdn_o_pre{n}", n * NV * DV, dtypes.float32)
    OA.csrc_call(kk, o, S, q, k, v, beta, decay).realize(); return o.reshape(n, NV, DV)
  def attn_decode(self, q, Kall, Vall, knew, vnew, idx, NH, NKV, HD, TMAX):
    """Decode attention for one token (q [NH, HD]) against the persistent cache stack, writing the new k / v rows; returns o [NH, HD]."""
    kk = self.key(f"attn_decode|{TMAX}", attn_decode_src(NH, NKV, HD, TMAX))
    o = self.buf("attn_o", NH * HD, dtypes.float32)
    OA.csrc_call(kk, o, q, Kall, Vall, knew, vnew, idx).realize(); return o.reshape(NH, HD)
  def attn_dec2(self, q_rows, kv_rows, qnw, knw, rope, Kall, Vall, idx, NH, NKV, HD, TMAX, ROT):
    """The fused decode attention (attn_dec2_src) -> the persistent o_rows [rows, NH*HD], the token in row 0."""
    if attn_gqa_on() and 4 * 1024 + 10 * HD * 4 + TMAX * 32 <= 32768 - 64 and NH // NKV <= 8:   # attn_dec3 (attn_gqa.py): byte-identical o_rows / cache
      import attn_gqa as AG
      U = int(os.environ.get("QWEN_ATTN_DEC_U", "1"))
      if f"attn_dec3_desc" not in self.bufs: self.bufs["attn_dec3_desc"] = Tensor(AG.attn_dec3_desc(NH, NKV, HD), device=DEV).realize()
      kk = self.key(f"attn_dec2|kv|{TMAX}|{U}", AG.attn_dec3_src(NH, NKV, HD, TMAX, ROT, self.eps, U=U))
      return self.run(kk, "o_rows", self.rows * NH * HD, dtypes.float32, q_rows, kv_rows, qnw, knw, rope, Kall, Vall, idx, self.bufs["attn_dec3_desc"]).reshape(self.rows, NH * HD)
    kk = self.key(f"attn_dec2|{TMAX}", attn_dec2_src(NH, NKV, HD, TMAX, ROT, self.eps))
    return self.run(kk, "o_rows", self.rows * NH * HD, dtypes.float32, q_rows, kv_rows, qnw, knw, rope, Kall, Vall, idx).reshape(self.rows, NH * HD)
  def gdn_lsr(self, Sall, idx, q, k, v, beta, decay, NV, DK, DV, n, diag=None):
    """The delta rule over n tokens on the state `idx` of `Sall` [L, NV, DK, DV], in place, the state in LSRAM; returns o [n, NV, DV]."""
    kk = self.key(f"gdn_lsr|{n}|{diag}", gdn_lsr_src(NV, DK, DV, n, diag=diag))
    if "gdn_lsr_desc" not in self.bufs: self.bufs["gdn_lsr_desc"] = Tensor(gdn_lsr_desc(DK, DV), device=DEV).realize()
    o = self.buf(f"gdn_lsr_o{n}", n * NV * DV, dtypes.float32)
    OA.csrc_call(kk, o, Sall, idx, q, k, v, beta, decay, self.bufs["gdn_lsr_desc"]).realize(); return o.reshape(n, NV, DV)
  def gdn_tokm(self, Sall, Call, idx, posb, qkv_ct, z_ct, xin, wab, adt, cwt, nw, banks, rawm, NV, NK, DK, DV, C, CONV, H, NG, xoff=0, zoff=0, rows=False, defer=None):
    """gdn_tok3 for the `real` rows as consecutive tokens (the verify pass) -> `a_o` rows 0..real-1; the per-token state banks and
    the raw rows for gdn_commit. M > GDN_WAVE: the tokens in waves (gdn_tokm_waves_src, its DMA-staged preparation always).
    `xoff` / `zoff`: the qkv / z tiles start at those groups of their buffers (the fused qkv|z GEMM: one buffer, z at zoff).
    `rows`: the outputs as fp32 rows in the persistent `o_rows` instead (a rotated-input model; gdn_tokm_src(rows_out=True))."""
    M = self.real; assert self.compact and 1 <= M <= 12
    diag = tuple(x for x in os.environ.get("GDN_TOKM_DIAG", "").split(",") if x)   # timing diagnostics (gdn_tokm_src)
    prep = gdn_prep()                                                              # "dma" (default) / "dma-lsarr": gdn_tokm_src's DMA-staged preparation; "fast" / "cached"
    if defer is not None and gdn_fast() != "off":   # (MS, accb): the deferred commit, fused sweeps (gdn_fast_src), M <= 8
      assert not diag                                # (not rows: o_proj's compact A straight from the kernel, gdn_fast_src(a_out))
      MS, accb = defer; sw = gdn_fast(); fd = tuple(x for x in os.environ.get("GDN_FAST_DIAG", "").split(",") if x)   # timing diagnostics
      kk = self.key(f"gdn_tokm|{M}|fast{MS}|{sw}|{fd}|{xoff}|{zoff}{'' if rows else '|a'}", ct_goff(gdn_fast_src(NV, NK, DK, DV, C, CONV, H, self.nrb, self.eps, M, NG, MS, sweeps=sw, diag=fd, a_out=not rows), self.nrb, (xoff, zoff), ("x", "z")))
      dk = f"gdn_fast_desc{M}|{MS}{'' if rows else '|a'}"
      if dk not in self.bufs: self.bufs[dk] = Tensor(gdn_fast_desc(DK, DV, H, gdn_hpt(NV), M, MS, C, CONV, NV=NV, a_out=not rows), device=DEV).realize()
      if not rows: return self.run(kk, "a_o", self.a_size(NV * DV), dtypes.uint16, Sall, Call, idx, posb, qkv_ct, z_ct, xin, wab, adt, cwt, nw, self.bufs[dk], banks, rawm, accb)
      return self.run(kk, "o_rows", self.rows * NV * DV, dtypes.float32, Sall, Call, idx, posb, qkv_ct, z_ct, xin, wab, adt, cwt, nw, self.bufs[dk], banks, rawm, accb).reshape(self.rows, NV * DV)
    if defer is not None:    # (MS, accb): the commit deferred to the next verify pass (gdn_defer_src; QWEN_GDN_FAST=off)
      assert rows and prep == "dma" and not diag
      MS, accb = defer
      kk = self.key(f"gdn_tokm|{M}|defer{MS}|{xoff}|{zoff}", ct_goff(gdn_defer_src(gdn_tokm_src(NV, NK, DK, DV, C, CONV, H, self.nrb, self.eps, M, NG, rows_out=True), NV, M, MS), self.nrb, (xoff, zoff), ("x", "z")))
      dk = f"gdn_tokm_desc{M}defer{MS}"
      if dk not in self.bufs: self.bufs[dk] = Tensor(np.concatenate([gdn_tokm_desc(DK, DV, H, gdn_hpt(NV), M, C=C, CONV=CONV), gdn_defer_desc(M, MS)]), device=DEV).realize()
      return self.run(kk, "o_rows", self.rows * NV * DV, dtypes.float32, Sall, Call, idx, posb, qkv_ct, z_ct, xin, wab, adt, cwt, nw, self.bufs[dk], banks, rawm, accb).reshape(self.rows, NV * DV)
    kk = self.key(f"gdn_tokm|{M}|{diag}|{gdn_prep_key(prep)}|{xoff}|{zoff}{'|rows' if rows else ''}",
                  ct_goff(gdn_tokm_src(NV, NK, DK, DV, C, CONV, H, self.nrb, self.eps, M, NG, diag=diag, rows_out=rows), self.nrb, (xoff, zoff), ("x", "z")))
    dk = f"gdn_tokm_desc{M}{prep if prep.startswith('dma') else ''}"
    if dk not in self.bufs:
      self.bufs[dk] = Tensor(gdn_tokm_desc(DK, DV, H, gdn_hpt(NV), M, **(dict(C=C, CONV=CONV) if prep.startswith("dma") or M > GDN_WAVE else {})), device=DEV).realize()
    if rows:                 # fp32 rows into the persistent o_rows (rows 0..M-1), for had_a32's o_proj A (gdn_tokm_src(rows_out=True))
      return self.run(kk, "o_rows", self.rows * NV * DV, dtypes.float32, Sall, Call, idx, posb, qkv_ct, z_ct, xin, wab, adt, cwt, nw, self.bufs[dk], banks, rawm).reshape(self.rows, NV * DV)
    return self.run(kk, "a_o", self.a_size(NV * DV), dtypes.uint16, Sall, Call, idx, posb, qkv_ct, z_ct, xin, wab, adt, cwt, nw, self.bufs[dk], banks, rawm)
  def gdn_commit(self, Sall, banks, Call, rawm, accb, posb, NV, DK, DV, C, CONV, M, NG, ring_only=False):
    """Every DeltaNet layer to the state after the accepted tokens (`accb` [1] = a) and their raw rows into the rings."""
    kk = self.key(f"gdn_commit|{M}{'|ring' if ring_only else ''}", gdn_commit_src(NV, DK, DV, C, CONV, M, NG, ring_only=ring_only))
    if "gdn_commit_desc" not in self.bufs: self.bufs["gdn_commit_desc"] = Tensor(gdn_commit_desc(), device=DEV).realize()
    OA.csrc_call(kk, Sall, banks, Call, rawm, accb, posb, self.bufs["gdn_commit_desc"]).realize()
  # ---- QWEN_SPEC_TREE (spec_tree.py): the tree verify's kernels -- gdn_tokm in waves with the rescue rows as one-token waves from
  # their parents' banks, attn_part with the tree table, and the commit along the committed path (the K / V rows moved too)
  def gdn_tokt(self, Sall, Call, idx, posb, qkv_ct, z_ct, xin, wab, adt, cwt, nw, banks, rawm, treeb, NV, NK, DK, DV, C, CONV, H, NG, xoff=0, zoff=0):
    """gdn_tokm for a tree of `real` rows (gdn_tokm_waves_src(tree=True): rows 0..GDN_WAVE-1 the chain, the rest rescue rows), the
    tree table in `treeb` (spec_tree.table, int32 [2 real])."""
    M = self.real; assert self.compact and GDN_WAVE < M <= 12, "the tree verify: GDN_WAVE + 1 .. 12 rows"
    diag = tuple(x for x in os.environ.get("GDN_TOKM_DIAG", "").split(",") if x)
    kk = self.key(f"gdn_tokt|{M}|{diag}|{xoff}|{zoff}", ct_goff(gdn_tokm_waves_src(NV, NK, DK, DV, C, CONV, H, self.nrb, self.eps, M, NG, diag=diag, tree=True), self.nrb, (xoff, zoff), ("x", "z")))
    dk = f"gdn_tokt_desc{M}"
    if dk not in self.bufs: self.bufs[dk] = Tensor(gdn_tokm_desc(DK, DV, H, gdn_hpt(NV), M, C=C, CONV=CONV, tree=True), device=DEV).realize()
    return self.run(kk, "a_o", self.a_size(NV * DV), dtypes.uint16, Sall, Call, idx, posb, qkv_ct, z_ct, xin, wab, adt, cwt, nw, self.bufs[dk], banks, rawm, treeb)
  def gdn_tokl(self, Sall, Call, idx, posb, qkv_ct, z_ct, xin, wab, adt, cwt, nw, banks, rawm, pathb, treeb, NV, NK, DK, DV, C, CONV, H, NG, MS, xoff=0, zoff=0, rows=True):
    """The leaf tree's DeltaNet on the `real` rows -> fp32 rows in the persistent o_rows (as gdn_tokm(rows, defer)); `pathb`: the
    previous pass's committed path (its pending updates), `treeb`: this pass's tree table. QWEN_GDN_FAST=asm|c (default asm):
    gdn_fast_src(tree=True) (the fused sweeps, M <= 8); off: gdn_tokl_src (M <= 5). The same arguments, slots and results.
    `rows` False (the fast kernel only): o_proj's compact A in `a_o` instead (gdn_fast_src(a_out); the models without a rotated input)."""
    M = self.real; assert self.compact and (rows or gdn_fast() != "off"), "gdn_tokl: o_proj's A only from the fast kernel"
    if gdn_fast() != "off":
      sw = gdn_fast()
      kk = self.key(f"gdn_tokl|{M}|fast{MS}|{sw}|{xoff}|{zoff}{'' if rows else '|a'}", ct_goff(gdn_fast_src(NV, NK, DK, DV, C, CONV, H, self.nrb, self.eps, M, NG, MS, sweeps=sw, tree=True, a_out=not rows), self.nrb, (xoff, zoff), ("x", "z")))
      dk = f"gdn_fast_desc{M}|{MS}{'' if rows else '|a'}"
      if dk not in self.bufs: self.bufs[dk] = Tensor(gdn_fast_desc(DK, DV, H, gdn_hpt(NV), M, MS, C, CONV, NV=NV, a_out=not rows), device=DEV).realize()
      if not rows: return self.run(kk, "a_o", self.a_size(NV * DV), dtypes.uint16, Sall, Call, idx, posb, qkv_ct, z_ct, xin, wab, adt, cwt, nw, self.bufs[dk], banks, rawm, pathb, treeb)
      return self.run(kk, "o_rows", self.rows * NV * DV, dtypes.float32, Sall, Call, idx, posb, qkv_ct, z_ct, xin, wab, adt, cwt, nw, self.bufs[dk], banks, rawm, pathb, treeb).reshape(self.rows, NV * DV)
    kk = self.key(f"gdn_tokl|{M}|{MS}|{xoff}|{zoff}", ct_goff(gdn_tokl_src(NV, NK, DK, DV, C, CONV, H, self.nrb, self.eps, M, MS, NG), self.nrb, (xoff, zoff), ("x", "z")))
    dk = f"gdn_tokm_desc{M}defer{MS}"
    if dk not in self.bufs: self.bufs[dk] = Tensor(np.concatenate([gdn_tokm_desc(DK, DV, H, gdn_hpt(NV), M, C=C, CONV=CONV), gdn_defer_desc(M, MS)]), device=DEV).realize()
    return self.run(kk, "o_rows", self.rows * NV * DV, dtypes.float32, Sall, Call, idx, posb, qkv_ct, z_ct, xin, wab, adt, cwt, nw, self.bufs[dk], banks, rawm, pathb, treeb).reshape(self.rows, NV * DV)
  def gdn_flush(self, Sall, banks, accp, NV, DK, DV, NG, MS, tree=False):
    """The deferred commit's pending updates (`accp`: accb, or the leaf tree's pathb) applied to every layer's state now
    (gdn_flush_src: before a plain decode step); the caller clears the pending count."""
    kk = self.key(f"gdn_flush|{MS}{'|tree' if tree else ''}", gdn_flush_src(NV, DK, DV, NG, MS, tree=tree))
    dk = f"gdn_flush_desc{MS}"
    if dk not in self.bufs: self.bufs[dk] = Tensor(gdn_flush_desc(DK, DV, MS), device=DEV).realize()
    OA.csrc_call(kk, Sall, banks, accp, self.bufs[dk]).realize()
  def gdn_commit_tree(self, Sall, banks, Call, rawm, Kall, Vall, pathb, posb, NV, DK, DV, C, CONV, M, NG, NATT, TMAX, NKV, HD, defer=False):
    """Every DeltaNet layer to the state after the committed path (`pathb`: spec_tree.path_words), the path's raw rows into the
    rings, and a committed rescue row's K / V cache rows to its position in every attention layer."""
    kk = self.key(f"gdn_commit_tree|{M}|{NATT}|{TMAX}{'|defer' if defer else ''}", gdn_commit_tree_src(NV, DK, DV, C, CONV, M, NG, NATT, TMAX, NKV, HD, defer=defer))
    if "gdn_commit_tree_desc" not in self.bufs: self.bufs["gdn_commit_tree_desc"] = Tensor(gdn_commit_tree_desc(NKV * HD * 4), device=DEV).realize()
    OA.csrc_call(kk, Sall, banks, Call, rawm, Kall, Vall, pathb, posb, self.bufs["gdn_commit_tree_desc"]).realize()
  def ring_commit(self, Call, rawm, Kall, Vall, words, C, CONV, NG, NATT, TMAX, NKV, HD):
    """The deferred commit's ring (and leaf K / V) copies for any verify geometry (ring_commit_src; `words`: ring_commit_words)."""
    kk = self.key(f"ring_commit|{NATT}|{TMAX}", ring_commit_src(C, CONV, NG, NATT, TMAX, NKV, HD))
    if "gdn_commit_tree_desc" not in self.bufs: self.bufs["gdn_commit_tree_desc"] = Tensor(gdn_commit_tree_desc(NKV * HD * 4), device=DEV).realize()
    OA.csrc_call(kk, Call, rawm, Kall, Vall, words, self.bufs["gdn_commit_tree_desc"]).realize()
  def attn_tree(self, q_rows, kv_rows, qnw, knw, rope, Kall, Vall, idx, treeb, NH, NKV, HD, TMAX, ROT):
    """attn_decm (attn_part + attn_comb) for a tree of `real` rows: attn_part_rg_src(tree=True) with the tree table `treeb`."""
    M = self.real; P = attn_parts(NH, HD, TMAX); assert 2 <= M <= 12, "the tree verify's attention: 2 .. 12 rows"
    if attn_gqa_on(): return self._attn_gqa(q_rows, kv_rows, qnw, knw, rope, Kall, Vall, idx, treeb, NH, NKV, HD, TMAX, ROT)
    if "attn_part_desc" not in self.bufs: self.bufs["attn_part_desc"] = Tensor(attn_part_desc(NKV, HD), device=DEV).realize()
    kp = self.key(f"attn_partt|{TMAX}|{M}", attn_part_rg_src(NH, NKV, HD, TMAX, ROT, self.eps, M, tree=True) if M > 7 else attn_part_tree_src(NH, NKV, HD, TMAX, ROT, self.eps, M))
    part = self.run(kp, "attn_part", NH * M * P * (HD + 16), dtypes.float32, q_rows, kv_rows, qnw, knw, rope, Kall, Vall, idx, self.bufs["attn_part_desc"], treeb)
    kc = self.key(f"attn_comb|{TMAX}|{M}", attn_comb_src(NH, HD, M, TMAX))
    return self.run(kc, "o_rows", self.rows * NH * HD, dtypes.float32, part, q_rows).reshape(self.rows, NH * HD)
  def attn_decm(self, q_rows, kv_rows, qnw, knw, rope, Kall, Vall, idx, NH, NKV, HD, TMAX, ROT):
    """attn_dec2 for the `real` rows at positions pos .. pos + real - 1 -> the persistent o_rows: attn_part + attn_comb (the cache
    streamed through LSRAM by (head, position slice) units); QWEN_ATTN=decm: the one-kernel attn_decm_src."""
    if os.environ.get("QWEN_ATTN", "split") == "split" and attn_gqa_on():
      return self._attn_gqa(q_rows, kv_rows, qnw, knw, rope, Kall, Vall, idx, None, NH, NKV, HD, TMAX, ROT)
    if os.environ.get("QWEN_ATTN", "split") == "split":
      M = self.real; P = attn_parts(NH, HD, TMAX)
      if "attn_part_desc" not in self.bufs: self.bufs["attn_part_desc"] = Tensor(attn_part_desc(NKV, HD), device=DEV).realize()
      kp = self.key(f"attn_part|{TMAX}|{M}", attn_part_src(NH, NKV, HD, TMAX, ROT, self.eps, M))
      part = self.run(kp, "attn_part", NH * M * P * (HD + 16), dtypes.float32, q_rows, kv_rows, qnw, knw, rope, Kall, Vall, idx, self.bufs["attn_part_desc"])
      kc = self.key(f"attn_comb|{TMAX}|{M}", attn_comb_src(NH, HD, M, TMAX))
      return self.run(kc, "o_rows", self.rows * NH * HD, dtypes.float32, part, q_rows).reshape(self.rows, NH * HD)
    kk = self.key(f"attn_decm|{TMAX}|{self.real}", attn_decm_src(NH, NKV, HD, TMAX, ROT, self.eps, self.real))
    return self.run(kk, "o_rows", self.rows * NH * HD, dtypes.float32, q_rows, kv_rows, qnw, knw, rope, Kall, Vall, idx).reshape(self.rows, NH * HD)
  def _attn_gqa(self, q_rows, kv_rows, qnw, knw, rope, Kall, Vall, idx, treeb, NH, NKV, HD, TMAX, ROT):
    """attn_part / attn_partt + attn_comb as the kv-head kernels (attn_gqa.py): the same records, the same o_rows, byte for byte
    (checked on the vendor simulator). `treeb` None: the chain rule; else the tree table."""
    import attn_gqa as AG
    M = self.real; P = attn_parts(NH, HD, TMAX); tree = treeb is not None; nte = attn_gqa_nte(M, NH, NKV, HD, TMAX)
    dk, ck = f"attn_gqa_desc{M}", f"attn_comb2_desc{M}"
    if dk not in self.bufs: self.bufs[dk] = Tensor(AG.attn_gqa_desc(NH, NKV, HD, TMAX, M), device=DEV).realize()
    if ck not in self.bufs: self.bufs[ck] = Tensor(AG.attn_comb2_desc(NH, HD, TMAX, M), device=DEV).realize()
    kp = self.key(f"{'attn_partt' if tree else 'attn_part'}|gqa|{TMAX}|{M}|{nte}", AG.attn_gqa_src(NH, NKV, HD, TMAX, ROT, self.eps, M, tree=tree, NTE=nte))
    part = self.run(kp, "attn_part", NH * M * P * (HD + 16), dtypes.float32, q_rows, kv_rows, qnw, knw, rope, Kall, Vall, idx, self.bufs[dk], *([treeb] if tree else []))
    kc = self.key(f"attn_comb|dma|{TMAX}|{M}", AG.attn_comb2_src(NH, HD, TMAX, M))
    return self.run(kc, "o_rows", self.rows * NH * HD, dtypes.float32, part, q_rows, self.bufs[ck]).reshape(self.rows, NH * HD)
  def attn_prefill(self, q_rows, kv_rows, qnw, knw, rope, Kall, Vall, idx, NH, NKV, HD, TMAX, ROT):
    """The prefill's causal attention over its `real` rows at positions 0..real-1, every row moved by DMA (attn_pkv: k / v into
    the cache of layer idx[0]; attn_ppart: the slice records; attn_pcomb: o with the gate) -> the persistent o_rows [rows, NH HD]
    (rows real.. untouched). QR / KR / OR: the (unpadded) row pitches (floats) of q_rows / kv_rows / o_rows."""
    n = self.real; QR, KR, OR = NH * 2 * HD, 2 * NKV * HD, NH * HD; P = attn_parts(NH, HD, TMAX)
    for k_, d in (("attn_pkv_desc", lambda: attn_pkv_desc(HD)), ("attn_ppart_desc", lambda: attn_ppart_desc(NKV, HD)), ("attn_pcomb_desc", lambda: attn_pcomb_desc(NH, HD, TMAX))):
      if k_ not in self.bufs: self.bufs[k_] = Tensor(d(), device=DEV).realize()
    kv = self.key(f"attn_pkv|{TMAX}|{n}|{KR}", attn_pkv_src(NKV, HD, TMAX, ROT, self.eps, n, KR))
    OA.csrc_call(kv, Kall, Vall, kv_rows, knw, rope, idx, self.bufs["attn_pkv_desc"]).realize()
    kp = self.key(f"attn_ppart|{TMAX}|{n}|{QR}", attn_ppart_src(NH, NKV, HD, TMAX, ROT, self.eps, n, QR))
    part = self.run(kp, "attn_ppart", NH * n * P * (HD + 16), dtypes.float32, q_rows, qnw, rope, Kall, Vall, idx, self.bufs["attn_ppart_desc"])
    kc = self.key(f"attn_pcomb|{TMAX}|{n}|{QR}|{OR}", attn_pcomb_src(NH, HD, TMAX, n, QR, OR))
    return self.run(kc, "o_rows_pre", self.rows * OR, dtypes.float32, part, q_rows, self.bufs["attn_pcomb_desc"]).reshape(self.rows, OR)
  def gdn_tok3(self, Sall, Call, idx, posb, qkv_ct, z_ct, xin, wab, adt, cwt, nw, NV, NK, DK, DV, C, CONV, H, xoff=0, zoff=0, rows=False):
    """gdn_tok2 with the tile reads, the a|b projection / beta / decay and o_proj's A layout folded in -> the persistent `a_o`
    (compact A of K = NV DV, the token in row 0). wab / adt: the stacks of gdn_tok3_src; qkv_ct / z_ct: the GEMMs' C tiles
    (`xoff` / `zoff`: their first groups in those buffers -- the fused qkv|z GEMM passes one buffer twice, z at zoff)."""
    assert self.compact and self.rt == 1, "gdn_tok3 is the one-token decode step"
    if rows:                 # fp32 rows into the persistent o_rows (row 0), for had_a32's o_proj A (gdn_tok3_src(rows_out=True))
      kk = self.key(f"gdn_tok3|{xoff}|{zoff}|rows", ct_goff(gdn_tok3_src(NV, NK, DK, DV, C, CONV, H, self.nrb, self.eps, rows_out=True), self.nrb, (xoff, zoff), ("x", "z")))
      if "gdn_tok3_desc" not in self.bufs: self.bufs["gdn_tok3_desc"] = Tensor(gdn_tok3_desc(DK, DV, H, gdn_hpt(NV)), device=DEV).realize()
      return self.run(kk, "o_rows", self.rows * NV * DV, dtypes.float32, Sall, Call, idx, posb, qkv_ct, z_ct, xin, wab, adt, cwt, nw, self.bufs["gdn_tok3_desc"]).reshape(self.rows, NV * DV)
    kk = self.key(f"gdn_tok3|{xoff}|{zoff}", ct_goff(gdn_tok3_src(NV, NK, DK, DV, C, CONV, H, self.nrb, self.eps), self.nrb, (xoff, zoff), ("x", "z")))
    if "gdn_tok3_desc" not in self.bufs: self.bufs["gdn_tok3_desc"] = Tensor(gdn_tok3_desc(DK, DV, H, gdn_hpt(NV)), device=DEV).realize()
    return self.run(kk, "a_o", self.a_size(NV * DV), dtypes.uint16, Sall, Call, idx, posb, qkv_ct, z_ct, xin, wab, adt, cwt, nw, self.bufs["gdn_tok3_desc"])
  def gdn_tok2(self, Sall, Call, idx, posb, qkv_rows, z_rows, bd, cwt, nw, NV, NK, DK, DV, C, CONV):
    """gdn_tok with the state streamed through LSRAM -> the persistent `o_rows` [rows, NV*DV], the token in row 0."""
    kk = self.key("gdn_tok2", gdn_tok2_src(NV, NK, DK, DV, C, CONV, self.eps))
    if "gdn_lsr_desc" not in self.bufs: self.bufs["gdn_lsr_desc"] = Tensor(gdn_lsr_desc(DK, DV), device=DEV).realize()
    return self.run(kk, "o_rows", self.rows * NV * DV, dtypes.float32, Sall, Call, idx, posb, qkv_rows, z_rows, bd, cwt, nw, self.bufs["gdn_lsr_desc"]).reshape(self.rows, NV * DV)

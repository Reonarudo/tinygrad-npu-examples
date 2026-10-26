"""Mamba-2 (SSD) token kernel for the Zhouyi TECs: NH heads of head_dim P over a state of N columns, G groups sharing B / C.

The mixer (HF `NemotronHMamba2Mixer`, `Mamba2Mixer`, Granite 4.0-H): the in_proj row splits into z [NH P], xBC [CD = NH P + 2 G N]
and dt [NH]. xBC goes through a causal conv1d (CONV taps, a bias) and SiLU, then splits into x [NH][P], B [G][N], C [G][N]. Per head h
(group g = h / (NH / G)): dt_h = softplus(dt_h + dt_bias_h), dA = exp(dt_h A_h) (A_h = -exp(A_log_h)), S_h = dA S_h + dt_h x_h (x) B_g,
y_h = S_h C_g + D_h x_h. Then y = rmsnorm_group(y * silu(z)) * w, the norm over groups of GS = NH P / G channels (gate before norm).

Kernels (fp32 rows in, fp32 rows out; a model port reads its GEMM tiles instead, as gdn_tok3 does):
  mamba2_tok   M <= MMAX tokens a launch. A unit = one head; a task takes heads core_id, core_id + NT, ... The head's state S_h
               [P][N] streams through LSRAM in PB-row blocks (double-buffered DMA, read once, written once); a row's N columns stay
               in registers across the M tokens. The conv is computed in-kernel from the raw rows and the CONV-deep ring of raw
               xBC rows (positions pos - CONV .. pos - 1), the ring then updated with the last CONV raw rows. The ring is PER HEAD
               (Call [L][NH][CONV][PR = P + 2 N]: the head's x slice, then its group's B and C), so no two tasks touch one line: a
               shared B / C ring raced (the group's first head rewrote the slices other heads were still reading, 2026-10-25,
               seen on the model's sequential cores). Out: yg = (y + D x)
               silu(z) [M][NH P] and ss [M][NH] = sum of yg^2 over the head (for the group norm).
  mamba2_gnorm the group norm from yg and ss: out = yg * rsqrt(sum_{heads of g} ss / GS + eps) * w.
Contract: `mamba2_ref` (float64).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
from zy import vec_f16 as V                                              # noqa: E402

NT = 12                                                                  # solo launch: all twelve TECs
MMAX = 12
LSRAM = 32768 - 64

HELPERS = """
#define F8(a, b) __builtin_shufflevector((a), (b), 0, 1, 2, 3, 4, 5, 6, 7)
static inline float8 exp2_d4(float8 t) {
  t = VMAX(t, BC(-126.0f)); t = VMIN(t, BC(126.0f));
  float8 k = VRINT(t); float8 f = t - k;
  float8 p = BC(9.618129e-3f);
  p = p * f + BC(5.550411e-2f); p = p * f + BC(2.402265e-1f); p = p * f + BC(6.931472e-1f); p = p * f + BC(1.0f);
  return VSCAL2(p, VI(k));
}
static inline float8 sw(float8 u) { float8 e = exp2_d4(BC(-1.4426950408889634f) * u); return u * VRCP(BC(1.0f) + e); }
static inline __attribute__((always_inline)) float ex1(float t) { return exp2_d4(BC(t * 1.4426950408889634f))[0]; }
static inline __attribute__((always_inline)) float softplus1(float t) {          /* max(t, 0) + log1p(exp(-|t|)), log1p = 2 atanh */
  float at = t < 0.0f ? -t : t; float y = ex1(-at); float s = y / (2.0f + y), s2 = s * s;
  float l = 2.0f * s * (1.0f + s2 * (0.3333333333333333f + s2 * (0.2f + s2 * (0.14285714285714285f + s2 * (0.1111111111111111f + s2 * (0.09090909090909091f + s2 * 0.07692307692307693f))))));
  return (t > 0.0f ? t : 0.0f) + l;
}
"""


def cfg(NH, P, N, G, CONV, M, PB=8):
  """The LSRAM plan: (PB, offsets) with the budget asserted. Blocks: 2 x PB N floats; per token: x P + B N + C N floats; dA / dx: M."""
  assert P % PB == 0 and N % 8 == 0 and P % 8 == 0 and NH % G == 0 and 1 <= M <= MMAX
  BLK = PB * N * 4; XS = 2 * BLK; BS = XS + M * P * 4; CS = BS + M * N * 4; END = CS + M * N * 4
  assert END <= LSRAM, (END, LSRAM)
  return dict(PB=PB, BLK=BLK, XS=XS, BS=BS, CS=CS, END=END)


def mamba2_tok_desc(NH, P, N, G, CONV, M, PB=8):
  """slot 0: a PB-row state block (contiguous PB N floats)."""
  c = cfg(NH, P, N, G, CONV, M, PB); return V._desc_slots((c["BLK"],))


def mamba2_tok_src(NH, P, N, G, CONV, M, PB=8):
  """args: yg [M][NH P], ss [M][NH], Sall [L][NH][P][N], Call [L][NH][CONV][PR], idx (int32 [1]: the layer), posb (int32 [1]: row 0's
  position), xbc [M][CD] (raw), dtr [M][NH] (raw), z [M][NH P], adt [L][3][NH] (-exp(A_log), dt_bias, D), cwt [L][CONV][CD],
  cbs [L][CD], desc (mamba2_tok_desc)."""
  c = cfg(NH, P, N, G, CONV, M, PB); PB, BLK, XS, BS, CS = c["PB"], c["BLK"], c["XS"], c["BS"], c["CS"]
  HPG = NH // G; CD = NH * P + 2 * G * N; BO, CO = NH * P, NH * P + G * N; NB = P // PB; NV = N // 8; PR = P + 2 * N
  conv = lambda dst, off, roff, n: f"""
    for (int m = 0; m < {M}; m++) for (int i = 0; i < {n}; i += 8) {{
      float8 acc = *(__global float8*)(cb + {off} + i);
      for (int t = 0; t < {CONV}; t++) {{
        int tok = m - {CONV - 1} + t; float8 v;
        if (tok >= 0) v = *(__global float8*)(xbc + tok * {CD} + {off} + i);
        else v = *(__global float8*)(Cr + ((pos + tok) % {CONV} + {CONV}) % {CONV} * {PR} + {roff} + i);
        acc += v * *(__global float8*)(cw + t * {CD} + {off} + i);
      }}
      *(__global float8*)({dst} + m * {n} + i) = sw(acc);
    }}"""
  ring = lambda off, roff, n: f"""
    for (int m = {max(0, M - CONV)}; m < {M}; m++) for (int i = 0; i < {n}; i += 8)
      *(__global float8*)(Cr + ((pos + m) % {CONV}) * {PR} + {roff} + i) = *(__global float8*)(xbc + m * {CD} + {off} + i);"""
  return V.FULL_H + HELPERS + f"""
__kernel void mamba2_tok(__global float* restrict yg, __global float* restrict ss, __global float* restrict Sall, __global float* restrict Call,
                         __global int* restrict idx, __global int* restrict posb, __global float* restrict xbc, __global float* restrict dtr,
                         __global float* restrict z, __global float* restrict adt, __global float* restrict cwt, __global float* restrict cbs,
                         __global int* restrict desc, const int core_id) {{
  int L = idx[0], pos = posb[0];
  __global float* S = Sall + L * {NH * P * N};
  __global float* cw = cwt + L * {CONV * CD}; __global float* cb = cbs + L * {CD}; adt += L * {3 * NH};
  __global float* xs = LSF({XS}); __global float* bs = LSF({BS}); __global float* cs = LSF({CS});
  for (int h = core_id; h < {NH}; h += {NT}) {{
    int g = h / {HPG}; __global float* Sh = S + h * {P * N}; __global float* Cr = Call + (L * {NH} + h) * {CONV * PR};
    /* the conv + SiLU of the M tokens: x of this head, B / C of its group (into LSRAM); the ring is the head's own */
    {conv("xs", f"h * {P}", 0, P)}
    {conv("bs", f"{BO} + g * {N}", P, N)}
    {conv("cs", f"{CO} + g * {N}", P + N, N)}
    float dA[{M}], dx[{M}], Dh = adt[{2 * NH} + h];
    for (int m = 0; m < {M}; m++) {{
      float dt = softplus1(dtr[m * {NH} + h] + adt[{NH} + h]);
      dA[m] = ex1(dt * adt[h]); dx[m] = dt;
    }}
    float sq[{M}]; for (int m = 0; m < {M}; m++) sq[m] = 0.0f;
    DMA_FILL(0, DESC(desc, 0), 0, (int)Sh);
    for (int b = 0; b < {NB}; b++) {{
      int p = b & 1;
      if (b + 1 < {NB}) {{
        if (p == 0) {{ DMA_WAIT(3); DMA_FILL(1, DESC(desc, 0), {BLK}, (int)(Sh + (b + 1) * {PB * N})); }}
        else        {{ DMA_WAIT(2); DMA_FILL(0, DESC(desc, 0), 0, (int)(Sh + (b + 1) * {PB * N})); }}
      }}
      if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
      __global float* B = LSF(p * {BLK});
      for (int r = 0; r < {PB}; r++) {{
        int pr = b * {PB} + r; __global float* Sr = B + r * {N};
        float8 s[{NV}]; for (int j = 0; j < {NV}; j++) s[j] = *(__global float8*)(Sr + 8 * j);
        for (int m = 0; m < {M}; m++) {{
          float x = xs[m * {P} + pr]; float8 a = BC(dA[m]), xb = BC(dx[m] * x), acc = BC(0.0f);
          __global float* Bm = bs + m * {N}; __global float* Cm = cs + m * {N};
          for (int j = 0; j < {NV}; j++) {{ s[j] = s[j] * a + xb * *(__global float8*)(Bm + 8 * j); acc += s[j] * *(__global float8*)(Cm + 8 * j); }}
          float y = __builtin_aipu_rpadd_tfp32_pw(acc, acc, ALL)[0] + Dh * x;
          float zz = z[m * {NH * P} + h * {P} + pr]; float yv = y * (zz * (1.0f / (1.0f + ex1(-zz))));
          yg[m * {NH * P} + h * {P} + pr] = yv; sq[m] += yv * yv;
        }}
        for (int j = 0; j < {NV}; j++) *(__global float8*)(Sr + 8 * j) = s[j];
      }}
      if (p == 0) DMA_DRAIN(2, DESC(desc, 0), 0, (int)(Sh + b * {PB * N})); else DMA_DRAIN(3, DESC(desc, 0), {BLK}, (int)(Sh + b * {PB * N}));
    }}
    for (int m = 0; m < {M}; m++) ss[m * {NH} + h] = sq[m];
    /* the head's ring: the last CONV raw rows (its x slice, its group's B and C) */
    {ring(f"h * {P}", 0, P)}
    {ring(f"{BO} + g * {N}", P, N)}
    {ring(f"{CO} + g * {N}", P + N, N)}
    DMA_WAIT_ALL();
  }}
}}"""


def mamba2_gnorm_src(NH, P, G, M, eps):
  """args: out [M][NH P], yg [M][NH P], ss [M][NH], w [L][NH P], idx. A unit = (token, group)."""
  HPG = NH // G; GS = HPG * P
  return V.FULL_H + f"""
__kernel void mamba2_gnorm(__global float* restrict out, __global float* restrict yg, __global float* restrict ss, __global float* restrict w,
                           __global int* restrict idx, const int core_id) {{
  w += idx[0] * {NH * P};
  for (int u = core_id; u < {M * G}; u += {NT}) {{
    int m = u / {G}, g = u % {G}; float acc = 0.0f;
    for (int h = 0; h < {HPG}; h++) acc += ss[m * {NH} + g * {HPG} + h];
    float8 inv = BC(1.0f / __builtin_sqrtf(acc * {1.0 / GS!r}f + {eps!r}f));
    for (int i = 0; i < {GS}; i += 8) {{
      int c = g * {GS} + i;
      *(__global float8*)(out + m * {NH * P} + c) = *(__global float8*)(yg + m * {NH * P} + c) * inv * *(__global float8*)(w + c);
    }}
  }}
}}"""


# ---- the Contract (float64)
def silu(x): return x / (1.0 + np.exp(-x))

def softplus(x): return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))

def mamba2_ref(S, ring, pos, xbc, dtr, z, Aneg, dt_bias, D, cw, cb, w, eps, NH, P, N, G, CONV):
  """One launch of M rows at positions pos .. pos + M - 1. S [NH][P][N], ring [NH][CONV][P + 2 N] (slot = position % CONV; a head's
  x slice, its group's B, its group's C), xbc [M][CD], dtr
  [M][NH], z [M][NH P]; cw [CONV][CD] (tap t multiplies position - (CONV - 1) + t), cb [CD]. Returns (yg, ss, out, S', ring')."""
  S = S.astype(np.float64).copy(); ring = ring.astype(np.float64).copy(); M = xbc.shape[0]; CD = xbc.shape[1]; HPG = NH // G
  xbc = xbc.astype(np.float64); cw = cw.astype(np.float64); cb = cb.astype(np.float64)
  def slices(h):                                                         # (column offset in xBC, offset in the head's ring row, width)
    g = h // HPG; return ((h * P, 0, P), (NH * P + g * N, P, N), (NH * P + G * N + g * N, P + N, N))
  x = np.zeros((M, NH, P)); Bm = np.zeros((M, NH, N)); Cm = np.zeros((M, NH, N))            # per head, from the head's own ring
  for m in range(M):
    for h in range(NH):
      for (off, roff, n), dst in zip(slices(h), (x, Bm, Cm)):
        acc = cb[off:off + n].copy()
        for t in range(CONV):
          tok = m - (CONV - 1) + t
          v = xbc[tok, off:off + n] if tok >= 0 else ring[h, (pos + tok) % CONV, roff:roff + n]
          acc = acc + v * cw[t, off:off + n]
        dst[m, h] = silu(acc)
  dt = softplus(dtr.astype(np.float64) + dt_bias.astype(np.float64)[None]); dA = np.exp(dt * Aneg.astype(np.float64)[None])
  yg = np.zeros((M, NH * P)); ss = np.zeros((M, NH)); zf = z.astype(np.float64)
  for m in range(M):
    for h in range(NH):
      S[h] = S[h] * dA[m, h] + np.outer(dt[m, h] * x[m, h], Bm[m, h])
      y = S[h] @ Cm[m, h] + D[h] * x[m, h]
      v = y * silu(zf[m, h * P:(h + 1) * P]); yg[m, h * P:(h + 1) * P] = v; ss[m, h] = (v * v).sum()
  for m in range(max(0, M - CONV), M):
    for h in range(NH):
      for off, roff, n in slices(h): ring[h, (pos + m) % CONV, roff:roff + n] = xbc[m, off:off + n]
  GS = HPG * P; out = np.zeros_like(yg)
  for m in range(M):
    for g in range(G):
      seg = yg[m, g * GS:(g + 1) * GS]; out[m, g * GS:(g + 1) * GS] = seg / np.sqrt((seg * seg).mean() + eps) * w.astype(np.float64)[g * GS:(g + 1) * GS]
  return yg, ss, out, S, ring

"""Kimi Delta Attention (Kimi Linear, Ling-3.0-tiny): the gated delta rule with a per-key-channel decay, NH heads of K x V state.

Per head h and token t (fla `fused_recurrent_kda`, float32): q, k = L2-normalised (eps 1e-6) SiLU(conv4(W x)); q *= K^-0.5; v = SiLU(conv4(W_v x));
alpha_i = exp(g_i) with (Kimi) g_i = -exp(A_log_h) softplus(graw_i + dt_bias_i) or (Ling, `gate="sigmoid"`) g_i = LB sigmoid(exp(A_log_h)
(graw_i + dt_bias_i)), LB = -5; beta = sigmoid(braw_h). Then S[i, :] *= alpha_i; kv = S^T k; delta = (v - kv) beta; S += k (x) delta;
o = S^T q; y = rmsnorm_V(o) * w * sigmoid(gout_h). Gated DeltaNet is the alpha_i = alpha (scalar) case.

  kda_tok   M <= MMAX tokens a launch (gdn_tokm's wave). A unit = one head; the state streams through LSRAM in K x W column blocks
            (2-D DMA, double-buffered, read once, written once); a block serves the M tokens in sequence: pass 1 decays its rows and
            accumulates kv[W] = sum_i k_i S[i, W], pass 2 adds k_i delta[W] and accumulates o[W]. The conv ring is per head
            (Call [L][NH][CONV][3 K]: q | k | v raw slices), as mamba2. Out: y [M][NH V] (the gated norm applied in-kernel: the norm
            is per head).
Contract: `kda_ref` (float64).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
from zy import vec_f16 as V                                              # noqa: E402
from mamba2 import HELPERS                                               # noqa: E402

NT = 12
MMAX = 6
LSRAM = 32768 - 64
W = 16

VHELP = """
static inline float8 sig8(float8 t) { return VRCP(BC(1.0f) + exp2_d4(BC(-1.4426950408889634f) * t)); }
static inline float8 softplus8(float8 t) {                              /* max(t, 0) + log1p(exp(-|t|)), log1p(y) = 2 atanh(y / (2 + y)) */
  float8 at = VMAX(t, BC(0.0f) - t); float8 y = exp2_d4(BC(-1.4426950408889634f) * at); float8 s = y * VRCP(BC(2.0f) + y), s2 = s * s;
  float8 l = BC(2.0f) * s * (BC(1.0f) + s2 * (BC(0.3333333333333333f) + s2 * (BC(0.2f) + s2 * (BC(0.14285714285714285f) + s2 * (BC(0.1111111111111111f) + s2 * (BC(0.09090909090909091f) + s2 * BC(0.07692307692307693f)))))));
  return VMAX(t, BC(0.0f)) + l;
}
static inline float8 exp8(float8 t) { return exp2_d4(BC(1.4426950408889634f) * t); }
"""


def cfg(NH, K, Vd, CONV, M):
  assert K % 8 == 0 and Vd % W == 0 and 1 <= M <= MMAX
  BLK = K * W * 4; QS = 2 * BLK; KS = QS + M * K * 4; AS = KS + M * K * 4; VS = AS + M * K * 4; OS = VS + M * Vd * 4; END = OS + M * Vd * 4
  assert END <= LSRAM, (END, LSRAM)
  return dict(BLK=BLK, QS=QS, KS=KS, AS=AS, VS=VS, OS=OS, END=END)


def kda_tok_desc(K, Vd):
  """slot 0: a state block: K rows of W floats (DDR pitch Vd floats, LSRAM contiguous)."""
  return V._desc_slots((K * W * 4, W * 4, Vd * 4, W * 4))


def kda_tok_src(NH, K, Vd, CONV, M, eps, gate="softplus", LB=-5.0):
  """args: y [M][NH Vd], Sall [L][NH][K][Vd], Call [L][NH][CONV][3 K], idx (int32 [1]), posb (int32 [1]), qkv [M][3 NH K] (raw q | k | v
  projections, head-major within each), graw [M][NH K], braw [M][NH], gout [M][NH Vd], prm [L][NH + NH K] (A_log, then dt_bias),
  cwt [L][CONV][3 NH K], nw [L][Vd] (the norm weight), desc (kda_tok_desc)."""
  c = cfg(NH, K, Vd, CONV, M); BLK, QS, KS, AS, VS, OS = (c[k] for k in ("BLK", "QS", "KS", "AS", "VS", "OS"))
  NB = Vd // W; R = 3 * K; KO, VO = NH * K, 2 * NH * K; K8 = K // 8
  assert Vd == K, "q / k and v of one width (the ring row is 3 K)"
  conv = lambda dst, off, roff, n, pitch: f"""
    for (int m = 0; m < {M}; m++) for (int i = 0; i < {n}; i += 8) {{
      float8 acc = BC(0.0f);
      for (int t = 0; t < {CONV}; t++) {{
        int tok = m - {CONV - 1} + t; float8 v;
        if (tok >= 0) v = *(__global float8*)(qkv + tok * {3 * NH * K} + {off} + i);
        else v = *(__global float8*)(Cr + ((pos + tok) % {CONV} + {CONV}) % {CONV} * {R} + {roff} + i);
        acc += v * *(__global float8*)(cw + t * {3 * NH * K} + {off} + i);
      }}
      *(__global float8*)({dst} + m * {pitch} + i) = sw(acc);
    }}"""
  ring = lambda off, roff, n: f"""
    for (int m = {max(0, M - CONV)}; m < {M}; m++) for (int i = 0; i < {n}; i += 8)
      *(__global float8*)(Cr + ((pos + m) % {CONV}) * {R} + {roff} + i) = *(__global float8*)(qkv + m * {3 * NH * K} + {off} + i);"""
  if gate == "softplus": gexp = f"exp8(BC(-ex1(prm[h])) * softplus8(*(__global float8*)(graw + m * {NH * K} + h * {K} + i) + *(__global float8*)(prm + {NH} + h * {K} + i)))"
  else: gexp = f"exp8(BC({LB!r}f) * sig8(BC(ex1(prm[h])) * (*(__global float8*)(graw + m * {NH * K} + h * {K} + i) + *(__global float8*)(prm + {NH} + h * {K} + i))))"
  return V.FULL_H + HELPERS + VHELP + f"""
static inline __attribute__((always_inline)) void l2n(__global float* restrict v, float scale) {{
  float8 s = BC(0.0f); for (int i = 0; i < {K}; i += 8) {{ float8 a = *(__global float8*)(v + i); s += a * a; }}
  float8 inv = BC(scale / __builtin_sqrtf(hsum8(s) + 1e-6f)); for (int i = 0; i < {K}; i += 8) *(__global float8*)(v + i) = *(__global float8*)(v + i) * inv;
}}
__kernel void kda_tok(__global float* restrict y, __global float* restrict Sall, __global float* restrict Call, __global int* restrict idx,
                      __global int* restrict posb, __global float* restrict qkv, __global float* restrict graw, __global float* restrict braw,
                      __global float* restrict gout, __global float* restrict prm, __global float* restrict cwt, __global float* restrict nw,
                      __global int* restrict desc, const int core_id) {{
  int L = idx[0], pos = posb[0];
  __global float* S = Sall + L * {NH * K * Vd}; __global float* cw = cwt + L * {CONV * 3 * NH * K}; prm += L * {NH + NH * K}; nw += L * {Vd};
  __global float* qs = LSF({QS}); __global float* ks = LSF({KS}); __global float* as = LSF({AS}); __global float* vs = LSF({VS}); __global float* os = LSF({OS});
  for (int h = core_id; h < {NH}; h += {NT}) {{
    __global float* Sh = S + h * {K * Vd}; __global float* Cr = Call + (L * {NH} + h) * {CONV * R};
    {conv("qs", f"h * {K}", 0, K, K)}
    {conv("ks", f"{KO} + h * {K}", K, K, K)}
    {conv("vs", f"{VO} + h * {K}", 2 * K, Vd, Vd)}
    float bt[{M}];
    for (int m = 0; m < {M}; m++) {{
      l2n(qs + m * {K}, {1.0 / K ** 0.5!r}f); l2n(ks + m * {K}, 1.0f);
      for (int i = 0; i < {K}; i += 8) *(__global float8*)(as + m * {K} + i) = {gexp};
      float b = braw[m * {NH} + h]; bt[m] = 1.0f / (1.0f + ex1(-b));
    }}
    DMA_FILL(0, DESC(desc, 0), 0, (int)Sh);
    for (int b = 0; b < {NB}; b++) {{
      int p = b & 1;
      if (b + 1 < {NB}) {{
        if (p == 0) {{ DMA_WAIT(3); DMA_FILL(1, DESC(desc, 0), {BLK}, (int)(Sh + (b + 1) * {W})); }}
        else        {{ DMA_WAIT(2); DMA_FILL(0, DESC(desc, 0), 0, (int)(Sh + (b + 1) * {W})); }}
      }}
      if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
      __global float* B = LSF(p * {BLK});
      for (int m = 0; m < {M}; m++) {{
        __global float* q = qs + m * {K}; __global float* k = ks + m * {K}; __global float* al = as + m * {K};
        float8 kv0 = BC(0.0f), kv1 = BC(0.0f);
        for (int i = 0; i < {K}; i++) {{                             /* S_i *= alpha_i; kv += k_i S_i */
          float8 ab = BC(al[i]), kb = BC(k[i]); float8 s0 = *(__global float8*)(B + i * {W}) * ab, s1 = *(__global float8*)(B + i * {W} + 8) * ab;
          *(__global float8*)(B + i * {W}) = s0; *(__global float8*)(B + i * {W} + 8) = s1; kv0 += kb * s0; kv1 += kb * s1;
        }}
        float8 bb = BC(bt[m]); float8 dl0 = (*(__global float8*)(vs + m * {Vd} + b * {W}) - kv0) * bb, dl1 = (*(__global float8*)(vs + m * {Vd} + b * {W} + 8) - kv1) * bb;
        float8 o0 = BC(0.0f), o1 = BC(0.0f);
        for (int i = 0; i < {K}; i++) {{                             /* S_i += k_i delta; o += q_i S_i */
          float8 kb = BC(k[i]), qb = BC(q[i]);
          float8 s0 = *(__global float8*)(B + i * {W}) + kb * dl0, s1 = *(__global float8*)(B + i * {W} + 8) + kb * dl1;
          *(__global float8*)(B + i * {W}) = s0; *(__global float8*)(B + i * {W} + 8) = s1; o0 += qb * s0; o1 += qb * s1;
        }}
        *(__global float8*)(os + m * {Vd} + b * {W}) = o0; *(__global float8*)(os + m * {Vd} + b * {W} + 8) = o1;
      }}
      if (p == 0) DMA_DRAIN(2, DESC(desc, 0), 0, (int)(Sh + b * {W})); else DMA_DRAIN(3, DESC(desc, 0), {BLK}, (int)(Sh + b * {W}));
    }}
    for (int m = 0; m < {M}; m++) {{                                 /* y = rmsnorm(o) nw sigmoid(gout) */
      __global float* o = os + m * {Vd}; float8 ss = BC(0.0f);
      for (int e = 0; e < {Vd}; e += 8) {{ float8 ov = *(__global float8*)(o + e); ss += ov * ov; }}
      float8 inv = BC(1.0f / __builtin_sqrtf(hsum8(ss) * {1.0 / Vd!r}f + {eps!r}f));
      for (int e = 0; e < {Vd}; e += 8)
        *(__global float8*)(y + m * {NH * Vd} + h * {Vd} + e) = *(__global float8*)(o + e) * inv * *(__global float8*)(nw + e) * sig8(*(__global float8*)(gout + m * {NH * Vd} + h * {Vd} + e));
    }}
    {ring(f"h * {K}", 0, K)}
    {ring(f"{KO} + h * {K}", K, K)}
    {ring(f"{VO} + h * {K}", 2 * K, Vd)}
    DMA_WAIT_ALL();
  }}
}}"""


# ---- the Contract (float64)
def silu(x): return x / (1.0 + np.exp(-x))
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def softplus(x): return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))

def kda_ref(S, ring, pos, qkv, graw, braw, gout, A_log, dt_bias, cw, nw, eps, NH, K, Vd, CONV, gate="softplus", LB=-5.0):
  """S [NH][K][Vd], ring [NH][CONV][3 K], qkv [M][3 NH K], graw [M][NH K], braw [M][NH], gout [M][NH Vd], A_log [NH], dt_bias [NH K],
  cw [CONV][3 NH K], nw [Vd]. -> (y [M][NH Vd], S', ring')."""
  S = S.astype(np.float64).copy(); ring = ring.astype(np.float64).copy(); M = qkv.shape[0]; qkv = qkv.astype(np.float64); cw = cw.astype(np.float64)
  y = np.zeros((M, NH * Vd))
  def slices(h): return ((h * K, 0, K), (NH * K + h * K, K, K), (2 * NH * K + h * K, 2 * K, Vd))
  for h in range(NH):
    q = np.zeros((M, K)); k = np.zeros((M, K)); v = np.zeros((M, Vd))
    for (off, roff, n), dst in zip(slices(h), (q, k, v)):
      for m in range(M):
        acc = np.zeros(n)
        for t in range(CONV):
          tok = m - (CONV - 1) + t
          src = qkv[tok, off:off + n] if tok >= 0 else ring[h, (pos + tok) % CONV, roff:roff + n]
          acc = acc + src * cw[t, off:off + n]
        dst[m] = silu(acc)
    for m in range(M):
      qm = q[m] / np.sqrt((q[m] ** 2).sum() + 1e-6) * K ** -0.5; km = k[m] / np.sqrt((k[m] ** 2).sum() + 1e-6)
      gr = graw[m, h * K:(h + 1) * K].astype(np.float64) + dt_bias[h * K:(h + 1) * K].astype(np.float64)
      g = -np.exp(A_log[h]) * softplus(gr) if gate == "softplus" else LB * sigmoid(np.exp(A_log[h]) * gr)
      alpha = np.exp(g); beta = sigmoid(float(braw[m, h]))
      S[h] = S[h] * alpha[:, None]; kv = S[h].T @ km; delta = (v[m] - kv) * beta; S[h] = S[h] + np.outer(km, delta); o = S[h].T @ qm
      yn = o / np.sqrt((o * o).mean() + eps) * nw.astype(np.float64) * sigmoid(gout[m, h * Vd:(h + 1) * Vd].astype(np.float64))
      y[m, h * Vd:(h + 1) * Vd] = yn
  for m in range(max(0, M - CONV), M):
    for h in range(NH):
      for off, roff, n in slices(h): ring[h, (pos + m) % CONV, roff:roff + n] = qkv[m, off:off + n]
  return y, S, ring

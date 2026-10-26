"""CCA / CCGQA (Zyphra, ZAYA1-8B): the preprocessing of a token's query and key / value rows for the latent attention.

transformers `modeling_zaya.py` (ZayaAttention), E hidden, NH query heads and NKV kv heads of HD, latent u = [q' (NH HD) | k' (NKV HD)]:
  qr = (q'_h + k'_{kv(h)}) / 2;  kr_kv = mean over the group's qr                                   (the QK-mean, pre-conv)
  d[c][s] = b1[c] + w1[c][0] u_{t-2+s}[c] + w1[c][1] u_{t-1+s}[c], s = 0, 1                        (depthwise conv, kernel 2, over the halo)
  e[c] = b2[c] + sum_j sum_s w2[c][j][s] d[head(c) HD + j][s]                                        (grouped conv: one group a head, kernel 2)
  q_h = e[q cols] + qr;  k_kv = e[k cols] + kr                                                      (the residual)
  v rows: kv head 0 = W_vc x_t (current), kv head 1 = W_vd x_{t-1} (the value shift: the delayed row is the previous token's)
  q_h *= sqrt(HD) / ||q_h||; k_kv *= sqrt(HD) / ||k_kv|| * temp[kv]; partial RoPE (rotate-half) on the first ROT dims
Then standard GQA attention (attn.py: NH, NKV, HD, scale HD^-0.5). The halo state a layer: the last 2 raw u rows; the delayed-v state: W_vd x
of the previous token.

  cca_prep  one token. Units: the NH + NKV conv heads (a unit = one head's e: a 128 x 256 matvec from the head's w2 through LSRAM).
            Query units also finish q_h; key units finish k_kv and the v rows. Out: q [NH][HD], k_new [NKV][HD], v_new [NKV][HD];
            the halo and the delayed v updated (by the k units: one writer a line).
Contract: `cca_ref` (float64).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
from zy import vec_f16 as V                                              # noqa: E402
from mamba2 import HELPERS                                               # noqa: E402

NT = 12
LSRAM = 32768 - 64


CH = 8                                                                   # w2 rows a DMA chunk (a row = HD x 2 floats)

def cca_prep_desc(HD):
  """slot 0: a chunk of CH rows of a head's w2 [HD][HD][2] (contiguous CH HD 2 floats)."""
  return V._desc_slots((CH * HD * 2 * 4,))


def cca_prep_src(NH, NKV, HD, ROT, eps=1e-6):
  """args: q [NH][HD], k_new [NKV][HD], v_new [NKV][HD], u [UW = (NH + NKV) HD] (the raw q' | k' row), vcd [2 HD] (W_vc x | W_vd x of this
  token), halo [L][2][UW] (raw u of t - 2, t - 1), vdel [L][HD] (W_vd x of t - 1), w1 [L][UW][2], b1 [L][UW], w2 [L][UW][HD][2], b2 [L][UW],
  temp [L][NKV], cs [TMAX][ROT] (cos [ROT / 2] | sin [ROT / 2] a position), idx (int32 [2]: layer, pos), desc."""
  UW = (NH + NKV) * HD; NU = NH + NKV; g = NH // NKV; W2 = CH * HD * 2 * 4; HR = ROT // 2; NCH = HD // CH
  assert 2 * W2 + 4 * HD * 4 <= LSRAM and ROT % 16 == 0 and HD % CH == 0
  return V.FULL_H + HELPERS + f"""
static inline __attribute__((always_inline)) void fin(__global float* restrict x, float mul, __global float* restrict cr) {{
  float8 s = BC(0.0f); for (int i = 0; i < {HD}; i += 8) {{ float8 a = *(__global float8*)(x + i); s += a * a; }}
  float n = __builtin_sqrtf(hsum8(s)); float8 f = BC(mul * {HD ** 0.5!r}f / (n > {eps!r}f ? n : {eps!r}f));
  for (int i = 0; i < {HD}; i += 8) *(__global float8*)(x + i) = *(__global float8*)(x + i) * f;
  for (int i = 0; i < {HR}; i += 8) {{
    float8 x1 = *(__global float8*)(x + i), x2 = *(__global float8*)(x + {HR} + i), c = *(__global float8*)(cr + i), sn = *(__global float8*)(cr + {HR} + i);
    *(__global float8*)(x + i) = x1 * c - x2 * sn; *(__global float8*)(x + {HR} + i) = x2 * c + x1 * sn;
  }}
}}
__kernel void cca_prep(__global float* restrict q, __global float* restrict k_new, __global float* restrict v_new, __global float* restrict u,
                       __global float* restrict vcd, __global float* restrict halo, __global float* restrict vdel, __global float* restrict w1,
                       __global float* restrict b1, __global float* restrict w2, __global float* restrict b2, __global float* restrict temp,
                       __global float* restrict cs, __global int* restrict idx, __global int* restrict desc, const int core_id) {{
  int L = idx[0], pos = idx[1]; __global float* H = halo + L * {2 * UW}; __global float* vd = vdel + L * {HD};
  w1 += L * {UW * 2}; b1 += L * {UW}; w2 += L * {UW * HD * 2}; b2 += L * {UW}; temp += L * {NKV}; __global float* cr = cs + pos * {ROT};
  __global float* d = LSF({2 * W2}); __global float* e = LSF({2 * W2 + 2 * HD * 4});
  for (int hu = core_id; hu < {NU}; hu += {NT}) {{
    int c0 = hu * {HD}; __global float* w2h = w2 + c0 * {HD * 2};
    DMA_FILL(0, DESC(desc, 0), 0, (int)w2h);
    for (int j = 0; j < {HD}; j++) {{                                   /* the depthwise conv: d[j][0] (t-2, t-1), d[j][1] (t-1, t) */
      int c = c0 + j; float w10 = w1[2 * c], w11 = w1[2 * c + 1];
      d[2 * j] = b1[c] + w10 * H[c] + w11 * H[{UW} + c]; d[2 * j + 1] = b1[c] + w10 * H[{UW} + c] + w11 * u[c];
    }}
    for (int ch = 0; ch < {NCH}; ch++) {{                               /* the grouped conv: e[i] = b2 + sum_j sum_s w2[i][j][s] d[j][s], CH rows a chunk */
      int p = ch & 1;
      if (ch + 1 < {NCH}) {{ if (p == 0) DMA_FILL(1, DESC(desc, 0), {W2}, (int)(w2h + (ch + 1) * {CH * HD * 2})); else DMA_FILL(0, DESC(desc, 0), 0, (int)(w2h + (ch + 1) * {CH * HD * 2})); }}
      if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
      __global float* wb = LSF(p * {W2});
      for (int r = 0; r < {CH}; r++) {{
        int i = ch * {CH} + r; float8 acc = BC(0.0f); __global float* wr = wb + r * {HD * 2};
        for (int j = 0; j < {HD * 2}; j += 8) acc += *(__global float8*)(wr + j) * *(__global float8*)(d + j);
        e[i] = __builtin_aipu_rpadd_tfp32_pw(acc, acc, ALL)[0] + b2[c0 + i];
      }}
    }}
    if (hu < {NH}) {{                                                   /* a query head: + qr, norm, rope */
      int kh = hu / {g}; __global float* qh = q + hu * {HD};
      for (int i = 0; i < {HD}; i += 8) *(__global float8*)(qh + i) = *(__global float8*)(e + i) + (*(__global float8*)(u + c0 + i) + *(__global float8*)(u + {NH * HD} + kh * {HD} + i)) * BC(0.5f);
      fin(qh, 1.0f, cr);
    }} else {{                                                          /* a kv head: + kr (the group mean of qr), norm, temp, rope; the v rows; the states */
      int kh = hu - {NH}; __global float* kk = k_new + kh * {HD};
      for (int i = 0; i < {HD}; i += 8) {{
        float8 acc = BC(0.0f);
        for (int r = 0; r < {g}; r++) acc += (*(__global float8*)(u + (kh * {g} + r) * {HD} + i) + *(__global float8*)(u + c0 + i)) * BC(0.5f);
        *(__global float8*)(kk + i) = *(__global float8*)(e + i) + acc * BC({1.0 / g!r}f);
      }}
      fin(kk, temp[kh], cr);
      __global float* vv = v_new + kh * {HD};
      if (kh == 0) for (int i = 0; i < {HD}; i += 8) *(__global float8*)(vv + i) = *(__global float8*)(vcd + i);
      else for (int i = 0; i < {HD}; i += 8) *(__global float8*)(vv + i) = *(__global float8*)(vd + i);
    }}
  }}
  DMA_WAIT_ALL();
  /* the states, after every unit has read them: task 0's units are done with the halo when it gets here, but other tasks may not be --
     so the halo / vdel update is a separate kernel (cca_state_src) in a chain, never here. */
}}"""


def cca_state_src(NH, NKV, HD):
  """args: halo [L][2][UW], vdel [L][HD], u [UW], vcd [2 HD], idx. The halo shifts (row 0 <- row 1 <- u) and vdel <- W_vd x. 4 tasks."""
  UW = (NH + NKV) * HD
  return V.FULL_H + f"""
__kernel void cca_state(__global float* restrict halo, __global float* restrict vdel, __global float* restrict u, __global float* restrict vcd,
                        __global int* restrict idx, const int core_id) {{
  int L = idx[0]; __global float* H = halo + L * {2 * UW};
  for (int i = core_id * 8; i < {UW}; i += 32) {{ *(__global float8*)(H + i) = *(__global float8*)(H + {UW} + i); *(__global float8*)(H + {UW} + i) = *(__global float8*)(u + i); }}
  if (core_id == 0) for (int i = 0; i < {HD}; i += 8) *(__global float8*)(vdel + L * {HD} + i) = *(__global float8*)(vcd + {HD} + i);
}}"""


# ---- the Contract (float64)
def cca_ref(u, vcd, halo, vdel, w1, b1, w2, b2, temp, cs, NH, NKV, HD, ROT, eps=1e-6):
  """u [UW], vcd [2 HD], halo [2][UW], vdel [HD], w1 [UW][2], b1 [UW], w2 [UW][HD][2], b2 [UW], temp [NKV], cs [ROT] (cos | sin of the
  position). -> (q [NH][HD], k_new [NKV][HD], v_new [NKV][HD], halo', vdel')."""
  f = lambda x: np.asarray(x, np.float64); u, vcd, halo, vdel, w1, b1, w2, b2, temp, cs = map(f, (u, vcd, halo, vdel, w1, b1, w2, b2, temp, cs))
  UW = (NH + NKV) * HD; g = NH // NKV; HR = ROT // 2
  d = np.stack([b1 + w1[:, 0] * halo[0] + w1[:, 1] * halo[1], b1 + w1[:, 0] * halo[1] + w1[:, 1] * u], 1)   # [UW][2]
  e = np.zeros(UW)
  for hu in range(NH + NKV):
    dd = d[hu * HD:(hu + 1) * HD].reshape(-1); e[hu * HD:(hu + 1) * HD] = b2[hu * HD:(hu + 1) * HD] + w2[hu * HD:(hu + 1) * HD].reshape(HD, -1) @ dd
  qp = u[:NH * HD].reshape(NH, HD); kp = u[NH * HD:].reshape(NKV, HD)
  qr = 0.5 * (qp + np.repeat(kp, g, 0)); kr = qr.reshape(NKV, g, HD).mean(1)
  q = e[:NH * HD].reshape(NH, HD) + qr; k = e[NH * HD:].reshape(NKV, HD) + kr
  def fin(x, mul):
    n = np.linalg.norm(x); x = x * (mul * HD ** 0.5 / max(n, eps)); x1, x2 = x[:HR].copy(), x[HR:ROT].copy(); c, s = cs[:HR], cs[HR:]
    x[:HR] = x1 * c - x2 * s; x[HR:ROT] = x2 * c + x1 * s; return x
  q = np.stack([fin(q[h], 1.0) for h in range(NH)]); k = np.stack([fin(k[j], temp[j]) for j in range(NKV)])
  v = np.stack([vcd[:HD], vdel]); halo2 = np.stack([halo[1], u]); vdel2 = vcd[HD:].copy()
  return q, k, v, halo2, vdel2

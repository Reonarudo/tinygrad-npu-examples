"""MoE routing on the device: the router's logits row -> the expert slots and weights the qdot GEMV heads and moe_comb2 read, so a layer
runs as one graph with no host sync. One task does the work (NE <= 512 logits; the others return at once). Variants:

  softmax      Qwen3 / Qwen3.5: p = softmax(l); top k; w = p[top] / sum                                     (norm_topk_prob)
  sigmoid_bias LFM2.5: s = sigmoid(l); top k of s + b; w = s[top] / max(sum, 6.1e-5)
  group        Ring / Ling 2.0: s = sigmoid(l); sel = s + b; NG groups, the TG with the largest top-2 sums kept; top k of sel;
               w = s[top] / max(sum, 6.1e-5) * scale
  topk_softmax gpt-oss: l += b; top k of l; w = softmax(l[top])
Ties: the first index wins (argsort stable on the host).
Out: sel_gu [16] (expert e -> e S_gu; -1 past k), sel_dn [16] (e S_dn), wt [16] (the weights at `woff` .. woff + k - 1; wt[0] = `w0` when
woff = 1: the shared expert's weight, 1.0 or sigmoid(g . x) with `shared_gate`). Contract: `route_ref`.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
from zy import vec_f16 as V                                              # noqa: E402
from mamba2 import HELPERS                                               # noqa: E402  (ex1: a scalar exp through the vector exp2 polynomial; the C compiler has no scalar expf)

NT = 4


def route_src(kind, NE, TOPK, S_gu, S_dn, NG=1, TG=1, scale=1.0, woff=0, w0=1.0):
  """args: sel_gu (int32 [16], out), sel_dn (int32 [16]), wt (float [16]), logits (float [>= NE]), bias (float [NE]; any buffer for
  softmax / topk variants without one), g (float [>= 1]: the shared expert's gate logit at g[0] when kind ends with "+sg": wt[0] = sigmoid(g[0]))."""
  base, sg = (kind[:-3], True) if kind.endswith("+sg") else (kind, False); assert base in ("softmax", "sigmoid_bias", "group", "topk_softmax")
  assert NE % NG == 0 and 1 <= TOPK <= 16 and woff + TOPK <= 16; GS = NE // NG
  if base == "softmax": score = "float m = L[0]; for (int e = 1; e < NE; e++) m = L[e] > m ? L[e] : m; float z = 0.0f; for (int e = 0; e < NE; e++) { s[e] = ex1(L[e] - m); z += s[e]; } for (int e = 0; e < NE; e++) { s[e] /= z; sel[e] = s[e]; }"
  elif base == "topk_softmax": score = "for (int e = 0; e < NE; e++) { s[e] = L[e] + B[e]; sel[e] = s[e]; }"
  else: score = "for (int e = 0; e < NE; e++) { s[e] = 1.0f / (1.0f + ex1(-L[e])); sel[e] = s[e] + B[e]; }"
  group = f"""
  for (int g = 0; g < {NG}; g++) {{                                       /* a group's score: its top-2 selection probs */
    float a = -3.0e38f, b = -3.0e38f;
    for (int e = g * {GS}; e < (g + 1) * {GS}; e++) {{ float v = sel[e]; if (v > a) {{ b = a; a = v; }} else if (v > b) b = v; }}
    gs[g] = a + b;
  }}
  for (int t = 0; t < {TG}; t++) {{ int bi = -1; float bv = -3.0e38f; for (int g = 0; g < {NG}; g++) if (!gk[g] && gs[g] > bv) {{ bv = gs[g]; bi = g; }} gk[bi] = 1; }}
  for (int g = 0; g < {NG}; g++) if (!gk[g]) for (int e = g * {GS}; e < (g + 1) * {GS}; e++) sel[e] = -3.0e38f;""" if base == "group" else ""
  if base == "topk_softmax": weights = "float mm = tv[0]; for (int k = 1; k < TOPK; k++) mm = tv[k] > mm ? tv[k] : mm; float zz = 0.0f; for (int k = 0; k < TOPK; k++) { tv[k] = ex1(tv[k] - mm); zz += tv[k]; } for (int k = 0; k < TOPK; k++) tv[k] /= zz;"
  elif base == "softmax": weights = "float zz = 0.0f; for (int k = 0; k < TOPK; k++) zz += tv[k]; for (int k = 0; k < TOPK; k++) tv[k] /= zz;"
  else: weights = f"float zz = 0.0f; for (int k = 0; k < TOPK; k++) zz += tv[k]; zz = zz > 6.103515625e-5f ? zz : 6.103515625e-5f; for (int k = 0; k < TOPK; k++) tv[k] = tv[k] / zz * {scale!r}f;"
  w0s = "1.0f / (1.0f + ex1(-g[0]))" if sg else f"{w0!r}f"
  return V.FULL_H + HELPERS + f"""
#define NE {NE}
#define TOPK {TOPK}
__kernel void moe_route(__global int* restrict sel_gu, __global int* restrict sel_dn, __global float* restrict wt, __global float* restrict L,
                        __global float* restrict B, __global float* restrict g, const int core_id) {{
  if (core_id != 0) return;
  float s[NE], sel[NE], gs[{NG}]; int gk[{NG}]; for (int i = 0; i < {NG}; i++) gk[i] = 0;
  {score}{group}
  int id[TOPK]; float tv[TOPK];
  for (int k = 0; k < TOPK; k++) {{                                      /* the largest left, the first index on ties */
    int bi = 0; float bv = -3.0e38f;
    for (int e = 0; e < NE; e++) if (sel[e] > bv) {{ bv = sel[e]; bi = e; }}
    id[k] = bi; tv[k] = {"sel[bi]" if base == "topk_softmax" else "s[bi]"}; sel[bi] = -3.0e38f;
  }}
  {weights}
  for (int i = 0; i < 16; i++) {{ sel_gu[i] = -1; sel_dn[i] = -1; wt[i] = 0.0f; }}
  for (int k = 0; k < TOPK; k++) {{ sel_gu[k] = id[k] * {S_gu}; sel_dn[k] = id[k] * {S_dn}; wt[{woff} + k] = tv[k]; }}
  {"wt[0] = " + w0s + ";" if woff else ""}
}}"""


def route_ref(kind, logits, bias, TOPK, NG=1, TG=1, scale=1.0, g=None):
  """-> (top [TOPK] in selection order, weights [TOPK], the shared weight or None), float64 (the host's rules)."""
  base, sg = (kind[:-3], True) if kind.endswith("+sg") else (kind, False); l = np.asarray(logits, np.float64)[:len(bias) if bias is not None else None]; NE = l.size
  if base == "softmax": s = np.exp(l - l.max()); s /= s.sum(); sel = s.copy()
  elif base == "topk_softmax": s = l + bias; sel = s.copy()
  else: s = 1.0 / (1.0 + np.exp(-l)); sel = s + bias
  if base == "group":
    GS = NE // NG; grp = np.sort(sel.reshape(NG, GS), -1)[:, -2:].sum(-1); keep = np.argsort(-grp, kind="stable")[:TG]
    mask = np.full(NG, -np.inf); mask[keep] = 0.0; sel = (sel.reshape(NG, GS) + mask[:, None]).reshape(NE)
  top = np.argsort(-sel, kind="stable")[:TOPK]; tv = sel[top] if base == "topk_softmax" else s[top]
  if base == "topk_softmax": w = np.exp(tv - tv.max()); w /= w.sum()
  elif base == "softmax": w = tv / tv.sum()
  else: w = tv / max(tv.sum(), 6.103515625e-5) * scale
  return top, w, (1.0 / (1.0 + np.exp(-float(g[0]))) if sg else None)

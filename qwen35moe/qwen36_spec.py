#!/usr/bin/env python3
"""Qwen3.6-35B-A3B speculative decoding on the NPU: the Qwen3.8 runtime (qwen38_generate: verify passes of M rows, the per-token
DeltaNet state banks and commits, the MTP drafter, the leaf tree) over qwen35moe_pack.py's cache completed by qwen36_spec_pack.py, with
the MoE in place of the dense MLP (MoELayers.mlp: the router and the shared gate as gemv32s of the rows, the top-8 routing on the device
for M rows, the experts' gate / up as qdot GEMV heads sharing a row's package, the down from their own rows, the shared expert, the
weighted combine) and Qwen3.6's own MTP layer -- a MoE layer too -- as layer 40. The head: the Q6_K head requantised to Q8_0 (the runtime
runs one GEMM mode per cache); plain decoding's ids are checked against qwen35moe_generate's exact path.
    python3 qwen36_spec.py "prompt" [--max-new 64]          (qwen38_generate's options; QWEN_DRAFT=none: plain greedy through the verify path)
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("QWEN_MODEL", "qwen3.5-35b-a3b")
os.environ.setdefault("QWEN_NPU", os.path.expanduser(os.environ.get("QWEN35MOE_NPU", "~/npu-packs/qwen36moe-npu")))
os.environ.setdefault("QWEN_DIR", os.environ.get("QWEN35MOE_GGUF", "/mnt/ssd/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"))
os.environ.setdefault("QWEN_TOK", "/mnt/ssd/ornith-9b-npu"); os.environ.setdefault("QWEN_PREFILL", "chunked"); os.environ.setdefault("QWEN_LAYER_BLOCK", "1")
os.environ.setdefault("QWEN_MTP_CACHE", os.path.join(os.environ["QWEN_NPU"], "mtp")); os.environ.setdefault("QWEN_SPEC", "6"); os.environ.setdefault("QWEN_SPEC_GEOS", "3,4,6"); os.environ.setdefault("QWEN_PIN_GB", "26")
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "tinygrad"))))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "qwen3.8-27b"), os.path.join(HERE, "..", "qwen3-30b-a3b"), os.path.join(HERE, "..", "ornith-9b"), os.path.join(HERE, "..", "seqmix")]
import numpy as np                                                        # noqa: E402
from tinygrad import Tensor, dtypes                                      # noqa: E402
from zy import vec_f16 as V                                              # noqa: E402
import qwen38_npu as QN, qwen38_ref as R, qwen38_generate as QG          # noqa: E402
import qwen3moe_qgemv as QGV                                             # noqa: E402
from mamba2 import HELPERS                                               # noqa: E402  (ex1)

NT = 12; TOPK = 8; KIND = {2: "q8", 5: "q4", 6: "q6n", 7: "q5n"}; DEV = "ZHOUYI"


def rms_rows_src(H, rows, stacked):
  """out[r] = rms(x[r]) w (w: a [H] vector, or row idx[0] of a stack [L][H]) for r < rows; rows of pitch H."""
  w = f"w + idx[0] * {H} + i" if stacked else "w + i"
  return V.FULL_H + f"""
__kernel void rms_rows(__global float* restrict out, __global float* restrict x, __global float* restrict w, __global int* restrict idx, const int core_id) {{
  for (int r = 0; r < {rows}; r++) {{
    __global float* xr = x + r * {H}; float8 s = BC(0.0f); for (int i = 0; i < {H}; i += 8) {{ float8 v = *(__global float8*)(xr + i); s += v * v; }}
    float8 inv = BC(1.0f / __builtin_sqrtf(hsum8(s) * {1.0 / H!r}f + {R.EPS!r}f));
    for (int i = core_id * 8; i < {H}; i += {8 * NT}) *(__global float8*)(out + r * {H} + i) = *(__global float8*)(xr + i) * inv * *(__global float8*)({w});
  }}
}}"""


def route_rows_src(NE, K_, rows, S_gu, S_dn):
  """Task r < rows: softmax over row r's logits, the top K_ (first index on ties), renormalised; the shared gate sigmoid(g[r][0]).
  sel_gu[r K_ + k] = e S_gu, sel_dn the same with S_dn (compact: head r K_ + k of the GEMVs), wt[r (K_ + 1)] = the shared gate, [1 + k] = w_k."""
  return V.FULL_H + HELPERS + f"""
__kernel void route_rows(__global int* restrict sel_gu, __global int* restrict sel_dn, __global float* restrict wt, __global float* restrict L, __global float* restrict g, const int core_id) {{
  if (core_id >= {rows}) return; int r = core_id; __global float* Lr = L + r * {NE};
  float s[{NE}]; float m = Lr[0]; for (int e = 1; e < {NE}; e++) m = Lr[e] > m ? Lr[e] : m;
  float z = 0.0f; for (int e = 0; e < {NE}; e++) {{ s[e] = ex1(Lr[e] - m); z += s[e]; }}
  int id[{K_}]; float tv[{K_}]; float zz = 0.0f;
  for (int k = 0; k < {K_}; k++) {{ int bi = 0; float bv = -1.0f; for (int e = 0; e < {NE}; e++) if (s[e] > bv) {{ bv = s[e]; bi = e; }} id[k] = bi; tv[k] = bv; zz += bv; s[bi] = -1.0f; }}
  for (int k = 0; k < {K_}; k++) {{ sel_gu[r * {K_} + k] = id[k] * {S_gu}; sel_dn[r * {K_} + k] = id[k] * {S_dn}; wt[r * {K_ + 1} + 1 + k] = tv[k] / zz; }}
  wt[r * {K_ + 1}] = 1.0f / (1.0f + ex1(-g[r * 8]));
}}"""


def comb_rows_src(H, NPH, K_, rows):
  """out[r] = x[r] + wt[r (K_ + 1)] ysd[r] + sum_k wt[r (K_ + 1) + 1 + k] y[r K_ + k]  (y rows of pitch NPH)."""
  terms = " + ".join(f"BC(wt[r * {K_ + 1} + {1 + k}]) * *(__global float8*)(y + (r * {K_} + {k}) * {NPH} + i)" for k in range(K_))
  return V.FULL_H + f"""
__kernel void comb_rows(__global float* restrict out, __global float* restrict x, __global float* restrict ysd, __global float* restrict y, __global float* restrict wt, const int core_id) {{
  for (int r = 0; r < {rows}; r++) for (int i = core_id * 8; i < {H}; i += {8 * NT})
    *(__global float8*)(out + r * {H} + i) = *(__global float8*)(x + r * {H} + i) + BC(wt[r * {K_ + 1}]) * *(__global float8*)(ysd + r * {NPH} + i) + {terms};
}}"""


class MoELayers(QN.Layers):
  """qwen38_npu.Layers over the qwen35moe cache: its 5-tuple meta (code, N, npad, K, S) -> (N, npad, K) + the codes / strides; the MoE MLP."""
  def _meta(self, l):
    if l not in self._metas:
      meta, small = super()._meta(l)
      if len(next(iter(v for k, v in meta.items() if not k.startswith("_")))) == 5:
        code = {k: v[0] for k, v in meta.items() if not k.startswith("_")}; S = {k: v[4] for k, v in meta.items() if not k.startswith("_")}
        meta = {k: (v[1], v[2], v[3]) for k, v in meta.items() if not k.startswith("_")}; meta["_code"] = code; meta["_S"] = S
      self._metas[l] = (meta, small)
    return self._metas[l]
  def kind(self, l): return tuple(sorted(self._meta(l)[0]["_code"].items()))
  def qgemv(self, kd, N, Kd, y, w, sel, pkg, heads=1, act_heads=1, act="pkg", gu_np=0, yoff=0, xs=None, nbuf=3):
    K = self.K; nbuf = min(nbuf, max(1, (32768 - Kd // 128 * QGV.SL_REC - 256 - (2 * Kd * 4 if act == "swiglu" else 0)) // QGV.SLB[kd]))
    key = f"moe_qgemv|{kd}|{N}|{Kd}|{heads}|{act_heads}|{act}|{gu_np}|{yoff}|{nbuf}"
    src = QGV.qgemvk_src(kd, N, Kd, heads=heads, act_heads=act_heads, act=act, gu_np=gu_np, yoff=yoff, eps=R.EPS, nbuf=nbuf)
    return K.call(key, src, y, pkg, w, sel, K.desc(f"q_desc|{kd}|{Kd}", lambda: QGV.qgemvk_desc(kd, Kd)), y, xs if xs is not None else pkg, pkg)
  def mlp(self, x, slot, meta, w1_post, out="x2"):
    K = self.K; M = self.n; H, FF = R.H, R.INTER; code, S = meta["_code"], meta["_S"]; NE = meta["router"][0]; NPF, NPH = QGV.qgemvk_np(FF), QGV.qgemvk_np(H); AB = H // 128 * QGV.SL_REC // 4
    xn = K.rows_buf("moe_xn", H); pk = K.buf("moe_pk", M * AB, dtypes.float32)
    y_gu, y_sgu = K.buf("moe_ygu", 2 * M * TOPK * NPF, dtypes.float32), K.buf("moe_ysgu", 2 * M * NPF, dtypes.float32); y_dn, y_sdn = K.buf("moe_ydn", M * TOPK * NPH, dtypes.float32), K.buf("moe_ysdn", M * NPH, dtypes.float32)
    sel_gu, sel_dn, wt = K.buf("moe_selgu", 16 * M, dtypes.int32), K.buf("moe_seldn", 16 * M, dtypes.int32), K.buf("moe_wt", 16 * M, dtypes.float32)
    sel0 = K.hold("moe_sel0", Tensor.zeros(16, dtype=dtypes.int32, device=DEV).contiguous().realize())
    if isinstance(w1_post, tuple): K.call(f"rms_rows|{H}|{M}|1", rms_rows_src(H, M, True), xn, x, w1_post[0], w1_post[1])
    else: K.call(f"rms_rows|{H}|{M}|0", rms_rows_src(H, M, False), xn, x, w1_post, sel0)
    K.call(f"qact4m|{H}|{M}|plain", QGV.qact4m_src(H, heads=M, mode="plain"), pk, xn, xn)
    rl = K.gemv(x, slot["router"], H, NE, "moe_rlog", norm=True); g = K.gemv(x, slot["shg"], H, 8, "moe_sgl", norm=True)
    K.call(f"route_rows|{NE}|{TOPK}|{M}|{S['gate']}|{S['down']}", route_rows_src(NE, TOPK, M, S["gate"], S["down"]), sel_gu, sel_dn, wt, rl, g)
    self.qgemv(KIND[code["gate"]], FF, H, y_gu, slot["gate"], sel_gu, pk, heads=M * TOPK, act_heads=M); self.qgemv(KIND[code["up"]], FF, H, y_gu, slot["up"], sel_gu, pk, heads=M * TOPK, act_heads=M, yoff=M * TOPK * NPF)
    self.qgemv(KIND[code["sgate"]], FF, H, y_sgu, slot["sgate"], sel0, pk, heads=M, act_heads=M, nbuf=2); self.qgemv(KIND[code["sup"]], FF, H, y_sgu, slot["sup"], sel0, pk, heads=M, act_heads=M, yoff=M * NPF, nbuf=2)
    self.qgemv(KIND[code["down"]], H, FF, y_dn, slot["down"], sel_dn, pk, heads=M * TOPK, act="swiglu", gu_np=NPF, xs=y_gu)
    self.qgemv(KIND[code["sdn"]], H, FF, y_sdn, slot["sdn"], sel0, pk, heads=M, act="swiglu", gu_np=NPF, xs=y_sgu)
    return K.call(f"comb_rows|{H}|{NPH}|{TOPK}|{M}", comb_rows_src(H, NPH, TOPK, M), K.rows_buf(out, H), x, y_sdn, y_dn, wt)


QG.Layers = MoELayers

if __name__ == "__main__": QG.main()

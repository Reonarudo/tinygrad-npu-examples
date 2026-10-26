"""Each Qwen3.6 kernel timed as a TinyJit replay on the board (device time a launch, min of 5 x 10): QWEN35MOE_GGUF=... TG=../tinygrad python3 bench_qwen36.py"""
import os, sys, time, functools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.environ["QWEN35MOE_JIT"] = "0"; os.environ["QWEN35MOE_GGUF"] = "/mnt/ssd/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
import qwen35moe_generate as G, numpy as np
from qwen35moe_generate import R, Device, QG, RT, NE, TOPK, FF, H, C_QKV, W, gemm, QN
from tinygrad.engine.jit import TinyJit
m = G.Model(os.environ.get("QWEN35MOE_NPU", os.path.expanduser("~/npu-packs/qwen36moe-npu")), tmax=512); K = m.K; dev = Device["ZHOUYI"]
def prep(l):
  m.layer_w(l); i = m.aidx.get(l, m.gidx.get(l)); 
  if l in m.aidx: G.poke(m.aix[i], np.array([i, 17], np.int32))
  G.poke(m.posb, np.array([17], np.int32)); return W(m.views[(l % 2, m.kind[l])], m.metas[l], m.sms[l])
cases = {}
for l in (0, 3):
  Wl = prep(l); x = m.xio[l % 2]; x1 = m.x1; kind = R.LAYER_TYPES[l]
  cases[f"L{l} whole layer ({kind})"] = (l, lambda l=l: m.whole(l, None))
  cases[f"L{l} rms_a"] = (l, lambda l=l, x=x, kind=kind: K.rms_a(x, (m.Win_g, m.sidx[m.gidx[l]]) if kind == "linear" else (m.Win_a, m.lidx[m.aidx[l]]), H, "a_h"))
  if kind == "linear":
    cases[f"L{l} gemm qkv+z (q8f)"] = (l, lambda l=l, Wl=Wl: (gemm(K, Wl, "qkv", K.bufs["a_h"] if "a_h" in K.bufs else K.rms_a(m.xio[l % 2], (m.Win_g, m.sidx[m.gidx[l]]), H, "a_h"), 1, K.bufs["ct_qkv"]), gemm(K, Wl, "z", K.rms_a(m.xio[l % 2], (m.Win_g, m.sidx[m.gidx[l]]), H, "a_h"), 1, K.bufs["ct_z"]))[-1])
    cases[f"L{l} gdn_tok3"] = (l, lambda l=l: K.gdn_tok3(m.Sall, m.Call, m.sidx[m.gidx[l]], m.posb, K.bufs["ct_qkv"], K.bufs["ct_z"], m.xio[l % 2], m.Wab, m.Adt, m.Cwt, m.Nw, R.NV, R.NK, R.DK, R.DV, C_QKV, R.CONV, H))
  else:
    cases[f"L{l} gemm q+kv (q8f)"] = (l, lambda l=l, Wl=Wl: (gemm(K, Wl, "q", K.rms_a(m.xio[l % 2], (m.Win_a, m.lidx[m.aidx[l]]), H, "a_h"), 1, K.bufs["ct_q"]), gemm(K, Wl, "kv", K.rms_a(m.xio[l % 2], (m.Win_a, m.lidx[m.aidx[l]]), H, "a_h"), 1, K.bufs["ct_kv"]))[-1])
    cases[f"L{l} unpack x2 + attn_dec2"] = (l, lambda l=l: K.attn_dec2(K.unpack(K.bufs["ct_q"], R.NH * 2 * R.HD, "q_rows"), K.unpack(K.bufs["ct_kv"], 2 * R.NKV * R.HD, "kv_rows"), m.QN, m.KN, m.ROPE, m.Kall, m.Vall, m.aix[m.aidx[l]], R.NH, R.NKV, R.HD, m.tmax, R.ROT))
  cases[f"L{l} gemm o (q8f) + resid"] = (l, lambda l=l, Wl=Wl, x=x: K.resid(x, gemm(K, Wl, "o", K.rms_a(K.bufs["o_rows"] if "o_rows" in K.bufs else x, None, R.NH * R.HD if kind == "full" else H, "a_o2", norm=False), 1, K.bufs["ct_o"]), H, "x1"))
  cases[f"L{l} gemv rlog 256 + sgl 8"] = (l, lambda l=l, x1=x1: (K.gemv(x1, m.router, H, NE, "rlog", norm=True, idx=m.lidx[l]), K.gemv(x1, m.shg8, H, 8, "sgl", norm=True, idx=m.lidx[l]))[-1])
  cases[f"L{l} route"] = (l, lambda l=l, Wl=Wl: K.call(f"route|softmax+sg|{NE}|{TOPK}|{Wl.meta['gate'][4]}|{Wl.meta['down'][4]}", RT.route_src("softmax+sg", NE, TOPK, Wl.meta["gate"][4], Wl.meta["down"][4], woff=1), m.sel_gu, m.sel_dn, m.wt, K.bufs["rlog"], K.bufs["rlog"], K.bufs["sgl"]))
  cases[f"L{l} gate+up q4m 8x512x2048"] = (l, lambda l=l, Wl=Wl, x1=x1: (m.qgemv("q4", FF, H, m.y_gu, Wl.st["gate"], m.sel_gu, x1, m.wpost_l[l], heads=TOPK), m.qgemv("q4", FF, H, m.y_gu, Wl.st["up"], m.sel_gu, x1, m.wpost_l[l], heads=TOPK, yoff=TOPK * m.NPF))[-1])
  cases[f"L{l} shared gate+up q8 512x2048"] = (l, lambda l=l, Wl=Wl, x1=x1: (m.qgemv("q8", FF, H, m.y_sgu, Wl.st["sgate"], m.sel1, x1, m.wpost_l[l], nbuf=2), m.qgemv("q8", FF, H, m.y_sgu, Wl.st["sup"], m.sel1, x1, m.wpost_l[l], yoff=m.NPF, nbuf=2))[-1])
  cases[f"L{l} down {QN.Layer.KIND[Wl.meta['down'][0]]} 8x2048x512"] = (l, lambda l=l, Wl=Wl, x1=x1: m.qgemv(QN.Layer.KIND[Wl.meta["down"][0]], H, FF, m.y_dn, Wl.st["down"], m.sel_dn, m.y_gu, x1, heads=TOPK, act="swiglu", gu_np=m.NPF, yoff=QG.qgemvk_np(H)))
  cases[f"L{l} shared down q8 2048x512"] = (l, lambda l=l, Wl=Wl, x1=x1: m.qgemv("q8", H, FF, m.y_dn, Wl.st["sdn"], m.sel1, m.y_sgu, x1, act="swiglu", gu_np=m.NPF))
  cases[f"L{l} moe_comb2 x9"] = (l, lambda l=l: K.call(f"moe_comb2|{H}|{QG.qgemvk_np(H)}|{TOPK + 1}", QG.moe_comb2_src(H, QG.qgemvk_np(H), TOPK + 1, nt=4), m.xio[(l + 1) % 2], m.x1, m.y_dn, m.wt, K.desc(f"moe_comb2_desc|{H}|{QG.qgemvk_np(H)}|{TOPK + 1}", lambda: QG.moe_comb2_desc(H, QG.qgemvk_np(H), TOPK + 1))))
cases["head q6 248k x 2048"] = (0, lambda: K.call(f"qgemv|head|{m.Vn}", QG.qgemvk_src("q6", m.Vn, H, act="norm", top=True, eps=R.EPS, nbuf=2), m.tops, m.pk_head, m.head_all, m.sel0, m.d_head, m.tops, m.xio[0], m.w_out))
reps = 10
for name, (l, fn) in cases.items():
  prep(l); m.whole(l, None); dev.synchronize()
  def f(_):
    for _ in range(reps): r = fn()
    return r
  j = TinyJit(f)
  for _ in range(2): j(m.jd)
  dev.synchronize(); ts = []
  for _ in range(5):
    t0 = time.perf_counter(); j(m.jd); dev.synchronize(); ts.append((time.perf_counter() - t0) / reps)
  print(f"  {name:36s} {min(ts) * 1e6:9.0f} us", flush=True)

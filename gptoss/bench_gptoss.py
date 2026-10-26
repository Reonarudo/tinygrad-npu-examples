"""Each gpt-oss kernel timed as a TinyJit replay on the board (device time a launch, min of 5 x 10): TG=../tinygrad python3 bench_gptoss.py"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); os.environ["OSS_JIT"] = "0"
import gptoss_npu as GN, numpy as np
from tinygrad import Device
from tinygrad.engine.jit import TinyJit
m = GN.Model(tmax=512); c = m.c; K = m.K; dev = Device["ZHOUYI"]; l = 1; W = m.views[(1, m.kind[1])]; meta = m.metas[1]
for ll in range(c.NL): GN.poke(m.aix[ll], np.array([ll, 17], np.int32))
m.raw.slot_map(m.slots[1][0], m.wbufs[1][0], 0, m.packs[1][2], drain=False)
KIND = GN.KIND; H, HD, NKV, NH = c.H, c.HD, c.NKV, c.NH
cases = {
  "rms_pad": lambda: m.normed(m.xio[1], m.w_attn[l]),
  "qact4m K2944": lambda: m.package(m.xn, m.pk, m.KP),
  "qact4m K2944 x4": lambda: m.package(m.hrows, m.pk4, m.KP, heads=c.TOPK),
  "q8 q 4096x2944": lambda: m.qgemv("q8", meta["q"][1], m.KP, m.y_q, W["q"], m.sel0, m.pk),
  "q8 k 512x2944": lambda: m.qgemv("q8", meta["k"][1], m.KP, m.y_k, W["k"], m.sel0, m.pk),
  "q8 o 2880x4096": lambda: m.qgemv("q8", H, meta["o"][3], m.y_o, W["o"], m.sel0, m.pk_o),
  "mx4 gate 4x2880x2944": lambda: m.qgemv("mx4", c.FF, m.KP, m.y_g, W["gate"], m.sel_e, m.pk, heads=c.TOPK),
  "mx4 down 4x2880x2944": lambda: m.qgemv("mx4", H, m.KP, m.y_dn, W["down"], m.sel_d, m.pk4, heads=c.TOPK, act_heads=c.TOPK),
  "qkv_rope": lambda: K.call(f"qkv_rope|{NH}|{NKV}|{HD}", GN.GKN.qkv_rope_src(NH, NKV, HD), m.q_f, m.k_f, m.v_f, m.y_q, m.y_k, m.y_v, m.bq[l], m.bk[l], m.bv[l], m.cstab, m.aix[l]),
  "attn_part full": lambda: K.call(f"attn_part|{NH}|{NKV}|{HD}|{m.tmax}|0|s", GN.SA.attn_part_src(NH, NKV, HD, 1.0 / np.sqrt(HD), m.tmax, window=0, sinks=True), m.part, m.q_f, m.Kl, m.Vl, m.k_f, m.v_f, m.sink, m.aix[l], m.d_attn),
  "attn_comb": lambda: K.call(f"attn_comb|{NH}|{NKV}|{HD}", GN.SA.attn_comb_src(NH, NKV, HD), m.o_f, m.part),
  "rowadd3": lambda: K.call(f"rowadd3|{H}", GN.RW.rowadd3_src(H), m.x1, m.xio[1], m.y_o, m.bo[l]),
  "rlog gemv32 3072x32": lambda: K.gemv(m.xn, m.router_p[l], m.KPR, c.NE, "rlog"),
  "route": lambda: K.call(f"route|topk_softmax|{c.NE}|{c.TOPK}|{meta['gate'][4]}|{meta['down'][4]}", GN.RT.route_src("topk_softmax", c.NE, c.TOPK, meta["gate"][4], meta["down"][4]), m.sel_e, m.sel_d, m.wt4, m.y_o, m.rbias[l], m.y_o),
  "oai_glu": lambda: K.call(f"oai_glu|{c.TOPK}|{c.FF}|{m.NPF}|{m.KP}|{meta['gate'][4]}", GN.GKN.oai_glu_src(c.TOPK, c.FF, m.NPF, m.KP, meta["gate"][4]), m.hrows, m.y_g, m.y_u, m.bg[l], m.bu[l], m.sel_e),
  "oss_comb": lambda: K.call(f"oss_comb|{c.TOPK}|{H}|{m.NPH}|{meta['down'][4]}", GN.GKN.oss_comb_src(c.TOPK, H, m.NPH, meta["down"][4]), m.xio[0], m.x1, m.y_dn, m.wt4, m.bd[l], m.sel_d),
  "head q8 201k x2944": lambda: m.qgemv("q8", m.Vn, m.KP, m.tops, m.head_all, m.sel0, m.pk, top=True, nbuf=2),
}
reps = 10
for name, fn in cases.items():
  def f(_):
    for _ in range(reps): r = fn()
    return r
  j = TinyJit(f)
  for _ in range(2): j(m.jd)
  dev.synchronize(); ts = []
  for _ in range(5):
    t0 = time.perf_counter(); j(m.jd); dev.synchronize(); ts.append((time.perf_counter() - t0) / reps)
  print(f"  {name:26s} {min(ts) * 1e6:9.0f} us", flush=True)

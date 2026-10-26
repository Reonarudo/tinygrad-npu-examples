"""The decode layer on the qdot GEMVs (qwen3moe_qgemv): every linear an integer dot product over the same gemm_gs streams the prefill
reads, the activations in int8 a 32 (llama.cpp's own CPU path quantises them to 8 bits too), and the layout steps folded into the
kernels next to them -- 9 launches a layer (10 where v is Q6_K):

    qgemv attn   the q | k (| v) GEMV, its package built in-kernel from rms(x) w_in (exported where a Q6_K v GEMV follows),
                 q -> q_rows, k | v -> kv_rows (the split output)
    qgemv v      (Q6_K v layers) into kv_rows + 512
    gattn_kvd    gemma4's (k norm + RoPE into the cache)
    gpart        gemma4's gattn_part for one row (the attention's slices) with 8-position blocks and vector reductions
    comb_pkg     gattn_comb that lays each head's output as a record of the o projection's package
    qgemv o      -> y_o
    router_x1    x1 = x + y_o, the router's logits and the experts' package (rms(x1) w_ffn)
    qgemv gu     gate | up of the 8 experts, one 16-head launch -> y_gu [16][768]; every task picks the top 8 from the logits
                 (task 0 writes down's slots and the 8 weights)
    qgemv down   the 8 experts, each head's package built in-kernel from silu(gate) up -> y_dn [8][2064]
    moe_comb2    (4 tasks) out = x1 + sum_h w_h y_dn[h]
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import qwen3moe_npu as QN                                                 # noqa: E402  (paths, Kernels, the gemma4 kernels)
from qwen3moe_npu import OA, dtypes, dev, GK, MK                          # noqa: E402
import qwen3moe_qgemv as Q                                                # noqa: E402

O_KP = int(os.environ.get("QWEN3MOE_O_KP", "1"))                           # the o projection's K parts (4: no faster on the board)
R16 = os.environ.get("QWEN3MOE_R16", "1") == "1"                           # the router's weights in fp16 (-8 us a layer; the same ids on the three prompts)
MC_NT = int(os.environ.get("QWEN3MOE_MC_NT", "4"))                      # moe_comb2's tasks
Q4N = os.environ.get("QWEN3MOE_Q4N", "1") == "1"                           # the attention's Q4_K linears over their decode-only 4.75-bit strip streams (attn_q4n, o_q4n) when packed
Q4N_CS = int(os.environ.get("QWEN3MOE_Q4N_CS", "4"))                       # slices a weight DMA for them (a strip's slice is 1216 B: 4 make a 4.9 KB request)
Q6N = os.environ.get("QWEN3MOE_Q6N", "1") == "1"                           # the Q6_K linears over their decode-only 6.6-bit copies (down_q6n, v_q6n) when packed
GP_KV = os.environ.get("QWEN3MOE_GP_KV", "1") == "1"                       # gattn_kvd folded into gpart (one launch fewer a layer)

FUSE = os.environ.get("QWEN3MOE_FUSE", "0") == "1"           # a layer as ONE launch, its kernels as phases with device barriers (no faster: off)
FUSE_TAIL = os.environ.get("QWEN3MOE_FUSE_TAIL", "1") == "1"  # the fused graph's one-task no-op (a chain job, not a lone program's)

_REG = {}
TRACE = None                                                              # a list: qlayer appends (name, thunk) instead of only running
_PHASES = None                                                            # a list while a fused layer is being collected


def call(K, name, src, out, *ins, nt=None):
  """K.call with the task count of the kernel (the C function name = the key's first field); while a fused layer is collected,
  the kernel is recorded as a phase instead."""
  if _PHASES is not None: _PHASES.append((name, src, (out,) + ins)); return out
  if TRACE is not None: TRACE.append((name, lambda: call.__wrapped__(K, name, src, out, *ins, nt=nt)))
  return call.__wrapped__(K, name, src, out, *ins, nt=nt)


# ---- the fused layer: one 12-task launch, the kernels as inline phases, a device barrier between phases
# The 3 groups of a 12-task launch run as one wave (docs/Core/Distributing.md), so the tasks can meet: each cleans its core's write-back
# L2 (the epilogue's ctrl0[0x82] writeback and drain poll), writes its arrival epoch to its own line by DMA and polls the twelve lines by
# DMA (check/bench_barrier.py: 2.6 us against ~12-28 us a launch). Every input a phase takes from an earlier phase it reads by DMA
# (never a stale cached line: no cache maintenance invalidates), but sel_dn (plain loads), whose lines no task of the job held before.
BAR_H = r"""
#define LSI(off) ((__global int*)(LSRAM_BASE + (off)))
static inline __attribute__((always_inline)) void l2_clean(void) {
  int v = __builtin_aipu_mfctrl0(0x82); __builtin_aipu_mtctrl0(v | 4, 0x82);
  while (__builtin_aipu_mfctrl0(0x82) & 4) ;
}
static inline __attribute__((always_inline)) void backoff(int n) { for (int i = 0; i < n; i++) (void)__builtin_aipu_mfctrl0(0x82); }
/* arrive (my epoch to my line), then task 0 alone polls the NT arrival lines and publishes the release epoch (line NT); the others
   poll that one line, with a backoff between polls: the early tasks must not steal the DDR port from the ones still streaming */
static inline __attribute__((always_inline)) int dev_barrier(__global int* bar, int e, int core_id, __global int* bdesc) {
  l2_clean();
  LSI(%(ME)d)[0] = e; DMA_DRAIN(3, DESC(bdesc, 1), %(ME)d, (int)(bar + 16 * core_id)); DMA_WAIT(3);
  if (core_id == 0) {
    for (int n = 1; n <= %(CAP)d; n++) {
      DMA_FILL(3, DESC(bdesc, 0), %(ALL)d, (int)bar); DMA_WAIT(3);
      int ok = 1; for (int t = 0; t < %(NT)d; t++) ok &= (LSI(%(ALL)d)[16 * t] - e) >= 0;
      if (ok) { DMA_DRAIN(3, DESC(bdesc, 1), %(ME)d, (int)(bar + 16 * %(NT)d)); DMA_WAIT(3); return 0; }
      backoff(%(BO0)d);
    }
    return 1;
  }
  for (int n = 1; n <= %(CAP)d; n++) {
    DMA_FILL(3, DESC(bdesc, 1), %(ALL)d, (int)(bar + 16 * %(NT)d)); DMA_WAIT(3);
    if ((LSI(%(ALL)d)[0] - e) >= 0) return 0;
    backoff(%(BO)d);
  }
  return 1;
}
"""
BAR_NT, BAR_ME, BAR_ALL, BAR_CAP = 12, 32704, 31872, 4000000
BAR_BO = int(os.environ.get("QWEN3MOE_BAR_BO", "64"))                     # backoff reads between a waiter's polls (task 0: a quarter)


def bar_desc(): return Q.V._desc_slots((BAR_NT * 64,), (64,))


def fuse_src(phases, hd=None, eps=None, name="qlayer_f", known=None):
  """One kernel from the phases' (name, src, args): each `__kernel` becomes a static inline function (restrict dropped: a phase may
  get one buffer twice), the headers once (`known`: the header pieces the sources are built from, in order), a barrier after every
  phase but the last. Returns (src, the unique argument tensors)."""
  import re
  known = known if known is not None else [Q.V.FULL_H, GK.EXP, Q.QH, GK._norm_rope(hd, eps)]
  uniq, idx = [], {}
  for _, _, args in phases:
    for a in args:
      if id(a) not in idx: idx[id(a)] = len(uniq); uniq.append(a)
  funcs, calls = [], []
  for i, (nm, src, args) in enumerate(phases):
    k = src.index("__kernel void"); hdr = src[:k]
    for piece in known: hdr = hdr.replace(piece, "", 1)
    assert not hdr.strip(), (nm, hdr[:200])
    m = re.match(r"__kernel void (\w+)\((.*?)\)\s*\{", src[k:], re.S); params = [p.strip() for p in m.group(2).split(",")]
    assert params[-1] == "const int core_id" and len(params) == len(args) + 1, (nm, params, len(args))
    types = [p.rsplit(" ", 1)[0].replace("restrict", "").strip() for p in params[:-1]]
    funcs.append(f"static inline __attribute__((always_inline)) void ph{i}(" + ", ".join(p.replace(" restrict", "") for p in params) + ")" + src[k + m.end() - 1:])
    calls.append(f"  ph{i}({', '.join(f'({t})a{idx[id(a)]}' for t, a in zip(types, args))}, core_id);  /* {nm} */")
    if i + 1 < len(phases): calls.append("  e += 1; fails += dev_barrier(bar, e, core_id, bdesc);")
  body = "\n".join(calls)
  return ("".join(known) + BAR_H % dict(ME=BAR_ME, ALL=BAR_ALL, CAP=BAR_CAP, NT=BAR_NT, BO=BAR_BO, BO0=BAR_BO // 4) + "\n".join(funcs) + f"""
__kernel void {name}({", ".join(f"__global uchar* a{j}" for j in range(len(uniq)))}, __global int* bar, __global int* bdesc, __global int* bstat, const int core_id) {{
  DMA_FILL(3, DESC(bdesc, 1), {BAR_ME}, (int)(bar + 16 * core_id)); DMA_WAIT(3); int e = LSI({BAR_ME})[0], fails = 0;
{body}
  if (fails) bstat[16 * core_id] += fails;
}}"""), uniq


def _call(K, name, src, out, *ins, nt=None):
  if nt is None: return K.call(name, src, out, *ins)
  if name not in _REG: _REG[name] = OA.register_csrc(name.split("|")[0], src, ntasks=nt)
  OA.csrc_call(_REG[name], out, *ins).realize(); return out
call.__wrapped__ = _call


class QBufs:
  """The qdot layer's buffers beyond Kernels' (q_rows, kv_rows, o_rows, x1, rlog) and the per-layer routing tables."""
  def __init__(self, K, c):
    H, QN_ = c.H, c.NH * c.HD; self.npd = Q.qgemvk_np(H)
    self.pkg_h = K.buf("q_pkg_h", H // 128 * Q.SL_REC // 4, dtypes.float32)
    self.pkg_o = K.buf("q_pkg_o", QN_ // 128 * Q.SL_REC // 4, dtypes.float32)
    self.pkg_f = K.buf("q_pkg_f", H // 128 * Q.SL_REC // 4, dtypes.float32)
    self.y_o = K.buf(f"q_y_o|{O_KP}", O_KP * self.npd + 64, dtypes.float32)
    self.y_gu = K.buf("q_y_gu", 2 * c.TOPK * c.FF + 64, dtypes.float32)
    self.y_dn = K.buf("q_y_dn", c.TOPK * self.npd + 64, dtypes.float32)
    self.sel0 = K.buf("q_sel0", 16, dtypes.int32); QN.poke(self.sel0, np.array([0] + [-1] * 15, np.int32))
    self.rows = lambda tag, n: K.rows_buf(tag, n)


def qlayer(K, QB, sel_gu, sel_dn, wt, st, meta, sm, c, x, Kc, Vc, cs, posb, out, d_up):
  """One decode row through the layer: x (rows buffer, row 0) -> out. `st`: the layer's streams (attn, v, o, gate, down; up at
  gate + d_up bytes), `meta` their (q8, N, npad, K, S, goff..). FUSE: one launch (the phases fused, device barriers between)."""
  global _PHASES
  if not FUSE or TRACE is not None: return _qlayer(K, QB, sel_gu, sel_dn, wt, st, meta, sm, c, x, Kc, Vc, cs, posb, out, d_up)
  _PHASES = []
  try: _qlayer(K, QB, sel_gu, sel_dn, wt, st, meta, sm, c, x, Kc, Vc, cs, posb, out, d_up); phases = _PHASES
  finally: _PHASES = None
  key = "qlayer_f|" + "|".join(n for n, _, _ in phases) + f"|{meta['attn'][2]}|{meta['down'][0]}|{BAR_BO}"
  if key not in _REG:
    src, _ = fuse_src(phases, c.HD, c.EPS); _REG[key] = (OA.register_csrc("qlayer_f", src, ntasks=BAR_NT), src)
  uniq = []; seen = set()
  for _, _, args in phases:
    for a in args:
      if id(a) not in seen: seen.add(id(a)); uniq.append(a)
  bar = K.buf("q_bar", 16 * (BAR_NT + 1), dtypes.int32); bstat = K.buf("q_bstat", 16 * BAR_NT, dtypes.int32)
  OA.csrc_call(_REG[key][0], uniq[0], *uniq[1:], bar, QN.desc(K, "q_bar_desc", bar_desc), bstat).realize()
  if FUSE_TAIL:   # a one-task no-op after it: the graph is then a chain job, not a lone program's (measured ~90 us a layer dearer)
    if "q_tail" not in _REG: _REG["q_tail"] = OA.register_csrc("qtail", "__kernel void qtail(__global int* restrict s, const int core_id) { }", ntasks=1)
    OA.csrc_call(_REG["q_tail"], bstat).realize()
  return out


def _qlayer(K, QB, sel_gu, sel_dn, wt, st, meta, sm, c, x, Kc, Vc, cs, posb, out, d_up):
  H, QN_, KVN, FF = c.H, c.NH * c.HD, c.NKV * c.HD, c.FF; vq6 = "v" in meta
  q_rows, kv_rows = K.rows_buf("q_rows", QN_), K.rows_buf("kv_rows", 2 * KVN); x1 = K.rows_buf("x1", H)
  na, goff = meta["attn"][2], meta["attn"][6]; split = goff * 48
  d42 = QN.desc(K, f"q_desc|q4|{H}", lambda: Q.qgemvk_desc("q4", H))
  ak, acs = ("q4n", Q4N_CS) if Q4N and "attn_q4n" in st else ("q4", 1)
  call(K, f"qgemv|attn|{ak}|{acs}|{na}|{split}|{int(vq6)}", Q.qgemvk_src(ak, na, H, split=split, act="norm", export=vq6, eps=c.EPS, cs=acs),
       q_rows, QB.pkg_h, st["attn_q4n" if ak == "q4n" else "attn"], QB.sel0, QN.desc(K, f"q_desc|{ak}|{H}|{acs}", lambda: Q.qgemvk_desc(ak, H, cs=acs)), kv_rows, x, sm["w_in"])
  if vq6:
    vk = "q6n" if Q6N and "v_q6n" in st else "q6"
    call(K, f"qgemv|v|{vk}|{KVN}", Q.qgemvk_src(vk, KVN, H, yoff=KVN), kv_rows, QB.pkg_h, st["v_q6n" if vk == "q6n" else "v"], QB.sel0,
         QN.desc(K, f"q_desc|{vk}|{H}", lambda: Q.qgemvk_desc(vk, H)), kv_rows, x, x)
  HG, BT, P, MR = GK.gattn_cfgr(c.NH, c.NKV, c.HD, 1); part = K.buf("ga_part|1", GK.gattn_part_size(c.NH, c.NKV, c.HD, 1), dtypes.float32)
  if GP_KV:   # the new position's k / v into the caches by gpart itself (gattn_kvd folded in)
    call(K, f"gpart|{c.NKV}|{c.HD}|{Q.GP_BT}|kv{2 * KVN}", Q.gpart_src(c.NH, c.NKV, c.HD, c.EPS, QN_, kv=2 * KVN), part, q_rows, Kc, Vc, sm["qnw"], cs, posb,
         QN.desc(K, f"gpart_desc|{c.NKV}|{c.HD}|{Q.GP_BT}", lambda: Q.gpart_desc(c.NKV, c.HD)), kv_rows, sm["knw"])
  else:
    call(K, f"gattn_kvd|{c.NKV}|{c.HD}|1|{2 * KVN}|nov", GK.gattn_kvd_src(c.NKV, c.HD, c.EPS, 1, 2 * KVN, vnorm=False), Kc, Vc, kv_rows, sm["knw"], cs, posb,
         QN.desc(K, f"gattn_kvd_desc|{2 * KVN}|{c.HD}", lambda: GK.gattn_kvd_desc(2 * KVN, c.HD)))
    call(K, f"gpart|{c.NKV}|{c.HD}|{Q.GP_BT}", Q.gpart_src(c.NH, c.NKV, c.HD, c.EPS, QN_), part, q_rows, Kc, Vc, sm["qnw"], cs, posb,
         QN.desc(K, f"gpart_desc|{c.NKV}|{c.HD}|{Q.GP_BT}", lambda: Q.gpart_desc(c.NKV, c.HD)))
  o_rows = K.rows_buf("o_rows", QN_)
  call(K, f"comb_pkg|{c.NKV}|{c.HD}|u", Q.comb_pkg_src(c.NH, c.NKV, c.HD), QB.pkg_o, part,
       QN.desc(K, f"comb_pkg_desc|{c.NKV}|{c.HD}", lambda: Q.comb_pkg_desc(c.NH, c.NKV, c.HD)), o_rows)
  ok_, ocs = ("q4n", Q4N_CS) if Q4N and "o_q4n" in st else ("q4", 1)
  call(K, f"qgemv|o|{ok_}|{ocs}|{H}|{QN_}|{O_KP}", Q.qgemvk_src(ok_, H, QN_, kp=O_KP, cs=ocs), QB.y_o, QB.pkg_o, st["o_q4n" if ok_ == "q4n" else "o"], QB.sel0,
       QN.desc(K, f"q_desc|{ok_}|{QN_}|{ocs}", lambda: Q.qgemvk_desc(ok_, QN_, cs=ocs)), QB.y_o, x, x)
  rlog = K.buf("q_rlog", 16 * c.NE // 16 + 64, dtypes.float32)
  wr = sm["router16"] if R16 else sm["router"]
  call(K, f"router_x1|{H}|{c.NE}|{O_KP}|{int(R16)}", Q.router_x1_src(H, c.NE, c.EPS, kp=O_KP, rs=QB.npd, w16=R16), rlog, x, QB.y_o, wr, sm["w_ffn"], x1, QB.pkg_f,
       QN.desc(K, f"router_x1_desc|{H}|{int(R16)}", lambda: Q.router_x1_desc(H, R16)))
  dk = "down_q6n" if Q6N and "down_q6n" in st else "down"; kd = "q6n" if dk == "down_q6n" else "q6" if meta["down"][0] == 4 else "q4"
  S_gu, S_dn = meta["gate"][4], meta[dk][4]              # the routing inside the gate | up launch (each task its top 8; task 0: sel_dn, wt)
  call(K, f"qgemv|gu|{FF}|{H}|{S_gu}|{d_up}|{S_dn}", Q.qgemvk_src("q4", FF, H, heads=2 * c.TOPK, route=(c.NE, S_gu, d_up, S_dn)), QB.y_gu, QB.pkg_f,
       st["gate"], rlog, d42, sel_dn, x, wt)
  call(K, f"qgemv|down|{kd}|{H}|{FF}", Q.qgemvk_src(kd, H, FF, heads=c.TOPK, act="swiglu", gu_np=FF), QB.y_dn, QB.pkg_f, st[dk], sel_dn,
       QN.desc(K, f"q_desc|{kd}|{FF}", lambda: Q.qgemvk_desc(kd, FF)), QB.y_dn, QB.y_gu, x)
  nt = 12 if _PHASES is not None else MC_NT       # (fused: every phase runs on the launch's 12 tasks)
  return call(K, f"moe_comb2|{H}|{QB.npd}|{nt}", Q.moe_comb2_src(H, QB.npd, c.TOPK, nt=nt), out, x1, QB.y_dn, wt,
              QN.desc(K, f"moe_comb2_desc|{H}|{QB.npd}", lambda: Q.moe_comb2_desc(H, QB.npd, c.TOPK)), nt=nt)


class QLayerW:
  """Layer l's streams for the gate (qwen3moe_npu.Layer): each file its own tensor, but gate and up in ONE (up at d_up bytes)."""
  def __init__(self, l, cache=QN.CACHE):
    self.l = l; self.meta, small = QN.read_meta(l, cache); self.vq6 = "v" in self.meta
    rd = lambda k: np.fromfile(os.path.join(cache, f"L{l}_{k}.bin"), np.uint8)
    self.st = {k: dev(rd(k)) for k in self.meta if k not in ("gate", "up")}
    g, u = rd("gate"), rd("up"); self.d_up = -(-g.nbytes // 4096) * 4096
    gu = np.zeros(self.d_up + u.nbytes, np.uint8); gu[:g.nbytes] = g; gu[self.d_up:] = u; self.st["gate"] = dev(gu)
    self.sm = {k: dev(v.astype(np.float32)) for k, v in small.items()}; self.sm["router16"] = dev(small["router"].astype(np.float16))

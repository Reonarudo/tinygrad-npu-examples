"""The batched one-token DeltaNet step: gdn_tok3 (the decode step's folded kernel: a | b projection, conv, SiLU, l2 norms, the delta rule
over the state blocks through LSRAM, the gated norm) over M INDEPENDENT rows -- M requests' next tokens -- in one launch. Row r is the
one-row kernel on tile row r of the C tiles (compact layout: tile r / 4, row r % 4), input row r of `xin` (pitch H), state / ring / small
stacks at index idx[r] (the per-user replicas: user x layers + layer), position posb[r]; its output is row r of the fp32 rows `out`
(pitch NV DV; gdn_tok3_src(rows_out)'s). Units = (row, head group) over the 12 tasks; per unit the one-row kernel's phases in its
order, so every row's results are gdn_tok3's bits (the simulator gate, kept with the hardware notes).
args: out, Sall, Call, idx [M], posb [M], qkv tiles, z tiles, xin [M rows], wab, adt, cwt, nw, desc (gdn_tok3_desc)."""
import os, re
from qwen38_kernels import gdn_tok3_src, gdn_hpt, ring_pitch, NT, V
# the a|b phase's chunk (columns of the task's 2 HPT a|b rows a DMA request): 128 (gdn_tok3's, 2 KiB a request, 16 requests one
# in flight -- latency-bound: ~50 us of a 150 us layer at 1 row on the 2B) -> 512 (8 KiB, 4 requests). The lane sums run in column
# order across the chunks either way: the same bits (the simulator gate against gdn_tok3 at 128). QWEN_GDN_KB overrides. 512 needs
# H x 4 + 4 HPT x 512 x 4 bytes of LSRAM: the 2B's (H 2048, 2 heads a task) fit, the 4B's (2560, 3) and the 9B's (4096, 3) do not --
# gdn_kb takes the largest of 512 / 256 / 128 that fits (the batched servers of the 4B / 9B failed at start-up since 2026-11-07)
GDN_KB = int(os.environ["QWEN_GDN_KB"]) if os.environ.get("QWEN_GDN_KB") else None
def gdn_kb(H, NV):
  if GDN_KB: return GDN_KB
  return next(kb for kb in (512, 256, 128) if H % kb == 0 and H * 4 + 2 * (2 * gdn_hpt(NV) * kb * 4) <= 32768 - 64)


def gdn_tokb_desc(DK, DV, H, HPT, C, CONV, KB=None, W=16):
  KB = (GDN_KB or 512) if KB is None else KB
  """gdn_tok3_desc's three slots, then 3: one 128-channel segment of the conv ring or of the taps (CONV rows of DK floats, DDR pitch
  ring_pitch(C) floats, LSRAM contiguous); 4-6: a row tile's strip pieces of one group (1-3 x 256 B at pitch 768 B: [4 column quads][4 rows][4]); 7: a 512-B row
  segment (the ring write) -- the conv's operands, the row's q / k / v / z segments and the ring writes by DMA instead of plain accesses."""
  return V._desc_slots((DK * W * 4, W * 4, DV * 4, W * 4), (H * 4,), (2 * HPT * KB * 4, KB * 4, H * 4, KB * 4), (CONV * DK * 4, DK * 4, ring_pitch(C) * 4, DK * 4),
                       (256, 256, 768, 256), (512, 256, 768, 256), (768, 256, 768, 256), (DK * 4,))   # 4-6: 1-3 strips' row-tile pieces (256 B: 4 column quads x 4 rows) of one group; 7: a 512-B segment


def gdn_tokb_src(NV, NK, DK, DV, C, CONV, H, nrb, eps, M, W=16, KB=None, stage=True, diag=()):
  """`stage` (default): the conv's ring rows and taps of a head's 128-channel segment come into LSRAM by two DMAs (gdn_tokb_desc slot 3)
  before conv_silu runs on them -- gdn_tok3 reads them by plain loads from DDR, ~250 a head, the batched kernel's dominant cost."""
  KB = gdn_kb(H, NV) if KB is None else KB
  src = gdn_tok3_src(NV, NK, DK, DV, C, CONV, H, nrb, eps, W, KB, rows_out=True); HPT = gdn_hpt(NV); NTU = -(-NV // HPT); LH = NV - (NTU - 1) * HPT
  def rep(a, b, n=1):
    nonlocal src; assert src.count(a) == n, (a[:60], src.count(a)); src = src.replace(a, b)
  rep("__kernel void gdn_tok3(", "__kernel void gdn_tokb(")
  # the tile reads at row r: LD8 takes the row offset `ro` (tile r / 4, row r % 4) -- conv_silu gets it as a parameter
  rep("#define LD8(ct, c) F8(*(__global float4*)((ct) + CTA(c)), *(__global float4*)((ct) + CTA(c) + 16))",
      "#define LD8(ct, c) F8(*(__global float4*)((ct) + CTA(c) + ro), *(__global float4*)((ct) + CTA(c) + ro + 16))")
  rep("void conv_silu(__global float* restrict ct, __global float* restrict Cr, __global float* restrict cwt, int pos, int off, __global float* restrict out) {",
      "void conv_silu(__global float* restrict ct, __global float* restrict Cr, __global float* restrict cwt, int pos, int off, __global float* restrict out, int ro) {")
  src = re.sub(r"conv_silu\(x, Cr, cwt, pos, ([^;]*?), (qs|ks|vs)\);", r"conv_silu(x, Cr, cwt, pos, \1, \2, ro);", src); assert src.count(", ro);") == 3
  # the task's early return -> the unit loop: units (row r, head group g) over the tasks
  early = f"  if (core_id >= {NTU}) return;                                       /* NTU tasks own the heads */\n" if NTU < NT else ""
  head = f"{early}  int L = idx[0], pos = posb[0];"
  rep(head, f"  for (int un = core_id; un < {M * NTU}; un += {NT}) {{ int r = un / {NTU}, g = un % {NTU}; int ro = (r / 4) * 64 + (r % 4) * 4;\n  int L = idx[r], pos = posb[r];")
  rep(f"int h0 = core_id * {HPT}, nh = core_id == {NTU - 1} ? {LH} : {HPT};", f"int h0 = g * {HPT}, nh = g == {NTU - 1} ? {LH} : {HPT};")
  # the stacks' per-layer offsets were compound assignments to the parameters (right for one unit a task): from the bases, per unit
  from qwen38_kernels import ring_pitch
  CP = ring_pitch(C)
  rep(f"cwt += L * {CONV * CP}; nw += L * {DV}; adt += L * {2 * NV};", f"__global float* cwt = cwt0 + L * {CONV * CP}; __global float* nw = nw0 + L * {DV}; __global float* adt = adt0 + L * {2 * NV};")
  rep("__global float* restrict cwt, __global float* restrict nw,", "__global float* restrict cwt0, __global float* restrict nw0,")
  rep("__global float* restrict adt, __global float* restrict cwt0", "__global float* restrict adt0, __global float* restrict cwt0")
  rep(f"+ core_id * {2 * HPT * H};", f"+ g * {2 * HPT * H};")
  rep("DMA_FILL(2, DESC(desc, 1), 0, (int)xin);", f"DMA_FILL(2, DESC(desc, 1), 0, (int)(xin + r * {H}));")
  rep(f"int u0 = core_id * {HPT * DV // W}, nu = nh * {DV // W};", f"int u0 = g * {HPT * DV // W}, nu = nh * {DV // W};")
  rep(f"*(__global float8*)(out + h * {DV} + c) =", f"*(__global float8*)(out + r * {NV * DV} + h * {DV} + c) =")
  rep("  DMA_WAIT_ALL();\n  #undef UBASE\n}", "  DMA_WAIT_ALL();\n  }\n  #undef UBASE\n}")
  if stage:
    CP = ring_pitch(C); QS = 2 * DK * W * 4; RSO = QS + 4 * DK * 4; TSO = RSO + CONV * DK * 4; assert TSO + CONV * DK * 4 <= 32768 - 64
    # conv_silu over the staged segment: Rs[(tok % CONV) * DK + i] (the ring's slot rows), Ts[t * DK + i] (the taps' rows)
    rep("void conv_silu(__global float* restrict ct, __global float* restrict Cr, __global float* restrict cwt, int pos, int off, __global float* restrict out, int ro) {",
        "void conv_silu(__global float* restrict ct, __global float* restrict Rs, __global float* restrict Ts, int pos, int off, __global float* restrict out, int ro) {")
    rep(f"float8 acc = LD8(ct, off + i) * *(__global float8*)(cwt + {(CONV - 1) * CP} + off + i);", f"float8 acc = LD8(ct, off + i) * *(__global float8*)(Ts + {(CONV - 1) * DK} + i);")
    rep(f"if (tok >= 0) acc += *(__global float8*)(Cr + (tok % {CONV}) * {CP} + off + i) * *(__global float8*)(cwt + t * {CP} + off + i);",
        f"if (tok >= 0) acc += *(__global float8*)(Rs + (tok % {CONV}) * {DK} + i) * *(__global float8*)(Ts + t * {DK} + i);")
    stg = (f"#define STAGE(seg) {{ DMA_WAIT(2); DMA_WAIT(3); DMA_FILL(2, DESC(desc, 3), {RSO}, (int)(Cr + (seg))); DMA_FILL(3, DESC(desc, 3), {TSO}, (int)(cwt + (seg))); DMA_WAIT(2); DMA_WAIT(3); }}\n"
           f"  __global float* RS = LSF({RSO}); __global float* TS = LSF({TSO});\n")
    rep("  __global float* qs = LSF(", stg + "  __global float* qs = LSF(")
    n0 = src.count("conv_silu(x, Cr, cwt, pos, ")
    # the row's q / k / v segments (and z's) from the C tiles by DMA, a group's strips a request; conv_silu reads the staged segment
    XQO, XKO, XVO, XZO = TSO + CONV * DK * 4, TSO + CONV * DK * 4 + 512, TSO + CONV * DK * 4 + 1024, TSO + CONV * DK * 4 + 1536; assert XZO + 512 <= 32768 - 64
    rep("float8 acc = LD8(ct, off + i) *", "float8 acc = *(__global float8*)(Xs + i) *")
    rep("void conv_silu(__global float* restrict ct, __global float* restrict Rs,", "void conv_silu(__global float* restrict Xs, __global float* restrict Rs,")
    STGO = XZO + 512; assert STGO + 2048 <= 32768 - 64      # the staging of a segment's 8 strips (256 B each: 4 rows), compacted into the row's 128 floats
    tiles = (f"#define TILES(base, seg, dst) {{ int c0_ = (seg), c1_ = (seg) + {DK}, o_ = 0, n_ = 0; for (int gg_ = c0_ / 48; gg_ * 48 < c1_; gg_++) {{ "
             f"int a0_ = gg_ * 48 > c0_ ? gg_ * 48 : c0_, a1_ = (gg_ + 1) * 48 < c1_ ? (gg_ + 1) * 48 : c1_; int k_ = (a1_ - a0_) / 16, st_ = (a0_ % 48) / 16; "
             f"if (n_ & 1) {{ DMA_FILL(3, DESC(desc, 3 + k_), {STGO} + o_ * 256, (int)((base) + (gg_ * {nrb * 3} + st_) * 192 + (r / 4) * 64)); DMA_WAIT(2); DMA_WAIT(3); }} "
             f"else DMA_FILL(2, DESC(desc, 3 + k_), {STGO} + o_ * 256, (int)((base) + (gg_ * {nrb * 3} + st_) * 192 + (r / 4) * 64)); n_++; o_ += k_; }} if (n_ & 1) DMA_WAIT(2); "
             f"for (int i_ = 0; i_ < {DK}; i_ += 8) *(__global float8*)((dst) + i_) = F8(*(__global float4*)(SG + (i_ / 16) * 64 + ((i_ % 16) / 4) * 16 + (r % 4) * 4), *(__global float4*)(SG + (i_ / 16) * 64 + ((i_ % 16) / 4) * 16 + 16 + (r % 4) * 4)); }}\n"
             f"  __global float* XQ = LSF({XQO}); __global float* XK = LSF({XKO}); __global float* XV = LSF({XVO}); __global float* XZ = LSF({XZO}); __global float* SG = LSF({STGO});\n")
    rep("  __global float* RS = LSF(", tiles + "  __global float* RS = LSF(")
    dst = {"qs": ("XQ", XQO), "ks": ("XK", XKO), "vs": ("XV", XVO)}
    def call(m):
      seg, o = m.group(1), m.group(2); X, XO = dst[o]
      return f"STAGE({seg}); TILES(x, {seg}, {X}); conv_silu({X}, RS, TS, pos, {seg}, {o}, ro);"
    src = re.sub(r"conv_silu\(x, Cr, cwt, pos, ([^;]*?), (qs|ks|vs), ro\);", call, src); assert src.count("STAGE(") == n0 + 1, src.count("STAGE(")
    # the ring writes: the staged segments drained (512 B each) instead of 16 plain stores
    QO, KO, VO = 0, NK * DK, 2 * NK * DK; REP = NV // NK
    rep(f"      for (int i = 0; i < {DV}; i += 8) *(__global float8*)(Cr + slot * {CP} + {VO} + h * {DV} + i) = LD8(x, {VO} + h * {DV} + i);\n"
        f"      if (h % {REP} == 0) for (int i = 0; i < {DK}; i += 8) {{\n"
        f"        *(__global float8*)(Cr + slot * {CP} + {QO} + j * {DK} + i) = LD8(x, {QO} + j * {DK} + i);\n"
        f"        *(__global float8*)(Cr + slot * {CP} + {KO} + j * {DK} + i) = LD8(x, {KO} + j * {DK} + i);\n"
        f"      }}",
        f"      DMA_DRAIN(3, DESC(desc, 7), {XVO}, (int)(Cr + slot * {CP} + {VO} + h * {DV}));\n"
        f"      if (h % {REP} == 0) {{ DMA_DRAIN(3, DESC(desc, 7), {XQO}, (int)(Cr + slot * {CP} + {QO} + j * {DK})); DMA_DRAIN(3, DESC(desc, 7), {XKO}, (int)(Cr + slot * {CP} + {KO} + j * {DK})); }}\n"
        f"      DMA_WAIT(3); TILES(z, h * {DV}, XZ);")
    rep(f"sw(LD8(z, h * {DV} + c), BC(1.0f))", "sw(*(__global float8*)(XZ + c), BC(1.0f))")
  assert "core_id *" not in src.split("for (int un")[1].split("DMA_WAIT_ALL")[0], "a task-indexed access left in the unit body"
  if "phases" in diag:    # cycle stamps a task into `dg` (int32 [NT][16], an extra argument before core_id): 0 total, 1 the a|b phase,
                          # 2 the prep (conv, tiles, ring rows) at a head's start, 3 the state fills' waits, 4 the drains' waits (before a refill),
                          # 5 the blocks' compute, 6 the output (gated norm), 7 units, 8 blocks
    def rep1(a, b):
      nonlocal src; assert src.count(a) == 1, (a[:60], src.count(a)); src = src.replace(a, b)
    rep1("__global int* restrict desc, const int core_id) {", "__global int* restrict desc, __global int* restrict dg, const int core_id) {")
    src = src.replace("#define CTA(c)", '#define CYC() ({ int c_; __asm__ volatile("mfctrl0 %0, 209" : "=r"(c_)); c_; })\n#define CTA(c)', 1)
    rep1("  for (int un = core_id;", "  int dg_[16]; for (int q_ = 0; q_ < 16; q_++) dg_[q_] = 0; int t00_ = CYC(), tp_ = 0;\n  for (int un = core_id;")
    rep1("  DMA_FILL(2, DESC(desc, 1), 0, (int)(xin + r *", "  dg_[7] += 1; tp_ = CYC(); DMA_FILL(2, DESC(desc, 1), 0, (int)(xin + r *")
    rep1("  float inv = 1.0f / __builtin_sqrtf(hsum8(sq)", "  dg_[1] += CYC() - tp_; float inv = 1.0f / __builtin_sqrtf(hsum8(sq)")
    rep1("    if (jb == 0) {", "    if (jb == 0) { tp_ = CYC();")
    rep1("    if (e + 1 < nu) {", "    if (jb == 0) dg_[2] += CYC() - tp_;\n    tp_ = CYC(); if (e + 1 < nu) {")
    rep1("    if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);\n    __global float* B = LSF(p *", "    dg_[4] += CYC() - tp_; tp_ = CYC(); if (p == 0) DMA_WAIT(0); else DMA_WAIT(1); dg_[3] += CYC() - tp_; tp_ = CYC(); dg_[8] += 1;\n    __global float* B = LSF(p *")
    rep1("    if (p == 0) DMA_DRAIN(2, DESC(desc, 0), 0, (int)UBASE(u)); else DMA_DRAIN(3, DESC(desc, 0), {BLK}, (int)UBASE(u));".replace("{BLK}", str(DK * W * 4)),
         "    dg_[5] += CYC() - tp_; if (p == 0) DMA_DRAIN(2, DESC(desc, 0), 0, (int)UBASE(u)); else DMA_DRAIN(3, DESC(desc, 0), %d, (int)UBASE(u)); tp_ = CYC();" % (DK * W * 4))
    rep1("  DMA_WAIT_ALL();\n  }\n  #undef UBASE", "  DMA_WAIT_ALL();\n  }\n  dg_[0] = CYC() - t00_; for (int q_ = 0; q_ < 16; q_++) dg[core_id * 16 + q_] = dg_[q_];\n  #undef UBASE")
  if "noab" in diag:      # timing only (wrong results): no a | b phase (its DMAs and dots), inv = 1
    i0 = src.index("  DMA_FILL(2, DESC(desc, 1), 0, (int)(xin + r *"); i1 = src.index("  float inv = 1.0f / __builtin_sqrtf(hsum8(sq)")
    decl = next(l for l in src[i0:i1].split("\n") if l.startswith("  float8 sq = ")); src = src[:i0] + decl + "\n  float inv = 1.0f;\n" + src[src.index("\n", i1) + 1:]
  if "nostate" in diag:   # timing only: no delta-rule phase (the conv staging, the state blocks, the output)
    i0 = src.index("  for (int e = 0; e < nu; e++) {"); i1 = src.index("  DMA_WAIT_ALL();\n  }\n  #undef"); src = src[:i0] + src[i1:]
  return src


# the decode kernel's phases at 1 row on the 2B (gdn_phases_bench, 2026-11-08, cycles a task of 165 k): the a|b projection 16 k, the
# heads' preparation 40 k (~20 DMA round trips a head, each waited for alone), the state blocks' compute 60 k, their drains' waits
# 30 k (the next fill waits for the previous block's drain, issued just before), the fills' waits 0.3 k. gdn_tokb2 takes the two
# DMA-latency items: the preparation's requests issued together, the state blocks triple-buffered
TOKB2 = os.environ.get("QWEN_GDN_TOKB2", "1") == "1"
# the units over the tasks: 0 as gdn_tokb (unit u on task u: 8 units fill cores 0 and 1, core 2 idle); 1: three a core (tasks 0-2,
# 4-6, 8-10), so every core's DMA engine carries a share (attn_gqa's cost model: one engine a core, ~7.2 GB/s)
SPREAD = os.environ.get("QWEN_GDN_SPREAD", "0") == "1"


def gdn_tokb2_src(NV, NK, DK, DV, C, CONV, H, nrb, eps, M, W=16, KB=None, diag=()):
  """gdn_tokb with a unit's head preparation hoisted before its state loop (a head's conv operands, its q / k / v / z tiles: every
  request issued at once, two waits; then the convs, norms and the ring rows) and the state blocks triple-buffered (the fill two
  blocks ahead is issued at the END of an iteration, after the previous block's drain is waited for behind this block's compute).
  The same arithmetic in the same order per head (the preparation does not depend on the state loop; a head's ring rows and the
  others' conv operands are disjoint channels): the same bits (gate: kernel-lib/sim/gdn_batch_sim.py GEN=gdn_tokb2_src).
  LSRAM: the a|b phase as gdn_tokb; the preparation's conv operands at 0 (3 segments x (ring 2 KiB + taps 2 KiB)), the tiles'
  staging at 12288 (4 x 2 KiB), the packed segments at 20480; the heads' q, k, v, o, z (2.5 KiB a head) from 24576 across both
  phases; the state blocks at 0 / 8192 / 16384. `diag` "phases": as gdn_tokb's (2 = the preparation)."""
  KB = gdn_kb(H, NV) if KB is None else KB; PH = "phases" in diag
  src = gdn_tokb_src(NV, NK, DK, DV, C, CONV, H, nrb, eps, M, W, KB, stage=True)
  mark = "  /* ---- the delta rule, as gdn_tok2 */"; assert src.count(mark) == 1; head = src[:src.index(mark)]
  REP = NV // NK; QO, KO, VO = 0, NK * DK, 2 * NK * DK; CP = ring_pitch(C); HPT = gdn_hpt(NV); NB = DV // W; BLK = DK * W * 4
  RS = [s * 2 * CONV * DK * 4 for s in range(3)]; TS = [r + CONV * DK * 4 for r in RS]; SGB = 3 * 2 * CONV * DK * 4; SG = [SGB + q * 2048 for q in range(4)]
  XQO = SGB + 4 * 2048; XKO, XVO = XQO + 512, XQO + 1024; PHB = 24576; HS = 5 * 512
  assert XVO + 512 <= PHB and 3 * BLK <= PHB and PHB + HPT * HS <= 32768 - 64 and DK == DV == 128 and W == 16, (XVO, PHB)
  def rep1(a, b):
    nonlocal head; assert head.count(a) == 1, (a[:60], head.count(a)); head = head.replace(a, b)
  if PH:
    rep1("__global int* restrict desc, const int core_id) {", "__global int* restrict desc, __global int* restrict dg, const int core_id) {")
    head = head.replace("#define CTA(c)", '#define CYC() ({ int c_; __asm__ volatile("mfctrl0 %0, 209" : "=r"(c_)); c_; })\n#define CTA(c)', 1)
    rep1("  for (int un = core_id;", "  int dg_[16]; for (int q_ = 0; q_ < 16; q_++) dg_[q_] = 0; int t00_ = CYC(), tp_ = 0;\n  for (int un = core_id;")
    rep1("  DMA_FILL(2, DESC(desc, 1), 0, (int)(xin + r *", "  dg_[7] += 1; tp_ = CYC(); DMA_FILL(2, DESC(desc, 1), 0, (int)(xin + r *")
    rep1("  float inv = 1.0f / __builtin_sqrtf(hsum8(sq)", "  dg_[1] += CYC() - tp_; float inv = 1.0f / __builtin_sqrtf(hsum8(sq)")
  if SPREAD:   # units 3 a core: task t = 4 c + i (i < 3) takes units 3 c + i, + 9, ...; task 4 c + 3 none
    HPTv = gdn_hpt(NV); NTUv = -(-NV // HPTv)
    a_ = f"  for (int un = core_id; un < {M * NTUv}; un += {NT}) {{"; assert head.count(a_) == 1, a_
    head = head.replace(a_, f"  int un0_ = (core_id & 3) == 3 ? {M * NTUv} : (core_id >> 2) * 3 + (core_id & 3);\n  for (int un = un0_; un < {M * NTUv}; un += 9) {{")
  T0 = (lambda: " tp_ = CYC();") if PH else (lambda: "")
  T = (lambda k: f" dg_[{k}] += CYC() - tp_;") if PH else (lambda k: "")
  fin = ("  dg_[0] = CYC() - t00_; for (int q_ = 0; q_ < 16; q_++) dg[core_id * 16 + q_] = dg_[q_];\n" if PH else "")
  hv = lambda hh: f"__global float* qs = LSF({PHB} + ({hh}) * {HS}); __global float* ks = qs + {DK}; __global float* vs = ks + {DK}; __global float* oh = vs + {DV}; __global float* XZ = oh + {DV};"
  body = f"""  /* ---- the heads' preparation: every request of a head at once (the conv operands on flag 0, the tiles on flag 1), then its convs, norms, ring rows */
#define TFILL(base, seg, sg, fl) {{ int c0_ = (seg), c1_ = (seg) + {DK}, o_ = 0; for (int gg_ = c0_ / 48; gg_ * 48 < c1_; gg_++) {{ int a0_ = gg_ * 48 > c0_ ? gg_ * 48 : c0_, a1_ = (gg_ + 1) * 48 < c1_ ? (gg_ + 1) * 48 : c1_; int k_ = (a1_ - a0_) / 16, st_ = (a0_ % 48) / 16; DMA_FILL(fl, DESC(desc, 3 + k_), (sg) + o_ * 256, (int)((base) + (gg_ * {nrb * 3} + st_) * 192 + (r / 4) * 64)); o_ += k_; }} }}
#define TPACK(sg, dst) {{ __global float* SG_ = LSF(sg); for (int i_ = 0; i_ < {DK}; i_ += 8) *(__global float8*)((dst) + i_) = F8(*(__global float4*)(SG_ + (i_ / 16) * 64 + ((i_ % 16) / 4) * 16 + (r % 4) * 4), *(__global float4*)(SG_ + (i_ / 16) * 64 + ((i_ % 16) / 4) * 16 + 16 + (r % 4) * 4)); }}
  __global float* XQ = LSF({XQO}); __global float* XK = LSF({XKO}); __global float* XV = LSF({XVO});
 {T0()} for (int hh = 0; hh < nh; hh++) {{
    int h = h0 + hh, j = h / {REP}; int sq_ = {QO} + j * {DK}, sk_ = {KO} + j * {DK}, sv_ = {VO} + h * {DV}; {hv("hh")}
    DMA_FILL(0, DESC(desc, 3), {RS[0]}, (int)(Cr + sq_)); DMA_FILL(0, DESC(desc, 3), {TS[0]}, (int)(cwt + sq_));
    DMA_FILL(0, DESC(desc, 3), {RS[1]}, (int)(Cr + sk_)); DMA_FILL(0, DESC(desc, 3), {TS[1]}, (int)(cwt + sk_));
    DMA_FILL(0, DESC(desc, 3), {RS[2]}, (int)(Cr + sv_)); DMA_FILL(0, DESC(desc, 3), {TS[2]}, (int)(cwt + sv_));
    TFILL(x, sq_, {SG[0]}, 1); TFILL(x, sk_, {SG[1]}, 1); TFILL(x, sv_, {SG[2]}, 1); TFILL(z, h * {DV}, {SG[3]}, 1);
    DMA_WAIT(0); DMA_WAIT(1);
    TPACK({SG[0]}, XQ); TPACK({SG[1]}, XK); TPACK({SG[2]}, XV); TPACK({SG[3]}, XZ);
    conv_silu(XQ, LSF({RS[0]}), LSF({TS[0]}), pos, sq_, qs, ro); l2n(qs, {1.0 / DK ** 0.5!r}f);
    conv_silu(XK, LSF({RS[1]}), LSF({TS[1]}), pos, sk_, ks, ro); l2n(ks, 1.0f);
    conv_silu(XV, LSF({RS[2]}), LSF({TS[2]}), pos, sv_, vs, ro);
    DMA_DRAIN(2, DESC(desc, 7), {XVO}, (int)(Cr + slot * {CP} + sv_));
    if (h % {REP} == 0) {{ DMA_DRAIN(2, DESC(desc, 7), {XQO}, (int)(Cr + slot * {CP} + sq_)); DMA_DRAIN(2, DESC(desc, 7), {XKO}, (int)(Cr + slot * {CP} + sk_)); }}
    DMA_WAIT(2);
  }}{T(2)}
  /* ---- the delta rule over the heads' state blocks, triple-buffered */
  int u0 = g * {HPT * NB}, nu = nh * {NB};
  #define UBASE(u) (S + ((u) / {NB}) * {DK * DV} + ((u) % {NB}) * {W})
  DMA_FILL(0, DESC(desc, 0), 0, (int)UBASE(u0)); if (nu > 1) DMA_FILL(1, DESC(desc, 0), {BLK}, (int)UBASE(u0 + 1));
  for (int e = 0; e < nu; e++) {{
    int u = u0 + e, p = e & 1, q = e % 3, h = u / {NB}, jb = u % {NB}, j0 = jb * {W}, hh = h - h0; {hv("hh")} float d = dd[hh], b = bb[hh];
   {T0()} if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);{T(3)}{T0()}
    __global float* B = LSF(q * {BLK});
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
    *(__global float8*)(oh + j0) = o0; *(__global float8*)(oh + j0 + 8) = o1;{T(5)}
    if (p == 0) DMA_DRAIN(2, DESC(desc, 0), q * {BLK}, (int)UBASE(u)); else DMA_DRAIN(3, DESC(desc, 0), q * {BLK}, (int)UBASE(u));
    if (jb == {NB - 1}) {{                                          /* o = rmsnorm(o) * nw * silu(z) -> the output row */
      float8 ss = BC(0.0f); for (int c = 0; c < {DV}; c += 8) {{ float8 ov = *(__global float8*)(oh + c); ss += ov * ov; }}
      float8 inv8 = BC(1.0f / __builtin_sqrtf(hsum8(ss) * {1.0 / DV!r}f + {eps!r}f));
      for (int c = 0; c < {DV}; c += 8) *(__global float8*)(out + r * {NV * DV} + h * {DV} + c) = *(__global float8*)(oh + c) * inv8 * *(__global float8*)(nw + c) * sw(*(__global float8*)(XZ + c), BC(1.0f));
    }}
   {T0()} if (e + 2 < nu) {{ if (e >= 1) {{ if (p == 0) DMA_WAIT(3); else DMA_WAIT(2); }} DMA_FILL(p, DESC(desc, 0), ((e + 2) % 3) * {BLK}, (int)UBASE(u + 2)); }}{T(4)}
  }}
  DMA_WAIT_ALL();
  }}
{fin}  #undef UBASE
}}"""
  return head + body

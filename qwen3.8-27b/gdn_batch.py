"""The batched one-token DeltaNet step: gdn_tok3 (the decode step's folded kernel: a | b projection, conv, SiLU, l2 norms, the delta rule
over the state blocks through LSRAM, the gated norm) over M INDEPENDENT rows -- M requests' next tokens -- in one launch. Row r is the
one-row kernel on tile row r of the C tiles (compact layout: tile r / 4, row r % 4), input row r of `xin` (pitch H), state / ring / small
stacks at index idx[r] (the per-user replicas: user x layers + layer), position posb[r]; its output is row r of the fp32 rows `out`
(pitch NV DV; gdn_tok3_src(rows_out)'s). Units = (row, head group) over the 12 tasks; per unit the one-row kernel's phases in its
order, so every row's results are gdn_tok3's bits (the simulator gate, kept with the hardware notes).
args: out, Sall, Call, idx [M], posb [M], qkv tiles, z tiles, xin [M rows], wab, adt, cwt, nw, desc (gdn_tok3_desc)."""
import re
from qwen38_kernels import gdn_tok3_src, gdn_hpt, ring_pitch, NT, V


def gdn_tokb_desc(DK, DV, H, HPT, C, CONV, KB=128, W=16):
  """gdn_tok3_desc's three slots, then 3: one 128-channel segment of the conv ring or of the taps (CONV rows of DK floats, DDR pitch
  ring_pitch(C) floats, LSRAM contiguous); 4-6: a row tile's strip pieces of one group (1-3 x 256 B at pitch 768 B: [4 column quads][4 rows][4]); 7: a 512-B row
  segment (the ring write) -- the conv's operands, the row's q / k / v / z segments and the ring writes by DMA instead of plain accesses."""
  return V._desc_slots((DK * W * 4, W * 4, DV * 4, W * 4), (H * 4,), (2 * HPT * KB * 4, KB * 4, H * 4, KB * 4), (CONV * DK * 4, DK * 4, ring_pitch(C) * 4, DK * 4),
                       (256, 256, 768, 256), (512, 256, 768, 256), (768, 256, 768, 256), (DK * 4,))   # 4-6: 1-3 strips' row-tile pieces (256 B: 4 column quads x 4 rows) of one group; 7: a 512-B segment


def gdn_tokb_src(NV, NK, DK, DV, C, CONV, H, nrb, eps, M, W=16, KB=128, stage=True, diag=()):
  """`stage` (default): the conv's ring rows and taps of a head's 128-channel segment come into LSRAM by two DMAs (gdn_tokb_desc slot 3)
  before conv_silu runs on them -- gdn_tok3 reads them by plain loads from DDR, ~250 a head, the batched kernel's dominant cost."""
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
  if "noab" in diag:      # timing only (wrong results): no a | b phase (its DMAs and dots), inv = 1
    i0 = src.index("  DMA_FILL(2, DESC(desc, 1), 0, (int)(xin + r *"); i1 = src.index("  float inv = 1.0f / __builtin_sqrtf(hsum8(sq)")
    decl = next(l for l in src[i0:i1].split("\n") if l.startswith("  float8 sq = ")); src = src[:i0] + decl + "\n  float inv = 1.0f;\n" + src[src.index("\n", i1) + 1:]
  if "nostate" in diag:   # timing only: no delta-rule phase (the conv staging, the state blocks, the output)
    i0 = src.index("  for (int e = 0; e < nu; e++) {"); i1 = src.index("  DMA_WAIT_ALL();\n  }\n  #undef"); src = src[:i0] + src[i1:]
  return src

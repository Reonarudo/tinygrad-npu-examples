"""The batched one-token DeltaNet step: gdn_tok3 (the decode step's folded kernel: a | b projection, conv, SiLU, l2 norms, the delta rule
over the state blocks through LSRAM, the gated norm) over M INDEPENDENT rows -- M requests' next tokens -- in one launch. Row r is the
one-row kernel on tile row r of the C tiles (compact layout: tile r / 4, row r % 4), input row r of `xin` (pitch H), state / ring / small
stacks at index idx[r] (the per-user replicas: user x layers + layer), position posb[r]; its output is row r of the fp32 rows `out`
(pitch NV DV; gdn_tok3_src(rows_out)'s). Units = (row, head group) over the 12 tasks; per unit the one-row kernel's phases in its
order, so every row's results are gdn_tok3's bits (the simulator gate, kept with the hardware notes).
args: out, Sall, Call, idx [M], posb [M], qkv tiles, z tiles, xin [M rows], wab, adt, cwt, nw, desc (gdn_tok3_desc)."""
import re
from qwen38_kernels import gdn_tok3_src, gdn_hpt, NT


def gdn_tokb_src(NV, NK, DK, DV, C, CONV, H, nrb, eps, M, W=16, KB=128):
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
  assert "core_id *" not in src.split("for (int un")[1].split("DMA_WAIT_ALL")[0], "a task-indexed access left in the unit body"
  return src

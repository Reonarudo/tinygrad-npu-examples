"""resid_had: resid32q and the next RMSNorm's had_a32d (had_fast v2, stacked, "rec") as ONE launch, one row (decode).

The pair runs 48 times a decode step of Bonsai 2 2B (the o projection's residual then the post norm; the down projection's then the
next layer's input norm): resid32q (one task: the row's C pieces stream through LSRAM by chunks, out = x + C, the sum of squares,
the record) and the had (4 tasks, a 512-block each: the record, the row block, the norm weights' block, the butterflies, the A
lines). Fused, every task of the had (block b = core_id) runs the whole residual itself -- the same chunks, additions and sums
in the same order, so the same row and the same 1 / rms in all of them (task 0 alone drains the row and the record: no two tasks
store into one line) -- and keeps its own block's values resident in LSRAM (copied as the chunk is built), then runs had_fast's
passes on them with the 1 / rms it just computed (the record's own expression), and drains its A lines. No cross-task
dependency, so no barrier; the C pieces are read nb times (43 groups x 768 B x 4 = 130 KB a call, while the port is idle anyway).

Bit-identical to resid32q + had_a32d(fast, v2, rec) by construction: the residual code is resid32q's quad(1) body, the
had is had_fast2's passes (the same schedules from the same cache), the 1 / rms the same scalar expression. Gated byte-for-byte
on the vendor simulator.

LSRAM (c = 2048, hb = 512): resid's two buffers [0, 15360) (a row chunk 1536 B + 8 groups' pieces 6144 B each), the record line
at 15360, the had's 4-row block from 16128 (512 B of garbage gap before it, rows 2560 B apart), its W block at 26368, the A lines at
28416..32512.
args: out (the A halves), xo (the residual rows out, fp32), x, ct, rec (fp32 [REC_N]), w (the norm stack [L, c]), idx, desc
      (resid_had_desc), core_id.
"""
import numpy as np
import had_fast as F
import qwen38_kernels as K
hasm = F.hasm; V = K.V; NT = K.NT; RQ_G = K.RQ_G

WS, AS = 14, 15                                        # the extra descriptor slots after resid32q_desc's 14: the W block, the A drain


def resid_had_desc(c, nrb, hb=512):
  """resid32q_desc(c, nrb, 1) + slot 14: the norm weights' block (hb floats); 15: a block's A lines (4 rows' halves)."""
  RB = hb * 4
  return np.concatenate([K.resid32q_desc(c, nrb, 1), V._desc_slots((RB,), (RB * 2,))])


def resid_had_src(c, nrb, eps, post=1.0, hb=512):
  HB = hb; assert HB in (512, 1024) and c % HB == 0 and c % 16 == 0; nb = c // HB; assert nb <= NT, "a block a task"
  real = 1; RB = HB * 4
  # resid32q at real = 1 (iqr 1, nl 1; the x buffer one row)
  W = 48 * RQ_G; ngr = -(-c // 48); nch = -(-ngr // RQ_G); WL = c - (nch - 1) * W; assert WL % 16 == 0
  XB = real * W * 4; CB = RQ_G * 768; BUF = XB + CB; REC = 2 * BUF
  sx, sr = RQ_G + 2, RQ_G + 5                         # resid32q's slots at nl rows: the rows (full / last chunk), the record line
  # the had's block: had_fast2's layout from REC + 256 (the 512-B gap before row 0 takes pass A's / B's garbage stores)
  L0 = REC + 256 + 512; LP = RB + 512; LPF = LP // 4; W0 = L0 + 4 * LP; OUT0 = W0 + RB
  assert OUT0 + 2 * RB <= F.INV0, (c, hb, OUT0)
  PST = post != 1.0; NC = 8 if HB == 1024 else 4
  sA = F.sch("A-rms-np", lambda: hasm.ops_passA("rms"), nopro=True); sB = F.sch("B-np", lambda: hasm.ops_radix8("B"), nopro=True)
  sC = F.sch("C" if NC == 8 else "C4", lambda: hasm.ops_radix8("C" if NC == 8 else "C4")); sV = F.sch("conv-post" if PST else "conv", lambda: hasm.ops_conv(post=PST))
  passA = hasm.c_block(sA, HB // 32, {"rl": "Lr", "rs": "Lr", "rw": "W"}, consts=[("ivv", "BC(iv)")])
  passB = hasm.c_block(sB, HB // 128, {"bl": "Lr + 8 * e", "bs": "Lr + 8 * e"})
  passC = hasm.c_block(sC, 16, {f"c{j}": f"Lr + {128 * j}" for j in range(NC)}, derive={f"c{j}": (f"c{j - 1}", 512) for j in range(1, NC)})
  conv = hasm.c_block(sV, HB // 16, {"c0": "Cv0", "c1": f"Cv0 + {LPF}", "c2": f"Cv0 + {2 * LPF}", "c3": f"Cv0 + {3 * LPF}", "o": "Ov"},
                      regs=[("lp", str(LP))], derive={f"c{r}": (f"c{r - 1}", "lp") for r in range(1, 4)}, consts=[("pvv", f"BC({post!r}f)")] if PST else [])
  # resid32q's chunk requests (quad(1)'s fill): the chunk's three sub-tile residues of every group, then the row chunk, all on flag p
  fill = lambda k, p: (f"{{ int ga_ = ({k}) * {RQ_G}; int gn_ = {ngr} - ga_ < {RQ_G} ? {ngr} - ga_ : {RQ_G};"
                       f" NOUNROLL for (int t_ = 0; t_ < 3; t_++) DMA_FILL({p}, DESC(desc, gn_ - 1), {p * BUF + XB} + t_ * 256, (int)(ct + (ga_ * {nrb * 3} + t_) * 192));"
                       f" DMA_FILL({p}, DESC(desc, ({k}) == {nch - 1} ? {sx + 1} : {sx}), {p * BUF}, (int)(x + ({k}) * {W})); }}")
  src = K.had_a32_src("rms", c, nrb, eps, real=real, compact=True, stacked=True, post=post, hb=HB)
  pre = src[:src.index("static inline __attribute__((always_inline)) float8 h8(")] + F.H16 + F.NOUNROLL + '#define NOHWL(i) __asm__ volatile("" : "+r"(i))\n'
  if "define F8(" not in pre: pre += K.TILE
  return pre + f"""__kernel void resid_had(__global half* restrict out, __global float* restrict xo, __global float* restrict x, __global float* restrict ct, __global float* restrict rec, __global float* restrict w, __global int* restrict idx, __global int* restrict desc, const int core_id) {{
  w += idx[0] * {c};
  if (core_id < {nb}) {{                                  /* one exit for every task (an early return beside DMA hung the simulator's job) */
    const int b = core_id; const int D0 = b == 0;
    __global float* L = LSF({L0}); __global float* W = LSF({W0}); __global half* O = LSH({OUT0});
    DMA_FILL(0, DESC(desc, {WS}), {W0}, (int)(w + b * {HB}));
    float8 acc0 = BC(0.0f);
    {fill(0, 0)}
    NOUNROLL for (int r = 1; r < 4; r++) NOUNROLL for (int k = 0; k < {HB}; k += 8) *(__global float8*)(L + r * {LPF} + k) = BC(0.0f);
    NOUNROLL for (int k = 0; k < {nch}; k++) {{
      int p = k & 1;
      if (k + 1 < {nch}) {{ if (p == 0) {{ if (k >= 1 && D0) DMA_WAIT(3); {fill("k + 1", 1)} }} else {{ if (D0) DMA_WAIT(2); {fill("k + 1", 0)} }} }}
      if (p == 0) DMA_WAIT(0); else DMA_WAIT(1);
      __global float* Xb = LSF(p * {BUF}); __global float* Cb = Xb + {XB // 4}; int nj = k == {nch - 1} ? {WL // 16} : {W // 16}; int kc = k * {W} - b * {HB};
      NOUNROLL for (int j = 0; j < nj; j++) {{
        __global float* q = Cb + j * 64; __global float* xk = Xb + 16 * j;
        float8 v00 = *(__global float8*)(xk) + F8(*(__global float4*)(q), *(__global float4*)(q + 16));
        *(__global float8*)(xk) = v00; acc0 += v00 * v00;
        float8 v01 = *(__global float8*)(xk + 8) + F8(*(__global float4*)(q + 32), *(__global float4*)(q + 48));
        *(__global float8*)(xk + 8) = v01; acc0 += v01 * v01;
        int cb = kc + 16 * j; if ((unsigned)cb < {HB}u) {{ *(__global float8*)(L + cb) = v00; *(__global float8*)(L + cb + 8) = v01; }}
      }}
      if (D0) {{ if (p == 0) DMA_DRAIN(2, DESC(desc, k == {nch - 1} ? {sx + 1} : {sx}), 0, (int)(xo + k * {W})); else DMA_DRAIN(3, DESC(desc, k == {nch - 1} ? {sx + 1} : {sx}), {BUF}, (int)(xo + k * {W})); }}
    }}
    float iv = 1.0f / __builtin_sqrtf(hsum8(acc0) * {1.0 / c!r}f + {eps!r}f);
    if (D0) {{ __global float* Rl = LSF({REC}); Rl[0] = iv; DMA_DRAIN(0, DESC(desc, {sr}), {REC}, (int)rec); }}
    {{
      __global float* Lr = L;
      {passA}
      NOUNROLL for (int e = 0; e < 2; e++) {{ {passB} NOHWL(e); }}
      {passC}
    }}
    {{ __global float* Cv0 = L; __global half* Ov = O; {conv} }}
    DMA_DRAIN(2, DESC(desc, {AS}), {OUT0}, (int)(out + b * {HB // 128 * 512}));
  }}
  DMA_WAIT_ALL();
}}"""

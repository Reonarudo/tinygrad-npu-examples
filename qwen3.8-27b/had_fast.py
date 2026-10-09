"""had_fast: had_a32d restructured for the TEC's instruction fetch and latencies, bit-identical by construction (every element sees
the same fp32 operations in the same order: the scale (x * inv) * w, stages 1, 2, 4, ..., 512 each a + b / a - b, the same fp16
conversion; rms's sums per row and lane in K order with the same hsum8 / rsqrt expression; swiglu's sw() as the compiler's own
instruction sequence). Gated byte-identical against had_a32d on the vendor simulator and on the board.

What changes against had_a32d (opts):
  * phase 1 (rms): one compact loop over the columns, the rows' fma chains interleaved (instead of a 260-bundle unrolled body
    re-fetched every chunk at 3.2 cycles a bundle); 8 KiB requests.  "asm": the loop as hand-scheduled bundles (had_asm.ops_p1),
    the rows' LSRAM pitch + 64 B so their loads pair.
  * the transform: pass A = scale + stages 1, 2, 4, 8 in registers (h16: four even / odd extract rounds on a 16-value pair,
    which return the natural order: no re-interleave); pass B = stages 16, 32, 64 (radix 8, vectors 2 apart); pass C = stages
    128, 256, 512 (radix 8, vectors 16 apart); the conversion four pieces a trip.  "asm": all four as modulo-scheduled bundles
    (had_asm), A and B without a prologue: L's rows sit 4608 B apart from byte 512, the 512-B gap before each row taking the
    garbage stores of the first trips.
  * swiglu "sx": the gate | up C sub-tiles by exact bytes (13-strip chunks, three requests a tensor: one residue of the strip
    index mod 3 each, 256-B elements a group apart in DDR and 768 B apart in LSRAM, so the strips land in natural order) instead
    of 9-slot requests (2.25x the bytes).  "swasm": also sw() as modulo-scheduled bundles (had_asm.ops_sw; implies sx).
  * diagnostics: "phases" (an extra argument dg, int32 [12][16]: per-phase cycle sums), "twice" (the body twice; the phases are
    the warm second run's), "spread" (unit u on task (u % 3) * 4 + u / 3); ablations "nosw", "nodma", "dmab" (timing only).
args as had_a32d: out, x, [w (stacked: [L, c]), [idx]], desc (had_fast_desc(mode, c, nrb, real, opts)), [dg], core_id.
"""
import os, sys, pathlib
try: import had_asm as hasm                                                   # the examples tree (qwen3.8-27b/had_asm.py)
except ImportError:                                                           # a standalone copy: hasm.py beside this file
  EX = pathlib.Path(os.environ.get("EX", "~/Developer/example-tinygrad-npu-uses")).expanduser()
  if str(EX / "qwen3.8-27b") not in sys.path: sys.path.insert(0, str(EX / "qwen3.8-27b"))
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
  import hasm                                                                 # noqa: E402
import qwen38_kernels as K                                                    # noqa: E402
_SCH = {}
def sch(name, mk, nopro=False):
  if name not in _SCH: _SCH[name] = hasm.best_schedule(mk, nopro=nopro)
  return _SCH[name]

HAD_B, NT = K.HAD_B, K.NT
V = K.V
CYCDEF = '#define CYC() ({ int c_; __asm__ volatile("mfctrl0 %0, 209" : "=r"(c_)); c_; })\n'
NOUNROLL = '#define NOUNROLL _Pragma("clang loop unroll(disable)")\n'
H16 = r"""
#define EXTE(a, b) __builtin_aipu_exte_tfp32_tfp32((a), (b))
#define EXTO(a, b) __builtin_aipu_exto_tfp32_tfp32((a), (b))
#define EXTL(a, b) __builtin_aipu_extl_tfp32_tfp32((a), (b))
#define EXTH(a, b) __builtin_aipu_exth_tfp32_tfp32((a), (b))
/* stages 1, 2, 4, 8 of the 16 values (v | w): each round pairs the current lowest index bit (even / odd lanes), E + O, E - O,
   and rotates that bit to the top; four rounds over four bits leave the natural order. */
#define H16R(v, w) { float8 e_ = EXTE(v, w), o_ = EXTO(v, w); v = e_ + o_; w = e_ - o_; }
#define H16(v, w) { H16R(v, w) H16R(v, w) H16R(v, w) H16R(v, w) }
"""


def p1_ch(c, real):
  """phase-1 columns a chunk: the largest of 1024 / 512 / 256 dividing c with real * ch * 4 <= 8 KiB."""
  for ch in (1024, 512, 256, 128):
    if c % ch == 0 and real * ch * 4 <= 8192: return ch
  raise ValueError((c, real))


def p1_pitch(c, real, opts):
  """phase 1's LSRAM row pitch (bytes): CH * 4, + 64 with "asm" (rows alternate banks, so the asm loop pairs their loads)."""
  return p1_ch(c, real) * 4 + (64 if "asm" in opts and real <= 8 else 0)


def had_fast_desc(mode, c, nrb, real, opts=()):
  """had_a32d_desc with slot 0 = phase 1's chunk (p1_ch columns of the real rows, pitch c; LSRAM pitch p1_pitch)."""
  if "v2" in opts: opts = tuple(opts) + ("asm", "swasm")
  iqr = min(3, -(-real // 4)); nl = real - 4 * (iqr - 1); RB = HAD_B * 4; ch = p1_ch(c, real); LP = RB + (512 if ("asm" in opts or "swasm" in opts) else 0)
  if "rec" in opts:   # the norm record's real lines (qwen38_kernels.resid32q), the last slot (rec_slot)
    base = had_fast_desc(mode, c, nrb, real, tuple(o for o in opts if o != "rec")); return __import__("numpy").concatenate([base, V._desc_slots((real * 64,))])
  return V._desc_slots((real * ch * 4, ch * 4, c * 4, p1_pitch(c, real, opts)), (4 * RB, RB, c * 4, LP), (nl * RB, RB, c * 4, LP), (RB,),
                       (8192,) if iqr == 1 else (8192, 32, 32 * iqr, 32),
                       *([((3 * nrb + 3) * 256 // (3 if "dmab" in opts else 1), 256, 768, 256)] if not ("sx" in opts or "swasm" in opts) else
                         [(256,)] + [(cnt * 256, 256, 3 * nrb * 768, 768) for cnt in range(1, 6)]))


def had_fast_src(mode, c, nrb, eps, real, stacked=False, post=1.0, opts=()):
  if "v2" in opts: return had_fast2_src(mode, c, nrb, eps, real, stacked=stacked, post=post, opts=opts)
  assert mode in ("rms", "plain", "swiglu") and c % HAD_B == 0 and 1 <= real <= 12; nb = c // HAD_B; iqr = min(3, -(-real // 4))
  RB = HAD_B * 4; SWB = (3 * nrb + 3) * 256
  # L: the quad's 4 rows. "asm": rows 4608 B apart from byte 512 (a 512-B gap before each row takes the prologue-free asm loops'
  # garbage stores); else contiguous from 0
  L0, LP = (512, RB + 512) if "asm" in opts else (0, RB); LPF = LP // 4
  W0 = L0 + 4 * LP; OUT0, IN0 = W0 + RB, W0
  CH = p1_ch(c, real); CHP = p1_pitch(c, real, opts); CHB = real * CHP; NCH = c // CH
  ASM = "asm" in opts or "swasm" in opts; HW = lambda v: f" NOHWL({v});" if ASM else ""      # an opaque counter: no hardware loop around an asm loop
  assert 2 * CHB <= 32768 - 64 and OUT0 + 8192 <= 32768 - 64 and IN0 + 4 * SWB <= 32768 - 64
  PH = "phases" in opts; UX = "((core_id & 3) * 3 + (core_id >> 2))" if "spread" in opts else "core_id"
  T0 = lambda v: f" int {v} = CYC();" if PH else ""
  T = lambda k, v: f" dg_[{k}] += CYC() - {v}; dg_[15] += 2;" if PH else ""
  scale = "" if post == 1.0 else f" * BC({post!r}f)"
  NOSW, NODMA = "nosw" in opts, "nodma" in opts           # ablations (timing only): no SiLU work / no chunk DMAs
  SX = "sx" in opts or "swasm" in opts; SWASM = "swasm" in opts; SXN = 13                             # swiglu "sx": exact-byte chunks of 13 strips (2 x 2 x 3328 B double-buffered)
  if mode == "swiglu":   # (as had_a32d) chunk ch: sub-tile columns j0..j0+3 of the gate (j = col / 16), the up's at j0 + c / 16
    if "sw16" in opts:      # 16 trips of two sw() each (a row pair of two column quads): a body under the 32-bundle loop buffer
      swloop = f"""NOUNROLL for (int t = 0; t < {0 if NOSW else 16}; t++) {{   /* (jj, cq, h) = (t / 4, 2 ((t / 2) % 2), t % 2): rows 2 h, 2 h + 1 */
        int jj = t >> 2, cq = ((t >> 1) & 1) * 2, h = t & 1; int j = j0 + jj, ju = ju0 + jj; int k = ch * 64 + jj * 16 + cq * 4;
        __global float* g = G + ((j / 3 - g0) * {3 * nrb} + j % 3) * 64 + cq * 16 + 8 * h; __global float* q = U + ((ju / 3 - u0) * {3 * nrb} + ju % 3) * 64 + cq * 16 + 8 * h;
        float8 a = sw(*(__global float8*)(g), *(__global float8*)(q)), bq = sw(*(__global float8*)(g + 16), *(__global float8*)(q + 16));
        __global float* Lh = L + h * {2 * LPF} + k;
        *(__global float8*)(Lh) = EXTL(a, bq); *(__global float8*)(Lh + {LPF}) = EXTH(a, bq);
      }}"""
    else:
      swloop = f"""NOUNROLL for (int t = 0; t < {0 if NOSW else 8}; t++) {{   /* (jj, cq) = (t / 2, 2 (t % 2)): column quads cq, cq + 1 of sub-tile jj */
        int jj = t >> 1, cq = (t & 1) * 2; int j = j0 + jj, ju = ju0 + jj; int k = ch * 64 + jj * 16 + cq * 4;
        __global float* g = G + ((j / 3 - g0) * {3 * nrb} + j % 3) * 64 + cq * 16; __global float* q = U + ((ju / 3 - u0) * {3 * nrb} + ju % 3) * 64 + cq * 16;
        float8 a01 = sw(*(__global float8*)(g), *(__global float8*)(q)), a23 = sw(*(__global float8*)(g + 8), *(__global float8*)(q + 8));
        float8 b01 = sw(*(__global float8*)(g + 16), *(__global float8*)(q + 16)), b23 = sw(*(__global float8*)(g + 24), *(__global float8*)(q + 24));
        *(__global float8*)(L + k) = EXTL(a01, b01); *(__global float8*)(L + {LPF} + k) = EXTH(a01, b01);
        *(__global float8*)(L + {2 * LPF} + k) = EXTL(a23, b23); *(__global float8*)(L + {3 * LPF} + k) = EXTH(a23, b23);
      }}"""
    def issue(ch, p):
      if NODMA: return ";"
      return (f"{{ int j0 = 64 * b + 4 * ({ch}); int ju = j0 + {c // 16};\n"
              f"        DMA_FILL({p}, DESC(desc, 5), {IN0} + {2 * SWB} * ({p}), (int)(x + (j0 / 3) * {3 * nrb * 192} + i * 64));\n"
              f"        DMA_FILL(({p}) + 2, DESC(desc, 5), {IN0} + {2 * SWB} * ({p}) + {SWB}, (int)(x + (ju / 3) * {3 * nrb * 192} + i * 64)); }}")
    load = f"""
   {T0("ts_")} {issue(0, 0)}
    NOUNROLL for (int ch = 0; ch < {HAD_B // 64}; ch++) {{
      int p = ch & 1;
      if (ch + 1 < {HAD_B // 64}) {issue("ch + 1", "p ^ 1")}
     {T0("tw_")} {"" if NODMA else "DMA_WAIT(p); DMA_WAIT(p + 2);"}{T(5, "tw_")}
      __global float* G = LSF({IN0} + p * {2 * SWB}); __global float* U = G + {SWB // 4};
      int j0 = 64 * b + 4 * ch, ju0 = j0 + {c // 16}; int g0 = j0 / 3, u0 = ju0 / 3;
      {swloop}
    }}{T(4, "ts_")}"""
    if SX:   # exact bytes: chunks of SXN strips; per tensor three requests (strips j0 + t, t = 0, 1, 2: one residue mod 3 each,
             # 256-B elements one group apart in DDR, 768 B apart in LSRAM), so the strips land in natural order, [strip][4 cq][16]
      NCK = -(-64 // SXN); SB = 2 * SXN * 256
      def sxissue(cc, bufo, fl="0"):   # gate requests on flag fl, up on fl + 2 (three on each: the flag stays busy while any is)
        if NODMA: return ";"
        lines = [f"{{ int j0 = 64 * b + {SXN} * ({cc}); int n = ({cc}) == {NCK - 1} ? {64 - SXN * (NCK - 1)} : {SXN};"]
        for t in range(3):
          lines.append(f"  {{ int js = j0 + {t}, ju = js + {c // 16}, cnt = (n - {t} + 2) / 3;"
                       f" DMA_FILL({fl}, DESC(desc, 5) + 64 * cnt, {IN0} + ({bufo}) + {256 * t}, (int)(x + ((js / 3) * {3 * nrb} + js % 3) * 192 + i * 64));"
                       f" DMA_FILL({fl} + 2, DESC(desc, 5) + 64 * cnt, {IN0} + ({bufo}) + {SXN * 256 + 256 * t}, (int)(x + ((ju / 3) * {3 * nrb} + ju % 3) * 192 + i * 64)); }}")
        return "\n        ".join(lines) + " }"
      swl = swloop.replace("((j / 3 - g0) * {3 * nrb} + j % 3) * 64".replace("{3 * nrb}", str(3 * nrb)), "jj * 64").replace("((ju / 3 - u0) * {3 * nrb} + ju % 3) * 64".replace("{3 * nrb}", str(3 * nrb)), "jj * 64")
      swl = swl.replace("int k = ch * 64 + jj * 16", f"int k = cc * {16 * SXN} + jj * 16")
      swl = swl.replace("int j = j0 + jj, ju = ju0 + jj; ", "")
      swl = swl.replace("for (int t = 0; t < 8; t++)", "for (int t = 0; t < 2 * n; t++)").replace("for (int t = 0; t < 16; t++)", "for (int t = 0; t < 4 * n; t++)")
      if SWASM:   # sw() in asm (hasm.ops_sw): two a trip, the row pairs h = 0, 1 in turn, 2 SXN trips (strip, column-quad pair)
        consts = [("cm", "BC(-1.4426950408889634f)"), ("cmn", "BC(-126.0f)"), ("cmx", "BC(126.0f)"), ("c0", "BC(9.618129e-3f)"),
                  ("c1", "BC(5.550411e-2f)"), ("c2", "BC(2.402265e-1f)"), ("c3", "BC(6.931472e-1f)"), ("c4", "BC(1.0f)")]
        blk = hasm.c_block(sch("sw", hasm.ops_sw), 2 * SXN, {"g": "G + 8 * h", "u": "U + 8 * h", "l0": f"L + 2 * h * {LPF} + cc * {16 * SXN}",
                           "l1": f"L + (2 * h + 1) * {LPF} + cc * {16 * SXN}"}, consts=consts)
        swasm = f"""NOUNROLL for (int h = 0; h < {0 if NOSW else 2}; h++) {{ {blk} NOHWL(h); }}"""
      load = f"""
   {T0("ts_")} {sxissue(0, 0, "0")}
    NOUNROLL for (int cc = 0; cc < {NCK}; cc++) {{
      int p = cc & 1; int n = cc == {NCK - 1} ? {64 - SXN * (NCK - 1)} : {SXN};
     {T0("tw_")} {"" if NODMA else "DMA_WAIT(p); DMA_WAIT(p + 2);"}{T(5, "tw_")}
      if (cc + 1 < {NCK}) {sxissue("cc + 1", f"(p ^ 1) * {SB}", "(p ^ 1)")}
      __global float* G = LSF({IN0} + p * {SB}); __global float* U = G + {SXN * 64};
      {swl if not SWASM else swasm}{HW("cc")}
    }}{T(4, "ts_")}"""
    phase1 = ""; ivs = None
  else:
    if ASM and real <= 8:
      blk = hasm.c_block(sch(f"p1-{real}", lambda: hasm.ops_p1(real)), CH // 8, {f"b{r}": f"bb + {r * CHP // 4}" for r in range(real)},
                         accs=[f"acc{r}" for r in range(real)])
      p1_inner = blk; accd = " ".join(f"float8 acc{r} = BC(0.0f);" for r in range(real)); accu = "".join(f" acc[{r}] = acc{r};" for r in range(real))
    else:
      p1_inner = f"""NOUNROLL for (int k = 0; k < {CH}; k += 8) {{
      {" ".join(f"float8 v{r} = *(__global float8*)(bb + {r * CHP // 4} + k);" for r in range(real))}
      {" ".join(f"acc[{r}] += v{r} * v{r};" for r in range(real))}
    }}"""; accd = ""; accu = ""
    phase1 = f"""
 {T0("tp1_")} float8 acc[{real}]; for (int r = 0; r < {real}; r++) acc[r] = BC(0.0f); {accd}
  DMA_FILL(0, DESC(desc, 0), 0, (int)x);
  NOUNROLL for (int ch = 0; ch < {NCH}; ch++) {{
    int p = ch & 1;
    if (ch + 1 < {NCH}) DMA_FILL(p ^ 1, DESC(desc, 0), (p ^ 1) * {CHB}, (int)(x + (ch + 1) * {CH}));
   {T0("tw_")} DMA_WAIT(p);{T(2, "tw_")}
    __global float* bb = LSF(p * {CHB});
    {p1_inner}{HW("ch")}
  }}
{accu}
  float inv[{real}];
  for (int r = 0; r < {real}; r++) inv[r] = 1.0f / __builtin_sqrtf(hsum8(acc[r]) * {1.0 / c!r}f + {eps!r}f);{T(1, "tp1_")}""" if mode == "rms" else ""
    load = f"""
   {T0("tl_")} if (i == 0) DMA_FILL(0, DESC(desc, 3), {W0}, (int)(w + b * {HAD_B}));
    DMA_FILL(2, DESC(desc, nr == 4 ? 1 : 2), {L0}, (int)(x + 4 * i * {c} + b * {HAD_B}));
    if (i == 0) DMA_WAIT(0);
    DMA_WAIT(2);{T(3, "tl_")}"""
    ivs = "inv[4 * i + r]" if mode == "rms" else None
  # pass A: (scale) + stages 1, 2, 4, 8; two 16-value pairs a body
  if mode == "swiglu":
    lda = lambda o: f"float8 v{o} = *(__global float8*)(Lr + k + {16 * o}), w{o} = *(__global float8*)(Lr + k + {16 * o + 8});"
  elif mode == "rms":
    lda = lambda o: (f"float8 v{o} = *(__global float8*)(Lr + k + {16 * o}) * BC(iv) * *(__global float8*)(W + k + {16 * o}), "
                     f"w{o} = *(__global float8*)(Lr + k + {16 * o + 8}) * BC(iv) * *(__global float8*)(W + k + {16 * o + 8});")
  else:   # plain: had_a32d's x * 1.0f * w (the 1.0f is exact)
    lda = lambda o: (f"float8 v{o} = *(__global float8*)(Lr + k + {16 * o}) * *(__global float8*)(W + k + {16 * o}), "
                     f"w{o} = *(__global float8*)(Lr + k + {16 * o + 8}) * *(__global float8*)(W + k + {16 * o + 8});")
  if ASM:
    am = mode if mode != "plain" else "plain"
    sA = sch(f"A-{am}-np", lambda: hasm.ops_passA(am), nopro=True); sB = sch("B-np", lambda: hasm.ops_radix8("B"), nopro=True); sC = sch("C", lambda: hasm.ops_radix8("C"))
    bA = {"rl": "Lr", "rs": "Lr"} | ({} if mode == "swiglu" else {"rw": "W"})
    passA_asm = hasm.c_block(sA, HAD_B // 32, bA, consts=[("ivv", "BC(iv)")] if mode == "rms" else [])
    passB_asm = hasm.c_block(sB, 8, {"bl": "Lr + 8 * e", "bs": "Lr + 8 * e"})
    passC_asm = hasm.c_block(sC, 16, {f"c{j}": f"Lr + {128 * j}" for j in range(8)})
    conv_asm = hasm.c_block(sch("conv", hasm.ops_conv), HAD_B // 16, {"c0": "L", "c1": f"L + {LPF}", "c2": f"L + {2 * LPF}", "c3": f"L + {3 * LPF}", "o": "O"}) if post == 1.0 else None
  passA = f"""
      NOUNROLL for (int k = 0; k < {HAD_B}; k += 32) {{
        {lda(0)} {lda(1)}
        H16(v0, w0) H16(v1, w1)
        *(__global float8*)(Lr + k) = v0; *(__global float8*)(Lr + k + 8) = w0; *(__global float8*)(Lr + k + 16) = v1; *(__global float8*)(Lr + k + 24) = w1;
      }}"""
  def radix8(base, step):   # stages on vectors step, 2 step, 4 step apart (a[j] = vector base + j * step)
    ld = " ".join(f"float8 a{j} = *(__global float8*)(Lr + {base} + {8 * step * j});" for j in range(8))
    st = " ".join(f"*(__global float8*)(Lr + {base} + {8 * step * j}) = a{j};" for j in range(8))
    stg = []
    for d in (1, 2, 4):
      for j in range(8):
        if j & d: continue
        stg.append(f"{{ float8 s_ = a{j} + a{j + d}, d_ = a{j} - a{j + d}; a{j} = s_; a{j + d} = d_; }}")
    return f"{ld}\n        {' '.join(stg)}\n        {st}"
  if ASM: passA = "\n      " + passA_asm
  passB = f"""
      NOUNROLL for (int g = 0; g < 16; g++) {{             /* stages 16, 32, 64: vectors 16 (g / 2) + g % 2 + 2 j */
        int o = (g >> 1) * 128 + (g & 1) * 8;
        {radix8("o", 2)}
      }}"""
  if ASM: passB = f"""
      NOUNROLL for (int e = 0; e < 2; e++) {{ {passB_asm}{HW("e")} }}"""
  passC = f"""
      NOUNROLL for (int g = 0; g < 16; g++) {{             /* stages 128, 256, 512: vectors g + 16 j */
        int o = g * 8;
        {radix8("o", 16)}
      }}"""
  if ASM: passC = "\n      " + passC_asm
  conv = f"""
   {T0("tc_")} NOUNROLL for (int pc = 0; pc < {HAD_B // 4}; pc += 4) {{       /* pieces pc .. pc + 3: columns 4 pc .. 4 pc + 15 */
      int k = 4 * pc;
      float8 r0 = *(__global float8*)(L + k), r1 = *(__global float8*)(L + {LPF} + k), r2 = *(__global float8*)(L + {2 * LPF} + k), r3 = *(__global float8*)(L + {3 * LPF} + k);
      float8 q0 = *(__global float8*)(L + k + 8), q1 = *(__global float8*)(L + {LPF} + k + 8), q2 = *(__global float8*)(L + {2 * LPF} + k + 8), q3 = *(__global float8*)(L + {3 * LPF} + k + 8);
      *(__global half16*)(O + pc * 16) = CVT16((EXTL(r0, r1){scale}), (EXTL(r2, r3){scale}));
      *(__global half16*)(O + pc * 16 + 16) = CVT16((EXTH(r0, r1){scale}), (EXTH(r2, r3){scale}));
      *(__global half16*)(O + pc * 16 + 32) = CVT16((EXTL(q0, q1){scale}), (EXTL(q2, q3){scale}));
      *(__global half16*)(O + pc * 16 + 48) = CVT16((EXTH(q0, q1){scale}), (EXTH(q2, q3){scale}));
    }}{T(9, "tc_")}"""
  if ASM and conv_asm: conv = f"""
   {T0("tc_")} {conv_asm}{T(9, "tc_")}"""
  zero = f"""
    NOUNROLL for (int r = nr; r < 4; r++) NOUNROLL for (int k = 0; k < {HAD_B}; k += 8) *(__global float8*)(L + r * {LPF} + k) = BC(0.0f);"""
  rows = f"""
   {T0("tA_")} NOUNROLL for (int r = 0; r < nr; r++) {{
      __global float* Lr = L + r * {LPF};{f" float iv = {ivs};" if ivs else ""}{passA}{HW("r")}
    }}{T(6, "tA_")}{T0("tB_")}
    NOUNROLL for (int r = 0; r < nr; r++) {{
      __global float* Lr = L + r * {LPF};{passB}{HW("r")}
    }}{T(7, "tB_")}{T0("tC_")}
    NOUNROLL for (int r = 0; r < nr; r++) {{
      __global float* Lr = L + r * {LPF};{passC}{HW("r")}
    }}{T(8, "tC_")}{zero}"""
  src = K.had_a32_src(mode, c, nrb, eps, real=real, compact=True, stacked=stacked, post=post)
  pre = src[:src.index("static inline __attribute__((always_inline)) float8 h8(")] + H16 + NOUNROLL + (CYCDEF if PH else "") + \
    '#define NOHWL(i) __asm__ volatile("" : "+r"(i))\n'

  dg_arg = ", __global int* restrict dg" if PH else ""
  TW = "twice" in opts                         # the body twice; the phases are the second (warm instruction fetch) run's
  rep0 = ("  for (int rep_ = 0; rep_ < 2; rep_++) {\n" + ("   if (rep_ == 1) { int s0_ = dg_[0] + CYC() - t00_; int e_ = dg_[12], u_ = dg_[13]; for (int q_ = 0; q_ < 16; q_++) dg_[q_] = 0; dg_[14] = s0_; dg_[12] = e_; dg_[13] = u_; t00_ = CYC(); }\n" if PH else "")) if TW else ""
  rep1 = "    if (pend) DMA_WAIT(3);\n  }\n" if TW else ""
  phd = "  int dg_[16]; for (int q_ = 0; q_ < 16; q_++) dg_[q_] = 0; int t00_ = CYC(); dg_[12] = t00_ & 0x7fffffff;\n" if PH else ""
  fin = (f"""{T0("tf_")} DMA_WAIT_ALL();{T(11, "tf_")}
  dg_[0] += CYC() - t00_;
  for (int q_ = 0; q_ < 16; q_++) dg[core_id * 16 + q_] = dg_[q_];""" if PH else "  DMA_WAIT_ALL();")
  return pre + f"""__kernel void had_a32d(__global half* restrict out, __global float* restrict x{"" if mode == "swiglu" else ", __global float* restrict w"}{", __global int* restrict idx" if stacked else ""}, __global int* restrict desc{dg_arg}, const int core_id) {{
{"  w += idx[0] * %d;" % c if stacked else ""}
{phd}  if ({UX} < {nb}) {{                                  /* one exit for every task: an early `return` beside DMA hung the simulator's job */
{rep0}  {phase1}
  __global float* L = LSF({L0}); __global half* O = LSH({OUT0}); int pend = 0;{"" if mode == "swiglu" else f" __global float* W = LSF({W0});"}
  for (int b = {UX}; b < {nb}; b += {NT}) {{{" dg_[13] += 1;" if PH else ""}
    for (int i = 0; i < {iqr}; i++) {{                 /* the unit's row quads in turn: one TEC writes all of a block's A lines */
    int nr = {real} - 4 * i; nr = nr > 4 ? 4 : nr;
   {T0("td_")} if (pend) {{ DMA_WAIT(3); pend = 0; }}{T(10, "td_")}
    {load}{rows}{conv}
    DMA_DRAIN(3, DESC(desc, 4), {OUT0}, (int)(out + b * {HAD_B // 128 * 512 * iqr} + i * 16)); pend = 1;{HW("i")}
    }}{HW("b")}
  }}
{rep1}  }}
{fin}
}}"""


# ---------------------------------------------------------------------------------------------------------------------------------
# v2 (opts "v2"): the same arithmetic in the same order as the "asm" + "swasm" kernel above (so as had_a32d),
# with less EXECUTED text -- a cold instruction fetch costs ~75-90 cycles per once-executed bundle, per task, per launch, and it
# does not overlap a DMA (both measured on the board):
#   * the glue specialised at generation time: one unit per task when nb <= 12 (no unit loop), one row quad when real <= 4 (no quad
#     loop, no pending drain, the row count a constant);
#   * one row loop running pass A, B (both halves) and C of a row in turn (three row loops before); constant trip counts;
#   * base registers derived inside the asm (pass C's c_j = c_0 + 512 j, the conversion's rows c_r = c_0 + 4608 r) instead of a
#     C constant each (the compiler spends a mov / movh pair, one op a bundle, on every LSRAM constant);
#   * rms: hsum8 / rsqrt of the quad's rows in vector lanes (an 8 x 8 lane transpose by exte / exto, the lanes added in hsum8's
#     order, the fma, one vector rsqrt; byte-identical to the scalar tail on the simulator and the board), into LSRAM;
#   * swiglu: ONE copy of the chunk-request code (step s issues chunk NCK-1-s and runs the SiLU of chunk NCK-s; the three requests
#     of a tensor as a loop), and the SiLU loop prologue-free: the chunks run in DESCENDING column order, so the first S-1 trips'
#     garbage stores (96 B before the chunk, in rows 2h and 2h+1) land in the chunk processed next (rewritten then) or, for
#     chunk 0, in the row's 512-B gap.
# Ablations (timing only, both slower -- P2.md): "warm" (rms / plain: a dry iteration r = -1 runs every pass and the conversion
# once, one trip each, on scratch while the row loads are in flight; the trip counts in registers, `loop rN`) and "p1w" (rms:
# phase 1 and the rows as one step loop, the dry trips inside phase 1's DMA waits). Diagnostics as v1: "phases", "twice".
INV0 = 32768 - 64 - 64                      # rms: the rows' 1 / rms (up to 12 floats) in LSRAM


def sw_sched():
  """swiglu's SiLU loop scheduled for the divide unit: its two rcp at cycles 0 and 21 -- the unit takes
  21 cycles a vector rcp and is not pipelined, so the second rcp's bundle stalls 20 cycles (no bundles there) -- with the measured
  latencies (rcp 21, fp32 mul / fma 4, add / sub / max / min / rint / cvt / scal2 2) and no fma beside an ext. 43 cycles a trip
  in 23 bundles: the candidate (II 43, seed 11) that ran fastest of 22 on the board, 46 cycles a trip against v1's 64."""
  if "sw-du" not in _SCH:
    old = hasm.NOFMAEXT; hasm.NOFMAEXT = True
    try: _SCH["sw-du"] = hasm.best_schedule(lambda: hasm.ops_sw(True, divunit=True), II0=43, IImax=43, tries=4000, nopro=True,
                                            blocked=lambda II: range(1, hasm.DIVU), seed=11)
    finally: hasm.NOFMAEXT = old
  return _SCH["sw-du"]


def _hsum_rsqrt_vec(real, c, eps):
  """C: acc0..acc{real-1} (float8) -> INVP[0..real-1] = 1 / sqrt(hsum8(acc_r) / c + eps), real <= 8, the rows in lanes: T_j holds
  lane j of every row (exte / exto rounds: (0, 1) (2, 3) ..., then (01, 23) ..., then (0123, 4567)), S = ((T0 + T1) + T2) + ... + T7
  (hsum8's order), rsqrt(S * (1 / c) + eps) (the scalar tail's fma)."""
  a = [f"acc{r}" if r < real else "Z_" for r in range(8)]
  n = 8 if real > 4 else 4
  s = ["float8 Z_ = BC(0.0f);"]
  for p in range(n // 2):
    s.append(f"float8 e1_{p} = EXTE({a[2 * p]}, {a[2 * p + 1]}), o1_{p} = EXTO({a[2 * p]}, {a[2 * p + 1]});")
  if n == 8:
    s.append("float8 ee0 = EXTE(e1_0, e1_1), eo0 = EXTO(e1_0, e1_1), ee1 = EXTE(e1_2, e1_3), eo1 = EXTO(e1_2, e1_3);")
    s.append("float8 oe0 = EXTE(o1_0, o1_1), oo0 = EXTO(o1_0, o1_1), oe1 = EXTE(o1_2, o1_3), oo1 = EXTO(o1_2, o1_3);")
    s.append("float8 T0 = EXTE(ee0, ee1), T4 = EXTO(ee0, ee1), T2 = EXTE(eo0, eo1), T6 = EXTO(eo0, eo1), "
             "T1 = EXTE(oe0, oe1), T5 = EXTO(oe0, oe1), T3 = EXTE(oo0, oo1), T7 = EXTO(oo0, oo1);")
  else:
    s.append("float8 ee0 = EXTE(e1_0, e1_1), eo0 = EXTO(e1_0, e1_1), oe0 = EXTE(o1_0, o1_1), oo0 = EXTO(o1_0, o1_1);")
    s.append("float8 T0 = EXTE(ee0, ee0), T4 = EXTO(ee0, ee0), T2 = EXTE(eo0, eo0), T6 = EXTO(eo0, eo0), "
             "T1 = EXTE(oe0, oe0), T5 = EXTO(oe0, oe0), T3 = EXTE(oo0, oo0), T7 = EXTO(oo0, oo0);")
  s.append("float8 S_ = T0 + T1; S_ = S_ + T2; S_ = S_ + T3; S_ = S_ + T4; S_ = S_ + T5; S_ = S_ + T6; S_ = S_ + T7;")
  s.append(f"*(__global float8*)INVP = __builtin_aipu_rsqrt_tfp32_pw(BC(0.0f), S_ * BC({1.0 / c!r}f) + BC({eps!r}f), (bool8)1);")
  return "\n  ".join(s)


def had_fast2_src(mode, c, nrb, eps, real, stacked=False, post=1.0, opts=()):
  assert mode in ("rms", "plain", "swiglu") and c % HAD_B == 0 and 1 <= real <= 12 and nrb >= 1
  assert post == 1.0, "v2: post == 1.0 only (the production setting)"
  nb = c // HAD_B; iqr = min(3, -(-real // 4)); RB = HAD_B * 4
  L0, LP = 512, RB + 512; LPF = LP // 4                   # the quad's rows 4608 B apart from byte 512 (A's / B's garbage gaps)
  W0 = L0 + 4 * LP; OUT0, IN0 = W0 + RB, W0; SCR = OUT0 + 512
  CH = p1_ch(c, real); CHP = p1_pitch(c, real, ("asm",)); CHB = real * CHP; NCH = c // CH
  assert 2 * CHB <= OUT0 and OUT0 + 8192 <= INV0 and SCR + 4096 + 128 <= OUT0 + 8192
  PH = "phases" in opts; WARM = "warm" in opts and mode != "swiglu"; P1W = "p1w" in opts and mode == "rms"
  T0 = lambda v: f" int {v} = CYC();" if PH else ""
  T = lambda k, v: f" dg_[{k}] += CYC() - {v}; dg_[15] += 2;" if PH else ""
  HW = lambda v: f" NOHWL({v});"
  ONEU = nb <= NT                                  # one unit per task: no unit loop
  ONEQ = iqr == 1                                  # one row quad: no quad loop, no pending drain, nr a constant
  MULTI = not (ONEU and ONEQ)
  SXN = 13; NCK = -(-64 // SXN); SB = 2 * SXN * 256
  zero = ("NOUNROLL for (int r = nr; r < 4; r++) NOUNROLL for (int k = 0; k < %d; k += 8) *(__global float8*)(L + r * %d + k) = BC(0.0f);" % (HAD_B, LPF)) \
    if (not ONEQ or real % 4) else ""
  if mode == "swiglu":
    assert IN0 + 2 * SB <= INV0
    consts = [("cm", "BC(-1.4426950408889634f)"), ("cmn", "BC(-126.0f)"), ("cmx", "BC(126.0f)"), ("c0", "BC(9.618129e-3f)"),
              ("c1", "BC(5.550411e-2f)"), ("c2", "BC(2.402265e-1f)"), ("c3", "BC(6.931472e-1f)"), ("c4", "BC(1.0f)")]
    blk = hasm.c_block(sw_sched() if "swv1" not in opts else sch("sw-np", hasm.ops_sw, nopro=True), "tsw", {"g": "G + 8 * h", "u": "U + 8 * h", "l0": f"L0p + 2 * h * {LPF}",
                       "l1": f"L0p + (2 * h + 1) * {LPF}"}, consts=consts)
    load = f"""
   {T0("ts_")} NOUNROLL for (int s = 0; s <= {NCK}; s++) {{
      if (s < {NCK}) {{ int ci = {NCK - 1} - s, q = s & 1; int j0 = 64 * b + {SXN} * ci; int n = ci == {NCK - 1} ? {64 - SXN * (NCK - 1)} : {SXN};
        NOUNROLL for (int t = 0; t < 3; t++) {{ int js = j0 + t, ju = js + {c // 16}, cnt = (n - t + 2) / 3;
          DMA_FILL(q, DESC(desc, 5) + 64 * cnt, {IN0} + q * {SB} + 256 * t, (int)(x + ((js / 3) * {3 * nrb} + js % 3) * 192 + i * 64));
          DMA_FILL(q + 2, DESC(desc, 5) + 64 * cnt, {IN0} + q * {SB} + {SXN * 256} + 256 * t, (int)(x + ((ju / 3) * {3 * nrb} + ju % 3) * 192 + i * 64)); NOHWL(t); }} }}
      if (s > 0) {{ int cp = {NCK} - s, p = (s - 1) & 1; int n = cp == {NCK - 1} ? {64 - SXN * (NCK - 1)} : {SXN};
       {T0("tw_")} DMA_WAIT(p); DMA_WAIT(p + 2);{T(5, "tw_")}
        __global float* G = LSF({IN0} + p * {SB}); __global float* U = G + {SXN * 64}; __global float* L0p = L + cp * {16 * SXN}; int tsw = 2 * n;
        NOUNROLL for (int h = 0; h < 2; h++) {{ {blk} NOHWL(h); }} }}{HW("s")}
    }}{T(4, "ts_")}"""
    waits = ""; phase1 = ""; ivs = None
  else:
    phase1 = ""
    if mode == "rms":
      if real <= 8:
        blk1 = hasm.c_block(sch(f"p1-{real}", lambda: hasm.ops_p1(real)), CH // 8, {f"b{r}": f"bb + {r * CHP // 4}" for r in range(real)},
                            accs=[f"acc{r}" for r in range(real)])
        tail = _hsum_rsqrt_vec(real, c, eps); accd = " ".join(f"float8 acc{r} = BC(0.0f);" for r in range(real))
      else:
        blk1 = f"""NOUNROLL for (int k = 0; k < {CH}; k += 8) {{
      {" ".join(f"float8 v{r} = *(__global float8*)(bb + {r * CHP // 4} + k);" for r in range(real))}
      {" ".join(f"acc[{r}] += v{r} * v{r};" for r in range(real))}
    }}"""
        accd = f"float8 acc[{real}]; for (int r = 0; r < {real}; r++) acc[r] = BC(0.0f);"
        tail = f"for (int r = 0; r < {real}; r++) INVP[r] = 1.0f / __builtin_sqrtf(hsum8(acc[r]) * {1.0 / c!r}f + {eps!r}f);"
      phase1 = f"""
 {T0("tp1_")} {accd}
  DMA_FILL(0, DESC(desc, 0), 0, (int)x);
  NOUNROLL for (int ch = 0; ch < {NCH}; ch++) {{
    int p = ch & 1;
    if (ch + 1 < {NCH}) DMA_FILL(p ^ 1, DESC(desc, 0), (p ^ 1) * {CHB}, (int)(x + (ch + 1) * {CH}));
   {T0("tw_")} DMA_WAIT(p);{T(2, "tw_")}
    __global float* bb = LSF(p * {CHB});
    {blk1} NOHWL(ch);
  }}
  {tail}{T(1, "tp1_")}"""
      if "rec" in opts:   # QWEN_SMALL2's norm records: the rows' 1 / rms from the residual's record (resid32q: the same
                          # sums, order and expression); idx[0] == 0 (the stack's first layer, its rows written outside the kernels) keeps phase 1
        assert stacked and not P1W
        rs = len(had_fast_desc(mode, c, nrb, real, tuple(o for o in opts if o != "rec"))) // 16
        phase1 = f"""
  if (idx[0] != 0) {{ DMA_FILL(0, DESC(desc, {rs}), 0, (int)rec); DMA_WAIT(0); __global float* R_ = LSF(0);
    NOUNROLL for (int r = 0; r < {real}; r++) INVP[r] = R_[16 * r]; }}
  else {{ {phase1} }}"""
    nrx = "nr == 4 ? 1 : 2" if not ONEQ else ("1" if real == 4 else "2")
    load = f"""
   {T0("tl_")} {"if (i == 0) " if not ONEQ else ""}DMA_FILL(0, DESC(desc, 3), {W0}, (int)(w + b * {HAD_B}));
    DMA_FILL(2, DESC(desc, {nrx}), {L0}, (int)(x + 4 * i * {c} + b * {HAD_B}));{T(3, "tl_")}"""
    waits = f"{T0('tlw_')} {'if (i == 0) ' if not ONEQ else ''}DMA_WAIT(0); DMA_WAIT(2);{T(3, 'tlw_')}"
    ivs = "INVP[4 * i + r]" if mode == "rms" else None
  # the passes: A (prologue-free), B (prologue-free, both halves), C (c_j derived), the conversion (rows derived)
  sA = sch(f"A-{mode}-np", lambda: hasm.ops_passA(mode), nopro=True); sB = sch("B-np", lambda: hasm.ops_radix8("B"), nopro=True)
  sC = sch("C", lambda: hasm.ops_radix8("C")); sV = sch("conv", hasm.ops_conv)
  bA = {"rl": "Lr", "rs": "Lr"} | ({} if mode == "swiglu" else {"rw": "W"})
  dC = {f"c{j}": (f"c{j - 1}", 512) for j in range(1, 8)}; dV = {f"c{r}": (f"c{r - 1}", "lp") for r in range(1, 4)}
  DRY = WARM or P1W
  tr = (lambda v, n: v) if DRY else (lambda v, n: n)      # the dry ablations: trip counts in registers (`loop rN`)
  passA = hasm.c_block(sA, tr("tA", HAD_B // 32), bA, consts=[("ivv", "BC(iv)")] if mode == "rms" else [])
  passB = hasm.c_block(sB, tr("tB", 8), {"bl": "Lr + 8 * e", "bs": "Lr + 8 * e"})
  passC = hasm.c_block(sC, tr("tC", 16), {f"c{j}": f"Lr + {128 * j}" for j in range(8)}, derive=dC)
  conv = hasm.c_block(sV, tr("tV", HAD_B // 16), {"c0": "Cv0", "c1": f"Cv0 + {LPF}", "c2": f"Cv0 + {2 * LPF}", "c3": f"Cv0 + {3 * LPF}", "o": "Ov"},
                      regs=[("lp", "lpv" if DRY else str(LP))], derive=dV)
  if not DRY:
    rows = f"""
    {zero}
    {waits}
    NOUNROLL for (int r = 0; r < nr; r++) {{
      __global float* Lr = L + r * {LPF};{f" float iv = {ivs};" if ivs else ""}
     {T0("tA_")} {passA}{T(6, "tA_")}{T0("tB_")}
      NOUNROLL for (int e = 0; e < 2; e++) {{ {passB}{HW("e")} }}{T(7, "tB_")}{T0("tC_")}
      {passC}{T(8, "tC_")}{HW("r")}
    }}
   {T0("tV_")} {{ __global float* Cv0 = L; __global half* Ov = O; {conv} }}{T(9, "tV_")}"""
  else:   # "warm": r = -1 is the dry iteration (first quad of the first unit); counters: real A / B / C / conv 6 / 7 / 8 / 9, dry 1 / 2 / 4 / 5
    r0 = ("(i == 0 && b == core_id) ? -1 : 0" if MULTI else "-1")
    PT = lambda a, b_, v: (f" if (dry) {{{T(a, v)}}} else {{{T(b_, v)}}}" if PH else "")
    rows = f"""
    {zero}
    NOUNROLL for (int r = {r0}; r <= nr; r++) {{
      int dry = r < 0;
      if (r == 0) {{ {waits} }}
      if (r < nr) {{
        __global float* Lr = dry ? LSF({SCR}) : L + r * {LPF};{f" float iv = INVP[4 * i + (dry ? 0 : r)];" if ivs else ""} int tA = dry ? 1 : {HAD_B // 32}, tB = dry ? 1 : 8, tC = dry ? 1 : 16;
       {T0("tA_")} {passA}{PT(1, 6, "tA_")}{T0("tB_")}
        NOUNROLL for (int e = 0; e < 2; e++) {{ {passB}{HW("e")} }}{PT(2, 7, "tB_")}{T0("tC_")}
        {passC}{PT(4, 8, "tC_")}
      }}
      if (dry | (r == nr)) {{
        __global float* Cv0 = dry ? LSF({SCR}) : L; __global half* Ov = dry ? LSH({SCR + 4096}) : O; int lpv = dry ? 0 : {LP}, tV = dry ? 1 : {HAD_B // 16};
       {T0("tV_")} {conv}{PT(5, 9, "tV_")}
      }}{HW("r")}
    }}"""
  drain = f"DMA_DRAIN(3, DESC(desc, 4), {OUT0}, (int)(out + b * {HAD_B // 128 * 512 * iqr} + i * 16));" + (" pend = 1;" if MULTI else "")
  if ONEQ:
    quad = f"""  {{ const int i = 0; const int nr = {real};{"" if not MULTI else f"{T0('td_')} if (pend) {{ DMA_WAIT(3); pend = 0; }}{T(10, 'td_')}"}
    {load}{rows}
    {drain}
    }}"""
  else:
    quad = f"""  for (int i = 0; i < {iqr}; i++) {{
    int nr = {real} - 4 * i; nr = nr > 4 ? 4 : nr;{T0("td_")} if (pend) {{ DMA_WAIT(3); pend = 0; }}{T(10, "td_")}
    {load}{rows}
    {drain}{HW("i")}
    }}"""
  if P1W:
    # phase 1 and the rows in ONE step loop: steps s < NCH are phase-1 chunks (issue s + 1; a dry piece; wait s; sum s), then per row
    # quad i five steps k = 0..3 (row k: A, B, C) and 4 (the conversion and the drain). Dry pieces s = 0 A, 1 B, 2 C, 3 the
    # conversion, on W's window (loaded after phase 1; A's / B's garbage in the 512-B gap before it).
    assert real <= 8 and ONEU and NCH >= 4 and 2 * CHB <= W0 - 512
    SCRW = W0; NS = NCH + 5 * iqr; phase1 = ""
    zero1 = zero.replace("int k = 0; k <", "int k2 = 0; k2 <").replace("k += 8", "k2 += 8").replace("+ k)", "+ k2)")
    unit = f"""  {{ const int b = core_id;{' dg_[13] += 1;' if PH else ''}
 {T0("tp1_")} {" ".join(f"float8 acc{r} = BC(0.0f);" for r in range(real))}
  DMA_FILL(0, DESC(desc, 0), 0, (int)x);{" int pend = 0;" if iqr > 1 else ""}
  NOUNROLL for (int s = 0; s < {NS}; s++) {{
    int p1s = s < {NCH}; int q = p1s ? 0 : s - {NCH}; int i = q / 5, k = q - 5 * i; int nr = {real} - 4 * i; nr = nr > 4 ? 4 : nr;
    if (p1s) {{ if (s + 1 < {NCH}) DMA_FILL((s + 1) & 1, DESC(desc, 0), ((s + 1) & 1) * {CHB}, (int)(x + (s + 1) * {CH})); }}
    else if (k == 0) {{
      if (i == 0) {{ {_hsum_rsqrt_vec(real, c, eps)}{T(1, "tp1_")} }}{" else { DMA_WAIT(3); }" if iqr > 1 else ""}
      {zero1}
     {T0("tl_")} if (i == 0) DMA_FILL(0, DESC(desc, 3), {W0}, (int)(w + b * {HAD_B}));
      DMA_FILL(2, DESC(desc, nr == 4 ? 1 : 2), {L0}, (int)(x + 4 * i * {c} + b * {HAD_B}));
      if (i == 0) DMA_WAIT(0); DMA_WAIT(2);{T(3, "tl_")}
    }}
    int dry = p1s; int dA = p1s ? s == 0 : k < nr, dB = p1s ? s == 1 : k < nr, dC = p1s ? s == 2 : k < nr, dVv = p1s ? s == 3 : k == 4;
    __global float* Lr = dry ? LSF({SCRW}) : L + k * {LPF}; float iv = INVP[dry ? 0 : 4 * i + (k & 3)];
    int tA = dry ? 1 : {HAD_B // 32}, tB = dry ? 1 : 8, tC = dry ? 1 : 16;
   {T0("tR_")} if (dA) {{ {passA} }}
    if (dB) {{ NOUNROLL for (int e = 0; e < 2; e++) {{ {passB}{HW("e")} }} }}
    if (dC) {{ {passC} }}
    if (dVv) {{
      __global float* Cv0 = dry ? LSF({SCRW}) : L; __global half* Ov = dry ? LSH({SCRW + 3712}) : O; int lpv = dry ? 0 : {LP}, tV = dry ? 1 : {HAD_B // 16};
      {conv}
      if (!dry) {{ DMA_DRAIN(3, DESC(desc, 4), {OUT0}, (int)(out + b * {HAD_B // 128 * 512 * iqr} + i * 16)); }}
    }}{T(6, "tR_")}
    if (p1s) {{
     {T0("tw_")} DMA_WAIT(s & 1);{T(2, "tw_")}
      __global float* bb = LSF((s & 1) * {CHB});
      {blk1}
    }}{HW("s")}
  }}
  }}"""
  elif ONEU: unit = f"  {{ const int b = core_id;{' dg_[13] += 1;' if PH else ''}{quad}\n  }}"
  else: unit = f"  for (int b = core_id; b < {nb}; b += {NT}) {{{' dg_[13] += 1;' if PH else ''}\n{quad}{HW('b')}\n  }}"
  src = K.had_a32_src(mode, c, nrb, eps, real=real, compact=True, stacked=stacked, post=post)
  pre = src[:src.index("static inline __attribute__((always_inline)) float8 h8(")] + H16 + NOUNROLL + (CYCDEF if PH else "") + \
    '#define NOHWL(i) __asm__ volatile("" : "+r"(i))\n'
  dg_arg = ", __global int* restrict dg" if PH else ""
  TW = "twice" in opts
  rep0 = ("  for (int rep_ = 0; rep_ < 2; rep_++) {\n" + ("   if (rep_ == 1) { int s0_ = dg_[0] + CYC() - t00_; int e_ = dg_[12], u_ = dg_[13]; for (int q_ = 0; q_ < 16; q_++) dg_[q_] = 0; dg_[14] = s0_; dg_[12] = e_; dg_[13] = u_; t00_ = CYC(); }\n" if PH else "")) if TW else ""
  rep1 = "    DMA_WAIT_ALL();\n  }\n" if TW else ""
  phd = "  int dg_[16]; for (int q_ = 0; q_ < 16; q_++) dg_[q_] = 0; int t00_ = CYC(); dg_[12] = t00_ & 0x7fffffff;\n" if PH else ""
  fin = (f"""{T0("tf_")} DMA_WAIT_ALL();{T(11, "tf_")}
  dg_[0] += CYC() - t00_;
  for (int q_ = 0; q_ < 16; q_++) dg[core_id * 16 + q_] = dg_[q_];""" if PH else "  DMA_WAIT_ALL();")
  return pre + f"""__kernel void had_a32d(__global half* restrict out, __global float* restrict x{"" if mode == "swiglu" else ", __global float* restrict w"}{", __global int* restrict idx" if stacked else ""}{", __global float* restrict rec" if "rec" in opts else ""}, __global int* restrict desc{dg_arg}, const int core_id) {{
{"  w += idx[0] * %d;" % c if stacked else ""}
{phd}  if (core_id < {nb}) {{
{rep0}  __global float* INVP = LSF({INV0});{phase1}
  __global float* L = LSF({L0}); __global half* O = LSH({OUT0});{" int pend = 0;" if MULTI and not P1W else ""}{"" if mode == "swiglu" else f" __global float* W = LSF({W0});"}
{unit}
{rep1}  }}
{fin}
}}"""

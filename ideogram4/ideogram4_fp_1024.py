#!/usr/bin/env python3
"""Ideogram 4 text-to-image at FULL PRECISION on the NPU, any 96-px-multiple size (1024 x 1024 = 4096 image
tokens): both transformers -- the conditional one over [text tokens | image tokens], the unconditional one
over the image tokens -- with classifier-free guidance, every block on the device: E4M3 weights unpacked
exactly to fp16 (`ops_zhouyi.e4m3_stream`), the linears and the attention as fp16 x fp16 -> fp32 on the
TEC matrix unit (`ops_zhouyi.gemm_gs`, 8-row-block pieces), the block's other math as DMA-fed vector
kernels (`ideogram4_vec.py`: rms_scale / qkv / softmax_stream / gather_o_sum / resid / swiglu). The host does
the per-step scalars (t-embedding, AdaLN, input projection, final layer, guidance, the Euler update) and,
once per image, the text conditioning (`ideogram4_text.py`).

    python3 ideogram4_fp_1024.py --text ~/ideogram4/text_rows.npz --steps 20 --size 1024 --seed 123 \\
        --out ~/ideogram4/cat_latents.npy
    GATE=1 ... --steps 1 --layers 0-0 --branch cond     (layer 0 op by op vs numpy, and vs the fp32 block)

Geometry (1024 px): 4096 image tokens (+ n text), rows padded to a multiple of 96 (ROWS = 4128 for n <= 32,
344 row blocks), keys padded the same (TP = ROWS); attention in three batches of 6 heads (the 6 heads' logits
are 409 MB: all 18 would not fit the 32-bit device address space beside the rest). The big fp32 scratch is
shared by the qkv linear's output, the logits and the w1|w3 linear's output (they are dead in turn).

Seed: the pipeline's `randn_tensor` on a CPU generator, replicated bit-for-bit (`torch_randn.py`).
"""
import argparse, ctypes, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
os.environ.setdefault("ZHOUYI_GM", "1")      # the GEMMs with GM (GemmGMRunner: the A chunks and the partials in GM, 2026-09-27); 0: the GSRAM runner
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ZHOUYI_CHAIN", "64")                 # one fused chain per layer (~27 kernels)
try: os.sched_setaffinity(0, {0, 1, 6, 7, 8, 9, 10, 11})
except (AttributeError, OSError): pass
from tinygrad import Tensor, dtypes                                      # noqa: E402
from zy import OA                            # noqa: E402
from zy import gemm_fp16 as G                 # noqa: E402
import ideogram4_vec as V                    # noqa: E402  (the model's vector kernels + the generic vec_f16 helpers)
import ideogram4_ref as R                                                # noqa: E402
import ideogram4_fp_backend as FB                                        # noqa: E402  (host_read, dev, zeros, _ulps)
from ideogram4_weights import Ideogram4Weights                           # noqa: E402
from torch_randn import randn as torch_randn                             # noqa: E402

DEV = "ZHOUYI"; NT = 12; KS, NS = 24, 6; PIECE = 8
ROWMAX = os.environ.get("ROWMAX", "1") == "1"               # the scores' GEMM writes each row block's row maxima after its C row (k_gemm_gs(rowmax=True)): the softmax skips its scan pass; ROWMAX=0: the scan
LIN48 = os.environ.get("LIN48", "1") == "1"
# V48=1: P V on the 48-deep 3-strip kernel (V^T written in its panel order by qkv_dma; the keys padded to a multiple of 192).
# Default 0 since the w2 padding change: with the GEMMs' partials in GM (ZHOUYI_GM=1) the 24-deep 6-strip kernel reads half the
# A (P) bytes a column and P V is DDR-bound: 81 -> 66 ms a 6-head batch at 4704 rows, step-1 latents bit-identical (PAD_W2.md)
V48 = LIN48 and os.environ.get("V48", "0") == "1"
QKV_FAST = os.environ.get("QKV_FAST", "1") == "1"         # qkv_dma(fast=True): loop-buffer bodies, the scale folded in, the half rotary tables double-buffered (bit-identical)
B8 = LIN48 and os.environ.get("B8", "1") == "1"            # the linears read the E4M3 codes and expand them in LSRAM (k_gemm_gs b8): half the weight bytes, no fp16 panels pass; C x 2^-8, the scales x 2^8                 # the linears on the 48-deep 3-strip k_gemm_gs (weights repacked, `p48/`); LIN48=0: the 24 x 6 kernel
GATE = bool(int(os.environ.get("GATE", "0"))); PROF = bool(int(os.environ.get("PROF", "0")))
C_, NH, DH, DHP, M_ = R.D, R.NH, R.HD, 288, R.FF
# the head dim's padding (DH 256 in DHP 288 = 3 x 96) skipped where it costs (gemm_gs ktail / nlive, ZHOUYI_GM=1; bit-identical):
# the scores' GEMM stops at DH -- the last of its K slices holds (DH mod 96) / 4 live k-steps (16), the rest Q's and K's zeros --
# and P V (V48=0) computes only DH columns (its last 96-column group's first 4 strips; the dead columns, never read by gather_o_sum,
# are left unwritten). S_KTAIL=0 / PV_NLIVE=0: the padded K / N
S_KTAIL = (DH - (DHP // 96 - 1) * 96) // 4 if os.environ.get("S_KTAIL", "1") == "1" and DH % 96 else 0
PV_NLIVE = DH if os.environ.get("PV_NLIVE", "1") == "1" else 0
N_QKV, N_W13 = 3 * NH * DH, 2 * M_
W_ORDER = ["qkv", "o", "w13", "w2"]
# W8A8 on the AIFF (`ideogram4_q8`; `--q8 PACKDIR`, set in main): qkv, o and w1|w3 as int8 per token x int8 per
# channel with SmoothQuant folded in, on the three AIFF engines; w2 on the TEC in fp16 as before unless `--q8-w2`. Off: today's path.
Q8 = False; Q8_W2 = False; Q8_PACK = None; Q8_DESC = None
Q8_GM = os.environ.get("Q8_GM", "1") == "1"                 # the AIFF reads the weights from GM slots the TECs stage into (w2: DDR); 0: from DDR
Q8_SWI = os.environ.get("Q8_SWI", "1") == "1"               # w1|w3's dequantiser fused into SwiGLU (swiglu_i16; both halves' partials in `big`); 0: dq_ct + swiglu_dma
W13S = QS = 1              # w1|w3's / qkv's channel parts on the AIFF path (the geometry's memory plan; set in main before the layers load)
CAL = None                 # calibration statistics (`--calib-stats`): "act" = per-column |max| of the linears' fp16 operands,
                           # "part" = per-column |max| of the AIFF's int16 partials (needs --q8); accumulated on the device per layer, read per step
Q8_LIN = ["qkv", "o", "w1", "w3"]                          # the AIFF parts (w1, w3: the two halves of w1|w3, 12288 columns each)
Q8_DQ = ("Cqkv0", "Cqkv", "Co", "C13a", "C13", "Cm")                # the dequantisers' outputs (the gate's names)
dev, zeros = FB.dev, FB.zeros


class Geo:
    """The sequence geometry shared by both transformers (the same buffers and GEMM plans)."""
    def __init__(self, n_img, n_text):
        self.n_img, self.n_text = n_img, n_text
        # the rows: a multiple of 96 with 8 | NRB (the attention GEMMs' 8-row-block pieces), the one with the fewest DDR bytes for the
        # linears (at the device's DMA ceiling): rows x (32 B of A + 128 or 64 (B8) B of weights / the lin48 piece) per unit, the piece
        # the largest even divisor <= 28 of NRB (4621 tokens -> 4704 rows, pieces of 28; 4096 -> 4224, 22)
        n = n_img + n_text; wb = 64 if B8 else 128
        piece = lambda nrb: max(p_ for p_ in range(2, 29, 2) if nrb % p_ == 0)
        self.ROWS = min((r for r in range(-(-n // 96) * 96, -(-n // 96) * 96 + 960, 96) if (r // 12) % 8 == 0), key=lambda r: r * (32 + wb / piece(r // 12)))
        if os.environ.get("I4_ROWS"):      # a forced row count (>= n, a multiple of 96, 8 | NRB): e.g. 4608 for 4416, whose fp16 w2 GEMM's 46 launches
            r = int(os.environ["I4_ROWS"]); assert r >= n and r % 96 == 0 and (r // 12) % 8 == 0, r; self.ROWS = r   # x graph clones exhaust the device memory
        self.NRB = self.ROWS // 12; self.TP = -(-self.ROWS // 192) * 192 if V48 else self.ROWS   # the keys (N of the scores, K of P V)
        self.QA, self.KB, self.VB, self.QKV_TOTAL = V.qkv_a16_sizes(NH, DH, DHP, self.NRB, KS, NS, self.TP)
        self.HC = 3 * self.NRB * 1152                         # floats per head of the P V tiles (N = 288)
        self.HS = (self.TP // 96) * self.NRB * (1200 if ROWMAX else 1152)   # floats per head of the logit tiles (N = TP; + the row maxima)
        # the memory plan counts the attention buffers at their V48=1 sizes (keys padded to a multiple of 192) whatever V48: the
        # plans stay those of ROWS_TEXT.md (V48=0's smaller TP would let q8-w2 at 6240 rows take 3-head batches 3 MiB under the budget)
        tpa = -(-self.ROWS // 192) * 192
        self.HSA, self.PA, self.QKVA = (tpa // 96) * self.NRB * (1200 if ROWMAX else 1152), self.ROWS * tpa, V.qkv_a16_sizes(NH, DH, DHP, self.NRB, KS, NS, tpa)[3]
        self.C48 = 48 * self.NRB * 1152                       # a linear with N = 4608
        if Q8:
            import ideogram4_q8 as Q
            self.qgm = {k: Q8_GM and k != "w2" for k in Q8_LIN + ["w2"]}            # w2's 1 MiB tiles: 6 slots do not fit the 4 MiB window
            hs = lambda k: int(os.environ.get("Q8_HSTEP_GM", "16")) if self.qgm[k] else 32   # h_step: 16 from GM (9 % faster, measured), 32 from DDR
            self.qp = dict(qkv=Q.Plan(self.ROWS, C_, N_QKV, hstep=hs("qkv")), o=Q.Plan(self.ROWS, C_, C_, hstep=hs("o")),
                           w1=Q.Plan(self.ROWS, C_, M_, hstep=hs("w1")), w3=Q.Plan(self.ROWS, C_, M_, hstep=hs("w3")))
            if Q8_W2: self.qp["w2"] = Q.Plan(self.ROWS, M_, C_, hstep=hs("w2"))
            assert len({self.qp[k].M_pad for k in Q8_LIN}) == 1, "qkv / o / w1|w3 share the int8 surfaces' row pitch"
            self.W13S = self.QS = 1
        self.plan_memory()
        if Q8: self._aiff_parts()

    def _aiff_parts(self):
        """The AIFF calls of a layer (after the memory plan): qkv, o, w1|w3 -- whole (w1, w3) or, `W13S == 2`, in channel halves
        (w1_0, w3_0, w1_1, w3_1: 6144 channels each, half the partials buffer) -- [, w2]: each its template, descriptor slot, shifts
        and (GM) staging offsets into its weight slot."""
        import ideogram4_q8 as Q
        self.aparts, self.qd, self.qargs, self.apl = [], {}, {}, {}
        self.splits = {"qkv": self.QS, "o": 1, "w1": self.W13S, "w3": self.W13S, "w2": 1}
        M2 = M_ // self.W13S
        qk = ["qkv"] if self.QS == 1 else ["qkv_0", "qkv_1"]
        for k in qk + ["o"] + (["w1", "w3"] if self.W13S == 1 else ["w1_0", "w3_0", "w1_1", "w3_1"]) + (["w2"] if Q8_W2 else []):
            w, h = (k.split("_")[0], int(k.split("_")[1])) if "_" in k else (k, 0)
            p = self.qp[w] if self.splits[w] == 1 else Q.Plan(self.ROWS, C_, self.qp[w].N // self.splits[w], hstep=self.qp[w].hstep)
            cat = dict(ntot=2 * M2, coff=M2 if w == "w3" else 0) if Q8_SWI and w in ("w1", "w3") else {}
            D = Q.Desc(Q.desc_path(p, Q8_DESC, gm=self.qgm[w], **cat))
            self.aparts.append((k, w, h)); self.qd[k] = D; self.apl[k] = p
            if not self.qgm[w]: self.qargs[k] = (D.chains.tolist(),); continue
            R = (p.engine_nodes[0][1] - p.engine_nodes[0][0]) * p.S; tile = 2 * p.kc * Q.UNIT; j0 = h * p.nodes        # the half's first node in w
            src = [[((r % p.S) * self.qp[w].nodes + j0 + p.engine_nodes[e][0] + r // p.S) * tile for r in range(R)] for e in range(3)]
            gch = [tuple(int(x) for x in D.info[i]) + tuple(int(x) for x in D.chains[i][1:]) for i in range(len(D.chains))]
            self.qargs[k] = (R, tile, gch, src)

    def part_sh(self, L, k):
        """A part's shift table [passes, its 128-channel units] from the layer's."""
        w, h = (k.split("_")[0], int(k.split("_")[1])) if "_" in k else (k, None)
        if h is None: return L.sh[w]
        u = self.qp[w].N // self.splits[w] // 128; return L.sh[w][:, h * u:(h + 1) * u]

    def part_cols(self, k):
        """A part's columns of its linear (a slice)."""
        w, h = (k.split("_")[0], int(k.split("_")[1])) if "_" in k else (k, 0)
        n = self.qp[w].N // self.splits[w]; return slice(h * n, (h + 1) * n)

    @property
    def p13(self) -> int:
        """w1|w3's partials buffer, both parts (bytes): [M_pad][S][2 M / W13S]."""
        return self.qp["w1"].M_pad * self.qp["w1"].S * 2 * (M_ // self.W13S) * 2

    # The device's buffers share ONE 3 GB IOVA window (the KMD's ASID 0: `V3: alloc_iova failed for ASID 0`). The big ones are exactly
    # `buf_bytes` + the weight pools; the rest (runners' launch arenas and their graph copies, the per-layer tables, the I/O and
    # modulation state) is OVERHEAD_MB, measured (ROWS_TEXT.md). The plan: the fewest attention batches (HB 6, then 3, 2) and two pool
    # slots (the prefetch overlap) if they fit; one slot (host copies serialised) only past that.
    BUDGET_MB = float(os.environ.get("I4_MEM_BUDGET", "2900"))
    def overhead_mb(self) -> float:
        return OVERHEAD_MB["q8" if Q8 else "fp8"](self.ROWS)
    def plan_memory(self):
        hb, sl = os.environ.get("I4_HB"), os.environ.get("I4_SLOTS")
        sp = (1, 2) if Q8 and Q8_SWI and CAL is None else (1,)
        # cheapest first: fewer attention batches, no channel splits (extra AIFF calls), and two pool slots (the prefetch overlap,
        # ~50 ms a layer) above all
        cost = lambda c: {6: 0, 3: 0.3, 2: 0.6}[c[0]] + (1.0 if c[1] == 1 else 0) + 0.2 * (c[2] - 1) + 0.2 * (c[3] - 1)
        cands = sorted([(h_, s_, w_, q_) for h_ in (6, 3, 2) for s_ in (2, 1) for w_ in sp for q_ in sp], key=cost)
        if hb or sl or os.environ.get("I4_W13S") or os.environ.get("I4_QS"):
            cands = [(int(hb or 6), int(sl or 2), int(os.environ.get("I4_W13S", "1")), int(os.environ.get("I4_QS", "1")))]
        for h_, s_, w_, q_ in cands:
            self.HB, self.SLOTS, self.W13S, self.QS = h_, s_, w_, q_
            est = (sum(buf_bytes(self).values()) + s_ * pool_bytes(self)) / 2**20 + self.overhead_mb()
            if est <= self.BUDGET_MB or len(cands) == 1: break
        self.mem_est = est
        print("   memory plan: %d rows, %d heads an attention batch, %d weight-pool slot(s), w1|w3 / qkv in %d / %d part(s): ~%.0f MiB of the %.0f-MiB budget (3072 window; "
              "buffers %.0f + pools %.0f + overhead %.0f)" % (self.ROWS, self.HB, self.SLOTS, getattr(self, "W13S", 1), getattr(self, "QS", 1), est, self.BUDGET_MB, sum(buf_bytes(self).values()) / 2**20,
              self.SLOTS * pool_bytes(self) / 2**20, self.overhead_mb()), flush=True)


class Kernels:
    """The vector kernels for one sequence (`treal` real rows): registered once per transformer branch."""
    def __init__(self, g: Geo, treal: int):
        reg = lambda name, src: OA.register_csrc(name, src, ntasks=NT)
        self.treal = treal
        self.rms = reg("rms_scale_dma", V.rms_scale_dma_src(treal, C_, R.EPS, KS, g.NRB, nt=NT)); self.d_rms = dev(V.rms_scale_dma_descs(C_, g.NRB))
        self.resid = reg("resid_dma", V.resid_dma_src(treal, C_, NS, g.NRB, R.EPS, nt=NT)); self.d_resid = dev(V.resid_dma_descs(C_, g.NRB))
        self.swiglu = reg("swiglu_dma", V.swiglu_dma_src(treal, M_, g.NRB, KS, nt=NT, interleaved=True)); self.d_swiglu = dev(V.swiglu_dma_descs(M_, g.NRB, KS, interleaved=True))
        self.qkv = reg("qkv_dma", V.qkv_dma_src(treal, NH, DH, DHP, g.NRB, KS, NS, g.TP, 1e-5, nt=NT, v48=V48, fast=QKV_FAST)); self.d_qkv = dev(V.qkv_dma_descs(g.NRB, KS, NS))
        SGB = int(os.environ.get("SOFTMAX_GB", "2" if (g.TP // 96) % 2 == 0 else "1"))   # groups per softmax DMA request (1: the single-group kernel)
        self.soft = [reg("softmax_stream", V.softmax_stream_src(treal, g.HB, g.TP, g.NRB, KS, 1.0 / math.sqrt(DH), hb=b * g.HB, nt=NT, gb=SGB, rowmax=ROWMAX)) for b in range(NH // g.HB)]
        self.d_soft = dev(V.softmax_stream_descs(g.NRB, SGB, rowmax=g.TP // 96 if ROWMAX else 0))
        self.gather = reg("gather_o_sum", V.gather_o_sum_src(treal, NH, DH, DHP, NS, g.NRB, KS, nt=NT)); self.d_gather = dev(V.gather_o_sum_descs(g.NRB))
        if CAL is not None:
            import ideogram4_q8 as Q
            if CAL == "act":
                self.cmax = {K: reg("colmax_a16", Q.colmax_a16_src(g.NRB, K, nt=NT)) for K in (C_, M_)}; self.d_cmax = dev(Q.colmax_a16_descs())
            else:
                self.pmax = {k: reg("colmax_i16", Q.colmax_i16_src(g.ROWS, W, nt=NT)) for k, W in cal_widths().items()}
                self.d_pmax = {k: dev(Q.colmax_i16_descs(g.ROWS, W)) for k, W in cal_widths().items()}
        if Q8:      # the quantiser (per K) and the dequantiser (per AIFF part: w1 / w3 land on alternate tile groups, swiglu's order)
            import ideogram4_q8 as Q
            mp = {C_: g.qp["qkv"].M_pad, M_: g.qp["w2"].M_pad if Q8_W2 else 0}      # the surfaces' row pitch per K (w2's plan may differ)
            self.q8 = {K: reg("q8_a16", Q.q8_a16_src(g.NRB, K, mp[K], nt=NT)) for K in ((C_, M_) if Q8_W2 else (C_,))}
            self.d_q8s = {K: dev(Q.q8_a16_descs(mp[K], g.NRB)) for K in self.q8}
            if Q8_SWI:
                p1 = g.qp["w1"]
                m2 = M_ // g.W13S
                self.swi = [reg("swiglu_i16", Q.swiglu_i16_src(g.NRB, p1.S, m2, p1.M_pad, nt=NT, gofs=h * m2 // 96)) for h in range(g.W13S)]
                self.d_swi = dev(Q.swiglu_i16_descs(p1.S, m2))
            gm = dict(qkv=(1, 0), o=(1, 0), w1=(2, 0), w3=(2, 1), w2=(1, 0))
            dqp = [(k, w, h) for k, w, h in g.aparts if not (Q8_SWI and w in ("w1", "w3"))]     # the parts a dequantiser follows
            self.dq = {k: reg("dq_ct", Q.dq_ct_src(g.NRB, g.apl[k].N // 96, g.apl[k].S, g.apl[k].N, g.apl[k].M_pad, gmul=gm[w][0],
                                                     gofs=gm[w][1] + h * g.apl[k].N // 96, nt=NT)) for k, w, h in dqp}
            self.d_dq = {k: dev(Q.dq_ct_descs(g.apl[k].S, g.apl[k].N)) for k, _, _ in dqp}


def zeros_r(n, dt):
    """A zeroed device buffer rounded up to 12 x 8 KiB (the verified host read's checksum granule)."""
    per = 98304 // dt.itemsize
    return zeros(-(-n // per) * per, dt)


def _r(n: int) -> int: return -(-n // 98304) * 98304          # zeros_r's rounding (bytes)


# The steady-state device memory besides `buf_bytes` and the pools (MiB, a function of the row count): the runners' launch arenas
# and their graph copies, the per-layer tables and modulation state, the I/O projections, kernel text. Measured on the board as the
# peak minus the two (ROWS_TEXT.md); a margin is in BUDGET_MB.
OVERHEAD_MB = {"fp8": lambda rows: 770.0, "q8": lambda rows: 820.0}   # measured max 759 / 804 (q8w2 686)


def buf_bytes(g: Geo) -> dict:
    """Every big buffer `make_bufs` allocates, in bytes (the same expressions)."""
    f, h = 4, 2
    big = max(144 * g.NRB * 1152, g.HB * g.HSA, 0 if Q8 and Q8_SWI else 256 * g.NRB * 1152, (g.p13 // 4) if Q8 and Q8_SWI else 0) * f
    b = dict(Ah=g.ROWS * C_ * h, big=big, QKV=g.QKVA * h, P=g.HB * g.PA * h, PV=NH * g.HC * f, sums=NH * g.NRB * 16 * f,
             Ao=g.ROWS * C_ * h, C48=g.C48 * f, x1=g.ROWS * C_ * f, Ah2=g.ROWS * C_ * h, Am=g.ROWS * M_ * h, out0=g.ROWS * C_ * f, out1=g.ROWS * C_ * f)
    if Q8:
        b["A8"] = max(p.a8_bytes for p in g.qp.values()); b["SX"] = g.NRB * 16 * f
        pb = max(p.p_bytes // (g.QS if k == "qkv" else 1) for k, p in g.qp.items() if not (Q8_SWI and k in ("w1", "w3")))
        if _r(b["P"]) < pb: b["PQ"] = pb                    # the partials in the attention's P buffer when it is big enough
    return {k: _r(v) for k, v in b.items()}


def pool_bytes(g: Geo) -> int:
    """One weight-pool slot (bytes)."""
    if not Q8: return N_QKV * C_ + C_ * C_ + N_W13 * C_ + C_ * M_
    dsc = sum(-(-d.nbytes // 98304) * 98304 for d in g.qd.values()) if getattr(g, "qd", None) else (6 + 2 * g.W13S) * (1 << 20)
    return sum(p.w_bytes for p in g.qp.values()) + (0 if Q8_W2 else C_ * M_) + dsc


def make_bufs(g: Geo):
    f32, u16 = dtypes.float32, dtypes.uint16
    zeros = zeros_r; sz = buf_bytes(g)
    b = dict(Ah=zeros(g.ROWS * C_, u16), big=zeros(sz["big"] // 4, f32), QKV=zeros(sz["QKV"] // 2, u16), P=zeros(sz["P"] // 2, u16),
             PV=zeros(NH * g.HC, f32), sums=zeros(NH * g.NRB * 16, f32), Ao=zeros(g.ROWS * C_, u16), C48=zeros(g.C48, f32),
             x1=zeros(g.ROWS * C_, f32), Ah2=zeros(g.ROWS * C_, u16), Am=zeros(g.ROWS * M_, u16),
             out0=zeros(g.ROWS * C_, f32), out1=zeros(g.ROWS * C_, f32))
    if not B8: b["panels"] = dict(qkv=zeros(N_QKV * C_, u16), o=zeros(C_ * C_, u16), w13=zeros(N_W13 * C_, u16), w2=zeros(C_ * M_, u16))
    if Q8:
        # the int8 surfaces (K = 12288 with an int8 w2), the per-token scales, and the AIFF's int16 partials: in the attention's P buffer
        # (dead from the PV GEMMs to the next layer's softmax) when it is big enough
        b["A8"] = zeros(sz["A8"], dtypes.uint8)
        b["SX"] = zeros(g.NRB * 16, f32)
        b["PQ"] = zeros(sz["PQ"] // 2, u16) if "PQ" in sz else b["P"]
    return b


def make_pool(g: Geo = None):
    """The weight-pool slots, one per layer parity (the prefetch worker fills the other while a layer runs); `g.SLOTS == 1`: ONE slot
    both parities use (no overlap: `forward` copies a layer's weights after the previous layer's job, on the submitting thread)."""
    if Q8:
        # int8 AIFF weights (+ w2's E4M3 codes for the TEC unless --q8-w2) and the parts' relocated descriptors, per layer parity
        def mk():
            s = {k: zeros(p.w_bytes, dtypes.uint8) for k, p in g.qp.items()}
            if not Q8_W2: s["w2"] = zeros(C_ * M_, dtypes.uint8)
            s.update({"d_" + k: zeros(-(-g.qd[k].nbytes // 98304) * 98304 // 4, dtypes.uint32) for k, _, _ in g.aparts})
            return s
    else:
        mk = lambda: dict(qkv=zeros(N_QKV * C_, dtypes.uint8), o=zeros(C_ * C_, dtypes.uint8), w13=zeros(N_W13 * C_, dtypes.uint8), w2=zeros(C_ * M_, dtypes.uint8))
    one = mk()
    return [one, mk() if g.SLOTS == 2 else one]


def slot_names(g: Geo):
    """The pool slots a layer's JIT takes, in order."""
    if not Q8: return list(W_ORDER)
    return list(Q8_LIN) + ["w2"] + ["d_" + k for k, _, _ in g.aparts]


def cal_widths() -> dict:
    """The statistics buffers' widths (floats) per name."""
    if CAL == "act": return {"qkv": C_, "o": C_, "w13": C_, "w2": M_}
    w = {"qkv": 2 * N_QKV, "o": 2 * C_, "w13": 2 * N_W13}
    if Q8_W2: w["w2"] = 3 * C_
    return w


def small_names(g: Geo):
    """The layer's resident tables a layer's JIT takes, in order."""
    return (["st_" + k for k in cal_widths()] if CAL is not None else []) + ["sc_qkv", "sc_o", "sc_w13", "sc_w2", "nq2", "nk2"] + (["cs_" + k for k, w, _ in g.aparts if not (Q8_SWI and w in ("w1", "w3"))] + ["cs13_%d" % h for h in range(g.W13S)] if Q8 else [])


class Layer:
    """One layer of one transformer: the packed codes (memory-mapped), the small tensors, the modulation."""
    def __init__(self, l, cache, q8dir=None):
        self.l = l
        sm = np.load(os.path.join(cache, f"L{l}_small.npz"))
        self.sc = {k: sm["sc_" + k] for k in W_ORDER}
        self.an1, self.an2, self.fn1, self.fn2 = (sm[k] for k in ("an1", "an2", "fn1", "fn2"))
        ac = sm["ada_codes"]; self.ada_w = (G.e4m3_table()[ac] * sm["ada_scale"][:, None]).astype(np.float32) if ac.dtype == np.uint8 else ac
        self.ada_b = sm["ada_b"]
        self._sm = sm
        self.norm_q, self.norm_k = sm["norm_q"], sm["norm_k"]
        self.codes = {k: np.memmap(os.path.join(cache, f"L{l}_{k}.bin"), np.uint8, "c") for k in W_ORDER}   # "c": copy-on-write, writable for ctypes (never written)
        # w1|w3 in the interleaved group order (swiglu_dma's), made once per cache (`L{l}_w13i.bin`)
        idir = os.path.join(os.path.expanduser(os.environ.get("W13I_DIR", "~/ideogram4/w13i")), os.path.basename(os.path.normpath(cache))); os.makedirs(idir, exist_ok=True)
        wi = os.path.join(idir, f"L{l}_w13i.bin"); si = os.path.join(idir, f"L{l}_sc_w13i.npy")
        if not os.path.exists(wi):
            c, sc = V.interleave_w13(np.asarray(self.codes["w13"]), self.sc["w13"], M_)
            np.save(si, sc); c.tofile(wi + ".tmp"); os.replace(wi + ".tmp", wi)
        self.codes["w13"] = np.memmap(wi, np.uint8, "c"); self.sc["w13"] = np.load(si)
        if LIN48:                  # the panels in `repack_panels_48x3` order, made once per cache (a byte permutation of the files above)
            pdir = os.path.join(os.path.expanduser(os.environ.get("P48_DIR", "~/ideogram4/p48")), os.path.basename(os.path.normpath(cache))); os.makedirs(pdir, exist_ok=True)
            for k, K in (("qkv", C_), ("o", C_), ("w13", C_), ("w2", M_)):
                pf = os.path.join(pdir, f"L{l}_{k}.bin")
                if not os.path.exists(pf):
                    G.repack_panels_48x3(np.asarray(self.codes[k]), K).tofile(pf + ".tmp"); os.replace(pf + ".tmp", pf)
                self.codes[k] = np.memmap(pf, np.uint8, "c")
        if B8: self.sc = {k: (v * np.float32(256)).astype(np.float32) for k, v in self.sc.items()}   # the GEMM's C is x 2^-8 (exact both ways)
        if Q8: self._q8_init(l, q8dir)
        self.t = dict(sc_qkv=dev(self.sc["qkv"]), sc_o=dev(self.sc["o"]), sc_w13=dev(self.sc["w13"]), sc_w2=dev(self.sc["w2"]),
                      nq2=dev(V.dup_quads(self._sm["norm_q"])), nk2=dev(V.dup_quads(self._sm["norm_k"])))
        if Q8:
            self.t.update({"cs_" + k: dev(v.ravel()) for k, v in self.cs.items()})
            if QS == 2: self.t.update({"cs_qkv_%d" % h: dev(np.ascontiguousarray(self.cs["qkv"][:, h * N_QKV // 2:(h + 1) * N_QKV // 2]).ravel()) for h in range(2)})
        if CAL is not None: self.t.update({"st_" + k: zeros_r(n, dtypes.float32) for k, n in cal_widths().items()})
        if Q8:   # swiglu_i16's [S][w1 | w3] per channel part (W13S, a module setting the geometry fixes before the layers load)
            m2 = M_ // W13S
            for h in range(W13S): self.t["cs13_%d" % h] = dev(np.concatenate([self.cs["w1"][:, h * m2:(h + 1) * m2], self.cs["w3"][:, h * m2:(h + 1) * m2]], 1).ravel())

    def _q8_init(self, l, q8dir):
        """The layer's int8 AIFF weights (memory-mapped, `ideogram4_q8_pack.py`) and tables. SmoothQuant's 1/lam is folded into the
        producers' per-channel multipliers: an1 (qkv's rms_scale), fn1 (w1|w3's), V's dequant columns (o's input is the attention
        output, linear in V) and, with --q8-w2, w3's dequant columns (w2's input is silu(w1 x) * w3 x). The dequantised C tiles are
        true values, so the consumers' column scales become 1 (w2's stays when it runs on the TEC)."""
        import ideogram4_q8 as Q
        z = np.load(os.path.join(q8dir, f"L{l}_q8.npz")); self.q8 = {k: z[k] for k in z.files}
        mm = lambda k: np.memmap(os.path.join(q8dir, f"L{l}_{k}.bin"), np.uint8, "c")
        w2_fp8 = self.codes["w2"]                                               # the TEC path's E4M3 panels (p48), kept without --q8-w2
        self.codes = {k: mm(k) for k in Q8_LIN}
        self.codes["w2"] = mm("w2") if Q8_W2 else w2_fp8
        z_ = self.q8
        self.an1 = (self.an1 / z_["lam_qkv"]).astype(np.float32); self.fn1 = (self.fn1 / z_["lam_w13"]).astype(np.float32)
        one = lambda n: np.ones(2 * n, np.float32)                              # dup-quads layout: 2 floats a column
        self.sc.update(qkv=one(N_QKV), o=one(C_), w13=one(N_W13))
        if Q8_W2: self.sc["w2"] = one(C_)
        vfold = np.concatenate([np.ones(2 * C_, np.float32), (1.0 / z_["lam_o"]).astype(np.float32)])        # V's columns carry o's 1/lam
        sh13, sw13 = z_["sh_w13"], z_["sw_w13"]; u = M_ // Q.UNIT
        w3x = (1.0 / z_["lam_w2"]).astype(np.float32) if Q8_W2 else None
        self.sh = dict(qkv=z_["sh_qkv"], o=z_["sh_o"], w1=sh13[:, :u], w3=sh13[:, u:])
        self.cs = dict(qkv=Q.scales_cs(z_["sw_qkv"], z_["sh_qkv"], vfold), o=Q.scales_cs(z_["sw_o"], z_["sh_o"]),
                       w1=Q.scales_cs(sw13[:M_], sh13[:, :u]), w3=Q.scales_cs(sw13[M_:], sh13[:, u:], w3x))
        if Q8_W2: self.sh["w2"] = z_["sh_w2"]; self.cs["w2"] = Q.scales_cs(z_["sw_w2"], z_["sh_w2"])

    def modulation(self, adaln):
        s1, g1, s2, g2 = np.split(self.ada_w @ adaln + self.ada_b, 4)
        return (1.0 + s1).astype(np.float32), np.tanh(g1).astype(np.float32), (1.0 + s2).astype(np.float32), np.tanh(g2).astype(np.float32)


def block_1024(g: Geo, K: Kernels, t: dict, cs, sn, x, ws1, k1, ws2, k2, codes: dict, bufs: dict, out, gate=None):
    """One block on the device: x `[ROWS, 4608]` fp32 -> `out`."""
    # the attention GEMMs' row pieces: the largest even divisor <= APIECE (14: 14 x 6 x 768 B of C = 64512 of the task's 64 KiB of
    # GSRAM) -- the K / V panels are re-read per piece, and the scores' GEMM sits near the DMA ceiling
    ap = max(p_ for p_ in range(2, int(os.environ.get("APIECE", "14")) + 1, 2) if g.NRB % p_ == 0)
    gm = lambda a, b, **kw: OA.gemm_gs(a, b, ks=KS, ns=NS, nrb=g.NRB, piece=ap, **kw)
    cs_ = OA.csrc_call
    chk = (lambda name, tt, fn=None: gate.check(name, tt, fn)) if gate is not None else (lambda name, tt, fn=None: tt)
    big = bufs["big"]
    pn = dict(codes) if B8 else {k: chk("panels_" + k, OA.e4m3_stream(codes[k], out=bufs["panels"][k])) for k in W_ORDER}
    Ah = chk("Ah", cs_(K.rms, bufs["Ah"], x, ws1, K.d_rms))
    cal_a = (lambda name, a16, kd: cs_(K.cmax[kd], t["st_" + name], a16.realize(), K.d_cmax).realize()) if CAL == "act" else (lambda *_: None)
    cal_a("qkv", Ah, C_)
    lin = lambda a_, b_, K, N, out: OA.gemm_gs(a_, b_, ks=KS, ns=NS, nrb=g.NRB, nslices=K // 96, ngroups=N // 96, lin48=LIN48, piece=0 if LIN48 else PIECE, b8=B8, out=out)
    if Q8:
        def q8lin(a16, kdim, parts, names, out):
            """a16 (the producer's fp16 A layout) -> int8 surfaces + s_x (TEC) -> each AIFF part's int16 partials -> its dequantised
            tiles in `out` (TEC). Every op realized in order: the parts share the partials buffer and the surfaces (no data edge
            orders part 2's AIFF after part 1's dequantiser)."""
            A8 = chk("A8_" + parts[0], cs_(K.q8[kdim], bufs["A8"], a16, bufs["SX"], K.d_q8s[kdim]).realize())
            for p, nm in zip(parts, names):
                w = p.split("_")[0]; aiff = OA.aiff_gm if g.qgm[w] else OA.aiff_lin
                Pp = chk("P_" + p, aiff(bufs["PQ"], A8, codes[w], codes["d_" + p], *g.qargs[p]).realize())
                if CAL == "part": cs_(K.pmax[p], t["st_" + p], Pp, K.d_pmax[p]).realize()
                C = chk(nm, cs_(K.dq[p], out, Pp, bufs["SX"], t["cs_" + p], K.d_dq[p]).realize())
            return C
    if Q8:
        qp_ = [k for k, w, _ in g.aparts if w == "qkv"]; Cq = q8lin(Ah, C_, qp_, ["Cqkv%d" % i for i in range(len(qp_) - 1)] + ["Cqkv"], big)
    else: Cq = chk("Cqkv", lin(Ah, pn["qkv"], C_, N_QKV, big))
    QKV = chk("QKV", cs_(K.qkv, bufs["QKV"], Cq, t["nq2"], t["nk2"], cs, sn, t["sc_qkv"], K.d_qkv))
    PV = bufs["PV"]
    # ⚠️ The batches REUSE the logit scratch and the P buffer, and nothing in the data flow orders batch b's
    # writes after batch b-1's reads (a write-after-read hazard): under the JIT's whole-block schedule the next
    # batch's GEMM overwrote the logits before the softmax read them (2026-09-25: layer outputs 20 % off the
    # eager ones, a black image). `after` edges on the same buffer are refused as a cycle, so each op is
    # REALIZED in order: the JIT records kernels in execution order and replays them so (still one fused chain).
    for b in range(NH // g.HB):
        S = chk("S%d" % b, gm(QKV, QKV, nslices=DHP // 96, ngroups=g.TP // 96, heads=g.HB, a_stride=g.QA, b_stride=g.KB, c_stride=g.HS,
                              a_off=b * g.HB * g.QA, b_off=NH * g.QA + b * g.HB * g.KB, rowmax=ROWMAX, ktail=S_KTAIL, out=big)).realize()
        Pb = chk("P%d" % b, cs_(K.soft[b], bufs["P"], S, bufs["sums"], K.d_soft)).realize()
        pv = dict(nslices=g.TP // 96, ngroups=DHP // 96, heads=g.HB, a_stride=g.ROWS * g.TP, b_stride=g.VB, c_stride=g.HC, b_off=NH * (g.QA + g.KB) + b * g.HB * g.VB, c_off=b * g.HB * g.HC, out=PV)
        PV = chk("PV%d" % b, OA.gemm_gs(Pb, QKV, ks=KS, ns=NS, nrb=g.NRB, lin48=True, piece=0, **pv) if V48 else gm(Pb, QKV, nlive=PV_NLIVE, **pv)).realize()
    Ao = chk("Ao", cs_(K.gather, bufs["Ao"], PV, bufs["sums"], K.d_gather))
    cal_a("o", Ao, C_)
    Co = q8lin(Ao, C_, ["o"], ["Co"], bufs["C48"]) if Q8 else chk("Co", lin(Ao, pn["o"], C_, C_, bufs["C48"]))
    x1 = chk("x1", cs_(K.resid, bufs["x1"], x, k1, Co, t["sc_o"], K.d_resid))
    Ah2 = chk("Ah2", cs_(K.rms, bufs["Ah2"], x1, ws2, K.d_rms))
    cal_a("w13", Ah2, C_)
    if Q8 and Q8_SWI:     # both halves' int16 partials into `big` ([M_pad][S][w1 | w3]), then SwiGLU reads them (no C tiles)
        A8 = chk("A8_w1", cs_(K.q8[C_], bufs["A8"], Ah2, bufs["SX"], K.d_q8s[C_]).realize())
        for h in range(g.W13S):           # W13S == 2: two channel halves, each its partials in half the buffer, then its SwiGLU slices
            for p in ("w1", "w3"):
                k = p if g.W13S == 1 else "%s_%d" % (p, h)
                aiff = OA.aiff_gm if g.qgm[p] else OA.aiff_lin
                P13 = chk("P_" + k, aiff(big, A8, codes[p], codes["d_" + k], *g.qargs[k]).realize())
            if CAL == "part": cs_(K.pmax["w13"], t["st_w13"], P13, K.d_pmax["w13"]).realize()
            Am = chk("Am" if h == g.W13S - 1 else "Am%d" % h, cs_(K.swi[h], bufs["Am"], big, bufs["SX"], t["cs13_%d" % h], K.d_swi).realize())
    else:
        C13 = q8lin(Ah2, C_, ["w1", "w3"], ["C13a", "C13"], big) if Q8 else chk("C13", lin(Ah2, pn["w13"], C_, N_W13, big))
        Am = chk("Am", cs_(K.swiglu, bufs["Am"], C13, t["sc_w13"], K.d_swiglu))
    cal_a("w2", Am, M_)
    Cm = q8lin(Am, M_, ["w2"], ["Cm"], bufs["C48"]) if Q8 and Q8_W2 else chk("Cm", lin(Am, pn["w2"], M_, C_, bufs["C48"]))
    return chk("out", cs_(K.resid, out, x1, k2, Cm, t["sc_w2"], K.d_resid))


class Branch:
    """One transformer (cond or uncond) over its sequence: the layers, the kernels, the rotary tables, the JITs."""
    def __init__(self, name, cache, weights, g: Geo, pos_ids, treal, layers):
        self.name, self.g, self.treal = name, g, treal
        self.W = Ideogram4Weights(weights); self.tw = R.Model(self.W)
        cos, sin = R.rope_tables(pos_ids)
        pad = lambda a: np.concatenate([a, np.zeros((g.ROWS - a.shape[0], a.shape[1]), np.float32)])
        self.cos, self.sin = cos, sin
        ct_, st_ = V.tile_rows(pad(cos), g.NRB), V.tile_rows(pad(sin), g.NRB)
        # QKV_FAST: qkv_dma's loop-buffer rewrite reads the tables' first halves packed per row group (`rope_csn`) through `cs`
        self.cs, self.sn = (dev(V.rope_csn(ct_, st_)) if QKV_FAST else dev(ct_)), dev(st_)
        self.K = Kernels(g, treal)
        self.layers = [Layer(l, cache, os.path.join(os.path.expanduser(Q8_PACK), name) if Q8 else None) for l in layers]
        self.jits = [None, None]
        self.qx = None

    def q8ctx(self, bufs):
        """The fixed device addresses the AIFF descriptors name (the surfaces, the partials) -- once."""
        if Q8 and self.qx is None:
            ad = lambda t: OA.Device[DEV].raw.dev_addr(t.uop.buffer._buf)
            import ideogram4_q8 as Q
            gm0 = OA.Device[DEV].raw.gm_window()[1]
            self.qx = dict(g=self.g, A8=ad(bufs["A8"]), PQ=ad(bufs["PQ"]), P13=ad(bufs["big"]), D=self.g.qd, ad=ad,
                           gmw={k: Q.gm_slot_bases(self.g.qd[k], gm0, self.g.apl[k]) for k, w, _ in self.g.aparts if self.g.qgm[w]})
        return self.qx

    @staticmethod
    def prep_layer(L: Layer, adaln, pool, qx=None):
        """A layer's host work: the modulation, its four small tables (numpy), and the codes into the layer's pool slot (a plain
        memmove into the mapped device buffer, as `copyin` does). No tinygrad call: `forward` runs this on a worker thread for the
        NEXT layer while the device runs this one (numpy's matmul and ctypes' memmove release the GIL; the slot of parity l % 2 is
        the one the running layer, of the other parity, does not read)."""
        if adaln is None: tabs = ()                                      # NPU_MOD: the tables are made on the device, in the layer's JIT (not None: `pre`)
        else:
            s1, g1, s2, g2 = L.modulation(adaln)
            q4 = lambda a: a.reshape(-1, 8)[:, :4].ravel()
            tabs = (V.dup_quads(L.an1 * s1), V.resid_dma_consts(L.an2, g1, q4(L.sc["o"])), V.dup_quads(L.fn1 * s2), V.resid_dma_consts(L.fn2, g2, q4(L.sc["w2"])))
        slots = pool[L.l % 2]
        for k in (Q8_LIN + ["w2"] if Q8 else W_ORDER):
            src = np.asarray(L.codes[k]); ctypes.memmove(slots[k].uop.buffer._buf.va, src.ctypes.data, src.nbytes)
        if Q8:      # the AIFF parts' descriptors: the template relocated to this slot's weights and descriptor buffer, the layer's shifts
            g = qx["g"]
            for p, wk, _ in g.aparts:
                d = slots["d_" + p]
                po = qx["P13"] if Q8_SWI and wk in ("w1", "w3") else qx["PQ"]
                w = qx["D"][p].build(qx["ad"](d), qx["A8"], qx["gmw"][p] if p in qx["gmw"] else qx["ad"](slots[wk]), po, g.part_sh(L, p))
                ctypes.memmove(d.uop.buffer._buf.va, w.ctypes.data, w.nbytes)
        return tabs

    def run_layer(self, L: Layer, x, adaln, bufs, pool, gate=None, pre=None):
        tc = time.perf_counter()
        mod = getattr(self, "mod", None) if gate is None else None
        if pre is None: OA.Device[DEV].raw.drain()   # host copies into a slot the last (in-flight, ZHOUYI_ASYNC) job may read: one slot, or no prefetch
        tabs = pre if pre is not None else self.prep_layer(L, None if mod is not None else adaln, pool, self.q8ctx(bufs))
        if PROF and pre is None: print("      %s layer %2d: the host work %.0f ms" % (self.name, L.l, (time.perf_counter() - tc) * 1e3), flush=True)
        slots = pool[L.l % 2]
        par = L.l % 2; out = bufs["out%d" % par]
        SN, SL = small_names(self.g), slot_names(self.g); ns = len(SN)
        small = [L.t[k] for k in SN]
        if mod is not None:                                              # NPU_MOD: mod_tables as the JIT's first op, from the layer's C
            if self.jits[par] is None:
                from tinygrad.engine.jit import TinyJit
                g, K, cs, sn, mk, md = self.g, self.K, self.cs, self.sn, mod.k, mod.desc
                def body_m(xx, cm_, kc_, sv_, out_, *rest):
                    tt = dict(zip(SN, rest[:ns]))
                    OA.csrc_call(mk, bufs["ws1"], cm_, kc_, bufs["k1"], bufs["ws2"], bufs["k2"], sv_, md).realize()
                    return block_1024(g, K, tt, cs, sn, xx, bufs["ws1"], bufs["k1"], bufs["ws2"], bufs["k2"], dict(zip(SL, rest[ns:])), bufs, out_).realize()
                self.jits[par] = TinyJit(body_m)
            return self.jits[par](x, L.cm, L.kc, mod.stepv[self.step], out, *small, *[slots[k] for k in SL])
        ws1, k1, ws2, k2 = (dev(t) for t in tabs)
        if gate is not None:
            return block_1024(self.g, self.K, L.t, self.cs, self.sn, x, ws1, k1, ws2, k2, slots, bufs, out, gate=gate).realize()
        if self.jits[par] is None:
            from tinygrad.engine.jit import TinyJit
            g, K, cs, sn = self.g, self.K, self.cs, self.sn
            def body(xx, ws1_, k1_, ws2_, k2_, out_, *rest):
                tt = dict(zip(SN, rest[:ns]))
                return block_1024(g, K, tt, cs, sn, xx, ws1_, k1_, ws2_, k2_, dict(zip(SL, rest[ns:])), bufs, out_).realize()
            self.jits[par] = TinyJit(body)
        return self.jits[par](x, ws1, k1, ws2, k2, out, *small, *[slots[k] for k in SL])

    def forward(self, x_rows, adaln, bufs, pool, gate_layer0=False, dump=None, dev_out=False):
        """x_rows `[treal, 4608]` fp32 (the embedded sequence; or the device rows `[ROWS][4608]` already, NPU_IO) -> the last hidden
        state `[treal, 4608]` (`dev_out`: the device rows)."""
        if isinstance(x_rows, Tensor): h = x_rows; hp = None
        else:
            hp = np.zeros((self.g.ROWS, C_), np.float32); hp[:self.treal] = x_rows
            h = dev(hp.ravel())
        # PREFETCH (default 1): layer l + 1's host work on a worker thread while the device runs layer l (the JIT path only)
        overlap = os.environ.get("PREFETCH", "1") == "1" and self.g.SLOTS == 2 and not (getattr(self, "diag", False) or os.environ.get("OPTIME") or gate_layer0)
        ex = ThreadPoolExecutor(1) if overlap else None
        ada_p = None if getattr(self, "mod", None) is not None else adaln           # NPU_MOD: the prefetch only moves the codes
        fut = ex.submit(self.prep_layer, self.layers[0], ada_p, pool, self.q8ctx(bufs)) if overlap else None
        for li, L in enumerate(self.layers):
            t0 = time.perf_counter()
            if getattr(self, "diag", False):
                fg = FiniteGate("%s layer %d" % (self.name, L.l)); h = self.run_layer(L, h, adaln, bufs, pool, gate=fg); fg.report()
            elif os.environ.get("OPTIME"):
                tm = OpTimer("%s layer %d, ms per op" % (self.name, L.l)); h = self.run_layer(L, h, adaln, bufs, pool, gate=tm); tm.report()
            elif gate_layer0 and L is self.layers[0] and hp is not None:
                gate = Gate1024(self, L, hp, adaln); h = self.run_layer(L, h, adaln, bufs, pool, gate=gate); gate.finish(h)
            elif overlap:
                pre = fut.result()
                # ⚠️ the worker memmoves layer l + 1's codes into slot (l + 1) % 2 -- layer l - 1's slot -- with raw ctypes, which
                # tinygrad's own host writes would drain first: under ZHOUYI_ASYNC (default since dc456c452, 2026-10-06) layer l - 1's
                # last job is still in flight here and would read half-overwritten weights (2026-10-09: step-1 velocity cosine
                # 0.14-0.18 against the Sep 28 run, the Oct 8 grey image). Drain it on this (the submitting) thread first.
                OA.Device[DEV].raw.drain()
                fut = ex.submit(self.prep_layer, self.layers[li + 1], ada_p, pool, self.q8ctx(bufs)) if li + 1 < len(self.layers) else None
                h = self.run_layer(L, h, adaln, bufs, pool, pre=pre)
            else: h = self.run_layer(L, h, adaln, bufs, pool)
            if PROF: h.realize(); print("      %s layer %2d: %.2f s" % (self.name, L.l, time.perf_counter() - t0), flush=True)
            if dump is not None: dump.append(FB.host_read(h).view(np.float32).reshape(self.g.ROWS, C_)[:self.treal].copy())
        if dev_out: return h
        return FB.host_read(h).view(np.float32).reshape(self.g.ROWS, C_)[:self.treal]


class OpTimer:
    """`OPTIME=1`: every op of the block realized and timed alone (eager, one layer at a time) -- where a
    layer's time goes. The first layer pays the compiles; read the second."""
    def __init__(self, tag): self.tag = tag; self.t = time.perf_counter(); self.rows = []
    def check(self, name, tt, fn=None):
        # drained: under ZHOUYI_ASYNC a realize returns with the op's last job in flight, whose time would land on the next op
        # (a layer's last ops on the next layer's first: its weight copies drain first)
        tt.realize(); OA.Device[DEV].raw.drain(); now = time.perf_counter(); self.rows.append((name, now - self.t)); self.t = now; return tt
    def report(self):
        tot = sum(d for _, d in self.rows)
        print("   %s: %.2f s = %s" % (self.tag, tot, ", ".join("%s %.0f" % (n, d * 1e3) for n, d in self.rows)), flush=True)


class FiniteGate:
    """`--diag-step`: every op of every layer realized and read back; the first op whose output holds inf or NaN
    is reported with the layer, and every op's max |value| is logged (fp16 outputs: how close to 65504)."""
    first = None
    def __init__(self, tag): self.tag = tag; self.rows = []
    def check(self, name, tt, fn=None):
        tt = tt.realize()
        raw = FB.host_read(tt); a = raw.view(np.uint16).view(np.float16) if tt.dtype == dtypes.uint16 else raw.view(np.float32)
        fin = np.isfinite(a); bad = int((~fin).sum()); mx = float(np.abs(a[fin]).max()) if fin.any() else float("nan")
        self.rows.append("%s %.3g%s" % (name, mx, (" NONFINITE %d" % bad) if bad else ""))
        if bad and FiniteGate.first is None: FiniteGate.first = "%s op %s: %d non-finite values (finite max |x| %.3g)" % (self.tag, name, bad, mx)
        return tt
    def report(self): print("   %s: %s" % (self.tag, ", ".join(self.rows)), flush=True)


class Gate1024:
    """Layer 0 op by op against numpy from the device's own inputs, on SAMPLED rows (the board's numpy
    has the reference BLAS: a full 4128-row reference GEMM takes minutes): every GEMM on `SR` rows (text
    rows, image rows spread over the sequence, the last real row), head 0 of batch 0 for the attention,
    the vector kernels on those rows through the full references where they are cheap. The layer's input
    and output go to `GATE_DUMP` (npz) for the whole-block fp32 comparison on a workstation (`ideogram4_ref.Block`)."""
    def __init__(self, br: Branch, L: Layer, hp, adaln):
        self.br, self.L, self.hp, self.ada = br, L, hp, adaln; self.bad = []; self.n = 0; self.cache = {}
        T = br.treal
        self.SR = np.unique(np.concatenate([np.arange(min(T, 8)), np.linspace(0, T - 1, 40).astype(int), [T - 1]]))

    def check(self, name, tt, fn):
        tt = tt.realize(); g = self.br.g
        raw = FB.host_read(tt)
        got = raw.view(np.uint16) if tt.dtype == dtypes.uint16 else raw.view(np.float32)
        self.cache[name] = got
        if Q8 and (name.startswith("A8_") or name.startswith("P_") or (name in Q8_DQ and (name != "Cm" or Q8_W2)) or (name.startswith("Am") and Q8_SWI)):
            self._q8_check(name, raw); return tt
        t0 = time.perf_counter(); ref = self._ref(name, got)
        if ref is None: print("   gate %-10s (covered downstream)" % name, flush=True); return tt
        got_, ref_ = ref
        got_, ref_ = np.asarray(got_).ravel()[:ref_.size], np.asarray(ref_).ravel()
        if got_.dtype == np.uint16:
            n, mx = FB._ulps(got_, ref_); big = FB._ulps_over(got_, ref_, 2); ok = big <= max(4, ref_.size // 50000)
            msg = "%d/%d differ (max %d ULP, %d beyond 2)" % (n, ref_.size, mx, big)
        else:
            d = np.abs(got_ - ref_); sc = np.abs(ref_).max() + 1e-30; big = int((d > 1e-4 * sc).sum()); ok = big <= max(4, ref_.size // 50000)
            msg = "max |d| %.3g (max |ref| %.3g), %d beyond 1e-4 relative" % (d.max(), sc, big)
        self.n += 1; self.bad += [] if ok else [name]
        print("   gate %-10s %s   %s   (ref %.1f s)" % (name, "ok " if ok else "BAD", msg, time.perf_counter() - t0), flush=True)
        return tt

    def _q8_check(self, name, raw):
        """The AIFF path's ops, exactly: the quantiser (int8 surfaces and s_x bit for bit, from the device's own fp16 operand), the
        engine (every column of the sampled rows bit for bit against int64 numpy from the device's own surfaces and the packed
        weights), the dequantiser (from the device's own partials, to fp32 rounding)."""
        import ideogram4_q8 as Q
        g, L, t0 = self.br.g, self.L, time.perf_counter()
        if name.startswith("A8_"):
            p = name[3:].split("_")[0]; pl = g.qp[p]; an = {"qkv": "Ah", "o": "Ao", "w1": "Ah2", "w2": "Am"}[p]
            A = self._arows(self.cache[an], pl.K, np.arange(g.ROWS))
            q, sx = Q.q8_ref(A)
            got = raw.view(np.int8)[:pl.a8_bytes].reshape(pl.K // 32, pl.M_pad, 32)
            sxd = FB.host_read(Gate1024.br_bufs["SX"]).view(np.float32)[:g.NRB * 16].reshape(g.NRB, 16)[:, :12].ravel()
            # live rows only: past ROWS the surfaces are never written (the buffer is shared by plans of different row pitch) and the
            # outputs of those rows are never read
            nq = int((got[:, :g.ROWS] != Q.to_surface(q, pl.M_pad)[:, :g.ROWS]).sum()); ns_ = int((sxd != sx).sum())
            self.q8c = dict(q=q, sx=sx, sxd=sxd)
            ok, msg = nq == 0 and ns_ == 0, "int8 %d / %d differ, s_x %d / %d differ" % (nq, got.size, ns_, sx.size)
        elif name.startswith("P_"):
            p = name[2:]; wk = p.split("_")[0]; h = int(p.split("_")[1]) if "_" in p else 0; pl = g.apl[p]; SR = self.SR
            if Q8_SWI and wk in ("w1", "w3"):
                m2 = M_ // g.W13S
                Pd = raw.view(np.int16)[:pl.M_pad * pl.S * 2 * m2].reshape(pl.M_pad, pl.S, 2 * m2)[:, :, (m2 if wk == "w3" else 0):][:, :, :m2]
                self.q8c["P_" + wk] = Pd
            else: Pd = raw.view(np.int16)[:pl.M_pad * pl.S * pl.N].reshape(pl.M_pad, pl.S, pl.N)
            wq = Q.unpack_weights(np.asarray(L.codes[wk]), g.qp[wk])[h * pl.N:(h + 1) * pl.N]
            ref = Q.aiff_ref(self.q8c["q"][SR], wq, g.part_sh(L, p), pl.kc)
            self.q8c["P"] = Pd; self.q8c["p"] = p
            nP = int((Pd[SR] != ref).sum()); sat = int((np.abs(ref.astype(np.int32)) >= 32767).sum())
            ok, msg = nP == 0, "%d / %d sampled outputs differ (%d rows x %d passes x %d), %d at the int16 limit" % (nP, ref.size, len(SR), pl.S, pl.N, sat)
        elif name.startswith("Am"):   # swiglu_i16: fp16(silu(a1) a3) from the device's partials of both parts (a channel part's slices)
            SR = self.SR; sxd = self.q8c["sxd"][SR]; h = int(name[2:]) if name[2:] else g.W13S - 1; m2 = M_ // g.W13S; cc = slice(h * m2, (h + 1) * m2)
            y = Q.swiglu_ref(Q.dq_ref(self.q8c["P_w1"][SR], sxd, L.cs["w1"][:, cc]), Q.dq_ref(self.q8c["P_w3"][SR], sxd, L.cs["w3"][:, cc]))
            got = self._arows(raw.view(np.uint16), M_, SR)[:, cc]
            gb = got.astype(np.float16).view(np.uint16); rb_ = y.astype(np.float16).view(np.uint16)
            n_, mx = FB._ulps(gb.ravel(), rb_.ravel()); big = FB._ulps_over(gb.ravel(), rb_.ravel(), 2)
            ok, msg = big <= max(4, rb_.size // 50000), "%d / %d differ (max %d ULP, %d beyond 2) on the sampled rows" % (n_, rb_.size, mx, big)
        else:
            p = self.q8c["p"]; w = p.split("_")[0]; pl = g.apl[p]; SR = self.SR
            gmul, gofs, ntot = {"qkv": (1, 0, N_QKV), "o": (1, 0, C_), "w1": (2, 0, N_W13), "w3": (2, 1, N_W13), "w2": (1, 0, C_)}[w]
            gofs += g.part_cols(p).start // 96
            y = Q.dq_ref(self.q8c["P"][SR], self.q8c["sxd"][SR], L.cs[w][:, g.part_cols(p)])
            full = self._crows(raw.view(np.float32), ntot, SR).reshape(len(SR), ntot // 96, 96)
            got = full[:, gofs::gmul][:, :pl.N // 96].reshape(len(SR), pl.N)
            d = np.abs(got - y); sc = np.abs(y).max() + 1e-30; big = int((d > 1e-6 * sc).sum())
            ok, msg = big == 0, "max |d| %.3g (max |ref| %.3g), %d beyond 1e-6 relative" % (d.max(), sc, big)
        self.n += 1; self.bad += [] if ok else [name]
        print("   gate %-10s %s   %s   (ref %.1f s)" % (name, "ok " if ok else "BAD", msg, time.perf_counter() - t0), flush=True)

    def _arows(self, h, K, rows):   # A layout (uint16) -> fp32 [len(rows), K]
        g = self.br.g; t = h.reshape(K // 96, g.NRB, KS, 3, 4, 4)
        rb, i, r = rows // 12, rows % 12 // 4, rows % 4
        return np.ascontiguousarray(t[:, rb, :, i, r, :]).reshape(len(rows), K).view(np.float16).astype(np.float32)   # separated advanced indices go first: [rows][slice][kk][k]

    def _b(self, hb, N, K, lin=False):   # B panels (uint16 fp16, or uint8 E4M3 codes) -> fp32 [N, K]; lin: the 48 x 3 order back to the 24 x 6 one first
        if lin and LIN48: hb = np.ascontiguousarray(hb.reshape(2, N // 96, K // 192, 3, 2, 24, 64).transpose(1, 2, 4, 0, 3, 5, 6)).ravel()
        v = G.e4m3_table()[hb].astype(np.float32) if hb.dtype == np.uint8 else hb.view(np.float16).astype(np.float32)
        return np.ascontiguousarray(v.reshape(N // 96, K // 96, NS, KS, 4, 4, 4).transpose(0, 2, 4, 5, 1, 3, 6)).reshape(N, K)

    def _crows(self, ct, N, rows):  # C tiles (fp32) -> [len(rows), N]
        g = self.br.g; t = ct[:(N // 96) * g.NRB * 1152].reshape(N // 96, g.NRB, NS, 3, 4, 4, 4)
        rb, i, r = rows // 12, rows % 12 // 4, rows % 4
        return np.ascontiguousarray(t[:, rb, :, i, :, r, :]).reshape(len(rows), N)          # [rows][group][s][jt][c]

    def _stiles(self, ct):          # a head's logits -> its C tiles (ROWMAX: each (group, row block)'s 192 B of maxima dropped)
        g = self.br.g
        if not ROWMAX: return ct[:g.HS]
        return np.ascontiguousarray(ct[:g.HS].reshape(g.TP // 96, g.NRB, 1200)[:, :, :1152]).ravel()

    def _smax(self, ct):            # ROWMAX: the GEMM's maxima against the max of its own tiles (the same values: bit-exact)
        g = self.br.g; T = self.br.treal; t = ct[:g.HS].reshape(g.TP // 96, g.NRB, 1200)
        got = t[:, :, 1152:].reshape(g.TP // 96, g.NRB, 3, 2, 2, 4)                      # [group][rb][i][h][row of pair][c % 4]
        want = t[:, :, :1152].reshape(g.TP // 96, g.NRB, 6, 3, 4, 2, 2, 4).max((2, 4))   # tiles [s][i][jt][h][row][c] -> [group][rb][i][h][row][c]
        ok = np.array_equal(got, want)
        print("   gate S0 maxima  %s   %d lanes" % ("ok " if ok else "BAD", got.size), flush=True)
        if not ok: self.bad.append("S0max")

    def _gemm(self, name, a_name, K, b_name, N):   # B8: the codes themselves, and the device's C x 2^8
        R_ = self.SR; A = self._arows(self.cache[a_name], K, R_)
        B = self._b(np.asarray(self.L.codes[b_name[7:]])[:N * K] if B8 else self.cache[b_name][:N * K], N, K, lin=True)
        return self._crows(self.cache[name], N, R_) * np.float32(256 if B8 else 1), A @ B.T

    def _ref(self, name, got):
        g, T, c, L = self.br.g, self.br.treal, self.cache, self.L
        q4 = lambda a: a.reshape(-1, 8)[:, :4].ravel()
        s1, g1, s2, g2 = L.modulation(self.ada)
        if name.startswith("panels_"):
            k = name[7:]; return got, G.e4m3_table()[np.asarray(L.codes[k])].astype(np.float16).view(np.uint16)
        if name == "Ah": return got, V.rms_scale_a16_ref(self.hp[:T], L.an1 * s1, np.ones(C_, np.float32), R.EPS, KS, g.NRB)
        if name == "Cqkv": return self._gemm("Cqkv", "Ah", C_, "panels_qkv", N_QKV)
        if name == "QKV":
            cf = V.c_from_tiles(c["Cqkv"][:144 * g.NRB * 1152], NS, g.NRB, N_QKV)[:T] * q4(L.sc["qkv"])
            return got, V.qkv_a16_ref(cf, L.norm_q, L.norm_k, self.br.cos, self.br.sin, 1e-5, NH, DH, DHP, g.NRB, KS, NS, g.TP, v48=V48)[0]
        if name == "S0":                # head 0
            q = c["QKV"]; A = self._arows(q[:g.QA], DHP, self.SR); B = self._b(q[NH * g.QA:NH * g.QA + g.KB], g.TP, DHP)
            if ROWMAX: self._smax(got)
            return self._crows(self._stiles(got), g.TP, self.SR), A @ B.T
        if name == "P0":                # head 0: the unnormalized P on the sampled rows, and all rows' sums
            lg = self._crows(self._stiles(c["S0"]), g.TP, np.arange(g.ROWS)); pr, inv = V.softmax_stream_ref(lg, 1.0 / math.sqrt(DH), T, KS, g.NRB)
            sums = FB.host_read(Gate1024.br_bufs["sums"]).view(np.float32)[:NH * g.NRB * 16].reshape(NH, g.NRB, 16)[0, :, :12].ravel()
            d = np.abs(sums - inv).max() / (np.abs(inv).max() + 1e-30)
            print("   gate sums0      %s   max rel |d| %.3g" % ("ok " if d < 1e-4 else "BAD", d), flush=True)
            if d >= 1e-4: self.bad.append("sums0")
            return got[:g.ROWS * g.TP], pr
        if name == "PV0":
            q = c["QKV"]; o = NH * (g.QA + g.KB)
            A = self._arows(c["P0"][:g.ROWS * g.TP], g.TP, self.SR); B = self._b(q[o:o + g.VB], DHP, g.TP, lin=V48)
            return self._crows(got[:g.HC], DHP, self.SR)[:, :DH], (A @ B.T)[:, :DH]     # the head's DH columns (PV_NLIVE: no others written)
        if name == "Co": return self._gemm("Co", "Ao", C_, "panels_o", C_)
        if name == "C13": return self._gemm("C13", "Ah2", C_, "panels_w13", N_W13)
        if name == "Cm": return self._gemm("Cm", "Am", M_, "panels_w2", C_)
        return None

    def finish(self, h):
        g, T = self.br.g, self.br.treal
        hn = FB.host_read(h).view(np.float32).reshape(g.ROWS, C_)[:T]
        path = os.path.expanduser(os.environ.get("GATE_DUMP", "~/ideogram4/gate_%s_L%d.npz" % (self.br.name, self.L.l)))
        np.savez(path, x=self.hp[:T], out=hn, ada=self.ada, cos=self.br.cos, sin=self.br.sin, layer=self.L.l)
        print("   layer %d (%s): %d ops gated, bad: %s; input/output -> %s (for an fp32 block check on a workstation: ideogram4_ref.Block)"
              % (self.L.l, self.br.name, self.n, self.bad or "none", path), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="ideogram4_text.py's npz (the text rows)")
    ap.add_argument("--cond", default="~/ideogram4/fpcache_cond"); ap.add_argument("--uncond", default="~/ideogram4/fpcache")
    ap.add_argument("--wcond", default="~/ideogram4/transformer"); ap.add_argument("--wuncond", default="~/ideogram4/unconditional_transformer")
    ap.add_argument("--steps", type=int, default=20); ap.add_argument("--size", type=int, default=1024); ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--mu", type=float, default=0.0); ap.add_argument("--std", type=float, default=1.5)
    ap.add_argument("--preset", default=None, choices=("V4_QUALITY_48", "V4_DEFAULT_20", "V4_TURBO_12"),
                    help="Ideogram's published sampler presets: steps, CFG schedule, mu, std")
    ap.add_argument("--guidance", default=None, help="comma list per step (default: 7.0, the last 3 steps 3.0 -- the pipeline's schedule shape)")
    ap.add_argument("--layers", default="0-33"); ap.add_argument("--branch", default="both", choices=("both", "cond", "uncond"))
    ap.add_argument("--out", required=True); ap.add_argument("--save-every", action="store_true", help="save the latents (+ `.step`) after every step")
    ap.add_argument("--resume-step", type=int, default=None, help="continue after step K from the history file `<out>.stepKK.npy`")
    ap.add_argument("--diag-step", type=int, default=None, help="run step K (1-based) EAGERLY with every op's output checked for inf / NaN (FiniteGate)")
    ap.add_argument("--stop-after", type=int, default=None, help="stop after step K")
    ap.add_argument("--resume", action="store_true", help="continue from `--out` and its `.step` (a run cut short: the schedule, seed and guidance must match)")
    ap.add_argument("--q8", default=None, help="W8A8 on the AIFF for qkv, o, w1|w3: the int8 pack root (cond/, uncond/: ideogram4_q8_pack.py); off: today's fp8 path")
    ap.add_argument("--q8-w2", action="store_true", help="with --q8: w2 int8 on the AIFF too (the pack must hold it); default: w2 in fp16 on the TEC")
    ap.add_argument("--q8desc", default="~/ideogram4/q8desc", help="with --q8: the AIFF descriptor templates (ideogram4_aiff_desc.py)")
    ap.add_argument("--calib-stats", default=None, help="save per-step calibration statistics (npz) for the steps in --calib-steps")
    ap.add_argument("--calib-kind", default="act", choices=("act", "part"), help="act: the linears' operands' column |max|; part: the AIFF partials' (--q8)")
    ap.add_argument("--calib-steps", default="0,3,6,9,11")
    ap.add_argument("--calib-from", default=None, help="run only --calib-steps, step i from the latents history <prefix>.step<i>.npy (i > 0; step 0 from the seed)")
    a = ap.parse_args()
    global Q8, Q8_W2, Q8_PACK, Q8_DESC, CAL
    CAL = a.calib_kind if a.calib_stats else None
    assert CAL != "part" or a.q8, "--calib-kind part needs --q8"
    Q8, Q8_W2, Q8_PACK, Q8_DESC = a.q8 is not None, a.q8 is not None and a.q8_w2, a.q8, a.q8desc
    grid = a.size // 16; n_img = grid * grid
    tx = np.load(os.path.expanduser(a.text)); text_rows = tx["rows"].astype(np.float32); n_text = text_rows.shape[0]
    g = Geo(n_img, n_text)
    global W13S, QS; W13S, QS = getattr(g, "W13S", 1), getattr(g, "QS", 1)
    lo, hi = (int(v) for v in a.layers.split("-")); layers = range(lo, hi + 1)
    img_pos = R.image_positions(grid, grid)
    txt_pos = R.text_positions(n_text)
    t0 = time.perf_counter()
    bufs, pool = {}, None                  # the big buffers come after the per-image setup below (its staging buffers are released first)
    br = {}
    if a.branch in ("both", "cond"): br["cond"] = Branch("cond", os.path.expanduser(a.cond), os.path.expanduser(a.wcond), g, np.concatenate([txt_pos, img_pos]), n_text + n_img, layers)
    if a.branch in ("both", "uncond"): br["uncond"] = Branch("uncond", os.path.expanduser(a.uncond), os.path.expanduser(a.wuncond), g, img_pos, n_img, layers)
    PRESETS = {"V4_QUALITY_48": (48, 3, 0.0, 1.5), "V4_DEFAULT_20": (20, 2, 0.0, 1.75), "V4_TURBO_12": (12, 1, 0.5, 1.75)}   # steps, polish steps at 3.0, mu, std
    if a.preset:
        a.steps, npol, a.mu, a.std = PRESETS[a.preset]; gw = [7.0] * (a.steps - npol) + [3.0] * npol
    else:
        gw = [float(v) for v in a.guidance.split(",")] if a.guidance else [7.0] * max(0, a.steps - 3) + [3.0] * min(3, a.steps)
    assert len(gw) == a.steps
    # the schedule of Ideogram's own sampler: model time t_k = LogitNormal(mu_res, std)(k / steps) on float32 intervals, float64
    # math, float32 out; step i goes from t(interval[N - i]) to t(interval[N - i - 1]) -- it ends at t_max = 0.99945, not at 1
    # (ideogram4_ref.sigmas ends at noise level 0, model time 1)
    mu = R.shifted_mu(a.size, a.size, a.mu)
    iv = np.linspace(0.0, 1.0, a.steps + 1, dtype=np.float32).astype(np.float64)
    def sched(u):
        y = mu + a.std * R.normal_quantile(u); t_ = 1.0 - 1.0 / (1.0 + math.exp(-y)) if np.isfinite(y) else (0.0 if y > 0 else 1.0)
        return np.float32(min(max(t_, 1.0 / (1 + math.exp(9.0))), 1.0 / (1 + math.exp(-7.5))))
    tm = [sched(iv[a.steps - k]) for k in range(a.steps + 1)]           # model time, noise -> data
    sig = [1.0 - float(t) for t in tm]
    x = torch_randn(a.seed, n_img * R.C_IN).reshape(n_img, R.C_IN)
    mode = ("W8A8 on the AIFF (qkv, o, w1|w3%s)" % (", w2" if Q8_W2 else "")) if Q8 else "FULL PRECISION"
    print("== Ideogram 4 at %s on the NPU: %r, %dx%d (%d image + %d text tokens -> %d rows), %d steps, seed %d, CFG %s; "
          "branches %s, layers %d-%d; set up in %.0f s ==" % (mode, str(tx["text"]).split("\n")[1] if "text" in tx else "?", a.size, a.size, n_img, n_text,
          g.ROWS, a.steps, a.seed, gw, list(br), lo, hi, time.perf_counter() - t0), flush=True)
    T0 = time.perf_counter(); start = 0
    if a.resume_step is not None:
        start = a.resume_step; x = np.load(os.path.expanduser(a.out)[:-4] + ".step%02d.npy" % start).astype(np.float32)
        print("   resuming after step %d of %d from the history" % (start, a.steps), flush=True)
    elif a.resume and os.path.exists(os.path.expanduser(a.out) + ".step"):
        start = int(open(os.path.expanduser(a.out) + ".step").read()); x = np.load(os.path.expanduser(a.out)).astype(np.float32)
        print("   resuming after step %d of %d from %s" % (start, a.steps, a.out), flush=True)
    # NPU_IO (default 1, 2026-09-28): the input projection and the final layer on the NPU (ideogram4_io_npu.py) -- in numpy on the
    # board's reference BLAS they were 1.8 + 2.0 s a branch, ~7.6 s a step; the rows stay on the device (layer 0 reads out1)
    npu_io = os.environ.get("NPU_IO", "1") == "1" and not GATE
    if npu_io:
        import ideogram4_io_npu as IOM
        for name, b in br.items(): b.io = IOM.IOProj(b.tw, g.ROWS, n_img, text_rows if name == "cond" else None)
    # every step's AdaLN input (t-embedding + 3 mat-vecs a branch, ~36 ms on the board's reference BLAS) depends only on the model time,
    # all known now: one worker computes them while the device runs step 1 (2026-09-28; the same arithmetic: bit-identical)
    # NPU_TEMB (default 1 with NPU_MOD / NPU_STEP): the t-embedding MLP on the device too (TembNPU) -- the host AdaLN inputs only
    # for the diagnostic modes that still take the host tables
    npu_temb = npu_io and os.environ.get("NPU_MOD", "1") == "1" and os.environ.get("NPU_STEP", "1") == "1" and os.environ.get("NPU_TEMB", "1") == "1" \
        and a.diag_step is None and not os.environ.get("OPTIME")
    ada_ex = ThreadPoolExecutor(1)
    ada_fut = {} if npu_temb else {(i_, name): ada_ex.submit(b.tw.cond, float(tm[i_])) for i_ in range(start, a.steps) for name, b in br.items()}
    # NPU_MOD (default 1, 2026-09-28): every layer's modulation for every step as one device GEMM a layer (ideogram4_io_npu.ModNPU),
    # the per-step tables made by `mod_tables` inside the layer's JIT -- no per-layer numpy mat-vec, tables or uploads
    if npu_io and os.environ.get("NPU_MOD", "1") == "1":
        tm0 = time.perf_counter()
        for k_, n_ in (("ws1", 9216), ("k1", 27648), ("ws2", 9216), ("k2", 27648)): bufs[k_] = zeros_r(n_, dtypes.float32)
        for name, b in br.items():
            b.mod = IOM.ModNPU(b.layers, a.steps)
            if npu_temb:
                b.temb = IOM.TembNPU(b.tw, a.steps); b.temb.run([tm[i_] for i_ in range(a.steps)], b.mod.a); b.mod.setup()
                b.io.use_temb(b.temb, b.mod.stepv)
            else: b.mod.setup([ada_fut[(i_, name)].result() if i_ >= start else b.tw.cond(float(tm[i_])) for i_ in range(a.steps)])
        print("   NPU_MOD: every layer's modulation for %d steps on the device in %.1f s" % (a.steps, time.perf_counter() - tm0), flush=True)
        OA.Device[DEV].raw.drain()          # the setup's last job may still be in flight (ZHOUYI_ASYNC): drain before its buffers go
        for b in br.values():               # setup-only staging (the t-embedding weights' and the modulation GEMM's): ~117 MB a branch
            b.mod.stage = b.mod.pan = None
            if getattr(b, "temb", None) is not None: b.temb.stage = None
        import gc; gc.collect()
    bufs.update(make_bufs(g)); pool = make_pool(g)
    # NPU_STEP (default 1 with NPU_IO, 2026-09-28): the latents stay on the device -- lat_pack, the text / pad rows from templates,
    # the velocity as device rows, guidance + Euler by cfg_euler; the host reads the 2 MB of latents back for the log line only
    S = None
    if npu_io and os.environ.get("NPU_STEP", "1") == "1":
        if not npu_temb:
            for name, b in br.items(): b.io.set_scales([ada_fut[(i_, name)].result() if i_ >= start else b.tw.cond(float(tm[i_])) for i_ in range(a.steps)])
        S = IOM.Sampler(x, br["cond"].io if "cond" in br else br["uncond"].io, br["uncond"].io if len(br) == 2 else None, gw, tm)
    csteps = [int(v) for v in a.calib_steps.split(",")]
    cstats: dict = {}
    def cal_collect(i):
        """read every layer's statistics buffers (verified host reads), file them under step i, zero them for the next step"""
        raw = OA.Device[DEV].raw; raw.drain()
        for name, b in br.items():
            for L in b.layers:
                for k, n in cal_widths().items():
                    t_ = L.t["st_" + k]
                    if i in csteps: cstats.setdefault("%s_%d_%s" % (name, L.l, k), []).append(FB.host_read(t_).view(np.float32)[:n].copy())
                    raw.drain(); ctypes.memset(t_.uop.buffer._buf.va, 0, t_.nbytes())
        if i in csteps:
            np.savez(os.path.expanduser(a.calib_stats), steps=np.array([s_ for s_ in csteps if s_ <= i]), **{k: np.stack(v) for k, v in cstats.items()})
    steps_run = range(start, a.steps) if not a.calib_from else csteps
    x0 = x
    for i in steps_run:
        if a.calib_from:      # this step from the saved trajectory: the sampler's latents set on the device (after the in-flight job)
            x = x0 if i == 0 else np.load(os.path.expanduser(a.calib_from) + ".step%02d.npy" % i).astype(np.float32)
            if S is not None:
                OA.Device[DEV].raw.drain(); xp = np.zeros(S.x.numel(), np.float32); xp[:x.size] = x.ravel()
                ctypes.memmove(S.x.uop.buffer._buf.va, xp.ctypes.data, xp.nbytes)
        ts = time.perf_counter()
        v = {}
        for name, b in br.items():
            b.diag = a.diag_step is not None and i + 1 == a.diag_step
            ada = ada_fut.pop((i, name)).result() if ada_fut else None
            Gate1024.br_bufs = bufs
            if npu_io:
                b.step = i
                hin = bufs["out%d" % (1 - lo % 2)]                               # the rows buffer the first layer does NOT write
                if S is not None:
                    b.io.final_dev(b.forward(b.io.embed_dev(S.x, hin, bufs["big"]), ada, bufs, pool, dev_out=True), i, bufs["Ah"])
                    continue
                hd = b.forward(b.io.embed(x, hin, bufs["big"]), ada, bufs, pool, dev_out=True)
                v[name] = b.io.final(hd, ada, bufs["Ah"])
                continue
            img = b.tw.embed(x)
            rows = np.concatenate([text_rows, img]) if name == "cond" else img
            hn = b.forward(rows, ada, bufs, pool, gate_layer0=GATE and i == 0)
            v[name] = b.tw.out(hn[-n_img:], ada).astype(np.float32)
        if S is not None:
            xp = x; S.step(i); x = S.read()                                        # z += v * (s - t), on the device
            vel = (x - xp) / np.float32(tm[i + 1] - tm[i])                          # (the log's |v|, from the two latents)
        else:
            if len(v) == 2: vel = np.float32(gw[i]) * v["cond"] + np.float32(1.0 - gw[i]) * v["uncond"]
            else: vel = next(iter(v.values()))
            x = (x + vel * np.float32(tm[i + 1] - tm[i])).astype(np.float32)          # z += v * (s - t)
        el = time.perf_counter() - ts
        print("   step %2d/%d sigma %.4f -> %.4f  guidance %.1f  %.1f s (total %.1f min)  |x| %.3f |v| %.3f" % (
            i + 1, a.steps, sig[i], sig[i + 1], gw[i], el, (time.perf_counter() - T0) / 60, np.abs(x).mean(), np.abs(vel).mean()), flush=True)
        if not np.isfinite(vel).all():
            print("   step %d: NON-FINITE velocity -- cond %s, uncond %s; first non-finite op: %s" % (i + 1, *(str(np.isfinite(v.get(k, np.zeros(1))).all()) for k in ("cond", "uncond")), FiniteGate.first), flush=True)
        if a.save_every:
            o = os.path.expanduser(a.out); np.save(o + ".tmp.npy", x); os.replace(o + ".tmp.npy", o)
            np.save(o[:-4] + ".step%02d.npy" % (i + 1), x)                        # the history: a bad step never destroys the last good one
            with open(o + ".step", "w") as f: f.write(str(i + 1))
        if CAL is not None: cal_collect(i)
        if a.calib_from:      # the step's output beside the run's --out (a one-step error against the trajectory's next latents)
            np.save(os.path.expanduser(a.out)[:-4] + ".from%02d.npy" % i, x); continue
        if a.stop_after is not None and i + 1 >= a.stop_after: print("   stopping after step %d" % (i + 1), flush=True); return
    np.save(os.path.expanduser(a.out), x)
    print("   latents -> %s  (%.1f min for %d steps)" % (a.out, (time.perf_counter() - T0) / 60, a.steps), flush=True)


if __name__ == "__main__":
    main()

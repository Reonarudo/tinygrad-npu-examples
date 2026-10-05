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

DEV = "ZHOUYI"; NT = 12; KS, NS = 24, 6; PIECE = 8; HB = 6
ROWMAX = os.environ.get("ROWMAX", "1") == "1"               # the scores' GEMM writes each row block's row maxima after its C row (k_gemm_gs(rowmax=True)): the softmax skips its scan pass; ROWMAX=0: the scan
LIN48 = os.environ.get("LIN48", "1") == "1"
V48 = LIN48 and os.environ.get("V48", "1") == "1"        # P V on the 48-deep 3-strip kernel (V^T written in its panel order by qkv_dma; the keys padded to a multiple of 192)
QKV_FAST = os.environ.get("QKV_FAST", "1") == "1"         # qkv_dma(fast=True): loop-buffer bodies, the scale folded in, the half rotary tables double-buffered (bit-identical)
B8 = LIN48 and os.environ.get("B8", "1") == "1"            # the linears read the E4M3 codes and expand them in LSRAM (k_gemm_gs b8): half the weight bytes, no fp16 panels pass; C x 2^-8, the scales x 2^8                 # the linears on the 48-deep 3-strip k_gemm_gs (weights repacked, `p48/`); LIN48=0: the 24 x 6 kernel
GATE = bool(int(os.environ.get("GATE", "0"))); PROF = bool(int(os.environ.get("PROF", "0")))
C_, NH, DH, DHP, M_ = R.HID, R.NH, R.HEAD_DIM, 288, R.MLP
N_QKV, N_W13 = 3 * NH * DH, 2 * M_
W_ORDER = ["qkv", "o", "w13", "w2"]
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
        self.NRB = self.ROWS // 12; self.TP = -(-self.ROWS // 192) * 192 if V48 else self.ROWS   # the keys (N of the scores, K of P V)
        self.QA, self.KB, self.VB, self.QKV_TOTAL = V.qkv_a16_sizes(NH, DH, DHP, self.NRB, KS, NS, self.TP)
        self.HC = 3 * self.NRB * 1152                         # floats per head of the P V tiles (N = 288)
        self.HS = (self.TP // 96) * self.NRB * (1200 if ROWMAX else 1152)   # floats per head of the logit tiles (N = TP; + the row maxima)
        self.C48 = 48 * self.NRB * 1152                       # a linear with N = 4608


class Kernels:
    """The vector kernels for one sequence (`treal` real rows): registered once per transformer branch."""
    def __init__(self, g: Geo, treal: int):
        reg = lambda name, src: OA.register_csrc(name, src, ntasks=NT)
        self.treal = treal
        self.rms = reg("rms_scale_dma", V.rms_scale_dma_src(treal, C_, R.NORM_EPS, KS, g.NRB, nt=NT)); self.d_rms = dev(V.rms_scale_dma_descs(C_, g.NRB))
        self.resid = reg("resid_dma", V.resid_dma_src(treal, C_, NS, g.NRB, R.NORM_EPS, nt=NT)); self.d_resid = dev(V.resid_dma_descs(C_, g.NRB))
        self.swiglu = reg("swiglu_dma", V.swiglu_dma_src(treal, M_, g.NRB, KS, nt=NT, interleaved=True)); self.d_swiglu = dev(V.swiglu_dma_descs(M_, g.NRB, KS, interleaved=True))
        self.qkv = reg("qkv_dma", V.qkv_dma_src(treal, NH, DH, DHP, g.NRB, KS, NS, g.TP, 1e-5, nt=NT, v48=V48, fast=QKV_FAST)); self.d_qkv = dev(V.qkv_dma_descs(g.NRB, KS, NS))
        SGB = int(os.environ.get("SOFTMAX_GB", "2" if (g.TP // 96) % 2 == 0 else "1"))   # groups per softmax DMA request (1: the single-group kernel)
        self.soft = [reg("softmax_stream", V.softmax_stream_src(treal, HB, g.TP, g.NRB, KS, 1.0 / math.sqrt(DH), hb=b * HB, nt=NT, gb=SGB, rowmax=ROWMAX)) for b in range(NH // HB)]
        self.d_soft = dev(V.softmax_stream_descs(g.NRB, SGB, rowmax=g.TP // 96 if ROWMAX else 0))
        self.gather = reg("gather_o_sum", V.gather_o_sum_src(treal, NH, DH, DHP, NS, g.NRB, KS, nt=NT)); self.d_gather = dev(V.gather_o_sum_descs(g.NRB))


def zeros_r(n, dt):
    """A zeroed device buffer rounded up to 12 x 8 KiB (the verified host read's checksum granule)."""
    per = 98304 // dt.itemsize
    return zeros(-(-n // per) * per, dt)


def make_bufs(g: Geo):
    f32, u16 = dtypes.float32, dtypes.uint16
    zeros = zeros_r
    big = max(144 * g.NRB * 1152, HB * g.HS, 256 * g.NRB * 1152)
    b = dict(Ah=zeros(g.ROWS * C_, u16), big=zeros(big, f32), QKV=zeros(g.QKV_TOTAL, u16), P=zeros(HB * g.ROWS * g.TP, u16),
             PV=zeros(NH * g.HC, f32), sums=zeros(NH * g.NRB * 16, f32), Ao=zeros(g.ROWS * C_, u16), C48=zeros(g.C48, f32),
             x1=zeros(g.ROWS * C_, f32), Ah2=zeros(g.ROWS * C_, u16), Am=zeros(g.ROWS * M_, u16),
             out0=zeros(g.ROWS * C_, f32), out1=zeros(g.ROWS * C_, f32))
    if not B8: b["panels"] = dict(qkv=zeros(N_QKV * C_, u16), o=zeros(C_ * C_, u16), w13=zeros(N_W13 * C_, u16), w2=zeros(C_ * M_, u16))
    return b


def make_pool():
    mk = lambda: dict(qkv=zeros(N_QKV * C_, dtypes.uint8), o=zeros(C_ * C_, dtypes.uint8), w13=zeros(N_W13 * C_, dtypes.uint8), w2=zeros(C_ * M_, dtypes.uint8))
    return [mk(), mk()]


class Layer:
    """One layer of one transformer: the packed codes (memory-mapped), the small tensors, the modulation."""
    def __init__(self, l, cache):
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
        self.t = dict(sc_qkv=dev(self.sc["qkv"]), sc_o=dev(self.sc["o"]), sc_w13=dev(self.sc["w13"]), sc_w2=dev(self.sc["w2"]),
                      nq2=dev(V.dup_quads(self._sm["norm_q"])), nk2=dev(V.dup_quads(self._sm["norm_k"])))

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
    lin = lambda a_, b_, K, N, out: OA.gemm_gs(a_, b_, ks=KS, ns=NS, nrb=g.NRB, nslices=K // 96, ngroups=N // 96, lin48=LIN48, piece=0 if LIN48 else PIECE, b8=B8, out=out)
    Cq = chk("Cqkv", lin(Ah, pn["qkv"], C_, N_QKV, big))
    QKV = chk("QKV", cs_(K.qkv, bufs["QKV"], Cq, t["nq2"], t["nk2"], cs, sn, t["sc_qkv"], K.d_qkv))
    PV = bufs["PV"]
    # ⚠️ The batches REUSE the logit scratch and the P buffer, and nothing in the data flow orders batch b's
    # writes after batch b-1's reads (a write-after-read hazard): under the JIT's whole-block schedule the next
    # batch's GEMM overwrote the logits before the softmax read them (2026-09-25: layer outputs 20 % off the
    # eager ones, a black image). `after` edges on the same buffer are refused as a cycle, so each op is
    # REALIZED in order: the JIT records kernels in execution order and replays them so (still one fused chain).
    for b in range(NH // HB):
        S = chk("S%d" % b, gm(QKV, QKV, nslices=DHP // 96, ngroups=g.TP // 96, heads=HB, a_stride=g.QA, b_stride=g.KB, c_stride=g.HS,
                              a_off=b * HB * g.QA, b_off=NH * g.QA + b * HB * g.KB, rowmax=ROWMAX, out=big)).realize()
        Pb = chk("P%d" % b, cs_(K.soft[b], bufs["P"], S, bufs["sums"], K.d_soft)).realize()
        pv = dict(nslices=g.TP // 96, ngroups=DHP // 96, heads=HB, a_stride=g.ROWS * g.TP, b_stride=g.VB, c_stride=g.HC, b_off=NH * (g.QA + g.KB) + b * HB * g.VB, c_off=b * HB * g.HC, out=PV)
        PV = chk("PV%d" % b, OA.gemm_gs(Pb, QKV, ks=KS, ns=NS, nrb=g.NRB, lin48=True, piece=0, **pv) if V48 else gm(Pb, QKV, **pv)).realize()
    Ao = chk("Ao", cs_(K.gather, bufs["Ao"], PV, bufs["sums"], K.d_gather))
    Co = chk("Co", lin(Ao, pn["o"], C_, C_, bufs["C48"]))
    x1 = chk("x1", cs_(K.resid, bufs["x1"], x, k1, Co, t["sc_o"], K.d_resid))
    Ah2 = chk("Ah2", cs_(K.rms, bufs["Ah2"], x1, ws2, K.d_rms))
    C13 = chk("C13", lin(Ah2, pn["w13"], C_, N_W13, big))
    Am = chk("Am", cs_(K.swiglu, bufs["Am"], C13, t["sc_w13"], K.d_swiglu))
    Cm = chk("Cm", lin(Am, pn["w2"], M_, C_, bufs["C48"]))
    return chk("out", cs_(K.resid, out, x1, k2, Cm, t["sc_w2"], K.d_resid))


class Branch:
    """One transformer (cond or uncond) over its sequence: the layers, the kernels, the rotary tables, the JITs."""
    def __init__(self, name, cache, weights, g: Geo, pos_ids, treal, layers):
        self.name, self.g, self.treal = name, g, treal
        self.W = Ideogram4Weights(weights); self.tw = R.TopWeights(self.W)
        cos, sin = R.mrope(pos_ids)
        pad = lambda a: np.concatenate([a, np.zeros((g.ROWS - a.shape[0], a.shape[1]), np.float32)])
        self.cos, self.sin = cos, sin
        ct_, st_ = V.tile_rows(pad(cos), g.NRB), V.tile_rows(pad(sin), g.NRB)
        # QKV_FAST: qkv_dma's loop-buffer rewrite reads the tables' first halves packed per row group (`rope_csn`) through `cs`
        self.cs, self.sn = (dev(V.rope_csn(ct_, st_)) if QKV_FAST else dev(ct_)), dev(st_)
        self.K = Kernels(g, treal)
        self.layers = [Layer(l, cache) for l in layers]
        self.jits = [None, None]

    @staticmethod
    def prep_layer(L: Layer, adaln, pool):
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
        for k in W_ORDER:
            src = np.asarray(L.codes[k]); ctypes.memmove(slots[k].uop.buffer._buf.va, src.ctypes.data, src.nbytes)
        return tabs

    def run_layer(self, L: Layer, x, adaln, bufs, pool, gate=None, pre=None):
        tc = time.perf_counter()
        mod = getattr(self, "mod", None) if gate is None else None
        tabs = pre if pre is not None else self.prep_layer(L, None if mod is not None else adaln, pool)
        if PROF and pre is None: print("      %s layer %2d: the host work %.0f ms" % (self.name, L.l, (time.perf_counter() - tc) * 1e3), flush=True)
        slots = pool[L.l % 2]
        par = L.l % 2; out = bufs["out%d" % par]
        small = [L.t[k] for k in ("sc_qkv", "sc_o", "sc_w13", "sc_w2", "nq2", "nk2")]
        if mod is not None:                                              # NPU_MOD: mod_tables as the JIT's first op, from the layer's C
            if self.jits[par] is None:
                from tinygrad.engine.jit import TinyJit
                g, K, cs, sn, mk, md = self.g, self.K, self.cs, self.sn, mod.k, mod.desc
                def body_m(xx, cm_, kc_, sv_, out_, *rest):
                    tt = dict(zip(("sc_qkv", "sc_o", "sc_w13", "sc_w2", "nq2", "nk2"), rest[:6]))
                    OA.csrc_call(mk, bufs["ws1"], cm_, kc_, bufs["k1"], bufs["ws2"], bufs["k2"], sv_, md).realize()
                    return block_1024(g, K, tt, cs, sn, xx, bufs["ws1"], bufs["k1"], bufs["ws2"], bufs["k2"], dict(zip(W_ORDER, rest[6:10])), bufs, out_).realize()
                self.jits[par] = TinyJit(body_m)
            return self.jits[par](x, L.cm, L.kc, mod.stepv[self.step], out, *small, *[slots[k] for k in W_ORDER])
        ws1, k1, ws2, k2 = (dev(t) for t in tabs)
        if gate is not None:
            return block_1024(self.g, self.K, L.t, self.cs, self.sn, x, ws1, k1, ws2, k2, slots, bufs, out, gate=gate).realize()
        if self.jits[par] is None:
            from tinygrad.engine.jit import TinyJit
            g, K, cs, sn = self.g, self.K, self.cs, self.sn
            def body(xx, ws1_, k1_, ws2_, k2_, out_, *rest):
                tt = dict(zip(("sc_qkv", "sc_o", "sc_w13", "sc_w2", "nq2", "nk2"), rest[:6]))
                return block_1024(g, K, tt, cs, sn, xx, ws1_, k1_, ws2_, k2_, dict(zip(W_ORDER, rest[6:10])), bufs, out_).realize()
            self.jits[par] = TinyJit(body)
        return self.jits[par](x, ws1, k1, ws2, k2, out, *small, *[slots[k] for k in W_ORDER])

    def forward(self, x_rows, adaln, bufs, pool, gate_layer0=False, dump=None, dev_out=False):
        """x_rows `[treal, 4608]` fp32 (the embedded sequence; or the device rows `[ROWS][4608]` already, NPU_IO) -> the last hidden
        state `[treal, 4608]` (`dev_out`: the device rows)."""
        if isinstance(x_rows, Tensor): h = x_rows; hp = None
        else:
            hp = np.zeros((self.g.ROWS, C_), np.float32); hp[:self.treal] = x_rows
            h = dev(hp.ravel())
        # PREFETCH (default 1): layer l + 1's host work on a worker thread while the device runs layer l (the JIT path only)
        overlap = os.environ.get("PREFETCH", "1") == "1" and not (getattr(self, "diag", False) or os.environ.get("OPTIME") or gate_layer0)
        ex = ThreadPoolExecutor(1) if overlap else None
        ada_p = None if getattr(self, "mod", None) is not None else adaln           # NPU_MOD: the prefetch only moves the codes
        fut = ex.submit(self.prep_layer, self.layers[0], ada_p, pool) if overlap else None
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
                fut = ex.submit(self.prep_layer, self.layers[li + 1], ada_p, pool) if li + 1 < len(self.layers) else None
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
        tt.realize(); now = time.perf_counter(); self.rows.append((name, now - self.t)); self.t = now; return tt
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
    and output go to `GATE_DUMP` (npz) for the whole-block fp32 comparison on a workstation (`ideogram4_ref.block`)."""
    def __init__(self, br: Branch, L: Layer, hp, adaln):
        self.br, self.L, self.hp, self.ada = br, L, hp, adaln; self.bad = []; self.n = 0; self.cache = {}
        T = br.treal
        self.SR = np.unique(np.concatenate([np.arange(min(T, 8)), np.linspace(0, T - 1, 40).astype(int), [T - 1]]))

    def check(self, name, tt, fn):
        tt = tt.realize(); g = self.br.g
        raw = FB.host_read(tt)
        got = raw.view(np.uint16) if tt.dtype == dtypes.uint16 else raw.view(np.float32)
        self.cache[name] = got
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
        if name == "Ah": return got, V.rms_scale_a16_ref(self.hp[:T], L.an1 * s1, np.ones(C_, np.float32), R.NORM_EPS, KS, g.NRB)
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
            return self._crows(got[:g.HC], DHP, self.SR), A @ B.T
        if name == "Co": return self._gemm("Co", "Ao", C_, "panels_o", C_)
        if name == "C13": return self._gemm("C13", "Ah2", C_, "panels_w13", N_W13)
        if name == "Cm": return self._gemm("Cm", "Am", M_, "panels_w2", C_)
        return None

    def finish(self, h):
        g, T = self.br.g, self.br.treal
        hn = FB.host_read(h).view(np.float32).reshape(g.ROWS, C_)[:T]
        path = os.environ.get("GATE_DUMP", "~/ideogram4/gate_%s_L%d.npz" % (self.br.name, self.L.l))
        np.savez(path, x=self.hp[:T], out=hn, ada=self.ada, cos=self.br.cos, sin=self.br.sin, layer=self.L.l)
        print("   layer %d (%s): %d ops gated, bad: %s; input/output -> %s (the fp32 block check: ideogram4_layer_check.py on the Mac)"
              % (self.L.l, self.br.name, self.n, self.bad or "none", path), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="ideogram4_text.py's npz (the text rows)")
    ap.add_argument("--cond", default="~/ideogram4/fpcache_cond"); ap.add_argument("--uncond", default="~/ideogram4/fpcache")
    ap.add_argument("--wcond", default="~/ideogram4/transformer"); ap.add_argument("--wuncond", default="~/ideogram4/unconditional_transformer")
    ap.add_argument("--steps", type=int, default=20); ap.add_argument("--size", type=int, default=1024); ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--mu", type=float, default=0.0); ap.add_argument("--std", type=float, default=1.5)
    ap.add_argument("--preset", default=None, choices=("V4_QUALITY_48", "V4_DEFAULT_20", "V4_TURBO_12"),
                    help="the official sampler presets (ideogram4/sampler_configs.py): steps, CFG schedule, mu, std")
    ap.add_argument("--guidance", default=None, help="comma list per step (default: 7.0, the last 3 steps 3.0 -- the pipeline's schedule shape)")
    ap.add_argument("--layers", default="0-33"); ap.add_argument("--branch", default="both", choices=("both", "cond", "uncond"))
    ap.add_argument("--out", required=True); ap.add_argument("--save-every", action="store_true", help="save the latents (+ `.step`) after every step")
    ap.add_argument("--resume-step", type=int, default=None, help="continue after step K from the history file `<out>.stepKK.npy`")
    ap.add_argument("--diag-step", type=int, default=None, help="run step K (1-based) EAGERLY with every op's output checked for inf / NaN (FiniteGate)")
    ap.add_argument("--stop-after", type=int, default=None, help="stop after step K")
    ap.add_argument("--resume", action="store_true", help="continue from `--out` and its `.step` (a run cut short: the schedule, seed and guidance must match)")
    a = ap.parse_args()
    grid = a.size // 16; n_img = grid * grid
    tx = np.load(os.path.expanduser(a.text)); text_rows = tx["rows"].astype(np.float32); n_text = text_rows.shape[0]
    g = Geo(n_img, n_text)
    lo, hi = (int(v) for v in a.layers.split("-")); layers = range(lo, hi + 1)
    img_pos = R.image_position_ids(grid, grid)
    txt_pos = np.repeat(np.arange(n_text)[:, None], 3, 1)
    t0 = time.perf_counter()
    bufs = make_bufs(g); pool = make_pool()
    br = {}
    if a.branch in ("both", "cond"): br["cond"] = Branch("cond", a.cond, os.path.expanduser(a.wcond), g, np.concatenate([txt_pos, img_pos]), n_text + n_img, layers)
    if a.branch in ("both", "uncond"): br["uncond"] = Branch("uncond", os.path.expanduser(a.uncond), os.path.expanduser(a.wuncond), g, img_pos, n_img, layers)
    PRESETS = {"V4_QUALITY_48": (48, 3, 0.0, 1.5), "V4_DEFAULT_20": (20, 2, 0.0, 1.75), "V4_TURBO_12": (12, 1, 0.5, 1.75)}   # steps, polish steps at 3.0, mu, std
    if a.preset:
        a.steps, npol, a.mu, a.std = PRESETS[a.preset]; gw = [7.0] * (a.steps - npol) + [3.0] * npol
    else:
        gw = [float(v) for v in a.guidance.split(",")] if a.guidance else [7.0] * max(0, a.steps - 3) + [3.0] * min(3, a.steps)
    assert len(gw) == a.steps
    # the official schedule (ideogram4/scheduler.py): model time t_k = LogitNormal(mu_res, std)(k / steps) on float32
    # intervals, float64 math, float32 out; step i goes from t(interval[N - i]) to t(interval[N - i - 1]) -- it ends at
    # t_max = 0.99945, not at 1 (the diffusers port's terminal sigma 0)
    mu = R.resolution_mu(a.size, a.size, a.mu)
    iv = np.linspace(0.0, 1.0, a.steps + 1, dtype=np.float32).astype(np.float64)
    def sched(u):
        y = mu + a.std * R._ndtri(np.array([u]))[0]; t_ = 1.0 - 1.0 / (1.0 + math.exp(-y)) if np.isfinite(y) else (0.0 if y > 0 else 1.0)
        return np.float32(min(max(t_, 1.0 / (1 + math.exp(9.0))), 1.0 / (1 + math.exp(-7.5))))
    tm = [sched(iv[a.steps - k]) for k in range(a.steps + 1)]           # model time, noise -> data
    sig = [1.0 - float(t) for t in tm]
    x = torch_randn(a.seed, n_img * R.IN_CH).reshape(n_img, R.IN_CH)
    print("== Ideogram 4 at FULL PRECISION on the NPU: %r, %dx%d (%d image + %d text tokens -> %d rows), %d steps, seed %d, CFG %s; "
          "branches %s, layers %d-%d; set up in %.0f s ==" % (str(tx["text"]).split("\n")[1] if "text" in tx else "?", a.size, a.size, n_img, n_text,
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
    ada_fut = {} if npu_temb else {(i_, name): ada_ex.submit(R.adaln_input, b.tw, float(tm[i_])) for i_ in range(start, a.steps) for name, b in br.items()}
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
            else: b.mod.setup([ada_fut[(i_, name)].result() if i_ >= start else R.adaln_input(b.tw, float(tm[i_])) for i_ in range(a.steps)])
        print("   NPU_MOD: every layer's modulation for %d steps on the device in %.1f s" % (a.steps, time.perf_counter() - tm0), flush=True)
    # NPU_STEP (default 1 with NPU_IO, 2026-09-28): the latents stay on the device -- lat_pack, the text / pad rows from templates,
    # the velocity as device rows, guidance + Euler by cfg_euler; the host reads the 2 MB of latents back for the log line only
    S = None
    if npu_io and os.environ.get("NPU_STEP", "1") == "1":
        if not npu_temb:
            for name, b in br.items(): b.io.set_scales([ada_fut[(i_, name)].result() if i_ >= start else R.adaln_input(b.tw, float(tm[i_])) for i_ in range(a.steps)])
        S = IOM.Sampler(x, br["cond"].io if "cond" in br else br["uncond"].io, br["uncond"].io if len(br) == 2 else None, gw, tm)
    for i in range(start, a.steps):
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
            img = (x @ b.tw.input_w.T + b.tw.input_b + b.tw.ind[1]).astype(np.float32)
            rows = np.concatenate([text_rows, img]) if name == "cond" else img
            hn = b.forward(rows, ada, bufs, pool, gate_layer0=GATE and i == 0)
            v[name] = R.final_layer(b.tw, hn[-n_img:], ada).astype(np.float32)
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
        if a.stop_after is not None and i + 1 >= a.stop_after: print("   stopping after step %d" % (i + 1), flush=True); return
    np.save(os.path.expanduser(a.out), x)
    print("   latents -> %s  (%.1f min for %d steps)" % (a.out, (time.perf_counter() - T0) / 60, a.steps), flush=True)


if __name__ == "__main__":
    main()

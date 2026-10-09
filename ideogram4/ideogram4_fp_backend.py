#!/usr/bin/env python3
"""Ideogram 4 at FULL PRECISION on the NPU: the unconditional transformer's 34 blocks with the E4M3
weights unpacked exactly to fp16 on the device (`ops_zhouyi.e4m3_stream`), the five linears and the
attention on the TEC's matrix unit in fp16 with fp32 accumulation (`ops_zhouyi.gemm_gs`), and the
block's other math -- RMSNorm + AdaLN scale, q/k norm + MRoPE, softmax, SwiGLU, the gated residual
adds -- as DMA-staged vector-fp32 C kernels (`ideogram4_vec.py`) that read the GEMM's fp32 tiles and write
its fp16 operands. Nothing is quantized: the activations entering a GEMM are fp16 (RNE), the residual
stream is fp32. The per-step scalars (the t-embedding, the AdaLN vectors, the input projection, the
final layer, the Euler update) run in numpy on the host (`ideogram4_ref.py`).

Needs the packed cache (`ideogram4_fp_pack.py`) and the checkpoint folder for the top-level tensors.

    python3 ideogram4_fp_backend.py --steps 12 --grid 16x16 --seed 0 --out ~/ideogram4/fp_latents.npy
        --ref ~/ideogram4/ref_latents_256.npy    (cosine of the velocity vs the fp32 reference per step)
        --layers 0-33   --dump step1.npz          (every layer's output of step 1)
    GATE=1 python3 ideogram4_fp_backend.py --layers 0-0 --steps 1      (layer 0 op by op vs numpy)

Layouts (`ideogram4_vec.py`): 256 tokens padded to 288 rows (24 row blocks of 12); A operands `pack_a_slices`
(uint16 halves), B panels `pack_b_group` (fp16, 6 strips per group), C tiles `[group][rb][s][192 f32]`.
The weights stream through a two-slot pool of E4M3 buffers (one per layer parity) and ONE shared set
of fp16 panel buffers the pass fills per layer; one TinyJit per layer parity (its captured buffers).
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: os.sched_setaffinity(0, {0, 1, 6, 7, 8, 9, 10, 11})
except (AttributeError, OSError): pass
from tinygrad import Tensor, dtypes                                      # noqa: E402
from zy import OA                            # noqa: E402
from zy import gemm_fp16 as G                 # noqa: E402
import ideogram4_vec as V                    # noqa: E402  (the model's vector kernels + the generic vec_f16 helpers)
import ideogram4_ref as R                                                # noqa: E402
from ideogram4_weights import Ideogram4Weights                           # noqa: E402

DEV = "ZHOUYI"
os.environ.setdefault("ZHOUYI_CHAIN", "32")                 # one fused chain per layer (18 kernels): never a three-job wave (one lost data on the board)
GATE = bool(int(os.environ.get("GATE", "0")))
PROF = bool(int(os.environ.get("PROF", "0")))
NT = 12
T = 256; NRB = 24; ROWS = 12 * NRB; KS, NS = 24, 6
C_, NH, DH, DHP, M_ = R.D, R.NH, R.HD, 288, R.FF
TP = 288
QA, KB, VB, QKV_TOTAL = V.qkv_a16_sizes(NH, DH, DHP, NRB, KS, NS, TP)
HC = 3 * NRB * 1152                                       # floats per head of the attention C tiles (288 x 288)
N_QKV, N_W13 = 3 * NH * DH, 2 * M_


_COPIES: dict = {}


_SALT = [0]


def host_read(t, tries: int = 200) -> np.ndarray:
    """A VERIFIED host read of a device-written tensor. The host's view of the device's writes lags -- data a DMA
    drain wrote is shadowed by a stale CLEAN line in the pages' one cacheable alias -- the kernel's linear
    map -- for seconds, and a second host read of the same bytes can be stale too (measured on the board,
    2026-09-24) -- and a lagging line in a step's output fed the next step and made runs
    differ. `_flush()` invalidates that alias. The device checksums stay as the verification: they also
    cover the lost device writes seen on the board, which no cache operation can fix. So: the device sums the buffer per task (`vec_f16.tec_sum`,
    by DMA: the device's view is consistent) with a per-call salt; the host reads the sums until the salt is
    the call's, then reads the data until its own per-task sums match. Returns the bytes as float32 (the
    caller reinterprets); sizes that are not a multiple of 12 x 8 KiB fall back to a plain read."""
    n = int(np.prod(t.shape)) * t.dtype.itemsize
    if n % (NT * 8192) or t.dtype.itemsize not in (2, 4): return t.numpy().ravel().view(np.float32) if t.dtype.itemsize in (2, 4) else t.numpy().ravel()
    e = _COPIES.get(n)
    if e is None:
        # (NT + 1) SLOTS, not ints: `tec_sum` gives every task its own 64-byte line, because two cores
        # storing into one 32-byte line lose data (measured on the board) and these are the
        # checksums the read below is trusted against.
        e = _COPIES[n] = (OA.register_csrc("tec_sum_%d" % n, V.tec_sum_src(n // 4, nt=NT), ntasks=NT),
                          zeros((NT + 1) * V.TEC_SUM_SLOT, dtypes.int32), dev(V.tec_copy_descs()))
    key, res, d = e
    src = t.reshape(-1)             # the kernel reads the bytes as ints whatever the dtype: ⚠️ a `bitcast` here materialized a COPY (409 MB at 1024 px: out of the 3 GB device address space)
    _SALT[0] += 1; salt = _SALT[0]; sv = np.zeros(2048, np.int32); sv[0] = salt
    r = OA.csrc_call(key, res, src, d, dev(sv)).realize()
    _flush(r, t)
    for _ in range(tries):
        sums = r.numpy()[::V.TEC_SUM_SLOT]
        if int(sums[NT]) == salt: break
        time.sleep(0.02); _flush(r, t)
    else: raise RuntimeError("host_read: the device's checksums never became visible")
    per = n // 4 // NT
    for _ in range(tries):
        raw = t.numpy().ravel().view(np.int32)
        mine = raw.reshape(NT, per).astype(np.int64).sum(1) & 0xFFFFFFFF
        want = sums[:NT].astype(np.int64) & 0xFFFFFFFF
        if np.array_equal(mine, want): return raw.view(np.float32)
        time.sleep(0.02); _flush(r, t)
    raise RuntimeError("host_read: the host's view never matched the device's checksums")


_FLUSH: list = []


def _flush(*ts):
    """Invalidate the CPU caches' stale clean lines for the tensors about to be read
    (`ops_zhouyi.host_invalidate`): ~0.1 ms per 2 MB against the ~15 ms of the 32 MB E4M3 eviction pass
    this replaces. ⚠️ Needs the driver fix of 2026-09-24; under an older KMD the ioctl is a no-op and
    the checksum loop below will spin, so the pass is still available as `IDEO_FLUSH_PASS=1`."""
    if os.environ.get("IDEO_FLUSH_PASS") == "1":
        if not _FLUSH:
            _FLUSH.append((Tensor.zeros(32 << 20, device=DEV, dtype=dtypes.uint8).contiguous().realize(), zeros(32 << 20, dtypes.uint16)))
        src, dst = _FLUSH[0]
        OA.e4m3_stream(src, out=dst).realize()
        return
    for t in ts: OA.host_invalidate(t)


def read_stable(t) -> np.ndarray:
    """The block output `[288, 4608]` fp32 through `host_read`."""
    return host_read(t).view(np.float32)


def dev(a, dt=None):
    t = Tensor(np.ascontiguousarray(a), device=DEV) if dt is None else Tensor(np.ascontiguousarray(a), device=DEV, dtype=dt)
    return t.realize()


def zeros(n, dt): return Tensor.zeros(n, device=DEV, dtype=dt).contiguous().realize()


class Kernels:
    """The six vector kernels (registered once) and their descriptor tensors."""
    def __init__(self):
        reg = lambda name, src: OA.register_csrc(name, src, ntasks=NT)
        self.rms = reg("rms_scale_dma", V.rms_scale_dma_src(T, C_, R.EPS, KS, NRB, nt=NT)); self.d_rms = dev(V.rms_scale_dma_descs(C_, NRB))
        self.resid = reg("resid_dma", V.resid_dma_src(T, C_, NS, NRB, R.EPS, nt=NT)); self.d_resid = dev(V.resid_dma_descs(C_, NRB))
        self.swiglu = reg("swiglu_dma", V.swiglu_dma_src(T, M_, NRB, KS, nt=NT)); self.d_swiglu = dev(V.swiglu_dma_descs(M_, NRB, KS))
        self.qkv = reg("qkv_dma", V.qkv_dma_src(T, NH, DH, DHP, NRB, KS, NS, TP, 1e-5, nt=NT)); self.d_qkv = dev(V.qkv_dma_descs(NRB, KS, NS))
        self.softmax = reg("softmax_dma", V.softmax_dma_src(T, NH, TP, NS, NRB, KS, 1.0 / np.sqrt(DH), nt=NT)); self.d_softmax = dev(V.softmax_dma_descs(NRB, KS))
        self.gather = reg("gather_o_dma", V.gather_o_dma_src(T, NH, DH, DHP, NS, NRB, KS, nt=NT)); self.d_gather = dev(V.gather_o_dma_descs(NRB))


def make_bufs():
    """The block's intermediates, one shared set (allocated, zero): the JIT owns none of them."""
    b = dict(Ah=zeros(ROWS * C_, dtypes.uint16), Cqkv=zeros(N_QKV // 96 * NRB * 1152, dtypes.float32), QKV=zeros(QKV_TOTAL, dtypes.uint16),
             S=zeros(NH * HC, dtypes.float32), P=zeros(NH * ROWS * TP, dtypes.uint16), PV=zeros(NH * HC, dtypes.float32),
             Ao=zeros(ROWS * C_, dtypes.uint16), Co=zeros(48 * NRB * 1152, dtypes.float32), x1=zeros(ROWS * C_, dtypes.float32),
             Ah2=zeros(ROWS * C_, dtypes.uint16), C13=zeros(N_W13 // 96 * NRB * 1152, dtypes.float32), Am=zeros(ROWS * M_, dtypes.uint16),
             Cm=zeros(48 * NRB * 1152, dtypes.float32), out0=zeros(ROWS * C_, dtypes.float32), out1=zeros(ROWS * C_, dtypes.float32))
    b["panels"] = dict(qkv=zeros(N_QKV * C_, dtypes.uint16), o=zeros(C_ * C_, dtypes.uint16), w13=zeros(N_W13 * C_, dtypes.uint16), w2=zeros(C_ * M_, dtypes.uint16))
    return b


def make_pool():
    """Two slots of E4M3 code buffers (one per layer parity)."""
    mk = lambda: dict(qkv=zeros(N_QKV * C_, dtypes.uint8), o=zeros(C_ * C_, dtypes.uint8), w13=zeros(N_W13 * C_, dtypes.uint8), w2=zeros(C_ * M_, dtypes.uint8))
    return [mk(), mk()]


BUF_ORDER = ["Ah", "Cqkv", "QKV", "S", "P", "PV", "Ao", "Co", "x1", "Ah2", "C13", "Am", "Cm"]
W_ORDER = ["qkv", "o", "w13", "w2"]


class LayerFP:
    """One layer: the packed codes from the cache, the small tensors, the modulation; `run` = the block."""
    def __init__(self, l, cache, cos, sin):
        self.l = l; self.cache = cache
        sm = np.load(os.path.join(cache, f"L{l}_small.npz"))
        self.sc = {k: sm["sc_" + k] for k in W_ORDER}
        self.an1, self.an2, self.fn1, self.fn2, self.norm_q, self.norm_k = (sm[k] for k in ("an1", "an2", "fn1", "fn2", "norm_q", "norm_k"))
        ac = sm["ada_codes"]; self.ada_w = (G.e4m3_table()[ac] * sm["ada_scale"][:, None]).astype(np.float32) if ac.dtype == np.uint8 else ac
        self.ada_b = sm["ada_b"]
        self.t = dict(sc_qkv=dev(self.sc["qkv"]), sc_o=dev(self.sc["o"]), sc_w13=dev(self.sc["w13"]), sc_w2=dev(self.sc["w2"]),
                      nq2=dev(V.dup_quads(self.norm_q)), nk2=dev(V.dup_quads(self.norm_k)),
                      cs=dev(V.tile_rows(cos, NRB)), sn=dev(V.tile_rows(sin, NRB)))
        self.codes = {k: np.fromfile(os.path.join(cache, f"L{l}_{k}.bin"), np.uint8) for k in W_ORDER}

    def modulation(self, adaln):
        mod = self.ada_w @ adaln + self.ada_b
        s1, g1, s2, g2 = np.split(mod, 4)
        return (1.0 + s1).astype(np.float32), np.tanh(g1).astype(np.float32), (1.0 + s2).astype(np.float32), np.tanh(g2).astype(np.float32)

    def weights(self, pool):
        slots = pool[self.l % 2]
        for k in W_ORDER: slots[k].uop.buffer.copyin(memoryview(self.codes[k]))
        return slots


def block_fp(K: Kernels, t: dict, x, ws1, k1, ws2, k2, codes: dict, bufs: dict, panels: dict, out, gate=None):
    """The block on the device: x `[288, 4608]` fp32 -> `out` (the same shape). `ws1` / `ws2` = dup quads
    of an1 * (1 + s1) / fn1 * (1 + s2), `k1` / `k2` = `resid_dma_consts(an2, g1, sc_o)` / `(fn2, g2, sc_w2)`.
    `gate` (a `Gate`) realizes and checks every op against numpy computed from the device's own inputs."""
    gm = lambda a, b, **kw: OA.gemm_gs(a, b, ks=KS, ns=NS, nrb=NRB, **kw)
    cs = lambda key, o, *ins: OA.csrc_call(key, o, *ins)
    g_ = (lambda name, tt, fn: gate.check(name, tt, fn)) if gate is not None else (lambda name, tt, fn: tt)
    # the weights: E4M3 codes -> fp16 panels
    pn = {k: g_("panels_" + k, OA.e4m3_stream(codes[k], out=panels[k]), lambda k=k: G.e4m3_table()[codes[k].numpy()].astype(np.float16).view(np.uint16)) for k in W_ORDER}
    # attention
    Ah = g_("Ah", cs(K.rms, bufs["Ah"], x, ws1, K.d_rms), lambda: V.rms_scale_a16_ref(x.numpy().reshape(ROWS, C_)[:T], np.ones(C_, np.float32), ws1.numpy().reshape(-1, 8)[:, :4].ravel(), R.EPS, KS, NRB))
    Cqkv = g_("Cqkv", gm(Ah, pn["qkv"], nslices=C_ // 96, ngroups=N_QKV // 96, out=bufs["Cqkv"]), lambda: _ref_gemm(Ah, pn["qkv"], C_, N_QKV))
    QKV = g_("QKV", cs(K.qkv, bufs["QKV"], Cqkv, t["nq2"], t["nk2"], t["cs"], t["sn"], t["sc_qkv"], K.d_qkv), lambda: _ref_qkv(Cqkv, t))
    S = g_("S", gm(QKV, QKV, nslices=3, ngroups=3, heads=NH, a_stride=QA, b_stride=KB, c_stride=HC, b_off=NH * QA, out=bufs["S"]), lambda: _ref_attn_gemm(QKV, 0, NH * QA))
    P = g_("P", cs(K.softmax, bufs["P"], S, K.d_softmax), lambda: V.softmax_dma_ref([V.c_from_tiles(S.numpy()[h * HC:(h + 1) * HC], NS, NRB, TP) for h in range(NH)], 1.0 / np.sqrt(DH), T, KS, NRB))
    PV = g_("PV", gm(P, QKV, nslices=3, ngroups=3, heads=NH, a_stride=ROWS * TP, b_stride=VB, c_stride=HC, b_off=(NH * QA + NH * KB), out=bufs["PV"]), lambda: _ref_attn_gemm(QKV, 0, NH * QA + NH * KB, a_buf=P))
    Ao = g_("Ao", cs(K.gather, bufs["Ao"], PV, K.d_gather), lambda: V.gather_o_a16_ref([V.c_from_tiles(PV.numpy()[h * HC:(h + 1) * HC], NS, NRB, DHP)[:T] for h in range(NH)], DH, KS, NRB))
    Co = g_("Co", gm(Ao, pn["o"], nslices=C_ // 96, ngroups=C_ // 96, out=bufs["Co"]), lambda: _ref_gemm(Ao, pn["o"], C_, C_))
    x1 = g_("x1", cs(K.resid, bufs["x1"], x, k1, Co, t["sc_o"], K.d_resid), lambda: _ref_resid(x, k1, Co))
    # the MLP
    Ah2 = g_("Ah2", cs(K.rms, bufs["Ah2"], x1, ws2, K.d_rms), lambda: V.rms_scale_a16_ref(x1.numpy().reshape(ROWS, C_)[:T], np.ones(C_, np.float32), ws2.numpy().reshape(-1, 8)[:, :4].ravel(), R.EPS, KS, NRB))
    C13 = g_("C13", gm(Ah2, pn["w13"], nslices=C_ // 96, ngroups=N_W13 // 96, out=bufs["C13"]), lambda: _ref_gemm(Ah2, pn["w13"], C_, N_W13))
    Am = g_("Am", cs(K.swiglu, bufs["Am"], C13, t["sc_w13"], K.d_swiglu), lambda: _ref_swiglu(C13, t["sc_w13"]))
    Cm = g_("Cm", gm(Am, pn["w2"], nslices=M_ // 96, ngroups=C_ // 96, out=bufs["Cm"]), lambda: _ref_gemm(Am, pn["w2"], M_, C_))
    return g_("out", cs(K.resid, out, x1, k2, Cm, t["sc_w2"], K.d_resid), lambda: _ref_resid(x1, k2, Cm))


# ---- the gate's references (numpy from the device's own inputs) ----
def _ref_gemm(a, b, K, N):
    A = _from_a(a.numpy(), K); B = b.numpy().view(np.float16).astype(np.float32)
    Bm = _from_b(B, N, K)
    return V.c_tiles_ref(A @ Bm.T, NS, NRB)


def _from_a(h, K):
    """the A layout (uint16) -> fp32 `[288, K]`."""
    return np.ascontiguousarray(h.reshape(K // 96, NRB, KS, 3, 4, 4).transpose(1, 3, 4, 0, 2, 5)).reshape(ROWS, K).view(np.float16).astype(np.float32) if h.dtype == np.uint16 else h


def _from_b(B, N, K):
    """the B panels (fp32 of the halves) -> `[N, K]`."""
    return np.ascontiguousarray(B.reshape(N // 96, K // 96, NS, KS, 4, 4, 4).transpose(0, 2, 4, 5, 1, 3, 6)).reshape(N, K)


def _ref_qkv(Cqkv, t):
    c = V.c_from_tiles(Cqkv.numpy(), NS, NRB, N_QKV)[:T] * t["sc_qkv"].numpy().reshape(-1, 8)[:, :4].ravel()
    nq = t["nq2"].numpy().reshape(-1, 8)[:, :4].ravel(); nk = t["nk2"].numpy().reshape(-1, 8)[:, :4].ravel()
    cos = _untile(t["cs"].numpy(), DH); sin = _untile(t["sn"].numpy(), DH)
    return V.qkv_a16_ref(c, nq, nk, cos[:T], sin[:T], 1e-5, NH, DH, DHP, NRB, KS, NS, TP)[0]


def _untile(a, K): return np.ascontiguousarray(a.reshape(3 * NRB, K // 4, 4, 4).transpose(0, 2, 1, 3)).reshape(ROWS, K)


def _ref_attn_gemm(QKV, a_off, b_off, a_buf=None):
    q = QKV.numpy(); out = []
    for h in range(NH):
        if a_buf is None: A = _from_a(q[a_off + h * QA:a_off + (h + 1) * QA], DHP)
        else: A = _from_a(a_buf.numpy()[h * ROWS * TP:(h + 1) * ROWS * TP], TP)
        B = _from_b(q[b_off + h * KB:b_off + (h + 1) * KB].view(np.float16).astype(np.float32), TP if a_buf is None else DHP, DHP if a_buf is None else TP)
        out.append(V.c_tiles_ref(A @ B.T, NS, NRB))
    return np.concatenate(out)


def _ref_resid(x, k3, Ct):
    k = k3.numpy().reshape(-1, 3, 8)[:, :, :4]; w, g, sc = k[:, 0].ravel(), k[:, 1].ravel(), k[:, 2].ravel()
    c = V.c_from_tiles(Ct.numpy(), NS, NRB, C_) * sc
    r = V.resid_ct_ref(x.numpy().reshape(ROWS, C_), w, g, c, R.EPS); r[T:] = 0
    return r.ravel()


def _ref_swiglu(C13, sc):
    c = V.c_from_tiles(C13.numpy(), NS, NRB, N_W13)[:T] * sc.numpy().reshape(-1, 8)[:, :4].ravel()
    return V.swiglu_a16_ref(c[:, :M_], c[:, M_:], KS, NRB)


def gate_read(tt) -> np.ndarray:
    """The gate's host read through the TEC copy: uint16 tensors come back as their bits."""
    raw = host_read(tt)
    return raw.view(np.uint16) if tt.dtype == dtypes.uint16 else raw.view(np.float32) if tt.dtype == dtypes.float32 else tt.numpy().ravel()


class Gate:
    def __init__(self): self.n = 0; self.bad = []
    def check(self, name, tt, fn):
        tt = tt.realize(); got = gate_read(tt); ref = np.asarray(fn()).ravel()
        # ⚠️ a few 64-byte pieces of a DMA drain were seen to come out unwritten (stale) under some system
        # condition (`stream_race*.py`, 2026-09-24: ~1 piece per 100 MB, never in the raw-launch probe);
        # the gate tolerates that (a layout bug is thousands of elements) and reports the count.
        if got.dtype == np.uint16:
            n, mx = _ulps(got, ref); big = _ulps_over(got, ref, 2)
            ok = big <= max(64, ref.size // 50000); msg = "%d/%d differ (max %d ULP, %d beyond 2 ULP)" % (n, ref.size, mx, big)
        else:
            d = np.abs(got - ref); scale = np.abs(ref).max() + 1e-30; big = int((d > 1e-4 * scale).sum()) if got.size == ref.size else ref.size
            ok = big <= max(64, ref.size // 50000)
            msg = "max |d| %.3g (max |ref| %.3g), %d elements beyond 1e-4 relative" % (d.max(), scale, big)
        if not ok:      # a host read right after the job can lag the device's writes: re-read after 0.3 s and say which it was
            time.sleep(0.3); got2 = gate_read(tt)
            big2 = _ulps_over(got2, ref, 2) if got.dtype == np.uint16 else int((np.abs(got2 - ref) > 1e-4 * (np.abs(ref).max() + 1e-30)).sum())
            msg += "; re-read after 0.3 s: %d %s" % (big2, "(LATE, not wrong)" if big2 <= max(64, ref.size // 50000) else "(still wrong)")
            ok = big2 <= max(64, ref.size // 50000)
        self.n += 1; self.bad += [] if ok else [name]
        print("   gate %-10s %s   %s" % (name, "ok " if ok else "BAD", msg), flush=True)
        if not ok and os.environ.get("DUMP_BAD"): np.savez(os.path.join(os.environ["DUMP_BAD"], name + ".npz"), got=got, ref=ref)
        return tt


def _ulps(got, ref):
    g = got.astype(np.int32); r = ref.astype(np.int32)
    g = np.where(g & 0x8000, -(g & 0x7FFF), g); r = np.where(r & 0x8000, -(r & 0x7FFF), r)
    d = np.abs(g - r); return int((d > 0).sum()), int(d.max())


def _ulps_over(got, ref, k):
    g = got.astype(np.int32); r = ref.astype(np.int32)
    g = np.where(g & 0x8000, -(g & 0x7FFF), g); r = np.where(r & 0x8000, -(r & 0x7FFF), r)
    return int((np.abs(g - r) > k).sum())


class Runner:
    """The per-parity TinyJit around `block_fp` and the plumbing of a layer's inputs."""
    def __init__(self, K, bufs, pool):
        self.K, self.bufs, self.pool = K, bufs, pool; self.jits = [None, None]

    def run(self, L: LayerFP, x, adaln, gate=None):
        t0 = time.perf_counter()
        s1, g1, s2, g2 = L.modulation(adaln)
        ws1, ws2 = dev(V.dup_quads(L.an1 * s1)), dev(V.dup_quads(L.fn1 * s2))
        gg1, gg2 = dev(V.resid_dma_consts(L.an2, g1, L.sc["o"].reshape(-1, 8)[:, :4].ravel())), dev(V.resid_dma_consts(L.fn2, g2, L.sc["w2"].reshape(-1, 8)[:, :4].ravel()))
        t1 = time.perf_counter()
        codes = L.weights(self.pool); t2 = time.perf_counter()
        par = L.l % 2; out = self.bufs["out%d" % par]
        small = [L.t[k] for k in ("sc_qkv", "sc_o", "sc_w13", "sc_w2", "nq2", "nk2", "cs", "sn")]
        if gate is not None:
            return block_fp(self.K, L.t, x, ws1, gg1, ws2, gg2, codes, self.bufs, self.bufs["panels"], out, gate=gate).realize()
        if self.jits[par] is None:
            from tinygrad.engine.jit import TinyJit
            K, bufs, panels = self.K, self.bufs, self.bufs["panels"]
            def body(xx, ws1_, g1_, ws2_, g2_, out_, *rest):
                tt = dict(zip(("sc_qkv", "sc_o", "sc_w13", "sc_w2", "nq2", "nk2", "cs", "sn"), rest[:8]))
                cd = dict(zip(W_ORDER, rest[8:12]))
                return block_fp(K, tt, xx, ws1_, g1_, ws2_, g2_, cd, bufs, panels, out_).realize()
            self.jits[par] = TinyJit(body)
        r = self.jits[par](x, ws1, gg1, ws2, gg2, out, *small, *[codes[k] for k in W_ORDER])
        if PROF: print("      layer %2d: modulation %.3f s, weights %.3f s, jit call %.3f s" % (L.l, t1 - t0, t2 - t1, time.perf_counter() - t2), flush=True)
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="~/ideogram4/unconditional_transformer"); ap.add_argument("--cache", default="~/ideogram4/fpcache")
    ap.add_argument("--steps", type=int, default=12); ap.add_argument("--grid", default="16x16"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu", type=float, default=0.5); ap.add_argument("--std", type=float, default=1.75); ap.add_argument("--layers", default="0-33")
    ap.add_argument("--out", default="~/ideogram4/fp_latents.npy"); ap.add_argument("--ref", default=None); ap.add_argument("--dump", default=None)
    a = ap.parse_args()
    gh, gw = (int(v) for v in a.grid.split("x")); assert gh * gw == T, "this harness is built for 256 tokens"
    lo, hi = (int(v) for v in a.layers.split("-"))
    M = R.Model(Ideogram4Weights(a.weights))
    cos, sin = R.rope_tables(R.image_positions(gh, gw))
    t0 = time.perf_counter()
    K = Kernels(); bufs = make_bufs(); pool = make_pool(); run = Runner(K, bufs, pool)
    layers = [LayerFP(l, os.path.expanduser(a.cache), cos, sin) for l in range(lo, hi + 1)]
    print("== Ideogram 4 at FULL PRECISION on the NPU: %d layers, %d steps, %dx%d tokens; set up in %.0f s ==" % (len(layers), a.steps, gh, gw, time.perf_counter() - t0), flush=True)
    sig = R.sigmas(a.steps, R.shifted_mu(gh * 16, gw * 16, a.mu), a.std)
    rng = np.random.RandomState(a.seed)
    x = rng.randn(T, R.C_IN).astype(np.float32)
    ref = np.load(os.path.expanduser(a.ref), allow_pickle=True) if a.ref else None
    from zy import timing as _TMG
    for i in range(a.steps):
        t_model = 1.0 - sig[i]
        ada = M.cond(t_model)
        hp = np.zeros((ROWS, C_), np.float32); hp[:T] = M.embed(x)
        h = dev(hp.ravel())
        _TMG.DEFAULT.reset(); t1 = time.perf_counter()
        dump = []
        for L in layers:
            if GATE and i == 0:
                gate = Gate(); h = run.run(L, h, ada, gate=gate)
                print("   layer %d: %d ops gated, bad: %s" % (L.l, gate.n, gate.bad or "none"), flush=True)
                if os.environ.get("GATE_REF", "1") == "1":
                    hn = h.numpy().reshape(ROWS, C_)[:T]; xin = hp[:T] if L.l == lo else None
                    if xin is not None:
                        want = M.block(L.l)(xin, (cos, sin), ada)
                        cosv = float((hn * want).sum() / np.sqrt((hn * hn).sum() * (want * want).sum()))
                        print("   layer %d output vs the fp32 reference block: cosine %.6f, max |d| %.3g (max |ref| %.3g)" % (L.l, cosv, np.abs(hn - want).max(), np.abs(want).max()), flush=True)
            else: h = run.run(L, h, ada)
            if a.dump and i == 0: dump.append(read_stable(h).reshape(ROWS, C_)[:T])
        if a.dump and i == 0: np.savez(os.path.expanduser(a.dump), h=np.stack(dump), ada=ada); print("   layer outputs of step 1 -> %s" % a.dump, flush=True)
        hn = read_stable(h).reshape(ROWS, C_)[:T]; wall = time.perf_counter() - t1
        rep = _TMG.DEFAULT.report()["categories"]; dev_ms = sum(v["seconds"] for c, v in rep.items() if c in ("T_submit", "T_AIFF_exec")) * 1e3
        nsub = rep.get("T_submit", {}).get("count", 0)
        v = M.out(hn, ada)
        x = x + np.float32(sig[i + 1] - sig[i]) * (-v)
        line = "   step %2d/%d sigma %.4f -> %.4f  blocks %.2f s (device %.0f ms in %d submits)  |x| %.3f |v| %.3f" % (i + 1, a.steps, sig[i], sig[i + 1], wall, dev_ms, nsub, np.abs(x).mean(), np.abs(v).mean())
        if ref is not None and i < len(ref):
            vr = ref[i]; line += "  v vs fp32: cosine %.4f" % float((v * vr).sum() / np.sqrt((v * v).sum() * (vr * vr).sum()))
        print(line, flush=True)
    np.save(os.path.expanduser(a.out), x)
    print("   latents -> %s" % a.out)


if __name__ == "__main__":
    main()

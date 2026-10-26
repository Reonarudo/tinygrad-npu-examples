#!/usr/bin/env python3
"""Ideogram 4's prompt conditioning on the NPU: the Qwen3-VL-8B text encoder's 36 decoder layers and the conditional transformer's
text projection, the same math as `ideogram4_text.py` (numpy, fp32) on the TEC's fp16 path -- exact fp16 x fp16 products, fp32
sums, nothing quantised -- from the packed cache `ideogram4_te_pack.py` makes (`te_p48/`, 7.2 GB).

Per layer: the fused linears (qkv, o, gate|up, down) as E4M3 -> fp16 GEMMs on the GM-staged `gemm_gs` (the codes expanded in
LSRAM, the weights streamed per layer through two device slots: a layer's copy lands while the previous layer runs); the qkv step
in one DMA-staged kernel (`qkv_te`: the scales, the q / k RMSNorms, RoPE -- a text prompt's MRoPE is plain RoPE -- Q in the A
layout, K and V^T panels per KV head, the head dim 128 padded to 192); QK^T and PV per query head on the same GEMM, the 8 KV heads
read by their 4 query heads each (`b_head_div`); the causal softmax (`softmax_stream`), the heads gathered and normalised
(`gather_o_sum`); the two RMSNorms (`rms_scale_dma`), the residual adds (`resid_dma`) and SwiGLU (`swiglu_dma`) as the DiT's
DMA-staged kernels. The residual stream is fp32 [R, 4224] (4096 real columns, the pad zero), R the tokens rounded up to 96.
The 13 taps (after layers 0, 3, ..., 33, 35) go straight into the text projection: tap t's own RMSNorm x its slice of
`llm_cond_norm` -> a GEMM against its 4224-K slice of `llm_cond_proj` -> accumulated x the tap's rms, then x the global
inverse rms over the 13 taps, x the scales, + the bias + the text indicator row: the text rows [n, 4608].

    TG=... python3 ideogram4_text_npu.py --prompt "..." --out out/x_text.npz [--cache ~/ideogram4/te_p48] [--compare cpu_text.npz]
    TG=... python3 ideogram4_text_npu.py --ids out/x_text.npz --gate states.npy [--layers 0-35]
        --gate: `ideogram4_text.py --dump-states` (every layer's input / output in fp32): each layer from the reference input vs the
                reference output (the relative error of the layer's own contribution, the cosine)
`ideogram4_text.py --npu` calls `encode` here; `run.sh` does so when TG is set (NPU_TEXT=0: the numpy path).
"""
import argparse, ctypes, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ZHOUYI_GM", "1")        # the GEMMs with GM (GemmGMRunner: the A chunks and the partials in GM)
os.environ.setdefault("ZHOUYI_CHAIN", "64")    # one fused chain per layer
from tinygrad import Tensor, dtypes                                      # noqa: E402
from zy import OA                            # noqa: E402
import ideogram4_vec as V                    # noqa: E402
from ideogram4_text import SafeDir, bf16_to_f32, TAPS                    # noqa: E402

DEV = "ZHOUYI"; H, HP, NQ, NKV, D, DP, INTER, NC = 4096, 4224, 32, 8, 128, 192, 12288, 4608
EPS, THETA, L = 1e-6, 5000000.0, 36
N_ = {"qkv": 6144, "o": HP, "gu": 24576, "dn": HP}; K_ = {"qkv": HP, "o": HP, "gu": HP, "dn": INTER}
OPTIME = os.environ.get("OPTIME", "0") == "1"        # every op of a layer realized, drained and timed alone (eager): ms per op
JIT = os.environ.get("JIT", "1") == "1" and not OPTIME
CAUSAL = os.environ.get("TE_CAUSAL", "spv")           # the causal trimming arms: s (the scores' key groups), p (P V's key slices), v (the softmax's groups)


def dev(a): return Tensor(np.ascontiguousarray(a), device=DEV).realize()
def zeros(n, dt): return Tensor.zeros(n, device=DEV, dtype=dt).contiguous().realize()


class TE:
    """The encoder for one token count `n`: its buffers, kernels and the two per-parity JITs."""
    def __init__(self, n, cache):
        # the rows: n rounded up to an even number of 12-row blocks with an attention piece of >= 8 row blocks (the GEMMs' GM C
        # regions hold up to 14; 681 tokens -> 720 rows, not 768); the keys TP: the rows rounded up to 96 (the key groups)
        piece = lambda nrb: max(p_ for p_ in range(2, 15, 2) if nrb % p_ == 0)
        self.NRB = next(nrb for nrb in range(-(-n // 12) + (-(-n // 12) % 2), 10 ** 6, 2) if piece(nrb) >= min(8, nrb))
        self.n = n; self.R = R = 12 * self.NRB; self.TP = TP = -(-R // 96) * 96; self.cache = cache; self.piece = piece(self.NRB)
        inv = 1.0 / (THETA ** (np.arange(0, D, 2, dtype=np.float64) / D)); f = np.arange(R, dtype=np.float64)[:, None] * inv[None]
        emb = np.concatenate([f, f], -1); cos, sin = np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)
        self.slots = [{k: zeros(N_[k] * K_[k], dtypes.uint8) for k in N_} for _ in range(2)]
        reg = lambda name, src: OA.register_csrc(name, src, ntasks=V.NT)
        # the attention: the fused qkv linear's tiles -> Q (A layout) / K / V^T panels; the causal softmax -> P (A layout) + 1 / sums;
        # the P V tiles x 1 / sum -> the o projection's A layout (K 4224: the pad columns never written, zero from the start)
        self.k_qkv = reg("qkv_te", V.qkv_te_src(n, self.NRB, TP, EPS)); self.d_qkv = dev(V.qkv_te_descs(self.NRB))
        self.csn = dev(V.qkv_te_csn(cos, sin, self.NRB))
        gb = 2 if (TP // 96) % 2 == 0 else 1                # groups per DMA request of the softmax (must divide the key groups)
        # causal: the scores' tasks cover the key groups up to their rows, the softmax reads and writes only those, P V runs only
        # those key slices (the P beyond stays zero from the start): the same values, about half the work
        # the heads in batches of HB (all 32 up to 1056 rows; 8 past that: a batch's logits are HB x R x TP fp32, 143 MB at
        # 2112 rows -- 571 MB for all 32 heads did not fit the device window beside the rest); the batches reuse the logits and P
        self.HB = HB = NQ if R <= 1056 else 8
        self.k_soft = [reg("softmax_stream", V.softmax_stream_src(n, HB, TP, self.NRB, 24, 1.0 / math.sqrt(D), hb=b * HB, gb=gb, causal=True, causal_trim="v" in CAUSAL)) for b in range(NQ // HB)]
        self.d_soft = dev(V.softmax_stream_descs(self.NRB, gb))
        self.k_gat = reg("gather_o_sum", V.gather_o_sum_src(n, NQ, D, DP, 6, self.NRB, 24)); self.d_gat = dev(V.gather_o_sum_descs(self.NRB, DP))
        self.QA, self.KB, self.VB, self.HC = R * DP, TP * DP, DP * TP, (DP // 96) * self.NRB * 1152
        self.b_qkv = zeros(NQ * self.QA + NKV * (self.KB + self.VB), dtypes.uint16)
        self.b_p, self.b_sums, self.b_o = zeros(HB * R * TP, dtypes.uint16), zeros(NQ * self.NRB * 16, dtypes.float32), zeros(R * HP, dtypes.uint16)
        self.b_pv = zeros(NQ * self.HC, dtypes.float32)
        # the residual stream [R, 4224] (pad columns 0): RMSNorm -> the A layout (the mean over 4096), x + C * sc, SwiGLU of the
        # interleaved gate / up tiles -> the down projection's A layout
        self.k_rms = reg("rms_scale_dma", V.rms_scale_dma_src(n, HP, EPS, 24, self.NRB, cdiv=H)); self.d_rms = dev(V.rms_scale_dma_descs(HP, self.NRB))
        self.k_res = reg("resid_dma", V.resid_dma_src(n, HP, 6, self.NRB, EPS, plain=True)); self.d_res = dev(V.resid_dma_descs(HP, self.NRB))
        self.k_swi = reg("swiglu_dma", V.swiglu_dma_src(n, INTER, self.NRB, 24, interleaved=True)); self.d_swi = dev(V.swiglu_dma_descs(INTER, self.NRB, 24, interleaved=True))
        self.b_a, self.b_m = zeros(R * HP, dtypes.uint16), zeros(R * INTER, dtypes.uint16)
        self.b_x = [zeros(R * HP, dtypes.float32) for _ in range(4)]
        # the text projection: the 13 taps' panels in one buffer (loaded once), their norm weights, scales, bias and indicator row
        z = np.load(os.path.join(cache, "cond_small.npz"))
        self.b_cond = zeros(13 * NC * HP, dtypes.uint8)
        OA.Device[DEV].raw.drain()                          # ⚠️ raw host writes into a device buffer: nothing may be in flight (ZHOUYI_ASYNC)
        for t in range(13):
            src = np.fromfile(os.path.join(cache, "cond_t%d.bin" % t), np.uint8); assert src.size == NC * HP, src.size
            ctypes.memmove(self.b_cond.uop.buffer._buf.va + t * NC * HP, src.ctypes.data, src.nbytes)
        self.norm_t = [dev(V.dup_quads(z["norm_t"][t])) for t in range(13)]
        self.cond_sc, self.cond_add = dev(z["sc"]), dev(z["bias"] + z["ind"])
        # a layer's small tables (norm weights, scales): one set per parity, host-written like the slots (sized from layer 0's)
        self.small = [{k: zeros(v.size, dtypes.float32) for k, v in self.tables(0).items()} for _ in range(2)]
        self.jits = [None, None]

    # the host side of a layer: its weights into slot l % 2, its small tables
    def tables(self, l):
        """Layer l's small tables (fp32 arrays, the kernels' layouts) from the cache."""
        z = np.load(os.path.join(self.cache, f"L{l}_small.npz"))
        pad = lambda v: np.concatenate([v, np.zeros(HP - v.size, np.float32)]); one = np.ones(HP, np.float32)
        tabs = {"ws_in": V.dup_quads(pad(z["input_layernorm"])), "ws_post": V.dup_quads(pad(z["post_attention_layernorm"])),
                "k3_o": V.resid_dma_consts(one, one, z["sc_o"]), "k3_dn": V.resid_dma_consts(one, one, z["sc_dn"]),
                "sc2_qkv": V.dup_quads(z["sc_qkv"]), "nq2": V.dup_quads(z["q_norm"]), "nk2": V.dup_quads(z["k_norm"]),
                "sc2_o": V.dup_quads(z["sc_o"]), "sc2_dn": V.dup_quads(z["sc_dn"]), "sc2_gu": V.dup_quads(z["sc_gu"])}
        return {k: np.ascontiguousarray(v, np.float32) for k, v in tabs.items()}

    def load(self, l):
        """Layer l's weights and tables into the parity-l % 2 buffers: raw host copies, nothing submitted (safe on a worker thread).
        ⚠️ Their last reader, layer l - 2, must be complete: `encode` drains before it hands layer l + 1's load to the worker and
        then runs layer l (one job is in flight at most, and it is layer l's), so the copy overlaps layer l on the device. A
        tinygrad copy-in would drain the device itself: no overlap."""
        s, sm = self.slots[l % 2], self.small[l % 2]
        for k in N_:
            src = np.fromfile(os.path.join(self.cache, f"L{l}_{k}.bin"), np.uint8); assert src.size == N_[k] * K_[k], (l, k, src.size)
            ctypes.memmove(s[k].uop.buffer._buf.va, src.ctypes.data, src.nbytes)
        for k, v in self.tables(l).items():
            assert v.nbytes == sm[k].nbytes(), (k, v.nbytes); ctypes.memmove(sm[k].uop.buffer._buf.va, v.ctypes.data, v.nbytes)
        return s, sm

    SMALL = ("ws_in", "ws_post", "k3_o", "k3_dn", "sc2_o", "sc2_dn", "sc2_gu", "sc2_qkv", "nq2", "nk2")

    def layer(self, x, l, loaded=None):
        """Layer l on x (flat [R x 4224] fp32) -> the next x (another buffer); one TinyJit per parity (its slot's buffers).
        `loaded`: `load(l)`'s result when a worker did it already."""
        t0 = time.perf_counter(); slot, sm = loaded or self.load(l); par = l % 2
        if OPTIME: OA.Device[DEV].raw.drain(); print("   layer %2d: weights loaded in %.0f ms" % (l, (time.perf_counter() - t0) * 1e3), flush=True)
        if not JIT: return self.body(x, par, slot, sm)
        if self.jits[par] is None:
            from tinygrad.engine.jit import TinyJit
            def fn(xx, *rest):
                return self.body(xx, par, dict(zip(N_, rest[len(self.SMALL):])), dict(zip(self.SMALL, rest[:len(self.SMALL)])))
            self.jits[par] = TinyJit(fn)
        return self.jits[par](x, *[sm[k] for k in self.SMALL], *[slot[k] for k in N_])

    def gemm(self, a, b, k, b_off=0):
        return OA.gemm_gs(a, b, ks=24, ns=6, nrb=self.NRB, nslices=K_[k] // 96, ngroups=N_[k] // 96, lin48=True, b8=True, b_off=b_off)

    def body(self, x, par, slot, sm):
        R, TP, NRB, pc = self.R, self.TP, self.NRB, self.piece
        tm = []; t_ = [time.perf_counter()]
        def T(name, t):            # OPTIME: realize + drain each op, its ms
            if OPTIME: t.realize(); OA.Device[DEV].raw.drain(); now = time.perf_counter(); tm.append((name, (now - t_[0]) * 1e3)); t_[0] = now
            return t
        a_in = T("rms", OA.csrc_call(self.k_rms, self.b_a, x, sm["ws_in"], self.d_rms))
        qo = T("qkv_te", OA.csrc_call(self.k_qkv, self.b_qkv, T("Cqkv", self.gemm(a_in, slot["qkv"], "qkv")), sm["sc2_qkv"], sm["nq2"], sm["nk2"], self.csn, self.d_qkv))
        HB, G = self.HB, NQ // NKV
        for b in range(NQ // HB):       # ⚠️ the batches reuse the logits and P: each op realized in order (the JIT records that order)
            st = T("S%d" % b, OA.gemm_gs(qo, qo, ks=24, ns=6, nrb=NRB, nslices=DP // 96, ngroups=TP // 96, heads=HB, a_stride=self.QA, b_stride=self.KB,
                                         a_off=b * HB * self.QA, b_off=NQ * self.QA + (b * HB // G) * self.KB, c_stride=(TP // 96) * NRB * 1152, piece=pc,
                                         b_head_div=G, causal=1 if "s" in CAUSAL else 0)).realize()
            p = T("softmax%d" % b, OA.csrc_call(self.k_soft[b], self.b_p, st, self.b_sums, self.d_soft)).realize()      # causal, unnormalised P
            pv = T("PV%d" % b, OA.gemm_gs(p, qo, ks=24, ns=6, nrb=NRB, nslices=TP // 96, ngroups=DP // 96, heads=HB, a_stride=R * TP, b_stride=self.VB,
                                          b_off=NQ * self.QA + NKV * self.KB + (b * HB // G) * self.VB, c_stride=self.HC, c_off=b * HB * self.HC, piece=pc,
                                          b_head_div=G, nlive=D, causal=2 if "p" in CAUSAL else 0, out=self.b_pv)).realize()
        oa = T("gather", OA.csrc_call(self.k_gat, self.b_o, pv, self.b_sums, self.d_gat))               # x 1 / sum -> the A layout
        x1 = T("resid", OA.csrc_call(self.k_res, self.b_x[0], x, sm["k3_o"], T("Co", self.gemm(oa, slot["o"], "o")), sm["sc2_o"], self.d_res))
        a2 = T("rms2", OA.csrc_call(self.k_rms, self.b_a, x1, sm["ws_post"], self.d_rms))
        am = T("swiglu", OA.csrc_call(self.k_swi, self.b_m, T("Cgu", self.gemm(a2, slot["gu"], "gu")), sm["sc2_gu"], self.d_swi))
        x2 = T("resid2", OA.csrc_call(self.k_res, self.b_x[2 + par], x1, sm["k3_dn"], T("Cdn", self.gemm(am, slot["dn"], "dn")), sm["sc2_dn"], self.d_res))
        if OPTIME: print("   layer, ms per op (%.0f): %s" % (sum(v for _, v in tm), ", ".join("%s %.1f" % kv for kv in tm)), flush=True)
        return x2.realize()

    # the text projection, a tap at a time
    def tap(self, x, t, acc, ssq):
        """Tap t (x: a layer's output, flat) -> acc += C_t x rms_t (the C tiles' layout), ssq += x's rows' sums of squares."""
        R, NRB = self.R, self.NRB
        xs = x.reshape(R, HP).square().sum(-1)                                                   # [R]
        a = OA.csrc_call(self.k_rms, self.b_a, x, self.norm_t[t], self.d_rms)                     # fp16(x / rms_t * w_t), K = 4224
        ct = OA.gemm_gs(a, self.b_cond, ks=24, ns=6, nrb=NRB, nslices=HP // 96, ngroups=NC // 96, lin48=True, b8=True, b_off=t * NC * HP)
        rms_t = (xs * (1.0 / H) + EPS).sqrt().reshape(1, NRB, 1, 3, 1, 4, 1)                      # the tiles' rows (rb, i, r)
        acc2 = (acc + ct.reshape(NC // 96, NRB, 6, 3, 4, 4, 4) * rms_t).contiguous().realize()
        return acc2, (ssq + xs).realize()

    def rows(self, acc, ssq):
        """The accumulated projection -> the text rows [R, 4608] fp32."""
        R, NRB = self.R, self.NRB
        inv = (ssq * (1.0 / (H * 13)) + 1e-6).rsqrt().reshape(1, NRB, 1, 3, 1, 4, 1)
        c = (acc * inv).permute(1, 3, 5, 0, 2, 4, 6).reshape(R, NC)                              # the C tiles -> [R, N]
        return (c * self.cond_sc + self.cond_add).realize()


_TE = {}


def encode(ids, cache, te_dir, log=print):
    """Token ids -> the text rows [n, 4608] (float32): everything on the NPU but the embedding lookup."""
    n = len(ids); cache, te_dir = os.path.expanduser(cache), os.path.expanduser(te_dir)
    te = _TE.get(n) or _TE.setdefault(n, TE(n, cache))         # one TE per length: its kernels, buffers and JITs reused
    S = SafeDir(te_dir); dt, shape, raw = S.raw("language_model.embed_tokens.weight"); row = shape[1] * 2
    x0 = np.zeros((te.R, HP), np.float32)
    x0[:n, :H] = np.stack([bf16_to_f32(np.frombuffer(raw[t * row:(t + 1) * row], np.uint16)) for t in ids])
    x = dev(x0.ravel())                                                                           # flat, as every layer's output
    acc, ssq = zeros((NC // 96) * te.NRB * 1152, dtypes.float32).reshape(NC // 96, te.NRB, 6, 3, 4, 4, 4), zeros(te.R, dtypes.float32)
    t0 = time.perf_counter()
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(1); fut = ex.submit(te.load, 0)        # layer l + 1's weights copied by a worker while layer l runs
    for l in range(L):
        loaded = fut.result()
        OA.Device[DEV].raw.drain()                                  # layer l - 1 complete: its buffers are layer l + 1's (see `load`)
        if l + 1 < L: fut = ex.submit(te.load, l + 1)
        x = te.layer(x, l, loaded)
        if l in TAPS: acc, ssq = te.tap(x, TAPS.index(l), acc, ssq)
    ex.shutdown()
    rows = te.rows(acc, ssq)
    out = OA.host_invalidate(rows).numpy()[:n]
    log("   text encoder on the NPU: %d tokens (%d rows) in %.1f s" % (n, te.R, time.perf_counter() - t0))
    return out


def compare(rows, ref_path):
    w = np.load(os.path.expanduser(ref_path))["rows"].astype(np.float64); g = rows.astype(np.float64)
    pc = (g * w).sum(1) / np.linalg.norm(g, axis=1) / np.linalg.norm(w, axis=1)
    return "cosine %.7f (worst row %.6f), max |d| %.3g of max |ref| %.3g, max rel |d| %.3g" % (
        float((g * w).sum() / np.linalg.norm(g) / np.linalg.norm(w)), pc.min(), np.abs(g - w).max(), np.abs(w).max(), np.abs(g - w).max() / np.abs(w).max())


def gate(ids, cache, states, layers):
    """Each layer from the reference input (fp32 states [37, n, 4096]: x0 and every layer's output) vs the reference output."""
    n = len(ids); te = TE(n, os.path.expanduser(cache)); lo, hi = (int(v) for v in layers.split("-"))
    ref = np.load(os.path.expanduser(states), mmap_mode="r")
    print("== the text encoder on the NPU, layer by layer: %d tokens -> %d rows, layers %d-%d" % (n, te.R, lo, hi), flush=True)
    bad = []
    for l in range(lo, hi + 1):
        xin = np.zeros((te.R, HP), np.float32); xin[:n, :H] = ref[l]
        t0 = time.perf_counter(); y = te.layer(dev(xin.ravel()), l); dt = time.perf_counter() - t0
        got = OA.host_invalidate(y).numpy().reshape(te.R, HP)[:n, :H]; want = np.asarray(ref[l + 1])
        d, dd = got - want, want - np.asarray(ref[l])                       # the layer's own contribution as the scale
        g64, w64 = got.astype(np.float64), want.astype(np.float64); cos = float((g64 * w64).sum() / np.linalg.norm(g64) / np.linalg.norm(w64))
        rel = np.linalg.norm(d) / np.linalg.norm(dd)
        if not (rel < 1e-2 and np.isfinite(got).all()): bad.append(l)
        print("   layer %2d: %.2f s  max |d| %.3g (max |ref| %.3g, max |ref delta| %.3g)  delta rel err %.3g  cosine %.7f%s" % (
            l, dt, np.abs(d).max(), np.abs(want).max(), np.abs(dd).max(), rel, cos, "" if np.isfinite(got).all() else "  NON-FINITE"), flush=True)
    print("   gate: %s" % ("every layer within 1e-2 of its contribution" if not bad else "BAD layers %s" % bad), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None); ap.add_argument("--prompt-file", default=None, help="read the prompt (e.g. a JSON caption) from a file")
    ap.add_argument("--ids", default=None, help="an npz with `ids` (a text file ideogram4_text.py made) instead of a prompt")
    ap.add_argument("--out", default=None); ap.add_argument("--cache", default="~/ideogram4/te_p48", help="ideogram4_te_pack.py's cache")
    ap.add_argument("--te", default="~/ideogram4/text_encoder"); ap.add_argument("--tok", default="~/ideogram4/tokenizer")
    ap.add_argument("--compare", default=None, help="a reference text npz: cosine / max |d| of the rows")
    ap.add_argument("--reps", type=int, default=1, help="encode this many times (the later ones: warm caches and compiles)")
    ap.add_argument("--gate", default=None, help="ideogram4_text.py --dump-states: layer by layer vs the numpy encoder"); ap.add_argument("--layers", default="0-35")
    a = ap.parse_args()
    if a.ids:
        z = np.load(os.path.expanduser(a.ids)); ids, text = [int(i) for i in z["ids"]], str(z["text"]) if "text" in z.files else ""
    else:
        from ideogram4_text import tokenize
        if a.prompt_file: a.prompt = open(os.path.expanduser(a.prompt_file)).read().strip()
        ids, text = tokenize(os.path.expanduser(a.tok), a.prompt)
    if a.gate: return gate(ids, a.cache, a.gate, a.layers)
    print("== the text encoder on the NPU: %d tokens" % len(ids), flush=True)
    for _ in range(a.reps):
        t0 = time.perf_counter(); rows = encode(ids, a.cache, a.te, log=lambda s_: print(s_, flush=True))
        print("   text rows %s in %.1f s" % (rows.shape, time.perf_counter() - t0), flush=True)
    if a.out: np.savez(os.path.expanduser(a.out), rows=rows.astype(np.float32), ids=np.array(ids), text=text)
    if a.compare: print("   vs %s: %s" % (a.compare, compare(rows, a.compare)), flush=True)


if __name__ == "__main__":
    main()

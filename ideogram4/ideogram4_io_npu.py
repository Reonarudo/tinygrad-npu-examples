#!/usr/bin/env python3
"""The DiT step's host matmuls on the NPU (2026-09-28): the input projection (latents [n_img, 128] -> the embedded image rows
[n_img, 4608], written straight into the device rows buffer layer 0 reads) and the final layer (LayerNorm x (1 + AdaLN scale) ->
[n_img, 128] velocity), which ran in numpy on the reference BLAS at ~1.3 GFLOP/s: 1.8 + 2.0 s a branch, ~7.6 s a step.

Precision: the weights are bf16 values that fp16 does not all hold (the smallest are below its normal range), so W x 2^10 is split
into fp16 hi + lo; the input projection also splits the latents (K = [x_hi | x_hi | x_lo] . [W_hi | W_lo | W_hi], ~22 bits), the
final layer stacks [W_hi ; W_lo] along N (its A -- the normalized rows -- fp16, like every linear of the blocks).

    python3 ideogram4_io_npu.py        (the probe: both ends vs numpy fp32 on the cond weights, and their times)
"""
import ctypes, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinygrad import Tensor, dtypes                                      # noqa: E402
from zy import OA                            # noqa: E402
from zy import gemm_fp16 as G                 # noqa: E402
import ideogram4_vec as V                    # noqa: E402  (the model's vector kernels + the generic vec_f16 helpers)

DEV = "ZHOUYI"; NT = 12; C_ = 4608; IN = 128; S10 = 2.0 ** 10


def _dev(a, dt=None): return (Tensor(np.ascontiguousarray(a), device=DEV) if dt is None else Tensor(np.ascontiguousarray(a), device=DEV, dtype=dt)).realize()
def _va(t): return t.uop.buffer.ensure_allocated()._buf.va
def _hilo(w):
    h = w.astype(np.float16); return h, (w - h.astype(np.float32)).astype(np.float16)
def _piece(nrb): return max(p for p in range(2, 15, 2) if nrb % p == 0)


class IOProj:
    """One branch's input projection and final layer. `rows` = the device rows buffer layer 0 reads (fp32 [ROWS][4608]),
    `treal` its real rows, the image rows the LAST n_img of them (after `text_rows`, the cond branch's)."""
    def __init__(self, tw, ROWS, n_img, text_rows=None):
        self.tw, self.ROWS, self.n_img = tw, ROWS, n_img
        self.text = None if text_rows is None else np.ascontiguousarray(text_rows, np.float32)
        self.n_text = 0 if text_rows is None else text_rows.shape[0]; self.treal = self.n_text + n_img
        wh, wl = _hilo(tw.input_w.astype(np.float32) * S10)                      # [4608, 128]
        self.b_in = _dev(V.b_layout_ref(np.concatenate([wh, wl, wh], 1).astype(np.float32), 24, 6))
        self.bias_in = _dev((tw.input_b + tw.ind[1]).astype(np.float32))
        fh, fl = _hilo(tw.fin_w.astype(np.float32) * S10)                        # [128, 4608]
        self.b_fin = _dev(V.b_layout_ref(np.concatenate([fh, fl, np.zeros((32, C_), np.float16)]).astype(np.float32), 24, 6))
        self.nrb_in = -(-n_img // 96) * 8                                        # the latents' rows padded to 96
        self.a_in = _dev(np.zeros(self.nrb_in * 12 * 3 * IN, np.uint16))
        self.nrb = ROWS // 12
        nc = 3 * self.nrb * 6 * 192; self.nc = nc                               # padded to host_read's 12 x 8 KiB: a VERIFIED read (a plain
        self.c_fin = _dev(np.zeros(-(-nc * 4 // 98304) * 98304 // 4, np.float32))   # numpy() of 5.4 MB came back with stale lines)
        self.k_rows = OA.register_csrc("ctile_rows", V.ctile_rows_src(self.nrb_in, n_img, C_, self.n_text, 1.0 / S10, nt=NT), ntasks=NT)
        self.d_rows = _dev(V.ctile_rows_descs(n_img, C_))
        self.k_ln = OA.register_csrc("rms_scale_dma", V.rms_scale_dma_src(self.treal, C_, 1e-6, 24, self.nrb, nt=NT, center=True), ntasks=NT)
        self.d_ln = _dev(V.rms_scale_dma_descs(C_, self.nrb))
        # the device-resident step (embed_dev / final_dev): the text rows and the pad rows' zeros from device templates, the latents
        # packed on the device, the velocity left as rows [ROWS][288] (hi + bias | lo) for cfg_euler
        self.k_pack = OA.register_csrc("lat_pack", V.lat_pack_src(self.nrb_in, nt=NT), ntasks=NT); self.d_pack = _dev(V.lat_pack_descs())
        if self.text is not None:
            self.text_dev = _dev(self.text.ravel())
            self.k_text = OA.register_csrc("dma_copy", V.dma_copy_src(self.text.nbytes, 0, nt=NT), ntasks=NT); self.d_text = _dev(V.dma_copy_descs(self.text.nbytes))
        npad = (ROWS - self.treal) * C_ * 4; self.npad = npad
        if npad:
            self.zero_dev = _dev(np.zeros(npad // 4, np.float32))
            self.k_pad = OA.register_csrc("dma_copy", V.dma_copy_src(npad, self.treal * C_ * 4, nt=NT), ntasks=NT); self.d_pad = _dev(V.dma_copy_descs(npad))
        self.vrows = _dev(np.zeros(ROWS * 288, np.float32))
        self.k_vrows = OA.register_csrc("ctile_rows", V.ctile_rows_src(self.nrb, ROWS, 288, 0, 1.0 / S10, nt=NT), ntasks=NT); self.d_vrows = _dev(V.ctile_rows_descs(ROWS, 288))
        self.bias_v = _dev(np.concatenate([tw.fin_b, np.zeros(160)]).astype(np.float32))
        self.ws2 = None

    def use_temb(self, temb, stepv):
        """the final layer's scale table made on the device each step (TembNPU.fin) instead of `set_scales`."""
        self.temb, self.stepv = temb, stepv; self.ws2buf = _dev(np.zeros(2 * C_, np.float32))

    def set_scales(self, adas):
        """The final layer's (1 + AdaLN scale) for every step, once an image (dup_quads, device)."""
        tw = self.tw
        self.ws2 = [_dev(V.dup_quads((1.0 + (tw.fin_ada_w @ (ad / (1.0 + np.exp(-ad))) + tw.fin_ada_b)).astype(np.float32))) for ad in adas]

    def embed_dev(self, xdev, rows, big):
        """the device latents (fp32 [12 nrb_in][128]) -> rows (text rows and pad zeros restored from the templates): no host work."""
        if self.text is not None: OA.csrc_call(self.k_text, rows, self.text_dev, self.d_text).realize()
        if self.npad: OA.csrc_call(self.k_pad, rows, self.zero_dev, self.d_pad).realize()
        OA.csrc_call(self.k_pack, self.a_in, xdev, self.d_pack).realize()
        ct = OA.gemm_gs(self.a_in, self.b_in, ks=24, ns=6, nrb=self.nrb_in, nslices=3 * IN // 96, ngroups=C_ // 96, piece=_piece(self.nrb_in), out=big).realize()
        OA.csrc_call(self.k_rows, rows, ct, self.bias_in, self.d_rows).realize()
        return rows

    def final_dev(self, h, step, a_buf):
        """the last hidden rows -> the velocity rows in `self.vrows` (device); `set_scales` (or `use_temb`) first."""
        if getattr(self, "temb", None) is not None: self.temb.fin(self.ws2buf, self.stepv[step]); ws2 = self.ws2buf
        else: ws2 = self.ws2[step]
        OA.csrc_call(self.k_ln, a_buf, h, ws2, self.d_ln).realize()
        ct = OA.gemm_gs(a_buf, self.b_fin, ks=24, ns=6, nrb=self.nrb, nslices=C_ // 96, ngroups=3, piece=_piece(self.nrb), out=self.c_fin).realize()
        OA.csrc_call(self.k_vrows, self.vrows, ct, self.bias_v, self.d_vrows).realize()
        return self.vrows


class Sampler:
    """The latents on the device and the guidance + Euler step there (cfg_euler); the host reads x back only when asked."""
    def __init__(self, x0, io_c, io_u, gw, tm):
        n = x0.shape[0]; self.n = n; rows = io_c.nrb_in * 12
        nb = -(-rows * IN * 4 // 98304) * 98304                                   # host_read's verified size
        xp = np.zeros(nb // 4, np.float32); xp[:n * IN] = x0.ravel(); self.x = _dev(xp)
        self.k = OA.register_csrc("cfg_euler", V.cfg_euler_src(n, io_c.treal - n, nt=NT), ntasks=NT); self.d = _dev(V.cfg_euler_descs())
        assert io_u is None or io_u.treal == n
        self.prm = [_dev(np.array([gw[i] if io_u is not None else 1.0, float(tm[i + 1] - tm[i])] + [0] * 14, np.float32)) for i in range(len(gw))]
        self.io_c, self.io_u = io_c, io_u

    def step(self, i):
        vu = self.io_u.vrows if self.io_u is not None else self.io_c.vrows
        OA.csrc_call(self.k, self.x, self.io_c.vrows, vu, self.prm[i], self.d).realize()

    def read(self):
        import ideogram4_fp_backend as FB
        return FB.host_read(self.x).view(np.float32)[:self.n * IN].reshape(self.n, IN).copy()

    def embed(self, x, rows, big):
        """latents [n_img, 128] -> rows[n_text:treal] (+ the text rows, the pad rows zeroed); `big` = an fp32 scratch >= nrb_in x 12 x 4608."""
        xh, xl = _hilo(x.astype(np.float32))
        a = np.zeros((self.nrb_in * 12, 3 * IN), np.float16); a[:self.n_img] = np.concatenate([xh, xh, xl], 1)
        pa = G.pack_a_slices(a, 24); ctypes.memmove(_va(self.a_in), pa.ctypes.data, pa.nbytes)
        base = _va(rows)                                                        # the host mapping is non-cacheable: the NPU reads these
        if self.text is not None: ctypes.memmove(base, self.text.ctypes.data, self.text.nbytes)
        ctypes.memset(base + self.treal * C_ * 4, 0, (self.ROWS - self.treal) * C_ * 4)
        ct = OA.gemm_gs(self.a_in, self.b_in, ks=24, ns=6, nrb=self.nrb_in, nslices=3 * IN // 96, ngroups=C_ // 96, piece=_piece(self.nrb_in), out=big).realize()
        OA.csrc_call(self.k_rows, rows, ct, self.bias_in, self.d_rows).realize()
        return rows

    def final(self, h, ada, a_buf):
        """the last hidden rows (device fp32 [ROWS][4608]) -> the velocity [n_img, 128] (host); `a_buf` = a uint16 scratch of ROWS x 4608."""
        tw = self.tw
        scale = (1.0 + (tw.fin_ada_w @ (ada / (1.0 + np.exp(-ada))) + tw.fin_ada_b)).astype(np.float32)
        OA.csrc_call(self.k_ln, a_buf, h, _dev(V.dup_quads(scale)), self.d_ln).realize()
        ct = OA.gemm_gs(a_buf, self.b_fin, ks=24, ns=6, nrb=self.nrb, nslices=C_ // 96, ngroups=3, piece=_piece(self.nrb), out=self.c_fin).realize()
        import ideogram4_fp_backend as FB
        c = FB.host_read(ct).view(np.float32)[:self.nc].reshape(3, self.nrb, 6, 3, 4, 4, 4).transpose(1, 3, 5, 0, 2, 4, 6).reshape(self.ROWS, 288)
        c = c[self.treal - self.n_img:self.treal]
        return ((c[:, :IN] + c[:, IN:2 * IN]) / S10 + tw.fin_b).astype(np.float32)


class ModNPU:
    """The layers' modulation and per-step tables on the device (2026-09-28). Per image: every layer's `ada_codes` (E4M3 [18432, 512],
    the per-row scale applied after) expanded to fp16 panels on the device (`e4m3_stream`) and multiplied by ALL steps' AdaLN inputs at
    once -- one gemm_gs a layer, A = [adaln_hi (sp rows) ; adaln_lo (sp rows)], ~fp32 -- into `L.cm`; per step, `mod_tables` (in the
    layer's JIT) turns that into the four tables the block reads. Replaces, per layer and step, a numpy 18432 x 512 mat-vec, the
    table building and four uploads (on the main thread)."""
    def __init__(self, layers, steps):
        self.sp = -(-steps // 12) * 12; self.layers = layers; sp = self.sp
        self.k = OA.register_csrc("mod_tables", V.mod_tables_src(sp, nt=NT), ntasks=NT); self.desc = _dev(V.mod_tables_descs())
        q4 = lambda a: a.reshape(-1, 8)[:, :4].ravel()
        for L in layers:
            ac = np.zeros((18432, 576), np.uint8); ac[:, :512] = L._sm["ada_codes"]
            L.ada_pk = np.ascontiguousarray(ac.reshape(192, 6, 4, 4, 6, 24, 4).transpose(0, 4, 1, 5, 2, 3, 6)).ravel()   # pack_b_group_e4m3, every group
            L.kc = _dev(V.mod_tables_consts(L._sm["ada_scale"], L.ada_b, L.an1, L.an2, L.fn1, L.fn2, q4(L.sc["o"]), q4(L.sc["w2"])))
            L.cm = _dev(np.zeros(192 * (2 * sp // 12) * 1152, np.float32))
        self.stage = _dev(np.zeros(18432 * 576, np.uint8)); self.pan = _dev(np.zeros(18432 * 576, np.uint16))
        self.a = _dev(np.zeros(2 * sp * 576, np.uint16))
        self.stepv = [_dev(np.array([s_] + [0] * 15, np.int32)) for s_ in range(steps)]

    def setup(self, adas=None):
        """adas: the steps' AdaLN inputs ([512] each; None: `self.a` already holds them, TembNPU's) -> every layer's `cm`."""
        sp = self.sp
        if adas is not None:
            A = np.zeros((2 * sp, 576), np.float16)
            for s_, ad in enumerate(adas): h, l = _hilo(np.asarray(ad, np.float32)); A[s_, :512] = h; A[sp + s_, :512] = l
            pa = G.pack_a_slices(A, 24); ctypes.memmove(_va(self.a), pa.ctypes.data, pa.nbytes)
        for L in self.layers:
            ctypes.memmove(_va(self.stage), L.ada_pk.ctypes.data, L.ada_pk.nbytes)
            OA.e4m3_stream(self.stage, out=self.pan).realize()
            OA.gemm_gs(self.a, self.pan, ks=24, ns=6, nrb=2 * sp // 12, nslices=6, ngroups=192, piece=_piece(2 * sp // 12), out=L.cm).realize()

    def tables(self, L, step, ws1, k1, ws2, k2):
        """(outside a JIT) the layer's tables for `step` into the four buffers."""
        return OA.csrc_call(self.k, ws1, L.cm, L.kc, k1, ws2, k2, self.stepv[step], self.desc).realize()


class TembNPU:
    """The t-embedding MLP -> the AdaLN inputs of every step, and the final layer's per-step scale, ON THE DEVICE (2026-09-28):
    host: the steps' sinusoidal embeddings (sin / cos of 12 x 4608) and one memmove of each weight's fp16 panels per image.
    Each weight W as [W_hi ; W_lo] (N-stacked, packed once a process), each activation as [x_hi rows ; x_lo rows]: `mlp_link`
    keeps hi.hi + hi.lo + lo.hi (~fp32). The last link writes `mod.a` (ModNPU's A) directly."""
    def __init__(self, tw, steps):
        self.sp = sp = -(-steps // 12) * 12; H2 = 2 * sp // 12
        def pan(w, n_pad=None, k_pad=None):
            w = w.astype(np.float32); N, K = w.shape; Np, Kp = n_pad or N, k_pad or K
            wp = np.zeros((Np, Kp), np.float32); wp[:N, :K] = w; h, l = _hilo(wp)
            return np.ascontiguousarray(V.b_layout_ref(np.concatenate([h, l]).astype(np.float32), 24, 6))
        pad = lambda v, n: np.concatenate([v.astype(np.float32), np.zeros(n - v.shape[0], np.float32)])
        self.W = [pan(tw.t_in_w), pan(tw.t_out_w), pan(tw.ada_w, n_pad=576), pan(tw.fin_ada_w, k_pad=576)]
        self.b = [_dev(tw.t_in_b.astype(np.float32)), _dev(tw.t_out_b.astype(np.float32)), _dev(pad(tw.ada_b, 576)), _dev(tw.fin_ada_b.astype(np.float32))]
        self.stage = _dev(np.zeros(max(w.size for w in self.W), np.uint16))
        self.c = _dev(np.zeros(96 * 2 * H2 * 1152, np.float32)); self.c4 = _dev(np.zeros(96 * 2 * H2 * 1152, np.float32))
        self.a1, self.a2, self.a3 = (_dev(np.zeros(2 * sp * C_, np.uint16)) for _ in range(3)); self.afin = _dev(np.zeros(2 * sp * 576, np.uint16))
        mk = lambda n, act: OA.register_csrc("mlp_link", V.mlp_link_src(sp, n, act, nt=NT), ntasks=NT)
        self.k_silu, self.k_none, self.k_ada, self.k_ada2 = mk(C_, "silu"), mk(C_, "none"), mk(576, "silu"), mk(576, "silu2")
        self.dl = _dev(V.mlp_link_descs())
        self.k_fin = OA.register_csrc("fin_tables", V.fin_tables_src(sp, C_, nt=NT), ntasks=NT); self.df = _dev(V.fin_tables_descs())
        self.H2 = H2

    def _gemm(self, a, wi, K, N, out):
        ctypes.memmove(_va(self.stage), self.W[wi].ctypes.data, self.W[wi].nbytes)
        return OA.gemm_gs(a, self.stage, ks=24, ns=6, nrb=self.H2, nslices=K // 96, ngroups=2 * N // 96, piece=_piece(self.H2), out=out).realize()

    def run(self, ts, a_mod):
        """ts: the steps' model times -> a_mod (ModNPU's A: the AdaLN inputs, hi / lo rows, K 576) and self.c4 (the final scales)."""
        import ideogram4_ref as R
        sp = self.sp; E = np.zeros((2 * sp, C_), np.float16)
        for s_, t in enumerate(ts): h, l = _hilo(R.sinusoidal(1e4 * float(t), C_)); E[s_] = h; E[sp + s_] = l
        pa = G.pack_a_slices(E, 24); ctypes.memmove(_va(self.a1), pa.ctypes.data, pa.nbytes)
        OA.csrc_call(self.k_silu, self.a2, self._gemm(self.a1, 0, C_, C_, self.c), self.b[0], self.dl).realize()
        OA.csrc_call(self.k_none, self.a3, self._gemm(self.a2, 1, C_, C_, self.c), self.b[1], self.dl).realize()
        c3 = self._gemm(self.a3, 2, C_, 576, self.c)
        OA.csrc_call(self.k_ada, a_mod, c3, self.b[2], self.dl).realize()
        OA.csrc_call(self.k_ada2, self.afin, c3, self.b[2], self.dl).realize()
        self._gemm(self.afin, 3, 576, C_, self.c4)

    def fin(self, ws2, stepv):
        return OA.csrc_call(self.k_fin, ws2, self.c4, self.b[3], stepv, self.df).realize()


def temb_probe():
    import ideogram4_ref as R
    from ideogram4_weights import Ideogram4Weights
    tw = R.TopWeights(Ideogram4Weights(os.path.expanduser("~/ideogram4/transformer")))
    ts = np.linspace(0.00055, 0.99945, 12).astype(np.float32)
    t0 = time.perf_counter(); T = TembNPU(tw, 12); t1 = time.perf_counter()
    amod = _dev(np.zeros(2 * 12 * 576, np.uint16))
    for _ in range(2): t2 = time.perf_counter(); T.run(ts, amod); t3 = time.perf_counter()
    A = OA.host_invalidate(amod).numpy().reshape(6, 2, 24, 3, 4, 4).transpose(1, 3, 4, 0, 2, 5).reshape(24, 576).view(np.float16).astype(np.float64)
    got = A[:12, :512] + A[12:, :512]
    want = np.stack([R.adaln_input(tw, float(t)) for t in ts]).astype(np.float64)
    t4 = time.perf_counter(); [R.adaln_input(tw, float(t)) for t in ts]; th = time.perf_counter() - t4
    print("packing %.1f s (a process)   chain %.0f ms (an image; host %.0f ms for 12 steps)" % (t1 - t0, (t3 - t2) * 1e3, th * 1e3))
    print("AdaLN inputs: max |d| %.3g (max |want| %.3g), rel %.3g" % (np.abs(got - want).max(), np.abs(want).max(), np.abs(got - want).max() / np.abs(want).max()))
    ws2 = _dev(np.zeros(9216, np.float32)); stv = [_dev(np.array([s_] + [0] * 15, np.int32)) for s_ in range(12)]; worst = 0.0
    for s_ in range(12):
        T.fin(ws2, stv[s_]); g_ = OA.host_invalidate(ws2).numpy()
        ad = want[s_].astype(np.float32); w_ = V.dup_quads((1.0 + (tw.fin_ada_w @ (ad / (1.0 + np.exp(-ad))) + tw.fin_ada_b)).astype(np.float32))
        worst = max(worst, float(np.abs(g_ - w_).max() / np.abs(w_).max()))
    print("final scales: max rel |d| %.3g over 12 steps" % worst)


def mod_probe():
    import ideogram4_ref as R, math
    from ideogram4_weights import Ideogram4Weights
    import ideogram4_fp_1024 as F
    tw = R.TopWeights(Ideogram4Weights(os.path.expanduser("~/ideogram4/transformer")))
    Ls = [F.Layer(l, "/mnt/ssd/ideogram4/fpcache_cond") for l in (0, 17, 33)]
    ts = np.linspace(0.02, 0.99, 12)
    adas = [R.adaln_input(tw, float(t)) for t in ts]
    t0 = time.perf_counter(); M = ModNPU(Ls, 12); t1 = time.perf_counter(); M.setup(adas); t2 = time.perf_counter()
    print("init %.2f s (host packing, once)   setup %.0f ms for %d layers (per image)" % (t1 - t0, (t2 - t1) * 1e3, len(Ls)))
    bufs = [_dev(np.zeros(n, np.float32)) for n in (9216, 27648, 9216, 27648)]
    q4 = lambda a: a.reshape(-1, 8)[:, :4].ravel(); worst = [0.0] * 4
    for L in Ls:
        for st in (0, 5, 11):
            t0 = time.perf_counter(); M.tables(L, st, *bufs); tt = time.perf_counter() - t0
            got = [OA.host_invalidate(b_).numpy() for b_ in bufs]
            s1, g1, s2, g2 = L.modulation(adas[st])
            want = (V.dup_quads(L.an1 * s1), V.resid_dma_consts(L.an2, g1, q4(L.sc["o"])), V.dup_quads(L.fn1 * s2), V.resid_dma_consts(L.fn2, g2, q4(L.sc["w2"])))
            for j in range(4):
                worst[j] = max(worst[j], float(np.abs(got[j] - want[j]).max() / max(np.abs(want[j]).max(), 1e-30)))
            gd = V.resid_dma_consts(L.an2, g1, q4(L.sc["o"])).reshape(-1, 3, 8)[:, 1]; gg = got[1].reshape(-1, 3, 8)[:, 1]
            print("layer %2d step %2d: tables %.1f ms   ws1 %.2g  k1 %.2g (gate max |d| %.2g)  ws2 %.2g  k2 %.2g  (max rel |d|)" % (L.l, st, tt * 1e3,
                  *[np.abs(got[j] - want[j]).max() / np.abs(want[j]).max() for j in range(2)], np.abs(gg - gd).max(), *[np.abs(got[j] - want[j]).max() / np.abs(want[j]).max() for j in (2, 3)]))
    t0 = time.perf_counter()
    for L in Ls: L.modulation(adas[0])
    print("host modulation: %.1f ms a layer" % ((time.perf_counter() - t0) / len(Ls) * 1e3))


def step_probe():
    """The device-resident step vs the host: embed_dev / final_dev / cfg_euler over 3 steps with random 'hidden rows' (h = the embedded
    rows + noise, standing in for the blocks) against the host's embed_image_tokens / final_layer / Euler in fp32."""
    import ideogram4_ref as R
    from ideogram4_weights import Ideogram4Weights
    import ideogram4_fp_backend as FB
    twc = R.TopWeights(Ideogram4Weights(os.path.expanduser("~/ideogram4/transformer"))); twu = R.TopWeights(Ideogram4Weights(os.path.expanduser("~/ideogram4/unconditional_transformer")))
    rng = np.random.RandomState(3); n_img, n_text = 4096, 525; ROWS = -(-(n_text + n_img) // 96) * 96
    text = (rng.randn(n_text, C_) * 0.5).astype(np.float32)
    ioc, iou = IOProj(twc, ROWS, n_img, text), IOProj(twu, ROWS, n_img)
    rows = _dev(np.zeros(ROWS * C_, np.float32)); big = _dev(np.zeros(ioc.nrb_in * 12 * C_, np.float32)); ab = _dev(np.zeros(ROWS * C_, np.uint16))
    x = rng.randn(n_img, IN).astype(np.float32); tm = [0.0, 0.1, 0.25, 0.4]; gw = [7.0, 7.0, 3.0]
    adas = [R.adaln_input(twc, t) for t in tm[:3]]; adau = [R.adaln_input(twu, t) for t in tm[:3]]
    ioc.set_scales(adas); iou.set_scales(adau)
    S = Sampler(x, ioc, iou, gw, tm); xh = x.copy(); noise = [(rng.randn(ROWS, C_) * 3).astype(np.float32) for _ in range(3)]
    for i in range(3):
        t0 = time.perf_counter(); v = {}
        for io, ad, nm in ((ioc, adas[i], "c"), (iou, adau[i], "u")):
            io.embed_dev(S.x, rows, big)
            hr = FB.host_read(rows).view(np.float32).reshape(ROWS, C_) + noise[i]            # "the blocks"
            h = _dev(hr.ravel()); io.final_dev(h, i, ab)
            # the host: the same rows from the host latents
            tw = io.tw; img = R.embed_image_tokens(tw, xh); hh = np.zeros((ROWS, C_), np.float32)
            if io.text is not None: hh[:n_text] = io.text
            hh[io.treal - n_img:io.treal] = img; hh += noise[i]
            v[nm] = R.final_layer(tw, hh[io.treal - n_img:io.treal], ad).astype(np.float32)
            print("   step %d %s: embedded rows vs host max |d| %.3g" % (i, nm, np.abs(hr - noise[i] - (hh - noise[i])).max()))
        S.step(i); tstep = time.perf_counter() - t0
        xh = (xh + (np.float32(gw[i]) * v["c"] + np.float32(1 - gw[i]) * v["u"]) * np.float32(tm[i + 1] - tm[i])).astype(np.float32)
        xd = S.read(); d = np.abs(xd - xh)
        print("step %d: latents device vs host max |d| %.3g mean |d| %.3g (|x| %.3g)" % (i, d.max(), d.mean(), np.abs(xh).mean()))
    # lat_pack's split vs numpy's fp16
    OA.csrc_call(ioc.k_pack, ioc.a_in, S.x, ioc.d_pack).realize()          # pack the CURRENT latents
    A = OA.host_invalidate(ioc.a_in).numpy()
    a = np.zeros((ioc.nrb_in * 12, 3 * IN), np.float16); xh16, xl16 = _hilo(xd); a[:n_img] = np.concatenate([xh16, xh16, xl16], 1)
    print("lat_pack vs numpy's hi / lo of the same x: %d of %d halves differ" % ((A.view(np.uint16) != G.pack_a_slices(a, 24).ravel().view(np.uint16)).sum(), A.size))


def probe():
    import ideogram4_ref as R
    from ideogram4_weights import Ideogram4Weights
    tw = R.TopWeights(Ideogram4Weights(os.path.expanduser("~/ideogram4/transformer")))
    rng = np.random.RandomState(1); n_img, n_text = 4096, 608; ROWS = -(-(n_text + n_img) // 96) * 96
    text = (rng.randn(n_text, C_) * 0.5).astype(np.float32)
    io = IOProj(tw, ROWS, n_img, text)
    rows = _dev(np.full(ROWS * C_, 7.0, np.float32)); big = _dev(np.zeros(io.nrb_in * 12 * C_, np.float32)); ab = _dev(np.zeros(ROWS * C_, np.uint16))
    import ideogram4_fp_backend as FB
    x = rng.randn(n_img, IN).astype(np.float32)
    for rep in range(3):
        t0 = time.perf_counter(); io.embed(x, rows, big); te = time.perf_counter() - t0
    got = FB.host_read(rows).view(np.float32).reshape(ROWS, C_)
    want = R.embed_image_tokens(tw, x)
    print("embed: %.1f ms   image rows max |d| %.3g (max |want| %.3g), rel %.3g   text rows exact %s   pad rows zero %s" % (te * 1e3,
          np.abs(got[n_text:n_text + n_img] - want).max(), np.abs(want).max(), np.abs(got[n_text:n_text + n_img] - want).max() / np.abs(want).max(),
          np.array_equal(got[:n_text], text), not got[n_text + n_img:].any()))
    # the final layer on hidden rows with a large per-row mean (the centring's case) and a realistic spread
    hid = (rng.randn(ROWS, C_) * rng.uniform(0.5, 20, (ROWS, 1)) + rng.uniform(-50, 50, (ROWS, 1))).astype(np.float32)
    hdev = _dev(hid.ravel()); ada = rng.randn(512).astype(np.float32)
    for rep in range(3):
        t0 = time.perf_counter(); v = io.final(hdev, ada, ab); tf = time.perf_counter() - t0
    vw = R.final_layer(tw, hid[n_text:n_text + n_img], ada).astype(np.float32)
    d = np.abs(v - vw)
    print("final: %.1f ms   velocity max |d| %.3g (max |want| %.3g), rel %.3g, mean rel %.3g" % (tf * 1e3, d.max(), np.abs(vw).max(), d.max() / np.abs(vw).max(), d.mean() / np.abs(vw).mean()))
    t0 = time.perf_counter(); R.embed_image_tokens(tw, x); R.final_layer(tw, hid[n_text:n_text + n_img], ada); print("host numpy, both: %.0f ms" % ((time.perf_counter() - t0) * 1e3))


if __name__ == "__main__":
    mod_probe() if "mod" in sys.argv[1:] else step_probe() if "step" in sys.argv[1:] else temb_probe() if "temb" in sys.argv[1:] else probe()

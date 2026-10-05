#!/usr/bin/env python3
"""Ideogram 4's VAE decoder on the NPU (2026-09-28): packed latents [4096, 128] -> the 1024-px RGB image. Every conv = conv_a (the
im2col into the ks-36 A layout; the GroupNorm affine + SiLU and the nearest-2x upsample fused; 1 x 1 shortcut slices appended to
conv2's K) -> gemm_gs (ks 36 x ns 4, fp16 weight panels) -> conv_c (bias, residual, pad columns zeroed, the next GroupNorm's
statistics), over bands of image rows (the A buffer <= ~320 MiB). Activations: guarded fp16 rows (ideogram4_vec.conv_guard).

    python3 vae_npu.py LATENTS.npy [--ref vae_ref.npz] [--out img.png]
"""
import argparse, ctypes, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ZHOUYI_GM", "1")
from tinygrad import Tensor, dtypes                                      # noqa: E402
from zy import OA                            # noqa: E402
import ideogram4_vec as V                    # noqa: E402  (the model's vector kernels + the generic vec_f16 helpers)
from zy import gemm_plan as GP              # noqa: E402
import ideogram4_vae as VA                                               # noqa: E402

DEV = "ZHOUYI"; NT = 12
GN_DEV = os.environ.get("GN_DEV", "1") == "1"                            # the GroupNorm affines made on the device (gn_prep)
PQ_DEV = os.environ.get("PQ_DEV", "1") == "1"                            # post_quant_conv on the device (conv1x1)
# solo 12-task csrc launches as ONE job, the three cores' groups overlapped behind a closing barrier task (ZHOUYI_SOLO12=chain): the
# conv_a / conv_c / gn_silu / softmax kernels here run eagerly, outside a JIT graph, and a solo launch otherwise runs its cores one after
# another (conv_a 1024^2 band 47 ms; the three-job WAVE 33; this 17). Safe for these kernels: no cache line is written by two tasks
# (bulk output by DMA; stats / sums one 64 B line per task or row block) -- wave_drop_probe PAD=16 under it: 0 / 1500 (2026-09-28)
os.environ.setdefault("ZHOUYI_SOLO12", "chain")
NSLOT = int(os.environ.get("NSLOT", "3"))                             # activation slots: x, the activated copy, the conv's out
A_MAX = int(os.environ.get("A_MB", "160")) << 20; C_MAX = int(os.environ.get("C_MB", "80")) << 20    # band scratch: 160 / 80 beat 320 / 160 (the 1024^2 GEMMs) and 80 / 40 (per-band cost)


def dev(a): return Tensor(np.ascontiguousarray(a), device=DEV).realize()
def zeros(n, dt): return Tensor.zeros(n, device=DEV, dtype=dt).contiguous().realize()
def rup(x, m): return -(-x // m) * m


class Pool:
    """Fixed-size activation slots, allocated ONCE, largest first. Per-shape allocations fragment the device pool: at the 1024^2
    stage 1.17 GB was free in 16 MB pieces but no 256 MB block was left, and a 259 / 544 MB activation took ENOMEM."""
    def __init__(self, n, elems):
        self.elems = rup(elems, 1 << 12); self.free = [Tensor.empty(self.elems, device=DEV, dtype=dtypes.float16) for _ in range(n)]
        for t in self.free: t.uop.buffer.ensure_allocated()
    def get(self, need):
        assert need <= self.elems, (need, self.elems)
        assert self.free, "out of activation slots (raise NSLOT)"
        return self.free.pop()
    def put(self, t): self.free.append(t)


POOL = None


class Act:
    """An activation: guarded fp16 rows of W x H x C (rows of Wp pixels), and its per-channel partial statistics (conv_c's).
    Without `t` it takes a POOL slot (its guards zeroed: the slot holds an earlier activation) and hands it back when dropped."""
    def __init__(self, W, H, C, t=None):
        self.W, self.H, self.C = W, H, C; self.Wp = rup(W, 12); self.G = V.conv_guard(self.Wp); self.slot = None
        if t is None and POOL is None: t = zeros((2 * self.G + H * self.Wp) * C, dtypes.float16)   # before make_pool (<= 128^2)
        if t is None:
            n = (2 * self.G + H * self.Wp) * C; t = self.slot = POOL.get(n)
            base = t.uop.buffer.ensure_allocated()._buf.va; g = self.G * C * 2                   # the host mapping is non-cacheable: a memset is what the NPU reads
            ctypes.memset(base, 0, g); ctypes.memset(base + (self.G + H * self.Wp) * C * 2, 0, g)
        self.t = t
        self.st = []                                                     # conv_c's stats tensors (one per band)
    def __del__(self):
        if self.slot is not None and POOL is not None: POOL.put(self.slot); self.slot = None
    def off(self): return self.G * self.C
    @staticmethod
    def from_chw(x):                                                     # numpy [C, H, W] -> an Act
        C, H, W = x.shape; Wp = rup(W, 12); G = V.conv_guard(Wp)
        r = np.zeros((2 * G + H * Wp, C), np.float16); r[G:G + H * Wp].reshape(H, Wp, C)[:, :W] = x.transpose(1, 2, 0)
        return Act(W, H, C, t=dev(r.ravel()))
    def rows(self, which=None):                                          # -> numpy [C, len(which), W]
        r = OA.host_invalidate(self.t).numpy().reshape(-1, self.C)[self.G:self.G + self.H * self.Wp].reshape(self.H, self.Wp, self.C)
        r = r if which is None else r[which]
        return r[:, :self.W].astype(np.float32).transpose(2, 0, 1)
    def gn(self, gamma, beta, groups=32, eps=1e-6):                      # the GroupNorm affine from the stats -> gab ([cb][32 a | 32 b])
        st = sum(OA.host_invalidate(s).numpy().astype(np.float64) for s in self.st)
        mean, rstd = V.conv_c_stats(st, self.C, self.W * self.H, groups, eps)
        a = np.repeat(rstd, self.C // groups) * gamma; b = beta - np.repeat(mean, self.C // groups) * a
        P = V.conv_perm32(); return dev(np.stack([a.reshape(-1, 32)[:, P], b.reshape(-1, 32)[:, P]], 1).ravel().astype(np.float32))


class VAE:
    def __init__(self, folder="~/ideogram4/vae", out_hw=1024):
        self.D = VA.VAEDecoder(folder); self.w = self.D._w
        self.k = {}; self.panels = {}; self.prof = {}; self.t_w = 0.0; self.t_gn = 0.0; self.t_chk = 0.0; self.t_host = 0.0
        self.out_hw = out_hw; self.make_pool()
        # persistent buffers: a kernel's returned tensor keeps its whole lazy graph (the inputs) alive, so results are realized and
        # DROPPED, the Acts keep plain buffers, and every band's A / C reuse these two
        self.abuf = zeros(A_MAX // 2, dtypes.uint16); self.cbuf = zeros(C_MAX // 4, dtypes.float32)

    def make_pool(self):
        """The activation slots, ONCE, before anything else is allocated (the pool fragments: allocated later they may find no block),
        sized for the largest activation, 256 ch at the output size; the mid attention borrows two for its scores / P."""
        global POOL
        Wp = rup(self.out_hw, 12)
        POOL = Pool(NSLOT, (2 * V.conv_guard(Wp) + self.out_hw * Wp) * max(self.D.cfg["block_out_channels"][:2]))

    def kern(self, name, src):
        if src not in self.k: self.k[src] = OA.register_csrc(name, src, ntasks=NT)
        return self.k[src]

    def gn_prep(self, x, gamma, beta, mode):
        """x's GroupNorm affine from its stats ON THE DEVICE (`ideogram4_vec.gn_prep`): "silu" -> gn_silu's gab, "plain" -> [a C | b C]."""
        key = ("gb", gamma.astype(np.float32).tobytes(), beta.astype(np.float32).tobytes())      # by value: an id can be reused
        if key not in self.panels: self.panels[key] = (dev(np.concatenate([gamma, beta]).astype(np.float32)),)
        nparts = int(np.prod(x.st[0].shape)) // (2 * x.C)
        k = self.kern("gn_prep", V.gn_prep_src(x.C, nparts, x.W * x.H, 1e-6, mode, nt=NT))
        dk = ("gnd", x.C, nparts)
        if dk not in self.panels: self.panels[dk] = dev(V.gn_prep_descs(x.C, nparts))
        gab = zeros(2 * x.C, dtypes.float32)
        return OA.csrc_call(k, gab, x.st[0], self.panels[key][0], self.panels[dk]).realize()

    def gn_silu(self, x, gamma, beta):
        """GroupNorm (x's stats) + SiLU, once, into a new Act in the unpack's order (`ideogram4_vec.gn_silu`); `.perm2` marks it."""
        th = time.perf_counter()
        if GN_DEV: gab = self.gn_prep(x, gamma, beta, "silu")
        else:
            st = sum(OA.host_invalidate(s_).numpy().astype(np.float64) for s_ in x.st)
            mean, rstd = V.conv_c_stats(st, x.C, x.W * x.H)
            a = np.repeat(rstd, x.C // 32) * gamma; b = beta - np.repeat(mean, x.C // 32) * a
            P = np.concatenate([V.conv_perm32() + 32 * k for k in range(2)])                       # within a 64-channel group
            gab = dev(np.stack([a.reshape(-1, 64)[:, P], b.reshape(-1, 64)[:, P]], 1).ravel().astype(np.float32))
        out = Act(x.W, x.H, x.C); out.perm2 = True; nrb = x.H * x.Wp // 12
        k = self.kern("gn_silu", V.gn_silu_src(x.Wp, x.H, x.C, nrb, nt=NT)); t0 = time.perf_counter(); self.t_host += t0 - th
        OA.csrc_call(k, out.t, x.t, gab, dev(V.gn_silu_descs(x.C)), dev(np.array([0, x.off()] + [0] * 14, np.int32))).realize()
        self.t_gn += time.perf_counter() - t0
        return out

    def weights(self, key, w, b, sc=None, perm2=False, gn=False):
        """Conv weights [O, C, 3, 3] (+ a 1 x 1 shortcut [O, Cs, 1, 1]) -> the fp16 panels (K = 9 C (+ Cs padded to 144), O padded to 64)."""
        if key not in self.panels:
            tw = time.perf_counter()
            O, C = w.shape[:2]; Op = rup(O, 64)
            Pm = V.conv_a_perm(perm2, gn)                                   # conv_a's channel order (per kernel mode / an activated copy)
            wk = np.ascontiguousarray(w.reshape(O, C // 32, 32, 3, 3)[:, :, Pm].transpose(0, 1, 3, 4, 2)).reshape(O, 9 * C)
            bias = b.copy()
            if sc is not None:
                ws, bs = sc; Cs = ws.shape[1]; Ks = rup(Cs, 144)
                wsk = np.zeros((O, Ks), np.float32); wsk[:, :Cs] = ws.reshape(O, Cs); wk = np.concatenate([wk, wsk], 1); bias = bias + bs
            wp = np.zeros((Op, wk.shape[1]), np.float32); wp[:O] = wk; bp = np.zeros(Op, np.float32); bp[:O] = bias
            self.panels[key] = (dev(V.b_layout_ref(wp, 36, 4)), dev(bp), Op, wk.shape[1])
            self.t_w += time.perf_counter() - tw
        return self.panels[key]

    def conv(self, key, x, w, b, gab=None, up=False, res=None, sc=None, scw=None, rgb=False):
        """out = conv3x3(act(x)) + b (+ res) (+ the 1 x 1 shortcut of `sc` with weights scw), `up`: x at half resolution."""
        bp, bias, Op, K = self.weights(key, w, b, scw, perm2=getattr(x, "perm2", False), gn=gab is not None)
        W, H = (2 * x.W, 2 * x.H) if up else (x.W, x.H)
        if rgb:                                                          # conv_out: the image, fp16 [H][Wp][4] in [0, 1]
            class _O: pass
            out = _O(); out.Wp = rup(W, 12); out.t = zeros(H * out.Wp * 4, dtypes.float16); out.off = lambda: 0
        else: out = Act(W, H, Op)
        Wp = out.Wp; nsl = K // 144; nsl3 = 9 * x.C // 144
        per_row = Wp * K * 2
        br = H
        while br * per_row > A_MAX or br * Wp * Op * 4 > C_MAX or (br * Wp // 12) % 16: br //= 2
        nrb = br * Wp // 12
        ka = self.kern("conv_a", V.conv_a_src(x.W, x.Wp, x.H, x.C, W, Wp, nrb, gn=gab is not None, up=up, nsl=nsl, nt=NT))
        k1 = self.kern("conv_a1", V.conv_a_src(sc.W, sc.Wp, sc.H, sc.C, W, Wp, nrb, k1=True, s0=nsl3, nsl=nsl, nt=NT)) if sc is not None else None
        kc = self.kern("conv_c", V.conv_c_src(W, Wp, H, Op, nrb, res=res is not None, nt=NT, rgb=rgb))
        da, dc = dev(V.conv_a_descs(x.C, up)), dev(V.conv_c_descs(Op, nrb))
        d1 = dev(V.conv_a_descs(sc.C)) if sc is not None else None
        gab_t = gab if gab is not None else dev(np.zeros(64, np.float32))
        stb = zeros(-(-H // br) * NT * Op * 2, dtypes.float32)
        for y0 in range(0, H, br):
            p0 = y0 * Wp
            # each call REALIZED, in order: a second lazy call on the same buffer drops the first from the graph (it never ran), and
            # chaining the tensors makes gemm_gs's .contiguous() copy the 320 MiB A
            t0 = time.perf_counter()
            OA.csrc_call(ka, self.abuf, x.t, gab_t, da, dev(np.array([p0, x.off()] + [0] * 14, np.int32))).realize()
            if sc is not None: OA.csrc_call(k1, self.abuf, sc.t, gab_t, d1, dev(np.array([p0, sc.off()] + [0] * 14, np.int32))).realize()
            t1 = time.perf_counter()
            ct = OA.gemm_gs(self.abuf, bp, ks=36, ns=4, nrb=nrb, nslices=nsl, ngroups=Op // 64, piece=16, out=self.cbuf).realize()
            t2 = time.perf_counter()
            OA.csrc_call(kc, out.t, ct, bias, res.t if res is not None else out.t, stb, dc, dev(np.array([p0, out.off(), y0 // br] + [0] * 13, np.int32))).realize()
            t3 = time.perf_counter()
            T = self.prof.setdefault((W, x.C, Op), [0.0, 0.0, 0.0, 0.0]); T[0] += t1 - t0; T[1] += t2 - t1; T[2] += t3 - t2; T[3] += 2 * W * H * K * Op / (H // br) / 1e12
        out.st = [stb]                                                   # [band][task][Op][2]: gn_prep reads them all at once
        return out

    def attention(self, x, p):
        """The mid block's attention (1 head of C = 512 over the n = W x H tokens): GroupNorm (the stats' affine, no SiLU) and A(xn) as
        tinygrad ops over the COMPACT tokens (the rows' pad columns dropped; tokens padded to R, masked as keys by softmax_stream's treal),
        ONE q | k | v GEMM (each padded to 576 columns), `attn_qkv` (C tiles + bias -> A(q) in query halves, B(k), B(v^T)), then per
        query half QK^T and PV on gemm_gs (ks 24 x ns 6) around softmax_stream (the scale in q, unnormalized P + 1 / sums) and `attn_o`
        (o x 1 / sum -> the padded rows) -> to_out as a 1 x 1 conv with the residual x."""
        w = self.w; C, W, H = x.C, x.W, x.H; n = W * H; R = rup(n, 96); NRB = R // 12; Kp = rup(C, 96)
        if GN_DEV:
            gp = self.gn_prep(x, w(p + ".group_norm.weight"), w(p + ".group_norm.bias"), "plain"); a_t, b_t = gp[:C], gp[C:]
        else:
            st = sum(OA.host_invalidate(s_).numpy().astype(np.float64) for s_ in x.st)
            mean, rstd = V.conv_c_stats(st, C, n)
            a_c = (np.repeat(rstd, C // 32) * w(p + ".group_norm.weight")).astype(np.float32); b_c = (w(p + ".group_norm.bias") - np.repeat(mean, C // 32) * a_c).astype(np.float32)
            a_t, b_t = dev(a_c), dev(b_c)
        rows = x.t[x.off():x.off() + H * x.Wp * C].reshape(H, x.Wp, C)[:, :W].reshape(n, C).cast(dtypes.float32)
        xn = (rows * a_t + b_t).pad(((0, R - n), (0, Kp - C)))
        A = lambda t: t.cast(dtypes.half).reshape(t.shape[0] // 12, 3, 4, t.shape[1] // 96, 24, 4).permute(3, 0, 4, 1, 2, 5).contiguous().bitcast(dtypes.uint16)
        if p not in self.panels:                                          # q | k | v each padded to 576 columns: a group is one of them
            wqp = np.zeros((3 * Kp, Kp), np.float32); bqp = np.zeros(3 * Kp, np.float32)
            for i_, (nm, sc) in enumerate((("to_q", 1 / np.sqrt(C)), ("to_k", 1.0), ("to_v", 1.0))):   # the scale in q
                wqp[i_ * Kp:i_ * Kp + C, :C] = w(p + "." + nm + ".weight") * sc; bqp[i_ * Kp:i_ * Kp + C] = w(p + "." + nm + ".bias") * sc
            self.panels[p] = (dev(V.b_layout_ref(wqp, 24, 6)), dev(bqp))
        wb, bb = self.panels[p]
        assert NRB % 2 == 0 and Kp == 576; NH = NRB // 2; RH = 12 * NH; SL = R // 96; pc = max(p_ for p_ in range(2, 15, 2) if NH % p_ == 0)
        if getattr(self, "att", None) is None or self.att["R"] != R:     # persistent: the kernels write them, the GEMMs read them
            G = V.conv_guard(x.Wp)
            self.att = dict(R=R, ct=zeros(R * 3 * Kp, dtypes.float32), aq=zeros(R * Kp, dtypes.uint16), bk=zeros(R * Kp, dtypes.uint16),
                            bv=zeros(R * Kp, dtypes.uint16), ot=zeros(RH * Kp, dtypes.float32), sums=[zeros(NH * 16, dtypes.float32) for _ in range(2)],
                            oa=zeros((2 * G + H * x.Wp) * C, dtypes.float16))
        at = self.att
        TP = [time.perf_counter()]; tick = (lambda: TP.append(time.perf_counter())) if os.environ.get("ATT_PROF") else (lambda: None)
        an = A(xn).realize(); tick()
        OA.gemm_gs(an, wb, ks=24, ns=6, nrb=NRB, nslices=Kp // 96, ngroups=3 * Kp // 96, piece=GP.plan_piece(Kp, 3 * Kp, NRB, ks=24, ns=6), out=at["ct"]).realize(); del an; tick()   # gemm_plan: piece 12 at 1024 px (8 was 7.5 % slower)
        kq = self.kern("attn_qkv", V.attn_qkv_src(NRB, NH, SL, nt=NT))
        OA.csrc_call(kq, at["aq"], at["ct"], bb, at["bk"], at["bv"], dev(V.attn_qkv_descs())).realize(); tick()
        # the queries in HALVES, each half's scores (539 MB fp32) and P (270 MB) in two borrowed activation slots (542 MB): the whole
        # 1.6 GB beside the slots fragmented the pool (a second decode could not place its slots, 2026-09-28)
        S_, P_ = POOL.get(NH * SL * 6 * 192 * 2), POOL.get(RH * R)
        ks_ = self.kern("softmax_stream", V.softmax_stream_src(n, 1, R, NH, 24, 1.0, nt=NT, gb=1)); sd = dev(V.softmax_stream_descs(NH, 1))
        ko = self.kern("attn_o", V.attn_o_src(NH, n, W, x.Wp, C, nt=NT)); od = dev(V.attn_o_descs(C))
        oa = Act(W, H, C, t=at["oa"])
        for h in range(2):
            st_ = OA.gemm_gs(at["aq"], at["bk"], ks=24, ns=6, nrb=NH, nslices=Kp // 96, ngroups=SL, piece=pc, a_off=h * RH * Kp, out=S_).realize(); tick()
            pa = OA.csrc_call(ks_, P_, st_, at["sums"][h], sd).realize(); tick()
            OA.gemm_gs(pa, at["bv"], ks=24, ns=6, nrb=NH, nslices=SL, ngroups=Kp // 96, piece=pc, out=at["ot"]).realize(); tick()
            OA.csrc_call(ko, oa.t, at["ot"], at["sums"][h], od, dev(np.array([h * RH, oa.off()] + [0] * 14, np.int32))).realize(); tick()
            del st_, pa
        POOL.put(S_); POOL.put(P_)
        r = self.conv1x1(p + ".to_out", oa, w(p + ".to_out.0.weight"), w(p + ".to_out.0.bias"), res=x); tick()
        if len(TP) > 1: print("   attention: " + "  ".join("%s %.0f" % (n_, (b_ - a_) * 1e3) for n_, a_, b_ in zip(("A(xn)", "qkv gemm", "attn_qkv") + ("QK^T", "softmax", "PV", "attn_o") * 2 + ("to_out",), TP, TP[1:])) + " ms", flush=True)
        return r

    def conv1x1(self, key, x, w, b, res=None):
        """A 1 x 1 conv (conv_a1 -> the GEMM -> conv_c): out = x W^T + b (+ res)."""
        if key not in self.panels:
            O, C = w.shape[:2]; Op = rup(O, 64); Kp = rup(C, 144); wk = np.zeros((Op, Kp), np.float32); wk[:O, :C] = w.reshape(O, C)
            bp_ = np.zeros(Op, np.float32); bp_[:O] = b
            self.panels[key] = (dev(V.b_layout_ref(wk, 36, 4)), dev(bp_), Op, Kp)
        bp, bias, Op, K = self.panels[key]
        out = Act(x.W, x.H, Op); Wp = out.Wp; nsl = K // 144; H = x.H
        br = H
        while br * Wp * K * 2 > A_MAX or br * Wp * Op * 4 > C_MAX or (br * Wp // 12) % 16: br //= 2
        nrb = br * Wp // 12
        k1 = self.kern("conv_a1", V.conv_a_src(x.W, x.Wp, x.H, x.C, x.W, Wp, nrb, k1=True, nsl=nsl, nt=NT))
        kc = self.kern("conv_c", V.conv_c_src(x.W, Wp, H, Op, nrb, res=res is not None, nt=NT))
        d1, dc = dev(V.conv_a_descs(x.C)), dev(V.conv_c_descs(Op, nrb)); gz = dev(np.zeros(64, np.float32))
        stb = zeros(-(-H // br) * NT * Op * 2, dtypes.float32)
        for y0 in range(0, H, br):
            p0 = y0 * Wp
            OA.csrc_call(k1, self.abuf, x.t, gz, d1, dev(np.array([p0, x.off()] + [0] * 14, np.int32))).realize()
            ct = OA.gemm_gs(self.abuf, bp, ks=36, ns=4, nrb=nrb, nslices=nsl, ngroups=Op // 64, piece=16, out=self.cbuf).realize()
            OA.csrc_call(kc, out.t, ct, bias, res.t if res is not None else out.t, stb, dc, dev(np.array([p0, out.off(), y0 // br] + [0] * 13, np.int32))).realize()
        out.st = [stb]                                                   # [band][task][Op][2]: gn_prep reads them all at once
        return out

    def resnet(self, x, p):
        w = self.w; has_sc = self.D.W.has(p + ".conv_shortcut.weight")
        if os.environ.get("FUSED_GN") == "1":                              # the GroupNorm + SiLU fused into conv_a (recomputed per neighbourhood)
            h = self.conv(p + ".conv1", x, w(p + ".conv1.weight"), w(p + ".conv1.bias"), gab=x.gn(w(p + ".norm1.weight"), w(p + ".norm1.bias")))
            g2, h2 = h.gn(w(p + ".norm2.weight"), w(p + ".norm2.bias")), h
        else:
            h = self.conv(p + ".conv1", self.gn_silu(x, w(p + ".norm1.weight"), w(p + ".norm1.bias")), w(p + ".conv1.weight"), w(p + ".conv1.bias"))
            g2, h2 = None, self.gn_silu(h, w(p + ".norm2.weight"), w(p + ".norm2.bias")); h = None     # h's slot back: x, h2, out
        if has_sc:
            return self.conv(p + ".conv2", h2, w(p + ".conv2.weight"), w(p + ".conv2.bias"), gab=g2, sc=x,
                             scw=(w(p + ".conv_shortcut.weight"), w(p + ".conv_shortcut.bias")))
        return self.conv(p + ".conv2", h2, w(p + ".conv2.weight"), w(p + ".conv2.bias"), gab=g2, res=x)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("latents"); ap.add_argument("--ref", default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--runs", type=int, default=1, help="decodes in one process: the first checked (--ref) and cold, the rest warm")
    a = ap.parse_args()
    vae = VAE()
    for run in range(a.runs): decode(vae, a.latents, np.load(a.ref) if a.ref and run == 0 else None, a.out if run == a.runs - 1 else None, run)


def decode(vae, latents, ref, out, run):
    D = vae.D; w = vae.w; vae.prof = {}; vae.t_w = vae.t_gn = vae.t_chk = vae.t_host = 0.0
    def check(name, act):
        tc = time.perf_counter(); _check(name, act); vae.t_chk += time.perf_counter() - tc
        if os.environ.get("STOP_AFTER") == name: raise SystemExit(0)
    def _check(name, act):
        if os.environ.get("MEMLOG"):
            from tinygrad.helpers import GlobalCounters
            print("   [mem] %-12s %.0f MB" % (name, GlobalCounters.mem_used_per_device.get(DEV, 0) / 2**20), flush=True)
        if ref is None: return
        if name in ref.files: want = ref[name]; got = act.rows()
        else: idx = ref[name + "_rowidx"]; want = ref[name + "_rows"]; got = act.rows(idx)
        got = got[:want.shape[0]]
        print("   %-12s %s  max rel |d| %.3g  mean rel |d| %.3g" % (name, (act.C, act.H, act.W), np.abs(got - want).max() / np.abs(want).max(),
              np.abs(got - want).mean() / np.abs(want).mean()), flush=True)
    t0 = time.perf_counter()
    z = D.unpack(np.load(latents), (64, 64))
    if PQ_DEV:                                                           # 1 x 1, 32 -> 32 (padded to 64) at 128^2 on the device; conv_in's
        z = vae.conv1x1("post_quant", Act.from_chw(z), w("post_quant_conv.weight"), w("post_quant_conv.bias"))   # input channels padded
        wci = w("decoder.conv_in.weight"); wci = np.concatenate([wci, np.zeros((wci.shape[0], z.C - wci.shape[1], 3, 3), np.float32)], 1)
    else:
        z = Act.from_chw(VA.conv2d(z, w("post_quant_conv.weight"), w("post_quant_conv.bias"), 0)); wci = w("decoder.conv_in.weight")
    t_pq = time.perf_counter() - t0
    x = vae.conv("conv_in", z, wci, w("decoder.conv_in.bias")); del z; check("conv_in", x)
    x = vae.resnet(x, "decoder.mid_block.resnets.0"); check("mid_res0", x)
    ta = time.perf_counter(); x = vae.attention(x, "decoder.mid_block.attentions.0"); ta = time.perf_counter() - ta; check("mid_attn", x)
    x = vae.resnet(x, "decoder.mid_block.resnets.1"); check("mid_res1", x)
    nblk = len(D.cfg["block_out_channels"])
    for i in range(nblk):
        for r in range(D.cfg["layers_per_block"] + 1): x = vae.resnet(x, f"decoder.up_blocks.{i}.resnets.{r}"); check(f"up{i}_res{r}", x)
        if i < nblk - 1:
            p = f"decoder.up_blocks.{i}.upsamplers.0.conv"
            x = vae.conv(p, x, w(p + ".weight"), w(p + ".bias"), up=True); check(f"up{i}_up", x)
    H_, W_ = x.H, x.W
    rgb = vae.conv("conv_out", vae.gn_silu(x, w("decoder.conv_norm_out.weight"), w("decoder.conv_norm_out.bias")), w("decoder.conv_out.weight"), w("decoder.conv_out.bias"), rgb=True)
    del x
    tr = time.perf_counter(); img = OA.host_invalidate(rgb.t).numpy().reshape(H_, rup(W_, 12), 4)[:, :W_, :3].astype(np.float32); tr = time.perf_counter() - tr
    print("   run %d (%s): decoded %s in %.1f s (%.1f s without the checks)" % (run, "cold" if run == 0 else "warm", img.shape, time.perf_counter() - t0, time.perf_counter() - t0 - vae.t_chk), flush=True)
    print("   host: latents + post-quant %.0f ms, GN tables %.0f ms, image read-back + clip %.0f ms" % (t_pq * 1e3, vae.t_host * 1e3, tr * 1e3))
    tot = [sum(v[i] for v in vae.prof.values()) for i in range(4)]
    print("   convs: im2col %.1f s, GEMM %.1f s (%.1f TFLOP), conv_c %.1f s; weights prep %.1f s; attention %.1f s; gn_silu %.1f s" % (tot[0], tot[1], tot[3], tot[2], vae.t_w, ta, vae.t_gn))
    for k_, v in sorted(vae.prof.items(), key=lambda kv: -sum(kv[1][:3])):
        print("     W %4d  %3d -> %3d: im2col %5.2f  GEMM %5.2f (%.2f TFLOP/s)  conv_c %5.2f" % (k_[0], k_[1], k_[2], v[0], v[1], v[3] / max(v[1], 1e-9), v[2]))
    if ref is not None:
        d = np.abs(img - ref["image"]); mse = (d ** 2).mean()
        print("   image vs the numpy decode: max |d| %.3g, PSNR %.1f dB" % (d.max(), 10 * np.log10(1 / mse)))
    if out: VA.save_png(img, out)


if __name__ == "__main__":
    try: main()
    except MemoryError:
        if not os.environ.get("MEMLOG"): raise
        raw = OA.Device[DEV].raw; got = []                              # what is left in the pool, and the largest single block
        for mb in (1024, 512, 256, 128, 64):
            try: b_ = raw.req_buf(mb << 20, OA.MM_REUSE); raw.free_buf(b_); print("   [mem] largest block >= %d MB" % mb); break
            except OSError: pass
        try:
            while True: got.append(raw.req_buf(16 << 20, OA.MM_REUSE))
        except OSError: pass
        print("   [mem] free in 16 MB pieces: %d MB" % (16 * len(got))); [raw.free_buf(b_) for b_ in got]
        raise

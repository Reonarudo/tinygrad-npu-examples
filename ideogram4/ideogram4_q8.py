#!/usr/bin/env python3
"""Ideogram 4's block linears as W8A8 on the AIFF int8 engines: the plan of one AIFF linear, the int8
weight pack, the descriptor templates' relocation, the two TEC glue kernels and their numpy references.

The scheme (the simulator's reference computes the same):

  activations  int8 per token: the producer's fp16 A operand `a` (SmoothQuant's 1/lam already folded into its per-channel
               multiplier) -> q = clamp(rint(a * (127 / amax_m)), +-127), s_x[m] = amax_m / 127      (`q8_a16`, a TEC kernel)
  weights      int8 per output channel: Wq = clamp(rint(W lam / s_w)), s_w = max_k |W lam| / 127              (`pack_weights`)
  the AIFF     per K pass s (2304 at K 4608, 4096 at K 12288) and 128-channel unit u:
               P_s = sat16(round_half_even(acc32 >> sh_su))    (int16, NHWC [M_pad][S][N]: a row's passes side by side)
  dequant      y[m, n] = s_x[m] * sum_s P_s[m, n] * cs_s[n],  cs_s[n] = s_w[n] 2^sh_s(n) (x 1/lam of the NEXT linear where it
               folds: the V columns of qkv carry o's, the w3 columns carry w2's when w2 is int8)  (`dq_ct`, a TEC kernel)
               -> the fp32 C tiles the existing consumers (qkv_dma, resid, swiglu) read, with their column scales set to 1.

AIFF linear geometry (`Plan`): the M rows are an h x 8 plane (w = 8, the prototype's), padded to a multiple of h_step x 8, in
chunks of <= 7 steps (h <= 255 is the engine's field width); N in nodes of 256 (two 128-channel units), spread over the three
engines (<= 32 nodes a chain); one descriptor chain per (engine, chunk, K pass). Activations [K/32][M_pad][32] int8 (the MTP's
surface layout), weights [S][node][unit][dense 256 x 32 tiles], outputs [M_pad][S][N] int16.
"""
from __future__ import annotations
import ctypes, os
import numpy as np

W_PLANE, SURF, UNIT, NODE = 8, 32, 128, 256
CHUNK_STEPS = 7                     # h = steps x h_step <= 255 (8-bit field) at h_step 32
ENGINES = 3


def pass_k(K: int) -> int:
    return {4608: 2304, 12288: 4096}.get(K, K)


class Plan:
    """One AIFF linear `[rows, K] x [K, N]` (see the module docstring)."""
    def __init__(self, rows: int, K: int, N: int, hstep: int = 32, engines: int = ENGINES):
        g = hstep * W_PLANE
        self.rows, self.K, self.N, self.hstep = rows, K, N, hstep
        self.M_pad = -(-rows // g) * g
        self.H = self.M_pad // W_PLANE
        self.kc = pass_k(K); assert K % self.kc == 0 and self.kc % 256 == 0
        self.S = K // self.kc
        assert N % NODE == 0; self.nodes = N // NODE
        T = self.H // hstep; nch = -(-T // (255 // hstep))                 # a chunk's h <= 255 (8-bit field)
        self.chunks, h0 = [], 0
        for i in range(nch):
            st = T // nch + (1 if i < T % nch else 0); self.chunks.append((h0, st * hstep)); h0 += st * hstep
        per = [self.nodes // engines + (1 if e < self.nodes % engines else 0) for e in range(engines)]
        assert max(per) <= 32, "a chain holds at most 32 nodes"
        self.engine_nodes, j = [], 0
        for p in per: self.engine_nodes.append((j, j + p)); j += p
        self.chains = [(e, c, s) for e in range(engines) for c in range(len(self.chunks)) for s in range(self.S)]

    @property
    def key(self) -> str: return "m%d_k%d_n%d_h%d" % (self.M_pad, self.K, self.N, self.hstep)
    @property
    def a8_bytes(self) -> int: return self.K * self.M_pad
    @property
    def p_bytes(self) -> int: return self.M_pad * self.S * self.N * 2
    @property
    def w_bytes(self) -> int: return self.K * self.N
    @property
    def macs(self) -> int: return self.M_pad * self.K * self.N


# ------------------------------------------------------------------------------------------------ weights ----
def quant_weights(W: np.ndarray, lam: np.ndarray):
    """W fp32 [N, K] (the dequantised checkpoint weights), lam fp32 [K] -> (Wq int8 [N, K], s_w fp32 [N]). The simulator's
    `dev_weights` in numpy: s_w = max|W lam| / 127 (fp32), Wq = clamp(rint(W lam / s_w))."""
    Ws = (W.astype(np.float32) * lam.astype(np.float32)[None, :]).astype(np.float32)
    sw = (np.maximum(np.abs(Ws).max(1), np.float32(1e-12)) / np.float32(127.0)).astype(np.float32)
    return np.clip(np.rint(Ws / sw[:, None]), -127, 127).astype(np.int8), sw


def pack_weights(wq: np.ndarray, plan: Plan) -> np.ndarray:
    """int8 [N, K] -> the AIFF weight buffer [S][node][unit][kc/256][128/32][256][32] (each unit's tile in the MTP's dense
    layout: `aiff_gen.dense_offsets` -- 256 input channels x 32 output channels blocks, output-channel block inner)."""
    N, K = wq.shape; assert (N, K) == (plan.N, plan.K)
    t = wq.reshape(plan.nodes, 2, UNIT // 32, 32, plan.S, plan.kc // 256, 256)        # [j][u][ob][oc][s][cb][ci]
    return np.ascontiguousarray(t.transpose(4, 0, 1, 5, 2, 6, 3)).reshape(-1).view(np.uint8)


def unpack_weights(buf: np.ndarray, plan: Plan) -> np.ndarray:
    """`pack_weights`' inverse: the AIFF weight buffer -> int8 [N, K]."""
    t = np.asarray(buf).view(np.int8).reshape(plan.S, plan.nodes, 2, plan.kc // 256, UNIT // 32, 256, 32)   # [s][j][u][cb][ob][ci][oc]
    return np.ascontiguousarray(t.transpose(1, 2, 4, 6, 0, 3, 5)).reshape(plan.N, plan.K)


# ------------------------------------------------------------------------------------------ descriptors ----
class Desc:
    """A linear's descriptor template (`ideogram4_aiff_desc.py`'s npz) and its relocation: every chain's words with the
    self-relative words, the activation / weight / output address words and the per-(pass, unit) shift fields patched."""
    def __init__(self, path: str):
        d = np.load(path)
        self.words = d["words"].astype(np.uint32)
        self.chains = d["chains"].astype(np.int64)                 # [n, 4]: engine, ctrl, param, act byte offsets
        self.rel = {k: d["rel_" + k].astype(np.int64) // 4 for k in ("self", "a", "w", "o")}
        self.t_word, self.t_shift, self.t_idx = d["trsh_off"].astype(np.int64) // 4, d["trsh_shift"].astype(np.int64), d["trsh_idx"].astype(np.int64)
        self.meta = {k: int(d[k]) for k in ("M_pad", "K", "N", "hstep")}
        self.nbytes = self.words.nbytes
        self.gm = "gm" in d.files                                   # GM-staged: one node a chain, weights in GM slots
        if self.gm: self.info, self.w_chain = d["chain_info"].astype(np.int64), d["rel_w_chain"].astype(np.int64)

    def build(self, desc_addr: int, a_addr: int, w_addr: int, o_addr: int, sh: np.ndarray) -> np.ndarray:
        """-> the uint32 words for a descriptor buffer at `desc_addr` (device addresses: ASID0 offsets), shifts `sh`
        int [S, N/128]."""
        w = self.words.astype(np.int64)
        for k, base in (("self", desc_addr), ("a", a_addr), ("o", o_addr)): w[self.rel[k]] += base
        w[self.rel["w"]] += np.asarray(w_addr, np.int64)[self.w_chain] if self.gm else w_addr     # GM: a base per chain (its node's slot)
        v = np.asarray(sh, np.int64).ravel()[self.t_idx]
        assert v.min() >= 0 and v.max() < 128, "a shift outside the 7-bit mul_trsh field"
        m = np.int64(0x7F) << self.t_shift
        w[self.t_word] = (w[self.t_word] & ~m) | (v << self.t_shift)
        return (w & 0xFFFFFFFF).astype(np.uint32)

    def kicks(self, desc_addr: int):
        """Per engine: the chains' (ctrl, param, act) device addresses, in order."""
        out = [[] for _ in range(int(self.chains[:, 0].max()) + 1)]
        for e, c, p, a in self.chains.tolist(): out[e].append((desc_addr + c, desc_addr + p, desc_addr + a))
        return out


def gm_slot_bases(D: "Desc", gm0: int, plan: Plan) -> np.ndarray:
    """GM-staged descriptors: each chain's weight base -- engine e's round r reads slot (e, r % 2) of 2 x kc x 128 bytes from gm0."""
    tile = 2 * plan.kc * UNIT
    return gm0 + (D.info[:, 0] * 2 + D.info[:, 1] % 2) * tile


def desc_path(plan: Plan, ddir: str, gm: bool = False, ntot: int|None = None, coff: int = 0) -> str:
    return os.path.join(os.path.expanduser(ddir), "aiffdesc_%s%s%s.npz" % (plan.key, "_gm" if gm else "", "_c%do%d" % (ntot, coff) if ntot else ""))


# --------------------------------------------------------------------------------------- numpy references ----
def to_surface(x: np.ndarray, M_pad: int) -> np.ndarray:
    """[rows, K] -> [K/32][M_pad][32] (rows past `rows` zero)."""
    rows, K = x.shape; out = np.zeros((K // SURF, M_pad, SURF), x.dtype)
    out[:, :rows] = x.reshape(rows, K // SURF, SURF).transpose(1, 0, 2); return out


def q8_ref(a: np.ndarray):
    """The quantiser on fp32 values (exactly representable fp16): [rows, K] -> (q int8 [rows, K], s_x fp32 [rows])."""
    a = a.astype(np.float32); amax = np.abs(a).max(1)
    r = np.where(amax > 0, np.float32(127.0) / np.where(amax > 0, amax, np.float32(1)), np.float32(0)).astype(np.float32)
    q = np.clip(np.rint(a * r[:, None]), -127, 127).astype(np.int8)
    return q, (amax / np.float32(127.0)).astype(np.float32)


def aiff_ref(q: np.ndarray, wq: np.ndarray, sh: np.ndarray, kc: int) -> np.ndarray:
    """The engine, exactly: q int8 [M, K], wq int8 [N, K], sh [S, N/128] -> P int16 [M, S, N]."""
    M, K = q.shape; N = wq.shape[0]; S = K // kc
    out = np.empty((M, S, N), np.int16)
    for s in range(S):
        acc = q[:, s * kc:(s + 1) * kc].astype(np.int64) @ wq[:, s * kc:(s + 1) * kc].astype(np.int64).T
        out[:, s] = np.clip(shift_round_even(acc, np.repeat(np.asarray(sh[s], np.int64), UNIT)[None, :]), -32768, 32767)
    return out


def shift_round_even(acc: np.ndarray, sh: np.ndarray) -> np.ndarray:
    """acc >> sh with round half to even (sh >= 0, per column)."""
    acc = np.asarray(acc, np.int64); sh = np.broadcast_to(np.asarray(sh, np.int64), acc.shape)
    q = acc >> sh; rem = acc - (q << sh); half = np.where(sh > 0, np.int64(1) << np.maximum(sh - 1, 0), 0)
    up = (sh > 0) & ((rem > half) | ((rem == half) & ((q & 1) == 1)))
    return q + up


def dq_ref(P: np.ndarray, sx: np.ndarray, cs: np.ndarray) -> np.ndarray:
    """P int16 [rows, S, N], sx fp32 [rows], cs fp32 [S, N] -> fp32 [rows, N]: acc = fma(P_s, cs_s, acc) per pass (one rounding
    each, as the TEC's contracted multiply-add), then x s_x."""
    acc = np.zeros(P.shape[::2], np.float32)
    for s in range(P.shape[1]): acc = (acc.astype(np.float64) + P[:, s].astype(np.float64) * cs[s].astype(np.float64)).astype(np.float32)
    return (acc * sx[:, None].astype(np.float32)).astype(np.float32)


def shifts_for(acc_max: np.ndarray, headroom: float = 2.0) -> np.ndarray:
    """|acc| maxima per channel [S, N] -> the per-(pass, unit) shift [S, N/128] (the simulator's rule)."""
    mx = acc_max.reshape(acc_max.shape[0], -1, UNIT).max(2).astype(np.float64) * headroom
    return np.maximum(0, np.ceil(np.log2(np.maximum(mx, 1.0) / 32767.0))).astype(np.int8)


def scales_cs(sw: np.ndarray, sh: np.ndarray, extra: np.ndarray|None = None) -> np.ndarray:
    """cs [S, N] fp32 = s_w 2^sh (x extra)."""
    cs = (sw.astype(np.float32)[None, :] * np.exp2(np.repeat(np.asarray(sh, np.float32), UNIT, axis=1))).astype(np.float32)
    if extra is not None: cs = (cs * extra.astype(np.float32)[None, :]).astype(np.float32)
    return np.ascontiguousarray(cs)


# ------------------------------------------------------------------------------------------- TEC kernels ----
def _hdr():
    from zy import vec_f16 as _V
    return _V.FULL_H, _V._desc_slots


def q8_a16_src(nrb: int, K: int, M_pad: int, nt: int = 12, spr: int = 4) -> str:
    """The quantiser: the fp16 A layout ([slice][nrb][ks 24][3][4][4] halves, `gemm_fp16.pack_a_slices`) -> the int8 surfaces
    `[K/32][M_pad][32]` and s_x `[nrb][16]` fp32 (a row block's 12, then 4 zeros). A unit = a row block: pass 1 streams its K/96
    slices (`spr` slices a request: one strided fill of spr x 2304 B, double-buffered, flags 0/1) for the rows' |max|, pass 2
    streams them again and writes each request's 3 spr surfaces (12 rows x 32 B each, one strided drain, flags 2/3).
    args: a8, a, sx, desc (`q8_a16_descs`)."""
    FULL_H, _ = _hdr()
    nsl = K // 96; assert nsl % spr == 0
    nrq, FB_, OB = nsl // spr, spr * 2304, spr * 1152
    o_out, o_sx = 2 * FB_, 2 * FB_ + 2 * OB
    assert o_sx + 64 <= 32768
    def rows_pair(rp):
        r0 = 2 * rp
        idx = []
        for half in (0, 16):                   # tile a (kq), tile b (kq + 1)
            for k in range(4): idx += [half + 4 * r0 + k, half + 4 * (r0 + 1) + k]
        return ", ".join(str(x) for x in idx)
    pair_code = ""
    for rp in range(2):
        pair_code += f"""
          {{ /* rows {2 * rp}, {2 * rp + 1} of the row group */
            float8 f0[4], f1[4];
            for (int tp = 0; tp < 4; tp++) {{
              half16 ta = *(__global half16*)(blk + (((8 * j + 2 * tp) * 3 + i) * 16)), tb = *(__global half16*)(blk + (((8 * j + 2 * tp + 1) * 3 + i) * 16));
              half16 hv = __builtin_shufflevector(ta, tb, {rows_pair(rp)});
              f0[tp] = __builtin_aipu_cvtue_tfp32_tfp16(hv); f1[tp] = __builtin_aipu_cvtuo_tfp32_tfp16(hv);
            }}
            float8 R0 = BC(RR[4 * i + {2 * rp}]), R1 = BC(RR[4 * i + {2 * rp + 1}]);
            *(__global char32*)(O + ((q * 3 + j) * 12 + 4 * i + {2 * rp}) * 32) = pack32(q8v(f0[0] * R0), q8v(f0[1] * R0), q8v(f0[2] * R0), q8v(f0[3] * R0));
            *(__global char32*)(O + ((q * 3 + j) * 12 + 4 * i + {2 * rp + 1}) * 32) = pack32(q8v(f1[0] * R1), q8v(f1[1] * R1), q8v(f1[2] * R1), q8v(f1[3] * R1));
          }}"""
    return FULL_H + f"""
__kernel void q8_a16(__global char* restrict a8, __global half* restrict a, __global float* restrict sx, __global int* restrict desc, const int core_id) {{
  for (int rb = core_id; rb < {nrb}; rb += {nt}) {{
    float8 me0 = BC(0.0f), me1 = BC(0.0f), me2 = BC(0.0f), mo0 = BC(0.0f), mo1 = BC(0.0f), mo2 = BC(0.0f);
    DMA_FILL(0, DESC(desc, 0), 0, (int)(a + rb * 1152));
    for (int rq = 0; rq < {nrq}; rq++) {{
      int s = rq & 1;
      if (rq + 1 < {nrq}) DMA_FILL(s ^ 1, DESC(desc, 0), (s ^ 1) * {FB_}, (int)(a + ((rq + 1) * {spr * nrb} + rb) * 1152));
      DMA_WAIT(s);
      __global half* blk0 = LSH(s * {FB_});
      for (int kq = 0; kq < {24 * spr}; kq++) {{
        half16 t0 = *(__global half16*)(blk0 + (kq * 3 + 0) * 16), t1 = *(__global half16*)(blk0 + (kq * 3 + 1) * 16), t2 = *(__global half16*)(blk0 + (kq * 3 + 2) * 16);
        float8 e0 = __builtin_aipu_cvtue_tfp32_tfp16(t0), o0 = __builtin_aipu_cvtuo_tfp32_tfp16(t0);
        float8 e1 = __builtin_aipu_cvtue_tfp32_tfp16(t1), o1 = __builtin_aipu_cvtuo_tfp32_tfp16(t1);
        float8 e2 = __builtin_aipu_cvtue_tfp32_tfp16(t2), o2 = __builtin_aipu_cvtuo_tfp32_tfp16(t2);
        me0 = VMAX(me0, VMAX(e0, -e0)); mo0 = VMAX(mo0, VMAX(o0, -o0));
        me1 = VMAX(me1, VMAX(e1, -e1)); mo1 = VMAX(mo1, VMAX(o1, -o1));
        me2 = VMAX(me2, VMAX(e2, -e2)); mo2 = VMAX(mo2, VMAX(o2, -o2));
      }}
    }}
    float RR[12];
    __global float* SXO = LSF({o_sx});
    for (int r = 0; r < 4; r++) {{
      float m0 = me0[2 * r] > me0[2 * r + 1] ? me0[2 * r] : me0[2 * r + 1]; float n0 = mo0[2 * r] > mo0[2 * r + 1] ? mo0[2 * r] : mo0[2 * r + 1]; m0 = m0 > n0 ? m0 : n0;
      float m1 = me1[2 * r] > me1[2 * r + 1] ? me1[2 * r] : me1[2 * r + 1]; float n1 = mo1[2 * r] > mo1[2 * r + 1] ? mo1[2 * r] : mo1[2 * r + 1]; m1 = m1 > n1 ? m1 : n1;
      float m2 = me2[2 * r] > me2[2 * r + 1] ? me2[2 * r] : me2[2 * r + 1]; float n2 = mo2[2 * r] > mo2[2 * r + 1] ? mo2[2 * r] : mo2[2 * r + 1]; m2 = m2 > n2 ? m2 : n2;
      RR[r] = m0 > 0.0f ? 127.0f / m0 : 0.0f; RR[4 + r] = m1 > 0.0f ? 127.0f / m1 : 0.0f; RR[8 + r] = m2 > 0.0f ? 127.0f / m2 : 0.0f;
      SXO[r] = m0 / 127.0f; SXO[4 + r] = m1 / 127.0f; SXO[8 + r] = m2 / 127.0f; SXO[12 + r] = 0.0f;
    }}
    DMA_DRAIN(2, DESC(desc, 2), {o_sx}, (int)(sx + rb * 16)); DMA_WAIT(2);
    DMA_FILL(0, DESC(desc, 0), 0, (int)(a + rb * 1152));
    for (int rq = 0; rq < {nrq}; rq++) {{
      int s = rq & 1;
      if (rq + 1 < {nrq}) DMA_FILL(s ^ 1, DESC(desc, 0), (s ^ 1) * {FB_}, (int)(a + ((rq + 1) * {spr * nrb} + rb) * 1152));
      DMA_WAIT(s); DMA_WAIT(2 + s);
      __global char* O = (__global char*)(LSRAM_BASE + {o_out} + s * {OB});
      for (int q = 0; q < {spr}; q++) {{
        __global half* blk = LSH(s * {FB_} + q * 2304);
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {{{pair_code}
        }}
      }}
      DMA_DRAIN(2 + s, DESC(desc, 1), {o_out} + s * {OB}, (int)(a8 + (rq * {3 * spr} * {M_pad} + rb * 12) * 32));
    }}
    DMA_WAIT_ALL();
  }}
}}"""


def q8_a16_descs(M_pad: int, nrb: int, spr: int = 4) -> np.ndarray:
    _, slots = _hdr()
    return slots((spr * 2304, 2304, nrb * 2304, 2304), (spr * 1152, 384, M_pad * 32, 384), (64,))


def dq_ct_src(nrb: int, ng: int, S: int, N: int, M_pad: int, gmul: int = 1, gofs: int = 0, nt: int = 12, mode: str = "full") -> str:
    """The dequantiser: the AIFF's int16 partials P `[M_pad][S][N]`, s_x `[nrb][16]`, cs `[S][N]` fp32 -> fp32 C tiles (the TEC
    GEMMs' layout, ns 6: a (group, row block) is 4608 contiguous bytes) of `ng` 96-column groups; the part's group g lands at
    tile group `gmul * g + gofs` (w1|w3: the two halves alternate, swiglu_dma's interleaved order). A unit = (g, rb): one
    strided fill of 12 rows x S passes x 192 B (flags 0/1), its sx / cs (flag 2/3, waited), one 4608-B drain (flags 2/3).
    args: ct, p, sx, cs, desc (`dq_ct_descs`)."""
    FULL_H, _ = _hdr()
    PB = S * 2304
    AUX = S * 384 + 64                                  # a unit's cs (S x 96 floats) then its sx (16 floats); double-buffered
    o_aux = 2 * PB
    o_out = (o_aux + 2 * AUX + 255) // 256 * 256
    assert o_out + 2 * 4608 <= 32768
    rows4 = ""
    for r in range(4):
        rows4 += "\n          { __global short* PR = PBUF + (4 * i + %d) * %d + 16 * q2; float8 l_ = BC(0.0f), h_ = BC(0.0f);" % (r, S * 96)
        for k in range(S):
            rows4 += " { short16 v = *(__global short16*)(PR + %d); l_ += VF(__builtin_aipu_sxtl_th(v)) * c%d0; h_ += VF(__builtin_aipu_sxth_th(v)) * c%d1; }" % (k * 96, k, k)
        rows4 += " float8 sv = BC(SX[4 * i + %d]); lo%d = l_ * sv; hi%d = h_ * sv; }" % (r, r, r)
    cload = " ".join("float8 c%d0 = *(__global float8*)(CS + %d + 16 * q2), c%d1 = *(__global float8*)(CS + %d + 16 * q2 + 8);" % (k, k * 96, k, k * 96) for k in range(S))
    return FULL_H + f"""
__kernel void dq_ct(__global float* restrict ct, __global short* restrict p, __global float* restrict sx, __global float* restrict cs, __global int* restrict desc, const int core_id) {{
  /* flags: 0 / 1 the P fills (double-buffered), 2 the drains, 3 the next unit's cs / sx (aux, double-buffered) */
  int k = 0, csg0 = -1, csg1 = -1;
  if (core_id < {ng * nrb}) {{
    int g = core_id / {nrb}, rb = core_id % {nrb};
    DMA_FILL(0, DESC(desc, 0), 0, (int)(p + (rb * 12 * {S} * {N}) + g * 96));
    DMA_FILL(3, DESC(desc, 1), {o_aux}, (int)(cs + g * 96)); DMA_WAIT(3); csg0 = g;
    DMA_FILL(3, DESC(desc, 2), {o_aux + S * 384}, (int)(sx + rb * 16));
  }}
  for (int u = core_id; u < {ng * nrb}; u += {nt}, k++) {{
    int g = u / {nrb}, rb = u % {nrb}, s = k & 1;
    DMA_WAIT(3); DMA_WAIT(s);
    int un = u + {nt};
    if (un < {ng * nrb}) {{
      int gn = un / {nrb}, rbn = un % {nrb};
      DMA_FILL(s ^ 1, DESC(desc, 0), (s ^ 1) * {PB}, (int)(p + (rbn * 12 * {S} * {N}) + gn * 96));
      int have = s ? csg0 : csg1;
      if (have != gn) {{ DMA_FILL(3, DESC(desc, 1), {o_aux} + (s ^ 1) * {AUX}, (int)(cs + gn * 96)); DMA_WAIT(3); if (s) csg0 = gn; else csg1 = gn; }}
      DMA_FILL(3, DESC(desc, 2), {o_aux} + (s ^ 1) * {AUX} + {S * 384}, (int)(sx + rbn * 16));
    }}
    __global short* PBUF = (__global short*)(LSRAM_BASE + s * {PB}); __global float* CS = LSF({o_aux} + s * {AUX}); __global float* SX = CS + {S * 96};
    __global float* OUT = LSF({o_out} + s * 4608);
    for (int q2 = 0; q2 < {6 if mode != "nomath" else 0}; q2++) {{
      {cload}
      for (int i = 0; i < 3; i++) {{
        float8 lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3;{rows4}
        __global float* T = OUT + (q2 * 12 + i * 4) * 16;
        *(__global float8*)(T) = __builtin_shufflevector(lo0, lo1, 0, 1, 2, 3, 8, 9, 10, 11); *(__global float8*)(T + 8) = __builtin_shufflevector(lo2, lo3, 0, 1, 2, 3, 8, 9, 10, 11);
        *(__global float8*)(T + 16) = __builtin_shufflevector(lo0, lo1, 4, 5, 6, 7, 12, 13, 14, 15); *(__global float8*)(T + 24) = __builtin_shufflevector(lo2, lo3, 4, 5, 6, 7, 12, 13, 14, 15);
        *(__global float8*)(T + 32) = __builtin_shufflevector(hi0, hi1, 0, 1, 2, 3, 8, 9, 10, 11); *(__global float8*)(T + 40) = __builtin_shufflevector(hi2, hi3, 0, 1, 2, 3, 8, 9, 10, 11);
        *(__global float8*)(T + 48) = __builtin_shufflevector(hi0, hi1, 4, 5, 6, 7, 12, 13, 14, 15); *(__global float8*)(T + 56) = __builtin_shufflevector(hi2, hi3, 4, 5, 6, 7, 12, 13, 14, 15);
      }}
    }}
    DMA_WAIT(2);
    DMA_DRAIN(2, DESC(desc, 3), {o_out} + s * 4608, (int)(ct + (({gmul} * g + {gofs}) * {nrb} + rb) * 1152));
  }}
  DMA_WAIT_ALL();
}}"""


def dq_ct_descs(S: int, N: int) -> np.ndarray:
    _, slots = _hdr()
    return slots((12 * S * 192, 192, N * 2, 192), (S * 384, 384, N * 4, 384), (64,), (4608,))


def ct_from_rows(y: np.ndarray, nrb: int, gmul: int = 1, gofs: int = 0, ng_total: int|None = None, ct: np.ndarray|None = None) -> np.ndarray:
    """numpy: fp32 [12 nrb, Np] (a part's columns) -> written into C tiles (`ct`, floats) at tile groups gmul g + gofs."""
    rows, Np = y.shape; ng = Np // 96
    if ct is None: ct = np.zeros((ng_total or ng) * nrb * 1152, np.float32)
    t = y.reshape(nrb, 3, 4, ng, 6, 4, 4).transpose(3, 0, 4, 1, 5, 2, 6).reshape(ng, nrb * 1152)   # [g][rb][s][i][jt][r][c]
    cv = ct.reshape(-1, nrb * 1152)
    for g in range(ng): cv[gmul * g + gofs] = t[g]
    return ct


LOG2E = 1.4426950408889634


def swiglu_i16_src(nrb: int, S: int, M: int, M_pad: int, nt: int = 12, gofs: int = 0) -> str:
    """w1|w3's dequantiser fused into SwiGLU: the AIFF partials of BOTH halves in one buffer P13 `[M_pad][S][2M]` (a row's passes
    side by side, each pass's w1 columns then w3's), s_x `[nrb][16]`, cs13 `[S][2M]` fp32 -> w2's fp16 A layout
    (`fp16(silu(a1) * a3)`, a1 / a3 = s_x sum_s P cs; `gofs`: the first output slice -- w1|w3 in channel halves). A unit = (96-column group g, row block): ONE strided fill of 12 rows x S
    passes x {w1, w3} x 192 B (the segments are M elements apart throughout; flags 0/1), its cs / s_x (aux, flag 3), one 2304-B
    A slice drained (flag 2). `swiglu_dma`'s exp2 (degree 4). args: out, p13, sx, cs13, desc (`swiglu_i16_descs`)."""
    FULL_H, _ = _hdr()
    ng = M // 96; NT2 = 2 * M
    PB = S * 2 * 2304                                   # a unit's partials: 12 rows x S x 2 x 192 B
    AUX = S * 2 * 384 + 64
    o_aux = 2 * PB; o_out = (o_aux + 2 * AUX + 255) // 256 * 256
    assert o_out + 2 * 2304 <= 32768
    rows4 = ""
    for r in range(4):
        rows4 += "\n          { __global short* PR = PBUF + (4 * i + %d) * %d + 16 * q2; float8 a_ = BC(0.0f), b_ = BC(0.0f), c_ = BC(0.0f), d_ = BC(0.0f);" % (r, S * 2 * 96)
        for k in range(S):
            for h, (lo, hi) in enumerate((("a_", "b_"), ("c_", "d_"))):
                rows4 += " { short16 v = *(__global short16*)(PR + %d); %s += VF(__builtin_aipu_sxtl_th(v)) * c%d%d0; %s += VF(__builtin_aipu_sxth_th(v)) * c%d%d1; }" % ((k * 2 + h) * 96, lo, k, h, hi, k, h)
        rows4 += " float8 sv = BC(SX[4 * i + %d]); lo%d = sw(a_ * sv, c_ * sv); hi%d = sw(b_ * sv, d_ * sv); }" % (r, r, r)
    cload = " ".join("float8 c%d%d0 = *(__global float8*)(CS + %d + 16 * q2), c%d%d1 = *(__global float8*)(CS + %d + 16 * q2 + 8);" % (k, h, (k * 2 + h) * 96, k, h, (k * 2 + h) * 96)
                     for k in range(S) for h in range(2))
    return FULL_H + f"""
static inline float8 exp2_d4(float8 t) {{
  t = VMAX(t, BC(-126.0f)); t = VMIN(t, BC(126.0f));
  float8 k = VRINT(t); float8 f = t - k;
  float8 p = BC(9.618129e-3f);
  p = p * f + BC(5.550411e-2f); p = p * f + BC(2.402265e-1f); p = p * f + BC(6.931472e-1f); p = p * f + BC(1.0f);
  return VSCAL2(p, VI(k));
}}
static inline float8 sw(float8 u, float8 q) {{ float8 e = exp2_d4(BC({-LOG2E!r}f) * u); return u * VRCP(BC(1.0f) + e) * q; }}
__kernel void swiglu_i16(__global half* restrict out, __global short* restrict p, __global float* restrict sx, __global float* restrict cs, __global int* restrict desc, const int core_id) {{
  int k = 0, csg0 = -1, csg1 = -1;
  if (core_id < {ng * nrb}) {{
    int g = core_id / {nrb}, rb = core_id % {nrb};
    DMA_FILL(0, DESC(desc, 0), 0, (int)(p + (rb * 12 * {S * NT2}) + g * 96));
    DMA_FILL(3, DESC(desc, 1), {o_aux}, (int)(cs + g * 96)); DMA_WAIT(3); csg0 = g;
    DMA_FILL(3, DESC(desc, 2), {o_aux + S * 2 * 384}, (int)(sx + rb * 16));
  }}
  for (int u = core_id; u < {ng * nrb}; u += {nt}, k++) {{
    int g = u / {nrb}, rb = u % {nrb}, s = k & 1;
    DMA_WAIT(3); DMA_WAIT(s);
    int un = u + {nt};
    if (un < {ng * nrb}) {{
      int gn = un / {nrb}, rbn = un % {nrb};
      DMA_FILL(s ^ 1, DESC(desc, 0), (s ^ 1) * {PB}, (int)(p + (rbn * 12 * {S * NT2}) + gn * 96));
      int have = s ? csg0 : csg1;
      if (have != gn) {{ DMA_FILL(3, DESC(desc, 1), {o_aux} + (s ^ 1) * {AUX}, (int)(cs + gn * 96)); DMA_WAIT(3); if (s) csg0 = gn; else csg1 = gn; }}
      DMA_FILL(3, DESC(desc, 2), {o_aux} + (s ^ 1) * {AUX} + {S * 2 * 384}, (int)(sx + rbn * 16));
    }}
    __global short* PBUF = (__global short*)(LSRAM_BASE + s * {PB}); __global float* CS = LSF({o_aux} + s * {AUX}); __global float* SX = CS + {S * 2 * 96};
    __global half* OUT = LSH({o_out} + s * 2304);
    for (int q2 = 0; q2 < 6; q2++) {{
      {cload}
      for (int i = 0; i < 3; i++) {{
        float8 lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3;{rows4}
        __global half* T = OUT + ((4 * q2) * 3 + i) * 16;
        *(__global half16*)(T) = CVT16(__builtin_shufflevector(lo0, lo1, 0, 1, 2, 3, 8, 9, 10, 11), __builtin_shufflevector(lo2, lo3, 0, 1, 2, 3, 8, 9, 10, 11));
        *(__global half16*)(T + 48) = CVT16(__builtin_shufflevector(lo0, lo1, 4, 5, 6, 7, 12, 13, 14, 15), __builtin_shufflevector(lo2, lo3, 4, 5, 6, 7, 12, 13, 14, 15));
        *(__global half16*)(T + 96) = CVT16(__builtin_shufflevector(hi0, hi1, 0, 1, 2, 3, 8, 9, 10, 11), __builtin_shufflevector(hi2, hi3, 0, 1, 2, 3, 8, 9, 10, 11));
        *(__global half16*)(T + 144) = CVT16(__builtin_shufflevector(hi0, hi1, 4, 5, 6, 7, 12, 13, 14, 15), __builtin_shufflevector(hi2, hi3, 4, 5, 6, 7, 12, 13, 14, 15));
      }}
    }}
    DMA_WAIT(2);
    DMA_DRAIN(2, DESC(desc, 3), {o_out} + s * 2304, (int)(out + ((g + {gofs}) * {nrb} + rb) * 1152));
  }}
  DMA_WAIT_ALL();
}}"""


def swiglu_i16_descs(S: int, M: int) -> np.ndarray:
    _, slots = _hdr()
    return slots((12 * S * 2 * 192, 192, M * 2, 192), (S * 2 * 384, 384, M * 4, 384), (64,), (2304,))


def swiglu_ref(y1: np.ndarray, y3: np.ndarray) -> np.ndarray:
    """fp32 silu(y1) * y3 (true exp: the kernel's degree-4 exp2 is ~1e-5 relative, under fp16's rounding)."""
    y1 = y1.astype(np.float32); return (y1 / (np.float32(1) + np.exp(-y1)) * y3.astype(np.float32)).astype(np.float32)


# ------------------------------------------------------------------------------------- calibration statistics ----
def colmax_a16_src(nrb: int, K: int, nt: int = 12, rbq: int = 4) -> str:
    """Per-column |max| of an fp16 A operand ([slice][nrb][1152] halves) folded into `out` [K] fp32 (out = max(out, new)): a
    task owns whole 96-column slices (its 384 B of `out` are its own lines). Rows past the operand's real ones are zero.
    args: out, a, desc (`colmax_a16_descs`)."""
    FULL_H, _ = _hdr()
    nsl = K // 96; assert nrb % rbq == 0
    nq, FB_ = nrb // rbq, rbq * 2304
    o_acc = 2 * FB_
    return FULL_H + f"""
__kernel void colmax_a16(__global float* restrict out, __global half* restrict a, __global int* restrict desc, const int core_id) {{
  __global float* ACC = LSF({o_acc});
  for (int sl = core_id; sl < {nsl}; sl += {nt}) {{
    for (int q = 0; q < 48; q++) *(__global float8*)(ACC + 8 * q) = BC(0.0f);
    DMA_FILL(0, DESC(desc, 0), 0, (int)(a + sl * {nrb * 1152}));
    for (int c = 0; c < {nq}; c++) {{
      int s = c & 1;
      if (c + 1 < {nq}) DMA_FILL(s ^ 1, DESC(desc, 0), (s ^ 1) * {FB_}, (int)(a + sl * {nrb * 1152} + (c + 1) * {rbq * 1152}));
      DMA_WAIT(s);
      __global half* B = LSH(s * {FB_});
      for (int kq = 0; kq < 24; kq++) {{
        float8 me = *(__global float8*)(ACC + 16 * kq), mo = *(__global float8*)(ACC + 16 * kq + 8);
        for (int t = 0; t < {3 * rbq}; t++) {{
          half16 h = *(__global half16*)(B + ((t / 3) * 72 + kq * 3 + t % 3) * 16);
          float8 e = __builtin_aipu_cvtue_tfp32_tfp16(h), o = __builtin_aipu_cvtuo_tfp32_tfp16(h);
          me = VMAX(me, VMAX(e, -e)); mo = VMAX(mo, VMAX(o, -o));
        }}
        *(__global float8*)(ACC + 16 * kq) = me; *(__global float8*)(ACC + 16 * kq + 8) = mo;
      }}
    }}
    __global float* O = LSF({o_acc + 1536});
    DMA_FILL(2, DESC(desc, 1), {o_acc + 1536}, (int)(out + sl * 96)); DMA_WAIT(2);
    for (int kq = 0; kq < 24; kq++) {{
      float8 me = *(__global float8*)(ACC + 16 * kq), mo = *(__global float8*)(ACC + 16 * kq + 8);
      float c0 = me[0], c1 = mo[0], c2 = me[1], c3 = mo[1];
      for (int l = 1; l < 4; l++) {{ c0 = c0 > me[2 * l] ? c0 : me[2 * l]; c2 = c2 > me[2 * l + 1] ? c2 : me[2 * l + 1];
                                      c1 = c1 > mo[2 * l] ? c1 : mo[2 * l]; c3 = c3 > mo[2 * l + 1] ? c3 : mo[2 * l + 1]; }}
      O[4 * kq] = O[4 * kq] > c0 ? O[4 * kq] : c0; O[4 * kq + 1] = O[4 * kq + 1] > c1 ? O[4 * kq + 1] : c1;
      O[4 * kq + 2] = O[4 * kq + 2] > c2 ? O[4 * kq + 2] : c2; O[4 * kq + 3] = O[4 * kq + 3] > c3 ? O[4 * kq + 3] : c3;
    }}
    DMA_DRAIN(2, DESC(desc, 1), {o_acc + 1536}, (int)(out + sl * 96)); DMA_WAIT(2);
  }}
}}"""


def colmax_a16_descs(rbq: int = 4) -> np.ndarray:
    _, slots = _hdr()
    return slots((rbq * 2304,), (384,))


def colmax_i16_src(rows: int, W: int, nt: int = 12, rq: int = 24) -> str:
    """Per-column |max| of int16 rows `[rows][W]` (the AIFF partials, W = passes x columns) folded into `out` [W] fp32. A unit
    = 192 columns (384 B), streamed in strided fills of `rq` rows (flags 0/1). args: out, p, desc (`colmax_i16_descs`)."""
    FULL_H, _ = _hdr()
    assert W % 192 == 0
    nu, nc, FB_ = W // 192, -(-rows // rq), rq * 384
    last = rows - (nc - 1) * rq
    o_acc = 2 * FB_
    return FULL_H + f"""
__kernel void colmax_i16(__global float* restrict out, __global short* restrict p, __global int* restrict desc, const int core_id) {{
  __global float* ACC = LSF({o_acc});
  for (int u = core_id; u < {nu}; u += {nt}) {{
    for (int q = 0; q < 24; q++) *(__global float8*)(ACC + 8 * q) = BC(0.0f);
    DMA_FILL(0, DESC(desc, {0 if nc > 1 else 1}), 0, (int)(p + u * 192));
    for (int c = 0; c < {nc}; c++) {{
      int s = c & 1;
      if (c + 1 < {nc}) DMA_FILL(s ^ 1, DESC(desc, c + 1 < {nc - 1} ? 0 : 1), (s ^ 1) * {FB_}, (int)(p + (c + 1) * {rq} * {W} + u * 192));
      DMA_WAIT(s);
      __global short* B = (__global short*)(LSRAM_BASE + s * {FB_});
      int nr = c + 1 < {nc} ? {rq} : {last};
      for (int q = 0; q < 12; q++) {{
        float8 lo = *(__global float8*)(ACC + 16 * q), hi = *(__global float8*)(ACC + 16 * q + 8);
        for (int r = 0; r < nr; r++) {{
          short16 v = *(__global short16*)(B + r * 192 + 16 * q);
          float8 a = VF(__builtin_aipu_sxtl_th(v)), b = VF(__builtin_aipu_sxth_th(v));
          lo = VMAX(lo, VMAX(a, -a)); hi = VMAX(hi, VMAX(b, -b));
        }}
        *(__global float8*)(ACC + 16 * q) = lo; *(__global float8*)(ACC + 16 * q + 8) = hi;
      }}
    }}
    __global float* O = LSF({o_acc + 768});
    DMA_FILL(2, DESC(desc, 2), {o_acc + 768}, (int)(out + u * 192)); DMA_WAIT(2);
    for (int q = 0; q < 24; q++) *(__global float8*)(O + 8 * q) = VMAX(*(__global float8*)(O + 8 * q), *(__global float8*)(ACC + 8 * q));
    DMA_DRAIN(2, DESC(desc, 2), {o_acc + 768}, (int)(out + u * 192)); DMA_WAIT(2);
  }}
}}"""


def colmax_i16_descs(rows: int, W: int, rq: int = 24) -> np.ndarray:
    _, slots = _hdr()
    nc = -(-rows // rq); last = rows - (nc - 1) * rq
    return slots((rq * 384, 384, W * 2, 384), (last * 384, 384, W * 2, 384), (768,))

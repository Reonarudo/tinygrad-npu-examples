"""Ideogram4's vector kernels: the DiT block's steps and the VAE decoder's, as C sources for `register_csrc`, with their
descriptor tables and numpy references. Moved here from the backend's `extra/zhouyi/vec_f16.py`, which keeps the generic
pieces these build on (the vector C header `FULL_H`, the GEMM operand layouts and their numpy references, `_desc_slots`,
`dup_quads`, `tile_rows`, `dma_copy`, `tec_sum`); they are re-exported here so the model code reads one namespace (`V`).

  K1..K6 / S1..S6: the DiT block -- RMSNorm + AdaLN scale, residual, SwiGLU, qkv with RMSNorm and RoPE, softmax (staged and
  streamed), the heads' gather; the I/O projections, modulation tables, timestep MLP, CFG / Euler step;
  the VAE decoder: 3x3 / 1x1 convs (`conv_a` / `conv_c`), GroupNorm (+ SiLU), the mid-block attention.

Import after the script has put its tinygrad tree on `sys.path` (as `zy` requires)."""
import numpy as np
from zy import vec_f16 as _V, gemm_fp16 as _G

# the generic helpers (extra/zhouyi/vec_f16.py), re-exported
NT, LOG2E, H, DMA_H, FULL_H, TEC_SUM_SLOT = _V.NT, _V.LOG2E, _V.H, _V.DMA_H, _V.FULL_H, _V.TEC_SUM_SLOT
a_layout_ref, b_layout_ref, c_tiles_ref, c_from_tiles = _V.a_layout_ref, _V.b_layout_ref, _V.c_tiles_ref, _V.c_from_tiles
_desc_slots, dup_quads, tile_rows = _V._desc_slots, _V.dup_quads, _V.tile_rows
dma_copy_src, dma_copy_descs, tec_copy_descs, tec_sum_src = _V.dma_copy_src, _V.dma_copy_descs, _V.tec_copy_descs, _V.tec_sum_src


# ---- K1..K6: the numpy references of the block's steps (the DMA-fed `*_dma_src` kernels below compute the same values) ----


# ---- K1: fp32 rows -> the A layout, RMSNorm + AdaLN scale ----


def rms_scale_a16_ref(x, w, sc, eps, ks, nrb):
    x = x.astype(np.float32); inv = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True, dtype=np.float32) + np.float32(eps))
    return a_layout_ref(x * inv * w.astype(np.float32) * sc.astype(np.float32), ks, nrb)


# ---- K2: the residual, x += g * rmsnorm(C tiles) * w ----


def resid_ct_ref(x, w, g, cfull, eps):
    """cfull: the linear's fp32 `[rows, c]` (the tiles' content)."""
    y = cfull.astype(np.float32); inv = 1.0 / np.sqrt((y * y).mean(-1, keepdims=True, dtype=np.float32) + np.float32(eps))
    return (x.astype(np.float32) + g.astype(np.float32) * (y * inv * w.astype(np.float32))).astype(np.float32)


# ---- K3: SwiGLU from two linears' C tiles -> the A layout ----


def swiglu_a16_ref(c1full, c3full, ks, nrb):
    u = c1full.astype(np.float32); y = u / (np.float32(1) + np.exp(-u)) * c3full.astype(np.float32)
    return a_layout_ref(y, ks, nrb)


# ---- K4: the qkv linear's C tiles -> Q (A layouts per head), K and V^T (B panels per head) ----


def qkv_a16_sizes(nh, dh, dhp, nrb, ks, nsa, tp):
    QA = 12 * nrb * dhp
    KB = (tp // (16 * nsa)) * (dhp // (4 * ks)) * nsa * ks * 64
    VB = (dhp // (16 * nsa)) * (tp // (4 * ks)) * nsa * ks * 64
    return QA, KB, VB, (QA + KB + VB) * nh


def qkv_a16_ref(cfull, nq, nk, cos, sin, eps, nh, dh, dhp, nrb, ks, nsa, tp, v48=False):
    """cfull: the qkv linear's fp32 `[rows, 3 nh dh]` -> (the whole output buffer (uint16), q, k, v `[rows, nh, dh]`).
    `v48`: V^T panels in `gemm_fp16.repack_panels_48x3` order."""
    rows = cfull.shape[0]; y = cfull.astype(np.float32).reshape(rows, 3, nh, dh); q, k, v = y[:, 0], y[:, 1], y[:, 2]
    rms = lambda x, w: x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True, dtype=np.float32) + np.float32(eps))) * w.astype(np.float32)
    q = rms(q, nq); k = rms(k, nk); hd = dh // 2
    def rope(x):
        c, s = cos[:, None, :], sin[:, None, :]
        return x * c + np.concatenate([-x[..., hd:], x[..., :hd]], -1) * s
    q = rope(q); k = rope(k)
    QA, KB, VB, total = qkv_a16_sizes(nh, dh, dhp, nrb, ks, nsa, tp)
    out = np.zeros(total, np.uint16)
    for h in range(nh):
        qp = np.zeros((rows, dhp), np.float32); qp[:, :dh] = q[:, h]
        out[h * QA:(h + 1) * QA] = a_layout_ref(qp, ks, nrb)
        kp = np.zeros((tp, dhp), np.float32); kp[:rows, :dh] = k[:, h]
        out[QA * nh + h * KB:QA * nh + (h + 1) * KB] = b_layout_ref(kp, ks, nsa)
        vt = np.zeros((dhp, tp), np.float32); vt[:dh, :rows] = v[:, h].T
        vb = b_layout_ref(vt, ks, nsa)
        if v48:
            vb = _G.repack_panels_48x3(vb, tp)
        out[(QA + KB) * nh + h * VB:(QA + KB) * nh + (h + 1) * VB] = vb
    return out, q, k, v


# ---- K5: softmax of the logits' C tiles -> P in the A layout ----


def softmax_a16_ref(cfull, scale, rows, ks, nrb):
    """cfull: the logits `[rows, tp]` fp32."""
    x = cfull[:rows, :rows].astype(np.float32) * np.float32(scale)
    e = np.exp(x - x.max(-1, keepdims=True)); p = e / e.sum(-1, keepdims=True)
    pp = np.zeros((rows, cfull.shape[1]), np.float32); pp[:, :rows] = p
    return a_layout_ref(pp, ks, nrb)


# ---- K6: the heads' P V tiles -> the o-projection's A layout ----


def gather_o_a16_ref(heads_full, dh, ks, nrb):
    """heads_full: list of nh fp32 `[rows, dhp]` (each head's P V); the first dh columns concatenated."""
    return a_layout_ref(np.concatenate([h[:, :dh] for h in heads_full], 1), ks, nrb)


# ---- S1: rms_scale, staged ----
def rms_scale_dma_src(rows: int, c: int, eps: float, ks: int, nrb: int, nt: int = NT, readback: bool = False, tail_spin: int = 0, cdiv: int|None = None, center: bool = False) -> str:
    """`A = fp16(rmsnorm(x) * w * sc)` in the A layout, a row block per unit. Phase A: half-row fills -> 1/rms per row.
    Phase B per slice: one strided fill of 12 x 384 B, tiles, one 2304-B drain (fills flags 0/1, drains 2/3).
    `ws2` = `dup_quads(w * sc)`. Needs 4 ks = 96. `cdiv`: the mean's divisor when x has zero pad columns.
    `center`: LayerNorm, summing x - K (K = mean of the row's first 8 values, against cancellation).
    args: out, x, ws2, desc (`rms_scale_dma_descs`)."""
    assert 4 * ks == 96 and c % 96 == 0 and (c % 18432 == 0 or c <= 4608) and (c // 2) % 8 == 0
    cdiv = c if cdiv is None else cdiv
    nsl = c // 96; half = 2 * c                                   # bytes of half a row
    if not center:
        PHASE_A = f"""for (int i = 0; i < {c // 2}; i += 8) {{ float8 v = *(__global float8*)(p + i); acc += v * v; }}
      if (t & 1) inv[t / 2] = (m < {rows}) ? 1.0f / __builtin_sqrtf((inv[t / 2] + hsum8(acc)) * {1.0 / cdiv!r}f + {eps!r}f) : 0.0f;
      else inv[t / 2] = hsum8(acc);"""
    else:
        PHASE_A = f"""if (!(t & 1)) kk[t / 2] = hsum8(*(__global float8*)(p)) * 0.125f;
      float8 kv = BC(kk[t / 2]), a1 = BC(0.0f);
      for (int i = 0; i < {c // 2}; i += 8) {{ float8 v = *(__global float8*)(p + i) - kv; a1 += v; acc += v * v; }}
      if (t & 1) {{
        float mu = (s1[t / 2] + hsum8(a1)) * {1.0 / c!r}f, var = (inv[t / 2] + hsum8(acc)) * {1.0 / c!r}f - mu * mu;
        inv[t / 2] = (m < {rows}) ? 1.0f / __builtin_sqrtf(var + {eps!r}f) : 0.0f; nb[t / 2] = -(kk[t / 2] + mu) * inv[t / 2];
      }} else {{ inv[t / 2] = hsum8(acc); s1[t / 2] = hsum8(a1); }}"""
    return FULL_H + f"""
__kernel void rms_scale_dma(__global half* restrict out, __global float* restrict x, __global float* restrict ws2, __global int* restrict desc, const int core_id) {{
  for (int rb = core_id; rb < {nrb}; rb += {nt}) {{
    float inv[12];{" float kk[12], s1[12], nb[12];" if center else ""}
    /* phase A: 24 half rows, double-buffered */
    DMA_FILL(0, DESC(desc, 0), 0, (int)(x + rb * 12 * {c}));
    for (int t = 0; t < 24; t++) {{
      int s = t & 1, m = rb * 12 + t / 2;
      if (t + 1 < 24) DMA_FILL(s ^ 1, DESC(desc, s ^ 1), (s ^ 1) * {half}, (int)(x + (rb * 12 + (t + 1) / 2) * {c} + ((t + 1) & 1) * {c // 2}));
      DMA_WAIT(s);
      __global float* p = LSF(s * {half}); float8 acc = BC(0.0f);
      {PHASE_A}
    }}
    /* phase B: per slice, the 12 rows' 96 columns -> 72 tiles */
    DMA_FILL(0, DESC(desc, 2), 0, (int)(x + rb * 12 * {c}));
    for (int sl = 0; sl < {nsl}; sl++) {{
      int s = sl & 1;
      if (sl + 1 < {nsl}) DMA_FILL(s ^ 1, DESC(desc, 2 + (s ^ 1)), (s ^ 1) * 4608, (int)(x + rb * 12 * {c} + (sl + 1) * 96));
      DMA_WAIT(s); DMA_WAIT(2 + s);
      DMA_FILL(2 + s, DESC(desc, 6), 14336, (int)(ws2 + sl * 192)); DMA_WAIT(2 + s);        /* the slice's 24 quads of w * sc: never through the cache */
      __global float* xr = LSF(s * 4608); __global half* o = LSH(9216 + s * 2304); __global float* scr = LSF(13824 + s * 64); __global float* WS = LSF(14336);
      for (int i = 0; i < 3; i++) {{
        float8 i01 = ROWS2(inv[4 * i], inv[4 * i + 1]), i23 = ROWS2(inv[4 * i + 2], inv[4 * i + 3]);{" float8 n01 = ROWS2(nb[4 * i], nb[4 * i + 1]), n23 = ROWS2(nb[4 * i + 2], nb[4 * i + 3]);" if center else ""}
        for (int kq = 0; kq < 24; kq++) {{
          *(__global float4*)(scr) = *(__global float4*)(xr + (4 * i) * 96 + 4 * kq);       *(__global float4*)(scr + 4) = *(__global float4*)(xr + (4 * i + 1) * 96 + 4 * kq);
          *(__global float4*)(scr + 8) = *(__global float4*)(xr + (4 * i + 2) * 96 + 4 * kq); *(__global float4*)(scr + 12) = *(__global float4*)(xr + (4 * i + 3) * 96 + 4 * kq);
          float8 w8 = *(__global float8*)(WS + kq * 8);
          float8 t01 = {"(*(__global float8*)(scr) * i01 + n01) * w8" if center else "*(__global float8*)(scr) * i01 * w8"}, t23 = {"(*(__global float8*)(scr + 8) * i23 + n23) * w8" if center else "*(__global float8*)(scr + 8) * i23 * w8"};
          *(__global half16*)(o + (kq * 3 + i) * 16) = CVT16(t01, t23);
        }}
      }}
      DMA_DRAIN(2 + s, DESC(desc, 4 + s), 9216 + s * 2304, (int)(out + (sl * {nrb} + rb) * {ks * 48}));
    }}
    DMA_WAIT_ALL();
    {"DMA_FILL(0, DESC(desc, 4), 13824, (int)(out + ((%d - 1) * %d + rb) * %d)); DMA_WAIT_ALL();" % (nsl, nrb, ks * 48) if readback else ""}
  }}
  {"{ int t0 = __builtin_aipu_mfctrl0(0xd1); while ((int)(__builtin_aipu_mfctrl0(0xd1) - t0) < %d) ; }" % tail_spin if tail_spin else ""}
}}"""


def rms_scale_dma_descs(c: int, nrb: int) -> np.ndarray:
    return _desc_slots((2 * c,), (2 * c,), (4608, 384, 4 * c, 384), (4608, 384, 4 * c, 384), (2304,), (2304,), (768,))


# ---- S2: the residual, staged ----
def resid_dma_src(rows: int, c: int, ns: int, nrb: int, eps: float, nt: int = NT, stamps: bool = False, fast: bool = True, plain: bool = False) -> str:
    """`out = x + g * (rmsnorm(C) * w)`, C the tiles of a linear with N = c (ns = 6), a row block per unit. Phase A: tile
    blocks, 2 groups per fill -> 1/rms per row. Phase B per group: tiles (flag 0), x rows (1), constants (3) in; out (2).
    `k3` = `resid_dma_consts(w, g, sc)`, `sc2` = `dup_quads(sc)` (per-column scale, applied before the norm).
    args: out, x, k3, ct, sc2, desc (+ st_ with `stamps`: per-task cycle counters at st_[16 task ..]).
    `fast`: phase B transposes in registers (bit-identical). `plain`: out = x + C * sc, no norm."""
    assert ns == 6 and c % 96 == 0
    ng = c // 96; assert ng % 2 == 0
    SLOWB = f'''      for (int i = 0; i < 3; i++) {{
        float8 i01 = ROWS2(inv[4 * i], inv[4 * i + 1]), i23 = ROWS2(inv[4 * i + 2], inv[4 * i + 3]);
        for (int st = 0; st < 6; st++) for (int jt = 0; jt < 4; jt++) {{
          int kq = st * 4 + jt; __global float* tp = cp + ((st * 3 + i) * 4 + jt) * 16;
          float8 w8 = *(__global float8*)(K3 + kq * 24) * *(__global float8*)(K3 + kq * 24 + 16), g8 = *(__global float8*)(K3 + kq * 24 + 8);
          *(__global float4*)(scr) = *(__global float4*)(xp + (4 * i) * 96 + 4 * kq);       *(__global float4*)(scr + 4) = *(__global float4*)(xp + (4 * i + 1) * 96 + 4 * kq);
          *(__global float4*)(scr + 8) = *(__global float4*)(xp + (4 * i + 2) * 96 + 4 * kq); *(__global float4*)(scr + 12) = *(__global float4*)(xp + (4 * i + 3) * 96 + 4 * kq);
          float8 o01 = *(__global float8*)(scr) + g8 * (*(__global float8*)(tp) * i01 * w8);
          float8 o23 = *(__global float8*)(scr + 8) + g8 * (*(__global float8*)(tp + 8) * i23 * w8);
          *(__global float8*)(scr) = o01; *(__global float8*)(scr + 8) = o23;
          *(__global float4*)(op + (4 * i) * 96 + 4 * kq) = *(__global float4*)(scr);         *(__global float4*)(op + (4 * i + 1) * 96 + 4 * kq) = *(__global float4*)(scr + 4);
          *(__global float4*)(op + (4 * i + 2) * 96 + 4 * kq) = *(__global float4*)(scr + 8); *(__global float4*)(op + (4 * i + 3) * 96 + 4 * kq) = *(__global float4*)(scr + 12);
        }}
      }}
'''
    FASTB = f'''      for (int i = 0; i < 3; i++) {{
        float8 i01 = ROWS2(inv[4 * i], inv[4 * i + 1]), i23 = ROWS2(inv[4 * i + 2], inv[4 * i + 3]);
        __global float* x0 = xp + (4 * i) * 96; __global float* o0 = op + (4 * i) * 96;
        for (int m = 0; m < 12; m++) {{                     /* column quads 2m, 2m + 1 of the four rows */
          float8 a0 = *(__global float8*)(x0 + 8 * m), a1 = *(__global float8*)(x0 + 96 + 8 * m);
          float8 a2 = *(__global float8*)(x0 + 192 + 8 * m), a3 = *(__global float8*)(x0 + 288 + 8 * m);
          int ke = 2 * m, ko = 2 * m + 1;
          __global float* te = cp + (((ke / 4) * 3 + i) * 4 + ke % 4) * 16; __global float* to = cp + (((ko / 4) * 3 + i) * 4 + ko % 4) * 16;
          float8 we = *(__global float8*)(K3 + ke * 24) * *(__global float8*)(K3 + ke * 24 + 16), ge = *(__global float8*)(K3 + ke * 24 + 8);
          float8 wo = *(__global float8*)(K3 + ko * 24) * *(__global float8*)(K3 + ko * 24 + 16), go = *(__global float8*)(K3 + ko * 24 + 8);
          float8 e01 = __builtin_shufflevector(a0, a1, 0, 1, 2, 3, 8, 9, 10, 11) + ge * (*(__global float8*)(te) * i01 * we);
          float8 e23 = __builtin_shufflevector(a2, a3, 0, 1, 2, 3, 8, 9, 10, 11) + ge * (*(__global float8*)(te + 8) * i23 * we);
          float8 d01 = __builtin_shufflevector(a0, a1, 4, 5, 6, 7, 12, 13, 14, 15) + go * (*(__global float8*)(to) * i01 * wo);
          float8 d23 = __builtin_shufflevector(a2, a3, 4, 5, 6, 7, 12, 13, 14, 15) + go * (*(__global float8*)(to + 8) * i23 * wo);
          *(__global float8*)(o0 + 8 * m) = __builtin_shufflevector(e01, d01, 0, 1, 2, 3, 8, 9, 10, 11);
          *(__global float8*)(o0 + 96 + 8 * m) = __builtin_shufflevector(e01, d01, 4, 5, 6, 7, 12, 13, 14, 15);
          *(__global float8*)(o0 + 192 + 8 * m) = __builtin_shufflevector(e23, d23, 0, 1, 2, 3, 8, 9, 10, 11);
          *(__global float8*)(o0 + 288 + 8 * m) = __builtin_shufflevector(e23, d23, 4, 5, 6, 7, 12, 13, 14, 15);
        }}
      }}
'''
    PHASEA = (f'''    float inv[12];
    for (int r = 0; r < 12; r++) inv[r] = (rb * 12 + r < {rows}) ? 1.0f : 0.0f;
''' if plain else f'''    float8 a01[3], a23[3];
    for (int i = 0; i < 3; i++) {{ a01[i] = BC(0.0f); a23[i] = BC(0.0f); }}
    DMA_FILL(0, DESC(desc, 0), 0, (int)(ct + (0 * {nrb} + rb) * 1152));
    for (int t = 0; t < {ng // 2}; t++) {{
      int s = t & 1;
      if (t + 1 < {ng // 2}) DMA_FILL(s ^ 1, DESC(desc, s ^ 1), (s ^ 1) * 9216, (int)(ct + ((t + 1) * 2 * {nrb} + rb) * 1152));
      {"{ int t_ = __builtin_aipu_mfctrl0(0xd1); acc_[1] += t_ - t0_; t0_ = t_; }" if stamps else ""}
      DMA_WAIT(s);
      DMA_FILL(2 + s, DESC(desc, 5), 18432 + s * 1536, (int)(sc2 + t * 2 * 192)); DMA_WAIT(2 + s);   /* the two groups' scale quads */
      {"{ int t_ = __builtin_aipu_mfctrl0(0xd1); acc_[0] += t_ - t0_; t0_ = t_; }" if stamps else ""}
      __global float* p = LSF(s * 9216); __global float* SC = LSF(18432 + s * 1536);
      for (int b = 0; b < 2; b++) for (int st = 0; st < 6; st++) for (int i = 0; i < 3; i++) for (int jt = 0; jt < 4; jt++) {{
        __global float* tp = p + ((b * 6 + st) * 12 + i * 4 + jt) * 16;
        float8 s8 = *(__global float8*)(SC + (b * 24 + st * 4 + jt) * 8);
        float8 v01 = *(__global float8*)(tp) * s8, v23 = *(__global float8*)(tp + 8) * s8;
        a01[i] += v01 * v01; a23[i] += v23 * v23;
      }}
    }}
    float inv[12];
    for (int i = 0; i < 3; i++) {{
      inv[4 * i] = a01[i][0] + a01[i][1] + a01[i][2] + a01[i][3]; inv[4 * i + 1] = a01[i][4] + a01[i][5] + a01[i][6] + a01[i][7];
      inv[4 * i + 2] = a23[i][0] + a23[i][1] + a23[i][2] + a23[i][3]; inv[4 * i + 3] = a23[i][4] + a23[i][5] + a23[i][6] + a23[i][7];
    }}
    for (int r = 0; r < 12; r++) inv[r] = (rb * 12 + r < {rows}) ? 1.0f / __builtin_sqrtf(inv[r] * {1.0 / c!r}f + {eps!r}f) : 0.0f;
''')
    return FULL_H + f"""
__kernel void resid_dma(__global float* restrict out, __global float* restrict x, __global float* restrict k3, __global float* restrict ct, __global float* restrict sc2, __global int* restrict desc{", __global int* restrict st_" if stamps else ""}, const int core_id) {{
  {"int acc_[4] = {0, 0, 0, 0}; int t0_ = __builtin_aipu_mfctrl0(0xd1); int st0_ = t0_;" if stamps else ""}
  for (int rb = core_id; rb < {nrb}; rb += {nt}) {{
{PHASEA}    /* phase B: per group: C tiles (4608 B at 0), x rows (4608 B at 4608), out rows (4608 B at 9216) */
    for (int g = 0; g < {ng}; g++) {{
      DMA_FILL(0, DESC(desc, 2), 0, (int)(ct + (g * {nrb} + rb) * 1152));
      DMA_FILL(1, DESC(desc, 3), 4608, (int)(x + rb * 12 * {c} + g * 96));
      DMA_FILL(3, DESC(desc, 6), 13888, (int)(k3 + g * 576));                                 /* the group's 24 quads of w | g | sc (`resid_dma_consts`); ⚠️ flag 3: the previous group's drain may still hold flag 2 */
      {"{ int t_ = __builtin_aipu_mfctrl0(0xd1); acc_[3] += t_ - t0_; t0_ = t_; }" if stamps else ""}
      DMA_WAIT_ALL();
      {"{ int t_ = __builtin_aipu_mfctrl0(0xd1); acc_[2] += t_ - t0_; t0_ = t_; }" if stamps else ""}
      __global float* cp = LSF(0); __global float* xp = LSF(4608); __global float* op = LSF(9216); __global float* scr = LSF(13824); __global float* K3 = LSF(13888);
{FASTB if fast else SLOWB}      DMA_DRAIN(2, DESC(desc, 4), 9216, (int)(out + rb * 12 * {c} + g * 96));
    }}
    {"{ int t_ = __builtin_aipu_mfctrl0(0xd1); acc_[3] += t_ - t0_; t0_ = t_; }" if stamps else ""}
    DMA_WAIT_ALL();
    {"{ int t_ = __builtin_aipu_mfctrl0(0xd1); acc_[2] += t_ - t0_; t0_ = t_; }" if stamps else ""}
  }}
  {"for (int k = 0; k < 4; k++) st_[core_id * 16 + k] = acc_[k]; st_[core_id * 16 + 4] = st0_; st_[core_id * 16 + 5] = __builtin_aipu_mfctrl0(0xd1); st_[core_id * 16 + 6] = __builtin_aipu_mfctrl0(0);" if stamps else ""}
}}"""


def resid_dma_descs(c: int, nrb: int) -> np.ndarray:
    return _desc_slots((9216, 4608, nrb * 4608, 4608), (9216, 4608, nrb * 4608, 4608), (4608,), (4608, 384, 4 * c, 384), (4608, 384, 4 * c, 384), (1536,), (2304,))


def resid_dma_consts(w: np.ndarray, g: np.ndarray, sc: np.ndarray) -> np.ndarray:
    """fp32 `[c]` each -> `[c/4 quads][3][8]`: per quad the duplicated quads of w, g and the weight scale (one 2304-B fill per group)."""
    return np.ascontiguousarray(np.stack([dup_quads(w).reshape(-1, 8), dup_quads(g).reshape(-1, 8), dup_quads(sc).reshape(-1, 8)], 1)).ravel()


# ---- S3: swiglu, staged ----
def swiglu_dma_src(rows: int, m_: int, nrb: int, ks: int, nt: int = NT, mode: str = "full", interleaved: bool = False) -> str:
    """`A = fp16(silu(C1) * C3)` (K = m_) from the fused w1|w3 linear's tiles (N = 2 m_, ns = 6; C3 after C1, or
    alternating if `interleaved`). A unit = (row block, slice): two tile blocks in, one 2304-B A slice out (fills flags
    0/1, drains 2/3). Needs 4 ks = 96 and zero pad rows. args: out, ct, sc2 (`dup_quads` of the column scales), desc
    (`swiglu_dma_descs`). `mode`: "full"; timing arms "nomath", "nodma"."""
    assert 4 * ks == 96 and m_ % 96 == 0 and mode in ("full", "nomath", "nodma")
    nsl = m_ // 96
    # `interleaved`: groups alternate w1, w3 so a slice's two tile blocks are one group apart (concatenated, the
    # m_/96-group distance can exceed the DMA's 16 MB stride field)
    gsl, g3 = (2, 1) if interleaved else (1, nsl)          # group of C1's slice sl = gsl * sl; C3's = that + g3
    D = (lambda x: x) if mode != "nodma" else (lambda x: ";")
    # degree-4 exp2 (1.3e-5 relative, enough for fp16): degree 6 makes this kernel compute-bound
    MATH = "float8 e = exp2_d4(BC(%rf) * u); return u * VRCP(BC(1.0f) + e) * q;" % -LOG2E if mode != "nomath" else "return u * q;"
    return FULL_H + f"""
static inline float8 exp2_d4(float8 t) {{
  t = VMAX(t, BC(-126.0f)); t = VMIN(t, BC(126.0f));
  float8 k = VRINT(t); float8 f = t - k;
  float8 p = BC(9.618129e-3f);
  p = p * f + BC(5.550411e-2f); p = p * f + BC(2.402265e-1f); p = p * f + BC(6.931472e-1f); p = p * f + BC(1.0f);
  return VSCAL2(p, VI(k));
}}
static inline float8 sw(float8 u, float8 q) {{ {MATH} }}
__kernel void swiglu_dma(__global half* restrict out, __global float* restrict ct, __global float* restrict sc2, __global int* restrict desc, const int core_id) {{
  for (int rb = core_id; rb < {nrb}; rb += {nt}) {{
    {D("DMA_FILL(0, DESC(desc, 0), 0, (int)(ct + (0 * %d + rb) * 1152));" % nrb)}
    for (int sl = 0; sl < {nsl}; sl++) {{
      int s = sl & 1;
      {D("if (sl + 1 < %d) DMA_FILL(s ^ 1, DESC(desc, s ^ 1), (s ^ 1) * 9216, (int)(ct + (%d * (sl + 1) * %d + rb) * 1152));" % (nsl, gsl, nrb))}
      {D("DMA_WAIT(s); DMA_WAIT(2 + s);")}
      {D("DMA_FILL(2 + s, DESC(desc, 4), 23040 + s * 1536, (int)(sc2 + %d * sl * 192)); DMA_WAIT(2 + s);" % gsl)}   /* the slice's scale quads of C1 and C3 (a strided fill) */
      __global float* c1 = LSF(s * 9216); __global float* c3 = c1 + 1152; __global half* o = LSH(18432 + s * 2304); __global float* SC = LSF(23040 + s * 1536);
      for (int i = 0; i < 3; i++) for (int st = 0; st < 6; st++) for (int jt = 0; jt < 4; jt += 2) {{
        /* two column tiles at once: 4 independent exp chains for the scheduler */
        int tb = ((st * 3 + i) * 4 + jt) * 16, kq = st * 4 + jt;
        float8 s1 = *(__global float8*)(SC + kq * 8), s3 = *(__global float8*)(SC + 192 + kq * 8);
        float8 r1 = *(__global float8*)(SC + kq * 8 + 8), r3 = *(__global float8*)(SC + 192 + kq * 8 + 8);
        float8 t01 = sw(*(__global float8*)(c1 + tb) * s1, *(__global float8*)(c3 + tb) * s3), t23 = sw(*(__global float8*)(c1 + tb + 8) * s1, *(__global float8*)(c3 + tb + 8) * s3);
        float8 u01 = sw(*(__global float8*)(c1 + tb + 16) * r1, *(__global float8*)(c3 + tb + 16) * r3), u23 = sw(*(__global float8*)(c1 + tb + 24) * r1, *(__global float8*)(c3 + tb + 24) * r3);
        *(__global half16*)(o + (kq * 3 + i) * 16) = CVT16(t01, t23);
        *(__global half16*)(o + ((kq + 1) * 3 + i) * 16) = CVT16(u01, u23);
      }}
      {D("DMA_DRAIN(2 + s, DESC(desc, 2 + s), 18432 + s * 2304, (int)(out + (sl * %d + rb) * %d));" % (nrb, ks * 48))}
    }}
    {D("DMA_WAIT_ALL();")}
  }}
}}"""


def swiglu_dma_descs(m_: int, nrb: int, ks: int, interleaved: bool = False) -> np.ndarray:
    g3 = 1 if interleaved else m_ // 96
    pitch = g3 * nrb * 4608
    return _desc_slots((9216, 4608, pitch, 4608), (9216, 4608, pitch, 4608), (2304,), (2304,), (1536, 768, g3 * 768, 768))


def interleave_w13(codes: np.ndarray, sc2: np.ndarray, m_: int):
    """Fused w1|w3 panels and scale quads -> the alternating group order of `swiglu_dma(interleaved=True)`."""
    ng = m_ // 96
    order = np.stack([np.arange(ng), np.arange(ng) + ng], 1).ravel()
    return codes.reshape(2 * ng, -1)[order].ravel(), sc2.reshape(2 * ng, -1)[order].ravel()


# ---- S4: qkv, staged ----
def head_strip_runs(h: int, dh: int, ns: int = 6, strip0: int = 0):
    """The head's dh / 16 strips (from strip `strip0 + h dh / 16`) as (group, first strip, count) runs: one fill each."""
    a0, n = strip0 + h * dh // 16, dh // 16; runs = []
    a = a0
    while a < a0 + n:
        g, st = a // ns, a % ns; cnt = min(ns - st, a0 + n - a); runs.append((g, st, cnt)); a += cnt
    return runs


def qkv_dma_src(rows: int, nh: int, dh: int, dhp: int, nrb: int, ks: int, nsa: int, tp: int, eps: float, nt: int = NT, mode: str = "full", v48: bool = False, fast: bool = False) -> str:
    """The qkv step (K4) staged, per (row block, head): Q_h -> A at `out + h*QA`; K_h -> B panels at `out + QA*nh + h*KB`;
    V_h^T -> B panels at `out + (QA+KB)*nh + h*VB`. `nq2` / `nk2` = `dup_quads` of the norm weights, `cs` / `sn` =
    `tile_rows(cos / sin)`, `sc2` = `dup_quads` of the column scales; all DMA'd.
    LSRAM: T 0, O 12288, sc 19200, nq/nk 21248, cos 23296, sin 27392..31488. Per component: head strips fill T (<= 3
    runs, flags 0..2), convert into O, drain <= 4 at a time. Pad token rows must be zero in the tiles; tp % 96 == 0,
    tp >= 12 nrb; tokens 12 nrb .. tp are host-zeroed. `v48`: V^T in `repack_panels_48x3` order (tp % 192 == 0).
    `mode`: "full"; timing arms "noV", "stamps" (extra arg `st`). `fast`: `_qkv_fast`."""
    assert dh == 256 and dhp == 288 and 4 * ks == 96 and nsa == 6 and tp % 96 == 0 and 12 * nrb <= tp
    assert mode in ("full", "noV", "stamps")
    QA, KB, VB, _ = qkv_a16_sizes(nh, dh, dhp, nrb, ks, nsa, tp)
    GB = (dhp // 96) * nsa * ks * 64                    # halves per K-panel group (K = dhp: 3 slices x 6 strips)
    GBV = (tp // 96) * nsa * ks * 64                    # halves per V^T-panel group (K = tp: tp / 96 slices x 6 strips)
    assert not v48 or (tp % 192 == 0 and ks == 24 and nsa == 6)
    if v48:   # (i, gv) -> two drains (hf = strips 3 hf .. 3 hf + 2): [hf][gv][tq / 48][s'][tq % 48] x 64 halves
        VDRAIN = f"""    for (int b = 0; b < 18; b += 4) {{
      for (int f = 0; f < 4 && b + f < 18; f++) {{
        int i = (b + f) / 6, gv = (b + f) % 6 / 2, hf = (b + f) % 2, tq = rb * 3 + i;
        DMA_DRAIN(f, DESC(desc, 12), 12288 + i * 2304 + gv * 768 + hf * 384, (int)(out + {(QA + KB) * nh} + h * {VB} + (hf * {dhp // 96} + gv) * {GBV // 2} + ((tq / 48) * 3 * 48 + tq % 48) * 64));
      }}
      DMA_WAIT_ALL();
    }}"""
    # per head a switch on its runs (descriptor slot = run length, 1..6 strips)
    def fills(comp):   # comp 0 q, 1 k, 2 v: strips start at comp * nh * 16
        body = []
        for h in range(nh):
            rr = head_strip_runs(h, dh, strip0=comp * nh * 16); off = 0; calls = []
            for f, (g, st, cnt) in enumerate(rr):
                calls.append("DMA_FILL(%d, DESC(desc, %d), %d, (int)(ct + ((%d * %d + rb) * 6 + %d) * 192));" % (f, cnt, off, g, nrb, st))
                off += cnt * 768
            body.append("      %s (h == %d) { %s }" % ("if" if h == 0 else "else if", h, " ".join(calls)))
        return "\n".join(body)
    hd = dh // 2
    src = FULL_H + f"""
static inline float8 hsq(float8 v) {{ return v * v; }}
__kernel void qkv_dma(__global half* restrict out, __global float* restrict ct, __global float* restrict nq2, __global float* restrict nk2, __global float* restrict cs, __global float* restrict sn, __global float* restrict sc2, __global int* restrict desc{", __global int* restrict st" if mode == "stamps" else ""}, const int core_id) {{
  {"int acc[12]; for (int k = 0; k < 12; k++) acc[k] = 0; int t0 = __builtin_aipu_mfctrl0(0xd1);" if mode == "stamps" else ""}
  for (int u = core_id; u < {nrb * nh}; u += {nt}) {{
    int rb = u / {nh}, h = u % {nh};
    __global float* T = LSF(0); __global half* O = LSH(12288);
    float inv[12];
    /* ---- Q: fill, norm, rope, A tiles ---- */
{fills(0)}
    DMA_WAIT_ALL();
    {"{ int t = __builtin_aipu_mfctrl0(0xd1); acc[0] += t - t0; t0 = t; }" if mode == "stamps" else ""}
    DMA_FILL(0, DESC(desc, 11), 19200, (int)(sc2 + (0 * {nh} + h) * 512)); DMA_FILL(1, DESC(desc, 11), 21248, (int)(nq2)); DMA_WAIT_ALL();    /* the head's 64 scale quads, the norm weights */
    for (int st = 0; st < 16; st++) for (int jt = 0; jt < 4; jt++) {{
      float8 s8 = *(__global float8*)(LSF(19200) + (st * 4 + jt) * 8);
      for (int i = 0; i < 3; i++) {{ __global float* tp = T + ((st * 3 + i) * 4 + jt) * 16; *(__global float8*)(tp) *= s8; *(__global float8*)(tp + 8) *= s8; }}
    }}
    for (int i = 0; i < 3; i++) {{
      float8 a01 = BC(0.0f), a23 = BC(0.0f);
      for (int st = 0; st < 16; st++) for (int jt = 0; jt < 4; jt++) {{ __global float* tp = T + ((st * 3 + i) * 4 + jt) * 16; a01 += hsq(*(__global float8*)(tp)); a23 += hsq(*(__global float8*)(tp + 8)); }}
      inv[4 * i] = a01[0] + a01[1] + a01[2] + a01[3]; inv[4 * i + 1] = a01[4] + a01[5] + a01[6] + a01[7]; inv[4 * i + 2] = a23[0] + a23[1] + a23[2] + a23[3]; inv[4 * i + 3] = a23[4] + a23[5] + a23[6] + a23[7];
    }}
    for (int r = 0; r < 12; r++) inv[r] = (rb * 12 + r < {rows}) ? 1.0f / __builtin_sqrtf(inv[r] * {1.0 / dh!r}f + {eps!r}f) : 0.0f;
    {"{ int t = __builtin_aipu_mfctrl0(0xd1); acc[1] += t - t0; t0 = t; }" if mode == "stamps" else ""}
    for (int i = 0; i < 3; i++) {{
      float8 i01 = ROWS2(inv[4 * i], inv[4 * i + 1]), i23 = ROWS2(inv[4 * i + 2], inv[4 * i + 3]);
      DMA_FILL(0, DESC(desc, 10), 23296, (int)(cs + (rb * 3 + i) * 1024)); DMA_FILL(1, DESC(desc, 10), 27392, (int)(sn + (rb * 3 + i) * 1024)); DMA_WAIT_ALL();
      __global float* CS = LSF(23296); __global float* SN = LSF(27392);     /* the row group's 64 cos / sin tiles (⚠️ cached loads missed every unit: 42 us) */
      for (int kq = 0; kq < 32; kq++) {{              /* the rotation pairs quad kq with kq + 32 */
        __global float* ta = T + (((kq / 4) * 3 + i) * 4 + kq % 4) * 16; __global float* tb = T + ((((kq + 32) / 4) * 3 + i) * 4 + kq % 4) * 16;
        int cidx = kq * 16, cidx2 = (kq + 32) * 16;
        float8 wa = *(__global float8*)(LSF(21248) + kq * 8), wb = *(__global float8*)(LSF(21248) + (kq + 32) * 8);
        float8 qa01 = *(__global float8*)(ta) * i01 * wa, qa23 = *(__global float8*)(ta + 8) * i23 * wa;
        float8 qb01 = *(__global float8*)(tb) * i01 * wb, qb23 = *(__global float8*)(tb + 8) * i23 * wb;
        float8 c01 = *(__global float8*)(CS + cidx), c23 = *(__global float8*)(CS + cidx + 8), s01 = *(__global float8*)(SN + cidx), s23 = *(__global float8*)(SN + cidx + 8);
        float8 d01 = *(__global float8*)(CS + cidx2), d23 = *(__global float8*)(CS + cidx2 + 8), u01 = *(__global float8*)(SN + cidx2), u23 = *(__global float8*)(SN + cidx2 + 8);
        int kkA = kq, kkB = kq + 32;                /* A layout of the head: slice kk / 24, tile ((kk % 24) * 3 + i) */
        *(__global half16*)(O + (kkA / 24) * 2304 / 2 + ((kkA % 24) * 3 + i) * 16) = CVT16(qa01 * c01 - qb01 * s01, qa23 * c23 - qb23 * s23);
        *(__global half16*)(O + (kkB / 24) * 2304 / 2 + ((kkB % 24) * 3 + i) * 16) = CVT16(qb01 * d01 + qa01 * u01, qb23 * d23 + qa23 * u23);
      }}
      for (int kk = 64; kk < 72; kk++) *(__global half16*)(O + 2 * 1152 + ((kk % 24) * 3 + i) * 16) = CVT16(BC(0.0f), BC(0.0f));
    }}
{"{ int t = __builtin_aipu_mfctrl0(0xd1); acc[2] += t - t0; t0 = t; }" if mode == "stamps" else ""}
    DMA_DRAIN(0, DESC(desc, 7), 12288, (int)(out + h * {QA} + (0 * {nrb} + rb) * {ks * 48}));
    DMA_DRAIN(1, DESC(desc, 7), 12288 + 2304, (int)(out + h * {QA} + (1 * {nrb} + rb) * {ks * 48}));
    DMA_DRAIN(2, DESC(desc, 7), 12288 + 4608, (int)(out + h * {QA} + (2 * {nrb} + rb) * {ks * 48}));
    DMA_WAIT_ALL();
    {"{ int t = __builtin_aipu_mfctrl0(0xd1); acc[3] += t - t0; t0 = t; }" if mode == "stamps" else ""}
    /* ---- K: fill, norm, rope, B-panel tiles: per (i, slice) 24 tiles at the panel's 128-B kk pitch ---- */
{fills(1)}
    DMA_WAIT_ALL();
    {"{ int t = __builtin_aipu_mfctrl0(0xd1); acc[4] += t - t0; t0 = t; }" if mode == "stamps" else ""}
    DMA_FILL(0, DESC(desc, 11), 19200, (int)(sc2 + (1 * {nh} + h) * 512)); DMA_FILL(1, DESC(desc, 11), 21248, (int)(nk2)); DMA_WAIT_ALL();    /* the head's 64 scale quads, the norm weights */
    for (int st = 0; st < 16; st++) for (int jt = 0; jt < 4; jt++) {{
      float8 s8 = *(__global float8*)(LSF(19200) + (st * 4 + jt) * 8);
      for (int i = 0; i < 3; i++) {{ __global float* tp = T + ((st * 3 + i) * 4 + jt) * 16; *(__global float8*)(tp) *= s8; *(__global float8*)(tp + 8) *= s8; }}
    }}
    for (int i = 0; i < 3; i++) {{
      float8 a01 = BC(0.0f), a23 = BC(0.0f);
      for (int st = 0; st < 16; st++) for (int jt = 0; jt < 4; jt++) {{ __global float* tp = T + ((st * 3 + i) * 4 + jt) * 16; a01 += hsq(*(__global float8*)(tp)); a23 += hsq(*(__global float8*)(tp + 8)); }}
      inv[4 * i] = a01[0] + a01[1] + a01[2] + a01[3]; inv[4 * i + 1] = a01[4] + a01[5] + a01[6] + a01[7]; inv[4 * i + 2] = a23[0] + a23[1] + a23[2] + a23[3]; inv[4 * i + 3] = a23[4] + a23[5] + a23[6] + a23[7];
    }}
    for (int r = 0; r < 12; r++) inv[r] = (rb * 12 + r < {rows}) ? 1.0f / __builtin_sqrtf(inv[r] * {1.0 / dh!r}f + {eps!r}f) : 0.0f;
    {"{ int t = __builtin_aipu_mfctrl0(0xd1); acc[5] += t - t0; t0 = t; }" if mode == "stamps" else ""}
    for (int i = 0; i < 3; i++) {{
      float8 i01 = ROWS2(inv[4 * i], inv[4 * i + 1]), i23 = ROWS2(inv[4 * i + 2], inv[4 * i + 3]);
      DMA_FILL(0, DESC(desc, 10), 23296, (int)(cs + (rb * 3 + i) * 1024)); DMA_FILL(1, DESC(desc, 10), 27392, (int)(sn + (rb * 3 + i) * 1024)); DMA_WAIT_ALL();
      __global float* CS = LSF(23296); __global float* SN = LSF(27392);
      for (int kq = 0; kq < 32; kq++) {{
        __global float* ta = T + (((kq / 4) * 3 + i) * 4 + kq % 4) * 16; __global float* tb = T + ((((kq + 32) / 4) * 3 + i) * 4 + kq % 4) * 16;
        int cidx = kq * 16, cidx2 = (kq + 32) * 16;
        float8 wa = *(__global float8*)(LSF(21248) + kq * 8), wb = *(__global float8*)(LSF(21248) + (kq + 32) * 8);
        float8 qa01 = *(__global float8*)(ta) * i01 * wa, qa23 = *(__global float8*)(ta + 8) * i23 * wa;
        float8 qb01 = *(__global float8*)(tb) * i01 * wb, qb23 = *(__global float8*)(tb + 8) * i23 * wb;
        float8 c01 = *(__global float8*)(CS + cidx), c23 = *(__global float8*)(CS + cidx + 8), s01 = *(__global float8*)(SN + cidx), s23 = *(__global float8*)(SN + cidx + 8);
        float8 d01 = *(__global float8*)(CS + cidx2), d23 = *(__global float8*)(CS + cidx2 + 8), u01 = *(__global float8*)(SN + cidx2), u23 = *(__global float8*)(SN + cidx2 + 8);
        int kkA = kq, kkB = kq + 32;                /* out region: [slice][i][kk % 24] tiles, 768 B per (slice, i) */
        *(__global half16*)(O + ((kkA / 24) * 3 + i) * 384 + (kkA % 24) * 16) = CVT16(qa01 * c01 - qb01 * s01, qa23 * c23 - qb23 * s23);
        *(__global half16*)(O + ((kkB / 24) * 3 + i) * 384 + (kkB % 24) * 16) = CVT16(qb01 * d01 + qa01 * u01, qb23 * d23 + qa23 * u23);
      }}
      for (int kk = 64; kk < 72; kk++) *(__global half16*)(O + (2 * 3 + i) * 384 + (kk % 24) * 16) = CVT16(BC(0.0f), BC(0.0f));
    }}
{"{ int t = __builtin_aipu_mfctrl0(0xd1); acc[6] += t - t0; t0 = t; }" if mode == "stamps" else ""}
    for (int b = 0; b < 9; b += 4) {{
      for (int f = 0; f < 4 && b + f < 9; f++) {{
        int sl = (b + f) / 3, i = (b + f) % 3, tq = rb * 3 + i;   /* the row group's token quad: group tq / 24, strip tq / 4 % 6, tile tq % 4 */
        DMA_DRAIN(f, DESC(desc, 8), 12288 + (b + f) * 768, (int)(out + {QA * nh} + h * {KB} + (tq / 24) * {GB} + ((sl * {nsa} + (tq / 4) % {nsa}) * {ks} * 4 + tq % 4) * 16));
      }}
      DMA_WAIT_ALL();
    }}
    {"{ int t = __builtin_aipu_mfctrl0(0xd1); acc[7] += t - t0; t0 = t; }" if mode == "stamps" else ""}
    /* ---- V^T: fill, transpose each tile, B-panel tiles: per i (a token quad = one kk) the 18 strips' 4 tiles ---- */
{fills(2)}
    DMA_WAIT_ALL();
    {"{ int t = __builtin_aipu_mfctrl0(0xd1); acc[8] += t - t0; t0 = t; }" if mode == "stamps" else ""}
    DMA_FILL(0, DESC(desc, 11), 19200, (int)(sc2 + (2 * {nh} + h) * 512)); DMA_WAIT_ALL();    /* the head's 64 scale quads */
    for (int st = 0; st < 16; st++) for (int jt = 0; jt < 4; jt++) {{
      float8 s8 = *(__global float8*)(LSF(19200) + (st * 4 + jt) * 8);
      for (int i = 0; i < 3; i++) {{ __global float* tp = T + ((st * 3 + i) * 4 + jt) * 16; *(__global float8*)(tp) *= s8; *(__global float8*)(tp + 8) *= s8; }}
    }}
    for (int i = 0; i < 3; i++) {{
      for (int dq = 0; dq < 72; dq++) {{                /* dim quad dq -> strip dq / 4, tile dq % 4; dims >= 256: zero */
        half16 hv;
        if (dq < 64) {{
          __global float* tp = T + (((dq / 4) * 3 + i) * 4 + dq % 4) * 16;              /* [r][c] */
          half16 h = TILE16(tp);                                                           /* the fp16 tile [r][c] */
          hv = __builtin_shufflevector(h, h, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);   /* [c][r]: one 16-lane permute (two float8 shuffles: 21.6 us a unit; 16 scalar copies through LSRAM: 27 us) */
        }} else hv = CVT16(BC(0.0f), BC(0.0f));
        *(__global half16*)(O + i * 1152 + dq * 16) = hv;                         /* out region: [i][strip][jt] tiles = [i][dq] */
      }}
    }}
{"{ int t = __builtin_aipu_mfctrl0(0xd1); acc[9] += t - t0; t0 = t; }" if mode == "stamps" else ""}
{VDRAIN if v48 else f"""    for (int b = 0; b < 9; b += 4) {{                   /* per (i, dim group gv): 6 strips x 4 tiles at the panel's ks x 128 B strip pitch */
      for (int f = 0; f < 4 && b + f < 9; f++) {{
        int i = (b + f) / 3, gv = (b + f) % 3, tq = rb * 3 + i;   /* the token quad = kk tq % 24 of slice tq / 24 */
        DMA_DRAIN(f, DESC(desc, 9), 12288 + i * 2304 + gv * 768, (int)(out + {(QA + KB) * nh} + h * {VB} + gv * {GBV} + ((tq / {ks}) * {nsa * ks} + tq % {ks}) * 64));
      }}
      DMA_WAIT_ALL();
    }}"""}
    {"{ int t = __builtin_aipu_mfctrl0(0xd1); acc[10] += t - t0; t0 = t; }" if mode == "stamps" else ""}
  }}
  {"for (int k = 0; k < 12; k++) st[core_id * 12 + k] = acc[k];" if mode == "stamps" else ""}
}}"""
    if mode == "noV":
        a_, b_ = src.index("    /* ---- V^T"), src.rindex("  }\n}"); src = src[:a_] + src[b_:]
    if fast: src = _qkv_fast(src, nh, dh, rows, eps, mode == "stamps")
    return src


def _qkv_fast(src: str, nh: int, dh: int, rows: int, eps: float, stamps: bool) -> str:
    """qkv_dma's Q / K / V bodies sized for the TEC's loop buffer, bit-identical: scale folded into each use, pointer
    increments, V^T transposed in fp32 before conversion. Spliced over the generated source."""
    ST = lambda k: ("      { int t = __builtin_aipu_mfctrl0(0xd1); acc[%d] += t - t0; t0 = t; }\n" % k) if stamps else ""
    def norm(c):
        return f"""    DMA_FILL(0, DESC(desc, 11), 19200, (int)(sc2 + ({c} * {nh} + h) * 512)); DMA_FILL(1, DESC(desc, 11), 21248, (int)({"nq2" if c == 0 else "nk2"})); DMA_WAIT_ALL();
    __global float* SC = LSF(19200); __global float* WN = LSF(21248);
    {"DMA_FILL(3, DESC(desc, 13), 23296, (int)(cs + (rb * 3) * 1024));    /* Q row group 0's half tables, behind the norm */" if c == 0 else ""}
    #pragma clang loop unroll(disable)
    for (int i = 0; i < 3; i++) {{
      float8 a01 = BC(0.0f), a23 = BC(0.0f);
      __global float* tp = T + i * 64; __global float* sp = SC;
      #pragma clang loop unroll(disable)
      for (int st = 0; st < 16; st++) {{
        for (int jt = 0; jt < 4; jt++) {{ float8 s8 = *(__global float8*)(sp + jt * 8); float8 v01 = *(__global float8*)(tp + jt * 16) * s8; float8 v23 = *(__global float8*)(tp + jt * 16 + 8) * s8; a01 += hsq(v01); a23 += hsq(v23); }}
        tp += 192; sp += 32;
      }}
      inv[4 * i] = a01[0] + a01[1] + a01[2] + a01[3]; inv[4 * i + 1] = a01[4] + a01[5] + a01[6] + a01[7]; inv[4 * i + 2] = a23[0] + a23[1] + a23[2] + a23[3]; inv[4 * i + 3] = a23[4] + a23[5] + a23[6] + a23[7];
    }}
    for (int r = 0; r < 12; r++) inv[r] = (rb * 12 + r < {rows}) ? 1.0f / __builtin_sqrtf(inv[r] * {1.0 / dh!r}f + {eps!r}f) : 0.0f;
"""
    def rope(c):
        if c == 0: oa = lambda k: (k // 24) * 1152 + (k % 24) * 48; os_, istr = 48, 16          # the A layout: [slice][kk % 24][3 i][16]
        else: oa = lambda k: (k // 24) * 3 * 384 + (k % 24) * 16; os_, istr = 16, 384          # the out region: [slice][i][kk % 24][16]
        segs = ""
        for k0, n in ((0, 16), (16, 8), (24, 8)):
            segs += f"""      {{
        __global float* ta = T + {k0 // 4 * 192} + i * 64; __global float* sa = SC + {k0 * 8}; __global float* wa = WN + {k0 * 8};
        __global float* ca = CS + {k0 * 16}; __global float* na = SN + {k0 * 16};
        __global half* oa = O + {oa(k0)} + i * {istr}; __global half* ob = O + {oa(k0 + 32)} + i * {istr};
        #pragma clang loop unroll(disable)
        for (int q = 0; q < {n // 2}; q++) {{
          for (int u = 0; u < 2; u++) {{
            float8 sA = *(__global float8*)(sa + u * 8), sB = *(__global float8*)(sa + 256 + u * 8);
            float8 va01 = *(__global float8*)(ta + u * 16) * sA; float8 va23 = *(__global float8*)(ta + u * 16 + 8) * sA;
            float8 vb01 = *(__global float8*)(ta + 1536 + u * 16) * sB; float8 vb23 = *(__global float8*)(ta + 1536 + u * 16 + 8) * sB;
            float8 wa8 = *(__global float8*)(wa + u * 8), wb8 = *(__global float8*)(wa + 256 + u * 8);
            float8 qa01 = va01 * i01 * wa8, qa23 = va23 * i23 * wa8;
            float8 qb01 = vb01 * i01 * wb8, qb23 = vb23 * i23 * wb8;
            float8 c01 = *(__global float8*)(ca + u * 16), c23 = *(__global float8*)(ca + u * 16 + 8), s01 = *(__global float8*)(na + u * 16), s23 = *(__global float8*)(na + u * 16 + 8);
            float8 d01 = c01, d23 = c23, u01 = s01, u23 = s23;         /* quad kq + 32's cos / sin = quad kq's (rope_csn) */
            *(__global half16*)(oa + u * {os_}) = CVT16(qa01 * c01 - qb01 * s01, qa23 * c23 - qb23 * s23);
            *(__global half16*)(ob + u * {os_}) = CVT16(qb01 * d01 + qa01 * u01, qb23 * d23 + qa23 * u23);
          }}
          int odd = q & 1; ta += 32 + odd * 128; sa += 16; wa += 16; ca += 32; na += 32; oa += {2 * os_}; ob += {2 * os_};
        }}
      }}
"""
        zero = ("      for (int kk = 64; kk < 72; kk++) *(__global half16*)(O + 2 * 1152 + ((kk % 24) * 3 + i) * 16) = CVT16(BC(0.0f), BC(0.0f));\n" if c == 0
                else "      for (int kk = 64; kk < 72; kk++) *(__global half16*)(O + (2 * 3 + i) * 384 + (kk % 24) * 16) = CVT16(BC(0.0f), BC(0.0f));\n")
        return f"""    for (int i = 0; i < 3; i++) {{
      float8 i01 = ROWS2(inv[4 * i], inv[4 * i + 1]), i23 = ROWS2(inv[4 * i + 2], inv[4 * i + 3]);
      {"{ int t = __builtin_aipu_mfctrl0(0xd1); acc[%d] += t - t0; t0 = t; }" % (2 if c == 0 else 6) if stamps else ""}
      DMA_WAIT(3);                                      /* this row group's half tables (issued one step ahead) */
      {"{ int t = __builtin_aipu_mfctrl0(0xd1); acc[11] += t - t0; t0 = t; }" if stamps else ""}
      int slot = ({c * 3} + i) & 1;
      {"" if c == 1 else "DMA_FILL(3, DESC(desc, 13), 23296 + (slot ^ 1) * 4096, (int)(cs + (rb * 3 + (i + 1) % 3) * 1024));   /* the next step's: Q's next row group, or K's first */"}
      {"if (i < 2) DMA_FILL(3, DESC(desc, 13), 23296 + (slot ^ 1) * 4096, (int)(cs + (rb * 3 + i + 1) * 1024));" if c == 1 else ""}
      __global float* CS = LSF(23296 + slot * 4096); __global float* SN = CS + 512;
{segs}{zero}    }}
"""
    def vt():
        return f"""    DMA_FILL(0, DESC(desc, 11), 19200, (int)(sc2 + (2 * {nh} + h) * 512)); DMA_WAIT_ALL();    /* the head's 64 scale quads */
    __global float* SC = LSF(19200);
    #pragma clang loop unroll(disable)
    for (int i = 0; i < 3; i++) {{
      __global float* tp = T + i * 64; __global float* sp = SC; __global half* o = O + i * 1152;
      #pragma clang loop unroll(disable)
      for (int st = 0; st < 16; st++) {{
        for (int jt = 0; jt < 4; jt++) {{
          float8 s8 = *(__global float8*)(sp + jt * 8); float8 t01 = *(__global float8*)(tp + jt * 16) * s8; float8 t23 = *(__global float8*)(tp + jt * 16 + 8) * s8;
          float8 e = __builtin_shufflevector(t01, t23, 0, 2, 4, 6, 8, 10, 12, 14), f = __builtin_shufflevector(t01, t23, 1, 3, 5, 7, 9, 11, 13, 15);
          *(__global half16*)(o + jt * 16) = CVT16(__builtin_shufflevector(e, f, 0, 2, 4, 6, 8, 10, 12, 14), __builtin_shufflevector(e, f, 1, 3, 5, 7, 9, 11, 13, 15));   /* [c][r] */
        }}
        tp += 192; sp += 32; o += 64;
      }}
      for (int dq = 64; dq < 72; dq++) *(__global half16*)(O + i * 1152 + dq * 16) = CVT16(BC(0.0f), BC(0.0f));
    }}
"""
    def cut(text, start, end_marker, new):
        a_ = text.index(start); b_ = text.index(end_marker, a_)
        return text[:a_] + new + text[b_:]
    q0 = "    DMA_FILL(0, DESC(desc, 11), 19200, (int)(sc2 + (0 * %d + h) * 512));" % nh
    src = cut(src, q0, "    DMA_DRAIN(0, DESC(desc, 7), 12288,", "    {\n" + norm(0) + ST(1) + rope(0) + ST(2) + "    }\n")
    k0 = "    DMA_FILL(0, DESC(desc, 11), 19200, (int)(sc2 + (1 * %d + h) * 512));" % nh
    src = cut(src, k0, "    for (int b = 0; b < 9; b += 4) {", "    {\n" + norm(1) + ST(5) + rope(1) + ST(6) + "    }\n")
    v0 = "    DMA_FILL(0, DESC(desc, 11), 19200, (int)(sc2 + (2 * %d + h) * 512));" % nh
    src = cut(src, v0, "    for (int b = 0; b < ", "    {\n" + vt() + ST(9) + "    }\n")
    return src


def qkv_dma_descs(nrb: int, ks: int, nsa: int) -> np.ndarray:
    """Slots: 1..6 a fill of n strips, 7 an A slice, 8 the K-panel drain, 9 the V^T drain, 10 cos / sin, 11 a head's
    scale / norm quads, 12 the v48 V^T drain, 13 `rope_csn` half tables."""
    return _desc_slots((64,), *[(n * 768,) for n in range(1, 7)], (2304,), (768, 32, 128, 32), (768, 128, ks * 128, 128), (4096,), (2048,), (384, 128, 48 * 128, 128), (4096,))


def rope_csn(cs_tiles: np.ndarray, sn_tiles: np.ndarray) -> np.ndarray:
    """`tile_rows(cos / sin)` -> per row group [cos quads 0..31 | sin quads 0..31]: `qkv_dma(fast)`'s `cs`. The tables are
    concat(f, f), so the second halves are redundant (asserted)."""
    c = cs_tiles.reshape(-1, 1024); s_ = sn_tiles.reshape(-1, 1024)
    assert np.array_equal(c[:, :512], c[:, 512:]) and np.array_equal(s_[:, :512], s_[:, 512:]), "rope_csn: the tables' halves differ"
    return np.ascontiguousarray(np.concatenate([c[:, :512], s_[:, :512]], 1)).ravel()


# ---- S5: softmax, staged ----
def softmax_dma_src(rows: int, nh: int, tp: int, nsa: int, nrb: int, ks: int, scale: float, nt: int = NT) -> str:
    """`P = fp16(softmax(C * scale))` per head (tp = 288; head h at `ct + h * LG`, `out + h * PA`), a (row block, head)
    per unit: one strided fill, one strided drain; columns >= rows zero. args: out, ct, desc (`softmax_dma_descs`)."""
    assert tp == 288 and 4 * ks == 96 and nsa == 6 and 12 * nrb <= tp
    scale = float(scale); LG = 3 * nrb * 1152; PA = 3 * nrb * ks * 48
    full = rows // 4                                 # column quads entirely < rows
    return FULL_H + f"""
__kernel void softmax_dma(__global half* restrict out, __global float* restrict ct, __global int* restrict desc, const int core_id) {{
  for (int u = core_id; u < {nrb * nh}; u += {nt}) {{
    int rb = u / {nh}, h = u % {nh};
    __global float* T = LSF(0); __global half* O = LSH(13824);
    DMA_FILL(0, DESC(desc, 0), 0, (int)(ct + h * {LG} + rb * 1152));
    DMA_WAIT_ALL();
    for (int i = 0; i < 3; i++) {{
      float8 m01 = BC(-3.0e38f), m23 = BC(-3.0e38f);
      for (int q = 0; q < {full}; q++) {{ __global float* tp = T + (((q / 4) * 3 + i) * 4 + q % 4) * 16; m01 = VMAX(m01, *(__global float8*)(tp)); m23 = VMAX(m23, *(__global float8*)(tp + 8)); }}
      float mx[4] = {{ m01[0] > m01[1] ? m01[0] : m01[1], m01[4] > m01[5] ? m01[4] : m01[5], m23[0] > m23[1] ? m23[0] : m23[1], m23[4] > m23[5] ? m23[4] : m23[5] }};
      mx[0] = mx[0] > m01[2] ? mx[0] : m01[2]; mx[0] = mx[0] > m01[3] ? mx[0] : m01[3]; mx[1] = mx[1] > m01[6] ? mx[1] : m01[6]; mx[1] = mx[1] > m01[7] ? mx[1] : m01[7];
      mx[2] = mx[2] > m23[2] ? mx[2] : m23[2]; mx[2] = mx[2] > m23[3] ? mx[2] : m23[3]; mx[3] = mx[3] > m23[6] ? mx[3] : m23[6]; mx[3] = mx[3] > m23[7] ? mx[3] : m23[7];
      float8 x01 = ROWS2(mx[0], mx[1]) * BC({scale * LOG2E!r}f), x23 = ROWS2(mx[2], mx[3]) * BC({scale * LOG2E!r}f);
      float8 s01 = BC(0.0f), s23 = BC(0.0f);
      for (int q = 0; q < {full}; q++) {{ __global float* tp = T + (((q / 4) * 3 + i) * 4 + q % 4) * 16; s01 += exp2_f8(*(__global float8*)(tp) * BC({scale * LOG2E!r}f) - x01); s23 += exp2_f8(*(__global float8*)(tp + 8) * BC({scale * LOG2E!r}f) - x23); }}
      int m0 = rb * 12 + 4 * i;
      float v0 = m0 < {rows} ? 1.0f / (s01[0] + s01[1] + s01[2] + s01[3]) : 0.0f, v1 = m0 + 1 < {rows} ? 1.0f / (s01[4] + s01[5] + s01[6] + s01[7]) : 0.0f;
      float v2 = m0 + 2 < {rows} ? 1.0f / (s23[0] + s23[1] + s23[2] + s23[3]) : 0.0f, v3 = m0 + 3 < {rows} ? 1.0f / (s23[4] + s23[5] + s23[6] + s23[7]) : 0.0f;
      float8 i01 = ROWS2(v0, v1), i23 = ROWS2(v2, v3);
      for (int q = 0; q < 72; q++) {{
        __global float* tp = T + (((q / 4) * 3 + i) * 4 + q % 4) * 16;
        float8 p01 = q < {full} ? exp2_f8(*(__global float8*)(tp) * BC({scale * LOG2E!r}f) - x01) * i01 : BC(0.0f);
        float8 p23 = q < {full} ? exp2_f8(*(__global float8*)(tp + 8) * BC({scale * LOG2E!r}f) - x23) * i23 : BC(0.0f);
        *(__global half16*)(O + (q / 24) * 1152 + ((q % 24) * 3 + i) * 16) = CVT16(p01, p23);
      }}
    }}
    DMA_DRAIN(1, DESC(desc, 1), 13824, (int)(out + h * {PA} + rb * {ks * 48}));
    DMA_WAIT_ALL();
  }}
}}"""


def softmax_dma_descs(nrb: int, ks: int) -> np.ndarray:
    return _desc_slots((13824, 4608, nrb * 4608, 4608), (6912, 2304, nrb * 2304, 2304))


def softmax_dma_ref(cfull_heads, scale, rows, ks, nrb):
    """cfull_heads: list of nh fp32 `[12 nrb, tp]`; the concatenated per-head P layouts."""
    return np.concatenate([softmax_a16_ref(c, scale, rows, ks, nrb) for c in cfull_heads])


# ---- S6: gather heads, staged ----
def gather_o_dma_src(rows: int, nh: int, dh: int, dhp: int, nsa: int, nrb: int, ks: int, nt: int = NT) -> str:
    """Each head's P V tiles (dhp = 288; `ct + h * HC`) -> the o-projection's A (K = nh dh), a (row block, head) per unit."""
    assert dh == 256 and dhp == 288 and 4 * ks == 96 and nsa == 6
    HC = 3 * nrb * 1152
    def drains():
        body = []
        for h in range(nh):
            q0 = 64 * h; calls = []; f = 0; q = q0
            while q < q0 + 64:
                sl, kk = q // 24, q % 24; cnt = min(24 - kk, q0 + 64 - q)
                calls.append("DMA_DRAIN(%d, DESC(desc, %d), %d, (int)(out + (%d * %d + rb) * %d + %d * 48));" % (f, cnt, 13824 + (q - q0) * 96, sl, nrb, ks * 48, kk))
                q += cnt; f += 1
            body.append("    %s (h == %d) { %s }" % ("if" if h == 0 else "else if", h, " ".join(calls)))
        return "\n".join(body)
    return FULL_H + f"""
__kernel void gather_o_dma(__global half* restrict out, __global float* restrict ct, __global int* restrict desc, const int core_id) {{
  for (int u = core_id; u < {nrb * nh}; u += {nt}) {{
    int rb = u / {nh}, h = u % {nh};
    __global float* T = LSF(0); __global half* O = LSH(13824);
    DMA_FILL(0, DESC(desc, 0), 0, (int)(ct + h * {HC} + rb * 1152));
    DMA_WAIT_ALL();
    for (int q = 0; q < 64; q++) for (int i = 0; i < 3; i++) {{
      __global float* tp = T + (((q / 4) * 3 + i) * 4 + q % 4) * 16;
      *(__global half16*)(O + (q * 3 + i) * 16) = TILE16(tp);
    }}
{drains()}
    DMA_WAIT_ALL();
  }}
}}"""


def gather_o_dma_descs(nrb: int) -> np.ndarray:
    """Slot 0 = the fill (3 tile blocks at the group pitch), slots 1..24 = a drain of n quads (n x 96 B)."""
    return _desc_slots((13824, 4608, nrb * 4608, 4608), *[(n * 96,) for n in range(1, 25)])



# ---- S5b: softmax over any key count, streamed ----
def _PASS1_SCAN(gb, ng, nrb, RB, treal, stamps, causal=False):
    """softmax_stream's pass 1 over the logits themselves (the row maxima, lane-wise)."""
    return f"""/* pass 1: the row maxima (lane-wise, reduced below) */
    DMA_FILL(0, DESC(desc, 0), 0, (int)base);
    for (int gc = 0; gc < {ng // gb}; gc++) {{
      int s = gc & 1;
      if (gc + 1 < {ng // gb}) DMA_FILL(s ^ 1, DESC(desc, 0), (s ^ 1) * {gb * 4608}, (int)(base + (gc + 1) * {gb * nrb * RB}));
      {"{ int t_ = __builtin_aipu_mfctrl0(0xd1); acc_[0] += t_ - t0_; t0_ = t_; }" if stamps else ""}
      DMA_WAIT(s);
      {"{ int t_ = __builtin_aipu_mfctrl0(0xd1); acc_[1] += t_ - t0_; t0_ = t_; }" if stamps else ""}
      for (int gg = 0; gg < {gb}; gg++) {{
      int g = gc * {gb} + gg;
      __global float* T = LSF(s * {gb * 4608} + gg * 4608);
      int full = {"(g * 96 + 95 <= rb * 12) && " if causal else ""}(g * 96 + 96 <= {treal});
      for (int st = 0; st < 6; st++) for (int i = 0; i < 3; i++) for (int jt = 0; jt < 4; jt++) {{
        __global float* tp_ = T + ((st * 3 + i) * 4 + jt) * 16;
        float8 v01 = *(__global float8*)(tp_), v23 = *(__global float8*)(tp_ + 8);
        if (!full) {{ {"v01 = v01 + kmask2(g, st, jt, rb * 12 + 4 * i); v23 = v23 + kmask2(g, st, jt, rb * 12 + 4 * i + 2);" if causal else "float8 mk = kmask(g, st, jt); v01 = v01 + mk; v23 = v23 + mk;"} }}
        m01[i] = VMAX(m01[i], v01); m23[i] = VMAX(m23[i], v23);
      }}
      }}
    }}"""


def softmax_stream_src(treal: int, nh: int, tp: int, nrb: int, ks: int, scale: float, hb: int = 0, nt: int = NT, gb: int = 2, stamps: bool = False, fast: bool = True, rowmax: bool = False, p2t: int = 5, causal: bool = False) -> str:
    """Unnormalized `P = fp16(exp(C * scale - rowmax))` for `nh` heads of logit tiles (N = tp; head h at `ct + h * LG`)
    in the A layout (head h at `out + h * PA`), and each row's `1 / sum` into `sums` (`[hb + h][rb][16]`: a 64-B line per
    unit), applied by `gather_o_sum`. Keys >= `treal` masked. A (row block, head) per unit, streamed twice by group:
    pass 1 row maxima, pass 2 exp, sums, A slices (fills flags 0/1, drains 2/3). `gb`: groups per DMA request.
    `rowmax`: logits from `gemm_gs(rowmax=True)`; pass 1 uses their maxima, scanning only groups with masked keys.
    `p2t`: pass-2 body (2: rolled two-tile, 3: `P2F`, 5: `P2P`, else 4-tile). `causal`: mask keys after the row.
    args: out, ct, sums, desc (`softmax_stream_descs(nrb, gb, rowmax=ng if rowmax)`)."""
    assert not (causal and rowmax), "causal: the GEMM's maxima include the future keys"
    KMASK2 = ("""
/* causal: lanes 0-3 are row `row`, 4-7 row `row + 1`; a key is kept if < treal and <= its row */
static inline float8 kmask2(int g, int st, int jt, int row) {
  float8 idx = (float8){0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 2.0f, 3.0f} + BC((float)(g * 96 + st * 16 + jt * 4));
  float8 rw = (float8){0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f} + BC((float)row);
  return VMIN(idx < BC(%rf) ? BC(0.0f) : BC(-3.0e38f), idx <= rw ? BC(0.0f) : BC(-3.0e38f));
}""" % float(treal)) if causal else ""
    assert tp % 96 == 0 and 4 * ks == 96 and 12 * nrb <= tp and (tp // 96) % gb == 0
    RB = 1200 if rowmax else 1152                   # floats per (group, row block)
    scale = float(scale); ng = tp // 96; LG = ng * nrb * RB; PA = nrb * 12 * tp
    ngf, ngp = treal // 96, -(-treal // 96)         # the groups wholly below treal; those with any key below it
    sl2 = scale * LOG2E
    EXP = "exp2_le0" if fast else "exp2_f8"
    PASS1 = _PASS1_SCAN(gb, ng, nrb, RB, treal, stamps, causal) if not rowmax else """/* pass 1: the GEMM's maxima of the full groups (one strided fill), then a scan of the groups with masked keys */
    DMA_FILL(0, DESC(desc, 3), 0, (int)(base + 1152)); DMA_WAIT(0);
    {
      __global float* MX = LSF(0);
      for (int g = 0; g < %d; g++) for (int i = 0; i < 3; i++) {
        m01[i] = VMAX(m01[i], *(__global float8*)(MX + g * 48 + 16 * i)); m23[i] = VMAX(m23[i], *(__global float8*)(MX + g * 48 + 16 * i + 8));
      }
    }
    for (int g = %d; g < %d; g++) {
      DMA_FILL(0, DESC(desc, 4), 0, (int)(base + g * %d)); DMA_WAIT(0);
      __global float* T = LSF(0);
      for (int st = 0; st < 6; st++) for (int i = 0; i < 3; i++) for (int jt = 0; jt < 4; jt++) {
        __global float* tp_ = T + ((st * 3 + i) * 4 + jt) * 16;
        float8 mk = kmask(g, st, jt);
        m01[i] = VMAX(m01[i], *(__global float8*)(tp_) + mk); m23[i] = VMAX(m23[i], *(__global float8*)(tp_ + 8) + mk);
      }
    }""" % (ngf, ngf, ngp, nrb * RB)
    P2A = f'''      for (int st = 0; st < 6; st++) for (int i = 0; i < 3; i++) {{
        /* the row group's 4 column tiles at once: 8 independent exp chains for the scheduler */
        __global float* tp_ = T + ((st * 3 + i) * 4) * 16;
        float8 a0 = *(__global float8*)(tp_) * BC({sl2!r}f), b0 = *(__global float8*)(tp_ + 8) * BC({sl2!r}f);
        float8 a1 = *(__global float8*)(tp_ + 16) * BC({sl2!r}f), b1 = *(__global float8*)(tp_ + 24) * BC({sl2!r}f);
        float8 a2 = *(__global float8*)(tp_ + 32) * BC({sl2!r}f), b2 = *(__global float8*)(tp_ + 40) * BC({sl2!r}f);
        float8 a3 = *(__global float8*)(tp_ + 48) * BC({sl2!r}f), b3 = *(__global float8*)(tp_ + 56) * BC({sl2!r}f);
        if (!full) {{
          {"int ra = rb * 12 + 4 * i; a0 += kmask2(g, st, 0, ra); b0 += kmask2(g, st, 0, ra + 2); a1 += kmask2(g, st, 1, ra); b1 += kmask2(g, st, 1, ra + 2); a2 += kmask2(g, st, 2, ra); b2 += kmask2(g, st, 2, ra + 2); a3 += kmask2(g, st, 3, ra); b3 += kmask2(g, st, 3, ra + 2);" if causal else "float8 m0 = kmask(g, st, 0), m1 = kmask(g, st, 1), m2 = kmask(g, st, 2), m3 = kmask(g, st, 3);" + chr(10) + "          a0 += m0; b0 += m0; a1 += m1; b1 += m1; a2 += m2; b2 += m2; a3 += m3; b3 += m3;"}
        }}
        float8 e0 = {EXP}(a0 - x01[i]), f0 = {EXP}(b0 - x23[i]), e1 = {EXP}(a1 - x01[i]), f1 = {EXP}(b1 - x23[i]);
        float8 e2 = {EXP}(a2 - x01[i]), f2 = {EXP}(b2 - x23[i]), e3 = {EXP}(a3 - x01[i]), f3 = {EXP}(b3 - x23[i]);
        s01[i] += (e0 + e1) + (e2 + e3); s23[i] += (f0 + f1) + (f2 + f3);
        __global half* o_ = O + ((st * 4) * 3 + i) * 16;
        *(__global half16*)(o_) = CVT16(e0, f0); *(__global half16*)(o_ + 48) = CVT16(e1, f1);
        *(__global half16*)(o_ + 96) = CVT16(e2, f2); *(__global half16*)(o_ + 144) = CVT16(e3, f3);
      }}
'''
    P2N = f'''      #pragma clang loop unroll(disable)
      for (int i = 0; i < 3; i++) {{
        float8 xa = x01[i], xb = x23[i], sa = s01[i], sb = s23[i];
        #pragma clang loop unroll(disable)
        for (int q = 0; q < 12; q++) {{                 /* strip q / 2, tiles 2 (q % 2) .. + 1 */
          int st = q >> 1, j0 = (q & 1) * 2;
          __global float* tp_ = T + ((st * 3 + i) * 4 + j0) * 16;
          float8 a0 = *(__global float8*)(tp_) * BC({sl2!r}f), b0 = *(__global float8*)(tp_ + 8) * BC({sl2!r}f);
          float8 a1 = *(__global float8*)(tp_ + 16) * BC({sl2!r}f), b1 = *(__global float8*)(tp_ + 24) * BC({sl2!r}f);
          if (!full) {{ float8 m0 = kmask(g, st, j0), m1 = kmask(g, st, j0 + 1); a0 += m0; b0 += m0; a1 += m1; b1 += m1; }}
          float8 e0 = {EXP}(a0 - xa), f0 = {EXP}(b0 - xb), e1 = {EXP}(a1 - xa), f1 = {EXP}(b1 - xb);
          sa += e0 + e1; sb += f0 + f1;
          __global half* o_ = O + ((st * 4 + j0) * 3 + i) * 16;
          *(__global half16*)(o_) = CVT16(e0, f0); *(__global half16*)(o_ + 48) = CVT16(e1, f1);
        }}
        s01[i] = sa; s23[i] = sb;
      }}
'''
    # p2t = 3: full groups through a branch-free two-tile pointer-increment loop, partial groups through the 4-tile code
    P2F = f'''      if (full) {{
      #pragma clang loop unroll(disable)
      for (int i = 0; i < 3; i++) {{
        float8 xa = x01[i], xb = x23[i], sa = s01[i], sb = s23[i];
        __global float* tq = T + i * 64; __global half* oq = O + i * 16;
        #pragma clang loop unroll(disable)
        for (int q = 0; q < 12; q++) {{                 /* tiles (st = q / 2, jt = 2 (q % 2) .. + 1): tq / oq step 32 / 96, then 160 / 96 */
          float8 a0 = *(__global float8*)(tq) * BC({sl2!r}f), b0 = *(__global float8*)(tq + 8) * BC({sl2!r}f);
          float8 a1 = *(__global float8*)(tq + 16) * BC({sl2!r}f), b1 = *(__global float8*)(tq + 24) * BC({sl2!r}f);
          float8 e0 = {EXP}(a0 - xa), f0 = {EXP}(b0 - xb), e1 = {EXP}(a1 - xa), f1 = {EXP}(b1 - xb);
          sa += e0 + e1; sb += f0 + f1;
          *(__global half16*)(oq) = CVT16(e0, f0); *(__global half16*)(oq + 48) = CVT16(e1, f1);
          int odd = q & 1; tq += 32 + odd * 128; oq += 96;
        }}
        s01[i] = sa; s23[i] = sb;
      }}
      }} else {{
{P2A}      }}
'''
    # p2t = 5: p2t=3 software-pipelined (stage A of the next two tiles beside stage B of these): same ops, same bits.
    # The scaled logits stay in their own statement so the compiler cannot fuse * and - into an fma.
    SA = lambda n: f"""float8 v{n}0 = *(__global float8*)(tq{n}) * BC({sl2!r}f), w{n}0 = *(__global float8*)(tq{n} + 8) * BC({sl2!r}f);
          float8 v{n}1 = *(__global float8*)(tq{n} + 16) * BC({sl2!r}f), w{n}1 = *(__global float8*)(tq{n} + 24) * BC({sl2!r}f);
          float8 t{n}0 = VMAX(v{n}0 - xa, BC(-126.0f)), u{n}0 = VMAX(w{n}0 - xb, BC(-126.0f)), t{n}1 = VMAX(v{n}1 - xa, BC(-126.0f)), u{n}1 = VMAX(w{n}1 - xb, BC(-126.0f));
          float8 k{n}0 = VRINT(t{n}0), l{n}0 = VRINT(u{n}0), k{n}1 = VRINT(t{n}1), l{n}1 = VRINT(u{n}1);
          float8 g{n}0 = t{n}0 - k{n}0, h{n}0 = u{n}0 - l{n}0, g{n}1 = t{n}1 - k{n}1, h{n}1 = u{n}1 - l{n}1;"""
    P5 = lambda f_: f"BC(9.618129e-3f) * {f_} + BC(5.550411e-2f)"
    POLY = lambda f_: f"((({P5(f_)}) * {f_} + BC(2.402265e-1f)) * {f_} + BC(6.931472e-1f)) * {f_} + BC(1.0f)"
    P2P = f'''      if (full) {{
      #pragma clang loop unroll(disable)
      for (int i = 0; i < 3; i++) {{
        float8 xa = x01[i], xb = x23[i], sa = s01[i], sb = s23[i];
        __global float* tqc = T + i * 64; __global half* oq = O + i * 16;
        {SA("c")}
        #pragma clang loop unroll(disable)
        for (int q = 0; q < 12; q++) {{
          int odd = q & 1; __global float* tqn = tqc + 32 + odd * 128;       /* the next two tiles (past the end on the last: read, unused) */
          {SA("n")}
          float8 e0 = VSCAL2({POLY("gc0")}, VI(kc0)), f0 = VSCAL2({POLY("hc0")}, VI(lc0));
          float8 e1 = VSCAL2({POLY("gc1")}, VI(kc1)), f1 = VSCAL2({POLY("hc1")}, VI(lc1));
          sa += e0 + e1; sb += f0 + f1;
          *(__global half16*)(oq) = CVT16(e0, f0); *(__global half16*)(oq + 48) = CVT16(e1, f1);
          oq += 96; tqc = tqn;
          gc0 = gn0; hc0 = hn0; gc1 = gn1; hc1 = hn1; kc0 = kn0; lc0 = ln0; kc1 = kn1; lc1 = ln1;
        }}
        s01[i] = sa; s23[i] = sb;
      }}
      }} else {{
{P2A}      }}
'''
    P2B = P2P if p2t == 5 else (P2F if p2t == 3 else (P2N if p2t == 2 else P2A))
    return FULL_H + f"""
/* 2^t for t <= 0 (s - rowmax, scaled): one clamp, degree-4 Taylor (1.3e-5 relative at |f| = 0.5, against fp16's
   4.9e-4: P is stored as fp16). ⚠️ the general exp2_f8 (two clamps, degree 6) made this pass compute-bound at
   ~217 cycles a 4x4 tile. */
static inline float8 exp2_le0(float8 t) {{
  t = VMAX(t, BC(-126.0f));
  float8 k = VRINT(t); float8 f = t - k;
  float8 p = BC(9.618129e-3f);
  p = p * f + BC(5.550411e-2f); p = p * f + BC(2.402265e-1f); p = p * f + BC(6.931472e-1f); p = p * f + BC(1.0f);
  return VSCAL2(p, VI(k));
}}
static inline float8 kmask(int g, int st, int jt) {{
  float8 idx = (float8){{0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 2.0f, 3.0f}} + BC((float)(g * 96 + st * 16 + jt * 4));
  return idx < BC({float(treal)!r}f) ? BC(0.0f) : BC(-3.0e38f);
}}{KMASK2}
__kernel void softmax_stream(__global half* restrict out, __global float* restrict ct, __global float* restrict sums, __global int* restrict desc{", __global int* restrict st_" if stamps else ""}, const int core_id) {{
  {"int acc_[4] = {0, 0, 0, 0}; int t0_ = __builtin_aipu_mfctrl0(0xd1);" if stamps else ""}
  for (int u = core_id; u < {nrb * nh}; u += {nt}) {{
    int rb = u / {nh}, h = u % {nh};
    __global float* base = ct + h * {LG} + rb * {RB};
    float8 m01[3], m23[3];
    for (int i = 0; i < 3; i++) {{ m01[i] = BC(-3.0e38f); m23[i] = BC(-3.0e38f); }}
    {PASS1}
    float8 x01[3], x23[3];
    for (int i = 0; i < 3; i++) {{
      float a0 = m01[i][0], a1 = m01[i][4], a2 = m23[i][0], a3 = m23[i][4];
      for (int l = 1; l < 4; l++) {{ a0 = a0 > m01[i][l] ? a0 : m01[i][l]; a1 = a1 > m01[i][4 + l] ? a1 : m01[i][4 + l]; a2 = a2 > m23[i][l] ? a2 : m23[i][l]; a3 = a3 > m23[i][4 + l] ? a3 : m23[i][4 + l]; }}
      x01[i] = ROWS2(a0, a1) * BC({sl2!r}f); x23[i] = ROWS2(a2, a3) * BC({sl2!r}f);
    }}
    /* pass 2: exp, sums, the fp16 A slices */
    float8 s01[3], s23[3];
    for (int i = 0; i < 3; i++) {{ s01[i] = BC(0.0f); s23[i] = BC(0.0f); }}
    DMA_FILL(0, DESC(desc, 0), 0, (int)base);
    for (int gc = 0; gc < {ng // gb}; gc++) {{
      int s = gc & 1;
      if (gc + 1 < {ng // gb}) DMA_FILL(s ^ 1, DESC(desc, 0), (s ^ 1) * {gb * 4608}, (int)(base + (gc + 1) * {gb * nrb * RB}));
      {"{ int t_ = __builtin_aipu_mfctrl0(0xd1); acc_[2] += t_ - t0_; t0_ = t_; }" if stamps else ""}
      DMA_WAIT(s); DMA_WAIT(2 + s);
      {"{ int t_ = __builtin_aipu_mfctrl0(0xd1); acc_[3] += t_ - t0_; t0_ = t_; }" if stamps else ""}
      for (int gg = 0; gg < {gb}; gg++) {{
      int g = gc * {gb} + gg;
      __global float* T = LSF(s * {gb * 4608} + gg * 4608); __global half* O = LSH({2 * gb * 4608} + s * {gb * 2304} + gg * 2304);
      int full = {"(g * 96 + 95 <= rb * 12) && " if causal else ""}(g * 96 + 96 <= {treal});
{P2B}      }}
      DMA_DRAIN(2 + s, DESC(desc, 1), {2 * gb * 4608} + s * {gb * 2304}, (int)(out + h * {PA} + (gc * {gb} * {nrb} + rb) * {ks * 48}));
    }}
    {"{ int t_ = __builtin_aipu_mfctrl0(0xd1); acc_[2] += t_ - t0_; t0_ = t_; }" if stamps else ""}
    DMA_WAIT_ALL();
    {"{ int t_ = __builtin_aipu_mfctrl0(0xd1); acc_[3] += t_ - t0_; t0_ = t_; }" if stamps else ""}
    __global float* SM = LSF({2 * gb * 4608 + 2 * gb * 2304});
    for (int i = 0; i < 3; i++) {{
      int m0 = rb * 12 + 4 * i;
      SM[4 * i + 0] = m0 + 0 < {treal} ? 1.0f / (s01[i][0] + s01[i][1] + s01[i][2] + s01[i][3]) : 0.0f;
      SM[4 * i + 1] = m0 + 1 < {treal} ? 1.0f / (s01[i][4] + s01[i][5] + s01[i][6] + s01[i][7]) : 0.0f;
      SM[4 * i + 2] = m0 + 2 < {treal} ? 1.0f / (s23[i][0] + s23[i][1] + s23[i][2] + s23[i][3]) : 0.0f;
      SM[4 * i + 3] = m0 + 3 < {treal} ? 1.0f / (s23[i][4] + s23[i][5] + s23[i][6] + s23[i][7]) : 0.0f;
    }}
    DMA_DRAIN(2, DESC(desc, 2), {2 * gb * 4608 + 2 * gb * 2304}, (int)(sums + (({hb} + h) * {nrb} + rb) * 16));
    DMA_WAIT_ALL();
  }}
  {"for (int k = 0; k < 4; k++) st_[core_id * 16 + k] = acc_[k];" if stamps else ""}
}}"""


def softmax_stream_descs(nrb: int = 0, gb: int = 1, rowmax: int = 0) -> np.ndarray:
    """0: fill of gb tile blocks, 1: drain of gb P slices, 2: the sums. `rowmax` = ng: 3: the ng maxima, 4: one group's tiles."""
    if rowmax:
        return _desc_slots((gb * 4608, 4608, nrb * 4800, 4608), (gb * 2304, 2304, nrb * 2304, 2304), (64,), (rowmax * 192, 192, nrb * 4800, 192), (4608,))
    if gb == 1: return _desc_slots((4608,), (2304,), (64,))
    return _desc_slots((gb * 4608, 4608, nrb * 4608, 4608), (gb * 2304, 2304, nrb * 2304, 2304), (64,))


def softmax_stream_ref(logits, scale, treal, ks, nrb):
    """logits `[12 nrb, tp]` fp32 -> (the unnormalized P A layout (uint16), the row inverse sums `[12 nrb]`)."""
    x = logits.astype(np.float32).copy(); x[:, treal:] = -np.inf
    x = x * np.float32(scale); m = x.max(-1, keepdims=True); e = np.exp(x - m); e[:, treal:] = 0
    inv = (1.0 / e.sum(-1)).astype(np.float32); inv[treal:] = 0
    return a_layout_ref(e, ks, nrb), inv


# ---- S6b: gather heads with the softmax row sums ----
def gather_o_sum_src(rows: int, nh: int, dh: int, dhp: int, nsa: int, nrb: int, ks: int, nt: int = NT) -> str:
    """`gather_o_dma` with rows scaled by `softmax_stream`'s inverse sums (`[nh][nrb][16]`): unnormalized P V -> softmax V.
    args: out, ct, sums, desc (`gather_o_sum_descs`)."""
    assert dh % 4 == 0 and dhp % 96 == 0 and dhp >= dh and 4 * ks == 96 and nsa == 6 and (dhp // 96) * 4608 <= 13824
    # any head dim: dhp / 96 groups a head; its dh / 4 quads land at quads dh / 4 x h of the out A (K = nh dh)
    NG_, DQ = dhp // 96, dh // 4
    HC = NG_ * nrb * 1152
    def drains():
        body = []
        for h in range(nh):
            q0 = DQ * h; calls = []; f = 0; q = q0
            while q < q0 + DQ:
                sl, kk = q // 24, q % 24; cnt = min(24 - kk, q0 + DQ - q)
                calls.append("DMA_DRAIN(%d, DESC(desc, %d), %d, (int)(out + (%d * %d + rb) * %d + %d * 48));" % (f, cnt, 13824 + (q - q0) * 96, sl, nrb, ks * 48, kk))
                q += cnt; f += 1
            body.append("    %s (h == %d) { %s }" % ("if" if h == 0 else "else if", h, " ".join(calls)))
        return "\n".join(body)
    return FULL_H + f"""
__kernel void gather_o_sum(__global half* restrict out, __global float* restrict ct, __global float* restrict sums, __global int* restrict desc, const int core_id) {{
  for (int u = core_id; u < {nrb * nh}; u += {nt}) {{
    int rb = u / {nh}, h = u % {nh};
    __global float* T = LSF(0); __global half* O = LSH(13824); __global float* SM = LSF(31744);
    DMA_FILL(0, DESC(desc, 0), 0, (int)(ct + h * {HC} + rb * 1152));
    DMA_FILL(1, DESC(desc, 25), 31744, (int)(sums + (h * {nrb} + rb) * 16));
    DMA_WAIT_ALL();
    for (int i = 0; i < 3; i++) {{
      float8 i01 = ROWS2(SM[4 * i], SM[4 * i + 1]), i23 = ROWS2(SM[4 * i + 2], SM[4 * i + 3]);
      for (int q = 0; q < {DQ}; q++) {{
        __global float* tp = T + (((q / 4) * 3 + i) * 4 + q % 4) * 16;
        *(__global half16*)(O + (q * 3 + i) * 16) = CVT16(*(__global float8*)(tp) * i01, *(__global float8*)(tp + 8) * i23);
      }}
    }}
{drains()}
    DMA_WAIT_ALL();
  }}
}}"""


def gather_o_sum_descs(nrb: int, dhp: int = 288) -> np.ndarray:
    """0: the fill (dhp / 96 tile blocks), 1..24: a drain of n quads, 25: the unit's sums."""
    return _desc_slots(((dhp // 96) * 4608, 4608, nrb * 4608, 4608), *[(n * 96,) for n in range(1, 25)], (64,))


# ---- the VAE decoder's convs ----
# Activations: fp16 rows [pixel][channel], rows of Wp = 12 ceil(W / 12) pixels (pad columns zero), with a zero guard
# of Wp + 16 pixels before and after (the im2col's fills read past the image; its masks drop those values).
def conv_guard(wp: int) -> int: return wp + 16


def conv_c_src(W: int, Wp: int, H: int, cout: int, nrb: int, res: bool = False, nt: int = NT, rgb: bool = False) -> str:
    """Conv GEMM C tiles (ks 36 x ns 4) for rows p0 .. p0 + 12 nrb -> fp16 rows `out[p][c]` = C + bias (+ `rsd` if `res`),
    pads zero; per-channel sum / sum of squares at `st[band][task][c][2]` for the next GroupNorm. A unit = (row block,
    group). args: out, ct, bias, rsd, st, desc (`conv_c_descs`), p0v (int32 [first pixel, pixel-0 offset, band]).
    `rgb`: out = fp16 [pixel][4], clamp(v / 2 + 1/2, 0, 1)."""
    assert cout % 64 == 0 and Wp % 12 == 0
    NG = cout // 64
    return FULL_H + f"""
__kernel void conv_c(__global half* restrict out, __global float* restrict ct, __global float* restrict bias, __global half* restrict rsd, __global float* restrict st, __global int* restrict desc, __global int* restrict p0v, const int core_id) {{
  /* DOUBLE-BUFFERED (2026-09-28): the next unit's C tiles (+ residual) fills and the previous unit's drain are issued before this
     unit's work, one DMA_WAIT_ALL per unit. LSRAM: T 2 x 3072 (0), O 2 x 1536 (6144), RS 2 x 1536 (9216), every bias (12288),
     ACC (14336), p0 (18432) */
  __global float* B = LSF(12288); __global float* ACC = LSF(14336);
  __global int* PV = (__global int*)LSF(18432);
  DMA_FILL(0, DESC(desc, 3), 18432, (int)p0v); DMA_FILL(1, DESC(desc, 5), 12288, (int)bias); DMA_WAIT_ALL();
  int p0 = PV[0], band = PV[2]; out = out + PV[1]; rsd = rsd + PV[1];        /* PV: the band's first pixel, the rows' pixel-0 offset (elements) */
  for (int k = 0; k < {cout * 2}; k += 8) *(__global float8*)(ACC + k) = BC(0.0f);        /* the per-channel partials, zeroed per launch */
  if (core_id < {nrb * NG}) {{
    int g = core_id / {nrb}, pr = p0 + (core_id % {nrb}) * 12;
    DMA_FILL(0, DESC(desc, 0), 0, (int)(ct + (g * {nrb} + core_id % {nrb}) * 768));
    {"DMA_FILL(2, DESC(desc, 1), 9216, (int)(rsd + pr * %d + g * 64));" % cout if res else ""}
    DMA_WAIT_ALL();
  }}
  int kk = 0;
  for (int u = core_id; u < {nrb * NG}; u += {nt}, kk++) {{
    int g = u / {nrb}, rb = u % {nrb}, pr = p0 + rb * 12, b = kk & 1;
    if (u + {nt} < {nrb * NG}) {{
      int un = u + {nt}, gn = un / {nrb}, rbn = un % {nrb};
      DMA_FILL(0, DESC(desc, 0), (b ^ 1) * 3072, (int)(ct + (gn * {nrb} + rbn) * 768));
      {"DMA_FILL(2, DESC(desc, 1), 9216 + (b ^ 1) * 1536, (int)(rsd + (p0 + rbn * 12) * %d + gn * 64));" % cout if res else ""}
    }}
    if (kk > 0) {{ int up_ = u - {nt}; {"DMA_DRAIN(3, DESC(desc, 6), 6144 + (b ^ 1) * 1536, (int)(out + (p0 + (up_ %% %d) * 12) * 4));" % nrb if rgb else "DMA_DRAIN(3, DESC(desc, 1), 6144 + (b ^ 1) * 1536, (int)(out + (p0 + (up_ %% %d) * 12) * %d + (up_ / %d) * 64));" % (nrb, cout, nrb)} }}
    __global float* T = LSF(b * 3072); __global half* O = LSH(6144 + b * 1536); __global half* RS = LSH(9216 + b * 1536);
    __global float* BG = B + g * 64;
    for (int px = 0; px < 12; px++) {{
      int p = pr + px, x = p % {Wp}, y = p / {Wp};
      int ok = (x < {W}) && (y < {H});
      int i = px / 4, r = px % 4;
      for (int s = 0; s < 4; s++) {{
        __global float* tb = T + ((s * 3 + i) * 4) * 16 + r * 4;               /* tile jt at tb + jt * 16: row r's 4 channels */
        float8 lo = __builtin_shufflevector(*(__global float4*)(tb), *(__global float4*)(tb + 16), 0, 1, 2, 3, 4, 5, 6, 7) + *(__global float8*)(BG + s * 16);
        float8 hi = __builtin_shufflevector(*(__global float4*)(tb + 32), *(__global float4*)(tb + 48), 0, 1, 2, 3, 4, 5, 6, 7) + *(__global float8*)(BG + s * 16 + 8);
        {"half16 rh = *(__global half16*)(RS + px * 64 + s * 16); lo += __builtin_aipu_cvtue_tfp32_tfp16(__builtin_shufflevector(rh, rh, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15)); hi += __builtin_aipu_cvtuo_tfp32_tfp16(__builtin_shufflevector(rh, rh, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15));" if res else ""}
        if (!ok) {{ lo = BC(0.0f); hi = BC(0.0f); }}
        {"if (s == 0) { float8 l2 = VMIN(VMAX(lo * BC(0.5f) + BC(0.5f), BC(0.0f)), BC(1.0f)); half16 h = CVT16(l2, l2); *(__global half4*)(O + px * 4) = __builtin_shufflevector(h, h, 0, 1, 2, 3); }" if rgb else "half16 h = CVT16(lo, hi); *(__global half16*)(O + px * 64 + s * 16) = h;"}
        float8 ql = lo, qh = hi;                                                  /* the stats of the fp32 values (masked) */
        __global float* ac = ACC + (g * 64 + s * 16) * 2;                         /* [c][sum, sumsq] as [16 sums][16 sumsqs] a strip */
        *(__global float8*)(ac) += ql; *(__global float8*)(ac + 8) += qh; *(__global float8*)(ac + 16) += ql * ql; *(__global float8*)(ac + 24) += qh * qh;
      }}
    }}
    DMA_WAIT_ALL();
  }}
  if (kk > 0) {{ int up_ = core_id + (kk - 1) * {nt}, b = (kk - 1) & 1;                 /* the rows: absolute pixel pr */
    {"DMA_DRAIN(3, DESC(desc, 6), 6144 + b * 1536, (int)(out + (p0 + (up_ %% %d) * 12) * 4));" % nrb if rgb else "DMA_DRAIN(3, DESC(desc, 1), 6144 + b * 1536, (int)(out + (p0 + (up_ %% %d) * 12) * %d + (up_ / %d) * 64));" % (nrb, cout, nrb)} }}
  DMA_DRAIN(0, DESC(desc, 4), 14336, (int)(st + (band * {nt} + core_id) * {cout * 2}));
  DMA_WAIT_ALL();
}}"""


def conv_c_descs(cout: int, nrb: int) -> np.ndarray:
    """0: a unit's C tiles, 1: 12 rows x 128 B (drain / residual fill), 2: unused, 3: p0v, 4: the stats, 5: all biases,
    6: the rgb drain."""
    return _desc_slots((3072,), (12 * 128, 128, cout * 2, 128), (256,), (64,), (cout * 8,), (cout * 4,), (96,))


def conv_c_stats(st: np.ndarray, cout: int, count: int, groups: int = 32, eps: float = 1e-6):
    """`conv_c`'s partials `[nt][cout][2]` as stored ([strip][16 sums][16 sumsqs] per 16 channels) -> per group (mean, rstd)."""
    s = st.reshape(-1, cout // 16, 2, 16).sum(0)                          # [strip][sum / sumsq][16]
    sums, sq = s[:, 0].ravel().astype(np.float64), s[:, 1].ravel().astype(np.float64)
    gs, gq = sums.reshape(groups, -1).sum(1), sq.reshape(groups, -1).sum(1); n = count * (cout // groups)
    mean = gs / n; var = np.maximum(gq / n - mean * mean, 0.0)
    return mean, 1.0 / np.sqrt(var + eps)


EXP2_D4 = """
static inline float8 exp2_d4(float8 t) {
  t = VMAX(t, BC(-126.0f)); t = VMIN(t, BC(126.0f));
  float8 k = VRINT(t); float8 f = t - k;
  float8 p = BC(9.618129e-3f);
  p = p * f + BC(5.550411e-2f); p = p * f + BC(2.402265e-1f); p = p * f + BC(6.931472e-1f); p = p * f + BC(1.0f);
  return VSCAL2(p, VI(k));
}
"""


def conv_a_src(Ws: int, Wsp: int, Hs: int, cin: int, W: int, Wp: int, nrb: int, k1: bool = False, gn: bool = False, up: bool = False,
               s0: int = 0, nsl: int = 0, nt: int = NT, stamps: bool = False) -> str:
    """im2col of a conv input (ks 36: slices of 144 k) for output rows p0 .. p0 + 12 nrb: slices `s0 ..` of an `nsl`-slice A.
    Source: guarded fp16 rows Ws x Hs (pitch Wsp); output pitch Wp. 3 x 3: k = (cb x 9 + tap) x 32 + j, j holding channel
    `conv_perm32()[j]` (the unpack's order). A unit = (row block, 32-channel block cb): three rows' neighbourhoods
    (`up`: nearest-2x upsample) filled, activated once into fp32 (`gn`: affine from `gab` [cb][32 a | 32 b] + SiLU),
    padding zeroed, 9 taps -> 2 slices. `k1`: 1 x 1 conv (K = cin padded to 144). args: out, src, gab, desc
    (`conv_a_descs`), p0v (int32 [first output pixel, src's pixel-0 offset])."""
    assert Wp % 12 == 0 and cin % 32 == 0 and not (k1 and (gn or up))
    SL = nrb * 1728                                          # halves in a slice of the A layout (nrb x 36 kk x 3 x 16)
    if k1:
        nj = -(-cin // 144)
        return FULL_H + f"""
__kernel void conv_a1(__global half* restrict out, __global half* restrict src, __global float* restrict gab, __global int* restrict desc, __global int* restrict p0v, const int core_id) {{
  __global half* F = LSH(0); __global half* O = LSH(10240); __global int* PV = (__global int*)LSF(20480);
  DMA_FILL(0, DESC(desc, 3), 20480, (int)p0v); DMA_WAIT_ALL();
  int p0 = PV[0]; src = src + PV[1];                          /* PV: the band's first output pixel, the source's pixel-0 offset */
  for (int u = core_id; u < {nrb * nj}; u += {nt}) {{
    int j = u / {nrb}, rb = u % {nrb}, pr = p0 + rb * 12;
    DMA_FILL(0, DESC(desc, 1), 0, (int)(src + pr * {cin} + j * 144)); DMA_WAIT_ALL();
    for (int q = 0; q < 36; q++) {{
      int live = (j * 144 + 4 * q < {cin});
      for (int i = 0; i < 3; i++) {{
        __global half* f = F + (4 * i) * 144 + 4 * q;
        half8 h01 = __builtin_shufflevector(*(__global half4*)(f), *(__global half4*)(f + 144), 0, 1, 2, 3, 4, 5, 6, 7);
        half8 h23 = __builtin_shufflevector(*(__global half4*)(f + 288), *(__global half4*)(f + 432), 0, 1, 2, 3, 4, 5, 6, 7);
        half16 t = __builtin_shufflevector(h01, h23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        if (!live) t = CVT16(BC(0.0f), BC(0.0f));
        *(__global half16*)(O + (q * 3 + i) * 16) = t;
      }}
    }}
    DMA_DRAIN(0, DESC(desc, 2), 10240, (int)(out + ({s0} + j) * {SL} + rb * 1728)); DMA_WAIT_ALL();
  }}
}}"""
    NPX = 8 if up else 14
    F32 = 4096
    SILU = "v = v * a8 + b8; v = v * VRCP(BC(1.0f) + exp2_d4(BC(%rf) * v));" % -LOG2E if gn else ""
    slot = ("((X0 + px + dx) >> 1) - xl0" if up else "px + dx + 1")
    GNC = ("ve = ve * *(__global float8*)(GB + h * 16) + *(__global float8*)(GB + 32 + h * 16); vo = vo * *(__global float8*)(GB + h * 16 + 8) + *(__global float8*)(GB + 40 + h * 16); "
           "ve = ve * VRCP(BC(1.0f) + exp2_d4(BC(%rf) * ve)); vo = vo * VRCP(BC(1.0f) + exp2_d4(BC(%rf) * vo));" % (-LOG2E, -LOG2E)) if gn else ""
    if not gn:                                   # the double-buffered body; the single-buffered one below serves gn
        return _conv_a_db_src(Ws, Wsp, Hs, cin, W, Wp, nrb, up, s0, nt, stamps, NPX, slot, SL)
    return FULL_H + EXP2_D4 + f"""
__kernel void conv_a(__global half* restrict out, __global half* restrict src, __global float* restrict gab, __global int* restrict desc, __global int* restrict p0v{", __global int* restrict st_" if stamps else ""}, const int core_id) {{
  {"int acc_[4] = {0, 0, 0, 0}; int t0_ = __builtin_aipu_mfctrl0(0xd1); int ts_ = t0_;" if stamps else ""}
  __global half* F = LSH(0); __global float* A = LSF({F32}); __global float* GB = LSF(9728); __global half* O = LSH(10240); __global int* PV = (__global int*)LSF(20480);
  DMA_FILL(0, DESC(desc, 3), 20480, (int)p0v); DMA_WAIT_ALL();
  int p0 = PV[0]; src = src + PV[1];                          /* PV: the band's first output pixel, the source's pixel-0 offset */
  for (int u = core_id; u < {nrb * (cin // 32)}; u += {nt}) {{
    int cb = u / {nrb}, rb = u % {nrb}, pr = p0 + rb * 12, Y = pr / {Wp}, X0 = pr % {Wp};
    int xl0 = {"(X0 - 1) >> 1" if up else "X0 - 1"};
    for (int d = 0; d < 3; d++) {{
      int row = {"(Y + d - 1) >> 1" if up else "Y + d - 1"};
      if (d == 0) DMA_FILL(0, DESC(desc, 0), {NPX * 64} * 0, (int)(src + (row * {Wsp} + xl0) * {cin} + cb * 32));
      else if (d == 1) DMA_FILL(1, DESC(desc, 0), {NPX * 64} * 1, (int)(src + (row * {Wsp} + xl0) * {cin} + cb * 32));
      else DMA_FILL(2, DESC(desc, 0), {NPX * 64} * 2, (int)(src + (row * {Wsp} + xl0) * {cin} + cb * 32));
    }}
    {"DMA_FILL(3, DESC(desc, 4), 9728, (int)(gab + cb * 64));" if gn else ""}
    DMA_WAIT_ALL();
    {"{ int t_ = __builtin_aipu_mfctrl0(0xd1); acc_[0] += t_ - t0_; t0_ = t_; }" if stamps else ""}
    /* activate the neighbourhoods once: fp16 -> fp32 by the even / odd unpack (positions [h][even 8 | odd 8]), the GroupNorm affine +
       SiLU, the padding zeroed */
    for (int d = 0; d < 3; d++) {{
      int row = {"(Y + d - 1) >> 1" if up else "Y + d - 1"}, rok = (row >= 0) && (row < {Hs});
      for (int p = 0; p < {NPX}; p++) {{
        int xs = xl0 + p; float okf = (rok && (xs >= 0) && (xs < {Ws})) ? 1.0f : 0.0f;
        __global half* fp = F + (d * {NPX} + p) * 32; __global float* ap = A + (d * {NPX} + p) * 32;
        for (int h = 0; h < 2; h++) {{
          half16 hv = *(__global half16*)(fp + h * 16);
          float8 ve = __builtin_aipu_cvtue_tfp32_tfp16(hv), vo = __builtin_aipu_cvtuo_tfp32_tfp16(hv);
          {GNC}
          *(__global float8*)(ap + h * 16) = ve * BC(okf); *(__global float8*)(ap + h * 16 + 8) = vo * BC(okf);
        }}
      }}
    }}
    {"{ int t_ = __builtin_aipu_mfctrl0(0xd1); acc_[1] += t_ - t0_; t0_ = t_; }" if stamps else ""}
    /* the 9 taps' tiles: quad kq = tap x 8 + q (positions 4q .. 4q + 3) -> slice kq / 36, kk kq % 36; a tile = CVT16 of two float8s,
       each two pixels' float4s */
    for (int tap = 0; tap < 9; tap++) {{
      int dy = tap / 3 - 1, dx = tap % 3 - 1;
      for (int i = 0; i < 3; i++) {{
        int px0 = 4 * i, px1 = px0 + 1, px2 = px0 + 2, px3 = px0 + 3;
        __global float* ar = A + (dy + 1) * {NPX * 32};
        __global float* r0 = ar + ({slot.replace("px", "px0")}) * 32; __global float* r1 = ar + ({slot.replace("px", "px1")}) * 32;
        __global float* r2 = ar + ({slot.replace("px", "px2")}) * 32; __global float* r3 = ar + ({slot.replace("px", "px3")}) * 32;
        for (int q = 0; q < 8; q++) {{
          float8 lo = __builtin_shufflevector(*(__global float4*)(r0 + 4 * q), *(__global float4*)(r1 + 4 * q), 0, 1, 2, 3, 4, 5, 6, 7);
          float8 hi = __builtin_shufflevector(*(__global float4*)(r2 + 4 * q), *(__global float4*)(r3 + 4 * q), 0, 1, 2, 3, 4, 5, 6, 7);
          int kq = tap * 8 + q;
          *(__global half16*)(O + (kq / 36) * 1728 + ((kq % 36) * 3 + i) * 16) = CVT16(lo, hi);
        }}
      }}
    }}
    {"{ int t_ = __builtin_aipu_mfctrl0(0xd1); acc_[2] += t_ - t0_; t0_ = t_; }" if stamps else ""}
    DMA_DRAIN(0, DESC(desc, 2), 10240, (int)(out + ({s0} + cb * 2) * {SL} + rb * 1728));
    DMA_DRAIN(1, DESC(desc, 2), 10240 + 3456, (int)(out + ({s0} + cb * 2 + 1) * {SL} + rb * 1728));
    DMA_WAIT_ALL();
    {"{ int t_ = __builtin_aipu_mfctrl0(0xd1); acc_[3] += t_ - t0_; t0_ = t_; }" if stamps else ""}
  }}
  {"for (int k = 0; k < 4; k++) st_[core_id * 16 + k] = acc_[k]; st_[core_id * 16 + 4] = ts_; st_[core_id * 16 + 5] = __builtin_aipu_mfctrl0(0xd1);" if stamps else ""}
}}"""


def conv_a_perm(activated: bool, gn: bool = False) -> np.ndarray:
    """Channel held by K position j of a 32-channel block: `conv_perm32()`, or pi o pi for a gn_silu-activated copy."""
    P = conv_perm32()
    return P[P] if activated else P


def _conv_a_db_src(Ws, Wsp, Hs, cin, W, Wp, nrb, up, s0, nt, stamps, NPX, slot, SL) -> str:
    """conv_a without GroupNorm, double-buffered: the next unit's fills (flags 0-2) and the previous unit's drains (flag 3)
    are issued before this unit's work; one DMA_WAIT_ALL per iteration. Stamps: issue, unpack, tiles, wait."""
    FB, OB = NPX * 3 * 64, 6912
    ST = (lambda k: "{ int t_ = __builtin_aipu_mfctrl0(0xd1); acc_[%d] += t_ - t0_; t0_ = t_; }" % k) if stamps else (lambda k: "")
    def fill(u, buf):
        return f"""{{ int cb_ = ({u}) / {nrb}, pr_ = p0 + (({u}) % {nrb}) * 12, Y_ = pr_ / {Wp}, X_ = pr_ % {Wp}, xl_ = {"(X_ - 1) >> 1" if up else "X_ - 1"};
      for (int d = 0; d < 3; d++) {{
        int row_ = {"(Y_ + d - 1) >> 1" if up else "Y_ + d - 1"};
        __global half* s_ = src + (row_ * {Wsp} + xl_) * {cin} + cb_ * 32;
        if (d == 0) DMA_FILL(0, DESC(desc, 0), ({buf}) * {FB} + {NPX * 64} * 0, (int)s_);
        else if (d == 1) DMA_FILL(1, DESC(desc, 0), ({buf}) * {FB} + {NPX * 64} * 1, (int)s_);
        else DMA_FILL(2, DESC(desc, 0), ({buf}) * {FB} + {NPX * 64} * 2, (int)s_);
      }} }}"""
    def drain(u, buf):
        return f"""{{ int cb_ = ({u}) / {nrb}, rb_ = ({u}) % {nrb};
      DMA_DRAIN(3, DESC(desc, 2), 10752 + ({buf}) * {OB}, (int)(out + ({s0} + cb_ * 2) * {SL} + rb_ * 1728));
      DMA_DRAIN(3, DESC(desc, 2), 10752 + ({buf}) * {OB} + 3456, (int)(out + ({s0} + cb_ * 2 + 1) * {SL} + rb_ * 1728)); }}"""
    N = nrb * (cin // 32)
    return FULL_H + f"""
__kernel void conv_a(__global half* restrict out, __global half* restrict src, __global float* restrict gab, __global int* restrict desc, __global int* restrict p0v{", __global int* restrict st_" if stamps else ""}, const int core_id) {{
  {"int acc_[4] = {0, 0, 0, 0}; int t0_ = __builtin_aipu_mfctrl0(0xd1); int ts_ = t0_;" if stamps else ""}
  __global float* A = LSF(5376); __global int* PV = (__global int*)LSF(24576);
  DMA_FILL(0, DESC(desc, 3), 24576, (int)p0v); DMA_WAIT_ALL();
  int p0 = PV[0]; src = src + PV[1];                          /* PV: the band's first output pixel, the source's pixel-0 offset */
  if (core_id < {N}) {{ {fill("core_id", "0")} DMA_WAIT_ALL(); }}
  int k = 0;
  for (int u = core_id; u < {N}; u += {nt}, k++) {{
    int b = k & 1;
    int cb = u / {nrb}, rb = u % {nrb}, pr = p0 + rb * 12, Y = pr / {Wp}, X0 = pr % {Wp};
    int xl0 = {"(X0 - 1) >> 1" if up else "X0 - 1"};
    if (u + {nt} < {N}) {fill(f"u + {nt}", "b ^ 1")}
    if (k > 0) {drain(f"u - {nt}", "b ^ 1")}
    {ST(0)}
    __global half* F = LSH(b * {FB}); __global half* O = LSH(10752 + b * {OB});
    for (int d = 0; d < 3; d++) {{
      int row = {"(Y + d - 1) >> 1" if up else "Y + d - 1"}, rok = (row >= 0) && (row < {Hs});
      for (int p = 0; p < {NPX}; p++) {{
        int xs = xl0 + p; float okf = (rok && (xs >= 0) && (xs < {Ws})) ? 1.0f : 0.0f;
        __global half* fp = F + (d * {NPX} + p) * 32; __global float* ap = A + (d * {NPX} + p) * 32;
        for (int h = 0; h < 2; h++) {{
          half16 hv = *(__global half16*)(fp + h * 16);
          float8 ve = __builtin_aipu_cvtue_tfp32_tfp16(hv), vo = __builtin_aipu_cvtuo_tfp32_tfp16(hv);
          *(__global float8*)(ap + h * 16) = ve * BC(okf); *(__global float8*)(ap + h * 16 + 8) = vo * BC(okf);
        }}
      }}
    }}
    {ST(1)}
    for (int tap = 0; tap < 9; tap++) {{
      int dy = tap / 3 - 1, dx = tap % 3 - 1;
      for (int i = 0; i < 3; i++) {{
        int px0 = 4 * i, px1 = px0 + 1, px2 = px0 + 2, px3 = px0 + 3;
        __global float* ar = A + (dy + 1) * {NPX * 32};
        __global float* r0 = ar + ({slot.replace("px", "px0")}) * 32; __global float* r1 = ar + ({slot.replace("px", "px1")}) * 32;
        __global float* r2 = ar + ({slot.replace("px", "px2")}) * 32; __global float* r3 = ar + ({slot.replace("px", "px3")}) * 32;
        for (int q = 0; q < 8; q++) {{
          float8 lo = __builtin_shufflevector(*(__global float4*)(r0 + 4 * q), *(__global float4*)(r1 + 4 * q), 0, 1, 2, 3, 4, 5, 6, 7);
          float8 hi = __builtin_shufflevector(*(__global float4*)(r2 + 4 * q), *(__global float4*)(r3 + 4 * q), 0, 1, 2, 3, 4, 5, 6, 7);
          int kq = tap * 8 + q;
          *(__global half16*)(O + (kq / 36) * 1728 + ((kq % 36) * 3 + i) * 16) = CVT16(lo, hi);
        }}
      }}
    }}
    {ST(2)}
    DMA_WAIT_ALL();
    {ST(3)}
  }}
  if (k > 0) {{ int u = core_id + (k - 1) * {nt}; int b = (k - 1) & 1; {drain("u", "b")} DMA_WAIT_ALL(); }}
  {"for (int k_ = 0; k_ < 4; k_++) st_[core_id * 16 + k_] = acc_[k_]; st_[core_id * 16 + 4] = ts_; st_[core_id * 16 + 5] = __builtin_aipu_mfctrl0(0xd1);" if stamps else ""}
}}"""


def attn_qkv_src(NRB: int, NH: int, SL: int, nt: int = NT) -> str:
    """VAE attention: q | k | v C tiles (one gemm_gs, 18 groups, each part padded to 576 columns) + bias -> fp16 operands:
    q -> A in two query halves aq[h][sl][rb'][kk][i][16]; k -> B bk[G][sl][s][kk][a][16] (key = 96 G + 16 s + 4 a + y);
    v -> v^T B bv[G][sl][s][kk][x][16] (4x4 blocks transposed). A unit = (group, row block); SL = key slices.
    args: aq, ct, bias (fp32 [1728]), bk, bv, desc (`attn_qkv_descs`)."""
    return FULL_H + f"""
__kernel void attn_qkv(__global half* restrict aq, __global float* restrict ct, __global float* restrict bias, __global half* restrict bk, __global half* restrict bv, __global int* restrict desc, const int core_id) {{
  __global float* T = LSF(0); __global float* BI = LSF(4608); __global half* O = LSH(11520);
  DMA_FILL(0, DESC(desc, 1), 4608, (int)bias); DMA_WAIT_ALL();
  for (int u = core_id; u < {18 * NRB}; u += {nt}) {{
    int g = u / {NRB}, rb = u % {NRB}, part = g / 6, gl = g % 6;
    DMA_FILL(0, DESC(desc, 0), 0, (int)(ct + (g * {NRB} + rb) * 1152)); DMA_WAIT_ALL();
    __global float* bg = BI + g * 96;
    for (int s = 0; s < 6; s++) for (int i = 0; i < 3; i++) for (int x = 0; x < 4; x++) {{
      __global float* blk = T + ((s * 3 + i) * 4 + x) * 16;
      float4 bq = *(__global float4*)(bg + s * 16 + x * 4);
      float8 bb = __builtin_shufflevector(bq, bq, 0, 1, 2, 3, 4, 5, 6, 7);
      float8 lo = *(__global float8*)(blk) + bb, hi = *(__global float8*)(blk + 8) + bb;      /* rows y 0-1 | 2-3, columns z */
      if (part == 0) *(__global half16*)(O + ((s * 4 + x) * 3 + i) * 16) = CVT16(lo, hi);
      else if (part == 1) *(__global half16*)(O + (i * 24 + s * 4 + x) * 16) = CVT16(lo, hi);
      else *(__global half16*)(O + ((i * 6 + s) * 4 + x) * 16) = CVT16(__builtin_shufflevector(lo, hi, 0, 4, 8, 12, 1, 5, 9, 13), __builtin_shufflevector(lo, hi, 2, 6, 10, 14, 3, 7, 11, 15));
    }}
    if (part == 0) {{
      DMA_DRAIN(0, DESC(desc, 2), 11520, (int)(aq + (rb / {NH}) * {6 * NH * 1152} + (gl * {NH} + rb % {NH}) * 1152));
    }} else {{
      for (int i = 0; i < 3; i++) {{
        int key = rb * 12 + i * 4;
        __global half* d = (part == 1) ? bk + ((((key / 96) * 6 + gl) * 6 + (key % 96) / 16) * 96 + (key % 16) / 4) * 16
                                       : bv + (((gl * {SL} + key / 96) * 144) * 4 + ((key % 96) / 4) * 4) * 16;
        if (i == 0) DMA_DRAIN(0, DESC(desc, part == 1 ? 3 : 4), 11520, (int)d);
        else if (i == 1) DMA_DRAIN(1, DESC(desc, part == 1 ? 3 : 4), 11520 + 768, (int)d);
        else DMA_DRAIN(2, DESC(desc, part == 1 ? 3 : 4), 11520 + 1536, (int)d);
      }}
    }}
    DMA_WAIT_ALL();
  }}
}}"""


def attn_qkv_descs() -> np.ndarray:
    """0: a unit's C tiles, 1: the biases, 2: a q A chunk, 3: the k drain (24 x 32 B at 128 B), 4: the v^T drain."""
    return _desc_slots((4608,), (1728 * 4,), (2304,), (768, 32, 128, 32), (768, 128, 3072, 128))


def attn_o_src(NH: int, n: int, W: int, Wp: int, C: int, nt: int = NT) -> str:
    """PV's C tiles of one query half x each row's 1 / sum -> fp16 activation rows (pixel = (q / W) Wp + q % W; channels
    >= C dropped). A unit = (group, row block). args: out, ot, sums, desc (`attn_o_descs`), p0v (int32 [first query,
    pixel-0 offset])."""
    return FULL_H + f"""
__kernel void attn_o(__global half* restrict out, __global float* restrict ot, __global float* restrict sums, __global int* restrict desc, __global int* restrict p0v, const int core_id) {{
  __global float* T = LSF(0); __global float* S = LSF(4608); __global half* O = LSH(4672); __global int* PV = (__global int*)LSF(7040);
  DMA_FILL(0, DESC(desc, 4), 7040, (int)p0v); DMA_WAIT_ALL();
  int q0 = PV[0]; out = out + PV[1];
  for (int u = core_id; u < {6 * NH}; u += {nt}) {{
    int g = u / {NH}, rb = u % {NH};
    DMA_FILL(0, DESC(desc, 0), 0, (int)(ot + (g * {NH} + rb) * 1152)); DMA_FILL(1, DESC(desc, 1), 4608, (int)(sums + rb * 16)); DMA_WAIT_ALL();
    for (int i = 0; i < 3; i++) for (int y = 0; y < 4; y++) {{
      float iv = S[i * 4 + y];
      for (int s = 0; s < 6; s++) {{
        __global float* b0 = T + ((s * 3 + i) * 4) * 16 + y * 4;
        float8 v01 = __builtin_shufflevector(*(__global float4*)(b0), *(__global float4*)(b0 + 16), 0, 1, 2, 3, 4, 5, 6, 7) * BC(iv);
        float8 v23 = __builtin_shufflevector(*(__global float4*)(b0 + 32), *(__global float4*)(b0 + 48), 0, 1, 2, 3, 4, 5, 6, 7) * BC(iv);
        *(__global half16*)(O + (i * 4 + y) * 96 + s * 16) = CVT16(v01, v23);
      }}
    }}
    for (int r = 0; r < 12; r++) {{
      int q = q0 + rb * 12 + r;
      if (q < {n}) {{
        __global half* d = out + ((q / {W}) * {Wp} + q % {W}) * {C} + g * 96;
        if (g * 96 + 96 <= {C}) DMA_DRAIN(0, DESC(desc, 2), 4672 + r * 192, (int)d);
        else DMA_DRAIN(0, DESC(desc, 3), 4672 + r * 192, (int)d);
      }}
    }}
    DMA_WAIT_ALL();
  }}
}}"""


def attn_o_descs(C: int) -> np.ndarray:
    """0: a unit's C tiles, 1: the row block's sums, 2: a row's 96 channels, 3: the last group's real channels, 4: p0v."""
    return _desc_slots((4608,), (64,), (192,), (max(C % 96, 1) * 2 if C % 96 else 192,), (64,))


def ctile_rows_src(nrb: int, n: int, c: int, row0: int, scale: float, nt: int = NT) -> str:
    """gemm_gs C tiles of N = c -> fp32 rows `out[row0 + r][col]` = C * scale + bias[col], r < n. A unit = (group, row
    block). args: out, ct, bias (fp32 [c]), desc (`ctile_rows_descs`)."""
    NG = c // 96; last = n % 12
    return FULL_H + f"""
__kernel void ctile_rows(__global float* restrict out, __global float* restrict ct, __global float* restrict bias, __global int* restrict desc, const int core_id) {{
  __global float* T = LSF(0); __global float* B = LSF(4608); __global float* O = LSF(5120);
  for (int u = core_id; u < {NG * nrb}; u += {nt}) {{
    int g = u / {nrb}, rb = u % {nrb};
    if (rb * 12 >= {n}) continue;
    DMA_FILL(0, DESC(desc, 0), 0, (int)(ct + (g * {nrb} + rb) * 1152)); DMA_FILL(1, DESC(desc, 1), 4608, (int)(bias + g * 96)); DMA_WAIT_ALL();
    for (int s = 0; s < 6; s++) for (int i = 0; i < 3; i++) for (int x = 0; x < 4; x++) {{
      float4 bq = *(__global float4*)(B + s * 16 + x * 4);
      __global float* blk = T + ((s * 3 + i) * 4 + x) * 16;
      for (int y = 0; y < 4; y++) *(__global float4*)(O + (i * 4 + y) * 96 + s * 16 + x * 4) = *(__global float4*)(blk + y * 4) * {float(scale)!r}f + bq;
    }}
    __global float* d = out + ({row0} + rb * 12) * {c} + g * 96;
    {"if (rb * 12 + 12 > %d) DMA_DRAIN(0, DESC(desc, 3), 5120, (int)d); else " % n if last else ""}DMA_DRAIN(0, DESC(desc, 2), 5120, (int)d);
    DMA_WAIT_ALL();
  }}
}}"""


def ctile_rows_descs(n: int, c: int) -> np.ndarray:
    """0: a unit's tiles, 1: its 96 biases, 2: 12 rows at the rows' pitch, 3: the last block's n % 12 rows."""
    last = n % 12 or 12
    return _desc_slots((4608,), (384,), (12 * 384, 384, 4 * c, 384), (last * 384, 384, 4 * c, 384))


EXP2_D7 = """
static inline float8 exp2_d7(float8 t) {
  t = VMAX(t, BC(-126.0f)); t = VMIN(t, BC(126.0f));
  float8 k = VRINT(t); float8 f = t - k;
  float8 p = BC(1.525273e-05f);
  p = p * f + BC(1.540353e-04f); p = p * f + BC(1.333356e-03f); p = p * f + BC(9.618129e-03f);
  p = p * f + BC(5.550411e-02f); p = p * f + BC(2.402265e-01f); p = p * f + BC(6.931472e-01f); p = p * f + BC(1.0f);
  return VSCAL2(p, VI(k));
}
static inline float8 tanh_d7(float8 x) {                      /* (1 - e) / (1 + e), e = exp(-2x): the reciprocal Newton-refined */
  float8 e = exp2_d7(x * BC(-2.885390081777927f)), d = BC(1.0f) + e, r = VRCP(d);
  r = r * (BC(2.0f) - d * r); return (BC(1.0f) - e) * r;
}
"""


def mod_tables_src(sp: int, nt: int = NT) -> str:
    """A DiT layer's four per-step tables from its modulation. `cm`: C tiles of `ada_codes @ [adaln_hi ; adaln_lo]` (2 sp
    rows: hi then lo; N = 192 groups), `kc` = `mod_tables_consts`, `stepv[0]` the step. m = (C_hi + C_lo) * scale + b;
    ws1 = dup_quads(an1 (1 + m_0)), k1 = resid_dma_consts(an2, tanh m_1, sc_o), ws2 = dup_quads(fn1 (1 + m_2)),
    k2 = (fn2, tanh m_3, sc_w2). A unit = a group. args: ws1, cm, kc, k1, ws2, k2, stepv, desc."""
    return FULL_H + EXP2_D7 + f"""
__kernel void mod_tables(__global float* restrict ws1, __global float* restrict cm, __global float* restrict kc, __global float* restrict k1, __global float* restrict ws2, __global float* restrict k2, __global int* restrict stepv, __global int* restrict desc, const int core_id) {{
  __global float* T = LSF(0); __global float* K = LSF(9216); __global float* O = LSF(10752); __global int* PV = (__global int*)LSF(13056);
  DMA_FILL(0, DESC(desc, 4), 13056, (int)stepv); DMA_WAIT_ALL();
  int st = PV[0], rb = st / 12, i = (st % 12) / 4, y = st % 4;
  for (int g = core_id; g < 192; g += {nt}) {{
    DMA_FILL(0, DESC(desc, 0), 0, (int)(cm + (g * {2 * sp // 12} + rb) * 1152)); DMA_FILL(1, DESC(desc, 0), 4608, (int)(cm + (g * {2 * sp // 12} + rb + {sp // 12}) * 1152));
    DMA_FILL(2, DESC(desc, 1), 9216, (int)(kc + g * 384)); DMA_WAIT_ALL();
    int seg = g / 48, gl = g % 48;
    for (int s = 0; s < 6; s++) for (int x = 0; x < 4; x += 2) {{
      int q = s * 4 + x;                                              /* the group's quads q, q + 1: columns 4q .. 4q + 7 */
      __global float* h0 = T + ((s * 3 + i) * 4 + x) * 16 + y * 4;
      float8 c = __builtin_shufflevector(*(__global float4*)(h0), *(__global float4*)(h0 + 16), 0, 1, 2, 3, 4, 5, 6, 7)
               + __builtin_shufflevector(*(__global float4*)(h0 + 1152), *(__global float4*)(h0 + 1168), 0, 1, 2, 3, 4, 5, 6, 7);
      float8 m = c * *(__global float8*)(K + 4 * q) + *(__global float8*)(K + 96 + 4 * q);
      float8 v1 = *(__global float8*)(K + 192 + 4 * q), v2 = *(__global float8*)(K + 288 + 4 * q);
      if (seg == 0 || seg == 2) {{
        float8 v = (BC(1.0f) + m) * v1;
        *(__global float8*)(O + q * 8) = __builtin_shufflevector(v, v, 0, 1, 2, 3, 0, 1, 2, 3);
        *(__global float8*)(O + q * 8 + 8) = __builtin_shufflevector(v, v, 4, 5, 6, 7, 4, 5, 6, 7);
      }} else {{
        float8 t = tanh_d7(m);
        *(__global float8*)(O + q * 24) = __builtin_shufflevector(v1, v1, 0, 1, 2, 3, 0, 1, 2, 3);
        *(__global float8*)(O + q * 24 + 8) = __builtin_shufflevector(t, t, 0, 1, 2, 3, 0, 1, 2, 3);
        *(__global float8*)(O + q * 24 + 16) = __builtin_shufflevector(v2, v2, 0, 1, 2, 3, 0, 1, 2, 3);
        *(__global float8*)(O + q * 24 + 24) = __builtin_shufflevector(v1, v1, 4, 5, 6, 7, 4, 5, 6, 7);
        *(__global float8*)(O + q * 24 + 32) = __builtin_shufflevector(t, t, 4, 5, 6, 7, 4, 5, 6, 7);
        *(__global float8*)(O + q * 24 + 40) = __builtin_shufflevector(v2, v2, 4, 5, 6, 7, 4, 5, 6, 7);
      }}
    }}
    if (seg == 0) DMA_DRAIN(0, DESC(desc, 2), 10752, (int)(ws1 + gl * 192));
    else if (seg == 1) DMA_DRAIN(0, DESC(desc, 3), 10752, (int)(k1 + gl * 576));
    else if (seg == 2) DMA_DRAIN(0, DESC(desc, 2), 10752, (int)(ws2 + gl * 192));
    else DMA_DRAIN(0, DESC(desc, 3), 10752, (int)(k2 + gl * 576));
    DMA_WAIT_ALL();
  }}
}}"""


def mod_tables_descs() -> np.ndarray:
    """0: a row block's tiles, 1: a group's constants, 2: a dup_quads drain, 3: a resid_dma_consts drain, 4: stepv."""
    return _desc_slots((4608,), (1536,), (768,), (2304,), (64,))


def mod_tables_consts(ada_scale, ada_b, an1, an2, fn1, fn2, sc_o, sc_w2) -> np.ndarray:
    """`mod_tables`' kc: [192 groups][96 ada_scale | 96 ada_b | 96 v1 | 96 v2] (fp32 [c] each; sc_o / sc_w2 the per-column scales)."""
    c = an1.shape[0]; z = np.zeros(c, np.float32)
    v1 = np.concatenate([an1, an2, fn1, fn2]).astype(np.float32); v2 = np.concatenate([z, sc_o, z, sc_w2]).astype(np.float32)
    return np.ascontiguousarray(np.stack([a.astype(np.float32).reshape(-1, 96) for a in (ada_scale, ada_b, v1, v2)], 1)).ravel()


def lat_pack_src(nrb: int, nt: int = NT) -> str:
    """DiT latents x (fp32 [12 nrb][128]) -> the input projection's A (K = [x_hi | x_hi | x_lo]); x_hi by Veltkamp's
    split (contraction off), x_lo = x - hi. args: out, x, desc (`lat_pack_descs`)."""
    return "#pragma OPENCL FP_CONTRACT OFF\n" + FULL_H + f"""
__kernel void lat_pack(__global half* restrict out, __global float* restrict x, __global int* restrict desc, const int core_id) {{
  __global float* X = LSF(0); __global float* HL = LSF(6144); __global half* O = LSH(18432);
  for (int rb = core_id; rb < {nrb}; rb += {nt}) {{
    DMA_FILL(0, DESC(desc, 0), 0, (int)(x + rb * 1536)); DMA_WAIT_ALL();
    for (int k = 0; k < 1536; k += 8) {{                        /* HL = [12 rows][hi 128 | lo 128] */
      float8 v = *(__global float8*)(X + k), c = v * BC(8193.0f), h = c - (c - v);
      int r = k / 128, cc = k % 128;
      *(__global float8*)(HL + r * 256 + cc) = h; *(__global float8*)(HL + r * 256 + 128 + cc) = v - h;
    }}
    for (int sl = 0; sl < 4; sl++) for (int kk = 0; kk < 24; kk++) {{
      int col = sl * 96 + kk * 4, part = col / 128, cc = col % 128, off = (part == 2 ? 128 : 0) + cc;
      for (int i = 0; i < 3; i++) {{
        __global float* r0 = HL + (i * 4) * 256 + off;
        float8 lo = __builtin_shufflevector(*(__global float4*)(r0), *(__global float4*)(r0 + 256), 0, 1, 2, 3, 4, 5, 6, 7);
        float8 hi = __builtin_shufflevector(*(__global float4*)(r0 + 512), *(__global float4*)(r0 + 768), 0, 1, 2, 3, 4, 5, 6, 7);
        *(__global half16*)(O + ((sl * 24 + kk) * 3 + i) * 16) = CVT16(lo, hi);
      }}
    }}
    for (int sl = 0; sl < 4; sl++) DMA_DRAIN(sl, DESC(desc, 1), 18432 + sl * 2304, (int)(out + (sl * {nrb} + rb) * 1152));
    DMA_WAIT_ALL();
  }}
}}"""


def lat_pack_descs() -> np.ndarray:
    """0: a row block of latents (12 x 512 B), 1: an A chunk (2304 B)."""
    return _desc_slots((6144,), (2304,))


def cfg_euler_src(n: int, rc0: int, nt: int = NT) -> str:
    """Sampler step: x[r] += dt (g vc[r] + (1 - g) vu[r]); vc / vu from `ctile_rows` (cols 0-127 hi + bias, 128-255 lo;
    cond rows from rc0); prm = [g, dt]. A unit = 8 rows. args: x (in place), vc, vu, prm, desc (`cfg_euler_descs`)."""
    return FULL_H + f"""
__kernel void cfg_euler(__global float* restrict x, __global float* restrict vc, __global float* restrict vu, __global float* restrict prm, __global int* restrict desc, const int core_id) {{
  __global float* A = LSF(0); __global float* B = LSF(8192); __global float* X = LSF(16384); __global float* P = LSF(20480);
  DMA_FILL(0, DESC(desc, 3), 20480, (int)prm); DMA_WAIT_ALL();
  float8 g = BC(P[0]), g1 = BC(1.0f - P[0]), dt = BC(P[1]);
  for (int u = core_id; u < {n // 8}; u += {nt}) {{
    DMA_FILL(0, DESC(desc, 0), 0, (int)(vc + ({rc0} + u * 8) * 288)); DMA_FILL(1, DESC(desc, 0), 8192, (int)(vu + u * 8 * 288));
    DMA_FILL(2, DESC(desc, 1), 16384, (int)(x + u * 1024)); DMA_WAIT_ALL();
    for (int r = 0; r < 8; r++) for (int c = 0; c < 128; c += 8) {{
      float8 a = *(__global float8*)(A + r * 256 + c) + *(__global float8*)(A + r * 256 + 128 + c);
      float8 b = *(__global float8*)(B + r * 256 + c) + *(__global float8*)(B + r * 256 + 128 + c);
      *(__global float8*)(X + r * 128 + c) = *(__global float8*)(X + r * 128 + c) + (g * a + g1 * b) * dt;
    }}
    DMA_DRAIN(0, DESC(desc, 1), 16384, (int)(x + u * 1024)); DMA_WAIT_ALL();
  }}
}}"""


def cfg_euler_descs() -> np.ndarray:
    """0: 8 velocity rows' first 256 columns (1152-B pitch), 1: 8 latent rows, 3: prm."""
    return _desc_slots((8192, 1024, 1152, 1024), (4096,), (64,), (64,))


def gn_prep_src(C: int, nparts: int, count: int, eps: float = 1e-6, mode: str = "silu", groups: int = 32, nt: int = NT) -> str:
    """GroupNorm affine from conv_c's partials st [nparts][C][2] and gb = [gamma C | beta C]: a = rstd gamma,
    b = beta - mean a. `mode` "silu": gn_silu's gab [C / 64][64 a | 64 b] in the unpack's order; "plain": [a C | b C].
    A unit = a 16-channel strip. args: gab, st, gb, desc (`gn_prep_descs`)."""
    cpg = C // groups; assert 16 % cpg == 0 or cpg % 16 == 0
    ch = max(c for c in range(1, 129) if nparts % c == 0)             # partials per fill (<= 128: 16 KiB of LSRAM)
    assert cpg <= 16, "a group within a strip"
    n = count * cpg
    if mode == "silu":
        pos_a = "(k / 4) * 128 + (k % 4) * 16"; pos_b = "(k / 4) * 128 + 64 + (k % 4) * 16"; perm = "2 * (j % 8) + j / 8"
    else:
        pos_a = "k * 16"; pos_b = "%d + k * 16" % C; perm = "j"
    return FULL_H + f"""
__kernel void gn_prep(__global float* restrict gab, __global float* restrict st, __global float* restrict gb, __global int* restrict desc, const int core_id) {{
  __global float* P = LSF(0); __global float* GB = LSF({ch * 128}); __global float* O = LSF({ch * 128 + 128});
  for (int k = core_id; k < {C // 16}; k += {nt}) {{
    DMA_FILL(1, DESC(desc, 1), {ch * 128}, (int)(gb + k * 16)); DMA_FILL(2, DESC(desc, 1), {ch * 128 + 64}, (int)(gb + {C} + k * 16));
    float8 s0 = BC(0.0f), s1 = BC(0.0f), q0 = BC(0.0f), q1 = BC(0.0f);
    for (int c0 = 0; c0 < {nparts}; c0 += {ch}) {{
      DMA_FILL(0, DESC(desc, 0), 0, (int)(st + c0 * {2 * C} + k * 32)); DMA_WAIT_ALL();
      for (int p = 0; p < {ch}; p++) {{
        __global float* pp = P + p * 32;
        s0 += *(__global float8*)(pp); s1 += *(__global float8*)(pp + 8); q0 += *(__global float8*)(pp + 16); q1 += *(__global float8*)(pp + 24);
      }}
    }}
    float sm[16], sq[16], a[16], b[16];
    *(float8*)(sm) = s0; *(float8*)(sm + 8) = s1; *(float8*)(sq) = q0; *(float8*)(sq + 8) = q1;
    for (int g0 = 0; g0 < 16; g0 += {cpg}) {{
      float gs = 0.0f, gq = 0.0f;
      for (int c = 0; c < {cpg}; c++) {{ gs += sm[g0 + c]; gq += sq[g0 + c]; }}
      float mean = gs * {1.0 / n!r}f, var = gq * {1.0 / n!r}f - mean * mean; var = var > 0.0f ? var : 0.0f;
      float rstd = 1.0f / __builtin_sqrtf(var + {eps!r}f);
      for (int c = 0; c < {cpg}; c++) {{ a[g0 + c] = rstd * GB[g0 + c]; b[g0 + c] = GB[16 + g0 + c] - mean * a[g0 + c]; }}
    }}
    for (int j = 0; j < 16; j++) {{ O[j] = a[{perm}]; O[16 + j] = b[{perm}]; }}
    DMA_DRAIN(0, DESC(desc, 1), {ch * 128 + 128}, (int)(gab + {pos_a})); DMA_DRAIN(1, DESC(desc, 1), {ch * 128 + 192}, (int)(gab + {pos_b})); DMA_WAIT_ALL();
  }}
}}"""


def gn_prep_descs(C: int, nparts: int) -> np.ndarray:
    """0: a strip's partials over a chunk of parts (<= 128 x 128 B at the C x 8-B pitch), 1: 64 B (16 floats)."""
    ch = max(c for c in range(1, 129) if nparts % c == 0)
    return _desc_slots((ch * 128, 128, C * 8, 128), (64,))


def mlp_link_src(sp: int, N: int, act: str = "none", nt: int = NT) -> str:
    """One link of an fp32-accurate MLP chain: C of `gemm_gs([x_hi ; x_lo], [W_hi ; W_lo])` -> y = x_hi W_hi + x_hi W_lo
    + x_lo W_hi + bias, `act`, -> the next A [y_hi ; y_lo] (Veltkamp split). args: out, ct, bias, desc (`mlp_link_descs`)."""
    NG = N // 96; H = sp // 12
    ACT = {"none": "", "silu": "v = SILU(v);", "silu2": "v = SILU(v); v = SILU(v);"}[act]
    return "#pragma OPENCL FP_CONTRACT OFF\n" + FULL_H + EXP2_D7 + f"""
static inline float8 SILU(float8 v) {{ float8 d = BC(1.0f) + exp2_d7(v * BC(-1.4426950408889634f)), r = VRCP(d); r = r * (BC(2.0f) - d * r); return v * r; }}
__kernel void mlp_link(__global half* restrict out, __global float* restrict ct, __global float* restrict bias, __global int* restrict desc, const int core_id) {{
  __global float* T0 = LSF(0); __global float* T1 = LSF(4608); __global float* T2 = LSF(9216); __global float* B = LSF(13824);
  __global half* OH = LSH(14336); __global half* OL = LSH(16640);
  for (int u = core_id; u < {NG * H}; u += {nt}) {{
    int g = u / {H}, h = u % {H};
    DMA_FILL(0, DESC(desc, 0), 0, (int)(ct + (g * {2 * H} + h) * 1152)); DMA_FILL(1, DESC(desc, 0), 4608, (int)(ct + ((g + {NG}) * {2 * H} + h) * 1152));
    DMA_FILL(2, DESC(desc, 0), 9216, (int)(ct + (g * {2 * H} + h + {H}) * 1152)); DMA_FILL(3, DESC(desc, 1), 13824, (int)(bias + g * 96)); DMA_WAIT_ALL();
    for (int s = 0; s < 6; s++) for (int i = 0; i < 3; i++) for (int x = 0; x < 4; x++) {{
      int o = ((s * 3 + i) * 4 + x) * 16;
      float4 bq = *(__global float4*)(B + s * 16 + x * 4); float8 bb = __builtin_shufflevector(bq, bq, 0, 1, 2, 3, 4, 5, 6, 7);
      for (int e = 0; e < 16; e += 8) {{
        float8 v = *(__global float8*)(T0 + o + e) + *(__global float8*)(T1 + o + e) + *(__global float8*)(T2 + o + e) + bb;
        {ACT}
        float8 c = v * BC(8193.0f), hi = c - (c - v);
        *(__global float8*)(T0 + o + e) = hi; *(__global float8*)(T1 + o + e) = v - hi;
      }}
      int t = ((s * 4 + x) * 3 + i) * 16;
      *(__global half16*)(OH + t) = CVT16(*(__global float8*)(T0 + o), *(__global float8*)(T0 + o + 8));
      *(__global half16*)(OL + t) = CVT16(*(__global float8*)(T1 + o), *(__global float8*)(T1 + o + 8));
    }}
    DMA_DRAIN(0, DESC(desc, 2), 14336, (int)(out + (g * {2 * H} + h) * 1152)); DMA_DRAIN(1, DESC(desc, 2), 16640, (int)(out + (g * {2 * H} + h + {H}) * 1152));
    DMA_WAIT_ALL();
  }}
}}"""


def mlp_link_descs() -> np.ndarray:
    """0: a tile block (4608 B), 1: 96 biases, 2: an A chunk (2304 B)."""
    return _desc_slots((4608,), (384,), (2304,))


def fin_tables_src(sp: int, N: int, nt: int = NT) -> str:
    """The DiT final layer's per-step table: ws2 = dup_quads(1 + m), m = the step's row of the hi / lo GEMM + bias.
    args: ws2 (out), ct, bias, stepv, desc (`fin_tables_descs`)."""
    NG = N // 96; H = sp // 12
    return FULL_H + f"""
__kernel void fin_tables(__global float* restrict ws2, __global float* restrict ct, __global float* restrict bias, __global int* restrict stepv, __global int* restrict desc, const int core_id) {{
  __global float* T0 = LSF(0); __global float* T1 = LSF(4608); __global float* T2 = LSF(9216); __global float* B = LSF(13824); __global float* O = LSF(14336);
  __global int* PV = (__global int*)LSF(15104);
  DMA_FILL(0, DESC(desc, 3), 15104, (int)stepv); DMA_WAIT_ALL();
  int st = PV[0], h = st / 12, i = (st % 12) / 4, y = st % 4;
  for (int g = core_id; g < {NG}; g += {nt}) {{
    DMA_FILL(0, DESC(desc, 0), 0, (int)(ct + (g * {2 * H} + h) * 1152)); DMA_FILL(1, DESC(desc, 0), 4608, (int)(ct + ((g + {NG}) * {2 * H} + h) * 1152));
    DMA_FILL(2, DESC(desc, 0), 9216, (int)(ct + (g * {2 * H} + h + {H}) * 1152)); DMA_FILL(3, DESC(desc, 1), 13824, (int)(bias + g * 96)); DMA_WAIT_ALL();
    for (int s = 0; s < 6; s++) for (int x = 0; x < 4; x++) {{
      int o = ((s * 3 + i) * 4 + x) * 16 + y * 4;
      float4 v = *(__global float4*)(T0 + o) + *(__global float4*)(T1 + o) + *(__global float4*)(T2 + o) + *(__global float4*)(B + s * 16 + x * 4);
      float4 w = v + 1.0f;
      *(__global float8*)(O + (s * 4 + x) * 8) = __builtin_shufflevector(w, w, 0, 1, 2, 3, 0, 1, 2, 3);
    }}
    DMA_DRAIN(0, DESC(desc, 2), 14336, (int)(ws2 + g * 192)); DMA_WAIT_ALL();
  }}
}}"""


def fin_tables_descs() -> np.ndarray:
    """0: a tile block (4608 B), 1: 96 biases, 2: a group's table (768 B), 3: stepv (64 B)."""
    return _desc_slots((4608,), (384,), (768,), (64,))


def conv_perm32() -> np.ndarray:
    """`conv_a`'s channel order in a 32-channel block: position j holds channel perm[j] (16-channel halves, even channels then odd)."""
    j = np.arange(32); return (j // 16) * 16 + 2 * (j % 8) + (j % 16) // 8


def conv_a_descs(cin: int, up: bool = False) -> np.ndarray:
    """0: a neighbourhood fill, 1: the 1 x 1 fill (12 pixels x 288 B), 2: an A slice (3456 B), 3: p0v, 4: the GN affine."""
    npx = 8 if up else 14
    return _desc_slots((npx * 64, 64, cin * 2, 64), (12 * 288, 288, cin * 2, 288), (3456,), (64,), (256,))


def gn_silu_src(Wp: int, H: int, C: int, nrb: int, nt: int = NT) -> str:
    """GroupNorm affine + SiLU of a conv input, once per pixel: fp16 rows -> fp16 rows in the unpack's channel order
    (conv_a reads positions pi(pi(j))); pad columns not zeroed. A unit = (row block, 64 channels). args: out, src, gab
    (fp32 [C / 64][64 a | 64 b]), desc (`gn_silu_descs`), p0v (int32 [first pixel, pixel-0 offset])."""
    assert C % 64 == 0
    NG = C // 64
    return FULL_H + EXP2_D4 + f"""
__kernel void gn_silu(__global half* restrict out, __global half* restrict src, __global float* restrict gab, __global int* restrict desc, __global int* restrict p0v, const int core_id) {{
  /* DOUBLE-BUFFERED (2026-09-28): the next unit's fill and the previous unit's drain issued before this unit's work, one DMA_WAIT_ALL
     a unit. LSRAM: F 2 x 1536 (0), O 2 x 1536 (3072), every group's affine (6144, C / 64 x 512 B), p0 (10240) */
  __global int* PV = (__global int*)LSF(10240);
  DMA_FILL(0, DESC(desc, 2), 10240, (int)p0v); DMA_FILL(1, DESC(desc, 3), 6144, (int)gab); DMA_WAIT_ALL();
  int p0 = PV[0]; out = out + PV[1]; src = src + PV[1];
  if (core_id < {nrb * NG}) {{ DMA_FILL(0, DESC(desc, 0), 0, (int)(src + (p0 + (core_id % {nrb}) * 12) * {C} + (core_id / {nrb}) * 64)); DMA_WAIT_ALL(); }}
  int kk = 0;
  for (int u = core_id; u < {nrb * NG}; u += {nt}, kk++) {{
    int g = u / {nrb}, rb = u % {nrb}, pr = p0 + rb * 12, b = kk & 1;
    if (u + {nt} < {nrb * NG}) {{ int un = u + {nt}; DMA_FILL(0, DESC(desc, 0), (b ^ 1) * 1536, (int)(src + (p0 + (un % {nrb}) * 12) * {C} + (un / {nrb}) * 64)); }}
    if (kk > 0) {{ int up_ = u - {nt}; DMA_DRAIN(3, DESC(desc, 0), 3072 + (b ^ 1) * 1536, (int)(out + (p0 + (up_ % {nrb}) * 12) * {C} + (up_ / {nrb}) * 64)); }}
    __global half* F = LSH(b * 1536); __global half* O = LSH(3072 + b * 1536); __global float* GB = LSF(6144 + g * 512);
    for (int px = 0; px < 12; px++) for (int h = 0; h < 4; h++) {{
      half16 hv = *(__global half16*)(F + px * 64 + h * 16);
      float8 ve = __builtin_aipu_cvtue_tfp32_tfp16(hv) * *(__global float8*)(GB + h * 16) + *(__global float8*)(GB + 64 + h * 16);
      float8 vo = __builtin_aipu_cvtuo_tfp32_tfp16(hv) * *(__global float8*)(GB + h * 16 + 8) + *(__global float8*)(GB + 72 + h * 16);
      ve = ve * VRCP(BC(1.0f) + exp2_d4(BC({-LOG2E!r}f) * ve)); vo = vo * VRCP(BC(1.0f) + exp2_d4(BC({-LOG2E!r}f) * vo));
      *(__global half16*)(O + px * 64 + h * 16) = CVT16(ve, vo);
    }}
    DMA_WAIT_ALL();
  }}
  if (kk > 0) {{ int up_ = core_id + (kk - 1) * {nt}, b = (kk - 1) & 1;
    DMA_DRAIN(3, DESC(desc, 0), 3072 + b * 1536, (int)(out + (p0 + (up_ % {nrb}) * 12) * {C} + (up_ / {nrb}) * 64)); DMA_WAIT_ALL(); }}
}}"""


def gn_silu_descs(C: int) -> np.ndarray:
    """0: 12 rows x 128 B at the rows' pitch (fill and drain), 1: unused, 2: p0v, 3: every group's affine."""
    return _desc_slots((12 * 128, 128, C * 2, 128), (512,), (64,), (C * 8,))

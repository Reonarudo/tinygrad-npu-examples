"""The decode GEMV as an integer dot product on the TEC's vector unit (`qdot`: 4 unsigned bytes x 4 signed bytes into an int32 lane,
8 lanes, two a bundle), over the SAME gemm_gs streams the prefill's matrix-unit GEMMs read (one copy of the 21.5 GB of weights),
instead of `gemm_gs` in rows mode, whose 12-row tile does 1 useful row in 12 at decode and whose 4-bit expand is its bound.

  Q4_K  w[n, k] = a[n, j] u[n, k] + b[n, j]     (per 32-weight sub-block j: a = d sc, b = -dmin m, u in 0..15)
  Q6_K  w[n, k] = d[n, i] q[n, k]               (per 16-weight block i, q in -32..31, stored as u = q + 128)
  x[k] ~= sx1[j] q1[k] + sx2[j] q2[k]           the activation in two int8 levels a 32: q1 = rint(x / sx1), sx1 = max|x| / 127, and
                                                the residual x - sx1 q1 the same way (LEV 2; LEV 1: q1 alone, llama.cpp-like)
  y[n]  = sum_j a sum_l sx_l (sum_k u q_l) + b (sum_k x)      -- the inner sums in int32 by qdot, the rest in fp32 (the b term with
          the true fp32 sum of x); Q6_K: d sum_l sx_l (sum_k u q_l - 128 sum_k q_l)

On the model's own hidden states (check/act_quant.py) one int8 level costs 0.4-2 % on q / k / v (llama.cpp's one scale a 256:
1-5 %), two levels 2e-5 - 8e-5 -- below the fp16 activations of the matrix-unit path.

The lanes are output columns, and the streams already lay them so (`gemm_fp16`):
  Q4_K (q8=5, `pack_b_group_q4m`): a group (48 columns = 3 strips of 16) is, per 128-wide K-slice, 3072 B of codes ([s][p][j][n][k]:
     the 32 B at s 1024 + p 64 + 32 gi are columns 16 s + 8 gi + 0..7 x the 4 k of k-quad 2 p (low nibbles) and 2 p + 1 (high):
     `c & 15` and `c >> 4` are qdot operands), then [s][sb][16 cols] fp16 a (zipl / ziph with itself + cvtue: the 8-column halves) and
     [s] b tiles (column c's sub-block sb at half 4 c + sb: cvtue / cvtuo give the even / odd sub-blocks of 4 columns, exte / exto of
     two such pair them up) -- 3840 B a (group, slice).
  Q6_K (q8=4, `pack_b_group_q8f(qb=16)`): per slice 6144 B of codes ([s][kk][j][n][k]: the 32 B at s 2048 + kk 64 + 32 gi are columns
     16 s + 8 gi + 0..7 x the 4 k of k-step kk) and [s][16-block][16 cols] fp16 d -- 6912 B.

The activation package: a 512-byte record a K-slice --
  [0:128] q1 of its 4 sub-blocks, [128:256] q2, [256:288] float8 [sm0 sm2 sm0 sm2 ..], [288:320] [sm1 sm3 ..] (Q4_K's b-term
  partners), [320:336] sx1[4], [336:352] sx2[4], [352:368] sm[4], [384:416] 128 x the 8 16-blocks' sums of q1, [416:448] of q2.
"""
import os
import numpy as np
from qwen3moe_kernels import V, NT

LEV = int(os.environ.get("QWEN3MOE_QLEV", "2"))   # int8 levels of the activation
SL_REC = 512                                      # package bytes a K-slice
SLB = {"q4": 3840, "q6": 6912, "q6n": 5088, "q4n": 1216, "q8": 6528, "q5n": 4416, "mx4": 3456, "q51": 4608}   # stream bytes a (unit, slice): a 48-column group, or a 16-column strip (q4n)
CU = {"q4": 48, "q6": 48, "q6n": 48, "q4n": 16, "q8": 48, "q5n": 48, "mx4": 48, "q51": 48}             # columns a unit
Q6N_MODE = os.environ.get("QWEN3MOE_Q6N_MODE", "shift")      # the q6n body: shift (a runtime 16-k block loop, the dense activation register re-pointed at the block by a
                                                             # 16-byte-offset load: no spills, the head at the port's 23.6 GB/s) | scratch (replicas from an LSRAM scratch: 18.9) | unrolled (20.4, ~56 spills)
O_Q2, O_SME, O_SMO, O_SX, O_SX2, O_SUM, O_S16, O_S16B = 128, 256, 288, 320, 336, 352, 384, 416
_R127 = np.float32(1.0 / 127.0)


# ---------------------------------------------------------------- host: the numpy references
def _q8(x):
  """float32 [n, 32] -> (int8 codes, sx) as the kernel: sx = max|x| (1/127), q = clip(rint(x 127 / max|x|))."""
  amax = np.abs(x).max(1); inv = np.where(amax > 0, np.float32(127.0) / np.where(amax > 0, amax, 1), 0).astype(np.float32)
  return np.clip(np.rint(x * inv[:, None]), -127, 127).astype(np.int8), (amax * _R127).astype(np.float32)


def qact_ref(x, lev=LEV):
  """fp32 [K] -> ([(q int8 [K], sx float32 [K/32]) a level], sm float32 [K/32])."""
  x = np.asarray(x, np.float32).reshape(-1, 32); out = []; r = x
  for _ in range(lev):
    q, sx = _q8(r); out.append((q.reshape(-1), sx)); r = (r - q.astype(np.float32) * sx[:, None]).astype(np.float32)
  return out, x.sum(1, dtype=np.float32)


def qact4m_ref(x, lev=LEV):
  """fp32 [K] -> the package bytes (from qact_ref's values)."""
  levels, sm = qact_ref(x, lev); nsl = sm.size // 4; out = np.zeros((nsl, SL_REC), np.uint8); s4 = np.ascontiguousarray(sm.reshape(nsl, 4))
  for li, (q, sx) in enumerate(levels):
    out[:, 128 * li:128 * li + 128] = q.reshape(nsl, 128).view(np.uint8)
    out[:, O_SX + 16 * li:O_SX + 16 * li + 16] = np.ascontiguousarray(sx.reshape(nsl, 4)).view(np.uint8)
    out[:, O_S16 + 32 * li:O_S16 + 32 * li + 32] = (q.reshape(nsl, 8, 16).astype(np.int32).sum(-1) * 128).astype(np.int32).view(np.uint8)
  out[:, O_SME:O_SME + 32] = np.ascontiguousarray(np.tile(s4[:, [0, 2]], 4)).view(np.uint8)
  out[:, O_SMO:O_SMO + 32] = np.ascontiguousarray(np.tile(s4[:, [1, 3]], 4)).view(np.uint8)
  out[:, O_SUM:O_SUM + 16] = s4.view(np.uint8); return out.reshape(-1)


def qgemv_ref(levels, sm, u, a, b):
  """Q4_K: y [N] in float64 from the activation levels and the factored weights with a / b rounded to fp16 (the kernel's operands)."""
  N, K = u.shape; a16, b16 = a.astype(np.float16).astype(np.float64), b.astype(np.float16).astype(np.float64); y = b16 @ sm.astype(np.float64)
  for q, sx in levels:
    P = (u.reshape(N, K // 32, 32).astype(np.int64) * q.reshape(1, K // 32, 32).astype(np.int64)).sum(-1); y = y + (a16 * sx[None].astype(np.float64) * P).sum(1)
  return y


def qgemv6_ref(levels, qw, d):
  """Q6_K: y [N] in float64 (qw int8 [N, K] in -32..31, d [N, K/16] rounded to fp16)."""
  N, K = qw.shape; d16 = d.astype(np.float16).astype(np.float64); y = np.zeros(N)
  for q, sx in levels:
    P = (qw.reshape(N, K // 16, 16).astype(np.int64) * q.reshape(1, K // 16, 16).astype(np.int64)).sum(-1)
    y = y + (d16 * np.repeat(sx, 2)[None].astype(np.float64) * P).sum(1)
  return y


def q6k_split_raw(blk):
  """GGUF Q6_K super-blocks [N, K/256, 210] -> (q int8 [N, K] in -32..31, sc int8 [N, K/16], d float32 [N, K/256]): w = d sc q."""
  N, nsb = blk.shape[:2]; f = blk.reshape(-1, 210)
  ql, qh = f[:, 0:128].astype(np.int32), f[:, 128:192].astype(np.int32)
  sc = f[:, 192:208].copy().view(np.int8); d = f[:, 208:210].copy().view(np.float16)[:, 0].astype(np.float32)
  q = np.empty((f.shape[0], 256), np.int32)
  for n in range(2):
    L, Hh = ql[:, 64 * n:64 * n + 64], qh[:, 32 * n:32 * n + 32]; o = 128 * n
    q[:, o:o + 32] = (L[:, :32] & 0xF) | ((Hh & 3) << 4); q[:, o + 32:o + 64] = (L[:, 32:] & 0xF) | (((Hh >> 2) & 3) << 4)
    q[:, o + 64:o + 96] = (L[:, :32] >> 4) | (((Hh >> 4) & 3) << 4); q[:, o + 96:o + 128] = (L[:, 32:] >> 4) | (((Hh >> 6) & 3) << 4)
  return (q - 32).astype(np.int8).reshape(N, nsb * 256), sc.reshape(N, nsb * 16), d.reshape(N, nsb)


def pack_b_group_q6n(q, sc, d, group, ns=3):
  """The decode-only Q6_K stream ("q6n", 6.625 bits a weight; nothing requantised): per (group of ns strips, 128-k slice)
    [0:3072]     the 6-bit codes' low nibbles, laid as the Q4_K stream's ([s][k-quad pair p][j][n][k]: quad 2p low, 2p + 1 high)
    [3072:4608]  their high 2 bits: [s][sub-block][gi][m][8 cols][4 k] bytes, bits 2 i for k-quad 4 m + i of the sub-block
    [4608:4992]  the blocks' int8 scales + 128: [s][gi][m][8 cols][4 blocks] (block 4 m + b of the slice's 8)
    [4992:5088]  [s][16 cols] fp16 d (the super-block's)
  so a code is u = q + 32 (0..63) and w = d sc (u - 32); the kernel reads sc with one qdot a block (a unit byte pattern)."""
  N, K = q.shape; assert K % 256 == 0 and sc.shape == (N, K // 16) and d.shape == (N, K // 256)
  r0, r1 = 16 * ns * group, 16 * ns * (group + 1); nsl = K // 128; nr = max(0, min(N, r1) - r0)
  u = np.full((16 * ns, K), 32, np.uint8); u[:nr] = (q[r0:r0 + nr].astype(np.int16) + 32).astype(np.uint8)
  t = u.reshape(ns, 4, 4, nsl, 16, 2, 4).transpose(3, 0, 4, 5, 1, 2, 6)                         # [slice][s][p][h][j][n][k]
  nib = np.ascontiguousarray((t[:, :, :, 0] & 15) | ((t[:, :, :, 1] & 15) << 4)).reshape(nsl, -1)
  hi = (u >> 4).reshape(ns, 2, 8, nsl, 4, 2, 4, 4).transpose(3, 0, 4, 1, 5, 2, 7, 6)             # [slice][s][sb][gi][m][l][k][i]
  hw = np.ascontiguousarray(hi[..., 0] | (hi[..., 1] << 2) | (hi[..., 2] << 4) | (hi[..., 3] << 6)).reshape(nsl, -1)
  s8 = np.full((16 * ns, K // 16), 128, np.uint8); s8[:nr] = (sc[r0:r0 + nr].astype(np.int16) + 128).astype(np.uint8)
  scw = np.ascontiguousarray(s8.reshape(ns, 2, 8, nsl, 2, 4).transpose(3, 0, 1, 4, 2, 5)).reshape(nsl, -1)   # [slice][s][gi][m][l][b]
  dd = np.zeros((16 * ns, K // 256), np.float16); dd[:nr] = d[r0:r0 + nr]
  d16 = np.ascontiguousarray(np.repeat(dd, 2, 1).reshape(ns, 16, nsl).transpose(2, 0, 1)).reshape(nsl, -1).view(np.uint8)   # [slice][s][16 cols]
  out = np.concatenate([nib, hw, scw, d16], 1); assert out.shape[1] == SLB["q6n"], out.shape
  return np.ascontiguousarray(out).ravel()


def qgemv6n_ref(levels, q, sc, d):
  """Q6_K on the q6n stream: y [N] in float64 with the block scale a = d sc rounded once to fp32 (the kernel's operand)."""
  N, K = q.shape; a = (np.repeat(d, 16, 1) * sc.astype(np.float32)).astype(np.float32).astype(np.float64); y = np.zeros(N)
  for qa, sx in levels:
    P = (q.reshape(N, K // 16, 16).astype(np.int64) * qa.reshape(1, K // 16, 16).astype(np.int64)).sum(-1)
    y = y + (a * np.repeat(sx, 2)[None].astype(np.float64) * P).sum(1)
  return y


def q4k_split_raw(blk):
  """GGUF Q4_K super-blocks [N, K/256, 144] -> (u uint8 [N, K] 0..15, sc uint8 [N, K/32] 0..63, m uint8 [N, K/32], d float32 [N, K/256],
  dmin float32 [N, K/256]): w = d sc u - dmin m per 32-weight sub-block."""
  N, nsb = blk.shape[:2]; f = blk.reshape(-1, 144)
  d = f[:, 0:2].copy().view(np.float16)[:, 0].astype(np.float32); dmin = f[:, 2:4].copy().view(np.float16)[:, 0].astype(np.float32)
  sc, qs = f[:, 4:16].astype(np.int32), f[:, 16:144]
  s_ = np.empty((f.shape[0], 8), np.int32); m = np.empty_like(s_)
  s_[:, :4], m[:, :4] = sc[:, 0:4] & 63, sc[:, 4:8] & 63
  s_[:, 4:] = (sc[:, 8:12] & 0xF) | ((sc[:, 0:4] >> 6) << 4); m[:, 4:] = (sc[:, 8:12] >> 4) | ((sc[:, 4:8] >> 6) << 4)
  q = qs.reshape(-1, 4, 32); u = np.stack([q & 0xF, q >> 4], 2).reshape(-1, 8, 32)
  return u.reshape(N, nsb * 256).astype(np.uint8), s_.astype(np.uint8).reshape(N, nsb * 8), m.astype(np.uint8).reshape(N, nsb * 8), d.reshape(N, nsb), dmin.reshape(N, nsb)


def pack_b_group_q4n(u, sc, m, d, dmin, strip):
  """The decode-only Q4_K stream ("q4n", 4.75 bits a weight; nothing requantised), strip-major: per (16-column strip, 128-k slice)
    [0:1024]     the codes as the Q4_K stream's strip ([k-quad pair p][j][n][k]: quad 2 p in the low nibble, 2 p + 1 in the high)
    [1024:1088]  the 6-bit sc as bytes: [gi][8 cols][4 sub-blocks]      (a sub-block's 8 scales by one qdot against a unit-byte pattern)
    [1088:1152]  the 6-bit m the same way
    [1152:1184]  [16 cols] fp16 d, [1184:1216] [16 cols] fp16 -dmin     (the super-block's; w = d sc u + (-dmin) m)
  A unit is one strip (balance: the o projection's 129 strips on 12 tasks), and a strip's slice is one contiguous DMA."""
  N, K = u.shape; nsl = K // 128; r0 = 16 * strip; nr = max(0, min(N, r0 + 16) - r0)
  uu = np.zeros((16, K), np.uint8); uu[:nr] = u[r0:r0 + nr]
  t = uu.reshape(4, 4, nsl, 16, 2, 4).transpose(2, 3, 4, 0, 1, 5)                                 # [slice][p][h][j][n][k]
  nib = np.ascontiguousarray(t[:, :, 0] | (t[:, :, 1] << 4)).reshape(nsl, -1)                      # [slice][p j n k] 1024 B
  def words(v):
    vv = np.zeros((16, K // 32), np.uint8); vv[:nr] = v[r0:r0 + nr]
    return np.ascontiguousarray(vv.reshape(2, 8, nsl, 4).transpose(2, 0, 1, 3)).reshape(nsl, -1)   # [slice][gi][l][sb] 64 B
  dd = np.zeros((2, 16, K // 256), np.float16); dd[0, :nr] = d[r0:r0 + nr]; dd[1, :nr] = -dmin[r0:r0 + nr]
  d16 = np.ascontiguousarray(np.repeat(dd, 2, 2).transpose(2, 0, 1)).reshape(nsl, -1).view(np.uint8)  # [slice][d | -dmin][16 cols] 64 B
  out = np.concatenate([nib, words(sc), words(m), d16], 1); assert out.shape[1] == SLB["q4n"], out.shape
  return np.ascontiguousarray(out).ravel()


def qgemv4n_ref(levels, sm, u, sc, m, d, dmin):
  """Q4_K on the q4n stream: y [N] in float64 with a = d sc and b = -dmin m each rounded once to fp32 (the kernel's operands)."""
  N, K = u.shape; r = lambda v: np.repeat(v, 8, 1)
  a = (r(d) * sc.astype(np.float32)).astype(np.float32).astype(np.float64); b = (r(-dmin) * m.astype(np.float32)).astype(np.float32).astype(np.float64)
  y = (b * sm[None].astype(np.float64)).sum(1)
  for q, sx in levels:
    P = (u.reshape(N, K // 32, 32).astype(np.int64) * q.reshape(1, K // 32, 32).astype(np.int64)).sum(-1); y = y + (a * sx[None].astype(np.float64) * P).sum(1)
  return y


def q5k_split_raw(blk):
  """GGUF Q5_K super-blocks [N, K/256, 176] -> (u uint8 [N, K] 0..31, sc uint8 [N, K/32] 0..63, m uint8 [N, K/32], d float32 [N, K/256],
  dmin float32 [N, K/256]): w = d sc u - dmin m per 32-weight sub-block (ggml's dequantize_row_q5_K: sub-block j's high bit is bit j
  of the 32 qh bytes, its low nibble the low (even j) / high (odd j) nibble of chunk j / 2)."""
  N, nsb = blk.shape[:2]; f = blk.reshape(-1, 176)
  d = f[:, 0:2].copy().view(np.float16)[:, 0].astype(np.float32); dmin = f[:, 2:4].copy().view(np.float16)[:, 0].astype(np.float32)
  sc, qh, qs = f[:, 4:16].astype(np.int32), f[:, 16:48].astype(np.int32), f[:, 48:176].astype(np.int32)
  s_ = np.empty((f.shape[0], 8), np.int32); m = np.empty_like(s_)
  s_[:, :4], m[:, :4] = sc[:, 0:4] & 63, sc[:, 4:8] & 63
  s_[:, 4:] = (sc[:, 8:12] & 0xF) | ((sc[:, 0:4] >> 6) << 4); m[:, 4:] = (sc[:, 8:12] >> 4) | ((sc[:, 4:8] >> 6) << 4)
  q = qs.reshape(-1, 4, 32); u = np.stack([q & 0xF, q >> 4], 2).reshape(-1, 8, 32); h = np.stack([(qh >> j) & 1 for j in range(8)], 1)
  return (u + 16 * h).astype(np.uint8).reshape(N, nsb * 256), s_.astype(np.uint8).reshape(N, nsb * 8), m.astype(np.uint8).reshape(N, nsb * 8), d.reshape(N, nsb), dmin.reshape(N, nsb)


def pack_b_group_q5n(u, sc, m, d, dmin, group, ns=3):
  """The decode-only Q5_K stream ("q5n", 5.75 bits a weight; nothing requantised), group-major: per (group of ns strips, 128-k slice)
    [0:3072]     the codes' low nibbles, laid as the Q4_K / q6n stream's ([s][k-quad pair p][j][n][k]: quad 2p low, 2p + 1 high)
    [3072:3840]  their high bits: [s][sub-block][j][n][k] bytes, bit q = k-quad q of the sub-block (eight weights a byte)
    [3840:4032]  the 6-bit sc as bytes: [s][gi][8 cols][4 sub-blocks]   (a sub-block's 8 scales by one qdot against a unit-byte pattern)
    [4032:4224]  the 6-bit m the same way
    [4224:4320]  [s][16 cols] fp16 d, [4320:4416] [s][16 cols] fp16 -dmin   (the super-block's; w = d sc u + (-dmin) m)"""
  N, K = u.shape; assert K % 256 == 0; nsl = K // 128; r0, r1 = 16 * ns * group, 16 * ns * (group + 1); nr = max(0, min(N, r1) - r0)
  uu = np.zeros((16 * ns, K), np.uint8); uu[:nr] = u[r0:r0 + nr]; lo, hi = uu & 15, uu >> 4
  t = lo.reshape(ns, 4, 4, nsl, 16, 2, 4).transpose(3, 0, 4, 5, 1, 2, 6)                        # [slice][s][p][h][j][n][k]
  nib = np.ascontiguousarray(t[:, :, :, 0] | (t[:, :, :, 1] << 4)).reshape(nsl, -1)
  th = hi.reshape(ns, 4, 4, nsl, 4, 8, 4).transpose(3, 0, 4, 1, 2, 6, 5)                         # [slice][s][sb][j][n][k][q]
  hb = np.zeros(th.shape[:-1], np.uint8)
  for q in range(8): hb |= (th[..., q] << q).astype(np.uint8)
  hb = np.ascontiguousarray(hb).reshape(nsl, -1)
  def words(v):
    vv = np.zeros((16 * ns, K // 32), np.uint8); vv[:nr] = v[r0:r0 + nr]
    return np.ascontiguousarray(vv.reshape(ns, 2, 8, nsl, 4).transpose(3, 0, 1, 2, 4)).reshape(nsl, -1)   # [slice][s][gi][l][sb]
  def halves(v):
    dd = np.zeros((16 * ns, K // 256), np.float16); dd[:nr] = v[r0:r0 + nr]
    return np.ascontiguousarray(np.repeat(dd, 2, 1).reshape(ns, 16, nsl).transpose(2, 0, 1)).reshape(nsl, -1).view(np.uint8)   # [slice][s][16 cols]
  out = np.concatenate([nib, hb, words(sc), words(m), halves(d), halves(-dmin)], 1); assert out.shape[1] == SLB["q5n"], out.shape
  return np.ascontiguousarray(out).ravel()


qgemv5n_ref = qgemv4n_ref                                                 # the same arithmetic (u in 0..31)


def q51_split_raw(blk):
  """GGUF Q5_1 blocks [N, K/32, 24] -> (u uint8 [N, K] 0..31, d float32 [N, K/32], m float32 [N, K/32]): w = d u + m a 32-weight block."""
  from gguf_read import q51_split
  N, nb = blk.shape[:2]; u, d, m = q51_split(blk.reshape(-1, 24)); return u.reshape(N, nb * 32), d.reshape(N, nb), m.reshape(N, nb)


def pack_b_group_q51(u, d, m, group, ns=3):
  """The Q5_1 qdot stream ("q51", 6 bits a weight; nothing requantised): the q5n stream's nibbles [0:3072] and high-bit plane
  [3072:3840], then [3840:4224] [s][sub-block][16 cols] fp16 d and [4224:4608] the same of m (the q8 / mx4 stream's table layout);
  w = d u + m a sub-block, so the kernel's a = d, b = m need no scale decode."""
  N, K = u.shape; assert K % 128 == 0; nsl = K // 128; r0, r1 = 16 * ns * group, 16 * ns * (group + 1); nr = max(0, min(N, r1) - r0)
  uu = np.zeros((16 * ns, K), np.uint8); uu[:nr] = u[r0:r0 + nr]; lo, hi = uu & 15, uu >> 4
  t = lo.reshape(ns, 4, 4, nsl, 16, 2, 4).transpose(3, 0, 4, 5, 1, 2, 6)
  nib = np.ascontiguousarray(t[:, :, :, 0] | (t[:, :, :, 1] << 4)).reshape(nsl, -1)
  th = hi.reshape(ns, 4, 4, nsl, 4, 8, 4).transpose(3, 0, 4, 1, 2, 6, 5); hb = np.zeros(th.shape[:-1], np.uint8)
  for q in range(8): hb |= (th[..., q] << q).astype(np.uint8)
  hb = np.ascontiguousarray(hb).reshape(nsl, -1)
  def table(v):
    dd = np.zeros((16 * ns, K // 32), np.float16); dd[:nr] = v[r0:r0 + nr]
    return np.ascontiguousarray(dd.reshape(ns, 16, nsl, 4).transpose(2, 0, 3, 1)).reshape(nsl, -1).view(np.uint8)      # [slice][s][sb][16 cols]
  out = np.concatenate([nib, hb, table(d), table(m)], 1); assert out.shape[1] == SLB["q51"], out.shape
  return np.ascontiguousarray(out).ravel()


def qgemv_q51_ref(levels, sm, u, d, m):
  """Q5_1 on the q51 stream: y [N] in float64 with the fp16 d and m as the kernel reads them."""
  N, K = u.shape; a = d.astype(np.float16).astype(np.float64); b = m.astype(np.float16).astype(np.float64); y = (b * sm[None].astype(np.float64)).sum(1)
  for q, sx in levels:
    P = (u.reshape(N, K // 32, 32).astype(np.int64) * q.reshape(1, K // 32, 32).astype(np.int64)).sum(-1); y = y + (a * sx[None].astype(np.float64) * P).sum(1)
  return y


def mxfp4_split_raw(blk):
  """GGUF MXFP4 blocks [N, K/32, 17] -> (code uint8 [N, K]: the e2m1 nibble (bit 3 the sign, bits 0-2 the magnitude index into
  {0, 1, 2, 3, 4, 6, 8, 12} = the value x 2), dh float32 [N, K/32] = 2^(e - 128), the halved E8M0 scale): w = dh kv[code]
  (ggml's dequantize_row_mxfp4: the low nibbles are weights 0..15 of the block, the high 16..31)."""
  N, nb = blk.shape[:2]; f = blk.reshape(-1, 17); e = f[:, 0].astype(np.int32); qs = f[:, 1:17]
  code = np.concatenate([qs & 15, qs >> 4], 1).astype(np.uint8); dh = np.ldexp(1.0, e - 128).astype(np.float32)
  return code.reshape(N, nb * 32), dh.reshape(N, nb)


KV_FP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.int32)


def pack_b_group_mx4(code, dh, group, ns=3):
  """The MXFP4 qdot stream ("mx4", 4.5 bits a weight; nothing requantised): per (group of ns strips, 128-k slice)
    [0:3072]     the e2m1 nibbles as the Q4_K stream lays them ([s][k-quad pair p][j][n][k]: quad 2p low, 2p + 1 high)
    [3072:3456]  [s][sub-block][16 cols] fp16 of the halved E8M0 scale (as the q8 stream's table)
  The kernel expands a nibble to u = 16 + kv[nibble] (4..28) by arithmetic and corrects 16 sum(x) out of the dot."""
  N, K = code.shape; assert K % 128 == 0; nsl = K // 128; r0, r1 = 16 * ns * group, 16 * ns * (group + 1); nr = max(0, min(N, r1) - r0)
  uu = np.zeros((16 * ns, K), np.uint8); uu[:nr] = code[r0:r0 + nr]
  t = uu.reshape(ns, 4, 4, nsl, 16, 2, 4).transpose(3, 0, 4, 5, 1, 2, 6)
  nib = np.ascontiguousarray(t[:, :, :, 0] | (t[:, :, :, 1] << 4)).reshape(nsl, -1)
  dd = np.zeros((16 * ns, K // 32), np.float16); dd[:nr] = dh[r0:r0 + nr]
  d16 = np.ascontiguousarray(dd.reshape(ns, 16, nsl, 4).transpose(2, 0, 3, 1)).reshape(nsl, -1).view(np.uint8)     # [slice][s][sb][16 cols]
  out = np.concatenate([nib, d16], 1); assert out.shape[1] == SLB["mx4"], out.shape
  return np.ascontiguousarray(out).ravel()


def qgemv_mx4_ref(levels, code, dh):
  """MXFP4 on the mx4 stream: y [N] in float64 (the halved scale exact in fp16)."""
  N, K = code.shape; v = KV_FP4[code].astype(np.int64); d16 = dh.astype(np.float16).astype(np.float64); y = np.zeros(N)
  for q, sx in levels:
    P = (v.reshape(N, K // 32, 32) * q.reshape(1, K // 32, 32).astype(np.int64)).sum(-1); y = y + (d16 * sx[None].astype(np.float64) * P).sum(1)
  return y


# ---------------------------------------------------------------- device
QH = r"""
typedef unsigned char uchar;
typedef unsigned char uchar32 __attribute__((__vector_size__(32)));
typedef bool bool32 __attribute__((ext_vector_type(32)));
#define ALLB ((bool32)(1))
#define QD(u, x) __builtin_aipu_qdot_tub_tb_pb(Z8, (u), (x), ALLB)
#define QA(p, u, x) __builtin_aipu_qdpa_tub_tb_pb((p), (u), (x), ALLB)
#define LSB(off) ((__global uchar*)(LSRAM_BASE + (off)))
#define BCI(v) ((int8v){(v), (v), (v), (v), (v), (v), (v), (v)})
static inline __attribute__((always_inline)) float8 vabs(float8 v) { return __builtin_aipu_abs_tfp32_pw(v, v, ALL); }
/* one int8 level of 32 values at o + qo: the codes, sx at o + so, the 16-blocks' sums x 128 at o + bo; the residual left in v0..v3 */
static inline __attribute__((always_inline)) void qlev(float8* v0, float8* v1, float8* v2, float8* v3, __global uchar* o, int qo, int so, int bo, int sb) {
  float8 m8 = VMAX(VMAX(vabs(*v0), vabs(*v1)), VMAX(vabs(*v2), vabs(*v3))); float amax = __builtin_aipu_rpmax_tfp32_pw(m8, m8, ALL)[0]; float inv = amax > 0.0f ? 127.0f / amax : 0.0f; float8 i8 = BC(inv);
  int8v q0 = q8v(*v0 * i8), q1 = q8v(*v1 * i8), q2 = q8v(*v2 * i8), q3 = q8v(*v3 * i8); float sx = amax * (1.0f / 127.0f); float8 s8 = BC(sx);
  *(__global char32*)(o + qo + sb * 32) = pack32(q0, q1, q2, q3);
  int8v c01 = q0 + q1, c23 = q2 + q3;
  ((__global int*)(o + bo))[2 * sb] = 128 * __builtin_aipu_rpadd_tw_pw(c01, c01, ALL)[0]; ((__global int*)(o + bo))[2 * sb + 1] = 128 * __builtin_aipu_rpadd_tw_pw(c23, c23, ALL)[0];
  ((__global float*)(o + so))[sb] = sx;
  *v0 = *v0 - VF(q0) * s8; *v1 = *v1 - VF(q1) * s8; *v2 = *v2 - VF(q2) * s8; *v3 = *v3 - VF(q3) * s8;
}
/* sub-block sb of a package record at o from its 32 values */
static inline __attribute__((always_inline)) void qsub(float8 v0, float8 v1, float8 v2, float8 v3, __global uchar* o, int sb) {
  float8 t8 = (v0 + v1) + (v2 + v3); ((__global float*)(o + %(O_SUM)d))[sb] = __builtin_aipu_rpadd_tfp32_pw(t8, t8, ALL)[0];
  qlev(&v0, &v1, &v2, &v3, o, 0, %(O_SX)d, %(O_S16)d, sb);%(LEV2)s
}
/* the record's b-term partners from its four sums */
static inline __attribute__((always_inline)) void qfin(__global uchar* o) {
  __global float* m = (__global float*)(o + %(O_SUM)d); float s0 = m[0], s1 = m[1], s2 = m[2], s3 = m[3];
  *(__global float8*)(o + %(O_SME)d) = (float8){s0, s2, s0, s2, s0, s2, s0, s2}; *(__global float8*)(o + %(O_SMO)d) = (float8){s1, s3, s1, s3, s1, s3, s1, s3};
}
static inline float8 silu8(float8 g) { return g * VRCP(BC(1.0f) + exp2_f8(g * BC(-1.4426950408889634f))); }
""" % dict(O_SUM=O_SUM, O_SX=O_SX, O_S16=O_S16, O_SME=O_SME, O_SMO=O_SMO,
           LEV2=f"\n  qlev(&v0, &v1, &v2, &v3, o, {O_Q2}, {O_SX2}, {O_S16B}, sb);" if LEV == 2 else "")


def qact4m_src(K, heads=1, mode="plain", eps=1e-6, nt=12, xs=None):
  """The package of `heads` rows as a kernel of its own (K / 128 records a row; a task a run of records; plain loads: the tests'
  reference producer -- the decode layer builds its packages inside the kernels around). Row h's values by `mode`:
    plain   x[h];   norm   rms(x[h]) w;   swiglu   silu(x[h]) x[heads + h]  (gate | up rows `xs` floats apart, default K)
  args: pkg (out), x, w (unused ones: any buffer)."""
  nsl = K // 128; U = heads * nsl; xs = xs or K; assert mode in ("plain", "norm", "swiglu")
  V8 = lambda e: f"*(__global float8*)({e})"
  nrm = f"""
    float8 s8 = BC(0.0f); for (int i = 0; i < {K}; i += 8) {{ float8 v = {V8("xr + i")}; s8 += v * v; }}
    iv = 1.0f / __builtin_sqrtf(hsum8(s8) * {1.0 / K!r}f + {eps!r}f);""" if mode == "norm" else ""
  if mode == "plain": ld = lambda i: V8(f"xs + {8 * i}")
  elif mode == "norm": ld = lambda i: f"{V8(f'xs + {8 * i}')} * {V8(f'ws + {8 * i}')} * BC(iv)"
  else: ld = lambda i: f"silu8({V8(f'xs + {8 * i}')}) * {V8(f'xs + {heads * xs + 8 * i}')}"
  return V.FULL_H + QH + f"""
__kernel void qact4m(__global uchar* restrict pkg, __global float* restrict x, __global float* restrict w, const int core_id) {{
  int u0 = core_id * {U} / {nt}, u1 = (core_id + 1) * {U} / {nt}; int hprev = -1; float iv = 1.0f;
  for (int uu = u0; uu < u1; uu++) {{
    int h = uu / {nsl}, sl = uu % {nsl}; __global float* xr = x + h * {xs};
    if (h != hprev) {{ hprev = h;{nrm} }}
    __global uchar* o = pkg + uu * {SL_REC};
    for (int sb = 0; sb < 4; sb++) {{
      __global float* xs = xr + sl * 128 + sb * 32; __global float* ws = w + sl * 128 + sb * 32;
      qsub({ld(0)}, {ld(1)}, {ld(2)}, {ld(3)}, o, sb);
    }}
    qfin(o);
  }}
}}"""


def qgemvk_desc(kind, K, ne=128, cs=1): return V._desc_slots((SLB[kind] * cs,), (K // 128 * SL_REC,), (K * 4,), (ne * 4,))   # (q4n: a strip's slice)


def qgemvk_np(N): return -(-N // 48) * 48


def qgemvk_src(kind, N, K, heads=1, act_heads=1, split=0, yoff=0, nt=12, nbuf=3, act="pkg", export=False, gu_np=0, eps=1e-6, top=False, route=None, cs=1, kp=1):
  """y [heads, NP] fp32 (NP = N rounded up to 48: the last group's padding columns come out 0) = W_h x for `heads` matrices in
  gemm_gs's stream (`kind` "q4": q8=5; "q6": q8=4, qb=16; ns 3, ks 32), head h's at w + sel[h] bytes (`sel` int32 [16]; < 0:
  skipped, not written). Column n of head h at y[yoff + h NP + n], or with `split` (one head; a multiple of 16) columns >= split
  at y2[n - split]. A task a run of (head, group) units; a (group, slice) chunk a DMA, `nbuf` deep; the 6 accumulators of a group
  (3 strips x 2 halves) in LSRAM across the slices, in registers across a strip's 4 sub-blocks. `act`, the package's source:
    pkg     `pkg` by DMA (package h // (heads / act_heads) for head h with act_heads > 1: a row's package shared by its experts)
    norm    built by every task from xs (fp32 [K]) and nw (the norm's weight): rms(xs) nw, before the weights stream (nw staged in
            their buffers); `export`: each task also writes its share of the records to `pkg`
    swiglu  built per head h from xs = the gate | up GEMV's rows (fp32 [2 heads][gu_np]): silu(xs[h]) xs[heads + h]
  `top` (the output head; one head): no logits stored -- each task keeps the top-1 of its columns online (max, its column: the first
  on ties; the sum of exp(logit - max)) and writes them at y + 4 core_id (head_top's partials for head_reduce).
  `kp` > 1: K in kp parts, a unit = (head, group, part) (the units balance over the tasks), part p's partial sums at
  y + p heads NP (the consumer adds them).
  `route` = (NE, S, D_up, S_dn) (the experts' gate | up launch, 16 heads): `sel` is the router's logits [NE] instead -- every task
  picks the top 8 itself (the first index on ties, as moe_route) and makes its slots (gate's e S, then up's D_up + e S); task 0
  also writes down's slots (e S_dn) to y2 and the 8 weights (their softmax) to nw.
  args: y (out), pkg, w, sel, desc (qgemvk_desc(kind, K)), y2, xs, nw (unused ones: any buffer)."""
  CUK, SPU = CU[kind], CU[kind] // 16; NG, NSL = -(-N // CUK), K // 128; NP = qgemvk_np(N); U = heads * NG * kp; AB = NSL * SL_REC; ACC = AB; BUF = AB + 256; SB1 = SLB[kind]; CB = SB1 * cs
  assert NSL % (cs * kp) == 0 and (kp == 1 or not (top or split)); NCH = NSL // cs; NCU = NCH // kp; NSP = NSL // kp
  SCR = BUF + nbuf * CB; plain = act == "plain"; act = "norm" if plain else act; assert act in ("pkg", "norm", "swiglu") and (not export or act == "norm")
  assert SCR + {"pkg": 0, "norm": K * 4, "swiglu": 2 * K * 4}[act] <= 32768 and (act != "norm" or plain or nbuf * CB >= 4 * K), "LSRAM"
  assert not split or (heads == 1 and split % 16 == 0)
  RT = SCR + {"pkg": 0, "norm": K * 4, "swiglu": 2 * K * 4}[act]
  if route:
    NE, S_, D_, S_dn = route; assert heads == 16 and NE % 8 == 0 and RT + NE * 4 + 64 <= 32768
    sel_src = f"""
  DMA_FILL(3, DESC(desc, 3), {RT}, (int)sel); DMA_WAIT(3);
  __global int* SEL = (__global int*)LSB({RT + NE * 4});
  {{ __global float* L = (__global float*)LSB({RT});
    float tv[8]; int id[8];
    for (int k = 0; k < 8; k++) {{         /* the largest left, the first index on ties; taken ones masked in LSRAM (a scalar store) */
      float8 m = *(__global float8*)L; for (int i = 1; i < {NE // 8}; i++) m = VMAX(m, *(__global float8*)(L + 8 * i));
      float b = __builtin_aipu_rpmax_tfp32_pw(m, m, ALL)[0]; int bi = 0;
      for (int e = 0; e < {NE}; e++) if (L[e] == b) {{ bi = e; break; }}
      L[bi] = -3.0e38f; tv[k] = b; id[k] = bi;
    }}
    for (int k = 0; k < 8; k++) {{ SEL[k] = id[k] * {S_}; SEL[8 + k] = {D_} + id[k] * {S_}; }}
    if (core_id == 0) {{
      float8 t = (float8){{tv[0], tv[1], tv[2], tv[3], tv[4], tv[5], tv[6], tv[7]}};
      float8 ex = exp2_f8((t - BC(tv[0])) * BC(1.4426950408889634f)); float8 wv = ex * BC(1.0f / __builtin_aipu_rpadd_tfp32_pw(ex, ex, ALL)[0]);
      *(__global float8*)nw = wv;
      for (int k = 0; k < 8; k++) {{ ((__global int*)y2)[k] = id[k] * {S_dn}; ((__global int*)y2)[8 + k] = -1; }}
    }}
  }}"""
  else: sel_src = "\n  __global int* SEL = sel;"
  per_head = act_heads > 1 or act == "swiglu"
  rec = lambda f: f"""for (int sl = 0; sl < {NSL}; sl++) {{ __global uchar* o = LSB(sl * {SL_REC});
        for (int sb = 0; sb < 4; sb++) {{ int k = sl * 128 + sb * 32; qsub({f(0)}, {f(1)}, {f(2)}, {f(3)}, o, sb); }}
        qfin(o); }}"""
  F8 = lambda p, i: f"*(__global float8*)({p} + k + {8 * i})"
  prefetch = f"""
  for (int c = 0; c < {nbuf - 1} && c < nc; c++) {{
    int uu = u0 + c / {NCU}; int off = SEL[uu / {kp * NG}];
    if (off >= 0) DMA_FILL(c, DESC(desc, 0), {BUF} + c * {CB}, (int)(w + off + (uu / {kp} % {NG}) * {NSL * SB1} + ((uu % {kp}) * {NCU} + c % {NCU}) * {CB}));
  }}"""
  if act == "norm":
    exp_ = f"""
    for (int sl = core_id * {NSL} / {nt}; sl < (core_id + 1) * {NSL} / {nt}; sl++)
      for (int i = 0; i < {SL_REC // 32}; i++) ((__global float8*)(pkg + sl * {SL_REC}))[i] = ((__global float8*)LSB(sl * {SL_REC}))[i];""" if export else ""
    produce0 = (f"""
  {{ DMA_FILL(0, DESC(desc, 2), {SCR}, (int)xs); DMA_WAIT(0); __global float* X = (__global float*)LSB({SCR});
    {rec(lambda i: F8('X', i))}{exp_}
  }}""" if plain else f"""
  {{ DMA_FILL(0, DESC(desc, 2), {SCR}, (int)xs); DMA_FILL(1, DESC(desc, 2), {BUF}, (int)nw); DMA_WAIT(0); DMA_WAIT(1);
    __global float* X = (__global float*)LSB({SCR}); __global float* NW = (__global float*)LSB({BUF});
    float8 s8 = BC(0.0f); for (int i = 0; i < {K}; i += 8) {{ float8 v = *(__global float8*)(X + i); s8 += v * v; }}
    float8 iv = BC(1.0f / __builtin_sqrtf(hsum8(s8) * {1.0 / K!r}f + {eps!r}f));
    {rec(lambda i: f"{F8('X', i)} * {F8('NW', i)} * iv")}{exp_}
  }}""") + prefetch
    load_act = ""
  elif act == "swiglu":
    produce0 = prefetch
    load_act = f"""if (h != hact) {{ hact = h;
      DMA_FILL(3, DESC(desc, 2), {SCR}, (int)(xs + h * {gu_np})); DMA_WAIT(3); DMA_FILL(3, DESC(desc, 2), {SCR + 4 * K}, (int)(xs + ({heads} + h) * {gu_np})); DMA_WAIT(3);
      __global float* G = (__global float*)LSB({SCR}); __global float* Uu = G + {K};
      {rec(lambda i: f"silu8({F8('G', i)}) * {F8('Uu', i)}")}
    }}"""
  else:
    produce0 = prefetch
    hsel = f"(h / {heads // act_heads})" if per_head else "0"                     # package h // (heads / act_heads): act_heads rows, heads / act_heads experts each
    load_act = f"""if ({hsel} != hact) {{ hact = {hsel}; DMA_FILL(3, DESC(desc, 1), 0, (int)(pkg + hact * {AB})); DMA_WAIT(3); }}"""
  def zp(e, hi): return f"__builtin_aipu_cvtue_tfp32_tfp16(__builtin_aipu_zip{'h' if hi else 'l'}_tfp16_tfp16({e}, {e}))"
  xs_ = ["x"] + (["z"] if LEV == 2 else [])
  regs = "" if kind == "q6n" else "".join(f"""
        int8v {v}q = *(__global int8v*)(A + {128 * li} + sb * 32);
        char32 {", ".join(f"{v}{t} = (char32)__builtin_aipu_replic_tw_i({v}q, {t})" for t in range(8))};
        float8 {v.upper()}S = BC(((__global float*)(A + {O_SX + 16 * li}))[sb]);""" for li, v in enumerate(xs_))
  if kind == "q4":
    def bt(q): return f"*(__global half16*)(T + {384 + 32 * q} + s * 128)"
    def pair(q0, q1, acc):
      return (f"{{ half16 b0 = {bt(q0)}, b1 = {bt(q1)};\n"
              f"          float8 e0 = __builtin_aipu_cvtue_tfp32_tfp16(b0) * SMe + __builtin_aipu_cvtuo_tfp32_tfp16(b0) * SMo;\n"
              f"          float8 e1 = __builtin_aipu_cvtue_tfp32_tfp16(b1) * SMe + __builtin_aipu_cvtuo_tfp32_tfp16(b1) * SMo;\n"
              f"          {acc} += __builtin_aipu_exte_tfp32_tfp32(e0, e1) + __builtin_aipu_exto_tfp32_tfp32(e0, e1); }}")
    def chain(gi, v):
      ops = []
      for r in range(4): ops += [(f"l{gi}{r}", f"{v}{2 * r}"), (f"h{gi}{r}", f"{v}{2 * r + 1}")]
      e = f"QD({ops[0][0]}, {ops[0][1]})"
      for u, x in ops[1:]: e = f"QA({e}, {u}, {x})"
      return e
    pre = f"""
      {pair(0, 1, "acc0")}
      {pair(2, 3, "acc1")}"""
    nib = "".join(f"""
        uchar32 l{gi}{r} = C[{2 * r + gi}] & (uchar)15, h{gi}{r} = C[{2 * r + gi}] >> (uchar)4;""" for gi in range(2) for r in range(4))
    terms = lambda gi: " + ".join(f"VF({chain(gi, v)}) * (a{gi} * {v.upper()}S)" for v in xs_)
    body = f"""
        half16 a = *(__global half16*)(T + s * 128 + sb * 32);
        __global uchar32* C = (__global uchar32*)(W + s * 1024 + sb * 256);{nib}
        float8 a0 = {zp("a", 0)}, a1 = {zp("a", 1)};
        acc0 += {terms(0)};
        acc1 += {terms(1)};"""
    tables = "__global uchar* T = W + 3072;"
  elif kind == "q8":
    # Q8_0 (gemma4's drafter): the code bytes u = q + 128 as the q6 stream lays them, one fp16 scale a sub-block ([s][sb][16 cols]):
    # y += d (sum_k u q - 128 sum_k q) sx over the sub-block's 8 quads (the package's two 16-block sums)
    def chain(gi, v):
      e = f"QD(C[{gi}], {v}0)"
      for t in range(1, 8): e = f"QA({e}, C[{2 * t + gi}], {v}{t})"
      return e
    pre = ""
    corr = "".join(f"""
        int8v {v}S = BCI(((__global int*)(A + {O_S16 + 32 * li}))[2 * sb] + ((__global int*)(A + {O_S16 + 32 * li}))[2 * sb + 1]);""" for li, v in enumerate(xs_))
    terms = lambda gi: " + ".join(f"VF({chain(gi, v)} - {v}S) * (d{gi} * {v.upper()}S)" for v in xs_)
    body = f"""
        half16 dh = *(__global half16*)(T + s * 128 + sb * 32); float8 d0 = {zp("dh", 0)}, d1 = {zp("dh", 1)};
        __global uchar32* C = (__global uchar32*)(W + s * 2048 + sb * 512);{corr}
        acc0 += {terms(0)};
        acc1 += {terms(1)};"""
    tables = "__global uchar* T = W + 6144;"
  elif kind == "q4n":
    # a strip's slice: codes as the Q4_K body's (one strip: s = 0), then the sub-block scales sc / m as bytes read by one qdot each
    # against the sub-block's unit-byte pattern, times the strip's d / -dmin: a = d sc, b = -dmin m; y += a (sum u q) sx + b (sum x)
    pre = f"""
      half16 dh = *(__global half16*)(W + 1152), mh = *(__global half16*)(W + 1184);
      float8 D0 = {zp("dh", 0)}, D1 = {zp("dh", 1)}, M0 = {zp("mh", 0)}, M1 = {zp("mh", 1)};"""
    def chain(gi, v):
      ops = []
      for r in range(4): ops += [(f"l{gi}{r}", f"{v}{2 * r}"), (f"h{gi}{r}", f"{v}{2 * r + 1}")]
      e = f"QD({ops[0][0]}, {ops[0][1]})"
      for u_, x_ in ops[1:]: e = f"QA({e}, {u_}, {x_})"
      return e
    nib = "".join(f"""
        uchar32 l{gi}{r} = C[{2 * r + gi}] & (uchar)15, h{gi}{r} = C[{2 * r + gi}] >> (uchar)4;""" for gi in range(2) for r in range(4))
    terms = lambda gi: " + ".join(f"VF({chain(gi, v)}) * (a{gi} * {v.upper()}S)" for v in xs_) + f" + b{gi} * SMs"
    body = f"""
        __global uchar32* C = (__global uchar32*)(W + sb * 256); __global uchar32* SCW = (__global uchar32*)(W + 1024); __global uchar32* MW = (__global uchar32*)(W + 1088);
        int8v E = BCI(1 << (8 * sb)); float8 SMs = BC(((__global float*)(A + {O_SUM}))[sb]);
        float8 a0 = VF(QD(SCW[0], (char32)E)) * D0, a1 = VF(QD(SCW[1], (char32)E)) * D1, b0 = VF(QD(MW[0], (char32)E)) * M0, b1 = VF(QD(MW[1], (char32)E)) * M1;{nib}
        acc0 += {terms(0)};
        acc1 += {terms(1)};"""
    tables = ""
  elif kind == "q5n":
    # q4n's body over a group's 3 strips with a 1-bit high plane merged into the nibbles (codes 0..31): a = d sc, b = -dmin m per
    # sub-block by one qdot each against the sub-block's unit-byte pattern; y += a (sum u q) sx + b (sum x)
    pre = f"""
      half16 dh = *(__global half16*)(T + s * 32), mh = *(__global half16*)(TM + s * 32);
      float8 D0 = {zp("dh", 0)}, D1 = {zp("dh", 1)}, M0 = {zp("mh", 0)}, M1 = {zp("mh", 1)};"""
    def chain(gi, v):
      ops = []
      for r in range(4): ops += [(f"l{gi}{r}", f"{v}{2 * r}"), (f"h{gi}{r}", f"{v}{2 * r + 1}")]
      e = f"QD({ops[0][0]}, {ops[0][1]})"
      for u_, x_ in ops[1:]: e = f"QA({e}, {u_}, {x_})"
      return e
    def hsh(gi, q): return f"(w{gi} << (uchar){4 - q})" if q < 4 else f"w{gi}" if q == 4 else f"(w{gi} >> (uchar){q - 4})"
    nib = "".join(f"""
        uchar32 w{gi} = HW[{gi}];""" + "".join(f"""
        uchar32 l{gi}{r} = (C[{2 * r + gi}] & (uchar)15) | ({hsh(gi, 2 * r)} & (uchar)0x10), h{gi}{r} = (C[{2 * r + gi}] >> (uchar)4) | ({hsh(gi, 2 * r + 1)} & (uchar)0x10);""" for r in range(4)) for gi in range(2))
    terms = lambda gi: " + ".join(f"VF({chain(gi, v)}) * (a{gi} * {v.upper()}S)" for v in xs_) + f" + b{gi} * SMs"
    body = f"""
        __global uchar32* C = (__global uchar32*)(W + s * 1024 + sb * 256); __global uchar32* HW = (__global uchar32*)(Hh + s * 256 + sb * 64);
        __global uchar32* SCW = (__global uchar32*)(SC + s * 64); __global uchar32* MW = (__global uchar32*)(MWb + s * 64);
        int8v E = BCI(1 << (8 * sb)); float8 SMs = BC(((__global float*)(A + {O_SUM}))[sb]);
        float8 a0 = VF(QD(SCW[0], (char32)E)) * D0, a1 = VF(QD(SCW[1], (char32)E)) * D1, b0 = VF(QD(MW[0], (char32)E)) * M0, b1 = VF(QD(MW[1], (char32)E)) * M1;{nib}
        acc0 += {terms(0)};
        acc1 += {terms(1)};"""
    tables = "__global uchar* Hh = W + 3072; __global uchar* SC = W + 3840; __global uchar* MWb = W + 4032; __global uchar* T = W + 4224; __global uchar* TM = W + 4320;"
  elif kind == "mx4":
    # MXFP4: the e2m1 nibble -> u = 16 + kv[nibble] by arithmetic (m = the magnitude bits: m, 2m - 4 or 4m - 16 by m's bits 2 and 1;
    # the sign bit negates), then the q8 body: y += d (sum u q - 16 sum q) sx a sub-block (the halved E8M0 scale as fp16)
    def chain(gi, v):
      ops = []
      for r in range(4): ops += [(f"l{gi}{r}", f"{v}{2 * r}"), (f"h{gi}{r}", f"{v}{2 * r + 1}")]
      e = f"QD({ops[0][0]}, {ops[0][1]})"
      for u_, x_ in ops[1:]: e = f"QA({e}, {u_}, {x_})"
      return e
    ex = lambda n, d: f"""
        {{ uchar32 m = {n} & (uchar)7, b2 = (m >> (uchar)2) & (uchar)1, b12 = b2 & (m >> (uchar)1) & (uchar)1, sg = {n} >> (uchar)3;
          uchar32 k2 = (uchar32)(((char32)(b2 << (uchar)7)) >> (char)7), k12 = (uchar32)(((char32)(b12 << (uchar)7)) >> (char)7), ks = (uchar32)(((char32)(sg << (uchar)7)) >> (char)7);
          uchar32 v2 = m + ((m - (uchar)4) & k2) + (((m << (uchar)1) - (uchar)12) & k12);
          {d} = (uchar)16 + ((v2 ^ ks) + sg); }}"""
    nib = "".join(f"""
        uchar32 l{gi}{r}, h{gi}{r};{ex(f"(C[{2 * r + gi}] & (uchar)15)", f"l{gi}{r}")}{ex(f"(C[{2 * r + gi}] >> (uchar)4)", f"h{gi}{r}")}""" for gi in range(2) for r in range(4))
    pre = ""
    corr = "".join(f"""
        int8v {v}S = BCI((((__global int*)(A + {O_S16 + 32 * li}))[2 * sb] + ((__global int*)(A + {O_S16 + 32 * li}))[2 * sb + 1]) >> 3);""" for li, v in enumerate(xs_))
    terms = lambda gi: " + ".join(f"VF({chain(gi, v)} - {v}S) * (d{gi} * {v.upper()}S)" for v in xs_)
    body = f"""
        half16 dh = *(__global half16*)(T + s * 128 + sb * 32); float8 d0 = {zp("dh", 0)}, d1 = {zp("dh", 1)};
        __global uchar32* C = (__global uchar32*)(W + s * 1024 + sb * 256);{corr}{nib}
        acc0 += {terms(0)};
        acc1 += {terms(1)};"""
    tables = "__global uchar* T = W + 3072;"
  elif kind == "q51":
    # Q5_1: the q5n nibble + high-bit plane (codes 0..31) with a = d, b = m per sub-block read straight from fp16 tables (no scale
    # decode): y += a (sum u q) sx + b (sum x)
    pre = ""
    def chain(gi, v):
      ops = []
      for r in range(4): ops += [(f"l{gi}{r}", f"{v}{2 * r}"), (f"h{gi}{r}", f"{v}{2 * r + 1}")]
      e = f"QD({ops[0][0]}, {ops[0][1]})"
      for u_, x_ in ops[1:]: e = f"QA({e}, {u_}, {x_})"
      return e
    def hsh(gi, q): return f"(w{gi} << (uchar){4 - q})" if q < 4 else f"w{gi}" if q == 4 else f"(w{gi} >> (uchar){q - 4})"
    nib = "".join(f"""
        uchar32 w{gi} = HW[{gi}];""" + "".join(f"""
        uchar32 l{gi}{r} = (C[{2 * r + gi}] & (uchar)15) | ({hsh(gi, 2 * r)} & (uchar)0x10), h{gi}{r} = (C[{2 * r + gi}] >> (uchar)4) | ({hsh(gi, 2 * r + 1)} & (uchar)0x10);""" for r in range(4)) for gi in range(2))
    terms = lambda gi: " + ".join(f"VF({chain(gi, v)}) * (a{gi} * {v.upper()}S)" for v in xs_) + f" + b{gi} * SMs"
    body = f"""
        half16 dh = *(__global half16*)(T + s * 128 + sb * 32), mh = *(__global half16*)(TM + s * 128 + sb * 32);
        float8 a0 = {zp("dh", 0)}, a1 = {zp("dh", 1)}, b0 = {zp("mh", 0)}, b1 = {zp("mh", 1)};
        __global uchar32* C = (__global uchar32*)(W + s * 1024 + sb * 256); __global uchar32* HW = (__global uchar32*)(Hh + s * 256 + sb * 64);
        float8 SMs = BC(((__global float*)(A + {O_SUM}))[sb]);{nib}
        acc0 += {terms(0)};
        acc1 += {terms(1)};"""
    tables = "__global uchar* Hh = W + 3072; __global uchar* T = W + 3840; __global uchar* TM = W + 4224;"
  elif kind == "q6n" and Q6N_MODE == "unrolled":
    # the 32 vector registers: a 16-k block (4 quads) at a time -- its 4 replicas a level, its 2 scales, its 2 correction sums, then
    # each 8-column half's 4 codes unpacked and consumed at once. Still ~56 spills a sub-block (the compiler hoists), but measured
    # the fastest: 18 GB/s at two levels (a runtime loop over the halves with LSRAM accumulators: 15.7; the q6 body's order: 17.3)
    pre = f"""
      half16 dh = *(__global half16*)(T + s * 32); float8 D0 = {zp("dh", 0)}, D1 = {zp("dh", 1)};"""
    def chain(gi, b, v):
      e = f"QD(u{gi}0, {v}{4 * b})"
      for t in range(1, 4): e = f"QA({e}, u{gi}{t}, {v}{4 * b + t})"
      return e
    blocks = ""
    for b in range(2):
      blocks += f"""
        {{ {" ".join(f"char32 {v}{4 * b + t} = (char32)__builtin_aipu_replic_tw_i({v}q, {4 * b + t});" for v in xs_ for t in range(4))}
          float8 a{b}0 = (VF(QD(SW[0], (char32)E{b})) - BC(128.0f)) * D0, a{b}1 = (VF(QD(SW[2], (char32)E{b})) - BC(128.0f)) * D1;
          {" ".join(f"int8v {v}S{b} = BCI(((__global int*)(A + {O_S16 + 32 * li}))[2 * sb + {b}] >> 2);" for li, v in enumerate(xs_))}"""
      for gi in range(2):
        blocks += f"""
          {{ uchar32 n0 = C[{4 * b + gi}], n1 = C[{4 * b + 2 + gi}], w = HW[{2 * gi + b}];
            uchar32 u{gi}0 = (n0 & (uchar)15) | ((w << (uchar)4) & (uchar)0x30), u{gi}1 = (n0 >> (uchar)4) | ((w << (uchar)2) & (uchar)0x30);
            uchar32 u{gi}2 = (n1 & (uchar)15) | (w & (uchar)0x30), u{gi}3 = (n1 >> (uchar)4) | ((w >> (uchar)2) & (uchar)0x30);
            acc{gi} += {" + ".join(f"VF({chain(gi, b, v)} - {v}S{b}) * (a{b}{gi} * {v.upper()}S)" for v in xs_)}; }}"""
      blocks += "\n        }"
    regs = "".join(f"""
        int8v {v}q = *(__global int8v*)(A + {128 * li} + sb * 32); float8 {v.upper()}S = BC(((__global float*)(A + {O_SX + 16 * li}))[sb]);""" for li, v in enumerate(xs_))
    body = f"""
        __global uchar32* C = (__global uchar32*)(W + s * 1024 + sb * 256); __global uchar32* HW = (__global uchar32*)(Hh + s * 512 + sb * 128);
        int eb = 1 << (16 * (sb & 1)); int8v E0 = BCI(eb), E1 = BCI(eb << 8); __global uchar32* SW = (__global uchar32*)(SC + s * 128 + (sb >> 1) * 32);{blocks}"""
    tables = "__global uchar* Hh = W + 3072; __global uchar* SC = W + 4608; __global uchar* T = W + 4992;"
  elif kind == "q6n":
    # a runtime loop over the 8 16-k blocks of a sub-block row (NOUNROLL: the compiler cannot hoist across it), ~28 live vectors:
    # the block's 4 replicas a level come from an LSRAM scratch written once a slice ("scratch": loads take offsets, so the block
    # index may be a variable) or by re-pointing the dense activation register at the block ("shift": a 16-byte-offset load, then
    # replic of lanes 0..3); its 2 scales, 2 correction sums, and each half's 4 codes unpacked and consumed at once.
    pre = f"""
      half16 dh = *(__global half16*)(T + s * 32); float8 D0 = {zp("dh", 0)}, D1 = {zp("dh", 1)};"""
    def chain(gi, v):
      e = f"QD(u{gi}0, {v}0)"
      for t in range(1, 4): e = f"QA({e}, u{gi}{t}, {v}{t})"
      return e
    if Q6N_MODE == "scratch":
      reps = "".join(f" char32 {v}{t} = R[{32 * li} + 4 * bk + {t}];" for li, v in enumerate(xs_) for t in range(4))
    else:
      reps = "".join(f" int8v {v}q = *(__global int8v*)(A + {128 * li} + sb * 32 + 16 * b);" for li, v in enumerate(xs_)) + \
             "".join(f" char32 {v}{t} = (char32)__builtin_aipu_replic_tw_i({v}q, {t});" for li, v in enumerate(xs_) for t in range(4))
    halves = ""
    for gi in range(2):
      halves += f"""
          {{ uchar32 n0 = C[{gi}], n1 = C[{2 + gi}], w = HW[{2 * gi}];
            uchar32 u{gi}0 = (n0 & (uchar)15) | ((w << (uchar)4) & (uchar)0x30), u{gi}1 = (n0 >> (uchar)4) | ((w << (uchar)2) & (uchar)0x30);
            uchar32 u{gi}2 = (n1 & (uchar)15) | (w & (uchar)0x30), u{gi}3 = (n1 >> (uchar)4) | ((w >> (uchar)2) & (uchar)0x30);
            acc{gi} += {" + ".join(f"VF({chain(gi, v)} - {v}S) * (a{gi} * {v.upper()}S)" for v in xs_)}; }}"""
    regs = ""
    body = f"""
        int sb = bk >> 1, b = bk & 1;
        __global uchar32* C = (__global uchar32*)(W + s * 1024 + sb * 256 + b * 128); __global uchar32* HW = (__global uchar32*)(Hh + s * 512 + sb * 128 + b * 32);
        int8v E = BCI(1 << ((bk & 3) * 8)); __global uchar32* SW = (__global uchar32*)(SC + s * 128 + (sb >> 1) * 32);
        float8 a0 = (VF(QD(SW[0], (char32)E)) - BC(128.0f)) * D0, a1 = (VF(QD(SW[2], (char32)E)) - BC(128.0f)) * D1;
        {" ".join(f"float8 {v.upper()}S = BC(((__global float*)(A + {O_SX + 16 * li}))[sb]); int8v {v}S = BCI(((__global int*)(A + {O_S16 + 32 * li}))[bk] >> 2);" for li, v in enumerate(xs_))}{reps}{halves}"""
    tables = "__global uchar* Hh = W + 3072; __global uchar* SC = W + 4608; __global uchar* T = W + 4992;"
  else:
    def chain(gi, b, v):
      e = f"QD(C[{8 * b + gi}], {v}{4 * b})"
      for t in range(1, 4): e = f"QA({e}, C[{8 * b + 2 * t + gi}], {v}{4 * b + t})"
      return e
    pre = ""
    corr = "".join(f"""
        int8v {v}S0 = BCI(((__global int*)(A + {O_S16 + 32 * li}))[2 * sb]), {v}S1 = BCI(((__global int*)(A + {O_S16 + 32 * li}))[2 * sb + 1]);""" for li, v in enumerate(xs_))
    terms = lambda gi: " + ".join(f"VF({chain(gi, b, v)} - {v}S{b}) * (d{b}{gi} * {v.upper()}S)" for v in xs_ for b in range(2))
    body = f"""
        half16 d0 = *(__global half16*)(T + s * 256 + sb * 64), d1 = *(__global half16*)(T + s * 256 + sb * 64 + 32);
        __global uchar32* C = (__global uchar32*)(W + s * 2048 + sb * 512);{corr}
        float8 d00 = {zp("d0", 0)}, d01 = {zp("d0", 1)}, d10 = {zp("d1", 0)}, d11 = {zp("d1", 1)};
        acc0 += {terms(0)};
        acc1 += {terms(1)};"""
    tables = "__global uchar* T = W + 6144;"
  if top:
    assert heads == 1 and not split
    pad = f"""
      if (g == {NG - 1}) for (int i = 0; i < {2 * SPU}; i++) for (int j = 0; j < 8; j++) if (g * {CUK} + 8 * i + j >= {N}) v[i][j] = -3.0e38f;""" if N % CUK else ""
    store = f"""
      float8 v[{2 * SPU}]; for (int i = 0; i < {2 * SPU}; i++) v[i] = accL[i];{pad}
      float8 m8 = v[0]; for (int i = 1; i < {2 * SPU}; i++) m8 = VMAX(m8, v[i]); float bm = __builtin_aipu_rpmax_tfp32_pw(m8, m8, ALL)[0];
      if (bm > tm) {{
        int f = 0; for (int i = 0; i < {2 * SPU} && !f; i++) for (int j = 0; j < 8; j++) if (v[i][j] == bm) {{ am = g * {CUK} + 8 * i + j; f = 1; break; }}
        ts = ts * exp2_f8(BC((tm - bm) * 1.4426950408889634f))[0]; tm = bm;
      }}
      float8 e = BC(0.0f); for (int i = 0; i < {2 * SPU}; i++) e += exp2_f8((v[i] - BC(tm)) * BC(1.4426950408889634f));
      ts += __builtin_aipu_rpadd_tfp32_pw(e, e, ALL)[0];"""
  elif split:
    store = f"""
      for (int i = 0; i < {2 * SPU}; i++) {{ int col = g * {CUK} + 8 * i;
        *(__global float8*)(col >= {split} ? y2 + (col - {split}) : y + {yoff} + col) = accL[i]; }}"""
  else:
    store = f"""
      __global float8* yo = (__global float8*)(y + {yoff} + kq * {heads * NP} + h * {NP} + g * {CUK});
      for (int i = 0; i < {2 * SPU}; i++) yo[i] = accL[i];"""
  blockloop = kind == "q6n" and Q6N_MODE != "unrolled"
  sbloop = "NOUNROLL for (int bk = 0; bk < 8; bk++)" if blockloop else "for (int sb = 0; sb < 4; sb++)"
  RS = RT + (NE * 4 + 64 if route else 0); scratch = kind == "q6n" and Q6N_MODE == "scratch"
  assert not scratch or RS + LEV * 1024 <= 32768, ("LSRAM: the replica scratch", RS)
  mkscr = f"""
    __global char32* R = (__global char32*)LSB({RS});
    {" ".join(f"R[{32 * li + 8 * sb_ + t}] = (char32)__builtin_aipu_replic_tw_i(*(__global int8v*)(A + {128 * li} + {32 * sb_}), {t});" for li in range(len(xs_)) for sb_ in range(4) for t in range(8))}""" if scratch else ""
  acc_in, acc_out = "float8 acc0 = st ? accL[2 * s] : BC(0.0f), acc1 = st ? accL[2 * s + 1] : BC(0.0f);", "accL[2 * s] = acc0; accL[2 * s + 1] = acc1;"
  tail = "\n  y[4 * core_id] = tm; y[4 * core_id + 1] = (float)am; y[4 * core_id + 2] = ts; y[4 * core_id + 3] = 0.0f;" if top else ""
  return V.FULL_H + QH + '#define NOUNROLL _Pragma("clang loop unroll(disable)")\n' + f"""
__kernel void qgemv(__global float* restrict y, __global uchar* restrict pkg, __global uchar* restrict w, __global int* restrict sel,
                    __global int* restrict desc, __global float* restrict y2, __global float* restrict xs, __global float* restrict nw, const int core_id) {{
  const int8v Z8 = (int8v){{0, 0, 0, 0, 0, 0, 0, 0}};
  int u0 = core_id * {U} / {nt}, u1 = (core_id + 1) * {U} / {nt};
  int nc = (u1 - u0) * {NCU}; int hact = -1; float tm = -3.0e38f, ts = 0.0f; int am = 0;
  __global float8* accL = (__global float8*)LSB({ACC});{sel_src}{produce0}
  for (int c = 0; c < nc; c++) {{
    int uu = u0 + c / {NCU}; int h = uu / {kp * NG}, g = uu / {kp} % {NG}, kq = uu % {kp}; int sp = kq * {NCU} + c % {NCU}; int bs = c % {nbuf};
    int cn = c + {nbuf - 1};
    if (cn < nc) {{
      int un = u0 + cn / {NCU}; int offn = SEL[un / {kp * NG}];
      if (offn >= 0) DMA_FILL(cn % {nbuf}, DESC(desc, 0), {BUF} + (cn % {nbuf}) * {CB}, (int)(w + offn + (un / {kp} % {NG}) * {NSL * SB1} + ((un % {kp}) * {NCU} + cn % {NCU}) * {CB}));
    }}
    if (SEL[h] < 0) continue;
    {load_act}
    DMA_WAIT(bs);
    for (int si = 0; si < {cs}; si++) {{
    int sl = sp * {cs} + si;
    __global uchar* W = LSB({BUF} + bs * {CB} + si * {SB1}); {tables} __global uchar* A = LSB(sl * {SL_REC});
    float8 SMe = *(__global float8*)(A + {O_SME}), SMo = *(__global float8*)(A + {O_SMO});{mkscr}
    for (int s = 0; s < {SPU}; s++) {{
      int st = sl != kq * {NSP}; {acc_in}{pre}
      {sbloop} {{{regs}{body}
      }}
      {acc_out}
    }}
    if (sl == kq * {NSP} + {NSP - 1}) {{{store}
    }}
    }}
  }}{tail}
  DMA_WAIT_ALL();
}}"""


# ---------------------------------------------------------------- the decode layer's other kernels around the qdot GEMVs
def comb_pkg_desc(NH, NKV, HD, M=1):
  import gemma4_kernels as GK
  HG, BT, P, MR = GK.gattn_cfgr(NH, NKV, HD, M); BLK = GK._ga_block(MR, HG, HD)
  return V._desc_slots((BLK * 4,), (HG * SL_REC,), (HG * HD * 4,))


def comb_pkg_src(NH, NKV, HD, M=1, rows=False):
  """gemma4's gattn_comb (the slices of gattn_part merged per q head) that lays head h's HD = 128 outputs as record h of the o
  projection's package. A unit = one (kv head, head group)'s HG q heads: its P slices' record blocks arrive by P DMAs at once
  (one round trip), the HG records leave by one; `rows`: the o row too (o_rows, as gattn_comb). One row (decode).
  args: pkg, part, desc, o_rows."""
  import gemma4_kernels as GK
  HG, BT, P, MR = GK.gattn_cfgr(NH, NKV, HD, M); g = NH // NKV; NHG = g // HG; NU = NKV * NHG; BLK = GK._ga_block(MR, HG, HD)
  SB = BLK * 4; RB = P * SB; OB = RB + HG * SL_REC; assert HD == 128 and M == 1 and P <= 4 and OB + HG * HD * 4 <= 32768 - 64
  LOG2E = 1.4426950408889634
  return V.FULL_H + GK.EXP + QH + f"""
__kernel void comb_pkg(__global uchar* restrict pkg, __global float* restrict part, __global int* restrict desc, __global float* restrict o_rows, const int core_id) {{
  for (int un = core_id; un < {NU}; un += {NT}) {{
    for (int c = 0; c < {P}; c++) DMA_FILL(c, DESC(desc, 0), c * {SB}, (int)(part + (un + c * {NU}) * {BLK}));
    {" ".join(f"DMA_WAIT({c});" for c in range(P))}
    for (int i = 0; i < {HG}; i++) {{
      float m = -3.0e38f; for (int c = 0; c < {P}; c++) {{ __global float* mm = LSF(c * {SB} + {HG * HD * 4}); if (mm[2 * i + 1] > 0.0f && mm[2 * i] > m) m = mm[2 * i]; }}
      float w[{P}], l = 0.0f;
      for (int c = 0; c < {P}; c++) {{ __global float* mm = LSF(c * {SB} + {HG * HD * 4}); w[c] = mm[2 * i + 1] > 0.0f ? exp2_d4(BC((mm[2 * i] - m) * {LOG2E!r}f))[0] : 0.0f; l += w[c] * mm[2 * i + 1]; }}
      float8 inv = BC(1.0f / l); __global float* ob = LSF({OB} + i * {HD * 4});
      for (int e = 0; e < {HD}; e += 8) {{ float8 o = BC(0.0f); for (int c = 0; c < {P}; c++) o += BC(w[c]) * *(__global float8*)(LSF(c * {SB} + i * {HD * 4}) + e); *(__global float8*)(ob + e) = o * inv; }}
      __global uchar* rec = LSB({RB} + i * {SL_REC});
      for (int sb = 0; sb < 4; sb++) {{ __global float8* v = (__global float8*)(ob + 32 * sb); qsub(v[0], v[1], v[2], v[3], rec, sb); }}
      qfin(rec);
    }}
    int h0 = (un / {NHG}) * {g} + (un % {NHG}) * {HG};
    DMA_DRAIN(0, DESC(desc, 1), {RB}, (int)(pkg + h0 * {SL_REC}));{f" DMA_DRAIN(1, DESC(desc, 2), {OB}, (int)(o_rows + h0 * {HD}));" if rows else ""}
    DMA_WAIT_ALL();
  }}
}}"""


RT_JC, RT_KB = 4, 512                      # router: experts a unit, K a weight block


def router_x1_desc(H, w16=False):
  e = 2 if w16 else 4; return V._desc_slots((H * 4,), (RT_JC * RT_KB * e, RT_KB * e, H * e, RT_KB * e))


def router_x1_src(H, NE, eps, nt=12, kp=1, rs=0, w16=False):
  """x1 = x + r (the o projection's row onto the residual; `kp` > 1: the sum of its kp K-partial rows, `rs` floats apart), rms(x1):
  the router's logits (W_r with the ffn norm's weight folded, fp32 or with `w16` fp16: `lg` [NE] = rms(x1) W_r^T, a unit 4
  experts, units t, t + 12, .. on task t, the weights by double-buffered DMA) and the experts' package (rms(x1) w_ffn, records t,
  t + 12, .. on task t); every task builds x1 whole in LSRAM and writes its slices.
  args: lg (out), x, r, wr, w_ffn, x1 (out), pkg (out), desc (router_x1_desc(H, w16))."""
  e = 2 if w16 else 4
  NSL, NU, NB, BLK = H // 128, NE // RT_JC, H // RT_KB, RT_JC * RT_KB * e; X, R, B0 = 0, H * 4, H * 4; PK = B0 + 2 * BLK
  assert PK + 2 * SL_REC <= 32768 and H % RT_KB == 0
  if w16:
    acc = "".join(f" {{ half16 hw = *(__global half16*)(WB + {j * RT_KB} + i); a{j} += __builtin_aipu_cvtue_tfp32_tfp16(hw) * xe + __builtin_aipu_cvtuo_tfp32_tfp16(hw) * xo; }}" for j in range(RT_JC))
    inner = f"""for (int i = 0; i < {RT_KB}; i += 16) {{ float8 x0 = *(__global float8*)(xr + i), x1_ = *(__global float8*)(xr + i + 8);
        float8 xe = __builtin_aipu_exte_tfp32_tfp32(x0, x1_), xo = __builtin_aipu_exto_tfp32_tfp32(x0, x1_);{acc} }}"""
    wt, wb = "half", "__global half* WB = LSH"
  else:
    acc = "".join(f" a{j} += xv * *(__global float8*)(WB + {j * RT_KB} + i);" for j in range(RT_JC))
    inner = f"for (int i = 0; i < {RT_KB}; i += 8) {{ float8 xv = *(__global float8*)(xr + i);{acc} }}"
    wt, wb = "float", "__global float* WB = LSF"
  more = "".join(f"""
  DMA_FILL(1, DESC(desc, 0), {R}, (int)(r + {p * rs})); DMA_WAIT(1);
  for (int i = 0; i < {H}; i += 8) *(__global float8*)(X1 + i) += *(__global float8*)(RR + i);""" for p in range(1, kp))
  return V.FULL_H + QH + f"""
__kernel void router_x1(__global float* restrict lg, __global float* restrict x, __global float* restrict r, __global {wt}* restrict wr,
                        __global float* restrict wf, __global float* restrict x1, __global uchar* restrict pkg, __global int* restrict desc, const int core_id) {{
  DMA_FILL(0, DESC(desc, 0), {X}, (int)x); DMA_FILL(1, DESC(desc, 0), {R}, (int)r); DMA_WAIT(0); DMA_WAIT(1);
  __global float* X1 = LSF({X}); __global float* RR = LSF({R});
  for (int i = 0; i < {H}; i += 8) *(__global float8*)(X1 + i) += *(__global float8*)(RR + i);{more}
  float8 s8 = BC(0.0f);
  for (int i = 0; i < {H}; i += 8) {{ float8 v = *(__global float8*)(X1 + i); s8 += v * v; }}
  float ivs = 1.0f / __builtin_sqrtf(hsum8(s8) * {1.0 / H!r}f + {eps!r}f); float8 iv = BC(ivs);
  DMA_FILL(0, DESC(desc, 0), {R}, (int)wf); DMA_WAIT(0);
  for (int sl = core_id; sl < {NSL}; sl += {nt}) {{
    for (int i = 0; i < 128; i += 8) *(__global float8*)(x1 + sl * 128 + i) = *(__global float8*)(X1 + sl * 128 + i);
    __global uchar* o = LSB({PK});
    for (int sb = 0; sb < 4; sb++) {{ int k = sl * 128 + sb * 32;
      qsub({", ".join(f"*(__global float8*)(X1 + k + {8 * i}) * *(__global float8*)(RR + k + {8 * i}) * iv" for i in range(4))}, o, sb); }}
    qfin(o);
    for (int i = 0; i < {SL_REC // 32}; i++) ((__global float8*)(pkg + sl * {SL_REC}))[i] = ((__global float8*)o)[i];
  }}
  for (int u = core_id; u < {NU}; u += {nt}) {{
    __global {wt}* wu = wr + u * {RT_JC * H};
    float8 {", ".join(f"a{j} = BC(0.0f)" for j in range(RT_JC))};
    DMA_FILL(0, DESC(desc, 1), {B0}, (int)wu);
    for (int b = 0; b < {NB}; b++) {{
      int s = b & 1;
      if (b + 1 < {NB}) DMA_FILL(s ^ 1, DESC(desc, 1), {B0} + (s ^ 1) * {BLK}, (int)(wu + (b + 1) * {RT_KB}));
      DMA_WAIT(s);
      {wb}({B0} + s * {BLK}); __global float* xr = X1 + b * {RT_KB};
      {inner}
    }}
    {" ".join(f"lg[u * {RT_JC} + {j}] = hsum8(a{j}) * ivs;" for j in range(RT_JC))}
  }}
  DMA_WAIT_ALL();
}}"""


def moe_route16_src(NE, TOPK, S_gu, D_up, S_dn):
  """moe_route (qwen3moe_kernels) with the up projection's 8 slots after the gate's: sel_gu[h] = e_h S_gu, sel_gu[8 + h] = D_up +
  e_h S_gu (up's stack D_up bytes after gate's: one 16-head GEMV launch over both), sel_dn[h] = e_h S_dn. Run as one task."""
  import qwen3moe_kernels as MK
  src = MK.moe_route_src(NE, TOPK, S_gu, S_dn)
  old = f"SG[k] = id[k] * {S_gu}; SD[k] = id[k] * {S_dn}; SG[k + 8] = 0; SD[k + 8] = 0;"
  assert old in src
  return src.replace(old, f"SG[k] = id[k] * {S_gu}; SD[k] = id[k] * {S_dn}; SG[k + 8] = {D_up} + id[k] * {S_gu}; SD[k + 8] = -1;")


MC_CH = int(os.environ.get("QWEN3MOE_MC_CH", "256"))   # moe_comb2: columns a unit (256 on 4 tasks: an 8-task launch costs ~45 us more)


def moe_comb2_desc(H, NP, heads=8): return V._desc_slots((MC_CH * 4,), (heads * MC_CH * 4, MC_CH * 4, NP * 4, MC_CH * 4), (64,))


def moe_comb2_src(H, NP, heads=8, nt=8):
  """out = x1 + sum_h wt[h] y[h] over the down GEMV's plain rows (y [heads][NP]); a unit MC_CH columns, its x1 and y pieces and the
  weights fetched together (one round trip), the sum back by one drain. args: out, x1, y, wt, desc (moe_comb2_desc)."""
  NU = H // MC_CH; XB, YB, WB = 0, MC_CH * 4, MC_CH * 4 * (1 + heads); assert H % MC_CH == 0
  return V.FULL_H + f"""
__kernel void moe_comb2(__global float* restrict out, __global float* restrict x1, __global float* restrict y, __global float* restrict wt,
                        __global int* restrict desc, const int core_id) {{
  __global float* W = LSF({WB});
  for (int u = core_id; u < {NU}; u += {nt}) {{
    DMA_FILL(0, DESC(desc, 0), {XB}, (int)(x1 + u * {MC_CH})); DMA_FILL(1, DESC(desc, 1), {YB}, (int)(y + u * {MC_CH}));
    if (u == core_id) {{ DMA_FILL(2, DESC(desc, 2), {WB}, (int)wt); DMA_WAIT(2); }}
    DMA_WAIT(0); DMA_WAIT(1);
    __global float* X = LSF({XB}); __global float* Y = LSF({YB});
    for (int i = 0; i < {MC_CH}; i += 8) {{
      float8 s = BC(0.0f);
      for (int h = 0; h < {heads}; h++) s += BC(W[h]) * *(__global float8*)(Y + h * {MC_CH} + i);
      *(__global float8*)(X + i) = *(__global float8*)(X + i) + s;
    }}
    DMA_DRAIN(0, DESC(desc, 0), {XB}, (int)(out + u * {MC_CH})); DMA_WAIT(0);
  }}
}}"""


GP_BT = int(os.environ.get("QWEN3MOE_GP_BT", "8"))   # gpart: cache positions a K / V DMA block (up to 10 with the folded kvd)


def gpart_desc(NKV, HD, BT=GP_BT):
  import gemma4_kernels as GK
  HG, _, _, MR = GK.gattn_cfgr(NKV * 8, NKV, HD, 1); return GK.gattn_part_desc(NKV, HD, MR, HG, BT)


def gpart_src(NH, NKV, HD, eps, QR, BT=GP_BT, kv=0):
  """gemma4's gattn_part for one decode row (no window, no ring), the same units, slices and record block (gattn_comb / comb_pkg read
  it unchanged), with BT-position K / V blocks (gattn_cfgr picks 4) and the scores / maxima reduced by rpadd / rpmax instead of
  8-lane scalar chains. args: as gattn_part (desc: gpart_desc).
  `kv` (the row pitch of kv_rows, k heads at 0, v heads at NKV HD): gattn_kvd folded in -- the new position's k (rms x knw, RoPE)
  and v (raw) come from kv_rows, are scored by the kv head's last slice and written to the caches by it; the other slices never see
  that position (its line in the cache is theirs only next token). args + kv_rows, knw."""
  import gemma4_kernels as GK
  HG, _, P, MR = GK.gattn_cfgr(NH, NKV, HD, 1); g = NH // NKV; NHG = g // HG; NU = NKV * NHG * P; LOG2E = 1.4426950408889634; BLK = GK._ga_block(1, HG, HD)
  assert MR == 1
  QS, OS = 0, HG * HD * 4; ML = OS + HG * HD * 4; CSO = OS + BLK * 4; QNO = CSO + HD * 4; KB = QNO + HD * 4; VB = KB + 2 * BT * HD * 4
  KNW = VB + 2 * BT * HD * 4; KR, VR, KN = KNW + HD * 4, KNW + 2 * HD * 4, KNW + 3 * HD * 4; END = KN + HD * 4 if kv else KNW
  assert END <= 32768 - 64 and BT <= 16, (END, BT)
  block = lambda kb, vb, bl: f"""
      for (int h = 0; h < {HG}; h++) {{
        __global float* q = qs + h * {HD}; float8 s8 = BC(-3.0e38f);
        for (int j = 0; j < {bl}; j++) {{
          float8 a = BC(0.0f); __global float* k = {kb} + j * {HD};
          for (int e = 0; e < {HD}; e += 8) a += *(__global float8*)(q + e) * *(__global float8*)(k + e);
          s8[j] = __builtin_aipu_rpadd_tfp32_pw(a, a, ALL)[0];
        }}
        float bm = __builtin_aipu_rpmax_tfp32_pw(s8, s8, ALL)[0];
        __global float* o = os + h * {HD};
        if (bm > mx[h]) {{
          float8 f = exp2_d4(BC((mx[h] - bm) * {LOG2E!r}f)); sm[h] *= f[0];
          for (int e = 0; e < {HD}; e += 8) *(__global float8*)(o + e) *= f;
          mx[h] = bm;
        }}
        float8 pr = exp2_d4((s8 - BC(mx[h])) * BC({LOG2E!r}f)); sm[h] += __builtin_aipu_rpadd_tfp32_pw(pr, pr, ALL)[0];
        for (int j = 0; j < {bl}; j++) {{
          float8 pb = BC(pr[j]); __global float* v = {vb} + j * {HD};
          for (int e = 0; e < {HD}; e += 8) *(__global float8*)(o + e) += pb * *(__global float8*)(v + e);
        }}
      }}"""
  if kv:   # the last slice's new row: fetched with q, normed / rotated, scored last, then into the caches (the drains wait at the end)
    kv_args = ", __global float* restrict kv_rows, __global float* restrict knw"
    kv_pre = f"""
  if (core_id < {NU}) {{ DMA_FILL(2, DESC(desc, {BT + 1}), {KNW}, (int)knw); DMA_WAIT(2); }}"""
    kv_fetch = f"""
    int last = c == {P - 1}; if (last) {{ t1 = pos; DMA_FILL(0, DESC(desc, {BT + 1}), {KR}, (int)(kv_rows + kh * {HD})); DMA_FILL(1, DESC(desc, {BT + 1}), {VR}, (int)(kv_rows + {NKV * HD} + kh * {HD})); }}"""
    kv_wait = " if (last) { DMA_WAIT(0); DMA_WAIT(1); }"
    kv_post = f"""
    if (last) {{
      norm_rope(LSF({KN}), LSF({KR}), LSF({KNW}), LSF({CSO}));
      DMA_DRAIN(0, DESC(desc, {BT + 1}), {KN}, (int)(Kc + (pos * {NKV} + kh) * {HD})); DMA_DRAIN(1, DESC(desc, {BT + 1}), {VR}, (int)(Vc + (pos * {NKV} + kh) * {HD}));
      {block(f"LSF({KN})", f"LSF({VR})", 1)}
    }}"""
    kv_end = " if (last) { DMA_WAIT(0); DMA_WAIT(1); }"
  else: kv_args = kv_pre = kv_fetch = kv_wait = kv_post = kv_end = ""
  return V.FULL_H + GK.EXP + GK._norm_rope(HD, eps) + f"""
__kernel void gpart(__global float* restrict part, __global float* restrict q_rows, __global float* restrict Kc, __global float* restrict Vc,
                    __global float* restrict qnw, __global float* restrict cs, __global int* restrict idx, __global int* restrict desc{kv_args}, const int core_id) {{
  int pos = idx[0];
  __global float* qs = LSF({QS}); __global float* os = LSF({OS}); __global float* ml = LSF({ML}); __global float* qw = LSF({QNO});
  if (core_id < {NU}) {{ DMA_FILL(3, DESC(desc, {BT + 1}), {QNO}, (int)qnw); DMA_WAIT(3); }}{kv_pre}
  for (int u = core_id; u < {NU}; u += {NT}) {{
    int kh = u / {NHG * P}, hg = (u / {P}) % {NHG}, c = u % {P}; int h0 = kh * {g} + hg * {HG};
    int T = pos + 1, t0 = c * T / {P}, t1 = (c + 1) * T / {P};
    DMA_FILL(2, DESC(desc, {BT}), {OS}, (int)(q_rows + h0 * {HD}));
    DMA_FILL(3, DESC(desc, {BT + 1}), {CSO}, (int)(cs + pos * {HD}));{kv_fetch} DMA_WAIT(2); DMA_WAIT(3);{kv_wait}
    for (int h = 0; h < {HG}; h++) norm_rope(qs + h * {HD}, os + h * {HD}, qw, LSF({CSO}));
    float mx[{HG}], sm[{HG}];
    for (int i = 0; i < {HG}; i++) {{ mx[i] = -3.0e38f; sm[i] = 0.0f; }}
    for (int i = 0; i < {HG * HD}; i += 8) *(__global float8*)(os + i) = BC(0.0f);
    #define BLEN(t) ({BT} < t1 - (t) ? {BT} : t1 - (t))
    #define ISSUE(t, buf) {{ int bl_ = BLEN(t); DMA_FILL(buf, DESC(desc, bl_ - 1), {KB} + (buf) * {BT * HD * 4}, (int)(Kc + (t) * {NKV * HD} + kh * {HD})); \\
                            DMA_FILL((buf) + 2, DESC(desc, bl_ - 1), {VB} + (buf) * {BT * HD * 4}, (int)(Vc + (t) * {NKV * HD} + kh * {HD})); }}
    int t = t0, p = 0;
    if (t < t1) ISSUE(t, 0);
    while (t < t1) {{
      int bl = BLEN(t), tn = t + bl;
      if (tn < t1) {{ if (p == 0) ISSUE(tn, 1) else ISSUE(tn, 0) }}
      if (p == 0) {{ DMA_WAIT(0); DMA_WAIT(2); }} else {{ DMA_WAIT(1); DMA_WAIT(3); }}
      __global float* kb = LSF({KB} + p * {BT * HD * 4}); __global float* vb = LSF({VB} + p * {BT * HD * 4});{block("kb", "vb", "bl")}
      t = tn; p ^= 1;
    }}{kv_post}
    for (int i = 0; i < {HG}; i++) {{ ml[2 * i] = mx[i]; ml[2 * i + 1] = sm[i]; }}
    DMA_DRAIN(2, DESC(desc, {BT + 2}), {OS}, (int)(part + (u / {P} + c * {NKV * NHG}) * {BLK})); DMA_WAIT(2);{kv_end}
    #undef ISSUE
    #undef BLEN
  }}
  DMA_WAIT_ALL();
}}"""

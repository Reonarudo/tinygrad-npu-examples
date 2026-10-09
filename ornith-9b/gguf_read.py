"""A minimal GGUF reader (version 3): the metadata, the tensor index, and F32 / F16 / BF16 / Q8_0 / Q4_K / Q6_K tensors from a
memory map (the K-quants dequantised only: Qwen3.5-9B's MTP head comes as Q4_K_M).

Q8_0 stores blocks of 32 weights along ne0 (the input dimension K): an fp16 scale d, then 32 int8 codes q; w = d * q.
`q8(name)` returns (q int8 [N, K], d float32 [N, K / 32]) -- what gemm_fp16.pack_b_group_q8 takes -- and `f32(name)` the
dequantised float32 [N, K] (GGUF lists dims ne0 first: a [ne0=K, ne1=N] tensor is an [N, K] row-major matrix)."""
import struct
import numpy as np

GGML = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34), 12: ("Q4_K", 256, 144), 14: ("Q6_K", 256, 210), 30: ("BF16", 1, 2)}


def _f16(b): return b.copy().view(np.float16)[..., 0].astype(np.float32)

def dequant_q4_k(blk):
  """[nb, 144] Q4_K super-blocks -> float32 [nb, 256], per the GGUF format's Q4_K layout (the values ggml's own dequantisation
  gives): d, dmin, 12 bytes of eight 6-bit (scale, min) pairs, 128 bytes of nibbles; sub-block j of 32: d * sc_j * q - dmin * m_j,
  the low nibbles then the high ones of each 32 bytes."""
  d, dmin, sc, qs = _f16(blk[:, 0:2]), _f16(blk[:, 2:4]), blk[:, 4:16].astype(np.int32), blk[:, 16:144]
  s = np.empty((blk.shape[0], 8), np.int32); m = np.empty_like(s)
  s[:, :4], m[:, :4] = sc[:, 0:4] & 63, sc[:, 4:8] & 63
  s[:, 4:] = (sc[:, 8:12] & 0xF) | ((sc[:, 0:4] >> 6) << 4); m[:, 4:] = (sc[:, 8:12] >> 4) | ((sc[:, 4:8] >> 6) << 4)
  q = qs.reshape(-1, 4, 32); q = np.stack([q & 0xF, q >> 4], 2).reshape(-1, 8, 32).astype(np.float32)   # sub-block 2i: low nibbles of chunk i
  return (d[:, None, None] * s[..., None] * q - dmin[:, None, None] * m[..., None]).reshape(-1, 256)

def dequant_q6_k(blk):
  """[nb, 210] Q6_K super-blocks -> float32 [nb, 256], per the GGUF format's Q6_K layout (the values ggml's own dequantisation
  gives): 128 bytes of low nibbles, 64 of high 2-bit pairs, 16 int8 scales (one per 16 weights), d; w = d * sc * (q - 32)."""
  ql, qh, sc, d = blk[:, 0:128].astype(np.int32), blk[:, 128:192].astype(np.int32), blk[:, 192:208].copy().view(np.int8).astype(np.float32), _f16(blk[:, 208:210])
  out = np.empty((blk.shape[0], 256), np.float32)
  for n in range(2):                                                          # two halves of 128 weights
    L, Hh, S = ql[:, 64 * n:64 * n + 64], qh[:, 32 * n:32 * n + 32], sc[:, 8 * n:8 * n + 8]
    qq = [(L[:, :32] & 0xF) | ((Hh & 3) << 4), (L[:, 32:] & 0xF) | (((Hh >> 2) & 3) << 4),
          (L[:, :32] >> 4) | (((Hh >> 4) & 3) << 4), (L[:, 32:] >> 4) | (((Hh >> 6) & 3) << 4)]
    for i in range(4):                                                        # weights 32i .. 32i + 31 of the half: scales 2i, 2i + 1
      sub = np.repeat(S[:, 2 * i:2 * i + 2], 16, 1)
      out[:, 128 * n + 32 * i:128 * n + 32 * i + 32] = d[:, None] * sub * (qq[i] - 32)
  return out


class GGUF:
  def __init__(self, path):
    self.path = path; self.mm = np.memmap(path, np.uint8, "r"); f = open(path, "rb")
    def rd(fmt): return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
    def rstr(): n = rd("Q"); return f.read(n).decode("utf-8", "replace")
    T = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    def rval(t):
      if t == 8: return rstr()
      if t == 9: et = rd("I"); n = rd("Q"); return [rval(et) for _ in range(n)]
      return rd(T[t])
    assert f.read(4) == b"GGUF"; self.version = rd("I"); nt = rd("Q"); nkv = rd("Q"); assert self.version == 3
    self.meta = {}
    for _ in range(nkv): k = rstr(); self.meta[k] = rval(rd("I"))
    self.tensors = {}
    for _ in range(nt):
      name = rstr(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; ty = rd("I"); off = rd("Q")
      self.tensors[name] = (dims, ty, off)
    align = self.meta.get("general.alignment", 32); self.data = -(-f.tell() // align) * align
  def raw(self, name):
    dims, ty, off = self.tensors[name]; kind, bl, bb = GGML[ty]; n = int(np.prod(dims))
    return kind, dims, self.mm[self.data + off:self.data + off + n // bl * bb]
  def q8(self, name):
    kind, dims, raw = self.raw(name); assert kind == "Q8_0", (name, kind)
    K, N = dims[0], (int(np.prod(dims[1:])) if len(dims) > 1 else 1)
    blk = raw.reshape(N, K // 32, 34)
    d = blk[:, :, :2].copy().view(np.float16)[..., 0].astype(np.float32); q = blk[:, :, 2:].copy().view(np.int8).reshape(N, K)
    return q, d
  def f32(self, name):
    kind, dims, raw = self.raw(name); shape = list(reversed(dims))
    if kind == "F32": return raw.view(np.float32).reshape(shape).copy()
    if kind == "F16": return raw.view(np.float16).reshape(shape).astype(np.float32)
    if kind == "BF16": return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
    if kind in ("Q4_K", "Q6_K"):
      nb = raw.size // GGML[self.tensors[name][1]][2]; bb = raw.reshape(nb, -1)
      return (dequant_q4_k(bb) if kind == "Q4_K" else dequant_q6_k(bb)).reshape(shape)
    q, d = self.q8(name); return (q.astype(np.float32).reshape(q.shape[0], -1, 32) * d[..., None]).reshape(shape)
  def q8_rows(self, name, rows):
    """Rows of a Q8_0 matrix, dequantised (the embedding lookup): float32 [len(rows), K]."""
    kind, dims, raw = self.raw(name); K = dims[0]; blk = raw.reshape(-1, K // 32, 34)[list(rows)]
    d = blk[:, :, :2].copy().view(np.float16)[..., 0].astype(np.float32); q = blk[:, :, 2:].copy().view(np.int8).astype(np.float32)
    return (q * d[..., None]).reshape(len(rows), K)

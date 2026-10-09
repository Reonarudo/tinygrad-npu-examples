"""PrismML Ternary Bonsai 2 27B GGUF: the PTQ1_0 decoder and the Hadamard contract (prism.hadamard.*), on top of ornith-9b's
GGUF reader.

PTQ1_0 (GGUF tensor type 143): 128 weights per 28-byte block, along ne0 (the input dimension K):
  qs[24]  five trits a byte -> elements 0..119
  qh[2]   four trits a byte -> elements 120..127
  d       fp16, at byte 26 (after qs and qh, unlike Q8_0 / TQ1_0 where d leads)
  w = d * t, t in {-1, 0, +1}
A byte holds trits base 3, most significant first, scaled to 256 (q = ceil(v * 256 / 243)); trit n of byte q is
((q * 3^n mod 256) * 3) >> 8, minus 1. The element order is TQ1_0's interleave with the stages generalised to 32/16/8
bytes: the 32-byte stage does not fit 24 bytes, so
  elements  n*16 + m       <- qs[m],      trit n   (n < 5, m < 16)
  elements  80 + n*8 + m   <- qs[16 + m], trit n   (n < 5, m < 8)
  elements  120 + n*2 + h  <- qh[h],      trit n   (n < 4, h < 2)
The Hadamard contract (the GGUF's prism.hadamard.* metadata): every weight W' in prism.hadamard.weight_names was folded offline, W' = W diag(s) Hb with Hb the blockwise
(block_size 1024) normalized Sylvester-Walsh-Hadamard matrix and s the sign vector of W's input width. The runtime feeds W'
with Hb (s * x): the signs first, then the rotation. Hb is symmetric and orthogonal, so
W' Hb (s * x) = W x. The token embedding is stored latent, z = Hb (s * e); after the row lookup the runtime restores
e = s * (Hb z) (the rotation first, then the signs).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ornith-9b"))
import gguf_read                                                               # noqa: E402
from gguf_read import GGUF                                                     # noqa: E402

PTQ1_0, QK, BLOCK_BYTES = 143, 128, 28
gguf_read.GGML[PTQ1_0] = ("PTQ1_0", QK, BLOCK_BYTES)                           # the type's block: 128 weights in 2 + 24 + 2 bytes

_POW3 = np.array([1, 3, 9, 27, 81], np.uint32)
_b = np.arange(256, dtype=np.uint32)[:, None]
TRIT5 = (((((_b * _POW3[None]) & 0xFF) * 3) >> 8).astype(np.int8) - 1)        # [256, 5]: trit n of a qs byte
TRIT4 = np.ascontiguousarray(TRIT5[:, :4])                                     # [256, 4]: trit n of a qh byte


def decode_ptq1_0(blk):
  """[nb, 28] uint8 PTQ1_0 blocks -> (trits int8 [nb, 128] in element order, d float32 [nb])."""
  blk = np.asarray(blk, np.uint8).reshape(-1, BLOCK_BYTES); nb = blk.shape[0]
  t = np.empty((nb, QK), np.int8)
  t[:, 0:80] = TRIT5[blk[:, 0:16]].transpose(0, 2, 1).reshape(nb, 80)           # [nb, m 16, n 5] -> [nb, n, m]
  t[:, 80:120] = TRIT5[blk[:, 16:24]].transpose(0, 2, 1).reshape(nb, 40)
  t[:, 120:128] = TRIT4[blk[:, 24:26]].transpose(0, 2, 1).reshape(nb, 8)
  d = blk[:, 26:28].copy().view(np.float16)[:, 0].astype(np.float32)
  return t, d


def dequant_ptq1_0(blk):
  """[nb, 28] PTQ1_0 blocks -> float32 [nb, 128] (= dequantize_row_ptq1_0: (t) * d, in float32)."""
  t, d = decode_ptq1_0(blk); return t.astype(np.float32) * d[:, None]


def hadamard_matrix(n):
  """The normalized Sylvester-Walsh-Hadamard matrix in natural order: H[r, c] = (-1)^popcount(r & c) / sqrt(n)
  (symmetric, orthogonal, its own inverse)."""
  r = np.arange(n)[:, None] & np.arange(n)[None]
  par = np.zeros_like(r)
  while r.any(): par ^= r & 1; r = r >> 1
  return np.where(par == 1, -1.0, 1.0).astype(np.float32) / np.float32(np.sqrt(n))


class Hadamard:
  """The prism.hadamard.* contract of a GGUF: `latent(x)` maps an activation to the folded weights' basis (Hb (s * x)), `primal(z)`
  undoes it for the latent embedding (s * (Hb z)). Both act on the last axis, blockwise; x may be any [..., K]."""
  def __init__(self, meta):
    self.version = int(meta["prism.hadamard.version"]); assert self.version == 1, self.version   # v2 (tied output) not handled
    self.block = int(meta["prism.hadamard.block_size"])
    assert meta["prism.hadamard.transform"] == "normalized-sylvester-walsh-hadamard"
    assert meta["prism.hadamard.axis"] == "input-last-dimension"
    self.sign_mode = meta["prism.hadamard.sign_mode"]
    self.weights = set(meta["prism.hadamard.weight_names"]); self.inverse = set(meta.get("prism.hadamard.inverse_weight_names", []))
    self.gdn_v_grouped = bool(meta.get("prism.hadamard.gdn_v_grouped", False))
    self.signs = {}
    if self.sign_mode == "explicit":                                            # one +-1 vector per input width, concatenated
      vals, off = np.asarray(meta["prism.hadamard.sign_values"], np.float32), 0
      for w in meta["prism.hadamard.sign_widths"]: self.signs[int(w)] = vals[off:off + w]; off += w
      assert off == vals.size and np.all(np.abs(vals) == 1)
    self.H = hadamard_matrix(self.block)
  def rotate(self, x):
    x = np.asarray(x, np.float32); K = x.shape[-1]; assert K % self.block == 0, K
    return (x.reshape(-1, K // self.block, self.block) @ self.H).reshape(x.shape)   # H symmetric: x_blk @ H == H x_blk
  def latent(self, x):
    s = self.signs.get(x.shape[-1]) if self.sign_mode == "explicit" else None
    return self.rotate(x * s if s is not None else x)
  def primal(self, z):
    s = self.signs.get(z.shape[-1]) if self.sign_mode == "explicit" else None
    e = self.rotate(z); return e * s if s is not None else e


Q8ACT = os.environ.get("BONSAI_Q8ACT", "0") == "1"

def q8_0_roundtrip(x):
  """x [..., K] through Q8_0 and back: d = amax / 127 per 32, stored fp16; q = round(x / d) (how llama.cpp quantises the activations
  of a Q8_0 matmul)."""
  b = np.asarray(x, np.float32).reshape(-1, 32); d = np.abs(b).max(-1, keepdims=True) / 127.0
  q = np.round(b * np.where(d == 0, 0, 1.0 / np.where(d == 0, 1, d))); d16 = d.astype(np.float16).astype(np.float32)
  return (q * d16).reshape(np.shape(x)).astype(np.float32)


class Bonsai2GGUF(GGUF):
  """The GGUF with PTQ1_0 tensors: `ptq(name)` (trits [N, K], d [N, K / 128]), `linear(name, rows)` the dequantised float32 rows
  (in the stored, latent basis), and `had` the Hadamard contract."""
  def __init__(self, path):
    super().__init__(os.path.expanduser(path))
    self.had = Hadamard(self.meta) if "prism.hadamard.version" in self.meta else None
  def blocks(self, name, rows=None):
    kind, dims, raw = self.raw(name); assert kind == "PTQ1_0", (name, kind)
    K = dims[0]; blk = raw.reshape(-1, K // QK, BLOCK_BYTES)
    return blk if rows is None else blk[rows]
  def ptq(self, name, rows=None):
    blk = self.blocks(name, rows); N, nb = blk.shape[:2]; t, d = decode_ptq1_0(blk.reshape(-1, BLOCK_BYTES))
    return t.reshape(N, nb * QK), d.reshape(N, nb)
  def linear(self, name, rows=None):
    blk = self.blocks(name, rows); N, nb = blk.shape[:2]
    return dequant_ptq1_0(blk.reshape(-1, BLOCK_BYTES)).reshape(N, nb * QK)
  def f32(self, name):
    if self.tensors[name][1] == PTQ1_0: return self.linear(name)
    return super().f32(name)
  def folded(self, name): return self.had is not None and name in self.had.weights
  def matmul(self, x, name, chunk=4096):
    """x [n, K] (primal basis) times the linear `name`: y [n, N] = x W^T, W' fed Hb(s * x) when folded (the contract's forward rule).
    The weight is dequantised `chunk` rows at a time (17408 x 5120 is 357 MB in float32). BONSAI_Q8ACT=1 rounds the (latent)
    activation to Q8_0 first, as PrismML's reference CPU kernel does, to attribute the differences against it."""
    xs = self.had.latent(x) if self.folded(name) else np.asarray(x, np.float32)
    if Q8ACT: xs = q8_0_roundtrip(xs)
    N = self.tensors[name][0][1]; out = np.empty((x.shape[0], N), np.float32)
    for i in range(0, N, chunk):
      w = self.linear(name, slice(i, min(N, i + chunk))); out[:, i:i + w.shape[0]] = xs @ w.T
    return out
  def embed(self, ids, name="token_embd.weight"):
    z = self.linear(name, list(ids))
    return self.had.primal(z) if self.had is not None and name in self.had.inverse else z

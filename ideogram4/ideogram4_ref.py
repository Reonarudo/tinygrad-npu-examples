#!/usr/bin/env python3
"""Ideogram 4's diffusion transformer and its sampling schedule in float32 numpy: the reference every device block is checked
against. No torch.

Written from the checkpoint -- its config.json and its tensors' names and shapes -- and the model's published architecture, with
the conventions its reference implementation in diffusers (`Ideogram4Transformer2DModel` / `Ideogram4Pipeline`) fixes: the
rotary layout, the norms' epsilons, the time embedding, the schedule. That implementation is the numerical reference.

The model. A sequence of tokens of width D = 4608 -- the text rows (ideogram4_text.py) then the image tokens, a 2 x 2 latent patch
(128 channels) each; the unconditional transformer sees the image tokens alone -- goes through 34 blocks, each modulated by the
time t (AdaLN):
    c = silu(W_ada (W_t2 silu(W_t1 sinus(1e4 t) + b) + b) + b)                    the time's conditioning, [512]
    (sa, ga, sf, gf) = W_mod,l c + b                                                block l's four modulation vectors, [D] each
    x += tanh(ga) rms(W_o attn(rope(rms_head(q)), rope(rms_head(k)), v), w_a2)     (q | k | v) = W_qkv (rms(x, w_a1) (1 + sa))
    x += tanh(gf) rms(W_2 (silu(W_1 h) * W_3 h), w_f2)                               h = rms(x, w_f1) (1 + sf)
18 heads of 256, full (non-causal) attention over the whole sequence; then the output layer
    v = W_out (layernorm(x) (1 + W_om silu(c) + b)) + b                             the velocity of the image tokens, [128]
Positions are (t, h, w) triples: image token (i, j) of the grid at (0, i, j) + 65536 on every axis, text token n at (n, n, n). The
rotary embedding turns each head's dim f with dim f + 128 by an angle that grows with one of the three coordinates (rope_tables).

Sampling is flow matching: the latents start as Gaussian noise at noise level sigma = 1 (model time t = 1 - sigma = 0) and an Euler
step moves them by (sigma_next - sigma) x (-v) -- the noise levels from a logit-normal schedule (sigmas).

    W = Ideogram4Weights("~/ideogram4/unconditional_transformer"); M = Model(W)
    v = M.velocity(latents, t, image_positions(16, 16))                 # one branch's velocity at model time t
    python3 ideogram4_ref.py --grid 16x16 --steps 12 --seed 0 --out ~/ideogram4/ref_latents_256.npy   # an unconditional sample's
                                                                        # per-step velocities (ideogram4_fp_backend.py --ref)
A block's cost at T tokens is 2 x 255M x T FLOP: 0.13 TFLOP at T = 256 (a workstation's numpy: ~0.4 s; the board's, without an
optimised BLAS: ~40 s).
"""
import argparse, math, os, sys
import numpy as np

# the shape (config.json; Model checks a checkpoint's against it)
D, NH, HD, FF, C_IN, NL, D_C = 4608, 18, 256, 12288, 128, 34, 512      # width, heads x head dim, MLP, latent channels, blocks, AdaLN
SECTIONS = (24, 20, 20)            # the rotary frequencies of the t, h and w axes (mrope_section)
THETA = 5_000_000                  # rope_theta
EPS = 1e-5                         # every RMSNorm of the blocks (norm_eps)
IMG_POS0 = 65536                   # added to every coordinate of an image token's position
T_MIN, T_MAX = 1.0 / (1.0 + math.exp(9.0)), 1.0 / (1.0 + math.exp(-7.5))   # the schedule's model times: log-SNR within [-15, 18]


def silu(x): return x / (1.0 + np.exp(-x))

def rms(x, w, eps=EPS):
  """RMSNorm over the last axis, in fp32: x / sqrt(mean(x^2) + eps) * w."""
  x = x.astype(np.float32)
  return x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True, dtype=np.float32) + np.float32(eps))) * w.astype(np.float32)


# ---- positions and the rotary embedding
def image_positions(gh, gw):
  """[gh gw, 3] int: the (t, h, w) positions of a gh x gw grid's tokens in row-major order, (0, i, j) + IMG_POS0."""
  i, j = np.divmod(np.arange(gh * gw), gw)
  return np.stack([np.zeros_like(i), i, j], 1) + IMG_POS0

def text_positions(n): return np.repeat(np.arange(n)[:, None], 3, 1)

def rope_tables(pos):
  """(cos, sin) [L, HD] float32 for positions [L, 3]. Frequency f of the HD / 2 (theta^(-2f / HD)) belongs to one axis: the h axis
  when f % 3 == 1 and f < 3 x SECTIONS[1], the w axis when f % 3 == 2 and f < 3 x SECTIONS[2], the t axis otherwise; its angle is
  that coordinate times the frequency. Each table holds the HD / 2 angles' values twice (dims f and f + HD / 2 turn together)."""
  f = np.arange(HD // 2)
  axis = np.where((f % 3 == 1) & (f < 3 * SECTIONS[1]), 1, np.where((f % 3 == 2) & (f < 3 * SECTIONS[2]), 2, 0))
  freq = (1.0 / (THETA ** (np.arange(0, HD, 2, dtype=np.float32) / HD))).astype(np.float32)
  ang = pos.astype(np.float32)[:, axis] * freq
  ang = np.concatenate([ang, ang], 1)
  return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

def rotate(x, cos, sin):
  """x [L, heads, HD] turned by the tables: dims (a, b) = (f, f + HD / 2) -> (a cos - b sin, b cos + a sin)."""
  h = HD // 2; a, b = x[..., :h], x[..., h:]; c, s = cos[:, None, :h], sin[:, None, :h]
  return np.concatenate([a * c - b * s, b * c + a * s], -1)


# ---- the time
def t_features(t, dim=D):
  """The sinusoidal features [dim] of model time t in [0, 1]: sin then cos of 1e4 t x 1e4^(-i / (dim / 2 - 1)), i < dim / 2."""
  n = dim // 2
  freq = np.exp(np.arange(n, dtype=np.float32) * -(math.log(1e4) / (n - 1)))
  a = np.float32(1e4 * t) * freq
  return np.concatenate([np.sin(a), np.cos(a)]).astype(np.float32)


# ---- a block
class Block:
  """Block l's tensors, the linears dequantised to float32 (~1 GB)."""
  def __init__(self, W, l):
    p = f"layers.{l}."; g = lambda n: W.get(p + n)
    self.l = l
    self.w_qkv, self.w_o = W.linear(p + "attention.qkv"), W.linear(p + "attention.o")          # [3 D, D], [D, D]
    self.w_1, self.w_3, self.w_2 = (W.linear(p + "feed_forward." + n) for n in ("w1", "w3", "w2"))   # [FF, D] x 2, [D, FF]
    self.n_q, self.n_k = g("attention.norm_q.weight"), g("attention.norm_k.weight")            # [HD]
    self.n_a1, self.n_a2 = g("attention_norm1.weight"), g("attention_norm2.weight")            # around the attention
    self.n_f1, self.n_f2 = g("ffn_norm1.weight"), g("ffn_norm2.weight")                        # around the MLP
    self.w_mod, self.b_mod = W.linear(p + "adaln_modulation"), g("adaln_modulation.bias")      # [4 D, D_C], [4 D]

  def modulation(self, c):
    """(1 + sa, tanh ga, 1 + sf, tanh gf) [D] each from the time's conditioning c [D_C]."""
    sa, ga, sf, gf = np.split(self.w_mod @ c + self.b_mod, 4)
    return 1.0 + sa, np.tanh(ga), 1.0 + sf, np.tanh(gf)

  def __call__(self, x, rope, c, hid=None):
    """x [L, D] float32 -> the block's output. `rope` = rope_tables(positions); `hid`, a dict, collects the intermediates the
    device's ops are checked against: h, q, k, v (after the norms and the rotation), att, ao, x1 (after the attention), h2, m."""
    ma, ga, mf, gf = self.modulation(c)
    L = x.shape[0]; cos, sin = rope
    h = rms(x, self.n_a1) * ma
    q, k, v = np.moveaxis((h @ self.w_qkv.T).reshape(L, 3, NH, HD), 1, 0)
    q, k = rotate(rms(q, self.n_q), cos, sin), rotate(rms(k, self.n_k), cos, sin)
    att = np.empty((L, NH, HD), np.float32); scale = np.float32(1.0 / math.sqrt(HD))
    for j in range(NH):
      s = (q[:, j] @ k[:, j].T) * scale
      e = np.exp(s - s.max(-1, keepdims=True))
      att[:, j] = (e / e.sum(-1, keepdims=True)) @ v[:, j]
    att = att.reshape(L, D); ao = att @ self.w_o.T
    x1 = x + ga * rms(ao, self.n_a2)
    h2 = rms(x1, self.n_f1) * mf
    m = (silu(h2 @ self.w_1.T) * (h2 @ self.w_3.T)) @ self.w_2.T
    if hid is not None: hid.update(h=h, q=q, k=k, v=v, att=att, ao=ao, x1=x1, h2=h2, m=m)
    return (x1 + gf * rms(m, self.n_f2)).astype(np.float32)


# ---- one transformer
class Model:
  """One of the two transformers (the checkpoint's `transformer`, conditional, or `unconditional_transformer`) on an
  Ideogram4Weights: the tensors outside the blocks loaded here, a block's when `block(l)` is called."""
  def __init__(self, W):
    cfg = W.config
    got = (cfg["num_layers"], cfg["num_attention_heads"], cfg["attention_head_dim"], cfg["intermediate_size"], cfg["in_channels"],
           cfg["adaln_dim"], tuple(cfg["mrope_section"]), cfg["rope_theta"], cfg["norm_eps"])
    assert got == (NL, NH, HD, FF, C_IN, D_C, SECTIONS, THETA, EPS), f"not the shape this module is written for: {got}"
    self.W = W; lin = lambda n: (W.linear(n), W.get(n + ".bias"))
    self.in_w, self.in_b = lin("input_proj")                                  # latents -> width: [D, C_IN]
    self.t1_w, self.t1_b = lin("t_embedding.mlp_in")                          # the time MLP: [D, D] x 2
    self.t2_w, self.t2_b = lin("t_embedding.mlp_out")
    self.c_w, self.c_b = lin("adaln_proj")                                    # -> the conditioning: [D_C, D]
    self.tag = W.get("embed_image_indicator.weight")                          # [2, D]: row 1 added to image tokens, row 0 to text
    self.om_w, self.om_b = lin("final_layer.adaln_modulation")                # the output norm's scale: [D, D_C]
    self.out_w, self.out_b = lin("final_layer.linear")                        # -> the velocity: [C_IN, D]

  def block(self, l): return Block(self.W, l)

  def cond(self, t):
    """The time's conditioning c [D_C] at model time t (0 = noise, 1 = data)."""
    e = silu(self.t1_w @ t_features(t) + self.t1_b)
    return silu(self.c_w @ (self.t2_w @ e + self.t2_b) + self.c_b)

  def embed(self, latents):
    """Image tokens [L, C_IN] -> rows [L, D]: the input projection plus the image tag."""
    return (latents.astype(np.float32) @ self.in_w.T + self.in_b + self.tag[1]).astype(np.float32)

  def out(self, x, c):
    """The image tokens' final rows [L, D] -> their velocity [L, C_IN]: LayerNorm (no affine, eps 1e-6) scaled by 1 + the
    conditioning's modulation, then the output projection."""
    mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return ((x - mu) / np.sqrt(var + 1e-6) * (1.0 + (self.om_w @ silu(c) + self.om_b))) @ self.out_w.T + self.out_b

  def velocity(self, latents, t, pos, text_rows=None, blocks=None, hid=None):
    """The velocity [L, C_IN] of the image tokens `latents` [L, C_IN] at model time t. `pos`: the sequence's positions (the text
    tokens' first when `text_rows` [n, D] lead it). `blocks`: Block objects held by the caller, else each loaded in turn (the
    resident set stays ~1 GB). `hid`: {l: block l's output}."""
    c = self.cond(t); rope = rope_tables(pos)
    x = self.embed(latents)
    if text_rows is not None: x = np.concatenate([text_rows.astype(np.float32), x])
    for l in range(NL):
      x = (blocks[l] if blocks is not None else self.block(l))(x, rope, c)
      if hid is not None: hid[l] = x
    return self.out(x[x.shape[0] - latents.shape[0]:], c)


# ---- the schedule
def normal_quantile(p):
  """The standard normal's inverse CDF, in float64: Newton's method on 0.5 erfc(-x / sqrt 2) = p from x = 0 (the CDF is convex
  left of 0 and concave right of it, so the iterates approach the root from the side of 0, monotonically; once a step is below
  1e-9 the next one lands at the float64 rounding); -inf at 0, +inf at 1. Slow only deep in the tails (the schedule's u are
  multiples of 1 / steps)."""
  def one(p):
    if p <= 0.0: return -math.inf
    if p >= 1.0: return math.inf
    x = 0.0
    for _ in range(10000):
      dx = (0.5 * math.erfc(-x / math.sqrt(2.0)) - p) / (math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi))
      x -= dx
      if abs(dx) < 1e-9 * max(1.0, abs(x)): break
    dx = (0.5 * math.erfc(-x / math.sqrt(2.0)) - p) / (math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi))
    return x - dx
  p = np.asarray(p, np.float64)
  return np.vectorize(one, otypes=[np.float64])(p) if p.ndim else one(float(p))

def shifted_mu(height, width, mu):
  """The schedule's mu at an image size: mu + log(pixels / 512^2) / 2 (larger images spend more steps at high noise)."""
  return mu + 0.5 * math.log(height * width / (512 * 512))

def sigmas(steps, mu, std):
  """The noise levels of a `steps`-step sample, [steps + 1] float64 from ~1 down to 0. The model time at u in [0, 1] is
  1 - sigmoid(mu + std x normal_quantile(u)), clipped to [T_MIN, T_MAX]; step i starts at noise level 1 - time(1 - i / steps),
  and the last step ends at 0."""
  y = mu + std * normal_quantile(np.linspace(0.0, 1.0, steps + 1))
  with np.errstate(over="ignore"): t = np.clip(1.0 - 1.0 / (1.0 + np.exp(-y)), T_MIN, T_MAX)
  return np.append((1.0 - t)[:0:-1], 0.0)

def sample_uncond(M, grid, steps=12, seed=0, mu=0.5, std=1.75, blocks=None, log=print):
  """An unconditional sample (the unconditional transformer alone, no guidance): the latents [gh gw, C_IN] after `steps` Euler
  steps from numpy's RandomState(seed) noise, and the per-step velocities."""
  gh, gw = grid; pos = image_positions(gh, gw)
  sig = sigmas(steps, shifted_mu(gh * 16, gw * 16, mu), std)
  x = np.random.RandomState(seed).randn(gh * gw, C_IN).astype(np.float32); vs = []
  for i in range(steps):
    v = M.velocity(x, 1.0 - sig[i], pos, blocks=blocks)
    x = x + np.float32(sig[i + 1] - sig[i]) * (-v); vs.append(v)
    log("   step %2d/%d sigma %.4f -> %.4f  |x| %.3f  |v| %.3f" % (i + 1, steps, sig[i], sig[i + 1], np.abs(x).mean(), np.abs(v).mean()))
  return x, vs


if __name__ == "__main__":
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  from ideogram4_weights import Ideogram4Weights
  ap = argparse.ArgumentParser(description="an unconditional sample's per-step velocities (ideogram4_fp_backend.py --ref)")
  ap.add_argument("--weights", default="~/ideogram4/unconditional_transformer"); ap.add_argument("--grid", default="16x16")
  ap.add_argument("--steps", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--mu", type=float, default=0.5); ap.add_argument("--std", type=float, default=1.75)
  ap.add_argument("--out", default="~/ideogram4/ref_latents_256.npy", help="the velocities [steps, tokens, 128]")
  ap.add_argument("--hold", action="store_true", help="keep the 34 dequantised blocks in memory (~34 GB) instead of reloading each step")
  a = ap.parse_args()
  M = Model(Ideogram4Weights(a.weights))
  blocks = [M.block(l) for l in range(NL)] if a.hold else None
  x, vs = sample_uncond(M, tuple(int(v) for v in a.grid.split("x")), a.steps, a.seed, a.mu, a.std, blocks)
  out = os.path.expanduser(a.out); np.save(out, np.stack(vs)); np.save(out[:-4] + "_x.npy", x)
  print("   the velocities -> %s, the latents -> %s" % (out, out[:-4] + "_x.npy"))

#!/usr/bin/env python3
"""Gemma 4 (the text decoder of E2B / E4B / 12B; the 26B's MoE block too) in numpy fp32, straight from a GGUF Q8_0 file: the
reference every device layer is checked against.

Written from the architecture as specified by llama.cpp (src/models/gemma4.cpp: the graph; src/llama-model.cpp: the KV
reuse rule; conversion/gemma.py: the GGUF conventions -- norms stored unshifted, `rope_freqs`, the per-layer head dims and
KV heads) and Hugging Face transformers (models/gemma4/modeling_gemma4.py: Gemma4RMSNorm, Gemma4TextAttention,
Gemma4TextRouter / Experts, Gemma4TextDecoderLayer, Gemma4TextModel.project_per_layer_inputs). No code is taken from either.

    M = Model("/mnt/ssd/models/gemma-4/gemma-4-E2B-it-Q8_0.gguf")
    logits, hid = M.forward(ids)                 # [n, 262144] soft-capped, {name: [n, ...]} per layer
    ids_out = M.greedy(ids, 16)                  # plain greedy decoding with a KV cache

A layer (x the residual stream, fp32):
    h = rms(x) w_attn; q = rope(rms_head(W_q h) w_qn); k = rope(rms_head(W_k h) w_kn); v = rms_head(W_v h)
    a = softmax(q k^T (scale 1) + causal / window mask) v;   x = x + rms(W_o a) w_post_attn
    x = x + rms(W_down(gelu_tanh(W_gate h2) * W_up h2)) w_post_ffw,  h2 = rms(x) w_ffn
    E-series: x = x + rms(W_proj(gelu_tanh(W_inp_gate x) * ple_l)) w_post_norm;   then x *= layer_output_scale
"""
import math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ornith-9b"))
from gguf_read import GGUF                                                 # noqa: E402

DEFAULT = os.environ.get("GEMMA_GGUF", "/mnt/ssd/models/gemma-4/gemma-4-E2B-it-Q8_0.gguf")
# GEMMA_REF_GGML=1: round where llama.cpp's CPU backend rounds, to tell its rounding from a modelling difference in a comparison:
# a Q8_0 matrix's input rows quantised to Q8_0 (ggml's quantize_row_q8_0: d = amax / 127 in fp16, q = round(x / d)), a BF16
# matrix's input to bf16, the K / V cache in fp16 (its default cache type). Off: plain fp32 activations (the reference proper).
GGML = os.environ.get("GEMMA_REF_GGML", "0") == "1"

def q8_act(x):
  """x [..., K] through Q8_0 and back (per 32 along K), as ggml quantises a Q8_0 matmul's activations: the codes from the fp32
  d = amax / 127 (rounded half away from zero), the stored scale fp16(d)."""
  b = x.reshape(*x.shape[:-1], -1, 32); d = (np.abs(b).max(-1, keepdims=True) / np.float32(127.0)).astype(np.float32)
  t = b * np.where(d == 0, 0, np.float32(1.0) / np.where(d == 0, 1, d)); q = np.sign(t) * np.floor(np.abs(t) + 0.5)
  return (q * d.astype(np.float16).astype(np.float32)).reshape(x.shape).astype(np.float32)
def bf16_round(x): u = np.ascontiguousarray(x, np.float32).view(np.uint32); return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)

def rms(x, w=None, eps=1e-6):
  """Gemma4RMSNorm over the last axis: x * rsqrt(mean(x^2) + eps) * w (w as stored -- Gemma 4 has no 1 + w); w None: no scale."""
  y = x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps))
  return y if w is None else y * w

def gelu_tanh(x):
  """gelu_pytorch_tanh: 0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))."""
  if GGML:                         # ggml's CPU GELU: a table over fp16 inputs holding fp16 outputs, exact pass-through beyond +-10
    h = x.astype(np.float16).astype(np.float32)
    t = (0.5 * h * (1.0 + np.tanh(np.float32(math.sqrt(2.0 / math.pi)) * (h + np.float32(0.044715) * h * h * h)))).astype(np.float16).astype(np.float32)
    return np.where(x <= -10, 0.0, np.where(x >= 10, x, t)).astype(np.float32)
  return 0.5 * x * (1.0 + np.tanh(np.float32(math.sqrt(2.0 / math.pi)) * (x + np.float32(0.044715) * x * x * x)))


class Config:
  """The model's shape from the GGUF metadata (arch `gemma4`, or `gemma4-assistant` for the MTP drafter)."""
  def __init__(self, g):
    a = g.meta["general.architecture"]; m = lambda k, d=None: g.meta.get(f"{a}.{k}", d)
    self.arch, self.NL, self.H = a, int(m("block_count")), int(m("embedding_length"))
    per = lambda v: list(v) if isinstance(v, list) else [v] * self.NL
    self.FF = [int(v) for v in per(m("feed_forward_length"))]
    self.NH = int(m("attention.head_count")); self.NKV = [int(v) for v in per(m("attention.head_count_kv"))]
    self.SWA = [bool(v) for v in m("attention.sliding_window_pattern")]          # True: a local (sliding-window) layer
    self.W = int(m("attention.sliding_window"))
    self.HD = [int(m("attention.key_length_swa")) if s else int(m("attention.key_length")) for s in self.SWA]
    self.ROT_SWA = int(m("rope.dimension_count_swa"))
    self.THETA, self.THETA_SWA = float(m("rope.freq_base")), float(m("rope.freq_base_swa"))
    self.EPS = float(m("attention.layer_norm_rms_epsilon")); self.CAP = m("final_logit_softcapping")
    self.PLE = int(m("embedding_length_per_layer_input", 0) or 0)
    self.NSHARED = int(m("attention.shared_kv_layers", 0) or 0)
    self.NE, self.NEU = m("expert_count"), m("expert_used_count")
    self.H_OUT = m("embedding_length_out")                                       # the drafter: the target's hidden size
    # KV sharing (SPEC 2b): layers from NKV_FROM on read the cache of the last non-shared layer of their type
    self.NKV_FROM = self.NL - self.NSHARED if a == "gemma4" else 0
    def src(l):
      if l < self.NKV_FROM: return l
      return max(j for j in range(self.NKV_FROM) if self.SWA[j] == self.SWA[l])
    self.KV_SRC = [src(l) for l in range(self.NL)] if a == "gemma4" else None


class Model:
  """The GGUF's tensors (dequantised on demand; the per-layer dense matrices cached when `cache_layers`)."""
  def __init__(self, path=DEFAULT, cache_layers=True):
    self.g = GGUF(os.path.expanduser(path)); self.c = Config(self.g); self.cache_layers = cache_layers; self._w = {}
    self.rope_freqs = self.g.f32("rope_freqs.weight")                           # [256]: 1 (64x), 1e30 (192x): global-layer factors
  def t(self, name):
    if name not in self._w:
      w = self.g.f32(name)
      if not self.cache_layers: return w
      self._w[name] = w
    return self._w[name]
  def has(self, name): return name in self.g.tensors
  def lin(self, x, name):
    """x [n, K] @ W^T, W the GGUF [N, K] matrix (dequantised: Q8_0 codes x their fp16 block scales)."""
    if GGML:
      kind = self.g.raw(name)[0]; x = q8_act(x) if kind == "Q8_0" else bf16_round(x) if kind == "BF16" else x
    return x @ self.t(name).T

  # ---- outside the layers
  def embed(self, ids):
    """The scaled token embedding: rows of token_embd x sqrt(H) (fp32 sqrt, as llama.cpp)."""
    return self.g.q8_rows("token_embd.weight", list(ids)) * np.float32(math.sqrt(self.c.H))
  def ple_inputs(self, ids, x0):
    """The per-layer inputs [n, NL, PLE] (E-series; SPEC 2e): (rms(proj(x0) / sqrt(H)) w + table[tok] sqrt(PLE)) / sqrt(2)."""
    c = self.c; n = len(ids)
    proj = self.lin(x0, "per_layer_model_proj.weight") * np.float32(1.0 / math.sqrt(c.H))
    proj = rms(proj.reshape(n, c.NL, c.PLE), self.t("per_layer_proj_norm.weight"), c.EPS)
    tok = self.g.q8_rows("per_layer_token_embd.weight", list(ids)).reshape(n, c.NL, c.PLE) * np.float32(math.sqrt(c.PLE))
    return (proj + tok) * np.float32(1.0 / math.sqrt(2.0))
  def final_norm(self, x): return rms(x, self.t("output_norm.weight"), self.c.EPS)
  def head(self, h, cap=True):
    """Tied head: h [n, H] @ token_embd^T in row blocks (the [262144, H] f32 matrix is not held), then the soft-cap."""
    q, d = self.g.q8("token_embd.weight"); out = np.empty((h.shape[0], q.shape[0]), np.float32)
    if GGML: h = q8_act(h)
    for i in range(0, q.shape[0], 16384):
      w = (q[i:i + 16384].astype(np.float32).reshape(-1, q.shape[1] // 32, 32) * d[i:i + 16384, :, None]).reshape(-1, q.shape[1])
      out[:, i:i + w.shape[0]] = h @ w.T
    if cap and self.c.CAP: out = np.float32(self.c.CAP) * np.tanh(out * np.float32(1.0 / self.c.CAP))
    return out

  # ---- RoPE (rotate-half pairing, SPEC 2c)
  def rope_cs(self, l, pos):
    """(cos, sin) [n, HD] of layer l at positions `pos`: local -- all dims, theta 1e4 over HD; global -- the proportional table
    (theta 1e6 over HD, frequency i x 1 / rope_freqs[i]: 0 for i >= 64, so those dims pass through)."""
    c = self.c; hd = c.HD[l]
    if c.SWA[l]:
      rot = c.ROT_SWA; inv = 1.0 / (c.THETA_SWA ** (np.arange(0, rot, 2, dtype=np.float64) / rot))
      assert rot == hd, (rot, hd)
    else: inv = (1.0 / (c.THETA ** (np.arange(0, hd, 2, dtype=np.float64) / hd))) / self.rope_freqs.astype(np.float64)
    f = np.asarray(pos, np.float64)[:, None] * inv[None]; emb = np.concatenate([f, f], -1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)
  @staticmethod
  def rope(x, cs):
    """x [n, heads, HD] rotated by cs = (cos, sin) [n, HD]: y = x cos + rotate_half(x) sin."""
    cos, sin = cs[0][:, None], cs[1][:, None]; h = x.shape[-1] // 2
    return x * cos + np.concatenate([-x[..., h:], x[..., :h]], -1) * sin

  # ---- the layer
  def layer(self, l, x, pos, kv, ple=None, hid=None):
    """Layer l over rows x [n, H] at positions `pos` (int array). kv: {owning layer: (K [T, NKV, HD], V [T, NKV, HD],
    positions [T])}, extended in place by the layers that own a cache; KV-shared layers read their source's. `ple` [n, PLE]
    (E-series). `hid`: a dict collecting intermediates (names as llama.cpp's graph callbacks)."""
    c, p = self.c, f"blk.{l}."; n = x.shape[0]; hd, nkv = c.HD[l], c.NKV[l]
    pos = np.asarray(pos)
    h = rms(x, self.t(p + "attn_norm.weight"), c.EPS)
    cs = self.rope_cs(l, pos)
    q = self.lin(h, p + "attn_q.weight").reshape(n, c.NH, hd)
    q = self.rope(rms(q, self.t(p + "attn_q_norm.weight"), c.EPS), cs)
    if c.KV_SRC is None or c.KV_SRC[l] == l:                                    # this layer owns a cache: its k / v
      kraw = self.lin(h, p + "attn_k.weight").reshape(n, nkv, hd)
      vraw = self.lin(h, p + "attn_v.weight").reshape(n, nkv, hd) if self.has(p + "attn_v.weight") else kraw   # V = K (12B / 26B global)
      k = self.rope(rms(kraw, self.t(p + "attn_k_norm.weight"), c.EPS), cs); v = rms(vraw, None, c.EPS)
      if GGML: k, v = k.astype(np.float16).astype(np.float32), v.astype(np.float16).astype(np.float32)
      if l in kv: K0, V0, P0 = kv[l]; kv[l] = (np.concatenate([K0, k]), np.concatenate([V0, v]), np.concatenate([P0, pos]))
      else: kv[l] = (k, v, pos.copy())
    K, V, P = kv[c.KV_SRC[l] if c.KV_SRC is not None else l]
    if hid is not None: hid[f"Qcur_pos-{l}"] = q
    # scores (scale 1), causal + the window for local layers: key t visible from query p iff p - W < t <= p
    vis = P[None, :] <= pos[:, None]
    if c.SWA[l]: vis &= P[None, :] > pos[:, None] - c.W
    mask = np.where(vis, 0.0, -np.inf).astype(np.float32)
    g = c.NH // K.shape[1]; o = np.empty((n, c.NH, hd), np.float32)
    f16 = (lambda t: t.astype(np.float16).astype(np.float32)) if GGML else (lambda t: t)   # ggml: K Q and KQ V with fp16 operands
    for hh in range(c.NH):
      s = f16(q[:, hh]) @ K[:, hh // g].T + mask
      s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True); o[:, hh] = f16(s) @ V[:, hh // g]
    a = rms(self.lin(o.reshape(n, c.NH * hd), p + "attn_output.weight"), self.t(p + "post_attention_norm.weight"), c.EPS)
    x = x + a
    if hid is not None: hid[f"attn_out-{l}"] = x
    h2 = rms(x, self.t(p + "ffn_norm.weight"), c.EPS)
    f = self.lin(gelu_tanh(self.lin(h2, p + "ffn_gate.weight")) * self.lin(h2, p + "ffn_up.weight"), p + "ffn_down.weight")
    if self.has(p + "ffn_gate_inp.weight"): f = rms(f, self.t(p + "post_ffw_norm_1.weight"), c.EPS) + self.moe(l, x)
    x = x + rms(f, self.t(p + "post_ffw_norm.weight"), c.EPS)
    if ple is not None:
      if hid is not None: hid[f"pe_in-{l}"] = x
      e = self.lin(gelu_tanh(self.lin(x, p + "inp_gate.weight")) * ple, p + "proj.weight")
      x = x + rms(e, self.t(p + "post_norm.weight"), c.EPS)
    if self.has(p + "layer_output_scale.weight"): x = x * self.t(p + "layer_output_scale.weight")[0]
    if hid is not None: hid[f"l_out-{l}"] = x
    return x

  def moe(self, l, r):
    """26B-A4B's expert block of the post-attention residual r (SPEC 2f): router on rms(r) x scale / sqrt(H), softmax, top-k,
    renormalised, x per-expert scale; experts gelu_tanh-gated on rms(r) w_pre2; summed, then rms(.) w_post2."""
    c, p = self.c, f"blk.{l}."; n = r.shape[0]
    t = rms(r, None, c.EPS) * self.t(p + "ffn_gate_inp.scale") * np.float32(1.0 / math.sqrt(c.H))
    z = t @ self.t(p + "ffn_gate_inp.weight").T; z = np.exp(z - z.max(-1, keepdims=True)); z /= z.sum(-1, keepdims=True)
    top = np.argsort(-z, -1)[:, :c.NEU]; wt = np.take_along_axis(z, top, -1); wt /= wt.sum(-1, keepdims=True)
    wt = wt * self.t(p + "ffn_down_exps.scale")[top]
    h2 = rms(r, self.t(p + "pre_ffw_norm_2.weight"), c.EPS); out = np.zeros_like(r)
    gu, dn = self.g.f32(p + "ffn_gate_up_exps.weight"), self.g.f32(p + "ffn_down_exps.weight")   # [E, 2 F, H], [E, H, F]
    F = dn.shape[-1]
    for i in range(n):
      for j, e in enumerate(top[i]):
        y = gu[e] @ h2[i]; out[i] += wt[i, j] * (dn[e] @ (gelu_tanh(y[:F]) * y[F:]))
    return rms(out, self.t(p + "post_ffw_norm_2.weight"), c.EPS)

  # ---- whole passes
  def forward(self, ids, pos0=0, kv=None, hid=None):
    """The decoder over `ids` at positions pos0.. -> (soft-capped logits [n, vocab], the post-final-norm hidden [n, H]);
    `kv` carries a previous prefix (extended in place)."""
    kv = {} if kv is None else kv; ids = list(ids); n = len(ids); pos = np.arange(pos0, pos0 + n)
    x = self.embed(ids)
    if hid is not None: hid["inp_scaled"] = x
    ple = self.ple_inputs(ids, x) if self.c.PLE else None
    if hid is not None and ple is not None: hid["inp_per_layer"] = ple
    for l in range(self.c.NL): x = self.layer(l, x, pos, kv, None if ple is None else ple[:, l], hid)
    h = self.final_norm(x)
    if hid is not None: hid["result_norm"] = h
    return self.head(h), h

  def greedy(self, ids, n_new, stop=(1, 106)):
    """Plain greedy decoding with the KV cache: the prompt's argmax, then one token a step (stops after a `stop` id)."""
    kv = {}; logits, _ = self.forward(ids, 0, kv); out = [int(logits[-1].argmax())]; pos = len(ids)
    while len(out) <= n_new and out[-1] not in stop:
      logits, _ = self.forward([out[-1]], pos, kv); pos += 1; out.append(int(logits[-1].argmax()))
    return out


class Drafter(Model):
  """The `gemma4-assistant` MTP head (SPEC 2g) over a target Model's KV caches: a draft step at the fixed position `pos` (the
  last accepted token's) from (token, h): x = pre_projection([embed_target(tok) sqrt(H_t) | h]); 4 q-only layers attending the
  target's last local / global caches; logits = its tied 256-wide embedding (no soft-cap, as llama.cpp's graph);
  h_next = post_projection(final_norm(x))."""
  def __init__(self, path, target):
    super().__init__(path); self.T = target; c = self.c; tc = target.c
    assert c.arch == "gemma4-assistant" and c.H_OUT == tc.H
    self.src = {True: max(j for j in range(tc.NKV_FROM) if tc.SWA[j]), False: max(j for j in range(tc.NKV_FROM) if not tc.SWA[j])}
  def step(self, tok, h, pos, kv_target):
    c = self.c
    x = np.concatenate([self.T.embed([tok]), h.reshape(1, -1)], -1) @ self.t("nextn.pre_projection.weight").T
    P = np.array([pos])
    for l in range(c.NL):
      p = f"blk.{l}."; hd = c.HD[l]; K, V, KP = kv_target[self.src[c.SWA[l]]]
      hh = rms(x, self.t(p + "attn_norm.weight"), c.EPS)
      q = self.rope(rms((hh @ self.t(p + "attn_q.weight").T).reshape(1, c.NH, hd), self.t(p + "attn_q_norm.weight"), c.EPS), self.rope_cs(l, P))
      vis = KP <= pos
      if c.SWA[l]: vis &= KP > pos - c.W
      g = c.NH // K.shape[1]; o = np.empty((1, c.NH, hd), np.float32)
      for i in range(c.NH):
        s = np.where(vis, q[:, i] @ K[:, i // g].T, -np.inf); s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
        o[:, i] = s @ V[:, i // g]
      x = x + rms(o.reshape(1, -1) @ self.t(p + "attn_output.weight").T, self.t(p + "post_attention_norm.weight"), c.EPS)
      h2 = rms(x, self.t(p + "ffn_norm.weight"), c.EPS)
      f = (gelu_tanh(h2 @ self.t(p + "ffn_gate.weight").T) * (h2 @ self.t(p + "ffn_up.weight").T)) @ self.t(p + "ffn_down.weight").T
      x = (x + rms(f, self.t(p + "post_ffw_norm.weight"), c.EPS)) * self.t(p + "layer_output_scale.weight")[0]
    hn = self.final_norm(x)
    return self.head(hn, cap=False)[0], (hn @ self.t("nextn.post_projection.weight").T)[0]


if __name__ == "__main__":
  import argparse, time
  ap = argparse.ArgumentParser(); ap.add_argument("--gguf", default=DEFAULT); ap.add_argument("--n", type=int, default=12)
  ap.add_argument("ids", nargs="*", type=int)
  a = ap.parse_args(); M = Model(a.gguf); t0 = time.perf_counter()
  ids = a.ids or [2, 105, 2364, 107, 3689, 563, 506, 5279, 529, 23613, 236881, 25685, 528, 886, 13315, 236761, 106, 107, 105, 4368, 107]
  print("greedy:", M.greedy(ids, a.n), f"({time.perf_counter() - t0:.1f} s)")

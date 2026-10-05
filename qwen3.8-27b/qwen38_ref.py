#!/usr/bin/env python3
"""Qwen3.8-27B (Qwen3.5 architecture) language-model layers in numpy fp32: the reference every device layer is checked against.

Reads the FP8 checkpoint's per-layer shards (`layers-N.safetensors`: bf16 small tensors, E4M3 linears with `weight_scale_inv`
per 128 x 128 block) and `outside.safetensors` (embeddings, final norm, lm_head) with numpy alone. The math transcribes
transformers' modeling_qwen3_5.py: Gated DeltaNet in its recurrent form (token by token, the exact semantics of the
chunked kernel), full attention with the q/k norms, RoPE and the sigmoid output gate, SwiGLU MLP.

    W = Weights("/mnt/ssd/qwen3.8-27b-fp8")
    x = W.embed(ids)                       # [n, 5120] float32
    for l in range(64): x, _ = layer(W, l, x)
    logits = W.lm_head(W.final_norm(x))
"""
import json, math, os, struct
import numpy as np

# QWEN_MODEL: the Qwen3.5-architecture model these modules run (qwen3.8-27b by default). ornith-9b (Ornith 1.0 9B, a GGUF Q8_0
# checkpoint: ../ornith-9b) has the same layer types at smaller sizes; its weights come through ornith_weights.Weights.
# bonsai2-27b (PrismML Ternary Bonsai 2 27B: Qwen3.8-27B's shapes, PTQ1_0 ternary linears with Hadamard-folded inputs:
# ../bonsai2-27b) comes through bonsai2_weights.Weights, whose `mm` applies each linear (the activation transform included).
MODEL = os.environ.get("QWEN_MODEL", "qwen3.8-27b")
_P = {"qwen3.8-27b": dict(H=5120, INTER=17408, NH=24, NKV=4, NK=16, NV=48, NL=64, Q8=False),
      "ornith-9b":   dict(H=4096, INTER=12288, NH=16, NKV=4, NK=16, NV=32, NL=32, Q8=True),
      "bonsai2-27b": dict(H=5120, INTER=17408, NH=24, NKV=4, NK=16, NV=48, NL=64, Q8=False, HAD=True)}[MODEL]
H, INTER, NH, NKV, HD, VOCAB, EPS, THETA = _P["H"], _P["INTER"], _P["NH"], _P["NKV"], 256, 248320, 1e-6, 1e7
NK, DK, NV, DV, CONV = _P["NK"], 128, _P["NV"], 128, 4       # Gated DeltaNet: key heads x 128, value heads x 128, conv kernel 4
NL = _P["NL"]                                                  # layers
Q8 = _P["Q8"]                                                  # linears as GGUF Q8_0 (int8, a scale per 32) instead of E4M3 128 x 128 blocks
# HAD: every linear's input is rotated (bonsai2-27b: a 1024-block Walsh-Hadamard transform with per-width signs, folded into the
# stored weights), so the device's A producers apply the transform (qwen38_kernels.had_a32_src) and the head is ternary
HAD = _P.get("HAD", False)
HEAD = "lm_head_q8" if Q8 else "lm_head_tern" if HAD else "lm_head_fp8"   # the packed block-scaled head's files
LAYER_TYPES = ["full" if (l + 1) % 4 == 0 else "linear" for l in range(NL)]

# E4M3 (fn): 1 sign, 4 exponent (bias 7), 3 mantissa; 0x7F / 0xFF are NaN, no infinities
_E4M3 = np.zeros(256, np.float32)
for _c in range(256):
  _s, _e, _m = -1.0 if _c & 0x80 else 1.0, (_c >> 3) & 0xF, _c & 7
  _E4M3[_c] = np.nan if (_c & 0x7F) == 0x7F else _s * (_m / 8.0 * 2.0 ** -6 if _e == 0 else (1 + _m / 8.0) * 2.0 ** (_e - 7))

def bf16_to_f32(u16: np.ndarray) -> np.ndarray: return (u16.astype(np.uint32) << 16).view(np.float32)

class SafeFile:
  """One safetensors file: the header, and tensors read on demand (memory-mapped)."""
  def __init__(self, path):
    self.path = os.path.expanduser(path)
    with open(self.path, "rb") as f:
      n = struct.unpack("<Q", f.read(8))[0]; self.h = json.loads(f.read(n)); self.base = 8 + n
    self.h.pop("__metadata__", None); self.mm = np.memmap(self.path, np.uint8, "r")
  def raw(self, name):
    e = self.h[name]; a, b = e["data_offsets"]; return e["dtype"], e["shape"], self.mm[self.base + a:self.base + b]
  def f32(self, name):
    dt, shape, raw = self.raw(name)
    if dt == "BF16": return bf16_to_f32(np.frombuffer(raw, np.uint16)).reshape(shape)
    if dt == "F32": return np.frombuffer(raw, np.float32).reshape(shape).copy()
    raise ValueError(f"{name}: {dt}")
  def codes(self, name):
    dt, shape, raw = self.raw(name); assert dt == "F8_E4M3", (name, dt)
    return np.frombuffer(raw, np.uint8).reshape(shape)

class Weights:
  def __init__(self, folder):
    self.folder = os.path.expanduser(folder); self.files = {}
    self.cfg = json.load(open(os.path.join(self.folder, "config.json")))["text_config"]
    assert self.cfg["hidden_size"] == H and self.cfg["num_hidden_layers"] == 64
  def file(self, name):
    if name not in self.files: self.files[name] = SafeFile(os.path.join(self.folder, name))
    return self.files[name]
  def shard(self, l): return self.file(f"layers-{l}.safetensors")
  def linear(self, sf, name):
    """E4M3 codes x the 128 x 128 block scale -> float32 [N, K] (exactly the checkpoint's dequantisation)."""
    c = sf.codes(name + ".weight"); s = sf.f32(name + ".weight_scale_inv"); N, K = c.shape
    assert s.shape == ((N + 127) // 128, (K + 127) // 128), (name, c.shape, s.shape)
    return _E4M3[c] * np.repeat(np.repeat(s, 128, 0), 128, 1)[:N, :K]
  def embed(self, ids):
    sf = self.file("outside.safetensors"); dt, shape, raw = sf.raw("model.language_model.embed_tokens.weight"); row = shape[1] * 2
    return np.stack([bf16_to_f32(np.frombuffer(raw[t * row:(t + 1) * row], np.uint16)) for t in ids])
  def final_norm(self, x): return rms(x, self.file("outside.safetensors").f32("model.language_model.norm.weight"))
  def lm_head(self, x):
    sf = self.file("outside.safetensors"); dt, shape, raw = sf.raw("lm_head.weight"); out = np.empty((x.shape[0], shape[0]), np.float32)
    for i in range(0, shape[0], 8192):   # 2.5 GB of bf16: in row blocks
      w = bf16_to_f32(np.frombuffer(raw[i * shape[1] * 2:(i + 8192) * shape[1] * 2], np.uint16)).reshape(-1, shape[1])
      out[:, i:i + w.shape[0]] = x @ w.T
    return out

def mm(W, sf, x, name):
  """x [n, K] times the linear `name` ([N, K]) -> [n, N]. A Weights object with an `mm` applies the linear itself (bonsai2: the
  Hadamard transform of x and the PTQ1_0 weight streamed in row blocks); else the dequantised float32 matrix."""
  return W.mm(sf, x, name) if hasattr(W, "mm") else x @ W.linear(sf, name).T

def rms(x, w, eps=EPS): return x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * (1.0 + w)   # Qwen3_5RMSNorm: (1 + weight)
def silu(x): return x / (1.0 + np.exp(-x))
def softplus(x): return np.logaddexp(0.0, x)
def l2norm(x, eps=1e-6): return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)   # transformers' l2norm (checked below)

ROT = int(HD * 0.25)   # partial_rotary_factor 0.25: the first 64 of the 256 head dims rotate, 32 frequencies over dim 64

def rope(x, pos):
  """x [n, heads, HD]: RoPE (theta 1e7) on the first ROT dims, the rest pass through. Text only: the three mrope axes carry the same
  positions, so the interleaved recomposition is the identity."""
  inv = 1.0 / (THETA ** (np.arange(0, ROT, 2, dtype=np.float64) / ROT)); f = pos[:, None].astype(np.float64) * inv[None]
  emb = np.concatenate([f, f], -1); cos, sin = np.cos(emb).astype(np.float32)[:, None], np.sin(emb).astype(np.float32)[:, None]
  xr = x[..., :ROT]; rot = np.concatenate([-xr[..., ROT // 2:], xr[..., :ROT // 2]], -1)
  return np.concatenate([xr * cos + rot * sin, x[..., ROT:]], -1)

def attention_layer(W, l, x, pos=None):
  sf, p = W.shard(l), f"model.language_model.layers.{l}."; n = x.shape[0]; pos = np.arange(n) if pos is None else pos
  h = rms(x, sf.f32(p + "input_layernorm.weight"))
  qg = mm(W, sf, h, p + "self_attn.q_proj").reshape(n, NH, 2 * HD)                  # [q | gate] per head
  q, gate = qg[..., :HD], qg[..., HD:]
  k = mm(W, sf, h, p + "self_attn.k_proj").reshape(n, NKV, HD); v = mm(W, sf, h, p + "self_attn.v_proj").reshape(n, NKV, HD)
  q = rms(q, sf.f32(p + "self_attn.q_norm.weight")); k = rms(k, sf.f32(p + "self_attn.k_norm.weight"))
  q, k = rope(q, pos), rope(k, pos)
  causal = np.triu(np.full((n, n), -np.inf, np.float32), 1); o = np.empty((n, NH, HD), np.float32); g = NH // NKV
  for hh in range(NH):
    a = (q[:, hh] @ k[:, hh // g].T) / np.float32(math.sqrt(HD)) + causal
    a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True); o[:, hh] = a @ v[:, hh // g]
  o = o * (1.0 / (1.0 + np.exp(-gate)))
  x = x + mm(W, sf, o.reshape(n, NH * HD), p + "self_attn.o_proj")
  return x + mlp(W, sf, p, rms(x, sf.f32(p + "post_attention_layernorm.weight")))

def mlp(W, sf, p, h):
  return mm(W, sf, silu(mm(W, sf, h, p + "mlp.gate_proj")) * mm(W, sf, h, p + "mlp.up_proj"), p + "mlp.down_proj")

def gdn_layer(W, l, x, state=None, conv_state=None):
  """Gated DeltaNet layer l over the tokens of x, in the recurrent form. `state` [NV, DK, DV] and `conv_state` [CONV-1, 2*NK*DK+NV*DV]
  carry a previous prefix (decode); returns (x_out, (state, conv_state))."""
  sf, p = W.shard(l), f"model.language_model.layers.{l}.linear_attn."; n = x.shape[0]
  h = rms(x, sf.f32(f"model.language_model.layers.{l}.input_layernorm.weight"))
  qkv = mm(W, sf, h, p + "in_proj_qkv")                                             # [n, 2*NK*DK + NV*DV] = [n, 10240]
  z = mm(W, sf, h, p + "in_proj_z").reshape(n, NV, DV)
  a = h @ sf.f32(p + "in_proj_a.weight").T; b = h @ sf.f32(p + "in_proj_b.weight").T    # [n, NV] each, bf16 weights
  # causal depthwise conv over the sequence (kernel CONV, the last CONV-1 inputs of the prefix in conv_state), then SiLU
  cw = sf.f32(p + "conv1d.weight")[:, 0, :]                                         # [10240, CONV]
  prev = np.zeros((CONV - 1, qkv.shape[1]), np.float32) if conv_state is None else conv_state
  ext = np.concatenate([prev, qkv], 0); conv = np.zeros_like(qkv)
  for j in range(CONV): conv += ext[j:j + n] * cw[:, j][None]
  qkv = silu(conv); new_conv_state = ext[-(CONV - 1):].copy()
  q = qkv[:, :NK * DK].reshape(n, NK, DK); k = qkv[:, NK * DK:2 * NK * DK].reshape(n, NK, DK); v = qkv[:, 2 * NK * DK:].reshape(n, NV, DV)
  q, k = l2norm(q), l2norm(k); q = q * (1.0 / math.sqrt(DK))       # the delta rule: unit q, k; q also by 1/sqrt(DK)
  beta = 1.0 / (1.0 + np.exp(-b)); g = -np.exp(sf.f32(p + "A_log")) * softplus(a + sf.f32(p + "dt_bias"))   # [n, NV]
  rep = NV // NK                                                                    # value head hv reads key head hv // rep
  S = np.zeros((NV, DK, DV), np.float32) if state is None else state.copy(); o = np.empty((n, NV, DV), np.float32)
  for t in range(n):
    kt, qt = np.repeat(k[t], rep, 0), np.repeat(q[t], rep, 0)                     # [NV, DK]
    S *= np.exp(g[t])[:, None, None]
    kv = np.einsum("hkv,hk->hv", S, kt)
    delta = (v[t] - kv) * beta[t][:, None]
    S += kt[:, :, None] * delta[:, None, :]
    o[t] = np.einsum("hkv,hk->hv", S, qt)
  # gated RMSNorm over each value head, silu(z) gate, then out_proj
  w = sf.f32(p + "norm.weight")
  o = o * (1.0 / np.sqrt((o * o).mean(-1, keepdims=True) + EPS)) * w * silu(z)
  x = x + mm(W, sf, o.reshape(n, NV * DV), p + "out_proj")
  p2 = f"model.language_model.layers.{l}."
  return x + mlp(W, sf, p2, rms(x, sf.f32(p2 + "post_attention_layernorm.weight"))), (S, new_conv_state)

def layer(W, l, x, **kw):
  if LAYER_TYPES[l] == "full": return attention_layer(W, l, x, **kw), None
  return gdn_layer(W, l, x, **kw)

if MODEL == "ornith-9b":                                       # the GGUF checkpoint under the HF names (the reference and packer as they are)
  import sys as _sys
  _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ornith-9b"))
  from ornith_weights import Weights                           # noqa: E402,F811

if MODEL == "bonsai2-27b":                                     # the PTQ1_0 / Hadamard GGUF under the HF names
  import sys as _sys
  _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bonsai2-27b"))
  from bonsai2_weights import Weights                          # noqa: E402,F811

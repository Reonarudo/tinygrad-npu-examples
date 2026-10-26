#!/usr/bin/env python3
"""Ring-mini-2.0 / Ling-mini-2.0 (GGUF `bailingmoe2`, 16B-A1.4B) in float32 numpy from the GGUF: the reference every NPU kernel and
layer is gated against (llama.cpp's graph, src/models/bailingmoe2.cpp + build_moe_ffn, mirrored).

    per layer:  h = rms(x) w_attn;  q | k | v = W_qkv h  (16 q heads, 4 kv heads x 128)
                q = rope(rms_head(q) w_qn), k = rope(rms_head(k) w_kn): RoPE on the first ROT = 64 dims of 128 (NeoX halves), theta 6e5
                x = x + W_o softmax(q k^T / sqrt(128) + causal) v
                m = rms(x) w_ffn
                layer 0 (dense):  x = x + W_down (silu(W_gate m) * W_up m)                         (ff 5120)
                else (MoE):       s = sigmoid(W_r m); sel = s + b (exp_probs_b); 8 groups of 32: the 4 groups with the largest
                                  (sum of their top-2 sel) kept, the others -inf; top 8 of sel; w = s[top] / sum(s[top]) * 2.5
                                  x = x + sum_e w_e down_e(silu(gate_e m) up_e m) + down_s(silu(gate_s m) up_s m)   (shared, ungated)
    logits = W_out rms(x) w_out

Every matrix is ggml's own dequantisation (Q4_K / Q5_K / Q6_K via gguf_read).

    python3 ringmini_ref.py "prompt" [--max-new 16]       (greedy; the chat template: <role>HUMAN</role>...<role>ASSISTANT</role><think>\\n)
"""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "ornith-9b"))
from gguf_read import GGUF, GGML, dequant_q4_k, dequant_q5_k, dequant_q6_k   # noqa: E402

DEFAULT = os.environ.get("RING_GGUF", "/mnt/ssd/models/Ring-mini-2.0-Q4_K_M.gguf")
TOK = os.environ.get("RING_TOK", "/mnt/ssd/models/ring-mini-tokenizer")
EOS = 156892
DEQ = {"Q4_K": dequant_q4_k, "Q5_K": dequant_q5_k, "Q6_K": dequant_q6_k}
ROPE_NEOX = os.environ.get("RING_ROPE", "neox") == "neox"                # the rotation's pairing (see rope)


class Config:
  def __init__(self, g):
    m, a = g.meta, "bailingmoe2."
    self.NL, self.H, self.FFD = m[a + "block_count"], m[a + "embedding_length"], m[a + "feed_forward_length"]
    self.NH, self.NKV, self.HD = m[a + "attention.head_count"], m[a + "attention.head_count_kv"], m[a + "attention.key_length"]
    self.NE, self.TOPK, self.FF, self.FFS = m[a + "expert_count"], m[a + "expert_used_count"], m[a + "expert_feed_forward_length"], m[a + "expert_shared_feed_forward_length"]
    self.NG, self.TG, self.SCALE, self.NORMW = m[a + "expert_group_count"], m[a + "expert_group_used_count"], float(m[a + "expert_weights_scale"]), bool(m[a + "expert_weights_norm"])
    self.THETA, self.EPS, self.ROT = float(m[a + "rope.freq_base"]), float(m[a + "attention.layer_norm_rms_epsilon"]), m[a + "rope.dimension_count"]
    self.NDENSE = m[a + "leading_dense_block_count"]; self.V = len(m["tokenizer.ggml.tokens"]); assert m[a + "expert_gating_func"] == 2


def rms(x, w, eps): return (x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps))) * w
def silu(x): return x / (1.0 + np.exp(-x))


class Model:
  def __init__(self, path=DEFAULT):
    self.g = GGUF(path); self.c = Config(self.g); self._mat = {}; self._exp = {}
  def kind(self, name): return GGML[self.g.tensors[name][1]][0]
  def blocks(self, name):
    kind, dims, raw = self.g.raw(name); bb = GGML[self.g.tensors[name][1]][2]
    return kind, np.asarray(raw).reshape(*reversed(dims[1:]), dims[0] // 256, bb)
  def mat(self, name):
    if name not in self._mat:
      kind = self.kind(name)
      if kind in DEQ: k_, b = self.blocks(name); self._mat[name] = DEQ[k_](b.reshape(-1, b.shape[-1])).reshape(b.shape[0], -1)
      else: self._mat[name] = self.g.f32(name)
    return self._mat[name]
  def expert(self, l, e):
    key = (l, e)
    if key not in self._exp:
      out = []
      for t in ("gate", "up", "down"):
        kind, b = self.blocks(f"blk.{l}.ffn_{t}_exps.weight"); out.append(DEQ[kind](b[e].reshape(-1, b.shape[-1])).reshape(b.shape[1], -1))
      if len(self._exp) > 96: self._exp.pop(next(iter(self._exp)))
      self._exp[key] = tuple(out)
    return self._exp[key]
  def embed(self, ids):
    kind, b = self.blocks("token_embd.weight"); return DEQ[kind](b[np.asarray(ids)].reshape(-1, b.shape[-1])).reshape(len(ids), -1)
  def rope_cs(self, pos):
    ROT = self.c.ROT; inv = self.c.THETA ** (-np.arange(0, ROT, 2, dtype=np.float64) / ROT); ang = np.asarray(pos, np.float64)[:, None] * inv[None]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
  def rope(self, x, cs):
    """x [n, heads, HD]: the first ROT dims rotated (NeoX halves: (x1, x2) -> (x1 c - x2 s, x2 c + x1 s); RING_ROPE=norm: adjacent pairs),
    the rest pass through."""
    ROT = self.c.ROT; c, s = cs[0][:, None], cs[1][:, None]; xr, xp = x[..., :ROT], x[..., ROT:]
    if ROPE_NEOX:
      h = ROT // 2; x1, x2 = xr[..., :h], xr[..., h:]; r = np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)
    else:
      x1, x2 = xr[..., 0::2], xr[..., 1::2]; r = np.stack([x1 * c - x2 * s, x2 * c + x1 * s], -1).reshape(xr.shape)
    return np.concatenate([r, xp], -1)

  def attn(self, l, x, pos, kv):
    c, p = self.c, f"blk.{l}."; n = len(x); h = rms(x, self.mat(p + "attn_norm.weight"), c.EPS)
    qkv = h @ self.mat(p + "attn_qkv.weight").T; QN, KN = c.NH * c.HD, c.NKV * c.HD
    q = qkv[:, :QN].reshape(n, c.NH, c.HD); k = qkv[:, QN:QN + KN].reshape(n, c.NKV, c.HD); v = qkv[:, QN + KN:].reshape(n, c.NKV, c.HD)
    cs = self.rope_cs(pos); q = self.rope(rms(q, self.mat(p + "attn_q_norm.weight"), c.EPS), cs); k = self.rope(rms(k, self.mat(p + "attn_k_norm.weight"), c.EPS), cs)
    K0, V0 = kv.get(l, (np.zeros((0, c.NKV, c.HD), np.float32),) * 2); Kc, Vc = np.concatenate([K0, k]), np.concatenate([V0, v]); kv[l] = (Kc, Vc); T = len(Kc)
    g = c.NH // c.NKV; qg = q.reshape(n, c.NKV, g, c.HD); s = np.einsum("nkgd,tkd->nkgt", qg, Kc) * np.float32(1.0 / np.sqrt(c.HD))
    allowed = np.arange(T)[None] <= np.asarray(pos)[:, None]; s = np.where(allowed[:, None, None], s, -np.inf)
    s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True); a = np.einsum("nkgt,tkd->nkgd", s, Vc).reshape(n, c.NH * c.HD)
    return a @ self.mat(p + "attn_output.weight").T

  def route(self, l, m):
    """m [n, H] (normed) -> (top [n, TOPK], weights [n, TOPK]): sigmoid scores, the bias for the selection, the group top-k, the top 8,
    the unbiased scores normalised and scaled."""
    c = self.c; s = 1.0 / (1.0 + np.exp(-(m @ self.mat(f"blk.{l}.ffn_gate_inp.weight").T))); sel = s + self.mat(f"blk.{l}.exp_probs_b.bias")[None]
    gsz = c.NE // c.NG; grp = np.sort(sel.reshape(len(m), c.NG, gsz), -1)[..., -2:].sum(-1)             # a group's score: its top-2 selection probs
    keep = np.argsort(-grp, -1, kind="stable")[:, :c.TG]; mask = np.full((len(m), c.NG), -np.inf); np.put_along_axis(mask, keep, 0.0, -1)
    sel = (sel.reshape(len(m), c.NG, gsz) + mask[..., None]).reshape(len(m), c.NE)
    top = np.argsort(-sel, -1, kind="stable")[:, :c.TOPK]; w = np.take_along_axis(s, top, -1)
    if c.NORMW: w = w / np.maximum(w.sum(-1, keepdims=True), 6.103515625e-5)
    return top, w * c.SCALE

  def ffn(self, l, x, hid=None):
    c, p = self.c, f"blk.{l}."; m = rms(x, self.mat(p + "ffn_norm.weight"), c.EPS)
    if l < c.NDENSE: return (silu(m @ self.mat(p + "ffn_gate.weight").T) * (m @ self.mat(p + "ffn_up.weight").T)) @ self.mat(p + "ffn_down.weight").T
    top, wt = self.route(l, m)
    if hid is not None: hid.setdefault("route", {})[l] = (top, wt)
    out = np.zeros_like(x)
    for r in range(len(x)):
      for e, w in zip(top[r], wt[r]):
        gw, uw, dw = self.expert(l, int(e)); out[r] += w * ((silu(m[r] @ gw.T) * (m[r] @ uw.T)) @ dw.T)
    return out + (silu(m @ self.mat(p + "ffn_gate_shexp.weight").T) * (m @ self.mat(p + "ffn_up_shexp.weight").T)) @ self.mat(p + "ffn_down_shexp.weight").T

  def layer(self, l, x, pos, kv, hid=None):
    x = x + self.attn(l, x, pos, kv); return x + self.ffn(l, x, hid)
  def head(self, x):
    h = rms(x, self.mat("output_norm.weight"), self.c.EPS); kind, b = self.blocks("output.weight"); out = np.empty((len(x), b.shape[0]), np.float32)
    for r0 in range(0, b.shape[0], 8192): out[:, r0:r0 + 8192] = h @ DEQ[kind](b[r0:r0 + 8192].reshape(-1, b.shape[-1])).reshape(-1, h.shape[1]).T
    return out
  def forward(self, ids, pos0=0, kv=None, hid=None):
    kv = {} if kv is None else kv; pos = np.arange(pos0, pos0 + len(ids)); x = self.embed(ids)
    for l in range(self.c.NL): x = self.layer(l, x, pos, kv, hid)
    return x, kv
  def greedy(self, ids, n_new, stop=(EOS,), log=None):
    kv = {}; x, _ = self.forward(ids, 0, kv); out = []; pos = len(ids)
    for _ in range(n_new):
      t = int(np.argmax(self.head(x[-1:])[0])); out.append(t)
      if log: log(t)
      if t in stop: break
      x, _ = self.forward([t], pos, kv); pos += 1
    return out


def chat_text(prompt, thinking=True):
  """The GGUF's chat template: <role>HUMAN</role>{prompt}<role>ASSISTANT</role><think>\\n (Ring is a thinking model; `thinking=False`
  closes the think block at once)."""
  return f"<role>HUMAN</role>{prompt}<role>ASSISTANT</role><think>\n" + ("" if thinking else "\n</think>\n")


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("prompt", nargs="?", default="What is the capital of Portugal? Answer in one sentence.")
  ap.add_argument("--max-new", type=int, default=16); ap.add_argument("--gguf", default=DEFAULT); ap.add_argument("--tok", default=TOK); ap.add_argument("--no-think", action="store_true")
  a = ap.parse_args(); M = Model(a.gguf)
  from tokenizers import Tokenizer
  tok = Tokenizer.from_file(os.path.join(a.tok, "tokenizer.json")); ids = tok.encode(chat_text(a.prompt, not a.no_think), add_special_tokens=False).ids
  print(f"{len(ids)} prompt tokens: {ids}", flush=True); t0 = time.perf_counter()
  out = M.greedy(ids, a.max_new, log=lambda t: print(t, repr(tok.decode([t])), flush=True))
  print(f"ids: {out}\n{time.perf_counter() - t0:.1f} s\n{tok.decode(out)!r}")


if __name__ == "__main__": main()

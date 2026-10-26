#!/usr/bin/env python3
"""Qwen3-30B-A3B (`qwen3moe`) in float32 numpy, straight from the GGUF Q4_K_M checkpoint: the reference every NPU kernel and layer
is gated against.

    x = embed(ids)                                                   (token_embd rows, Q4_K dequantised)
    per layer:  a = softmax(q k^T / sqrt(HD) + causal) v              q = rope(rms(W_q h) w_qn), k = rope(rms(W_k h) w_kn) per head,
                                                                     h = rms(x) w_attn; GQA 32 q / 4 kv heads of 128; NeoX RoPE, theta 1e6
                x = x + W_o a
                p = softmax(W_r m), m = rms(x) w_ffn;  top 8 of 128, renormalised (norm_topk_prob)
                x = x + sum_e p_e W_down,e (silu(W_gate,e m) * W_up,e m)
    logits = W_out rms(x) w_out

Every matrix is ggml's own dequantisation of its Q4_K / Q6_K blocks (`gguf_read`), so the reference sees exactly the weights the
NPU's streams carry before their fp16 rounding.

    python3 qwen3moe_ref.py "prompt" [--max-new 16]     (greedy; the chat template with thinking off)
"""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ornith-9b"))
from gguf_read import GGUF, GGML, dequant_q4_k, dequant_q6_k   # noqa: E402

DEFAULT = os.environ.get("QWEN3MOE_GGUF", "/mnt/ssd/models/Qwen3-30B-A3B-Q4_K_M.gguf")
IM_START, IM_END, EOT = 151644, 151645, 151643


class Config:
  def __init__(self, g):
    m, a = g.meta, "qwen3moe."
    self.NL, self.H = m[a + "block_count"], m[a + "embedding_length"]
    self.NH, self.NKV, self.HD = m[a + "attention.head_count"], m[a + "attention.head_count_kv"], m[a + "attention.key_length"]
    self.NE, self.TOPK, self.FF = m[a + "expert_count"], m[a + "expert_used_count"], m[a + "expert_feed_forward_length"]
    self.THETA, self.EPS = float(m[a + "rope.freq_base"]), float(m[a + "attention.layer_norm_rms_epsilon"])
    self.V = len(m["tokenizer.ggml.tokens"])


def rms(x, w, eps):
  return (x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps))) * w


def dequant(kind, blk):
  """Super-blocks [..., bytes] of one kind -> float32 [..., 256 a block]."""
  sh = blk.shape[:-1]; f = blk.reshape(-1, blk.shape[-1])
  w = dequant_q4_k(f) if kind == "Q4_K" else dequant_q6_k(f) if kind == "Q6_K" else None
  assert w is not None, kind
  return w.reshape(*sh, 256)


class Model:
  def __init__(self, path=DEFAULT):
    self.g = GGUF(path); self.c = Config(self.g); self._mat = {}; self._exp = {}
  def kind(self, name): return GGML[self.g.tensors[name][1]][0]
  def t(self, name): return self.g.f32(name)
  def blocks(self, name):
    """A K-quant tensor's super-blocks [*outer, N, K / 256, bytes] (outer: the experts of a stacked tensor)."""
    kind, dims, raw = self.g.raw(name); bb = GGML[self.g.tensors[name][1]][2]
    return kind, np.asarray(raw).reshape(*reversed(dims[1:]), dims[0] // 256, bb)
  def mat(self, name):
    """A 2-D matrix [N, K] as float32 (cached: the attention projections and the norms are small)."""
    if name not in self._mat:
      kind = self.kind(name)
      if kind in ("Q4_K", "Q6_K"): k_, b = self.blocks(name); self._mat[name] = dequant(k_, b).reshape(b.shape[0], -1)
      else: self._mat[name] = self.t(name)
    return self._mat[name]
  def expert(self, l, e):
    """Expert e of layer l: (gate [FF, H], up [FF, H], down [H, FF]) float32, an LRU of the last 64 (19 MB each)."""
    key = (l, e)
    if key not in self._exp:
      out = []
      for t in ("gate", "up", "down"):
        kind, b = self.blocks(f"blk.{l}.ffn_{t}_exps.weight"); out.append(dequant(kind, b[e]).reshape(b.shape[1], -1))
      if len(self._exp) > 64: self._exp.pop(next(iter(self._exp)))
      self._exp[key] = tuple(out)
    return self._exp[key]

  def embed(self, ids):
    kind, b = self.blocks("token_embd.weight")
    return dequant(kind, b[np.asarray(ids)]).reshape(len(ids), -1)
  def rope_cs(self, pos):
    """cos, sin [len(pos), HD / 2] of the NeoX rotation (theta^(-2i / HD), float64 -> float32, as llama.cpp)."""
    HD = self.c.HD; inv = self.c.THETA ** (-np.arange(0, HD, 2, dtype=np.float64) / HD)
    ang = np.asarray(pos, np.float64)[:, None] * inv[None]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
  @staticmethod
  def rope(x, cs):
    """x [n, heads, HD] rotated in halves (NeoX): (x1, x2) -> (x1 c - x2 s, x2 c + x1 s)."""
    c, s = cs[0][:, None], cs[1][:, None]; h = x.shape[-1] // 2; x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)

  def attn(self, l, x, pos, kv):
    """The attention block's output W_o a [n, H] (not added): kv[l] = (K, V) [T, NKV, HD] grows by the n new rows."""
    c, p = self.c, f"blk.{l}."; n = len(x)
    h = rms(x, self.mat(p + "attn_norm.weight"), c.EPS)
    q = (h @ self.mat(p + "attn_q.weight").T).reshape(n, c.NH, c.HD)
    k = (h @ self.mat(p + "attn_k.weight").T).reshape(n, c.NKV, c.HD)
    v = (h @ self.mat(p + "attn_v.weight").T).reshape(n, c.NKV, c.HD)
    cs = self.rope_cs(pos)
    q = self.rope(rms(q, self.mat(p + "attn_q_norm.weight"), c.EPS), cs)
    k = self.rope(rms(k, self.mat(p + "attn_k_norm.weight"), c.EPS), cs)
    K0, V0 = kv.get(l, (np.zeros((0, c.NKV, c.HD), np.float32),) * 2)
    Kc, Vc = np.concatenate([K0, k]), np.concatenate([V0, v]); kv[l] = (Kc, Vc); T = len(Kc)
    g = c.NH // c.NKV; qg = q.reshape(n, c.NKV, g, c.HD)
    s = np.einsum("nkgd,tkd->nkgt", qg, Kc) * np.float32(1.0 / np.sqrt(c.HD))
    allowed = np.arange(T)[None] <= np.asarray(pos)[:, None]                 # causal: key t visible from query p iff t <= p
    s = np.where(allowed[:, None, None], s, -np.inf); s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
    a = np.einsum("nkgt,tkd->nkgd", s, Vc).reshape(n, c.NH * c.HD)
    return a @ self.mat(p + "attn_output.weight").T

  def route(self, l, m):
    """Router on the normed rows m [n, H] -> (top [n, TOPK] expert ids, weights [n, TOPK]): softmax over the 128, the top 8 (largest
    first), renormalised."""
    r = m @ self.mat(f"blk.{l}.ffn_gate_inp.weight").T
    p = np.exp(r - r.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    top = np.argsort(-p, -1, kind="stable")[:, :self.c.TOPK]; w = np.take_along_axis(p, top, -1)
    return top, w / w.sum(-1, keepdims=True)

  def moe(self, l, x, hid=None):
    """The expert block's output [n, H] (not added)."""
    c = self.c; m = rms(x, self.mat(f"blk.{l}.ffn_norm.weight"), c.EPS); top, wt = self.route(l, m)
    if hid is not None: hid.setdefault("route", {})[l] = (top, wt)
    out = np.zeros_like(x)
    for r in range(len(x)):
      for e, w in zip(top[r], wt[r]):
        gw, uw, dw = self.expert(l, int(e)); gt = m[r] @ gw.T
        out[r] += w * ((gt / (1.0 + np.exp(-gt)) * (m[r] @ uw.T)) @ dw.T)
    return out

  def layer(self, l, x, pos, kv, hid=None):
    x = x + self.attn(l, x, pos, kv)
    return x + self.moe(l, x, hid)
  def head(self, x):
    """Logits [n, V] of the last hidden rows: the Q6_K head dequantised in 8192-row chunks (2.4 GB as float32 whole)."""
    h = rms(x, self.mat("output_norm.weight"), self.c.EPS); kind, b = self.blocks("output.weight"); out = np.empty((len(x), b.shape[0]), np.float32)
    for r0 in range(0, b.shape[0], 8192): out[:, r0:r0 + 8192] = h @ dequant(kind, b[r0:r0 + 8192]).reshape(-1, h.shape[1]).T
    return out
  def forward(self, ids, pos0=0, kv=None, hid=None):
    kv = {} if kv is None else kv; pos = np.arange(pos0, pos0 + len(ids)); x = self.embed(ids)
    for l in range(self.c.NL): x = self.layer(l, x, pos, kv, hid)
    if hid is not None: hid["final"] = x
    return x, kv
  def greedy(self, ids, n_new, stop=(IM_END, EOT), log=None):
    kv = {}; x, _ = self.forward(ids, 0, kv); out = []; pos = len(ids)
    for _ in range(n_new):
      t = int(np.argmax(self.head(x[-1:])[0])); out.append(t)
      if log: log(t)
      if t in stop: break
      x, _ = self.forward([t], pos, kv); pos += 1
    return out


def chat_text(prompt, thinking=False):
  """Qwen3's chat template, one user turn + the generation prompt (enable_thinking=False appends the empty think block)."""
  return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n" + ("" if thinking else "<think>\n\n</think>\n\n")


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("prompt", nargs="?", default="What is the capital of Portugal? Answer in one sentence.")
  ap.add_argument("--max-new", type=int, default=16); ap.add_argument("--gguf", default=DEFAULT); ap.add_argument("--ids", default="")
  ap.add_argument("--tok", default=os.environ.get("QWEN3MOE_TOK", "/mnt/ssd/models/qwen3-30b-a3b-tokenizer"))
  a = ap.parse_args(); M = Model(a.gguf); tok = None
  if a.ids: ids = [int(v) for v in a.ids.split(",")]
  else:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.tok, "tokenizer.json")); ids = tok.encode(chat_text(a.prompt), add_special_tokens=False).ids
  print(f"{len(ids)} prompt tokens: {ids}", flush=True); t0 = time.perf_counter()
  out = M.greedy(ids, a.max_new, log=lambda t: print(t, repr(tok.decode([t])) if tok else "", flush=True))
  print(f"ids: {out}\n{time.perf_counter() - t0:.1f} s")
  if tok: print(repr(tok.decode(out)))


if __name__ == "__main__": main()

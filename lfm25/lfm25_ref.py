#!/usr/bin/env python3
"""LFM2.5-24B-A2B (GGUF `lfm2moe`) in float32 numpy from the GGUF: the reference every NPU kernel and layer is gated against
(llama.cpp's graph, src/models/lfm2.cpp + build_moe_ffn, mirrored).

    per layer:  h = rms(x) w_attn
      conv layer (30 of 40):  b | c | g = W_in h (3 x H); u = b * g; v_t = sum_j w[j] u_{t-2+j} (depthwise, 3 taps, the 2 previous u rows
                              the state); y = W_out (c * v)
      attention layer (10):   q (32 x 64), k, v (8 x 64) = W h; q, k per-head RMSNorm, RoPE (NeoX, all 64 dims, theta 1e7); GQA, scale 1/8; W_o
      x = x + y;  m = rms(x) w_ffn
      dense (layers 0, 1):    x = x + W_down (silu(W_gate m) W_up m)                   (ff 11776)
      MoE:                    s = sigmoid(W_r m); top 4 of s + b; w = s[top] / sum(s[top]);  x = x + sum_e w_e expert_e(m)
    logits = token_embd rms(x) w_out   (tied: token_embd_norm is the final norm)

    python3 lfm25_ref.py "prompt" [--max-new 16]
"""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "ornith-9b"))
from gguf_read import GGUF, GGML, dequant_q4_k, dequant_q5_k, dequant_q6_k   # noqa: E402

DEFAULT = os.environ.get("LFM_GGUF", "/mnt/ssd/models/LFM2.5-24B-A2B-Q4_K_M.gguf")
TOK = os.environ.get("LFM_TOK", "/mnt/ssd/models/lfm25-tokenizer")
BOS, IM_START, IM_END = 124894, 124899, 124900
DEQ = {"Q4_K": dequant_q4_k, "Q5_K": dequant_q5_k, "Q6_K": dequant_q6_k}


class Config:
  def __init__(self, g):
    m, a = g.meta, "lfm2moe."
    self.NL, self.H, self.FFD = m[a + "block_count"], m[a + "embedding_length"], m[a + "feed_forward_length"]
    self.NH, self.KVL = m[a + "attention.head_count"], list(m[a + "attention.head_count_kv"]); self.NKV = max(self.KVL); self.HD = self.H // self.NH
    self.NE, self.TOPK, self.FF = m[a + "expert_count"], m[a + "expert_used_count"], m[a + "expert_feed_forward_length"]
    self.THETA, self.EPS, self.NDENSE, self.LC = float(m[a + "rope.freq_base"]), float(m[a + "attention.layer_norm_rms_epsilon"]), m[a + "leading_dense_block_count"], m[a + "shortconv.l_cache"]
    self.V = m[a + "vocab_size"]; assert m[a + "expert_gating_func"] == 2
    self.ATTN = [l for l in range(self.NL) if self.KVL[l] > 0]


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
      if len(self._exp) > 64: self._exp.pop(next(iter(self._exp)))
      self._exp[key] = tuple(out)
    return self._exp[key]
  def embed(self, ids):
    kind, b = self.blocks("token_embd.weight"); return DEQ[kind](b[np.asarray(ids)].reshape(-1, b.shape[-1])).reshape(len(ids), -1)
  def rope_cs(self, pos):
    HD = self.c.HD; inv = self.c.THETA ** (-np.arange(0, HD, 2, dtype=np.float64) / HD); ang = np.asarray(pos, np.float64)[:, None] * inv[None]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
  @staticmethod
  def rope(x, cs):
    c, s = cs[0][:, None], cs[1][:, None]; h = x.shape[-1] // 2; x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)

  def conv(self, l, h, st):
    """The short-conv block of layer l on the normed rows h [n, H]; st[l] = the last 2 u rows (the state), updated."""
    c, p = self.c, f"blk.{l}."; n = len(h); bcx = h @ self.mat(p + "shortconv.in_proj.weight").T; H = c.H
    b, cc, g = bcx[:, :H], bcx[:, H:2 * H], bcx[:, 2 * H:]; u = b * g; w = self.mat(p + "shortconv.conv.weight").T     # the GGUF's [H, 3] -> [3, H]: tap j multiplies u_{t - 2 + j}
    prev = st.get(l, np.zeros((c.LC - 1, H), np.float32)); ext = np.concatenate([prev, u]); st[l] = ext[-(c.LC - 1):].copy()
    v = sum(ext[j:j + n] * w[j][None] for j in range(c.LC))
    return (cc * v) @ self.mat(p + "shortconv.out_proj.weight").T

  def attn(self, l, h, pos, kv):
    c, p = self.c, f"blk.{l}."; n = len(h)
    q = (h @ self.mat(p + "attn_q.weight").T).reshape(n, c.NH, c.HD); k = (h @ self.mat(p + "attn_k.weight").T).reshape(n, c.NKV, c.HD); v = (h @ self.mat(p + "attn_v.weight").T).reshape(n, c.NKV, c.HD)
    cs = self.rope_cs(pos); q = self.rope(rms(q, self.mat(p + "attn_q_norm.weight"), c.EPS), cs); k = self.rope(rms(k, self.mat(p + "attn_k_norm.weight"), c.EPS), cs)
    K0, V0 = kv.get(l, (np.zeros((0, c.NKV, c.HD), np.float32),) * 2); Kc, Vc = np.concatenate([K0, k]), np.concatenate([V0, v]); kv[l] = (Kc, Vc); T = len(Kc)
    g = c.NH // c.NKV; qg = q.reshape(n, c.NKV, g, c.HD); s = np.einsum("nkgd,tkd->nkgt", qg, Kc) * np.float32(1.0 / np.sqrt(c.HD))
    allowed = np.arange(T)[None] <= np.asarray(pos)[:, None]; s = np.where(allowed[:, None, None], s, -np.inf)
    s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True); a = np.einsum("nkgt,tkd->nkgd", s, Vc).reshape(n, c.NH * c.HD)
    return a @ self.mat(p + "attn_output.weight").T

  def route(self, l, m):
    c = self.c; s = 1.0 / (1.0 + np.exp(-(m @ self.mat(f"blk.{l}.ffn_gate_inp.weight").T))); sel = s + self.mat(f"blk.{l}.exp_probs_b.bias")[None]
    top = np.argsort(-sel, -1, kind="stable")[:, :c.TOPK]; w = np.take_along_axis(s, top, -1); return top, w / np.maximum(w.sum(-1, keepdims=True), 6.103515625e-5)

  def ffn(self, l, x, hid=None):
    c, p = self.c, f"blk.{l}."; m = rms(x, self.mat(p + "ffn_norm.weight"), c.EPS)
    if l < c.NDENSE: return (silu(m @ self.mat(p + "ffn_gate.weight").T) * (m @ self.mat(p + "ffn_up.weight").T)) @ self.mat(p + "ffn_down.weight").T
    top, wt = self.route(l, m)
    if hid is not None: hid.setdefault("route", {})[l] = (top, wt)
    out = np.zeros_like(x)
    for r in range(len(x)):
      for e, w in zip(top[r], wt[r]):
        gw, uw, dw = self.expert(l, int(e)); out[r] += w * ((silu(m[r] @ gw.T) * (m[r] @ uw.T)) @ dw.T)
    return out

  def layer(self, l, x, pos, kv, st, hid=None):
    h = rms(x, self.mat(f"blk.{l}.attn_norm.weight"), self.c.EPS)
    x = x + (self.attn(l, h, pos, kv) if l in self.c.ATTN else self.conv(l, h, st)); return x + self.ffn(l, x, hid)
  def head(self, x):
    h = rms(x, self.mat("token_embd_norm.weight"), self.c.EPS); kind, b = self.blocks("token_embd.weight"); out = np.empty((len(x), b.shape[0]), np.float32)
    for r0 in range(0, b.shape[0], 8192): out[:, r0:r0 + 8192] = h @ DEQ[kind](b[r0:r0 + 8192].reshape(-1, b.shape[-1])).reshape(-1, h.shape[1]).T
    return out
  def forward(self, ids, pos0, kv, st, hid=None):
    pos = np.arange(pos0, pos0 + len(ids)); x = self.embed(ids)
    for l in range(self.c.NL): x = self.layer(l, x, pos, kv, st, hid)
    return x
  def greedy(self, ids, n_new, stop=(IM_END,), log=None):
    kv, st = {}, {}; x = self.forward(ids, 0, kv, st); out = []; pos = len(ids)
    for _ in range(n_new):
      t = int(np.argmax(self.head(x[-1:])[0])); out.append(t)
      if log: log(t)
      if t in stop: break
      x = self.forward([t], pos, kv, st); pos += 1
    return out


def chat_text(prompt): return f"<|startoftext|><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("prompt", nargs="?", default="What is the capital of Portugal? Answer in one sentence.")
  ap.add_argument("--max-new", type=int, default=16); ap.add_argument("--gguf", default=DEFAULT); ap.add_argument("--tok", default=TOK); a = ap.parse_args(); M = Model(a.gguf)
  from tokenizers import Tokenizer
  tok = Tokenizer.from_file(os.path.join(a.tok, "tokenizer.json")); ids = tok.encode(chat_text(a.prompt), add_special_tokens=False).ids
  print(f"{len(ids)} prompt tokens: {ids}", flush=True); t0 = time.perf_counter()
  out = M.greedy(ids, a.max_new, log=lambda t: print(t, repr(tok.decode([t])), flush=True))
  print(f"ids: {out}\n{time.perf_counter() - t0:.1f} s\n{tok.decode(out)!r}")


if __name__ == "__main__": main()

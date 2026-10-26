#!/usr/bin/env python3
"""gpt-oss-20b (GGUF `gpt-oss`, MXFP4 experts) in float32 numpy from the GGUF: the reference every NPU kernel and layer is gated against
(llama.cpp's graph, src/models/openai-moe.cpp + build_moe_ffn, mirrored).

    per layer:  h = rms(x) w_attn;  q (64 x 64), k, v (8 x 64) = W h + b; q, k RoPE (NeoX, YaRN: factor 32 over 4096, beta 32 / 1, the
                magnitude scale 1 + 0.1 ln 32); scores / 8; even layers see only the last 128 positions; the softmax has a per-head sink
                logit in its denominator; x = x + W_o a + b_o
                m = rms(x) w_post;  l = W_r m + b_r; top 4 of l; w = softmax(l[top])
                x = x + sum_e w_e (W_down,e g(gate_e, up_e) + b_down,e), gate_e = W_gate,e m + b, up_e = W_up,e m + b,
                g(a, u) = min(a, 7) sigmoid(1.702 min(a, 7)) (clamp(u, -7, 7) + 1)
    logits = W_out rms(x) w_out

    python3 gptoss_ref.py "prompt" [--max-new 16]      (the raw harmony turn: <|start|>user<|message|>...<|end|><|start|>assistant)
"""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "ornith-9b"))
from gguf_read import GGUF, GGML, dequant_mxfp4   # noqa: E402

DEFAULT = os.environ.get("OSS_GGUF", "/mnt/ssd/models/gpt-oss-20b-MXFP4.gguf")
TOK = os.environ.get("OSS_TOK", "/mnt/ssd/models/gpt-oss-tokenizer")
RETURN, END = 200002, 200007
ALPHA, LIMIT = 1.702, 7.0


class Config:
  def __init__(self, g):
    m, a = g.meta, "gpt-oss."
    self.NL, self.H, self.NH, self.NKV, self.HD = m[a + "block_count"], m[a + "embedding_length"], m[a + "attention.head_count"], m[a + "attention.head_count_kv"], m[a + "attention.key_length"]
    self.NE, self.TOPK, self.FF, self.SWA = m[a + "expert_count"], m[a + "expert_used_count"], m[a + "expert_feed_forward_length"], m[a + "attention.sliding_window"]
    self.THETA, self.EPS = float(m[a + "rope.freq_base"]), float(m[a + "attention.layer_norm_rms_epsilon"])
    self.YF, self.YCTX, self.BF, self.BS = float(m[a + "rope.scaling.factor"]), m[a + "rope.scaling.original_context_length"], float(m[a + "rope.scaling.yarn_beta_fast"]), float(m[a + "rope.scaling.yarn_beta_slow"])
    self.SWA_LAYERS = [l for l in range(self.NL) if l % 2 == 0]           # llama.cpp: swa_layers for gpt-oss = even layers


def rms(x, w, eps): return (x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps))) * w


class Model:
  def __init__(self, path=DEFAULT):
    self.g = GGUF(path); self.c = Config(self.g); self._mat = {}; self._exp = {}
  def kind(self, name): return GGML[self.g.tensors[name][1]][0]
  def blocks(self, name):
    kind, dims, raw = self.g.raw(name); bl, bb = GGML[self.g.tensors[name][1]][1:]
    return kind, np.asarray(raw).reshape(*reversed(dims[1:]), dims[0] // bl, bb)
  def mat(self, name):
    if name not in self._mat: self._mat[name] = self.g.f32(name)
    return self._mat[name]
  def expert(self, l, e):
    key = (l, e)
    if key not in self._exp:
      out = []
      for t in ("gate", "up", "down"):
        kind, b = self.blocks(f"blk.{l}.ffn_{t}_exps.weight"); assert kind == "MXFP4"; out.append(dequant_mxfp4(b[e].reshape(-1, 17)).reshape(b.shape[1], -1))
      if len(self._exp) > 48: self._exp.pop(next(iter(self._exp)))
      self._exp[key] = tuple(out)
    return self._exp[key]
  def embed(self, ids): return self.g.q8_rows("token_embd.weight", list(ids)) if hasattr(self.g, "q8_rows") else self.mat("token_embd.weight")[np.asarray(ids)]

  def rope_cs(self, pos):
    """YaRN (ggml's rope_yarn): per pair i (dims i, i + 32): theta_extrap = pos base^(-2i / HD); theta = interp (1 - r) + extrap r, r the ramp
    between the correction dims; cos / sin scaled by 1 + 0.1 ln(factor)."""
    c = self.c; HD = c.HD; i0 = np.arange(0, HD, 2, dtype=np.float64); extrap = np.asarray(pos, np.float64)[:, None] * (c.THETA ** (-i0 / HD))[None]
    fs = 1.0 / c.YF; interp = fs * extrap
    cd = lambda nrot: HD * np.log(c.YCTX / (nrot * 2 * np.pi)) / (2 * np.log(c.THETA))
    lo, hi = max(0.0, np.floor(cd(c.BF))), min(HD - 1.0, np.ceil(cd(c.BS)))
    ramp = 1.0 - np.clip((i0 / 2 - lo) / max(0.001, hi - lo), 0.0, 1.0); theta = interp * (1 - ramp) + extrap * ramp
    ms = 1.0 + 0.1 * np.log(1.0 / fs); return (np.cos(theta) * ms).astype(np.float32), (np.sin(theta) * ms).astype(np.float32)
  @staticmethod
  def rope(x, cs):
    c, s = cs[0][:, None], cs[1][:, None]; h = x.shape[-1] // 2; x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)

  def attn(self, l, x, pos, kv):
    c, p = self.c, f"blk.{l}."; n = len(x); h = rms(x, self.mat(p + "attn_norm.weight"), c.EPS)
    q = (h @ self.mat(p + "attn_q.weight").T + self.mat(p + "attn_q.bias")).reshape(n, c.NH, c.HD)
    k = (h @ self.mat(p + "attn_k.weight").T + self.mat(p + "attn_k.bias")).reshape(n, c.NKV, c.HD); v = (h @ self.mat(p + "attn_v.weight").T + self.mat(p + "attn_v.bias")).reshape(n, c.NKV, c.HD)
    cs = self.rope_cs(pos); q, k = self.rope(q, cs), self.rope(k, cs)
    K0, V0 = kv.get(l, (np.zeros((0, c.NKV, c.HD), np.float32),) * 2); Kc, Vc = np.concatenate([K0, k]), np.concatenate([V0, v]); kv[l] = (Kc, Vc); T = len(Kc)
    g = c.NH // c.NKV; qg = q.reshape(n, c.NKV, g, c.HD); s = np.einsum("nkgd,tkd->nkgt", qg, Kc) * np.float32(1.0 / np.sqrt(c.HD))
    pos = np.asarray(pos); allowed = np.arange(T)[None] <= pos[:, None]
    if l in c.SWA_LAYERS: allowed &= np.arange(T)[None] > pos[:, None] - c.SWA
    s = np.where(allowed[:, None, None], s, -np.inf); sink = self.mat(p + "attn_sinks.weight").reshape(1, c.NKV, g, 1)
    mx = np.maximum(s.max(-1, keepdims=True), sink); e = np.exp(s - mx); a = np.einsum("nkgt,tkd->nkgd", e, Vc) / (e.sum(-1, keepdims=True) + np.exp(sink - mx))
    return a.reshape(n, c.NH * c.HD) @ self.mat(p + "attn_output.weight").T + self.mat(p + "attn_output.bias")

  def route(self, l, m):
    c = self.c; lg = m @ self.mat(f"blk.{l}.ffn_gate_inp.weight").T + self.mat(f"blk.{l}.ffn_gate_inp.bias"); top = np.argsort(-lg, -1, kind="stable")[:, :c.TOPK]
    w = np.take_along_axis(lg, top, -1); w = np.exp(w - w.max(-1, keepdims=True)); return top, w / w.sum(-1, keepdims=True)
  @staticmethod
  def act(a, u):
    a = np.minimum(a, LIMIT); u = np.clip(u, -LIMIT, LIMIT); return a / (1.0 + np.exp(-ALPHA * a)) * (u + 1.0)
  def moe(self, l, x, hid=None):
    c, p = self.c, f"blk.{l}."; m = rms(x, self.mat(p + "post_attention_norm.weight"), c.EPS)
    top, wt = self.route(l, m)
    if hid is not None: hid.setdefault("route", {})[l] = (top, wt)
    bg, bu, bd = self.mat(p + "ffn_gate_exps.bias"), self.mat(p + "ffn_up_exps.bias"), self.mat(p + "ffn_down_exps.bias"); out = np.zeros_like(x)
    for r in range(len(x)):
      for e, w in zip(top[r], wt[r]):
        gw, uw, dw = self.expert(l, int(e)); out[r] += w * (self.act(m[r] @ gw.T + bg[e], m[r] @ uw.T + bu[e]) @ dw.T + bd[e])
    return out

  def layer(self, l, x, pos, kv, hid=None):
    x = x + self.attn(l, x, pos, kv); return x + self.moe(l, x, hid)
  def head(self, x):
    h = rms(x, self.mat("output_norm.weight"), self.c.EPS); q, d = self.g.q8("output.weight"); out = np.empty((len(x), q.shape[0]), np.float32)
    for r0 in range(0, q.shape[0], 16384):
      w = (q[r0:r0 + 16384].astype(np.float32).reshape(-1, q.shape[1] // 32, 32) * d[r0:r0 + 16384, :, None]).reshape(-1, q.shape[1]); out[:, r0:r0 + w.shape[0]] = h @ w.T
    return out
  def forward(self, ids, pos0, kv, hid=None):
    pos = np.arange(pos0, pos0 + len(ids)); x = self.embed(ids)
    for l in range(self.c.NL): x = self.layer(l, x, pos, kv, hid)
    return x
  def greedy(self, ids, n_new, stop=(RETURN, END), log=None):
    kv = {}; x = self.forward(ids, 0, kv); out = []; pos = len(ids)
    for _ in range(n_new):
      t = int(np.argmax(self.head(x[-1:])[0])); out.append(t)
      if log: log(t)
      if t in stop: break
      x = self.forward([t], pos, kv); pos += 1
    return out


def chat_text(prompt): return f"<|start|>user<|message|>{prompt}<|end|><|start|>assistant"


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("prompt", nargs="?", default="What is the capital of Portugal? Answer in one sentence.")
  ap.add_argument("--max-new", type=int, default=16); ap.add_argument("--gguf", default=DEFAULT); ap.add_argument("--tok", default=TOK); a = ap.parse_args(); M = Model(a.gguf)
  from tokenizers import Tokenizer
  tok = Tokenizer.from_file(os.path.join(a.tok, "tokenizer.json")); ids = tok.encode(chat_text(a.prompt), add_special_tokens=False).ids
  print(f"{len(ids)} prompt tokens: {ids}", flush=True); t0 = time.perf_counter()
  out = M.greedy(ids, a.max_new, log=lambda t: print(t, repr(tok.decode([t])), flush=True))
  print(f"ids: {out}\n{time.perf_counter() - t0:.1f} s\n{tok.decode(out)!r}")


if __name__ == "__main__": main()

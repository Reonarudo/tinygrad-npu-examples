#!/usr/bin/env python3
"""Gemma 4 26B-A4B in float32 numpy from its GGUF (unsloth UD-Q4_K_M: Q8_0 attention / dense FFN / head, Q4_K fused gate|up experts,
Q5_1 down experts): gemma4_ref.Model with the expert block dequantising only the 8 selected experts (the whole tensors are 2 GB a layer
in fp32) and exposing the routing (`hid["route"][l]`). The reference every kernel and layer of gemma26_npu.py is gated against.

    python3 gemma26_ref.py "prompt" [--max-new 8]
"""
import argparse, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, "..", "gemma4"), os.path.join(HERE, "..", "ornith-9b")]
import gemma4_ref as GR                                                    # noqa: E402
from gemma4_ref import rms, gelu_tanh                                      # noqa: E402
from gguf_read import GGML, dequant_q4_k, dequant_q5_1                     # noqa: E402

DEFAULT = os.environ.get("G26_GGUF", os.path.expanduser("~/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"))


class Model(GR.Model):
  def __init__(self, path=DEFAULT, cache_layers=True):
    super().__init__(path, cache_layers); self._exp = {}; self.hid = None
  def blocks(self, name):
    """A quantised tensor's blocks [*outer dims, K / bl, bb] (uint8)."""
    kind, dims, raw = self.g.raw(name); bl, bb = GGML[self.g.tensors[name][1]][1:]
    return kind, np.asarray(raw).reshape(*reversed(dims[1:]), dims[0] // bl, bb)
  def expert(self, l, e):
    """(gate|up [2 F, H], down [H, F]) of expert e of layer l, dequantised (the last 64 kept)."""
    key = (l, e)
    if key not in self._exp:
      p = f"blk.{l}."; kind, b = self.blocks(p + "ffn_gate_up_exps.weight"); assert kind == "Q4_K"
      gu = dequant_q4_k(b[e].reshape(-1, b.shape[-1])).reshape(b.shape[1], -1)
      kind, b = self.blocks(p + "ffn_down_exps.weight")
      dn = {"Q5_1": dequant_q5_1, "Q8_0": lambda x: (x[:, 2:].view(np.int8).astype(np.float32) * x[:, :2].copy().view(np.float16).astype(np.float32))}[kind](b[e].reshape(-1, b.shape[-1])).reshape(b.shape[1], -1)
      if len(self._exp) > 64: self._exp.pop(next(iter(self._exp)))
      self._exp[key] = (gu, dn)
    return self._exp[key]
  def route(self, l, r):
    """-> (top [n, NEU], weights [n, NEU] incl. the per-expert scale): the router on rms(r) scale / sqrt(H), softmax, top-k, renormalised."""
    c, p = self.c, f"blk.{l}."
    t = rms(r, None, c.EPS) * self.t(p + "ffn_gate_inp.scale") * np.float32(1.0 / math.sqrt(c.H))
    z = t @ self.t(p + "ffn_gate_inp.weight").T; z = np.exp(z - z.max(-1, keepdims=True)); z /= z.sum(-1, keepdims=True)
    top = np.argsort(-z, -1, kind="stable")[:, :c.NEU]; wt = np.take_along_axis(z, top, -1); wt /= wt.sum(-1, keepdims=True)
    return top, wt * self.t(p + "ffn_down_exps.scale")[top]
  def moe(self, l, r):
    c, p = self.c, f"blk.{l}."; top, wt = self.route(l, r)
    if self.hid is not None: self.hid.setdefault("route", {})[l] = (top, wt)
    h2 = rms(r, self.t(p + "pre_ffw_norm_2.weight"), c.EPS); out = np.zeros_like(r)
    for i in range(r.shape[0]):
      for j, e in enumerate(top[i]):
        gu, dn = self.expert(l, int(e)); F = dn.shape[-1]; y = gu @ h2[i]; out[i] += wt[i, j] * (dn @ (gelu_tanh(y[:F]) * y[F:]))
    return rms(out, self.t(p + "post_ffw_norm_2.weight"), c.EPS)


def main():
  sys.path.insert(0, os.path.join(HERE, "..", "gemma4")); import gemma4_tokenize as GT
  ap = argparse.ArgumentParser(); ap.add_argument("prompt", nargs="?", default="What is the capital of Portugal? Answer in one sentence."); ap.add_argument("--max-new", type=int, default=8); ap.add_argument("--gguf", default=DEFAULT); a = ap.parse_args()
  M = Model(a.gguf); tok = GT.Tok(M.g.meta); ids = tok.encode(GT.chat_messages([{"role": "user", "content": a.prompt}]))
  print(f"{len(ids)} prompt tokens: {ids}", flush=True); t0 = time.perf_counter(); out = M.greedy(ids, a.max_new)
  print(f"ids: {out}\n{time.perf_counter() - t0:.0f} s\n{tok.decode(out)!r}")


if __name__ == "__main__": main()

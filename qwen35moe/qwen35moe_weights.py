"""Qwen3.5-35B-A3B / Qwen3.6-35B-A3B (GGUF `qwen35moe`) under the Hugging Face names: the DeltaNet / attention parts exactly as
Ornith's adapter serves them (the same 16 key x 32 value head layout, the GGUF converter's reordering undone), plus the MoE block:

  router   ffn_gate_inp [NE, H] (F32): p = softmax(m W_r^T), the top 8 renormalised (norm_topk_prob)
  experts  ffn_gate_exps / ffn_up_exps [NE][FF][H] (Q4_K), ffn_down_exps [NE][H][FF] (Q5_K in the Q4_K_M quant): the stacked K-quant
           super-blocks, dequantised per expert on demand (ggml's own dequantisation: gguf_read)
  shared   ffn_gate_shexp / ffn_up_shexp [FF][H], ffn_down_shexp [H][FF] (Q8_0), gated by sigmoid(m . ffn_gate_inp_shexp)
  out = sum_e p_e down_e(silu(gate_e m) up_e m) + sigmoid(g) down_s(silu(gate_s m) up_s m)

`moe(sf, p, h)` is what qwen38_ref.mlp calls for a MoE model (QWEN_MODEL=qwen3.5-35b-a3b); `route(l, m)` exposes the routing for
the gates; `blocks(name)` the raw super-blocks for the packer."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "ornith-9b"))
import ornith_weights as OW                                               # noqa: E402
from gguf_read import GGML, dequant_q4_k, dequant_q5_k, dequant_q6_k      # noqa: E402

DEQ = {"Q4_K": dequant_q4_k, "Q5_K": dequant_q5_k, "Q6_K": dequant_q6_k}


def silu(x): return x / (1.0 + np.exp(-x))


class Weights(OW.Weights):
  def __init__(self, path):
    super().__init__(path); m = self.g.meta; a = "qwen35moe."
    self.NL, self.H, self.NE, self.TOPK, self.FF = m[a + "block_count"], m[a + "embedding_length"], m[a + "expert_count"], m[a + "expert_used_count"], m[a + "expert_feed_forward_length"]
    self._exp = {}; self.last_route = {}
  def f32(self, name):
    """ornith_weights' f32, plus the DeltaNet's a / b projections stored as F32 (Qwen3.6's GGUF) rather than Q8_0 (the 3.5 quant)."""
    rest = name.split(".", 4)[-1] if name.startswith("model.language_model.layers.") else name
    if rest in ("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"):
      l = int(name.split("layers.")[1].split(".")[0]); t = f"blk.{l}.{OW.LIN[rest[:-len('.weight')]]}.weight"
      if self.g.raw(t)[0] != "Q8_0": return self.g.f32(t)[OW.PERM].astype(np.float32)
    return super().f32(name)
  def blocks(self, name):
    """A K-quant tensor's super-blocks [*outer, N, K / 256, bytes] (outer: the experts of a stacked tensor)."""
    kind, dims, raw = self.g.raw(name); bb = GGML[self.g.tensors[name][1]][2]
    return kind, np.asarray(raw).reshape(*reversed(dims[1:]), dims[0] // 256, bb)
  def expert(self, l, e):
    """Expert e of layer l: (gate [FF, H], up [FF, H], down [H, FF]) float32; an LRU of the last 96."""
    key = (l, e)
    if key not in self._exp:
      out = []
      for t in ("gate", "up", "down"):
        kind, b = self.blocks(f"blk.{l}.ffn_{t}_exps.weight"); out.append(DEQ[kind](b[e].reshape(-1, b.shape[-1])).reshape(b.shape[1], -1))
      if len(self._exp) > 96: self._exp.pop(next(iter(self._exp)))
      self._exp[key] = tuple(out)
    return self._exp[key]
  def route(self, l, m):
    """m [n, H] (normed) -> (top [n, TOPK], weights [n, TOPK]): softmax over the NE, the top TOPK (largest first), renormalised."""
    r = m @ self.g.f32(f"blk.{l}.ffn_gate_inp.weight").T
    p = np.exp(r - r.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    top = np.argsort(-p, -1, kind="stable")[:, :self.TOPK]; w = np.take_along_axis(p, top, -1)
    return top, w / w.sum(-1, keepdims=True)
  def moe(self, sf, p, m):
    """The expert block's output [n, H] for the normed rows m (qwen38_ref.mlp's hook)."""
    l = int(p.split("layers.")[1].split(".")[0]); top, wt = self.route(l, m); self.last_route[l] = (top, wt)
    out = np.zeros_like(m)
    for r in range(len(m)):
      for e, w in zip(top[r], wt[r]):
        gw, uw, dw = self.expert(l, int(e)); out[r] += w * ((silu(m[r] @ gw.T) * (m[r] @ uw.T)) @ dw.T)
    q = f"blk.{l}.ffn_"; sg = 1.0 / (1.0 + np.exp(-(m @ self.g.f32(q + "gate_inp_shexp.weight").reshape(-1))))
    sh = (silu(m @ self.g.f32(q + "gate_shexp.weight").T) * (m @ self.g.f32(q + "up_shexp.weight").T)) @ self.g.f32(q + "down_shexp.weight").T
    return out + sg[:, None] * sh

  def lm_head(self, x):
    """Logits [n, V] from the Q6_K head (Ornith's is Q8_0): dequantised in 8192-row chunks."""
    kind, b = self.blocks("output.weight"); out = np.empty((x.shape[0], b.shape[0]), np.float32)
    for r0 in range(0, b.shape[0], 8192):
      w = DEQ[kind](b[r0:r0 + 8192].reshape(-1, b.shape[-1])).reshape(-1, x.shape[1]); out[:, r0:r0 + w.shape[0]] = x @ w.T
    return out

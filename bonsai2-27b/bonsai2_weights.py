"""Ternary Bonsai 2 27B (GGUF, architecture `qwen35`, PTQ1_0 + prism.hadamard) served under the Hugging Face Qwen3.5 names, so
qwen3.8-27b's numpy reference (qwen38_ref.py with QWEN_MODEL=bonsai2-27b) runs it unchanged.

Every linear goes through `mm(sf, x, name)` rather than a dequantised [N, K] matrix: the stored weights are in a rotated basis
(W' = W diag(s) Hb), so `mm` feeds them Hb (s * x) exactly as the format's contract prescribes,
dequantising 4096 rows at a time. The DeltaNet's ssm_alpha / ssm_beta are BF16 and not folded (absent from weight_names).

The converter's qwen35 conventions are undone as for Ornith (../ornith-9b/ornith_weights.py): zero-centred RMSNorm weights stored as 1 + w, A_log stored as -exp(A_log), the conv kernel as [C, 4], and the V heads
of in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / A_log / dt_bias / the conv reordered from grouped (value head h reads key
head h // 3) to tiled (h' = r * 16 + g): restored here on the output rows. out_proj is the exception: with a Hadamard fold a
column permutation cannot be folded back, so the converter keeps it in grouped order (GGUF
prism.hadamard.gdn_v_grouped = True) and a runtime in tiled order permutes its activation to grouped before the transform. The reference's activation is already grouped (HF order): no permutation either way."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bonsai2_gguf import Bonsai2GGUF

NK, DK, NV, DV = 16, 128, 48, 128
REP = NV // NK
PERM = np.array([(h % REP) * NK + h // REP for h in range(NV)])            # HF (grouped) head h <- GGUF (tiled) head PERM[h]
def _rows(D): return (PERM[:, None] * D + np.arange(D)[None]).ravel()
QK = 2 * NK * DK

LIN = {"linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate", "linear_attn.out_proj": "ssm_out",
       "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
       "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}
OUT_ROWS = {"linear_attn.in_proj_qkv": np.concatenate([np.arange(QK), QK + _rows(DV)]), "linear_attn.in_proj_z": _rows(DV)}
VEC = {"input_layernorm.weight": ("attn_norm.weight", -1), "post_attention_layernorm.weight": ("post_attention_norm.weight", -1),
       "self_attn.q_norm.weight": ("attn_q_norm.weight", -1), "self_attn.k_norm.weight": ("attn_k_norm.weight", -1),
       "linear_attn.norm.weight": ("ssm_norm.weight", 0)}


class Weights:
  """The HF-named view: `shard(l)` is the weights object itself; `mm(sf, x, name)` applies a linear, `f32(name)` the small tensors."""
  def __init__(self, path):
    path = os.path.expanduser(path)
    if os.path.isdir(path): path = next(os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith("PTQ1_0.gguf"))
    self.g = Bonsai2GGUF(path); h = self.g.had
    assert h is not None and h.gdn_v_grouped and h.sign_mode == "explicit", "expected the Bonsai 2 Hadamard contract"
  def shard(self, l): return self
  @staticmethod
  def _split(name):
    p = "model.language_model.layers."; assert name.startswith(p), name
    l, rest = name[len(p):].split(".", 1); return int(l), rest
  def gguf_name(self, name):
    l, rest = self._split(name); return f"blk.{l}.{LIN[rest]}.weight", rest
  def mm(self, sf, x, name):
    t, rest = self.gguf_name(name); assert self.g.folded(t), t      # every trunk linear is folded in this checkpoint
    y = self.g.matmul(np.asarray(x, np.float32), t)
    return y[:, OUT_ROWS[rest]] if rest in OUT_ROWS else y
  def linear(self, sf, name):
    raise NotImplementedError("Bonsai 2 linears are stored in a rotated basis: use mm()")
  def f32(self, name):
    if name == "model.language_model.norm.weight": return self.g.f32("output_norm.weight") - 1.0
    l, rest = self._split(name)
    if rest in VEC: t, off = VEC[rest]; return self.g.f32(f"blk.{l}.{t}") + off
    if rest == "linear_attn.in_proj_a.weight": return self.g.f32(f"blk.{l}.ssm_alpha.weight")[PERM]
    if rest == "linear_attn.in_proj_b.weight": return self.g.f32(f"blk.{l}.ssm_beta.weight")[PERM]
    if rest == "linear_attn.A_log": return np.log(-self.g.f32(f"blk.{l}.ssm_a"))[PERM].astype(np.float32)
    if rest == "linear_attn.dt_bias": return self.g.f32(f"blk.{l}.ssm_dt.bias")[PERM]
    if rest == "linear_attn.conv1d.weight":
      c = self.g.f32(f"blk.{l}.ssm_conv1d.weight")                         # [C, 4]
      return np.concatenate([c[:QK], c[QK:][_rows(DV)]])[:, None, :]       # [C, 1, 4]
    raise KeyError(name)
  # ---- outside the layers
  def embed(self, ids):                                                    # the latent rows, then s * (Hb z): ~2.7 ms a row on the host,
    c = self.__dict__.setdefault("_emb", {})                               # so rows are kept (a verify pass re-embeds the drafted tokens)
    miss = [t for t in dict.fromkeys(ids) if t not in c]
    if miss:
      if len(c) + len(miss) > 4096: c.clear()
      c.update(zip(miss, self.g.embed(miss)))
    return np.stack([c[t] for t in ids])
  def final_norm(self, x):
    w = self.f32("model.language_model.norm.weight"); return x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)) * (1.0 + w)
  def lm_head(self, x): return self.g.matmul(np.asarray(x, np.float32), "output.weight", chunk=8192)

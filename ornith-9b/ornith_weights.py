"""Ornith 1.0 9B (GGUF, architecture `qwen35`) served under the Hugging Face Qwen3.5 names and conventions, so qwen3.8-27b's numpy
reference and packer read it unchanged.

The GGUF converter (its Qwen3.5 text model conversion) changed the
checkpoint, and this undoes it:
- the zero-centred RMSNorm weights (input / post-attention / q / k / final norms) are stored as 1 + w: w = stored - 1
  (the DeltaNet's gated `linear_attn.norm` is stored as is);
- A_log is stored as -exp(A_log): A_log = log(-stored); dt_bias is `ssm_dt.bias`; the conv kernel is [C, 4];
- with 32 value heads over 16 key heads, the V heads are reordered from grouped (value head h reads key head h // 2) to
  tiled (h' = r * 16 + g) in in_proj_qkv's V rows, in_proj_z, in_proj_a / _b, A_log, dt_bias, the conv's V channels and
  out_proj's columns: restored here (the kernels keep the grouped convention).
Linears are GGUF Q8_0 (int8 codes, an fp16 scale per 32 weights along K): `q8(name)` gives (codes, scales) for the packer,
`linear()` the dequantised float32 [N, K] for the reference.

`MTP` serves the multi-token-prediction head of Qwen3.5-9B (Ornith's base model; Ornith's own GGUF has none) from a GGUF that
keeps it (the last block's `nextn.*`, e.g. the bf16 conversion) under the HF names (`mtp.*`), for the speculative drafts."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_read import GGUF

NK, DK, NV, DV = 16, 128, 32, 128
REP = NV // NK
PERM = np.array([(h % REP) * NK + h // REP for h in range(NV)])            # HF (grouped) head h <- GGUF (tiled) head PERM[h]
def _rows(D): return (PERM[:, None] * D + np.arange(D)[None]).ravel()        # the row permutation of NV heads of D rows each
QK = 2 * NK * DK                                                           # in_proj_qkv: q | k rows are not reordered

LIN = {"linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate", "linear_attn.out_proj": "ssm_out",
       "linear_attn.in_proj_a": "ssm_alpha", "linear_attn.in_proj_b": "ssm_beta",
       "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
       "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}
VEC = {"input_layernorm.weight": ("attn_norm.weight", -1), "post_attention_layernorm.weight": ("post_attention_norm.weight", -1),
       "self_attn.q_norm.weight": ("attn_q_norm.weight", -1), "self_attn.k_norm.weight": ("attn_k_norm.weight", -1),
       "linear_attn.norm.weight": ("ssm_norm.weight", 0)}


class Weights:
  """The HF-named view: `shard(l)` is the weights object itself (names carry the layer); `W.linear(sf, name)`, `sf.f32(name)`."""
  def __init__(self, path):
    path = os.path.expanduser(path)
    if os.path.isdir(path): path = next(os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".gguf"))
    self.g = GGUF(path)
  def shard(self, l): return self
  @staticmethod
  def _split(name):
    p = "model.language_model.layers."; assert name.startswith(p), name
    l, rest = name[len(p):].split(".", 1); return int(l), rest
  def q8(self, name):
    """(codes int8 [N, K], scales float32 [N, K / 32]) of an HF-named linear, the V heads in HF order."""
    l, rest = self._split(name); q, d = self.g.q8(f"blk.{l}.{LIN[rest]}.weight")
    if rest == "linear_attn.in_proj_qkv":
      r = np.concatenate([np.arange(QK), QK + _rows(DV)]); q, d = q[r], d[r]
    elif rest == "linear_attn.in_proj_z": q, d = q[_rows(DV)], d[_rows(DV)]
    elif rest in ("linear_attn.in_proj_a", "linear_attn.in_proj_b"): q, d = q[PERM], d[PERM]
    elif rest == "linear_attn.out_proj":                                   # columns (K) in heads of DV = 4 Q8_0 blocks: whole blocks move
      c = _rows(DV); q = q[:, c]; d = d[:, c.reshape(-1, 32)[:, 0] // 32]
    return q, d
  def linear(self, sf, name):
    q, d = self.q8(name); return (q.astype(np.float32).reshape(q.shape[0], -1, 32) * d[..., None]).reshape(q.shape)
  def f32(self, name):
    if name == "model.language_model.norm.weight": return self.g.f32("output_norm.weight") - 1.0
    l, rest = self._split(name)
    if rest in VEC: t, off = VEC[rest]; return self.g.f32(f"blk.{l}.{t}") + off
    if rest in ("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"): return self.linear(self, name[:-len(".weight")])
    if rest == "linear_attn.A_log": return np.log(-self.g.f32(f"blk.{l}.ssm_a"))[PERM].astype(np.float32)
    if rest == "linear_attn.dt_bias": return self.g.f32(f"blk.{l}.ssm_dt.bias")[PERM]
    if rest == "linear_attn.conv1d.weight":
      c = self.g.f32(f"blk.{l}.ssm_conv1d.weight")                         # [C, 4]
      return np.concatenate([c[:QK], c[QK:][_rows(DV)]])[:, None, :]       # [C, 1, 4]
    raise KeyError(name)
  # ---- outside the layers
  def embed(self, ids): return self.g.q8_rows("token_embd.weight", list(ids))
  def final_norm(self, x):
    w = self.f32("model.language_model.norm.weight"); return x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6)) * (1.0 + w)
  def lm_head(self, x):
    q, d = self.g.q8("output.weight"); out = np.empty((x.shape[0], q.shape[0]), np.float32)
    for i in range(0, q.shape[0], 16384):
      w = (q[i:i + 16384].astype(np.float32).reshape(-1, q.shape[1] // 32, 32) * d[i:i + 16384, :, None]).reshape(-1, q.shape[1])
      out[:, i:i + w.shape[0]] = x @ w.T
    return out


# ---- the MTP head (the last block of a qwen35 GGUF converted with nextn_predict_layers = 1)
MTP_LIN = {"mtp.fc": "nextn.eh_proj", "mtp.layers.0.self_attn.q_proj": "attn_q", "mtp.layers.0.self_attn.k_proj": "attn_k",
           "mtp.layers.0.self_attn.v_proj": "attn_v", "mtp.layers.0.self_attn.o_proj": "attn_output", "mtp.layers.0.mlp.gate_proj": "ffn_gate",
           "mtp.layers.0.mlp.up_proj": "ffn_up", "mtp.layers.0.mlp.down_proj": "ffn_down"}
# every MTP norm is stored as 1 + w (the 27B's raw HF values minus these agree: enorm -0.46 / 0.52, hnorm -0.16 / 0.77, norm 1.25 / 2.43)
MTP_VEC = {"mtp.pre_fc_norm_embedding.weight": "nextn.enorm", "mtp.pre_fc_norm_hidden.weight": "nextn.hnorm", "mtp.norm.weight": "nextn.shared_head_norm",
           "mtp.layers.0.input_layernorm.weight": "attn_norm", "mtp.layers.0.post_attention_layernorm.weight": "post_attention_norm",
           "mtp.layers.0.self_attn.q_norm.weight": "attn_q_norm", "mtp.layers.0.self_attn.k_norm.weight": "attn_k_norm"}


class MTP:
  """The MTP head under its HF names: `linear(name)` float32 [N, K] (any GGUF type), `q8(name)` Q8_0 (codes, fp16-exact scales)
  for the packer (as stored when the GGUF has Q8_0, else quantised here: amax / 127 a block of 32), `f32(name)` the norms."""
  def __init__(self, path):
    self.g = GGUF(os.path.expanduser(path)); self.l = int(self.g.meta["qwen35.block_count"]) - 1
    assert int(self.g.meta.get("qwen35.nextn_predict_layers", 0)) == 1 and f"blk.{self.l}.nextn.eh_proj.weight" in self.g.tensors, "no MTP head in this GGUF"
  def _t(self, name, table): return f"blk.{self.l}.{table[name[:-len('.weight')] if table is MTP_LIN and name.endswith('.weight') else name]}.weight"
  def linear(self, name): return self.g.f32(self._t(name, MTP_LIN))
  def f32(self, name): return self.g.f32(self._t(name, MTP_VEC)) - 1.0
  def q8(self, name):
    t = self._t(name, MTP_LIN)
    if self.g.raw(t)[0] == "Q8_0": return self.g.q8(t)
    w = self.linear(name); N, K = w.shape; b = w.reshape(N, K // 32, 32)
    d = (np.abs(b).max(-1) / 127.0).astype(np.float16).astype(np.float32)            # the scale as the packer will store it (fp16)
    q = np.clip(np.rint(b / np.where(d == 0, 1, d)[..., None]), -127, 127).astype(np.int8).reshape(N, K)
    return q, d

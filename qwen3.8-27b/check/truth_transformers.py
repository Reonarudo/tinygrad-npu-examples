"""Ground truth for layers 0 (Gated DeltaNet) and 3 (attention) from transformers' own modules with the checkpoint's dequantised weights."""
import json, struct, sys, numpy as np, torch
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SHARDS = os.environ.get("QWEN_SHARDS", "/tmp/qwen38")   # a folder with config.json and layers-0 / layers-3 .safetensors
import qwen38_ref as R
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
cfg = AutoConfig.from_pretrained("" + SHARDS + "/config.json").text_config
cfg._attn_implementation = "eager"
torch.manual_seed(0); n = 12
x = (torch.randn(1, n, R.H) * 1.0).float()
rot = m.Qwen3_5TextRotaryEmbedding(cfg)
pos = torch.arange(n)[None].expand(3, 1, n)          # text only: the three mrope axes coincide
cos, sin = rot(x, pos)
W = R.Weights(SHARDS)     # config.json + layers-0/3 shards
def load(layer, l):
  sf, p = W.shard(l), f"model.language_model.layers.{l}."
  sd = {}
  for k in sf.h:
    if not k.startswith(p) or k.endswith("weight_scale_inv"): continue
    short = k[len(p):]
    if sf.h[k]["dtype"] == "F8_E4M3": sd[short] = torch.from_numpy(W.linear(sf, k[:-len(".weight")]))
    else: sd[short] = torch.from_numpy(sf.f32(k))
  missing, unexpected = layer.load_state_dict(sd, strict=False)
  print("layer", l, "missing", missing, "unexpected", unexpected)
out = {"x": x[0].numpy()}
for l in (0, 3):
  layer = m.Qwen3_5DecoderLayer(cfg, l).float().eval(); load(layer, l)
  mask = torch.full((1, 1, n, n), float("-inf")).triu(1) if l == 3 else None
  with torch.no_grad():
    y = layer(x, attention_mask=mask, position_embeddings=(cos, sin), position_ids=pos)
  y = y[0] if isinstance(y, tuple) else y
  out[f"y{l}"] = y[0].float().numpy(); print("layer", l, "out", out[f"y{l}"].shape, "|y| max", float(np.abs(out[f"y{l}"]).max()))
np.savez("" + SHARDS + "/truth.npz", **out); print("saved", SHARDS + "/truth.npz")

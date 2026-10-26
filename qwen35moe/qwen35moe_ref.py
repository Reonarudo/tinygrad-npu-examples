#!/usr/bin/env python3
"""Qwen3.5-35B-A3B (and 3.6) in float32 numpy from its GGUF: qwen3.8-27b's Qwen3.5 reference (QWEN_MODEL=qwen3.5-35b-a3b) with the
MoE block from qwen35moe_weights. Greedy generation, the chat template with thinking off.

    QWEN35MOE_GGUF=/mnt/ssd/models/Qwen3.5-35B-A3B-Q4_K_M.gguf python3 qwen35moe_ref.py "prompt" [--max-new 16]
"""
import argparse, os, sys, time
os.environ["QWEN_MODEL"] = "qwen3.5-35b-a3b"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, "..", "qwen3.8-27b")]
import numpy as np                                                       # noqa: E402
import qwen38_ref as R                                                   # noqa: E402

DEFAULT = os.environ.get("QWEN35MOE_GGUF", "/mnt/ssd/models/Qwen3.5-35B-A3B-Q4_K_M.gguf")
IM_START, IM_END = 248044, 248046                                        # <|im_start|>, <|im_end|> (the GGUF's eos)


def chat_text(prompt, thinking=False):
  return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n" + ("" if thinking else "<think>\n\n</think>\n\n")


PROMPT = "What is the capital of Portugal? Answer in one sentence."


def forward_layer(W, l, x, pos0, states):
  """x [n, H] through layer l at positions pos0 ..; `states[l]` (a (S, conv_state) pair or a {"kv": (K, V)} dict) advanced in place."""
  n = x.shape[0]; pos = np.arange(pos0, pos0 + n)
  if R.LAYER_TYPES[l] == "full": return attention_cached(W, l, x, pos, states.setdefault(l, {}))
  s = states.get(l); x, s2 = R.gdn_layer(W, l, x, state=None if s is None else s[0], conv_state=None if s is None else s[1]); states[l] = s2
  return x


def forward(W, x, pos0, states):
  """x [n, H] through every layer; `states` carries the DeltaNet states / conv states and the attention caches per layer."""
  for l in range(R.NL): x = forward_layer(W, l, x, pos0, states)
  return x


def attention_cached(W, l, x, pos, st):
  """qwen38_ref.attention_layer with a K / V cache (the reference's own is prefill-only)."""
  import math
  sf, p = W.shard(l), f"model.language_model.layers.{l}."; n = x.shape[0]
  h = R.rms(x, sf.f32(p + "input_layernorm.weight"))
  qg = R.mm(W, sf, h, p + "self_attn.q_proj").reshape(n, R.NH, 2 * R.HD); q, gate = qg[..., :R.HD], qg[..., R.HD:]
  k = R.mm(W, sf, h, p + "self_attn.k_proj").reshape(n, R.NKV, R.HD); v = R.mm(W, sf, h, p + "self_attn.v_proj").reshape(n, R.NKV, R.HD)
  q = R.rms(q, sf.f32(p + "self_attn.q_norm.weight")); k = R.rms(k, sf.f32(p + "self_attn.k_norm.weight")); q, k = R.rope(q, pos), R.rope(k, pos)
  K0, V0 = st.get("kv", (np.zeros((0, R.NKV, R.HD), np.float32),) * 2); Kc, Vc = np.concatenate([K0, k]), np.concatenate([V0, v]); st["kv"] = (Kc, Vc)
  T = len(Kc); g = R.NH // R.NKV; o = np.empty((n, R.NH, R.HD), np.float32)
  allowed = np.arange(T)[None] <= pos[:, None]
  for hh in range(R.NH):
    a = (q[:, hh] @ Kc[:, hh // g].T) / np.float32(math.sqrt(R.HD)); a = np.where(allowed, a, -np.inf)
    a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True); o[:, hh] = a @ Vc[:, hh // g]
  o = o * (1.0 / (1.0 + np.exp(-gate)))
  x = x + R.mm(W, sf, o.reshape(n, R.NH * R.HD), p + "self_attn.o_proj")
  return x + R.mlp(W, sf, p, R.rms(x, sf.f32(p + "post_attention_layernorm.weight")))


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("prompt", nargs="?", default=PROMPT)
  ap.add_argument("--max-new", type=int, default=12); ap.add_argument("--gguf", default=DEFAULT); ap.add_argument("--ids", default="")
  ap.add_argument("--tok", default=os.environ.get("QWEN35MOE_TOK", "")); a = ap.parse_args()
  W = R.Weights(a.gguf); tok = None
  if a.ids: ids = [int(v) for v in a.ids.split(",")]
  else:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.tok, "tokenizer.json")); ids = tok.encode(chat_text(a.prompt), add_special_tokens=False).ids
  print(f"{len(ids)} prompt tokens: {ids}", flush=True); t0 = time.perf_counter(); states = {}
  x = forward(W, W.embed(ids), 0, states); out = []; pos = len(ids)
  for _ in range(a.max_new):
    t = int(np.argmax(W.lm_head(W.final_norm(x[-1:]))[0])); out.append(t); print(t, repr(tok.decode([t])) if tok else "", flush=True)
    if t == IM_END: break
    x = forward(W, W.embed([t]), pos, states); pos += 1
  print(f"ids: {out}\n{time.perf_counter() - t0:.1f} s"); 
  if tok: print(repr(tok.decode(out)))


if __name__ == "__main__": main()

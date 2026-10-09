#!/usr/bin/env python3
"""Ideogram 4's prompt conditioning, once per image, on the host CPU (numpy): the Qwen3-VL-8B text
encoder's decoder layers over the chat-formatted prompt, the hidden states tapped after layers
(0, 3, ..., 33, 35) concatenated per token as `[n, 4096 x 13]` (feature d * 13 + tap: the order the conditional
transformer's `llm_cond_proj` expects), then the conditional transformer's `llm_cond_norm` +
`llm_cond_proj` + the text indicator embedding -> the text rows `[n, 4608]` that open the packed
sequence. ⚠️ The pipeline left-pads to 2048 tokens; the pads are a separate attention segment and the
encoder is causal, so dropping them is exact (the text positions are 0..n-1 either way).

    python3 ideogram4_text.py --prompt "..." --out ~/ideogram4/text_rows.npz
        --te ~/ideogram4/text_encoder --tok ~/ideogram4/tokenizer --weights ~/ideogram4/transformer

Weights: E4M3 codes with per-row F32 `weight_scale` (the fp8 checkpoint), BF16 norms and embeddings;
computed in float32 (the pipeline runs bf16: expect ~1e-2 relative differences in the features).
"""
import argparse, json, math, os, struct, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ideogram4_weights import Ideogram4Weights, bf16_to_f32, _E4M3_TABLE   # noqa: E402

TAPS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)
LLM_TOKEN, IMAGE = 3, 2


class SafeDir:
    """A (possibly sharded) safetensors checkpoint, memory-mapped; tensors by name as float32."""
    def __init__(self, folder):
        self.files = {}; self.where = {}
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".safetensors"): continue
            p = os.path.join(folder, fn)
            with open(p, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))
            h.pop("__metadata__", None)
            self.files[p] = (8 + n, h, np.memmap(p, np.uint8, "r"))
            for k in h: self.where[k] = p
    def raw(self, name):
        p = self.where[name]; base, h, mm = self.files[p]; e = h[name]; a, b = e["data_offsets"]
        return e["dtype"], e["shape"], mm[base + a:base + b]
    def get(self, name):
        dt, shape, r = self.raw(name)
        if dt == "BF16": return bf16_to_f32(np.frombuffer(r, np.uint16)).reshape(shape)
        if dt == "F32": return np.frombuffer(r, np.float32).reshape(shape).copy()
        if dt == "F8_E4M3": return _E4M3_TABLE[np.frombuffer(r, np.uint8)].reshape(shape)
        raise ValueError(name + ": " + dt)
    def linear(self, prefix):
        w = self.get(prefix + ".weight")
        if prefix + ".weight_scale" in self.where: w = w * self.get(prefix + ".weight_scale").reshape(-1, 1)
        return w.astype(np.float32)


def rms(x, w, eps): return (x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + np.float32(eps)))) * w


def rot_half(x): h = x.shape[-1] // 2; return np.concatenate([-x[..., h:], x[..., :h]], -1)


def encode(te_dir, token_ids, log=print):
    """Token ids `[n]` -> the 13 tapped hidden states `[13, n, 4096]` (float32)."""
    cfg = json.load(open(os.path.join(te_dir, "config.json")))["text_config"]
    H, NHq, NKV, D = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    eps, theta, L = cfg["rms_norm_eps"], cfg["rope_parameters"]["rope_theta"], cfg["num_hidden_layers"]
    S = SafeDir(te_dir); p = "language_model."
    emb_dt, emb_shape, emb_raw = S.raw(p + "embed_tokens.weight")
    assert emb_dt == "BF16", emb_dt
    row = emb_shape[1] * 2
    x = np.stack([bf16_to_f32(np.frombuffer(emb_raw[t * row:(t + 1) * row], np.uint16)) for t in token_ids]).astype(np.float32)
    n = len(token_ids)
    inv = 1.0 / (theta ** (np.arange(0, D, 2, dtype=np.float64) / D))
    f = np.arange(n, dtype=np.float64)[:, None] * inv[None]            # text positions: t = h = w, so the interleaved MRoPE is plain RoPE
    emb = np.concatenate([f, f], -1); cos, sin = np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)
    causal = np.triu(np.full((n, n), -np.inf, np.float32), 1)
    taps = {}
    for l in range(L):
        t0 = time.perf_counter(); q_ = f"{p}layers.{l}."
        h = rms(x, S.get(q_ + "input_layernorm.weight"), eps)
        q = (h @ S.linear(q_ + "self_attn.q_proj").T).reshape(n, NHq, D)
        k = (h @ S.linear(q_ + "self_attn.k_proj").T).reshape(n, NKV, D)
        v = (h @ S.linear(q_ + "self_attn.v_proj").T).reshape(n, NKV, D)
        q = rms(q, S.get(q_ + "self_attn.q_norm.weight"), eps); k = rms(k, S.get(q_ + "self_attn.k_norm.weight"), eps)
        q = q * cos[:, None] + rot_half(q) * sin[:, None]; k = k * cos[:, None] + rot_half(k) * sin[:, None]
        o = np.empty((n, NHq, D), np.float32); g = NHq // NKV
        for hh in range(NHq):
            a = (q[:, hh] @ k[:, hh // g].T) / np.float32(math.sqrt(D)) + causal
            a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
            o[:, hh] = a @ v[:, hh // g]
        x = x + o.reshape(n, NHq * D) @ S.linear(q_ + "self_attn.o_proj").T
        h = rms(x, S.get(q_ + "post_attention_layernorm.weight"), eps)
        gt = h @ S.linear(q_ + "mlp.gate_proj").T; up = h @ S.linear(q_ + "mlp.up_proj").T
        x = x + ((gt / (1.0 + np.exp(-gt))) * up) @ S.linear(q_ + "mlp.down_proj").T
        if l in TAPS: taps[l] = x.copy()
        log("   text encoder layer %2d: %.1f s  |x| max %.3g" % (l, time.perf_counter() - t0, float(np.abs(x).max())))
    return np.stack([taps[l] for l in TAPS])


def tokenize(tok_dir, prompt):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))
    text = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"   # chat_template.jinja, one user text message, add_generation_prompt
    return tk.encode(text, add_special_tokens=False).ids, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--te", default="~/ideogram4/text_encoder"); ap.add_argument("--tok", default="~/ideogram4/tokenizer")
    ap.add_argument("--weights", default="~/ideogram4/transformer", help="the conditional transformer's checkpoint folder, or a `cond_proj.npz` (norm, codes, scale, bias, ind)")
    ap.add_argument("--prompt-file", default=None, help="read the prompt (e.g. a JSON caption) from a file")
    a = ap.parse_args()
    if a.prompt_file: a.prompt = open(os.path.expanduser(a.prompt_file)).read().strip()
    ids, text = tokenize(a.tok, a.prompt)
    print("== prompt %r -> %d tokens: %s" % (a.prompt, len(ids), ids), flush=True)
    t0 = time.perf_counter()
    taps = encode(a.te, ids, log=lambda s: print(s, flush=True))                 # [13, n, 4096]
    feat = np.ascontiguousarray(taps.transpose(1, 2, 0)).reshape(len(ids), -1)    # [n, 4096 * 13]: feature d * 13 + tap
    if a.weights.endswith(".npz"):
        z = np.load(os.path.expanduser(a.weights))
        nw, pb, ind = z["norm"], z["bias"], z["ind"]; pw = _E4M3_TABLE[z["codes"]] * z["scale"][:, None]
    else:
        W = Ideogram4Weights(a.weights)
        nw = W.get("llm_cond_norm.weight"); pw = W.linear("llm_cond_proj"); pb = W.get("llm_cond_proj.bias")
        ind = W.get("embed_image_indicator.weight")                                # [2, 4608]: row 0 = not an output-image token
    rows = rms(feat.astype(np.float32), nw, 1e-6) @ pw.T + pb + ind[0]
    np.savez(os.path.expanduser(a.out), rows=rows.astype(np.float32), ids=np.array(ids), feat_absmax=np.abs(feat).max(), text=text)
    print("   text rows %s (|max| %.3g) in %.0f s -> %s" % (rows.shape, float(np.abs(rows).max()), time.perf_counter() - t0, a.out), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The Bonsai 2 / Qwen3.5 vision tower in numpy, from the mmproj GGUF (BF16 or Q8_0), as llama.cpp's clip_graph_qwen3vl computes it:
  image -> smart resize (multiples of 32, min / max pixels) -> normalise ((x / 255 - 0.5) / 0.5) -> 16 x 16 patches in 2 x 2 block
  order -> patch embedding (the two temporal conv weights summed: a still image is both frames) + bias + the learned position
  embedding interpolated bilinearly (align corners) to the patch grid -> 24 blocks (LN1, attention with 2-D rope on half the head
  dims (16 freqs for y, 16 for x), LN2, GELU MLP) -> post LN -> the merger: 4 consecutive patches (a 2 x 2 block) concatenated ->
  mm.0 -> GELU -> mm.2 -> [n_tokens, H] embeddings (H = the language model's width) for the language model, plus the token grid (nx, ny) for m-RoPE.
  python3 vit_ref.py IMAGE [--mmproj PATH] [--max-tokens 1024] [--out emb.npy]"""
import argparse, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "ornith-9b"))
import gguf_read as G                                                    # noqa: E402

PATCH, MERGE, THETA_V = 16, 2, 10000.0


def smart_size(w, h, min_tokens=8, max_tokens=1024):
  """llama.cpp's calc_size_preserved_ratio with align 32 and the token limits (a token = a 32 x 32 pixel block)."""
  f = PATCH * MERGE; r = lambda x: max(f, int(round(x / f)) * f); ce = lambda x: int(math.ceil(x / f)) * f; fl = lambda x: max(f, int(math.floor(x / f)) * f)
  wb, hb = r(w), r(h); mx, mn = max_tokens * f * f, min_tokens * f * f
  if hb * wb > mx: beta = math.sqrt(h * w / mx); hb, wb = fl(h / beta), fl(w / beta)
  elif hb * wb < mn: beta = math.sqrt(mn / (h * w)); hb, wb = ce(h * beta), ce(w * beta)
  return wb, hb


def load_image(path, max_tokens):
  from PIL import Image
  im = Image.open(path).convert("RGB"); wb, hb = smart_size(im.width, im.height, max_tokens=max_tokens)
  im = im.resize((wb, hb), Image.BICUBIC); x = (np.asarray(im, np.float32) / 255.0 - 0.5) / 0.5    # [H, W, 3]
  return x, wb // PATCH, hb // PATCH


def patches_in_block_order(x, pw, ph):
  """[H, W, 3] -> [n_patches, 3 * 16 * 16] (channel-major within a patch, as the conv weight is [out][c][ky][kx]), the patches in
  2 x 2 block order: block (by, bx) raster, then (dy, dx) within it."""
  p = x.reshape(ph, PATCH, pw, PATCH, 3).transpose(0, 2, 4, 1, 3)        # [ph, pw, c, ky, kx]
  p = p.reshape(ph // 2, 2, pw // 2, 2, 3 * PATCH * PATCH).transpose(0, 2, 1, 3, 4)   # [by, bx, dy, dx, 768]
  return np.ascontiguousarray(p).reshape(-1, 3 * PATCH * PATCH)


def pos_emb_interp(pe, pw, ph):
  """The learned [S*S, C] grid bilinearly interpolated (align corners) to [ph, pw] -> [ph*pw, C] in raster order."""
  S = int(round(math.sqrt(pe.shape[0]))); g = pe.reshape(S, S, -1).astype(np.float64)
  ys = np.linspace(0, S - 1, ph) if ph > 1 else np.zeros(1); xs = np.linspace(0, S - 1, pw) if pw > 1 else np.zeros(1)
  y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int); y1 = np.minimum(y0 + 1, S - 1); x1 = np.minimum(x0 + 1, S - 1)
  fy = (ys - y0)[:, None, None]; fx = (xs - x0)[None, :, None]
  out = (g[y0][:, x0] * (1 - fy) * (1 - fx) + g[y0][:, x1] * (1 - fy) * fx + g[y1][:, x0] * fy * (1 - fx) + g[y1][:, x1] * fy * fx)
  return out.reshape(ph * pw, -1).astype(np.float32)


def block_order_index(pw, ph):
  """The raster index of each patch in 2 x 2 block order (to reorder raster arrays)."""
  idx = []
  for by in range(0, ph, 2):
    for bx in range(0, pw, 2):
      for dy in range(2):
        for dx in range(2): idx.append((by + dy) * pw + bx + dx)
  return np.array(idx)


def layer_norm(x, w, b, eps=1e-6):
  m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True); return (x - m) / np.sqrt(v + eps) * w + b


def gelu(x):   # ggml's tanh approximation
  return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def vision_rope(q, ys, xs):
  """q [n, heads, D]: the 2-D rope as pairs (i, i + D/2): pairs 0 .. D/4 - 1 turn by y x theta_i, the next D/4 by x x theta_(i - D/4),
  theta_i = 10000^(-2 i / (D / 2)) (the 2B's 64-dim heads: 16 + 16 pairs over 32; the 9B's 72: 18 + 18 over 36)."""
  D = q.shape[-1]; hr, q4 = D // 2, D // 4; inv = THETA_V ** (-np.arange(0, q4, dtype=np.float64) * 2 / hr)
  ang = np.concatenate([ys[:, None] * inv[None], xs[:, None] * inv[None]], 1)          # [n, D/2]
  c, s = np.cos(ang).astype(np.float32)[:, None, :], np.sin(ang).astype(np.float32)[:, None, :]
  a, b = q[..., :hr], q[..., hr:]; return np.concatenate([a * c - b * s, a * s + b * c], -1)


class ViT:
  def __init__(self, mmproj):
    self.g = G.GGUF(mmproj); md = self.g.meta if hasattr(self.g, "meta") else self.g.kv
    self.n_layer = int(md["clip.vision.block_count"]); self.n_head = int(md["clip.vision.attention.head_count"]); self.eps = float(md["clip.vision.attention.layer_norm_epsilon"])
    self.C = int(md["clip.vision.embedding_length"]); t = lambda n: np.asarray(self.g.f32(n), np.float32)
    self.W_patch = (t("v.patch_embd.weight").reshape(self.C, -1) + t("v.patch_embd.weight.1").reshape(self.C, -1))   # [1024, 768]
    self.b_patch = t("v.patch_embd.bias"); self.pos = t("v.position_embd.weight")                                        # [S*S, 1024]
    self.L = [{k: t(f"v.blk.{i}.{k}") for k in ("ln1.weight", "ln1.bias", "attn_qkv.weight", "attn_qkv.bias", "attn_out.weight", "attn_out.bias", "ln2.weight", "ln2.bias", "ffn_up.weight", "ffn_up.bias", "ffn_down.weight", "ffn_down.bias")} for i in range(self.n_layer)]
    self.post_w, self.post_b = t("v.post_ln.weight"), t("v.post_ln.bias")
    self.mm0_w, self.mm0_b, self.mm2_w, self.mm2_b = t("mm.0.weight"), t("mm.0.bias"), t("mm.2.weight"), t("mm.2.bias")
  def forward(self, x, pw, ph, dump=None):
    """x [H, W, 3] normalised -> (embeddings [pw ph / 4, H], (nx, ny) = (pw / 2, ph / 2))."""
    P = patches_in_block_order(x, pw, ph); n = P.shape[0]; H = self.n_head; D = self.C // H
    idx = block_order_index(pw, ph); ys = (idx // pw).astype(np.float64); xs = (idx % pw).astype(np.float64)
    h = P @ self.W_patch.T + self.b_patch + pos_emb_interp(self.pos, pw, ph)[idx]
    if dump is not None: dump["inp"] = h.copy()
    for i, L in enumerate(self.L):
      a = layer_norm(h, L["ln1.weight"], L["ln1.bias"], self.eps)
      qkv = a @ L["attn_qkv.weight"].T + L["attn_qkv.bias"]
      q, k, v = (qkv[:, j * self.C:(j + 1) * self.C].reshape(n, H, D) for j in range(3))
      q, k = vision_rope(q, ys, xs), vision_rope(k, ys, xs)
      s = np.einsum("nhd,mhd->hnm", q, k) / math.sqrt(D); s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
      o = np.einsum("hnm,mhd->nhd", s, v).reshape(n, self.C)
      h = h + o @ L["attn_out.weight"].T + L["attn_out.bias"]
      a = layer_norm(h, L["ln2.weight"], L["ln2.bias"], self.eps)
      h = h + gelu(a @ L["ffn_up.weight"].T + L["ffn_up.bias"]) @ L["ffn_down.weight"].T + L["ffn_down.bias"]
      if dump is not None and i in (0, self.n_layer - 1): dump[f"blk{i}"] = h.copy()
    h = layer_norm(h, self.post_w, self.post_b, self.eps)
    m = h.reshape(n // 4, 4 * self.C)                                       # a 2 x 2 block's four patches, in (dy, dx) order
    e = gelu(m @ self.mm0_w.T + self.mm0_b) @ self.mm2_w.T + self.mm2_b
    return e.astype(np.float32), (pw // 2, ph // 2)


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("image"); ap.add_argument("--mmproj", default=os.environ.get("BONSAI_MMPROJ", "/tmp/vision/Ternary-Bonsai-2-2B-mmproj-BF16.gguf"))
  ap.add_argument("--max-tokens", type=int, default=1024); ap.add_argument("--out", default=None); a = ap.parse_args()
  t0 = time.perf_counter(); V = ViT(a.mmproj); print(f"mmproj: {V.n_layer} blocks, C {V.C}, {V.n_head} heads; loaded in {time.perf_counter() - t0:.1f} s", flush=True)
  x, pw, ph = load_image(a.image, a.max_tokens); print(f"image -> {pw * PATCH} x {ph * PATCH}: {pw} x {ph} patches, {pw * ph // 4} tokens ({pw // 2} x {ph // 2})", flush=True)
  t0 = time.perf_counter(); e, grid = V.forward(x, pw, ph); print(f"embeddings {e.shape} in {time.perf_counter() - t0:.1f} s; |e| mean {np.abs(e).mean():.3f} max {np.abs(e).max():.2f}; grid {grid}", flush=True)
  if a.out: np.save(a.out, e); print("saved", a.out)


if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Ideogram 4's VAE decoder in float32 numpy: packed latents `[T, 128]` -> an RGB image. Written from the checkpoint's `vae`
folder -- its config.json (`AutoencoderKLFlux2`: latent_channels 32, patch_size 2 x 2, block_out_channels, layers_per_block)
and its tensors' names and shapes -- and the standard KL-VAE decoder architecture they describe:

    z = latents x sqrt(running_var + eps) + running_mean   (the latents' per-channel BatchNorm statistics, undone)
    z: [T, 128] -> [32, gh*2, gw*2]                      (a token holds a 2 x 2 patch: channel (py * 2 + px) * 32 + c)
    post_quant_conv (1x1) -> conv_in (3x3, 32 -> 512)
    mid: resnet, attention (1 head of 512, GroupNorm, residual), resnet
    up blocks over [512, 512, 256, 128]: 3 resnets each, nearest 2x + 3x3 conv between blocks
    GroupNorm(32, eps 1e-6) -> silu -> conv_out (3x3, 128 -> 3) -> (x/2 + 0.5) clipped to [0, 1]

A resnet: GN -> silu -> conv1 -> GN -> silu -> conv2, + the input (through the 1x1 `conv_shortcut` when the width changes).
Convolutions by im2col on a workstation's BLAS: a 256 px image decodes in seconds; the board's numpy (no BLAS) would take
minutes.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ideogram4_weights import Ideogram4Weights   # noqa: E402

EPS = 1e-6


def silu(x): return x / (1.0 + np.exp(-x))


def group_norm(x, g, b, groups=32, eps=EPS):
    """x `[C, H, W]`."""
    c, h, w = x.shape
    xg = x.reshape(groups, -1)
    m = xg.mean(1, keepdims=True); v = xg.var(1, keepdims=True)
    return ((xg - m) / np.sqrt(v + eps)).reshape(c, h, w) * g[:, None, None] + b[:, None, None]


def conv2d(x, w, b, pad):
    """x `[C, H, W]`, w `[O, C, k, k]` -> `[O, H, W]` (stride 1, SAME for k=3 / pad=1, k=1 / pad=0)."""
    o, c, k, _ = w.shape
    _, h, wd = x.shape
    if k == 1: return (w.reshape(o, c) @ x.reshape(c, -1)).reshape(o, h, wd) + b[:, None, None]
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    # im2col in horizontal strips of <= ~256 MB of columns (a 1024-px image's 128-channel convs would need 4.8 GB at once)
    rows = max(1, min(h, (256 << 20) // (c * k * k * wd * 4)))
    y = np.empty((o, h, wd), np.float32); wm = w.reshape(o, -1)
    for r0 in range(0, h, rows):
        hr = min(rows, h - r0)
        cols = np.empty((c, k, k, hr, wd), np.float32)
        for i in range(k):
            for j in range(k): cols[:, i, j] = xp[:, r0 + i:r0 + i + hr, j:j + wd]
        y[:, r0:r0 + hr] = (wm @ cols.reshape(c * k * k, hr * wd)).reshape(o, hr, wd)
    return y + b[:, None, None]


class VAEDecoder:
    def __init__(self, folder="~/ideogram4/vae"):
        self.W = W = Ideogram4Weights(folder)
        self.cfg = W.config
        # a packed channel's statistics: the latents are stored normalised by them
        self.lat_mean, self.lat_scale = W.get("bn.running_mean"), np.sqrt(W.get("bn.running_var") + self.cfg["batch_norm_eps"])
        self.patch = tuple(self.cfg["patch_size"]); self.latent_channels = self.cfg["latent_channels"]

    def _w(self, n): return self.W.get(n)

    def resnet(self, x, p):
        h = conv2d(silu(group_norm(x, self._w(p + ".norm1.weight"), self._w(p + ".norm1.bias"))), self._w(p + ".conv1.weight"), self._w(p + ".conv1.bias"), 1)
        h = conv2d(silu(group_norm(h, self._w(p + ".norm2.weight"), self._w(p + ".norm2.bias"))), self._w(p + ".conv2.weight"), self._w(p + ".conv2.bias"), 1)
        if self.W.has(p + ".conv_shortcut.weight"): x = conv2d(x, self._w(p + ".conv_shortcut.weight"), self._w(p + ".conv_shortcut.bias"), 0)
        return x + h

    def attention(self, x, p):
        c, h, w = x.shape
        hs = group_norm(x, self._w(p + ".group_norm.weight"), self._w(p + ".group_norm.bias")).reshape(c, -1).T      # [HW, C]
        q = hs @ self._w(p + ".to_q.weight").T + self._w(p + ".to_q.bias")
        k = hs @ self._w(p + ".to_k.weight").T + self._w(p + ".to_k.bias")
        v = hs @ self._w(p + ".to_v.weight").T + self._w(p + ".to_v.bias")
        pv = np.empty_like(q)
        for q0 in range(0, q.shape[0], 2048):              # query blocks: a 1024-px image's mid block has 16384 tokens
            a = (q[q0:q0 + 2048] @ k.T) / np.sqrt(np.float32(c)); a = a - a.max(-1, keepdims=True); e = np.exp(a)
            pv[q0:q0 + 2048] = (e / e.sum(-1, keepdims=True)) @ v
        o = pv @ self._w(p + ".to_out.0.weight").T + self._w(p + ".to_out.0.bias")
        return x + o.T.reshape(c, h, w)

    def unpack(self, latents, grid):
        """`[T, 128]` packed -> `[32, gh*2, gw*2]`: the statistics restored, then each token's patch put in its place -- the
        token of grid cell (i, j) holds pixel (i ph + py, j pw + px) of channel c at (py pw + px) C + c."""
        gh, gw = grid; ph, pw = self.patch; C = self.latent_channels
        tok = (latents.astype(np.float32) * self.lat_scale + self.lat_mean).reshape(gh, gw, ph * pw, C)
        z = np.empty((C, gh * ph, gw * pw), np.float32)
        for py in range(ph):
            for px in range(pw): z[:, py::ph, px::pw] = tok[:, :, py * pw + px].transpose(2, 0, 1)
        return z

    def decode(self, latents, grid, log=None):
        z = self.unpack(latents, grid)
        z = conv2d(z, self._w("post_quant_conv.weight"), self._w("post_quant_conv.bias"), 0)
        x = conv2d(z, self._w("decoder.conv_in.weight"), self._w("decoder.conv_in.bias"), 1)
        x = self.resnet(x, "decoder.mid_block.resnets.0")
        x = self.attention(x, "decoder.mid_block.attentions.0")
        x = self.resnet(x, "decoder.mid_block.resnets.1")
        nblk = len(self.cfg["block_out_channels"])
        for i in range(nblk):
            for r in range(self.cfg["layers_per_block"] + 1): x = self.resnet(x, f"decoder.up_blocks.{i}.resnets.{r}")
            if i < nblk - 1:
                x = x.repeat(2, axis=1).repeat(2, axis=2)
                x = conv2d(x, self._w(f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"), self._w(f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"), 1)
            if log: log("   up block %d -> %s" % (i, x.shape))
        x = silu(group_norm(x, self._w("decoder.conv_norm_out.weight"), self._w("decoder.conv_norm_out.bias")))
        x = conv2d(x, self._w("decoder.conv_out.weight"), self._w("decoder.conv_out.bias"), 1)
        return np.clip(x / 2 + 0.5, 0, 1).transpose(1, 2, 0)          # [H, W, 3] in [0, 1]


def save_png(img01, path):
    """`[H, W, 3]` in [0, 1] -> an 8-bit PNG (zlib + the PNG chunks; no PIL needed)."""
    import struct, zlib
    a = (np.clip(img01, 0, 1) * 255 + 0.5).astype(np.uint8); h, w, _ = a.shape
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


if __name__ == "__main__":
    import time
    D = VAEDecoder()
    gh = gw = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    z = np.random.RandomState(0).randn(gh * gw, 128).astype(np.float32)
    t0 = time.perf_counter(); img = D.decode(z, (gh, gw), log=print)
    print("decoded %s in %.1f s; range %.3f..%.3f" % (img.shape, time.perf_counter() - t0, img.min(), img.max()))
    save_png(img, "/tmp/vae_noise.png"); print("wrote /tmp/vae_noise.png")

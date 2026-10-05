#!/usr/bin/env python3
"""Ideogram 4's fp8 checkpoint (`ideogram-ai/ideogram-4-fp8`) read with numpy alone: the safetensors
header, the `F8_E4M3` matrices with their per-OUTPUT-ROW `weight_scale` (F32 `[out]`), the BF16
norms / biases / embeddings -- every tensor as float32, no torch anywhere.

    layout on disk (both transformers, 670 tensors each, 9.29 GB):
      <linear>.weight        F8_E4M3 [out, in]      dequantized = fp8 * weight_scale[out, None]
      <linear>.weight_scale  F32     [out]
      <linear>.bias          BF16    [out]           (adaln / input_proj / final / t_embedding only)
      *_norm*.weight         BF16    [dim]
      embed_image_indicator  BF16    [2, 4608]

E4M3 here is the OCP `float8_e4m3fn` (no infinities, 0x7F / 0xFF NaN, bias 7, subnormals at
exponent 0). The decode is a 256-entry table; `_E4M3_TABLE` is asserted against the defining
values (1.0 = 0x38, 1.5 = 0x3C, 2^-6 = 0x08, 448 = 0x7E, -1 = 0xB8).

    W = Ideogram4Weights("~/ideogram4/unconditional_transformer")
    w = W.linear("layers.0.attention.qkv")          # float32 [13824, 4608]
    g = W.get("layers.0.attention_norm1.weight")    # float32 [4608]
"""
import json, os, struct
import numpy as np


def _e4m3_table() -> np.ndarray:
    t = np.empty(256, np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF; m = b & 7
        if e == 0: v = (m / 8.0) * 2.0 ** -6
        elif e == 15 and m == 7: v = np.nan
        else: v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = s * v
    return t


_E4M3_TABLE = _e4m3_table()
assert _E4M3_TABLE[0x38] == 1.0 and _E4M3_TABLE[0x3C] == 1.5 and _E4M3_TABLE[0x08] == 2.0 ** -6 \
    and _E4M3_TABLE[0x7E] == 448.0 and _E4M3_TABLE[0xB8] == -1.0 and np.isnan(_E4M3_TABLE[0x7F]) and _E4M3_TABLE[0x00] == 0.0


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    """uint16 bf16 bits -> float32 (the high half of the float32 word)."""
    return (raw.astype(np.uint32) << 16).view(np.float32)


class Ideogram4Weights:
    def __init__(self, folder: str):
        self.folder = os.path.expanduser(folder)
        self.path = os.path.join(self.folder, "diffusion_pytorch_model.safetensors")
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.base = 8 + n
        self.header.pop("__metadata__", None)
        self.mm = np.memmap(self.path, np.uint8, "r")
        self.config = json.load(open(os.path.join(self.folder, "config.json")))

    def names(self): return sorted(self.header)

    def _raw(self, name: str):
        e = self.header[name]; a, b = e["data_offsets"]
        return e["dtype"], e["shape"], self.mm[self.base + a:self.base + b]

    def get(self, name: str) -> np.ndarray:
        """Any tensor as float32 (fp8 WITHOUT its scale: use `linear` for a dequantized matrix)."""
        dt, shape, raw = self._raw(name)
        if dt == "F32": return np.frombuffer(raw, np.float32).reshape(shape).copy()
        if dt == "BF16": return bf16_to_f32(np.frombuffer(raw, np.uint16)).reshape(shape)
        if dt == "F8_E4M3": return _E4M3_TABLE[np.frombuffer(raw, np.uint8)].reshape(shape)
        raise ValueError(f"{name}: dtype {dt}")

    def linear(self, prefix: str) -> np.ndarray:
        """`prefix.weight` dequantized by `prefix.weight_scale`: float32 `[out, in]`."""
        w = self.get(prefix + ".weight")
        s = self.get(prefix + ".weight_scale")
        return w * s[:, None]

    def has(self, name: str) -> bool: return name in self.header


if __name__ == "__main__":
    import sys
    W = Ideogram4Weights(sys.argv[1] if len(sys.argv) > 1 else "~/ideogram4/unconditional_transformer")
    print(W.config)
    print(len(W.names()), "tensors")
    w = W.linear("layers.0.attention.qkv")
    print("qkv", w.shape, w.dtype, "absmax %.4g mean|w| %.4g nan %d" % (np.abs(w).max(), np.abs(w).mean(), int(np.isnan(w).sum())))
    print("norm1", W.get("layers.0.attention_norm1.weight")[:6])
    print("indicator", W.get("embed_image_indicator.weight")[:, :4])

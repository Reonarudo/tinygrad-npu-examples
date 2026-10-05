#!/usr/bin/env python3
"""The FULL-PRECISION weight cache for `ideogram4_fp_backend.py`: every block's five linears as E4M3
codes in `k_gemm_gs`'s group-panel order (`gemm_fp16.pack_b_group_e4m3`: [group][slice][strip][k-step]
[4 tiles][4 cols][4 k] bytes -- the device's E4M3 -> fp16 pass turns them into the GEMM's fp16 panels
in place), w1 | w3 fused along N, the per-row weight scales as tile quads (`vec_f16.dup_quads`), and
the small tensors (norms, the AdaLN modulation's E4M3 codes + scale + bias) in an npz.

    python3 ideogram4_fp_pack.py --weights ~/ideogram4/unconditional_transformer --out ~/ideogram4/fpcache --layers 0-33

Per layer: 254 MB of codes (qkv 63.7, o 21.2, w13 113.2, w2 56.6) + ~10 MB small; 8.6 GB for the 34.
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ideogram4_weights import Ideogram4Weights                           # noqa: E402

KS, NS = 24, 6


def pack_codes(codes: np.ndarray) -> np.ndarray:
    """E4M3 `[N, K]` -> all groups' panels, [g][slice][s][kk][j][n][k] (`pack_b_group_e4m3` per group, concatenated)."""
    N, K = codes.shape; assert N % 96 == 0 and K % 96 == 0
    t = codes.reshape(N // 96, NS, 4, 4, K // 96, KS, 4)                  # [g][s][j][n][slice][kk][k]
    return np.ascontiguousarray(t.transpose(0, 4, 1, 5, 2, 3, 6)).ravel()


def dup_quads(v): q = v.astype(np.float32).reshape(-1, 4); return np.ascontiguousarray(np.concatenate([q, q], 1)).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="~/ideogram4/unconditional_transformer"); ap.add_argument("--out", default="~/ideogram4/fpcache"); ap.add_argument("--layers", default="0-33")
    a = ap.parse_args()
    W = Ideogram4Weights(a.weights); out = os.path.expanduser(a.out); os.makedirs(out, exist_ok=True)
    lo, hi = (int(v) for v in a.layers.split("-"))
    raw = lambda name: W._raw(name)[2]
    for l in range(lo, hi + 1):
        t0 = time.perf_counter(); p = f"layers.{l}."
        codes = lambda n, shape: np.frombuffer(raw(p + n + ".weight"), np.uint8).reshape(shape)
        scale = lambda n: W.get(p + n + ".weight_scale").astype(np.float32)
        qkv = codes("attention.qkv", (13824, 4608)); o = codes("attention.o", (4608, 4608))
        w1 = codes("feed_forward.w1", (12288, 4608)); w3 = codes("feed_forward.w3", (12288, 4608)); w2 = codes("feed_forward.w2", (4608, 12288))
        for name, c in (("qkv", qkv), ("o", o), ("w13", np.concatenate([w1, w3], 0)), ("w2", w2)):
            pack_codes(c).tofile(os.path.join(out, f"L{l}_{name}.bin"))
        ada_dt, ada_shape, ada_raw = W._raw(p + "adaln_modulation.weight")
        np.savez(os.path.join(out, f"L{l}_small.npz"),
                 sc_qkv=dup_quads(scale("attention.qkv")), sc_o=dup_quads(scale("attention.o")), sc_w13=dup_quads(np.concatenate([scale("feed_forward.w1"), scale("feed_forward.w3")])), sc_w2=dup_quads(scale("feed_forward.w2")),
                 an1=W.get(p + "attention_norm1.weight"), an2=W.get(p + "attention_norm2.weight"), fn1=W.get(p + "ffn_norm1.weight"), fn2=W.get(p + "ffn_norm2.weight"),
                 norm_q=W.get(p + "attention.norm_q.weight"), norm_k=W.get(p + "attention.norm_k.weight"),
                 ada_codes=np.frombuffer(ada_raw, np.uint8).reshape(ada_shape) if ada_dt == "F8_E4M3" else W.get(p + "adaln_modulation.weight"),
                 ada_scale=W.get(p + "adaln_modulation.weight_scale") if W.has(p + "adaln_modulation.weight_scale") else np.ones(ada_shape[0], np.float32), ada_b=W.get(p + "adaln_modulation.bias"))
        print("   layer %2d packed in %.1f s" % (l, time.perf_counter() - t0), flush=True)


if __name__ == "__main__":
    main()

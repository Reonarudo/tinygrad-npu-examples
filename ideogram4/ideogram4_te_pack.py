#!/usr/bin/env python3
"""The Qwen3-VL-8B text encoder's weights -> the NPU GEMM's cache (`ideogram4_text_npu.py`): per layer the fused linears as E4M3
codes padded to the GEMM's multiples (N: 96, K: 192; pads are code 0 = 0.0) in `k_gemm_gm`'s 48 x 3 panel order
(`pack_b_group_e4m3` + `repack_panels_48x3`), their per-row scales x 2^8 (the b8 GEMM's C is x 2^-8), and the norms as fp32:
    qkv: q | k | v rows (6144 x 4224)   o: 4224 x 4224   gu: gate / up rows interleaved per 96 (swiglu_dma's order, 24576 x 4224)
    dn: 4224 x 12288   (o / dn padded to 4224 columns: the residual stream is [R, 4224] with zero pad columns)
and the conditional transformer's text projection (`--cond`): `llm_cond_proj` [4608, 4096 x 13] as 13 per-tap GEMMs of K = 4224
(feature d * 13 + t -> tap t's columns d), `cond_t{t}.bin`, with `cond_small.npz` (the scales, `llm_cond_norm` per tap, the bias,
the text indicator row). Asserts no E4M3 NaN code (0x7f / 0xff): the in-LSRAM expansion is exact for the other 254.
~7.2 GB; a few minutes.

    TG=... python3 ideogram4_te_pack.py --te ~/ideogram4/text_encoder --cond ~/ideogram4/transformer --out ~/ideogram4/te_p48 [--layers 0-35]
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zy import gemm_fp16 as G                                    # noqa: E402
from ideogram4_text import SafeDir                               # noqa: E402

HID, INTER, KP_H = 4096, 12288, 4224


def rup(x, m): return -(-x // m) * m


def pack(codes, scale, K_pad, N_pad=None):
    """codes uint8 [N, K], scale f32 [N] -> (the 48 x 3 panels, bytes; the padded scale x 256)."""
    N, K = codes.shape
    assert not np.any((codes & 0x7F) == 0x7F), "an E4M3 NaN code"
    Np = rup(N, 96) if N_pad is None else N_pad
    c = np.zeros((Np, K_pad), np.uint8); c[:N, :K] = codes
    p24 = np.concatenate([G.pack_b_group_e4m3(c, g, 6, 24).ravel() for g in range(Np // 96)])
    sc = np.zeros(Np, np.float32); sc[:N] = scale * np.float32(256)
    return G.repack_panels_48x3(p24, K_pad), sc


def pack_cond(weights, out):
    """The conditional transformer's text projection -> `cond_t{t}.bin` (13 taps) and `cond_small.npz`."""
    from ideogram4_weights import Ideogram4Weights
    W = Ideogram4Weights(weights)
    dt, shape, raw = W._raw("llm_cond_proj.weight"); assert dt == "F8_E4M3", dt
    codes = np.asarray(raw).reshape(shape); scale = W.get("llm_cond_proj.weight_scale").ravel()
    sc = None
    for t in range(13):
        pt, sc = pack(np.ascontiguousarray(codes[:, t::13]), scale, KP_H)
        pt.tofile(os.path.join(out, "cond_t%d.bin" % t))
    nw = W.get("llm_cond_norm.weight").astype(np.float32)
    np.savez(os.path.join(out, "cond_small.npz"), sc=sc, norm=nw, norm_t=np.stack([np.concatenate([nw[t::13], np.zeros(KP_H - HID, np.float32)]) for t in range(13)]),
             bias=W.get("llm_cond_proj.bias").astype(np.float32), ind=W.get("embed_image_indicator.weight")[0].astype(np.float32))
    print("   llm_cond_proj %s packed as 13 taps of K %d" % (list(shape), KP_H), flush=True)


def pack_layers(te, out, lo, hi):
    S = SafeDir(te); p = "language_model."
    code = lambda n: np.asarray(S.raw(n)[2]).reshape(S.raw(n)[1])
    for l in range(lo, hi + 1):
        t0 = time.perf_counter(); q = f"{p}layers.{l}."
        lin = lambda names: (np.concatenate([code(q + n + ".weight") for n in names]), np.concatenate([S.get(q + n + ".weight_scale").ravel() for n in names]))
        small = {}
        for name, parts, K in (("qkv", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"), KP_H), ("o", ("self_attn.o_proj",), KP_H),
                               ("gu", ("mlp.gate_proj", "mlp.up_proj"), KP_H), ("dn", ("mlp.down_proj",), INTER)):
            c, s = lin(parts)
            if name == "gu":       # interleave: groups of 96 rows gate 0, up 0, gate 1, up 1, ...
                ng = c.shape[0] // 192; order = np.stack([np.arange(ng), np.arange(ng) + ng], 1).ravel()
                c = c.reshape(2 * ng, 96, -1)[order].reshape(c.shape); s = s.reshape(2 * ng, 96)[order].ravel()
            panels, sc = pack(c, s, K, 4224 if name in ("o", "dn") else None)
            f = os.path.join(out, f"L{l}_{name}.bin"); panels.tofile(f + ".tmp"); os.replace(f + ".tmp", f)
            small["sc_" + name] = sc
        for n in ("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
            small[n.split(".")[-1]] = S.get(q + n + ".weight").astype(np.float32)
        np.savez(os.path.join(out, f"L{l}_small.npz"), **small)
        print("   layer %2d packed in %.1f s" % (l, time.perf_counter() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--te", default="~/ideogram4/text_encoder"); ap.add_argument("--out", default="~/ideogram4/te_p48")
    ap.add_argument("--layers", default="0-35"); ap.add_argument("--cond", default="~/ideogram4/transformer", help="the conditional transformer's checkpoint folder (llm_cond_proj); '' to skip")
    a = ap.parse_args()
    a.te, a.out, a.cond = (os.path.expanduser(v) for v in (a.te, a.out, a.cond))
    lo, hi = (int(x) for x in a.layers.split("-"))
    os.makedirs(a.out, exist_ok=True)
    pack_layers(a.te, a.out, lo, hi)
    if a.cond: pack_cond(a.cond, a.out)


if __name__ == "__main__":
    main()

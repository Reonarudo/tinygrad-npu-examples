#!/usr/bin/env python3
"""Pack one Ideogram 4 transformer's block linears as int8 for the AIFF (`ideogram4_q8`): per layer, per
linear, the weights W (the checkpoint's E4M3 codes x their row scales, fp32) smoothed by the calibration's SmoothQuant factors
(W lam) and quantised per output channel (s_w = max|W lam| / 127), in the AIFF's dense layout; w1|w3 as its two halves (one
lam and one s_w table over the fused [24576, 4608] matrix). The calibration holds lam [K] and the int16 shifts [passes, N/128]
per (branch, layer, linear); the default is the shipped `q8_calib.npz`, made by `ideogram4_q8_calib.py` from the 15-prompt set
in prompts/calib (maxima over steps 0/3/6/9/11 of both branches, shifts with 2x headroom). w2 is packed when the calibration has
it (q8_calib.npz does; the pipeline uses it only with --q8-w2). The existing fp8 cache is not touched.

    python3 ideogram4_q8_pack.py --weights ~/ideogram4/transformer --branch cond --out ~/ideogram4/q8pack/cond
    python3 ideogram4_q8_pack.py --weights ~/ideogram4/unconditional_transformer --branch uncond --out ~/ideogram4/q8pack/uncond

Per layer: L{l}_{qkv,o,w1,w3[,w2]}.bin (int8, `ideogram4_q8.pack_weights`) and L{l}_q8.npz (lam_*, sw_*, sh_*).
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ideogram4_q8 as Q                         # noqa: E402
from ideogram4_weights import Ideogram4Weights   # noqa: E402

ROWS_ANY = 4704      # the pack does not depend on the row count (only the descriptors do)


def e4m3_table():
    """The 256 E4M3 (fn) code values as fp32."""
    c = np.arange(256, dtype=np.uint32); s = (c >> 7) & 1; e = (c >> 3) & 0xF; m = c & 7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e.astype(np.int32) - 7))
    v = np.where((e == 15) & (m == 7), np.nan, v)
    return np.where(s == 1, -v, v).astype(np.float32)


TAB = e4m3_table()


def ckpt_weight(W, name):
    dt, shape, raw = W._raw(name + ".weight")
    codes = np.frombuffer(bytes(raw), np.uint8).reshape(shape)
    return (TAB[codes] * W.get(name + ".weight_scale").astype(np.float32)[:, None]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--branch", required=True, choices=("cond", "uncond"))
    ap.add_argument("--calib", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "q8_calib.npz")); ap.add_argument("--out", required=True); ap.add_argument("--layers", default=None)
    ap.add_argument("--only", default=None, help="comma list of linears to (re)pack (qkv,o,w13,w2); the layer's tables are merged into its npz")
    a = ap.parse_args()
    W = Ideogram4Weights(os.path.expanduser(a.weights)); C = np.load(os.path.expanduser(a.calib))
    nl = W.config["num_layers"]; lo, hi = (int(v) for v in a.layers.split("-")) if a.layers else (0, nl - 1)
    out = os.path.expanduser(a.out); os.makedirs(out, exist_ok=True)
    has_w2 = ("%s_0_w2_lam" % a.branch) in C.files
    T0 = time.time()
    for l in range(lo, hi + 1):
        t0 = time.time(); p = "layers.%d." % l; small = {}
        want = a.only.split(",") if a.only else ["qkv", "o", "w13"] + (["w2"] if has_w2 else [])
        sp = os.path.join(out, "L%d_q8.npz" % l)
        if a.only and os.path.exists(sp): small = dict(np.load(sp))
        load = {"qkv": lambda: ckpt_weight(W, p + "attention.qkv"), "o": lambda: ckpt_weight(W, p + "attention.o"),
                "w13": lambda: np.concatenate([ckpt_weight(W, p + "feed_forward.w1"), ckpt_weight(W, p + "feed_forward.w3")]),
                "w2": lambda: ckpt_weight(W, p + "feed_forward.w2")}
        mats = {k: load[k]() for k in want}
        for k, Wm in mats.items():
            key = "%s_%d_%s" % (a.branch, l, k)
            lam = C[key + "_lam"].astype(np.float32); sh = C[key + "_sh"].astype(np.int8)
            wq, sw = Q.quant_weights(Wm, lam)
            small["lam_" + k], small["sw_" + k], small["sh_" + k] = lam, sw, sh
            if k == "w13":
                half = Wm.shape[0] // 2
                for nm, part in (("w1", wq[:half]), ("w3", wq[half:])):
                    Q.pack_weights(np.ascontiguousarray(part), Q.Plan(ROWS_ANY, part.shape[1], part.shape[0])).tofile(os.path.join(out, "L%d_%s.bin.tmp" % (l, nm)))
                    os.replace(os.path.join(out, "L%d_%s.bin.tmp" % (l, nm)), os.path.join(out, "L%d_%s.bin" % (l, nm)))
            else:
                Q.pack_weights(wq, Q.Plan(ROWS_ANY, wq.shape[1], wq.shape[0])).tofile(os.path.join(out, "L%d_%s.bin.tmp" % (l, k)))
                os.replace(os.path.join(out, "L%d_%s.bin.tmp" % (l, k)), os.path.join(out, "L%d_%s.bin" % (l, k)))
        np.savez(os.path.join(out, "L%d_q8.npz" % l), **small)
        print("%s layer %2d packed (%s) in %.1f s (total %.1f min)" % (a.branch, l, ",".join(mats), time.time() - t0, (time.time() - T0) / 60), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""The W8A8 path's calibration from a prompt set, from the device's own statistics
(`ideogram4_fp_1024.py --calib-stats`):

  lam     python3 ideogram4_q8_calib.py lam --act act_*.npz --old calib_dev.npz --out calib_lam.npz [--combine max|pNN]
          phase 1: per (branch, layer, linear) the per-channel |max| of the linear's input over prompts x steps x rows
          (`--calib-kind act`, run on the W8A8 path with the OLD calibration: its operands are x / lam_old, multiplied
          back here; w2's operand is not smoothed without --q8-w2) -> lam = sqrt(xmax) / sqrt(wmax), clamped as the
          simulator does; plus PROVISIONAL shifts from the weights' bound 127 sum|Wq| (no int16 saturation possible).
  shifts  python3 ideogram4_q8_calib.py shifts --part part_*.npz --lam calib_lam.npz --out calib_new.npz [--headroom 2]
          [--patch PACKDIR]
          phase 2: the AIFF partials' per-column |max| (`--calib-kind part`, the packs of calib_lam.npz, i.e. the
          provisional shifts) x 2^shift = the accumulators' |max| -> per (pass, 128-unit) shift with headroom;
          `--patch` rewrites the packs' L*_q8.npz sh_* (the int8 weights do not depend on the shifts).

Keys: "<branch>_<layer>_<lin>_lam" [K], "_sh" [passes, N/128] (the simulator's calibration format).
"""
import argparse, glob, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ideogram4_q8 as Q                         # noqa: E402
from ideogram4_weights import Ideogram4Weights   # noqa: E402
from ideogram4_q8_pack import ckpt_weight        # noqa: E402

KS = {"qkv": 4608, "o": 4608, "w13": 4608, "w2": 12288}
NS = {"qkv": 13824, "o": 4608, "w13": 24576, "w2": 4608}
WN = {"qkv": ["attention.qkv"], "o": ["attention.o"], "w13": ["feed_forward.w1", "feed_forward.w3"], "w2": ["feed_forward.w2"]}
WDIR = {"cond": "~/ideogram4/transformer", "uncond": "~/ideogram4/unconditional_transformer"}


def weights(W, l, k):
    return np.concatenate([ckpt_weight(W, "layers.%d.%s" % (l, n)) for n in WN[k]])


def combine(samples: np.ndarray, how: str) -> np.ndarray:
    """samples [n, K] (one row per prompt x step) -> [K]."""
    if how == "max": return samples.max(0)
    return np.percentile(samples, float(how[1:]), axis=0).astype(np.float32)


def bound_shifts(wq: np.ndarray, kc: int) -> np.ndarray:
    """[passes, N/128]: the smallest shift with 127 sum_k |Wq| <= 32767 for every channel of the unit (never saturates)."""
    N, K = wq.shape; S = K // kc
    b = np.stack([127.0 * np.abs(wq[:, s * kc:(s + 1) * kc].astype(np.int64)).sum(1) for s in range(S)])
    return np.maximum(0, np.ceil(np.log2(np.maximum(b.reshape(S, -1, Q.UNIT).max(2), 1.0) / 32767.0))).astype(np.int8)


def do_lam(a):
    files = sorted(sum((glob.glob(os.path.expanduser(p)) for p in a.act), []))
    old = np.load(os.path.expanduser(a.old))
    st = [np.load(f) for f in files]
    keys = sorted(k for k in st[0].files if k != "steps")
    print("phase 1: %d prompts x steps %s, %d (branch, layer, linear) keys; combine %s" % (len(files), list(st[0]["steps"]), len(keys), a.combine), flush=True)
    out, Ws, rat = {}, {}, []
    t0 = time.time()
    for key in keys:
        b, l, k = key.split("_"); l = int(l)
        smp = np.concatenate([z[key] for z in st]).astype(np.float32)               # [prompts x steps, K]: the operands' |max|
        if k != "w2" or a.w2_smoothed: smp = smp * old["%s_%d_%s_lam" % (b, l, k)][None, :]   # back to the linear's input x
        s_sorted = np.sort(smp, 0); rat.append(float(np.median(s_sorted[-1] / np.maximum(s_sorted[-2], 1e-12))))
        xm = np.maximum(combine(smp, a.combine), 1e-5)
        if b not in Ws: Ws[b] = Ideogram4Weights(os.path.expanduser(WDIR[b]))
        W = weights(Ws[b], l, k)
        wm = np.maximum(np.abs(W).max(0), 1e-5)
        lam = np.clip(np.sqrt(xm) / np.sqrt(wm), 1e-2, 1e3).astype(np.float32)
        wq, _ = Q.quant_weights(W, lam)
        out[key + "_lam"] = lam; out[key + "_sh"] = bound_shifts(wq, Q.pass_k(KS[k]))
        out[key + "_xmax"] = xm.astype(np.float32)
    np.savez(os.path.expanduser(a.out), **out)
    print("median over keys of (largest / second-largest sample) per channel: %.3f (max %.3f); %d keys in %.0f s -> %s"
          % (float(np.median(rat)), float(np.max(rat)), len(keys), time.time() - t0, a.out), flush=True)


def do_shifts(a):
    files = sorted(sum((glob.glob(os.path.expanduser(p)) for p in a.part), []))
    lam = np.load(os.path.expanduser(a.lam)); st = [np.load(f) for f in files]
    keys = sorted(k for k in st[0].files if k != "steps")
    out = {k: lam[k] for k in lam.files if k.endswith("_lam")}
    rows = []
    for key in keys:
        b, l, k = key.split("_")
        smp = np.concatenate([z[key] for z in st]).astype(np.float64)               # [n, S x N] |max| of P
        S, N = KS[k] // Q.pass_k(KS[k]), NS[k]
        sh0 = lam[key + "_sh"].astype(np.float64)                                    # the shifts these partials were made with
        acc = smp.reshape(-1, S, N) * np.exp2(np.repeat(sh0, Q.UNIT, axis=1))[None]  # the accumulators' |max|, to 2^sh0
        m = acc.max(0).reshape(S, -1, Q.UNIT).max(2)
        sh = np.maximum(0, np.ceil(np.log2(np.maximum(m * a.headroom, 1.0) / 32767.0))).astype(np.int8)
        out[key + "_sh"] = sh
        rows.append((key, float(sh0.mean() - sh.mean()), int((smp.reshape(-1, S, N).max(0) >= 32767).sum())))
    np.savez(os.path.expanduser(a.out), **out)
    gain = np.array([r[1] for r in rows]); satk = [r for r in rows if r[2]]
    print("phase 2: %d prompts; shifts %.2f bits finer than the bound on average (min %.2f, max %.2f); provisional partials at the "
          "int16 limit in %d keys -> %s" % (len(files), gain.mean(), gain.min(), gain.max(), len(satk), a.out), flush=True)
    if a.patch:
        n = 0
        for b in ("cond", "uncond"):
            for f in sorted(glob.glob(os.path.join(os.path.expanduser(a.patch), b, "L*_q8.npz"))):
                l = int(os.path.basename(f)[1:].split("_")[0]); z = dict(np.load(f))
                for k in ("qkv", "o", "w13", "w2"):
                    key = "%s_%d_%s_sh" % (b, l, k)
                    if "sh_" + k in z and key in out: z["sh_" + k] = out[key]
                np.savez(f + ".tmp.npz", **z); os.replace(f + ".tmp.npz", f); n += 1
        print("patched the shifts of %d layer tables under %s" % (n, a.patch), flush=True)


def do_variant(a):
    """A lam variant (another combine or alpha) with its shifts DERIVED from phase 2's measurements under the base lam: acc ~ 127^2
    y / (a_m s_n) (a_m the token's |max| of x / lam, s_n the row's |max| of W lam). For lam_var <= lam_base elementwise, a_m can only
    grow, so max|acc_var| <= max|acc_base| x s_base,n / s_var,n -- the shifts this gives never saturate where the measured ones do not."""
    files = sorted(sum((glob.glob(os.path.expanduser(p)) for p in a.act), []))
    pfiles = sorted(sum((glob.glob(os.path.expanduser(p)) for p in a.part), []))
    old, base = np.load(os.path.expanduser(a.old)), np.load(os.path.expanduser(a.base))
    st, sp = [np.load(f) for f in files], [np.load(f) for f in pfiles]
    out, Ws, d_lam, d_sh = {}, {}, [], []
    for key in sorted(k for k in st[0].files if k != "steps"):
        b, l, k = key.split("_"); l = int(l)
        smp = np.concatenate([z[key] for z in st]).astype(np.float32)
        if k != "w2": smp = smp * old["%s_%d_%s_lam" % (b, l, k)][None, :]
        xm = np.maximum(combine(smp, a.combine), 1e-5)
        if b not in Ws: Ws[b] = Ideogram4Weights(os.path.expanduser(WDIR[b]))
        W = weights(Ws[b], l, k); wm = np.maximum(np.abs(W).max(0), 1e-5)
        lam = np.clip(xm ** a.alpha / wm ** (1 - a.alpha), 1e-2, 1e3).astype(np.float32)
        lb = base[key + "_lam"]; lam = np.minimum(lam, lb)                         # the derivation needs lam <= the base
        S, N = KS[k] // Q.pass_k(KS[k]), NS[k]
        s_b = np.abs(W * lb[None, :]).max(1); s_v = np.abs(W * lam[None, :]).max(1)
        pm = np.concatenate([z[key] for z in sp]).astype(np.float64).reshape(-1, S, N).max(0)
        acc = pm * np.exp2(np.repeat(base[key + "_sh"].astype(np.float64), Q.UNIT, axis=1)) * (s_b / s_v)[None, :]
        sh = np.maximum(0, np.ceil(np.log2(np.maximum(acc.reshape(S, -1, Q.UNIT).max(2) * a.headroom, 1.0) / 32767.0))).astype(np.int8)
        out[key + "_lam"], out[key + "_sh"] = lam, sh
        d_lam.append(float(np.median(lam / lb))); d_sh.append(float(sh.mean() - base[key + "_sh"].mean()))
    np.savez(os.path.expanduser(a.out), **out)
    print("variant combine %s alpha %.2f: lam / base median %.3f; shifts minus base %+.2f bits (mean) -> %s" % (a.combine, a.alpha, np.median(d_lam), np.mean(d_sh), a.out), flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("lam"); p.add_argument("--act", nargs="+", required=True); p.add_argument("--old", required=True)
    p.add_argument("--out", required=True); p.add_argument("--combine", default="max")
    p.add_argument("--w2-smoothed", action="store_true", help="the phase-1 run used --q8-w2 (w2's operand carries 1/lam_old)")
    p = sub.add_parser("shifts"); p.add_argument("--part", nargs="+", required=True); p.add_argument("--lam", required=True)
    p.add_argument("--out", required=True); p.add_argument("--headroom", type=float, default=2.0); p.add_argument("--patch", default=None)
    p = sub.add_parser("variant"); p.add_argument("--act", nargs="+", required=True); p.add_argument("--part", nargs="+", required=True)
    p.add_argument("--old", required=True); p.add_argument("--base", required=True, help="the calibration the partials were measured with")
    p.add_argument("--out", required=True); p.add_argument("--combine", default="max"); p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--headroom", type=float, default=2.0)
    a = ap.parse_args()
    dict(lam=do_lam, shifts=do_shifts, variant=do_variant)[a.mode](a)


if __name__ == "__main__":
    main()

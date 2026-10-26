#!/usr/bin/env python3
"""One W8A8 linear of Ideogram 4 end to end on the board: the TEC quantiser, the AIFF linear, the TEC
dequantiser, each gated against numpy (`ideogram4_q8`'s references: the quantiser and the engine bit-exact, the dequantiser
to fp32 rounding), then timed. Real layer weights from the checkpoint; activations synthetic (a normal with outlier channels,
fp16, the A layout) unless --act gives a [rows, K] fp32 npy.

    python3 q8_check.py --lin w13 --layer 0 --rows 4704 --desc ~/ideogram4/q8desc [--reps 3]
"""
import argparse, ctypes, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad"))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinygrad import Tensor, dtypes                  # noqa: E402
from zy import OA, gemm_fp16 as G                    # noqa: E402
import ideogram4_q8 as Q                             # noqa: E402
import ideogram4_fp_backend as FB                    # noqa: E402
from ideogram4_weights import Ideogram4Weights       # noqa: E402

DEV = "ZHOUYI"
LIN = {"qkv": ("attention.qkv", 4608, 13824), "o": ("attention.o", 4608, 4608), "w13": ("feed_forward.w1", 4608, 12288),
       "w3": ("feed_forward.w3", 4608, 12288), "w2": ("feed_forward.w2", 12288, 4608)}


def addr(t): return OA.Device[DEV].raw.dev_addr(t.uop.buffer._buf)
def va(t): return t.uop.buffer._buf.va


def ckpt_weight(W, name):
    dt, shape, raw = W._raw(name + ".weight")
    codes = np.frombuffer(bytes(raw), np.uint8).reshape(shape)
    return (G.e4m3_table()[codes] * W.get(name + ".weight_scale")[:, None]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lin", default="w13", choices=list(LIN)); ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--weights", default="~/ideogram4/transformer"); ap.add_argument("--rows", type=int, default=4704)
    ap.add_argument("--treal", type=int, default=4621); ap.add_argument("--desc", default="~/ideogram4/q8desc")
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--sample", type=int, default=96); ap.add_argument("--act", default=None); ap.add_argument("--dq-mode", default=None); ap.add_argument("--gm", action="store_true"); ap.add_argument("--hstep", type=int, default=32)
    a = ap.parse_args()
    name, K, N = LIN[a.lin]; rows, nrb = a.rows, a.rows // 12
    plan = Q.Plan(rows, K, N, hstep=a.hstep); D = Q.Desc(Q.desc_path(plan, a.desc, gm=a.gm))
    print("== %s layer %d: [%d x %d] x [%d x %d], M_pad %d, %d passes of %d, %d chains, chunks %s" % (a.lin, a.layer, rows, K, K, N, plan.M_pad, plan.S, plan.kc, len(plan.chains), plan.chunks), flush=True)
    t0 = time.time()
    W = ckpt_weight(Ideogram4Weights(os.path.expanduser(a.weights)), "layers.%d.%s" % (a.layer, name))
    rng = np.random.RandomState(1)
    if a.act: x = np.load(os.path.expanduser(a.act)).astype(np.float32)[:rows]
    else:
        x = rng.randn(a.treal, K).astype(np.float32) * 0.5
        x[:, rng.choice(K, 16, replace=False)] *= 20.0                       # outlier channels
    xr = np.zeros((rows, K), np.float32); xr[:x.shape[0]] = x
    lam = np.sqrt(np.maximum(np.abs(xr).max(0), 1e-5)) / np.sqrt(np.maximum(np.abs(W).max(0), 1e-5))
    lam = np.clip(lam, 1e-2, 1e3).astype(np.float32)
    a16 = (xr / lam[None, :]).astype(np.float16)                              # the producer's smoothed fp16 operand
    q, sx = Q.q8_ref(a16.astype(np.float32))
    wq, sw = Q.quant_weights(W, lam)
    SR = np.unique(np.concatenate([np.linspace(0, x.shape[0] - 1, a.sample).astype(int), [0, x.shape[0] - 1]]))
    acc = np.stack([q[SR, s * plan.kc:(s + 1) * plan.kc].astype(np.float64) @ wq[:, s * plan.kc:(s + 1) * plan.kc].astype(np.float64).T for s in range(plan.S)])
    sh = Q.shifts_for(np.abs(acc).max(1))
    cs = Q.scales_cs(sw, sh)
    print("   host prep %.1f s; shifts %s..%s; |acc| max %.3g" % (time.time() - t0, sh.min(), sh.max(), np.abs(acc).max()), flush=True)

    # device buffers
    A16 = FB.dev(G.pack_a_slices(a16, 24).ravel().view(np.uint16))
    A8 = FB.zeros(plan.a8_bytes, dtypes.uint8); SX = FB.zeros(-(-nrb * 16 // 24576) * 24576, dtypes.float32)
    P = FB.zeros(plan.p_bytes // 2, dtypes.int16)
    Wd = FB.dev(Q.pack_weights(wq, plan)); CS = FB.dev(cs.ravel())
    DESC = FB.zeros(-(-D.nbytes // 98304) * 98304 // 4, dtypes.uint32)
    if a.gm:
        gm0 = OA.Device[DEV].raw.gm_window()[1]
        words = D.build(addr(DESC), addr(A8), Q.gm_slot_bases(D, gm0, plan), addr(P), sh)
        R = (plan.engine_nodes[0][1] - plan.engine_nodes[0][0]) * plan.S; tile = 2 * plan.kc * Q.UNIT
        src = [[((r % plan.S) * plan.nodes + plan.engine_nodes[e][0] + r // plan.S) * tile for r in range(R)] for e in range(3)]
        gch = [tuple(int(x) for x in D.info[i]) + tuple(int(x) for x in D.chains[i][1:]) for i in range(len(D.chains))]
    else: words = D.build(addr(DESC), addr(A8), addr(Wd), addr(P), sh)
    ctypes.memmove(va(DESC), words.ctypes.data, words.nbytes)
    CT = FB.zeros((N // 96) * nrb * 1152, dtypes.float32)
    kq = OA.register_csrc("q8_a16", Q.q8_a16_src(nrb, K, plan.M_pad), ntasks=12); dq8 = FB.dev(Q.q8_a16_descs(plan.M_pad, nrb))
    kd = OA.register_csrc("dq_ct", Q.dq_ct_src(nrb, N // 96, plan.S, N, plan.M_pad), ntasks=12); ddq = FB.dev(Q.dq_ct_descs(plan.S, N))
    run_q = lambda: OA.csrc_call(kq, A8, A16, SX, dq8).realize()
    run_a = (lambda: OA.aiff_gm(P, A8, Wd, DESC, R, tile, gch, src).realize()) if a.gm else (lambda: OA.aiff_lin(P, A8, Wd, DESC, D.chains.tolist()).realize())
    run_d = lambda: OA.csrc_call(kd, CT, P, SX, CS, ddq).realize()
    print("   device buffers ready (%.1f s): A8 %d B, P %d B, W %d B, descriptors %d B" % (time.time() - t0, plan.a8_bytes, plan.p_bytes, plan.w_bytes, D.nbytes), flush=True)

    bad = []
    # 1. the quantiser
    run_q()
    a8 = FB.host_read(A8).view(np.int8).reshape(K // 32, plan.M_pad, 32)
    sxd = FB.host_read(SX).view(np.float32).reshape(-1, 16)[:nrb, :12].ravel()
    a8_ref = Q.to_surface(q, plan.M_pad)
    nq = int((a8 != a8_ref).sum()); nsx = int((sxd != sx).sum())
    print("   gate q8_a16  %s  int8 %d / %d differ (max |d| %d), s_x %d / %d differ" % ("ok " if nq == 0 and nsx == 0 else "BAD", nq, a8.size,
          int(np.abs(a8.astype(np.int32) - a8_ref).max()), nsx, sx.size), flush=True)
    if nq or nsx: bad.append("q8_a16")
    # 2. the AIFF linear: poison P, run, compare the sampled rows (every column) bit for bit, count poison left in the live rows
    ctypes.memset(va(P), 0x5A, plan.p_bytes)
    OA.Device[DEV].raw.cache_flush(P.uop.buffer._buf) if hasattr(OA.Device[DEV].raw, "cache_flush") else None
    run_a()
    Pd = FB.host_read(P).view(np.int16).reshape(plan.M_pad, plan.S, N)
    Pref = np.stack([np.clip(Q.shift_round_even(acc[s].astype(np.int64), np.repeat(sh[s].astype(np.int64), Q.UNIT)[None, :]), -32768, 32767) for s in range(plan.S)], 1).astype(np.int16)
    nP = int((Pd[SR] != Pref).sum()); poison = int((Pd[:rows].view(np.uint16) == 0x5A5A).sum())
    sat = int((np.abs(Pref.astype(np.int32)) >= 32767).sum())
    print("   gate aiff    %s  %d / %d sampled outputs differ (%d rows x %d x %d); %d poison words left in the %d live rows; %d saturated in the ref" % (
          "ok " if nP == 0 else "BAD", nP, Pref.size, len(SR), plan.S, N, poison, rows, sat), flush=True)
    if nP: bad.append("aiff")
    if nP:
        d = np.nonzero(Pd[SR] != Pref); print("      first diffs (row, pass, col): %s" % list(zip(SR[d[0][:8]], d[1][:8], d[2][:8])), flush=True)
    # 3. the dequantiser (reads the device's own P)
    run_d()
    ct = FB.host_read(CT).view(np.float32)
    y = Q.dq_ref(Pd[:rows], np.concatenate([sx, np.zeros(rows - len(sx), np.float32)]) if len(sx) < rows else sx[:rows], cs)
    yref = Q.ct_from_rows(y, nrb)
    dd = np.abs(ct - yref); scale = np.abs(yref).max() + 1e-30
    nbig = int((dd > 1e-6 * scale).sum())
    print("   gate dq_ct   %s  max |d| %.3g (max |ref| %.3g), %d beyond 1e-6 relative" % ("ok " if nbig <= 64 else "BAD", dd.max(), scale, nbig), flush=True)
    if nbig > 64: bad.append("dq_ct")
    # accuracy vs fp32 on the sampled rows
    yf = (xr[SR] @ W.T).astype(np.float64); yd = y[SR].astype(np.float64)
    rel = np.linalg.norm(yd - yf) / np.linalg.norm(yf)
    print("   W8A8 vs fp32 on the sampled rows: rel %.4f, cos %.6f" % (rel, float((yd * yf).sum() / np.linalg.norm(yd) / np.linalg.norm(yf))), flush=True)
    # 4. timing
    for nm, f, macs in (("q8_a16", run_q, 0), ("aiff", run_a, plan.macs), ("dq_ct", run_d, 0)):
        ts = []
        for _ in range(a.reps):
            s0 = time.perf_counter(); f(); ts.append(time.perf_counter() - s0)
        t = min(ts)
        print("   time %-7s %.2f ms (min of %d)%s" % (nm, t * 1e3, a.reps, ("  %.2f TMAC/s (padded M)" % (macs / t / 1e12)) if macs else ""), flush=True)
    if a.dq_mode:
        kd2 = OA.register_csrc("dq_ct_" + a.dq_mode, Q.dq_ct_src(nrb, N // 96, plan.S, N, plan.M_pad, mode=a.dq_mode), ntasks=12)
        ts = []
        for _ in range(a.reps):
            s0 = time.perf_counter(); OA.csrc_call(kd2, CT, P, SX, CS, ddq).realize(); ts.append(time.perf_counter() - s0)
        print("   time dq_ct(%s) %.2f ms" % (a.dq_mode, min(ts) * 1e3), flush=True)
    if a.gm and os.environ.get("NOSTAGE_ARM") == "1":
        os.environ["ZHOUYI_AIFF_NOSTAGE"] = "1"; ts = []
        for _ in range(a.reps):
            s0 = time.perf_counter(); run_a(); ts.append(time.perf_counter() - s0)
        print("   time aiff without staging (timing arm, wrong output) %.2f ms" % (min(ts) * 1e3), flush=True)
        del os.environ["ZHOUYI_AIFF_NOSTAGE"]
    print("== %s: %s" % (a.lin, "ALL GATES PASS" if not bad else "BAD: %s" % bad), flush=True)


if __name__ == "__main__":
    main()

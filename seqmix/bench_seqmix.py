#!/usr/bin/env python3
"""The library kernels timed on the board at their models' shapes (device time of a TinyJit replay of `reps` launches, the min of 5;
functional correctness is the simulator gates' -- here only finiteness is checked). Governor `performance` is asserted on every policy
for the run and `schedutil` restored after (the measurement protocol); the driver's srcversion is printed.
    TG=../tinygrad python3 bench_seqmix.py [reps]"""
import os, sys, time, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "tinygrad")))); sys.path.insert(0, HERE)
from tinygrad import Tensor, Device, dtypes                              # noqa: E402
from tinygrad.engine.jit import TinyJit                                   # noqa: E402
from zy import OA                                                         # noqa: E402
import mamba2 as MB, mla as ML, kda as KD, cca as CC, attn as AT, route as RT   # noqa: E402

DEV = "ZHOUYI"; reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
dev = lambda a: Tensor(np.ascontiguousarray(a), device=DEV).realize()
f32 = lambda *s: dev(np.random.default_rng(0).standard_normal(s).astype(np.float32))
z32 = lambda *s: Tensor.zeros(*s, device=DEV).contiguous().realize()


def gov(mode):
  subprocess.run(["sudo", "sh", "-c", f"for p in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do echo {mode} > $p; done"], check=False)
  return sorted(set(open(p).read().strip() for p in __import__("glob").glob("/sys/devices/system/cpu/cpufreq/policy*/scaling_governor")))


def timed(name, src, ntasks, out, *ins):
  key = OA.register_csrc(name, src, ntasks=ntasks)
  def f(_):
    for _ in range(reps): OA.csrc_call(key, out, *ins).realize()
    return out
  j = TinyJit(f); jd = z32(16)
  for _ in range(2): j(jd)
  Device[DEV].synchronize(); ts = []
  for _ in range(5):
    t0 = time.perf_counter(); j(jd); Device[DEV].synchronize(); ts.append((time.perf_counter() - t0) / reps)
  fin = bool(np.isfinite(OA.host_invalidate(out).numpy()).all()); return min(ts), fin


def main():
  print("driver", open("/sys/module/aipu/srcversion").read().strip(), "| governor", gov("performance"), flush=True); rows = []
  try:
    # Mamba-2: Nemotron 3 Nano / 3.5 Lightning (64 heads x 64, N 128, 8 groups), one token
    NH, P, N, G, CONV, M = 64, 64, 128, 8, 4, 1; CD = NH * P + 2 * G * N
    t, ok = timed("mamba2_tok", MB.mamba2_tok_src(NH, P, N, G, CONV, M), 12, z32(M * NH * P), z32(M * NH), f32(2, NH, P, N), f32(2, NH, CONV, P + 2 * N), dev(np.array([1], np.int32)), dev(np.array([5], np.int32)),
                  f32(M, CD), f32(M, NH), f32(M, NH * P), dev(np.stack([-np.ones((2, NH)), np.zeros((2, NH)), np.ones((2, NH))], 1).astype(np.float32)), f32(2, CONV, CD), f32(2, CD), dev(MB.mamba2_tok_desc(NH, P, N, G, CONV, M)))
    rows.append(("mamba2_tok  Nemotron 64x64 N128 G8, M 1 (a layer: 2 MB of state read + written)", t, ok))
    t, ok = timed("mamba2_tok", MB.mamba2_tok_src(NH, P, N, G, CONV, 12), 12, z32(12 * NH * P), z32(12 * NH), f32(2, NH, P, N), f32(2, NH, CONV, P + 2 * N), dev(np.array([1], np.int32)), dev(np.array([5], np.int32)),
                  f32(12, CD), f32(12, NH), f32(12, NH * P), dev(np.stack([-np.ones((2, NH)), np.zeros((2, NH)), np.ones((2, NH))], 1).astype(np.float32)), f32(2, CONV, CD), f32(2, CD), dev(MB.mamba2_tok_desc(NH, P, N, G, CONV, 12)))
    rows.append(("mamba2_tok  the same, M 12 (12 tokens a launch)", t, ok))
    # MLA: GLM-4.7-Flash (20 heads, latent 512 + 64), 256 cached positions
    NHm, KVL, RD, TM, pos = 20, 512, 64, 512, 256; HG, BT, Pm = ML.cfg(NHm, KVL, RD)
    t, ok = timed("mla_part", ML.mla_part_src(NHm, KVL, RD, 1 / 16.0, TM), 12, z32((NHm // HG) * Pm * ML.block_floats(HG, KVL)), f32(NHm, KVL + RD), f32(2, TM, KVL + RD), f32(KVL + RD), dev(np.array([1, pos], np.int32)), dev(ML.mla_part_desc(NHm, KVL, RD)))
    rows.append((f"mla_part    GLM-4.7-Flash 20 heads, latent 576, pos {pos} (HG {HG} BT {BT} P {Pm})", t, ok))
    # KDA: Ling-3.0-tiny / Kimi (16 heads x 128), one token
    NHk, Kd, Vd = 16, 128, 128
    t, ok = timed("kda_tok", KD.kda_tok_src(NHk, Kd, Vd, CONV, 1, 1e-5), 12, z32(NHk * Vd), f32(2, NHk, Kd, Vd), f32(2, NHk, CONV, 3 * Kd), dev(np.array([1], np.int32)), dev(np.array([5], np.int32)),
                  f32(1, 3 * NHk * Kd), f32(1, NHk * Kd), f32(1, NHk), f32(1, NHk * Vd), f32(2, NHk + NHk * Kd), f32(2, CONV, 3 * NHk * Kd), f32(2, Vd), dev(KD.kda_tok_desc(Kd, Vd)))
    rows.append(("kda_tok     16 heads x 128 x 128, M 1 (a layer: 1 MB of state)", t, ok))
    # CCA: ZAYA1-8B (8 q / 2 kv x 128)
    t, ok = timed("cca_prep", CC.cca_prep_src(8, 2, 128, 64), 12, z32(8 * 128), z32(2 * 128), z32(2 * 128), f32(10 * 128), f32(256), f32(2, 2, 1280), f32(2, 128), f32(2, 1280, 2), f32(2, 1280), f32(2, 1280, 128, 2), f32(2, 1280), f32(2, 2), f32(64, 64), dev(np.array([1, 37], np.int32)), dev(CC.cca_prep_desc(128)))
    rows.append(("cca_prep    ZAYA1 8 / 2 x 128 (the 10 heads' grouped convs, 128 KB of weights each)", t, ok))
    # sinks + window attention: gpt-oss (64 / 8 x 64), pos 256, window 128
    NHa, NKV, HD = 64, 8, 64; HGa, BTa, Pa = AT.cfg(NHa, NKV, HD)
    t, ok = timed("attn_part", AT.attn_part_src(NHa, NKV, HD, 1 / 8.0, TM, window=128, sinks=True), 12, z32(NKV * (NHa // NKV // HGa) * Pa * AT.block_floats(HGa, HD)), f32(NHa, HD), f32(2, TM, NKV, HD), f32(2, TM, NKV, HD), f32(NKV, HD), f32(NKV, HD), f32(2, NHa), dev(np.array([1, pos], np.int32)), dev(AT.attn_part_desc(NHa, NKV, HD)))
    rows.append((f"attn_part   gpt-oss 64 / 8 x 64, window 128, sinks (HG {HGa} BT {BTa} P {Pa})", t, ok))
    t, ok = timed("attn_part", AT.attn_part_src(NHa, NKV, HD, 1 / 8.0, TM, window=0, sinks=True), 12, z32(NKV * (NHa // NKV // HGa) * Pa * AT.block_floats(HGa, HD)), f32(NHa, HD), f32(2, TM, NKV, HD), f32(2, TM, NKV, HD), f32(NKV, HD), f32(NKV, HD), f32(2, NHa), dev(np.array([1, pos], np.int32)), dev(AT.attn_part_desc(NHa, NKV, HD)))
    rows.append((f"attn_part   the same, full attention over {pos} positions", t, ok))
    # routing: Ring's grouped sigmoid router, 256 experts
    t, ok = timed("moe_route", RT.route_src("group", 256, 8, 1000, 2000, 8, 4, 2.5, woff=1), 4, Tensor.zeros(16, device=DEV, dtype=dtypes.int32).contiguous().realize(), Tensor.zeros(16, device=DEV, dtype=dtypes.int32).contiguous().realize(), z32(16), f32(256), f32(256), f32(2))
    rows.append(("moe_route   Ring's grouped sigmoid router, 256 experts top 8", t, ok))
  finally:
    print("governor restored", gov("schedutil"), flush=True)
  print(f"\n{'kernel':78s} {'us a launch':>12s}  finite"); [print(f"{n:78s} {t * 1e6:12.1f}  {ok}") for n, t, ok in rows]


if __name__ == "__main__": main()

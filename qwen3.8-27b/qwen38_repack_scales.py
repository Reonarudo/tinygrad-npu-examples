#!/usr/bin/env python3
"""An existing NPU cache (qwen38_pack.py: the block-scaled GEMM streams with every fp32 scale stored twice) -> the same cache with
each scale stored once (`gemm_fp16.bscale_stream_single`: 6528 -> 6336 B a K-slice, -2.9 %; the codes untouched, the same scale
values, bit-identical C from `gemm_gs(scales="single")`), written to a NEW directory with the `scales` marker. No checkpoint read:
the streams are transformed record by record, so the fused files (q|kv, qkv|z: whole groups concatenated) convert the same way
and `L{l}_fused.npz` / `L{l}_small.npz` / `lm_head_fp8.npz` / `outside_small.npz` / `fused` are copied as they are. The fp16
lm_head panels (`lm_head.bin`, no scales) are copied unchanged. Never writes into --src.

    python3 qwen38_repack_scales.py [--src /mnt/ssd/qwen3.8-27b-npu] [--out /mnt/ssd/qwen3.8-27b-npu-s1] [--only 'L0_*,L1_*'] [--dry-run]

Sizes: every stream shrinks by 192 / 6528 (0.9706 x); a 27B cache of ~27 GB of E4M3 streams writes ~26.2 GB. Time: the transform
runs at ~3 GB/s on a laptop core (numpy, 32-bit-lane copies); on the board it is bounded by the NVMe read + write of the cache
(the same disk both ways: at ~1-3 GB/s combined, 27 GB in and 26 GB out is ~20-50 min) -- --only converts a subset to time it first.
The output is checked as it is written: each converted file's size is the expected 6336 / 6528 of the source's, and the first and
last record of every file are compared against the host packer's identity (the duplicated quads must agree, or the source was not
a dup-layout stream)."""
import argparse, os, shutil, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad")))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zy import gemm_fp16 as G                                             # noqa: E402

KS, NS = 32, 3
REC_DUP, REC_ONE = NS * KS * 64 + NS * 128, NS * KS * 64 + NS * 64      # a K-slice's record in each layout (6528 / 6336 B)
CHUNK = (64 << 20) // REC_DUP * REC_DUP                                  # 64 MB of whole records a step
COPY_AS_IS = ("lm_head.bin",)                                            # the fp16 lm_head panels (ks 24, no scale table)


def check_record(rec):
  """A dup-layout record's scale table has every quad twice: [s j][c0..c3 c0..c3]."""
  sc = np.frombuffer(rec[NS * KS * 64:], np.uint32).reshape(NS * 4, 8)
  return bool(np.array_equal(sc[:, :4], sc[:, 4:]))


def convert(src, dst, dry):
  size = os.path.getsize(src)
  assert size % REC_DUP == 0, f"{src}: {size} B is not whole 6528-B records (not a dup-layout block-scale stream?)"
  n = size // REC_DUP; want = n * REC_ONE
  if dry: return size, want
  with open(src, "rb") as fi:
    first = fi.read(REC_DUP); fi.seek(size - REC_DUP); last = fi.read(REC_DUP)
  assert check_record(first) and check_record(last), f"{src}: the scale quads are not duplicated (already single, or not a bscale stream)"
  tmp = dst + ".tmp"
  with open(src, "rb") as fi, open(tmp, "wb") as fo:
    while True:
      b = fi.read(CHUNK)
      if not b: break
      fo.write(G.bscale_stream_single(np.frombuffer(b, np.uint8), NS, KS).tobytes())
  got = os.path.getsize(tmp); assert got == want, (src, got, want)
  os.replace(tmp, dst)
  return size, want


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--src", default=os.environ.get("QWEN_NPU", "/mnt/ssd/qwen3.8-27b-npu")); ap.add_argument("--out", default=None, help="default: <src>-s1")
  ap.add_argument("--only", default=None, help="comma list of glob patterns of files to convert (the rest is skipped), e.g. 'L0_*,L1_*,lm_head_fp8_0.bin'")
  ap.add_argument("--dry-run", action="store_true", help="sizes only, nothing written")
  a = ap.parse_args(); src = os.path.expanduser(a.src); out = os.path.expanduser(a.out or src.rstrip("/") + "-s1")
  assert os.path.isdir(src) and os.path.realpath(src) != os.path.realpath(out), (src, out)
  marker = os.path.join(src, "scales")
  assert not os.path.exists(marker) or open(marker).read().split()[0] == "dup", f"{src} is not a dup-layout cache ({open(marker).read().split()[0]})"
  import fnmatch
  names = sorted(os.listdir(src)); only = a.only.split(",") if a.only else None
  pick = lambda n: only is None or any(fnmatch.fnmatch(n, p) for p in only)
  if not a.dry_run: os.makedirs(out, exist_ok=True)
  t0 = time.perf_counter(); tin = tout = 0; nconv = ncopy = 0
  for n in names:
    s, d = os.path.join(src, n), os.path.join(out, n)
    if not os.path.isfile(s) or n == "scales": continue
    if n.endswith(".bin") and n not in COPY_AS_IS:
      if not pick(n): continue
      if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s) // REC_DUP * REC_ONE: print(f"   {n}: present"); continue
      t1 = time.perf_counter(); si, so = convert(s, d, a.dry_run); tin += si; tout += so; nconv += 1
      print(f"   {n}: {si / 1e9:.3f} -> {so / 1e9:.3f} GB" + ("" if a.dry_run else f" in {time.perf_counter() - t1:.1f} s"), flush=True)
    else:                                                                  # .npz, the `fused` marker, lm_head.bin: as they are
      if not a.dry_run and not (os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s)): shutil.copyfile(s, d)
      ncopy += 1
  if not a.dry_run and nconv:
    with open(os.path.join(out, "scales"), "w") as fh: fh.write(f"single\nqwen38_repack_scales.py from {src}: gemm_fp16.bscale_stream_single (each fp32 block scale once), gemm_gs(scales='single')\n")
  print(f"{'would convert' if a.dry_run else 'converted'} {nconv} streams: {tin / 1e9:.2f} -> {tout / 1e9:.2f} GB ({tout / max(tin, 1):.4f} x), {ncopy} files copied as they are, "
        f"in {time.perf_counter() - t0:.0f} s -> {out}" + ("" if a.dry_run else " (marker `scales` = single; point QWEN_NPU at it)"), flush=True)


if __name__ == "__main__":
  main()

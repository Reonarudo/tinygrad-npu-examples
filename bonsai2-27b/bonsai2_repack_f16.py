#!/usr/bin/env python3
"""An existing ternary NPU cache (bonsai2_pack.py: every stream gemm_fp16.pack_b_group_tern's, the scale table single-layout fp32
s x 2^18, 64 B a strip and K-slice) -> the same cache with fp16 tables (s x 2^15, 32 B a strip: 1728 -> 1632 B a K-slice record,
-5.6 %, 2.125 bits a weight), written to a NEW directory with the marker `tscale` = f16 (qwen38_npu.TSCALE -> gemm_gs(tscale="f16"),
a backend whose gemm_gs takes tscale=). The codes are untouched; every scale must convert EXACTLY -- fp32 v = s x 2^18 -> fp16 v x 2^-3,
finite, normal or +0 (the kernel widens it by fmae against fp16 2^3: a subnormal or -0 would not come back bit for bit) -- or the
tool stops (nothing half-written is left under the final name). With exact tables the GEMMs' C is bit-identical to the source
cache's, so the model's outputs are the same.

No GGUF read: the streams are converted record by record, so the fused files (q|kv, qkv|z: whole groups concatenated) and the head
parts (lm_head_tern_{p}.bin) convert the same way. Everything else -- *.npz, the markers `tern` / `scales` / `fused`, tokenizer.json,
and a subfolder that is not a ternary cache (bonsai2_pack.py --mtp's `mtp`: E4M3 streams, no `tern` marker) -- is hard-linked
(copied when the filesystem refuses links); a ternary subfolder is converted recursively. Never writes into --src.

    python3 bonsai2_repack_f16.py [--src /mnt/ssd/bonsai2-npu] [--out /mnt/ssd/bonsai2-npu-f16] [--only 'L0_*,lm_head_tern_0.bin'] [--dry-run]

Each output stream is checked as it is written: its size is 1632 / 1728 of the source's, and it is read back and every record's
codes compared with the source's and every fp16 table widened (x 2^15) compared with the source's fp32 table, bit for bit.
"""
import argparse, fnmatch, os, shutil, sys, time
import numpy as np

KS, NS = 32, 3
CB = NS * KS * 16                                                         # a K-slice's 2-bit codes (1536 B)
REC32, REC16 = CB + NS * 64, CB + NS * 32                                 # a K-slice record: fp32 table (1728 B) / fp16 table (1632 B)
CHUNK = (64 << 20) // REC32 * REC32                                       # 64 MB of whole records a step


def table_f16(t32):
  """[n, NS x 64] uint8 fp32 tables (s x 2^18) -> [n, NS x 32] uint8 fp16 tables (s x 2^15; lane 2i = column i, 2i + 1 = column 8 + i).
  Raises ValueError unless every value is exact (gemm_fp16.tern_table_f16's rule)."""
  v = np.ascontiguousarray(t32).view(np.float32).reshape(-1, NS, 16)
  h = (v * np.float32(2.0 ** -3)).astype(np.float16); hb = h.view(np.uint16)
  back = h.astype(np.float32) * np.float32(2.0 ** 3)
  bad = (back.view(np.uint32) != v.view(np.uint32)) | ((hb & 0x7C00) == 0x7C00) | (((hb & 0x7C00) == 0) & (hb != 0))
  if bad.any():
    i = tuple(np.argwhere(bad)[0])
    raise ValueError(f"{int(bad.sum())} scale(s) not exact as a normal fp16 s x 2^15 (first: record {i[0]} strip {i[1]} column {i[2]}: "
                     f"fp32 {float(v[i])!r}, bits {int(v.view(np.uint32)[i]):#010x})")
  return np.ascontiguousarray(h.reshape(-1, NS, 2, 8).transpose(0, 1, 3, 2)).view(np.uint8).reshape(-1, NS * 32)


def widen(t16):
  """The kernel's view of fp16 tables: [n, NS x 32] uint8 -> the fp32 tables [n, NS x 64] uint8 (fp16 x 2^3, lanes reordered)."""
  h = np.ascontiguousarray(t16).view(np.float16).reshape(-1, NS, 8, 2).transpose(0, 1, 3, 2).reshape(-1, NS * 16)
  return np.ascontiguousarray(h.astype(np.float32) * np.float32(2.0 ** 3)).view(np.uint8).reshape(-1, NS * 64)


def convert_records(b):
  rec = np.frombuffer(b, np.uint8).reshape(-1, REC32); out = np.empty((rec.shape[0], REC16), np.uint8)
  out[:, :CB] = rec[:, :CB]; out[:, CB:] = table_f16(rec[:, CB:])
  return out.tobytes()


def convert(src, dst, dry):
  size = os.path.getsize(src)
  if size % REC32: raise ValueError(f"{src}: {size} B is not whole {REC32}-B records (not an fp32-table ternary stream?)")
  want = size // REC32 * REC16
  if dry: return size, want
  tmp = dst + ".tmp"
  try:
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
      off = 0
      while (b := fi.read(CHUNK)):
        try: fo.write(convert_records(b))
        except ValueError as e: raise ValueError(f"{src} (records from {off // REC32}): {e}") from None
        off += len(b)
    got = os.path.getsize(tmp); assert got == want, (src, got, want)
    with open(src, "rb") as fi, open(tmp, "rb") as fo:                    # read back: codes equal, tables widen to the source's bits
      while (b := fi.read(CHUNK)):
        r32 = np.frombuffer(b, np.uint8).reshape(-1, REC32); r16 = np.frombuffer(fo.read(r32.shape[0] * REC16), np.uint8).reshape(-1, REC16)
        assert np.array_equal(r32[:, :CB], r16[:, :CB]) and np.array_equal(widen(r16[:, CB:]), r32[:, CB:]), f"{dst}: read-back mismatch"
    os.replace(tmp, dst)
  finally:
    if os.path.exists(tmp): os.unlink(tmp)
  return size, want


def link(s, d):
  if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s): return
  if os.path.exists(d): os.unlink(d)
  try: os.link(s, d)
  except OSError: shutil.copyfile(s, d)


def marker(d, n):
  f = os.path.join(d, n); return open(f).read().split()[0] if os.path.exists(f) else None


def repack(src, out, a, st, rel=""):
  tern = os.path.exists(os.path.join(src, "tern"))
  if tern:
    assert marker(src, "scales") == "single", f"{src}: a ternary cache holds the single-layout table (`scales` = single), got {marker(src, 'scales')}"
    assert marker(src, "tscale") in (None, "f32"), f"{src} is already an fp16-table cache (tscale = {marker(src, 'tscale')})"
  if not a.dry_run: os.makedirs(out, exist_ok=True)
  only = a.only.split(",") if a.only else None
  for n in sorted(os.listdir(src)):
    s, d = os.path.join(src, n), os.path.join(out, n)
    if os.path.isdir(s): repack(s, d, a, st, rel + n + "/"); continue
    if not os.path.isfile(s) or n == "tscale": continue
    if tern and n.endswith(".bin"):
      if only and not any(fnmatch.fnmatch(rel + n, p) for p in only): continue
      if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s) // REC32 * REC16: print(f"   {rel}{n}: present"); continue
      t1 = time.perf_counter(); si, so = convert(s, d, a.dry_run); st["in"] += si; st["out"] += so; st["conv"] += 1
      print(f"   {rel}{n}: {si / 1e9:.3f} -> {so / 1e9:.3f} GB" + ("" if a.dry_run else f" in {time.perf_counter() - t1:.1f} s"), flush=True)
    else:
      if not a.dry_run: link(s, d)
      st["link"] += 1
  if tern and not a.dry_run:                                              # the marker last: a cache without it reads as fp32
    with open(os.path.join(out, "tscale"), "w") as fh:
      fh.write(f"f16\nbonsai2_repack_f16.py from {src}: the ternary scale tables as fp16 s x 2^15 (exact), gemm_gs(tscale='f16')\n")
    st["marked"].append(out)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--src", default=os.environ.get("QWEN_NPU", "/mnt/ssd/bonsai2-npu")); ap.add_argument("--out", default=None, help="default: <src>-f16")
  ap.add_argument("--only", default=None, help="comma list of glob patterns (relative to --src) of the streams to convert, e.g. 'L0_*,lm_head_tern_0.bin'")
  ap.add_argument("--dry-run", action="store_true", help="sizes only, nothing written")
  a = ap.parse_args(); src = os.path.expanduser(a.src); out = os.path.expanduser(a.out or src.rstrip("/") + "-f16")
  assert os.path.isdir(src) and os.path.exists(os.path.join(src, "tern")), f"{src}: not a ternary cache (no `tern` marker)"
  assert os.path.realpath(src) != os.path.realpath(out) and not os.path.realpath(out).startswith(os.path.realpath(src) + "/"), (src, out)
  st = dict(conv=0, link=0, marked=[], **{"in": 0, "out": 0}); t0 = time.perf_counter()
  try: repack(src, out, a, st)
  except ValueError as e: print(f"FAILED: {e}", flush=True); sys.exit(1)
  print(f"{'would convert' if a.dry_run else 'converted'} {st['conv']} streams: {st['in'] / 1e9:.2f} -> {st['out'] / 1e9:.2f} GB "
        f"({st['out'] / max(st['in'], 1):.4f} x), {st['link']} files linked as they are, in {time.perf_counter() - t0:.0f} s -> {out}"
        + ("" if a.dry_run else f" (marker `tscale` = f16 in {st['marked']}; point QWEN_NPU at it)"), flush=True)


if __name__ == "__main__":
  main()

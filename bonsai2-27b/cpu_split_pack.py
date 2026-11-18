#!/usr/bin/env python3
"""The CPU half's weights for the NPU + CPU column split (cpu_split.c): each layer's linear streams (gemm_fp16.pack_b_group_tern,
as bonsai2_pack.py wrote them; the fused input projection -- qkvz on a DeltaNet layer, qkv = q|kv on an attention layer -- as
the cache's fused stream) decoded into the int8 kernel's column-interleaved 2-bit layout --
  <cache>/cpu_split/L{l}_{k}.c2c  uint8 [K/128][npad/8 column octets][4 b][4 s][16 B]: byte j's field f (bits 2f, 2f + 1) is
                                  u = w + 1 of column 8 co + 2 f + j / 8 at k = 32 b + 8 s + j % 8
  <cache>/cpu_split/L{l}_{k}.sct  fp32  [K/128][npad]:     the scale of (slice, column), as the NPU's table holds it
-- for ALL columns (the share is chosen at run time; only the used columns' pages are ever read). numpy only.
--mtp <q4t MTP cache>: the drafter's 4-bit linears as <mtp>/cpu_split/L<l>_<k>.c4 (cpu_split.c fmt 1) / .sct.
--mtp8 <fp8 MTP cache>: an E4M3 drafter's linears (single-layout scales) as <mtp>/cpu_split/L<l>_<k>.f8c (cpu_split.c fmt 2: the
  cache's own codes, re-ordered [K/128][npad][128]) / .sct (the stream's scales / 256).
--head: the ternary head's parts (lm_head_tern_{i}.bin, K = H) the same way, as <cache>/cpu_split/head_{i}.c2c / .sct.
--gtail <dst>: a cache whose attention layers' fused q|kv stream has the output gate's columns at its tail (gtail()), every other
  file a symlink into --cache; then its .c2c / .sct of those streams.
  python3 cpu_split_pack.py [--cache /mnt/ssd/bonsai2-npu] [--layers 0-63] [--kinds gu,dn,o,fused] [--head] [--gtail <dst>]"""
import argparse, os, time
import numpy as np
NS, KS, QK = 3, 32, 128

def unpack_stream(buf, npad, K):   # bonsai2_pack.unpack_stream (fp16 tables only)
  ng, nsl = npad // (16 * NS), K // QK; cb = NS * KS * 16; tb = len(buf) // (ng * nsl) - cb
  assert tb == NS * 32 and len(buf) == ng * nsl * (cb + tb), (len(buf), npad, K, tb)
  b = np.frombuffer(buf, np.uint8).reshape(ng, nsl, cb + tb)
  codes = b[..., :cb].reshape(ng, nsl, NS, KS // 2, 2, 16)
  j = np.stack([(codes >> (6 - 2 * jj)) & 3 for jj in range(4)], 5)
  c = j.reshape(ng, nsl, NS, KS // 2, 2, 4, 4, 4).transpose(0, 2, 5, 6, 1, 3, 4, 7).reshape(npad, K)
  h = np.ascontiguousarray(b[..., cb:]).view(np.float16).reshape(ng, nsl, NS, 8, 2).transpose(0, 1, 2, 4, 3).reshape(ng, nsl, 16 * NS)
  return c, (h.astype(np.float32) * np.float32(2.0 ** -15)).transpose(0, 2, 1).reshape(npad, nsl)

def unpack_q4t(buf, npad, K):
  """gemm_fp16.pack_b_group_q4t's stream (tscale f16) -> (u uint8 [npad, K] in 0..15, scale fp32 [npad, K/128])."""
  ng, nsl = npad // (16 * NS), K // QK; cb = NS * KS * 32; tb = NS * 32
  assert len(buf) == ng * nsl * (cb + tb), (len(buf), npad, K)
  b = np.frombuffer(buf, np.uint8).reshape(ng, nsl, cb + tb)
  c = b[..., :cb].reshape(ng, nsl, NS, KS // 2, 2, 2, 4, 4)                       # [g][slice][s][j][unit][h][n][k]
  q = np.stack([c[..., 0, :, :, :] & 15, c[..., 0, :, :, :] >> 4, c[..., 1, :, :, :] & 15, c[..., 1, :, :, :] >> 4], 5)   # [g][slice][s][j][h][q][n][k]
  u = q.transpose(0, 2, 5, 6, 1, 3, 4, 7).reshape(npad, K)                         # row s 16 + q 4 + n, k slice 128 + j 8 + h 4 + k
  h = np.ascontiguousarray(b[..., cb:]).view(np.float16).reshape(ng, nsl, NS, 8, 2).transpose(0, 1, 2, 4, 3).reshape(ng, nsl, 16 * NS)
  return u, (h.astype(np.float32) * np.float32(2.0 ** -17)).transpose(0, 2, 1).reshape(npad, nsl)

def stream_cols(buf, npad, K):
  """unpack_stream without arithmetic: (codes uint8 [npad, K] 0..3, the tables' raw fp16 bits uint16 [npad, K / 128])."""
  ng, nsl = npad // (16 * NS), K // QK; cb = NS * KS * 16; tb = NS * 32; assert len(buf) == ng * nsl * (cb + tb), (len(buf), npad, K)
  b = np.frombuffer(buf, np.uint8).reshape(ng, nsl, cb + tb)
  codes = b[..., :cb].reshape(ng, nsl, NS, KS // 2, 2, 16)
  j = np.stack([(codes >> (6 - 2 * jj)) & 3 for jj in range(4)], 5)
  c = j.reshape(ng, nsl, NS, KS // 2, 2, 4, 4, 4).transpose(0, 2, 5, 6, 1, 3, 4, 7).reshape(npad, K)
  h = np.ascontiguousarray(b[..., cb:]).view(np.uint16).reshape(ng, nsl, NS, 8, 2).transpose(0, 1, 2, 4, 3).reshape(ng, nsl, 16 * NS)
  return c, h.transpose(0, 2, 1).reshape(npad, nsl)

def cols_stream(c, h):
  """stream_cols' inverse: (codes [npad, K], raw tables [npad, K / 128]) -> the stream's bytes."""
  npad, K = c.shape; ng, nsl = npad // (16 * NS), K // QK
  v = c.reshape(ng, NS, 4, 4, nsl, KS // 2, 2, 4).transpose(0, 4, 1, 5, 6, 2, 3, 7).astype(np.uint8)   # [g][slice][s][i][u][jj][a][b]
  codes = (v[..., 0, :, :] << 6) | (v[..., 1, :, :] << 4) | (v[..., 2, :, :] << 2) | v[..., 3, :, :]   # [g][slice][s][i][u][a][b]
  t = h.reshape(ng, 16 * NS, nsl).transpose(0, 2, 1).reshape(ng, nsl, NS, 2, 8).transpose(0, 1, 2, 4, 3)
  return np.concatenate([codes.reshape(ng, nsl, -1), np.ascontiguousarray(t).view(np.uint8).reshape(ng, nsl, -1)], 2).reshape(-1)

def gtail(src, dst, layers):
  """dst: the cache src with each attention layer's fused q|kv stream (L{l}_qkv.bin: q = NH heads x [HD q | HD gate], then k|v)
  reordered to [the NH heads' q | k|v | the NH heads' gates], each from a group boundary (q|gate's padding split between q's
  end and the gates' when NH HD is not a whole number of groups: the 2B) -- a permutation of whole columns (codes and scale tables moved as they
  are: no requantization) -- and its L{l}_fused.npz with q|kv at group offsets 0 / NH HD / 48 and `gtail` = the gates' first group
  (qwen38_npu.Layers._meta: q's offset becomes the pair (0, gtail); the consumers interleave the rows back or read q and the gates
  apart). Every other file of src (and every directory but cpu_split) is a symlink. Each stream is checked: the reordering's
  inverse gives the source's bytes back."""
  os.makedirs(os.path.join(dst, "cpu_split"), exist_ok=True); att = []
  for l in layers:
    fz = os.path.join(src, f"L{l}_fused.npz")
    if os.path.exists(fz) and str(np.load(fz)["name"]) == "qkv":
      assert "gtail" not in np.load(fz).files, f"{src}: layer {l}'s q|kv already has the gates at the tail (a second permutation would scramble it)"
      att.append(l)
  own = {f"L{l}_{x}" for l in att for x in ("qkv.bin", "fused.npz")}
  for f in os.listdir(src):
    if f not in own and f != "cpu_split" and not os.path.lexists(os.path.join(dst, f)): os.symlink(os.path.join(src, f), os.path.join(dst, f))
  for f in os.listdir(os.path.join(src, "cpu_split")):
    if not any(f.startswith(f"L{l}_qkv.") for l in att) and not os.path.lexists(os.path.join(dst, "cpu_split", f)):
      os.symlink(os.path.join(src, "cpu_split", f), os.path.join(dst, "cpu_split", f))
  for l in att:
    f = np.load(os.path.join(src, f"L{l}_fused.npz")); subs = [str(k) for k in f["subs"]]; goff = [int(g) for g in f["goff"]]
    assert subs == ["q", "kv"] and goff[0] == 0, (l, subs, goff)
    sm = np.load(os.path.join(src, f"L{l}_small.npz")); meta = dict(zip([str(n) for n in sm["meta_names"]], sm["meta"].tolist()))
    HD = int(sm["self_attn_q_norm_weight"].shape[0]); Nq, npq, K = meta["q"]; NH = Nq // (2 * HD); G = 16 * NS
    N, npad, K2 = [int(v) for v in f["meta"]]; assert K2 == K and npq == goff[1] * G and Nq == 2 * NH * HD, (l, meta["q"], f["meta"], goff)
    gq = -(-NH * HD // G); p1 = gq * G - NH * HD; assert p1 <= npq - Nq, (l, p1, npq - Nq)   # q's pad (the 2B: 2048 q of 8 x 256 + 16) from q|gate's
    t0 = time.time(); raw = np.fromfile(os.path.join(src, f"L{l}_qkv.bin"), np.uint8); c, h = stream_cols(raw, npad, K)
    assert np.array_equal(cols_stream(c, h), raw), f"L{l}: the stream's decode / encode is not exact"
    hq = np.arange(Nq).reshape(NH, 2, HD)   # [q | q's pad to a group | k|v (with its pad) | the gates | the rest of q|gate's pad]
    perm = np.concatenate([hq[:, 0].reshape(-1), np.arange(Nq, Nq + p1), np.arange(npq, npad), hq[:, 1].reshape(-1), np.arange(Nq + p1, npq)])
    assert np.array_equal(np.sort(perm), np.arange(npad)), (l, len(perm), npad)
    out = cols_stream(c[perm], h[perm]); inv = np.argsort(perm)
    c2, h2 = stream_cols(out, npad, K); assert np.array_equal(cols_stream(c2[inv], h2[inv]), raw), f"L{l}: the reordering's inverse"
    out.tofile(os.path.join(dst, f"L{l}_qkv.bin.tmp")); os.replace(os.path.join(dst, f"L{l}_qkv.bin.tmp"), os.path.join(dst, f"L{l}_qkv.bin"))
    np.savez(os.path.join(dst, f"L{l}_fused.npz"), **{k: f[k] for k in f.files if k != "goff"}, goff=np.array([0, gq], f["goff"].dtype), gtail=np.int64(gq + (npad - npq) // G))
    print(f"L{l} qkv: q {NH} x {HD} (+{p1}) | kv {npad - npq} | gates from group {gq + (npad - npq) // G} -- {time.time() - t0:.1f} s", flush=True)
  return att

def q4_layout(u):
  """u [npad, K] (0..15) -> cpu_split.c's fmt 1 codes [K/128][npad/2][8][16]: byte j of block b = column 2cp + j / 8 at
  k = 16 b + j % 8 (bits 3:0) and 16 b + 8 + j % 8 (bits 7:4)."""
  npad, K = u.shape; U = u.reshape(npad // 2, 2, K // QK, 8, 2, 8)                 # [cp][c][slice][b][f][kk]
  return np.ascontiguousarray((U[..., 0, :] | U[..., 1, :] << 4).transpose(2, 0, 3, 1, 4)).astype(np.uint8)   # [slice][cp][b][c][kk]

def pack_q4(src, dst, npad, K, label):
  """A q4t stream -> dst.c4 / dst.sct (skipped if both exist)."""
  f4, fs = dst + ".c4", dst + ".sct"
  if os.path.exists(f4) and os.path.exists(fs): return
  t0 = time.time(); u, sc = unpack_q4t(np.fromfile(src, np.uint8), npad, K)
  q4_layout(u).tofile(f4 + ".tmp"); np.ascontiguousarray(sc.T).astype(np.float32).tofile(fs + ".tmp"); os.replace(f4 + ".tmp", f4); os.replace(fs + ".tmp", fs)
  print(f"{label} (pad {npad}) K {K} -> {os.path.getsize(f4) / 1e6:.1f} + {os.path.getsize(fs) / 1e6:.1f} MB in {time.time() - t0:.1f} s", flush=True)

def unpack_f8(buf, npad, K):
  """gemm_fp16.pack_b_group_bscale(scales="single")'s stream -> (codes uint8 [K/128][npad][128], scale fp32 [K/128][npad])."""
  ng, nsl = npad // (16 * NS), K // QK; cb = NS * KS * 64; tb = NS * 64
  assert len(buf) == ng * nsl * (cb + tb), (len(buf), npad, K)
  b = np.frombuffer(buf, np.uint8).reshape(ng, nsl, cb + tb)
  c = b[..., :cb].reshape(ng, nsl, NS, KS, 4, 4, 4)                               # [g][slice][s][kk][j][n][k]: row 48 g + 16 s + 4 j + n, k = 128 slice + 4 kk + k
  codes = np.ascontiguousarray(c.transpose(1, 0, 2, 4, 5, 3, 6)).reshape(nsl, npad, QK)
  sc = np.ascontiguousarray(b[..., cb:]).view(np.float32).reshape(ng, nsl, 16 * NS)   # [g][slice][16 s + 4 j + c] (x 2^8)
  return codes, np.ascontiguousarray(sc.transpose(1, 0, 2)).reshape(nsl, npad) * np.float32(1.0 / 256.0)

def pack_f8(src, dst, npad, K, label):
  """An E4M3 single-scale stream -> dst.f8c / dst.sct (skipped if both exist)."""
  f8, fs = dst + ".f8c", dst + ".sct"
  if os.path.exists(f8) and os.path.exists(fs): return
  t0 = time.time(); codes, sc = unpack_f8(np.fromfile(src, np.uint8), npad, K); assert not (codes & 0x7f == 0x7f).any(), "a NaN code"
  codes.tofile(f8 + ".tmp"); sc.astype(np.float32).tofile(fs + ".tmp"); os.replace(f8 + ".tmp", f8); os.replace(fs + ".tmp", fs)
  print(f"{label} (pad {npad}) K {K} -> {os.path.getsize(f8) / 1e6:.1f} + {os.path.getsize(fs) / 1e6:.1f} MB in {time.time() - t0:.1f} s", flush=True)

def main():
  global CHECK
  ap = argparse.ArgumentParser(); ap.add_argument("--cache", default="/mnt/ssd/bonsai2-npu"); ap.add_argument("--layers", default="0-63")
  ap.add_argument("--kinds", default="gu,dn", help="the linears: gu, dn, o, fused (the layer's fused input projection)")
  ap.add_argument("--head", action="store_true", help="the ternary head's parts only")
  ap.add_argument("--mtp", help="a q4t MTP cache (pack_mtp_q4t.py): its linears (fc q kv o gu dn) as <mtp>/cpu_split/L<l>_<k>.c4 / .sct")
  ap.add_argument("--mtp8", help="an fp8 (E4M3, single-layout scales) MTP cache: its linears as <mtp>/cpu_split/L<l>_<k>.f8c / .sct")
  ap.add_argument("--check", action="store_true", help="re-derive each existing .c2c / .sct and compare (no writes)")
  ap.add_argument("--gtail", help="a cache dir to build: the attention layers' q|kv streams with the gates at the tail (gtail()), then their .c2c / .sct")
  a = ap.parse_args(); CHECK = a.check
  if a.gtail:
    lo, hi = (int(v) for v in a.layers.split("-"))
    for l in gtail(a.cache, a.gtail, range(lo, hi + 1)):
      m = [int(v) for v in np.load(os.path.join(a.gtail, f"L{l}_fused.npz"))["meta"]]
      pack(os.path.join(a.gtail, f"L{l}_qkv.bin"), os.path.join(a.gtail, "cpu_split", f"L{l}_qkv"), m[1], m[2], f"L{l} qkv (gates at the tail): N {m[0]}")
    return
  if a.mtp8:
    d = os.path.join(a.mtp8, "cpu_split"); os.makedirs(d, exist_ok=True)
    assert open(os.path.join(a.mtp8, "scales")).read().split()[0] == "single" and not os.path.exists(os.path.join(a.mtp8, "q4t")), "--mtp8: an fp8 cache with single-layout scales"
    small = [f for f in os.listdir(a.mtp8) if f.endswith("_small.npz")][0]; l = small.split("_")[0]
    z = np.load(os.path.join(a.mtp8, small)); meta = dict(zip([str(n) for n in z["meta_names"]], z["meta"].tolist()))
    meta["fc"] = np.load(os.path.join(a.mtp8, f"{l}_fc.npz"))["meta"].tolist()
    for k in ("fc", "q", "kv", "o", "gu", "dn"):
      N, npad, K = meta[k]; pack_f8(os.path.join(a.mtp8, f"{l}_{k}.bin"), os.path.join(d, f"{l}_{k}"), npad, K, f"MTP {l} {k}: N {N}")
    return
  if a.mtp:
    d = os.path.join(a.mtp, "cpu_split"); os.makedirs(d, exist_ok=True); assert os.path.exists(os.path.join(a.mtp, "q4t")), "--mtp: a q4t cache"
    small = [f for f in os.listdir(a.mtp) if f.endswith("_small.npz")][0]; l = small.split("_")[0]
    z = np.load(os.path.join(a.mtp, small)); meta = dict(zip([str(n) for n in z["meta_names"]], z["meta"].tolist()))
    meta["fc"] = np.load(os.path.join(a.mtp, f"{l}_fc.npz"))["meta"].tolist()
    for k in ("fc", "q", "kv", "o", "gu", "dn"):
      N, npad, K = meta[k]; pack_q4(os.path.join(a.mtp, f"{l}_{k}.bin"), os.path.join(d, f"{l}_{k}"), npad, K, f"MTP {l} {k}: N {N}")
    return
  out = os.path.join(a.cache, "cpu_split"); os.makedirs(out, exist_ok=True); lo, hi = (int(v) for v in a.layers.split("-"))
  if a.head:
    z = np.load(os.path.join(a.cache, "lm_head_tern.npz")); K = int(np.load(os.path.join(a.cache, "L0_small.npz"))["meta"][0][2])
    for i, (g0, g1) in enumerate(z["parts"]):
      pack(os.path.join(a.cache, f"lm_head_tern_{i}.bin"), os.path.join(out, f"head_{i}"), int(g1 - g0) * 16 * NS, K, f"head {i}")
    return
  for l in range(lo, hi + 1):
    z = np.load(os.path.join(a.cache, f"L{l}_small.npz")); meta = dict(zip([str(n) for n in z["meta_names"]], z["meta"].tolist()))
    kinds = a.kinds.split(",")
    if "fused" in kinds:
      f = np.load(os.path.join(a.cache, f"L{l}_fused.npz")); fn = str(f["name"]); meta[fn] = [int(v) for v in f["meta"]]; kinds[kinds.index("fused")] = fn
    for k in kinds:
      N, npad, K = meta[k]; pack(os.path.join(a.cache, f"L{l}_{k}.bin"), os.path.join(out, f"L{l}_{k}"), npad, K, f"L{l} {k}: N {N}")

def codes2c(c, npad, K):
  """t = w + 1 [npad][K] (0..2) -> .c2c [K/128][npad/8][4 b][4 s][16 B] (cpu_split.c cols_c2c: byte j's field f = column
  8 co + 2 f + j / 8 at k = 32 b + 8 s + j % 8)."""
  t = c.reshape(npad // 8, 4, 2, K // QK, 4, 4, 8).transpose(3, 0, 4, 5, 2, 6, 1)   # [sl][co][b][s][jhi][jlo][f]
  return (t[..., 0] | (t[..., 1] << 2) | (t[..., 2] << 4) | (t[..., 3] << 6)).astype(np.uint8)
CHECK = False
def pack(src, dst, npad, K, label):
  """One stream (src) -> dst.c2c / dst.sct (skipped if both exist; CHECK: compared with the existing files instead)."""
  fc, fs = dst + ".c2c", dst + ".sct"
  if not CHECK and os.path.exists(fc) and os.path.exists(fs): return
  t0 = time.time()
  c, sc = unpack_stream(np.fromfile(src, np.uint8), npad, K); assert c.max() <= 2
  c2, s2 = codes2c(c.astype(np.uint8), npad, K), np.ascontiguousarray(sc.T).astype(np.float32)
  if CHECK:
    ok = np.array_equal(np.fromfile(fc, np.uint8), c2.reshape(-1)) and np.array_equal(np.fromfile(fs, np.float32), s2.reshape(-1))
    print(f"{label}: {'identical' if ok else 'DIFFERS'}", flush=True); assert ok; return
  c2.tofile(fc + ".tmp"); s2.tofile(fs + ".tmp"); os.replace(fc + ".tmp", fc); os.replace(fs + ".tmp", fs)
  print(f"{label} (pad {npad}) K {K} -> {os.path.getsize(fc) / 1e6:.1f} + {os.path.getsize(fs) / 1e6:.1f} MB in {time.time() - t0:.1f} s", flush=True)

if __name__ == "__main__": main()

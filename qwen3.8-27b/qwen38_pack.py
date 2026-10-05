#!/usr/bin/env python3
"""The checkpoint's per-layer shards -> the NPU cache `qwen38_npu.py` streams per token: every fp8 linear as the block-scaled
GEMM stream (`gemm_fp16.pack_b_group_bscale`: ks 32, 3 strips a group, each K-slice's E4M3 codes then its per-column scales),
k|v and gate|up fused along N, plus `L{l}_small.npz` with the layer's bf16 tensors as float32.

    python3 qwen38_pack.py [--dir /mnt/ssd/qwen3.8-27b-fp8] [--out /mnt/ssd/qwen3.8-27b-npu] [--layers 0-63]
    python3 qwen38_pack.py --fuse          # then: q | k|v and qkv | z as one stream file per layer, alongside (FUSED)
    python3 qwen38_pack.py --scales single --out /mnt/ssd/qwen3.8-27b-npu-s1 ...   # each block scale once (a `scales` marker)

`--scales single` (gemm_fp16.pack_b_group_bscale(scales="single"), k_gemm_gs(scales="single")): the fp32 block scales stored once
instead of twice -- 6336 B a slice against 6528 (-2.9 % of the stream), the same values, bit-identical C. A cache is one layout
throughout, named by the file `scales` in it (absent = dup); `qwen38_repack_scales.py` converts an existing cache without the checkpoint.
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad")))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zy import gemm_fp16 as G                                             # noqa: E402
import qwen38_ref as R                                                    # noqa: E402

KS, NS = 32, 3
SCALES = "dup"                          # the block-scale table's layout for every stream packed here (--scales; see stream())
LINEARS = {"linear": [("qkv", ("linear_attn.in_proj_qkv",)), ("z", ("linear_attn.in_proj_z",)), ("o", ("linear_attn.out_proj",)),
                      ("gu", ("mlp.gate_proj", "mlp.up_proj")), ("dn", ("mlp.down_proj",))],
           "full": [("q", ("self_attn.q_proj",)), ("kv", ("self_attn.k_proj", "self_attn.v_proj")), ("o", ("self_attn.o_proj",)),
                    ("gu", ("mlp.gate_proj", "mlp.up_proj")), ("dn", ("mlp.down_proj",))]}
SMALL = {"linear": ["input_layernorm.weight", "post_attention_layernorm.weight", "linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.conv1d.weight",
                    "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight", "linear_attn.norm.weight"],
         "full": ["input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight"]}

def stream(codes, sinv):
  """[N, K] codes + [N/128, K/128] scales -> the whole B stream (all groups), and N padded to the group size."""
  ngroups = -(-codes.shape[0] // (16 * NS))
  return np.concatenate([G.pack_b_group_bscale(codes, sinv, g, NS, KS, scales=SCALES) for g in range(ngroups)]), 16 * NS * ngroups

def scales_marker(out, scales):
  """The cache's layout marker (file `scales`): written for "single"; a cache already holding streams of the other layout refuses."""
  f = os.path.join(out, "scales"); have = open(f).read().split()[0] if os.path.exists(f) else "dup"
  bins = [n for n in os.listdir(out) if n.endswith(".bin") and n != "lm_head.bin"] if os.path.isdir(out) else []
  assert not bins or have == scales, f"{out} holds {have!r}-layout streams; pack {scales!r} into a fresh --out"
  if scales != "dup" and have != scales: open(f, "w").write(f"{scales}\nqwen38_pack.py --scales {scales}: gemm_fp16.pack_b_group_bscale(scales={scales!r}), gemm_gs(scales={scales!r})\n")

# The same-input projections fused into one GEMM (`--fuse`): the attention layers' q and k|v (both of the normalised residual),
# the DeltaNet layers' qkv and z (both of the normalised residual). A fused stream is the parts' streams one after the other:
# a stream is whole groups of 48 columns (N padded per part), so every part starts at a group boundary and the fused GEMM's C
# tiles ([group][rb][strip][192]) are the parts' C buffers back to back -- bit-identical, read at the part's group offset.
FUSED = {"full": ("qkv", ("q", "kv")), "linear": ("qkvz", ("qkv", "z"))}

def fuse_layer(out, l, kind):
  """L{l}_<fused>.bin (the parts' stream files concatenated) and L{l}_fused.npz (name, subs, goff = each part's first group, meta =
  the fused (N, npad, K)) alongside the per-linear files, which stay for the unfused build. Returns the fused file's bytes."""
  import shutil
  name, subs = FUSED[kind]
  z = np.load(os.path.join(out, f"L{l}_small.npz")); meta = {str(k): tuple(int(v) for v in m) for k, m in zip(z["meta_names"], z["meta"])}
  f = os.path.join(out, f"L{l}_{name}.bin"); goff, g, per = [], 0, None
  with open(f + ".tmp", "wb") as fh:
    for s in subs:
      N, npad, K = meta[s]; ng = npad // (16 * NS); src = os.path.join(out, f"L{l}_{s}.bin"); sz = os.path.getsize(src)
      assert npad % (16 * NS) == 0 and sz % ng == 0 and per in (None, sz // ng) and K == meta[subs[0]][2], (l, s, meta[s], sz)   # one group size throughout
      per = sz // ng; goff.append(g); g += ng
      with open(src, "rb") as sf: shutil.copyfileobj(sf, fh, 64 << 20)
  os.replace(f + ".tmp", f)
  np.savez(os.path.join(out, f"L{l}_fused.npz"), name=np.array(name), subs=np.array(subs), goff=np.array(goff, np.int64),
           meta=np.array([g * 16 * NS, g * 16 * NS, meta[subs[0]][2]], np.int64))
  return os.path.getsize(f)

def fuse_all(out, layers):
  """`--fuse` over the layers present (and the MTP head, layer R.NL, if packed): the fused files, a `fused` marker, the totals."""
  t0 = time.perf_counter(); tot = 0; done = []
  for l in list(layers) + ([R.NL] if os.path.exists(os.path.join(out, f"L{R.NL}_small.npz")) else []):
    if not os.path.exists(os.path.join(out, f"L{l}_small.npz")): print(f"   layer {l:2d} not packed: skipped"); continue
    kind = "full" if l == R.NL else R.LAYER_TYPES[l]; t1 = time.perf_counter(); sz = fuse_layer(out, l, kind); tot += sz; done.append(l)
    print(f"   layer {l:2d} ({kind}): {FUSED[kind][0]} = {' | '.join(FUSED[kind][1])}, {sz / 1e6:.1f} MB in {time.perf_counter() - t1:.2f} s", flush=True)
  with open(os.path.join(out, "fused"), "w") as fh: fh.write(f"qwen38_pack.py --fuse: layers {done} ({FUSED})\n")
  print(f"   fused {len(done)} layers: {tot / 1e9:.2f} GB written alongside the per-linear files in {time.perf_counter() - t0:.0f} s (QWEN_FUSE=0 ignores them)", flush=True)

def pack_layer(W, l, out):
  sf, kind, p = W.shard(l), R.LAYER_TYPES[l], f"model.language_model.layers.{l}."
  meta = {}
  for name, parts in LINEARS[kind]:
    codes = np.concatenate([sf.codes(p + q + ".weight") for q in parts]); sinv = np.concatenate([sf.f32(p + q + ".weight_scale_inv") for q in parts])
    st, npad = stream(codes, sinv); f = os.path.join(out, f"L{l}_{name}.bin"); st.tofile(f + ".tmp"); os.replace(f + ".tmp", f)
    meta[name] = (codes.shape[0], npad, codes.shape[1])
  small = {n.replace(".", "_"): sf.f32(p + n) for n in SMALL[kind]}
  small["meta_names"] = np.array([k for k in meta]); small["meta"] = np.array([meta[k] for k in meta], np.int64)
  np.savez(os.path.join(out, f"L{l}_small.npz"), **small)

MTP_L = 64
def pack_mtp(W, out):
  """The MTP head (mtp.safetensors) as layer 64 in the attention layers' format -- its q / k|v / o / gate|up / down have their
  shapes, so the zero-copy views of an attention layer read it -- plus L64_fc (mtp.fc, bf16 [5120, 10240], quantised like the fp8
  lm_head: E4M3 with 128 x 128 block scales) with its own meta, and the MTP norms (pre_fc_norm_embedding / _hidden, norm)."""
  sf, p = W.file("mtp.safetensors"), "mtp.layers.0."; meta = {}
  for name, parts in LINEARS["full"]:
    codes = np.concatenate([sf.codes(p + q + ".weight") for q in parts]); sinv = np.concatenate([sf.f32(p + q + ".weight_scale_inv") for q in parts])
    st, npad = stream(codes, sinv); f = os.path.join(out, f"L{MTP_L}_{name}.bin"); st.tofile(f + ".tmp"); os.replace(f + ".tmp", f)
    meta[name] = (codes.shape[0], npad, codes.shape[1])
  small = {n.replace(".", "_"): sf.f32(p + n) for n in SMALL["full"]}
  for n in ("mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight"): small[n.replace(".", "_")] = sf.f32(n)
  small["meta_names"] = np.array([k for k in meta]); small["meta"] = np.array([meta[k] for k in meta], np.int64)
  np.savez(os.path.join(out, f"L{MTP_L}_small.npz"), **small)
  w = sf.f32("mtp.fc.weight"); N, K = w.shape; nb = N // 128
  sinv = np.maximum(np.abs(w.reshape(nb, 128, K // 128, 128)).max((1, 3)), 1e-12).astype(np.float32) / 448.0
  codes = e4m3_encode(w / np.repeat(np.repeat(sinv, 128, 0), 128, 1))
  st, npad = stream(codes, sinv); f = os.path.join(out, f"L{MTP_L}_fc.bin"); st.tofile(f + ".tmp"); os.replace(f + ".tmp", f)
  np.savez(os.path.join(out, f"L{MTP_L}_fc.npz"), meta=np.array([N, npad, K], np.int64))

def pack_lm_head(W, out):
  """lm_head (bf16 [248320, 5120]) -> fp16 GEMM panels (`pack_b_group`, ks 24, 6 strips a group), N padded to 248352 (2587 groups)."""
  sf = W.file("outside.safetensors"); dt, shape, raw = sf.raw("lm_head.weight"); N, K = shape; npad = -(-N // 96) * 96; kpad = -(-K // 96) * 96
  f = os.path.join(out, "lm_head.bin")
  with open(f + ".tmp", "wb") as fh:
    for g0 in range(0, npad // 96, 64):                                   # 64 groups (6144 rows) at a time: 64 MB of fp16 panels
      r0, r1 = 96 * g0, min(96 * (g0 + 64), npad)
      w = np.zeros((r1 - r0, kpad), np.float32); n = min(r1, N) - r0
      w[:n, :K] = R.bf16_to_f32(np.frombuffer(raw[r0 * K * 2:(r0 + n) * K * 2], np.uint16)).reshape(n, K)
      h = w.astype(np.float16).view(np.uint16)
      panels = h.reshape((r1 - r0) // 96, 6, 4, 4, kpad // 96, 24, 4).transpose(0, 4, 1, 5, 2, 3, 6)   # [g][slice][s][kk][j][n][k] = pack_b_group per group
      fh.write(np.ascontiguousarray(panels).tobytes())
  os.replace(f + ".tmp", f)
  np.savez(os.path.join(out, "outside_small.npz"), norm=sf.f32("model.language_model.norm.weight"), lm_head_n=np.array([N, npad, kpad]))

def e4m3_encode(x):
  """float32 -> E4M3 (fn) codes, round to nearest (|x| <= 448 expected)."""
  vals = G.e4m3_table()[:0x7F].astype(np.float64)                         # the 127 finite non-negative codes, ascending
  mid = (vals[1:] + vals[:-1]) / 2
  c = np.searchsorted(mid, np.abs(x).astype(np.float64)).astype(np.uint8)
  return c | np.where(x < 0, 0x80, 0).astype(np.uint8)

HEAD_PARTS = 4
def pack_lm_head_fp8(W, out):
  """lm_head (bf16 [248320, 5120]) -> E4M3 with a scale per 128 x 128 block (amax / 448), the layers' block-scaled GEMM stream
  (ks 32, 3 strips a group), in HEAD_PARTS files `lm_head_fp8_{p}.bin` of whole groups (each mapped into a slot on its own)."""
  sf = W.file("outside.safetensors"); dt, shape, raw = sf.raw("lm_head.weight"); N, K = shape; assert N % 128 == 0 and K % 128 == 0
  gs = 16 * NS; ngroups = -(-N // gs); per = -(-ngroups // HEAD_PARTS); per = -(-per // 8) * 8          # groups a part, a multiple of 8 (384 rows)
  parts = []
  for p_ in range(HEAD_PARTS):
    g0, g1 = p_ * per, min(ngroups, (p_ + 1) * per); f = os.path.join(out, f"lm_head_fp8_{p_}.bin")
    with open(f + ".tmp", "wb") as fh:
      for c0 in range(g0, g1, 64):                                          # 64 groups (3072 rows, 24 scale blocks) at a time
        c1 = min(g1, c0 + 64); r0, r1 = c0 * gs, min(c1 * gs, N)
        w = R.bf16_to_f32(np.frombuffer(raw[r0 * K * 2:r1 * K * 2], np.uint16)).reshape(r1 - r0, K)
        nb = -(-(r1 - r0) // 128); wp = np.zeros((nb * 128, K), np.float32); wp[:r1 - r0] = w
        amax = np.abs(wp.reshape(nb, 128, K // 128, 128)).max((1, 3)); sinv = np.maximum(amax, 1e-12).astype(np.float32) / 448.0
        codes = e4m3_encode(wp / np.repeat(np.repeat(sinv, 128, 0), 128, 1))[:r1 - r0]
        fh.write(np.concatenate([G.pack_b_group_bscale(codes, sinv, g, NS, KS, scales=SCALES) for g in range(c1 - c0)]).tobytes())
    os.replace(f + ".tmp", f); parts.append((g0, g1))
  np.savez(os.path.join(out, "lm_head_fp8.npz"), parts=np.array(parts, np.int64), n=np.array([N, K]))

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--dir", default=os.environ.get("QWEN_DIR", "/mnt/ssd/qwen3.8-27b-fp8")); ap.add_argument("--out", default=os.environ.get("QWEN_NPU", "/mnt/ssd/qwen3.8-27b-npu"))
  ap.add_argument("--layers", default="0-63"); ap.add_argument("--lm-head", action="store_true", help="pack lm_head + the final norm (outside.safetensors)"); ap.add_argument("--lm-head-fp8", action="store_true", help="lm_head as E4M3 with 128 x 128 block scales (the layers' GEMM path)")
  ap.add_argument("--mtp", action="store_true", help="the MTP head (speculative decoding's draft model) as layer 64")
  ap.add_argument("--fuse", action="store_true", help="the same-input projections (q | k|v, qkv | z) as one stream file per layer, alongside (no checkpoint read)")
  ap.add_argument("--scales", default="dup", choices=("dup", "single"), help="the block-scale table's layout in every stream (single: each scale once, a `scales` marker in the cache)")
  a = ap.parse_args()
  os.makedirs(a.out, exist_ok=True); lo, hi = (int(v) for v in a.layers.split("-"))
  if a.fuse: fuse_all(a.out, range(lo, hi + 1)); sys.exit(0)                      # a byte concatenation: either layout
  SCALES = a.scales; scales_marker(a.out, SCALES)
  W = R.Weights(a.dir)
  if a.lm_head:
    t0 = time.perf_counter(); pack_lm_head(W, a.out); print(f"   lm_head packed in {time.perf_counter() - t0:.0f} s", flush=True); sys.exit(0)
  if a.lm_head_fp8:
    t0 = time.perf_counter(); pack_lm_head_fp8(W, a.out); print(f"   lm_head (fp8, block scales) packed in {time.perf_counter() - t0:.0f} s", flush=True); sys.exit(0)
  if a.mtp:
    t0 = time.perf_counter(); pack_mtp(W, a.out); print(f"   the MTP head packed as layer {MTP_L} in {time.perf_counter() - t0:.0f} s", flush=True); sys.exit(0)
  for l in range(lo, hi + 1):
    if os.path.exists(os.path.join(a.out, f"L{l}_small.npz")): print(f"   layer {l:2d} present"); continue
    t0 = time.perf_counter(); pack_layer(W, l, a.out); print(f"   layer {l:2d} ({R.LAYER_TYPES[l]}) packed in {time.perf_counter() - t0:.1f} s", flush=True)

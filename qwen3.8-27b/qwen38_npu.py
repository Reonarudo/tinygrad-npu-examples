#!/usr/bin/env python3
"""Qwen3.8-27B's language-model layers on the NPU. The linears run on the TEC matrix unit through `gemm_gs(b8, bscale)` (the
E4M3 codes with their 128 x 128 block scales, streamed per layer from `qwen38_pack.py`'s cache through two device slots);
norms, RoPE, the attention softmax, SwiGLU, the gates and the GEMM layouts are tinygrad ops (the backend's generic kernels).
The residual stream is fp32 [R, 5120] with R = 12 * nrb rows (nrb even); GEMM inputs are fp16.

    python3 qwen38_npu.py --gate 3          (layer 3 on a random input vs the numpy reference)
"""
import argparse, ctypes, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad")))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinygrad import Tensor, dtypes                                      # noqa: E402
from zy import OA, gemm_plan as GP                                       # noqa: E402
import qwen38_ref as R                                                   # noqa: E402
from qwen38_kernels import Kernels, NT, gdn_hpt, ring_pitch                       # noqa: E402
GDN_TOK = os.environ.get("QWEN_GDN_TOK", "3")      # the decode DeltaNet step: 3 = gdn_tok3 (folded), 2 = gdn_tok2, 1 = gdn_tok

# Q8MODE (the Q8_0 models): 1 = gemm_gs(q8) -- a K-slice one 32-weight scale block, corrected and scaled per slice --; 2 = the
# weights built in fp16 by the expand (gemm_gs q8=2: `ornith_pack.py --q8f`'s cache, marked by a file `q8f` in it), 32-step slices
Q8MODE = 0 if not R.Q8 else 2 if os.path.exists(os.path.join(os.path.expanduser(os.environ.get("QWEN_NPU", "")), "q8f")) else 1
DEV = "ZHOUYI"; KS, NS = (8 if Q8MODE == 1 else 32), 3     # k-steps a K-slice (E4M3: one 128 x 128 scale block; Q8_0: 32 or 128 weights)
# SCALES (the E4M3 caches): the block-scale table's layout in the streams -- "dup" (each fp32 scale twice, 128 B a strip and
# slice, the original `pack_b_group_bscale`) or "single" (once, 64 B: qwen38_pack.py --scales single / qwen38_repack_scales.py,
# marked by a file `scales` holding the word in the cache). Passed to gemm_gs(scales=): the marker alone decides the layout.
def _scales_marker(cache):
  f = os.path.join(os.path.expanduser(cache), "scales")
  return open(f).read().split()[0] if os.path.exists(f) else "dup"
SCALES = "dup" if R.Q8 else _scales_marker(os.environ.get("QWEN_NPU", "/mnt/ssd/qwen3.8-27b-npu-s1"))
# TERN (a ternary cache, e.g. bonsai2_pack.py's, marked by a file `tern`): every stream is gemm_fp16.pack_b_group_tern's (2-bit codes,
# the single-layout scale table) and the GEMMs run gemm_gs(tern=True); the keyword is passed only then (older backends lack it)
TERN = not R.Q8 and os.path.exists(os.path.join(os.path.expanduser(os.environ.get("QWEN_NPU", "/mnt/ssd/qwen3.8-27b-npu-s1")), "tern"))
# TSCALE (a ternary cache's scale tables): "f32" (bonsai2_pack.py's single-layout fp32 s x 2^18, 64 B a strip and slice) or "f16"
# (fp16 s x 2^15, 32 B: 2.125 bits a weight, C bit-identical; bonsai2_repack_f16.py's / bonsai2_pack.py --tscale f16's cache, marked by
# a file `tscale` holding the word). Passed to gemm_gs(tscale=) only for f16 (older backends lack the keyword).
def _tscale_marker(cache):
  f = os.path.join(os.path.expanduser(cache), "tscale")
  return open(f).read().split()[0] if os.path.exists(f) else "f32"
TSCALE = _tscale_marker(os.environ.get("QWEN_NPU", "/mnt/ssd/qwen3.8-27b-npu-s1")) if TERN else "f32"
assert TSCALE in ("f32", "f16"), f"`tscale` marker {TSCALE!r}: f32 | f16"
TERN_KW = ({"tern": True, "tscale": "f16"} if TSCALE == "f16" else {"tern": True}) if TERN else {}
if TERN: assert SCALES == "single", "a ternary cache holds the single-layout scale table (its `scales` marker)"

def had_signs(cache):
  """A rotated-input model's cache (R.HAD): hadamard.npz -> ({width: the +-1 signs (float32)}, the factor had_a32 applies after its
  transform: 1.0 when the cache's scale tables carry the 1 / sqrt(block), else that factor)."""
  z = np.load(os.path.join(os.path.expanduser(cache), "hadamard.npz"))
  return {int(k[1:]): z[k].astype(np.float32) for k in z.files if k.startswith("s")}, (1.0 if int(z["fold_norm"]) else float(1.0 / np.sqrt(int(z["block"]))))

DEBUG_LOAD = os.environ.get("QWEN_DEBUG", "0") == "1"
# QWEN_FUSE (default 1): the same-input projections as ONE GEMM where the cache has them (qwen38_pack.py --fuse: L{l}_qkv.bin =
# q | k|v for the attention layers, L{l}_qkvz.bin = qkv | z for the DeltaNet layers, each with L{l}_fused.npz); the consumers read
# their linear's C tiles at its group offset in the one buffer. 0: the per-linear files, whatever the cache holds.
FUSE = os.environ.get("QWEN_FUSE", "1") != "0"

def read_into(path, mv, size, nt=8, drop=False):
  """The file's bytes into `mv` (any writable buffer, e.g. a device mapping) by `nt` parallel preadv's over page-aligned slices.
  `drop`: release the file's page cache afterwards (the bytes now live in `mv`; the cache would crowd out contiguous memory)."""
  import threading
  fd = os.open(path, os.O_RDONLY)
  def part(lo, hi):
    while lo < hi: lo += os.preadv(fd, [mv[lo:hi]], lo)
  b = [(size * i // nt) & ~4095 for i in range(nt)] + [size]
  ths = [threading.Thread(target=part, args=(b[i], b[i + 1])) for i in range(nt)]
  for t in ths: t.start()
  for t in ths: t.join()
  if drop: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
  os.close(fd)

def memmove_threads(dst, src, size, nt=4):
  import threading
  b = [(size * i // nt) & ~63 for i in range(nt)] + [size]
  ths = [threading.Thread(target=ctypes.memmove, args=(dst + b[i], src + b[i], b[i + 1] - b[i])) for i in range(nt)]
  for t in ths: t.start()
  for t in ths: t.join()

def dev(a, dt=None): return Tensor(np.ascontiguousarray(a), device=DEV, **({} if dt is None else {"dtype": dt})).realize()
def sync():
  """Wait for a job the backend left in flight (zhouyi's async graph tail; a no-op on a backend without one)."""
  from tinygrad import Device
  Device[DEV].synchronize()
def slot_map_nodrain(raw, sid, wid, off, n):
  """SLOT_MAP without waiting for a job left in flight: only for a slot that job does not read (the layer loop's slot l % 2: the
  in-flight job is layer l - 1's, on the other slot). An older backend has no `drain` keyword (and no async tail)."""
  try: raw.slot_map(sid, wid, off, n, drain=False)
  except TypeError: raw.slot_map(sid, wid, off, n)
def poke(t, a):
  """The host array `a` into the start of the realised device tensor `t`'s buffer by a memmove into its host mapping -- what an
  assign of a host tensor ends in (the allocator's _copyin) without scheduling a copy and waiting for it (~1.5 ms a round trip).
  Only between jobs: a job left in flight is waited for first (`sync`), so no kernel is reading the buffer meanwhile."""
  sync(); a = np.ascontiguousarray(a); b = t.uop.base.buffer; b.ensure_allocated(); raw = b._buf
  assert a.nbytes <= b.nbytes and raw.va, (a.nbytes, b.nbytes)
  ctypes.memmove(raw.va, a.ctypes.data, a.nbytes)
def rup(x, m): return -(-x // m) * m

class Layers:
  def __init__(self, n, cache, fp8=False):
    """`fp8`: a geometry over another model's E4M3 cache next to a ternary one (bonsai2-27b's drafter: Qwen3.8-27B's MTP layer) --
    its GEMMs without the ternary stream, its A producers without the Hadamard transform, its norms without the signs."""
    self.n, self.cache, self.fp8 = n, cache, fp8
    # SCALES was read from QWEN_NPU at import; the cache opened here (qwen38_generate.py --cache, check scripts) must be the same
    # layout, or the kernels would read the other layout's codes as scales -- silently (nothing downstream checks it)
    assert R.Q8 or _scales_marker(cache) == SCALES, f"{cache}: `scales` marker {_scales_marker(cache)!r} but SCALES={SCALES!r} (set QWEN_NPU to this cache)"
    assert R.Q8 or os.path.exists(os.path.join(cache, "tern")) == (TERN and not fp8), f"{cache}: its `tern` marker disagrees with TERN={TERN} (set QWEN_NPU to this cache)"
    assert fp8 or not TERN or _tscale_marker(cache) == TSCALE, f"{cache}: `tscale` marker {_tscale_marker(cache)!r} but TSCALE={TSCALE!r} (set QWEN_NPU to this cache)"
    self.tern_kw = {} if fp8 else TERN_KW; self.had = R.HAD and not fp8
    self.nrb = rup(rup(n, 12) // 12, 2); self.R = 12 * self.nrb
    self.slots = [{}, {}]                                                 # two slots of device buffers per linear name
    self._metas, self._inflight, self.pinned, self._small_dev, self.stage = {}, {}, {}, {}, {}
    self.zc = {}                                                                   # zero-copy state (pin), shared with the other geometry
    self.K = Kernels(self.nrb, self.R, R.EPS, real=n)                              # the hand-written layout kernels (ks 32)
    if self.had:                 # rotated linear inputs: the producers transform (Kernels.had), the norms' 1 + w carry the signs (_meta)
      assert TERN, f"{cache}: QWEN_MODEL={R.MODEL} needs its ternary cache (a `tern` marker; set QWEN_NPU to it)"
      self.signs, self.K.had_post = had_signs(cache); self.K.had = {w: dev(v) for w, v in self.signs.items()}
    self.piece = self.nrb if self.nrb < 4 else 0     # rows mode (decode, the fp8 head): one piece; full row blocks ask gemm_plan (lin_ct)
    inv = 1.0 / (R.THETA ** (np.arange(0, R.ROT, 2, dtype=np.float64) / R.ROT)); f = np.arange(self.R, dtype=np.float64)[:, None] * inv[None]
    emb = np.concatenate([f, f], -1)
    self.cos, self.sin = dev(np.cos(emb).astype(np.float32).reshape(self.R, 1, R.ROT)), dev(np.sin(emb).astype(np.float32).reshape(self.R, 1, R.ROT))
    i, j = np.arange(self.R)[:, None], np.arange(self.R)[None]
    self.mask = dev(np.where((j <= i) & (j < n), 0.0, -1e30).astype(np.float32))

  # ---- weights: layer l's streams into slot parity l % 2 (`prefetch(l)` starts the copy on a thread; `load(l)` waits for it).
  # The streams of the first `pin_gb` GB of layers are read into process RAM once (`pin`) and memmoved from there (4 threads,
  # ~18 GB/s); the other layers are read from the NVMe straight into the device buffers by 8 parallel preadv's (~3 GB/s):
  # per token the 26 GB do not fit the page cache, so relying on it means re-reading everything at one stream's 1.2 GB/s.
  def _meta(self, l):
    if l not in self._metas:
      z = np.load(os.path.join(self.cache, f"L{l}_small.npz")); names = [str(k) for k in z["meta_names"]]
      small = {k: z[k] for k in z.files if not k.startswith("meta")}
      for k in ("input_layernorm_weight", "post_attention_layernorm_weight"): small[k] = (1.0 + small[k]).astype(np.float32)   # rms_a32 takes 1 + w
      if "linear_attn_in_proj_a_weight" in small:      # a | b stacked, the input norm's (1 + w) folded in (gemv normalises the rows itself); the conv taps as [CONV, C]
        small["linear_attn_in_proj_ab_weight"] = np.ascontiguousarray(np.concatenate([small["linear_attn_in_proj_a_weight"], small["linear_attn_in_proj_b_weight"]], 0) * small["input_layernorm_weight"][None])
        small["linear_attn_conv1d_weight_t"] = np.ascontiguousarray(small["linear_attn_conv1d_weight"].reshape(-1, R.CONV).T)
        del small["linear_attn_in_proj_a_weight"], small["linear_attn_in_proj_b_weight"]   # only their concatenation is used
      if self.had:               # the linears' input signs into the norms that produce their inputs (after a|b took the plain 1 + w)
        for k in ("input_layernorm_weight", "post_attention_layernorm_weight"): small[k] = (small[k] * self.signs[R.H]).astype(np.float32)
      meta = {k: tuple(int(v) for v in m) for k, m in zip(names, z["meta"])}
      fz = os.path.join(self.cache, f"L{l}_fused.npz")
      if FUSE and os.path.exists(fz):     # the fused linear replaces its parts in the file list; the parts keep (N, npad, K) + (parent, group offset) under "_sub"
        f = np.load(fz); name = str(f["name"]); subs = [str(k) for k in f["subs"]]; goff = [int(g) for g in f["goff"]]
        assert all(k in meta for k in subs), (l, subs, list(meta))
        fused, sub = {}, {}
        for k in meta:
          if k == subs[0]: fused[name] = tuple(int(v) for v in f["meta"])
          if k in subs: sub[k] = (name, goff[subs.index(k)]) + meta[k]
          else: fused[k] = meta[k]
        meta = dict(fused, _sub=sub)
      self._metas[l] = (meta, small)
    return self._metas[l]
  @staticmethod
  def files(meta): return [k for k in meta if not k.startswith("_")]        # the linears with a stream file (a fused one for its parts)
  def _slot_buf(self, l, k, size):
    slot = self.slots[l % 2]
    if k not in slot or slot[k].shape[0] != size:
      slot[k] = Tensor.empty(size, device=DEV, dtype=dtypes.uint8); slot[k].uop.buffer.ensure_allocated()
      if os.environ.get("QWEN_PA"): print(f"[slot] {l % 2} {k} n={size} pa={slot[k].uop.buffer._buf.pa:#x}", flush=True)
    return slot[k].uop.buffer._buf.va
  def _sizes(self, l):
    meta, _ = self._meta(l); return {k: os.path.getsize(os.path.join(self.cache, f"L{l}_{k}.bin")) for k in self.files(meta)}
  # ---- zero-copy (the KMD's weight buffers and slots): a layer's five linears packed in one weight buffer (each 2 MiB-aligned,
  # 16 B of load-overhang slack after it); slot l % 2 re-pointed at it by load(l); the linears' tensors are fixed views into the
  # slot per (parity, layer type), so the JIT sees the same buffers every call. Streamed layers: two staging weight buffers,
  # refilled only after the layer mapped from them has run (the NPU reads them in place).
  def _pack(self, l):
    sizes = self._sizes(l); offs, o = {}, 0
    for k, sz in sizes.items(): offs[k] = o; o += -(-(sz + 16) // (2 << 20)) * (2 << 20)
    return sizes, offs, o
  def _pin_zc(self, gb):
    from tinygrad import Device
    raw = Device[DEV].raw; t0 = time.perf_counter(); zc = self.zc
    packs = {l: self._pack(l) for l in range(R.NL)}; tot = {l: p_[2] for l, p_ in packs.items()}; big = max(tot.values())
    try: slots = [raw.slot_alloc(big) for _ in range(2)]
    except OSError as e: print(f"   zero-copy unavailable ({e}); copying the weights", flush=True); return False
    budget = gb * 1e9 - 2 * big                                                   # the two staging buffers are inside the budget
    staging = [raw.wbuf_alloc(big) for _ in range(2)]                            # first, while contiguous memory is plentiful
    stream = []
    if sum(tot.values()) > budget:
      k = 1
      while sum(sorted(tot.values())[:R.NL - k]) > budget: k += 1
      stream = sorted({min(63, int(64 * (j + 0.5) / k)) for j in range(k)})
    wbufs, held = {}, 0
    for l in range(R.NL):
      if l in stream: continue
      sizes, offs, n = packs[l]
      try: wid, mm = raw.wbuf_alloc(n)
      except OSError: stream.append(l); continue                                  # no contiguous chunks left: stream this layer
      for k_, sz in sizes.items(): read_into(os.path.join(self.cache, f"L{l}_{k_}.bin"), memoryview(mm)[offs[k_]:offs[k_] + sz], sz, drop=True)
      wbufs[l] = (wid, mm); held += n
    stream = sorted(stream)
    if not stream:
      for wid, mm in staging: mm.close(); raw.wbuf_free(wid)
      staging = []
    views = {}
    for par in range(2):
      for lt in ("linear", "full"):
        l0 = R.LAYER_TYPES.index(lt); sizes, offs, _ = packs[l0]
        views[(par, lt)] = {k_: Tensor.from_blob(slots[par][1] + offs[k_], (sz,), dtype=dtypes.uint8, device=DEV) for k_, sz in sizes.items()}
    zc.update(raw=raw, slots=slots, views=views, packs=packs, wbufs=wbufs, stream=stream, staging=staging, st_layer=[None, None],
              st_thread=[None, None], st_free=[], st_next=0)
    for i in range(len(staging)): self._stage_fill(i)
    if DEBUG_LOAD: print(f"   zero-copy: {len(wbufs)} layers in weight buffers, {held / 1e9:.1f} GB in {time.perf_counter() - t0:.0f} s; streamed "
                         f"through two {big / 1e9:.2f} GB staging buffers: layers {stream}", flush=True)
    return True
  def _stage_fill(self, i):
    """Read the next streamed layer (cyclically) into staging buffer i on a thread."""
    import threading
    zc = self.zc; l = zc["stream"][zc["st_next"] % len(zc["stream"])]; zc["st_next"] += 1
    sizes, offs, _ = zc["packs"][l]; mm = zc["staging"][i][1]
    def read():
      for k_, sz in sizes.items(): read_into(os.path.join(self.cache, f"L{l}_{k_}.bin"), memoryview(mm)[offs[k_]:offs[k_] + sz], sz)
    zc["st_layer"][i] = l; zc["st_thread"][i] = threading.Thread(target=read, daemon=True); zc["st_thread"][i].start()
  def _map_zc(self, l):
    """Point slot l % 2 at layer l's weights (on the prefetch thread: the slot's previous layer, l - 2, has run)."""
    zc = self.zc; src = zc.setdefault("slot_src", [None, None])
    if l in zc["wbufs"]: wid = zc["wbufs"][l][0]; src[l % 2] = None
    else:
      if l in zc["st_layer"]: i = zc["st_layer"].index(l); zc["st_thread"][i].join()
      else:                                                                      # out of the cyclic order (tests, profilers): read it now
        i = next(j for j in range(len(zc["staging"])) if src[1 - l % 2] != j); zc["st_thread"][i].join()
        sizes, offs, _ = zc["packs"][l]; mm = zc["staging"][i][1]
        for k_, sz in sizes.items(): read_into(os.path.join(self.cache, f"L{l}_{k_}.bin"), memoryview(mm)[offs[k_]:offs[k_] + sz], sz)
        zc["st_layer"][i] = l
      wid = zc["staging"][i][0]; src[l % 2] = i
    # no drain: slot l % 2 was last read by layer l - 2, which has run (a job left in flight is layer l - 1's, on the other slot;
    # the prefetch thread starts only after a sync); and a thread must not drain
    slot_map_nodrain(zc["raw"], zc["slots"][l % 2][0], wid, 0, zc["packs"][l][2])
  # ---- layer blocks (QWEN_LAYER_BLOCK, Model.verify): k consecutive layers as ONE TinyJit call (one job), so each of the 2 x k
  # positions (block parity, layer in block) has a window slot of its own; position (bp, 0) is the plain path's slot bp. The
  # layers' linears are views into their position's slot, fixed per (position, layer type): the JIT sees the same buffers.
  def block_slots(self, k):
    """The 2 k block slots (allocated once, shared with the other geometries through `zc`) -> {(bp, j): (sid, pa)}."""
    zc = self.zc
    if zc.get("bslots") is None or len(zc["bslots"]) != 2 * k:
      big = max(p_[2] for p_ in zc["packs"].values()); raw = zc["raw"]
      zc["bslots"] = {(bp, j): (zc["slots"][bp] if j == 0 else raw.slot_alloc(big)) for bp in range(2) for j in range(k)}
      zc["bviews"], zc["bsize"] = {}, big
    return zc["bslots"]
  def block_views(self, bp, j, lt):
    zc = self.zc; key = (bp, j, lt)
    if key not in zc["bviews"]:
      sizes, offs, _ = zc["packs"][R.LAYER_TYPES.index(lt)]; pa = zc["bslots"][(bp, j)][1]
      zc["bviews"][key] = {k_: Tensor.from_blob(pa + offs[k_], (sz,), dtype=dtypes.uint8, device=DEV) for k_, sz in sizes.items()}
    return zc["bviews"][key]
  def blob_view(self, bp, j, n):
    """A uint8 view of n bytes at block slot (bp, j)'s start (e.g. a head part mapped there), cached."""
    zc = self.zc; key = ("blob", bp, j, n)
    if key not in zc["bviews"]: zc["bviews"][key] = Tensor.from_blob(zc["bslots"][(bp, j)][1], (n,), dtype=dtypes.uint8, device=DEV)
    return zc["bviews"][key]
  def map_block(self, l, bp, j):
    """Layer l (pinned zero-copy) into block slot (bp, j), without waiting for a job left in flight: the caller's in-flight job
    is the other block parity's (this slot's last reader, two blocks back, has run) -> (its linears' views, meta)."""
    zc = self.zc; slot_map_nodrain(zc["raw"], zc["bslots"][(bp, j)][0], zc["wbufs"][l][0], 0, zc["packs"][l][2])
    return self.block_views(bp, j, R.LAYER_TYPES[l]), self._meta(l)[0]
  def _load_zc(self, l):
    zc = self.zc; prev = (l - 1) % R.NL
    for i, sl in enumerate(zc["st_layer"]):                                     # layer l - 1 has run: its staging buffer is free
      if sl == prev: sync(); self._stage_fill(i)                               # (its job may still be in flight: wait for it)
    if l in self._inflight: self._inflight.pop(l).join()
    else: self._map_zc(l)
    return zc["views"][(l % 2, R.LAYER_TYPES[l])]

  def pin(self, gb):
    if os.environ.get("QWEN_ZC", "1") != "0" and self._pin_zc(gb): return
    """Hold `gb` GB of layer streams in RAM. The layers that do not fit are spread evenly over the model and read ahead
    from the NVMe through a one-layer staging buffer (inside the budget): the read of the next unpinned layer starts as
    soon as the previous one has been consumed, so each read has ~1/k of a step to finish instead of one layer's time."""
    t0 = time.perf_counter(); sizes = {l: self._sizes(l) for l in range(R.NL)}; tot = {l: sum(v.values()) for l, v in sizes.items()}
    stage_bytes = max(tot.values()); budget = gb * 1e9
    if sum(tot.values()) <= budget: out = []
    else:
      budget -= stage_bytes; k = 1
      while sum(sorted(tot.values())[:R.NL - k]) > budget: k += 1                      # how many layers must stream
      out = sorted({min(63, int(64 * (j + 0.5) / k)) for j in range(k)})
    held = 0
    for l in range(R.NL):
      if l in out: continue
      self.pinned[l] = {}
      for k_, size in sizes[l].items():
        a_ = np.empty(size, np.uint8); read_into(os.path.join(self.cache, f"L{l}_{k_}.bin"), memoryview(a_), size); self.pinned[l][k_] = a_; held += size
    self.stage.update(order=out, buf=np.empty(stage_bytes, np.uint8) if out else None, layer=None, thread=None, offs={})
    if out: self._stage_start(out[0])
    if DEBUG_LOAD: print(f"   pinned {len(self.pinned)} layers, {held / 1e9:.1f} GB in {time.perf_counter() - t0:.0f} s; streamed from the NVMe through a "
                         f"{stage_bytes / 1e9:.2f} GB staging buffer: layers {out}", flush=True)
  def _stage_start(self, l):
    """Read layer l's streams into the staging buffer on a thread."""
    import threading
    st = self.stage; offs, o = {}, 0
    for k_, size in self._sizes(l).items(): offs[k_] = (o, size); o += size
    def read():
      for k_, (o_, size) in offs.items(): read_into(os.path.join(self.cache, f"L{l}_{k_}.bin"), memoryview(st["buf"])[o_:o_ + size], size)
    st.update(layer=l, offs=offs, thread=threading.Thread(target=read, daemon=True)); st["thread"].start()
  def _copy(self, l):
    if self.zc: return                                                             # zero-copy: nothing to move
    meta, _ = self._meta(l); st = self.stage
    if l not in self.pinned and st.get("layer") == l:                              # staged: wait for the read, copy from RAM, stage the next
      st["thread"].join()
      for k in self.files(meta): memmove_threads(self._slot_buf(l, k, st["offs"][k][1]), st["buf"].ctypes.data + st["offs"][k][0], st["offs"][k][1])
      order = st["order"]; self._stage_start(order[(order.index(l) + 1) % len(order)]); return
    for k in self.files(meta):
      f = os.path.join(self.cache, f"L{l}_{k}.bin"); size = os.path.getsize(f); va = self._slot_buf(l, k, size)
      if l in self.pinned: memmove_threads(va, self.pinned[l][k].ctypes.data, size)
      else: read_into(f, memoryview((ctypes.c_uint8 * size).from_address(va)), size)
  def prefetch(self, l):
    """Start layer l's weight copy into slot l % 2 on a thread (it overlaps the device work of layer l - 1)."""
    import threading
    if l in self._inflight or os.environ.get("QWEN_NOPREFETCH"): return       # QWEN_NOPREFETCH=1: copy / map in load()
    # a layer pinned zero-copy only needs its slot mapped (one ioctl): a thread for it cost more than it hid (bonsai2-27b, 4-row
    # verify: -10 ms a pass without it). QWEN_PREFETCH=1: the thread anyway. Streamed layers keep it
    if self.zc and l in self.zc["wbufs"] and os.environ.get("QWEN_PREFETCH") != "1": return
    sync()                                                                         # slot l % 2's last reader (layer l - 2) may still be in flight
    t = threading.Thread(target=self._map_zc if self.zc else self._copy, args=(l,), daemon=True); t.start(); self._inflight[l] = t
  def load(self, l):
    if self.zc: slot = self._load_zc(l)
    else:
      if l in self._inflight: self._inflight.pop(l).join()
      else: self._copy(l)
      slot = self.slots[l % 2]
    meta, small = self._meta(l)
    if l not in self._small_dev: self._small_dev[l] = {k: dev(v) for k, v in small.items()}   # uploaded once, device-resident after
    return slot, meta, self._small_dev[l]

  # ---- the GEMM's layouts (ks 32: 128-wide K slices of 32 k-quads; 3 strips of 16 columns a group)
  def A(self, x):
    Rr, K = x.shape
    return x.cast(dtypes.half).reshape(Rr // 12, 3, 4, K // (4 * KS), KS, 4).permute(3, 0, 4, 1, 2, 5).contiguous().bitcast(dtypes.uint16)
  def C(self, ct, ngroups):
    return ct.reshape(ngroups, self.nrb, NS, 3, 4, 4, 4).permute(1, 3, 5, 0, 2, 4, 6).reshape(self.R, 16 * NS * ngroups)
  def lin_ct(self, a, slot, meta, k, full=False):
    """The GEMM of a prebuilt A layout: the raw fp32 C tiles [ngroups][nrb][3][192], in the persistent buffer `ct_<k>`."""
    N, npad, K = meta[k]; ng = npad // (16 * NS)
    out = self.K.buf(f"ct_{k}", ng * self.nrb * NS * 192, dtypes.float32)
    rows = 0 if full or not self.K.compact else self.n        # decode / short prompts: only the real rows (A in the compact layout)
    # the row piece from gemm_plan's board-fitted model: one piece of all the row blocks up to 28 (each B slice reused across
    # more rows) -- 3.5-23 % faster than half the row blocks at 4-24 blocks, and nrb = 6 no longer asks for an odd piece of 3
    piece = self.nrb if (rows or R.Q8) else GP.plan_piece(K, npad, self.nrb, ks=KS, ns=NS, b8=True, bscale=True, rows=rows)
    return OA.gemm_gs(a, slot[k], ks=KS, ns=NS, nrb=self.nrb, nslices=K // (4 * KS), ngroups=ng, b8=True, bscale=True, q8=Q8MODE, scales=SCALES, piece=piece, rows=rows, out=out, **self.tern_kw).realize()
  def lin_cts(self, a, slot, meta, names, full=False):
    """The GEMMs of the linears `names` on one A -> {name: (C tiles buffer, first group)}. Linears fused in the cache (meta "_sub":
    name -> (parent, group offset, N, npad, K)) run as their parent's one GEMM; each consumer reads its slice at the offset."""
    sub = meta.get("_sub", {}); done, out = {}, {}
    for n in names:
      if n in sub:
        parent, goff = sub[n][:2]
        if parent not in done: done[parent] = self.lin_ct(a, slot, meta, parent, full)
        out[n] = (done[parent], goff)
      else: out[n] = (self.lin_ct(a, slot, meta, n, full), 0)
    return out
  def lin(self, x, slot, meta, k, a=None):
    """Linear k on fp32 rows x -> fp32 [R, N] (tinygrad views; a part of a fused linear: its parent's GEMM, the columns from its group)."""
    parent, goff, N, npad, K = meta["_sub"][k] if k in meta.get("_sub", {}) else (k, 0) + meta[k]
    if a is None:
      if x.shape[1] < K: x = x.pad(((0, 0), (0, K - x.shape[1])))
      a = self.A(x)
    c0 = 16 * NS * goff; return self.C(self.lin_ct(a, slot, meta, parent, full=True), meta[parent][1] // (16 * NS))[:, c0:c0 + N]

  @staticmethod
  def rms(x, w, eps=R.EPS): return x * (x.square().mean(-1, keepdim=True) + eps).rsqrt() * (1.0 + w)   # Qwen3_5RMSNorm

  # ---- the layers (prefill: all R rows at once)
  def attention_layer(self, x, l):
    slot, meta, sm = self.load(l); Rr = self.R
    x = self.K.rows_buf("x_in", R.H).assign(x).realize()                    # the residual into a persistent buffer (the kernels' rule, README)
    a_h = self.K.rms_a(x, self.K.hold("w_in", sm["input_layernorm_weight"]), R.H, "a_h")
    ct = self.lin_cts(a_h, slot, meta, ("q", "kv"))                           # one GEMM when the cache fuses q | k|v
    qg = self.K.unpack(ct["q"][0], R.NH * 2 * R.HD, "q_rows", goff=ct["q"][1]).reshape(Rr, R.NH, 2 * R.HD); q, gate = qg[..., :R.HD], qg[..., R.HD:]
    kv = self.K.unpack(ct["kv"][0], 2 * R.NKV * R.HD, "kv_rows", goff=ct["kv"][1]); k = kv[:, :R.NKV * R.HD].reshape(Rr, R.NKV, R.HD); v = kv[:, R.NKV * R.HD:].reshape(Rr, R.NKV, R.HD)
    q = self.rms(q, sm["self_attn_q_norm_weight"]); k = self.rms(k, sm["self_attn_k_norm_weight"])
    def rope(t):
      tr, tp = t[..., :R.ROT], t[..., R.ROT:]
      rot = Tensor.cat(-tr[..., R.ROT // 2:], tr[..., :R.ROT // 2], dim=-1)
      return Tensor.cat(tr * self.cos + rot * self.sin, tp, dim=-1)
    q, k = rope(q), rope(k)
    g = R.NH // R.NKV
    kr = k.reshape(Rr, R.NKV, 1, R.HD).expand(Rr, R.NKV, g, R.HD).reshape(Rr, R.NH, R.HD).permute(1, 2, 0)      # [NH, HD, R]
    vr = v.reshape(Rr, R.NKV, 1, R.HD).expand(Rr, R.NKV, g, R.HD).reshape(Rr, R.NH, R.HD).permute(1, 0, 2)      # [NH, R, HD]
    s = q.permute(1, 0, 2) @ kr * (1.0 / math.sqrt(R.HD)) + self.mask                                            # [NH, R, R]
    o = (s.softmax(-1) @ vr).permute(1, 0, 2) * gate.sigmoid()                                                   # [R, NH, HD]
    oc = self.K.rows_buf("o_rows", R.NH * R.HD).assign(o.reshape(Rr, R.NH * R.HD)).realize()      # into a persistent buffer (README)
    x = self.K.resid(x, self.lin_ct(self.K.rms_a(oc, None, R.NH * R.HD, "a_o", norm=False), slot, meta, "o"), R.H, "x1")
    return self.mlp(x, slot, meta, sm["post_attention_layernorm_weight"]).realize()

  _gate_l = None
  def gdn_layer(self, x, l, state=None, conv_state=None):
    """Gated DeltaNet layer l over the first n rows (recurrent form: a Python loop over the tokens, each step a few small
    tinygrad ops on the [48, 128, 128] state). `state` / `conv_state` carry a previous prefix; returns (x_out, (state, conv_state))."""
    slot, meta, sm = self.load(l); self._gate_l = l
    return self.gdn_body(x, slot, meta, sm, state, conv_state)

  @staticmethod
  def ab3(wab):
    """The a|b rows [2 NV, H] (norm weight folded) -> gdn_tok3's per-task layout [NTU][2 x HPT][H] (task t: a rows then b rows of
    its HPT heads, the last task's padded with zero rows): HPT = gdn_hpt(NV), NTU = ceil(NV / HPT)."""
    hpt = gdn_hpt(R.NV); ntu = -(-R.NV // hpt); w = np.zeros((2, ntu * hpt, wab.shape[-1]), np.float32); w[:, :R.NV] = wab.reshape(2, R.NV, -1)
    return np.ascontiguousarray(w.reshape(2, ntu, hpt, -1).transpose(1, 0, 2, 3)).reshape(-1)
  @staticmethod
  def ab3_size():
    """floats of one layer's a|b weights as the token kernels read them (ab3: padded to whole tasks; else 2 NV H)."""
    hpt = gdn_hpt(R.NV); return 2 * (-(-R.NV // hpt)) * hpt * R.H if GDN_TOK == "3" else 2 * R.NV * R.H
  @staticmethod
  def adt(small):
    """gdn_tok3's per-head constants [2][NV]: -exp(A_log), dt_bias."""
    return np.stack([-np.exp(small["linear_attn_A_log"].astype(np.float64)), small["linear_attn_dt_bias"]]).astype(np.float32).reshape(-1)

  def gdn_body(self, x, slot, meta, sm, state=None, conv_state=None, xin=None, out="x2"):
    """n == 1 (decode): `state` = (Sall, Call, idx, posb, Wab, Cwt, Nw[, Adt, Win, Wpost]) -- the layer's delta-rule state and conv
    ring live in persistent stacks (updated in place by one kernel), as do the small weights (a|b projection -- gdn_tok3's layout
    when GDN_TOK is 3 --, conv taps, output norm, the -exp(A_log) | dt_bias pair, the two RMSNorms' 1 + w) indexed by idx;
    returns (x_out, (Sall, Call)). n > 1 (prefill): `state` / `conv_state` are tensors of a previous prefix;
    returns (x_out, (S [NV, DK, DV], conv_state [CONV-1, C])). `xin`: the input already in a persistent buffer (read directly);
    `out`: the tag of the persistent output buffer."""
    Rr, n, C = self.R, self.n, 2 * R.NK * R.DK + R.NV * R.DV; CP = ring_pitch(C)   # the ring / taps row pitch
    x = xin if xin is not None else self.K.rows_buf("x_in", R.H).assign(x).realize()
    tok3 = n == 1 and GDN_TOK == "3"
    if tok3:                  # decode, folded: the layer's small weights and norms from stacks (a fresh one-layer set for the gate test)
      if state is None:
        Sall = self.K.buf("S_scratch", R.NV * R.DK * R.DV, dtypes.float32); Sall.assign(Tensor.zeros(R.NV * R.DK * R.DV, device=DEV)).realize()
        Call = self.K.buf("C_scratch", R.CONV * CP, dtypes.float32); Call.assign(Tensor.zeros(R.CONV * CP, device=DEV)).realize()
        idx = self.K.buf("idx_zero", 1, dtypes.int32); posb = self.K.buf("pos_zero", 1, dtypes.int32); _, small = self._meta(self._gate_l)
        Wab, Adt = self.K.buf("w_ab3", self.ab3_size(), dtypes.float32), self.K.buf("adt1", 2 * R.NV, dtypes.float32)
        Wab.assign(dev(self.ab3(small["linear_attn_in_proj_ab_weight"]))).realize(); Adt.assign(dev(self.adt(small))).realize()
        Nw, Win, Wpost = (self.K.hold(t, sm[k_]) for t, k_ in (("nw", "linear_attn_norm_weight"), ("w_in", "input_layernorm_weight"), ("w_post", "post_attention_layernorm_weight")))
        Cwt = self.K.hold("cw_t", sm["linear_attn_conv1d_weight_t"].pad(((0, 0), (0, CP - C))))               # the taps' rows at pitch CP
      else: Sall, Call, idx, posb, Wab, Cwt, Nw, Adt, Win, Wpost = state; idx = self.K.hold("idx_held", idx)
      a_h = self.K.rms_a(x, (Win, idx), R.H, "a_h")
      ct = self.lin_cts(a_h, slot, meta, ("qkv", "z"))                      # one GEMM when the cache fuses qkv | z
      if R.HAD:              # o_proj's A from the fp32 rows (had_a32 transforms whole 1024-blocks across heads)
        o = self.K.gdn_tok3(Sall, Call, idx, posb, ct["qkv"][0], ct["z"][0], x, Wab, Adt, Cwt, Nw, R.NV, R.NK, R.DK, R.DV, C, R.CONV, R.H, xoff=ct["qkv"][1], zoff=ct["z"][1], rows=True)
        a_o = self.K.rms_a(o, None, R.NV * R.DV, "a_o", norm=False)
      else: a_o = self.K.gdn_tok3(Sall, Call, idx, posb, ct["qkv"][0], ct["z"][0], x, Wab, Adt, Cwt, Nw, R.NV, R.NK, R.DK, R.DV, C, R.CONV, R.H, xoff=ct["qkv"][1], zoff=ct["z"][1])
      x = self.K.resid(x, self.lin_ct(a_o, slot, meta, "o"), R.H, "x1")
      return self.mlp(x, slot, meta, (Wpost, idx), out), (Sall, Call)
    a_h = self.K.rms_a(x, self.K.hold("w_in", sm["input_layernorm_weight"]), R.H, "a_h")
    ct = self.lin_cts(a_h, slot, meta, ("qkv", "z"))                          # one GEMM when the cache fuses qkv | z
    qkv_rows = self.K.unpack(ct["qkv"][0], C, "qkv_rows", goff=ct["qkv"][1])                       # [R, 10240] raw (pre-conv)
    z_rows = self.K.unpack(ct["z"][0], R.NV * R.DV, "z_rows", goff=ct["z"][1])
    if n == 1:                # decode: everything after the projections in one kernel, on the persistent stacks
      if state is None:       # a fresh state (the gate test): zeroed one-layer stacks at position 0, the small weights held
        Sall = self.K.buf("S_scratch", R.NV * R.DK * R.DV, dtypes.float32); Sall.assign(Tensor.zeros(R.NV * R.DK * R.DV, device=DEV)).realize()
        Call = self.K.buf("C_scratch", R.CONV * CP, dtypes.float32); Call.assign(Tensor.zeros(R.CONV * CP, device=DEV)).realize()
        idx = self.K.buf("idx_zero", 1, dtypes.int32); posb = self.K.buf("pos_zero", 1, dtypes.int32)
        Wab, Nw = (self.K.hold(t, sm[k_]) for t, k_ in (("w_ab", "linear_attn_in_proj_ab_weight"), ("nw", "linear_attn_norm_weight")))
        Cwt = self.K.hold("cw_t", sm["linear_attn_conv1d_weight_t"].pad(((0, 0), (0, CP - C))))               # the taps' rows at pitch CP
      else: Sall, Call, idx, posb, Wab, Cwt, Nw = state[:7]; idx = self.K.hold("idx_held", idx)   # idx is a JIT input: held (the kernels' rule)
      ab = self.K.gemv(x, Wab, R.H, 2 * R.NV, "ab", norm=True, idx=idx)[:1]
    else: ab = self.K.gemv(x, self.K.hold("w_ab", sm["linear_attn_in_proj_ab_weight"]), R.H, 2 * R.NV, "ab", norm=True)[:n]
    a, b = ab[:, :R.NV], ab[:, R.NV:]                                                                                      # [n, 48]
    beta = b.sigmoid(); g = -sm["linear_attn_A_log"].exp() * (a + sm["linear_attn_dt_bias"]).softplus(); decay = g.exp()       # [n, 48]
    if n == 1:
      bd = self.K.buf("bd", 2 * R.NV, dtypes.float32).assign(Tensor.cat(beta[0], decay[0])).realize()
      tok = self.K.gdn_tok if os.environ.get("QWEN_GDN_TOK") == "1" else self.K.gdn_tok2        # gdn_tok2: the state streamed through LSRAM
      o = tok(Sall, Call, idx, posb, qkv_rows, z_rows, bd, Cwt, Nw, R.NV, R.NK, R.DK, R.DV, C, R.CONV)
      x = self.K.resid(x, self.lin_ct(self.K.rms_a(o, None, R.NV * R.DV, "a_o", norm=False), slot, meta, "o"), R.H, "x1")
      return self.mlp(x, slot, meta, sm["post_attention_layernorm_weight"], out), (Sall, Call)
    qkv = qkv_rows; z = z_rows[:n].reshape(n, R.NV, R.DV)
    cw = sm["linear_attn_conv1d_weight"].reshape(-1, R.CONV)                              # [10240, 4]
    prev = Tensor.zeros(R.CONV - 1, qkv.shape[1], device=DEV) if conv_state is None else conv_state
    ext = Tensor.cat(prev, qkv[:n], dim=0)                                                # [3 + n, 10240]
    conv = sum(ext[j:j + n] * cw[:, j].reshape(1, -1) for j in range(R.CONV))
    qkv = conv.silu(); new_conv_state = ext[-(R.CONV - 1):].contiguous()
    l2 = lambda t: t * (t.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
    q = l2(qkv[:, :R.NK * R.DK].reshape(n, R.NK, R.DK)) * (1.0 / math.sqrt(R.DK)); k = l2(qkv[:, R.NK * R.DK:2 * R.NK * R.DK].reshape(n, R.NK, R.DK))
    v = qkv[:, 2 * R.NK * R.DK:].reshape(n, R.NV, R.DV)
    rep = R.NV // R.NK
    q = q.reshape(n, R.NK, 1, R.DK).expand(n, R.NK, rep, R.DK).reshape(n, R.NV, R.DK); k = k.reshape(n, R.NK, 1, R.DK).expand(n, R.NK, rep, R.DK).reshape(n, R.NV, R.DK)
    decay, q, k, v, beta = decay.realize(), q.realize(), k.realize(), v.realize(), beta.realize()
    Sw = self.K.buf("S_work", R.NV * R.DK * R.DV, dtypes.float32)                        # prefill: the recurrence over the n tokens in one kernel
    Sw.assign(Tensor.zeros(R.NV * R.DK * R.DV, device=DEV) if state is None else state.reshape(-1)).realize()
    idx0 = self.K.buf("idx_zero", 1, dtypes.int32)                                  # S_work is a one-layer stack
    o = self.K.gdn_lsr(Sw, idx0, self.K.hold("q_n", q), self.K.hold("k_n", k), self.K.hold("v_n", v), self.K.hold("beta_n", beta), self.K.hold("decay_n", decay), R.NV, R.DK, R.DV, n)
    S = Sw.reshape(R.NV, R.DK, R.DV)
    o = o * (o.square().mean(-1, keepdim=True) + R.EPS).rsqrt() * sm["linear_attn_norm_weight"] * z.silu()
    o = self.K.rows_buf("o_rows", R.NV * R.DV).assign(o.reshape(n, R.NV * R.DV).pad(((0, Rr - n), (0, 0)))).realize()
    x = self.K.resid(x, self.lin_ct(self.K.rms_a(o, None, R.NV * R.DV, "a_o", norm=False), slot, meta, "o"), R.H, "x1")
    return self.mlp(x, slot, meta, sm["post_attention_layernorm_weight"]), (S, new_conv_state)

  def layer(self, x, l, **kw):
    if R.LAYER_TYPES[l] == "full": return self.attention_layer(x, l), None
    return self.gdn_layer(x, l, **kw)

  def mlp(self, x, slot, meta, w1_post, out="x2"):
    """x + down(silu(gate(h)) * up(h)), h = rmsnorm(x) * w1_post: three hand-written kernels around the two GEMMs; into the persistent `out`.
    `w1_post`: the 1 + w vector, or (stack, idx) -- read on the device (no per-layer copy)."""
    ct_gu = self.lin_ct(self.K.rms_a(x, w1_post if isinstance(w1_post, tuple) else self.K.hold("w_post", w1_post), R.H, "a_post"), slot, meta, "gu")
    return self.K.resid(x, self.lin_ct(self.K.swiglu_a(ct_gu, R.INTER, "a_sw"), slot, meta, "dn"), R.H, out)

def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--gate", type=int, default=3, help="layer to run on a random input vs the numpy reference")
  ap.add_argument("--n", type=int, default=12); ap.add_argument("--dir", default=os.environ.get("QWEN_DIR", "/mnt/ssd/qwen3.8-27b-fp8"))
  ap.add_argument("--cache", default=os.environ.get("QWEN_NPU", "/mnt/ssd/qwen3.8-27b-npu-s1"))
  ap.add_argument("--x", help="the input rows: a float32 [m, H] file (e.g. a reference's l_out-<gate-1> dump; its first n rows) instead of random ones")
  a = ap.parse_args()
  l, n = a.gate, a.n
  rng = np.random.default_rng(0); x = rng.standard_normal((n, R.H)).astype(np.float32) if a.x is None else np.fromfile(a.x, np.float32).reshape(-1, R.H)[:n].copy()
  W = R.Weights(a.dir); t0 = time.perf_counter(); want, _ = R.layer(W, l, x); print(f"numpy reference ({R.LAYER_TYPES[l]}): {time.perf_counter() - t0:.1f} s", flush=True)
  L = Layers(n, a.cache); xin = np.zeros((L.R, R.H), np.float32); xin[:n] = x
  for rep in range(2):
    t0 = time.perf_counter(); y, _ = L.layer(dev(xin), l); dt = time.perf_counter() - t0
    got = OA.host_invalidate(y).numpy()[:n]
    d = got - want; dd = want - x
    print(f"layer {l} on the NPU ({L.R} rows): {dt:.2f} s | max|d| {np.abs(d).max():.3g} (max|y| {np.abs(want).max():.3g}) | rel err of the layer's delta {np.linalg.norm(d)/np.linalg.norm(dd):.3g} | cosine {float((got*want).sum()/np.linalg.norm(got)/np.linalg.norm(want)):.7f} | finite {bool(np.isfinite(got).all())}", flush=True)

if __name__ == "__main__": main()

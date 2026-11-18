"""The NPU + CPU column split of a verify geometry's ternary GEMMs: the host side (cpu_split.c's orchestrator and pool, the CPU
half's weights from cpu_split_pack.py's files). `CpuSplit(L, share)` on a Layers `L` (<= 4 rows, rows mode) makes L.lin_ct run each
split linear on its first column groups only, bracketed by the chain's hand-off kernels (qwen38_npu.Layers.lin_ct: cpu_req after
the A, cpu_ack before the C's consumer); the CPU computes the other groups into the same C tiles. `arm()` before a pass over the
layers, `wait()` after it. The schedule is each layer's split linears in the chain's order: the fused input projection (qkvz on
a DeltaNet layer, qkv = q|kv on an attention layer), o, gate|up, down.
  QWEN_CPU_SPLIT=<share of each split linear's column groups on the CPU, e.g. 0.31; or per linear: gu=0.31,dn=0.33,o=0.3,qkvz=0.3,qkv=0.3
                  (the linears named are the ones split)>
  head=<share> (in the per-linear form): the ternary head's parts too, in the verify head's job and in each draft's (the head's
  CPU files: cpu_split_pack.py --head)
  QWEN_CPU_SPLIT_HEAD_NOCOMP=1: the head's CPU steps acknowledged at once (no compute; the head's outputs wrong): the NPU's cost alone
  QWEN_CPU_SPLIT_LIN=gu,dn (default) | all (= qkvz,qkv,o,gu,dn): the linears a bare share applies to
  QWEN_CPU_SPLIT_CPUS (default: the big cores the host is not on -- pool_cpus; 1, 6-11 on the O6N; never the A520s)
  QWEN_CPU_SPLIT_SNAP=1 (default): the NPU's group count snapped to a multiple of 12 -- the GEMM runs 12 tasks of ceil(groups / 12)
  groups each, so its time moves in steps of 12 groups; any other count leaves NPU time idle that the CPU's columns pay for
  QWEN_CPU_SPLIT_SHADOW=1|2|3: controls -- the NPU computes every column (as without the split), the outputs are the plain ones:
    1  the CPU does its split work into a scratch buffer, the hand-off kernels in the chain as with the split
    2  the hand-off kernels in the chain, the CPU acknowledges at once (no copy, no compute): their own cost
    3  the CPU does its split work into scratch each pass, free-running, no hand-off kernels: the DDR contention alone
    4  nothing but the split's buffer placement (its A's in DDR, not the GEMM's GM staging): no hand-off kernels, no CPU work
  QWEN_CPU_SPLIT_AGM=1 (default): the GEMMs' A stay in GM staging (the producers' place: off the DDR port) and the hand-off kernel
  copies each to a DDR mirror the CPU reads (qwen38_kernels.cpu_req_copy); 0: the A's themselves kept in DDR (12.8 ms a pass dearer)
  The CPU's kernel (cpu_split.c): the int8 matrix-multiply kernel (USMMLA, A in two int8 limbs: 1.3e-5 relative rms against exact
  sums) as hand-allocated assembly (cpu_split_asm.py) on the column-interleaved 2-bit weight files (.c2c and .sct: cpu_split_pack.py,
  and cpu_split_pack.py --head for the head's parts). QWEN_CPU_SPLIT_KERNEL: cs_set_kernel's bits, of which only bit 5 is still read
  (the narrow linears, o and down, K-split in core pairs: another order of sums, gated); default 7461, bit 5 set (the name of the
  2026-11-15 kernel: the other bits chose the variants since removed -- the lookup-table kernel, the .c2 int8 kernels).
  QWEN_CPU_SPLIT_EVEN=1: the columns shared evenly among the pool (default: by each core's top clock -- 2.2-2.6 GHz)
  The deferred steps (always, in a chain verify): a DeltaNet layer's qkvz step in two (when the gdn kernel can leave the gated norm
  to after it: value heads a multiple of the 12 tasks -- the 27B's 48, not the 2B's 16) -- the CPU's qkv columns ACKed before the
  recurrence, its z columns (no REQ: the same A) computed while the NPU runs it and ACKed before the gated output norm, which the
  gdn kernel then leaves to a kernel after it (gdn_fast_src(raw), gdn_onormd_src) -- so the qkvz share can grow (0.70); and on a
  gate-tail cache (the q|kv streams with the gates at the tail: cpu_split_pack.py --gtail) the same for an attention layer's qkv
  step -- its gate columns computed while the NPU runs attn_part and ACKed before attn_comb, which reads them from the C tiles
  The hand-off kernels: the REQ is raised by the A's producer itself (had_fast v2 "req": it also writes the A to the DDR mirror;
  else cpu_req_copy), the ACK is a one-task kernel (cpu_ack)
  QWEN_CPU_SPLIT_NOCOMP=1: every CPU step acknowledged at once (timing only, outputs wrong): the NPU's own path under the split
  bonsai2_generate.py's default on the 27B (when the cache has its cpu_split files): the fastest setting measured (the 10-prompt
  gate: 233.1 ms a pass, 11.42 tok/s, 10 / 10 identical) -- seven workers with the host on
  the NPU's interrupt CPU alone, qkvz=0.70,qkv=0.65,gu=0.50,o=0.44,dn=0.44,head=0.50 (qkv 0.50 on a cache without the gate tail),
  QWEN_CPU_SPLIT_MTP=0.55, the chain verify. QWEN_CPU_SPLIT=0: off (NPU only)
  QWEN_CPU_SPLIT_HEAD_EARLY=1 (default 0): the head's parts 1.. as steps without a REQ (part 0's A): computed back to back
  QWEN_CPU_SPLIT_MTP on an fp8 drafter cache (cpu_split_pack.py --mtp8: .f8c files): cpu_split.c fmt 2, the E4M3 codes as they are
  (fp16 products; 4.5e-4 relative rms -- the drafts only). peek(): a device buffer's words read through the host mapping
  QWEN_CPU_GLUE=1 (default 0; qwen38_generate, the chain verify in layer blocks, 512-block models): inside a block a layer's closing
  residual and the next layer's input norm + rotation on the CPU (set_glue; cpu_split.c GLUE_RESID_NORM: the NPU kernels' operations
  in their order -- the same bits), the NPU handing off with one one-task kernel (qwen38_kernels.cpu_sync). The device buffers are
  mapped uncached for the host, so the glue copies what it reads in whole lines, a share a worker. +0.6 % on the 2B
  QWEN_CPU_SPLIT_STATS=1: at exit, the orchestrator's per-pass medians (waiting for REQ, the A copies, the compute; by linear) on stderr"""
import atexit, ctypes, os, subprocess, sys
import numpy as np
from tinygrad import dtypes
import qwen38_ref as R

NS = 3; HERE = os.path.dirname(os.path.abspath(__file__)); NKIND = 16
KINDS = ("qkvz", "qkv", "o", "gu", "dn", "head", "mtp", "z", "g", "glue")   # the stats' buckets (head: the ternary head's parts; mtp: the drafter's linears; z: qkvz's z columns under QWEN_CPU_SPLIT_ZDEFER; g: qkv's gate columns under QWEN_CPU_SPLIT_GDEFER)
MTP_LIN = ("fc", "q", "kv", "o", "gu", "dn")      # the drafter's MTP layer's linears in its chain's order (qwen38_generate.mtp_step)
A_TAG = {"qkvz": "a_h", "qkv": "a_h", "o": "a_o", "gu": "a_post", "dn": "a_sw"}   # each linear's A buffer (qwen38_generate.vgdn_step / vatt_step, Layers.mlp)
class Step(ctypes.Structure):
  _fields_ = [("codes4", ctypes.c_void_p), ("sct", ctypes.c_void_p), ("a", ctypes.c_void_p), ("ct", ctypes.c_void_p),
              ("K", ctypes.c_int32), ("N", ctypes.c_int32), ("n0", ctypes.c_int32), ("gstride", ctypes.c_int32), ("kind", ctypes.c_int32), ("pad_", ctypes.c_int32), ("codes2", ctypes.c_void_p),
              ("fmt", ctypes.c_int32), ("rows", ctypes.c_int32), ("n1", ctypes.c_int32), ("flags", ctypes.c_int32)]

class Glue(ctypes.Structure):   # cpu_split.c's glue_t
  _fields_ = [("kind", ctypes.c_int32), ("rows", ctypes.c_int32), ("c", ctypes.c_int32), ("hb", ctypes.c_int32), ("n0", ctypes.c_int32), ("gstride", ctypes.c_int32),
              ("post", ctypes.c_float), ("eps", ctypes.c_float), ("ct", ctypes.c_void_p), ("xin", ctypes.c_void_p), ("xout", ctypes.c_void_p), ("w", ctypes.c_void_p), ("mirror", ctypes.c_void_p)]

_lib = None
def lib(cpus):
  global _lib
  if _lib is None:
    so, src = os.path.join(HERE, "libcpu_split.so"), os.path.join(HERE, "cpu_split.c")
    if not os.path.exists(so) or os.path.getmtime(so) < max(os.path.getmtime(src), os.path.getmtime(os.path.join(HERE, "cpu_split_asm.h"))):
      subprocess.check_call(["gcc", "-O3", "-ffp-contract=off", "-march=armv8.2-a+fp16+i8mm", "-shared", "-fPIC", "-pthread", "-o", so, src])   # (no fused multiply-add but the written ones: the glue's bits)
    _lib = ctypes.CDLL(so); _lib.cs_wait.restype = ctypes.c_int; _lib.cs_init(len(cpus), (ctypes.c_int * len(cpus))(*cpus))
    _lib.cs_set_kernel(int(os.environ.get("QWEN_CPU_SPLIT_KERNEL", "7461")))
    if os.environ.get("QWEN_CPU_SPLIT_EVEN", "0") == "1": _lib.cs_set_weights(None)
  return _lib

def peek(t, n, dt=np.int32):
  """The first n elements of the persistent device tensor `t` (a buffer of its own, as poke's), read through its host mapping after
  the job in flight has completed and the CPU's cached lines of those bytes are dropped (cs_inval: dc civac from user space, as the
  hand-off words are read) -- OA.host_invalidate(t).numpy() without the tensor's realize, the driver's block-wide invalidate and
  the copy's schedule (~0.3 ms a read on the 2B's decode pass). None before the library is loaded."""
  if _lib is None: return None
  from qwen38_npu import sync
  sync(); b = t.uop.base.buffer; raw = b._buf; nb = n * np.dtype(dt).itemsize; assert raw.va and nb <= b.nbytes, (nb, b.nbytes)
  _lib.cs_inval(ctypes.c_void_p(raw.va), ctypes.c_size_t(nb))
  return np.frombuffer(ctypes.string_at(raw.va, nb), dt).copy()
def c2c(path):   # a .c2c file (cpu_split_pack.py writes them)
  assert os.path.exists(path), f"{path}: missing -- python3 cpu_split_pack.py --kinds gu,dn,o,fused (--head for the head's parts)"
  return np.memmap(path, np.uint8, "r")
def c2_view(c, nsl, npad, n0):   # the CPU's columns of a .c2c file (prefault): [nsl][npad / 8 octets][256]
  return c.reshape(nsl, npad // 8, 256)[:, n0 // 8:, :]
_W = {}   # (cache, l, k) -> (codes memmap, sct memmap): shared by every geometry
def weights(cache, l, k):
  """A layer linear's CPU-side weights: the .c2c codes and the .sct scales (cpu_split_pack.py)."""
  if (cache, l, k) not in _W:
    d = os.path.join(cache, "cpu_split")
    _W[cache, l, k] = (c2c(os.path.join(d, f"L{l}_{k}.c2c")), np.memmap(os.path.join(d, f"L{l}_{k}.sct"), np.float32, "r"))
  return _W[cache, l, k]

def parse_share(s):
  """QWEN_CPU_SPLIT's value -> {linear: share}."""
  if "=" in s: return {k: float(v) for k, v in (kv.split("=") for kv in s.split(","))}
  lin = os.environ.get("QWEN_CPU_SPLIT_LIN", "gu,dn"); lin = KINDS if lin == "all" else lin.split(",")
  return {k: float(s) for k in lin}

def pool_cpus():
  """QWEN_CPU_SPLIT_CPUS (default: every big core the host is not pinned to). The little cores (the lowest cpu_capacity: the A520s,
  CPUs 2-5 on the Sky1) are never used -- they keep the system responsive under the pool's load; the host's own CPUs (pin_cpus:
  the NPU's interrupt CPU) stay free for it. On the O6N: 1, 6-11."""
  if (v := os.environ.get("QWEN_CPU_SPLIT_CPUS")): return [int(c) for c in v.split(",")]
  n = os.cpu_count() or 1
  def cap(c):
    try: return int(open(f"/sys/devices/system/cpu/cpu{c}/cpu_capacity").read())
    except (OSError, ValueError): return 1024
  caps = {c: cap(c) for c in range(n)}; lo = min(caps.values())
  host = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else set()
  big = [c for c in range(n) if caps[c] > lo] or list(range(n))   # (no little cores: every core)
  cpus = [c for c in big if c not in host]
  assert cpus, f"cpu_split: no big core left for the pool (the host is on {sorted(host)})"
  return cpus

class CpuSplit:
  def __init__(self, L, share, cpus=None, layers=None, prefault=True, chain=True):
    """`chain`: the generator verifies a chain (not a tree): the deferred z / gate steps need its kernels (vgdn_step / vatt_step)."""
    assert L.K.compact and L.K.rt == 1 and not L.fp8, "the CPU split: a ternary rows-mode geometry of <= 4 rows"
    cpus = cpus or pool_cpus()
    shares = share if isinstance(share, dict) else parse_share(str(share)); snap = os.environ.get("QWEN_CPU_SPLIT_SNAP", "1") != "0"
    assert set(shares) <= set(KINDS), ("QWEN_CPU_SPLIT: linears", set(shares) - set(KINDS))
    shares = dict(shares); self.head_share = shares.pop("head", None)
    self.agm = os.environ.get("QWEN_CPU_SPLIT_AGM", "1") == "1"
    self.shadow = int(os.environ.get("QWEN_CPU_SPLIT_SHADOW", "0") or 0); assert self.shadow in (0, 1, 2, 3, 4), self.shadow
    self.L, self.lib, self.share = L, lib(cpus), shares; K = L.K; layers = list(range(R.NL)) if layers is None else list(layers); self.layers = layers
    # per linear: its shape (the same on every layer that has it), the NPU's groups
    self.gn, self.npad, shp, order = {}, {}, {}, []
    for l in layers:
      meta = L._meta(l)[0]; ks = [k for k in ("qkvz", "qkv", "o", "gu", "dn") if k in meta and k in shares]
      if {"qkvz", "qkv"} & set(shares): assert "qkvz" in meta or "qkv" in meta, f"cpu_split: layer {l}'s input projection is not fused (QWEN_FUSE)"
      order.append((l, ks))
      for k in ks:
        N, npad, Kk = meta[k]; ng = npad // (16 * NS)
        if k in shp: assert shp[k] == (Kk, ng) and self.npad[k] == npad, (l, k, meta[k]); continue
        gn = ng - int(round(shares[k] * ng))
        if snap: gn = 12 * int(round(gn / 12))
        self.gn[k] = max(1, min(ng - 1, gn)); self.npad[k] = npad; shp[k] = (Kk, ng)
    # the buffers both sides touch, in DDR and allocated now (their addresses go into the schedule): the A's, the C's, the words
    va = lambda t: t.uop.buffer._buf.va
    if self.agm:      # the CPU reads a DDR mirror of each A, written by the hand-off kernel
      self.mirror = {k: K.buf(f"cpu_mirror_{k}", K.a_size(shp[k][0]), dtypes.uint16) for k in shp}; self.a = self.mirror
    else:
      tags = {A_TAG[k] for k in shp}
      assert not (tags & set(K.bufs)), f"cpu_split: set up before the geometry's first pass (its A buffers {tags & set(K.bufs)} are already allocated, in GM)"
      K.no_gm |= tags
      self.a = {k: K.buf(A_TAG[k], K.a_size(shp[k][0]), dtypes.uint16) for k in shp}
    self.ct = {k: K.buf(f"ct_{k}", shp[k][1] * L.nrb * NS * 192, dtypes.float32) for k in shp}
    if self.shadow:   # the CPU's columns into host scratch with the C tiles' layout (64-byte aligned), not into the NPU's C
      self._scratch = {k: np.zeros(shp[k][1] * L.nrb * NS * 192 + 16, np.float32) for k in shp}
      cva = {k: (self._scratch[k].ctypes.data + 63) // 64 * 64 for k in shp}
    else: cva = {k: va(self.ct[k]) for k in shp}
    from qwen38_kernels import NT as KNT
    self.req, self.ack = K.buf("cpu_req_w", 16 * KNT, dtypes.int32), K.buf("cpu_ack_w", 16, dtypes.int32)
    self.lib.cs_set_req_lines(KNT if self.agm else 1)
    # The deferred steps (chain verify): a DeltaNet layer's qkvz step in two -- the CPU's qkv columns (ACKed before the recurrence)
    # and its z columns (no REQ: the same A; ACKed before gdn_onormd, the gated norm the gdn kernel then leaves to after it:
    # gdn_fast_src(raw)), so the z columns are computed while the NPU runs the recurrence. On a gate-tail cache (the attention
    # layers' gates at the fused tail: cpu_split_pack.py --gtail) an attention layer's qkv step in two the same way, its gate
    # columns ACKed between attn_part and attn_comb (qwen38_generate.vatt_step)
    gtail = any(isinstance(L._meta(l)[0].get("_sub", {}).get("q", (None, 0))[1], tuple) for l, ks in order if "qkv" in ks)
    # the z step: where gdn_tokm reads z after the recurrence -- the raw rows (the 27B; the 2B at <= 4 rows) or the 2B's split plan
    # (> 4 rows). QWEN_CPU_SPLIT_ZDEFER=0 / 1 overrides (default: on for NV % 12 == 0, the 27B). On the 2B it is lossless but slower
    # at every qkvz share (2026-11-17: -1.2 to -2.6 %; the raw rows' gdn_onormd launch a layer costs more than its z columns free)
    _zd = os.environ.get("QWEN_CPU_SPLIT_ZDEFER", "1" if R.NV % KNT == 0 else "0") == "1"
    self.zdefer = chain and "qkvz" in self.gn and R.HAD and _zd
    self.gdefer = chain and "qkv" in self.gn and gtail and L.K.attn_gt_ok()
    dc = {}                                                     # (layer, linear) -> (its first deferred column, the deferred step's kind)
    for l, ks in order:
      sub = L._meta(l)[0].get("_sub", {})
      if self.zdefer and "qkvz" in ks: dc[l, "qkvz"] = (16 * NS * sub["z"][1], "z")
      if self.gdefer and "qkv" in ks:
        assert isinstance(sub["q"][1], tuple), f"cpu_split: layer {l}'s q|kv is not gate-tail but another layer's is (a mixed cache)"
        dc[l, "qkv"] = (16 * NS * sub["q"][1][1], "g")
    self.steps = (Step * (sum(len(ks) for _, ks in order) + len(dc)))(); self._keep = []; self.first = []; i = 0; self._sk = []; self._cva = cva
    for l, ks in order:
      self.first.append(i)
      for k in ks:
        self._sk.append((l, k))
        if (l, k) in dc: self._sk.append((l, dc[l, k][1]))
        c2, sct = weights(L.cache, l, k); self._keep += [c2, sct]; n0 = self.gn[k] * 16 * NS
        self.steps[i] = Step(None, sct.ctypes.data, va(self.a[k]), cva[k], shp[k][0], self.npad[k], n0, L.nrb * NS * 192, KINDS.index(k), 0, c2.ctypes.data, 0, L.K.real); i += 1
        if (l, k) in dc:   # the step's columns end at the deferred ones' first; the deferred step after it (flags 1: the same A)
          c0, dk = dc[l, k]; assert c0 % 16 == 0 and c0 < self.npad[k], (l, k, c0); self.steps[i - 1].n1 = max(n0, c0)
          self.steps[i] = Step(None, sct.ctypes.data, va(self.a[k]), cva[k], shp[k][0], self.npad[k], max(n0, c0), L.nrb * NS * 192, KINDS.index(dk), 0, c2.ctypes.data, 0, L.K.real, 0, 1); i += 1
        if prefault:   # the CPU's columns' pages in (page cache), so no pass waits on the disk
          nsl = shp[k][0] // 128; c2_view(c2, nsl, self.npad[k], n0).max(); np.asarray(sct).reshape(nsl, self.npad[k])[:, n0:].max()
    self.first.append(i); assert i == len(self.steps), (i, len(self.steps))
    L.zdefer_ack = L.K.cpu_ack if self.zdefer and self.shadow not in (3, 4) else None   # (qwen38_generate.vgdn_step: the z step's ACK before gdn_onorm)
    L.gdefer_ack = L.K.cpu_ack if self.gdefer and self.shadow not in (3, 4) else None   # (vatt_step: the gate step's ACK before attn_comb)
    # the ternary head's parts (head=<share>): steps after the layers', served by arm_head (the verify head's job, a draft's job);
    # qwen38_generate._head_part runs part i's first head_gn[i] groups, bracketed by the same hand-off kernels
    self.head_gn, self.head_steps = [], None; self.head_early = os.environ.get("QWEN_CPU_SPLIT_HEAD_EARLY", "0") == "1"
    if self.head_share and not self.shadow:
      z = np.load(os.path.join(L.cache, "lm_head_tern.npz")); Kh = R.H; self.mirror_head = K.buf("cpu_mirror_head", K.a_size(Kh), dtypes.uint16)
      parts = [int(g1 - g0) for g0, g1 in z["parts"]]; self.head_steps = (Step * len(parts))()
      for j, ng in enumerate(parts):
        gn = ng - int(round(self.head_share * ng)); gn = 12 * int(round(gn / 12)) if snap else gn; gn = max(1, min(ng - 1, gn)); self.head_gn.append(gn)
        d = os.path.join(L.cache, "cpu_split"); npad = ng * 16 * NS; n0 = gn * 16 * NS
        c2, sct = c2c(os.path.join(d, f"head_{j}.c2c")), np.memmap(os.path.join(d, f"head_{j}.sct"), np.float32, "r"); self._keep += [c2, sct]
        ct = K.buf(f"ct_head{j}", ng * L.nrb * NS * 192, dtypes.float32)   # (qwen38_generate._head_part's C tiles)
        # QWEN_CPU_SPLIT_HEAD_EARLY=1 (default 0): parts 1.. are steps without a REQ (flags 1: part 0's A, its copy and limbs) -- the
        # CPU computes the parts back to back instead of waiting for the NPU's chain between them (qwen38_generate._head_part
        # raises no REQ for them)
        self.head_steps[j] = Step(None, sct.ctypes.data, va(self.mirror_head), va(ct), Kh, npad, n0, L.nrb * NS * 192, KINDS.index("head"), 0, c2.ctypes.data, 0, K.real, 0, 1 if self.head_early and j else 0)
        if prefault: nsl = Kh // 128; c2_view(c2, nsl, npad, n0).max(); np.asarray(sct).reshape(nsl, npad)[:, n0:].max()
    self.acc_head, self.acc_draft, self.pre, self._draft_steps, self._ph_steps = [], [], [], {}, {}
    self._req_p, self._ack_p = ctypes.c_void_p(va(self.req)), ctypes.c_void_p(va(self.ack))
    if self.shadow in (3, 4): self._req_p = self._ack_p = None     # the schedule runs at once, no handshake
    self.handoff = self.shadow not in (3, 4); self.lib.cs_set_mode(1 if self.shadow in (2, 4) or os.environ.get("QWEN_CPU_SPLIT_NOCOMP") == "1" else 2 if os.environ.get("QWEN_CPU_SPLIT_HEAD_NOCOMP") == "1" else 0)   # NOCOMP: the CPU acknowledges every step at once (timing only: the NPU's own path; outputs wrong)
    L.cpu_split = self; self.acc = []
    if os.environ.get("QWEN_CPU_SPLIT_STATS"): atexit.register(self._report)
  def _report(self):
    if self.acc_draft:
      m = {k: float(np.median([a[k] for a in self.acc_draft])) for k in ("wait_ms", "copy_ms", "compute_ms")}
      by = {k: tuple(float(np.median([a["by"][k][j] for a in self.acc_draft])) for j in range(2)) for k in ("mtp", "head")}
      print(f"cpu_split draft jobs ({self.L.n} rows): {len(self.acc_draft)}, per job (median) wait {m['wait_ms']:.2f} ms, A copies {m['copy_ms']:.2f} ms, "
            f"compute {m['compute_ms']:.2f} ms; " + ", ".join(f"{k} wait {w:.2f} compute {c:.2f} ms" for k, (w, c) in by.items()), file=sys.stderr, flush=True)
    if self.acc_head:
      m = {k: float(np.median([a[k] for a in self.acc_head])) for k in ("wait_ms", "copy_ms", "compute_ms")}
      print(f"cpu_split head share {self.head_share} NPU groups {self.head_gn}: {len(self.acc_head)} head jobs, per job (median) wait {m['wait_ms']:.2f} ms, "
            f"A copies {m['copy_ms']:.2f} ms, compute {m['compute_ms']:.2f} ms", file=sys.stderr, flush=True)
    if not self.acc: return
    if getattr(self, "glue_on", False):
      g = (ctypes.c_double * 6)(); self.lib.cs_glue_stats(g)
      print(f"cpu_split glue ({self.L.n} rows): {int(g[5])} calls, us a call: invalidate {g[0] / 1e3:.1f}, copies in {g[1] / 1e3:.1f}, compute {g[2] / 1e3:.1f}, "
            f"rows out {g[3] / 1e3:.1f}, A out {g[4] / 1e3:.1f}", file=sys.stderr, flush=True)
    m = {k: float(np.median([a[k] for a in self.acc])) for k in self.acc[0] if k != "by"}
    by = {k: tuple(float(np.median([a["by"][k][j] for a in self.acc])) for j in range(2)) for k in self.acc[0]["by"]}
    print(f"cpu_split shares {self.share} NPU groups {self.gn}: {len(self.acc)} passes, per pass (median) wait {m['wait_ms']:.2f} ms, "
          f"A copies {m['copy_ms']:.2f} ms, compute {m['compute_ms']:.2f} ms (longest step {m['max_compute_ms']:.3f} ms), {m['steps']:.0f} steps; "
          + ", ".join(f"{k} wait {w:.2f} compute {c:.2f} ms" for k, (w, c) in by.items()), file=sys.stderr, flush=True)
  def set_glue(self, win, blk):
    """QWEN_CPU_GLUE: inside each block of `blk` layers, layer l's closing residual and layer l + 1's input norm + rotation on the
    CPU (cpu_split.c GLUE_RESID_NORM) -- after l's dn step (now without an ACK) a glue step (its REQ: the NPU's dn part is done;
    it writes the layer's output rows and the next layer's first linear's A, then ACKs), and l + 1's first step takes that A
    (no REQ, no copy). `win`: {layer: its input norm's (1 + weight) x signs [H], host}. Needs dn and the layers' first linears
    split, the hand-off kernels, no deferred steps, a 512-block model at <= 4 rows."""
    L, K = self.L, self.L.K; ok = (self.handoff and self.agm and not self.shadow and not self.zdefer and not self.gdefer and "dn" in self.gn and K.had_b == 512
                                   and K.had_post != 1.0 and L.K.real <= 4 and all(l in win for l in self.layers))
    if not ok: print("   QWEN_CPU_GLUE: not available for this split (needs dn and the first linears split, no deferred steps, 512 blocks): off", flush=True); return
    va = lambda t: t.uop.buffer._buf.va
    xin = va(K.buf("x1", K.rows * R.H, dtypes.float32)); xo = [va(K.buf(f"xiov{p}", K.rows * R.H, dtypes.float32)) for p in (0, 1)]
    old = [(self._sk[i], self.steps[i]) for i in range(len(self.steps))]; new = []; self._glues = []; self._win = {l: np.ascontiguousarray(w, np.float32) for l, w in win.items()}
    have = set(self.layers); take = set()                                 # take: the layers whose first step reads the glue's A
    for (l, k), st in old:
      nk = None
      if k == "dn" and l + 1 in have and (l + 1) % blk != 0:
        nk = next((kk for (ll, kk), _ in old if ll == l + 1), None); nk = nk if nk in ("qkvz", "qkv") else None
      c = Step.from_buffer_copy(st)
      if (l, k) == next(((ll, kk) for (ll, kk), _ in old if ll == l), None) and l in take: c.flags |= 2
      if nk is not None:
        c.flags |= 4; new.append(((l, k), c))
        g = Glue(0, K.real, R.H, K.had_b, self.gn["dn"] * 16 * NS, L.nrb * NS * 192, float(K.had_post), float(K.eps), self._cva["dn"], xin, xo[1 - l % 2],
                 self._win[l + 1].ctypes.data, va(self.mirror[nk])); self._glues.append(g)
        new.append(((l, "glue"), Step(ctypes.addressof(g), None, None, None, R.H, 0, 0, 0, KINDS.index("glue"), 0, None, 0, K.real, 0, 8))); take.add(l + 1)
      else: new.append(((l, k), c))
    self.steps = (Step * len(new))(); self._sk = [x for x, _ in new]; self.first = []
    for i, (lk, c) in enumerate(new):
      self.steps[i] = c
      if i == 0 or lk[0] != new[i - 1][0][0]: self.first.append(i)
    self.first.append(len(new)); self._ph_steps = {}; self.glue_on = True
    print(f"   QWEN_CPU_GLUE: {len(self._glues)} of {len(self.layers)} layers' closing residual + next input norm on the CPU ({K.real} rows)", flush=True)
  def arm(self, i0=0, n=None):
    """Serve the next pass: layers[i0 : i0 + n] in order (default all)."""
    n = len(self.layers) - i0 if n is None else n; s0, s1 = self.first[i0], self.first[i0 + n]
    self.lib.cs_arm(ctypes.byref(self.steps, s0 * ctypes.sizeof(Step)), s1 - s0, self._req_p, self._ack_p)
  def arm_pass_head(self, n):
    """Serve a whole verify pass and its head job's parts 0..n-1 as one schedule (QWEN_HEAD_OVERLAP: the head's job is submitted
    while the last block still runs, so the pool cannot be re-armed between them); one wait() after the head's job."""
    if n not in self._ph_steps:
      st = (Step * (len(self.steps) + n))()
      for i, x in enumerate(list(self.steps) + [self.head_steps[j] for j in range(n)]): st[i] = x
      self._ph_steps[n] = st
    self.lib.cs_arm(self._ph_steps[n], len(self.steps) + n, self._req_p, self._ack_p); self._armed = None
  def arm_head(self, n):
    """Serve the head's parts 0..n-1 (the verify head's job); False if the head is not split."""
    if not self.head_gn: return False
    self.lib.cs_arm(self.head_steps, n, self._req_p, self._ack_p); self._armed = self.acc_head; return True
  def arm_draft(self, n):
    """Serve a draft job: the MTP layer's split linears (MtpSplit's, `pre`), then the head's parts 0..n-1 (0: the job has no split
    head); False if the job has no split step."""
    n = n if self.head_gn else 0
    if not self.pre and not n: return False
    if n not in self._draft_steps:
      st = (Step * (len(self.pre) + n))()
      for i, x in enumerate(list(self.pre) + [self.head_steps[j] for j in range(n)]): st[i] = x
      self._draft_steps[n] = st
    self.lib.cs_arm(self._draft_steps[n], len(self.pre) + n, self._req_p, self._ack_p); self._armed = self.acc_draft; return True
  def wait(self):
    rc = self.lib.cs_wait(); acc = getattr(self, "_armed", None); acc = self.acc if acc is None else acc; self._armed = None
    if rc: raise RuntimeError(f"cpu_split: step {rc - 1}'s request never came ({'the head / draft job' if acc is not self.acc else 'the pass'}: the chain and the schedule disagree)")
    if os.environ.get("QWEN_CPU_SPLIT_STATS"): acc.append(self.stats())
  def stats(self):
    s = (ctypes.c_double * (5 + 2 * NKIND))(); self.lib.cs_stats(s)
    return dict(wait_ms=s[0] / 1e6, copy_ms=s[1] / 1e6, compute_ms=s[2] / 1e6, max_compute_ms=s[3] / 1e6, steps=int(s[4]),
                by={k: (s[5 + KINDS.index(k)] / 1e6, s[5 + NKIND + KINDS.index(k)] / 1e6) for k in list(self.gn) + ["head", "mtp"] + (["z"] if self.zdefer else []) + (["g"] if self.gdefer else []) + (["glue"] if getattr(self, "glue_on", False) else [])})


class MtpSplit:
  """The drafter's MTP layer split as the verify's linears (a q4t MTP cache, or an fp8 one with its .f8c files, on its own geometry, qwen38_npu.Layers(fp8=True)): each
  split linear's first column groups on the NPU (Layers.lin_ct's hook, which reads gn / npad / mirror / agm / shadow / handoff), the
  rest on the CPU with cpu_split.c's 4-bit kernel (fmt 1; the files from cpu_split_pack.py --mtp). The hand-off words are the head
  geometry's (hs: the verify geometry of the same rows, whose CpuSplit serves the whole draft job -- these steps, then the head's
  parts: CpuSplit.arm_draft).
    QWEN_CPU_SPLIT_MTP=<share> | fc=..,q=..,kv=..,o=..,gu=..,dn=.. (the linears named are the ones split)"""
  def __init__(self, Lm, meta, cache, layer, share, hs, prefault=True):
    shares = parse_share(share) if "=" in str(share) else {k: float(share) for k in MTP_LIN}
    assert set(shares) <= set(MTP_LIN), ("QWEN_CPU_SPLIT_MTP: linears", set(shares) - set(MTP_LIN))
    K = Lm.K; assert K.compact and K.rt == 1, "the MTP split: a rows-mode geometry of <= 4 rows"
    assert "cpu_req_w" not in K.bufs and "cpu_ack_w" not in K.bufs, "MtpSplit: set up before the geometry's first pass"
    K.bufs["cpu_req_w"], K.bufs["cpu_ack_w"] = hs.req, hs.ack          # one pair of words for the draft job (the head's geometry's)
    self.shadow, self.agm, self.handoff, self.share = 0, True, True, shares; snap = os.environ.get("QWEN_CPU_SPLIT_SNAP", "1") != "0"
    va = lambda t: t.uop.buffer._buf.va; d = os.path.join(cache, "cpu_split"); self.gn, self.npad, self.mirror, self._keep, steps = {}, {}, {}, [], []
    for k in MTP_LIN:
      if k not in shares: continue
      N, npad, Kk = meta[k]; ng = npad // (16 * NS); gn = ng - int(round(shares[k] * ng))
      if snap: gn = 12 * int(round(gn / 12))
      gn = max(1, min(ng - 1, gn)); self.gn[k], self.npad[k] = gn, npad; n0 = gn * 16 * NS
      self.mirror[k] = K.buf(f"cpu_mirror_{k}", K.a_size(Kk), dtypes.uint16); ct = K.buf(f"ct_{k}", ng * Lm.nrb * NS * 192, dtypes.float32)
      f8 = os.path.join(d, f"L{layer}_{k}.f8c"); fmt = 2 if os.path.exists(f8) else 1   # (an fp8 cache: cpu_split_pack.py --mtp8; a q4t one: --mtp)
      c4 = np.memmap(f8 if fmt == 2 else os.path.join(d, f"L{layer}_{k}.c4"), np.uint8, "r"); sct = np.memmap(os.path.join(d, f"L{layer}_{k}.sct"), np.float32, "r"); self._keep += [c4, sct]
      steps.append(Step(c4.ctypes.data, sct.ctypes.data, va(self.mirror[k]), va(ct), Kk, npad, n0, Lm.nrb * NS * 192, KINDS.index("mtp"), 0, None, fmt, K.real))
      if prefault:
        nsl = Kk // 128; np.asarray(sct).reshape(nsl, npad)[:, n0:].max()
        if fmt == 2: c4.reshape(nsl, npad, 128)[:, n0:, :].max()
        else: c4.reshape(nsl, npad // 2, 128)[:, n0 // 2:, :].max()
    hs.pre = steps; hs._draft_steps = {}; Lm.cpu_split = self

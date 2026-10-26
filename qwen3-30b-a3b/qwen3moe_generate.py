#!/usr/bin/env python3
"""Qwen3-30B-A3B text generation on the NPU (greedy), from qwen3moe_pack.py's cache: the 48 layers of qwen3moe_npu.py and the
output head (Q6_K, 4 parts, the top-1 on the device), the host embedding the token (token_embd's Q4_K row) and routing the experts.

Weights: every layer's streams in a KMD weight buffer of its own (zero-copy, ~21 GB pinned); two window slots, layer l mapped into
slot l % 2 before it runs (each layer syncs for its routing, so the slot's previous layer has run); the linears are fixed views
into the slot per (slot, layer kind) -- the 24 layers whose v and down are Q6_K are one kind, the others another.

    python3 qwen3moe_generate.py "prompt" [--max-new 64] [--thinking]
"""
import argparse, functools, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import qwen3moe_npu as QN                                                  # noqa: E402  (puts tinygrad and the other examples on the path)
import qwen3moe_kernels as MK                                              # noqa: E402
import qwen3moe_ref as QR                                                  # noqa: E402
from qwen3moe_npu import Kernels, Tensor, Device, OA, dtypes, dev, poke   # noqa: E402

TOK = os.environ.get("QWEN3MOE_TOK", "/mnt/ssd/models/qwen3-30b-a3b-tokenizer")
JIT = os.environ.get("QWEN3MOE_JIT", "1") != "0"
PROF = os.environ.get("QWEN3MOE_PROF", "0") == "1"
DEVTOK = os.environ.get("QWEN3MOE_DEVTOK", "1") == "1"       # the decode loop on the device: the head's token, its embedding row, the position
READ_EVERY = int(os.environ.get("QWEN3MOE_READ_EVERY", "8"))
SYNC_PROBE = os.environ.get("QWEN3MOE_SYNC_PROBE", "0") == "1"   # a host wait + 512 B read after every layer's job (what a CPU share would pay)
LPJ = int(os.environ.get("QWEN3MOE_LPJ", "1"))                # layers a job (a graph): 2 LPJ weight slots, mapped a job ahead (2: the 4 slots overflow the NPU window -- MemoryError)
HEAD_JOIN = os.environ.get("QWEN3MOE_HEAD_JOIN", "1") == "1"   # the head and embed_next in the last layer group's job
HEAD_Q6N = os.environ.get("QWEN3MOE_HEAD_Q6N", "1") == "1"    # the head over its decode-only 6.6-bit stream (head_q6n.bin, when packed)  # (DEVTOK: the host reads the token ring every this many tokens -- the stop check)
PREFILL = os.environ.get("QWEN3MOE_PREFILL", "batch")        # "batch": qwen3moe_prefill (chunks of 128 tokens), "step": the decode step a token        # per-phase wall times (a sync after each phase: slower, for the breakdown)          # each layer's two halves (attention + router, experts) and the head as TinyJit graphs

# ---- the host's CPU set and latency request (gemma4_generate's / qwen38_generate's pin_cpus / hold_cpu_latency): the NPU interrupt's
# CPU and one more of its kind, kept out of deep idle (a 0 us PM QoS request while the process runs) -- the gaps between jobs shrink
def pin_cpus():
  v = os.environ.get("QWEN3MOE_CPUS", os.environ.get("QWEN_CPUS", "auto"))
  if v == "off" or not hasattr(os, "sched_setaffinity"): return None
  if v != "auto": cpus = {int(c) for c in v.split(",")}
  else:
    try:
      line = next(l for l in open("/proc/interrupts") if l.rstrip().endswith(" aipu"))
      counts = [int(x) for x in line.split()[1:1 + os.cpu_count()]]; c0 = max(range(len(counts)), key=counts.__getitem__)
      fmax = lambda c: open(f"/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq").read().strip()
      cpus = {c0} | set([c for c in range(os.cpu_count()) if c != c0 and fmax(c) == fmax(c0)][:1])
    except (StopIteration, OSError, ValueError): return None
  os.sched_setaffinity(0, cpus); return sorted(cpus)
_LAT = []
def hold_cpu_latency():
  v = os.environ.get("QWEN3MOE_CPU_LATENCY", os.environ.get("QWEN_CPU_LATENCY", "0"))
  if v == "off": return None
  try: f = open("/dev/cpu_dma_latency", "wb", buffering=0); f.write(int(v).to_bytes(4, "little")); _LAT.append(f); return int(v)
  except OSError as e: print(f"   (no PM QoS request -- {e.strerror})", flush=True); return None

def one_job():
  """The capturing TinyJit call's kernels in one graph (JIT_BATCH_SIZE=0) and one chain (ZHOUYI_CHAIN_CAP): one job (gemma4's)."""
  from tinygrad.helpers import Context, ContextVar
  return Context(**{k: v for k, v in (("JIT_BATCH_SIZE", 0), ("ZHOUYI_CHAIN_CAP", 4096)) if k in ContextVar._cache})
CHUNK = 2 << 20


def read_into(path, mv, size, nt=8, drop=False):
  """The file's bytes into `mv` (a weight buffer's mapping) by `nt` parallel preadv's; `drop`: release its page cache afterwards
  (qwen3.8-27b's qwen38_npu.read_into)."""
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


class W:
  """A layer's linears as views into a slot, its meta and its small tensors (qwen3moe_npu.gemm's interface)."""
  def __init__(self, st, meta, sm, vq6): self.st, self.meta, self.sm, self.vq6 = st, meta, sm, vq6


class Model:
  def __init__(self, cache=QN.CACHE, gguf=QR.DEFAULT, tmax=1024, log=print):
    t0 = time.perf_counter(); self.M = QR.Model(gguf); c = self.c = self.M.c; self.tmax = tmax; self.cache = cache
    self.K = Kernels(2, 24, c.EPS, real=1); self.R = 24; self.B = QN.Bufs(self.K, c)
    self.metas, self.sms, self.kind = [], [], []
    for l in range(c.NL):
      meta, small = QN.read_meta(l, cache); self.metas.append(meta); self.kind.append("v" in meta)
      self.sms.append({k: dev(v.astype(np.float32)) for k, v in small.items()}); self.sms[-1]["router16"] = dev(small["router"].astype(np.float16))
    # zero-copy: a weight buffer a layer, its files 2 MiB-aligned (+16 B of load overhang slack each); the same offsets per kind
    raw = self.raw = Device[QN.DEV].raw; self.packs = {}
    for l in range(c.NL):
      sizes = {k: os.path.getsize(os.path.join(cache, f"L{l}_{k}.bin")) for k in sorted(self.metas[l])}; offs, o = {}, 0
      for k, sz in sizes.items(): offs[k] = o; o += -(-(sz + 16) // CHUNK) * CHUNK
      self.packs[l] = (sizes, offs, o)
    for l in range(c.NL):   # the views are per kind: every layer of a kind must pack the same streams at the same offsets
      assert self.packs[l][1] == self.packs[self.kind.index(self.kind[l])][1], (l, "a layer's streams differ from its kind's first layer's (an unfinished q6n pack?)")
    big = max(p[2] for p in self.packs.values()); self.NSLOT = 2 * LPJ; self.slots = [raw.slot_alloc(big) for _ in range(self.NSLOT)]
    self.wbufs = {}
    for l in range(c.NL):
      sizes, offs, n = self.packs[l]; wid, mm = raw.wbuf_alloc(n)
      for k, sz in sizes.items(): read_into(os.path.join(cache, f"L{l}_{k}.bin"), memoryview(mm)[offs[k]:offs[k] + sz], sz, drop=True)
      self.wbufs[l] = (wid, mm)
    self.views = {}
    for par in range(self.NSLOT):
      for kd in (False, True):
        l0 = self.kind.index(kd); sizes, offs, _ = self.packs[l0]
        self.views[(par, kd)] = {k: Tensor.from_blob(self.slots[par][1] + offs[k], (sz,), dtype=dtypes.uint8, device=QN.DEV) for k, sz in sizes.items()}
    log(f"weights: {c.NL} layers in weight buffers, {sum(p[2] for p in self.packs.values()) / 1e9:.1f} GB, two {big / 1e6:.0f} MB slots "
        f"({time.perf_counter() - t0:.0f} s)")
    z = np.load(os.path.join(cache, "head.npz")); self.head_parts = [tuple(int(v) for v in r) for r in z["parts"]]
    self.w_out = dev(z["w_out"]); hb = [np.fromfile(os.path.join(cache, f"head{j}.bin"), np.uint8) for j in range(len(self.head_parts))]
    if QN.QDOT:                    # the qdot head: the parts' streams as one (whole groups each), one launch, the top-1 kept per task
      p6 = os.path.join(cache, "head_q6n.bin"); self.head_kind = "q6n" if os.path.exists(p6) and HEAD_Q6N else "q6"   # the decode-only 6.6-bit stream when packed
      self.head_all = dev(np.fromfile(p6, np.uint8) if self.head_kind == "q6n" else np.concatenate(hb)); n_ = sum(p[1] for p in self.head_parts); self.head_parts = [(4, n_, -(-n_ // 48) * 48, c.H, 0)]
    else: self.head = [dev(b) for b in hb]
    self.Kc = [Tensor.zeros(tmax * c.NKV * c.HD, device=QN.DEV).contiguous().realize() for _ in range(c.NL)]
    self.Vc = [Tensor.zeros(tmax * c.NKV * c.HD, device=QN.DEV).contiguous().realize() for _ in range(c.NL)]
    self.cs = dev(QN.rope_cs(self.M, tmax)); self.posb = self.K.buf("posb", 16, dtypes.int32)
    self.xio = [self.K.rows_buf("xio0", c.H), self.K.rows_buf("xio1", c.H)]
    self.jits, self.jd = {}, Tensor.zeros(16, device=QN.DEV).contiguous().realize()
    if DEVTOK:                     # token_embd's Q4_K rows on the device (175 MB) and the ring of generated ids
      kind, b = self.M.blocks("token_embd.weight"); assert kind == "Q4_K"
      self.emb = dev(np.array(b, copy=True).reshape(-1)); self.ring = Tensor.zeros(tmax + 16, device=QN.DEV, dtype=dtypes.int32).contiguous().realize()
      self.d_emb = dev(MK.embed_next_desc(c.H))
    # device routing (QN.DEVROUTE): each layer's own slots and weights, so no buffer a kernel reads by its cache is written twice in
    # a job (the GEMM reads its slot with a scalar load)
    self.lb = []
    for l in range(c.NL):
      b = type("LB", (), {})(); b.__dict__.update(self.B.__dict__)
      b.sel_gu, b.sel_dn, b.wt = (Tensor.zeros(n_, device=QN.DEV, dtype=dt).contiguous().realize() for n_, dt in ((16, dtypes.int32), (16, dtypes.int32), (12 * c.TOPK, dtypes.float32)))
      self.lb.append(b)
    log(f"ready in {time.perf_counter() - t0:.0f} s")

  # ---- the device work, as functions of a dummy input (each a TinyJit graph under JIT: captured on its second call, replayed after)
  def f_attn(self, l, _):
    """Layer l's attention and its router: xio[l % 2] -> x1 (Kernels' buffer), the router's logits (returned)."""
    kd = self.kind[l]; Wl = W(self.views[(l % self.NSLOT, kd)], self.metas[l], self.sms[l], kd)
    x1 = QN.attn_body(self.K, self.B, Wl, self.c, 1, self.xio[l % 2], self.Kc[l], self.Vc[l], self.cs, self.posb)
    return QN.router(self.K, Wl, self.c, x1)
  def f_moe(self, l, _):
    """Layer l's experts on the slots and weights poked: x1 -> xio[(l + 1) % 2]."""
    kd = self.kind[l]; Wl = W(self.views[(l % self.NSLOT, kd)], self.metas[l], self.sms[l], kd)
    return QN.moe_body(self.K, self.B, Wl, self.c, 1, self.K.bufs["x1"].reshape(self.R, self.c.H), self.xio[(l + 1) % 2])
  def f_layer(self, l, _):
    """Layer l whole (QN.DEVROUTE): attention, router, moe_route, experts: xio[l % 2] -> xio[(l + 1) % 2]."""
    kd = self.kind[l]; Wl = W(self.views[(l % self.NSLOT, kd)], self.metas[l], self.sms[l], kd); K, c, B = self.K, self.c, self.lb[l]
    if QN.QDOT:                    # the qdot decode layer (qwen3moe_qdec): the same streams, integer GEMVs, 10-11 launches
      import qwen3moe_qdec as QD
      if not hasattr(self, "QB"): self.QB = QD.QBufs(K, c)
      offs = self.packs[self.kind.index(kd)][1]
      return QD.qlayer(K, self.QB, B.sel_gu, B.sel_dn, B.wt, Wl.st, Wl.meta, Wl.sm, c, self.xio[l % 2], self.Kc[l], self.Vc[l], self.cs, self.posb,
                       self.xio[(l + 1) % 2], offs["up"] - offs["gate"])
    x1 = QN.attn_body(K, B, Wl, c, 1, self.xio[l % 2], self.Kc[l], self.Vc[l], self.cs, self.posb)
    QN.dev_route(K, B, Wl, c, QN.router(K, Wl, c, x1))
    return QN.moe_body(K, B, Wl, c, 1, x1, self.xio[(l + 1) % 2])
  def f_head(self, _):
    """The final norm, the head's 4 Q6_K parts and each part's top-1 partials (top_head{j})."""
    K, c = self.K, self.c
    if QN.QDOT:
      import qwen3moe_qgemv as Q, qwen3moe_qdec as QD
      if not hasattr(self, "QB"): self.QB = QD.QBufs(K, c)
      N = self.head_parts[0][1]; tops = K.buf("top_head0", 48, dtypes.float32)
      hk = self.head_kind
      out = QD.call(K, f"qgemv|head|{hk}|{N}", Q.qgemvk_src(hk, N, c.H, act="norm", top=True, eps=c.EPS, nbuf=2), tops, self.QB.pkg_h, self.head_all, self.QB.sel0,
                    QN.desc(K, f"q_desc|{hk}|{c.H}", lambda: Q.qgemvk_desc(hk, c.H)), tops, self.xio[c.NL % 2], self.w_out)
    else: out = self.f_head_gs()
    if DEVTOK:                     # the token on the device: the parts' top-1 combined, then its embedding row as the next input
      ids = K.head_reduce(tuple(r0 for (_, _, _, _, r0) in self.head_parts))
      out = K.call(f"embed_next|{c.H}", MK.embed_next_src(c.H), self.xio[0], self.posb, self.ring, ids, self.emb, self.d_emb)
    return out
  def f_head_gs(self):
    K, c = self.K, self.c; a = K.rms_a(self.xio[c.NL % 2], self.w_out, c.H, "a_head"); out = None
    for j, (q8, N, npad, Kd, r0) in enumerate(self.head_parts):
      ng = npad // 48; ct = K.buf(f"ct_head{ng}", ng * QN.CS(K), dtypes.float32)
      OA.gemm_gs(a, self.head[j], ks=QN.KS, ns=QN.NS, nrb=K.nrb, nslices=Kd // (4 * QN.KS), ngroups=ng, b8=True, bscale=True, q8=q8, piece=K.nrb, rows=1, out=ct).realize()
      out = K.head_top(ct, N, f"top_head{j}")
    return out
  def run(self, key, fn):
    if not JIT: return fn(None)
    if key not in self.jits:
      from tinygrad.engine.jit import TinyJit
      self.jits[key] = TinyJit(fn)
    j = self.jits[key]
    if j.cnt < 2:
      with one_job(): return j(self.jd)
    return j(self.jd)

  def layer_w(self, l, drain=True):
    """Point slot l % NSLOT at layer l's weights. drain=False: the job in flight (if any) is the previous group's, on the other
    LPJ slots (this slot's last reader ran before it: one job at a time)."""
    if drain: self.raw.slot_map(self.slots[l % self.NSLOT][0], self.wbufs[l][0], 0, self.packs[l][2])
    else:
      try: self.raw.slot_map(self.slots[l % self.NSLOT][0], self.wbufs[l][0], 0, self.packs[l][2], drain=False)
      except TypeError: self.raw.slot_map(self.slots[l % self.NSLOT][0], self.wbufs[l][0], 0, self.packs[l][2])
    kd = self.kind[l]; return W(self.views[(l % self.NSLOT, kd)], self.metas[l], self.sms[l], kd)

  def step(self, tok, pos):
    """One token at position `pos` through the layers and the head -> the next token (greedy) and its probability."""
    K, B, c = self.K, self.B, self.c; self.prof = getattr(self, "prof", {"route": 0.0, "layers": 0.0, "head": 0.0})
    poke(self.posb, np.array([pos] + [0] * 15, np.int32))
    x0 = np.zeros((self.R, c.H), np.float32); x0[0] = self.M.embed([tok])[0]; poke(self.xio[0], x0)
    t0 = time.perf_counter(); P = self.prof
    def tick(k, t):
      if PROF: Device[QN.DEV].synchronize(); P[k] = P.get(k, 0.0) + time.perf_counter() - t
      return time.perf_counter()
    for l in range(c.NL if QN.DEVROUTE else 0):
      t = time.perf_counter(); self.layer_w(l, drain=PROF); t = tick("p_map", t)
      self.run(("layer", l), functools.partial(self.f_layer, l)); t = tick("p_layer", t)
    for l in range(0 if QN.DEVROUTE else c.NL):
      t = time.perf_counter(); Wl = self.layer_w(l); t = tick("p_map", t)
      rl = self.run(("attn", l), functools.partial(self.f_attn, l)); t = tick("p_attn", t)
      t1 = time.perf_counter(); lg = OA.host_invalidate(rl).numpy().reshape(self.R, c.NE)[:1]; t = tick("p_read", t)
      top, wt = QN.route(c, lg, 1); QN.set_route(B, Wl, c, top, wt); self.prof["route"] += time.perf_counter() - t1; t = tick("p_route", t)
      self.run(("moe", l), functools.partial(self.f_moe, l)); t = tick("p_moe", t)
    Device[QN.DEV].synchronize(); t1 = time.perf_counter(); self.prof["layers"] += t1 - t0
    self.run("head", self.f_head); best = None
    for j, (q8, N, npad, Kd, r0) in enumerate(self.head_parts):
      tp = OA.host_invalidate(K.bufs[f"top_head{j}"]).numpy().reshape(-1, 1, 4)[:, 0]
      t = int(np.argmax(tp[:, 0]))                                          # the first task with the largest max (its first column on ties)
      if best is None or tp[t, 0] > best[0]: best = (float(tp[t, 0]), r0 + int(tp[t, 1]), tp)
    self.prof["head"] += time.perf_counter() - t1
    m, tok, _ = best; return tok

  def f_group(self, l0, _):
    """Layers l0 .. l0 + LPJ - 1 as one graph (one job); the last group also runs the head and embed_next (HEAD_JOIN)."""
    out = None
    for l in range(l0, min(l0 + LPJ, self.c.NL)): out = self.f_layer(l, _)
    if HEAD_JOIN and l0 + LPJ >= self.c.NL: out = self.f_head(_)
    return out
  def run_token(self):
    """One decode step entirely on the device (DEVTOK): the 48 layers on xio[0] at position posb[0], the head, embed_next (the next
    token's row into xio[0], posb + 1, the id into ring[posb]). The host maps the slots and submits; it reads nothing. LPJ layers a
    job: their slots are mapped while the previous group's job runs (it reads the other LPJ slots)."""
    for l0 in range(0, self.c.NL, LPJ):
      for l in range(l0, min(l0 + LPJ, self.c.NL)): self.layer_w(l, drain=PROF)
      self.run(("layers", l0), functools.partial(self.f_group, l0))
      if SYNC_PROBE: Device[QN.DEV].synchronize(); OA.host_invalidate(self.K.bufs["q_rlog"]).numpy()   # (the cost of a host sync + read a layer)
    if not HEAD_JOIN: self.run("head", self.f_head)
  def decode_dev(self, first_tok, pos, n_new, stop, on_token=None):
    """`first_tok` at `pos` (poked once), then n_new tokens on the device; the ring read every READ_EVERY tokens (a stop token ends the
    loop there: up to READ_EVERY - 1 tokens past it are computed and dropped). -> the generated ids."""
    c, K = self.c, self.K; out = []            # the first token's row by embed_next too: ids = it, posb = pos - 1 (embed_next advances it)
    poke(K.bufs["ids_head"], np.array([first_tok] + [0] * 15, np.int32)); poke(self.posb, np.array([pos - 1] + [0] * 15, np.int32))
    K.call(f"embed_next|{c.H}", MK.embed_next_src(c.H), self.xio[0], self.posb, self.ring, K.bufs["ids_head"], self.emb, self.d_emb)
    for i in range(n_new):
      self.run_token()
      if (i + 1) % READ_EVERY == 0 or i + 1 == n_new:
        ring = OA.host_invalidate(self.ring).numpy(); new = [int(t) for t in ring[pos + 1 + len(out):pos + 2 + i]]
        for t in new:
          out.append(t)
          if on_token: on_token(t)
          if t in stop: return out
    return out
  def warmup(self):
    """Capture every graph (TinyJit: the second call captures, the third replays): two throwaway steps at position 0 (the prompt
    rewrites the cache row they touched)."""
    t0 = time.perf_counter()
    if DEVTOK and JIT:             # the device step's own graphs (run_token's keys), so the prompt's first tokens replay, not capture
      c, K = self.c, self.K; ids = K.buf("ids_head", 16, dtypes.int32)       # (head_reduce's own buffer, made here first)
      for _ in range(2):
        poke(ids, np.array([QR.IM_START] + [0] * 15, np.int32)); poke(self.posb, np.array([-1] + [0] * 15, np.int32))
        K.call(f"embed_next|{c.H}", MK.embed_next_src(c.H), self.xio[0], self.posb, self.ring, K.bufs["ids_head"], self.emb, self.d_emb)
        self.run_token()
      Device[QN.DEV].synchronize()
    else:
      for _ in range(2 if JIT else 0): self.step(QR.IM_START, 0)
    if PREFILL == "batch":                                                 # the prefill's kernels and GEMM runners (eager: built once, here)
      import qwen3moe_prefill as QP
      self.pre = QP.Prefill(self); self.pre.chunk([QR.IM_START] * 4, 0)
      self.pre_s = QP.Prefill(self, QP.M_SHORT) if QP.M_SHORT < QP.M_CHUNK else None   # (short prompts: chunks of M_SHORT)
      if self.pre_s: self.pre_s.chunk([QR.IM_START] * 4, 0)
    self.prof = {"route": 0.0, "layers": 0.0, "head": 0.0}
    return time.perf_counter() - t0
  def generate(self, ids, n_new, stop=(QR.IM_END, QR.EOT), on_token=None):
    if DEVTOK:
      t0 = time.perf_counter()
      if PREFILL == "batch" and len(ids) > int(os.environ.get("QWEN3MOE_PREFILL_MIN", "24")):
        (self.pre_s if self.pre_s and len(ids) - 1 <= self.pre_s.M else self.pre).prefill(ids[:-1], 0)
      else:                        # a short prompt: the device step a token, the host writing only the token's id and position
        c, K = self.c, self.K
        for p, t in enumerate(ids[:-1]):
          poke(K.bufs["ids_head"], np.array([t] + [0] * 15, np.int32)); poke(self.posb, np.array([p - 1] + [0] * 15, np.int32))
          K.call(f"embed_next|{c.H}", MK.embed_next_src(c.H), self.xio[0], self.posb, self.ring, K.bufs["ids_head"], self.emb, self.d_emb)
          self.run_token()
      Device[QN.DEV].synchronize(); t_pre = time.perf_counter() - t0; t1 = time.perf_counter()
      out = self.decode_dev(ids[-1], len(ids) - 1, n_new, stop, on_token)
      return out, t_pre, time.perf_counter() - t1
    t0 = time.perf_counter(); nxt = None
    if PREFILL == "batch" and len(ids) > int(os.environ.get("QWEN3MOE_PREFILL_MIN", "24")):   # (a chunk reads ~every expert: ~0.9 s at least)
      (self.pre_s if self.pre_s and len(ids) - 1 <= self.pre_s.M else self.pre).prefill(ids[:-1], 0); nxt = self.step(ids[-1], len(ids) - 1)
    else:
      for p, t in enumerate(ids): nxt = self.step(t, p)
    t_pre = time.perf_counter() - t0; out = [nxt]; pos = len(ids); t1 = time.perf_counter()
    if on_token: on_token(nxt)
    while len(out) < n_new and out[-1] not in stop:
      out.append(self.step(out[-1], pos)); pos += 1
      if on_token: on_token(out[-1])
    return out, t_pre, time.perf_counter() - t1


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("prompt", nargs="?", default="What is the capital of Portugal? Answer in one sentence.")
  ap.add_argument("--max-new", type=int, default=32); ap.add_argument("--thinking", action="store_true"); ap.add_argument("--tmax", type=int, default=1024)
  ap.add_argument("--cache", default=QN.CACHE); ap.add_argument("--out", default=""); a = ap.parse_args()
  from tokenizers import Tokenizer
  cpus, lat = pin_cpus(), hold_cpu_latency(); print(f"host: CPUs {cpus}, CPU-latency request {lat} us", flush=True)
  tk = Tokenizer.from_file(os.path.join(TOK, "tokenizer.json")); ids = tk.encode(QR.chat_text(a.prompt, a.thinking), add_special_tokens=False).ids
  m = Model(a.cache, tmax=a.tmax); print(f"graphs {'captured in %.0f s' % m.warmup() if JIT else 'off'}; {len(ids)} prompt tokens", flush=True)
  out, t_pre, t_dec = m.generate(ids, a.max_new, on_token=lambda t: print(tk.decode([t]), end="", flush=True))
  print(f"\n\nids: {out}\nprefill {len(ids)} tokens in {t_pre:.1f} s ({len(ids) / t_pre:.2f} tok/s, {("chunks of %d" % (m.pre_s.M if m.pre_s and len(ids) - 1 <= m.pre_s.M else m.pre.M)) if PREFILL == "batch" and len(ids) > int(os.environ.get("QWEN3MOE_PREFILL_MIN", "24")) else "one at a time"}); decode {len(out) - (0 if DEVTOK else 1)} tokens in {t_dec:.1f} s "
        f"({(len(out) - (0 if DEVTOK else 1)) / max(t_dec, 1e-9):.2f} tok/s) | " + " ".join(f"{k[2:]} {v / (len(ids) + len(out) - 1) / 48 * 1e3:.2f}" for k, v in m.prof.items() if k.startswith("p_")) + " ms a layer | " + f"per step: layers {m.prof['layers'] / (len(ids) + len(out) - 1) * 1e3:.0f} ms (routing reads {m.prof['route'] / (len(ids) + len(out) - 1) * 1e3:.0f}), head {m.prof['head'] / (len(ids) + len(out) - 1) * 1e3:.0f} ms")
  if a.out: np.savez(a.out, ids=np.array(ids), out=np.array(out))


if __name__ == "__main__": main()

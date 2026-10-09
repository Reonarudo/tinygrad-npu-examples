#!/usr/bin/env python3
"""Gemma 4 (E2B) greedy generation on the NPU from gemma4_pack.py's cache.

Per token the host looks up the scaled embedding row (and, GEMMA_PLE=host, the per-layer inputs) and writes it, with the position,
into fixed device buffers; then the device runs
  - (GEMMA_PLE=dev, default) the per-layer inputs: the token's PLE table row (host lookup) + the Q8_0 projection of the embedding,
    normed per layer (gemma4_kernels.ple_mix32) into the stack of every layer's PLE rows;
  - the 35 layers (gemma4_npu.layer_body), GEMMA_LAYER_BLOCK layers a TinyJit call -- each captured once as one frozen graph and
    one job (the zhouyi backend's ZhouyiGraph: replays poke nothing but changed words, the last job left in flight while the host
    goes on); the residual rows ping-pong between two persistent buffers, nothing is copied between layers;
  - (GEMMA_HEAD=dev, default) the tied head: the final norm into the A layout, the Q8_0 GEMM over the 262144-row embedding, the
    top-1 by DMA (head_topd) and the token's id on the device (head_reduce): only the id comes back. The final soft-cap is monotone
    and greedy needs no probabilities: not applied. GEMMA_HEAD=host: the logits' C tiles read back and argmax'd on the host.
The weights are zero-copy: each layer's streams in a weight buffer (KMD WBUF, pinned RAM outside the NPU window), mapped once into a
window slot of its own (E2B's 2.4 GB fit the window, so no slot is ever re-mapped).
The prompt runs through the rows-mode graphs, --prefill-m tokens a pass. Measured on the board (published backend f7e4601, one
token a pass), the 22-token Fibonacci prompt + 40 tokens, ids identical to llama.cpp CPU: 124 ms a token, 8.05 tok/s (the
bring-up path: 1765 ms, 0.57 tok/s); at position ~1074 132 ms a token.

    python3 gemma4_pack.py                                         (once: the cache, 2.43 GB, ~15 s)
    python3 gemma4_generate.py [--n-new 40] [--cache /mnt/ssd/gemma4-e2b-npu] [--tmax 1024] [--check] ID ...
    python3 gemma4_generate.py --spec 6 --tau 0.5 ID ...            (speculative decoding with the MTP drafter: gemma4_mtp.py packs it)
Measured on the board (2026-10-09), 3 prompts pooled: E2B plain 8.16 tok/s, --spec 6 --tau 0.5
18.1 tok/s; E4B (GEMMA_LAYER_BLOCK=4 GEMMA_SPEC_GEOS=3,5,7) 4.27 -> 11.0 tok/s; ids identical to plain greedy. Prefill 8 tokens a
pass (--prefill-m): 1074 tokens 23.3 s (E2B) / 39.0 s (E4B).

Env: GEMMA_LAYER_BLOCK (layers a job, default 5: 1 / 5 / 7 / 35 measured 159.4 / 157.5 / 157.4 / 157.7 ms a token), GEMMA_JIT=0 (no graphs: every kernel its own job), GEMMA_HEAD, GEMMA_PLE,
GEMMA_CPUS / GEMMA_CPU_LATENCY (as qwen38_generate's QWEN_CPUS / QWEN_CPU_LATENCY), GEMMA_ZC=0 (weights copied into ordinary
device buffers instead of zero-copy slots).
"""
import argparse, functools, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gemma4_npu as GN                                                   # noqa: E402  (puts TG and qwen3.8-27b on sys.path)
from gemma4_npu import dev, poke, NS, KS, OA, Tensor, dtypes, Kernels, SharedK, LayerW, layer_body, PLE_ROWS   # noqa: E402
import gemma4_kernels as GK                                              # noqa: E402
import gemma4_ref as GR                                                  # noqa: E402
from tinygrad.engine.jit import TinyJit                                  # noqa: E402
from qwen38_npu import read_into, slot_map_nodrain                       # noqa: E402

DEFAULT_CACHE = os.environ.get("GEMMA_NPU", "/mnt/ssd/gemma4-e2b-npu")

# ---- the host's CPU set and latency request: copies of qwen38_generate's pin_cpus / hold_cpu_latency / one_job (that module imports
# the Qwen tokenizer); see their notes there (the NPU interrupt's CPU kept awake: the gaps between a job's kernels ~8 instead of 45-75 us)
def pin_cpus():
  v = os.environ.get("GEMMA_CPUS", os.environ.get("QWEN_CPUS", "auto"))
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
  v = os.environ.get("GEMMA_CPU_LATENCY", os.environ.get("QWEN_CPU_LATENCY", "0"))
  if v == "off": return None
  try: f = open("/dev/cpu_dma_latency", "wb", buffering=0); f.write(int(v).to_bytes(4, "little")); _LAT.append(f); return int(v)
  except OSError as e: print(f"   (no PM QoS request -- {e.strerror})", flush=True); return None
def one_job():
  """The capturing TinyJit call's kernels in one graph (JIT_BATCH_SIZE=0) and one chain (ZHOUYI_CHAIN_CAP): one job."""
  from tinygrad.helpers import Context, ContextVar
  return Context(**{k: v for k, v in (("JIT_BATCH_SIZE", 0), ("ZHOUYI_CHAIN_CAP", 4096)) if k in ContextVar._cache})

class Weights:
  """gemma4_pack.py's cache on the device. Zero-copy (GEMMA_ZC, default 1): each layer's streams (2 MiB-aligned, +16 B of load
  overhang) in one KMD weight buffer, a window slot of the same size mapped onto it once; the linears are fixed views into the slot.
  Else (or when the driver refuses): the streams in ordinary device buffers.
  Re-mapped (GEMMA_REMAP: auto -- when one slot a layer would pass GEMMA_WINDOW_GB of the NPU's 3 GB window, E4B's 4.96 GB --, 1, 0):
  every layer in its weight buffer, unmapped; 2 x `block` window slots of the largest layer's size, layer l's linears fixed views
  into slot (block parity, position in block); map_block(b) points block b's slots at its layers' buffers while block b - 1 (the
  other parity's slots) runs -- the Qwen family's QWEN_LAYER_BLOCK slots (qwen38_npu.Layers.block_slots / map_block)."""
  def __init__(self, c, cache, block=5):
    self.c, self.cache = c, cache; t0 = time.perf_counter()
    from tinygrad import Device
    self.raw = Device[GN.DEV].raw; self.zc = os.environ.get("GEMMA_ZC", "1") != "0"; self.held = []
    self.layers = []; self.block = block
    pk = lambda files: sum(-(-(os.path.getsize(f) + 16) // (2 << 20)) * (2 << 20) for f in files.values())
    lfiles, metas = [], []
    for l in range(c.NL):
      z = np.load(os.path.join(cache, f"L{l}_small.npz")); names = [str(k) for k in z["meta_names"]]; metas.append(z)
      lfiles.append({k: os.path.join(cache, f"L{l}_{k}.bin") for k in names})
    tot = sum(pk(f) for f in lfiles) + sum(os.path.getsize(os.path.join(cache, f)) for f in ("head_q8.bin", "ple_proj_q8.bin") if os.path.exists(os.path.join(cache, f)))
    rm = os.environ.get("GEMMA_REMAP", "auto"); self.remap = self.zc and (rm == "1" or (rm == "auto" and tot > float(os.environ.get("GEMMA_WINDOW_GB", "2.6")) * 1e9))
    if self.remap:
      big = max(pk(f) for f in lfiles); self.bslots = {(bp, j): self.raw.slot_alloc(big) for bp in range(2) for j in range(block)}; self.lbuf = {}
    for l in range(c.NL):
      z, files = metas[l], lfiles[l]; names = list(files)
      views = self.layer_blob(l, files) if self.remap else self.blob(files)
      lin = {k: (views[k], int(m[0]), int(m[1])) for k, m in zip(names, z["meta"])}
      goff = {k: [int(v) for v in z[f"goff_{k}"]] for k in names}
      sm = {k: dev(z[k]) for k in z.files if not k.startswith(("meta", "goff_", "vsame"))}
      self.layers.append(LayerW(c, l, lin, goff, sm, vsame=bool(int(z["vsame"]))))
    hm = np.load(os.path.join(cache, "head_q8.npz"))["meta"]; self.vocab, self.head_npad, hk = (int(v) for v in hm)
    self.head = (self.blob({"head": os.path.join(cache, "head_q8.bin")})["head"], self.head_npad, hk)
    o = np.load(os.path.join(cache, "outside_small.npz")); self.w_out = dev(o["output_norm"])
    if c.PLE:
      pm = o["ple_meta"]; self.ple = (self.blob({"p": os.path.join(cache, "ple_proj_q8.bin")})["p"], int(pm[1]), int(pm[2]))
      self.ple_norm = dev(o["per_layer_proj_norm"])
    gb = sum(os.path.getsize(os.path.join(cache, f)) for f in os.listdir(cache) if f.endswith(".bin")) / 1e9
    how = (f"zero-copy, re-mapped: {2 * block} block slots of {big / 1e6:.0f} MB" if self.remap else "zero-copy (weight buffers, a slot each)") if self.zc else "copied into device buffers"
    print(f"   weights: {gb:.2f} GB {how} in {time.perf_counter() - t0:.1f} s", flush=True)
  def slot_of(self, l): return ((l // self.block) % 2, l % self.block)
  def layer_blob(self, l, files):
    """Layer l's streams in a weight buffer of its own (not mapped) -> views into its block slot (fixed per layer)."""
    sizes = {k: os.path.getsize(f) for k, f in files.items()}; offs, o = {}, 0
    for k, sz in sizes.items(): offs[k] = o; o += -(-(sz + 16) // (2 << 20)) * (2 << 20)
    wid, mm = self.raw.wbuf_alloc(o)
    for k, f in files.items(): read_into(f, memoryview(mm)[offs[k]:offs[k] + sizes[k]], sizes[k], drop=True)
    self.lbuf[l] = (wid, mm, o); pa = self.bslots[self.slot_of(l)][1]
    return {k: Tensor.from_blob(pa + offs[k], (sz,), dtype=dtypes.uint8, device=GN.DEV) for k, sz in sizes.items()}
  def map_block(self, ls):
    """The block's layers into their slots, without waiting for the job in flight: that is the previous block's, on the other
    parity's slots (every submit waits for the job before it, so this parity's last reader, two blocks back, has run)."""
    for l in ls: wid, _, n = self.lbuf[l]; slot_map_nodrain(self.raw, self.bslots[self.slot_of(l)][0], wid, 0, n)
  def blob(self, files):
    """The files' streams on the device -> {name: a uint8 tensor over each}."""
    sizes = {k: os.path.getsize(f) for k, f in files.items()}
    if self.zc:
      offs, o = {}, 0
      for k, sz in sizes.items(): offs[k] = o; o += -(-(sz + 16) // (2 << 20)) * (2 << 20)
      try:
        wid, mm = self.raw.wbuf_alloc(o); sid, pa = self.raw.slot_alloc(o)
      except OSError as e:
        print(f"   zero-copy unavailable ({e}); copying the weights", flush=True); self.zc = False
      else:
        for k, f in files.items(): read_into(f, memoryview(mm)[offs[k]:offs[k] + sizes[k]], sizes[k], drop=True)
        self.raw.slot_map(sid, wid, 0, o); self.held.append((wid, mm, sid))
        return {k: Tensor.from_blob(pa + offs[k], (sz,), dtype=dtypes.uint8, device=GN.DEV) for k, sz in sizes.items()}
    return {k: dev(np.fromfile(f, np.uint8)) for k, f in files.items()}

class Geo:
  """One row geometry: m rows a pass (1: a decode step; 2..12: a verify pass or a prefill chunk) -- its kernels (Kernels(real=m):
  the rows-mode GEMM and the small kernels at m rows), their persistent scratch, the residual rows' two ping-pong buffers and its
  graphs. The weights, the caches, the RoPE tables, the position and the PLE stack are the model's (shared by every geometry)."""
  def __init__(self, c, m):
    self.m = m; self.K = SharedK(2, 24, c.EPS, real=m); self.xb = [self.K.rows_buf("xio0", c.H), self.K.rows_buf("xio1", c.H)]; self.jits = {}

def zeros(n, dt): return Tensor.zeros(n, device=GN.DEV, dtype=dt).contiguous().realize()

class Model:
  def __init__(self, M, cache=DEFAULT_CACHE, tmax=1024):
    self.M, self.c, self.tmax = M, M.c, tmax; c = M.c; t0 = time.perf_counter()
    k = int(os.environ.get("GEMMA_LAYER_BLOCK", "5")); self.Wt = Weights(c, cache, k); self.vocab = self.Wt.vocab
    # the caches of the layers that own one: global layers TMAX rows; local layers (GEMMA_ATT_DMA) a ring of W + 16 rows, any length
    self.ring = {l: (GN.ring_rows(c) if c.SWA[l] and GN.ATT_DMA else 0) for l in range(c.NL)}
    rows = lambda l: self.ring[l] or tmax
    self.cache = {l: (zeros(rows(l) * c.NKV[l] * c.HD[l], dtypes.float32), zeros(rows(l) * c.NKV[l] * c.HD[l], dtypes.float32))
                  for l in range(c.NL) if c.KV_SRC[l] == l}
    l_loc, l_glo = c.SWA.index(True), c.SWA.index(False)
    self.cs = {True: dev(GN.rope_cs(M, l_loc, tmax)), False: dev(GN.rope_cs(M, l_glo, tmax))}
    self.posb = zeros(16, dtypes.int32)
    self.pl = zeros(c.NL * PLE_ROWS * max(c.PLE, 1), dtypes.float32)
    self.pidx = [dev(np.array([l * PLE_ROWS * c.PLE], np.int32)) for l in range(c.NL)]
    self.jd = dev(np.zeros(16, np.int32))
    self.head_mode = os.environ.get("GEMMA_HEAD", "dev"); self.ple_mode = os.environ.get("GEMMA_PLE", "dev") if c.PLE else "none"
    assert self.head_mode in ("dev", "host") and self.ple_mode in ("dev", "host", "none")
    if self.ple_mode == "dev": self.ple_tok = zeros(PLE_ROWS * c.NL * c.PLE, dtypes.float32)
    if self.ple_mode == "host": M.t("per_layer_model_proj.weight") if M.cache_layers else None
    self.jit = os.environ.get("GEMMA_JIT", "1") != "0"
    assert self.jit or not self.Wt.remap, "the re-mapped slots need the block graphs (GEMMA_JIT=1)"
    self.blocks = [list(range(b, min(c.NL, b + k))) for b in range(0, c.NL, k)]
    self.geos = {}; self.w_out_h = np.load(os.path.join(cache, "outside_small.npz"))["output_norm"].astype(np.float32)
    print(f"   model set-up: {time.perf_counter() - t0:.1f} s; {len(self.blocks)} layer blocks of {k}, graphs {'on' if self.jit else 'off'}, "
          f"head {self.head_mode}, per-layer inputs {self.ple_mode}", flush=True)
  def geo(self, m):
    if m not in self.geos: self.geos[m] = Geo(self.c, m)
    return self.geos[m]

  # ---- the device work of geometry G, as functions of a dummy input (each a TinyJit when GEMMA_JIT)
  def f_block(self, G, ls, _):
    out = None
    for l in ls:
      Kc, Vc = self.cache[self.c.KV_SRC[l]]
      out = layer_body(G.K, self.Wt.layers[l], G.m, G.xb[l % 2], G.xb[1 - l % 2], Kc, Vc, self.cs[self.c.SWA[l]], self.posb, self.tmax, self.pl, self.pidx[l],
                       ring=self.ring[self.c.KV_SRC[l]])
    return out
  def f_head(self, G, _):
    K, c = G.K, self.c; b, npad, hk = self.Wt.head; ng = npad // (16 * NS)
    a = K.rms_a(G.xb[c.NL % 2], self.Wt.w_out, c.H, "a_fin")
    ct = OA.gemm_gs(a, b, ks=KS, ns=NS, nrb=K.nrb, nslices=hk // (4 * KS), ngroups=ng, b8=True, bscale=True, q8=1, scales="dup",
                    piece=K.nrb, rows=G.m, out=K.buf("ct_head", ng * K.nrb * NS * 192, dtypes.float32)).realize()
    if self.head_mode == "host": return ct
    K.head_top(ct, self.vocab, "top_head0")
    return K.head_reduce((0,))
  def f_ple(self, G, _):
    K, c = G.K, self.c; b, npad, pk = self.Wt.ple; ng = npad // (16 * NS)
    a = K.rms_a(G.xb[0], None, c.H, "a_x0", norm=False)
    ct = OA.gemm_gs(a, b, ks=KS, ns=NS, nrb=K.nrb, nslices=pk // (4 * KS), ngroups=ng, b8=True, bscale=True, q8=1, scales="dup",
                    piece=K.nrb, rows=G.m, out=K.buf("ct_ple", ng * K.nrb * NS * 192, dtypes.float32)).realize()
    return K.call(f"ple_mix32|{c.NL}|{c.PLE}|{G.m}", GK.ple_mix32_src(c.NL, c.PLE, c.H, K.nrb, c.EPS, G.m, PLE_ROWS), self.pl, ct, self.ple_tok, self.Wt.ple_norm)
  def run(self, G, key, fn):
    if not self.jit: return fn(None)
    if key not in G.jits: G.jits[key] = TinyJit(fn)
    j = G.jits[key]
    if j.cnt < 2:
      with one_job(): return j(self.jd)
    return j(self.jd)

  def forward(self, toks, pos, head=True):
    """The tokens `toks` (m <= 12) at positions pos .. pos + m - 1 through the layers on geometry m (the caches get their K / V rows),
    then (head) the head on every row -> (ids [m]: each row's greedy next token, probabilities [m] of the head's softmax without the
    soft-cap); head=False: None."""
    c, M = self.c, self.M; m = len(toks); G = self.geo(m)
    assert pos + m <= self.tmax, f"positions {pos}..{pos + m - 1}: the global layers' caches hold --tmax {self.tmax} rows"
    x0 = M.embed(toks); poke(G.xb[0], x0); poke(self.posb, np.array([pos] + [0] * 15, np.int32))
    if self.ple_mode == "host":
      pz = np.zeros((c.NL, PLE_ROWS, c.PLE), np.float32); pz[:, :m] = M.ple_inputs(toks, x0).transpose(1, 0, 2); poke(self.pl, pz)
    elif self.ple_mode == "dev":
      poke(self.ple_tok, M.g.q8_rows("per_layer_token_embd.weight", list(toks)) * np.float32(math.sqrt(c.PLE)))
      self.run(G, "ple", functools.partial(self.f_ple, G))
    for b, ls in enumerate(self.blocks):
      if self.Wt.remap: self.Wt.map_block(ls)
      self.run(G, ("blk", b), functools.partial(self.f_block, G, ls))
    if not head: return None
    o = self.run(G, "head", functools.partial(self.f_head, G))
    if self.head_mode == "host":
      lg = GK.ct_rows(OA.host_invalidate(o).numpy(), G.K.nrb, self.vocab)[:m]
      e = np.exp(lg - lg.max(-1, keepdims=True)); return lg.argmax(-1).astype(np.int64), (1.0 / e.sum(-1)).astype(np.float32)
    o = OA.host_invalidate(G.K.bufs["ids_head"]).numpy(); return o[:m].astype(np.int64), o[m:2 * m].view(np.float32).copy()
  def hidden(self, m, row):
    """The last layer's output row `row` of geometry m's last pass, final-normed on the host (the MTP drafter's input h)."""
    x = OA.host_invalidate(self.geo(m).xb[self.c.NL % 2]).numpy()[row].astype(np.float32)
    return (x * (1.0 / np.sqrt(np.mean(x * x) + self.c.EPS)) * self.w_out_h).astype(np.float32)
  def step(self, tok, pos):
    """Token `tok` at position `pos` through the layers and the head -> the next token's id (greedy)."""
    return int(self.forward([tok], pos)[0][0])

  def warmup(self, ms=(1,)):
    """Capture geometry m's graphs (TinyJit: the second call captures, the third replays) before the prompt: two throwaway passes at
    position 0 (the prompt rewrites every cache row they touched)."""
    t0 = time.perf_counter()
    for m in ms:
      for _ in range(2 if self.jit else 0): self.forward([2] * m, 0)
    if self.jit: print(f"   graphs of geometries {list(ms)} captured in {time.perf_counter() - t0:.1f} s", flush=True)

  def prefill(self, ids, pm=1):
    """The prompt -> the first generated token. pm > 1: chunks of pm tokens a pass through geometry pm (the last chunk padded with its
    last token: the padding rows' K / V land at positions past the prompt, rewritten before any row reads them); the head runs on
    the last chunk only. pm = 1: one token a pass."""
    n = len(ids); nxt = None
    for c0 in range(0, n, pm):
      ch = list(ids[c0:c0 + pm]); a = len(ch); last = c0 + pm >= n
      r = self.forward(ch + [ch[-1]] * (pm - a), c0, head=last)
      if last: nxt = int(r[0][a - 1]); self.last_rows = (pm, a - 1)
    return nxt

  def generate(self, ids, n_new, stop=(1, 106), timing=None, pm=1):
    t0 = time.perf_counter(); ids = list(ids)
    nxt = self.prefill(ids, pm)
    out = [nxt]; pos = len(ids); t1 = time.perf_counter()
    print(f"   prefill {len(ids)} tokens: {t1 - t0:.2f} s ({'one token a pass' if pm == 1 else f'{pm} tokens a pass'}; {len(ids) / (t1 - t0):.1f} tok/s)", flush=True)
    ts = []
    while len(out) < n_new and out[-1] not in stop:
      ta = time.perf_counter(); out.append(self.step(out[-1], pos)); pos += 1; ts.append(time.perf_counter() - ta)
    dt = time.perf_counter() - t1
    if ts: print(f"   decode {len(ts)} tokens: {dt:.2f} s, {len(ts) / dt:.2f} tok/s ({1e3 * dt / len(ts):.1f} ms a token; median step {1e3 * float(np.median(ts)):.1f} ms)", flush=True)
    if timing is not None: timing.update(prefill_s=t1 - t0, decode_s=dt, steps=ts)
    return out

  def spec_generate(self, ids, n_new, dr, kmax, tau=0.0, stop=(1, 106), timing=None, pm=8, stats=None, geos=None):
    """Greedy speculative decoding with the MTP drafter `dr` (gemma4_mtp.Drafter): each pass verifies [cur, d1 .. dk] (k <= kmax
    drafts chained at cur's position; the chain stops after a draft below tau) on geometry 1 + k, keeps the drafts the model agrees
    with plus its own next token. The commit is the K / V rows alone: the accepted rows stay, the rejected ones sit at positions
    the next pass rewrites before any row reads them (the local rings hold W + 16 >= W + 11 rows; the KV-shared layers read their
    sources' rows of the same pass). The ids are those of plain greedy decoding up to the geometries' rounding (near-ties)."""
    st = {} if stats is None else stats; t0 = time.perf_counter(); ids = list(ids)
    nxt = self.prefill(ids, pm); pos = len(ids); h = self.hidden(*self.last_rows); out, cur = [nxt], nxt; t1 = time.perf_counter()
    print(f"   prefill {len(ids)} tokens: {t1 - t0:.2f} s ({pm} tokens a pass; {len(ids) / (t1 - t0):.1f} tok/s)", flush=True)
    st.update(passes=0, accepted=0, drafted=0, rows=[], draft_log=[], t_draft=0.0, t_verify=0.0)
    while len(out) < n_new and cur not in stop:
      td = time.perf_counter(); d, pr = dr.chain(cur, h, pos, kmax, tau); td = time.perf_counter() - td
      nr = 1 + len(d); m = min(x for x in (geos or [nr]) if x >= nr); toks = [cur] + d + [d[-1]] * (m - nr)   # (padded to a kept geometry)
      if pos + m > self.tmax: break
      tv = time.perf_counter(); g, _ = self.forward(toks, pos); tv = time.perf_counter() - tv
      a = 1
      while a < nr and toks[a] == int(g[a - 1]): a += 1
      new = toks[1:a] + [int(g[a - 1])]
      st["draft_log"].extend((j, pr[j], j < a - 1) for j in range(len(d))); st["rows"].append(m)
      st["passes"] += 1; st["accepted"] += a - 1; st["drafted"] += len(d); st["t_draft"] += td; st["t_verify"] += tv
      for t in new:
        out.append(t)
        if t in stop or len(out) >= n_new: break
      if out[-1] in stop: break                                            # (an accepted draft can be the end of turn)
      cur = int(g[a - 1]); h = self.hidden(m, a - 1); pos += a
    dt = time.perf_counter() - t1; nt = len(out) - 1
    if nt: print(f"   speculative decode {nt} tokens: {dt:.2f} s, {nt / dt:.2f} tok/s | {st['passes']} passes, {nt / max(1, st['passes']):.2f} tokens a pass, "
                 f"{st['accepted']} of {st['drafted']} drafts accepted ({st['accepted'] / max(1, st['drafted']):.0%}) | draft {1e3 * st['t_draft'] / max(1, st['passes']):.1f} ms, "
                 f"verify {1e3 * st['t_verify'] / max(1, st['passes']):.1f} ms a pass", flush=True)
    if timing is not None: timing.update(prefill_s=t1 - t0, decode_s=dt, **{k: v for k, v in st.items()})
    return out


def GM_default_mtp(gguf):
  import gemma4_mtp as GM
  return GM.default_mtp(gguf)

def main():
  cpus = pin_cpus(); lat = hold_cpu_latency()
  ap = argparse.ArgumentParser(); ap.add_argument("--cache", default=DEFAULT_CACHE); ap.add_argument("--n-new", type=int, default=40)
  ap.add_argument("--tmax", type=int, default=1024); ap.add_argument("--check", action="store_true", help="the numpy reference's greedy ids alongside")
  ap.add_argument("--prefill-m", type=int, default=int(os.environ.get("GEMMA_PREFILL_M", "8")), help="prompt tokens a pass (1..12; 1: one token a pass)")
  ap.add_argument("--spec", type=int, default=None, help="speculative decoding: the MTP drafter's drafts a pass at most (0: plain greedy; default "
                  "GEMMA_SPEC, else 6 when the drafter's GGUF is found beside the model)")
  ap.add_argument("--tau", type=float, default=float(os.environ.get("GEMMA_DRAFT_TAU", "0.5")), help="the draft chain stops after a draft below this probability")
  ap.add_argument("--mtp", default=None, help="the drafter's GGUF (default: mtp-<the target's GGUF name> beside it)")
  ap.add_argument("--json", help="write ids and timings here"); ap.add_argument("ids", nargs="*", type=int)
  a = ap.parse_args(); meta = json.load(open(os.path.join(a.cache, "gemma4.json")))
  if a.spec is None:                             # speculative by default when the drafter is there (k 6, tau 0.5: the measured best)
    a.spec = int(os.environ["GEMMA_SPEC"]) if "GEMMA_SPEC" in os.environ else 6 if os.path.exists(a.mtp or GM_default_mtp(meta["gguf"])) else 0
  if a.spec and "E4B" in os.path.basename(meta["gguf"]):   # E4B: the window holds ~5 captured geometries next to its weight slots
    os.environ.setdefault("GEMMA_LAYER_BLOCK", "4"); os.environ.setdefault("GEMMA_SPEC_GEOS", "3,5,7")
  M = GR.Model(meta["gguf"], cache_layers=os.environ.get("GEMMA_PLE", "dev") == "host"); ids = a.ids or GN.PROMPT
  print(f"== {len(ids)} prompt tokens; greedy, up to {a.n_new} new; prefill {a.prefill_m} tokens a pass; host CPUs {cpus}; CPU latency {lat}", flush=True)
  m = Model(M, a.cache, a.tmax); tm = {}
  if a.spec:
    import gemma4_mtp as GM
    # GEMMA_SPEC_GEOS: the verify geometries kept (a pass rounds 1 + its drafts up to the next, padded with the last draft); default
    # every 2 .. spec + 1. Each geometry's graphs take launch arenas in the NPU window: E4B (2 x 4 block slots of 122 MB + its head)
    # captures at most ~5 geometries and the drafter
    sg = sorted({int(x) for x in os.environ.get("GEMMA_SPEC_GEOS", ",".join(str(x) for x in range(2, a.spec + 2))).split(",")} | {a.spec + 1})
    dr = GM.Drafter(m, a.mtp or GM.default_mtp(meta["gguf"]), os.path.join(a.cache, "mtp")); m.warmup(sorted({a.prefill_m} | set(sg)))
    t0 = time.perf_counter(); dr.chain(2, np.zeros(m.c.H, np.float32), 1, 2); print(f"   drafter graph captured in {time.perf_counter() - t0:.1f} s", flush=True)
    out = m.spec_generate(ids, a.n_new, dr, a.spec, a.tau, timing=tm, pm=a.prefill_m, geos=sg); print("NPU speculative: ", out, flush=True)
  else:
    m.warmup(sorted({1, a.prefill_m})); out = m.generate(ids, a.n_new, timing=tm, pm=a.prefill_m); print("NPU greedy:      ", out, flush=True)
  if a.json: json.dump(dict(ids=ids, out=out, prefill_m=a.prefill_m, env={k: v for k, v in os.environ.items() if k.startswith(("GEMMA_", "ZHOUYI_", "QWEN_"))}, **tm), open(a.json, "w"))
  if a.check:
    ref = M.greedy(ids, len(out) - 1); print("numpy greedy:    ", ref); print("identical:", ref[:len(out)] == out)

if __name__ == "__main__": main()

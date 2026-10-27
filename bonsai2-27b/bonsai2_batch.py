#!/usr/bin/env python3
"""Batched decode of N independent requests on Ternary Bonsai 2 (the 2B by default; `QWEN_MODEL=bonsai2-4b | -9b | -27b` too): the
linears (the weight bytes) run once for all N rows on the GEMM's rows mode (a Layers(N) geometry of the Qwen3.8 runtime, its A
producers, residuals and the dense MLP as the verify pass uses them); the sequence mixers run per user on the one-row kernels
(gdn_tok3 / attn_dec2) with the user's own DeltaNet state, conv ring and K / V cache (the per-layer stacks replicated per user, the
index = user x layers + layer), fed by small scatter / gather kernels: the GEMM's C tiles' row u into a per-user tile buffer, the
unpacked rows' row u into a per-user row, the per-user outputs back into the N-row buffer. Every row is plain greedy decoding of its
own request (checked against the single-request path); the prompts go through the same batched step a token (in lockstep).

    python3 bonsai2_batch.py --users 4 --max-new 32            # the throughput at N users (prompts trimmed to one length)
    python3 bonsai2_batch.py --users 1 --check PLAIN.npz        # user 0's ids against a bonsai2_generate run (--out)
"""
import argparse, functools, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("QWEN_MODEL", "bonsai2-2b"); os.environ.setdefault("QWEN_DRAFT", "none"); os.environ.setdefault("QWEN_SPEC_GEOS", "2,3,4")
import bonsai2_generate                                                   # noqa: E402,F401  (the paths and the small sizes' settings)
sys.path.insert(0, os.path.join(HERE, "..", "seqmix"))
from tinygrad import Tensor, dtypes                                      # noqa: E402
from tinygrad.engine.jit import TinyJit                                  # noqa: E402
from zy import OA, vec_f16 as V                                          # noqa: E402
import qwen38_generate as QG, qwen38_ref as R, qwen38_kernels as QK      # noqa: E402
import gdn_batch as GB, attn_batch as AB                                  # noqa: E402  (the batched one-token mixers: M rows a launch)
BATCHK = os.environ.get("BATCH_KERNELS", "1") != "0"                       # 0: the mixers per user on the one-row kernels (scatter / gather)
BLK = int(os.environ.get("BATCH_BLOCK", "4"))                              # layers a JIT graph / job (the runtime's layer blocks: their window slots), 1 = a layer a job
from qwen38_generate import poke, dev, DEV                               # noqa: E402

NT = 12


# ---- the scatter / gather kernels (12 tasks; plain loads: a few KB a user)
def rows_scatter_src(C, N):
  """out_u[0:C] = rows[u C + .] for u < N (the unpacked rows' row u into user u's row-0 buffer)."""
  outs = ", ".join(f"__global float* restrict o{u}" for u in range(N)); body = "".join(f" *(__global float8*)(o{u} + i) = *(__global float8*)(rows + {u} * {C} + i);" for u in range(N))
  return V.FULL_H + f"""
__kernel void rows_scatter({outs}, __global float* restrict rows, const int core_id) {{
  for (int i = core_id * 8; i < {C}; i += {8 * NT}) {{{body} }}
}}"""


def rows_gather_src(C, N):
  """out[u C + .] = in_u[0:C] (the users' row-0 outputs into the N-row buffer)."""
  ins = ", ".join(f"__global float* restrict i{u}" for u in range(N)); body = "".join(f" *(__global float8*)(out + {u} * {C} + i) = *(__global float8*)(i{u} + i);" for u in range(N))
  return V.FULL_H + f"""
__kernel void rows_gather(__global float* restrict out, {ins}, const int core_id) {{
  for (int i = core_id * 8; i < {C}; i += {8 * NT}) {{{body} }}
}}"""


def tile_rows_scatter_src(nblk, N):
  """The compact C tiles' row u into user u's tile buffer's row 0: a 192-float block holds rows (a, c) at a 64 + b 16 + c 4 + d
  (a: the 4-row tile, c: the row in it, b / d: the 16 columns); row u = (u / 4, u % 4) -> row 0 = (0, 0)."""
  outs = ", ".join(f"__global float* restrict o{u}" for u in range(N))
  body = "".join(f" *(__global float4*)(o{u} + base + {b * 16}) = *(__global float4*)(ct + base + {(u // 4) * 64 + b * 16 + (u % 4) * 4});" for u in range(N) for b in range(4))
  return V.FULL_H + f"""
__kernel void tile_rows_scatter({outs}, __global float* restrict ct, const int core_id) {{
  for (int k = core_id; k < {nblk}; k += {NT}) {{ int base = k * 192;{body} }}
}}"""


class Batch:
  def __init__(self, M, N, tmax):
    self.M, self.N, self.TMAX = M, N, tmax; L = self.L = M._ver(N); K = L.K; self.X = M.xbvs[N]; H = R.H
    C = self.C = 2 * R.NK * R.DK + R.NV * R.DV; self.CP = QK.ring_pitch(C); NG, NA = M.NGDN, M.NATT
    rep = lambda t, n: dev(np.tile(OA.host_invalidate(t).numpy().reshape(n, -1), (N, 1)).reshape(-1))        # a stack replicated per user
    self.Wab, self.Adt, self.Cwt, self.Nw = rep(M.Wab, NG), rep(M.Adt, NG), rep(M.Cwt, NG), rep(M.Nw, NG); self.QN, self.KN = rep(M.QN, NA), rep(M.KN, NA)
    self.Sall = Tensor.zeros(N * NG * R.NV * R.DK * R.DV, device=DEV).contiguous().realize(); self.Call = Tensor.zeros(N * NG * R.CONV * self.CP, device=DEV).contiguous().realize()
    self.kvs = M.kvs                                                       # KV slots (QWEN_KV_SLOTS=1, QWEN_KV_USERS >= N): the users' caches are the model's, in host memory behind two window slots
    if self.kvs: assert self.kvs.users >= N and self.kvs.tmax == tmax, (self.kvs.users, self.kvs.tmax, N, tmax); self.Kall = self.Vall = None
    else: ksz = (N * NA * tmax + QK.ATT_BT) * R.NKV * R.HD; self.Kall, self.Vall = (Tensor.zeros(ksz, device=DEV).contiguous().realize() for _ in range(2))
    self.ci = (lambda u, i: u) if self.kvs else (lambda u, i: u * NA + i)   # a user's cache index at attention layer i
    self.sidx = [[dev(np.array([u * NG + i], np.int32)) for i in range(NG)] for u in range(N)]; self.aix = [[dev(np.array([i, 0, self.ci(u, i)], np.int32)) for i in range(NA)] for u in range(N)]
    self.posb = [dev(np.array([0], np.int32)) for u in range(N)]; self.aix0 = [dev(np.array([i, 0], np.int32)) for i in range(NA)]
    self.K1 = [QK.Kernels(2, 24, R.EPS, real=1) for _ in range(N)]          # the users' one-row kernel sets (their own row-0 buffers)
    self.x1 = [k.rows_buf("x_in", H) for k in self.K1]; self.O = K.rows_buf("o_gather", max(R.NV * R.DV, R.NH * R.HD))
    self.ct_u = {}; self.jits = {}
    self.gtab = [dev(np.array([u * NG + i for u in range(N)], np.int32)) for i in range(NG)]            # the users' state indices a layer
    self.ptab = dev(np.zeros(N, np.int32)); self.atab = [dev(np.zeros(3 * N, np.int32)) for i in range(NA)]   # positions; [layer, pos, cache index] a user, a layer

  def tiles_for(self, name, n):
    if name not in self.ct_u: self.ct_u[name] = [k.buf(f"ct_{name}", n, dtypes.float32) for k in self.K1]
    return self.ct_u[name]

  def gdn_b(self, par, sidx, gtab, ptab, *views):
    """A DeltaNet layer on the N rows with the batched kernel: gtab (int32 [N]: the users' state indices), ptab [N] (their positions)."""
    M, L, K, N, H = self.M, self.L, self.L.K, self.N, R.H; X = self.X[par]; slot = dict(zip(M.GDN_LIN, views)); meta = self.meta_gdn
    si = K.hold("b_sidx", sidx); gt = K.hold("b_gtab", gtab); pt = K.hold("b_ptab", ptab)
    a_h = K.rms_a(X, (M.Win_g, si), H, "a_h"); ct = L.lin_cts(a_h, slot, meta, ("qkv", "z")); xoff, zoff = ct["qkv"][1], ct["z"][1]
    src = QK.ct_goff(GB.gdn_tokb_src(R.NV, R.NK, R.DK, R.DV, self.C, R.CONV, H, K.nrb, R.EPS, N), K.nrb, (xoff, zoff), ("x", "z"))
    O = K.rows_buf("o_gather", max(R.NV * R.DV, R.NH * R.HD))
    K.call(f"gdn_tokb|{N}|{xoff}|{zoff}", src, O, self.Sall, self.Call, gt, pt, ct["qkv"][0], ct["z"][0], X, self.Wab, self.Adt, self.Cwt, self.Nw, K.desc("gdn_tokb_desc", lambda: GB.gdn_tokb_desc(R.DK, R.DV, H, QK.gdn_hpt(R.NV), self.C, R.CONV)))
    a_o = K.rms_a(O, None, R.NV * R.DV, "a_o", norm=False)
    x1 = K.resid(X, L.lin_ct(a_o, slot, meta, "o"), H, "x1")
    return L.mlp(x1, slot, meta, (M.Wpost_g, si), f"xiov{1 - par}").realize()

  def kv(self, cp): return self.kvs.kv(cp) if self.kvs else (self.Kall, self.Vall)
  def att_b(self, par, cp, aix0, atab, *views):
    """An attention layer on the N rows with the batched kernel: atab (int32 [N][2]: the users' [cache index, position]); cp: the cache
    parity (KV slots: the slot the layer's caches are mapped in; the JIT is keyed by it)."""
    M, L, K, N, H = self.M, self.L, self.L.K, self.N, R.H; X = self.X[par]; slot = dict(zip(M.ATT_LIN, views)); meta = self.meta_att; Kall, Vall = self.kv(cp)
    a0 = K.hold("b_aix0", aix0); at = K.hold("b_atab", atab)
    a_h = K.rms_a(X, (M.Win_a, a0), H, "a_h"); ct = L.lin_cts(a_h, slot, meta, ("q", "kv")); QR, KR = R.NH * 2 * R.HD, 2 * R.NKV * R.HD
    q_rows = K.unpack(ct["q"][0], QR, "q_rows", goff=ct["q"][1]); kv_rows = K.unpack(ct["kv"][0], KR, "kv_rows", goff=ct["kv"][1])
    O = K.rows_buf("o_gather", max(R.NV * R.DV, R.NH * R.HD))
    K.call(f"attn_decb|{N}|{self.TMAX}|nidx", AB.attn_decb_src(R.NH, R.NKV, R.HD, self.TMAX, R.ROT, R.EPS, N, nidx=True), O, q_rows, kv_rows, self.QN, self.KN, M.ROPE, Kall, Vall, at, K.desc("attn_dec3_desc", lambda: __import__("attn_gqa").attn_dec3_desc(R.NH, R.NKV, R.HD)))
    a_o = K.rms_a(O, None, R.NH * R.HD, "a_o", norm=False)
    x1 = K.resid(X, L.lin_ct(a_o, slot, meta, "o"), H, "x1")
    return L.mlp(x1, slot, meta, (M.Wpost_a, a0), f"xiov{1 - par}").realize()

  def gdn(self, par, sidx, usidx, posb, *views):
    """A DeltaNet layer on the N rows: sidx (int32 [1]: the layer's index into the per-type stacks), usidx [N] (the users' indices into
    the replicated stacks / states), posb [N] (the users' positions), the layer's linears' views (M.GDN_LIN's order)."""
    M, L, K, N, H = self.M, self.L, self.L.K, self.N, R.H; X = self.X[par]; slot = dict(zip(M.GDN_LIN, views)); meta = self.meta_gdn
    si = K.hold("b_sidx", sidx); us = [self.K1[u].hold("sidx", usidx[u]) for u in range(N)]; pb = [self.K1[u].hold("posb", posb[u]) for u in range(N)]
    a_h = K.rms_a(X, (M.Win_g, si), H, "a_h"); ct = L.lin_cts(a_h, slot, meta, ("qkv", "z"))
    K.call(f"rows_scatter|{H}|{N}", rows_scatter_src(H, N), self.x1[0], *self.x1[1:], X)        # the users' input rows (gdn_tok3 projects a | b from them)
    tiles = {}
    for key, (ctb, goff) in ct.items():
      if id(ctb) in tiles: continue
      n = ctb.shape[0] if len(ctb.shape) == 1 else int(np.prod(ctb.shape)); nblk = n // 192; tu = self.tiles_for(f"{key}{n}", n); tiles[id(ctb)] = tu
      K.call(f"tile_rows_scatter|{nblk}|{N}", tile_rows_scatter_src(nblk, N), tu[0], *tu[1:], ctb)
    for u in range(N):
      k1 = self.K1[u]; o = k1.gdn_tok3(self.Sall, self.Call, us[u], pb[u], tiles[id(ct["qkv"][0])][u], tiles[id(ct["z"][0])][u], self.x1[u], self.Wab, self.Adt, self.Cwt, self.Nw,
                                       R.NV, R.NK, R.DK, R.DV, self.C, R.CONV, H, xoff=ct["qkv"][1], zoff=ct["z"][1], rows=True)
      outs = (outs if u else []) + [k1.bufs["o_rows"]]
    K.call(f"rows_gather|{R.NV * R.DV}|{N}", rows_gather_src(R.NV * R.DV, N), self.O, *outs)
    a_o = K.rms_a(self.O, None, R.NV * R.DV, "a_o", norm=False)
    x1 = K.resid(X, L.lin_ct(a_o, slot, meta, "o"), H, "x1")
    return L.mlp(x1, slot, meta, (M.Wpost_g, si), f"xiov{1 - par}").realize()

  def att(self, par, cp, aix0, uaix, *views):
    """An attention layer on the N rows: aix0 (int32 [2]: [the layer's attention index, 0]), uaix [N] ([user x NATT + index, the user's
    position]: the user's cache and norm stacks), the linears' views (M.ATT_LIN's order)."""
    M, L, K, N, H = self.M, self.L, self.L.K, self.N, R.H; X = self.X[par]; slot = dict(zip(M.ATT_LIN, views)); meta = self.meta_att
    a0 = K.hold("b_aix0", aix0); ua = [self.K1[u].hold("aix", uaix[u]) for u in range(N)]
    a_h = K.rms_a(X, (M.Win_a, a0), H, "a_h"); ct = L.lin_cts(a_h, slot, meta, ("q", "kv"))
    QR, KR = R.NH * 2 * R.HD, 2 * R.NKV * R.HD
    q_rows = K.unpack(ct["q"][0], QR, "q_rows", goff=ct["q"][1]); kv_rows = K.unpack(ct["kv"][0], KR, "kv_rows", goff=ct["kv"][1])
    q1 = [k.rows_buf("q_rows", QR) for k in self.K1]; kv1 = [k.rows_buf("kv_rows", KR) for k in self.K1]
    K.call(f"rows_scatter|{QR}|{N}", rows_scatter_src(QR, N), q1[0], *q1[1:], q_rows); K.call(f"rows_scatter|{KR}|{N}", rows_scatter_src(KR, N), kv1[0], *kv1[1:], kv_rows)
    outs = [self.K1[u].attn_dec2(q1[u], kv1[u], self.QN, self.KN, M.ROPE, *self.kv(cp), ua[u], R.NH, R.NKV, R.HD, self.TMAX, R.ROT, nidx=True) for u in range(N)]
    K.call(f"rows_gather|{R.NH * R.HD}|{N}", rows_gather_src(R.NH * R.HD, N), self.O, *[self.K1[u].bufs["o_rows"] for u in range(N)])
    a_o = K.rms_a(self.O, None, R.NH * R.HD, "a_o", norm=False)
    x1 = K.resid(X, L.lin_ct(a_o, slot, meta, "o"), H, "x1")
    return L.mlp(x1, slot, meta, (M.Wpost_a, a0), f"xiov{1 - par}").realize()

  def blk(self, types, bp, ptab, *args):
    """BLK consecutive layers as one graph: the users' positions once (ptab), then the layers' index buffers and views concatenated
    (gdn: sidx, gtab + M.GDN_LIN views; att: aix0, atab + M.ATT_LIN views), each layer on parity j % 2 (blocks start at even layers)."""
    M = self.M; a = list(args); out = None
    for j, t in enumerate(types):
      if t == "full": n = 2 + len(M.ATT_LIN); out = self.att_b(j % 2, bp, *a[:n]); a = a[n:]
      else: n = 2 + len(M.GDN_LIN); out = self.gdn_b(j % 2, a[0], a[1], ptab, *a[2:n]); a = a[n:]
    return out

  def step(self, toks, pos):
    """toks [N] at positions pos [N] -> the next token of each user."""
    M, L, N = self.M, self.L, self.N; t_in = time.perf_counter()
    x = np.zeros((L.R, R.H), np.float32); x[:N] = M.W.embed([int(t) for t in toks]); poke(self.X[0], x)
    poke(self.ptab, np.array(pos, np.int32))
    for i in range(M.NATT): poke(self.atab[i], np.array(sum(([i, pos[u], self.ci(u, i)] for u in range(N)), []), np.int32))
    for u in range(N):
      poke(self.posb[u], np.array([pos[u]], np.int32))
      for i in range(M.NATT): poke(self.aix[u][i], np.array([i, pos[u], self.ci(u, i)], np.int32))
    if BATCHK and BLK > 1 and R.NL % BLK == 0 and BLK % 2 == 0:            # layer blocks: BLK layers a job, each in a window slot of its own
      from qwen38_generate import one_job
      prof = os.environ.get("BATCH_PROF") == "1"; tp = [time.perf_counter()]
      if prof: from tinygrad import Device; Device[DEV].synchronize(); tp.append(time.perf_counter())
      L.block_slots(BLK)
      for b in range(R.NL // BLK):
        bp = b % 2; types = tuple(R.LAYER_TYPES[b * BLK + j] for j in range(BLK)); args = []
        for j in range(BLK):
          l = b * BLK + j; views, meta = L.map_block(l, bp, j); M._kvmap(l, bp)
          if types[j] == "full": i = M.aidx[l]; self.meta_att = meta; args += [self.aix0[i], self.atab[i], *[views[k] for k in M.ATT_LIN]]
          else: i = M.gidx[l]; self.meta_gdn = meta; args += [M.sidx[i], self.gtab[i], *[views[k] for k in M.GDN_LIN]]
        key = ("blk", bp, types, tuple(L.kind(b * BLK + j) for j in range(BLK)))
        if key not in self.jits: self.jits[key] = TinyJit(functools.partial(self.blk, types, bp))
        jit = self.jits[key]
        if jit.cnt < 2:
          with one_job(): jit(self.ptab, *args)
        else: jit(self.ptab, *args)
        if prof:
          Device[DEV].synchronize(); tp.append(time.perf_counter())
          if b == 0 and jit.cnt >= 2 and not getattr(self, "_prof_once", False):
            import collections; self._prof_once = True; names = collections.Counter(getattr(getattr(ei, "prg", None), "name", type(ei).__name__)[:28] for ei in jit.jit_cache)
            print(f"      prof: a block of {BLK} layers ({types}) = {len(jit.jit_cache)} launches: {dict(names)}", flush=True)
      ids, _ = M.head_top(self.X[R.NL % 2], L)
      if prof:
        Device[DEV].synchronize(); tp.append(time.perf_counter()); d = np.diff(tp) * 1e3
        print(f"      prof: host (embed + pokes) {d[0]:.1f} ms | blocks {' '.join(f'{v:.1f}' for v in d[1:-1])} ms | head {d[-1]:.1f} ms | total {(tp[-1] - tp[0]) * 1e3:.1f}", flush=True)
      return [int(v) for v in ids[:N]]
    L.prefetch(0); M._kvmap(0)
    for l in range(R.NL):
      if l + 1 < R.NL: L.prefetch(l + 1); M._kvmap(l + 1)
      slot, meta, sm = L.load(l); par = l % 2; kind = L.kind(l)
      if R.LAYER_TYPES[l] == "full":
        i = M.aidx[l]; self.meta_att = meta; cp = self.kvs.cp(i) if self.kvs else 0
        if BATCHK:
          if (key := ("attb", par, kind, cp)) not in self.jits: self.jits[key] = TinyJit(functools.partial(self.att_b, par, cp))
          self.jits[key](self.aix0[i], self.atab[i], *[slot[k] for k in M.ATT_LIN])
        else:
          if (key := ("att", par, kind, cp)) not in self.jits: self.jits[key] = TinyJit(functools.partial(self.att, par, cp))
          self.jits[key](self.aix0[i], [self.aix[u][i] for u in range(N)], *[slot[k] for k in M.ATT_LIN])
      else:
        i = M.gidx[l]; self.meta_gdn = meta
        if BATCHK:
          if (key := ("gdnb", par, kind)) not in self.jits: self.jits[key] = TinyJit(functools.partial(self.gdn_b, par))
          self.jits[key](M.sidx[i], self.gtab[i], self.ptab, *[slot[k] for k in M.GDN_LIN])
        else:
          if (key := ("gdn", par, kind)) not in self.jits: self.jits[key] = TinyJit(functools.partial(self.gdn, par))
          self.jits[key](M.sidx[i], [self.sidx[u][i] for u in range(N)], self.posb, *[slot[k] for k in M.GDN_LIN])
    ids, _ = M.head_top(self.X[R.NL % 2], L)
    return [int(v) for v in ids[:N]]


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--users", type=int, default=4); ap.add_argument("--max-new", type=int, default=32); ap.add_argument("--tmax", type=int, default=512)
  ap.add_argument("--check", default=None, help="a bonsai2_generate --out npz: user 0's ids must equal its out_ids"); ap.add_argument("--pin-gb", type=float, default=12); a = ap.parse_args()
  os.environ.setdefault("QWEN_KV_USERS", str(a.users))                     # KV slots: the model's caches hold every user
  from qwen38_tokenize import Tok, chat_text
  cache, dir_ = os.environ["QWEN_NPU"], os.environ["QWEN_DIR"]; tok = Tok(os.environ.get("QWEN_TOK", cache))
  prompts = ["The capital of France is", "Write a Python function that returns the n-th Fibonacci number.", "Explain in two sentences why the sky is blue.", "List three uses of copper.",
             "What is the boiling point of water in Kelvin?", "Name the largest planet of the solar system.", "Translate 'good morning' into Spanish.", "Who wrote the Odyssey?"]
  prompts = (prompts * (a.users // len(prompts) + 1))[:a.users]; ids = [tok.encode(chat_text(p)) for p in prompts]; n = min(len(v) for v in ids); ids = [v[-n:] if len(v) > n else v for v in ids]   # one length: the tail (the template's end kept)
  ids[0] = tok.encode(chat_text(prompts[0]))[:n] if len(tok.encode(chat_text(prompts[0]))) >= n else ids[0]
  t0 = time.perf_counter(); M = QG.Model(1, cache, dir_, a.tmax, a.pin_gb); B = Batch(M, a.users, a.tmax); print(f"   model set up in {time.perf_counter() - t0:.0f} s; {a.users} users, prompts of {n} tokens", flush=True)
  N = a.users; out = [[] for _ in range(N)]; pos = [0] * N
  t0 = time.perf_counter()
  for j in range(n):                                                      # the prompts in lockstep, a token a step
    nxt = B.step([ids[u][j] for u in range(N)], pos); pos = [p + 1 for p in pos]
  tp = time.perf_counter() - t0; print(f"   prefill {n} tokens x {N} users: {tp:.1f} s ({n * N / tp:.1f} tok/s aggregate)", flush=True)
  for u in range(N): out[u].append(nxt[u])
  t0 = time.perf_counter(); steps = []
  for j in range(a.max_new - 1):
    t1 = time.perf_counter(); nxt = B.step([out[u][-1] for u in range(N)], pos); steps.append(time.perf_counter() - t1); pos = [p + 1 for p in pos]
    for u in range(N): out[u].append(nxt[u])
  td = time.perf_counter() - t0
  for u in range(N): print(f"   user {u}: {tok.decode(out[u])!r}", flush=True)
  print(f"   decode {a.max_new - 1} steps x {N} users: {td:.1f} s -> {np.median(steps) * 1e3:.0f} ms a step (median), {N / np.median(steps):.1f} tok/s aggregate, {1 / np.median(steps):.2f} a user", flush=True)
  if a.check:
    ref = np.load(a.check)["out_ids"].tolist(); m = min(len(ref), len(out[0])); print("   user 0 vs the single-request run:", "identical" if out[0][:m] == ref[:m] else f"DIFFER at {next(j for j in range(m) if out[0][j] != ref[j])}", f"({m} tokens)")
  np.savez(os.path.expanduser(f"~/bonsai_batch_{N}.npz"), out_ids=np.array(out), prompt_ids=np.array(ids))


if __name__ == "__main__": main()

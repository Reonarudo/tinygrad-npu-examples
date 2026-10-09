#!/usr/bin/env python3
"""Qwen3.8-27B text generation on the NPU: prompt -> chat template -> token ids -> prefill through the 64 layers (the Gated
DeltaNet states and the attention layers' K / V kept on the device) -> greedy decode one token at a time -> text.

    python3 qwen38_generate.py "What is the capital of Portugal?" [--max-new 32] [--thinking] [--tmax 512]
        [--dir /mnt/ssd/qwen3.8-27b-fp8] [--cache /mnt/ssd/qwen3.8-27b-npu]
Every token streams the 64 layers' weights (~26 GB) from the page cache into the device slots, so decode is seconds per token.
The decode step of each layer type runs under TinyJit (one per weight-slot parity): the K / V caches have a fixed length
`--tmax` so the graphs keep their shapes.
"""
import argparse, ctypes, functools, json, math, os, sys, threading, time
import numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad")))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# a layer's launches as one job (the backend's default 8 makes two): Qwen3.8-27B's 132 jobs a verify pass -> 68 (bonsai2 / ornith set it too)
os.environ.setdefault("ZHOUYI_CHAIN_MAX", "16")
from tinygrad import Tensor, dtypes                                      # noqa: E402
from tinygrad.engine.jit import TinyJit                                  # noqa: E402
from zy import OA                                                        # noqa: E402
import qwen38_ref as R                                                   # noqa: E402
from qwen38_kernels import ATT_BT, ring_pitch                            # noqa: E402
import qwen38_kernels as QK                                              # noqa: E402
from qwen38_npu import Layers, dev, poke, DEV, read_into, memmove_threads, GDN_TOK, KS, Q8MODE, SCALES, TERN_KW, slot_map_nodrain   # noqa: E402
from qwen38_tokenize import Tok, chat_text, EOS                         # noqa: E402

def one_job():
  """A context for the TinyJit call that captures: its kernels in one graph (JIT_BATCH_SIZE=0) and one chain (the zhouyi backend's
  ZHOUYI_CHAIN_CAP), i.e. one job however many launches; an older backend without the override keeps its ZHOUYI_CHAIN_MAX."""
  from tinygrad.helpers import Context, ContextVar
  return Context(**{k: v for k, v in (("JIT_BATCH_SIZE", 0), ("ZHOUYI_CHAIN_CAP", 4096)) if k in ContextVar._cache})

def draft_parts(v):
  """QWEN_DRAFT_PARTS = <n>[,thresh:<x>] -> (n, x or None, the parts policy as the spec logs' header records it: "all" | "thresh:<x>").
  <n>: the draft head reads the head's first n parts; thresh:<x>: parts 1.. only when part 0's top logit on the row drafted
  from is <= x (without it every one of the n parts is read)."""
  n, *rest = [f.strip() for f in v.split(",")]
  assert len(rest) <= 1 and all(r.startswith("thresh:") for r in rest), f"QWEN_DRAFT_PARTS={v!r}: <n>[,thresh:<x>]"
  return int(n), (float(rest[0].split(":", 1)[1]) if rest else None), (rest[0] if rest else "all")
import spec_tree                                                         # noqa: E402

def combine_tops(tops, c0s, top3=False, need_feats=False):
  """The head's partials (per part read: [NT, m, 4] of head_top, or [NT, m, 8] of head_top3; `c0s` the parts' first columns)
  -> (ids [m], probabilities [m] = 1 / the sum of exp over the parts read, ptop [parts, m] each part's top logit,
  feats = (top logit, 2nd-largest (part, task) maximum) [m] or None, t3 = per row per part ([3 ids], [3 logits]) or None).
  The argmax is the first (part, task) holding the largest max and that task's column: the smallest id among ties."""
  t = np.stack(tops); m, P = t.shape[2], t.shape[0]; se = 6 if top3 else 2   # [parts, NT, m, W]: (max, column, ..., sum exp(. - max))
  mx = t[..., 0].max((0, 1)); first = (t[..., 0] == mx).reshape(-1, m).argmax(0)   # the first (part, task) holding the max
  feats = None
  if need_feats:                                                          # telemetry: the top logit and the
    tm = np.sort(t[..., 0].reshape(-1, m), 0)                             # 2nd-largest (part, task) maximum (a lower bound on the runner-up)
    feats = (mx, tm[-2] if tm.shape[0] > 1 else np.full(m, -np.inf))
  col = t[..., 1].reshape(-1, m)[first, np.arange(m)].astype(np.int64)
  c0 = np.array(c0s)[first // t.shape[1]]
  pr = 1.0 / (t[..., se] * np.exp(t[..., 0] - mx)).sum((0, 1))
  t3 = None
  if top3:
    lg, cs = t[..., 0:6:2], t[..., 1:6:2]                                  # [parts, NT, m, 3]: the tasks' candidates
    t3 = []
    for r in range(m):
      row = []
      for p in range(P):
        l = lg[p, :, r].ravel(); c = cs[p, :, r].ravel().astype(np.int64) + c0s[p]; o = np.lexsort((c, -l))[:3]
        row.append(([int(v) for v in c[o]], [float(v) for v in l[o]]))
      t3.append(row)
  return c0 + col, pr, t[..., 0].max(1), feats, t3

def top3_rank(t3, g):
  """The rank (0..2) of token g in a draft's merged top-3 (`t3`: per part ([ids], [logits])), -1 when absent."""
  cand = sorted(((-l, i) for ids, lgs in t3 for i, l in zip(ids, lgs)))[:3]
  return next((k for k, (_, i) in enumerate(cand) if i == g), -1)

def read_into_at(path, va, off, size, nt=8):
  """`size` bytes of the file from `off` into the device mapping at `va`, by parallel preadv's."""
  import threading
  fd = os.open(path, os.O_RDONLY); mv = memoryview((ctypes.c_uint8 * size).from_address(va))
  def part(lo, hi):
    while lo < hi: lo += os.preadv(fd, [mv[lo:hi]], off + lo)
  b = [(size * i // nt) & ~4095 for i in range(nt)] + [size]
  ths = [threading.Thread(target=part, args=(b[i], b[i + 1])) for i in range(nt)]
  for t in ths: t.start()
  for t in ths: t.join()
  os.close(fd)

def rope_tables(pos0, Rr):
  pos = np.arange(pos0, pos0 + Rr, dtype=np.float64); inv = 1.0 / (R.THETA ** (np.arange(0, R.ROT, 2, dtype=np.float64) / R.ROT))
  f = pos[:, None] * inv[None]; emb = np.concatenate([f, f], -1)
  return dev(np.cos(emb).astype(np.float32).reshape(Rr, 1, R.ROT)), dev(np.sin(emb).astype(np.float32).reshape(Rr, 1, R.ROT))

# QWEN_ATTN_PREFILL: the full-attention prefill's attention (default csrc). The tinygrad attention below
# (`none`, or empty) hung silently on the board (2026-10-03: the score kernel r_2_*_3_3_3_64_4, 4 tasks on one core; its q / k rows
# and the softmax-V kernel's q / v / o rows sit at pitches that are multiples of 8 KiB -- a data-cache set-conflict hazard).
#   csrc  the attention by attn_pkv / attn_ppart / attn_pcomb (Kernels.attn_prefill): every row by DMA, no generated kernel; the
#         k / v rows go into the cache from the kernel (prefill's cache copy is skipped)
ATT_PRE = os.environ.get("QWEN_ATTN_PREFILL", "csrc") or "none"
NPU_IDS = os.environ.get("QWEN_NPU_IDS", "0") == "1"                     # the token ids chosen, embedded and accepted on the NPU (head_reduce, ...)
# The decode loop's jobs and host work:
#   QWEN_RING_COMMIT (1): the deferred commit's copies by ring_commit (one kernel for every geometry, the same bytes as gdn_commit_tree(defer) /
#                         gdn_commit(ring_only))
#   QWEN_COMMIT_FOLD (1): that commit runs inside the next draft pass's job instead of a job of its own
#   QWEN_DRAFT_JOB (1):   a draft pass (the input rows, the MTP layer, the draft head's parts) as one job
#   QWEN_STREAM_LATE (1): a streamed layer's staging refill and prefetch issued after the next layer's submit, not behind a sync()
#   QWEN_MTPIN (dma):     the device's MTP input rows by mtpin_d_src (12 tasks, DMA) | task (mtpin_src, one task a row); the same bits
RING_COMMIT = os.environ.get("QWEN_RING_COMMIT", "1") == "1"
COMMIT_FOLD = os.environ.get("QWEN_COMMIT_FOLD", "1") == "1"
STREAM_LATE = os.environ.get("QWEN_STREAM_LATE", "1") == "1"          # the verify pass's streamed layers: Layers.after_submit (no sync in load)
DRAFT_JOB = os.environ.get("QWEN_DRAFT_JOB", "1") == "1"
MTPIN_DMA = os.environ.get("QWEN_MTPIN", "dma") == "dma"
assert ATT_PRE in ("none", "csrc"), f"QWEN_ATTN_PREFILL={ATT_PRE}: none | csrc"

def qkv_proj(L, x, slot, meta, sm, cos, sin, xin=None):
  """RMSNorm -> q (24 heads x [256 | gate]), k, v (4 heads) with the q / k norms and the partial RoPE (`xin`: x already in a persistent buffer)."""
  Rr = L.R; x = xin if xin is not None else L.K.rows_buf("x_in", R.H).assign(x).realize(); a_h = L.K.rms_a(x, L.K.hold("w_in", sm["input_layernorm_weight"]), R.H, "a_h")
  QC, KC = R.NH * 2 * R.HD, 2 * R.NKV * R.HD
  ct = L.lin_cts(a_h, slot, meta, ("q", "kv"))                               # one GEMM when the cache fuses q | k|v
  qg = L.K.unpack(ct["q"][0], QC, "q_rows", goff=ct["q"][1]).reshape(Rr, R.NH, 2 * R.HD)
  kv = L.K.unpack(ct["kv"][0], KC, "kv_rows", goff=ct["kv"][1])
  q, gate = qg[..., :R.HD], qg[..., R.HD:]
  k = kv[:, :R.NKV * R.HD].reshape(Rr, R.NKV, R.HD); v = kv[:, R.NKV * R.HD:].reshape(Rr, R.NKV, R.HD)
  q = L.rms(q, sm["self_attn_q_norm_weight"]); k = L.rms(k, sm["self_attn_k_norm_weight"])
  def rope(t):
    tr, tp = t[..., :R.ROT], t[..., R.ROT:]; rot = Tensor.cat(-tr[..., R.ROT // 2:], tr[..., :R.ROT // 2], dim=-1)
    return Tensor.cat(tr * cos + rot * sin, tp, dim=-1)
  q, k = rope(q), rope(k)
  return q, k, v, gate

def attend(L, x, q, gate, K_, V_, mask, slot, meta, sm):
  """Scores over the cached K / V (T rows), the sigmoid gate, o_proj, the MLP; returns the layer's output."""
  Rr, T, g = L.R, int(K_.shape[0]), R.NH // R.NKV
  kr = K_.reshape(T, R.NKV, 1, R.HD).expand(T, R.NKV, g, R.HD).reshape(T, R.NH, R.HD).permute(1, 2, 0)       # [NH, HD, T]
  vr = V_.reshape(T, R.NKV, 1, R.HD).expand(T, R.NKV, g, R.HD).reshape(T, R.NH, R.HD).permute(1, 0, 2)       # [NH, T, HD]
  s = q.permute(1, 0, 2) @ kr * (1.0 / math.sqrt(R.HD)) + mask
  o = (s.softmax(-1) @ vr).permute(1, 0, 2) * gate.sigmoid()
  OC = R.NH * R.HD
  oc = L.K.rows_buf("o_rows", OC).assign(o.reshape(Rr, OC)).realize(); a_o = L.K.rms_a(oc, None, OC, "a_o", norm=False)
  x = L.K.resid(x, L.lin_ct(a_o, slot, meta, "o"), R.H, "x1")
  return L.mlp(x, slot, meta, sm["post_attention_layernorm_weight"])

class Model:
  """The two row geometries share the weight slots: `pre` for the prompt (R = prompt rows), `dec` for one token (R = 24)."""
  GDN_SMALL = ["input_layernorm_weight", "post_attention_layernorm_weight", "linear_attn_A_log", "linear_attn_dt_bias", "linear_attn_conv1d_weight",
               "linear_attn_norm_weight", "linear_attn_in_proj_ab_weight", "linear_attn_conv1d_weight_t"]
  ATT_SMALL = ["input_layernorm_weight", "post_attention_layernorm_weight", "self_attn_q_norm_weight", "self_attn_k_norm_weight"]
  # GDN_LIN / ATT_LIN: a layer type's stream files, the JIT inputs of its step (the cache's: ["qkv", "z", "o", "gu", "dn"] and
  # ["q", "kv", "o", "gu", "dn"], or with the fused linears ["qkvz", "o", "gu", "dn"] and ["qkv", "o", "gu", "dn"]; QWEN_FUSE)

  def __init__(self, n_prompt, cache, dir_, tmax=512, pin_gb=24.0):
    self.cache, self.W, self.TMAX = cache, R.Weights(dir_), tmax
    # QWEN_DUMP_HIDDEN=path: the last layer's output (before the final norm) at every position, saved with the ids (the MTP
    # head's input, from which its draft acceptance can be measured)
    self.dump = [] if os.environ.get("QWEN_DUMP_HIDDEN") else None
    self.pre = Layers(n_prompt, cache); self.dec = Layers(1, cache); self.dec.slots = self.pre.slots; self.dec._inflight = self.pre._inflight
    self.GDN_LIN = Layers.files(self.pre._meta(R.LAYER_TYPES.index("linear"))[0]); self.ATT_LIN = Layers.files(self.pre._meta(R.LAYER_TYPES.index("full"))[0])
    if os.environ.get("QWEN_DEBUG"): print(f"   linears: DeltaNet {self.GDN_LIN}, attention {self.ATT_LIN}", flush=True)
    self.pre.pin(pin_gb); self.dec.pinned, self.dec._small_dev, self.dec.stage, self.dec.zc = self.pre.pinned, self.pre._small_dev, self.pre.stage, self.pre.zc
    z = np.load(os.path.join(cache, "outside_small.npz")); self.norm_w = dev(z["norm"]); self.N, self.npad, self.K = (int(v) for v in z["lm_head_n"])
    self.lm_path = os.path.join(cache, "lm_head.bin"); self.lm_size = os.path.getsize(self.lm_path) if os.path.exists(self.lm_path) else 0
    # QWEN_HEAD: fp8 (default when packed: E4M3 with 128 x 128 block scales like every other linear of the checkpoint, zero-copy, the
    # decoders' GEMM; measured on 41 decode steps: the same argmax every step, logits within ~1.5 %) | fp16 (the bf16 head, exact to
    # fp16) | both (runs both, prints the comparison, fp16 drives)
    have8 = bool(self.pre.zc) and os.path.exists(os.path.join(cache, f"{R.HEAD}.npz"))                 # the block-scaled head (Q8_0 for ornith-9b)
    self.head = os.environ.get("QWEN_HEAD", "fp8" if have8 else "fp16")
    if self.head != "fp16" and not have8:
      print(f"   QWEN_HEAD={self.head}: needs zero-copy and lm_head_fp8 (qwen38_pack.py --lm-head-fp8); using fp16", flush=True); self.head = "fp16"
    self.lm_pin = np.empty(self.lm_size if self.head != "fp8" else 0, np.uint8)
    if self.head != "fp8": read_into(self.lm_path, memoryview(self.lm_pin), self.lm_size)   # 2.5 GB held in RAM
    if self.head != "fp16": self._head_fp8_init(cache)
    self.LMG = 256                                                        # groups (24576 rows) a chunk: the head streams through two ~255 MB slots,
    self.gbytes = 6 * 24 * 128 * (self.K // 96)                           # one copied while the other's GEMM runs; fp16 panel bytes per group
    self.lm_slots = []                                                    # the fp16 head's two ~255 MB staging slots: only when it can run
    if self.head != "fp8":
      self.lm_slots = [Tensor.empty(self.LMG * self.gbytes, device=DEV, dtype=dtypes.uint8) for _ in range(2)]
      for t in self.lm_slots: t.uop.buffer.ensure_allocated()
    self.states = [None] * 64; self.kv = [None] * 64                     # per layer: conv_state / (K [TMAX, 4, 256], V [TMAX, 4, 256])
    self.NGDN = sum(1 for t in R.LAYER_TYPES if t == "linear"); self.gidx = {l: i for i, l in enumerate(l for l in range(R.NL) if R.LAYER_TYPES[l] == "linear")}
    self.NATT = R.NL - self.NGDN; self.aidx = {l: i for i, l in enumerate(l for l in range(R.NL) if R.LAYER_TYPES[l] == "full")}
    # QWEN_MTP_CACHE: the cache holding the MTP head, packed as layer NL (default: the model's own cache, if its checkpoint has one).
    # bonsai2-27b has none: its drafter is Qwen3.8-27B's MTP layer, an E4M3 cache of its own (default the subfolder `mtp` of the model's
    # cache: bonsai2_pack.py --mtp; or a full Qwen3.8-27B cache packed with --scales single), run on its own fp8 geometries
    # (Layers(fp8=True): no ternary GEMM, no Hadamard) and fed Bonsai's hidden states (bonsai2-27b/README.md)
    self.mtp_cache = os.path.expanduser(os.environ.get("QWEN_MTP_CACHE", os.path.join(cache, "mtp") if R.HAD else cache))
    self.mtp_x = os.path.realpath(self.mtp_cache) != os.path.realpath(cache)                  # the drafter from another model's cache
    have_mtp = os.path.exists(os.path.join(self.mtp_cache, f"L{R.NL}_small.npz"))
    # QWEN_SPEC_TREE=off (default) | rescue2 | fixed:<j1,j2> (spec_tree.py): the tree verify -- always TREE_M = 8 rows (the 6-row chain
    # + 2 rescue rows, the rank-2 siblings), its own kernels (gdn_tokt / attn_partt / gdn_commit_tree); off leaves every default path as it was
    # QWEN_SPEC_TREE=leaf (any model): the leaf tree (spec_tree.build_leaf) -- rows up to QWEN_SPEC, the deferred commit's DeltaNet
    # (gdn_tokl: gdn_fast_src(tree=True), o_proj's A straight from it on the models without a rotated input), any row count a pass;
    # rescue2 / fixed (not the rotated-input models): the 8-row bank-path tree (gdn_tokt has no fp32-rows output)
    self.tree = spec_tree.mode_from_env(os.environ); self.leaf = bool(self.tree) and self.tree[0] == "leaf"
    assert not self.tree or self.leaf or not R.HAD, f"QWEN_SPEC_TREE={os.environ.get('QWEN_SPEC_TREE')} with QWEN_MODEL={R.MODEL}: rescue2 / fixed are the bank-path tree, not the rotated-input models'"
    spec_default = "4" if self.leaf else str(spec_tree.TREE_M) if self.tree else ("6" if R.Q8 else "0")
    self.MTPL = R.NL if have_mtp and os.environ.get("QWEN_SPEC", spec_default) != "0" and os.environ.get("QWEN_DRAFT", "mtp") == "mtp" else None
    natt = self.NATT + (1 if self.MTPL else 0)                                               # the MTP head is attention index NATT
    self.mtpg = {1: Layers(1, self.mtp_cache, fp8=True)} if self.MTPL and self.mtp_x else {}   # the MTP layer's fp8 geometries (rows -> Layers)
    self.mtp_src = self.mtpg[1] if self.mtpg else self.dec                                    # whose _meta(MTPL) is the MTP layer's
    # (+ ATT_BT rows: attn_part's last DMA block of the last layer may run past its final row)
    ksz = (natt * tmax + ATT_BT) * R.NKV * R.HD
    K = self.dec.K; self.Kall = K.buf("K_all", ksz, dtypes.float32); self.Vall = K.buf("V_all", ksz, dtypes.float32)
    self.aix = [K.buf(f"aidx{i}", 2, dtypes.int32) for i in range(natt)]                   # per attention layer: (layer, pos), pos set per step
    # the fused decode attention's constants: q / k norm weights (1 + w) per attention layer, cos | sin per position
    qn, kn = np.zeros((natt, R.HD), np.float32), np.zeros((natt, R.HD), np.float32)
    for l, i in list(self.aidx.items()) + ([(self.MTPL, self.NATT)] if self.MTPL else []):
      _, small = (self.mtp_src if l == self.MTPL else self.dec)._meta(l); qn[i] = 1.0 + small["self_attn_q_norm_weight"]; kn[i] = 1.0 + small["self_attn_k_norm_weight"]
    self.QN = K.buf("qn_all", qn.size, dtypes.float32).assign(dev(qn.ravel())).realize(); self.KN = K.buf("kn_all", kn.size, dtypes.float32).assign(dev(kn.ravel())).realize()
    inv = 1.0 / (R.THETA ** (np.arange(0, R.ROT, 2, dtype=np.float64) / R.ROT)); f = np.arange(tmax, dtype=np.float64)[:, None] * inv[None]; emb = np.concatenate([f, f], -1)
    self.ROPE = K.buf("rope_tab", tmax * 2 * R.ROT, dtypes.float32).assign(dev(np.concatenate([np.cos(emb), np.sin(emb)], -1).astype(np.float32).ravel())).realize()
    self.Sall = K.buf("S_all", self.NGDN * R.NV * R.DK * R.DV, dtypes.float32)             # the 48 DeltaNet states, one persistent stack (151 MB)
    self.sidx = [K.buf(f"sidx{i}", 1, dtypes.int32) for i in range(self.NGDN)]            # each layer's index into it
    for i, t in enumerate(self.sidx): t.assign(Tensor([i], device=DEV, dtype=dtypes.int32)).realize()
    self.C = 2 * R.NK * R.DK + R.NV * R.DV; self.CP = ring_pitch(self.C)          # a raw qkv row; the row pitch of the rings, taps, raw rows
    self.Call = K.buf("C_all", self.NGDN * R.CONV * self.CP, dtypes.float32)               # the conv rings: the last CONV-1 raw qkv rows per layer, [L][CONV][CP]
    self.posb = K.buf("pos_b", 1, dtypes.int32)                                             # the decode step's position (shared by the layers)
    szab, szc = Layers.ab3_size(), R.CONV * self.CP                                              # the DeltaNet layers' small weights, stacked (94 MB): no per-layer copies
    self.Wab, self.Cwt, self.Nw = K.buf("Wab_all", self.NGDN * szab, dtypes.float32), K.buf("Cwt_all", self.NGDN * szc, dtypes.float32), K.buf("Nw_all", self.NGDN * R.DV, dtypes.float32)
    self.Adt = K.buf("Adt_all", self.NGDN * 2 * R.NV, dtypes.float32)                        # gdn_tok3: -exp(A_log) | dt_bias per layer
    # the RMSNorms' 1 + w per layer type (indexed by the layer's sidx / aix): the decode layers read them in place, no copy kernel
    self.Win_g, self.Wpost_g = K.buf("Win_gdn", self.NGDN * R.H, dtypes.float32), K.buf("Wpost_gdn", self.NGDN * R.H, dtypes.float32)
    self.Win_a, self.Wpost_a = K.buf("Win_att", natt * R.H, dtypes.float32), K.buf("Wpost_att", natt * R.H, dtypes.float32)
    for l, g in self.gidx.items():
      _, small = self.dec._meta(l)
      wab = small["linear_attn_in_proj_ab_weight"]; wab = Layers.ab3(wab) if GDN_TOK == "3" else wab
      for stack, sz, v in ((self.Wab, szab, wab), (self.Cwt, szc, np.pad(small["linear_attn_conv1d_weight_t"], ((0, 0), (0, self.CP - self.C)))), (self.Nw, R.DV, small["linear_attn_norm_weight"]),
                           (self.Adt, 2 * R.NV, Layers.adt(small)), (self.Win_g, R.H, small["input_layernorm_weight"]), (self.Wpost_g, R.H, small["post_attention_layernorm_weight"])):
        stack[g * sz:(g + 1) * sz].assign(dev(np.ascontiguousarray(v).reshape(-1))).realize()
    for l, i in list(self.aidx.items()) + ([(self.MTPL, self.NATT)] if self.MTPL else []):
      _, small = (self.mtp_src if l == self.MTPL else self.dec)._meta(l)
      for stack, key in ((self.Win_a, "input_layernorm_weight"), (self.Wpost_a, "post_attention_layernorm_weight")):
        stack[i * R.H:(i + 1) * R.H].assign(dev(np.ascontiguousarray(small[key]).reshape(-1))).realize()
    self.xb = [K.rows_buf("xio0", R.H), K.rows_buf("xio1", R.H)]                             # decode: the residual stream ping-pongs between two persistent buffers
    self.jit_gdn, self.jit_att = [None, None], [None, None]
    self.cos_dec, self.sin_dec = {}, {}
    # speculative decoding: M rows (the token + M - 1 drafts) a verify pass; the Q8_0 models prefill through the same geometries
    self.M = int(os.environ.get("QWEN_SPEC", spec_default))
    if R.Q8 and not self.M:                # QWEN_SPEC=0 on a Q8_0 model: plain decoding (no drafter: MTPL is None), but its prompt is
      self.M = int(spec_default)           # prefilled through the verify geometries (prefill_chunked), so they are made all the same
      print(f"   QWEN_SPEC=0 ({R.MODEL}): plain decoding; the prompt prefilled through the {self.M}-row verify geometries", flush=True)
    if self.tree and not self.leaf and self.M != spec_tree.TREE_M:
      print(f"   QWEN_SPEC_TREE={os.environ['QWEN_SPEC_TREE']}: the tree verify runs {spec_tree.TREE_M} rows (QWEN_SPEC={self.M} ignored)", flush=True); self.M = spec_tree.TREE_M
    # QWEN_SPEC_LOG=path.jsonl: a line per verify pass (the drafts' probabilities and head features, which were accepted, the
    # geometry, host-side wall times at the existing sync points) appended to the file
    self.spec_log, self.spec_meta = os.environ.get("QWEN_SPEC_LOG") or None, {}
    if self.M: self._spec_init(self.M)

  # ---- speculative decoding: the verify pass (M consecutive tokens through the 64 layers, one streaming of the weights)
  def _spec_init(self, M):
    assert 2 <= M <= 12 and self.head == "fp8", "QWEN_SPEC: 2..12 rows (the GEMMs' rows mode), the fp8 head"
    self.vers, self.xbvs = {}, {}; self.ver = self._ver(M)                   # the verify geometries (rows = tokens), made on demand
    K = self.ver.K; SZ = R.NV * R.DK * R.DV
    # QWEN_GDN_DEFER (default on): the DeltaNet commit deferred to the next verify pass. The chain: gdn_fast_src (any M <= 8;
    # QWEN_GDN_FAST=off: gdn_defer_src, M <= 4, the rotated-input models' fp32 rows only -- the others then take the bank path,
    # gdn_tokm_src); the leaf tree: gdn_fast_src(tree=True) (M <= 8; QWEN_GDN_FAST=off: gdn_tokl_src, M <= 5). No per-token banks
    # (1 GB at M = 4 on the 27B): `banks` holds each head's update inputs instead ([NG][NV][M] slots of GD_SLOT floats; +
    # gdn_fast_scratch, a task's v rows when the fast kernel keeps half rows). QWEN_GDN_DEFER=0 or QWEN_GDN_FAST=off: the
    # per-token banks and the full commit (gdn_tokm_src + gdn_commit), as before the port, for the models without a rotated input
    fast = QK.gdn_fast() != "off"
    self.defer = (R.HAD or fast or self.leaf) and (not self.tree or self.leaf) and M <= (QK.GF_MS if fast else 5 if self.leaf else 4) and os.environ.get("QWEN_GDN_DEFER", "1") == "1"
    self.pend = False                                                     # (defer: a verify pass's accepted updates not yet in Sall; step() flushes them)
    assert self.defer or not self.leaf, f"QWEN_SPEC_TREE=leaf: the deferred DeltaNet commit (QWEN_GDN_DEFER=1), QWEN_SPEC <= {QK.GF_MS if fast else 5}"
    if self.defer: self.banks = K.buf("gdn_kvb", self.NGDN * R.NV * M * QK.GD_SLOT + (QK.gdn_fast_scratch(M) if fast else 0), dtypes.float32)
    else: self.banks = K.buf("gdn_banks", (M - 1) * self.NGDN * SZ, dtypes.float32)      # the state after tokens 0..M-2, every DeltaNet layer
    self.rawm = K.buf("gdn_raw", self.NGDN * M * self.CP, dtypes.float32)             # the M raw conv rows, every DeltaNet layer: [NG][M][CP]
    self.accb = K.buf("acc_b", 1, dtypes.int32)
    self.jit_vgdn, self.jit_vatt, self.jit_head, self.jit_d1 = {}, {}, {}, {}
    self.pcommit = None                                                   # a commit left for the next draft pass's job (commit(fold=True))
    # adaptive draft depth: after the first draft, the chain goes on only while the draft head's probability of its draft is
    # >= QWEN_DRAFT_TAU; a verify pass then takes 1 + the drafts' rows (its own geometry)
    self.tau = float(os.environ.get("QWEN_DRAFT_TAU", "0.6"))
    # (QWEN_DRAFT_TAU=0: always QWEN_DRAFT_MAX drafts.) Only the number of drafts changes: the verify pass decides the tokens.
    # QWEN_DRAFT_MAX: drafts a pass at most. The Q8_0 GEMM is compute-bound, so a verify pass past 4 rows (a second row tile)
    # costs ~1.35x one of 2-4 (ornith-9b: 0.88 vs 0.60-0.69 s) and the 4th / 5th drafts do not pay for it: 3 (2.7-3.8 tok/s on
    # three prompts vs 2.4-3.3 with 5). The E4M3 models' GEMM streams at the DDR's rate whatever the rows: M - 1.
    self.kmax = min(M - 1, int(os.environ.get("QWEN_DRAFT_MAX", "3" if R.Q8 else str(M - 1))))
    # QWEN_SPEC_GEOS: the verify geometries (row counts) kept; a pass's 1 + drafts rounds up to the next (padded with the last
    # draft). Rows 5.. cost nearly what 6 do (the second row tile), so 3 / 4 / 6 is within 1 % of every count (simulated).
    self.geos = sorted({int(g) for g in os.environ.get("QWEN_SPEC_GEOS", "3,4,6").split(",") if 2 <= int(g) <= M} | {M})
    if self.tree:                                                         # the tree verify: every pass TREE_M rows (the chain's 6 + the rescue rows)
      if os.environ.get("QWEN_SPEC_GEOS"): print(f"   QWEN_SPEC_TREE: the verify geometry is {M} rows alone (QWEN_SPEC_GEOS ignored)", flush=True)
      if self.leaf:                                                       # the leaf tree: every row count 2..M (QWEN_SPEC_GEOS ignored)
        self.geos = list(range(2, M + 1)); self.NC = M
        # QWEN_TREE_TC: the drafter drafts on while the chain's path probability p_1 ... p_j >= it (QWEN_DRAFT_TAU unused);
        # QWEN_TREE_TL: a leaf candidate's least path probability (spec_tree.leaf_pick). Defaults: the best of a host-side replay at 4-5 rows
        self.tc, self.tl = float(os.environ.get("QWEN_TREE_TC", "0.4")), float(os.environ.get("QWEN_TREE_TL", "0.05"))
      else: self.geos = [M]; self.NC = M - spec_tree.TREE_NR; self.kmax = min(self.kmax, self.NC - 1)
      self.treeb = K.buf("tree_b", 2 * M, dtypes.int32); self.pathb = K.buf("path_b", 4 + M, dtypes.int32)   # spec_tree.table / path_words
      if self.leaf: poke(self.pathb, np.zeros(4 + M, np.int32))           # (gdn_tokl: the previous pass's path, none yet)
    # ring_commit (QWEN_RING_COMMIT): the deferred commits only -- the leaf tree's gdn_commit_tree(defer), the chain's gdn_commit(ring_only)
    self.ringc = RING_COMMIT and self.defer and (self.leaf or not self.tree)
    if self.ringc: self.cwords = K.buf("commit_w", QK.RC_W + M, dtypes.int32); poke(self.cwords, np.zeros(QK.RC_W + M, np.int32))
    # QWEN_LAYER_BLOCK=k (default 1; bonsai2_generate: 4): the verify pass's layers k at a time, each block one TinyJit call and so
    # one job (with ZHOUYI_CHAIN_MAX >= its launches), its k layers in k window slots of their own (2 k slots: the next block's are
    # mapped while this one runs); the head's parts in one job over the block slots too. Needs every layer pinned zero-copy (no
    # streamed layer), an even k dividing NL, the chain verify. 1: a job per layer (and per head part), as before.
    k = int(os.environ.get("QWEN_LAYER_BLOCK", "1")); zc = self.pre.zc
    ok = k > 1 and k % 2 == 0 and R.NL % k == 0 and bool(zc) and not zc.get("stream") and (not self.tree or self.leaf)
    if k > 1 and not ok: print(f"   QWEN_LAYER_BLOCK={k}: needs an even k dividing {R.NL}, every layer pinned zero-copy, the chain verify; a job per layer", flush=True)
    self.blk = k if ok else 1; self.jit_vblk = {}
    if self.blk > 1:
      self.pre.block_slots(self.blk)
      print(f"   layer blocks: {self.blk} layers a job, {2 * self.blk} window slots of {zc['bsize'] / 1e6:.0f} MB", flush=True)
    self.need_feats = bool(self.spec_log)                                 # the head's (top logit, 2nd max) per row: telemetry
    self.fnorm = np.load(os.path.join(self.cache, "outside_small.npz"))["norm"]               # host: the final norm (MTP input)
    if self.MTPL: self._mtp_init()
    if NPU_IDS: self._ids_init()
    if os.environ.get("QWEN_SPEC_WARM", "1") == "1": self.spec_warmup()
  def _ids_init(self):
    """QWEN_NPU_IDS: the decode loop's token ids stay on the device -- the verify pass's ids (vt: [cur, drafts], padded), the drafter's
    (nxt: the accepted tokens; dt: a chained draft), embedded there (embed_tern: the GGUF's PTQ1_0 table, uploaded once), accepted
    there (accept: accb, the next cur) and drafted into (pick_id). The host reads ids (to print them, to choose the next geometry)
    but writes none after the prompt."""
    assert R.HAD and (not self.tree or self.leaf) and self.MTPL, "QWEN_NPU_IDS: the ternary embedding (bonsai2-27b), the chain verify or the leaf tree, the MTP drafter"
    g = self.W.g; kind, dims, raw = g.raw("token_embd.weight"); assert kind == "PTQ1_0" and dims[0] == R.H, (kind, dims)
    self.e_vocab = int(dims[1]); self.etab = dev(np.array(np.frombuffer(raw, np.uint8)))   # (a copy: the GGUF map is read-only)
    s_ = g.had.signs.get(R.H) if "token_embd.weight" in g.had.inverse else None
    self.esign = dev(np.ones(R.H, np.float32) if s_ is None else np.asarray(s_, np.float32)); self.edesc = dev(QK.embed_tern_desc(R.H))
    K = self.ver.K
    self.vtok, self.dtok, self.nxtb, self.acco = (K.buf(t, 16, dtypes.int32) for t in ("ids_vt", "ids_dt", "ids_nxt", "ids_acc"))
    self.embrows = K.buf("emb_rows", 16 * R.H, dtypes.float32)
    self.ne1, self.nh1, self.fn1 = (dev((1.0 + w).astype(np.float32)) for w in (self.mtp_ne, self.mtp_nh, self.fnorm))
    self.jd = dev(np.zeros(16, np.int32)); self.jit_ids = {}
  def _embed_dev(self, K, m, out, ids):
    return K.call(f"embed_tern|{m}|{R.H}|{self.e_vocab}", QK.embed_tern_src(m, R.H, self.e_vocab, R.H), out, ids, self.etab, self.esign, self.edesc)
  def _jid(self, key, fn):
    """The small id / row kernels as one JIT submission each (`self.jd`: a dummy input; no kernel reads a JIT input)."""
    if key not in self.jit_ids: self.jit_ids[key] = TinyJit(fn)
    return self.jit_ids[key](self.jd)
  def _row_move(self, t, src, dst):
    """Row src of the persistent [rows, H] fp32 buffer t into row dst (host mapping, between jobs: poke's rule)."""
    from qwen38_npu import sync
    sync(); raw = t.uop.base.buffer._buf; OA.host_invalidate(t); n = R.H * 4
    ctypes.memmove(raw.va + dst * n, raw.va + src * n, n)
  def _vemb(self, m):
    L = self.vers[m]; return self._jid(("vemb", m), lambda _: self._embed_dev(L.K, m, L.K.bufs["xiov0"], self.vtok))
  def _vacc(self, m):
    L = self.vers[m]
    return self._jid(("vacc", m), lambda _: L.K.call(f"accept|{m}", QK.accept_src(m), self.acco, self.vtok, L.K.bufs["ids_head"], self.accb, self.nxtb))
  def _mtpin(self, K, r, mp, hoff, catch, X):
    """The MTP layer's input rows (mtpin_src) into geometry mp's `mtp_in`: QWEN_MTPIN=dma (default) mtpin_d_src (12 tasks, the rows
    by DMA; the same bits, checked on the vendor simulator) | task (mtpin_src: one task a row)."""
    if MTPIN_DMA:
      if not hasattr(self, "mi_desc"): self.mi_desc = dev(QK.mtpin_d_desc(R.H))
      return K.call(f"mtp_in_d|{r}|{mp}|{hoff}|{int(catch)}", QK.mtpin_d_src(r, mp, R.H, R.EPS, hoff, catch), K.bufs["mtp_in"], self.embrows, X, self.ne1, self.nh1, self.fn1, self.mi_desc)
    return K.call(f"mtp_in|{r}|{mp}|{hoff}|{int(catch)}", QK.mtpin_src(r, mp, R.H, R.EPS, hoff, catch), K.bufs["mtp_in"], self.embrows, X, self.ne1, self.nh1, self.fn1)
  def _dcatch_fn(self, m, a, mp):
    """The drafter's catch-up input rows (rows 0..a-1: the accepted tokens' embeddings | the verify pass's hidden rows) on the
    device -> (key, the kernels' call)."""
    Lm, Lv = self._mtpL(mp), self.vers[m]
    def f(_):
      self._embed_dev(Lm.K, a, self.embrows, self.nxtb)
      return self._mtpin(Lm.K, a, mp, 0, True, Lv.K.bufs["xiov0"])
    return ("dcatch", m, a, mp), f
  def _dchain_fn(self, mprev, row, mp):
    """A chained draft's input row (the last draft's embedding | the MTP layer's output row `row` of geometry mprev) on the device."""
    Lm, Lp = self._mtpL(mp), self._mtpL(mprev)
    def f(_):
      self._embed_dev(Lm.K, 1, self.embrows, self.dtok)
      return self._mtpin(Lm.K, 1, mp, row, False, Lp.K.bufs["mtp_out"])
    return ("dchain", mprev, row, mp), f
  def _dcatch(self, m, a, mp): return self._jid(*self._dcatch_fn(m, a, mp))
  def _dchain(self, mprev, row, mp): return self._jid(*self._dchain_fn(mprev, row, mp))
  def _draft1_ok(self, hp, parts):
    """QWEN_DRAFT_JOB=1 (default): a draft pass as ONE job -- the pending commit (if any), the input rows, the MTP layer and the
    draft head's parts -- when the MTP layer has a slot of its own and every part a slot of its own (the layer blocks' parity-0
    slots, or the two layer slots); else (QWEN_DRAFT_JOB=0, or the parts take turns in one slot) the jobs as before."""
    zc = self.pre.zc
    if not DRAFT_JOB or hp["sid"] is not None or getattr(self, "mtp_own", None) is None: return False
    if self.blk > 1: return len(parts) <= self.blk and all(p_["n"] <= zc["bsize"] for p_ in parts)
    return len(parts) <= 2
  def _dhead_views(self, hp, parts):
    """The draft head's parts mapped into their one-job slots (no drain: the caller has waited for the job in flight) -> their views."""
    zc = self.pre.zc; raw = zc["raw"]
    if self.blk > 1:
      for i, p_ in enumerate(parts): slot_map_nodrain(raw, zc["bslots"][(0, i)][0], p_["wid"], 0, p_["n"])
      return [self.pre.blob_view(0, i, p_["n"]) for i, p_ in enumerate(parts)]
    for i, p_ in enumerate(parts): slot_map_nodrain(raw, zc["slots"][i][0], p_["wid"], 0, p_["n"])
    zc.setdefault("slot_src", [None, None])[0] = None; zc["slot_src"][1] = None
    return [p_["views"][i] for i, p_ in enumerate(parts)]
  def _mtp_map_own(self):
    """The MTP layer into its own slot (no drain: nothing is in flight -- the caller poked first) -> its views."""
    zc = self.pre.zc; slot_map_nodrain(zc["raw"], self.mtp_own, self.mtp_w[0], 0, self.mtp_w[2]); return self.mtp_own_views
  def _draft1(self, m, rows_fn, commit_fn, L, hp, n, c0s, pick, x3, top3, devhead, aix, *views):
    """ONE draft pass as one JIT (one job): the pending commit, the input rows (rows_fn: the device's mtp_in; None: poked by the
    host), the MTP layer on geometry m (-> mtp_out), then the draft head's first n parts: `devhead` (QWEN_NPU_IDS) _head_parts
    (head_reduce, pick_id); else each part's partials (top_head<i> / top3_head<i>), as head_top's per-part JITs leave them."""
    if commit_fn is not None: commit_fn()
    if rows_fn is not None: rows_fn(None)
    nm = len(self.MTP_LIN); out = self.mtp_step(m, aix, *views[:nm])
    if devhead: return self._head_parts(L, out, self.mtp_norm_w1, hp, n, c0s, pick, x3, *views[nm:])
    for i in range(n): self._head_part(L, i, out, self.mtp_norm_w1, views[nm + i], top3=top3, hp=hp)
    return out
  def _take_commit(self):
    """The commit left pending for the next draft job (commit(fold=True)) -> (key, its kernel's call) or None; cleared."""
    c = getattr(self, "pcommit", None); self.pcommit = None; return c
  def mtp_pass_dev(self, r, pos, pick, rows_in=None, rows=None):
    """mtp_pass with the input rows already on the device (_dcatch / _dchain) and the draft's id picked into vt (pick = (j, row)).
    `rows`: (key, call) of the input rows' kernels (_dcatch_fn / _dchain_fn): with the one-job draft (QWEN_DRAFT_JOB) the pending
    commit, they, the MTP layer and the head parts are one JIT; else `rows_in` (the call that runs them as their own job) is
    made here after the position's poke (a poke waits for a job in flight), so the MTP layer's JIT is prepared while it runs."""
    zc = self.pre.zc; m = min(g for g in self.mtp_geos if g >= r); t0 = time.perf_counter()
    poke(self.aix[self.NATT], np.array([self.NATT, pos], np.int32))
    hp = self.dhead; parts = hp["parts"][:self.draft_parts]
    if rows is not None and self._draft1_ok(hp, parts):
      L = self._dheadL(m); c0s = tuple(p_["c0"] for p_ in parts); x3 = pick if self.leaf else False; cm = self._take_commit()
      own = self._mtp_map_own(); views = [own[k] for k in self.MTP_LIN] + self._dhead_views(hp, parts)
      key = ("d1", rows[0], m, pick, x3, cm[0] if cm else None)
      if key not in self.jit_d1: self.jit_d1[key] = TinyJit(functools.partial(self._draft1, m, rows[1], cm[1] if cm else None, L, hp, len(parts), c0s, pick, x3, False, True))
      with one_job(): o = self.jit_d1[key](self.aix[self.NATT], *views)
      ids, pr = self._head_out(o, L, parts, c0s, x3)
      self._draft_feats(r); self.prof["mtp_dev"] = self.prof.get("mtp_dev", 0.0) + time.perf_counter() - t0
      return m, ids, pr
    self._commit_flush()
    if rows_in is None and rows is not None: rows_in = lambda: self._jid(*rows)
    if rows_in is not None: rows_in()
    if self.mtp_x: slot_map_nodrain(zc["raw"], self.mtp_sid, self.mtp_w[0], 0, self.mtp_w[2]); views = self.mtp_views   # (in flight: the input rows' job)
    else:
      zc["raw"].slot_map(zc["slots"][0][0], self.mtp_w[0], 0, self.mtp_w[2]); zc.setdefault("slot_src", [None, None])[0] = None
      views = zc["views"][(0, "full")]
    if m not in self.jit_mtp: self.jit_mtp[m] = TinyJit(functools.partial(self.mtp_step, m))
    with one_job(): out = self.jit_mtp[m](self.aix[self.NATT], *[views[k] for k in self.MTP_LIN])   # (the capture: one job)
    ids, pr = self._head_top_dev(out, self._dheadL(m), self.mtp_norm_w1, self.dhead, self.dhead["parts"][:self.draft_parts], pick=pick, x3=pick if self.leaf else False)
    self._draft_feats(r); self.prof["mtp_dev"] = self.prof.get("mtp_dev", 0.0) + time.perf_counter() - t0
    return m, ids, pr
  def mtp_draft_dev(self, m, a, pos, k):
    """mtp_draft on the device's ids: the catch-up on the a accepted tokens (nxt) and the verify pass's hidden rows, then up to
    k - 1 chained drafts while the draft probability stays >= tau. Each draft's id goes into vt on the device; the host reads
    the ids and probabilities only (the geometry of the next pass, the log)."""
    t0 = time.perf_counter(); mp = min(g for g in self.mtp_geos if g >= a)
    mp, ids, ps = self.mtp_pass_dev(a, pos, (0, a - 1), rows=self._dcatch_fn(m, a, mp)); d, pr = [int(ids[a - 1])], [float(ps[a - 1])]
    log = self.spec_log is not None; mprev, row = mp, a - 1
    if log: tl, fx, px = [time.perf_counter() - t0], [self.mtp_feats[a - 1]], [self.mtp_ptop[a - 1]]
    t3 = [self.mtp_t3[a - 1]] if self.leaf else []                       # (the leaf tree: each draft's top-3, its leaves)
    for j in range(1, k):
      if not (len(pr) < self.kmax and (float(np.prod(pr)) >= self.tc if self.leaf else pr[-1] >= self.tau)): break
      t1 = time.perf_counter(); m1 = min(self.mtp_geos)
      _, ids, ps = self.mtp_pass_dev(1, pos + a - 1 + j, (j, 0), rows=self._dchain_fn(mprev, row, m1)); d.append(int(ids[0])); pr.append(float(ps[0])); mprev, row = m1, 0
      if self.leaf: t3.append(self.mtp_t3[0])
      if log: tl.append(time.perf_counter() - t1); fx.append(self.mtp_feats[0]); px.append(self.mtp_ptop[0])
    self.draft_p, self.draft_t3 = pr, t3
    if log: self.draft_rec = dict(td=tl, f=fx, fp=px, catch_rows=a, **(dict(t3=t3) if self.leaf else {}))
    return d
  def spec_warmup(self):
    """Capture every verify geometry's (and the MTP layer's) JITs up front -- a capture pass costs ~1.2 s more than a replay --
    with throwaway passes at position 0, before any prefill: prefill rebuilds every state they touch (the DeltaNet states, the
    caches' low positions; the rings are never written by a verify pass)."""
    t0 = time.perf_counter()
    for m in self.geos:
      for _ in range(2): self.verify([0] * m, 0)                       # TinyJit: the second call captures, the third replays
    if self.MTPL:
      for g in self.mtp_geos:
        for _ in range(2): self.mtp_pass(np.zeros((g, 2 * R.H), np.float32), 0)
      if os.environ.get("QWEN_DRAFT_WARM", "1") == "1": self._draft_warmup()
    gb = sum(t.nbytes() for V_ in self.vers.values() for t in V_.K.bufs.values()) / 1e9   # every verify / draft geometry's persistent buffers
    self.prof = {}; print(f"   speculative decoding: geometries {self.geos} (+ the MTP layer) captured in {time.perf_counter() - t0:.0f} s; their device buffers "
                          f"{gb:.2f} GB (of which the state banks {self.banks.nbytes() / 1e9:.2f} GB)", flush=True)
  def _draft_warmup(self):
    """QWEN_DRAFT_WARM (default 1): capture the decode loop's draft JITs too, so no capture (~0.5-1.5 s each) lands inside a
    generation: QWEN_NPU_IDS -- every catch-up (verify geometry m, a accepted rows; with the folded commit) and chained (the row it
    drafts from, its depth) draft of mtp_draft_dev; the host path -- the one-job pass per MTP geometry, with and without the folded
    commit. Throwaway passes at position 0 before any prefill, as spec_warmup's (the commit's words copy nothing: a = 0)."""
    t0 = time.perf_counter(); n = 0; fold = COMMIT_FOLD and self.ringc
    def with_commit():
      if fold: poke(self.cwords, np.array([0, self.M, 0, -1, -1] + [0] * self.M, np.int32)); self.pcommit = self._commit_call(self.M)
    if NPU_IDS:
      m1 = min(self.mtp_geos); seen = set()
      for m in self.geos:
        for a in range(1, min(m, self.kmax + 1) + 1):
          mp = min(g for g in self.mtp_geos if g >= a)
          for _ in range(2): with_commit(); self.mtp_pass_dev(a, 0, (0, a - 1), rows=self._dcatch_fn(m, a, mp)); n += 1
          for j in range(1, self.kmax):
            src = (mp, a - 1) if j == 1 else (m1, 0)
            if (src, j) in seen: continue
            seen.add((src, j))
            for _ in range(2): self.mtp_pass_dev(1, 0, (j, 0), rows=self._dchain_fn(src[0], src[1], m1)); n += 1
    elif self._draft1_ok(self.dhead, self.dhead["parts"][:self.draft_parts if self.parts_thresh is None else 1]):
      for g in self.mtp_geos:                                             # (spec_warmup's mtp_pass captured the passes without a commit)
        if fold:
          for _ in range(2): with_commit(); self.mtp_pass(np.zeros((g, 2 * R.H), np.float32), 0); n += 1
    self.pcommit = None
    if self.ringc: poke(self.cwords, np.zeros(QK.RC_W + self.M, np.int32))
    print(f"   draft JITs warmed: {n} passes in {time.perf_counter() - t0:.0f} s", flush=True)
  def _ver(self, m):
    """The verify geometry for m tokens (Layers(m): compact rows), sharing the weight slots; its two ping-pong row buffers."""
    if m not in self.vers:
      V_ = Layers(m, self.cache); V_.slots, V_._inflight = self.pre.slots, self.pre._inflight
      V_.pinned, V_._small_dev, V_.stage, V_.zc = self.pre.pinned, self.pre._small_dev, self.pre.stage, self.pre.zc
      self.vers[m] = V_; self.xbvs[m] = [V_.K.rows_buf("xiov0", R.H), V_.K.rows_buf("xiov1", R.H)]
    return self.vers[m]
  def _mtp_init(self):
    """The MTP head: its attention-format linears in one weight buffer (mapped into slot 0 between verify passes: the attention
    layers' views read it), fc (E4M3, 128 x 128 blocks) in a device buffer, its norms."""
    zc = self.pre.zc; assert zc, "the MTP draft head needs zero-copy weights"; raw = zc["raw"]; src = self.mtp_src
    sizes, offs, n = src._pack(self.MTPL); l0 = R.LAYER_TYPES.index("full")
    # QWEN_DRAFT_PARTS = <n>[,thresh:<x>] (draft_parts()). <n>: the draft head reads the block-scaled head's first n parts only
    # (2 of 4 = ids < 124416: every token of the measured English / code generations; ornith-9b 2 of 6 = ids < 82944: the same
    # drafts accepted as with 3, 4 % faster). thresh:<x>: parts 1.. are read only when part 0's top logit on the row drafted from
    # is <= x (without it all n parts: the default on ornith-9b); 2,thresh:25 the default on qwen3.8-27b:
    # +2.1 % projected, the ids those of plain greedy. A part's GEMM streams ~318 MB of head weights (18-19 ms on the 27B); a
    # draft from part 0 alone is wrong when the true argmax lies elsewhere (the verify pass then rejects it).
    self.draft_parts, self.parts_thresh, self.parts_policy = draft_parts(os.environ.get("QWEN_DRAFT_PARTS", "2,thresh:25" if R.MODEL == "qwen3.8-27b" else "2"))
    self.MTP_LIN = Layers.files(src._meta(self.MTPL)[0]) if self.mtp_x else self.ATT_LIN
    if self.mtp_x:
      # another model's MTP layer (bonsai2-27b): the trunk's slots are sized for its ternary layers (a quarter of this fp8 layer's
      # bytes), so the layer gets a window slot of its own, re-pointed at it before each draft pass -- the fp8 draft head's parts
      # (QWEN_MTP_HEAD=qwen) take turns in the same slot.
      # QWEN_MTP_HEAD: the draft head -- bonsai (default: the target's own ternary parts, on the verify geometry's had_a32 with
      #   mtp.norm's 1 + w times the signs) | qwen (Qwen3.8-27B's E4M3 head parts from QWEN_MTP_CACHE, the head the MTP layer was
      #   trained with: a full Qwen3.8-27B cache packed with --lm-head-fp8 --scales single).
      # QWEN_MTP_EMBED: the drafted token's embedding -- bonsai (default: the GGUF's row, decoded and un-rotated) | qwen (Qwen3.8-27B's
      #   bf16 table: outside.safetensors in QWEN_MTP_DIR). Board, 10 prompts (bonsai2-27b/README.md): bonsai | bonsai 2.39 tok/s,
      #   qwen | qwen 2.47.
      self.mtp_head = os.environ.get("QWEN_MTP_HEAD", "bonsai"); assert self.mtp_head in ("bonsai", "qwen"), f"QWEN_MTP_HEAD={self.mtp_head}: bonsai | qwen"
      self.mtp_emb = os.environ.get("QWEN_MTP_EMBED", "bonsai"); assert self.mtp_emb in ("bonsai", "qwen"), f"QWEN_MTP_EMBED={self.mtp_emb}: bonsai | qwen"
      qh = os.path.join(self.mtp_cache, "lm_head_fp8.npz")
      assert self.mtp_head != "qwen" or os.path.exists(qh), f"QWEN_MTP_HEAD=qwen: {qh} not found (set QWEN_MTP_CACHE to a Qwen3.8-27B cache with its fp8 head)"
      zq = np.load(qh) if self.mtp_head == "qwen" else None
      hsz = [os.path.getsize(os.path.join(self.mtp_cache, f"lm_head_fp8_{i}.bin")) for i in range(min(self.draft_parts, len(zq["parts"])))] if zq is not None else []
      self.mtp_sid, sva = raw.slot_alloc(max([n] + hsz))
      self.mtp_views = {k_: Tensor.from_blob(sva + offs[k_], (sz,), dtype=dtypes.uint8, device=DEV) for k_, sz in sizes.items()}
      if zq is not None:                                                  # the first draft_parts parts only (a draft reads no others)
        parts = []
        for i, (g0, g1) in enumerate(zq["parts"][:len(hsz)]):
          f = os.path.join(self.mtp_cache, f"lm_head_fp8_{i}.bin"); hid_, hmm = raw.wbuf_alloc(hsz[i]); read_into(f, memoryview(hmm), hsz[i], drop=True)
          v = Tensor.from_blob(sva, (hsz[i],), dtype=dtypes.uint8, device=DEV)
          parts.append(dict(wid=hid_, mm=hmm, n=hsz[i], groups=int(g1 - g0), views=[v, v], c0=int(g0) * 48))
        self.dhead = dict(parts=parts, kw={}, sid=self.mtp_sid, n=int(zq["n"][0]))
      else: self.dhead = self.mhead                                       # the target's own (ternary) head parts
      if self.mtp_emb == "qwen":                                          # Qwen3.8-27B's bf16 embedding rows (outside.safetensors)
        assert os.environ.get("QWEN_MTP_DIR"), "QWEN_MTP_EMBED=qwen: set QWEN_MTP_DIR to Qwen3.8-27B-FP8's folder (its outside.safetensors)"
        self.qemb = R.SafeFile(os.path.join(os.path.expanduser(os.environ["QWEN_MTP_DIR"]), "outside.safetensors")).raw("model.language_model.embed_tokens.weight")
      print(f"   drafter: {self.mtp_cache}'s MTP layer ({n / 1e9:.2f} GB, fp8) | embedding {self.mtp_emb} | draft head {self.mtp_head} "
            f"({sum(p_['n'] for p_ in self.dhead['parts'][:self.draft_parts]) / 1e9:.2f} GB in {min(self.draft_parts, len(self.dhead['parts']))} parts)", flush=True)
    else:
      assert offs == zc["packs"][l0][1] and n <= zc["packs"][l0][2], "the MTP pack does not match an attention layer's layout"
      self.dhead = self.mhead
    # the MTP layer's own window slot (the one-job draft, QWEN_DRAFT_JOB: the head's parts take the layer slots meanwhile): another
    # model's MTP layer has one already; else one more slot of the layer's size, if the window has room (else the jobs as before)
    self.mtp_own = self.mtp_own_views = None
    if self.mtp_x: self.mtp_own, self.mtp_own_views = self.mtp_sid, self.mtp_views
    elif DRAFT_JOB:
      try:
        sid_, sva_ = raw.slot_alloc(n); self.mtp_own = sid_
        self.mtp_own_views = {k_: Tensor.from_blob(sva_ + offs[k_], (sz,), dtype=dtypes.uint8, device=DEV) for k_, sz in sizes.items()}
      except OSError as e: print(f"   (QWEN_DRAFT_JOB: no window slot for the MTP layer -- {e}; a draft pass keeps its jobs)", flush=True)
    wid, mm = raw.wbuf_alloc(n)
    for k_, sz in sizes.items(): read_into(os.path.join(self.mtp_cache, f"L{self.MTPL}_{k_}.bin"), memoryview(mm)[offs[k_]:offs[k_] + sz], sz)
    self.mtp_w = (wid, mm, n); self.meta_mtp, small = src._meta(self.MTPL)
    self.meta_mtp = dict(self.meta_mtp, fc=tuple(int(v) for v in np.load(os.path.join(self.mtp_cache, f"L{self.MTPL}_fc.npz"))["meta"]))
    self.mtp_fc = dev(np.fromfile(os.path.join(self.mtp_cache, f"L{self.MTPL}_fc.bin"), np.uint8))
    self.mtp_ne, self.mtp_nh = small["mtp_pre_fc_norm_embedding_weight"], small["mtp_pre_fc_norm_hidden_weight"]
    w1 = 1.0 + small["mtp_norm_weight"]
    if R.HAD and self.dhead is self.mhead: w1 = (w1 * self.dec.signs[R.H]).astype(np.float32)   # the ternary head's input signs (as the final norm's)
    K = self.ver.K; self.mtp_norm_w1 = K.buf("mtp_norm_w1", R.H, dtypes.float32).assign(dev(w1)).realize()
    # a draft pass runs on the smallest geometry that holds its rows: 1 (a chained draft) or a verify geometry (the catch-up on the
    # accepted tokens, <= QWEN_DRAFT_MAX + 1 rows) -- not the widest, whose second row tile costs the compute-bound GEMMs ~1.35x
    # (the tree verify keeps the chain's draft geometries 3 / 4 / 6: a catch-up pass has at most NC = 6 rows, the widest a rescue hit commits)
    gs = [3, 4, 6] if self.tree and not self.leaf else self.geos
    self.mtp_geos = sorted({1} | {g for g in gs if g <= self.kmax + 1} | {min(g for g in gs if g >= self.kmax + 1)})
    for g in self.mtp_geos:
      self._ver(g)
      if self.mtp_x and g not in self.mtpg: self.mtpg[g] = Layers(g, self.mtp_cache, fp8=True)
    self.jit_mtp = {}; self.prof = {}
    # QWEN_HEAD_TOP3=1: the draft head's kernel returns each part's top-3 (logit, column) per row (head_top3_src); the drafts
    # are unchanged (the top-1 of the merged lists); QWEN_SPEC_LOG gets them and the model's token's rank among them
    self.head_top3 = os.environ.get("QWEN_HEAD_TOP3", "0") == "1" or self.tree is not None   # the tree's rescue rows are the drafts' rank-2 tokens
  def _mtpL(self, m): return self.mtpg[m] if self.mtp_x else self.vers[m]       # the MTP layer's geometry for m rows
  def _dheadL(self, m): return self.vers[m] if self.dhead is self.mhead else self.mtpg[m]   # the draft head's (the ternary head: had_a32)
  def mtp_step(self, m, aix, *views):
    """The MTP layer on geometry m's rows: fc([normed embedding | normed hidden]) -> gated attention (its own cache, attention
    index NATT) -> the dense MLP -> the persistent `mtp_out` rows (before mtp.norm)."""
    L = self._mtpL(m); slot = dict(zip(self.MTP_LIN, views)); ah = L.K.hold("aix_mtp", aix)
    u = L.K.unpack(L.lin_ct(L.K.rms_a(L.K.rows_buf("mtp_in", 2 * R.H), None, 2 * R.H, "a_fc", norm=False), {"fc": self.mtp_fc}, self.meta_mtp, "fc"), R.H, "mtp_u")
    a_h = L.K.rms_a(u, (self.Win_a, ah), R.H, "a_h"); ct = L.lin_cts(a_h, slot, self.meta_mtp, ("q", "kv"))
    q_rows = L.K.unpack(ct["q"][0], R.NH * 2 * R.HD, "q_rows", goff=ct["q"][1])
    kv_rows = L.K.unpack(ct["kv"][0], 2 * R.NKV * R.HD, "kv_rows", goff=ct["kv"][1])
    oc = L.K.attn_decm(q_rows, kv_rows, self.QN, self.KN, self.ROPE, self.Kall, self.Vall, ah, R.NH, R.NKV, R.HD, self.TMAX, R.ROT)
    x = L.K.resid(u, L.lin_ct(L.K.rms_a(oc, None, R.NH * R.HD, "a_o", norm=False), slot, self.meta_mtp, "o"), R.H, "x1")
    return L.mlp(x, slot, self.meta_mtp, (self.Wpost_a, ah), "mtp_out").realize()
  def mtp_pass(self, rows, pos):
    """rows [r, 2 H] (normed embedding | normed hidden) at positions pos.. -> (the layer's output rows [r, H], the draft head's
    (ids [r], probabilities [r]) over the draft vocabulary). Runs on the smallest MTP geometry m >= r (padded with the last row)."""
    zc = self.pre.zc; r = len(rows); m = min(g for g in self.mtp_geos if g >= r); L = self._mtpL(m); pf = self.prof; t0 = time.perf_counter()
    x = np.empty((m, 2 * R.H), np.float32); x[:r] = rows; x[r:] = rows[-1]
    poke(L.K.rows_buf("mtp_in", 2 * R.H), x)                              # only the m real rows (the kernels read no others)
    poke(self.aix[self.NATT], np.array([self.NATT, pos], np.int32))
    hp = self.dhead; parts = hp["parts"][:self.draft_parts]; n0 = len(parts) if self.parts_thresh is None else 1   # (thresh: part 0, then maybe more)
    if self._draft1_ok(hp, parts[:n0]):                                    # one job: the pending commit, the MTP layer, the first n0 parts
      cm = self._take_commit(); own = self._mtp_map_own(); Ld = self._dheadL(m); top3 = self.head_top3
      views = [own[k] for k in self.MTP_LIN] + self._dhead_views(hp, parts[:n0])
      key = ("h1", m, n0, top3, cm[0] if cm else None)
      if key not in self.jit_d1: self.jit_d1[key] = TinyJit(functools.partial(self._draft1, m, None, cm[1] if cm else None, Ld, hp, n0, None, None, False, top3, False))
      t1 = time.perf_counter()
      with one_job(): out = self.jit_d1[key](self.aix[self.NATT], *views)
      hx = OA.host_invalidate(out).numpy()[:r].copy()
      pre = [OA.host_invalidate(Ld.K.bufs[f"top3_head{i}" if top3 else f"top_head{i}"]).numpy().reshape(-1, Ld.n, 8 if top3 else 4).copy() for i in range(n0)]
      t2 = time.perf_counter(); ids, pr = self.head_top(out, Ld, self.mtp_norm_w1, self.draft_parts, thresh=self.parts_thresh, row=r - 1, top3=top3, hp=hp, pre=pre)
      lg = (ids[:r], pr[:r]); t3 = time.perf_counter(); self._draft_feats(r)
      for k_, v in (("mtp_in", t1 - t0), ("mtp_layer", t2 - t1), ("mtp_head", t3 - t2)): pf[k_] = pf.get(k_, 0.0) + v
      return hx, lg
    self._commit_flush()
    if self.mtp_x: zc["raw"].slot_map(self.mtp_sid, self.mtp_w[0], 0, self.mtp_w[2]); views = self.mtp_views   # its own slot
    else:
      zc["raw"].slot_map(zc["slots"][0][0], self.mtp_w[0], 0, self.mtp_w[2]); zc.setdefault("slot_src", [None, None])[0] = None
      views = zc["views"][(0, "full")]
    if m not in self.jit_mtp: self.jit_mtp[m] = TinyJit(functools.partial(self.mtp_step, m))
    t1 = time.perf_counter()
    with one_job(): out = self.jit_mtp[m](self.aix[self.NATT], *[views[k] for k in self.MTP_LIN])
    hx = OA.host_invalidate(out).numpy()[:r].copy()
    t2 = time.perf_counter(); ids, pr = self.head_top(out, self._dheadL(m), self.mtp_norm_w1, self.draft_parts, thresh=self.parts_thresh, row=r - 1, top3=self.head_top3, hp=self.dhead)
    lg = (ids[:r], pr[:r]); t3 = time.perf_counter(); self._draft_feats(r)
    for k_, v in (("mtp_in", t1 - t0), ("mtp_layer", t2 - t1), ("mtp_head", t3 - t2)): pf[k_] = pf.get(k_, 0.0) + v
    return hx, lg
  def _draft_feats(self, r):
    """After the draft head on r rows: the telemetry features of its rows -- mtp_feats (top logit, 2nd (part, task)
    max), mtp_ptop (each part's top logit, the parts read), mtp_t3 (each part's top-3 (ids, logits), QWEN_HEAD_TOP3)."""
    if self.need_feats: self.mtp_feats = [(float(a), float(b)) for a, b in zip(*(f[:r] for f in self.head_feats))]
    self.mtp_ptop = [[float(v) for v in col] for col in self.head_ptop[:, :r].T]
    if self.head_top3: self.mtp_t3 = self.head_t3[:r]
  def mtp_rows(self, hid, toks, chained=False):
    """The MTP inputs: [rms(embed(tok), pre_fc_norm_embedding) | rms(h, pre_fc_norm_hidden)], h the 27B's final-normed hidden
    (or, chained, the MTP layer's own output)."""
    h = hid if chained else R.rms(hid, self.fnorm)
    return np.concatenate([R.rms(self.mtp_embed(list(toks)), self.mtp_ne), R.rms(h, self.mtp_nh)], 1).astype(np.float32)
  def mtp_embed(self, ids):
    """The drafted tokens' embeddings: the target's (W.embed), or QWEN_MTP_EMBED=qwen: Qwen3.8-27B's bf16 rows."""
    if not self.mtp_x or self.mtp_emb != "qwen": return self.W.embed(ids)
    _, shape, raw = self.qemb; row = shape[1] * 2
    return np.stack([R.bf16_to_f32(np.frombuffer(raw[t * row:(t + 1) * row], np.uint16)) for t in ids])
  def mtp_draft(self, hid, nxt, pos, k):
    """The MTP cache caught up on the true positions pos .. pos + len(nxt) - 1 (hidden rows `hid`, the tokens after them `nxt`),
    then k chained drafts for the positions after them."""
    t0 = time.perf_counter(); rows = self.mtp_rows(hid, nxt); self.prof["mtp_rows"] = self.prof.get("mtp_rows", 0.0) + time.perf_counter() - t0
    a = len(nxt); hx, (ids, ps) = self.mtp_pass(rows, pos)                # the draft and its probability (draft vocabulary)
    d, pr = [int(ids[a - 1])], [float(ps[a - 1])]; h = hx[a - 1]
    log = self.spec_log is not None
    t3 = [self.mtp_t3[a - 1]] if self.head_top3 else []                  # each draft's merged top-3 per part read (the log; the tree's rescue rows)
    if log: tl, fx, px = [time.perf_counter() - t0], [self.mtp_feats[a - 1]], [self.mtp_ptop[a - 1]]   # telemetry per draft: wall time, (top logit, 2nd max), the parts' tops
    for j in range(1, k):
      tp = time.perf_counter(); go = len(pr) < self.kmax and (float(np.prod(pr)) >= self.tc if self.leaf else pr[-1] >= self.tau)   # (leaf: the path probability)
      t1 = time.perf_counter(); self.prof["policy"] = self.prof.get("policy", 0.0) + t1 - tp
      if not go: break                                                    # the threshold rule: not confident (or kmax drafts)
      hx, (ids, ps) = self.mtp_pass(self.mtp_rows(h[None], [d[-1]], chained=True), pos + a - 1 + j)
      d.append(int(ids[0])); pr.append(float(ps[0])); h = hx[0]
      if self.head_top3: t3.append(self.mtp_t3[0])
      if log: tl.append(time.perf_counter() - t1); fx.append(self.mtp_feats[0]); px.append(self.mtp_ptop[0])
    self.draft_p, self.draft_t3 = pr, t3
    if log: self.draft_rec = dict(td=tl, f=fx, fp=px, catch_rows=a, **(dict(t3=t3) if self.head_top3 else {}))
    return d
  def vgdn_step(self, m, par, idx, *rest):
    L = self.vers[m]; slot = dict(zip(self.GDN_LIN, rest[:len(self.GDN_LIN)])); x = self.xbvs[m][par]; idh = L.K.hold("idx_held", idx)
    a_h = L.K.rms_a(x, (self.Win_g, idh), R.H, "a_h")
    ct = L.lin_cts(a_h, slot, self.meta_gdn, ("qkv", "z"))                   # one GEMM when the cache fuses qkv | z
    if self.leaf:                                                         # the leaf tree: leaves read-only from their parent's state, the deferred commit (QWEN_GDN_FAST: gdn_fast_src(tree=True))
      rows = R.HAD or QK.gdn_fast() == "off"                              # (else o_proj's A straight from the kernel)
      a_o = L.K.gdn_tokl(self.Sall, self.Call, idh, self.posb, ct["qkv"][0], ct["z"][0], x, self.Wab, self.Adt, self.Cwt, self.Nw, self.banks, self.rawm, self.pathb, self.treeb,
                         R.NV, R.NK, R.DK, R.DV, self.C, R.CONV, R.H, self.NGDN, self.M, xoff=ct["qkv"][1], zoff=ct["z"][1], rows=rows)
      if rows: a_o = L.K.rms_a(a_o, None, R.NV * R.DV, "a_o", norm=False)
    elif self.tree:                                                       # the tree verify: the rescue rows as one-token waves from their parents' banks
      a_o = L.K.gdn_tokt(self.Sall, self.Call, idh, self.posb, ct["qkv"][0], ct["z"][0], x, self.Wab, self.Adt, self.Cwt, self.Nw, self.banks, self.rawm, self.treeb,
                         R.NV, R.NK, R.DK, R.DV, self.C, R.CONV, R.H, self.NGDN, xoff=ct["qkv"][1], zoff=ct["z"][1])
    else:
      a_o = L.K.gdn_tokm(self.Sall, self.Call, idh, self.posb, ct["qkv"][0], ct["z"][0], x, self.Wab, self.Adt, self.Cwt, self.Nw, self.banks, self.rawm,
                         R.NV, R.NK, R.DK, R.DV, self.C, R.CONV, R.H, self.NGDN, xoff=ct["qkv"][1], zoff=ct["z"][1], rows=R.HAD,
                         defer=(self.M, self.accb) if self.defer else None)
      if R.HAD: a_o = L.K.rms_a(a_o, None, R.NV * R.DV, "a_o", norm=False)   # o_proj's A from the fp32 rows (had_a32: whole 1024-blocks across heads)
    x = L.K.resid(x, L.lin_ct(a_o, slot, self.meta_gdn, "o"), R.H, "x1")
    return L.mlp(x, slot, self.meta_gdn, (self.Wpost_g, idh), f"xiov{1 - par}").realize()
  def vblk_step(self, m, types, *args):
    """A block of layers (types: their LAYER_TYPES; the block starts at an even layer) as one JIT: per layer its index buffer (sidx /
    aix) then its linears' views, in order; layer j reads xbvs[m][j % 2] and writes the other, as the per-layer steps do."""
    i = 0; out = None
    for j, lt in enumerate(types):
      nl = len(self.ATT_LIN if lt == "full" else self.GDN_LIN)
      out = (self.vatt_step if lt == "full" else self.vgdn_step)(m, j % 2, *args[i:i + 1 + nl]); i += 1 + nl
    return out
  def vatt_step(self, m, par, aix, *rest):
    L = self.vers[m]; slot = dict(zip(self.ATT_LIN, rest[:len(self.ATT_LIN)])); x = self.xbvs[m][par]; ah = L.K.hold("aix_held", aix)
    a_h = L.K.rms_a(x, (self.Win_a, ah), R.H, "a_h"); ct = L.lin_cts(a_h, slot, self.meta_att, ("q", "kv"))   # one GEMM when fused
    q_rows = L.K.unpack(ct["q"][0], R.NH * 2 * R.HD, "q_rows", goff=ct["q"][1])
    kv_rows = L.K.unpack(ct["kv"][0], 2 * R.NKV * R.HD, "kv_rows", goff=ct["kv"][1])
    if self.tree: oc = L.K.attn_tree(q_rows, kv_rows, self.QN, self.KN, self.ROPE, self.Kall, self.Vall, ah, self.treeb, R.NH, R.NKV, R.HD, self.TMAX, R.ROT)
    else: oc = L.K.attn_decm(q_rows, kv_rows, self.QN, self.KN, self.ROPE, self.Kall, self.Vall, ah, R.NH, R.NKV, R.HD, self.TMAX, R.ROT)
    x = L.K.resid(x, L.lin_ct(L.K.rms_a(oc, None, R.NH * R.HD, "a_o", norm=False), slot, self.meta_att, "o"), R.H, "x1")
    return L.mlp(x, slot, self.meta_att, (self.Wpost_a, ah), f"xiov{1 - par}").realize()
  def _head_part(self, L, i, x, norm_w1, view, top3=False, hp=None, red=None, pick=None, x3=False):
    """Head part i on geometry L's rows (under TinyJit: one submission) -> its top-1 partials (`top3`: top-3); part 0 first takes the norm.
    `hp`: the head (parts, GEMM keywords, slot, vocabulary; default the model's own, `mhead`)."""
    hp = self.mhead if hp is None else hp; K = L.K; p_ = hp["parts"][i]
    a = K.rms_a(x, norm_w1, R.H, "a_head") if i == 0 else K.buf("a_head", K.a_size(R.H), dtypes.uint16)
    ct = OA.gemm_gs(a, view, ks=KS, ns=3, nrb=L.nrb, nslices=R.H // (4 * KS), ngroups=p_["groups"], b8=True, bscale=True, q8=Q8MODE, scales=SCALES, piece=L.piece,
                    rows=L.n, out=K.buf(f"ct_head{i}", p_["groups"] * L.nrb * 3 * 192, dtypes.float32), **hp["kw"])
    t = K.head_top(ct, min(p_["groups"] * 48, hp["n"] - p_["c0"]), f"top3_head{i}" if top3 else f"top_head{i}", top3=top3)
    if x3: K.head_top3_row(ct, min(p_["groups"] * 48, hp["n"] - p_["c0"]), f"top3r_head{i}", pick[1])   # (the leaf tree's drafts: each part's top-3 on the pick row)
    if not red: return t
    o = K.head_reduce(red)                                                # the last part read: the token on the device too
    if pick is not None and red: K.call(f"pick_id|{pick[0]}|{pick[1]}", QK.pick_src(*pick), self.vtok, o, self.dtok)   # a draft: into vt
    return o
  def head_top(self, x, L, norm_w1=None, parts=None, thresh=None, row=None, top3=False, hp=None, pre=None):
    """The block-scaled head's top-1 on geometry L's rows of x (after the final norm, or `norm_w1`'s) -> (ids [m], probabilities
    [m]); `parts`: the head's first parts only (a draft needs no exact vocabulary: the verify pass decides). Per part the GEMM
    and head_top run as one JIT submission (their slot mapped first); only the 12 tasks' partials come back.
    `thresh`: parts 1.. are skipped when part 0's top logit on row `row` (every row if None) exceeds it (QWEN_DRAFT_PARTS' thresh:<x>).
    `top3`: the top-3 kernel; `head_t3` then holds per row, per part read, ([3 global ids], [3 logits]) in rank order.
    Also set: `head_ptop` [parts read, m] (each part's top logit) and, when `need_feats`, `head_feats`. `hp`: as _head_part's.
    `pre`: the first parts' partials, already computed (the one-job draft pass, _draft1)."""
    zc = self.pre.zc; raw = zc["raw"]; nw = self.norm_w1 if norm_w1 is None else norm_w1; tops = list(pre or []); hp = self.mhead if hp is None else hp
    if NPU_IDS and thresh is None and not top3 and not pre: return self._head_top_dev(x, L, nw, hp, hp["parts"][:parts])
    if not pre and thresh is None and DRAFT_JOB and getattr(self, "blk", 1) > 1 and hp["sid"] is None and len(hp["parts"][:parts]) <= 2 * self.blk \
       and all(p_["n"] <= zc["bsize"] for p_ in hp["parts"][:parts]):
      tops = self._head_tops_blk(x, L, nw, hp, hp["parts"][:parts], top3)   # layer blocks: every part in a block slot, one job
    for i, p_ in enumerate(hp["parts"][:parts]):
      if i >= len(tops):
        par = i % 2; raw.slot_map(zc["slots"][par][0] if hp["sid"] is None else hp["sid"], p_["wid"], 0, p_["n"]); key = (L.n, id(L), i, id(x), id(nw), top3, id(hp))
        if key not in self.jit_head: self.jit_head[key] = TinyJit(functools.partial(self._head_part, L, i, x, nw, top3=top3, hp=hp))
        tops.append(OA.host_invalidate(self.jit_head[key](p_["views"][par])).numpy().reshape(-1, L.n, 8 if top3 else 4).copy())
      if i == 0 and thresh is not None:                                   # the parts policy: part 0's top on the decision row(s)
        p0 = tops[0][..., 0].max(0)
        if (p0[row] if row is not None else p0.min()) > thresh: break
    ids, pr, self.head_ptop, feats, self.head_t3 = combine_tops(tops, [p_["c0"] for p_ in hp["parts"][:len(tops)]], top3, getattr(self, "need_feats", False))
    if feats is not None: self.head_feats = feats
    return ids, pr
  def _head_tops_blk(self, x, L, nw, hp, parts, top3):
    """head_top's parts as ONE job over the layer blocks' slots (QWEN_LAYER_BLOCK): parts 0..k-1 into the parity-0 block slots
    without a drain (the job in flight is the last block's, parity 1), the rest into the parity-1 slots after it -> each part's
    partials, as the per-part JITs leave them."""
    zc = self.pre.zc; raw = zc["raw"]; k = self.blk; views = []
    for i, p_ in enumerate(parts):
      bp, j = (0, i) if i < k else (1, i - k)
      if i < k: slot_map_nodrain(raw, zc["bslots"][(bp, j)][0], p_["wid"], 0, p_["n"])
      else: raw.slot_map(zc["bslots"][(bp, j)][0], p_["wid"], 0, p_["n"])   # (drains: the last block may read that slot)
      views.append(self.pre.blob_view(bp, j, p_["n"]))
    key = ("hall", L.n, id(L), id(x), id(nw), id(hp), len(parts), top3)
    if key not in self.jit_head: self.jit_head[key] = TinyJit(functools.partial(self._head_parts_host, L, x, nw, hp, len(parts), top3))
    with one_job(): self.jit_head[key](*views)
    return [OA.host_invalidate(L.K.bufs[f"top3_head{i}" if top3 else f"top_head{i}"]).numpy().reshape(-1, L.n, 8 if top3 else 4).copy() for i in range(len(parts))]
  def _head_parts_host(self, L, x, nw, hp, n, top3, *views):
    for i in range(n): t = self._head_part(L, i, x, nw, views[i], top3=top3, hp=hp)
    return t
  def _head_top_dev(self, x, L, nw, hp, parts, pick=None, x3=False):
    """head_top with the token chosen on the device (QWEN_NPU_IDS): the last part's job ends with head_reduce, and only its
    [ids | probabilities] come back. Telemetry (need_feats: QWEN_SPEC_LOG) still reads the partials and checks the ids."""
    zc = self.pre.zc; raw = zc["raw"]; c0s = tuple(p_["c0"] for p_ in parts); m = L.n
    if getattr(self, "blk", 1) > 1 and hp["sid"] is None and len(parts) <= self.blk and all(p_["n"] <= zc["bsize"] for p_ in parts):
      # layer blocks: every part in a block slot of parity 0 (the last block, parity 1, may still run) -> one JIT, one job
      for i, p_ in enumerate(parts): slot_map_nodrain(raw, zc["bslots"][(0, i)][0], p_["wid"], 0, p_["n"])
      key = ("all", L.n, id(L), id(x), id(nw), id(hp), len(parts), pick, x3)
      if key not in self.jit_head: self.jit_head[key] = TinyJit(functools.partial(self._head_parts, L, x, nw, hp, len(parts), c0s, pick, x3))
      with one_job(): o = self.jit_head[key](*[self.pre.blob_view(0, i, p_["n"]) for i, p_ in enumerate(parts)])
    else:
      o = self._head_parts_seq(L, x, nw, hp, parts, c0s, pick, x3)
    return self._head_out(o, L, parts, c0s, x3)
  def _head_parts(self, L, x, nw, hp, n, c0s, pick, x3, *views):
    """Head parts 0..n-1 (each in its own slot: views) as one JIT, the last ending with head_reduce (and pick_id)."""
    o = None
    for i in range(n): o = self._head_part(L, i, x, nw, views[i], hp=hp, red=c0s if i == n - 1 else None, pick=pick if i == n - 1 or x3 else None, x3=x3)
    return o
  def _head_parts_seq(self, L, x, nw, hp, parts, c0s, pick, x3=False):
    zc = self.pre.zc; raw = zc["raw"]
    for i, p_ in enumerate(parts):
      par = i % 2; raw.slot_map(zc["slots"][par][0] if hp["sid"] is None else hp["sid"], p_["wid"], 0, p_["n"]); red = c0s if i == len(parts) - 1 else None
      pk = pick if red or x3 else None; key = (L.n, id(L), i, id(x), id(nw), False, id(hp), red, pk, x3)
      if key not in self.jit_head: self.jit_head[key] = TinyJit(functools.partial(self._head_part, L, i, x, nw, hp=hp, red=red, pick=pk, x3=x3))
      o = self.jit_head[key](p_["views"][par])
    return o
  def _head_out(self, o, L, parts, c0s, x3=False):
    m = L.n; o = OA.host_invalidate(o).numpy(); ids, pr = o[:m].astype(np.int64), o[m:2 * m].view(np.float32).copy()
    if x3:                                                                # the parts' top-3 partials on the pick row -> head_t3 [m] (that row's; combine_tops' merge)
      t3s = [OA.host_invalidate(L.K.bufs[f"top3r_head{i}"]).numpy().reshape(-1, 1, 8).copy() for i in range(len(parts))]
      t3s = [np.concatenate([t[..., :6], np.ones_like(t[..., 6:7]), t[..., 7:]], -1) for t in t3s]   # (no sum of exp: a dummy 1)
      h3, _, _, _, r3 = combine_tops(t3s, list(c0s), True, False); self.head_t3 = r3 * m
      assert int(h3[0]) == int(ids[x3[1]]), ("head_top3r's top-1 differs from head_reduce", ids, h3)
    if getattr(self, "need_feats", False):
      tops = [OA.host_invalidate(L.K.bufs[f"top_head{i}"]).numpy().reshape(-1, m, 4).copy() for i in range(len(parts))]
      hid, _, self.head_ptop, self.head_feats, t3_ = combine_tops(tops, list(c0s), False, True)
      if not x3: self.head_t3 = t3_
      assert np.array_equal(hid, ids), ("head_reduce ids differ from combine_tops", ids, hid)
    else: self.head_ptop = np.zeros((len(parts), m), np.float32)
    return ids, pr
  def logits_rows(self, x, norm_w1=None, parts=None, L=None):
    """The fp8 head on rows 0..M-1 of the verify geometry (after the final norm, or `norm_w1`'s) -> [M, vocab]; `parts`: only the
    head's first parts (a draft needs no exact vocabulary: the verify pass decides) -> [M, their ids]."""
    L = self.ver if L is None else L; K = L.K; zc = self.pre.zc; raw = zc["raw"]; M = L.n
    a = K.rms_a(x, self.norm_w1 if norm_w1 is None else norm_w1, R.H, "a_head"); out = []
    for i, p_ in enumerate(self.head_parts[:parts]):
      par = i % 2; raw.slot_map(zc["slots"][par][0], p_["wid"], 0, p_["n"])
      ct = OA.gemm_gs(a, p_["views"][par], ks=KS, ns=3, nrb=L.nrb, nslices=R.H // (4 * KS), ngroups=p_["groups"], b8=True, bscale=True, q8=Q8MODE, scales=SCALES, piece=L.piece,
                      rows=M, out=K.buf(f"ct_head{i}", p_["groups"] * L.nrb * 3 * 192, dtypes.float32), **TERN_KW).realize()
      out.append(OA.host_invalidate(K.unpack(ct, p_["groups"] * 48, f"head_rows{i}")).numpy()[:M].copy())
    return np.concatenate(out, 1)[:, :self.head_n]
  def verify(self, toks, pos, head="top", tree=None):
    """toks (M ids) at positions pos .. pos + M - 1 through the layers -> head "top": the model's next token after each [M]
    (head_top), "full": logits [M, vocab], None: no head; the attention caches get the M rows, every DeltaNet layer its
    per-token state banks (commit() picks the accepted one). QWEN_SPEC_TREE: `tree` (a spec_tree.Tree over the M rows; None =
    the chain) is the kernels' table -- rows at pos + depth, attending their ancestors; the rescue rows' K / V at pos + row."""
    M = len(toks); L = self._ver(M); xbv = self.xbvs[M]; assert 2 <= M <= self.M and pos + M <= self.TMAX
    self._commit_flush()                                                  # (a folded commit no draft pass took)
    if NPU_IDS and not getattr(self, "_prompt_pass", False):              # vt holds [cur, drafts] (accept / pick_id wrote them)
      if self.need_feats: assert OA.host_invalidate(self.vtok).numpy()[:M].tolist() == [int(t) for t in toks], ("vt", OA.host_invalidate(self.vtok).numpy()[:M], toks)
      self._vemb(M)
    else: poke(xbv[0], self.W.embed(list(toks)).astype(np.float32))       # rows M.. stay zero (no kernel writes them)
    for i, t in enumerate(self.aix): poke(t, np.array([i, pos], np.int32))
    poke(self.posb, np.array([pos], np.int32))
    if self.tree: assert M == self.M or self.leaf; poke(self.treeb, spec_tree.table(tree) if tree is not None else spec_tree.chain_table(M))
    if self.blk > 1: self._verify_blocks(M, L)
    else:
      zc = self.pre.zc; late = STREAM_LATE and bool(zc) and bool(zc.get("stream"))   # streamed layers: refill / prefetch after the next submit
      L.prefetch(0)
      if late: zc["late_refill"] = True
      for l in range(R.NL):
        if l + 1 < R.NL and not late: L.prefetch(l + 1)
        slot, meta, sm = L.load(l); par = l % 2
        if R.LAYER_TYPES[l] == "full":
          self.meta_att = meta
          if (M, par) not in self.jit_vatt: self.jit_vatt[(M, par)] = TinyJit(functools.partial(self.vatt_step, M, par))
          self.jit_vatt[(M, par)](self.aix[self.aidx[l]], *[slot[k] for k in self.ATT_LIN])
        else:
          self.meta_gdn = meta
          if (M, par) not in self.jit_vgdn: self.jit_vgdn[(M, par)] = TinyJit(functools.partial(self.vgdn_step, M, par))
          self.jit_vgdn[(M, par)](self.sidx[self.gidx[l]], *[slot[k] for k in self.GDN_LIN])
        if late: L.after_submit(l)
      if late: zc["late_refill"] = False
    if not NPU_IDS or self.dump is not None or getattr(self, "_prompt_pass", False): self.v_hidden = OA.host_invalidate(xbv[0]).numpy()[:M].copy()   # the MTP draft head's input
    if head is None: return None                                          # (the last layer, parity 1, wrote xbv[0])
    return self.head_top(xbv[0], L)[0] if head == "top" else self.logits_rows(xbv[0], L=L)
  def _verify_blocks(self, M, L):
    """The 64 layers QWEN_LAYER_BLOCK at a time (vblk_step): block b's layers mapped into the slots of block parity b % 2 while block
    b - 1 runs (the backend's async tail), then the block as one JIT call."""
    k = self.blk
    for b in range(R.NL // k):
      bp = b % 2; types = tuple(R.LAYER_TYPES[b * k + j] for j in range(k)); args = []
      for j in range(k):
        l = b * k + j; views, meta = L.map_block(l, bp, j)
        if types[j] == "full": self.meta_att = meta; args += [self.aix[self.aidx[l]], *[views[n] for n in self.ATT_LIN]]
        else: self.meta_gdn = meta; args += [self.sidx[self.gidx[l]], *[views[n] for n in self.GDN_LIN]]
      if (key := (M, bp, types)) not in self.jit_vblk: self.jit_vblk[key] = TinyJit(functools.partial(self.vblk_step, M, types))
      jit = self.jit_vblk[key]
      if jit.cnt < 2:                                                     # the capture: the whole block as one graph, one chain
        with one_job(): jit(*args)
      else: jit(*args)
  def commit(self, a, pos, m=None, path=None, fold=False):
    """After verify(toks, pos) accepted a of its M tokens: the DeltaNet states and rings as after toks[:a] (posb still = pos).
    QWEN_SPEC_TREE: `path` = the committed rows (spec_tree.accept; None = the chain's first a): the state from the path's last
    row's bank, the ring from the path's raw rows, a committed rescue row's K / V rows moved to its position (gdn_commit_tree).
    The deferred commit (the state's updates wait for the next verify pass) leaves only copies: ring_commit (QWEN_RING_COMMIT=1,
    default) does them for any geometry from `cwords`, the same bytes as gdn_commit_tree(defer) / gdn_commit(ring_only). `fold`
    (the decode loop, a draft pass next): the kernel is not run here but by the next draft pass's job (QWEN_COMMIT_FOLD, `pcommit`);
    anything else that runs first (verify, flush, prefill) runs it on its own (_commit_flush)."""
    m = self.M if m is None else m; self._commit_flush()
    self.pend = self.defer                                                # (the deferred commit: the accepted updates wait for the next verify pass)
    nc = None
    if self.tree:                                                         # (leaf: pathb is also the next pass's pending path, gdn_tokl)
      nc = getattr(self, "pass_nc", m) if self.leaf and path is not None else (m if self.leaf else self.NC)
      poke(self.pathb, spec_tree.path_words(list(range(a)) if path is None else path, self.M if self.leaf else m, nc))
    elif not NPU_IDS or getattr(self, "_prompt_pass", False): poke(self.accb, np.array([a], np.int32))   # QWEN_NPU_IDS: accept wrote it (not for a prompt pass)
    if self.ringc: poke(self.cwords, QK.ring_commit_words(a, m, pos, path, nc if self.tree else None, self.M))
    c = self._commit_call(m)
    if fold and COMMIT_FOLD and self.ringc: self.pcommit = c
    else: c[1]()
  def _commit_call(self, m):
    """The commit's kernel for a pass on geometry m -> (its key, the call that issues it)."""
    if self.ringc:
      return ("ring",), lambda: self.ver.K.ring_commit(self.Call, self.rawm, self.Kall, self.Vall, self.cwords, self.C, R.CONV, self.NGDN, self.NATT, self.TMAX, R.NKV, R.HD)
    if self.tree:
      return ("tree", m), lambda: self.vers[m].K.gdn_commit_tree(self.Sall, self.banks, self.Call, self.rawm, self.Kall, self.Vall, self.pathb, self.posb,
                                                                 R.NV, R.DK, R.DV, self.C, R.CONV, m, self.NGDN, self.NATT, self.TMAX, R.NKV, R.HD, defer=self.leaf)
    return ("chain", m), lambda: self.vers[m].K.gdn_commit(self.Sall, self.banks, self.Call, self.rawm, self.accb, self.posb, R.NV, R.DK, R.DV, self.C, R.CONV, m, self.NGDN, ring_only=self.defer)
  def _commit_flush(self):
    """A commit left for the next draft job (commit(fold=True)) that has not run: run it now, on its own."""
    c = self._take_commit()
    if c is not None: c[1]()
  def spec_generate(self, first, pos, max_new, drafter, stats=None, cb=None):
    """Greedy speculative decoding from `first` (generated, not yet fed) at `pos`: each pass verifies [cur, d1..d(M-1)]
    (drafter(cur, pos, out) -> M - 1 ids), keeps the drafts the model agrees with plus its own next token. Returns the new ids
    (the same as plain greedy decoding, up to the kernels' rounding)."""
    out, cur = [first], first; st = stats if stats is not None else {}
    mtp = drafter == "mtp"; t_start = time.perf_counter()
    flog = open(self.spec_log, "a") if self.spec_log and mtp else None
    if flog:
      flog.write(json.dumps({"run": dict(self.spec_meta, model=R.MODEL, M=self.M, geos=self.geos, kmax=self.kmax, tau=self.tau,
        policy=f"leaf tc {self.tc} tl {self.tl}" if self.leaf else "threshold", draft_parts=self.draft_parts, parts_policy=self.parts_policy, head_top3=self.head_top3,
        tree=os.environ.get("QWEN_SPEC_TREE", "off"),
        head_c0=[p_["c0"] for p_ in self.dhead["parts"]], pos0=pos,
        **(dict(drafter=dict(cache=self.mtp_cache, embed=self.mtp_emb, head=self.mtp_head)) if self.mtp_x else {}), max_new=max_new, time=time.time(),
        env={k: v for k, v in os.environ.items() if k.startswith("QWEN_")})}) + "\n")
    if mtp:                                                               # the MTP cache over the prompt, then the first drafts
      t0 = time.perf_counter(); n = len(self.pre_hidden); nxt_all = self.ids_prompt[1:] + [first]
      cm = max(self.mtp_geos)
      for c0 in range(0, n, cm):
        c1 = min(n, c0 + cm)
        if c1 < n: self.mtp_pass(self.mtp_rows(self.pre_hidden[c0:c1], nxt_all[c0:c1]), c0)
        else: dnext = self.mtp_draft(self.pre_hidden[c0:c1], nxt_all[c0:c1], c0, self.kmax)
      st["t_draft"] = st.get("t_draft", 0.0) + time.perf_counter() - t0
      if NPU_IDS: poke(self.vtok, np.array(([first] + dnext + [dnext[-1]] * 16)[:16], np.int32))   # the prompt's: the last host write of ids
    while len(out) < max_new and cur not in EOS:
      if flog: drec = self.draft_rec
      if mtp: d = dnext
      else: d = list(drafter(cur, pos, out))[:self.M - 1]; d += [cur] * (self.M - 1 - len(d))
      nd = len(d); tree = None
      if self.leaf:                                                       # the leaf tree: the chain + the leaves (best path probability) in the rows left
        tree = spec_tree.build_leaf(cur, d[:nd], self.draft_t3, self.draft_p, self.M, self.tl); m = tree.M; toks = tree.toks; self.pass_nc = 1 + nd
      elif self.tree:                                                     # the tree: the chain (padded to NC rows) + the rescue rows from the drafts' top-3
        tree = spec_tree.build(cur, d[:nd], self.draft_t3, self.tree, self.M, spec_tree.TREE_NR); m = self.M; toks = tree.toks; d = toks[1:self.NC]
      else:
        m = min(g for g in self.geos if g >= 1 + nd); d = d + [d[-1]] * (m - 1 - nd); toks = [cur] + d   # padded to a kept geometry
      if pos + m > self.TMAX: break
      if self.leaf and NPU_IDS: poke(self.vtok, np.array((toks + [toks[-1]] * 16)[:16], np.int32))   # the leaf tree: vt = the tree's rows (the leaves are host picks)
      t0v = t0 = time.perf_counter(); g = self.verify(toks, pos, tree=tree); tv = time.perf_counter() - t0
      if tree is not None:                                                # the longest root path the model agrees with (spec_tree.accept)
        path, new, rej = spec_tree.accept(tree, g); a = len(new); ac = (rej or (self.pass_nc if self.leaf else self.NC)) - 1   # ac: chain drafts accepted
        if self.leaf and NPU_IDS:                                         # the drafter's catch-up inputs: nxt = the new tokens, the path's hidden rows
          poke(self.nxtb, np.array((new + [new[-1]] * 16)[:16], np.int32))
          if path[-1] != a - 1: self._row_move(self.xbvs[m][0], path[-1], a - 1)   # (a leaf's row into the path's last position)
      else:
        a = 1
        while a < m and d[a - 1] == int(g[a - 1]): a += 1
        path, new, ac = list(range(a)), d[:a - 1] + [int(g[a - 1])], a - 1
        if NPU_IDS:                                                       # the device's acceptance (accb, vt[0], nxt), checked against the host's
          self._vacc(m); o_ = OA.host_invalidate(self.acco).numpy(); ad = int(o_[0])
          assert (ad, [int(v) for v in o_[2:2 + ad]]) == (a, new), ("accept differs", o_.tolist(), a, new)
      if mtp:                                                             # (draft depth, probability, accepted) for the threshold's choice
        st.setdefault("draft_log", []).extend((j, self.draft_p[j], j < ac) for j in range(nd))
      st.setdefault("rows", []).append(m)
      n0 = len(out)
      for t in new:
        out.append(t)
        if t in EOS or len(out) >= max_new: break
      more = mtp and len(out) < max_new and out[-1] not in EOS            # a draft pass next: the commit rides in its job (QWEN_COMMIT_FOLD)
      tc = time.perf_counter(); self.commit(a, pos, m, path, fold=more); tc = time.perf_counter() - tc
      if cb is not None: cb(out[n0:], tv)
      if more:                                                            # the MTP cache caught up on the committed path's hidden rows
        t0 = time.perf_counter(); dnext = self.mtp_draft_dev(m, a, pos, self.kmax) if NPU_IDS else self.mtp_draft(self.v_hidden[path], new, pos, self.kmax)
        st["t_draft"] = st.get("t_draft", 0.0) + time.perf_counter() - t0
      if tree is not None:
        hit = path[-1] >= (self.pass_nc if self.leaf else self.NC); st.setdefault("rescue_hits", 0); st["rescue_hits"] += int(hit)
        trec = dict(rescue=[(r, j, k, tree.toks[r]) for r, j, k in tree.rescue], path=path, hit=int(path[-1]) if hit else -1)
      if flog:                                                            # (pass i's drafts were drawn at the end of pass i - 1)
        if "t3" in drec: drec["r3"] = [top3_rank(drec["t3"][j], int(g[j])) for j in range(nd)]   # the model's token's rank in draft j's top-3 (-1: absent)
        flog.write(json.dumps(dict(i=st.get("passes", 0), pos=pos, m=m, nd=nd, a=a, p=self.draft_p, d=d[:nd], g=[int(v) for v in g[:m]],
          tv=tv, tc=tc, t_rest=time.perf_counter() - t0v - tv, **drec, **(dict(tree=trec) if tree is not None else {}))) + "\n")
      cur, pos = out[-1], pos + a
      st.setdefault("passes", 0); st["passes"] += 1; st.setdefault("accepted", 0); st["accepted"] += a - 1; st.setdefault("t_verify", 0.0); st["t_verify"] += tv
    self._commit_flush()                                                  # (a commit no draft pass took: run it)
    if flog:
      flog.write(json.dumps({"end": dict(tokens=len(out), dt=time.perf_counter() - t_start, passes=st.get("passes", 0), out_ids=[int(v) for v in out])}) + "\n"); flog.close()
    return out

  # ---- the head
  def _head_fp8_init(self, cache):
    """lm_head as E4M3 with 128 x 128 block scales (qwen38_pack.py --lm-head-fp8): each part in a weight buffer, mapped into a slot
    in turn and run through the decoders' block-scaled GEMM (rows mode, compact A from the final norm)."""
    zc = self.pre.zc; raw = zc["raw"]; z = np.load(os.path.join(cache, f"{R.HEAD}.npz")); self.head_parts = []
    for i, (g0, g1) in enumerate(z["parts"]):
      f = os.path.join(cache, f"{R.HEAD}_{i}.bin"); n = os.path.getsize(f); wid, mm = raw.wbuf_alloc(n); read_into(f, memoryview(mm), n, drop=True)
      views = [Tensor.from_blob(zc["slots"][par][1], (n,), dtype=dtypes.uint8, device=DEV) for par in range(2)]
      self.head_parts.append(dict(wid=wid, mm=mm, n=n, groups=int(g1 - g0), views=views, c0=int(g0) * 48))
    self.head_n = int(z["n"][0]); self.mhead = dict(parts=self.head_parts, kw=TERN_KW, sid=None, n=self.head_n); K = self.dec.K; w1 = 1.0 + np.load(os.path.join(cache, "outside_small.npz"))["norm"]
    if R.HAD: w1 = (w1 * self.dec.signs[R.H]).astype(np.float32)          # the head's input signs in the final norm's 1 + w (as the layers')
    self.norm_w1 = K.buf("head_norm_w1", R.H, dtypes.float32).assign(dev(w1)).realize()
  def logits_fp8(self, L, x, row):
    K = self.dec.K; zc = self.pre.zc; raw = zc["raw"]
    xr = K.rows_buf("head_x", R.H).assign(x[row:row + 1].pad(((0, self.dec.R - 1), (0, 0)))).realize()   # the row as a decode row 0
    a = K.rms_a(xr, self.norm_w1, R.H, "a_head"); out = []
    for i, p_ in enumerate(self.head_parts):
      par = i % 2; raw.slot_map(zc["slots"][par][0], p_["wid"], 0, p_["n"])
      ct = OA.gemm_gs(a, p_["views"][par], ks=KS, ns=3, nrb=self.dec.nrb, nslices=R.H // (4 * KS), ngroups=p_["groups"], b8=True, bscale=True, q8=Q8MODE, scales=SCALES, piece=self.dec.piece,
                      rows=1, out=K.buf(f"ct_head{i}", p_["groups"] * self.dec.nrb * 3 * 192, dtypes.float32), **TERN_KW).realize()
      out.append(OA.host_invalidate(K.unpack(ct, p_["groups"] * 48, f"head_rows{i}")).numpy()[0].copy())
    return np.concatenate(out)[:self.head_n]
  def logits(self, L, x, row):
    if self.head == "fp8": return self.logits_fp8(L, x, row)
    lg = self.logits_fp16(L, x, row)
    if self.head == "both":
      l8 = self.logits_fp8(L, x, row); t16, t8 = np.argsort(-lg)[:8], np.argsort(-l8)[:8]
      print(f"       head fp8 vs fp16: argmax {'same' if t16[0] == t8[0] else 'DIFFERENT'} | top-8 overlap {len(set(t16) & set(t8))}/8 | "
            f"max|d| {np.abs(l8 - lg).max():.3f} (max|logit| {np.abs(lg).max():.1f}) | top-1 margin fp16 {lg[t16[0]] - lg[t16[1]]:.2f}", flush=True)
    return lg
  def logits_fp16(self, L, x, row):
    """The last real row's logits: final norm -> lm_head on the fp16 GEMM (ks 24, 6 strips), the panels streamed in chunks -> [N] float32."""
    h = L.rms(x, self.norm_w)[row:row + 1]                                # [1, 5120] -> a 24-row A (nrb 2), K padded to 5184
    a = h.pad(((0, 23), (0, self.K - R.H))).cast(dtypes.half).reshape(2, 3, 4, self.K // 96, 24, 4).permute(3, 0, 4, 1, 2, 5).contiguous().bitcast(dtypes.uint16).realize()
    ng = self.npad // 96; out = np.empty(96 * ng, np.float32); vas = [t.uop.buffer._buf.va for t in self.lm_slots]
    chunks = [(g0, min(self.LMG, ng - g0)) for g0 in range(0, ng, self.LMG)]
    def copy(i): g0, n = chunks[i]; memmove_threads(vas[i % 2], self.lm_pin.ctypes.data + g0 * self.gbytes, n * self.gbytes)
    th = threading.Thread(target=copy, args=(0,)); th.start()
    for i, (g0, n) in enumerate(chunks):
      th.join()                                                           # chunk i is in slot i % 2; the other slot's GEMM (i - 1) is done
      if i + 1 < len(chunks): th = threading.Thread(target=copy, args=(i + 1,)); th.start()
      ct = OA.gemm_gs(a, self.lm_slots[i % 2], ks=24, ns=6, nrb=2, nslices=self.K // 96, ngroups=n, piece=2)
      c = ct.reshape(n, 2, 6, 3, 4, 4, 4).permute(1, 3, 5, 0, 2, 4, 6).reshape(24, 96 * n)[0]
      out[96 * g0:96 * (g0 + n)] = OA.host_invalidate(c.realize()).numpy()
    return out[:self.N]

  # ---- prefill: the prompt's rows at once, the layer bodies under TinyJit per (type, slot parity) like decode
  def pre_gdn_step(self, x, *rest):
    L = self.pre; nl = len(self.GDN_LIN); slot = dict(zip(self.GDN_LIN, rest[:nl])); sm = dict(zip(self.GDN_SMALL, rest[nl:]))
    y, (S, cs) = L.gdn_body(x, slot, self.meta_gdn, sm)
    return y.realize(), S.realize(), cs.realize()

  def pre_att_step(self, x, cos, sin, mask, *rest):
    L = self.pre; nl = len(self.ATT_LIN); ns = len(self.ATT_SMALL); slot = dict(zip(self.ATT_LIN, rest[:nl])); sm = dict(zip(self.ATT_SMALL, rest[nl:nl + ns]))
    if ATT_PRE == "csrc": return self.pre_att_csrc(L, x, slot, sm, rest[nl + ns])
    q, k, v, gate = qkv_proj(L, x, slot, self.meta_att, sm, cos, sin); x = L.K.rows_buf("x_in", R.H)
    return attend(L, x, q, gate, k, v, mask, slot, self.meta_att, sm).realize(), k.realize(), v.realize()

  def pre_att_csrc(self, L, x, slot, sm, aix):
    """QWEN_ATTN_PREFILL=csrc: the prefill attention layer with the attention by Kernels.attn_prefill (every row by DMA; the k / v
    rows written into the cache of the layer aix[0] by the kernel). `aix` = the layer's (index, 0), poked by prefill()."""
    meta = self.meta_att; x = L.K.rows_buf("x_in", R.H).assign(x).realize()
    a_h = L.K.rms_a(x, L.K.hold("w_in", sm["input_layernorm_weight"]), R.H, "a_h")
    QC, KC, OC = R.NH * 2 * R.HD, 2 * R.NKV * R.HD, R.NH * R.HD
    ct = L.lin_cts(a_h, slot, meta, ("q", "kv"))                               # one GEMM when the cache fuses q | k|v
    L.K.unpack(ct["q"][0], QC, "q_rows", goff=ct["q"][1]); L.K.unpack(ct["kv"][0], KC, "kv_rows", goff=ct["kv"][1])
    ah = L.K.hold("aix_held", aix)
    oc = L.K.attn_prefill(L.K.bufs["q_rows"], L.K.bufs["kv_rows"], self.QN, self.KN, self.ROPE, self.Kall, self.Vall, ah, R.NH, R.NKV, R.HD, self.TMAX, R.ROT)
    x = L.K.resid(x, L.lin_ct(L.K.rms_a(oc, None, OC, "a_o", norm=False), slot, meta, "o"), R.H, "x1")
    return L.mlp(x, slot, meta, sm["post_attention_layernorm_weight"]).realize()

  def prefill_chunked(self, ids):
    """The prompt through the verify path, up to max(geometries) tokens a pass (each pass one streaming of the weights; the
    DeltaNet states and caches carried like a decode), the last chunk padded and committed to its real length -> the logits of
    the last prompt token. The rows-mode (compact) layouts only: the Q8_0 models' prefill."""
    self.ids_prompt = list(ids); n = len(ids); m = max(self.geos); lg = None
    self.reset()                                                          # (the first chunk reads Sall as the state before position 0: zero it --
    #                                                                        spec_warmup's bank-path passes, a previous prompt or plain steps left one there)
    self._prompt_pass = True                                              # QWEN_NPU_IDS: the prompt's ids / rows come from the host
    for c0 in range(0, n, m):
      chunk = list(ids[c0:c0 + m]); a = len(chunk); mm = min(g for g in self.geos if g >= max(2, a))
      lg = self.verify(chunk + [chunk[-1]] * (mm - a), c0, head="full" if c0 + m >= n else None); self.commit(a, c0, mm)
      if self.dump is not None or self.MTPL: self.pre_hidden = np.concatenate([getattr(self, "pre_hidden", np.zeros((0, R.H), np.float32))[:c0], self.v_hidden[:a]])
    if self.dump is not None: self.dump.extend(self.pre_hidden)           # QWEN_DUMP_HIDDEN: every prompt position
    self._prompt_pass = False
    return lg[a - 1]
  def reset(self):
    """Before an unrelated prompt in the same process (a server's next request): the DeltaNet states to zero (a chunked prefill
    starts from them; the default prefill overwrites them anyway). The rings and the K / V caches are only read at positions the
    new prompt has written."""
    self.Sall.assign(Tensor.zeros(self.Sall.shape[0], device=DEV, dtype=dtypes.float32)).realize()
  def prefill(self, ids, nlayers=None):
    self._commit_flush()
    if getattr(self, "defer", False): poke(self.accb, np.zeros(1, np.int32)); self.pend = False   # no update pending for the next verify pass (gdn_defer_src)
    if getattr(self, "leaf", False) and self.M: poke(self.pathb, np.zeros(4 + self.M, np.int32))   # (the leaf tree's gdn_tokl: its pending path)
    if R.Q8 or os.environ.get("QWEN_PREFILL") == "chunked": return self.prefill_chunked(ids)
    nlayers = R.NL if nlayers is None else nlayers
    n = len(ids); L = self.pre; assert n <= self.TMAX; self.ids_prompt = list(ids)
    x = np.zeros((L.R, R.H), np.float32); x[:n] = self.W.embed(ids); x = dev(x)
    cos, sin = rope_tables(0, L.R); i, j = np.arange(L.R)[:, None], np.arange(L.R)[None]
    mask = dev(np.where((j <= i) & (i < n), 0.0, -1e30).astype(np.float32))
    jg, ja = [None, None], [None, None]
    csrc = ATT_PRE == "csrc"
    if csrc:                                                              # attn_pkv writes layer aix[0]'s cache rows 0..n-1
      for i_, t in enumerate(self.aix): poke(t, np.array([i_, 0], np.int32))
    L.prefetch(0)
    for l in range(nlayers):
      if l + 1 < nlayers: L.prefetch(l + 1)
      slot, meta, sm = L.load(l); par = l % 2
      if os.environ.get("QWEN_DEBUG"):
        j = (ja if R.LAYER_TYPES[l] == "full" else jg)[par]; print(f"   prefill layer {l:2d} {R.LAYER_TYPES[l]:6s} jit call #{j.cnt if j is not None else 0}", flush=True)
      if R.LAYER_TYPES[l] == "full":
        self.meta_att = meta
        if ja[par] is None: ja[par] = TinyJit(self.pre_att_step)
        if csrc: x = ja[par](x, cos, sin, mask, *[slot[k_] for k_ in self.ATT_LIN], *[sm[k_] for k_ in self.ATT_SMALL], self.aix[self.aidx[l]])
        else:
          x, k, v = ja[par](x, cos, sin, mask, *[slot[k_] for k_ in self.ATT_LIN], *[sm[k_] for k_ in self.ATT_SMALL])
          i = self.aidx[l]; base = i * self.TMAX * R.NKV * R.HD; rowsz = R.NKV * R.HD
          self.Kall[base:base + n * rowsz].assign(k[:n].reshape(-1)).realize(); self.Vall[base:base + n * rowsz].assign(v[:n].reshape(-1)).realize()
      else:
        self.meta_gdn = meta
        if jg[par] is None: jg[par] = TinyJit(self.pre_gdn_step)
        x, S, cs = jg[par](x, *[slot[k_] for k_ in self.GDN_LIN], *[sm[k_] for k_ in self.GDN_SMALL])
        g = self.gidx[l]; self.Sall[g * R.NV * R.DK * R.DV:(g + 1) * R.NV * R.DK * R.DV].assign(S.reshape(-1)).realize()
        for k_ in range(R.CONV - 1):                                       # the last CONV-1 raw rows into the layer's ring (token t at slot t % CONV)
          if (tok := n - (R.CONV - 1) + k_) >= 0: self.Call[(r := (g * R.CONV + tok % R.CONV) * self.CP):r + self.C].assign(cs[k_]).realize()
      x = x.clone().realize()
    if self.dump is not None or getattr(self, "MTPL", None): self.pre_hidden = OA.host_invalidate(x).numpy()[:n].copy()
    if self.dump is not None: self.dump.extend(self.pre_hidden)   # QWEN_DUMP_HIDDEN: every prompt position
    return self.logits(L, x, n - 1)

  # ---- decode: one token, the layer bodies under TinyJit per (type, slot parity)
  def gdn_step(self, par, idx, *rest):
    """Decode (layer parity `par`: reads xb[par], writes xb[1-par]): the layer's state and conv ring live in the persistent stacks
    at `idx` (a 1-int persistent buffer), the position in `posb`, the small weights in their stacks."""
    L = self.dec; nl = len(self.GDN_LIN); slot = dict(zip(self.GDN_LIN, rest[:nl])); sm = dict(zip(self.GDN_SMALL, rest[nl:]))
    y, _ = L.gdn_body(None, slot, self.meta_gdn, sm, (self.Sall, self.Call, idx, self.posb, self.Wab, self.Cwt, self.Nw, self.Adt, self.Win_g, self.Wpost_g), xin=self.xb[par], out=f"xio{1 - par}")
    return y.realize()

  def att_step(self, par, aix, cos, sin, *rest):
    """Decode (layer parity `par`: reads xb[par], writes xb[1-par]): q / k / v of the one real row, the hand-written attention
    against the cache stack (the new k / v rows written at pos by the kernel), the gate, o_proj and the MLP."""
    L = self.dec; nl = len(self.ATT_LIN); slot = dict(zip(self.ATT_LIN, rest[:nl])); sm = dict(zip(self.ATT_SMALL, rest[nl:])); x = self.xb[par]
    ah = L.K.hold("aix_held", aix)                                                     # aix[0]: the attention layer's index into the stacks
    a_h = L.K.rms_a(x, (self.Win_a, ah), R.H, "a_h"); ct = L.lin_cts(a_h, slot, self.meta_att, ("q", "kv"))   # one GEMM when fused
    q_rows = L.K.unpack(ct["q"][0], R.NH * 2 * R.HD, "q_rows", goff=ct["q"][1])
    kv_rows = L.K.unpack(ct["kv"][0], 2 * R.NKV * R.HD, "kv_rows", goff=ct["kv"][1])
    oc = L.K.attn_dec2(q_rows, kv_rows, self.QN, self.KN, self.ROPE, self.Kall, self.Vall, ah, R.NH, R.NKV, R.HD, self.TMAX, R.ROT)
    x = L.K.resid(x, L.lin_ct(L.K.rms_a(oc, None, R.NH * R.HD, "a_o", norm=False), slot, self.meta_att, "o"), R.H, "x1")
    return L.mlp(x, slot, self.meta_att, (self.Wpost_a, ah), f"xio{1 - par}").realize()

  def flush(self):
    """The deferred commit's pending updates into Sall now (gdn_flush), none left pending: before a plain decode step, which reads
    Sall -- the Q8_0 models' prompt goes through the verify path (prefill_chunked), and a step may follow a verify pass."""
    self._commit_flush()
    if not getattr(self, "pend", False): return
    K = self.ver.K; acc = self.pathb if self.leaf else self.accb
    K.gdn_flush(self.Sall, self.banks, acc, R.NV, R.DK, R.DV, self.NGDN, self.M, tree=self.leaf)
    poke(acc, np.zeros(4 + self.M if self.leaf else 1, np.int32)); self.pend = False
  def step(self, tok, pos):
    L = self.dec; assert pos < self.TMAX
    self.flush()
    x = np.zeros((L.R, R.H), np.float32); x[0] = self.W.embed([tok])[0]; self.xb[0].assign(dev(x)).realize()
    if pos not in self.cos_dec: self.cos_dec[pos], self.sin_dec[pos] = rope_tables(pos, L.R)
    for i, t in enumerate(self.aix): t.assign(Tensor([i, pos], device=DEV, dtype=dtypes.int32)).realize()
    self.posb.assign(Tensor([pos], device=DEV, dtype=dtypes.int32)).realize()
    L.prefetch(0)
    for l in range(R.NL):
      if l + 1 < R.NL: L.prefetch(l + 1)
      slot, meta, sm = L.load(l); par = l % 2
      if R.LAYER_TYPES[l] == "full":
        self.meta_att = meta
        if self.jit_att[par] is None: self.jit_att[par] = TinyJit(functools.partial(self.att_step, par))
        self.jit_att[par](self.aix[self.aidx[l]], self.cos_dec[pos], self.sin_dec[pos], *[slot[k] for k in self.ATT_LIN], *[sm[k] for k in self.ATT_SMALL])
      else:
        self.meta_gdn = meta
        if self.jit_gdn[par] is None: self.jit_gdn[par] = TinyJit(functools.partial(self.gdn_step, par))
        self.jit_gdn[par](self.sidx[self.gidx[l]], *[slot[k] for k in self.GDN_LIN], *[sm[k] for k in self.GDN_SMALL])
    if self.dump is not None: self.dump.append(OA.host_invalidate(self.xb[0]).numpy()[0].copy())
    return self.logits(L, self.xb[0], 0)                                   # 64 layers: the last (parity 1) wrote xb[0]

def pin_cpus():
  """QWEN_CPUS (default auto): run the host on the CPU that takes the NPU's interrupt (/proc/interrupts' "aipu" line) and one other
  core of its speed. The driver completes every job group on that CPU: with the host there it stays awake, and a 4-row verify
  pass of bonsai2-27b took 0.513 s pinned against 0.565 s unpinned (the gaps between a job's kernels ~0.45 ms shorter a layer,
  and a fast core for the host). off: no pinning; a list "0,1": those CPUs."""
  v = os.environ.get("QWEN_CPUS", "auto")
  if v == "off" or not hasattr(os, "sched_setaffinity"): return None
  if v != "auto": cpus = {int(c) for c in v.split(",")}
  else:
    try:
      line = next(l for l in open("/proc/interrupts") if l.rstrip().endswith(" aipu"))
      counts = [int(x) for x in line.split()[1:1 + os.cpu_count()]]
      c0 = max(range(len(counts)), key=counts.__getitem__)
      fmax = lambda c: open(f"/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq").read().strip()
      cpus = {c0} | set([c for c in range(os.cpu_count()) if c != c0 and fmax(c) == fmax(c0)][:1])
    except (StopIteration, OSError, ValueError): return None
  os.sched_setaffinity(0, cpus); return sorted(cpus)

_LAT = []
def hold_cpu_latency():
  """QWEN_CPU_LATENCY (default 0, in us; off: none): a PM QoS request on /dev/cpu_dma_latency, held while the process lives. The NPU
  interrupt's CPU otherwise drops into idle states with 360-500 us exit latency between a job's groups, and the gaps between a
  job's kernels go from ~8 to 45-75 us (bonsai2-27b, 4-row verify 0.526 s -> 0.484 s with the host on CPU 1).
  The device is root-only by default: a udev rule, e.g. KERNEL=="cpu_dma_latency", GROUP="users", MODE="0660", lets a user hold it."""
  v = os.environ.get("QWEN_CPU_LATENCY", "0")
  if v == "off": return None
  try: f = open("/dev/cpu_dma_latency", "wb", buffering=0); f.write(int(v).to_bytes(4, "little")); _LAT.append(f); return int(v)
  except OSError as e:
    print(f"   (QWEN_CPU_LATENCY: no PM QoS request -- {e.strerror}; the gaps between a job's kernels may be ~10x longer)", flush=True); return None

def main():
  cpus = pin_cpus(); lat = hold_cpu_latency()
  ap = argparse.ArgumentParser(); ap.add_argument("prompt"); ap.add_argument("--max-new", type=int, default=32); ap.add_argument("--thinking", action="store_true")
  ap.add_argument("--tmax", type=int, default=512, help="the K / V caches' length (prompt + generated tokens)")
  ap.add_argument("--pin-gb", type=float, default=float(os.environ.get("QWEN_PIN_GB", "24")), help="GB of layer weights held in RAM (the rest streams from the NVMe each token)")
  ap.add_argument("--dir", default=os.environ.get("QWEN_DIR", "/mnt/ssd/qwen3.8-27b-fp8")); ap.add_argument("--cache", default=os.environ.get("QWEN_NPU", "/mnt/ssd/qwen3.8-27b-npu-s1"))
  ap.add_argument("--out", default=None, help="save the generated ids and every step's top-8 logits (npz) for the online comparison")
  ap.add_argument("--raw", action="store_true", help="the prompt as plain text (no chat template)")
  ap.add_argument("--ids", help="the prompt as comma-separated token ids (the `prompt` argument is then only a label; no tokenizer needed)")
  a = ap.parse_args()
  if a.ids:                                                               # e.g. another implementation's tokenisation, to compare its greedy tokens
    ids = [int(t) for t in a.ids.split(",")]
    try: tok = Tok(os.environ.get("QWEN_TOK", a.dir))
    except Exception: tok = type("Ids", (), {"decode": staticmethod(lambda ids: " ".join(str(i) for i in ids))})()
  else:
    tok = Tok(os.environ.get("QWEN_TOK", a.dir)); text = a.prompt if a.raw else chat_text(a.prompt, a.thinking); ids = tok.encode(text)   # QWEN_TOK: tokenizer.json's folder
  print(f"== {len(ids)} prompt tokens; greedy, up to {a.max_new} new{f'; host CPUs {cpus}' if cpus else ''}{f'; CPU latency {lat} us' if lat is not None else ''}", flush=True)
  t0 = time.perf_counter(); M = Model(len(ids), a.cache, a.dir, a.tmax, a.pin_gb); print(f"   model set up in {time.perf_counter() - t0:.0f} s", flush=True)
  M.spec_meta = dict(prompt=a.prompt, n_prompt=len(ids), thinking=a.thinking, prompt_ids=[int(v) for v in ids])   # QWEN_SPEC_LOG's run header
  t0 = time.perf_counter(); lg = M.prefill(ids); tp = time.perf_counter() - t0
  print(f"   prefill {len(ids)} tokens: {tp:.1f} s ({len(ids) / tp:.2f} tok/s)", flush=True)
  out, tops = [], []
  if M.M and M.MTPL:                                                       # speculative decoding: tokens arrive a verify pass at a time
    first = int(lg.argmax()); print(f"   [0] {first} {tok.decode([first])!r}", flush=True); st = {}; t0 = time.perf_counter()
    def cb(new, tv): print(f"       +{len(new)} {tok.decode(new)!r} (verify {tv:.2f} s)", flush=True)
    out = M.spec_generate(first, len(ids), a.max_new, "mtp" if M.MTPL else (lambda cur, pos, o: [cur] * (M.M - 1)), st, cb)
    dt = time.perf_counter() - t0
    print(f"== answer: {tok.decode(out)!r}\n   {len(out)} tokens in {dt:.1f} s after the first: {dt / max(1, len(out) - 1):.2f} s / token "
          f"({max(1, len(out) - 1) / dt:.2f} tok/s) | {st['passes']} verify passes, {st['accepted']} drafts accepted"
          + (f", {st.get('rescue_hits', 0)} rescue-row hits (QWEN_SPEC_TREE={os.environ.get('QWEN_SPEC_TREE')})" if M.tree else ""), flush=True)
    if getattr(M, "prof", None): print("   drafting: " + " ".join(f"{k} {v:.2f} s" for k, v in M.prof.items()), flush=True)
    if a.out: np.savez(a.out, prompt_ids=np.array(ids), out_ids=np.array(out), draft_log=np.array(st.get("draft_log", []), np.float64).reshape(-1, 3),
                       rows=np.array(st.get("rows", [])))
    return
  for i in range(a.max_new):
    nxt = int(lg.argmax()); top = np.argsort(-lg)[:8]; tops.append((top, lg[top])); out.append(nxt)
    print(f"   [{i}] {nxt} {tok.decode([nxt])!r}", flush=True)
    if nxt in EOS: break
    t0 = time.perf_counter(); lg = M.step(nxt, len(ids) + i); print(f"       step {time.perf_counter() - t0:.2f} s", flush=True)
  print("== answer:", repr(tok.decode(out)))
  if M.dump is not None:
    np.savez(os.environ["QWEN_DUMP_HIDDEN"], ids=np.array(ids + out), hidden=np.stack(M.dump).astype(np.float32), n_prompt=len(ids))
  if a.out: np.savez(a.out, prompt_ids=np.array(ids), out_ids=np.array(out), top_ids=np.array([t[0] for t in tops]), top_logits=np.array([t[1] for t in tops]))

if __name__ == "__main__": main()

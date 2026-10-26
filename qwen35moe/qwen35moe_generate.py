#!/usr/bin/env python3
"""Qwen3.5-35B-A3B (and 3.6) text generation on the NPU (greedy), from qwen35moe_pack.py's cache: the 40 layers as one TinyJit graph
each (the DeltaNet / attention mixers of qwen3.8-27b on the q8f GEMM tiles; the routing on the device -- seqmix/route.py "softmax+sg":
softmax, top 8, renormalised, the shared expert's sigmoid gate; the experts' gate / up on the qdot GEMV over their q4m streams, the
down on their q5n / q6n streams from silu(g) u, the shared expert's q8f the same as head 0) and the output head (Q6_K, the top-1 on the
device); the host embeds the token (token_embd's Q8_0 row) and maps the weights. Nothing syncs mid-token.

Weights: every layer's streams in a KMD weight buffer of its own (zero-copy, ~22 GB pinned); two window slots, layer l mapped into
slot l % 2 before it runs; the linears are fixed views into the slot per (slot, layer kind), a kind = (mixer type, the down stream's
code). The prompt goes through the decode step a token (the prefill proper is the next step).

    python3 qwen35moe_generate.py "prompt" [--max-new 32] [--thinking] [--cache /mnt/ssd/qwen35moe-npu]
"""
import argparse, functools, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import qwen35moe_npu as QN                                                 # noqa: E402  (the paths, QWEN_MODEL, Kernels, gemm, poke)
from qwen35moe_npu import R, Tensor, Device, OA, dtypes, dev, poke, gemm, Layers, ring_pitch, ATT_BT, QG, NE, TOPK, FF, H, C_QKV   # noqa: E402
import qwen35moe_ref as Q                                                  # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "seqmix")); import route as RT   # noqa: E402,E702
from tinygrad.engine.jit import TinyJit                                    # noqa: E402

JIT = os.environ.get("QWEN35MOE_JIT", "1") != "0"

CHUNK = 2 << 20
TOK = os.environ.get("QWEN35MOE_TOK", "/mnt/ssd/ornith-9b-npu")
IM_END = Q.IM_END


def read_into(path, mv, size, nt=8):
  """The file into the memoryview with nt parallel preads (the weight buffers are 2 MiB-aligned, written once)."""
  import threading
  def part(lo, hi):
    with open(path, "rb", buffering=0) as f:
      while lo < hi: lo += os.preadv(f.fileno(), [mv[lo:min(hi, lo + (64 << 20))]], lo)
  bounds = [size * i // nt for i in range(nt + 1)]; th = [threading.Thread(target=part, args=(bounds[i], bounds[i + 1])) for i in range(nt)]
  for t in th: t.start()
  for t in th: t.join()


class W:
  def __init__(self, st, meta, sm): self.st, self.meta, self.sm = st, meta, sm


class Model:
  def __init__(self, cache=QN.CACHE, gguf=Q.DEFAULT, tmax=1024, log=print):
    t0 = time.perf_counter(); self.Wt = R.Weights(gguf); self.tmax = tmax; self.cache = cache; NL = R.NL
    K = self.K = QN.Kernels(2, 24, R.EPS, real=1); self.Rr = 24
    self.metas, self.sms, self.kind = [], [], []
    for l in range(NL):
      z = np.load(os.path.join(cache, f"L{l}_small.npz")); meta = {str(k): tuple(int(v) for v in m) for k, m in zip(z["meta_names"], z["meta"])}
      self.metas.append(meta); self.sms.append({k: z[k] for k in z.files if not k.startswith("meta")}); self.kind.append((R.LAYER_TYPES[l], meta["down"][0]))
    raw = self.raw = Device[QN.DEV].raw; self.packs = {}
    for l in range(NL):
      sizes = {k: os.path.getsize(os.path.join(cache, f"L{l}_{k}.bin")) for k in sorted(self.metas[l])}; offs, o = {}, 0
      for k, sz in sizes.items(): offs[k] = o; o += -(-(sz + 16) // CHUNK) * CHUNK
      self.packs[l] = (sizes, offs, o)
    for l in range(NL): assert self.packs[l][1] == self.packs[self.kind.index(self.kind[l])][1], (l, "a layer's streams differ from its kind's first layer's")
    big = max(p[2] for p in self.packs.values()); self.NSLOT = 2; self.slots = [raw.slot_alloc(big) for _ in range(self.NSLOT)]
    self.wbufs = {}
    for l in range(NL):
      sizes, offs, n = self.packs[l]; wid, mm = raw.wbuf_alloc(n)
      for k, sz in sizes.items(): read_into(os.path.join(cache, f"L{l}_{k}.bin"), memoryview(mm)[offs[k]:offs[k] + sz], sz)
      self.wbufs[l] = (wid, mm)
    self.views = {}
    for par in range(self.NSLOT):
      for kd in set(self.kind):
        l0 = self.kind.index(kd); sizes, offs, _ = self.packs[l0]
        self.views[(par, kd)] = {k: Tensor.from_blob(self.slots[par][1] + offs[k], (sz,), dtype=dtypes.uint8, device=QN.DEV) for k, sz in sizes.items()}
    log(f"weights: {NL} layers in weight buffers, {sum(p[2] for p in self.packs.values()) / 1e9:.1f} GB, two {big / 1e6:.0f} MB slots ({time.perf_counter() - t0:.0f} s)", flush=True)
    # the small weights as stacks the kernels index by the layer's position in its type (gdn_tok3 / attn_dec2's conventions)
    self.gidx = {l: i for i, l in enumerate(l for l in range(NL) if R.LAYER_TYPES[l] == "linear")}; self.aidx = {l: i for i, l in enumerate(l for l in range(NL) if R.LAYER_TYPES[l] == "full")}
    NG, NA = len(self.gidx), len(self.aidx); CP = self.CP = ring_pitch(C_QKV)
    self.Sall = K.buf("S_all", NG * R.NV * R.DK * R.DV, dtypes.float32); self.Call = K.buf("C_all", NG * R.CONV * CP, dtypes.float32)
    self.sidx = [K.buf(f"sidx{i}", 1, dtypes.int32) for i in range(NG)]; self.aix = [K.buf(f"aidx{i}", 2, dtypes.int32) for i in range(NA)]
    for i, t in enumerate(self.sidx): poke(t, np.array([i], np.int32))
    ksz = (NA * tmax + ATT_BT) * R.NKV * R.HD; self.Kall, self.Vall = K.buf("K_all", ksz, dtypes.float32), K.buf("V_all", ksz, dtypes.float32)
    wab = np.zeros((NG, Layers.ab3_size()), np.float32); adt = np.zeros((NG, 2 * R.NV), np.float32); cwt = np.zeros((NG, R.CONV, CP), np.float32); nw = np.zeros((NG, R.DV), np.float32)
    win_g, wpost_g = np.zeros((NG, H), np.float32), np.zeros((NG, H), np.float32); win_a, wpost_a = np.zeros((NA, H), np.float32), np.zeros((NA, H), np.float32)
    qn, kn = np.zeros((NA, R.HD), np.float32), np.zeros((NA, R.HD), np.float32); router = np.zeros((NL, NE, H), np.float32); self.shg = []
    for l in range(NL):
      sm = self.sms[l]; router[l] = sm["router"]; self.shg.append(sm["shexp_gate"].astype(np.float64))
      if R.LAYER_TYPES[l] == "linear":
        i = self.gidx[l]; win_g[i], wpost_g[i] = 1.0 + sm["input_layernorm_weight"], 1.0 + sm["post_attention_layernorm_weight"]
        wab[i] = Layers.ab3((np.concatenate([sm["linear_attn_in_proj_a_weight"], sm["linear_attn_in_proj_b_weight"]]) * (1.0 + sm["input_layernorm_weight"])[None]).astype(np.float32))
        adt[i] = Layers.adt(sm); cwt[i, :, :C_QKV] = sm["linear_attn_conv1d_weight"][:, 0, :].T; nw[i] = sm["linear_attn_norm_weight"]
      else:
        i = self.aidx[l]; win_a[i], wpost_a[i] = 1.0 + sm["input_layernorm_weight"], 1.0 + sm["post_attention_layernorm_weight"]
        qn[i], kn[i] = 1.0 + sm["self_attn_q_norm_weight"], 1.0 + sm["self_attn_k_norm_weight"]
    hold = lambda tag, a: K.hold(tag, dev(np.ascontiguousarray(a)))
    self.Wab, self.Adt, self.Cwt, self.Nw = hold("Wab_all", wab), hold("Adt_all", adt), hold("Cwt_all", cwt), hold("Nw_all", nw)
    self.Win_g, self.Wpost_g, self.Win_a, self.Wpost_a = hold("Win_g", win_g), hold("Wpost_g", wpost_g), hold("Win_a", win_a), hold("Wpost_a", wpost_a)
    self.QN, self.KN = hold("qn_all", qn), hold("kn_all", kn); self.router = hold("router_all", router)
    shg8 = np.zeros((NL, 8, H), np.float32); shg8[:, 0] = np.stack(self.shg); self.shg8 = hold("shg_all", shg8)             # the shared gate as a gemv32 of 8 columns
    self.wpost_l = [dev((1.0 + self.sms[l]["post_attention_layernorm_weight"]).astype(np.float32)) for l in range(NL)]
    inv = 1.0 / (R.THETA ** (np.arange(0, R.ROT, 2, dtype=np.float64) / R.ROT)); f = np.arange(tmax, dtype=np.float64)[:, None] * inv[None]; emb = np.concatenate([f, f], -1)
    self.ROPE = hold("rope_tab", np.concatenate([np.cos(emb), np.sin(emb)], -1).astype(np.float32).ravel())
    self.lidx = [K.buf(f"lidx{l}", 1, dtypes.int32) for l in range(NL)]
    for l, t in enumerate(self.lidx): poke(t, np.array([l], np.int32))
    # the head: the Q6_K parts as one q6 stream, the top-1 kept per task (test_qhead's path)
    z = np.load(os.path.join(cache, "head.npz")); parts = [tuple(int(v) for v in r) for r in z["parts"]]; self.Vn = sum(p[1] for p in parts)
    self.head_all = dev(np.concatenate([np.fromfile(os.path.join(cache, f"head{j}.bin"), np.uint8) for j in range(len(parts))])); self.w_out = dev(z["w_out"].astype(np.float32))
    self.sel0 = dev(np.array([0] + [-1] * 15, np.int32)); self.d_head = dev(QG.qgemvk_desc("q6", H)); self.tops = K.buf("top_head0", 48, dtypes.float32)
    self.pk_head = Tensor.zeros(H // 128 * QG.SL_REC // 4, device=QN.DEV).contiguous().realize()
    # the layer buffers (qwen35moe_npu.Layer's), one set
    cs = QN.CS(K); self.ng_gu = self.metas[0]["gate"][2] // 48
    self.sel_gu, self.sel_dn, self.sel1, self.wt = K.buf("sel_gu", 16, dtypes.int32), K.buf("sel_dn", 16, dtypes.int32), K.buf("sel1", 16, dtypes.int32), K.buf("moe_w", 12 * (TOPK + 1), dtypes.float32)
    poke(self.sel1, np.array([0] + [-1] * 15, np.int32)); self.posb = K.buf("posb", 1, dtypes.int32)
    self.xio = [K.rows_buf("xio0", H), K.rows_buf("xio1", H)]; self.x1 = K.rows_buf("x1", H)
    self.NPF = QG.qgemvk_np(FF); self.y_gu = K.buf("y_gu", 2 * TOPK * self.NPF, dtypes.float32); self.y_sgu = K.buf("y_sgu", 2 * self.NPF, dtypes.float32)
    self.y_dn = K.buf("y_dn", (TOPK + 1) * QG.qgemvk_np(H), dtypes.float32); self.pk = Tensor.zeros(H // 128 * QG.SL_REC // 4, device=QN.DEV).contiguous().realize()
    self.jits = {}; self.jd = Tensor.zeros(16, device=QN.DEV).contiguous().realize()
    for k, n in (("qkv", C_QKV), ("z", R.NV * R.DV), ("o", H), ("q", R.NH * 2 * R.HD), ("kv", 2 * R.NKV * R.HD)): K.buf(f"ct_{k}", -(-n // 48) * cs, dtypes.float32)
    log(f"ready in {time.perf_counter() - t0:.0f} s", flush=True)

  def layer_w(self, l):
    try: self.raw.slot_map(self.slots[l % self.NSLOT][0], self.wbufs[l][0], 0, self.packs[l][2], drain=False)
    except TypeError: self.raw.slot_map(self.slots[l % self.NSLOT][0], self.wbufs[l][0], 0, self.packs[l][2])
    return W(self.views[(l % self.NSLOT, self.kind[l])], self.metas[l], self.sms[l])

  def qgemv(self, kd, N, Kd, y, w, sel, xs, nw, heads=1, act="norm", gu_np=0, yoff=0, nbuf=3):
    K = self.K; key = f"qgemv|{kd}|{N}|{Kd}|{heads}|{act}|{gu_np}|{yoff}|{nbuf}"
    src = QG.qgemvk_src(kd, N, Kd, heads=heads, act=act, gu_np=gu_np, yoff=yoff, eps=R.EPS, nbuf=nbuf)
    return K.call(key, src, y, self.pk, w, sel, K.desc(f"q_desc|{kd}|{Kd}", lambda: QG.qgemvk_desc(kd, Kd)), y, xs, nw)

  def whole(self, l, _):
    """The layer as kernels on persistent buffers (one TinyJit graph; a dummy input): xio[l % 2] -> xio[(l + 1) % 2]. The position is
    in posb / aix (poked before), the layer's weights in slot l % 2 (mapped before)."""
    K = self.K; Wl = W(self.views[(l % self.NSLOT, self.kind[l])], self.metas[l], self.sms[l]); out = self.xio[(l + 1) % 2]; x = self.xio[l % 2]
    if R.LAYER_TYPES[l] == "linear":
      i = self.gidx[l]; a_h = K.rms_a(x, (self.Win_g, self.sidx[i]), H, "a_h")
      ct_qkv = gemm(K, Wl, "qkv", a_h, 1, K.bufs["ct_qkv"]); ct_z = gemm(K, Wl, "z", a_h, 1, K.bufs["ct_z"])
      a_o = K.gdn_tok3(self.Sall, self.Call, self.sidx[i], self.posb, ct_qkv, ct_z, x, self.Wab, self.Adt, self.Cwt, self.Nw, R.NV, R.NK, R.DK, R.DV, C_QKV, R.CONV, H)
    else:
      i = self.aidx[l]; a_h = K.rms_a(x, (self.Win_a, self.lidx[i]), H, "a_h")   # lidx[i] holds i
      ct_q = gemm(K, Wl, "q", a_h, 1, K.bufs["ct_q"]); ct_kv = gemm(K, Wl, "kv", a_h, 1, K.bufs["ct_kv"])
      q_rows = K.unpack(ct_q, R.NH * 2 * R.HD, "q_rows"); kv_rows = K.unpack(ct_kv, 2 * R.NKV * R.HD, "kv_rows")
      oc = K.attn_dec2(q_rows, kv_rows, self.QN, self.KN, self.ROPE, self.Kall, self.Vall, self.aix[i], R.NH, R.NKV, R.HD, self.tmax, R.ROT)
      a_o = K.rms_a(oc, None, R.NH * R.HD, "a_o", norm=False)
    x1 = K.resid(x, gemm(K, Wl, "o", a_o, 1, K.bufs["ct_o"]), H, "x1")
    # the routing on the device: the router's logits and the shared gate's logit (gemv32 on rms(x1), w_post folded), then route
    rl = K.gemv(x1, self.router, H, NE, "rlog", norm=True, idx=self.lidx[l]); g = K.gemv(x1, self.shg8, H, 8, "sgl", norm=True, idx=self.lidx[l])
    S_gu, S_dn = Wl.meta["gate"][4], Wl.meta["down"][4]
    K.call(f"route|softmax+sg|{NE}|{TOPK}|{S_gu}|{S_dn}", RT.route_src("softmax+sg", NE, TOPK, S_gu, S_dn, woff=1), self.sel_gu, self.sel_dn, self.wt, rl, rl, g)
    kg, kd = QN.Layer.KIND[Wl.meta["gate"][0]], QN.Layer.KIND[Wl.meta["down"][0]]; NP = QG.qgemvk_np(H); NPF = self.NPF
    self.qgemv(kg, FF, H, self.y_gu, Wl.st["gate"], self.sel_gu, x1, self.wpost_l[l], heads=TOPK); self.qgemv(QN.Layer.KIND[Wl.meta["up"][0]], FF, H, self.y_gu, Wl.st["up"], self.sel_gu, x1, self.wpost_l[l], heads=TOPK, yoff=TOPK * NPF)
    self.qgemv("q8", FF, H, self.y_sgu, Wl.st["sgate"], self.sel1, x1, self.wpost_l[l], nbuf=2); self.qgemv("q8", FF, H, self.y_sgu, Wl.st["sup"], self.sel1, x1, self.wpost_l[l], yoff=NPF, nbuf=2)   # q8 at K 2048 + the norm row: 2 buffers fit LSRAM
    self.qgemv(kd, H, FF, self.y_dn, Wl.st["down"], self.sel_dn, self.y_gu, x1, heads=TOPK, act="swiglu", gu_np=NPF, yoff=NP)        # heads 1 .. TOPK
    self.qgemv("q8", H, FF, self.y_dn, Wl.st["sdn"], self.sel1, self.y_sgu, x1, act="swiglu", gu_np=NPF)                              # the shared expert: head 0
    return K.call(f"moe_comb2|{H}|{NP}|{TOPK + 1}", QG.moe_comb2_src(H, NP, TOPK + 1, nt=4), out, x1, self.y_dn, self.wt, K.desc(f"moe_comb2_desc|{H}|{NP}|{TOPK + 1}", lambda: QG.moe_comb2_desc(H, NP, TOPK + 1)))

  def layer(self, l, x, pos):
    """x (a persistent rows buffer, the token in row 0) -> the layer's output into xio[(l + 1) % 2]."""
    self.layer_w(l)
    if R.LAYER_TYPES[l] != "linear" and not getattr(self, "_pos_set", False): i = self.aidx[l]; poke(self.aix[i], np.array([i, pos], np.int32))
    if JIT:
      if l not in self.jits: self.jits[l] = TinyJit(functools.partial(self.whole, l))
      return self.jits[l](self.jd)
    return self.whole(l, None)

  def step(self, tok, pos):
    """One token at `pos` through the 40 layers and the head -> the next token (greedy)."""
    K = self.K; poke(self.posb, np.array([pos], np.int32))
    x0 = np.zeros((self.Rr, H), np.float32); x0[0] = self.Wt.embed([tok])[0]; poke(self.xio[0], x0)
    for l, i in self.aidx.items(): poke(self.aix[i], np.array([i, pos], np.int32))
    self._pos_set = True; x = self.xio[0]
    for l in range(R.NL): x = self.layer(l, x, pos)
    self._pos_set = False
    K.call(f"qgemv|head|{self.Vn}", QG.qgemvk_src("q6", self.Vn, H, act="norm", top=True, eps=R.EPS, nbuf=2), self.tops, self.pk_head, self.head_all, self.sel0, self.d_head, self.tops, x, self.w_out)
    ids = K.head_reduce((0,)); return int(OA.host_invalidate(ids).numpy()[0])

  def generate(self, ids, n_new, stop=(IM_END,), on_token=None):
    t0 = time.perf_counter(); nxt = None
    for p, t in enumerate(ids): nxt = self.step(t, p)
    t_pre = time.perf_counter() - t0; out = [nxt]; pos = len(ids); t1 = time.perf_counter()
    if on_token: on_token(nxt)
    while len(out) < n_new and out[-1] not in stop:
      out.append(self.step(out[-1], pos)); pos += 1
      if on_token: on_token(out[-1])
    return out, t_pre, time.perf_counter() - t1


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("prompt", nargs="?", default=Q.PROMPT); ap.add_argument("--max-new", type=int, default=16)
  ap.add_argument("--thinking", action="store_true"); ap.add_argument("--tmax", type=int, default=512); ap.add_argument("--cache", default=QN.CACHE); a = ap.parse_args()
  from tokenizers import Tokenizer
  tk = Tokenizer.from_file(os.path.join(TOK, "tokenizer.json")); ids = tk.encode(Q.chat_text(a.prompt, a.thinking), add_special_tokens=False).ids
  m = Model(a.cache, tmax=a.tmax); print(f"{len(ids)} prompt tokens", flush=True)
  out, t_pre, t_dec = m.generate(ids, a.max_new, on_token=lambda t: print(tk.decode([t]), end="", flush=True))
  print(f"\n\nids: {out}\nprefill {len(ids)} tokens in {t_pre:.1f} s ({len(ids) / t_pre:.2f} tok/s, one at a time); decode {len(out) - 1} tokens in {t_dec:.1f} s ({(len(out) - 1) / max(t_dec, 1e-9):.2f} tok/s)")


if __name__ == "__main__": main()

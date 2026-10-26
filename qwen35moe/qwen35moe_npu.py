#!/usr/bin/env python3
"""One Qwen3.5-35B-A3B decoder layer on the NPU (decode: one row), from qwen35moe_pack.py's cache, against the numpy reference.

  DeltaNet layer:  rms_a -> the qkv and z GEMMs (q8f) -> gdn_tok3 (qwen3.8-27b's folded step: conv, a|b, the delta rule on the state
                   stack, the gated norm, o_proj's A) -> the o GEMM -> resid (x1)
  attention layer: rms_a -> the q (q | gate per head) and k|v GEMMs -> attn_dec2 (q / k norms, partial RoPE, the cache, the sigmoid gate)
                   -> the o GEMM -> resid (x1)
  then the MoE:    the router's gemv on rms(x1) (w_post folded) -> the host's softmax / top 8 / renormalisation and the shared expert's
                   sigmoid gate -> gate and up of the 8 experts as gemm_gs(bsel) heads 0-7 and the shared expert's as head 8 (q8f) into
                   one tile buffer [gate x 9 | up x 9] -> swiglu_a (9 x FF columns) -> down x 9 -> moe_comb (x1 + sum_h w_h C_h, w_8 the gate)

    python3 qwen35moe_npu.py --gate 0,3 [--tmax 64]   (the prompt's last token through those layers; the states / caches from the reference)
"""
import argparse, ctypes, os, sys, time
import numpy as np
os.environ["QWEN_MODEL"] = "qwen3.5-35b-a3b"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "tinygrad"))))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "qwen3.8-27b"), os.path.join(HERE, "..", "qwen3-30b-a3b"), os.path.join(HERE, "..", "gemma4"), os.path.join(HERE, "..", "ornith-9b")]
from tinygrad import Tensor, Device, dtypes                              # noqa: E402
from zy import OA                                                         # noqa: E402
import qwen38_ref as R                                                    # noqa: E402
from qwen38_kernels import Kernels, gdn_hpt, ring_pitch, ATT_BT          # noqa: E402
from qwen38_npu import Layers                                             # noqa: E402  (ab3 / adt: gdn_tok3's small-weight layouts)
import qwen3moe_kernels as MK                                             # noqa: E402
import qwen3moe_qgemv as QG                                               # noqa: E402  (the qdot GEMV: the experts' down on their q5n / q6n streams)
import qwen35moe_ref as Q                                                 # noqa: E402

DEV, NS, KS = "ZHOUYI", 3, 32
CACHE = os.environ.get("QWEN35MOE_NPU", "/mnt/ssd/qwen35moe-npu")
CS = lambda K: K.nrb * NS * 192                                           # floats of one group's C tiles
NE, TOPK, FF, H = 256, 8, 512, R.H
C_QKV = 2 * R.NK * R.DK + R.NV * R.DV                                     # the DeltaNet projection row (8192)


def dev(a, dt=None): return Tensor(np.ascontiguousarray(a), device=DEV, **({} if dt is None else {"dtype": dt})).realize()
def poke(t, a):
  Device[DEV].synchronize(); a = np.ascontiguousarray(a); b = t.uop.base.buffer; b.ensure_allocated(); raw = b._buf
  assert a.nbytes <= b.nbytes and raw.va, (a.nbytes, b.nbytes); ctypes.memmove(raw.va, a.ctypes.data, a.nbytes)


class LayerW:
  """Layer l's packed weights on the device: `st` name -> stream, `meta` name -> (q8, N, npad, K, S), `sm` the small tensors (numpy)."""
  def __init__(self, l, cache=CACHE):
    z = np.load(os.path.join(cache, f"L{l}_small.npz")); self.l = l
    self.meta = {str(k): tuple(int(v) for v in m) for k, m in zip(z["meta_names"], z["meta"])}
    self.sm = {k: z[k] for k in z.files if not k.startswith("meta")}
    self.st = {k: dev(np.fromfile(os.path.join(cache, f"L{l}_{k}.bin"), np.uint8)) for k in self.meta}


def gemm(K, W, k, a, n, out, heads=1, a_stride=0, c_stride=0, c_off=0, bsel=None):
  q8, _, npad, Kd, _ = W.meta[k]; ng = npad // (16 * NS)
  return OA.gemm_gs(a, W.st[k], ks=KS, ns=NS, nrb=K.nrb, nslices=Kd // (4 * KS), ngroups=ng, heads=heads, a_stride=a_stride, c_stride=c_stride,
                    c_off=c_off, b8=True, bscale=True, q8=q8, piece=K.nrb, rows=n, bsel=bsel, out=out).realize()


class Layer:
  """Layer l alone: its weights, its own buffers and a one-layer state stack / cache. L(x, pos, state) -> the output rows."""
  def __init__(self, l, tmax, cache=CACHE):
    self.l, self.tmax, self.kind = l, tmax, R.LAYER_TYPES[l]; K = self.K = Kernels(2, 24, R.EPS, real=1); self.R = 24; W = self.W = LayerW(l, cache)
    cs = CS(K); sm = W.sm
    self.ng_gu, self.ng_dn = W.meta["gate"][2] // 48, W.meta["down"][2] // 48
    self.ct_gu = K.buf("ct_gu", 2 * (TOPK + 1) * self.ng_gu * cs, dtypes.float32); self.ct_dn = K.buf("ct_dn", (TOPK + 1) * self.ng_dn * cs, dtypes.float32)
    self.sel_gu, self.sel_dn, self.wt = K.buf("sel_gu", 16, dtypes.int32), K.buf("sel_dn", 16, dtypes.int32), K.buf("moe_w", 12 * (TOPK + 1), dtypes.float32)
    self.win = K.hold("w_in", dev(1.0 + sm["input_layernorm_weight"])); self.wpost = K.hold("w_post", dev(1.0 + sm["post_attention_layernorm_weight"]))
    self.router = dev(sm["router"]); self.shg = sm["shexp_gate"].astype(np.float64)
    self.xin, self.xout = K.rows_buf("xio0", H), K.rows_buf("xio1", H)
    if self.kind == "linear":
      CP = self.CP = ring_pitch(C_QKV)
      self.Sall = K.buf("S_all", R.NV * R.DK * R.DV, dtypes.float32); self.Call = K.buf("C_all", R.CONV * CP, dtypes.float32)
      self.idx = K.buf("sidx0", 1, dtypes.int32); self.posb = K.buf("posb", 1, dtypes.int32)
      wab = np.concatenate([sm["linear_attn_in_proj_a_weight"], sm["linear_attn_in_proj_b_weight"]]) * (1.0 + sm["input_layernorm_weight"])[None]
      self.Wab = K.hold("Wab_all", dev(Layers.ab3(wab.astype(np.float32)))); self.Adt = K.hold("Adt_all", dev(Layers.adt(sm)))
      cw = sm["linear_attn_conv1d_weight"][:, 0, :].T.astype(np.float32)                        # [CONV, C] taps, position - (CONV - 1) + t
      self.Cwt = K.hold("Cwt_all", dev(np.pad(cw, ((0, 0), (0, CP - C_QKV))))); self.Nw = K.hold("Nw_all", dev(sm["linear_attn_norm_weight"]))
      for k in ("qkv", "z", "o"): K.buf(f"ct_{k}", W.meta[k][2] // 48 * cs, dtypes.float32)
    else:
      ksz = (tmax + ATT_BT) * R.NKV * R.HD
      self.Kall, self.Vall = K.buf("K_all", ksz, dtypes.float32), K.buf("V_all", ksz, dtypes.float32); self.aix = K.buf("aidx0", 2, dtypes.int32)
      self.QN = K.hold("qn_all", dev(1.0 + sm["self_attn_q_norm_weight"])); self.KN = K.hold("kn_all", dev(1.0 + sm["self_attn_k_norm_weight"]))
      inv = 1.0 / (R.THETA ** (np.arange(0, R.ROT, 2, dtype=np.float64) / R.ROT)); f = np.arange(tmax, dtype=np.float64)[:, None] * inv[None]; emb = np.concatenate([f, f], -1)
      self.ROPE = K.hold("rope_tab", dev(np.concatenate([np.cos(emb), np.sin(emb)], -1).astype(np.float32).ravel()))
      for k in ("q", "kv", "o"): K.buf(f"ct_{k}", W.meta[k][2] // 48 * cs, dtypes.float32)

  def attn_part(self, x, pos):
    K, W = self.K, self.W
    a_h = K.rms_a(x, self.win, H, "a_h")
    if self.kind == "linear":
      ct_qkv = gemm(K, W, "qkv", a_h, 1, K.bufs["ct_qkv"]); ct_z = gemm(K, W, "z", a_h, 1, K.bufs["ct_z"])
      a_o = K.gdn_tok3(self.Sall, self.Call, self.idx, self.posb, ct_qkv, ct_z, x, self.Wab, self.Adt, self.Cwt, self.Nw, R.NV, R.NK, R.DK, R.DV, C_QKV, R.CONV, H)
    else:
      ct_q = gemm(K, W, "q", a_h, 1, K.bufs["ct_q"]); ct_kv = gemm(K, W, "kv", a_h, 1, K.bufs["ct_kv"])
      q_rows = K.unpack(ct_q, R.NH * 2 * R.HD, "q_rows"); kv_rows = K.unpack(ct_kv, 2 * R.NKV * R.HD, "kv_rows")
      oc = K.attn_dec2(q_rows, kv_rows, self.QN, self.KN, self.ROPE, self.Kall, self.Vall, self.aix, R.NH, R.NKV, R.HD, self.tmax, R.ROT)
      a_o = K.rms_a(oc, None, R.NH * R.HD, "a_o", norm=False)
    return K.resid(x, gemm(K, W, "o", a_o, 1, K.bufs["ct_o"]), H, "x1")

  def route(self, x1):
    """The host's routing from the device's x1 row: (top, weights, the shared expert's gate) as the reference."""
    K = self.K; rl = K.gemv(x1, self.router, H, NE, "rlog", norm=True); lg = OA.host_invalidate(rl).numpy().reshape(self.R, NE)[0].astype(np.float64)
    p = np.exp(lg - lg.max()); p /= p.sum(); top = np.argsort(-p, kind="stable")[:TOPK]; w = p[top]; w /= w.sum()
    xr = OA.host_invalidate(x1).numpy().reshape(self.R, H)[0].astype(np.float64); m = xr / np.sqrt((xr * xr).mean() + R.EPS)
    sg = 1.0 / (1.0 + np.exp(-(m @ self.shg)))                                                 # (w_post is folded into shexp_gate)
    return top, w, sg, lg

  KIND = {5: "q4", 6: "q6n", 7: "q5n", 2: "q8"}                           # a stream's meta code -> the qdot GEMV kind that reads it

  def moe(self, x1, top, w, sg, out):
    """Gate and up as gemm_gs(bsel) heads (the q4m streams the prefill will read), their tiles unpacked to rows; the down projections on
    the qdot GEMV from those rows (act="swiglu" builds each head's package from silu(g) u): the experts' on their q5n / q6n streams
    (heads 1 .. TOPK of y_dn), the shared expert's on its q8f stream (kind "q8", head 0); moe_comb2 sums the rows with the weights
    (wt[0] = the shared expert's sigmoid gate)."""
    K, W = self.K, self.W; cs = CS(K); gs = self.ng_gu * cs; NP = QG.qgemvk_np(H)
    poke(self.sel_gu, np.array(list(top * W.meta["gate"][4]) + [0] * (16 - TOPK), np.int32))
    poke(self.sel_dn, np.array(list(top * W.meta["down"][4]) + [-1] * (16 - TOPK), np.int32))
    wt = np.zeros(12 * (TOPK + 1), np.float32); wt[0] = sg; wt[1:TOPK + 1] = w; poke(self.wt, wt)
    a_f = K.rms_a(x1, self.wpost, H, "a_f")
    ct_g = K.buf("ct_g8", TOPK * gs, dtypes.float32); ct_u = K.buf("ct_u8", TOPK * gs, dtypes.float32); ct_sg = K.buf("ct_sg", gs, dtypes.float32); ct_su = K.buf("ct_su", gs, dtypes.float32)
    gemm(K, W, "gate", a_f, 1, ct_g, heads=TOPK, c_stride=gs, bsel=self.sel_gu); gemm(K, W, "up", a_f, 1, ct_u, heads=TOPK, c_stride=gs, bsel=self.sel_gu)
    gemm(K, W, "sgate", a_f, 1, ct_sg); gemm(K, W, "sup", a_f, 1, ct_su)
    rows = [K.unpack(ct_g, FF, f"g{h}", goff=h * self.ng_gu)[0] for h in range(TOPK)] + [K.unpack(ct_u, FF, f"u{h}", goff=h * self.ng_gu)[0] for h in range(TOPK)]
    xs8 = K.buf("xs8", 2 * TOPK * FF, dtypes.float32).assign(Tensor.cat(*rows)).realize()
    xs1 = K.buf("xs1", 2 * FF, dtypes.float32).assign(Tensor.cat(K.unpack(ct_sg, FF, "g_sh")[0], K.unpack(ct_su, FF, "u_sh")[0])).realize()
    y_dn = K.buf("y_dn", (TOPK + 1) * NP, dtypes.float32); kd = self.KIND[W.meta["down"][0]]
    K.call(f"qgemv|down|{kd}|{H}|{FF}|{TOPK}", QG.qgemvk_src(kd, H, FF, heads=TOPK, act="swiglu", gu_np=FF, yoff=NP), y_dn, y_dn, W.st["down"], self.sel_dn,
           K.desc(f"q_desc|{kd}|{FF}", lambda: QG.qgemvk_desc(kd, FF)), y_dn, xs8, x1)
    sel1 = K.buf("sel1", 16, dtypes.int32); poke(sel1, np.array([0] + [-1] * 15, np.int32))
    K.call(f"qgemv|sdn|q8|{H}|{FF}", QG.qgemvk_src("q8", H, FF, heads=1, act="swiglu", gu_np=FF), y_dn, y_dn, W.st["sdn"], sel1,
           K.desc(f"q_desc|q8|{FF}", lambda: QG.qgemvk_desc("q8", FF)), y_dn, xs1, x1)
    return K.call(f"moe_comb2|{H}|{NP}|{TOPK + 1}", QG.moe_comb2_src(H, NP, TOPK + 1, nt=4), out, x1, y_dn, self.wt, K.desc(f"moe_comb2_desc|{H}|{NP}|{TOPK + 1}", lambda: QG.moe_comb2_desc(H, NP, TOPK + 1)))

  def __call__(self, x, pos):
    poke(self.xin, np.asarray(x, np.float32).reshape(self.R, H))
    if self.kind == "linear": poke(self.posb, np.array([pos], np.int32))
    else: poke(self.aix, np.array([0, pos], np.int32))
    x1 = self.attn_part(self.xin, pos); top, w, sg, lg = self.route(x1); self.last_route = (top, w, sg, lg)
    return self.moe(x1, top, w, sg, self.xout)


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--gguf", default=Q.DEFAULT); ap.add_argument("--cache", default=CACHE)
  ap.add_argument("--gate", default="0,3", help="layers: a list (0,3) or a range (0-39)"); ap.add_argument("--tmax", type=int, default=512)
  ap.add_argument("--tol", type=float, default=5e-3); ap.add_argument("--tok", default=os.environ.get("QWEN35MOE_TOK", "/mnt/ssd/ornith-9b-npu")); a = ap.parse_args()
  from tokenizers import Tokenizer
  W = R.Weights(a.gguf); ids = Tokenizer.from_file(os.path.join(a.tok, "tokenizer.json")).encode(Q.chat_text(Q.PROMPT), add_special_tokens=False).ids; p0 = len(ids) - 1
  layers = list(range(int(a.gate.split("-")[0]), int(a.gate.split("-")[1]) + 1)) if "-" in a.gate else [int(v) for v in a.gate.split(",")]
  t0 = time.perf_counter(); states = {}; xs = []
  x = W.embed(ids[:p0]); Q.forward(W, x, 0, states)                                  # the prefix: the states / caches at position p0
  xl = W.embed(ids[p0:]); xs = [xl]                                                   # the last token, layer by layer (the states advance)
  st2 = {l: (None if s is None else ((s[0].copy(), s[1].copy()) if isinstance(s, tuple) else {k: (v[0].copy(), v[1].copy()) for k, v in s.items()})) for l, s in states.items()}
  for l in range(max(layers) + 1):
    xl = Q.forward_layer(W, l, xl, p0, st2); xs.append(xl)
  print(f"numpy reference, {len(ids)} tokens through layer {max(layers)}: {time.perf_counter() - t0:.0f} s", flush=True); worst = 0.0
  for l in layers:
    t0 = time.perf_counter(); L = Layer(l, a.tmax, a.cache); tl = time.perf_counter() - t0; s = states[l]
    if L.kind == "linear":
      S, conv = s; poke(L.Sall, S.astype(np.float32).reshape(-1)); ring = np.zeros((R.CONV, L.CP), np.float32)
      for j in range(R.CONV - 1): ring[(p0 - (R.CONV - 1) + j) % R.CONV, :C_QKV] = conv[j]
      poke(L.Call, ring.reshape(-1)); poke(L.idx, np.array([0], np.int32))
    else:
      Kc, Vc = s["kv"]; kk = np.zeros((a.tmax + ATT_BT, R.NKV, R.HD), np.float32); vv = np.zeros_like(kk); kk[:p0], vv[:p0] = Kc[:p0], Vc[:p0]
      poke(L.Kall, kk.reshape(-1)); poke(L.Vall, vv.reshape(-1))
    xin = np.zeros((L.R, H), np.float32); xin[0] = xs[l][0]; want = xs[l + 1][0]
    t0 = time.perf_counter(); y = L(xin, p0); got = OA.host_invalidate(y).numpy().reshape(L.R, H)[0]; dt = time.perf_counter() - t0
    r = float(np.linalg.norm(got - want) / np.linalg.norm(want - xs[l][0])); top, w, sg, _ = L.last_route; rt, rw = W.last_route[l]
    same = bool((np.sort(top) == np.sort(rt[0])).all())
    # the two parts: x1 (after the mixer) from the device against the reference's (its MoE switched off on a copy of the prefix state)
    x1d = OA.host_invalidate(L.K.bufs["x1"]).numpy().reshape(L.R, H)[0]; moe_save = W.moe; W.moe = lambda sf, p, m: np.zeros_like(m)
    st3 = {l: (states[l][0].copy(), states[l][1].copy()) if L.kind == "linear" else {"kv": (states[l]["kv"][0].copy(), states[l]["kv"][1].copy())}}
    x1r = Q.forward_layer(W, l, xs[l].copy(), p0, st3)[0]; W.moe = moe_save
    ra = float(np.linalg.norm(x1d - x1r) / np.linalg.norm(x1r - xs[l][0])); rm = float(np.linalg.norm((got - x1d) - (want - x1r)) / np.linalg.norm(want - x1r))
    print(f"layer {l:2d} ({L.kind}) row at {p0}: {dt * 1e3:.0f} ms (load {tl:.1f} s) | rel err of the layer's delta {r:.3g} (mixer {ra:.3g}, experts {rm:.3g}) | experts {'= the reference' if same else f'{sorted(top.tolist())} vs {sorted(rt[0].tolist())}'} "
          f"(weights max|d| {np.abs(np.sort(w) - np.sort(rw[0])).max():.2g}) | finite {bool(np.isfinite(got).all())}", flush=True)
    worst = max(worst, r if np.isfinite(r) else np.inf); del L
  print(f"GATE {'PASS' if worst < a.tol else 'FAIL'}: worst rel err of a layer's delta {worst:.3g} over layers {layers}", flush=True)


if __name__ == "__main__": main()

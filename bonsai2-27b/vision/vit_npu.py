#!/usr/bin/env python3
"""The Bonsai 2 vision tower on the NPU: the Q8_0 mmproj's linears as gemm_gs Q8 panels (packed once into <cache>/vision), the
LayerNorms, GELUs, residuals, the 2-D rope and the bidirectional attention as the hand-written kernels of vision_kernels.py, the
A layouts by the runtime's producer. Same math as vit_ref.py (llama.cpp's clip_graph_qwen3vl).
  python3 vit_npu.py IMAGE [--ref ref_dumps.npz]       # the embeddings, timed; --ref: against the numpy reference's dumps"""
import argparse, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, ".."), os.path.join(HERE, "..", "..", "qwen3.8-27b"), os.path.join(HERE, "..", "..", "ornith-9b")]
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "..", "tinygrad"))))
from tinygrad import Tensor, dtypes                                       # noqa: E402
from zy import OA, gemm_fp16 as GF                                       # noqa: E402
import qwen38_kernels as QK                                              # noqa: E402
import gguf_read as G                                                    # noqa: E402
import vision_kernels as VK                                              # noqa: E402
import vit_ref as VR                                                     # noqa: E402

DEV = "ZHOUYI"; NS, KS = 3, 32; EPS = 1e-6                                 # the q8f panels (gemm_gs q8=2): 128-wide K slices, the A layout the producers write
LINEARS = ["attn_qkv", "attn_out", "ffn_up", "ffn_down"]


def requant_q8(w):
  """fp32 [N, K] -> GGUF Q8_0 (q int8 [N, K], d [N, K/32])."""
  N, K = w.shape; b = w.reshape(N, K // 32, 32); d = np.abs(b).max(-1) / 127.0; d = np.where(d == 0, 1.0, d)
  q = np.clip(np.rint(b / d[..., None]), -127, 127).astype(np.int8).reshape(N, K); return q, d.astype(np.float32)


def pack_vision(mmproj, out):
  """The mmproj -> <out>/*.bin q8f panel streams + vision_small.npz (norms, biases, position embedding, the metas, the geometry). Every
  linear is requantised to Q8_0 from fp32 (Q8_0 tensors round-trip exactly; the 9B's F16 ones get quantised) after padding: each head
  to HDP = 16 | HD dims as [first rope half | zeros | second half | zeros] (each half padded to HDP/2, so the kernels' pairs
  (i, i + HDP/2) are the model's (i, i + HD/2)), in the qkv rows, the qkv bias and attn_out's columns; the MLP to a multiple of 128 (ffn_up's rows and bias, ffn_down's columns). The
  2B / 4B tower (1024 wide, 64-dim heads, MLP 4096) is unpadded; the 9B's (1152, 72, 4304) becomes 1280-wide qkv heads of 80 and an MLP of 4352."""
  os.makedirs(out, exist_ok=True); g = G.GGUF(mmproj); md = g.meta if hasattr(g, "meta") else g.kv; meta = {}; small = {}
  nl = int(md["clip.vision.block_count"]); C = int(md["clip.vision.embedding_length"]); NH = int(md["clip.vision.attention.head_count"]); HD = C // NH
  HDP = -(-HD // 16) * 16; MLP = int(md["clip.vision.feed_forward_length"]); MLPP = -(-MLP // 128) * 128; eps = float(md.get("clip.vision.attention.layer_norm_epsilon", EPS))
  f32 = lambda name: np.asarray(g.f32(name), np.float32)
  def headpad(w, axis):
    if HDP == HD: return w
    w = np.moveaxis(w, axis, 0); hr = HD // 2; wh = w.reshape(NH, HD, *w.shape[1:]); o = np.zeros((NH, HDP) + w.shape[1:], w.dtype)
    o[:, :hr] = wh[:, :hr]; o[:, HDP // 2:HDP // 2 + hr] = wh[:, hr:]; return np.moveaxis(o.reshape(NH * HDP, *w.shape[1:]), 0, axis)   # [a | 0 | b | 0]
  def pad_rows(w, n): return w if w.shape[0] == n else np.concatenate([w, np.zeros((n - w.shape[0],) + w.shape[1:], w.dtype)])
  def stream(q, d):
    ng = -(-q.shape[0] // (16 * NS)); return np.concatenate([GF.pack_b_group_q8f(q, d, k, NS, KS) for k in range(ng)]), 16 * NS * ng
  def put(name, w):
    q, d = requant_q8(w); st, npad = stream(q, d); st.tofile(os.path.join(out, f"{name}.bin")); meta[name] = (q.shape[0], npad, q.shape[1])
  for i in range(nl):
    b = f"v.blk.{i}."; w = f32(b + "attn_qkv.weight").reshape(3, C, C)
    put(f"blk{i}_attn_qkv", headpad(w, 1).reshape(3 * NH * HDP, C)); small[f"blk{i}_attn_qkv_bias"] = headpad(f32(b + "attn_qkv.bias").reshape(3, C), 1).ravel()
    put(f"blk{i}_attn_out", headpad(f32(b + "attn_out.weight"), 1)); put(f"blk{i}_ffn_up", pad_rows(f32(b + "ffn_up.weight"), MLPP)); put(f"blk{i}_ffn_down", pad_rows(f32(b + "ffn_down.weight").T, MLPP).T.copy())
    small[f"blk{i}_ffn_up_bias"] = pad_rows(f32(b + "ffn_up.bias"), MLPP)
    for k in ("attn_out.bias", "ffn_down.bias", "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"): small[f"blk{i}_{k.replace('.', '_')}"] = f32(b + k)
  put("patch", f32("v.patch_embd.weight").reshape(C, -1) + f32("v.patch_embd.weight.1").reshape(C, -1))
  put("mm0", f32("mm.0.weight")); put("mm2", f32("mm.2.weight"))
  for k in ("v.patch_embd.bias", "v.position_embd.weight", "v.post_ln.weight", "v.post_ln.bias", "mm.0.bias", "mm.2.bias"): small[k.replace(".", "_")] = f32(k)
  small["meta_names"] = np.array(list(meta)); small["meta"] = np.array([meta[k] for k in meta], np.int64); small["n_layer"] = nl
  small["cfg"] = np.array([C, NH, HD, HDP, MLPP], np.int64); small["eps"] = np.array(eps, np.float64)
  np.savez(os.path.join(out, "vision_small.npz"), **small); print(f"   vision pack: {nl} blocks, width {C}, {NH} heads of {HD} (padded to {HDP}), MLP {MLP} (to {MLPP}) -> {out}", flush=True)


class VisionNPU:
  def __init__(self, cache):
    self.cache = cache; z = np.load(os.path.join(cache, "vision_small.npz")); self.nl = int(z["n_layer"])
    self.C, self.NH, self.HD, self.HDP, self.MLPP = (int(v) for v in z["cfg"]) if "cfg" in z.files else (1024, 16, 64, 64, 4096); self.eps = float(z["eps"]) if "eps" in z.files else EPS
    self.meta = {str(n): tuple(int(v) for v in m) for n, m in zip(z["meta_names"], z["meta"])}
    dev = lambda a: Tensor(np.ascontiguousarray(a, np.float32).ravel(), device=DEV).realize()
    self.small = {k: dev(z[k]) for k in z.files if k not in ("meta_names", "meta", "n_layer", "cfg", "eps") and k != "v_position_embd_weight"}
    self.pos_np = z["v_position_embd_weight"]; self.mm2_bias = dev(np.pad(z["mm_2_bias"], (0, self.meta["mm2"][1] - z["mm_2_bias"].shape[0])))   # the bias over the padded columns
    # The ~100 linears' panels (300-450 MB) stay on the HOST by default and go to one device staging buffer right before their GEMM
    # (~0.1 s an image): the NPU's ASID 0 window (~2.75 GB for every process's buffers and window slots) is what bounds how many
    # servers run side by side. QWEN_VISION_RESIDENT=1: the panels resident on the device, as before (no upload a GEMM).
    self.resident = os.environ.get("QWEN_VISION_RESIDENT") == "1"
    self.Wh = {n: np.fromfile(os.path.join(cache, f"{n}.bin"), np.uint8) for n in self.meta}
    if self.resident: self.W = {n: Tensor(w, device=DEV).realize() for n, w in self.Wh.items()}
    else: self.stage = Tensor.zeros(max(w.size for w in self.Wh.values()), dtype=dtypes.uint8, device=DEV).contiguous().realize()
    self.geo = {}
  def panels(self, name):
    """The linear's packed panels as a device tensor: resident, or uploaded into the staging buffer (its prefix view)."""
    if self.resident: return self.W[name]
    w = self.Wh[name]; st = self.stage[:w.size]; st.assign(Tensor(w, device=DEV)).realize(); return st
  def _geo(self, n):
    if n not in self.geo:
      nrb = max(2, -(-(-(-n // 12)) // 2) * 2); self.geo[n] = (nrb, 12 * nrb, QK.Kernels(nrb, 12 * nrb, self.eps, real=None))
    return self.geo[n]
  @staticmethod
  def piece(nrb):
    """The GEMM's row piece: the largest even divisor of nrb within the runner's bound (piece x ns x 768 B of C staging <= 64 KiB: 28)."""
    return max(p for p in range(2, min(nrb, 28) + 1, 2) if nrb % p == 0)
  def gemm(self, K, nrb, a, name):
    N, npad, Kin = self.meta[name]; ng = npad // 48; out = K.buf(f"ct_{name.split('_', 1)[-1] if name.startswith('blk') else name}", ng * nrb * NS * 192, dtypes.float32)
    return OA.gemm_gs(a, self.panels(name), ks=KS, ns=NS, nrb=nrb, nslices=Kin // (4 * KS), ngroups=ng, b8=True, bscale=True, q8=2, scales="dup", piece=self.piece(nrb), out=out).realize(), N, npad
  def rows_add(self, K, n, c, nrb, x, ct, bias, tag):
    k = K.key(f"residb32|{c}|{nrb}|{n}", VK.residb32_src(c, nrb, n)); return K.run(k, tag, 12 * nrb * c, dtypes.float32, x, ct, bias).reshape(12 * nrb, c)
  def ln(self, K, n, c, x, w, b, tag):
    k = K.key(f"ln32|{c}|{n}", VK.ln32_src(c, self.eps, n)); return K.run(k, tag, x.shape[0] * c, dtypes.float32, x, w, b).reshape(x.shape[0], c)
  def gelu_rows(self, K, n, c, nrb, ct, bias, tag):
    k = K.key(f"geluc32|{c}|{nrb}|{n}", VK.geluc32_src(c, nrb, n)); return K.run(k, tag, 12 * nrb * c, dtypes.float32, ct, bias).reshape(12 * nrb, c)
  def forward(self, x, pw, ph, dump=None):
    """x [H, W, 3] normalised -> (embeddings [pw ph / 4, the language model's width], (pw / 2, ph / 2))."""
    P = VR.patches_in_block_order(x, pw, ph); n = P.shape[0]; nrb, Rr, K = self._geo(n); C, NH, HD, MLPP = self.C, self.NH, self.HDP, self.MLPP; QW = 3 * NH * HD; t0 = time.perf_counter()
    idx = VR.block_order_index(pw, ph); pe = VR.pos_emb_interp(self.pos_np, pw, ph)[idx]
    xin = np.zeros((Rr, 768), np.float32); xin[:n] = P; pos = np.zeros((Rr, C), np.float32); pos[:n] = pe
    X = K.buf("v_in", Rr * 768, dtypes.float32).assign(Tensor(xin.ravel(), device=DEV)).realize().reshape(Rr, 768); POS = K.buf("v_pos", Rr * C, dtypes.float32).assign(Tensor(pos.ravel(), device=DEV)).realize().reshape(Rr, C)
    zero = K.buf("v_zero", Rr * QW, dtypes.float32).reshape(Rr, QW)
    ct, _, _ = self.gemm(K, nrb, K.rms_a(X, None, 768, "a_v768", norm=False), "patch"); h = self.rows_add(K, n, C, nrb, POS, ct, self.small["v_patch_embd_bias"], "v_h")
    if dump is not None: dump["inp"] = OA.host_invalidate(h).numpy()[:n].copy()
    ys = (idx // pw).astype(np.float64); xs = (idx % pw).astype(np.float64); hr, q4 = self.HD // 2, self.HD // 4; inv = VR.THETA_V ** (-np.arange(0, q4, dtype=np.float64) * 2 / hr)
    ang = np.zeros((n, HD // 2)); ang[:, :q4] = ys[:, None] * inv[None]; ang[:, q4:2 * q4] = xs[:, None] * inv[None]   # the padded pairs turn by 0
    tab = np.zeros((Rr, HD), np.float32); tab[:n] = np.concatenate([np.cos(ang), np.sin(ang)], 1)
    TAB = K.buf("v_rope", Rr * HD, dtypes.float32).assign(Tensor(tab.ravel(), device=DEV)).realize()
    k_rope = K.key(f"rope_rows|{n}", VK.rope_rows_src(n, NH, HD, QW)); k_attn = K.key(f"vit_attn|{n}", VK.vit_attn_src(NH, HD, n, QW, hd_real=self.HD)); dsc = K.desc("vit_attn_desc", lambda: VK.vit_attn_desc(HD, QW, NH * HD, 8, 3))
    for i in range(self.nl):
      s = lambda k: self.small[f"blk{i}_{k}"]
      a = K.rms_a(self.ln(K, n, C, h, s("ln1_weight"), s("ln1_bias"), "v_ln"), None, C, f"a_v{C}", norm=False)
      ct, _, _ = self.gemm(K, nrb, a, f"blk{i}_attn_qkv"); qkv = self.rows_add(K, n, QW, nrb, zero, ct, s("attn_qkv_bias"), "v_qkv")
      if dump is not None and i == 0: dump["qkv0_pre"] = OA.host_invalidate(qkv).numpy()[:n].copy(); dump["tab"] = tab[:n].copy()
      K.run(k_rope, "v_flag", 16, dtypes.float32, qkv, TAB); o = K.run(k_attn, "v_o", Rr * NH * HD, dtypes.float32, qkv, dsc).reshape(Rr, NH * HD)
      ct, _, _ = self.gemm(K, nrb, K.rms_a(o, None, NH * HD, f"a_v{NH * HD}", norm=False), f"blk{i}_attn_out"); h = self.rows_add(K, n, C, nrb, h, ct, s("attn_out_bias"), "v_h2")
      if dump is not None and i == 0: dump["qkv0"] = OA.host_invalidate(qkv).numpy()[:n].copy(); dump["o0"] = OA.host_invalidate(o).numpy()[:n].copy(); dump["hatt0"] = OA.host_invalidate(h).numpy()[:n].copy()
      a = K.rms_a(self.ln(K, n, C, h, s("ln2_weight"), s("ln2_bias"), "v_ln"), None, C, f"a_v{C}", norm=False)
      ct, _, _ = self.gemm(K, nrb, a, f"blk{i}_ffn_up"); g = self.gelu_rows(K, n, MLPP, nrb, ct, s("ffn_up_bias"), "v_g")
      ct, _, _ = self.gemm(K, nrb, K.rms_a(g, None, MLPP, f"a_v{MLPP}", norm=False), f"blk{i}_ffn_down"); h = self.rows_add(K, n, C, nrb, h, ct, s("ffn_down_bias"), "v_h")
      if dump is not None and i in (0, self.nl - 1): dump[f"blk{i}"] = OA.host_invalidate(h).numpy()[:n].copy()
    hn = self.ln(K, n, C, h, self.small["v_post_ln_weight"], self.small["v_post_ln_bias"], "v_ln")
    # the merger on n / 4 rows of 4 C (4 consecutive patches = a 2 x 2 block)
    n2 = n // 4; nrb2, R2, K2 = self._geo(n2); M = K2.buf("v_m", R2 * 4 * C, dtypes.float32).reshape(R2, 4 * C)
    H, Hpad = self.meta["mm2"][:2]                                          # the language model's width (2048 / 2560 / 4096), its 48-padded columns
    M[:n2].assign(hn[:n].reshape(n2, 4 * C)).realize(); zero2 = K2.buf("v_zero2", R2 * Hpad, dtypes.float32).reshape(R2, Hpad)
    N0 = self.meta["mm0"][0]                                                # the merger's width (4 C)
    ct, _, _ = self.gemm(K2, nrb2, K2.rms_a(M, None, 4 * C, "a_m", norm=False), "mm0"); g = self.gelu_rows(K2, n2, N0, nrb2, ct, self.small["mm_0_bias"], "v_g2")
    ct, _, npad = self.gemm(K2, nrb2, K2.rms_a(g, None, N0, "a_m", norm=False), "mm2"); e = self.rows_add(K2, n2, npad, nrb2, zero2, ct, self.mm2_bias, "v_e")
    out = OA.host_invalidate(e).numpy()[:n2, :H].copy(); self.t_last = time.perf_counter() - t0
    return out, (pw // 2, ph // 2)


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("image"); ap.add_argument("--mmproj", default=os.environ.get("BONSAI_MMPROJ", "/mnt/ssd/bonsai2-2b/Ternary-Bonsai-2-2B-mmproj-Q8_0.gguf"))
  ap.add_argument("--cache", default=os.path.join(os.environ.get("QWEN_NPU", "/mnt/ssd/bonsai2-2b-npu"), "vision")); ap.add_argument("--max-tokens", type=int, default=1024); ap.add_argument("--ref", default=None); a = ap.parse_args()
  if not os.path.exists(os.path.join(a.cache, "vision_small.npz")): pack_vision(a.mmproj, a.cache)
  t0 = time.perf_counter(); V = VisionNPU(a.cache); print(f"   vision tower loaded in {time.perf_counter() - t0:.1f} s", flush=True)
  x, pw, ph = VR.load_image(a.image, a.max_tokens); d = {}
  for rep in range(2): e, grid = V.forward(x, pw, ph, dump=d if rep == 0 else None); print(f"   {pw * ph} patches -> {e.shape[0]} tokens {grid} in {V.t_last:.2f} s", flush=True)
  if a.ref:
    r = np.load(a.ref); rel = lambda g, w: float(np.linalg.norm(g.astype(np.float64) - w) / np.linalg.norm(w))
    for k in ("inp", "blk0", f"blk{V.nl - 1}"): print(f"   {k}: rel err vs numpy {rel(d[k], r[k]):.3e}")
    print(f"   embeddings: rel err vs numpy {rel(e, r['emb']):.3e}; per-token cosine min {np.min((e * r['emb']).sum(1) / (np.linalg.norm(e, axis=1) * np.linalg.norm(r['emb'], axis=1))):.4f}")


if __name__ == "__main__": main()

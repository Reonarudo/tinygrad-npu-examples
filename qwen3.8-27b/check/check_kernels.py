"""qwen38_kernels vs the tinygrad-op versions in qwen38_npu (random data, decode and a 96-row geometry), with timings."""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tinygrad")))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tinygrad import Tensor
from zy import OA
import qwen38_npu as Q, qwen38_ref as R
from qwen38_kernels import Kernels, ring_pitch
def rd(t): return OA.host_invalidate(t.realize()).numpy().ravel()
def timeit(fn, reps=5):
  fn().realize(); t0 = time.perf_counter()
  for _ in range(reps): fn().realize()
  return (time.perf_counter() - t0) / reps * 1e3
for n in (12,):
  L = Q.Layers(n, "/mnt/ssd/qwen3.8-27b-npu-s1"); K = Kernels(L.nrb, L.R, R.EPS); rng = np.random.default_rng(0)
  x = rng.standard_normal((L.R, 5120)).astype(np.float32); w = rng.standard_normal(5120).astype(np.float32) * 0.1
  xt, wt = Q.dev(x), Q.dev(w)
  a_ref = rd(L.A(L.rms(xt, wt))); a_new = rd(K.rms_a(xt, Q.dev(1.0 + w), 5120, "t"))
  print(f"n={n} (R={L.R}) rms_a32 vs tinygrad: max|d| {np.abs(a_ref.view(np.float16).astype(np.float32) - a_new.view(np.float16).astype(np.float32)).max():.3g}  |  {timeit(lambda: K.rms_a(xt, Q.dev(1.0 + w), 5120, "t")):.1f} ms vs {timeit(lambda: L.A(L.rms(xt, wt))):.1f} ms")
  a_ref = rd(L.A(xt)); a_new = rd(K.rms_a(xt, wt, 5120, "t2", norm=False))
  print(f"   plain A: max|d| {np.abs(a_ref.view(np.float16).astype(np.float32) - a_new.view(np.float16).astype(np.float32)).max():.3g}  |  {timeit(lambda: K.rms_a(xt, wt, 5120, "t2", norm=False)):.1f} ms vs {timeit(lambda: L.A(xt)):.1f} ms")
  m_ = 17408; ng = (2 * m_ + 47) // 48; ct = Q.dev(rng.standard_normal((ng * L.nrb * 3 * 192,)).astype(np.float32) * 2)
  g = L.C(ct, ng); ref = rd(L.A(g[:, :m_].silu() * g[:, m_:2 * m_])); new = rd(K.swiglu_a(ct, m_, "t3"))
  d = np.abs(ref.view(np.float16).astype(np.float32) - new.view(np.float16).astype(np.float32))
  print(f"   swiglu_a32: max|d| {d.max():.3g} (max |A| {np.abs(ref.view(np.float16).astype(np.float32)).max():.3g})  |  {timeit(lambda: K.swiglu_a(ct, m_, "t3")):.1f} ms vs {timeit(lambda: L.A(g[:, :m_].silu() * g[:, m_:2 * m_])):.1f} ms")
  c = 5120; ng = (c + 47) // 48; ct2 = Q.dev(rng.standard_normal((ng * L.nrb * 3 * 192,)).astype(np.float32))
  ref = rd(xt + L.C(ct2, ng)[:, :c]); new = rd(K.resid(xt, ct2, c, "t4"))
  print(f"   resid32: max|d| {np.abs(ref - new).max():.3g}  |  {timeit(lambda: K.resid(xt, ct2, c, "t4")):.1f} ms vs {timeit(lambda: xt + L.C(ct2, ng)[:, :c]):.1f} ms")
# ---- the DeltaNet / row kernels (decode geometry)
L = Q.Layers(1, "/mnt/ssd/qwen3.8-27b-npu-s1"); K = Kernels(L.nrb, L.R, R.EPS); rng = np.random.default_rng(3)
c = 10240; ng = (c + 47) // 48; ct = Q.dev(rng.standard_normal((ng * L.nrb * 3 * 192,)).astype(np.float32))
ref = rd(L.C(ct, ng)[:, :c]); new = rd(K.unpack(ct, c, "t5")); print(f"rows32 (c {c}): max|d| {np.abs(ref - new).max():.3g}  |  {timeit(lambda: K.unpack(ct, c, 't5')):.1f} ms vs {timeit(lambda: L.C(ct, ng)[:, :c].contiguous()):.1f} ms")
x = rng.standard_normal((L.R, 5120)).astype(np.float32); w = rng.standard_normal((96, 5120)).astype(np.float32) * 0.02
xt, wt = Q.dev(x), Q.dev(w); ref = (x @ w.T).ravel(); new = rd(K.gemv(xt, wt, 5120, 96, "t6"))
print(f"gemv32 [24 x 5120] @ [96 x 5120]^T: max|d| {np.abs(ref - new).max():.3g} of {np.abs(ref).max():.3g}  |  {timeit(lambda: K.gemv(xt, wt, 5120, 96, 't6')):.1f} ms vs {timeit(lambda: xt @ wt.T):.1f} ms")
NV, DK, DV = R.NV, R.DK, R.DV
S0 = rng.standard_normal((NV, DK, DV)).astype(np.float32) * 0.1; q = rng.standard_normal((NV, DK)).astype(np.float32); k = rng.standard_normal((NV, DK)).astype(np.float32)
v = rng.standard_normal((NV, DV)).astype(np.float32); beta = rng.random(NV).astype(np.float32); decay = rng.random(NV).astype(np.float32)
S_ref = S0 * decay[:, None, None]; kv = np.einsum("hkv,hk->hv", S_ref, k); delta = (v - kv) * beta[:, None]; S_ref = S_ref + k[:, :, None] * delta[:, None, :]; o_ref = np.einsum("hkv,hk->hv", S_ref, q)
St = K.buf("gdn_S_test", 2 * NV * DK * DV, Q.dtypes.float32); St[NV * DK * DV:].assign(Q.dev(S0.ravel())).realize(); idx = K.buf("idx1", 1, Q.dtypes.int32); idx.assign(Q.dev(np.array([1], np.int32))).realize()
o = rd(K.gdn_step(St, idx, Q.dev(q), Q.dev(k), Q.dev(v), Q.dev(beta), Q.dev(decay), NV, DK, DV))
S_new = OA.host_invalidate(St).numpy()[NV * DK * DV:].reshape(NV, DK, DV)
print(f"gdn_step: o max|d| {np.abs(o.reshape(NV, DV) - o_ref).max():.3g} of {np.abs(o_ref).max():.3g} | S max|d| {np.abs(S_new - S_ref).max():.3g}")
t0 = time.time(); [K.gdn_step(St, idx, Q.dev(q), Q.dev(k), Q.dev(v), Q.dev(beta), Q.dev(decay), NV, DK, DV) for _ in range(5)]; print(f"   gdn_step {(time.time() - t0) / 5 * 1e3:.1f} ms (incl. the input copies)")
# ---- the decode attention vs numpy (T = 512 cache, pos 300)
NH, NKV, HD, TMAX = R.NH, R.NKV, R.HD, 512; rng = np.random.default_rng(7); pos = 300
Kc = rng.standard_normal((TMAX, NKV, HD)).astype(np.float32); Vc = rng.standard_normal((TMAX, NKV, HD)).astype(np.float32)
q = rng.standard_normal((NH, HD)).astype(np.float32); kn = rng.standard_normal((NKV, HD)).astype(np.float32); vn = rng.standard_normal((NKV, HD)).astype(np.float32)
Kf, Vf = Kc.copy(), Vc.copy(); Kf[pos], Vf[pos] = kn, vn; g = NH // NKV
sref = np.einsum("hd,thd->ht", q, np.repeat(Kf[:pos + 1], g, 1)) / np.sqrt(HD); pw = np.exp(sref - sref.max(1, keepdims=True)); pw /= pw.sum(1, keepdims=True)
oref = np.einsum("ht,thd->hd", pw, np.repeat(Vf[:pos + 1], g, 1))
Ka = K.buf("K_t", 2 * TMAX * NKV * HD, Q.dtypes.float32); Va = K.buf("V_t", 2 * TMAX * NKV * HD, Q.dtypes.float32)
Ka[TMAX * NKV * HD:].assign(Q.dev(Kc.ravel())).realize(); Va[TMAX * NKV * HD:].assign(Q.dev(Vc.ravel())).realize()
ix = K.buf("aix_t", 2, Q.dtypes.int32); ix.assign(Q.dev(np.array([1, pos], np.int32))).realize()
o = rd(K.attn_decode(Q.dev(q), Ka, Va, Q.dev(kn), Q.dev(vn), ix, NH, NKV, HD, TMAX)).reshape(NH, HD)
Kb = OA.host_invalidate(Ka).numpy()[TMAX * NKV * HD:].reshape(TMAX, NKV, HD)
print(f"attn_decode (T {TMAX}, pos {pos}): max|d| {np.abs(o - oref).max():.3g} of {np.abs(oref).max():.3g} | cache row written {bool(np.abs(Kb[pos] - kn).max() == 0)}")
t0 = time.time(); [K.attn_decode(Q.dev(q), Ka, Va, Q.dev(kn), Q.dev(vn), ix, NH, NKV, HD, TMAX) for _ in range(5)]; print(f"   attn_decode {(time.time() - t0) / 5 * 1e3:.1f} ms (incl. the input copies)")
# ---- the fused DeltaNet token step vs numpy (a random ring at pos 5: tokens 2, 3, 4 in slots 2, 3, 0; slot 1 gets the new row)
NV, NK, DK, DV, CONV = R.NV, R.NK, R.DK, R.DV, R.CONV; C = 2 * NK * DK + NV * DV; CP = ring_pitch(C); rng = np.random.default_rng(11); pos = 5; eps = R.EPS
ring = rng.standard_normal((CONV, C)).astype(np.float32); xr = rng.standard_normal(C).astype(np.float32); zr = rng.standard_normal(NV * DV).astype(np.float32)
cw = rng.standard_normal((C, CONV)).astype(np.float32) * 0.5; nw = (1.0 + 0.1 * rng.standard_normal(DV)).astype(np.float32)
beta = rng.uniform(0.1, 0.9, NV).astype(np.float32); decay = rng.uniform(0.8, 1.0, NV).astype(np.float32); S0 = rng.standard_normal((NV, DK, DV)).astype(np.float32) * 0.1
ext = np.stack([ring[(pos - 3) % CONV], ring[(pos - 2) % CONV], ring[(pos - 1) % CONV], xr]); u = (ext * cw.T).sum(0); u = u / (1.0 + np.exp(-u))
l2 = lambda t: t / np.sqrt((t * t).sum(-1, keepdims=True) + 1e-6)
q = np.repeat(l2(u[:NK * DK].reshape(NK, DK)) / np.sqrt(DK), NV // NK, 0); k = np.repeat(l2(u[NK * DK:2 * NK * DK].reshape(NK, DK)), NV // NK, 0); v = u[2 * NK * DK:].reshape(NV, DV)
S1 = S0 * decay[:, None, None]; kv = np.einsum("hi,hij->hj", k, S1); delta = (v - kv) * beta[:, None]; S2 = S1 + np.einsum("hi,hj->hij", k, delta); o = np.einsum("hi,hij->hj", q, S2)
oref = o / np.sqrt((o * o).mean(-1, keepdims=True) + eps) * nw * (zr.reshape(NV, DV) / (1.0 + np.exp(-zr.reshape(NV, DV))))
pad = lambda a: np.pad(a, ((0, 0), (0, CP - C)))                          # the ring / taps rows at the kernels' pitch CP
St = K.buf("S_tok_t", 2 * NV * DK * DV, Q.dtypes.float32); St[NV * DK * DV:].assign(Q.dev(S0.ravel())).realize()
Ct = K.buf("C_tok_t", 2 * CONV * CP, Q.dtypes.float32); Ct[CONV * CP:].assign(Q.dev(pad(ring).ravel())).realize()
ix = K.buf("tidx", 1, Q.dtypes.int32); ix.assign(Q.dev(np.array([1], np.int32))).realize(); pb = K.buf("tpos", 1, Q.dtypes.int32); pb.assign(Q.dev(np.array([pos], np.int32))).realize()
xrows = np.zeros((L.R, C), np.float32); xrows[0] = xr; zrows = np.zeros((L.R, NV * DV), np.float32); zrows[0] = zr
xb = K.buf("qkv_t", L.R * C, Q.dtypes.float32); xb.assign(Q.dev(xrows.ravel())).realize(); zb = K.buf("z_t", L.R * NV * DV, Q.dtypes.float32); zb.assign(Q.dev(zrows.ravel())).realize()
bd = K.buf("bd_t", 2 * NV, Q.dtypes.float32); bd.assign(Q.dev(np.concatenate([beta, decay]))).realize()
cw_st = Q.dev(np.stack([np.zeros((CONV, CP), np.float32), pad(cw.T)])); nw_st = Q.dev(np.stack([np.zeros_like(nw), nw]))
o_rows = rd(K.gdn_tok(St, Ct, ix, pb, xb, zb, bd, cw_st, nw_st, NV, NK, DK, DV, C, CONV)).reshape(L.R, NV * DV)
# the same inputs through gdn_tok2 (state streamed through LSRAM): fresh copies of the state and ring
St2 = K.buf("S_tok_t2", 2 * NV * DK * DV, Q.dtypes.float32); St2[NV * DK * DV:].assign(Q.dev(S0.ravel())).realize()
Ct2 = K.buf("C_tok_t2", 2 * CONV * CP, Q.dtypes.float32); Ct2[CONV * CP:].assign(Q.dev(pad(ring).ravel())).realize()
o2 = rd(K.gdn_tok2(St2, Ct2, ix, pb, xb, zb, bd, cw_st, nw_st, NV, NK, DK, DV, C, CONV)).reshape(L.R, NV * DV)
S2g = OA.host_invalidate(St2).numpy()[NV * DK * DV:].reshape(NV, DK, DV); r2 = OA.host_invalidate(Ct2).numpy()[CONV * CP:].reshape(CONV, CP)[:, :C]
S_new = OA.host_invalidate(St).numpy()[NV * DK * DV:].reshape(NV, DK, DV); ring_new = OA.host_invalidate(Ct).numpy()[CONV * CP:].reshape(CONV, CP)[:, :C]
print(f"gdn_tok: o max|d| {np.abs(o_rows[0] - oref.ravel()).max():.3g} of {np.abs(oref).max():.3g} | rows 1.. zero {bool(np.abs(o_rows[1:]).max() == 0)} | S max|d| {np.abs(S_new - S2).max():.3g}"
      f" | ring: new row at slot {pos % CONV} {bool((ring_new[pos % CONV] == xr).all())}, others kept {bool(all((ring_new[s] == ring[s]).all() for s in range(CONV) if s != pos % CONV))}")
t0 = time.time(); [K.gdn_tok(St, Ct, ix, pb, xb, zb, bd, cw_st, nw_st, NV, NK, DK, DV, C, CONV) for _ in range(5)]; print(f"   gdn_tok {(time.time() - t0) / 5 * 1e3:.1f} ms")
print(f"gdn_tok2: o max|d| {np.abs(o2[0] - oref.ravel()).max():.3g} | S max|d| {np.abs(S2g - S2).max():.3g} | ring new row {bool((r2[pos % CONV] == xr).all())}, others kept {bool(all((r2[s_] == ring[s_]).all() for s_ in range(CONV) if s_ != pos % CONV))}")
# ---- gemv with the row norm inside
x = rng.standard_normal((L.R, 5120)).astype(np.float32); w = rng.standard_normal((96, 5120)).astype(np.float32)
ref = ((x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) @ w.T).ravel(); new = rd(K.gemv(Q.dev(x), Q.dev(w), 5120, 96, "t7", norm=True))
print(f"gemv32 norm: max|d| {np.abs(new - ref).max():.3g} of {np.abs(ref).max():.3g}")
w_st = Q.dev(np.stack([np.zeros_like(w), w])); new2 = rd(K.gemv(Q.dev(x), w_st, 5120, 96, "t8", norm=True, idx=ix))
xt8 = Q.dev(x); t0 = time.time(); [K.gemv(xt8, w_st, 5120, 96, "t8", norm=True, idx=ix) for _ in range(5)]
print(f"gemv32 norm stacked (idx 1): max|d| {np.abs(new2 - ref).max():.3g} | {(time.time() - t0) / 5 * 1e3:.1f} ms")

# ---- the fused decode attention (norms, partial RoPE, cache write, attention, gate) vs numpy
NH, NKV, HD, TMAX, ROT = R.NH, R.NKV, R.HD, 512, R.ROT; rng = np.random.default_rng(13); pos = 200; g = NH // NKV
qg = rng.standard_normal((NH, 2 * HD)).astype(np.float32) * 2; kv = rng.standard_normal(2 * NKV * HD).astype(np.float32) * 2
qn, kn = (1 + 0.1 * rng.standard_normal(HD)).astype(np.float32), (1 + 0.1 * rng.standard_normal(HD)).astype(np.float32)
Kc = rng.standard_normal((TMAX, NKV, HD)).astype(np.float32); Vc = rng.standard_normal((TMAX, NKV, HD)).astype(np.float32)
def nr(x, w):
  y = x / np.sqrt((x * x).mean(-1, keepdims=True) + R.EPS) * w; return R.rope(y[None], np.array([pos]))[0]
qh = nr(qg[:, :HD], qn); kh = nr(kv[:NKV * HD].reshape(NKV, HD), kn); vh = kv[NKV * HD:].reshape(NKV, HD)
Kf, Vf = Kc.copy(), Vc.copy(); Kf[pos], Vf[pos] = kh, vh
sref = np.einsum("hd,thd->ht", qh, np.repeat(Kf[:pos + 1], g, 1)) / np.sqrt(HD); pw = np.exp(sref - sref.max(1, keepdims=True)); pw /= pw.sum(1, keepdims=True)
oref = np.einsum("ht,thd->hd", pw, np.repeat(Vf[:pos + 1], g, 1)) / (1 + np.exp(-qg[:, HD:]))
Ka = K.buf("K2_t", 2 * TMAX * NKV * HD, Q.dtypes.float32); Va = K.buf("V2_t", 2 * TMAX * NKV * HD, Q.dtypes.float32)
Ka[TMAX * NKV * HD:].assign(Q.dev(Kc.ravel())).realize(); Va[TMAX * NKV * HD:].assign(Q.dev(Vc.ravel())).realize()
qr = np.zeros((L.R, NH * 2 * HD), np.float32); qr[0] = qg.ravel(); kvr = np.zeros((L.R, 2 * NKV * HD), np.float32); kvr[0] = kv
qrb = K.buf("qr2_t", qr.size, Q.dtypes.float32); qrb.assign(Q.dev(qr.ravel())).realize(); kvb = K.buf("kv2_t", kvr.size, Q.dtypes.float32); kvb.assign(Q.dev(kvr.ravel())).realize()
qnb = K.buf("qn2_t", 2 * HD, Q.dtypes.float32); qnb.assign(Q.dev(np.concatenate([np.zeros(HD, np.float32), qn]))).realize()
knb = K.buf("kn2_t", 2 * HD, Q.dtypes.float32); knb.assign(Q.dev(np.concatenate([np.zeros(HD, np.float32), kn]))).realize()
inv = 1.0 / (R.THETA ** (np.arange(0, ROT, 2, dtype=np.float64) / ROT)); f = np.arange(TMAX, dtype=np.float64)[:, None] * inv[None]; emb = np.concatenate([f, f], -1)
rp = K.buf("rope2_t", TMAX * 2 * ROT, Q.dtypes.float32); rp.assign(Q.dev(np.concatenate([np.cos(emb), np.sin(emb)], -1).astype(np.float32).ravel())).realize()
ix2 = K.buf("aix2_t", 2, Q.dtypes.int32); ix2.assign(Q.dev(np.array([1, pos], np.int32))).realize()
o = rd(K.attn_dec2(qrb, kvb, qnb, knb, rp, Ka, Va, ix2, NH, NKV, HD, TMAX, ROT)).reshape(L.R, NH * HD)
Kb = OA.host_invalidate(Ka).numpy()[TMAX * NKV * HD:].reshape(TMAX, NKV, HD)
print(f"attn_dec2 (T {TMAX}, pos {pos}): max|d| {np.abs(o[0] - oref.ravel()).max():.3g} of {np.abs(oref).max():.3g} | cache k row max|d| {np.abs(Kb[pos] - kh).max():.3g}")
t0 = time.time(); [K.attn_dec2(qrb, kvb, qnb, knb, rp, Ka, Va, ix2, NH, NKV, HD, TMAX, ROT) for _ in range(5)]; print(f"   attn_dec2 {(time.time() - t0) / 5 * 1e3:.2f} ms")

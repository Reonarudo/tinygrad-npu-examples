"""Speculative decoding's multi-token kernels against M sequential runs of the single-token decode kernels, on random inputs:
gdn_tokm (outputs, the per-token state banks, the final state, the raw rows), gdn_commit (a accepted of M: state + ring),
attn_decm (outputs, the new cache rows). python3 check_spec_kernels.py [M]"""
import os, sys, numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tinygrad")))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tinygrad import Tensor, dtypes
from zy import OA
import qwen38_ref as R
from qwen38_npu import dev, Layers
from qwen38_kernels import Kernels, NT, ring_pitch

M = int(sys.argv[1]) if len(sys.argv) > 1 else 4
rng = np.random.default_rng(1); NRB = 2; ROWS = 24; POS = int(os.environ.get("CHECK_POS", "9"))   # CHECK_POS: the verify position
C = 2 * R.NK * R.DK + R.NV * R.DV; CP = ring_pitch(C); SZ = R.NV * R.DK * R.DV   # the ring / taps / raw rows: pitch CP
K1, KM = Kernels(NRB, ROWS, R.EPS, real=1), Kernels(NRB, ROWS, R.EPS, real=M)
host = lambda t: OA.host_invalidate(t).numpy().copy()
def i32(v): return Tensor(np.array(v, np.int32), device="ZHOUYI").realize()

def tiles(rows, N):
  """fp32 rows [<=4, N] -> the GEMM's C tiles [groups][nrb][3][192] with the rows in row block 0 (N padded to 48)."""
  ng = -(-N // 48); t = np.zeros((ng, NRB, 3, 192), np.float32)
  for r in range(rows.shape[0]):
    for c in range(N): t[c // 48, 0, (c % 48) // 16, (r // 4) * 64 + ((c % 16) // 4) * 16 + (r % 4) * 4 + c % 4] = rows[r, c]
  return t.ravel()
def a_rows(half_u16, K, M_):
  """compact A (uint16 halves, rt = ceil(M_ / 4) row tiles) -> fp32 rows [M_, K]."""
  rt = -(-M_ // 4); h = half_u16.view(np.float16).reshape(K // 128, 32, rt, 4, 4).astype(np.float32)   # [slice][kq][tile][row][4 cols]
  return np.stack([h[:, :, r // 4, r % 4, :].reshape(-1) for r in range(M_)])
def pad(a): return np.pad(a.reshape(-1, C), ((0, 0), (0, CP - C))).ravel()      # rows of C -> rows at pitch CP (zero padding)
def rel(a, b): return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))

# ---- DeltaNet: one layer (a stack of 1), random state, ring, rows, weights
S0 = (rng.standard_normal(SZ) * 0.05).astype(np.float32); Ring0 = pad(rng.standard_normal(R.CONV * C).astype(np.float32))
qkv = rng.standard_normal((M, C)).astype(np.float32); zr = rng.standard_normal((M, R.NV * R.DV)).astype(np.float32)
xin = np.zeros((ROWS, R.H), np.float32); xin[:M] = rng.standard_normal((M, R.H))
wab = Layers.ab3((rng.standard_normal((2 * R.NV, R.H)) * 0.02).astype(np.float32))
adt = np.stack([-np.exp(rng.standard_normal(R.NV) * 0.3), rng.standard_normal(R.NV) * 0.5]).astype(np.float32).ravel()
cwt = pad((rng.standard_normal((R.CONV, C)) * 0.4).astype(np.float32)); nw = (1 + rng.standard_normal(R.DV) * 0.1).astype(np.float32)
Wab, Adt, Cwt, Nw, idx0 = dev(wab), dev(adt), dev(cwt), dev(nw), i32([0])

# sequential: gdn_tok3 on each row in turn (row t moved to row 0), the state and ring carried
Sd, Cd = dev(S0), dev(Ring0); seq_out, seq_S = [], []
for t in range(M):
  x1 = np.zeros((ROWS, R.H), np.float32); x1[0] = xin[t]
  o = K1.gdn_tok3(Sd, Cd, idx0, i32([POS + t]), dev(tiles(qkv[t:t + 1], C)), dev(tiles(zr[t:t + 1], R.NV * R.DV)), dev(x1), Wab, Adt, Cwt, Nw,
                  R.NV, R.NK, R.DK, R.DV, C, R.CONV, R.H)
  seq_out.append(a_rows(host(o), R.NV * R.DV, 1)[0]); seq_S.append(host(Sd))
seq_ring = host(Cd)
# multi-token
Sm, Cm = dev(S0), dev(Ring0); banks = dev(np.zeros(max(1, M - 1) * SZ, np.float32)); rawm = dev(np.zeros(M * CP, np.float32))
om = KM.gdn_tokm(Sm, Cm, idx0, i32([POS]), dev(tiles(qkv, C)), dev(tiles(zr, R.NV * R.DV)), dev(xin), Wab, Adt, Cwt, Nw, banks, rawm,
                 R.NV, R.NK, R.DK, R.DV, C, R.CONV, R.H, 1)
got = a_rows(host(om), R.NV * R.DV, M); B = host(banks).reshape(-1, SZ); Sfin = host(Sm)
print(f"gdn_tokm M={M}:")
for t in range(M): print(f"   token {t}: output rel err {rel(got[t], seq_out[t]):.2e}" + (f" | state bank {rel(B[t], seq_S[t]):.2e}" if t < M - 1 else f" | final state {rel(Sfin, seq_S[t]):.2e}"))
print(f"   raw rows {rel(host(rawm).reshape(M, CP)[:, :C], qkv):.2e} | ring untouched: {bool((host(Cm) == Ring0).all())}")
# commit: a accepted -> the state after a tokens, the ring after a tokens
for a in sorted({1, max(1, M // 2), M}):
  Sc, Cc = dev(Sfin), dev(Ring0)
  KM.gdn_commit(Sc, banks, Cc, rawm, i32([a]), i32([POS]), R.NV, R.DK, R.DV, C, R.CONV, M, 1)
  ring_a = Ring0.reshape(R.CONV, CP).copy()
  for i in range(a): ring_a[(POS + i) % R.CONV, :C] = qkv[i]
  print(f"   commit a={a}: state {rel(host(Sc), seq_S[a - 1]):.2e} | ring exact {bool((host(Cc).reshape(R.CONV, CP) == ring_a).all())}")

# ---- attention: one layer's cache (a stack of 1) filled below POS, random rows
TMAX = max(64, POS + 16); QR, KR = R.NH * 2 * R.HD, 2 * R.NKV * R.HD
q = rng.standard_normal((M, QR)).astype(np.float32); kv = rng.standard_normal((M, KR)).astype(np.float32)
K0 = np.zeros(TMAX * R.NKV * R.HD, np.float32); V0 = np.zeros_like(K0); K0[:POS * R.NKV * R.HD] = rng.standard_normal(POS * R.NKV * R.HD); V0[:POS * R.NKV * R.HD] = rng.standard_normal(POS * R.NKV * R.HD)
qn, kn = dev((1 + rng.standard_normal(R.HD) * 0.1).astype(np.float32)), dev((1 + rng.standard_normal(R.HD) * 0.1).astype(np.float32))
inv = 1.0 / (R.THETA ** (np.arange(0, R.ROT, 2, dtype=np.float64) / R.ROT)); f = np.arange(TMAX, dtype=np.float64)[:, None] * inv[None]; emb = np.concatenate([f, f], -1)
rope = dev(np.concatenate([np.cos(emb), np.sin(emb)], -1).astype(np.float32).ravel())
def rows(a, n): z = np.zeros((ROWS, a.shape[1]), np.float32); z[:n] = a; return dev(z)
Kd, Vd = dev(K0), dev(V0); seq_o = []
for r in range(M):
  o = K1.attn_dec2(rows(q[r:r + 1], 1), rows(kv[r:r + 1], 1), qn, kn, rope, Kd, Vd, i32([0, POS + r]), R.NH, R.NKV, R.HD, TMAX, R.ROT)
  seq_o.append(host(o)[0])
Km, Vm = dev(K0), dev(V0)
om = KM.attn_decm(rows(q, M), rows(kv, M), qn, kn, rope, Km, Vm, i32([0, POS]), R.NH, R.NKV, R.HD, TMAX, R.ROT)
go = host(om)
print(f"attn_decm M={M}: " + " ".join(f"row {r} {rel(go[r], seq_o[r]):.2e}" for r in range(M))
      + f" | K cache {rel(host(Km), host(Kd)):.2e} V cache {rel(host(Vm), host(Vd)):.2e}")

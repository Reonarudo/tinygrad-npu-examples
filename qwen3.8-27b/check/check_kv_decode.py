"""The attention layer's decode path (qwen38_generate: fixed-length K / V cache, the jitted `att_step`): a 12-token prefill of
layer 3 then one decode step vs the numpy reference over the 13 tokens at once. python3 check_kv_decode.py"""
import os, sys, numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tinygrad")))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tinygrad import Tensor
from tinygrad.engine.jit import TinyJit
from zy import OA
import qwen38_ref as R
from qwen38_npu import dev, DEV
from qwen38_generate import Model, rope_tables, qkv_proj, attend
l, n, TMAX = 3, 12, 64
rng = np.random.default_rng(0); x = rng.standard_normal((n + 1, R.H)).astype(np.float32)
dir_, cache = os.environ.get("QWEN_DIR", "/mnt/ssd/qwen3.8-27b-fp8"), os.environ.get("QWEN_NPU", "/mnt/ssd/qwen3.8-27b-npu-s1")
want = R.attention_layer(R.Weights(dir_), l, x)
M = Model(n, cache, dir_, tmax=TMAX)
# prefill of layer 3 (eager, as Model.prefill does it)
L = M.pre; xp = np.zeros((L.R, R.H), np.float32); xp[:n] = x[:n]
cos, sin = rope_tables(0, L.R); i, j = np.arange(L.R)[:, None], np.arange(L.R)[None]
mask = dev(np.where((j <= i) & (i < n), 0.0, -1e30).astype(np.float32))
slot, meta, sm = L.load(l); q, k, v, gate = qkv_proj(L, dev(xp), slot, meta, sm, cos, sin)
yp = attend(L, dev(xp), q, gate, k, v, mask, slot, meta, sm).realize(); got_p = OA.host_invalidate(yp).numpy()[:n]
Kc = Tensor.cat(k[:n], Tensor.zeros(TMAX - n, R.NKV, R.HD, device=DEV), dim=0).contiguous().realize()
Vc = Tensor.cat(v[:n], Tensor.zeros(TMAX - n, R.NKV, R.HD, device=DEV), dim=0).contiguous().realize()
# one decode step through the jitted body, twice (the second call is the JIT's replay)
L = M.dec; slot, meta, sm = L.load(l); M.meta_att = meta; jit = TinyJit(M.att_step); pos = n
xd = np.zeros((L.R, R.H), np.float32); xd[0] = x[n]
onehot = dev((np.arange(TMAX) == pos).astype(np.float32).reshape(TMAX, 1, 1))
dmask = dev(np.where(np.arange(TMAX)[None] <= pos, 0.0, -1e30).astype(np.float32).reshape(1, TMAX).repeat(L.R, 0))
cosd, sind = rope_tables(pos, L.R)
for rep in range(2):
  yd, K2, V2 = jit(dev(xd), Kc, Vc, onehot, dmask, cosd, sind, *[slot[k_] for k_ in M.ATT_LIN], *[sm[k_] for k_ in M.ATT_SMALL])
  got_d = OA.host_invalidate(yd).numpy()[0]
  d = got_d - want[n]; dd = want[n] - x[n]
  print(f"decode row (call {rep}): max|d| {np.abs(d).max():.3g} of max|y| {np.abs(want[n]).max():.3g} | rel err of the layer's delta {np.linalg.norm(d)/np.linalg.norm(dd):.3g}")
d = got_p - want[:n]; dd = want[:n] - x[:n]
print(f"prefill rows: max|d| {np.abs(d).max():.3g} | rel err of the layer's delta {np.linalg.norm(d)/np.linalg.norm(dd):.3g}")
print("cache row 12 written:", bool(np.abs(OA.host_invalidate(K2).numpy()[pos]).max() > 0), "| row 13 still zero:", bool(np.abs(OA.host_invalidate(K2).numpy()[pos + 1]).max() == 0))

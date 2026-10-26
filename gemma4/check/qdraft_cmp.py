"""The qdot drafter (gemma4_draft.QDrafter) against the matrix-unit one (gemma4_mtp.Drafter) on the same inputs after a real prefill:
the drafts' ids and probabilities over a chain, h_next's error, and the time of a draft. python3 qdraft_cmp.py [k]"""
import os, sys, time, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import gemma4_npu as GN, gemma4_generate as GG, gemma4_mtp as GM, gemma4_draft as GD, gemma4_ref as GR
from tinygrad import Device
k = int(sys.argv[1]) if len(sys.argv) > 1 else 6
cache = GG.DEFAULT_CACHE; meta = json.load(open(os.path.join(cache, "gemma4.json"))); M = GR.Model(meta["gguf"], cache_layers=False)
T = GG.Model(M, cache, 1024); mtp = GM.default_mtp(meta["gguf"])
d0 = GM.Drafter(T, mtp, os.path.join(cache, "mtp")); d1 = GD.QDrafter(T, mtp, os.path.join(cache, "mtp"))
ids = GN.PROMPT; T.warmup((8,)); nxt = T.prefill(ids, 8); P = len(ids); h = T.hidden(*T.last_rows)
for name, dr in (("matrix-unit", d0), ("qdot", d1)):
  for _ in range(2): dr.chain(nxt, h, P, 2)                    # capture
  t0 = time.perf_counter(); d, pr = dr.chain(nxt, h, P, k, 0.0); Device[GN.DEV].synchronize(); dt = time.perf_counter() - t0
  print(f"{name:12s}: drafts {d} probabilities {[round(p, 3) for p in pr]} ({1e3 * dt / len(d):.2f} ms a draft over {len(d)})", flush=True)
# h_next of one draft from the same (tok, h)
d0.set_pos(P); d1.set_pos(P); t0, p0, h0 = d0.step(nxt, h); t1, p1, h1 = d1.step(nxt, h)
print(f"one draft from the same input: ids {t0} / {t1}, probabilities {p0:.4f} / {p1:.4f}, h_next rel diff {np.linalg.norm(h1 - h0) / np.linalg.norm(h0):.2e}", flush=True)
def timed(dr, n=20):
  ts = []
  for _ in range(3):
    t0 = time.perf_counter()
    for _ in range(n): dr.step(nxt, h)
    ts.append((time.perf_counter() - t0) / n)
  return min(ts)
print(f"a draft (host loop, {20} steps): matrix-unit {1e3 * timed(d0):.2f} ms, qdot {1e3 * timed(d1):.2f} ms", flush=True)

#!/usr/bin/env python3
"""The MTP drafter on the NPU (gemma4_mtp.Drafter) against the numpy reference (gemma4_ref.Drafter), on the same input: the prompt
is prefilled on the NPU (chunks of --pm rows), then a chain of --d drafts at the next position runs on both -- the NPU's over the
NPU's caches, numpy's over its own (gemma4_ref.Model.forward of the prompt) -- each step fed the NPU's previous draft and h. Reports
per draft both ids, the NPU's probability, and the relative error of h_next; and the final-normed hidden h the chain starts from.

    python3 check/mtp_npu.py [--cache /mnt/ssd/gemma4-e2b-npu] [--d 4] [--pm 8] ID ...
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gemma4_generate as GG                                              # noqa: E402
import gemma4_mtp as GM                                                  # noqa: E402
import gemma4_ref as GR                                                  # noqa: E402

def rel(a, b): return float(np.linalg.norm(np.asarray(a, np.float64) - b) / max(np.linalg.norm(np.asarray(b, np.float64)), 1e-30))

def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--cache", default=GG.DEFAULT_CACHE); ap.add_argument("--d", type=int, default=4)
  ap.add_argument("--pm", type=int, default=8); ap.add_argument("ids", nargs="*", type=int); a = ap.parse_args()
  meta = json.load(open(os.path.join(a.cache, "gemma4.json"))); ids = a.ids or GG.GN.PROMPT
  T = GR.Model(meta["gguf"], cache_layers=False); mtp = GM.default_mtp(meta["gguf"])
  m = GG.Model(T, a.cache, 1024); dr = GM.Drafter(m, mtp, os.path.join(a.cache, "mtp")); m.warmup([a.pm])
  first = m.prefill(ids, a.pm); h = m.hidden(*m.last_rows); P = len(ids)
  t0 = time.perf_counter(); kv = {}; lg, hn = T.forward(ids, 0, kv); D = GR.Drafter(mtp, T)
  print(f"numpy reference over the prompt: {time.perf_counter() - t0:.0f} s; first token NPU {first} numpy {int(lg[-1].argmax())}; h rel {rel(h, hn[-1]):.3g}", flush=True)
  dr.set_pos(P); tok, hh, worst, same = first, h, 0.0, True
  for j in range(a.d):
    did, p, hnext = dr.step(tok, hh)
    rl, rh = D.step(tok, hh, P, kv); rid = int(rl.argmax()); e = np.exp(rl - rl.max()); rp = float(e[rid] / e.sum())
    r = rel(hnext, rh); worst = max(worst, r); same &= did == rid
    print(f"draft {j + 1}: NPU {did} (p {p:.4f}) numpy {rid} (p {rp:.4f}) | h_next rel {r:.3g}", flush=True)
    tok, hh = did, hnext
  print(f"DRAFT CHECK {'PASS' if same and worst < 2e-2 else 'FAIL'}: ids {'identical' if same else 'DIFFER'}, worst h_next rel {worst:.3g}", flush=True)

if __name__ == "__main__": main()

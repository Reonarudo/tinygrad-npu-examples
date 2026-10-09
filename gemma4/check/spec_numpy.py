#!/usr/bin/env python3
"""gemma4_generate.Model.spec_generate's bookkeeping on the numpy reference, no NPU: the same method run against a stand-in whose
forward(toks, pos) is gemma4_ref.Model's (the K / V rows past an accepted prefix dropped, as the NPU's next pass rewrites them),
whose hidden() is the reference's final-normed row, and a drafter that is gemma4_ref.Drafter. Its ids must equal the reference's
plain greedy ids exactly (one arithmetic for both) -- a check of the acceptance, the positions, the drafter's (token, h, P).

    python3 check/spec_numpy.py [--gguf T.gguf] [--n 24] [--k 3] [--pm 8] ID ...
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gemma4_generate as GG                                              # noqa: E402
import gemma4_ref as GR                                                  # noqa: E402

class RefNPU:
  """The pieces of gemma4_generate.Model that spec_generate calls, on the numpy reference."""
  def __init__(self, T): self.T, self.kv, self.tmax, self.h = T, {}, 4096, {}
  def forward(self, toks, pos, head=True):
    for l, (K, V, P) in list(self.kv.items()):                            # rows at positions >= pos: the rejected rows a pass rewrites
      keep = P < pos; self.kv[l] = (K[keep], V[keep], P[keep])
    lg, hn = self.T.forward(list(toks), pos, self.kv); self.h[len(toks)] = hn
    return lg.argmax(-1), None
  def hidden(self, m, row): return self.h[m][row]
  def prefill(self, ids, pm=1):
    n = len(ids)
    for c0 in range(0, n, pm):
      ch = list(ids[c0:c0 + pm]); g, _ = self.forward(ch, c0); a = len(ch)
    self.last_rows = (a, a - 1); return int(g[a - 1])

class RefDrafter:
  def __init__(self, D, npu): self.D, self.npu = D, npu
  def chain(self, tok, h, P, k, tau=0.0):
    d, pr = [], []
    for _ in range(k):
      lg, h = self.D.step(tok, h, P, self.npu.kv); tok = int(lg.argmax()); e = np.exp(lg - lg.max()); p = float(e[tok] / e.sum())
      d.append(tok); pr.append(p)
      if p < tau: break
    return d, pr

def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--gguf", default=GR.DEFAULT); ap.add_argument("--n", type=int, default=24)
  ap.add_argument("--k", type=int, default=3); ap.add_argument("--pm", type=int, default=8); ap.add_argument("--tau", type=float, default=0.0)
  ap.add_argument("ids", nargs="*", type=int); a = ap.parse_args(); ids = a.ids or GG.GN.PROMPT
  T = GR.Model(a.gguf, cache_layers=os.environ.get("REF_CACHE", "0") == "1"); D = GR.Drafter(os.path.join(os.path.dirname(a.gguf), "mtp-" + os.path.basename(a.gguf)), T)
  t0 = time.perf_counter(); ref = T.greedy(ids, a.n - 1); print(f"reference greedy ({time.perf_counter() - t0:.0f} s): {ref}", flush=True)
  npu = RefNPU(T); st = {}
  out = GG.Model.spec_generate(npu, ids, a.n, RefDrafter(D, npu), a.k, a.tau, pm=a.pm, stats=st)
  print(f"spec_generate on the reference: {out}", flush=True)
  print(f"SPEC NUMPY {'PASS' if out == ref[:len(out)] and len(out) == min(a.n, len(ref)) else 'FAIL'}: {st['passes']} passes, "
        f"{st['accepted']} of {st['drafted']} drafts accepted", flush=True)

if __name__ == "__main__": main()

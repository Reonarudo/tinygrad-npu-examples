#!/usr/bin/env python3
"""The MTP draft head (gemma4_ref.Drafter, the `gemma4-assistant` GGUF) against its target in numpy: greedy decoding of the
target, and at every step a chain of D drafts from (the token just picked, the target's post-final-norm hidden of the row that
picked it) at that token's position over the target's caches; the k-th draft is accepted when the first k drafts all equal the
target's next greedy tokens. High acceptance = the drafter's reference is wired right (input concat, shared caches, constant
position, the 256-wide tied head, post-projection).

    python3 check/mtp_accept.py [--gguf target.gguf] [--mtp mtp.gguf] [--n 32] [--d 3] ID ...
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gemma4_ref as G                                                     # noqa: E402

def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--gguf", default=G.DEFAULT)
  ap.add_argument("--mtp", default=os.path.join(os.path.dirname(G.DEFAULT), "mtp-" + os.path.basename(G.DEFAULT)))
  ap.add_argument("--n", type=int, default=32); ap.add_argument("--d", type=int, default=3); ap.add_argument("ids", nargs="+", type=int)
  a = ap.parse_args(); T = G.Model(a.gguf); D = G.Drafter(a.mtp, T); t0 = time.perf_counter()
  kv = {}; logits, h = T.forward(a.ids, 0, kv); pos = len(a.ids); out = [int(logits[-1].argmax())]; drafts = []
  while len(out) < a.n:
    tok, hh, ch = out[-1], h[-1], []
    for _ in range(a.d):                                        # the chain: all at the position of `tok` (not yet in the caches)
      lg, hh = D.step(tok, hh, pos, kv); tok = int(lg.argmax()); ch.append(tok)
    drafts.append((len(out), ch))
    logits, h = T.forward([out[-1]], pos, kv); pos += 1; out.append(int(logits[-1].argmax()))
  acc = np.zeros(a.d); tot = np.zeros(a.d)
  for i, ch in drafts:
    for k in range(a.d):
      if i + k >= len(out): break
      tot[k] += 1
      if ch[:k + 1] == out[i:i + k + 1]: acc[k] += 1
  print(f"target greedy ({time.perf_counter() - t0:.0f} s): {out}")
  print("acceptance of draft k given drafts < k accepted:", " ".join(f"{k + 1}: {acc[k] / max(1, acc[k - 1] if k else tot[k]):.0%}" for k in range(a.d)),
        "| cumulative:", " ".join(f"{acc[k] / tot[k]:.0%}" for k in range(a.d)), f"({int(tot[0])} positions)")

if __name__ == "__main__": main()

"""Scheduling the drafts by the target's confidence: from spec runs' --json logs (pass_log: the target's top-1 probability of the token a
pass drafts from, drafts made, drafts accepted, rows, the drafter's probabilities), the acceptance by confidence bin, and policies
k(p) simulated on the same passes (a pass costs verify(rows) + drafts x t_draft; the chain's acceptance is the recorded prefix).
python3 sched_an.py run1.json run2.json ..."""
import sys, json, numpy as np
VER = {1: 119.0, 2: 120.4, 3: 122.3, 4: 123.0, 5: 128.4, 6: 130.6, 7: 130.7, 8: 132.2}; TD = 4.5   # ms (geo_bench; the qdot draft)
logs = []
for f in sys.argv[1:]:
  j = json.load(open(f)); logs.append((f, j["pass_log"]))
allp = [(p, nd, acc, m, prs) for _, L in logs for (p, nd, acc, m, prs) in L]
print(f"{len(allp)} passes from {len(logs)} runs; drafts a pass {np.mean([x[1] for x in allp]):.2f}, accepted {np.mean([x[2] for x in allp]):.2f}")
bins = [0, 0.3, 0.5, 0.7, 0.9, 0.99, 1.01]
for lo, hi in zip(bins, bins[1:]):
  sel = [x for x in allp if lo <= x[0] < hi]
  if sel: print(f"  target p in [{lo:.2f}, {hi:.2f}): {len(sel):3d} passes, drafts made {np.mean([x[1] for x in sel]):.2f}, accepted {np.mean([x[2] for x in sel]):.2f} (all {np.mean([x[2] == x[1] for x in sel]):.0%})")
def simulate(policy, tau=0.5, name=""):
  """Each recorded pass replayed with k = policy(p): the drafts made are the first k of the recorded chain (cut at the recorded tau
  stop); accepted = min(recorded accepted, made) (the recorded chain's prefix agrees as it did)."""
  tot_tok = tot_ms = 0.0; n = 0
  for f, L in logs:
    for p, nd, acc, m, prs in L:
      k = policy(p); made = 0
      for j in range(min(k, len(prs))):
        made += 1
        if prs[j] < tau: break
      made_acc = min(acc, made); rows = 1 + made; tok = 1 + made_acc
      tot_tok += tok; tot_ms += VER[min(8, max(1, rows))] + made * TD; n += 1
  print(f"  policy {name:28s}: {tot_tok / n:.2f} tokens a pass, {tot_ms / n:.1f} ms a pass -> {1e3 * tot_tok / tot_ms:.2f} tok/s")
print("policies (replayed on the recorded chains; the recorded runs made up to 7 drafts at tau 0):")
for k in (4, 6, 7): simulate(lambda p, k=k: k, name=f"k {k}, tau 0.5")
simulate(lambda p: 7 if p >= 0.9 else 5 if p >= 0.5 else 3, name="7 / 5 / 3 by p 0.9 / 0.5")
simulate(lambda p: 7 if p >= 0.7 else 3, name="7 if p >= 0.7 else 3")
simulate(lambda p: 7 if p >= 0.5 else 2, name="7 if p >= 0.5 else 2")
simulate(lambda p: 7 if p >= 0.9 else 4 if p >= 0.6 else 1, name="7 / 4 / 1 by p 0.9 / 0.6")
print("the drafter's own threshold (constant k 7):")
for tau in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8): simulate(lambda p: 7, tau=tau, name=f"k 7, tau {tau}")
for tau in (0.5, 0.7): simulate(lambda p: 6, tau=tau, name=f"k 6, tau {tau}")

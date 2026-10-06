#!/usr/bin/env python3
"""The verify-shaped multi-row passes (speculative decoding's: compact rows mode, rows 3 / 4 / 6 / 8) against plain greedy decoding,
on the NPU. One process: prefill + `--n` plain greedy steps (decode, 1 row) -> the reference ids g; prefill again, then
Model.spec_generate's loop (verify + commit) with ORACLE drafts -- g's continuation, every third pass with its last draft replaced
by a wrong token (so passes reject too and commit() keeps a shorter prefix) -- the geometry cycling through QWEN_SPEC_GEOS.
Every token the verify passes emit is a verify row's argmax: the run is exact iff the ids equal g.
QWEN_SPEC_TREE=leaf (the leaf tree's kernels: gdn_tokl, attn_partt, gdn_commit_tree(defer); geometries 2..QWEN_SPEC): every other
pass is a TREE -- the chain's draft at depth k wrong, a leaf at depth k holding the right token (and, room permitting, a wrong leaf
before it), so the pass commits the path through the leaf (spec_tree.accept, its K / V rows moved, its update slot pending).

    QWEN_SPEC=8 QWEN_SPEC_GEOS=3,4,6,8 python3 check/verify_oracle.py [--ids 760,...] [--n 40]
"""
import argparse, os, sys, time
os.environ.setdefault("QWEN_MODEL", "bonsai2-27b"); os.environ.setdefault("QWEN_NPU", "/mnt/ssd/bonsai2-npu")
os.environ.setdefault("QWEN_DIR", os.environ.get("BONSAI_GGUF", "/mnt/ssd/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf"))
os.environ.setdefault("QWEN_SPEC", "8"); os.environ.setdefault("QWEN_SPEC_GEOS", "3,4,6,8")
os.environ.setdefault("QWEN_DRAFT", "none")                               # oracle drafts: no MTP drafter loaded
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "qwen3.8-27b"))
import qwen38_generate as G                                              # noqa: E402
import spec_tree as ST                                                   # noqa: E402

def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--ids", default="760,6511,314,9338,369,11751,13,561,6511,314,9564,369")
  ap.add_argument("--n", type=int, default=40); a = ap.parse_args()
  ids = [int(t) for t in a.ids.split(",")]
  M = G.Model(len(ids), os.environ["QWEN_NPU"], os.environ["QWEN_DIR"])
  assert M.M, "QWEN_SPEC: the verify geometries"
  print(f"== {len(ids)} prompt tokens; verify geometries {M.geos}", flush=True)
  lg = M.prefill(ids); g = [int(lg.argmax())]
  for i in range(a.n - 1): g.append(int(M.step(g[-1], len(ids) + i).argmax()))
  print(f"   plain greedy ({a.n}): {g}", flush=True)
  lg = M.prefill(ids); first = int(lg.argmax()); assert first == g[0], (first, g[0])
  out, cur, pos, rows, acc, i = [first], first, len(ids), [], 0, 0
  t0 = time.perf_counter()
  while len(out) < a.n:                                                  # Model.spec_generate's loop, the geometry chosen per pass
    m = M.geos[i % len(M.geos)]; d = list(g[len(out):len(out) + m - 1]); d += [d[-1] if d else cur] * (m - 1 - len(d))
    if getattr(M, "leaf", False) and i % 2 == 1 and m >= 3:              # a tree: chain of m - 1 - nl rows, the right token on a leaf
      nl = 1 if (m == 3 or i % 4 == 1) else 2; nc = m - nl; k = 1 + (i // 4) % (nc - 1)   # the wrong chain draft's depth k (1 .. nc - 1)
      ch = list(d[:nc - 1]); right = ch[k - 1]; ch[k - 1] = (right + 7) % 1000
      parent = [-1] + list(range(nc - 1)); toks = [cur] + ch; resc = []
      if nl == 2: toks.append((right + 13) % 1000); parent.append(k - 1); resc.append((len(toks) - 1, k, 2))   # a wrong leaf first
      toks.append(right); parent.append(k - 1); resc.append((len(toks) - 1, k, 1))
      tree = ST.Tree(toks, parent, resc); M.pass_nc = nc
      v = [int(t) for t in M.verify(toks, pos, tree=tree)]
      path, new, _ = ST.accept(tree, v); a_ = len(new); assert path[-1] >= nc or len(path) < k + 1, (path, k)
      M.commit(a_, pos, m, path); out += new; cur, pos = out[-1], pos + a_
      rows.append((m, a_, "leaf" if path[-1] >= nc else "chain")); acc += a_ - 1; i += 1; continue
    if i % 3 == 2: d[-1] = (d[-1] + 1) % 1000                             # a wrong last draft: the pass rejects it
    v = [int(t) for t in M.verify([cur] + d, pos)]
    a_ = 1
    while a_ < m and d[a_ - 1] == v[a_ - 1]: a_ += 1
    M.commit(a_, pos, m); out += d[:a_ - 1] + [v[a_ - 1]]; cur, pos = out[-1], pos + a_
    rows.append((m, a_)); acc += a_ - 1; i += 1
  out = out[:a.n]; dt = time.perf_counter() - t0
  print(f"   verify path ({i} passes, (rows, accepted) {rows}, {dt:.1f} s): {out}", flush=True)
  same = sum(x == y for x, y in zip(out, g))
  print(f"VERIFY vs plain greedy: {same}/{len(g)} equal, first difference at {next((i for i, (x, y) in enumerate(zip(out, g)) if x != y), None)} -> "
        f"{'EXACT' if out == g else 'DIFFERENT'}", flush=True)
  sys.exit(0 if out == g else 1)

if __name__ == "__main__": main()

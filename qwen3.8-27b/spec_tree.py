"""Tree verification for speculative decoding (QWEN_SPEC_TREE): a verify pass of M = 8 rows holding the 6-row chain (the pass's
token + 5 drafts) and NR = 2 rescue rows, each the draft head's rank-2 token at one chain position (a sibling of that chain row:
the same parent, chain row j - 1). Pure Python on a few ints: the generator (qwen38_generate.py) calls these.

  QWEN_SPEC_TREE=off (default: the chain verify)
                 rescue2       the two rescue rows at the two chain positions whose draft has the lowest top-1 - top-2 margin
                               (the rule that replayed best on logged passes of the 27B: +0.215 token / pass)
                 fixed:<j1,j2> the rescue rows at chain positions j1 and j2 (1-based: position j = draft j - 1 = chain row j)
                 leaf          (the rotated-input models, bonsai2-27b) a pass of any row count up to QWEN_SPEC: the drafter's chain
                               (drafting on while the chain's path probability p_1 ... p_j >= QWEN_TREE_TC) and, in the rows left,
                               LEAVES -- the rank-2 / rank-3 candidates of the drafts, best path probability first, those >= QWEN_TREE_TL
                               (build_leaf; the kernels gdn_tokl / attn_partt / gdn_commit_tree(defer))

Rows and positions. Chain row k (0 <= k <= NC - 1, NC = 6) is the token at sequence position pos + k: row 0 the pass's token
`cur`, row k >= 1 draft k - 1. A rescue row r (NC <= r < M) for chain position j (1 <= j <= NC - 1) holds another candidate for
position pos + j: its parent is chain row j - 1, its depth is j, so it attends (and its DeltaNet state follows) rows 0..j-1 and
itself, and no chain row >= j. Each row's model token g[r] is the greedy next token after the path to r; the accepted path is the
longest root path on which every token equals the model's token at its parent, and the pass commits the path's tokens plus the
model's token at its last row -- exactly plain greedy decoding, as the chain verify (greedy
is deterministic, so a row's token depends only on its ancestors).

The kernels take the tree as one int32 table `table(tree)` = [depth[0..M-1], anc[0..M-1]] (anc[r]: a bit per row that row r
attends, itself included); the chain is `chain_table(M)` (depth r, bits 0..r), under which the tree kernels are bit-identical to
the chain kernels (the simulator gate). `gdn_commit_tree` takes `path_words(...)`: the committed path's rows.
"""
import numpy as np

MODES = ("off", "rescue2", "fixed", "leaf")
TREE_M, TREE_NR = 8, 2                                                    # rows a tree verify pass; of them the rescue rows (chain NC = 6)

def mode_from_env(env):
  """QWEN_SPEC_TREE -> None (off) or ("rescue2", None) | ("fixed", (j1, j2))."""
  v = env.get("QWEN_SPEC_TREE", "off").strip()
  if v in ("", "off", "0"): return None
  if v == "rescue2": return ("rescue2", None)
  if v == "leaf": return ("leaf", None)
  if v.startswith("fixed:"):
    js = tuple(int(x) for x in v[6:].split(",") if x); assert len(js) == 2 and js[0] != js[1] and all(1 <= j for j in js), f"QWEN_SPEC_TREE={v!r}: fixed:<j1,j2>, two distinct chain positions >= 1"
    return ("fixed", js)
  raise ValueError(f"QWEN_SPEC_TREE={v!r}: off | rescue2 | fixed:<j1,j2> | leaf")

def merged_top3(t3k):
  """A draft's top-3 over the head parts read (`t3k`: per part ([ids], [logits])) -> [(logit, id)] in rank order (the generator's
  top3_rank: ties by the larger logit first, then the smaller id -- sorted() on (-logit, id))."""
  return [(l, i) for _, l, i in sorted((-l, l, i) for ids, lgs in t3k for i, l in zip(ids, lgs))][:3]

def margins(t3):
  """Per draft: (the rank-2 token or None, top-1 - top-2 margin or +inf when the head returned fewer than 2 candidates)."""
  out = []
  for t3k in t3:
    m = merged_top3(t3k)
    out.append((int(m[1][1]), float(m[0][0] - m[1][0])) if len(m) >= 2 else (None, float("inf")))
  return out

class Tree:
  """toks[r], parent[r] (-1 for row 0), depth[r], and `rescue` = [(row, chain position j, rank)] for the rescue rows."""
  def __init__(self, toks, parent, rescue):
    self.toks, self.parent, self.rescue = [int(t) for t in toks], list(parent), list(rescue); self.M = len(toks)
    self.depth = [0] * self.M
    for r in range(1, self.M): self.depth[r] = self.depth[self.parent[r]] + 1
    self.anc = [0] * self.M                                                # a bit per attended row (ancestors and itself)
    for r in range(self.M):
      bits, q = 0, r
      while q >= 0: bits |= 1 << q; q = self.parent[q]
      self.anc[r] = bits
    for r in range(1, self.M): assert self.parent[r] < r, "parents precede their children (the kernels' row order)"
  def ancestors(self, r):
    """The root path to r inclusive, root first."""
    path = []
    while r >= 0: path.append(r); r = self.parent[r]
    return path[::-1]

def chain(toks):
  """The chain tree: row r's parent is r - 1."""
  return Tree(toks, [-1] + list(range(len(toks) - 1)), [])

def positions(mode, nd, t3):
  """The rescue rows' chain positions (1-based, distinct, each <= nd) from the mode and the drafts' head top-3 -> [(j, rank)]:
  rescue2: the two drafts with the lowest top-1 - top-2 margin (ties: the earlier position -- np.argsort kind="stable", as the
  replay's rule), each the rank-2 token; fixed: the given positions (a position past the drafts drawn falls back to the lowest
  margins); fewer than 2 drafts: the second rescue is the rank-3 token of position 1 (never both the same (j, rank))."""
  kind, js = mode; mg = margins(t3[:nd])
  order = [int(j) + 1 for j in np.argsort([m for _, m in mg], kind="stable")]
  if kind == "fixed": want = [j for j in js if 1 <= j <= nd] + [j for j in order if j not in js]
  else: want = order
  out = [(j, 1) for j in want[:2]]
  if len(out) < 2 and nd >= 1: out.append((want[0], 2))                   # one draft only: its rank-2 and rank-3
  return out

def build(cur, d, t3, mode, M=8, NR=2):
  """The pass's tree: rows 0..M-NR-1 the chain (cur, then the drafts padded with the last one, as the chain verify pads), rows
  M-NR.. the rescue rows (positions(mode, ...)); the rank-k sibling of chain position j is draft j - 1's merged top-3 entry k
  (absent from the head's top-3: the chain token itself, a harmless duplicate that cannot commit more than the chain)."""
  NC = M - NR; nd = len(d); assert 1 <= nd <= NC - 1, (nd, NC)
  toks = [cur] + list(d) + [d[-1]] * (NC - 1 - nd); parent = [-1] + list(range(NC - 1)); rescue = []
  for r, (j, rank) in zip(range(NC, M), positions(mode, nd, t3)):
    m3 = merged_top3(t3[j - 1]); tok = int(m3[rank][1]) if len(m3) > rank else int(toks[j])
    toks.append(tok); parent.append(j - 1); rescue.append((r, j, rank))
  while len(toks) < M: toks.append(toks[NC - 1]); parent.append(NC - 2); rescue.append((len(toks) - 1, NC - 1, 0))   # (never: NR rescues)
  return Tree(toks, parent, rescue)

def leaf_pick(pr, t3, B, tl):
  """The leaf tree's leaves for a chain of nd = len(pr) drafts (their top-1 probabilities `pr`, merged top-3 lists `t3`) within B
  rows: every draft's rank-2 / rank-3 candidate is a leaf candidate with the path probability p_1 ... p_(j-1) * p_j exp(l_r - l_1)
  (the draft head's softmax over its top-3 logits); those >= tl, best first (ties: the shallower, then rank 2), fill the B - 1 - nd
  rows the chain leaves -> [(j, rank)] (rank 1 = the 2nd candidate). The `dyn2` rule of a host-side replay of bonsai2-27b's drafts."""
  cand, P = [], 1.0
  for j, (p, t3k) in enumerate(zip(pr, t3), 1):
    m3 = merged_top3(t3k)
    for r in (1, 2):
      if r < len(m3):
        q = P * p * float(np.exp(m3[r][0] - m3[0][0]))
        if q >= tl: cand.append((-q, j, r))
    P *= p
  return [(j, r) for _, j, r in sorted(cand)[:max(0, B - 1 - len(pr))]]

def build_leaf(cur, d, t3, pr, B, tl):
  """The leaf tree's pass: rows 0..nd the chain (cur, the nd drafts; no padding: the pass's geometry is its row count), then the
  leaves of leaf_pick, each a sibling of chain row j (parent j - 1)."""
  nd = len(d); assert 1 <= nd <= B - 1 and len(t3) >= nd, (nd, B, len(t3))
  toks, parent, rescue = [cur] + list(d), [-1] + list(range(nd)), []
  for j, r in leaf_pick(pr[:nd], t3[:nd], B, tl):
    toks.append(int(merged_top3(t3[j - 1])[r][1])); parent.append(j - 1); rescue.append((len(toks) - 1, j, r))
  return Tree(toks, parent, rescue)

def accept(tree, g):
  """The model's token per row `g` -> (path rows root..last, the committed tokens, the chain's first rejected position or 0).
  The path: the chain's accepted prefix (rows 0..a-1, a = 1 + the drafts the model agreed with), extended by a rescue row whose
  parent is chain row a - 1 and whose token is the model's token there (the first such row); the committed tokens are the
  path's tokens after row 0... i.e. the path's tokens (row 0 = the pass's token, already in `out`) plus g at the path's last row."""
  M = tree.M; NC = M - len(tree.rescue); a = 1
  while a < NC and tree.toks[a] == int(g[a - 1]): a += 1
  path = list(range(a)); rej = a if a < NC else 0
  if rej:
    for r, j, _ in tree.rescue:
      if j == a and tree.toks[r] == int(g[a - 1]): path.append(r); break
  new = [tree.toks[r] for r in path[1:]] + [int(g[path[-1]])]
  return path, new, rej

def table(tree):
  """int32 [2 M]: depth[r], then anc[r] (the kernels' tree table)."""
  return np.array(tree.depth + tree.anc, np.int32)

def chain_table(M):
  return np.array(list(range(M)) + [(2 << r) - 1 for r in range(M)], np.int32)

def path_words(path, M, NC):
  """gdn_commit_tree's int32 [4 + M]: a (the tokens committed = len(path)), the path's last row, the K / V cache copy (source row,
  destination row; -1 when the path ends on a chain row), then the path's rows (padded with the last)."""
  a, last = len(path), path[-1]; src, dst = (last, a - 1) if last >= NC else (-1, -1)
  rows = list(path) + [last] * (M - len(path))
  return np.array([a, last, src, dst] + rows, np.int32)

"""The fused-sweep loops of gdn_fast_src as hand-scheduled TEC bundles (inline asm in the C kernel).

A sweep runs over the DK = 128 rows of a DK x 16 fp32 state block in LSRAM (a row = two float8 halves, 64 B), U rows a loop
iteration, and per row does a fixed list of vector ops (see SWEEPS). The k / q scalars of a row reach the lanes by `replic` from
float8 vectors of a token's rows (KQN: new tokens, [4-row group][token] (k rows 4g..4g+3 | q rows 4g..4g+3); KP: pending tokens,
[8-row group][slot] k rows 8g..8g+7).

`modulo_schedule` places the ops of one iteration in II bundles with two stages (stage 0: loads, replics and temporaries of the
NEXT iteration; stage 1: the accumulations and the stores of this one), so that the loop needs a prologue (stage 0 of iteration
0) and NO epilogue: the last trip's stage-0 ops only load past the block (inside LSRAM) and write temporaries. Rules of the TEC
(measured on the board): two ALU ops a bundle (fma / mul / scalar add), two memory ops (vld / st / replic) with replic
and st in slot 2 only (one of them a bundle: the assembler refuses {st; replic}); a load pair must straddle bit 6 (the bank), a load beside a store must share
it; at most two stores in any three consecutive bundles (1.5 cycles a store); latencies vld 3, replic 2, fma / mul 4, add 1;
a loop body (+ loopend) of at most 32 bundles. Every reuse of a physical register comes strictly after the last read of its
previous value, so the schedule is right whether or not the pipeline interlocks (and on a functional simulator)."""

import random
LAT = {"v": 4, "ld": 3, "rep": 2, "s": 1, "st": 0}
TRIES = 40                                                  # randomised retries a (II, recurrence placement) before II + 1
ALU, MEM = ("v", "s"), ("ld", "st", "rep")


class Op:
  def __init__(self, kind, fmt, dst=None, srcs=(), base=None, off=0, bank=None, late=False, tie=False):
    self.kind, self.fmt, self.dst, self.srcs, self.base, self.off, self.bank = kind, fmt, dst, tuple(srcs), base, off, bank
    self.late, self.tie, self.sigma = late, tie, None       # late: stage 1 only (accumulations, stores); tie: dst = srcs[0]'s register


def sweep_ops(kind, U, M, MS):
  """The op list of one iteration (U rows) of sweep `kind`, and the base registers {name: stride bytes}.
  Values: '%x' = a C operand (accumulators %o0 %o1 %k0 %k1, constants %dl0 %dl1 %dn %dc); others temporaries.
  Bases: rb (the block's rows, loads and stores; U = 8: rl loads, rs stores), rk (KQN), rp / rq (KP)."""
  ops, bases = [], {}
  if U == 4: bases["rb"] = 256
  else: bases["rl"] = bases["rs"] = 512
  lb, sb = ("rb", "rb") if U == 4 else ("rl", "rs")
  # the k / q vectors: (base, offset) of the vector holding row r's lane, and the lane
  if kind in ("N1", "N", "NL", "K0n", "PB", "NB", "N1B", "LF", "LFD"):
    bases["rk"] = M * 32 * (U // 4)
  if kind in ("K0p", "P", "PB"):
    bases["rp"] = MS * 32
  def kq(t, r):                                             # new token t (relative to rk), row r: (base, off, lane) of k; q = lane + 4
    return "rk", (r // 4) * M * 32 + t * 32, r % 4
  def kp(u, r):                                             # pending slot u (relative to rp), row r (U = 8)
    return "rp", u * 32, r
  vec = {}                                                  # (base, off) -> vector value name, loaded once
  def vload(base, off, r, use):                            # a vector a (4-row half, use): short live ranges
    if (base, off, r // 4, use) not in vec:
      n = f"V{base}{off}_{r // 4}{use}"; vec[(base, off, r // 4, use)] = n
      ops.append(Op("ld", "ld {d}, [{b}+{o}]", n, (), base=base, off=off, bank=None))
    return vec[(base, off, r // 4, use)]
  def rep(src, lane, name):
    ops.append(Op("rep", "replic {d}.w, {s0}.w, %d" % lane, name, (src,)))
    return name
  srcs_k = {"N1": ("kq", 0, "kq", 1), "N": ("kq", 0, "kq", 1), "NL": ("kq", 0, None, None), "K0n": (None, None, "kq", 0),
            "K0p": (None, None, "kp", 0), "P": ("kp", 0, "kp", 1), "PB": ("kp", 0, "kq", 0),
            "NB": ("kq", 0, "kq", 1), "N1B": ("kq", 0, "kq", 1), "LF": ("kq", 0, None, None), "LFD": ("kq", 0, None, None)}[kind]
  for r in range(U):
    b0, b1 = f"b0_{r}", f"b1_{r}"
    ops.append(Op("ld", "ld {d}, [{b}+{o}]", b0, (), base=lb, off=64 * r, bank=r & 1))
    ops.append(Op("ld", "ld {d}, [{b}+{o}]", b1, (), base=lb, off=64 * r + 32, bank=r & 1))
    cur, ci, nxt, ni = srcs_k
    kb = qb = nb = None
    if cur:
      base, off, lane = (kq if cur == "kq" else kp)(ci, r)
      kb = rep(vload(base, off, r, "k"), lane, f"kb_{r}")
      if kind in ("N1", "N", "NL", "NB", "N1B", "LF", "LFD"): qb = rep(vload(base, off, r, "q"), lane + 4, f"qb_{r}")
    if nxt:
      base, off, lane = (kq if nxt == "kq" else kp)(ni, r)
      nb = rep(vload(base, off, r, "n"), lane, f"nb_{r}")
    def fma(acc, a, b, late=False, tie=False, name=None):
      ops.append(Op("v", "fma {d}.fp32, {s1}.fp32, {s2}.fp32, p7.w", name or acc, (acc, a, b), late=late, tie=tie))
    def mul(d, a, b, tie=False):
      ops.append(Op("v", "mul {d}.fp32, {s0}.fp32, {s1}.fp32, p7.w", d, (a, b), tie=tie))
    def st(v, half, early=False):
      ops.append(Op("st", "st {s0}, [{b}+{o}]", None, (v,), base=sb, off=64 * r + 32 * half, bank=r & 1, late="early" if early else True))
    s0, s1 = f"s0_{r}", f"s1_{r}"
    if kind in ("K0n", "K0p"):                             # d = S dn -> stored; kv += nb d
      mul(f"d0_{r}", b0, "%dn"); mul(f"d1_{r}", b1, "%dn"); st(f"d0_{r}", 0); st(f"d1_{r}", 1)
      fma("%k0", nb, f"d0_{r}", late=True); fma("%k1", nb, f"d1_{r}", late=True)
      continue
    if kind in ("LF", "LFD"):                              # a leaf (read only): kq += kl S, qq += ql S; LFD: d = S dn -> stored
      fma("%k0", kb, b0, late=True); fma("%k1", kb, b1, late=True); fma("%o0", qb, b0, late=True); fma("%o1", qb, b1, late=True)
      if kind == "LFD": mul(f"d0_{r}", b0, "%dn"); mul(f"d1_{r}", b1, "%dn"); st(f"d0_{r}", 0); st(f"d1_{r}", 1)
      continue
    if kind in ("N1", "N1B"):                               # S stored undecayed: S *= dc (in place) first
      mul(f"c0_{r}", b0, "%dc", tie=True); mul(f"c1_{r}", b1, "%dc", tie=True); b0, b1 = f"c0_{r}", f"c1_{r}"
    fma(b0, kb, "%dl0", tie=True, name=s0); fma(b1, kb, "%dl1", tie=True, name=s1)       # s = S + k dl (in place)
    if qb is not None: fma("%o0", qb, s0, late=True); fma("%o1", qb, s1, late=True)
    if kind == "NL": continue
    if kind in ("PB", "NB", "N1B"): st(s0, 0, True); st(s1, 1, True)   # the committed state (PB) / a leaves' parent's state (NB, N1B)
    mul(f"d0_{r}", s0, "%dn"); mul(f"d1_{r}", s1, "%dn")
    if kind not in ("PB", "NB", "N1B"): st(f"d0_{r}", 0); st(f"d1_{r}", 1)
    fma("%k0", nb, f"d0_{r}", late=True); fma("%k1", nb, f"d1_{r}", late=True)
  used = {o.base for o in ops if o.base}
  return ops, {b: v for b, v in bases.items() if b in used}


def _deps(ops):
  """RAW edges (src op, dst op, delay, distance) by program order; accumulators ('%' values written by ops) also loop-carried."""
  last, first_w, edges = {}, {}, []
  for j, o in enumerate(ops):
    for s in o.srcs:
      if s in last: edges.append((last[s], j, LAT[ops[last[s]].kind], 0))
    if o.dst is not None:
      if o.dst.startswith("%"): first_w.setdefault(o.dst, j)
      last[o.dst] = j
  for v, j0 in first_w.items():                             # the accumulator chains across the back edge
    edges.append((last[v], j0, LAT["v"], 1))
  return edges


def modulo_schedule(ops, bases, II, pool, A=None, rnd=None):
  """-> (bundles of the kernel [II][texts], prologue bundles, {base: bias}) or None. The accumulator recurrences go first, row r's
  pair (both halves, one bundle) at A[acc] + 4 r in stage 1; then every other op as late as its consumers allow (stores: as early
  as their value allows, in stage 1)."""
  edges = _deps(ops); preds = {j: [] for j in range(len(ops))}; succs = {j: [] for j in range(len(ops))}
  for a, b, d, dist in edges: preds[b].append((a, d, dist)); succs[a].append((b, d, dist))
  for o in ops: o.sigma = None
  res = [dict(alu=0, mem=0, rep=0, st=0, banks=[]) for _ in range(II)]
  def fits(o, k):
    r = res[k]
    if o.kind in ALU: return r["alu"] < 2
    if r["mem"] >= 2: return False
    if o.kind == "rep": return r["rep"] == 0 and r["st"] == 0   # replic and st both want slot 2 (the assembler refuses the pair)
    if o.kind == "st":
      if r["st"] or r["rep"] or any(kind == "ld" and (bk is None or bk != o.bank) for kind, bk in r["banks"]): return False
      if sum(res[(k + dk) % II]["st"] for dk in (-2, -1)) >= 2 or sum(res[(k + dk) % II]["st"] for dk in (-1, 1)) >= 2 or \
         sum(res[(k + dk) % II]["st"] for dk in (1, 2)) >= 2: return False
      return True
    # a load: beside another load only across banks, beside a store only on its bank; an unknown bank alone (a replic is no access)
    for kind, bk in r["banks"]:
      if o.bank is None or bk is None: return False
      if kind == "ld" and bk == o.bank: return False
      if kind == "st" and bk != o.bank: return False
    return True
  def take(o, k):
    r = res[k]
    if o.kind in ALU: r["alu"] += 1
    else:
      r["mem"] += 1
      if o.kind == "rep": r["rep"] += 1
      else: r["banks"].append((o.kind, o.bank))
      if o.kind == "st": r["st"] += 1
  # 1. the recurrences: accumulator ops grouped by row pair (o0 / o1 of row r together)
  chains = {}
  for j, o in enumerate(ops):
    if o.dst is not None and o.dst.startswith("%"): chains.setdefault(o.dst[:-1], []).append(j)
  names = sorted(chains)
  for nm in names:
    js = chains[nm]; half = len(js) // 2; pairs = [(js[2 * r], js[2 * r + 1]) for r in range(half)]
    start = A[nm]
    for r, (j0, j1) in enumerate(pairs):
      s_ = start + 4 * r
      if not (II <= s_ < 2 * II) or res[s_ % II]["alu"] > 0: return None
      for j in (j0, j1): ops[j].sigma = s_; take(ops[j], s_ % II)
  # 2. the stores, next to the recurrence that reads the same row (row r's k op), then the rest as late as their consumers allow
  kop = [ops[j].sigma for j in chains.get("%k", [])[0::2]]  # row r's kv op (it reads the stored d; PB: s, 4 cycles earlier)
  for j, o in enumerate(ops):
    if o.kind != "st": continue
    anchor = kop[o.off // 64] - (4 if o.late == "early" else 0) + (rnd.randrange(3) if rnd else 0)
    for s_ in list(range(anchor, 2 * II)) + list(range(anchor - 1, II - 1, -1)):
      if s_ >= II and fits(o, s_ % II): o.sigma = s_; take(o, s_ % II); break
    else: return None
  for j in range(len(ops) - 1, -1, -1):
    o = ops[j]
    if o.sigma is not None: continue
    lim = 2 * II - 1
    for b, d, dist in succs[j]:
      if dist == 0:
        if ops[b].sigma is None: return None
        lim = min(lim, ops[b].sigma - d)
    if rnd and rnd.random() < 0.3: lim -= rnd.randrange(1, 4)
    for s_ in range(lim, max(-1, lim - II), -1):
      if fits(o, s_ % II): o.sigma = s_; take(o, s_ % II); break
    else: return None
  for a, b, d, dist in edges:                               # every dependence holds (the chains' back edge included)
    if ops[b].sigma + II * dist < ops[a].sigma + d: return None
  # the base increments: a free ALU slot each, such that the base's offsets fit 0..511 after a bias
  incs, bias = {}, {}
  for b, stride in bases.items():
    users = [o for o in ops if o.base == b]
    best = None
    for si in range(2 * II):
      if res[si % II]["alu"] >= 2: continue
      if any(o.sigma - si <= -II for o in users): continue   # (the prologue's iteration 0 must see the same base)
      offs = [o.off - stride * -(-(o.sigma - si) // II) for o in users]
      span = max(offs) - min(offs)
      if span <= 511 and (best is None or span < best[0]): best = (span, si, -min(offs))
    if best is None: return None
    _, si, bs = best; incs[b] = si; bias[b] = bs; res[si % II]["alu"] += 1
  # registers: temporaries' live ranges (a tied chain = one range) on the circle of II cycles
  rng = {}                                                  # root value -> [start, end]
  root = {}
  for o in ops:
    if o.dst is None or o.dst.startswith("%"): continue
    r_ = root[o.srcs[0]] if o.tie else o.dst; root[o.dst] = r_
    lo, hi = rng.get(r_, [o.sigma, o.sigma]); rng[r_] = [min(lo, o.sigma), max(hi, o.sigma)]
  for o in ops:
    for s in o.srcs:
      if s in root: rng[root[s]][1] = max(rng[root[s]][1], o.sigma)
  reg, occ = {}, {}
  for v, (lo, hi) in sorted(rng.items(), key=lambda kv: -(kv[1][1] - kv[1][0])):
    if hi - lo >= II: return None
    cyc = {c % II for c in range(lo, hi + 1)}
    for p in pool:
      if not (occ.get(p, set()) & cyc): reg[v] = p; occ.setdefault(p, set()).update(cyc); break
    else: return None
  def R(v):
    if v.startswith("%"): return "%[" + v[1:] + "]"
    return reg[root[v]]
  def text(o):
    kw = {"d": R(o.dst) if o.dst else ""}
    for i, s in enumerate(o.srcs): kw[f"s{i}"] = R(s)
    if o.base is not None:
      cnt = -(-(o.sigma - incs[o.base]) // II); kw["b"] = "%[" + o.base + "]"; kw["o"] = o.off - bases[o.base] * cnt + bias[o.base]
      assert 0 <= kw["o"] <= 511
    return o.fmt.format(**kw)
  kern = [[] for _ in range(II)]; pro = []                  # pro: (sigma, order, text, kind, bank, reads, writes) of iteration 0's stage 0
  for k, o in enumerate(ops):
    kern[o.sigma % II].append(text(o))
    if o.sigma < II:
      rd = {R(x) for x in o.srcs} | ({"%[" + o.base + "]"} if o.base else set()); wr = {R(o.dst)} if o.dst else set()
      pro.append((o.sigma, k, text(o), o.kind, o.bank, rd, wr))
  for b, si in incs.items():
    t = "add %%[%s], %%[%s], %d" % (b, b, bases[b]); kern[si % II].append(t)
    if si < II: pro.append((si, len(ops), t, "s", None, {"%[" + b + "]"}, {"%[" + b + "]"}))   # after the same cycle's loads (they read the old base)
  pro.sort(key=lambda x: (x[0], x[1]))
  # the prologue packed: the ops in schedule order into as few bundles as the slots allow, never two ops of a bundle sharing a
  # register unless both only read it (so neither the hardware's interlocks nor a sequential simulator see a reordering)
  packed = []
  for sg, k, t, kind, bank, rd, wr in pro:
    if packed:
      b = packed[-1]; alu = sum(x[0] in ALU for x in b["ops"]); mem = sum(x[0] in MEM for x in b["ops"])
      slot2 = sum(x[0] in ("rep", "st") for x in b["ops"])
      ok = (kind in ALU and alu < 2) or (kind in MEM and mem < 2 and not (kind in ("rep", "st") and slot2) and
                                          not (kind == "ld" and any(x[0] == "ld" and (x[1] is None or bank is None or x[1] == bank) for x in b["ops"])) and
                                          not (kind == "ld" and any(x[0] == "st" for x in b["ops"])) and not (kind == "st" and any(x[0] == "ld" for x in b["ops"])))
      ok = ok and not (wr & (b["rd"] | b["wr"])) and not (rd & b["wr"])
      if ok: b["ops"].append((kind, bank)); b["txt"].append(t); b["rd"] |= rd; b["wr"] |= wr; continue
    packed.append(dict(ops=[(kind, bank)], txt=[t], rd=set(rd), wr=set(wr)))
  pro = [b["txt"] for b in packed]
  return kern, pro, bias


def sweep_asm(kind, M, MS, pool=tuple(f"t{i}" for i in range(22)), N=None):
  """(asm text, {base: bias}, II, U) of sweep `kind` (K0p K0n P PB N1 N NL; the leaf tree: NB N1B LF LFD) for M new tokens and MS pending slots."""
  U = 4 if kind in ("N1", "N", "NB", "N1B") else 8
  ops0, bases = sweep_ops(kind, U, M, MS)
  nalu = sum(o.kind in ALU for o in ops0) + len(bases); nmem = sum(o.kind in MEM for o in ops0)
  nst = sum(o.kind == "st" for o in ops0); nrep = sum(o.kind == "rep" for o in ops0)
  II = max(-(-nalu // 2), -(-nmem // 2), nrep + nst, -(-3 * nst // 2), 4 * U)   # replic / st: slot 2 only; 4 U: an accumulator's fma a row (latency 4)
  out = None; rnd = random.Random(1234)
  accs = sorted({o.dst[:-1] for o in ops0 if o.dst and o.dst.startswith("%")})
  while II <= 32 and out is None:
    combos = [(a0, a1) for a0 in range(II, 2 * II) for a1 in ([a for a in range(II, 2 * II) if a != a0] if len(accs) == 2 else [None])]
    for tries in range(TRIES):
      for a0, a1 in combos:
        ops, bases = sweep_ops(kind, U, M, MS)
        out = modulo_schedule(ops, bases, II, pool, dict(zip(accs, (a0, a1))), rnd if tries else None)
        if out is not None: break
      if out is not None: break
    else: II += 1
  if out is None: raise AssertionError(f"sweep {kind}: no schedule within 32 bundles")
  kern, pro, bias = out
  B = lambda b: "{\n" + "".join(f" {x}\n" for x in (b or ["nop"])) + "}\n"
  n = (128 // U) if N is None else N
  asm = "".join(B(b) for b in pro) + f"loop {n - 1}\n" + "".join(B(b) for b in kern) + "loopend\n"
  return asm, bias, II, U


SIG = {"K0p": ("kp",), "K0n": ("kq",), "P": ("kp",), "PB": ("kp", "kq"), "N1": ("kq",), "N": ("kq",), "NL": ("kq",),
       "NB": ("kq",), "N1B": ("kq",), "LF": ("kq",), "LFD": ("kq",)}
CONST = {"K0p": ("dn",), "K0n": ("dn",), "P": ("dl0", "dl1", "dn"), "PB": ("dl0", "dl1", "dn"), "N1": ("dl0", "dl1", "dc", "dn"),
         "N": ("dl0", "dl1", "dn"), "NL": ("dl0", "dl1"), "NB": ("dl0", "dl1", "dn"), "N1B": ("dl0", "dl1", "dc", "dn"), "LF": (), "LFD": ("dn",)}


_cache = {}
def sweep_c(kind, M, MS, name=None):
  """A C function `sa_<kind>(B, k pointers..., constants..., float8* kv, float8* o)` running the sweep in asm. Pointers: B the
  block; kq = the KQN vector of the sweep's current token (K0n: the next token's; PB: new token 0), kp = the KP column of the
  current pending slot (K0p: slot 0)."""
  key = (kind, M, MS)
  if key not in _cache: _cache[key] = sweep_asm(kind, M, MS)
  asm, bias, II, U = _cache[key]
  ptrs = SIG[kind]; consts = CONST[kind]; accs = (["k0", "k1"] if kind != "NL" else []) + (["o0", "o1"] if kind in ("N1", "N", "NL", "NB", "N1B", "LF", "LFD") else [])
  args = ["__global float* restrict B"] + [f"__global float* restrict {p}" for p in ptrs] + [f"float8 {c}" for c in consts] + ["float8* kv", "float8* o"]
  regs = {"rb": "B", "rl": "B", "rs": "B", "rk": "kq", "rp": "kp"}
  lines = [f"static inline __attribute__((always_inline)) void {name or 'sa_' + kind}({', '.join(args)}) {{  /* II {II} a {U} rows */"]
  lines.append("  " + " ".join(f"float8 {a} = BC(0.0f);" for a in accs))
  lines.append("  " + " ".join(f"int {b} = (int){regs[b]} - {bias[b]};" for b in bias))
  body = "".join(f'"{l}\\n"\n    ' for l in asm.strip().split("\n"))
  outs = ", ".join([f'[{a}] "+t"({a})' for a in accs] + [f'[{b}] "+r"({b})' for b in bias])
  ins = ", ".join(f'[{c}] "t"({c})' for c in consts)
  clob = ", ".join(f'"t{i}"' for i in range(22)) + ', "memory"'
  lines.append(f"  __asm__ volatile (\n    {body}: {outs}\n    : {ins}\n    : {clob});")
  if "k0" in accs: lines.append("  kv[0] = k0; kv[1] = k1;")
  if "o0" in accs: lines.append("  o[0] = o0; o[1] = o1;")
  lines.append("}")
  return "\n".join(lines) + "\n"

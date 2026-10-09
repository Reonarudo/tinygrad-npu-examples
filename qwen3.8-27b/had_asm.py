"""had_fast's inner loops as hand-scheduled TEC bundles (inline asm), by a small modulo scheduler.

Machine rules (measured on the board; the same ones gdn_asm.py uses):
  * a bundle: two ALU slots (vector add / sub / mul / fma / cvt, scalar add), slot 2 (a store, an extract, or the first load),
    slot 3 (a load). So at most one of {st, ext} a bundle, at most two memory ops (loads) a bundle.
  * two loads in a bundle straddle bit 6 (the LSRAM banks); a load beside a store shares its bank; an unknown bank goes alone.
  * at most two stores in any three consecutive bundles (1.5 cycles a store).
  * latencies: vld 3, ext 2, fp add / mul / fma / cvt 4; a base register is usable the bundle after its increment.
  * the loop body at most 32 bundles (the loop buffer), `loop n-1` ... `loopend` with nothing beside loopend.
Pipelining: an iteration's ops are placed modulo II in S stages; every store in the last stage, so the kernel runs N trips after
a prologue (stages < S-1 of the first S-1 iterations) and NO epilogue: the last trips' early stages only load past the data
(inside LSRAM) and write temporaries. Every reuse of a physical register comes strictly after the last read of its previous value,
so the schedule is right whether or not the pipeline interlocks (and on a functional simulator)."""
import random

LAT = {"ld": 3, "x": 2, "v": 4, "st": 0, "m": 2}   # m: mov.pre (a memory-slot move: beside an ext / store, or a load, not two loads)
NOFMAEXT = False                                    # the fma-beside-ext rule (set by the measured-latency schedules)
ALU, POOL = ("v",), tuple(f"t{i}" for i in range(10, 32))


# measured result latencies (on the board): fp32 mul / fma 4; add / sub / max / min / abs / rint / scal2 / cvt 2;
# rcp / rsqrt / sqrt / ext* / replic / sel / mov.pre 2 (slot 2). And a pairing cost: fma beside an ext / sel costs 2 cycles.
MLAT = {"mul": 4, "fma": 4, "fnma": 4, "add": 2, "sub": 2, "max": 2, "min": 2, "abs": 2, "rint": 2, "scal2": 2, "cvt": 2, "rcp": 2}


class Op:
  def __init__(self, kind, fmt, dst=None, srcs=(), base=None, off=0, bank=None, tie=False, after=None, lat=None):
    self.kind, self.fmt, self.dst, self.srcs, self.base, self.off, self.bank = kind, fmt, dst, tuple(srcs), base, off, bank
    self.tie = tie                                   # dst lives in srcs[0]'s register (an accumulating fma)
    self.after = after                               # (value, delay): not before that value's producer + delay (a late load)
    self.lat = LAT[kind] if lat is None else lat
    mn = fmt.split()[0]
    self.slot = None                                  # a fixed modulo slot (the divide unit's ops: see ops_sw(divunit=True))
    self.fmaop = mn in ("fma", "fnma")                # never beside an ext / sel (2 cycles, measured)
    self.extop = mn in ("exte", "exto", "extl", "exth", "sel")
    self.sigma = None


def _fits(res, II, o, k):
  r = res[k % II]
  if NOFMAEXT and ((o.fmaop and r["ext"]) or (o.extop and r["fma"])): return False
  if o.kind in ALU: return r["alu"] < 2
  if o.kind == "m": return r["nld"] + r["nm"] + (1 if r["s2"] else 0) < 2 and not (r["nld"] and r["s2"])
  if o.kind in ("st", "x"):
    if r["s2"] or r["nld"] + r["nm"] >= 2: return False
    if o.kind == "st":
      if any(bk is None or o.bank is None or bk != o.bank for bk in r["ldb"]): return False
      if II >= 3:
        cnt = lambda ks: sum(res[(k + d) % II]["st"] for d in ks)
        if cnt((-2, -1)) >= 2 or cnt((-1, 1)) >= 2 or cnt((1, 2)) >= 2: return False
      elif any(res[x]["st"] for x in range(II)): return False
    return True
  # a load
  if r["nld"] + r["nm"] + (1 if r["s2"] else 0) >= 2: return False
  if r["nm"] and r["s2"]: return False
  if r["stb"] is not None and (o.bank is None or o.bank != r["stb"]): return False
  if r["s2"] and r["stb"] is None and r["nld"] >= 1: return False          # ext in slot 2: one load (slot 3)
  for bk in r["ldb"]:
    if bk is None or o.bank is None or bk == o.bank: return False
  return True


def _take(res, II, o, k):
  r = res[k % II]
  r["fma"] = r["fma"] or o.fmaop; r["ext"] = r["ext"] or o.extop
  if o.kind in ALU: r["alu"] += 1
  elif o.kind == "st": r["s2"] = True; r["st"] += 1; r["stb"] = o.bank
  elif o.kind == "x": r["s2"] = True
  elif o.kind == "m": r["nm"] += 1
  else: r["nld"] += 1; r["ldb"].append(o.bank)


def schedule(ops, bases, II, rnd=None, pool=POOL, acc_lat=4, nopro=False, blocked=()):
  """-> dict(kern=[II][texts], pro=[bundles], bias={base: b}, S) or None. `nopro`: no prologue -- the kernel runs N + S - 1 trips
  from the first, so its first S - 1 trips run the late stages of iterations -S+1..-1 (loads before the data, stores of garbage
  into the S - 1 strides before it: the caller leaves that gap), and the offsets count every trip's increment. `bases`: {name: stride in bytes a trip}. Values named
  '%x' are asm operands (accumulators read and written in place: a loop-carried recurrence of latency acc_lat)."""
  n = len(ops); last = {}; preds = [[] for _ in range(n)]
  for j, o in enumerate(ops):
    for s in o.srcs:
      if s in last: preds[j].append((last[s], ops[last[s]].lat))
    if o.after: preds[j].append((last[o.after[0]], o.after[1]))
    if o.dst is not None: last[o.dst] = j
  res = [dict(alu=0, s2=False, st=0, stb=None, nld=0, nm=0, ldb=[], fma=False, ext=False) for _ in range(II)]
  for k in blocked: res[k].update(alu=2, s2=True, nld=2, nm=2, st=0)      # stall cycles (the divide unit's wait): no bundle there
  for o in ops:                                                             # fixed-slot ops reserve their slot first
    if o.slot is not None: _take(res, II, o, o.slot)
  for o in ops: o.sigma = None
  for j, o in enumerate(ops):
    t0 = max([ops[p].sigma + d for p, d in preds[j]] + [0])
    if rnd is not None and o.kind != "st" and rnd.random() < 0.25: t0 += rnd.randrange(0, 3)
    for k in range(t0, t0 + II):
      if o.slot is not None:
        if k % II == o.slot: o.sigma = k; break
        continue
      if _fits(res, II, o, k): o.sigma = k; _take(res, II, o, k); break
    else: return None
  # accumulator recurrences ('%' written in place): the next trip's write after this one's latency
  wr = {}
  for o in ops:
    if o.dst and o.dst.startswith("%"): wr.setdefault(o.dst, []).append(o.sigma)
  for v, ss in wr.items():
    if max(ss) - min(ss) + acc_lat > II: return None
  S = max(o.sigma for o in ops) // II + 1
  for o in ops:                                                             # stores / accumulations moved to the last stage
    if (o.kind == "st" or (o.dst or "").startswith("%")) and o.sigma // II != S - 1: o.sigma += II * (S - 1 - o.sigma // II)
  if any((o.kind == "st" or (o.dst or "").startswith("%")) and o.sigma // II != S - 1 for o in ops):   # stores / accumulations only in
    return None                                                                                       # the last stage (no epilogue)
  # base increments: a free ALU slot each, all of the base's offsets in 0..511 after a bias
  incs, bias = {}, {}
  for b, stride in bases.items():
    users = [o for o in ops if o.base == b]; best = None
    for si in range(S * II):
      if res[si % II]["alu"] >= 2: continue
      if not nopro and any(o.sigma - si <= -II for o in users): continue
      if nopro and si >= II: break
      offs = [o.off - stride * (_cnt(o.sigma, si, II) if nopro else -(-(o.sigma - si) // II)) for o in users]
      span = max(offs) - min(offs)
      if span <= 511 and (best is None or span < best[0]): best = (span, si, -min(offs))
    if best is None: return None
    _, si, bs = best; incs[b] = si; bias[b] = bs; res[si % II]["alu"] += 1
  # registers: each temporary's live range [def, last use] on the circle of II cycles
  rng, root = {}, {}
  for o in ops:
    if o.dst and not o.dst.startswith("%"):
      r_ = root[o.srcs[0]] if o.tie else o.dst; root[o.dst] = r_
      lo, hi = rng.get(r_, [o.sigma, o.sigma]); rng[r_] = [min(lo, o.sigma), max(hi, o.sigma)]
  for o in ops:
    for s in o.srcs:
      if s in root: rng[root[s]][1] = max(rng[root[s]][1], o.sigma)
  reg, occ = {}, {}
  order = sorted(rng.items(), key=lambda kv: (-(kv[1][1] - kv[1][0]), kv[1][0]))
  for v, (lo, hi) in order:
    if hi - lo >= II: return None
    cyc = {c % II for c in range(lo, hi + 1)}
    cand = list(pool)
    if rnd is not None: rnd.shuffle(cand)
    for p in cand:
      if not (occ.get(p, set()) & cyc): reg[v] = p; occ.setdefault(p, set()).update(cyc); break
    else: return None
  R = lambda v: "%[" + v[1:] + "]" if v.startswith("%") else reg[root[v]]
  def text(o):
    kw = {"d": R(o.dst) if o.dst else ""}
    for i, s in enumerate(o.srcs): kw[f"s{i}"] = R(s)
    if o.base is not None:
      cnt = _cnt(o.sigma, incs[o.base], II) if nopro else -(-(o.sigma - incs[o.base]) // II); kw["b"] = "%[" + o.base + "]"; kw["o"] = o.off - bases[o.base] * cnt + bias[o.base]
      assert 0 <= kw["o"] <= 511, (o.fmt, kw["o"])
    return o.fmt.format(**kw)
  kern = [[] for _ in range(II)]
  BL = set(blocked)
  if nopro: S_pro = 1                        # no prologue trips
  pro = []                                   # (time, order, text, kind, bank, reads, writes) of the prologue (trips 0..S-2, partial)
  for k, o in enumerate(ops):
    kern[o.sigma % II].append(text(o))
    st_ = o.sigma // II
    for T in range(st_, (S - 1) if not nopro else 0):   # the prologue trip T runs iteration T - stage of every op whose stage <= T
      rd = {R(x) for x in o.srcs} | ({"%[" + o.base + "]"} if o.base else set()); wr_ = {R(o.dst)} if o.dst else set()
      pro.append((T * II + o.sigma % II, k, text(o), o.kind, o.bank, rd, wr_))
  for b, si in incs.items():
    t = "add %%[%s], %%[%s], %d" % (b, b, bases[b]); kern[si % II].append(t)
    for T in range(si // II, (S - 1) if not nopro else 0):
      pro.append((T * II + si % II, len(ops), t, "s", None, {"%[" + b + "]"}, {"%[" + b + "]"}))
  pro.sort(key=lambda x: (x[0], x[1]))
  packed = []
  for _, _, t, kind, bank, rd, wr_ in pro:   # in time order into as few bundles as the slots allow, no register shared by a writer
    if packed:
      b = packed[-1]; kinds = [x[0] for x in b["ops"]]
      alu = sum(x in ("v", "s") for x in kinds); nld = kinds.count("ld"); s2 = sum(x in ("st", "x") for x in kinds)
      if kind in ("v", "s"): ok = alu < 2
      elif kind in ("st", "x"): ok = s2 == 0 and nld <= 1 and not (kind == "st" and any(x[0] == "ld" and x[1] != bank for x in b["ops"]))
      else: ok = nld + s2 < 2 and not any(x[0] == "ld" and (x[1] is None or bank is None or x[1] == bank) for x in b["ops"]) and \
                 not any(x[0] == "st" and x[1] != bank for x in b["ops"])
      ok = ok and not (wr_ & (b["rd"] | b["wr"])) and not (rd & b["wr"])
      if ok: b["ops"].append((kind, bank)); b["txt"].append(t); b["rd"] |= rd; b["wr"] |= wr_; continue
    packed.append(dict(ops=[(kind, bank)], txt=[t], rd=set(rd), wr=set(wr_)))
  assert not any(kern[k] for k in BL), "an op in a stall slot"
  kern = [b for k, b in enumerate(kern) if k not in BL]
  return dict(kern=kern, pro=[b["txt"] for b in packed], bias=bias, S=S, II=II, nopro=nopro, cycles=II)


def _cnt(sigma, si, II):
  """no prologue: the increments a trip-T op of stage sigma // II has seen beyond its iteration's (i = T - stage) own count i:
  the base at trip T holds start + stride T (+1 if the increment sits earlier in the bundle order), i.e. i + stage (+1)."""
  return sigma // II + (1 if si % II < sigma % II else 0)


def best_schedule(mk, II0=1, tries=60, IImax=32, nopro=False, blocked=lambda II: (), seed=7, skip=0):
  """mk() -> (ops, bases); the smallest II with a schedule (deterministic first, then randomised retries). `blocked(II)`: modulo
  slots that are stall cycles, not bundles (the divide unit's wait); II then counts cycles and the body has II - len(blocked)."""
  II = II0; rnd = random.Random(seed)
  while II <= IImax:
    for t in range(tries):
      ops, bases = mk()
      out = schedule(ops, bases, II, rnd if t else None, nopro=nopro, blocked=blocked(II))
      if out is not None:
        if skip: skip -= 1; continue                   # the skip-th schedule found (candidates for an on-board pick)
        out["ops"] = ops; return out
    II += 1
  raise AssertionError("no schedule within 32 bundles")


def asm_text(sch, trips):
  """`trips` an int (loop N - 1) or a C expression (`loop %[ntr]`, the register holding trips - 1: see c_block)."""
  B = lambda b: "{\n" + "".join(f" {x}\n" for x in (b or ["nop"])) + "}\n"
  if isinstance(trips, str): lp = "loop %[ntr]\n"
  else:
    if sch.get("nopro"): trips += sch["S"] - 1
    lp = f"loop {trips - 1}\n"
  return "".join(B(b) for b in sch["pro"]) + lp + "".join(B(b) for b in sch["kern"]) + "loopend\n"


def c_block(sch, trips, bases_c, consts=(), accs=(), clob_pool=POOL, regs=(), derive=None):
  """The asm statement: bases_c {base: C pointer expression}, consts [(name, C float8 expr)] as "t" inputs, accs [names] as "+t",
  regs [(name, C int expr)] as "r" inputs. `trips` a C expression: the trip count in a register (`loop rN`, N =
  trips - 1; the nopro schedule's S - 1 extra trips added here), so one copy of the text runs any count -- 1 for a warm-up."""
  asm = asm_text(sch, trips)
  derive = dict(derive or {})               # {base: (from base, +delta)}: computed in the asm, no C constant per base (each costs
  if derive:                                # the compiler a mov / movh pair); delta an int (the expressions' difference) or a regs name
    pk, done = [], set(n for n in bases_c if n not in derive)
    todo = [n for n in bases_c if n in derive]
    while todo:
      b_ = [n for n in todo if derive[n][0] in done][:2]; assert b_, derive
      txt, txt2 = [], []
      for n in b_:
        f, d = derive[n]; dd = -(sch["bias"][n] - sch["bias"][f])
        if isinstance(d, str):
          txt.append(f"add %[{n}], %[{f}], %[{d}]")
          if dd: txt2.append(f"{'add' if dd > 0 else 'sub'} %[{n}], %[{n}], {abs(dd)}")
        else:
          dd += d; assert abs(dd) <= 1023, dd
          txt.append(f"{'add' if dd >= 0 else 'sub'} %[{n}], %[{f}], {abs(dd)}")
      for t_ in (txt, txt2):
        if t_: pk.append("{\n" + "".join(f" {t}\n" for t in t_) + "}\n")
      done.update(b_); todo = [n for n in todo if n not in b_]
    asm = "".join(pk) + asm
  body = "".join(f'"{l}\\n"\n      ' for l in asm.strip().split("\n"))
  lines = ["{"]
  lines.append("  " + " ".join(f"int {b};" if b in derive else f"int {b} = (int)({e}) - {sch['bias'][b]};" for b, e in bases_c.items()))
  lines.append("  " + " ".join(f"float8 {n} = {e};" for n, e in consts))
  if isinstance(trips, str): regs = list(regs) + [("ntr", f"({trips}) + {sch['S'] - 1 if sch.get('nopro') else 0} - 1")]
  lines.append("  " + " ".join(f"int {n}_ = {e};" for n, e in regs))
  outs = ", ".join([f'[{a}] "+t"({a})' for a in accs] + [f'[{b}] "{"=&r" if b in derive else "+r"}"({b})' for b in bases_c])
  ins = ", ".join([f'[{n}] "t"({n})' for n, _ in consts] + [f'[{n}] "r"({n}_)' for n, _ in regs])
  clob = ", ".join(f'"{r}"' for r in clob_pool) + ', "memory"'
  lines.append(f"  __asm__ volatile (\n      {body}: {outs}\n      : {ins}\n      : {clob});")
  lines.append("}")
  return "\n".join(lines)


# ---- the loop bodies ----
def ops_passA(mode, U=2):
  """U 16-value pairs a trip (64 B each, bank = pair & 1): load (and scale), stages 1, 2, 4, 8 (h16), store in place.
  bases: rl (loads of L), rs (stores into L), rw (W, rms / plain). Constants: %iv (rms)."""
  ops = []; bases = {"rl": 64 * U, "rs": 64 * U}
  if mode != "swiglu": bases["rw"] = 64 * U
  for p in range(U):
    o = 64 * p; bk = p & 1; v, w = f"x{p}a", f"x{p}b"
    ops.append(Op("ld", "ld {d}, [{b}+{o}]", v, (), "rl", o, bk)); ops.append(Op("ld", "ld {d}, [{b}+{o}]", w, (), "rl", o + 32, bk))
    if mode != "swiglu":
      ops.append(Op("ld", "ld {d}, [{b}+{o}]", f"w{p}a", (), "rw", o, bk)); ops.append(Op("ld", "ld {d}, [{b}+{o}]", f"w{p}b", (), "rw", o + 32, bk))
      if mode == "rms":
        ops.append(Op("v", "mul {d}.fp32, {s0}.fp32, {s1}.fp32, p7.w", f"m{p}a", (v, "%ivv"))); ops.append(Op("v", "mul {d}.fp32, {s0}.fp32, {s1}.fp32, p7.w", f"m{p}b", (w, "%ivv")))
        v, w = f"m{p}a", f"m{p}b"
      ops.append(Op("v", "mul {d}.fp32, {s0}.fp32, {s1}.fp32, p7.w", f"y{p}a", (v, f"w{p}a"))); ops.append(Op("v", "mul {d}.fp32, {s0}.fp32, {s1}.fp32, p7.w", f"y{p}b", (w, f"w{p}b")))
      v, w = f"y{p}a", f"y{p}b"
    for rd in range(4):
      e, od = f"e{p}_{rd}", f"o{p}_{rd}"
      ops.append(Op("x", "exte {d}.w, {s0}.w, {s1}.w", e, (v, w))); ops.append(Op("x", "exto {d}.w, {s0}.w, {s1}.w", od, (v, w)))
      nv, nw = f"s{p}_{rd}", f"d{p}_{rd}"
      ops.append(Op("v", "add {d}.fp32, {s0}.fp32, {s1}.fp32, p7.w", nv, (e, od))); ops.append(Op("v", "sub {d}.fp32, {s0}.fp32, {s1}.fp32, p7.w", nw, (e, od)))
      v, w = nv, nw
    ops.append(Op("st", "st {s0}, [{b}+{o}]", None, (v,), "rs", o, bk)); ops.append(Op("st", "st {s0}, [{b}+{o}]", None, (w,), "rs", o + 32, bk))
  return ops, bases


def ops_radix8(kind):
  """Pass B ("B": stages 16, 32, 64; one group a trip: vectors at 64 j, a trip 512 B; the caller runs it twice, from L and L + 32 B) or pass C ("C": stages
  128, 256, 512; one group a trip: vectors at 512 j, a trip 32 B; base c_j per j, shared by the loads and the stores).
  Stages on j's bits 0, 1, 2 in turn (a + b at the lower j, a - b at the upper)."""
  ops = []
  if kind == "B": bases = {"bl": 512, "bs": 512}; groups = 1
  else: bases = {f"c{j}": 32 for j in range(8)}; groups = 1
  for q in range(groups):
    a = {}; offs = {}
    for j in range(8):
      if kind == "B": off, bl, bs = 32 * q + 64 * j, "bl", "bs"
      else: off, bl, bs = 0, f"c{j}", f"c{j}"
      offs[j] = (off, bs); bk = ((32 * q + 64 * j) >> 6) & 1 if kind == "B" else 0
      a[j] = f"a{q}_{j}"; ops.append(Op("ld", "ld {d}, [{b}+{o}]", a[j], (), bl, off, bk))
    for dd in (1, 2, 4):
      for j in range(8):
        if j & dd: continue
        s_, d_ = f"a{q}_{j}_{dd}", f"a{q}_{j + dd}_{dd}"
        ops.append(Op("v", "add {d}.fp32, {s0}.fp32, {s1}.fp32, p7.w", s_, (a[j], a[j + dd]))); ops.append(Op("v", "sub {d}.fp32, {s0}.fp32, {s1}.fp32, p7.w", d_, (a[j], a[j + dd])))
        a[j], a[j + dd] = s_, d_
    for j in range(8):
      off, bs = offs[j]; bk = ((32 * q + 64 * j) >> 6) & 1 if kind == "B" else 0
      ops.append(Op("st", "st {s0}, [{b}+{o}]", None, (a[j],), bs, off, bk))
  return ops, bases


def ops_conv():
  """4 pieces a trip (16 columns of the 4 rows: r0..r3 at +0, q0..q3 at +32; rows 4 KiB apart): extl / exth row pairs, cvt.d, 4
  stores of 32 B (O advances 128 B a trip). bases: c0..c3 (row loads), o (stores)."""
  ops = []; bases = {"c0": 64, "c1": 64, "c2": 64, "c3": 64, "o": 128}
  for h in range(2):
    rr = [f"r{h}_{r}" for r in range(4)]
    for r in range(4): ops.append(Op("ld", "ld {d}, [{b}+{o}]", rr[r], (), f"c{r}", 32 * h, 0))
    for sel, nm in (("extl", "L"), ("exth", "H")):
      x01, x23 = f"{nm}{h}01", f"{nm}{h}23"
      ops.append(Op("x", sel + " {d}.w, {s0}.w, {s1}.w", x01, (rr[0], rr[1]))); ops.append(Op("x", sel + " {d}.w, {s0}.w, {s1}.w", x23, (rr[2], rr[3])))
      cv = f"cv{nm}{h}"; ops.append(Op("v", "cvt.d {d}.fp16, {s0}.fp32, {s1}.fp32", cv, (x01, x23)))
      ops.append(Op("st", "st {s0}, [{b}+{o}]", None, (cv,), "o", 64 * h + (0 if nm == "L" else 32), None))
  return ops, bases


def ops_p1(R):
  """phase 1, one float8 column of the R rows a trip (bases b0..b{R-1}, 32 B a trip; the rows' LSRAM pitch is CH * 4 + 64 so
  that rows r and r + 1 sit in different banks and their loads pair): acc[r] += v * v."""
  ops = []; bases = {f"b{r}": 32 for r in range(R)}
  for r in range(R):
    ops.append(Op("ld", "ld {d}, [{b}+{o}]", f"v{r}", (), f"b{r}", 0, r & 1))
    ops.append(Op("v", "fma {d}.fp32, {s0}.fp32, {s1}.fp32, p7.w", f"%acc{r}", (f"v{r}", f"v{r}")))
  return ops, bases


SWC = ("cm", "cmn", "cmx", "c0", "c1", "c2", "c3", "c4")   # -1.4427, -126, 126, 9.618e-3, 5.55e-2, 0.2402, 0.6931, 1.0


DIVU = 21                                           # the vector divide unit (rcp / div / sqrt): 21 cycles a vector op, not pipelined


def ops_sw(measured=False, divunit=False):
  """Two sw a trip: one row pair h of a column-quad pair (the caller runs the loop twice, h = 0, 1: g / u + 32 h, rows 2h, 2h + 1).
  a = col quad cq (+0), b = cq + 1 (+64 B); row 2h = EXTL(a, b) (base l0), row 2h + 1 = EXTH(a, b) (base l1); 128 B / 32 B a trip."""
  ops = []; bases = {"g": 128, "u": 128, "l0": 32, "l1": 32}
  V = lambda f, d, *s_, tie=False: ops.append(Op("v", f, d, s_, tie=tie, lat=MLAT.get(f.split()[0]) if measured else None))
  out = {}
  for nm, off in (("a", 0), ("b", 64)):
    _sw_ops(ops, V, nm, off, measured or divunit); out[nm] = f"y{nm}"
  if divunit:   # the two rcp at cycles 0 and DIVU (the second waits for the unit: the stall slots 1..DIVU-1 hold no bundle), result latency DIVU
    rc = [o for o in ops if o.fmt.startswith("rcp")]
    for o, sl in zip(rc, (0, DIVU)): o.slot = sl; o.lat = DIVU
  for r, sel in enumerate(("extl", "exth")):
    ops.append(Op("x", sel + " {d}.w, {s0}.w, {s1}.w", f"o{r}", (out["a"], out["b"])))
    ops.append(Op("st", "st {s0}, [{b}+{o}]", None, (f"o{r}",), f"l{r}", 0, None))
  return ops, bases


def _sw_ops(ops, V, nm, off, measured=False):
  u, q = f"u{nm}", f"q{nm}"; bk = (off >> 6) & 1
  ops.append(Op("ld", "ld {d}, [{b}+{o}]", u, (), "g", off, bk))
  V("mul {d}.fp32, {s0}.fp32, {s1}.fp32, p7.w", f"t1{nm}", u, "%cm")
  V("max {d}.fp32, {s0}.fp32, {s1}.fp32, p7.w", f"t2{nm}", f"t1{nm}", "%cmn")
  V("min {d}.fp32, {s0}.fp32, {s1}.fp32, p7.w", f"t3{nm}", f"t2{nm}", "%cmx")
  V("rint {d}.fp32, {s0}.fp32, p7.w", f"k{nm}", f"t3{nm}")
  V("sub {d}.fp32, {s0}.fp32, {s1}.fp32, p7.w", f"f{nm}", f"t3{nm}", f"k{nm}")
  prev = None
  for i, cst in enumerate(("%c1", "%c2", "%c3", "%c4")):
    p0 = f"p{i}i{nm}"; ops.append(Op("m", "mov.pre {d}, {s0}", p0, (cst,), after=((f"f{nm}" if i == 0 else prev), 1)))
    a, b = (f"f{nm}", "%c0") if i == 0 else (prev, f"f{nm}")
    V("fma {d}.fp32, {s1}.fp32, {s2}.fp32, p7.w", f"p{i}{nm}", p0, a, b, tie=True); prev = f"p{i}{nm}"
  ops.append(Op("v", "cvt {d}.w, {s0}.fp32", f"ki{nm}", (f"k{nm}",), after=(f"p2{nm}", 0), lat=2 if measured else None))
  V("scal2 {d}.fp32, {s0}.fp32, {s1}.w, p7.w", f"e{nm}", prev, f"ki{nm}")
  V("add {d}.fp32, {s0}.fp32, {s1}.fp32, p7.w", f"s{nm}", f"e{nm}", "%c4")
  ops.append(Op("x", "rcp {d}.fp32, {s0}.fp32, p7.w", f"r{nm}", (f"s{nm}",), lat=2 if measured else 4))   # rcp issues in slot 2 (refused beside a store / ext / rcp)
  ops.append(Op("ld", "ld {d}, [{b}+{o}]", u + "2", (), "g", off, bk, after=(f"s{nm}", 1)))
  ops.append(Op("ld", "ld {d}, [{b}+{o}]", q, (), "u", off, bk, after=(f"s{nm}", 1)))
  V("mul {d}.fp32, {s0}.fp32, {s1}.fp32, p7.w", f"m{nm}", u + "2", f"r{nm}")
  V("mul {d}.fp32, {s0}.fp32, {s1}.fp32, p7.w", f"y{nm}", q, f"m{nm}")

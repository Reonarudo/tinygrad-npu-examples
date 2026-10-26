#!/usr/bin/env python3
"""Device-memory accounting of an `ideogram4_fp_1024.py` run: every REQ_BUF / free of the process, labelled by its owner
(a runner class, or the pipeline function that allocated; `graph:` when a graph's clone did), the peak of the live total
(page-rounded: what the KMD's 3 GB ASID-0 IOVA window holds) and the breakdown at the peak -- or at the failure.

    python3 rows_probe.py --report out.json -- <ideogram4_fp_1024.py arguments>
"""
import json, os, sys, traceback, collections, runpy

here = os.path.dirname(os.path.abspath(__file__))
argv = sys.argv[1:]; i = argv.index("--"); rep = argv[argv.index("--report") + 1]; args = argv[i + 1:]
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(here, "..", "tinygrad")))); sys.path.insert(0, here)
from tinygrad.runtime.support.zhouyi import dev as D   # noqa: E402

live, st, addr = {}, dict(total=0, peak=0, n=0, peak_by={}), {}
OWN = ("Runner", "Launch", "TextGM", "Sampler", "IOProj", "ModNPU", "TembNPU", "Graph")


def label():
    f, graph, who = sys._getframe(2), False, None
    while f is not None:
        fn = f.f_code.co_filename
        if "graph/zhouyi" in fn: graph = True
        s = f.f_locals.get("self")
        if who is None and s is not None and not fn.endswith(("support/zhouyi/dev.py", "support/zhouyi/launch.py")):
            c = type(s).__name__
            if any(o in c for o in OWN) or "Program" in c: who = c
        if fn.startswith(here) and "rows_probe" not in fn and f.f_code.co_name not in ("<module>", "<lambda>", "<dictcomp>", "<listcomp>", "<genexpr>"):
            name = "%s:%s" % (os.path.basename(fn)[:-3].replace("ideogram4_", ""), f.f_code.co_name)
            if who is None: who = name
            elif who.count("<") < 1 and not who.endswith(name): who = who + "<" + name          # its caller in our code
        f = f.f_back
    return ("graph:" if graph else "") + (who or "?")


def by_label():
    d = collections.Counter()
    for b, (n, l) in live.items(): d[l] += n
    return {k: v for k, v in d.most_common()}


req0, free0 = D.RawDevice.req_buf, D.RawDevice.free_buf
def req(self, nbytes, *a, **k):
    b = req0(self, nbytes, *a, **k)
    n = (b.nbytes + self.LOAD_OVERHANG + 4095) // 4096 * 4096
    live[id(b)] = (n, label()); st["total"] += n; st["n"] += 1; addr[id(b)] = (self.dev_addr(b), n)
    if st["total"] > st["peak"]:
        st["peak"] = st["total"]; st["peak_n"] = len(live)
        if st["total"] - st.get("snap_at", 0) > 8 << 20 or st["total"] > (2900 << 20):   # the breakdown at the peak (re-taken every 8 MiB)
            st["snap_at"] = st["total"]; st["peak_by"] = by_label()
    return b
def free(self, b, *a, **k):
    e = live.pop(id(b), None); addr.pop(id(b), None)
    if e: st["total"] -= e[0]
    return free0(self, b, *a, **k)
D.RawDevice.req_buf, D.RawDevice.free_buf = req, free

ok, err = True, None
sys.argv = [os.path.join(here, "ideogram4_fp_1024.py")] + args
try: runpy.run_path(sys.argv[0], run_name="__main__")
except BaseException as e:
    ok, err = False, "".join(traceback.format_exception_only(type(e), e)).strip()
out = dict(ok=ok, err=err, peak_MiB=round(st["peak"] / 2**20, 1), live_at_end_MiB=round(st["total"] / 2**20, 1), buffers=st["n"],
           live_by_MiB={k: round(v / 2**20, 1) for k, v in by_label().items()},
           peak_by_MiB={k: round(v / 2**20, 1) for k, v in st["peak_by"].items()},
           big_buffers=sorted([(hex(a_), round(n / 2**20, 1)) for a_, n in addr.values() if n >= 16 << 20]))
json.dump(out, open(os.path.expanduser(rep), "w"), indent=1)
print("ROWS_PROBE %s peak %.1f MiB (at the peak: %s); %s live at end/failure" % ("OK" if ok else "FAIL " + str(err)[:80], out["peak_MiB"],
      ", ".join("%s %.0f" % kv for kv in list(out["peak_by_MiB"].items())[:12]), out["live_at_end_MiB"]), flush=True)

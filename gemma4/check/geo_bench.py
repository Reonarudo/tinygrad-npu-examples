#!/usr/bin/env python3
"""The wall time of a pass (the layers + the head, as Model.forward runs them) at each row geometry -- the verify cost table of the
speculative shapes -- and of a drafter step. After a prefill of the
prompt; each geometry --reps passes at the prompt's end (the caches' rows there are rewritten each pass: no state is changed
that a later pass reads).      python3 check/geo_bench.py [--cache ...] [--geos 1,2,3,4,5,6,7,8,12] [--reps 16] [--draft] ID ..."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gemma4_generate as GG                                              # noqa: E402
import gemma4_ref as GR                                                  # noqa: E402

def main():
  cpus = GG.pin_cpus(); GG.hold_cpu_latency()
  ap = argparse.ArgumentParser(); ap.add_argument("--cache", default=GG.DEFAULT_CACHE); ap.add_argument("--geos", default="1,2,3,4,5,6,7,8")
  ap.add_argument("--reps", type=int, default=16); ap.add_argument("--draft", action="store_true"); ap.add_argument("--json"); ap.add_argument("ids", nargs="*", type=int)
  a = ap.parse_args(); meta = json.load(open(os.path.join(a.cache, "gemma4.json"))); ids = a.ids or GG.GN.PROMPT; geos = [int(g) for g in a.geos.split(",")]
  T = GR.Model(meta["gguf"], cache_layers=False); m = GG.Model(T, a.cache, 1024); m.warmup(sorted(set(geos))); res = {}
  first = m.prefill(ids, max(geos)); P = len(ids)
  for g in geos:
    ts = []
    for _ in range(a.reps): t0 = time.perf_counter(); m.forward([first] * g, P); ts.append(time.perf_counter() - t0)
    res[g] = float(np.median(ts)) * 1e3; print(f"geometry {g:2d}: {res[g]:.1f} ms a pass (median of {a.reps}; mean {1e3 * np.mean(ts):.1f})", flush=True)
  if a.draft:
    import gemma4_mtp as GM
    dr = GM.Drafter(m, GM.default_mtp(meta["gguf"]), os.path.join(a.cache, "mtp")); h = m.hidden(max(geos), 0); dr.chain(first, h, P, 2)
    ts = []
    for _ in range(a.reps): t0 = time.perf_counter(); dr.chain(first, h, P, 1); ts.append(time.perf_counter() - t0)
    res["draft"] = float(np.median(ts)) * 1e3; print(f"drafter step: {res['draft']:.2f} ms (median of {a.reps})", flush=True)
  print("COST " + ",".join(f"{g}:{res[g]:.1f}" for g in geos) + (f" DRAFT {res['draft']:.2f}" if a.draft else ""), flush=True)
  if a.json: json.dump(dict(cpus=cpus, ms=res, ids=ids), open(a.json, "w"))

if __name__ == "__main__": main()

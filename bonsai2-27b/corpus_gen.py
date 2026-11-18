#!/usr/bin/env python3
"""A corpus for training a drafter: Bonsai 2 27B
regenerates a prompt set by its own speculative decoding (the deployed loop, greedy, so the text is its own), and at every committed
position dumps the last layer's hidden row (the drafter's input; QWEN_DUMP_HIDDEN) and two DeltaNet layers' inputs (the residual row
and the raw conv row; QWEN_DUMP_GDN) for the ReaDy arm. One process, the model loaded once; the prompt through the chunked prefill
(the verify path, so every position's rows come the same way).

    python3 corpus_gen.py prompts.json <out dir> [first] [count] [--max-new N]
prompts.json: [{id, src, split, text}]. Writes <out>/<id>.npz (ids, n_prompt, hidden fp16 [T, H], x<l> / raw<l> fp16 per layer),
<out>/<id>.jsonl (QWEN_SPEC_LOG: the passes), and a line per prompt to <out>/manifest.jsonl.
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "qwen3.8-27b"))
os.environ.setdefault("QWEN_PREFILL", "chunked"); os.environ.setdefault("QWEN_SPEC_TREE", "off"); os.environ["QWEN_DUMP_HIDDEN"] = "1"
os.environ.setdefault("QWEN_DUMP_GDN", "32,60")
os.environ["QWEN_SPEC_LOG"] = "1"      # the pass log's telemetry on at model build; the file is set per prompt (M.spec_log)
import bonsai2_generate  # noqa: F401,E402  (the profile's defaults: QWEN_MODEL=bonsai2-27b, its caches, QWEN_SPEC=4, the drafter)
import numpy as np  # noqa: E402
import qwen38_generate as QG  # noqa: E402

def main():
  args = [a for a in sys.argv[1:] if not a.startswith("--")]; max_new = int(sys.argv[sys.argv.index("--max-new") + 1]) if "--max-new" in sys.argv else 256
  prompts = json.load(open(args[0])); out = args[1]; first = int(args[2]) if len(args) > 2 else 0; cnt = int(args[3]) if len(args) > 3 else 10**9
  prompts = prompts[first:first + cnt]; os.makedirs(out, exist_ok=True)
  layers = [int(v) for v in os.environ["QWEN_DUMP_GDN"].split(",") if v]
  QG.pin_cpus(); QG.hold_cpu_latency(); tok = QG.Tok(os.environ["QWEN_TOK"]); t0 = time.perf_counter()
  M = QG.Model(16, os.environ["QWEN_NPU"], os.environ["QWEN_DIR"], 512, float(os.environ.get("QWEN_PIN_GB", "24")))
  print(f"model up in {time.perf_counter() - t0:.0f} s: spec {M.M} geos {M.geos} kmax {M.kmax} tau {M.tau} tree {os.environ.get('QWEN_SPEC_TREE')}; dump GDN {layers}; max_new {max_new}", flush=True)
  man = open(os.path.join(out, "manifest.jsonl"), "a"); tot_tok = tot_t = 0
  for p in prompts:
    if os.path.exists(os.path.join(out, f"{p['id']:04d}.npz")): continue
    ids = tok.encode(QG.chat_text(p["text"], False)); n = len(ids)
    if n + max_new + 8 > M.TMAX: print(f"  {p['id']:4d} skipped: {n} prompt tokens", flush=True); continue
    M.reset(); M.dump = []; M.gdump = {l: [] for l in layers}; M._gd_x, M._gd_raw = {}, {}
    M.spec_log = os.path.join(out, f"{p['id']:04d}.jsonl"); M.spec_meta = dict(prompt=p["text"], n_prompt=n, thinking=False, prompt_ids=[int(v) for v in ids], src=p["src"], split=p["split"], id=p["id"])
    t = time.perf_counter(); lg = M.prefill(ids); tp = time.perf_counter() - t; first_ = int(lg.argmax()); st = {}
    t = time.perf_counter(); gen = M.spec_generate(first_, n, max_new, "mtp", st); tg = time.perf_counter() - t
    seq = list(ids) + [int(v) for v in gen]; T = len(seq)
    hid = np.stack(M.dump).astype(np.float32); assert hid.shape[0] >= T - 1, (hid.shape, T)
    d = dict(ids=np.array(seq, np.int32), n_prompt=n, hidden=hid[:T].astype(np.float16), passes=st.get("passes", 0), t_prefill=tp, t_gen=tg)
    for l in layers:
      xs = np.concatenate([x for x, _ in M.gdump[l]]); assert xs.shape[0] == hid.shape[0], (l, xs.shape, hid.shape)
      d[f"x{l}"] = xs[:T].astype(np.float16)
      if M.gdump[l][0][1] is not None: d[f"raw{l}"] = np.concatenate([r for _, r in M.gdump[l]])[:T].astype(np.float16)
    np.savez(os.path.join(out, f"{p['id']:04d}.npz"), **d)
    rec = dict(id=p["id"], src=p["src"], split=p["split"], n_prompt=n, n_gen=len(gen), passes=st.get("passes", 0), t_prefill=round(tp, 2), t_gen=round(tg, 2), eos=int(gen[-1]) in QG.EOS if gen else False)
    man.write(json.dumps(rec) + "\n"); man.flush(); tot_tok += len(gen); tot_t += tg
    print(f"  {p['id']:4d} {p['src']:5s} {p['split']:5s} prompt {n:3d} gen {len(gen):3d} in {tg:5.1f} s ({len(gen) / max(tg, 1e-9):.2f} tok/s, {st.get('passes', 0)} passes, {len(gen) / max(1, st.get('passes', 0)):.2f} a pass)"
          f" | {tok.decode(list(gen))[:70]!r} | total {tot_tok} tokens {tot_t / 3600:.2f} h", flush=True)

if __name__ == "__main__": main()

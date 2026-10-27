#!/usr/bin/env python3
"""Ternary Bonsai 2 behind the OpenAI- / Ollama-compatible endpoint with concurrent requests BATCHED into one decode step: up to
`--users` requests advance together (bonsai2_batch.py's loop: the linears once for all rows, the batched one-token mixers), each
with its own DeltaNet states, conv rings, K / V cache and position in a slot of its own. A new request takes a free slot (its state
zeroed), its prompt goes through the same steps a token at a time (the slot's rows discarded until the prompt is in), then it decodes
plain greedy until <eos>, a stop string or max_tokens; its tokens stream as they come. Idle slots run dummy rows (the step's cost is
the geometry's, ~66 ms for 4 users on the 2B, ~85 ms for 8), so one user sees 1 / step and N users N / step.

  QWEN_MODEL=bonsai2-2b python3 bonsai2_batch_serve.py [--users 4] [--host 0.0.0.0] [--port 8000] [--tmax 512]
  (the single-request speculative server is bonsai2_serve.py: faster for one user, one request at a time)
"""
import argparse, os, queue, sys, threading, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("QWEN_MODEL", "bonsai2-2b")
import bonsai2_batch as BB                                               # noqa: E402  (sets the environment, imports the runtime)
import numpy as np                                                       # noqa: E402
import qwen38_generate as QG, qwen38_ref as R, qwen38_serve as S         # noqa: E402
from qwen38_tokenize import Tok                                          # noqa: E402
from tinygrad import Tensor                                              # noqa: E402

EOS = S.EOS


class BatchEngine:
  def __init__(self, a):
    self.cpus, self.lat = QG.pin_cpus(), QG.hold_cpu_latency()
    self.tok = Tok(os.environ.get("QWEN_TOK", a.dir)); self.tmax, self.N, self.jobs = a.tmax, a.users, queue.Queue()
    t0 = time.perf_counter(); self.M = QG.Model(1, a.cache, a.dir, a.tmax, a.pin_gb); self.B = BB.Batch(self.M, a.users, a.tmax)
    self.slots = [None] * a.users; self.lock = threading.Lock()
    NG = self.M.NGDN; self.ssz, self.csz = NG * R.NV * R.DK * R.DV, NG * R.CONV * self.B.CP
    self.B.step([0] * a.users, [0] * a.users)                              # the graphs captured before the first request
    print(f"   model set up in {time.perf_counter() - t0:.0f} s; {a.users} slots; host CPUs {self.cpus}; CPU latency {self.lat}", flush=True)
  def submit(self, ids, max_new, stop):
    q = queue.Queue(); self.jobs.put((list(ids), max_new, stop, q)); return q
  def _free(self, u, job):
    """Slot u takes a job: its DeltaNet states and rings zeroed, its position 0."""
    ids, max_new, stop, q = job
    if len(ids) + 1 > self.tmax: q.put(("error", f"the prompt ({len(ids)} tokens) does not fit the context ({self.tmax})")); return
    self.B.Sall[u * self.ssz:(u + 1) * self.ssz].assign(Tensor.zeros(self.ssz, device=BB.DEV)).realize()
    self.B.Call[u * self.csz:(u + 1) * self.csz].assign(Tensor.zeros(self.csz, device=BB.DEV)).realize()
    self.slots[u] = dict(ids=ids, k=0, out=[], max_new=max(1, min(max_new, self.tmax - len(ids) - 1)), stop=stop, q=q, pos=0, sent="", t0=time.perf_counter(), tf=None)
  def _emit(self, s, t):
    """Token t into slot s's output: the text's new suffix streamed; done on <eos>, a stop string or max_tokens -> True when finished."""
    s["out"].append(int(t)); fin = None
    if s["tf"] is None: s["tf"] = time.perf_counter()
    text = self.tok.decode([v for v in s["out"] if v not in EOS])
    if not text.endswith("�"):
      for st in s["stop"]:
        if st and (i := text.find(st)) >= 0: text = text[:i]; fin = "stop"; break
      if len(text) > len(s["sent"]): s["q"].put(("delta", text[len(s["sent"]):])); s["sent"] = text
    if t in EOS: fin = "stop"
    if fin is None and len(s["out"]) >= s["max_new"]: fin = "length"
    if fin is not None:
      t1 = time.perf_counter(); s["q"].put(("done", len([v for v in s["out"] if v not in EOS]), fin, s["tf"] - s["t0"], t1 - s["tf"])); return True
    return False
  def serve_jobs(self):
    """The main thread's loop: a step whenever a slot is busy; waiting requests fill the free slots between steps."""
    N = self.N
    while True:
      if all(s is None for s in self.slots): job = self.jobs.get(); self._free(self.slots.index(None), job)
      while any(s is None for s in self.slots) and not self.jobs.empty(): self._free(self.slots.index(None), self.jobs.get_nowait())
      toks, pos = [], []
      for s in self.slots:
        if s is None: toks.append(0); pos.append(0)
        elif s["k"] < len(s["ids"]): toks.append(s["ids"][s["k"]]); pos.append(s["pos"])
        else: toks.append(s["out"][-1]); pos.append(s["pos"])
      try: nxt = self.B.step(toks, pos)
      except Exception as e:
        for u, s in enumerate(self.slots):
          if s is not None: s["q"].put(("error", f"{type(e).__name__}: {e}")); self.slots[u] = None
        print(f"   step failed: {e!r}", flush=True); continue
      for u, s in enumerate(self.slots):
        if s is None: continue
        s["pos"] += 1
        if s["k"] < len(s["ids"]):
          s["k"] += 1
          if s["k"] < len(s["ids"]): continue                             # the prompt still going in: the row's output discarded
        if self._emit(s, nxt[u]): self.slots[u] = None


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
  ap.add_argument("--users", type=int, default=int(os.environ.get("BATCH_USERS", "4")), help="slots: requests decoded together (the step's geometry; 8 at most pays)")
  ap.add_argument("--model-name", default=os.environ.get("QWEN_SERVE_NAME", os.environ["QWEN_MODEL"])); ap.add_argument("--tmax", type=int, default=512, help="the context (prompt + generated tokens): the batched attention kernel's scores live in LSRAM, 512 at most")
  ap.add_argument("--pin-gb", type=float, default=float(os.environ.get("QWEN_PIN_GB", "12")))
  ap.add_argument("--dir", default=os.environ["QWEN_DIR"]); ap.add_argument("--cache", default=os.environ["QWEN_NPU"]); a = ap.parse_args()
  assert a.tmax <= 512, "--tmax: the batched attention kernel holds a row's scores in LSRAM (512 positions)"
  eng = BatchEngine(a); srv = S.ThreadingHTTPServer((a.host, a.port), S.make_handler(eng, a.model_name))
  print(f"== serving {a.model_name} on http://{a.host}:{a.port}/v1 (OpenAI- / Ollama-compatible; greedy; {a.users} requests a step; context {a.tmax} tokens)", flush=True)
  threading.Thread(target=srv.serve_forever, daemon=True).start()
  try: eng.serve_jobs()
  except KeyboardInterrupt: pass


if __name__ == "__main__": main()

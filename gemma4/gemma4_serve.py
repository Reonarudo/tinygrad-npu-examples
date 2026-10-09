#!/usr/bin/env python3
"""Gemma 4 (E2B / E4B) behind an OpenAI- and Ollama-compatible HTTP endpoint: qwen3.8-27b/qwen38_serve.py's HTTP side and request
queue, with Gemma's model, its tokenizer and chat template (gemma4_tokenize: from the GGUF itself) and gemma4_generate's decoding
-- speculative with the MTP drafter when its GGUF is beside the model (as gemma4_generate: 6 drafts a pass, tau 0.5), the text
streamed as each verify pass accepts it. Greedy; thinking off.

  python3 gemma4_serve.py [--host 0.0.0.0] [--port 8000] [--cache /mnt/ssd/gemma4-e2b-npu] [--tmax 1024] [--spec 6] [--tau 0.5]
  curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \\
       -d '{"model": "gemma4-e2b", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
  OLLAMA_HOST=http://<board>:8000 ollama run --verbose gemma4-e2b "Hello"
"""
import argparse, json, os, queue, sys, threading, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np                                                        # noqa: E402
import gemma4_generate as GG                                              # noqa: E402  (puts qwen3.8-27b on sys.path)
import gemma4_ref as GR                                                   # noqa: E402
from gemma4_tokenize import Tok, chat_messages, STOP                     # noqa: E402
import qwen38_serve as QS                                                 # noqa: E402  (its HTTP side; the Qwen model is not imported)
from http.server import ThreadingHTTPServer                               # noqa: E402

class Engine(QS.Engine):
  """The model, the drafter and the one-request-at-a-time generation; the queue (submit / serve_jobs) is qwen38_serve's."""
  def __init__(self, a):
    self.cpus, self.lat = GG.pin_cpus(), GG.hold_cpu_latency()
    meta = json.load(open(os.path.join(a.cache, "gemma4.json"))); gguf = meta["gguf"]; mtp = a.mtp or GG.GM_default_mtp(gguf)
    self.spec = a.spec if a.spec is not None else int(os.environ["GEMMA_SPEC"]) if "GEMMA_SPEC" in os.environ else 6 if os.path.exists(mtp) else 0
    self.e4b = "E4B" in os.path.basename(gguf)
    if self.spec and self.e4b: os.environ.setdefault("GEMMA_LAYER_BLOCK", "4"); os.environ.setdefault("GEMMA_SPEC_GEOS", "3,5,7")   # as gemma4_generate
    t0 = time.perf_counter(); R = GR.Model(gguf, cache_layers=os.environ.get("GEMMA_PLE", "dev") == "host")
    self.tok = Tok(R.g.meta); self.tmax, self.lock, self.jobs, self.tau, self.pm = a.tmax, threading.Lock(), queue.Queue(), a.tau, a.prefill_m
    self.m = GG.Model(R, a.cache, a.tmax); self.dr = None
    if self.spec:
      import gemma4_mtp as GM
      self.geos = sorted({int(x) for x in os.environ.get("GEMMA_SPEC_GEOS", ",".join(str(x) for x in range(2, self.spec + 2))).split(",")} | {self.spec + 1})
      self.dr = GM.Drafter(self.m, mtp, os.path.join(a.cache, "mtp")); self.m.warmup(sorted({self.pm} | set(self.geos)))
      self.dr.chain(2, np.zeros(self.m.c.H, np.float32), 1, 2)                                     # (the drafter's graph captured)
    else: self.m.warmup(sorted({1, self.pm}))
    print(f"   model set up in {time.perf_counter() - t0:.0f} s; {'speculative, ' + str(self.spec) + ' drafts a pass' if self.spec else 'plain greedy'}; "
          f"host CPUs {self.cpus}; CPU latency {self.lat}", flush=True)
  def run(self, ids, max_new, stop, on_delta):
    """Greedy generation after `ids`; on_delta(text) per accepted chunk. Returns (completion tokens, finish_reason)."""
    m, tok = self.m, self.tok; out, sent, finish = [], [""], ["length"]; self.t_first = None
    room = self.tmax - len(ids) - max(self.pm, self.spec + 1)                 # (the last prefill chunk and a verify pass are padded)
    if room < 1: raise ValueError(f"the prompt ({len(ids)} tokens) does not fit the context ({self.tmax})")
    max_new = max(1, min(max_new, room))
    def emit(new):
      if self.t_first is None: self.t_first = time.perf_counter()
      out.extend(int(t) for t in new)
      text = tok.decode([t for t in out if t not in STOP])
      if text.endswith("�"): return                                     # an incomplete UTF-8 sequence: wait for its rest
      for s in stop:
        if s and (i := text.find(s)) >= 0: text = text[:i]; finish[0] = "stop"; break
      if len(text) > len(sent[0]): on_delta(text[len(sent[0]):]); sent[0] = text
      if any(t in STOP for t in new): finish[0] = "stop"
      if finish[0] == "stop": raise QS.Stop
    with self.lock:
      try:
        if self.dr: m.spec_generate(ids, max_new, self.dr, self.spec, self.tau, pm=self.pm, geos=self.geos, on_new=emit)
        else:
          nxt = m.prefill(ids, self.pm); emit([nxt])
          for i in range(1, max_new): emit([m.step(out[-1], len(ids) + i - 1)])
      except QS.Stop: pass
    return min(len(out), max_new), finish[0]

def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
  ap.add_argument("--cache", default=GG.DEFAULT_CACHE); ap.add_argument("--model-name", default=os.environ.get("GEMMA_SERVE_NAME"))
  ap.add_argument("--tmax", type=int, default=1024, help="the context: prompt + generated tokens (the global layers' cache length)")
  ap.add_argument("--prefill-m", type=int, default=int(os.environ.get("GEMMA_PREFILL_M", "8")), help="prompt tokens a pass")
  ap.add_argument("--spec", type=int, default=None, help="drafts a pass (0: plain greedy; default GEMMA_SPEC, else 6 when the drafter is found)")
  ap.add_argument("--tau", type=float, default=float(os.environ.get("GEMMA_DRAFT_TAU", "0.5"))); ap.add_argument("--mtp", default=None)
  a = ap.parse_args(); eng = Engine(a); size = "E4B" if eng.e4b else "E2B"; name = a.model_name or f"gemma4-{size.lower()}"
  details = {"format": "npu", "family": "gemma4", "families": ["gemma4"], "parameter_size": size, "quantization_level": "Q8_0"}
  srv = ThreadingHTTPServer((a.host, a.port), QS.make_handler(eng, name, chat_messages, details))
  print(f"== serving {name} on http://{a.host}:{a.port}/v1 (OpenAI- and Ollama-compatible; greedy; context {a.tmax} tokens)", flush=True)
  threading.Thread(target=srv.serve_forever, daemon=True).start()
  try: eng.serve_jobs()
  except KeyboardInterrupt: pass

if __name__ == "__main__": main()

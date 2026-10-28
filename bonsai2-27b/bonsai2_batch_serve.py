#!/usr/bin/env python3
"""Ternary Bonsai 2 behind the OpenAI- / Ollama-compatible endpoint with concurrent requests BATCHED into one decode step: up to
`--users` requests advance together (bonsai2_batch.py's loop: the linears once for all rows, the batched one-token mixers), each
with its own DeltaNet states, conv rings, K / V cache and position in a slot of its own. A new request takes a free slot (its state
zeroed), its prompt goes through the same steps a token at a time (the slot's rows discarded until the prompt is in), then it decodes
plain greedy until <eos>, a stop string or max_tokens; its tokens stream as they come. Idle slots run dummy rows (the step's cost is
the geometry's, ~66 ms for 4 users on the 2B, ~85 ms for 8), so one user sees 1 / step and N users N / step.

The prompt: a short one (< BATCH_PREFILL_MIN, 48 tokens) goes through the batch steps a token at a time in its slot (the row's
outputs discarded until it is in: no other user slows down); a long one through the model's chunked prefill (the verify path,
max(geometries) = 4 tokens a pass on the model's own states and cache), one request at a time, a pass between two batch steps while
others decode (back to back when no slot is busy); when it is in, the DeltaNet states, rings and the prompt's K / V rows are copied
into the slot and the first token goes out. The context
(`--tmax`, 65536 by default) sets the K / V caches: (1 + users) x attention layers x tmax x 2 KiB (2B: 1.5 GB a user at 64k;
the 4B / 9B 4.3 GB a user -- fewer users or a shorter context there).

With the caches in the NPU's ASIDs 1-3 (QWEN_KV_ASID=1, bonsai2_generate's default for the smaller sizes: three more 3 GiB address
spaces the attention kernels reach by DMA) the ASID0 window is not the bound: `auto` is the largest context whose fullest ASID holds its
layers' caches for every user (QWEN_TMAX_MAX caps it, 65536), and the prefill writes straight into the user's cache.

  QWEN_MODEL=bonsai2-2b python3 bonsai2_batch_serve.py [--users 4] [--host 0.0.0.0] [--port 8000] [--tmax auto]
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
PREFILL_MIN = int(os.environ.get("BATCH_PREFILL_MIN", "48"))
PRE = "prefilling"                                                           # a slot reserved for the prompt in the chunked prefill                   # prompts this long or longer go through the chunked prefill


class BatchEngine:
  def __init__(self, a):
    self.cpus, self.lat = QG.pin_cpus(), QG.hold_cpu_latency()
    self.tok = Tok(os.environ.get("QWEN_TOK", a.dir)); self.tmax, self.N, self.jobs, self.dir = a.tmax, a.users, queue.Queue(), a.dir
    t0 = time.perf_counter(); self.M = QG.Model(1, a.cache, a.dir, a.tmax, a.pin_gb); self.B = BB.Batch(self.M, a.users, a.tmax)
    self.slots = [None] * a.users; self.lock = threading.Lock()
    NG = self.M.NGDN; self.ssz, self.csz = NG * R.NV * R.DK * R.DV, NG * R.CONV * self.B.CP
    self.B.step([0] * a.users, [0] * a.users)                              # the graphs captured before the first request
    print(f"   model set up in {time.perf_counter() - t0:.0f} s; {a.users} slots; host CPUs {self.cpus}; CPU latency {self.lat}", flush=True)
  vision, decode_images, embed_images, prompt = S.Engine.vision, S.Engine.decode_images, S.Engine.embed_images, S.Engine.prompt   # images: the vision tower, the template with the placeholders (qwen38_serve)
  def submit(self, ids, max_new, stop, vis=None):
    """`vis`: the prompt's images [(start, pixels, nx, ny)] (prompt()); they go through the chunked prefill whatever the length."""
    q = queue.Queue(); self.jobs.put((list(ids), max_new, stop, q, vis)); return q
  def _short(self, u, job):
    """Slot u takes a short prompt: its DeltaNet states and rings zeroed, the prompt's ids go in a token a step from position 0."""
    ids, max_new, stop, q, _ = job
    self.B.Sall[u * self.ssz:(u + 1) * self.ssz].assign(Tensor.zeros(self.ssz, device=BB.DEV)).realize()
    self.B.Call[u * self.csz:(u + 1) * self.csz].assign(Tensor.zeros(self.csz, device=BB.DEV)).realize()
    self.slots[u] = dict(ids=ids, k=0, out=[], max_new=max(1, min(max_new, self.tmax - len(ids) - 1)), stop=stop, q=q, pos=0, delta=0, sent="", t0=time.perf_counter(), tf=None)
  def _start(self, job):
    """A job's prompt starts through the model's chunked prefill (one at a time) for the free slot it reserves (PRE): -> the
    in-flight record. The caches in ASIDs: the model prefills straight into that user's cache (kv_user)."""
    ids, max_new, stop, q, vis = job; u = self.slots.index(None); self.slots[u] = PRE
    M = self.M; M.kv_user = u; M.set_vision(ids, self.embed_images(vis)); M.prefill_prelude(); M.spec_meta = dict(prompt="(server)", n_prompt=len(ids), thinking=False, prompt_ids=[int(v) for v in ids])
    return dict(job=job, gen=M.prefill_chunks(ids), t0=time.perf_counter(), u=u, delta=M.pos_delta)   # delta: the m-RoPE counter's lag after the images (0 without)
  def _place(self, pre):
    """The prefilled prompt into slot u: the model's DeltaNet states, rings and the prompt's K / V rows copied (the model's own
    stacks were the prefill's), the first token emitted; the slot decodes from position n."""
    ids, max_new, stop, q, _ = pre["job"]; M, B, n, u = self.M, self.B, len(pre["job"][0]), pre["u"]; NA = M.NATT; rs = R.NKV * R.HD
    M.flush()                                                             # (the deferred commit: the last pass's state updates into Sall)
    B.Sall[u * self.ssz:(u + 1) * self.ssz].assign(M.Sall).realize(); B.Call[u * self.csz:(u + 1) * self.csz].assign(M.Call).realize()
    if not M.kvs:
      for i in range(NA):
        src, dst = i * M.TMAX * rs, (u * NA + i) * self.tmax * rs
        B.Kall[dst:dst + n * rs].assign(M.Kall[src:src + n * rs]).realize(); B.Vall[dst:dst + n * rs].assign(M.Vall[src:src + n * rs]).realize()
    first = int(np.argmax(M.prefill_lg))
    s = self.slots[u] = dict(ids=ids, k=n, out=[], max_new=max(1, min(max_new, self.tmax - n - 1)), stop=stop, q=q, pos=n, delta=pre["delta"], sent="", t0=pre["t0"], tf=None)
    if self._emit(s, first): self.slots[u] = None
  def _emit(self, s, t):
    """Token t into slot s's output: the text's new suffix streamed; done on <eos>, a stop string or max_tokens -> True when finished."""
    s["out"].append(int(t)); fin = None
    if s["tf"] is None: s["tf"] = time.perf_counter()
    text = self.tok.decode([v for v in s["out"] if v not in EOS])
    if not text.endswith("\ufffd"):
      for st in s["stop"]:
        if st and (i := text.find(st)) >= 0: text = text[:i]; fin = "stop"; break
      if len(text) > len(s["sent"]): s["q"].put(("delta", text[len(s["sent"]):])); s["sent"] = text
    if t in EOS: fin = "stop"
    if fin is None and len(s["out"]) >= s["max_new"]: fin = "length"
    if fin is not None:
      t1 = time.perf_counter(); s["q"].put(("done", len([v for v in s["out"] if v not in EOS]), fin, s["tf"] - s["t0"], t1 - s["tf"])); return True
    return False
  def _take(self):
    """The next waiting job: rejected (too long), or placed in a free slot (short), or started on the chunked prefill (-> its record)."""
    job = self.jobs.get_nowait(); ids = job[0]
    if len(ids) + 1 > self.tmax: job[3].put(("error", f"the prompt ({len(ids)} tokens) does not fit the context ({self.tmax})")); return None
    if len(ids) < PREFILL_MIN and job[4] is None: self._short(self.slots.index(None), job); return None
    return self._start(job)
  def _busy(self): return [s for s in self.slots if s is not None and s is not PRE]
  def serve_jobs(self):
    """The main thread's loop: a prefill pass (the one long prompt in flight) and a batch step (when a slot is busy) in turn; a prompt
    that is in takes a free slot; waiting requests start while a slot is free (and the prefill is, for a long one)."""
    pre = None
    while True:
      if pre is None and all(s is None for s in self.slots) and self.jobs.empty(): self.jobs.put(self.jobs.get())   # (block until a request comes)
      while None in self.slots and not self.jobs.empty() and (pre is None or len(self.jobs.queue[0][0]) < PREFILL_MIN): pre = self._take() or pre
      if pre is not None:
        try:
          if next(pre["gen"]) >= len(pre["job"][0]): self._place(pre); pre = None   # (the last pass yields the prompt's length)
        except Exception as e: pre["job"][3].put(("error", f"{type(e).__name__}: {e}")); print(f"   prefill failed: {e!r}", flush=True); self.slots[pre["u"]] = None; pre = None
      if not self._busy(): continue
      toks, pos, rpos = [], [], []
      for s in self.slots:
        if s is None or s is PRE: toks.append(0); pos.append(0); rpos.append(0)
        elif s["k"] < len(s["ids"]): toks.append(s["ids"][s["k"]]); pos.append(s["pos"]); rpos.append(s["pos"] - s["delta"])
        else: toks.append(s["out"][-1]); pos.append(s["pos"]); rpos.append(s["pos"] - s["delta"])
      try: nxt = self.B.step(toks, pos, rpos)
      except Exception as e:
        for u, s in enumerate(self.slots):
          if s is not None and s is not PRE: s["q"].put(("error", f"{type(e).__name__}: {e}")); self.slots[u] = None
        print(f"   step failed: {e!r}", flush=True); continue
      for u, s in enumerate(self.slots):
        if s is None or s is PRE: continue
        s["pos"] += 1
        if s["k"] < len(s["ids"]):
          s["k"] += 1
          if s["k"] < len(s["ids"]): continue                             # the prompt still going in: the row's output discarded
        if self._emit(s, nxt[u]): self.slots[u] = None


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
  ap.add_argument("--users", type=int, default=int(os.environ.get("BATCH_USERS", "4")), help="slots: requests decoded together (the step's geometry; 8 at most pays)")
  ap.add_argument("--model-name", default=os.environ.get("QWEN_SERVE_NAME", os.environ["QWEN_MODEL"])); ap.add_argument("--tmax", default=os.environ.get("QWEN_TMAX", "auto"), help="the context (prompt + generated tokens): the K / V caches' length; `auto` (default): the largest up to 65536 whose (1 + users) caches fit QWEN_KV_GB (2.0) of NPU-visible memory (the 2B: 16384 at 4 users, 8192 at 8)")
  ap.add_argument("--pin-gb", type=float, default=float(os.environ.get("QWEN_PIN_GB", "12")))
  ap.add_argument("--dir", default=os.environ["QWEN_DIR"]); ap.add_argument("--cache", default=os.environ["QWEN_NPU"]); a = ap.parse_args()
  auto = a.tmax == "auto" or os.environ.get("QWEN_TMAX_AUTO") == "1"
  os.environ.setdefault("QWEN_KV_USERS", str(a.users)); asid = os.environ.get("QWEN_KV_ASID", "0") == "1"
  # the batched server is for throughput: fp32 caches unless an explicit --tmax needs fp16 to fit (QWEN_KV_DTYPE overrides)
  if "QWEN_KV_DTYPE" not in os.environ:
    if a.tmax == "auto" or S.asid_tmax(a.users, R.NL // 4, R.NKV, R.HD, False, int(a.tmax))[0] >= int(a.tmax): os.environ["QWEN_KV_DTYPE"] = "f32"
    else: os.environ["QWEN_KV_DTYPE"] = "f16"
  a.tmax, kv_gb = S.auto_tmax(a.users, float(os.environ.get("QWEN_KV_GB", "2.0")), R.NL // 4, R.NKV, R.HD, want=int(os.environ.get("QWEN_TMAX_MAX", "262144")), mtp=False) if a.tmax == "auto" else (int(a.tmax), (S.asid_tmax(a.users, R.NL // 4, R.NKV, R.HD, False, int(a.tmax))[1] if asid else S.kv_window_bytes(a.users, R.NL // 4, R.NKV, R.HD, int(a.tmax), False) / 1e9))
  print(f"   context {a.tmax} tokens: K / V caches {kv_gb:.2f} GB {'in the fullest of ASIDs 1-3 (3 GiB each; none of the ASID0 window; ' + os.environ.get('QWEN_KV_DTYPE', 'f16') + ')' if asid else f'of the NPU window ({1 + a.users} x {R.NL // 4} attention layers as buffers; ~2.75 GB in all)'}", flush=True)
  eng = S.engine_fit(BatchEngine, a, auto); srv = S.ThreadingHTTPServer((a.host, a.port), S.make_handler(eng, a.model_name))
  print(f"== serving {a.model_name} on http://{a.host}:{a.port}/v1 (OpenAI- / Ollama-compatible; greedy; {a.users} requests a step; context {a.tmax} tokens)", flush=True)
  threading.Thread(target=srv.serve_forever, daemon=True).start()
  try: eng.serve_jobs()
  except KeyboardInterrupt: pass


if __name__ == "__main__": main()

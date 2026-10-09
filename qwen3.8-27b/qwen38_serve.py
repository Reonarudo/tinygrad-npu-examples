#!/usr/bin/env python3
"""An OpenAI-compatible HTTP endpoint for the NPU generator (Python's standard library only): the model is loaded once, requests
run one at a time on the NPU (a lock; others wait in line), and the tokens stream as the speculative decoding's verify passes
accept them.

  GET  /v1/models
  POST /v1/chat/completions   {"messages": [...], "max_tokens" | "max_completion_tokens", "stream", "stop", ...}
  POST /v1/completions        {"prompt": "...", ...}           (raw text, no chat template)
and Ollama's API for Ollama clients (e.g. Enchanted): HEAD /, GET /api/tags, GET /api/version, POST /api/show, POST /api/chat,
POST /api/generate (newline-delimited JSON streaming, the default; "options": {"num_predict", "stop"}). Every route is also
accepted under a /v1 prefix (a client configured with the OpenAI base URL).

tinygrad is not thread-safe, so the model lives on the main thread, which serves a queue of requests; the HTTP handler threads
only queue a request and relay its text. Decoding is greedy (the verify pass's argmax): temperature / top_p / seed are accepted and ignored, n must be 1. Thinking is off
unless the request sets "chat_template_kwargs": {"enable_thinking": true} (or "enable_thinking": true). Prompt + output must fit
--tmax tokens. The prompt goes through the verify path (QWEN_PREFILL=chunked, any length; 4 tokens a pass).

  python3 qwen38_serve.py [--host 0.0.0.0] [--port 8000] [--model-name NAME] [--tmax 2048]    (model paths: as qwen38_generate)
  curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \\
       -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
"""
import argparse, json, os, queue, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
os.environ.setdefault("QWEN_PREFILL", "chunked")
os.environ.setdefault("QWEN_SPEC", "6")                   # the server runs the verify path: speculative decoding up to 6 rows unless set
                                                          # (bonsai2_serve / ornith set their own defaults before this import)
import numpy as np                                                        # noqa: E402
from qwen38_tokenize import Tok, chat_messages, EOS                       # noqa: E402

class Stop(Exception): pass

OLLAMA_DETAILS = {"format": "npu", "family": "qwen35", "families": ["qwen35"], "parameter_size": "27B", "quantization_level": "ternary"}

class Engine:
  """The model and its one-request-at-a-time generation (`run` yields text deltas)."""
  def __init__(self, a):
    import qwen38_generate as G                                           # (imported here: gemma4_serve uses this module for its HTTP side only)
    self.cpus, self.lat = G.pin_cpus(), G.hold_cpu_latency()
    self.tok = Tok(os.environ.get("QWEN_TOK", a.dir)); self.tmax, self.lock, self.jobs = a.tmax, threading.Lock(), queue.Queue()
    t0 = time.perf_counter(); self.M = G.Model(12, a.cache, a.dir, a.tmax, a.pin_gb)
    assert self.M.M, "the server runs the verify path (QWEN_SPEC >= 2 geometries)"
    print(f"   model set up in {time.perf_counter() - t0:.0f} s; host CPUs {self.cpus}; CPU latency {self.lat}", flush=True)
  def submit(self, ids, max_new, stop):
    """Queue a request (from any thread) -> a queue of ("delta", text) ... then ("done", tokens, finish) or ("error", message)."""
    q = queue.Queue(); self.jobs.put((ids, max_new, stop, q)); return q
  def serve_jobs(self):
    """The main thread's loop: one request at a time, on the thread that built the model."""
    while True:
      ids, max_new, stop, q = self.jobs.get()
      try:
        t0 = time.perf_counter(); r = self.run(ids, max_new, stop, lambda d: q.put(("delta", d))); t1 = time.perf_counter()
        tf = self.t_first or t1; q.put(("done",) + r + (tf - t0, t1 - tf))      # + the prompt's seconds, the generation's
      except ValueError as e: q.put(("error", str(e)))
      except Exception as e: q.put(("error", f"{type(e).__name__}: {e}")); print(f"   request failed: {e!r}", flush=True)
  def run(self, ids, max_new, stop, on_delta):
    """Greedy generation after `ids`; on_delta(text) per accepted chunk. Returns (completion tokens, finish_reason)."""
    M, tok = self.M, self.tok; out, sent = [], [""]; self.t_first = None
    if len(ids) + 1 > self.tmax: raise ValueError(f"the prompt ({len(ids)} tokens) does not fit the context ({self.tmax})")
    max_new = max(1, min(max_new, self.tmax - len(ids) - 1)); finish = ["length"]
    def emit(new):
      if self.t_first is None: self.t_first = time.perf_counter()
      out.extend(int(t) for t in new)
      text = tok.decode([t for t in out if t not in EOS])
      if text.endswith("�"): return                                   # an incomplete UTF-8 sequence: wait for its rest
      for s in stop:
        if s and (i := text.find(s)) >= 0: text = text[:i]; finish[0] = "stop"; break
      if len(text) > len(sent[0]): on_delta(text[len(sent[0]):]); sent[0] = text
      if any(t in EOS for t in new): finish[0] = "stop"
      if finish[0] == "stop": raise Stop
    with self.lock:
      M.reset(); M.spec_meta = dict(prompt="(server)", n_prompt=len(ids), thinking=False, prompt_ids=[int(v) for v in ids])
      lg = M.prefill(ids); first = int(np.argmax(lg))
      try:
        emit([first])
        if first not in EOS and max_new > 1:
          if M.MTPL: M.spec_generate(first, len(ids), max_new, "mtp", {}, lambda new, tv: emit(new))
          else:
            for i in range(1, max_new):
              lg = M.step(out[-1], len(ids) + i - 1); emit([int(np.argmax(lg))])
      except Stop: pass
    return min(len(out), max_new), finish[0]

def make_handler(eng, name, chat_messages=chat_messages, details=OLLAMA_DETAILS):
  class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, fmt, *args): print("   " + fmt % args, flush=True)
    def _json(self, code, obj):
      b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def _err(self, code, msg): self._json(code, {"error": {"message": msg, "type": "invalid_request_error"}})
    def _route(self): return "/" + self.path.split("?")[0].strip("/").removeprefix("v1").strip("/")
    def do_HEAD(self):
      self.send_response(200); self.send_header("Content-Type", "text/plain"); self.send_header("Content-Length", "0"); self.end_headers()
    def _ndjson_start(self):
      self.send_response(200); self.send_header("Content-Type", "application/x-ndjson"); self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
    def _chunk(self, o):
      b = (json.dumps(o) + "\n").encode(); self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n"); self.wfile.flush()
    def _ollama(self, route):
      """Ollama's API: /api/chat (messages) and /api/generate (prompt [+ system]; "raw": no template), streaming by default."""
      try: req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
      except ValueError: return self._json(400, {"error": "the body is not JSON"})
      if route == "/api/show":
        return self._json(200, {"modelfile": "", "parameters": "", "template": "", "details": details, "model_info": {}, "capabilities": ["completion"]})
      if (route == "/api/generate" and not req.get("prompt")) or (route == "/api/chat" and not req.get("messages")):   # a client's "load the model" call
        return self._json(200, {"model": name, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()), "done": True, "done_reason": "load",
                                **({"message": {"role": "assistant", "content": ""}} if route == "/api/chat" else {"response": ""})})
      opts = req.get("options") or {}; think = bool(req.get("think", False))
      if route == "/api/chat":
        if not req.get("messages"): return self._json(400, {"error": "messages is required"})
        text = chat_messages(req["messages"], think)
      else:
        p = req.get("prompt", "")
        text = p if req.get("raw") else chat_messages(([{"role": "system", "content": req["system"]}] if req.get("system") else []) + [{"role": "user", "content": p}], think)
      ids = eng.tok.encode(text); n0 = int(opts.get("num_predict") or 512); max_new = 512 if n0 < 0 else n0
      stop = opts.get("stop") or []; stop = [stop] if isinstance(stop, str) else list(stop)
      t0 = time.perf_counter(); q = eng.submit(ids, max_new, stop); chat = route == "/api/chat"
      now = lambda: time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
      piece = lambda d: {"message": {"role": "assistant", "content": d}} if chat else {"response": d}
      def final(n, finish, tp, tg):
        # Ollama's rate is eval_count / eval_duration: the tokens after the first over the time after it (the first token
        # comes with the prompt's processing, counted in prompt_eval_duration)
        return dict({"model": name, "created_at": now()}, **piece(""), done=True, done_reason=finish, total_duration=int((time.perf_counter() - t0) * 1e9),
                    load_duration=0, prompt_eval_count=len(ids), prompt_eval_duration=int(tp * 1e9), eval_count=max(0, n - 1), eval_duration=max(1, int(tg * 1e9)))
      if req.get("stream", True):
        self._ndjson_start()
        try:
          while (m := q.get())[0] == "delta": self._chunk(dict({"model": name, "created_at": now()}, **piece(m[1]), done=False))
          self._chunk({"error": m[1]} if m[0] == "error" else final(*m[1:]))
          self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError): pass
        return
      parts = []
      while (m := q.get())[0] == "delta": parts.append(m[1])
      if m[0] == "error": return self._json(400, {"error": m[1]})
      out = final(*m[1:]); out.update(piece("".join(parts))); self._json(200, out)
    def do_GET(self):
      route = self._route()
      if route == "/api/tags":
        return self._json(200, {"models": [{"name": name, "model": name, "modified_at": "2026-10-05T00:00:00Z", "size": 0, "digest": "0" * 64, "details": details}]})
      if route == "/api/version": return self._json(200, {"version": "0.5.0"})
      if route == "/api/ps": return self._json(200, {"models": [{"name": name, "model": name, "size": 0, "digest": "0" * 64, "details": details}]})
      if route == "/": return self._json(200, {"status": "ok"})
      if self.path.rstrip("/") in ("/v1/models", "/models"):
        return self._json(200, {"object": "list", "data": [{"id": name, "object": "model", "created": 0, "owned_by": "local"}]})
      if self.path in ("/health", "/"): return self._json(200, {"status": "ok"})
      self._err(404, f"no route {self.path}")
    def do_POST(self):
      route = self._route()
      if route in ("/api/chat", "/api/generate", "/api/show"): return self._ollama(route)
      if route not in ("/chat/completions", "/completions"): return self._err(404, f"no route {self.path}")
      try: req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
      except ValueError: return self._err(400, "the body is not JSON")
      chat = route == "/chat/completions"
      if int(req.get("n", 1)) != 1: return self._err(400, "n must be 1 (greedy decoding)")
      kw = req.get("chat_template_kwargs") or {}; think = bool(kw.get("enable_thinking", req.get("enable_thinking", False)))
      if chat:
        if not req.get("messages"): return self._err(400, "messages is required")
        text = chat_messages(req["messages"], think)
      else:
        text = req.get("prompt", ""); text = text[0] if isinstance(text, list) else text
      ids = eng.tok.encode(text); max_new = int(req.get("max_completion_tokens") or req.get("max_tokens") or 512)
      stop = req.get("stop") or []; stop = [stop] if isinstance(stop, str) else list(stop)
      rid, created = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24], int(time.time())
      obj = "chat.completion" if chat else "text_completion"
      def choice(text, finish, delta=False):
        if chat: return {"index": 0, ("delta" if delta else "message"): ({"content": text} if delta else {"role": "assistant", "content": text}), "finish_reason": finish}
        return {"index": 0, "text": text, "finish_reason": finish, "logprobs": None}
      if req.get("stream"):
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        def sse(o):
          b = f"data: {o if isinstance(o, str) else json.dumps(o)}\n\n".encode()
          self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n"); self.wfile.flush()
        base = {"id": rid, "object": obj + ".chunk" if chat else obj, "created": created, "model": name}
        if chat: sse(dict(base, choices=[{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]))
        q = eng.submit(ids, max_new, stop)
        try:
          while (m := q.get())[0] == "delta": sse(dict(base, choices=[choice(m[1], None, delta=True)]))
          if m[0] == "error": sse({"error": {"message": m[1], "type": "invalid_request_error"}})
          else:
            n, finish = m[1], m[2]; last = dict(base, choices=[choice("", finish, delta=True)])
            if (req.get("stream_options") or {}).get("include_usage"): last["usage"] = {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}
            sse(last); sse("[DONE]")
        except (BrokenPipeError, ConnectionResetError): return     # the client left; its request still runs to its end
        self.wfile.write(b"0\r\n\r\n"); self.wfile.flush(); return
      parts, q = [], eng.submit(ids, max_new, stop)
      while (m := q.get())[0] == "delta": parts.append(m[1])
      if m[0] == "error": return self._err(400, m[1])
      n, finish = m[1], m[2]
      self._json(200, {"id": rid, "object": obj, "created": created, "model": name, "choices": [choice("".join(parts), finish)],
                       "usage": {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}})
  return H

def main(default_name="qwen3.8-27b"):
  ap = argparse.ArgumentParser(); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
  ap.add_argument("--model-name", default=os.environ.get("QWEN_SERVE_NAME", default_name))
  ap.add_argument("--tmax", type=int, default=2048, help="the context: prompt + generated tokens (the K / V caches' length)")
  ap.add_argument("--pin-gb", type=float, default=float(os.environ.get("QWEN_PIN_GB", "24")))
  ap.add_argument("--dir", default=os.environ.get("QWEN_DIR", "/mnt/ssd/qwen3.8-27b-fp8")); ap.add_argument("--cache", default=os.environ.get("QWEN_NPU", "/mnt/ssd/qwen3.8-27b-npu-s1"))
  a = ap.parse_args(); eng = Engine(a)
  srv = ThreadingHTTPServer((a.host, a.port), make_handler(eng, a.model_name))
  print(f"== serving {a.model_name} on http://{a.host}:{a.port}/v1 (OpenAI-compatible; greedy; context {a.tmax} tokens)", flush=True)
  threading.Thread(target=srv.serve_forever, daemon=True).start()
  try: eng.serve_jobs()
  except KeyboardInterrupt: pass

if __name__ == "__main__": main()

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
import argparse, json, os, queue, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
os.environ.setdefault("QWEN_PREFILL", "chunked")
os.environ.setdefault("QWEN_SPEC", "6")                   # the server runs the verify path: speculative decoding up to 6 rows unless set
                                                          # (bonsai2_serve / ornith set their own defaults before this import)
import numpy as np                                                        # noqa: E402
from qwen38_tokenize import Tok, chat_messages, image_parts, parse_tool_calls, IMAGE_PAD, EOS   # noqa: E402
VISION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bonsai2-27b", "vision")

class Stop(Exception): pass

OLLAMA_DETAILS = {"format": "npu", "family": "qwen35", "families": ["qwen35"], "parameter_size": "27B", "quantization_level": "ternary"}

class PrefixCache:
  """What the model's K / V caches hold and where it can resume: `ids` = the tokens whose rows are in the cache (positions 0..), and
  snapshots of the recurrent state (snapshot_state) at a few of those positions. A new prompt resumes from the largest snapshot
  within its common prefix with `ids` -- Hermes-style agents resend ~16k tokens of system prompt and tools every turn, and a turn
  that continues a conversation shares everything up to the previous answer. Snapshots: after the first message (the system turn),
  at the prompt's end, at the generation's end; QWEN_PREFIX_SNAPS (6) kept, the earliest position always."""
  def __init__(self, M, keep=int(os.environ.get("QWEN_PREFIX_SNAPS", "6"))): self.M, self.keep, self.ids, self.snaps, self.order = M, keep, [], {}, []
  def invalidate(self): self.ids, self.snaps, self.order = [], {}, []
  def shared(self, ids):
    n = min(len(ids), len(self.ids)); k = 0
    while k < n and ids[k] == self.ids[k]: k += 1
    return k
  def resume_at(self, ids):
    """The position a prompt could resume from here: the largest snapshot inside its shared prefix (0: none)."""
    k = self.shared(ids); return max((q for q in self.snaps if q <= min(k, len(ids) - 1)), default=0)
  def plan(self, ids):
    """-> the position to resume the prompt from (the state restored; 0 = from scratch). Snapshots past the shared prefix go."""
    k = self.shared(ids); p = self.resume_at(ids)
    for q in [q for q in self.snaps if q > k]: self.drop(q)
    if p > 0: self.M.restore_state(self.snaps[p])
    return p
  def add(self, pos, snap):
    if pos in self.snaps: self.order.remove(pos)
    self.snaps[pos] = snap; self.order.append(pos)
    while len(self.snaps) > self.keep:
      lo = min(self.snaps); victim = next(q for q in self.order if q != lo); self.drop(victim)
  def drop(self, pos): self.snaps.pop(pos, None); self.order.remove(pos)
  def commit(self, ids, snap=None):
    """After a request: the cache holds `ids` (the prompt and the fed generated tokens); a snapshot at its end (`snap`: the state
    from elsewhere -- a batched slot's; default the model's own)."""
    self.ids = list(ids)
    try: self.add(len(ids), snap if snap is not None else self.M.snapshot_state())
    except Exception as e: print(f"   prefix cache: no end snapshot ({e!r})", flush=True)
  @staticmethod
  def boundaries(ids, im_end):
    """Message boundaries: the position after each <|im_end|> + newline pair (the first one: after the system turn)."""
    return [i + 2 for i in range(len(ids) - 1) if ids[i] == im_end]


class Engine:
  """The model and its one-request-at-a-time generation (`run` yields text deltas)."""
  def __init__(self, a):
    import qwen38_generate as G                                           # (imported here: gemma4_serve uses this module for its HTTP side only)
    self.cpus, self.lat = G.pin_cpus(), G.hold_cpu_latency()
    self.tok = Tok(os.environ.get("QWEN_TOK", a.dir)); self.tmax, self.lock, self.jobs, self.dir = a.tmax, threading.Lock(), queue.Queue(), a.dir
    t0 = time.perf_counter(); self.M = G.Model(12, a.cache, a.dir, a.tmax, a.pin_gb)
    assert self.M.M, "the server runs the verify path (QWEN_SPEC >= 2 geometries)"
    print(f"   model set up in {time.perf_counter() - t0:.0f} s; host CPUs {self.cpus}; CPU latency {self.lat}", flush=True)
    # the prefix cache: a region a prefix (QWEN_PREFIX_REGIONS, 2: an agent's side requests -- titles, summaries -- land in the other
    # region and leave the conversation's 16k-token prefix in place); the regions are the model's KV users (QWEN_KV_USERS, set by main)
    self.regions = self.M.kvs.users if self.M.kvs else 1
    self.pfx = [PrefixCache(self.M) for _ in range(self.regions)] if os.environ.get("QWEN_PREFIX_CACHE", "1") == "1" else None; self.im_end = self.tok.encode("<|im_end|>")[0]
  def submit(self, ids, max_new, stop, vis=None):
    """Queue a request (from any thread) -> a queue of ("delta", text) ... then ("done", tokens, finish) or ("error", message).
    `vis`: the prompt's images, [(embeddings, nx, ny)] in order, placed at the ids' <|image_pad|> runs."""
    q = queue.Queue(); self.jobs.put((ids, max_new, stop, q, vis)); return q
  def prompt(self, messages, think, tools=None):
    """An OpenAI / Ollama message list (+ OpenAI tool definitions) -> (ids, vis): the images decoded (the tower runs on the engine's
    thread), the template with their token counts."""
    parts = image_parts(messages)
    if not parts: return self.tok.encode(chat_messages(messages, think, tools=tools)), None
    imgs = self.decode_images(parts); pad = self.tok.encode(IMAGE_PAD)[0]
    ids = self.tok.encode(chat_messages(messages, think, image_tokens=[nx * ny for x, nx, ny in imgs], tools=tools))
    starts = [i for i in range(len(ids)) if ids[i] == pad and (i == 0 or ids[i - 1] != pad)]
    assert len(starts) == len(imgs), "image placeholder runs vs images"
    return ids, [(s, x, nx, ny) for s, (x, nx, ny) in zip(starts, imgs)]
  def serve_jobs(self):
    """The main thread's loop: one request at a time, on the thread that built the model."""
    while True:
      ids, max_new, stop, q, vis = self.jobs.get()
      try:
        t0 = time.perf_counter(); r = self.run(ids, max_new, stop, lambda d: q.put(("delta", d)), vis=self.embed_images(vis)); t1 = time.perf_counter()
        tf = self.t_first or t1; q.put(("done",) + r + (tf - t0, t1 - tf))      # + the prompt's seconds, the generation's
      except ValueError as e: q.put(("error", str(e)))
      except Exception as e: q.put(("error", f"{type(e).__name__}: {e}")); print(f"   request failed: {e!r}", flush=True)
  def vision(self):
    """The size's vision tower on the NPU (bonsai2-27b/vision), loaded on the first image; the mmproj (BONSAI_MMPROJ, else the model
    directory's *mmproj*Q8_0.gguf) packed into <cache>/vision on first use. Called on the engine's thread only."""
    if getattr(self, "_vision", None) is None:
      import glob; sys.path.insert(0, VISION_DIR); import vit_npu
      cache = os.path.join(self.M.cache, "vision"); d = self.dir if os.path.isdir(self.dir) else os.path.dirname(self.dir)   # Bonsai: the GGUF's directory
      if not os.path.exists(os.path.join(cache, "vision_small.npz")):
        mm = os.environ.get("BONSAI_MMPROJ") or next(iter(sorted(glob.glob(os.path.join(d, "*mmproj*Q8_0.gguf"))) + sorted(glob.glob(os.path.join(d, "*mmproj*.gguf")))), None)
        if not mm: raise ValueError(f"no mmproj GGUF in {d} (set BONSAI_MMPROJ)")
        print(f"   packing the vision tower from {mm} ...", flush=True); vit_npu.pack_vision(mm, cache)
      self._vision = vit_npu.VisionNPU(cache)
    return self._vision
  def decode_images(self, parts, max_tokens=1024):
    """The request's images -> [(pixels [H, W, 3] in [-1, 1], nx, ny)]: bicubic smart resize to multiples of 32, at most `max_tokens`
    tokens (32 x 32 pixel blocks) each. Runs on the handler's thread; the tower runs later, on the engine's."""
    import base64, io, urllib.request; from PIL import Image; sys.path.insert(0, VISION_DIR); import vit_ref as VR
    out = []
    for kind, v in parts:
      if kind == "b64" or (kind == "url" and v.startswith("data:")): data = base64.b64decode(v.split(",", 1)[1] if v.startswith("data:") else v)
      elif kind == "url": data = urllib.request.urlopen(v, timeout=30).read()
      else: data = v
      im = Image.open(io.BytesIO(data)).convert("RGB"); wb, hb = VR.smart_size(im.width, im.height, max_tokens=max_tokens)
      x = (np.asarray(im.resize((wb, hb), Image.BICUBIC), np.float32) / 255.0 - 0.5) / 0.5
      out.append((x, wb // (VR.PATCH * VR.MERGE), hb // (VR.PATCH * VR.MERGE)))
    return out
  def embed_images(self, vis):
    """[(start, pixels, nx, ny)] -> [(start, embeddings, nx, ny)] through the tower (the engine's thread)."""
    if not vis: return None
    out = []
    for s_, x, nx, ny in vis:
      t0 = time.perf_counter(); e, (nx_, ny_) = self.vision().forward(x, x.shape[1] // 16, x.shape[0] // 16); assert (nx_, ny_) == (nx, ny)
      print(f"   image {x.shape[1]}x{x.shape[0]} -> {len(e)} tokens ({nx}x{ny}) in {time.perf_counter() - t0:.2f} s", flush=True); out.append((s_, e, nx, ny))
    return out
  def run(self, ids, max_new, stop, on_delta, vis=None):
    """Greedy generation after `ids`; on_delta(text) per accepted chunk; `vis`: the prompt's images (set_vision). Returns (completion tokens, finish_reason)."""
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
      M.spec_meta = dict(prompt="(server)", n_prompt=len(ids), thinking=False, prompt_ids=[int(v) for v in ids]); pfx = None
      if self.pfx:                                                        # the region: the furthest resume point, else the one holding the least
        sh = [c.resume_at(ids) for c in self.pfx]; u = max(range(self.regions), key=lambda r: (sh[r], -len(self.pfx[r].ids)))
        M.kv_user = u; pfx = self.pfx[u]
        if vis: pfx.invalidate(); pfx = None                              # (image rows in the cache: not a token prefix)
      p = pfx.plan(ids) if pfx else 0; t0 = time.perf_counter()
      if p == 0: M.reset()
      b = PrefixCache.boundaries(ids, self.im_end); snap_at = ([b[0]] if b else []) + [len(ids)] if pfx else ()
      try: lg = M.prefill(ids, vis=vis, start=p, snap_at=snap_at, on_snap=pfx.add if pfx else None)
      except Exception:
        if pfx: pfx.invalidate()
        raise
      first = int(np.argmax(lg))
      if pfx: print(f"   prompt {len(ids)} tokens, {len(ids) - p} prefilled ({p} from the prefix cache, region {M.kv_user}; the regions hold {[len(c.ids) for c in self.pfx]}) in {time.perf_counter() - t0:.1f} s", flush=True)
      try:
        emit([first])
        if first not in EOS and max_new > 1:
          if M.MTPL: M.spec_generate(first, len(ids), max_new, "mtp", {}, lambda new, tv: emit(new))
          else:
            for i in range(1, max_new):
              lg = M.step(out[-1], len(ids) + i - 1); emit([int(np.argmax(lg))])
      except Stop: pass
      if pfx:                                                             # the rows written: the prompt and every generated token fed back
        try: pfx.commit(list(ids) + [int(t) for t in out[:-1]])
        except Exception as e: print(f"   prefix cache: commit failed ({e!r})", flush=True); pfx.invalidate()
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
        ids, vis = eng.prompt(req["messages"], think)
      else:
        p = req.get("prompt", ""); msgs = ([{"role": "system", "content": req["system"]}] if req.get("system") else []) + [{"role": "user", "content": p, "images": req.get("images") or []}]
        ids, vis = (eng.tok.encode(p), None) if req.get("raw") else eng.prompt(msgs, think)
      n0 = int(opts.get("num_predict") or 512); max_new = 512 if n0 < 0 else n0
      stop = opts.get("stop") or []; stop = [stop] if isinstance(stop, str) else list(stop)
      t0 = time.perf_counter(); q = eng.submit(ids, max_new, stop, vis); chat = route == "/api/chat"
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
      tools = req.get("tools") if chat and req.get("tool_choice") != "none" else None
      if os.environ.get("QWEN_SERVE_DUMP"):                               # every request's body appended as one JSON line (what agents send)
        with open(os.environ["QWEN_SERVE_DUMP"], "a") as f: f.write(json.dumps({"t": time.time(), "route": route, "body": req}) + "\n")
      if chat:
        if not req.get("messages"): return self._err(400, "messages is required")
        ids, vis = eng.prompt(req["messages"], think, tools=tools or None)
      else:
        text = req.get("prompt", ""); text = text[0] if isinstance(text, list) else text; ids, vis = eng.tok.encode(text), None
      max_new = int(req.get("max_completion_tokens") or req.get("max_tokens") or 512)
      def tool_calls_of(text):
        """The completion's <tool_call> blocks as OpenAI tool_calls (with tools in the request) -> (content, tool_calls or None)."""
        if not tools: return text, None
        content, calls = parse_tool_calls(text)
        return content, ([{"index": i, "id": "call_" + uuid.uuid4().hex[:24], "type": "function", "function": c} for i, c in enumerate(calls)] or None)
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
        q = eng.submit(ids, max_new, stop, vis); acc = ""; sent = 0; TAG = "<tool_call>"
        def nextmsg():
          while True:
            try: return q.get(timeout=20)
            except queue.Empty: sse(dict(base, choices=[choice("", None, delta=True)]))   # keep-alive during a long prefill
        try:
          while (m := nextmsg())[0] == "delta":
            if not tools: sse(dict(base, choices=[choice(m[1], None, delta=True)])); continue
            acc += m[1]                                                   # with tools: stream the text before a <tool_call>, hold back a tag's possible start
            if TAG in acc: cut = acc.index(TAG)
            else: cut = len(acc) - max((k for k in range(1, len(TAG)) if acc.endswith(TAG[:k])), default=0)
            if cut > sent: sse(dict(base, choices=[choice(acc[sent:cut], None, delta=True)])); sent = cut
          if m[0] == "error": sse({"error": {"message": m[1], "type": "invalid_request_error"}})
          else:
            n, finish = m[1], m[2]
            if tools:
              content, calls = tool_calls_of(acc)
              if calls: sse(dict(base, choices=[{"index": 0, "delta": {"tool_calls": calls}, "finish_reason": None}])); finish = "tool_calls"
              elif len(acc) > sent: sse(dict(base, choices=[choice(acc[sent:], None, delta=True)]))
            last = dict(base, choices=[choice("", finish, delta=True)])
            if (req.get("stream_options") or {}).get("include_usage"): last["usage"] = {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}
            sse(last); sse("[DONE]")
        except (BrokenPipeError, ConnectionResetError): return     # the client left; its request still runs to its end
        self.wfile.write(b"0\r\n\r\n"); self.wfile.flush(); return
      parts, q = [], eng.submit(ids, max_new, stop, vis)
      while (m := q.get())[0] == "delta": parts.append(m[1])
      if m[0] == "error": return self._err(400, m[1])
      n, finish = m[1], m[2]; content, calls = tool_calls_of("".join(parts)); ch = choice(content, "tool_calls" if calls else finish)
      if calls: ch["message"]["tool_calls"] = calls; ch["message"]["content"] = content or None
      self._json(200, {"id": rid, "object": obj, "created": created, "model": name, "choices": [ch],
                       "usage": {"prompt_tokens": len(ids), "completion_tokens": n, "total_tokens": len(ids) + n}})
  return H


def kv_window_bytes(users, natt, nkv, hd, tmax, slots, mtp=True):
  """The K / V caches' bytes in the NPU's window: as buffers (1 + users) x natt layers x 2 x nkv hd floats a position; with KV slots
  two slots of max(1, users) users' [K | V] (+ the drafter's own slot when `mtp`)."""
  row = 2 * nkv * hd * 4
  return (2 * max(1, users) + (1 if mtp else 0)) * row * tmax if slots else (1 + users) * natt * row * tmax


def asid_count():
  """How many of ASIDs 1..3 allocate right now (qwen38_generate.KVAsid.available on the device)."""
  from tinygrad import Device; import qwen38_generate as G_
  return G_.KVAsid.available(Device["ZHOUYI"].raw)


def asid_tmax(users, natt, nkv, hd, mtp=True, want=65536, nasid=None):
  """The caches in ASIDs 1-3 (QWEN_KV_ASID): the largest multiple of 1024 up to `want` whose fullest ASID (ceil(natt / nasid) layers x
  max(1, users) regions) holds its [K | V] in 3 GiB -> (tmax, that ASID's GB); the drafter's cache takes a buffer of its own where
  there is room. nasid: the ASIDs that allocate (asid_count; the stock KMD: 1). QWEN_KV_DTYPE: f32 | f16 | auto (default) -- auto
  takes fp32 when `want` fits in it and fp16 otherwise (fp16 doubles the positions and costs ~10-20 % of the step: the part's fp16
  FMAs issue slower than its fp32 ones); the choice is written back to QWEN_KV_DTYPE for the model."""
  na = asid_count() if nasid is None else nasid; dt = os.environ.get("QWEN_KV_DTYPE", "auto")
  regs = max(-(-natt // na) * max(1, users), 1)
  def fit(es):
    row = nkv * hd * es; cap = 3 * 2**30 - (16 << 20) - 2 * 4 * row                              # (the drafter's region has a buffer of its own; the ATT_BT margin rows)
    tm = min(want, cap // (2 * regs * row) // 1024 * 1024); return max(1024, tm), 2 * regs * row * tm / 1e9
  if dt == "auto": dt = "f32" if fit(4)[0] >= want else "f16"
  os.environ["QWEN_KV_DTYPE"] = dt
  return fit(2 if dt == "f16" else 4)


def auto_tmax(users, budget_gb, natt, nkv, hd, want=65536, slots=False, mtp=True):
  """The context: `want`, or the largest multiple of 1024 whose K / V caches (kv_window_bytes) fit budget_gb -- the NPU addresses a
  4 GiB window of which ~2.75 GB can hold buffers and slots (the rest of the process's buffers take ~0.3-0.6 GB). With the caches
  in ASIDs 1-3 (QWEN_KV_ASID=1) the window is not the bound: asid_tmax's."""
  if os.environ.get("QWEN_KV_ASID", "0") == "1": return asid_tmax(users, natt, nkv, hd, mtp, want)
  per_pos = kv_window_bytes(users, natt, nkv, hd, 1, slots, mtp); t = min(want, int(budget_gb * 1e9) // per_pos // 1024 * 1024)
  return max(1024, t), per_pos * t / 1e9


def engine_fit(cls, a, auto):
  """cls(a); when the NPU-visible memory runs out under `auto` (the model's own buffers vary by size: the drafter, the head parts,
  the window slots), the process restarts itself with the context shortened by a quarter (QWEN_TMAX; down to 4096) -- a fresh
  process, since a half-built model leaves buffers behind."""
  import sys
  try: return cls(a)
  except MemoryError as e:
    if not auto or a.tmax <= 4096: raise
    t = max(4096, a.tmax * 3 // 4 // 1024 * 1024); print(f"   NPU memory ran out ({str(e)[:60]}): restarting with a context of {t} tokens", flush=True)
    argv = [v for i, v in enumerate(sys.argv) if v != "--tmax" and not (i and sys.argv[i - 1] == "--tmax")]
    os.environ["QWEN_TMAX"] = str(t); os.environ["QWEN_TMAX_AUTO"] = "1"; os.execv(sys.executable, [sys.executable] + argv)


def main(default_name="qwen3.8-27b", default_tmax=2048):
  ap = argparse.ArgumentParser(); ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
  ap.add_argument("--model-name", default=os.environ.get("QWEN_SERVE_NAME", default_name))
  ap.add_argument("--tmax", default=os.environ.get("QWEN_TMAX", str(default_tmax)), help="the context: prompt + generated tokens (the K / V caches' length); `auto`: the largest up to 65536 whose caches fit QWEN_KV_GB (2.0) of NPU-visible memory")
  ap.add_argument("--pin-gb", type=float, default=float(os.environ.get("QWEN_PIN_GB", "24")))
  ap.add_argument("--dir", default=os.environ.get("QWEN_DIR", "/mnt/ssd/qwen3.8-27b-fp8")); ap.add_argument("--cache", default=os.environ.get("QWEN_NPU", "/mnt/ssd/qwen3.8-27b-npu-s1"))
  a = ap.parse_args()
  import qwen38_ref as R_
  auto = a.tmax == "auto" or os.environ.get("QWEN_TMAX_AUTO") == "1"
  asid = os.environ.get("QWEN_KV_ASID", "0") == "1"
  nreg = int(os.environ.get("QWEN_PREFIX_REGIONS", "2")) if asid and os.environ.get("QWEN_PREFIX_CACHE", "1") == "1" else 1   # the prefix cache's regions = KV users
  os.environ["QWEN_KV_USERS"] = str(nreg); ureg = nreg if nreg > 1 else 0
  a.tmax, gb = auto_tmax(ureg, float(os.environ.get("QWEN_KV_GB", "2.0")), R_.NL // 4, R_.NKV, R_.HD, want=int(os.environ.get("QWEN_TMAX_MAX", "262144"))) if a.tmax == "auto" else (int(a.tmax), (asid_tmax(ureg, R_.NL // 4, R_.NKV, R_.HD, True, int(a.tmax))[1] if asid else kv_window_bytes(0, R_.NL // 4, R_.NKV, R_.HD, int(a.tmax), False) / 1e9))
  print(f"   context {a.tmax} tokens: K / V caches {gb:.2f} GB {'in the fullest of ASIDs 1-3 (3 GiB each; none of the ASID0 window; ' + os.environ.get('QWEN_KV_DTYPE', 'f16') + ')' if asid else 'of the NPU window'}", flush=True); eng = engine_fit(Engine, a, auto)
  srv = ThreadingHTTPServer((a.host, a.port), make_handler(eng, a.model_name))
  print(f"== serving {a.model_name} on http://{a.host}:{a.port}/v1 (OpenAI-compatible; greedy; context {a.tmax} tokens)", flush=True)
  threading.Thread(target=srv.serve_forever, daemon=True).start()
  try: eng.serve_jobs()
  except KeyboardInterrupt: pass

if __name__ == "__main__": main()

#!/usr/bin/env python3
"""Gemma 4's text -> token ids and back, from the GGUF's own vocabulary (no other files): `tokenizer.ggml.model` is "gemma4", a
BPE over raw UTF-8 with SentencePiece-style spaces (each " " is "▁" before the merges), the merges in rank order
(`tokenizer.ggml.merges`), "<0xXX>" byte tokens for any piece not in the vocabulary, and no space prefix. The text is split
at newline runs only (a run of newlines that is itself a token stays one token); control and user-defined tokens written in
the text (`<bos>`, `<|turn>`, `<turn|>`, ...) are taken as themselves. And the chat template's text-only form (the GGUF's
`tokenizer.chat_template`, thinking off): <bos>[<|turn>system\\n...<turn|>\\n]<|turn>user\\n...<turn|>\\n ... <|turn>model\\n.

    python3 gemma4_tokenize.py "prompt" [--gguf /mnt/ssd/models/gemma-4/gemma-4-E2B-it-Q8_0.gguf]   (the chat prompt's ids)
"""
import argparse, heapq, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "ornith-9b"))

BOS, EOS, TURN_END = 2, 1, 106          # <bos>; <eos> and <turn|> end a reply
STOP = (EOS, TURN_END)
NORMAL, CONTROL, USER, BYTE = 1, 3, 4, 6  # tokenizer.ggml.token_type

def chat_messages(messages, thinking: bool = False) -> str:
  """The chat template's rendering of an OpenAI-style message list ([{"role": system | developer | user | assistant, "content":
  str or a list of {"type": "text", "text": ...} parts}]) + the generation prompt. A leading system message becomes the system
  turn; assistant turns are "model" turns without their thought channel. Thinking is not supported (it is off)."""
  def text(c): return c if isinstance(c, str) else "".join(p.get("text", "") for p in (c or []) if isinstance(p, dict))
  def strip_thought(c):                                                  # the template's strip_thinking
    return "".join(p.split("<|channel>")[0] for p in c.split("<channel|>")).strip()
  s, ms = "<bos>", list(messages)
  if ms and ms[0].get("role") in ("system", "developer"): s += f"<|turn>system\n{text(ms.pop(0).get('content')).strip()}<turn|>\n"
  for m in ms:
    role = "model" if m.get("role") == "assistant" else m.get("role", "user")
    c = text(m.get("content", "")); c = strip_thought(c) if role == "model" else c.strip()
    s += f"<|turn>{role}\n{c}<turn|>\n"
  return s + "<|turn>model\n"

class Tok:
  """encode(text) -> ids (no <bos> added: the chat text carries it), decode(ids) -> text (control tokens dropped)."""
  def __init__(self, meta):
    self.pieces = meta["tokenizer.ggml.tokens"]; self.types = meta["tokenizer.ggml.token_type"]
    self.ids = {p: i for i, p in enumerate(self.pieces)}
    self.ranks = {}
    for i, m in enumerate(meta["tokenizer.ggml.merges"]):
      j = m.find(" ", 1)
      if j > 0: self.ranks[(m[:j], m[j + 1:])] = i
    sp = sorted((p for p, t in zip(self.pieces, self.types) if t in (CONTROL, USER) and len(p) > 1), key=len, reverse=True)
    self.special = re.compile("(" + "|".join(re.escape(p) for p in sp) + ")")
  @classmethod
  def from_gguf(cls, path):
    from gguf_read import GGUF
    return cls(GGUF(os.path.expanduser(path)).meta)

  def _bpe(self, word):
    """One newline-free (or all-newline) piece -> ids: the merges by rank over its characters, then bytes for what is left over."""
    if word.strip("\n") == "" and word in self.ids: return [self.ids[word]]
    sym = list(word); nxt = list(range(1, len(sym))) + [-1]; prv = list(range(-1, len(sym) - 1)); heap = []
    def push(a):
      b = nxt[a] if a >= 0 else -1
      if a >= 0 and b >= 0 and (r := self.ranks.get((sym[a], sym[b]))) is not None: heapq.heappush(heap, (r, a, sym[a] + sym[b]))
    for a in range(len(sym) - 1): push(a)
    while heap:
      _, a, text = heapq.heappop(heap); b = nxt[a]
      if b < 0 or not sym[a] or sym[a] + sym[b] != text: continue             # outdated
      sym[a] = text; sym[b] = ""; nxt[a] = nxt[b]
      if nxt[b] >= 0: prv[nxt[b]] = a
      push(prv[a]); push(a)
    out = []
    for s in sym:
      if not s: continue
      if s in self.ids: out.append(self.ids[s])
      else: out.extend(self.ids[f"<0x{c:02X}>"] for c in s.encode("utf-8") if f"<0x{c:02X}>" in self.ids)
    return out
  def encode(self, text: str) -> list[int]:
    out = []
    for frag in self.special.split(text):
      if not frag: continue
      if frag in self.ids and self.types[self.ids[frag]] in (CONTROL, USER): out.append(self.ids[frag]); continue
      for w in re.findall(r"[^\n]+|\n+", frag.replace(" ", "▁")): out.extend(self._bpe(w))
    return out
  def decode(self, ids) -> str:
    b = bytearray()
    for i in ids:
      t, p = self.types[i], self.pieces[i]
      if t == BYTE: b.append(int(p[3:5], 16))
      elif t == NORMAL: b += p.replace("▁", " ").encode("utf-8")
      elif t == USER: b += p.encode("utf-8")
    return b.decode("utf-8", "replace")

if __name__ == "__main__":
  ap = argparse.ArgumentParser(); ap.add_argument("prompt")
  ap.add_argument("--gguf", default=os.environ.get("GEMMA_GGUF", "/mnt/ssd/models/gemma-4/gemma-4-E2B-it-Q8_0.gguf")); a = ap.parse_args()
  t = Tok.from_gguf(a.gguf); text = chat_messages([{"role": "user", "content": a.prompt}]); ids = t.encode(text)
  print(repr(text)); print(len(ids), "tokens:", " ".join(map(str, ids)))

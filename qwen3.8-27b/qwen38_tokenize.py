#!/usr/bin/env python3
"""Qwen3.8's prompt -> token ids and back, with the checkpoint's `tokenizer.json` (byte-level BPE, 248320 ids) through the
`tokenizers` library, and the chat template's single-turn form written out (chat_template.jinja: one user message, the
generation prompt; thinking on or off).

    python3 qwen38_tokenize.py "prompt" [--thinking] [--dir /mnt/ssd/qwen3.8-27b-fp8]
"""
import argparse, os
from tokenizers import Tokenizer

IM_START, IM_END, EOT = 248045, 248046, 248044          # <|im_start|>, <|im_end|>, <|endoftext|>: generation stops on either of the last two
EOS = (IM_END, EOT)

def chat_text(prompt: str, thinking: bool = False, system: str | None = None) -> str:
  """The template's rendering of [system?, user] + add_generation_prompt."""
  s = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
  s += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
  return s + ("<think>\n" if thinking else "<think>\n\n</think>\n\n")

VISION_START, VISION_END, IMAGE_PAD = "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"


TOOLS_HEAD = ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures "
              "within <tools></tools> XML tags:\n<tools>\n")
TOOLS_TAIL = ("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> "
              "XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")


def chat_messages(messages, thinking: bool = False, image_tokens=None, tools=None) -> str:
  """The template's rendering of an OpenAI-style message list ([{"role": system | user | assistant, "content": str or a list of
  {"type": "text", "text": ...} / {"type": "image_url", ...} parts}]) + add_generation_prompt: earlier assistant turns keep their
  text without any <think> ... </think> block (as the template drops the reasoning of past turns). `image_tokens`: the token count
  of each image part in order of appearance (the vision tower's grid, nx x ny): the part renders as <|vision_start|> + that many
  <|image_pad|> + <|vision_end|>, where the model's embedding rows are replaced by the image's. `tools`: OpenAI function definitions,
  rendered Qwen's way (the "# Tools" section with the JSON signatures in the system turn, the model's calls as <tool_call> JSON, an
  assistant turn's tool_calls re-rendered so, "tool" results as <tool_response> blocks in a user turn)."""
  import json
  it = iter(image_tokens or [])
  def text(c):
    if isinstance(c, str): return c
    out = []
    for p in (c or []):
      if not isinstance(p, dict): continue
      if p.get("type") in ("image_url", "image", "input_image"): out.append(VISION_START + IMAGE_PAD * next(it) + VISION_END)
      else: out.append(p.get("text", ""))
    return "".join(out)
  s = ""; messages = list(messages)
  if tools:                                                                # the tools section goes into the (first) system turn
    sig = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    if messages and messages[0].get("role") in ("system", "developer"):
      sysc = text(messages[0].get("content", "")); messages = messages[1:]; s += f"<|im_start|>system\n{sysc}\n\n{TOOLS_HEAD}{sig}{TOOLS_TAIL}<|im_end|>\n"
    else: s += f"<|im_start|>system\n{TOOLS_HEAD}{sig}{TOOLS_TAIL}<|im_end|>\n"
  i = 0
  while i < len(messages):
    m = messages[i]; role, c = m.get("role", "user"), text(m.get("content", "")); i += 1
    if role == "tool":                                                     # consecutive tool results: one user turn of <tool_response> blocks
      blocks = [c]
      while i < len(messages) and messages[i].get("role") == "tool": blocks.append(text(messages[i].get("content", ""))); i += 1
      s += "<|im_start|>user\n" + "\n".join(f"<tool_response>\n{b}\n</tool_response>" for b in blocks) + "<|im_end|>\n"; continue
    if role == "user" and m.get("images"):                                 # Ollama's per-message base64 images: before the text
      c = "".join(VISION_START + IMAGE_PAD * next(it) + VISION_END for _ in m["images"]) + c
    if role == "assistant" and "</think>" in c: c = c.split("</think>", 1)[1].lstrip("\n")
    if role == "assistant" and not thinking: c = "<think>\n\n</think>\n\n" + c   # as the model generated it (the generation prompt's empty block): the prefix cache continues a conversation from its last answer
    if role == "assistant" and m.get("tool_calls"):
      calls = []
      for tc in m["tool_calls"]:
        f = tc.get("function", tc); a = f.get("arguments", {})
        if isinstance(a, str):
          try: a = json.loads(a)
          except ValueError: pass
        calls.append("<tool_call>\n" + json.dumps({"name": f.get("name", ""), "arguments": a}, ensure_ascii=False) + "\n</tool_call>")
      c = (c + "\n" if c else "") + "\n".join(calls)
    if role == "developer": role = "system"
    s += f"<|im_start|>{role}\n{c}<|im_end|>\n"
  s += "<|im_start|>assistant\n"
  return s + ("<think>\n" if thinking else "<think>\n\n</think>\n\n")


def parse_tool_calls(text: str):
  """A completion's <tool_call> JSON blocks -> (the text before the first call, [{"name", "arguments": JSON string}]). A block whose JSON
  does not parse stays in the text."""
  import json, re
  calls = []; out = text
  for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
    try: o = json.loads(m.group(1))
    except ValueError: continue
    if not isinstance(o, dict) or "name" not in o: continue
    a = o.get("arguments", {}); calls.append({"name": str(o["name"]), "arguments": a if isinstance(a, str) else json.dumps(a, ensure_ascii=False)})
  if calls: out = text[:text.index("<tool_call>")]
  return out.rstrip() if calls else text, calls


def image_parts(messages):
  """The images of an OpenAI / Ollama message list in order of appearance: a list of ("url", str) | ("b64", str) | ("bytes", bytes)."""
  out = []
  for m in messages:
    c = m.get("content", "")
    if m.get("role") == "user" and m.get("images"): out += [("b64", b) for b in m["images"]]
    if isinstance(c, list):
      for p in c:
        if not isinstance(p, dict): continue
        if p.get("type") == "image_url": u = p["image_url"]["url"] if isinstance(p.get("image_url"), dict) else p.get("image_url"); out.append(("url", u))
        elif p.get("type") in ("image", "input_image"): out.append(("url", p.get("image_url") or p.get("url") or p.get("source", {}).get("data", "")))
  return out


class Tok:
  def __init__(self, folder):
    self.tk = Tokenizer.from_file(os.path.join(os.path.expanduser(folder), "tokenizer.json"))
  def encode(self, text: str) -> list[int]: return self.tk.encode(text, add_special_tokens=False).ids
  def decode(self, ids) -> str: return self.tk.decode(list(ids), skip_special_tokens=False)

if __name__ == "__main__":
  ap = argparse.ArgumentParser(); ap.add_argument("prompt"); ap.add_argument("--thinking", action="store_true")
  ap.add_argument("--dir", default=os.environ.get("QWEN_DIR", "/mnt/ssd/qwen3.8-27b-fp8")); a = ap.parse_args()
  t = Tok(a.dir); text = chat_text(a.prompt, a.thinking); ids = t.encode(text)
  print(repr(text)); print(len(ids), "tokens:", ids); print("round trip ok:", t.decode(ids) == text)

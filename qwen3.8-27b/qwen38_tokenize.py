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


def chat_messages(messages, thinking: bool = False, image_tokens=None) -> str:
  """The template's rendering of an OpenAI-style message list ([{"role": system | user | assistant, "content": str or a list of
  {"type": "text", "text": ...} / {"type": "image_url", ...} parts}]) + add_generation_prompt: earlier assistant turns keep their
  text without any <think> ... </think> block (as the template drops the reasoning of past turns). `image_tokens`: the token count
  of each image part in order of appearance (the vision tower's grid, nx x ny): the part renders as <|vision_start|> + that many
  <|image_pad|> + <|vision_end|>, where the model's embedding rows are replaced by the image's."""
  it = iter(image_tokens or [])
  def text(c):
    if isinstance(c, str): return c
    out = []
    for p in (c or []):
      if not isinstance(p, dict): continue
      if p.get("type") in ("image_url", "image", "input_image"): out.append(VISION_START + IMAGE_PAD * next(it) + VISION_END)
      else: out.append(p.get("text", ""))
    return "".join(out)
  s = ""
  for m in messages:
    role, c = m.get("role", "user"), text(m.get("content", ""))
    if role == "user" and m.get("images"):                                 # Ollama's per-message base64 images: before the text
      c = "".join(VISION_START + IMAGE_PAD * next(it) + VISION_END for _ in m["images"]) + c
    if role == "assistant" and "</think>" in c: c = c.split("</think>", 1)[1].lstrip("\n")
    if role == "developer": role = "system"
    s += f"<|im_start|>{role}\n{c}<|im_end|>\n"
  s += "<|im_start|>assistant\n"
  return s + ("<think>\n" if thinking else "<think>\n\n</think>\n\n")


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

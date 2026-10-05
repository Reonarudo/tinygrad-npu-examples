"""Speculative decoding's verify / commit path against plain greedy decoding: the same prompt decoded N tokens plainly, then
(prefill again: every state rebuilt) with an oracle drafter (the plain run's own tokens: every draft accepted) and a bad one
(the current token repeated: mostly rejected, the rollback exercised). The ids must match.
  QWEN_SPEC=4 python3 spec_check.py [N]"""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tinygrad")))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from qwen38_generate import Model
from qwen38_tokenize import Tok, chat_text

N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
d_, c_ = os.environ.get("QWEN_DIR", "/mnt/ssd/qwen3.8-27b-fp8"), os.environ.get("QWEN_NPU", "/mnt/ssd/qwen3.8-27b-npu-s1")
tok = Tok(d_); ids = tok.encode(chat_text("What is the capital of Portugal? Answer in one sentence, then name two famous landmarks there.", False))
M = Model(len(ids), c_, d_, 512, float(os.environ.get("QWEN_PIN_GB", "24")))
def plain():
  lg = M.prefill(ids); out = [int(lg.argmax())]; t0 = time.perf_counter()
  while len(out) < N:
    lg = M.step(out[-1], len(ids) + len(out) - 1); out.append(int(lg.argmax()))
  return out, (time.perf_counter() - t0) / (N - 1)
base, tp = plain()
print(f"plain: {tok.decode(base)!r} | {tp:.2f} s / token", flush=True)
runs = [("oracle", lambda cur, pos, out: base[len(out):len(out) + M.M - 1]), ("bad", lambda cur, pos, out: [cur] * (M.M - 1))]
if M.MTPL: runs.append(("mtp", "mtp"))
for name, drafter in runs:
  lg = M.prefill(ids); first = int(lg.argmax()); st = {}; t0 = time.perf_counter()
  got = M.spec_generate(first, len(ids), N, drafter, st); tt = time.perf_counter() - t0
  same = got[:N] == base[:len(got[:N])]
  print(f"{name:6s}: ids {'IDENTICAL' if same else 'DIFFERENT'} ({len(got)} tokens) | {st['passes']} passes, {st['accepted']} drafts accepted | "
        f"{st['t_verify'] / st['passes']:.2f} s a verify pass" + (f", drafting {st['t_draft'] / st['passes']:.2f} s" if "t_draft" in st else "")
        + f" | {tt / max(1, len(got) - 1):.2f} s / token", flush=True)
  if not same: print("   got ", got[:N], "\n   base", base[:N])

"""Ornith 1.0 9B text generation on the NPU: qwen3.8-27b's generator with QWEN_MODEL=ornith-9b (the same Qwen3.5 layer kernels
at Ornith's sizes, the linears and the head on the Q8_0 GEMM, the prompt prefilled through the multi-token decode path).
  TG=... python3 ornith_generate.py "prompt" [--max-new N] [--thinking] [--out ids.npz]
Paths: ORNITH_GGUF (the checkpoint; the embedding is read from it), QWEN_NPU (the packed cache, ornith_pack.py), QWEN_TOK
(tokenizer.json's folder: Qwen3.5's tokenizer, identical to Ornith's vocabulary and merges; default the cache)."""
import os, sys
os.environ.setdefault("QWEN_MODEL", "ornith-9b")
os.environ.setdefault("QWEN_NPU", "/mnt/ssd/ornith-9b-npu")
os.environ.setdefault("QWEN_DIR", os.environ.get("ORNITH_GGUF", "/mnt/ssd/models/ornith-1.0-9b-Q8_0.gguf"))
os.environ.setdefault("QWEN_TOK", os.environ["QWEN_NPU"])
os.environ.setdefault("QWEN_PIN_GB", "12")
# speculative decoding: the leaf tree (spare rows take the drafts' rank-2/3 candidates) on the fast deferred-commit DeltaNet kernel,
# up to 8 rows and 5 drafts. Board, 3 prompts x 120 tokens, a backend with the Q8_0 GEMM's rows 5-8 rescheduled: 6.59 tok/s
# pooled vs 5.46 for the chain of <= 3 drafts (the previous default) on the same kernel, every run identical to plain greedy. On an
# older backend, whose rows 5+ cost ~1.35x, QWEN_SPEC=4 (the leaf tree at <= 4 rows: +6 % over the chain) is the better choice
# QWEN_SPEC=0 (plain decoding) keeps the chain's geometries: the prompt prefilled in 6-row chunks, as before the leaf default (the leaf
# tree's 4-row chunks cost ~40 % more prefill and move the prompt's rounding)
if os.environ.get("QWEN_SPEC") != "0":
  os.environ.setdefault("QWEN_SPEC_TREE", "leaf")
  os.environ.setdefault("QWEN_SPEC", "8")
  os.environ.setdefault("QWEN_DRAFT_MAX", "5")
  # the verify pass's layers 4 at a time, a block one job (and the head's 6 parts one job), every layer pinned zero-copy (12 GB):
  # 70 jobs a pass -> 10
  os.environ.setdefault("QWEN_LAYER_BLOCK", "4")
  os.environ.setdefault("ZHOUYI_CHAIN_MAX", "16")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qwen3.8-27b")); sys.path.insert(0, HERE)
import qwen38_generate                                                   # noqa: E402

if __name__ == "__main__": qwen38_generate.main()

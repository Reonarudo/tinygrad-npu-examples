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
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qwen3.8-27b")); sys.path.insert(0, HERE)
import qwen38_generate                                                   # noqa: E402

if __name__ == "__main__": qwen38_generate.main()

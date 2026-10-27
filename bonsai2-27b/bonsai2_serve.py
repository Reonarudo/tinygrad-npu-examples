#!/usr/bin/env python3
"""Ternary Bonsai 2 behind an OpenAI- (and Ollama-) compatible HTTP endpoint (qwen3.8-27b/qwen38_serve.py with bonsai2_generate's
defaults: the size's paths, its drafter and the decoding settings). The 27B by default; `QWEN_MODEL=bonsai2-9b | -4b | -2b` serves the
smaller sizes (speculative decoding with their own drafters: 2B ~30, 4B ~24, 9B ~14 tok/s on code). One request runs at a time;
others wait in line (bonsai2_batch_serve.py batches concurrent requests into one step). The context: `--tmax auto` by default on
the smaller sizes = the largest up to 65536 whose K / V caches fit the NPU's ASID 1 (3 GiB, reached by the DMA's base code; QWEN_KV_ASID=1):
65536 on the 2B, 43008 on the 4B / 9B; the attention kernels stream any length in chunks. 2048 on the 27B (its one-pass prefill).

  [QWEN_MODEL=bonsai2-2b] python3 bonsai2_serve.py [--host 0.0.0.0] [--port 8000] [--tmax 65536]
  curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \\
       -d '{"model": "bonsai2-27b", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
"""
import bonsai2_generate                                                  # noqa: F401  (sets the environment, imports qwen38_generate)
import qwen38_serve                                                      # noqa: E402

if __name__ == "__main__":
  model = __import__("os").environ.get("QWEN_MODEL", "bonsai2-27b"); qwen38_serve.main(model, 2048 if model == "bonsai2-27b" else "auto")

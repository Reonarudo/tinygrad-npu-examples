#!/usr/bin/env python3
"""Ternary Bonsai 2 27B behind an OpenAI-compatible HTTP endpoint (qwen3.8-27b/qwen38_serve.py with bonsai2_generate's defaults:
the same paths, the drafter and the decoding settings).

  python3 bonsai2_serve.py [--host 0.0.0.0] [--port 8000] [--tmax 2048]
  curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \\
       -d '{"model": "bonsai2-27b", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
"""
import bonsai2_generate                                                  # noqa: F401  (sets the environment, imports qwen38_generate)
import qwen38_serve                                                      # noqa: E402

if __name__ == "__main__": qwen38_serve.main("bonsai2-27b")

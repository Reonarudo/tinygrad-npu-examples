# gpt-oss-20b on the NPU

OpenAI's gpt-oss-20b (MXFP4 experts top 4 of 32 with a bias router and the clamped SwiGLU, Q8_0 attention with biases and sinks, sliding
windows of 128 on even layers, YaRN) from its GGUF: `gptoss_ref.py` (mirrors llama.cpp's graph), `gptoss_pack.py` (K 2880 padded to
2944 / 3072; `--experts q5n` repacks the MXFP4 experts **losslessly** into the 5.75-bit q5n stream -- their values are 5-bit integers x a
power of two; the 4.5-bit `mx4` kind is exact too but its FP4 decoder runs at 5 GB/s against q5n's 20; `check_q5pack.py`),
`gptoss_kernels.py` (the biases, the YaRN row, the gating, the weighted combine), `gptoss_npu.py --gate 0,1,23` or `"prompt"`,
`bench_gptoss.py` (each kernel timed). The attention is the library's sinks + window kernel (`seqmix/attn.py`); one TinyJit graph a layer,
the routing on the device ("topk_softmax"). Board o6n: gate worst 1.3e-4, llama.cpp's tokens (one near tie in 24 moved by the summation
order), **4.9 tok/s** decode (0.61 eager), 14.6 GB resident; the 201k-row Q8_0 head alone is 26 ms of the 205 ms token.

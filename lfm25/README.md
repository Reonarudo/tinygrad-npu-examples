# LFM2.5-24B-A2B on the NPU

Liquid's LFM2.5-24B-A2B (30 short-conv layers: in_proj -> b | c | g, u = b g, three taps over a 2-row state, out_proj(c v); 10 attention
layers 32 / 8 x 64 with q / k RMSNorm and full NeoX RoPE; 2 dense 11776-wide layers; 64 experts top 4 with a sigmoid router and a
selection bias; the tied Q6_K head) from its Q4_K_M GGUF, nothing requantised: `lfm25_pack.py` (q4m, q6n, the dense down as q8f for the
rows-mode GEMM), `lfm25_ref.py`, `lfm25_npu.py --gate 0,2,3,39` or `"prompt"`, `lfm25_kernels.py` (the conv with its state shifted in
place, the dense SwiGLU). One TinyJit graph a layer, the routing on the device ("sigmoid_bias"). Board o6n: gate worst 1.1e-3,
llama.cpp's tokens, **9.2 tok/s** decode (0.84 eager), prefill 5.1 tok/s token by token.

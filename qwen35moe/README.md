# Qwen3.6-35B-A3B (and Qwen3.5-35B-A3B) on the NPU

The UD-Q4_K_M GGUF, nothing requantised: `qwen35moe_pack.py` (q8f for the Q8_0 mixers, q4m gate / up, the native 5- / 6-bit `q5n` /
`q6n` streams for the Q5_K / Q6_K down experts, the shared expert's q8f, the Q6_K head in parts; `QWEN35MOE_GGUF`, `QWEN35MOE_NPU`),
`qwen35moe_ref.py` (qwen3.8-27b's reference with the MoE), `qwen35moe_npu.py --gate 0,3` (a layer against the reference),
`qwen35moe_generate.py "prompt"` (one TinyJit graph a layer: the DeltaNet / attention mixers of qwen3.8-27b on the GEMM tiles, the
routing on the device with the shared expert's sigmoid gate, the experts on the qdot GEMV), `bench_qwen36.py` (each kernel timed).
Board o6n: all 40 layers gated (worst 1.05e-3), llama.cpp's ids, 4.7 tok/s decode at 23.6 GB resident. The router's fp32 GEMV is a
quarter of the token (1.35 ms a layer); a DMA router GEMV is the next step.

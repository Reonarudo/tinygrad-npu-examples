# Gemma 4 26B-A4B on the NPU

The unsloth UD-Q4_K_M GGUF, nothing requantised: `gemma26_pack.py` (q8 attention / dense / head, q4m fused gate|up experts, the new exact
`q51` qdot kind for the Q5_1 down experts), `gemma26_ref.py` (gemma4_ref with per-expert dequantisation), `gemma26_kernels.py` (the row
kernels), `gemma26_npu.py` (one TinyJit graph a layer; `--gate 0,5,29` against the reference, or a prompt). Board o6n: gate worst 2.2e-4,
the reference's ids, 3.7 tok/s decode at 18.1 GB resident.

# Qwen3.6-35B-A3B (and Qwen3.5-35B-A3B) on the NPU

The UD-Q4_K_M GGUF, nothing requantised: `qwen35moe_pack.py` (q8f for the Q8_0 mixers, q4m gate / up, the native 5- / 6-bit `q5n` /
`q6n` streams for the Q5_K / Q6_K down experts, the shared expert's q8f, the Q6_K head in parts; `QWEN35MOE_GGUF`, `QWEN35MOE_NPU`),
`qwen35moe_ref.py` (qwen3.8-27b's reference with the MoE), `qwen35moe_npu.py --gate 0,3` (a layer against the reference),
`qwen35moe_generate.py "prompt"` (one TinyJit graph a layer: the DeltaNet / attention mixers of qwen3.8-27b on the GEMM tiles, the
routing on the device with the shared expert's sigmoid gate, the experts on the qdot GEMV), `bench_qwen36.py` (each kernel timed).
Board o6n: all 40 layers gated (worst 1.05e-3), llama.cpp's ids, 4.7 tok/s decode at 23.6 GB resident. The router's fp32 GEMV is a
quarter of the token (1.35 ms a layer); a DMA router GEMV is the next step.

## Speculative decoding: `qwen36_spec.py`

The Qwen3.8 runtime (qwen3.8-27b/qwen38_generate.py: verify passes of up to 6 rows, the DeltaNet state banks and commits, the MTP
drafter, the leaf tree) over this cache, completed by `qwen36_spec_pack.py`: the router / shared-gate tables as files mapped with each
layer, every layer's files padded to one size per name (the runtime's slot layout is fixed per layer type; the q6n down layers are
bigger), the Q6_K head requantised to Q8_0 for the runtime's head GEMM, and **Qwen3.6's own MTP layer** -- a MoE layer with 256
experts of its own -- as layer 40 in `mtp/` (its 19 `mtp.*` tensors, 1.7 GB of bf16, fetched from the HF checkpoint by range requests;
the GGUF drops them; every linear quantised to Q8_0). `MoELayers.mlp` replaces the dense MLP for 1 to 6 rows: the router and the shared
gate as gemv32s of the rows, the top-8 routing on the device (`route_rows`), the experts' gate / up as qdot GEMV heads sharing a row's
package (`act_heads`), the down from their own rows, the shared expert, the weighted combine (`comb_rows`); the layer graphs are keyed
by the layer's stream kinds. Board o6n, 2026-10-26, three prompts, 64 new tokens:

| | plain (this runtime, `QWEN_DRAFT=none`) | speculative (6 drafts a pass, leaf tree) |
| --- | --- | --- |
| decode | 2.69 tok/s (0.37 s a step) | **5.5 / 8.5 / 6.3 tok/s** (code 8.5; 3.0 / 1.8 drafts accepted a pass) |
| prefill (24 tokens) | 11 tok/s (the verify path, 6 rows a pass) | the same |
| ids | the exact one-row generator's ("The capital of Portugal is Lisbon.") | identical to plain on two prompts; one near tie moved at token 13 of the third |

The one-row generator (`qwen35moe_generate.py`, 4.7 tok/s) stays the exact path: this runtime's plain step is slower (its MoE row
kernels and router GEMV are plain loads, its head six GEMM parts), so the speculative gain over it is 2-3x and over the best plain
decode 1.2-1.8x. The MTP layer costs ~38 ms a draft (its q8 experts); the verify pass of 6 rows 0.39 s.

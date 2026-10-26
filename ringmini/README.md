# Ring-mini-2.0 on the NPU

inclusionAI's Ring-mini-2.0 (bailingmoe2: 20 layers, 16 / 4 x 128 heads with NeoX RoPE on 64 dims, 256 experts x 512 top 8 in 8 groups
with a sigmoid router + bias and a shared expert, a dense first layer) from its Q4_K_M GGUF, nothing requantised: `ringmini_pack.py`
(q5n qkv, q4m, q6n), `ringmini_ref.py`, `ringmini_npu.py --gate 0,1,19` or `"prompt"`. An all-GEMV layer as one TinyJit graph: the
routing on the device (`seqmix/route.py` "group"), the [q | gate] rows and the residuals as row kernels (`seqmix/rows.py`), the k | v
row straight from the GEMV. Board o6n: the reference's tokens, **12.5 tok/s** decode (1.6 eager), prefill 9.1 tok/s token by token.
This is the model the speed recipe was worked out on; the other ports follow it.

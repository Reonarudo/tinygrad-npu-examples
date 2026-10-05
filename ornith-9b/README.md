# Ornith 1.0 9B on the Zhouyi NPU

Ornith 1.0 9B (a coding assistant built on the Qwen3.5 architecture) from its GGUF **Q8_0** checkpoint, running on the NPU of
the Radxa Orion O6N through tinygrad's `ZHOUYI` backend (the repository's `tinygrad/` submodule). Every layer, the output head
and the prompt's prefill run on the NPU; the host tokenises, looks up the embedding and picks tokens.

**Status: it generates.** *"What is the capital of Portugal? Answer in one sentence."* ->
`The capital of Portugal is Lisbon.<|im_end|>` (greedy). Measured on the board: model set-up 16 s, prefill ~7.8 tokens/s
(35 tokens in 4.5 s), **decode ~4.4 tok/s across prompts and ~5.4 tok/s on a code prompt** with speculative decoding and
the `--q8f` cache (below; 1.7 tok/s, 0.58 s a token, without speculative decoding).

## The model

| | |
| --- | --- |
| layers | 32: 24 Gated DeltaNet (linear attention) + 8 full attention (every 4th) |
| hidden / MLP | 4096 / 12 288 (SwiGLU) |
| full attention | GQA 16 query / 4 kv heads, head dim 256, RoPE on 64 dims (theta 1e7), q/k RMSNorm, sigmoid output gate |
| Gated DeltaNet | 16 key heads x 128, 32 value heads x 128, causal conv1d (kernel 4) + SiLU, gated delta rule, gated RMSNorm |
| vocabulary | 248 320 (Qwen3.5's tokenizer: identical vocabulary and merges), untied output head |
| weights | GGUF Q8_0: int8 with an fp16 scale per 32 weights (the linears, the embedding, the head); norms and DeltaNet parameters f32 |

It is the same layer structure as [qwen3.8-27b](../qwen3.8-27b/) at smaller sizes, so it runs **that example's modules** with
`QWEN_MODEL=ornith-9b` (the dimensions, the layer count and the weight format come from a profile in `qwen38_ref.py`).

## What was new for this model

- **A Q8_0 GEMM** (`k_gemm_gs(q8=True)` in the backend, `tinygrad/extra/zhouyi/`): the block-scaled GEMM with a 32-deep K slice
  (one Q8_0 block). The packer stores each code as u = q + 128 (unsigned), so the expand is one zip: the byte in the low half
  of a 16-bit lane IS the fp16 **subnormal** u x 2^-24 (exact). The +128 comes back out per slice: a pre-pass multiplies the
  A slice by a constant tile of 2^-17 on the matrix unit, which gives each row's sum in the C tiles' layout, and C -= it before
  the scale. The scales are stored as fp16 (d x 2^10, 32 B a strip) and widened in registers. It relies on the matrix unit
  taking subnormal fp16 inputs exactly, which the board confirms (3e-7 vs float64 on Ornith's real weights:
  `check/check_q8_gemm.py`). The one-zip expand and the 4x smaller scale table
  took the MLP's up/gate GEMM from 6.3 to 5.0 ms a token (25 GB/s of weights); the kernel is still compute-bound.
- **DeltaNet kernels at 32 value heads**: 4 heads a task on 8 of the 12 tasks (the 27B's 48 heads fill all 12).
- **The converter undone** (`ornith_weights.py`): the GGUF converter reordered the DeltaNet's value heads from grouped to
  tiled order (for its runtime's broadcast), stored the zero-centred norms as 1 + w and A_log as -exp(A_log). The packer restores the
  Hugging Face conventions, so the kernels and the numpy reference read it like any Qwen3.5 checkpoint.
- **Prefill through the multi-token decode path**: the prompt goes through the speculative-decoding verify pass in chunks of
  up to 6 tokens (each chunk one streaming of the weights, the DeltaNet states and caches carried as in decoding).

### `--q8f`: the weights built in fp16 (optional, faster, not exact)

`ornith_pack.py --q8f --out /mnt/ssd/ornith-9b-npu-q8f` packs the same Q8_0 weights for `k_gemm_gs(q8f)`: the expand builds each
weight in fp16 with the TEC's 16-lane fp16 ALU (the code u = q + 128 zipped under 0x64 is 1024 + u; `sub.fp16` 1152 gives q
exactly; `mul.fp16` by the block scale d gives fp16(q·d)), so a K-slice spans four scale blocks and the per-32-weight
correction and scale pass is gone. The runtime picks the mode from a `q8f` file in the cache (`QWEN_NPU=...-q8f`).

`ornith_pack.py --fuse --out <cache>` (either cache) adds the fused same-input projections of qwen3.8-27b's `--fuse` next to the
per-linear files: `q | k|v` (8192 + 2048 -> one GEMM of 10240, k|v's tiles at group 171) and `qkv | z` (8192 + 4096 -> 12288, z at
group 171), the MTP head too; ~1.7 GB more on disk, nothing read from the GGUF, `QWEN_FUSE=0` ignores them.

| | exact kernel (default) | `--q8f` |
| --- | --- | --- |
| GEMM vs float64 (5 real tensors) | 3e-7 | 2e-4 (one fp16 rounding per weight, as fp16 weights) |
| GEMM time (4 rows) | compute-bound | DDR-bound: compute -20 to -25 %, full -3.4 to -5.7 % |
| decode, three prompts x 64 tokens | 3.50 / 4.55 / 3.69 tok/s | **3.63 / 4.86 / 3.90 tok/s** |
| decode, 300 tokens | 4.45 tok/s | 4.50 tok/s |
| prefill | ~7.8 tok/s | **~10.5 tok/s** (its 6-row passes were the most compute-bound) |
| tokens | — | identical to the exact kernel on all 492 generated tokens |

There is no fixed-function dequantisation on this NPU to use instead: the vendor's GPTQ hardware (group scales, fp16×int8/int4)
is X3-only, compiled into the X2 simulator but hard-wired off.

## Speculative decoding: Qwen3.5-9B's MTP head drafts for Ornith

Ornith's GGUF has no multi-token-prediction head, but Ornith is a fine-tune of Qwen3.5-9B (its weights are within 2-3 % of the
base model's; unrelated weights would differ by ~140 %), and the base model's MTP head -- one full-attention layer + MLP, kept
in the bf16 GGUF conversion as `blk.32.nextn.*` -- drafts well for it. `ornith_pack.py --mtp` packs it (Q8_0) as layer 32 and
qwen3.8-27b's speculative path runs as it does for the 27B: each pass verifies the token plus up to 3 drafts in one streaming
of the weights (`QWEN_DRAFT_MAX`; the chain stops early when the draft head is unsure).

| | |
| --- | --- |
| draft acceptance (numpy, 192 generated positions) | 1st draft 80 %, 2nd 75 %, 3rd 69 % |
| decode, three prompts x 64 tokens | **3.50 / 4.55 / 3.69 tok/s** (plain: 1.72; 2.74 / 3.58 / 2.93 before the host overhead and the TEC work below) |
| decode, 300 tokens (a binary-search-tree module) | **4.45 tok/s** (3.89 with the old attention kernel; the same 300 tokens) |
| tokens | identical to plain greedy decoding on all three |

A verify pass of up to 4 rows costs about one plain step; 5-6 rows cost ~1.45x (the Q8_0 GEMM is compute-bound), which is why
the drafts stop at 3. Where a pass's time goes (4 rows): 0.56 s = the layers 0.48 s
(NPU-bound: ~15 ms a layer, of which the GEMMs ~11.9 ms; launched as one graph a layer, host 0.5 ms) + the head 0.066 s (its
GEMMs, the top-1 on the device) + 0.014 s host; a draft pass 36-40 ms (the MTP layer ~17 ms, a third of the head ~20 ms), nearly
all NPU. Fusing the small kernels (residual + norm) or dropping the index copies measured no gain: in a graph their launches
are cheap.

**All 12 TECs.** The DeltaNet token kernels give each task ceil(32 / 12) = 3 value heads (11 tasks; 4 heads on 8 tasks before):
`gdn_tokm` 1.94 -> 1.58 ms. The verify attention was the kernel that grew with the context: a unit per (row, head) read its KV
head's cache with plain loads, every row 16 times (14.6 ms a layer at position 250). `attn_part` + `attn_comb` give a unit
each (head, position slice) -- 16 x 3 = 48 units, 4 a task -- stream the cache through LSRAM by DMA once per unit, and merge the
slices' softmax partials: 0.98 / 1.58 / 2.2 ms at positions 30 / 250 / 480 (was 2.2 / 14.6 / ~28). The GEMMs' 48-column groups
leave the N = 4096 linears' last task short (86 groups: 8 a task on 11), but 32-column groups measured no better (each strip's
fixed cost eats the balance). On one near-tie (plain decoding's logits 24.954 vs 24.935) an earlier setting took the
other token; the float32 reference sided with the speculative path.

## Run it

The paths below (`/mnt/ssd/...`) are examples and the scripts' defaults: set `ORNITH_GGUF`, `ORNITH_MTP` and `QWEN_NPU` to your
own paths.

You need two GGUF files, fetched yourself: Ornith's checkpoint `ornith-1.0-9b-Q8_0.gguf` (9.5 GB) and, for speculative decoding,
a Qwen3.5-9B GGUF that keeps the `nextn` (MTP) tensors, e.g. its bf16 conversion. You also need Qwen3.5's `tokenizer.json`
(Ornith's vocabulary and merges are identical; Qwen3.8's file works too, e.g. the one `bonsai2-27b/download.sh` fetches).

```sh
cd ornith-9b
export ORNITH_GGUF=/mnt/ssd/models/ornith-1.0-9b-Q8_0.gguf ORNITH_MTP=/mnt/ssd/models/Qwen3.5-9B-bf16.gguf
export QWEN_NPU=/mnt/ssd/ornith-9b-npu-q8f
python3 ornith_pack.py --q8f                       # the 32 layers and the head, ~1 min, 8.0 GB
python3 ornith_pack.py --q8f --mtp                 # the draft head from $ORNITH_MTP (optional: without it, plain decoding)
python3 ornith_pack.py --fuse                      # the fused projections, ~1.7 GB more (optional)
cp /path/to/tokenizer.json $QWEN_NPU/
python3 ornith_generate.py "Write a Python function that returns the n-th Fibonacci number." --max-new 64
```

Leave out `--q8f` for the exact kernel's cache (a separate folder: a cache holds one format). `ornith_generate.py` is
qwen3.8-27b's `qwen38_generate.py` with the profile set; the same options apply (`--max-new`, `--thinking`, `--out`); with the MTP
head packed, decoding is speculative (`QWEN_SPEC=0`: plain). Environment: `ORNITH_GGUF` (the checkpoint, read for the
embedding), `QWEN_NPU` (the packed cache), `QWEN_TOK` (tokenizer.json's folder, default the cache), `QWEN_PIN_GB` (default 12).
qwen3.8-27b's defaults apply here too (`QWEN_GDN_PREP=dma`, `QWEN_FUSE=1` when the cache was packed `--fuse`), except the draft
head's parts, which stay `QWEN_DRAFT_PARTS=2` (both read); `QWEN_ATTN_PREFILL` is not used (Ornith prefills through the verify
path). The [root README](../README.md#configuration-qwen38-27b-bonsai-2-and-ornith) has the full table.

## How correctness is checked

- `check/check_q8_gemm.py [tensor] [rows]` (on the board): the Q8_0 GEMM on a real Ornith linear against float64 -- the test that
  the matrix unit honours the subnormal fp16 inputs (`Q8F=1`: the `q8f` kernel).
- `check/ref_compare.py prompt.txt out.npz [other.txt] [n]` (no device): the NPU's greedy tokens, saved by
  `ornith_generate.py --raw --out out.npz "$(cat prompt.txt)"` (the file is tokenised as is, without the chat
  template), against a float32 numpy forward of the whole model on the dequantised weights: at every generated position, does fp32 pick the NPU's token, and how close is its runner-up. With `other.txt`, also where another
  implementation's text diverged and which side fp32 takes.
- Speculative decoding returns plain greedy decoding's tokens by construction; the measured runs above were compared id for id.

## Files

| file | what |
| --- | --- |
| `gguf_read.py` | a minimal GGUF reader: metadata, tensor index, F32 / F16 / BF16 / Q8_0 / Q4_K / Q6_K tensors from a memory map |
| `ornith_weights.py` | the checkpoint under the Hugging Face Qwen3.5 names and conventions (the converter's changes undone); `MTP`: the draft head |
| `ornith_pack.py` | GGUF -> the per-layer NPU cache (Q8_0 GEMM streams, small tensors, the head in 6 parts); `--mtp`: the draft head |
| `ornith_generate.py` | generation (qwen38_generate.py with `QWEN_MODEL=ornith-9b`) |
| `check/check_q8_gemm.py` | the Q8_0 GEMM on a real Ornith linear vs float64 (`Q8F=1`: the q8f kernel; `Q8_DBG=unit_d,zero_q,ones_a`: debug arms) |
| `check/ref_compare.py` | the NPU's greedy tokens vs a float32 numpy forward of the whole model (and where another implementation diverged) |

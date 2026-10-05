# Qwen3.8-27B on the Zhouyi NPU

Text generation with [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) (Apache-2.0), all 64
language-model layers, the output head and the multi-token-prediction (MTP) draft head on the NPU through the `ZHOUYI` backend.
Text only: the vision tower is not used.

**Status: it generates.** *"What is the capital of Portugal? Answer in one sentence."* is answered with
`The capital of Portugal is Lisbon.<|im_end|>` (greedy). Measured on the board (24-token prompt, a 512-slot K/V cache): prefill
~21 s, then **1.52 s per generated token (~0.66 tok/s)** with plain decoding, and **~1.9-3.0 tok/s** with speculative decoding
(`QWEN_SPEC=6`), with the same tokens. What makes it run at this speed, roughly in order of effect: hand-written kernels for
everything the generic kernels did badly; decode GEMMs that compute only the real rows and prefetch their operands;
**zero-copy weights** (the kernel driver's weight buffers and slots: the NPU reads each layer where it already sits in RAM,
instead of the CPU copying 26 GB into the NPU's window every token); a fused decode attention and DeltaNet step (`gdn_tok3`
reads the GEMMs' output tiles, computes the a|b projection, beta and decay itself and writes o_proj's input layout); and an fp8
lm_head (below).

This folder also holds the modules the other two text examples run: [bonsai2-27b](../bonsai2-27b/) and
[ornith-9b](../ornith-9b/) set `QWEN_MODEL` and reuse them. All settings are listed in the
[root README](../README.md#configuration-qwen38-27b-bonsai-2-and-ornith).

## The model

| | |
| --- | --- |
| layers | 64: 48 Gated DeltaNet (linear attention) + 16 full attention (every 4th) |
| hidden / MLP | 5120 / 17 408 (SwiGLU) |
| full attention | GQA 24 query / 4 kv heads, head dim 256, RoPE (theta 1e7), q/k RMSNorm, sigmoid output gate |
| Gated DeltaNet | 16 key heads x 128, 48 value heads x 128, causal conv1d (kernel 4) + SiLU, gated delta rule, gated RMSNorm |
| vocab | 248 320, untied `lm_head` |
| weights | fp8 (E4M3) linears with 128 x 128 block scales; `embed_tokens`, `lm_head`, norms and the DeltaNet's small parameters in bf16 |

## What the board imposes

The NPU addresses a 4 GB window, so the 27 GB of weights cannot stay resident as device buffers. With the driver's zero-copy
weight buffers (the WBUF / SLOT ioctls) each layer is mapped into a slot of that window where it already sits in RAM; without
them (`QWEN_ZC=0`) the CPU copies every layer into the window every token, ~2-3 s a token. The board has 29 GB of RAM:
`QWEN_PIN_GB` (default 24) GB of layers are pinned in RAM, the rest is read from the NVMe SSD each token. Decoding is bound by
streaming the weights through the NPU's ~24 GB/s DDR port, which is why speculative decoding pays: a pass through the weights
costs about the same for 1 to 8 rows.

## Run it

The paths below (`/mnt/ssd/...`) are examples and the scripts' defaults: set `QWEN_DIR` (the checkpoint) and `QWEN_NPU` (the
packed cache) to your own paths, on a disk with ~60 GB free (27 GB of checkpoint, ~25 GB of layers, the heads, the drafter and the fused streams).

```sh
cd qwen3.8-27b
export QWEN_DIR=/mnt/ssd/qwen3.8-27b-fp8 QWEN_NPU=/mnt/ssd/qwen3.8-27b-npu-s1
bash download.sh                                  # the checkpoint (27 GB) into $QWEN_DIR; resumable
python3 qwen38_pack.py --scales single            # the per-layer NPU cache (~25 GB) into $QWEN_NPU, ~1.3 s a layer
python3 qwen38_pack.py --scales single --lm-head  # lm_head as fp16 GEMM panels (2.5 GB) + the final norm
python3 qwen38_pack.py --scales single --lm-head-fp8   # the fp8 head (the default head; needed by the verify pass)
python3 qwen38_pack.py --scales single --mtp      # the checkpoint's MTP head as layer 64 (0.5 GB), the drafter
python3 qwen38_pack.py --fuse                     # the same-input projections as one stream each (no checkpoint read)
python3 qwen38_generate.py "What is the capital of Portugal? Answer in one sentence." --max-new 32
QWEN_SPEC=6 python3 qwen38_generate.py "Write a Python function that returns the n-th Fibonacci number." --max-new 80
```

`--scales single` stores each block scale once (2.9 % smaller streams, the same numbers); the cache records its layout in a
`scales` file, and `qwen38_repack_scales.py` converts a cache packed without it.

`qwen38_generate.py` renders the chat template (thinking off by default; `--thinking` for the model's default mode), prefills
the prompt through the 64 layers, then decodes greedily until `<|im_end|>` / `<|endoftext|>` or `--max-new`. `--raw` sends the
prompt without the template, `--ids 1,2,...` gives the prompt as token ids, `--tmax` sets the K/V cache length (default 512), and
`--out x.npz` keeps the generated ids and each step's top-8 logits.

To serve the model over the OpenAI and Ollama APIs, `qwen38_serve.py` takes the same environment (it needs the verify path,
`QWEN_SPEC` >= 2): `QWEN_SPEC=6 python3 qwen38_serve.py --host 0.0.0.0 --port 8000`. Its endpoints, the service set-up and the
clients are described in [bonsai2-27b/README.md](../bonsai2-27b/README.md#serve-openai--and-ollama-compatible-api).

## Speculative decoding (`QWEN_SPEC=6`)

Decoding is bound by streaming the 26 GB of weights once per token, and a GEMM costs about the same for 1 to 8 rows. So each
pass through the weights verifies the current token plus up to 5 drafts from the checkpoint's own multi-token-prediction head
(`mtp.safetensors`: one attention layer + MLP, run on the NPU). The model keeps the drafts it agrees with plus its own next
token; the output is **identical to plain greedy decoding** (checked id for id). The DeltaNet states roll back to the last
accepted token (`gdn_tokm` keeps the state after each token; `gdn_commit` restores it); the attention caches need no rollback.

**Adaptive depth.** The draft chain goes on only while the draft head's probability of its draft is >= `QWEN_DRAFT_TAU` (0.6);
a pass verifies 3, 4 or 6 rows (`QWEN_SPEC_GEOS`, the count rounded up; rows 5.. cost nearly what 6 do). `QWEN_DRAFT_MAX` caps the
drafts a pass (default `QWEN_SPEC` - 1 here; 3 for the Q8_0 and ternary models, whose compute-bound GEMM makes a 5-6-row pass
costlier). Measured on 350 drafts: probability >= 0.95 -> 72 % accepted, < 0.5 -> 12 %. Every geometry's JIT is captured at
start-up (~15 s, `QWEN_SPEC_WARM`).

The head returns only its top-1: per head part the GEMM and a `head_top` kernel (each row's max, its column and the sum of exp)
run as one submission, and the small uploads are memmoves into the buffers' mappings. The verify attention is split over the
TECs (`attn_part` + `attn_comb`: (head, position slice) units streaming the cache through local memory by DMA). On the
Fibonacci prompt: **2.60 tok/s** (80 tokens, the same ids as plain decoding).

**The draft head's parts (`QWEN_DRAFT_PARTS=<n>[,thresh:<x>]`).** A chained draft is ~57-59 ms on the 27B (~35 ms on Ornith),
most of it the head: each of the n = 2 parts streams ~318 MB of E4M3 weights (18-19 ms on the 27B). `2,thresh:<x>` reads part 0
and then parts 1.. only when part 0's top logit on the row drafted from is <= x (a confident part 0 is taken as the argmax);
`2` reads both. The 27B's default is `2,thresh:25` (Ornith keeps `2`). Verification decides the tokens, so the policies only
change which drafts are proposed.

**Tree verification (`QWEN_SPEC_TREE`, off by default).** `QWEN_SPEC_TREE=rescue2` makes every verify pass 8 rows: the chain of
5 drafts plus two "rescue" rows, each holding the draft head's second-choice token at one of the two chain positions whose draft
had the lowest top-1 - top-2 margin (`fixed:<j1,j2>`: at the positions given). A rescue row has the same parent as the chain row
it replaces, so it attends (and its DeltaNet state follows) only its ancestors; when the chain is rejected at that position and
the rescue token is the model's, the pass commits one more token. The output is still plain greedy decoding's. With the tree on,
`QWEN_SPEC` / `QWEN_SPEC_GEOS` are forced to 8, `QWEN_HEAD_TOP3` is on, `QWEN_DRAFT_MAX` <= 5, and the per-row DeltaNet state
banks take 1.06 GB. The gain is about 1 % on average. The kernels (`gdn_tokt`, `attn_partt`, `gdn_commit_tree`) are the chain
kernels' sources with the tree edits conditional, so with the tree off the chain kernels are unchanged; `spec_tree.py` holds the
rows, the table the kernels read and the acceptance along the tree.

**Telemetry (`QWEN_SPEC_LOG=path.jsonl`).** Opt-in; it appends JSON lines:

- a `run` header: the prompt and its ids, the model, M, the geometries, kmax, tau, and the `QWEN_*` environment;
- one line per verify pass: `pos`, the rows `m`, the drafts `nd`, the tokens accepted `a` (padding rows included), each draft's
  raw `p`, `f` = (top logit, the 2nd-largest (part, task) maximum), the drafts `d`, the model's tokens `g`, and host wall times:
  `tv` (verify, its head's top-1 included), `tc` (commit), `td` (each draft) and `t_rest` (verify end to iteration end);
  per draft `fp`, each head part's top logit in the order read, and with `QWEN_HEAD_TOP3=1`, `t3` (per part read: its three
  largest logits' ids and the logits) and `r3` (the rank, 0-2, of the model's token among the merged top-3; -1 when absent);
- an `end` line: the tokens, the wall time, and `out_ids`.

The times come from `time.perf_counter()` at the path's existing sync points; no sync is added. With the log off, the extra
cost is two `perf_counter` calls per draft decision and two per commit.

## Fused projections (`qwen38_pack.py --fuse`, `QWEN_FUSE`)

Two linears of every layer read the same normalised residual: the attention layers' `q` (12288 columns, q and its gate) and
`k|v` (2048), the DeltaNet layers' `qkv` (10240) and `z` (6144). `--fuse` writes each pair as ONE stream file next to the
per-linear ones (`L{l}_qkv.bin` = `L{l}_q.bin` + `L{l}_kv.bin`, `L{l}_qkvz.bin` = `qkv` + `z`, the MTP layer too; `L{l}_fused.npz`
records the parts' group offsets): a stream is whole 48-column groups, so the concatenation is the parts' streams back to back
and the fused GEMM's output tiles are the parts' back to back -- bit-identical, no re-quantisation and nothing read from the
checkpoint (5.6 GB more on disk). The runtime uses the fused files when every layer has them (`QWEN_FUSE=0`: the per-linear
files): one `gemm_gs` call instead of two, every consumer reading its linear's tiles at its group offset in the one buffer.

## The lm_head in fp8

The checkpoint stores `lm_head` in bf16 (2.5 GB). `qwen38_pack.py --lm-head-fp8` re-quantises it to E4M3 with a scale per
128 x 128 block (amax / 448) -- the format every other linear of the checkpoint already uses -- so it runs through the same
GEMM, zero-copy: ~65 ms instead of ~335 ms a token. Measured against the fp16 head on 41 decode steps of two prompts: the same
argmax at every step (including steps where the top two differed by 0.2), top-8 sets agreeing 7-8 of 8, logits within ~0.45
of values around 25. It is the default when packed; `QWEN_HEAD=fp16` restores the exact head, `QWEN_HEAD=both` prints the
comparison each step.

## How correctness is checked

| check | where | what |
| --- | --- | --- |
| `check/truth_transformers.py`, then `check/check_ref.py` | a workstation with `torch` and `transformers` | `qwen38_ref.py`, the numpy fp32 reference, against transformers' own Qwen3.5 modules on layers 0 (Gated DeltaNet) and 3 (attention); `QWEN_SHARDS` = a folder with `config.json` and the two layers' `.safetensors`. The reference matches to fp32 noise |
| `python3 qwen38_npu.py --gate L` | the board | layer L on a random input against the numpy reference |
| `check/check_gemm_bscale.py` | the board | the block-scaled GEMM on a real layer-0 linear against numpy (`ROWS=4,8,...`: the verify pass's row mode) |
| `check/check_kv_decode.py` | the board | the attention layer's decode path (the K/V cache): a 12-token prefill then one decode step against the reference over 13 tokens |
| `check/check_kernels.py` | the board | the hand-written kernels against the tinygrad-op versions, with timings |
| `check/check_spec_kernels.py [M]` | the board | the multi-token kernels (`gdn_tokm`, `gdn_commit`, `attn_decm`) against M sequential single-token runs |
| `QWEN_SPEC=4 python3 check/spec_check.py [N]` | the board | the verify / commit path against plain greedy decoding, with an oracle drafter (every draft accepted) and a bad one (the rollback exercised): the ids must match |

Accuracy on the board (random 12-token inputs, against the numpy fp32 reference):

| | rel err of the layer's contribution | note |
| --- | ---: | --- |
| block-scaled GEMM alone (`in_proj_qkv`, `down_proj`, ...) | 1.3e-6 .. 2.4e-6 | fp32 accumulation noise |
| attention layer (layer 3), prefill and the K/V-cache decode row | 3.9e-4 / 4.1e-4 | fp16 GEMM inputs |
| Gated DeltaNet layer (layer 0) | 1.4e-4 | |

## A hardware limitation found here: 16-byte loads read 16 bytes past their end

Generation hung at a place that moved with unrelated allocations. The cause, measured on the board and checked on the vendor
simulator:

- On silicon, a 16-byte vector load (measured with `float4`; the compiler emits it as `ld tN`) reads **16 bytes past
  its end**. Scalar loads, 32-byte loads and stores of either width stay in bounds. The simulator executes the same
  instruction as exactly 16 bytes (its access trace shows no read past the buffer, and its output is correct), so it
  cannot show this: the overhang is below the ISA level it models.
- When the last 16-byte load of a buffer sits at the buffer's end and the next page is not mapped, the SMMU faults the
  overhang (`dmesg`: `F_TRANSLATION ... iova: <the buffer's end>`) and the TEC waits forever: the job times out with no
  NPU exception. Whether the next page is mapped depends on the allocation layout, hence the moving hang.
- When the buffer ends at the top of the NPU window (`0xC0000000`, where a process's first allocation lands), the load
  returns garbage instead: the last 4 floats of `rms_a32`'s input came back as the weight vector's.

The fix is in the backend, not here: every `REQ_BUF` mapping carries 16 readable bytes after its logical size
(`RawDevice.LOAD_OVERHANG`; it adds a page only when a buffer ends within 16 bytes of a page boundary). With it set to 0
both failures reproduce; with 16 neither does.

## Files

| file | role |
| --- | --- |
| `download.sh` | the checkpoint into `$QWEN_DIR` (default `/mnt/ssd/qwen3.8-27b-fp8`; resumable) |
| `qwen38_pack.py` | the per-layer cache: every fp8 linear as the block-scaled GEMM stream (`gemm_gs(b8, bscale)`, K-slices of 128 = the scale blocks, k\|v and gate\|up fused), the bf16 tensors as float32; `--lm-head` / `--lm-head-fp8` the head; `--mtp` the drafter; `--fuse` the same-input pairs as one stream file each, alongside |
| `qwen38_repack_scales.py` | converts a cache to single-layout scales without the checkpoint |
| `qwen38_npu.py` | the layers on the device: Gated DeltaNet, full attention, SwiGLU, the weight streaming; `--gate L` checks layer L against numpy |
| `qwen38_kernels.py` | the model's hand-written NPU kernels |
| `qwen38_generate.py` | the driver: embedding, prefill, the K/V cache and the DeltaNet states, the head, greedy and speculative decoding |
| `qwen38_tokenize.py` | tokenizer (`tokenizers`), the single-turn chat template and the message-list form (`chat_messages`) |
| `qwen38_serve.py` | the OpenAI- and Ollama-compatible HTTP server (standard library only) |
| `spec_tree.py` | `QWEN_SPEC_TREE`: the tree verify's rows, the kernels' table, the acceptance along the tree and the committed path |
| `qwen38_ref.py` | the numpy fp32 reference of every layer type, and the model profiles (Qwen3.8-27B, Bonsai 2, Ornith) |
| `check/` | the correctness checks above |
| `zy.py` | resolves the backend's custom ops from the tinygrad tree `TG` points at |

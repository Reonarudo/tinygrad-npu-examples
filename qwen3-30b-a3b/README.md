# Qwen3-30B-A3B on the Zhouyi NPU

Qwen3-30B-A3B (a 128-expert mixture of experts, 3.3 B parameters active a token) from the **same GGUF the Radxa forum's llama.cpp
benchmark uses** ([`Qwen/Qwen3-30B-A3B-GGUF`](https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF), `Qwen3-30B-A3B-Q4_K_M.gguf`,
17.3 GiB; [forum post](https://forum.radxa.com/t/llama-cpp-benchmarks/27813/13)), running on the NPU of the Radxa Orion O6N
through our tinygrad fork, in the checkpoint's own 4- and 6-bit formats. **Everything runs on the NPU** -- the embedding lookup,
every layer (attention, the router, the routing, the 8 experts), the head and its argmax: the host tokenises, writes the prompt's ids,
maps each layer's weights into the NPU's window (an ioctl a layer: 21.5 GB of weights, a 3 GB window), submits the jobs and reads the
generated ids from a ring on the device every 8 tokens. Nothing is read back or computed on the host inside a prompt chunk or a token.

**Status: it generates the reference's tokens.** *"What is the capital of Portugal? Answer in one sentence."* ->
`The capital of Portugal is Lisbon.<|im_end|>` (greedy). On the board: **prefill 62.4 tok/s** in chunks of 512 tokens (llama.cpp
on the same board's 8 big cores: 31.3), **decode 9.6 tok/s** (llama.cpp: 17.4; ~10 with one int8 activation level, `QWEN3MOE_QLEV=1`).
Set-up ~30 s (26 GB of weights pinned: the prefill's streams and the decode-only copies -- 6.6-bit Q6_K linears, 4.75-bit attention).

## Speed against the CPU

Measured on this board, back to back, the board otherwise idle, CPU governor `schedutil` (as found), NPU clocks at their defaults.
The NPU column is `qwen3moe_prefill.py --bench` (512-token chunks of random ids from position 0, as `llama-bench`'s pp512) and the
decode of `qwen3moe_generate.py` (120 tokens, greedy, after the first).

| | llama.cpp, CPU (8 x A720, `llama-bench -t 8`) | this example, NPU |
| --- | ---: | ---: |
| prefill (pp512) | 31.3 tok/s | **62.4 tok/s** |
| decode (tg128) | **17.4 tok/s** | 9.6 tok/s |

(The forum's numbers for this file, on another O6 board: pp512 34.0, tg128 21.1 with llama.cpp; ik_llama 75.4 / 14.8.)

Prefill 62.4 tok/s since the routing moved to the device (58.7 before, with a host read-back a layer). Decode 9.6 tok/s since it runs
on integer dot products (`qdot`, below; 6.5 on the matrix unit's rows mode) over decode-only streams for the Q6_K linears (6.6 bits)
and the attention (4.75); 8.9 over the prefill's streams alone.

**Prefill is the NPU's**: a chunk of 512 tokens reads each expert's weights about twice (a slot of 24 tokens a read), and the work is
GEMMs on the matrix unit. **Decode is the CPU's, and stays so**: a decode token reads ~2.1 GB of weights in the streams the prefill
reads too (the 8 routed experts of every layer, the attention, the head; Q4_K at 5.0 bits a weight, Q6_K at 9.1), and the NPU reaches
DDR through one 24 GB/s port (~23 GB/s of reads measured) -- ~10.8 tok/s at best with these streams -- while llama.cpp streams ~31-38
GB/s through the CPU's memory path. The Q6_K linears (v and down in 24 layers, the head: a third of a token's bytes) now come from a
decode-only 6.6-bit copy (4.3 GB more RAM) read at the port's rate (the head 23.6 GB/s), and the attention's Q4_K (0.57 GB a token) from a
4.75-bit strip copy (0.5 GB more); the experts' Q4_K (0.75 GB a token at 5.0 bits, 15 GB in all) stays shared with the prefill, since a
second copy does not fit and the native 4.5-bit layout would need the prefill's matrix-unit expand rewritten. 26 of 29 GB are pinned. What is left over the port's ~85 ms a
token is ~0.25 ms a layer of small kernels and launches (gpart ~0.1 ms at position 100), measured, not launch-bound (a fused layer
is no faster; two layers a job needs 4 weight slots, which overflow the NPU window). Measured dead ends: prompt-lookup speculation
(a verify pass yields 1.3-1.5 tokens for 2-3x the expert bytes on code, nothing on prose: `check/pl_accept.py`); the CPU as a second
memory client (`check/bench_concurrent.py`, `host_gemv.c`, `QWEN3MOE_SYNC_PROBE`): the DDR does sustain ~60 GB/s for both, but
llama.cpp's own Q4_K / Q6_K dot kernels reach 4-11 GB/s on 4 A720 cores here (8 threads halve the NPU's share), and a host wait +
read after every layer's job costs 0.4 ms (9.55 -> 8.08 tok/s with nothing computed): a per-layer expert share would lose ~10-30 ms
a token to gain ~9; a split head (one sync a token) is worth ~2.5 ms. Not taken.

## The model

| | |
| --- | --- |
| layers | 48, each attention + a mixture of experts |
| hidden | 2048 |
| attention | GQA 32 query / 4 kv heads, head dim 128, q/k RMSNorm, NeoX RoPE (theta 1e6) |
| experts | 128, the top 8 a token (softmax, renormalised), SwiGLU with FFN 768; no shared expert |
| vocabulary | 151 936, untied output head |
| weights (Q4_K_M) | Q4_K: q, k, o, the gate / up experts, the embedding, v and down in 24 layers; **Q6_K**: v and down in the other 24 and the head |

## How it runs

**The K-quants on the TEC GEMM** (`k_gemm_gs` in the tinygrad fork, `extra/zhouyi/`), nothing requantised:

- **Q4_K -> `gemm_gs(q8=5)`** (`pack_b_group_q4m`, 5.0 bits a weight): w = d sc u - dmin m factored as a u + b per 32-weight
  sub-block. The expand widens each code byte by a byte multiply, splits the nibbles with one `and` and an integer `sub`, and makes
  each an exact fp16 by two full-rate multiplies (u as an fp16 subnormal, x 2^12 / 2^8, x a 2^12): fp16(a u), one rounding a weight,
  list-scheduled in one loop a K-slice position. b enters through **one extra `mma` k-step a 128-wide slice**: A = each row's four
  sub-block sums (summed in the kernel against selector tiles), B = the slice's b as a panel tile.
- **Q6_K -> `gemm_gs(q8=4)`** (`pack_b_group_q8f(qb=16)`, 9 bits a weight): the Q8_0 fp16 build with a scale per 16 weights.

**Decode** (`qwen3moe_generate.py`, `qwen3moe_qdec.py`): one TinyJit graph a layer and one for the head, whose last kernels pick the
token (`head_reduce`) and lay its embedding row (`embed_next`: the Q4_K row dequantised on the device, the id into a ring, the
position + 1) as the next step's input -- the host submits and reads the ring every 8 tokens (a stop token ends the loop there). The
21.5 GB of weights live in one KMD weight buffer a layer; two window slots are re-pointed at the next layer while the current one runs.

**The decode GEMVs are integer dot products** (`qwen3moe_qgemv.py`): `qdot` (4 unsigned code bytes x 4 signed activation bytes into an
int32 lane, 8 lanes, two a bundle) over the **same gemm_gs streams** the prefill reads -- their code bytes already lie as 8 output
columns x 4 k a 32-byte register, so the lanes are output columns and nothing is transposed or reduced across lanes:

- Q4_K (q8=5): `c & 15` / `c >> 4` are the two k-quads of a code byte; y = sum_j a_j (sum_k u q) + b_j (sum_k x) per 32-weight
  sub-block, a and b from the stream's fp16 tables (lane shuffles: `zipl`/`ziph` + `cvtue`, `cvtue`/`cvtuo` + `exte`/`exto`).
- Q4_K attention (q | k | v, o): by default from the decode-only stream `q4n` (`pack_b_group_q4n`, 4.75 bits: the codes as the shared
  stream's, the 6-bit sc / m as bytes read by one `qdot` each against a unit-byte pattern, fp16 d / -dmin a super-block; `qwen3moe_pack.py
  --q4n`: `L{l}_attn_q4n.bin`, `L{l}_o_q4n.bin`), strip-major: a unit is one 16-column strip (the o projection's 129 strips balance on
  12 tasks where its 43 groups did not), a strip's slice one contiguous DMA, 4 slices a request. `QWEN3MOE_Q4N=0`: the shared stream.
- Q6_K: by default from the decode-only stream `q6n` (`pack_b_group_q6n`, 6.6 bits a weight: the codes' low nibbles as the Q4_K
  stream lays them, their 2 high bits four to a byte, the 16-block int8 scales + 128 read by one `qdot` against a unit-byte pattern,
  the super-block's fp16 d; `qwen3moe_pack.py --q6n`, `--head-q6n`: `L{l}_down_q6n.bin`, `L{l}_v_q6n.bin`, `head_q6n.bin`); the
  prefill's q8f stream (q8=4: u = q + 128 a byte, 9 bits) is read when the copy is absent or `QWEN3MOE_Q6N=0`. Its body is a
  runtime loop over 16-k blocks (`QWEN3MOE_Q6N_MODE=shift`): the dense activation register is re-pointed at the block by a 16-byte-
  offset LSRAM load and its 4 quads replicated there, so ~28 vectors are live and nothing spills (a fully unrolled sub-block had ~54
  live and ~56 spills: 18-20 GB/s; a per-slice replica scratch in LSRAM: 18.9). The head runs at 23.6 GB/s, the down experts at 19.7.
- The activation in **two int8 levels a 32** (q1 = rint(x / sx1), then the residual): 2e-5 - 8e-5 on the model's q / k / v
  (`check/act_quant.py`; one level: 0.4-2 %, llama.cpp's Q8_K 1-5 %). One level (`QWEN3MOE_QLEV=1`) is 9.6 tok/s (+6 %) with the
  same ids on the Fibonacci and Portugal prompts and an earlier near-tie flip on the lighthouse one; the default keeps two.
- A layer is 8 launches (9 where v is Q6_K): the q | k | v GEMV builds its activation package from rms(x) itself and writes q and
  k | v straight into the attention's rows; `gpart` (gemma4's `gattn_part` with 8-position blocks and vector reductions, and
  `gattn_kvd` folded in: the kv head's last slice norms / rotates the new k, scores the new row from LSRAM and writes it and v into
  the caches); `comb_pkg` (gattn_comb per kv head: its 3 slices by 3 DMAs at once, its 8 records of the o projection's package by
  one); the o GEMV; `router_x1` (x1 = x + o, the router's logits over fp16 weights, the experts' package); gate | up of the 8 experts
  as one 16-head GEMV that also picks the top 8 from the logits; down, building each expert's package from silu(gate) up;
  `moe_comb2` (256-column units on 4 tasks: an 8-task launch measured ~45 us dearer). The GEMVs run at 18-22 GB/s (the launch
  included), a layer 1.81 (Q4_K) - 2.05 (Q6_K down) ms.
- The head: one launch over its decode-only stream with the top-1 kept per task (10.9 ms, 23.6 GB/s; 14.7 over the 9-bit parts),
  then `head_reduce`.

A layer as ONE launch is in too (`QWEN3MOE_FUSE=1`): the kernels become inline phases of a 12-task kernel and meet at a device
barrier -- each task cleans its core's write-back L2, writes its arrival epoch by DMA and polls by DMA (no hardware sync spans the
three cores; `check/bench_barrier.py`: 2.6 us a barrier, every word that crossed it right). It is correct (the same gate and ids) and
no faster (8.84 against 8.88 tok/s then): a kernel boundary inside a chained job costs only ~6 us, and the ~0.3 ms a layer beyond the
GEMVs' DDR time is the small kernels' own latency (gpart ~0.1 ms, router_x1 ~0.07) and the GEMVs' imbalance, not launches. A lone
12-task program as a graph's only member is submitted ~90 us dearer than a chain, so the fused graph carries a one-task no-op.

The matrix unit's rows-mode decode is kept (`QWEN3MOE_QDOT=0`, 6.5 tok/s): there the 12-row tile does 1 useful row in 12 and the
4-bit expand bounds the GEMMs at 12-16 GB/s.

**Prefill** (`qwen3moe_prefill.py`): chunks of 512 tokens (128 for prompts up to 128; the decode step under 24), one layer at a time:

- attention in the GEMM's full row-block layout, **on the matrix unit**: Q / K / V re-laid as fp16 tiles (`qrope_c` straight from
  the q|k|v GEMM's tiles, `k_panels`, `v_panels`), QK^T and PV as fp16 `gemm_gs` (32 heads, GQA by `b_head_div`), the causal
  softmax on tile pairs (`softmax_a`), the result straight into the o projection's A (`o_a`) -- for a chunk from position 0 (a later
  chunk: gemma4's `gattn_part`);
- the prompt rows from the token ids (`embed_rows`), the router for the chunk's rows (`gemv32m`), each token's top 8 (`route_top`)
  and the tokens grouped by expert into slots of 24 rows (`route_slots`: the tables every later step reads; unused slots marked -1,
  which the expert GEMMs skip) -- one graph a layer, nothing read back;
- the experts as one `gemm_gs(bsel)` launch of ~230 slot heads (24 rows each) for gate, up and down, the slot rows gathered and
  normed by `rms_gather`, the results weighted into rows (`slot_rows`) and summed onto the residual by DMA (`rows_sum`);
- every layout step by DMA (`inv_rms` / `rms_ad`, `resid_d`, `swiglu_d`); a layer's two halves are TinyJit graphs.

Where a 512-token chunk's ~180 ms a layer goes: the GEMMs ~135 ms (q|k|v 18, o 16, the experts' gate / up / down ~29 each), all
three DMA-bound -- an A operand is streamed once per 48-column group, and C makes a GSRAM round trip a K-slice. A slices-outer
order with A loaded once a block of groups (`k_gemm_gs(gblock)`) is in the fork but slower on the board (its C working set and
exposed latencies cost more than the A it saves), so it is off.

## Accuracy

| check | result |
| --- | --- |
| the numpy reference (`qwen3moe_ref.py`, float32 from ggml's dequantisation) vs llama.cpp (CPU, greedy) | identical text, two prompts (8 and 40 tokens) |
| Q4_K / Q6_K GEMMs on the model's tensors vs float64 (`check/check_qk_gemm.py`) | Q4_K 0.9-1.2e-3, Q6_K 2.5e-4 (rows 1, 4, 12; the head 2.5e-4) |
| every decode layer vs the reference (`qwen3moe_npu.py --gate 0-47`, the qdot layer with the q6n streams and the fp16 router, the routing on the device) | the same 8 experts in all 48 layers; worst error of a layer's update 9.3e-4 (the matrix unit's rows mode: 2.3e-3) |
| the qdot GEMVs on the model's tensors vs float64 (`check/test_qgemv.py`) | Q4_K 6.7e-4, Q6_K 1.9e-4 (two int8 activation levels; vs their integer reference 3-6e-6); the q6n stream 2.2e-5 and the q4n stream 2.6e-5 (`QG_KIND=q6n` / `q4n`: the scales in fp32, not fp16) |
| every prefill layer vs the reference on its input rows (`qwen3moe_prefill.py --gate 0-47`, 64 tokens) | 1e-3 a layer; 8 of 3072 (row, layer) pairs route a near-tied expert differently (worst row 17 %) |
| generation vs the reference | identical ids: the Portugal answer, the Fibonacci answer's first 40 tokens, a 64-token prompt's first 32 |
| qdot decode vs the matrix unit's (`QWEN3MOE_QDOT=0`) | identical ids: the Fibonacci answer (120 tokens), the Portugal answer, a 72-token prompt's 64 |
| the q6n / q4n streams, the folded kvd, the fp16 router (the defaults) vs the qdot path before them | identical ids on the Fibonacci (120) and Portugal prompts; on the lighthouse prompt (`check/ref_ids.py`) both qdot paths drop one near-tied token the reference and the matrix-unit path keep (token 19 of 64) |

The prefill attention rounds Q and K to fp16 for the matrix unit, the prefill GEMMs take fp16 activations and the decode GEMVs two
int8 levels; llama.cpp's own CPU path quantises activations to one 8-bit level a 256, so none is the float32 reference exactly. The rare routing flips are near-ties of two
experts' router scores.

## Run it

```sh
# the checkpoint (17.3 GiB, SHA-256 0d003f66...a8f5) and the model's tokenizer
curl -L -o /mnt/ssd/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf
mkdir -p /mnt/ssd/models/qwen3-30b-a3b-tokenizer
curl -L -o /mnt/ssd/models/qwen3-30b-a3b-tokenizer/tokenizer.json https://huggingface.co/Qwen/Qwen3-30B-A3B/resolve/main/tokenizer.json

python3 qwen3moe_pack.py                     # -> /mnt/ssd/qwen3-30b-a3b-npu, ~75 s with 4 jobs, 21 GB
python3 qwen3moe_generate.py "Write a Python function that returns the n-th Fibonacci number." --max-new 120
python3 qwen3moe_prefill.py --bench 3        # the pp512-style prefill benchmark
```

Needs ~22 GB of free RAM for the pinned weights (the board has 29 GB) and the driver's zero-copy weight buffers (the WBUF / SLOT
ioctls). The first prompt of a process captures the prefill's graphs (a few seconds). Options: `--max-new`, `--thinking` (Qwen3's
thinking block; off by default), `--tmax` (cache positions, 1024). Environment: `QWEN3MOE_GGUF` (the checkpoint: the embedding and the
reference read it), `QWEN3MOE_NPU` (the packed cache), `QWEN3MOE_TOK` (tokenizer.json's folder), `QWEN3MOE_PREFILL=step` (the
decode step a prompt token), `QWEN3MOE_JIT=0` (no graphs), `QWEN3MOE_PROF=1` (per-phase times), `QWEN3MOE_CPUS` /
`QWEN3MOE_CPU_LATENCY` (the host's CPUs and latency request, as the other examples), `QWEN3MOE_QDOT=0` (decode on the matrix unit),
`QWEN3MOE_QLEV=1` (one int8 activation level).

## Files

| file | what |
| --- | --- |
| `qwen3moe_ref.py` | the float32 numpy model from the GGUF (ggml's dequantisation): the reference every check uses |
| `qwen3moe_pack.py` | GGUF -> the per-layer NPU cache (each linear in its own format; the 128 experts as equal streams; the head in 4 parts) |
| `qwen3moe_npu.py` | one decode layer on the NPU; `--gate` checks layers against the reference |
| `qwen3moe_prefill.py` | the batched prefill; `--gate` (layers vs the reference), `--bench` (512-token chunks) |
| `qwen3moe_kernels.py` | the model's own TEC kernels: routing, the experts' gather / scatter, the matrix-unit attention's layouts, the DMA layout steps |
| `qwen3moe_qgemv.py` | the decode GEMVs as integer dot products over the gemm_gs streams, their activation packages, the decode layer's other kernels |
| `qwen3moe_qdec.py` | the decode layer on them (`QWEN3MOE_QDOT`, the default) |
| `qwen3moe_generate.py` | generation: the weights pinned zero-copy, a graph a decode layer, the head with the device top-1 |
| `check/` | the GEMMs on the model's tensors (`check_qk_gemm.py`), the expert slots (`check_bsel.py`), the kernels vs numpy (`test_kernels.py`, `test_attn.py`, `test_layout.py`), the qdot GEMVs and head (`test_qgemv.py`, `test_qhead.py`; `asm_qgemv.py`: their compiled bundles and spills, no device), the activation's int8 error (`act_quant.py`), the reference's ids for a prompt (`ref_ids.py`), the routing's statistics over generations (`route_stats.py`), prompt-lookup drafting's acceptance on recorded outputs (`pl_accept.py`), NPU and CPU DDR bandwidth alone and together (`bench_concurrent.py`, `membw.c`), CPU GEMVs on the model's rows (`host_head.c` NEON, `host_gemv.c` on llama.cpp's libggml-cpu: `test_host_head.py`, `test_host_gemv.py`), the device barrier (`bench_barrier.py`), timings (`bench_gemm.py`, `bench_layer.py`, `bench_qlayer.py`, `bench_gemm_full.py`, `bench_prefill_attn.py`, `bench_empty2.py`: a launch's floor) |

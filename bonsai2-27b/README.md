# Ternary Bonsai 2 27B on the Zhouyi NPU

Text generation with PrismML's [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) on the NPU of
the Radxa Orion O6N, through the `ZHOUYI` backend of tinygrad (the repository's `tinygrad/` submodule).

Bonsai 2 27B is [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) with its architecture unchanged and every linear
layer, the embedding and the output head stored as **ternary weights**: each weight is -1, 0 or +1, with one fp16 scale per
128 weights, in a blockwise Hadamard-rotated basis. The `PTQ1_0` GGUF is 5.9 GB, against 27 GB for Qwen3.8-27B in FP8. The model
and its GGUF are PrismML's, under the Apache-2.0 license; see their [model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf),
[website](https://prismml.com) and [whitepaper](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/bonsai-2-27b-whitepaper.pdf).

**Status: it generates.** Plain greedy decoding runs at 2.1 tok/s. With speculative decoding (the default) it runs at
**~6.2 tok/s**, and ~7.5 tok/s on code, with the same tokens as plain greedy decoding.

![Bonsai 2 27B served from the board, asked through the Ollama CLI on a Mac](serve-demo.gif)

*Served from the board ([Serve](#serve-openai--and-ollama-compatible-api)) and asked with `ollama run --verbose` on a Mac: 208 tokens at 4.8 tok/s.*

## What runs where

- **On the NPU:** all 64 layers (48 Gated DeltaNet + 16 full attention), the output head, the prompt's prefill, and the draft
  model of speculative decoding. Every linear runs on a **ternary GEMM** that reads the 2-bit codes directly (0.28 bytes a
  weight against FP8's 1.03), so a token streams ~8 GB of weights instead of 26 GB.
- **On the host:** the tokenizer, the prompt's embedding and its first token, and reading the generated ids to print them. In
  speculative decoding every later token id stays on the NPU (`QWEN_NPU_IDS=1`, the default): the head picks it
  (`head_reduce`), the embedding is decoded and rotated there (`embed_tern`), the drafts are accepted (`accept`) and fed back to the
  drafter (`pick_id`, `mtp_in`); the host only reads the ids and the draft probabilities to choose the next pass's size.

The model runs on the same modules as [qwen3.8-27b](../qwen3.8-27b/) with `QWEN_MODEL=bonsai2-27b`. Only the weight format and
the input transform differ (see "How it works" below).

## Requirements

The backend's requirements, listed in the [root README](../README.md#requirements-on-the-board):

- the Zhouyi NPU kernel driver with its header `armchina_aipu.h` and its zero-copy weight buffers (the WBUF / SLOT ioctls),
- libclang for the first run (`LIBCLANG_PATH` if your distribution ships only a versioned library),
- the vendor's AIPU toolchain library,
- Python 3.12 or newer with `numpy` and `tokenizers` (`pip install numpy tokenizers`).

On top of that: **~15 GB of disk** (5.9 GB GGUF, 0.5 GB of Qwen3.8-27B files, 8.2 GB packed cache) and **~10 GB of free RAM**
(the packed layers are held in RAM; the board has 29 GB).

The paths below (`/mnt/ssd/...`) are examples and the scripts' defaults: set `BONSAI_DIR`, `BONSAI_GGUF` and `QWEN_NPU` to your
own paths.

## Download

```sh
cd bonsai2-27b
BONSAI_DIR=/mnt/ssd/bonsai2 bash download.sh      # about 6.4 GB; resumable; --dry-run lists the files
```

This fetches two things into `$BONSAI_DIR` (default `/mnt/ssd/bonsai2`; set it to your path):

- from [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf): `Ternary-Bonsai-2-27B-PTQ1_0.gguf`
  and its license files;
- from [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8), into `$BONSAI_DIR/qwen3.8-27b-mtp`: only
  `mtp.safetensors` (0.48 GB, the draft model's weights; see below), `tokenizer.json` and the small config and license files.
  Not the 27 GB checkpoint.

A Hugging Face token is not needed; if `~/.hf_token` exists, it is sent.

## Pack

The packer converts the GGUF into the per-layer cache the NPU streams (`QWEN_NPU`, default `/mnt/ssd/bonsai2-npu`; set it to
your path). Times are on the board:

```sh
export BONSAI_GGUF=/mnt/ssd/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf QWEN_NPU=/mnt/ssd/bonsai2-npu
python3 bonsai2_pack.py                                    # the 64 layers, ~3 min
python3 bonsai2_pack.py --head                             # the output head and the final norm, ~10 s
python3 bonsai2_pack.py --fuse                             # the same-input projections as one GEMM each, ~1 s
python3 bonsai2_pack.py --mtp /mnt/ssd/bonsai2/qwen3.8-27b-mtp   # the drafter and tokenizer.json, ~4 s
```

`--mtp` packs Qwen3.8-27B's multi-token-prediction (MTP) layer into the subfolder `$QWEN_NPU/mtp`, where the generator looks for
it, and copies the tokenizer into the cache. The MTP layer stays in FP8 (block-scaled E4M3), so it has its own folder with its
own format markers.

The scale tables are stored as fp16 (2.125 bits a weight instead of 2.25 for fp32, 5.6 % fewer bytes a pass; the GEMM's output
is bit-identical, so the tokens are the same). `--tscale f32` packs the older fp32 tables. A cache packed with fp32 tables can be
converted into a new folder (no GGUF read, ~15 s on the board; the tool stops if any scale is not exact in fp16):

```sh
python3 bonsai2_repack_f16.py --src /mnt/ssd/bonsai2-npu --out /mnt/ssd/bonsai2-npu-f16   # then QWEN_NPU=/mnt/ssd/bonsai2-npu-f16
```

The cache's `tscale` file selects the format; a cache without it is read as fp32.

## Run

```sh
python3 bonsai2_generate.py "Write a Python function that returns the n-th Fibonacci number." --max-new 120
QWEN_SPEC=0 python3 bonsai2_generate.py "Explain why the sky is blue to a ten-year-old." --max-new 120   # plain decoding
```

The prompt goes through the chat template (thinking off; `--thinking` turns it on). Decoding is greedy until `<|im_end|>` or
`--max-new`. `--raw` skips the template, and `--out ids.npz` saves the generated ids. Model set-up takes ~20 s a run (9 s for
plain decoding), and the prefill runs at ~1.6 tok/s (24 tokens in 15 s).

Without the drafter in `$QWEN_NPU/mtp`, the generator says so and decodes plainly.

## Serve (OpenAI- and Ollama-compatible API)

`bonsai2_serve.py` loads the model once and serves it over HTTP with Python's standard library (no extra packages). It speaks
two APIs on one port:

- **OpenAI:** `GET /v1/models`, `POST /v1/chat/completions` (system / user / assistant messages; streaming with `"stream": true`,
  and the token counts at the end with `stream_options.include_usage`) and `POST /v1/completions` (raw text, no chat template).
- **Ollama:** `HEAD /`, `GET /api/tags`, `GET /api/version`, `POST /api/show`, `POST /api/chat` and `POST /api/generate`
  (newline-delimited JSON, streaming by default; `"stream": false` for one reply; `options.num_predict` and `options.stop`).
  Every route is also accepted under a `/v1` prefix, so a client given the OpenAI base URL still finds them.

The generic server is [`qwen3.8-27b/qwen38_serve.py`](../qwen3.8-27b/qwen38_serve.py); `bonsai2_serve.py` is that server with
this example's paths, drafter and decoding defaults (the [Settings](#settings) below).

### Start

```sh
cd bonsai2-27b
export BONSAI_GGUF=/mnt/ssd/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf QWEN_NPU=/mnt/ssd/bonsai2-npu   # your paths
python3 bonsai2_serve.py --host 0.0.0.0 --port 8000 [--tmax 2048]
```

- `--host` defaults to `127.0.0.1` (this machine only); `0.0.0.0` accepts connections from the network.
- `--tmax` is the context in tokens, prompt plus output (default 2048).
- `--model-name` (or `QWEN_SERVE_NAME`) sets the name the server reports (default `bonsai2-27b`).

The model is ready when the server prints `== serving bonsai2-27b on http://...`.

### Run it as a service

An example systemd unit, `/etc/systemd/system/bonsai2.service` (replace `<user>` and `<path>`):

```ini
[Unit]
Description=Ternary Bonsai 2 27B on the Zhouyi NPU (OpenAI / Ollama API)
After=network-online.target

[Service]
Type=simple
User=<user>
WorkingDirectory=<path>/tinygrad-npu-examples/bonsai2-27b
ExecStart=/usr/bin/python3 bonsai2_serve.py --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

If your model files are not at the default paths, add them to `[Service]`, e.g.
`Environment=BONSAI_GGUF=/data/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf QWEN_NPU=/data/bonsai2-npu`. Then:

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now bonsai2.service
journalctl -u bonsai2.service -f          # the log; the model is ready at "== serving ..."
```

### The CPU-latency request

While it runs, the decode process holds a 0 µs CPU-latency request on `/dev/cpu_dma_latency` (`QWEN_CPU_LATENCY`, default `0`;
`off` disables it). Without it, the CPU that takes the NPU's interrupt drops into deep idle states between a job's kernel groups,
and decoding is slower. The device is root-only by default; the process then prints a note
(`QWEN_CPU_LATENCY: no PM QoS request ...`) and runs without it. To let the `users` group hold the request, add a udev rule:

```sh
echo 'KERNEL=="cpu_dma_latency", GROUP="users", MODE="0660"' | sudo tee /etc/udev/rules.d/60-cpu-dma-latency.rules
sudo udevadm control --reload && sudo udevadm trigger --name-match=cpu_dma_latency
```

The user running the server must be in the `users` group (`id <user>`; `sudo usermod -aG users <user>`, then log in again).
The request is released when the process exits.

### Reaching the board by name

With Avahi (mDNS) running on the board (`sudo apt install avahi-daemon` if it is not), clients on the same network reach it by
its host name as `http://<hostname>.local:8000`. Below, `<board>` stands for that name or the board's address.

### Consuming with Ollama clients

The Ollama CLI talks to the board when `OLLAMA_HOST` points at it:

```sh
export OLLAMA_HOST=http://<board>:8000
ollama list                               # bonsai2-27b
ollama run bonsai2-27b                    # an interactive chat
ollama run --verbose bonsai2-27b "Write a Python function that returns the n-th Fibonacci number."   # + the eval rate
```

`--verbose` prints the prompt's processing time and the generation's eval rate (the tokens after the first over the time after
the first token).

GUI apps that speak Ollama's API (e.g. Enchanted): set the server URL to `http://<board>:8000`.

With curl:

```sh
curl http://<board>:8000/api/chat -d '{"model": "bonsai2-27b", "messages": [{"role": "user", "content": "Hello"}]}'
curl http://<board>:8000/api/chat -d '{"model": "bonsai2-27b", "messages": [{"role": "user", "content": "Hello"}], "stream": false}'
curl http://<board>:8000/api/generate -d '{"model": "bonsai2-27b", "prompt": "Why is the sky blue?", "options": {"num_predict": 128}}'
```

`/api/generate` applies the chat template (with `system` as the system turn, if given); `"raw": true` sends the prompt as is.

### Consuming with OpenAI clients

With curl, streaming and not:

```sh
curl http://<board>:8000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model": "bonsai2-27b", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
curl http://<board>:8000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model": "bonsai2-27b", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 128}'
```

With the `openai` Python package (`pip install openai`); the API key is not checked, any value works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://<board>:8000/v1", api_key="none")
stream = client.chat.completions.create(
    model="bonsai2-27b",
    messages=[{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Behaviour

- **Greedy decoding.** `temperature`, `top_p` and `seed` are accepted and ignored; `n` must be 1. `max_tokens` /
  `max_completion_tokens` (Ollama: `options.num_predict`; default 512) and `stop` work.
- **One request at a time** runs on the NPU; others wait in line. The tokens stream as the verify passes accept them.
- **The context is `--tmax`** tokens (default 2048), prompt plus output; a longer prompt is refused, and the output stops at the
  context's end.
- **Prompt speed.** The prompt goes through the verify path, 4 tokens a pass, at ~7 tok/s, so a long chat history delays the first
  token (each request processes the whole conversation again). Its tokens can differ from the one-pass prefill's where two
  logits nearly tie; on the Fibonacci prompt the 120 tokens equal plain greedy decoding's.
- **Generation speed** depends on how predictable the text is: ~7.5 tok/s on code, ~4.8 tok/s on free prose.
- **Thinking** is off unless the request turns it on: `"chat_template_kwargs": {"enable_thinking": true}` (OpenAI) or
  `"think": true` (Ollama).

## Speed

Measured on the board with the NPU clocks at their defaults, the host on CPUs 0 and 1 (`QWEN_CPUS=0,1`: CPU 0 takes the NPU's
interrupt; the default `auto` picks it and one CPU of the same speed) and holding the 0 µs CPU-latency request
([above](#the-cpu-latency-request)), 120 new tokens each, wall time after the first token, the prefill excluded. Every
speculative run returned exactly the plain greedy tokens.

The published backend and the generator's defaults (fp16 scale tables, the leaf tree, kernel code in GM), on three of
qwen3.8-27b's exploration prompts (plain: 40 tokens), one session, the board shared with other jobs (not running at the same
time):

| prompt | plain | speculative, default drafter |
| --- | ---: | ---: |
| e0 Fibonacci in Python | 2.16 | 7.47 |
| e4 a short story's opening | 2.13 | 4.78 |
| e8 a train journey's length, step by step | 2.13 | 6.96 |
| **the three together** | **2.14** | **6.17** |

The previous release's build, for comparison:

| prompt | plain (40 tokens) | speculative, defaults |
| --- | ---: | ---: |
| Fibonacci in Python | 2.03 | 6.71 |
| a short story's opening | | 4.28 |
| a train journey's length, step by step | | 6.28 |
| **the three together** | | **5.58** |

Over ten varied prompts that build averaged 5.34 tok/s.

Code and step-by-step reasoning draft best; free prose drafts worst.

## Speculative decoding

The GGUF has no draft model, so the drafter is Qwen3.8-27B's MTP layer (see "How it works"). Each pass verifies up to 4 rows: the
current token, the drafter's chain (it keeps drafting while the chain's probability is at least `QWEN_TREE_TC`), and in the rows
left, the drafts' second and third candidates as extra leaves, which cost no extra draft pass. Every emitted token is the verify
pass's own argmax, so the output is the same as plain greedy decoding.

### Settings

`bonsai2_generate.py` sets these defaults. Everything else is shared with qwen3.8-27b and documented in the
[root README](../README.md#configuration-qwen38-27b-bonsai-2-and-ornith).

| variable | default | what it does |
| --- | --- | --- |
| `BONSAI_GGUF` | `/mnt/ssd/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf` | the checkpoint (the host reads the embedding rows from it). Set it to your path |
| `QWEN_NPU` | `/mnt/ssd/bonsai2-npu` | the packed cache. Set it to your path |
| `QWEN_TOK` | the cache | the folder holding `tokenizer.json` |
| `QWEN_MTP_CACHE` | `$QWEN_NPU/mtp` | the drafter's cache: Qwen3.8-27B's MTP layer, in FP8 with single-layout scales |
| `QWEN_MTP_HEAD` | `bonsai` | the draft head: Bonsai's own ternary head, or `qwen`: Qwen3.8-27B's FP8 head, read from `QWEN_MTP_CACHE` |
| `QWEN_MTP_EMBED` | `bonsai` | the drafted token's embedding: Bonsai's, or `qwen`: Qwen3.8-27B's table, from `QWEN_MTP_DIR` |
| `QWEN_MTP_DIR` | unset | Qwen3.8-27B-FP8's folder (its `outside.safetensors`), needed only with `QWEN_MTP_EMBED=qwen` |
| `QWEN_SPEC` | `4` | rows of a verify pass at most (the token + up to 3 drafts); `0` decodes plainly |
| `QWEN_SPEC_GEOS` | `2,3,4` | the verify pass sizes prepared at start-up (with the leaf tree: every size from 2 to `QWEN_SPEC`) |
| `QWEN_DRAFT_MAX` | `3` | drafts a pass at most |
| `QWEN_SPEC_TREE` | `leaf` | the verify rows the chain leaves free take the drafts' 2nd / 3rd candidates; `off`: the chain only |
| `QWEN_TREE_TC` | `0.5` | with the leaf tree: keep drafting while the chain's probability is at least this |
| `QWEN_DRAFT_TAU` | `0.2` (`0.4` with the Qwen head) | without the leaf tree (`QWEN_SPEC_TREE=off`): keep drafting while the draft head's probability is at least this |
| `QWEN_DRAFT_PARTS` | `2` (`2,thresh:20` with the Qwen head) | the draft head reads the first 2 of the head's 4 parts (token ids < 124 416) |
| `QWEN_CPUS`, `QWEN_CPU_LATENCY` | `auto`, `0` | the host's CPU pinning and the CPU-latency request ([above](#the-cpu-latency-request)) |

`QWEN_SPEC_TREE` is not available for this model.

**The Qwen-head option (+3 %).** If you already have Qwen3.8-27B packed for the [qwen3.8-27b](../qwen3.8-27b/) example, with
single-layout scales (`qwen38_pack.py --scales single`, or `qwen38_repack_scales.py`), its FP8 head (`--lm-head-fp8`) and its MTP
layer (`--mtp`), the drafter can use Qwen3.8-27B's own head, which the MTP layer was trained with:

```sh
QWEN_MTP_CACHE=/mnt/ssd/qwen3.8-27b-npu-s1 QWEN_MTP_HEAD=qwen QWEN_MTP_EMBED=qwen QWEN_MTP_DIR=/mnt/ssd/qwen3.8-27b-fp8 \
  python3 bonsai2_generate.py "Write a Python function that returns the n-th Fibonacci number." --max-new 120
```

It needs the 27 GB checkpoint and its pack, plus 0.66 GB more RAM for the head's two parts. Measured on ten prompts with an
earlier, slower ternary GEMM: plain 1.07 tok/s, the default drafter 2.39 tok/s, the Qwen head 2.47 tok/s. The embedding switch
alone is worth under 1 %.

## How correctness is checked

- **The reference.** `bonsai2_ref.py` is a numpy fp32 forward pass of the GGUF. On the reference prompt (12 tokens) its greedy
  continuation matches PrismML's reference implementation for 10 of 10 tokens, and its PTQ1_0 decoder is bit-identical to
  that implementation's.
- **The pack.** `check/check_pack.py --cache $QWEN_NPU [--layers 0-3] [--head]` (no device) unpacks every stream of the cache and
  requires the GGUF's decoded weights back after the documented folding: the codes, the scales, the padding, the fused files,
  the small tensors, the Hadamard signs and the head parts.
- **The NPU's arithmetic against the reference.** `check/emulate_layer.py --cache $QWEN_NPU --layers 0,3` (no device) emulates
  what the device does with the cache -- the Hadamard-rotated fp16 inputs, the ternary GEMM's accumulation order and scales --
  inside the reference's layer math, and compares it with the GGUF reference. On the board, single layers are within 2.2e-4 of
  the reference's layer output at 1 to 21 rows, and greedy decoding on the NPU gives the same 10 tokens 10/10, with every step's
  top logits within 0.002 of the numpy reference's.
- **Speculative against plain.** Every token is the verify pass's own argmax, so speculative decoding must return exactly the
  ids of plain greedy decoding. `check/verify_oracle.py` (on the board) decodes 40 tokens plainly, then runs the verify / commit
  loop with oracle drafts (every third pass with a wrong last draft, so passes reject too) over the verify geometries, and
  prints `EXACT` when the ids are equal. On the board this held on 70 runs (7 drafter configurations x 10 prompts) and on every
  run of the speed table.

```sh
python3 check/check_pack.py --cache $QWEN_NPU --head
python3 check/emulate_layer.py --cache $QWEN_NPU --layers 0,3
QWEN_SPEC=8 QWEN_SPEC_GEOS=3,4,6,8 python3 check/verify_oracle.py
```

## The smaller sizes: Ternary Bonsai 2 2B, 4B and 9B

The same runtime with `QWEN_MODEL=bonsai2-2b | -4b | -9b` (`bonsai2_weights.PATHS` holds each size's GGUF and cache; the Hadamard block is
the GGUF's: 512 / 512 / 4096). Their drafters are MTP layers of their own size: the 9B's and 4B's from PrismML's
`Ternary-Bonsai-2-{9B,4B}-PQ2_0-MTP-Q8_0.gguf` (`bonsai2_pack.py --mtp-gguf`), the 2B's from Qwen3.5-2B's own `mtp.*` tensors
(the Bonsai 2 2B is Qwen3.5-2B's shape exactly; the 122 MB of bf16 tensors fetched from its checkpoint by range requests into a
safetensors file, then `--mtp-gguf` on that file). Board o6n, 2026-10-26, three prompts (a 9-token answer, a 64-token code answer, a
~60-token explanation), the speculative ids identical to plain greedy's on every prompt:

| size | resident | prefill tok/s | decode plain | decode speculative | drafts accepted a pass |
| --- | --- | --- | --- | --- | --- |
| 2B | 0.6 GB | 50-60 | 10.0 (the verify-geometry plain path; 22.8 on the one-row batch loop) | **28-35** (23.6-29.5 before the producer rework) | 1.1-1.9 of 3 (leaf tree, 4 rows) |
| 4B | 1.5 GB | 29-35 | 6.9 | **16.2-24.2** | 1.0-2.0 |
| 9B | 2.7 GB | 15-17 | 4.5 | **9.2-10.7** | 1.5-1.9 |

The prompt goes through the verify path (`QWEN_PREFILL=chunked`: the one-pass prefill's DeltaNet kernel wants 48 value heads). Plain
decoding on these sizes is `QWEN_DRAFT=none` (the verify path without drafts); `QWEN_SPEC=0` has no geometries for the chunked prefill.

## Batched decode: several requests a step (`bonsai2_batch.py`)

N independent requests through one decode step: the linears -- the weight bytes -- run once for all N rows on the GEMM's rows mode (a
Layers(N) geometry of the runtime: its A producers, residuals and dense MLP as the verify pass uses them), and the sequence mixers run
as ONE launch a layer on batched one-token kernels: `gdn_tokb` (qwen3.8-27b/gdn_batch.py) and `attn_decb` (attn_batch.py) are the
one-row kernels gdn_tok3 / attn_dec3 over M independent rows -- units (row, head group) / (row, kv head) over the 12 tasks, row r
reading tile row r of the C tiles, its own input row, its user's state / ring / cache (the per-layer stacks replicated per user, index =
user x layers + layer) and position, writing row r of the output rows. Each row's results are the one-row kernel's bits (simulator
gates gdn_batch_sim.py / attn_batch_sim.py), so each user is plain greedy decoding of its own request: user 0's ids equal the
single-request run's at every N. One JIT graph a layer per (type, parity). `BATCH_KERNELS=0`: the first form, the mixers per user on
the one-row kernels joined by scatter / gather kernels. Board o6n, 2026-10-26, Bonsai 2 2B, prompts of 17 tokens in lockstep, 24 new tokens:

| users | a step | aggregate tok/s | a user | prefill (lockstep) | per-user mixers (BATCH_KERNELS=0) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 40-52 ms (53 before the reworks) | 19-25 | 19-25 | 8.4 tok/s | 52 ms, 19.2 |
| 2 | | | | | 61 ms, 33.0 |
| 4 | 48-62 ms (80 before the reworks) | **64-84** | 16-21 | 32.9 | 108 ms, 37.1 |
| 8 | 71-85 ms (123 before the reworks) | **94-112** | 12-14 | 38.6 | 182 ms, 43.9 |
| 12 | 216 ms | 55.7 | 4.6 | 37.2 | |

Eight users is the sweet spot: 3.4x one stream's throughput. Past 8 the GEMM's third row tile (rows 9-12) and the 96 DeltaNet units
over 12 tasks cost more than the extra rows return. The remaining growth, ~9 ms a user a step at 4-8 users, is the DeltaNet units
(each ~0.3 ms: the a | b projection and the 1 MB state through LSRAM, 8 head groups a user over 12 tasks) and the GEMMs' second row
tile; the single-user step (53 ms, 18.8 tok/s) is also the 2B's best plain decode (the plain path through the 4-row verify
geometry measured 100 ms).

**Where a row costs, and the producer rework** (`gemm_parts_bench.py`: the 2B's kernels alone, a JIT replay of 10 launches, ms a launch;
before -> after `had_rows`):

| rows | gu GEMM (12288 x 2048) | dn GEMM | qkv\|z GEMM | o GEMM | had_a32 (an A producer) | swiglu_a | resid | unpack |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.39 | 0.25 | 0.28 | 0.14 | 0.14 -> **0.09** | 0.18 -> 0.19 | 0.06 | 0.06 |
| 4 | 0.37 | 0.24 | 0.24 | 0.14 | 0.31 -> **0.13** | 0.20 -> 0.24 | 0.04 | 0.03 |
| 8 | 0.44 | 0.28 | 0.32 | 0.12 | 0.57 -> **0.16** | 0.41 -> **0.33** | 0.07 | 0.04 |
| 12 | 1.03 | 0.61 | 0.70 | 0.27 | 0.76 -> **0.22** | 0.48 -> **0.45** | 0.07 | 0.09 |

The ternary GEMMs are flat from 1 to 8 rows (the codes are expanded once a K-slice and reused across the row tiles; gu moves its 6.3
MB of codes at ~16 GB/s) and double at 12 rows, where the third row tile leaves the LSRAM-accumulator fast path (rt <= 2: the 32
vector registers hold 8 accumulators a row tile beside the operands and the expand, so a third tile does not fit). What grew with the
rows was the A producers: the Hadamard transform's unit was one 1024-block of K (2 for H 2048) running every row quad in turn, so the
rows serialised on 2 of the 12 tasks. `had_rows` (qwen38_kernels.py, `QWEN_HAD_ROWS`, the default for the 512-block sizes) makes a
unit a (row, block): the row by one DMA into LSRAM, the block transformed there in had_a32's exact order, drained by one DMA into the
row's own range; the A layout is then the plain DMA packer's. "swiglu" keeps (row quad, block) units with the gate | up tiles gathered
by DMA a group at a time. Bit-identical to had_a32 on the simulator in every mode (the gate `had_rows_sim.py`). Board, the 2B:
batched decode 8 users 123 -> **101 ms a step (79 tok/s aggregate)**, one user 53 -> 44 ms (22.8 tok/s); speculative decode 24 / 29.5 /
24.3 -> **28 / 35 / 30 tok/s**; the 4B's 21.4 -> 24.2 on code. The 27B's 1024-block producers (had_a32d / had_fast) stay faster than this
path (7.3 vs 6.6 tok/s), so there it is opt-in.

**The batched DeltaNet kernel's data movement** (`mixers_bench.py`: gdn_tokb alone at 1 / 4 / 8 rows): the first form read the conv's ring
rows and taps, the row's q / k / v / z tile segments and the gated-norm's z by plain DDR loads and wrote the ring by plain stores -- some
320 cache-line round trips a unit, the kernel's dominant cost (1.38 ms a layer at 8 rows). `gdn_tokb_src(stage=True)` moves them by DMA:
the ring and taps segment (two requests, gdn_tokb_desc slot 3), the row's segments as whole row-tile strip pieces (256 B: 4 column quads
x 4 rows, one request a group, slots 4-6) compacted in LSRAM, the ring writes as 512-B drains (slot 7). Bit-identical on the simulator
(gdn_batch_sim.py); on the board 0.33 / 0.76 / 1.38 -> 0.24 / 0.54 / 1.02 ms at 1 / 4 / 8 rows. What remains a unit is its a | b
weight chunks (16 requests a row: the same weights every row of a head group) and the state blocks' fill / drain pairs.

**Layer blocks** (`BATCH_BLOCK`, default 4): as the runtime's verify pass, four layers run as one JIT graph and one job, each layer's
weights in a window slot of its own (`Layers.block_slots` / `map_block`), so a token costs 6 jobs instead of 24. At 8 users 93-94 ->
71-85 ms a step (one layer a job vs blocks, back to back); at 1 user the gain is within the noise. The board's step times vary by
~15 % between runs of the same configuration (71 / 84 / 85 ms in three 8-user runs), so the figures above are ranges over 2-3 runs.

**The 4B and the 9B on the same loop** (board o6n, 2026-10-27, prompts of 17 tokens, 32 new; user 0's ids = the single-request run's
at every N). The producers now also move the norm weights by DMA (a sub-block at a time, accumulated in the mix's order) and gather
the 9B's four mixed sub-blocks by DMA, so the 9B's 4096-block producers went from 1.31 / 1.42 ms (had_a32 / swiglu_a at 8 rows) to
0.24 / 0.94, and `QWEN_HAD_ROWS` is on by default for them too (the 27B's tuned 1024-block producers remain the exception):

| size | 1 user | 4 users | 8 users | speculative (code prompt) |
| --- | ---: | ---: | ---: | ---: |
| 2B | 49 ms, 20.5 tok/s | 48-62 ms, 64-84 tok/s | 86 ms, 93 tok/s | 35 tok/s |
| 4B | 88 ms, 11.4 tok/s | 116 ms, 34.5 tok/s | 165 ms, 48.5 tok/s | 24.2 tok/s |
| 9B | 145 ms, 6.9 tok/s | 171 ms, 23.4 tok/s | 282 ms (399 before the producer DMA), 28.4 tok/s | 14.0 tok/s (10.7 before) |

Where the 9B's 282 ms at 8 users go: the GEMMs ~120 ms over its 32 layers (its 2.5 GB of codes a token are ~110 ms at the port: the
floor), the batched DeltaNet kernel ~50 (2.1 ms a layer: 11 head groups x 8 rows of units, each re-fetching its a | b weight chunks),
the producers ~45 (swiglu_a's four-sub-block gather 0.94 ms a layer), the rest residuals, unpacks and the jobs' gaps.

**Where the step stands against the port** (`BATCH_PROF=1`: the host, each 4-layer job and the head timed with a sync after each; the
2B at 8 users: jobs 13.4-13.7 ms each, the head 11.2, the host ~2.5; at 1 user jobs 7.3, the head 7.3). A step's bytes are the weights
(0.54 GB for the 2B, 1.2 the 4B, 2.2 the 9B) plus the DeltaNet states, read and written once a user a layer (1 MB a layer on the 2B,
2 MB on the 4B / 9B: 0.29 / 0.6 / 0.77 GB at 8 users), so the 8-user floors at 23 GB/s are 36 / 78 / 130 ms against 85-90 / 165 / 282
measured: ~40-47 %. Inside that: the GEMMs run at 16-20 GB/s (the gu GEMM 0.43 ms for 6.3 MB at 8 rows), the batched DeltaNet kernel
at ~80 % of the port for its state traffic (its a | b phase is 10-20 % of it: `gdn_diag_bench.py`), the head at its bytes at 1 row and
+4 ms for 8. The rest is the producers (~12 ms a step on the 2B), residuals / unpacks (~2), and the gaps between the ~15 launches a layer
(~0.6 ms a layer, ~15 % of the step; eight layers a job instead of four changes nothing, so the gaps are per launch, not per job).
Pairing the conv staging's requests on two flags changed nothing either: that phase is its bytes.

**Why batching and speculative decoding are not combined here.** A row spent on a draft returns a token ~45 % of the time, a row
spent on another user's token always; with the row costs above, plain batching beats any mix of users and drafts in a pass on this
model. The combination pays on models whose rows are nearly free (the Q8_0 Gemma 4 E2B: a 7-row verify pass costs 1.07 plain steps).

## Serving the smaller sizes, and several users at once

`QWEN_MODEL=bonsai2-2b python3 bonsai2_serve.py --port 8000` (or -4b / -9b) serves that size over the same OpenAI- and Ollama-compatible
endpoint as the 27B, with its own drafter (speculative decoding, the tokens of plain greedy): one request at a time, others queue.
Board o6n, 2026-10-27, a 22-token prompt and a 23-token answer: the 9B in 3.1 s wall, the 4B 2.1 s, the 2B under 2 s.

`bonsai2_batch_serve.py` is the same endpoint with concurrent requests **batched into one decode step** (the loop of `bonsai2_batch.py`
behind a slot scheduler): up to `--users` requests advance together, each in a slot with its own DeltaNet states, conv ring, K / V cache
and position; a new request takes a free slot (its state zeroed) and decodes plain greedy, streaming. A short prompt (under 48 tokens)
goes in a token a step in its slot; a longer one goes through the model's chunked prefill (4 tokens a pass, one prompt at a time, a
pass between two steps while the others decode) and its states and cache rows are copied into the slot when it is in. The step costs
the geometry's whatever the number of busy slots (the 2B: ~50 ms at 4 slots, ~75 at 8), so one user sees one token a step and N users
N. Board o6n, the 2B, `batch_load.sh PORT N MAX_TOKENS` (N `curl`s in parallel, the aggregate tok/s printed), wall time including each
request's prompt:

| slots | requests | completion tokens | wall | aggregate (prefill included) | decode alone |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 22 | 2.2 s | 10 tok/s | 20 tok/s a step |
| 4 | 4 | 166 | 3.6 s | **46.5 tok/s** | 60-84 tok/s |
| 4 | 6 (two wait) | 181 | 5.2 s | 34.8 tok/s | |
| 4 | 8 (four wait) | 288 | 6.7 s | 43.3 tok/s | |
| 8 | 8 | 346 | 6.3-6.8 s | **51-55 tok/s** | 94-112 tok/s |
| 8 | 1 | 22 | 3.1 s | 7.1 tok/s | 12 tok/s a step |

So: `bonsai2_serve.py` for one user at a time (speculative, ~30 tok/s on the 2B), `bonsai2_batch_serve.py --users 4` when several agents
or users share the board (each at 15-20 tok/s while 4 run, ~60 in all). The two are separate processes; the NPU time-slices them.

### Long contexts

The attention kernels used to hold a row's scores in LSRAM (32 KiB a task): 512 positions for the batched one-token kernel, about 4k
for the sliced verify-pass kernel (its slices multiply until the combiner's records overflow). Both now stream the positions in
**chunks with an online softmax** -- the chunk's scores in LSRAM, a running max and sum per head, the partial output rescaled by
exp(m_old - m_new) when the max grows -- so the kernels take any context (`attn_plan`: the exact path up to 16 slices, then 12 slices
in chunks of 168; the one-row kernel chunks of 256 past 512). Simulator gate: the chunked kernels against float64 (rel. error 1-2e-5),
the batched kernel bit for bit against the one-row kernel. The 2B answers this correctly at a 64k context, a 6283-token prompt
("how many days are logged above?" over 120 repeated paragraphs: 120), prefilled in 69 s (batched server, 90 tok/s) to 87 s (the
speculative server, 72 tok/s): the prefill is 4 tokens a pass, so long prompts take minutes.

**The NPU's address window, and the caches in the other ASIDs.** Every device pointer a kernel holds is a 32-bit offset in a 4 GiB
window (ASID 0), and about 2.75 GB of it can hold buffers and the slots the weights stream through. As fp32 buffers there, the K / V
caches (24 KiB a position on the 2B, 64 KiB on the 4B / 9B) would have bounded the context at 64k for one user of the 2B, ~21k for the
9B, 15k for 4 batched users. But the part has **four** such address spaces, 3 GiB usable each, and the TEC's DMA instruction carries a
base code (`kGlobal1..3`) that selects one: a transfer with that code reads or writes the matching ASID at the 32-bit address. Every
attention kernel moves the cache by DMA alone, so the caches now live in **ASIDs 1-3** (`QWEN_KV_ASID=1`, the default for the smaller
sizes): buffers requested in those ASIDs (the KMD's `REQ_BUF` takes the ASID; the chain head carries the four bases,
`ZHOUYI_ASID_BASES`), attention layer i in ASID 1 + i mod 3, the kernels' cache transfers with the base code, the layer's region as the
kernels' cache index, and nothing of the ASID 0 window spent. Probes on the board: a kernel's DMA with base 1 reads the ASID 1 buffer
bit for bit and its writes reach the host, the same address through base 0 reads something else; the caches in the ASIDs equal the
buffers' row for row and the tokens are the same (2B at 512, 4096 and 65536). `--tmax auto` is then the largest context whose fullest
ASID holds its layers' caches (`QWEN_TMAX_MAX` caps it, 65536 by default):

| server, caches in ASIDs 1-3 | 2B | 4B / 9B |
| --- | ---: | ---: |
| `bonsai2_serve.py` (speculative, one user) | 65536 (262144 would fit) | 97280 with `QWEN_TMAX_MAX=131072` |
| `bonsai2_batch_serve.py --users 4` | 65536 | 31744 |
| `bonsai2_batch_serve.py --users 8` | 48128 | ~15k |

Board o6n, 2026-10-27: the 2B's batched server at 65536 (4 users) answers the 6283-token prompt in 68 s and runs 4 concurrent requests
at 46 tok/s, 8 users at 48128 at 53 tok/s; the 9B at 97280 and the 4B at 65536 answer the 3142-token prompt ("60 days") in 127 s and
78 s; the 9B's batched server (4 users, 31744) runs 4 concurrent requests at 14 tok/s. One price: the speculative server runs the
chain verify, not the leaf tree (its commit moves K / V rows in every layer's cache through ASID 0 pointers): about 4 % fewer tokens a
pass.

**A driver fix was needed for ASIDs 2 and 3.** The KMD's IOVA allocator searches downward from its last 64-bit insertion, device-wide,
so once anything sat in ASID 1 a request for ASID 2 or 3 was placed in ASID 1's range and refused by the range check. The slim driver
now starts the search at the domain's top for ASIDs above 0 (the never-map zones bound each slot from below); the code spreads the
layers over whatever ASIDs allocate, so with the stock allocator it falls back to ASID 1 alone (a third of the capacity). The 27B keeps
its 2048 default and its buffers (its one-pass prefill's part kernels are the exact path's).

## How it works

- **The ternary GEMM.** The backend's `gemm_gs(tern=True)` streams each weight as a 2-bit code with one fp16 scale per row and
  128-wide K block. The GGUF's `PTQ1_0` block is 128 weights along K with one scale, so it maps onto one K slice of the GEMM and
  its scale is stored as is. The kernel expands the codes to fp16 before multiplying; at 1 to 4 rows it keeps up with the
  weight stream. A verify pass through the 64 layers costs 0.37-0.39 s at 2 to 4 rows.
- **The Hadamard producers.** Bonsai stores each weight as W' = W diag(s) H, with H a 1024-point Walsh-Hadamard transform
  and s fixed +-1 signs per input width, so each linear's input must be rotated the same way. The kernels that already write
  each GEMM's fp16 input (after the RMSNorms, after SwiGLU, after attention) run the 10 butterfly stages of the transform in
  local memory on the way (`had_a32` in `qwen38_kernels.py`; `had_a32d` in decode and verify, which moves every operand
  by DMA and does the in-register butterfly stages with lane extracts: 39-106 us a call at 4 rows). The 1/32 normalisation is
  folded into the weight scales. The signs are folded into the norm weights where that is exact and multiplied in where it is
  not. The cost is under 0.1 % of the GEMMs' work.
- **The borrowed MTP drafter.** The GGUF has no draft model. Bonsai is Qwen3.8-27B's architecture in the same residual
  coordinates (the rotation sits only inside the linears), so Qwen3.8-27B's MTP layer can be fed Bonsai's final hidden state
  without any change of basis. It runs on the NPU in FP8, in its own memory slot, and drafts through Bonsai's own ternary head.
  74 % of first drafts are accepted, and 64-69 % of the second and third.
- **The leaf tree, and the 5-row step.** A verify pass of 2, 3 or 4 rows costs 0.37-0.39 s (less than a plain decode step's
  0.49 s, because the decode path's kernels differ). The drafter drafts while its chain's probability stays >= 0.5
  (`QWEN_TREE_TC`), and the rows left of the 4 take the drafts' second and third candidates, which need no extra draft pass
  (`QWEN_SPEC_TREE=leaf`, the default: 2.63 tokens a pass instead of 2.57). The ternary GEMM works in tiles of 4 rows, and a
  5th row costs ~+80 ms a pass (rows 6-8 then ~+10 ms each), more than the deeper drafts or extra leaves win back, so the
  default stays at 4 rows.

## Files

| file | what |
| --- | --- |
| `download.sh` | the GGUF, and the Qwen3.8-27B MTP weights and tokenizer |
| `bonsai2_pack.py` | the GGUF -> the NPU cache (ternary streams, the Hadamard signs, the head in parts); `--mtp`: the drafter |
| `bonsai2_repack_f16.py` | an fp32-table cache -> a new cache with fp16 scale tables (exact, or it stops) |
| `bonsai2_generate.py` | generation on the NPU (`qwen3.8-27b/qwen38_generate.py` with `QWEN_MODEL=bonsai2-27b` and the defaults above) |
| `bonsai2_serve.py` | the OpenAI- and Ollama-compatible server (`qwen3.8-27b/qwen38_serve.py` with these defaults) |
| `bonsai2_gguf.py`, `bonsai2_weights.py` | the GGUF reader, the `PTQ1_0` decoder and the Hadamard contract; the GGUF under the HF tensor names |
| `bonsai2_ref.py` | the numpy reference forward pass |
| `serve-demo.gif` | the recording above |
| `check/check_pack.py` | the pack round trip (no device) |
| `check/emulate_layer.py` | a host emulation of the NPU's layers and head from the cache, against the reference (no device) |
| `check/verify_oracle.py` | the verify path against plain greedy decoding, on the board |

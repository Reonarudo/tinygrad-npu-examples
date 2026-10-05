# Running LLMs and an image model on the Zhouyi NPU with tinygrad

Worked examples that run real models on the Arm China Zhouyi X2 NPU of the CIX P1 (CD8180), as found on the Radxa Orion O6N,
through the `ZHOUYI` backend of [tinygrad](https://github.com/Reonarudo/tinygrad/tree/zhouyi). The backend is included as the
`tinygrad/` submodule (branch `zhouyi`). Each example downloads its model, packs the weights into the layout the NPU streams,
and runs it; the text models can also be served over the OpenAI and Ollama APIs.

## Examples

| example | what it does | speed on the board |
| --- | --- | --- |
| [bonsai2-27b/](bonsai2-27b/) | PrismML's Ternary Bonsai 2 27B (Qwen3.8-27B with ternary weights) from its 5.9 GB GGUF: all 64 layers and the head on a ternary GEMM, speculative decoding with Qwen3.8-27B's MTP layer as the draft; an OpenAI- and Ollama-compatible server (`bonsai2_serve.py`) | 1.52 tok/s plain; **~4.3 tok/s** with speculative decoding (~5.6 on code) |
| [qwen3.8-27b/](qwen3.8-27b/) | Qwen3.8-27B text generation from its FP8 checkpoint: all 64 layers, the head and the multi-token-prediction (MTP) draft head on the NPU, weights mapped zero-copy from RAM | ~0.7 tok/s plain; **~1.9-3.0 tok/s** with speculative decoding (`QWEN_SPEC=6`), same tokens |
| [ornith-9b/](ornith-9b/) | Ornith 1.0 9B (Qwen3.5 architecture) from its GGUF Q8_0 checkpoint, on the same modules as qwen3.8-27b; speculative decoding with Qwen3.5-9B's MTP head as the draft | ~4.4 tok/s across prompts, ~5.4 tok/s on a code prompt |
| [ideogram4/](ideogram4/) | Ideogram 4 text-to-image at full precision: both DiT transformers and the VAE decoder on the NPU | ~170 s a sampling step at 1024 x 1024 (12 steps with `V4_TURBO_12`, ~35 min an image) |

Speeds are measured on the board with the NPU clocks at their defaults; text-generation speed depends on the prompt (code and
other predictable text drafts better).

- **Qwen3.8-27B / Ornith:** the host only tokenises, embeds and picks tokens. Decoding is bound by streaming the weights through
  the NPU's ~24 GB/s DDR port (27 GB a pass for the 27B), so speculative decoding verifies up to 6 tokens per pass through the
  weights. The output is identical to plain greedy decoding by construction.
- **Bonsai 2 27B:** the ternary weights cut a pass's weight traffic to ~7 GB, which streams through the same port; the GEMM works
  in tiles of 4 rows, so a verify pass costs about the same for 1 to 4 tokens and speculative decoding drafts at most 3.
- **Ideogram 4:** the text encoder runs on the host CPU (~2.5 min a prompt); everything after it runs on the NPU.

## Requirements

- A Radxa Orion O6N (or another CIX P1 board) running Linux with the Zhouyi NPU kernel driver (details below).
- **Python 3.12 or newer** with `numpy` and `tokenizers` (`pip install -r requirements.txt`). tinygrad comes from the submodule;
  nothing else is installed from this repository.
- Disk and RAM for the model you run (below, and in each example's README).

### Requirements on the board

More detail in [`tinygrad/extra/zhouyi/README.md`](tinygrad/extra/zhouyi/README.md).

- **The Zhouyi NPU kernel driver with its UAPI header `armchina_aipu.h`.** On first use the backend generates its bindings from
  that header. It looks in the driver's usual install locations; set `ZHOUYI_AIPU_HEADER=/path/to/armchina_aipu.h` otherwise.
- **libclang**, for that one-time generation. Distributions that ship only a versioned library (e.g. Debian / Ubuntu
  `libclang-NN-dev`) need `LIBCLANG_PATH=/usr/lib/llvm-NN/lib/libclang-NN.so` for the first run.
- **The vendor's AIPU compiler toolchain library** (`libaiputoolchain.so`). Default location `/usr/share/cix/lib/onnxruntime`;
  set `ZHOUYI_TOOLCHAIN_DIR` otherwise.
- **For the 27B models:** the driver's zero-copy weight buffers (the WBUF / SLOT ioctls). Without them the weights are copied
  every token (~2-3 s a token).
- **NPU clocks at their defaults** (`npuclk` 1.2 GHz, `npu_memclk` 750 MHz) to reproduce the numbers above.
- **Optional, for the speeds above:** write access to `/dev/cpu_dma_latency` (see
  [bonsai2-27b/README.md](bonsai2-27b/README.md#the-cpu-latency-request)); without it the text models run slower and print a note.
- **Disk:** Qwen3.8 about 27 GB of checkpoint + 27 GB of packed layers; Bonsai 2 about 6.4 GB of downloads + 8.7 GB packed;
  Ideogram 4 about 45 GB, plus ~16 GB of repacked panels (set `P48_DIR` / `W13I_DIR` to a disk with room).

## Getting started

```sh
git clone --recursive https://github.com/Reonarudo/tinygrad-npu-examples
cd tinygrad-npu-examples
pip install -r requirements.txt
```

If you cloned without `--recursive`, run `git submodule update --init` to fetch `tinygrad/`.

The scripts import tinygrad from the `tinygrad/` submodule. To use another checkout, set `TG=/path/to/tinygrad`.

### Paths

The scripts default to model and cache paths under `/mnt/ssd/...` (and `~/ideogram4` for Ideogram 4). These are examples: set
the environment variables each README lists (`BONSAI_DIR`, `BONSAI_GGUF`, `QWEN_DIR`, `QWEN_NPU`, `ORNITH_GGUF`, `IDEOGRAM4_DIR`,
...) to your own paths, on a disk with room.

### First run

Each example's README walks through download, pack and run. For example, Bonsai 2 27B:

```sh
cd bonsai2-27b
export BONSAI_DIR=/mnt/ssd/bonsai2 BONSAI_GGUF=/mnt/ssd/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf QWEN_NPU=/mnt/ssd/bonsai2-npu
bash download.sh
python3 bonsai2_pack.py && python3 bonsai2_pack.py --head && python3 bonsai2_pack.py --fuse
python3 bonsai2_pack.py --mtp $BONSAI_DIR/qwen3.8-27b-mtp
python3 bonsai2_generate.py "Write a Python function that returns the n-th Fibonacci number." --max-new 120
```

### Serving

`bonsai2-27b/bonsai2_serve.py` (and the generic `qwen3.8-27b/qwen38_serve.py`) keeps the model loaded and serves the OpenAI API
(`/v1/chat/completions`, `/v1/completions`, `/v1/models`) and Ollama's API (`/api/chat`, `/api/generate`, `/api/tags`, ...), so
the Ollama CLI, Ollama GUI apps and OpenAI clients can use the board. Starting it, running it as a systemd service and the
client set-up are in [bonsai2-27b/README.md](bonsai2-27b/README.md#serve-openai--and-ollama-compatible-api).

### Troubleshooting

| message | fix |
| --- | --- |
| `the Zhouyi KMD's UAPI header armchina_aipu.h was not found` | install the NPU driver's headers, or set `ZHOUYI_AIPU_HEADER` |
| `failed to load library libclang: try setting LIBCLANG_PATH?` | set `LIBCLANG_PATH` to your versioned `libclang-NN.so` for the first run |
| an error about the AIPU toolchain library | install the vendor toolchain package, or set `ZHOUYI_TOOLCHAIN_DIR` |
| `QWEN_CPU_LATENCY: no PM QoS request` | optional: allow access to `/dev/cpu_dma_latency` ([how](bonsai2-27b/README.md#the-cpu-latency-request)) |
| a job reports `EXCEPTION` after ~30 s | the NPU did not finish a job; `ZHOUYI_HANG_ID=1` names the kernel, its tasks and buffers |

## Layout

Each example is a folder with a `README.md` (prerequisites, stages, expected numbers and timings), its scripts and, where the
model is fetched from Hugging Face, a `download.sh`. Ideogram 4 has a `run.sh` for the whole pipeline. The `check/` folders hold
the correctness checks each README describes.

Model-specific NPU kernels live with their model (e.g. `qwen3.8-27b/qwen38_kernels.py`, `ideogram4/ideogram4_vec.py`). The
generic backend lives in the submodule: the device under `tinygrad/runtime/`, and the custom ops and generic hand-written
kernels under `extra/zhouyi/`. The small `zy.py` next to each example resolves them.

## Licences

This repository bundles no third-party code. tinygrad is a git submodule under its own licence (MIT); the NPU driver, the vendor
toolchain library, libclang and the Python packages are installed by you from their own sources. The model weights are not
included: each `download.sh` fetches them from Hugging Face, under each model's own licence (Qwen3.8-27B and Ternary Bonsai 2
27B: Apache-2.0; Ideogram 4: gated, accept its licence on Hugging Face first).

## Configuration: Qwen3.8-27B, Bonsai 2 and Ornith

The three text examples run the same modules (`bonsai2-27b/` sets `QWEN_MODEL=bonsai2-27b`, `ornith-9b/` sets
`QWEN_MODEL=ornith-9b`). All settings are environment variables; the defaults are the tuned ones. Bonsai 2's own defaults are in
[its README](bonsai2-27b/README.md#settings).

### Common settings

| variable | default | what it does |
| --- | --- | --- |
| `QWEN_SPEC` | `0` on qwen3.8-27b (plain decoding), `6` on ornith-9b | rows of a verify pass (1 + drafts); `0` decodes plainly. `6` is the measured setting |
| `QWEN_DIR` | `/mnt/ssd/qwen3.8-27b-fp8` (Ornith: `ORNITH_GGUF`) | the checkpoint. Set it to your path |
| `QWEN_NPU` | `/mnt/ssd/qwen3.8-27b-npu-s1` (Ornith: `/mnt/ssd/ornith-9b-npu`) | the packed cache written by the pack script. Set it to your path. For Ornith, pack with `ornith_pack.py --q8f` and point `QWEN_NPU` at that cache: it is the faster format, and the one the speeds above were measured with |
| `ORNITH_GGUF`, `ORNITH_MTP` | `/mnt/ssd/models/ornith-1.0-9b-Q8_0.gguf`, `/mnt/ssd/qbench/models/Qwen_Qwen3.5-9B-bf16.gguf` | Ornith's checkpoint and the Qwen3.5-9B GGUF its draft head comes from. Set them to your paths |
| `QWEN_TOK` | the checkpoint folder (Ornith: the cache) | where `tokenizer.json` is |
| `QWEN_PIN_GB` | `24` (Ornith: `12`) | GB of layers pinned in RAM |
| `QWEN_CPUS` | `auto` | run the host on the CPU that takes the NPU's interrupt (and one other core of its speed); `off`, or a list such as `0,1` |
| `QWEN_CPU_LATENCY` | `0` | the CPU-latency request (µs) held on `/dev/cpu_dma_latency` while the process runs; `off`: none |

### Speculative decoding

| variable | default | what it does |
| --- | --- | --- |
| `QWEN_DRAFT` | `mtp` | the drafter: the checkpoint's MTP head, or `none` (verify path without drafts) |
| `QWEN_DRAFT_TAU` | `0.6` | keep drafting while the draft head's probability is at least tau; `0` always drafts `QWEN_DRAFT_MAX` |
| `QWEN_DRAFT_MAX` | `QWEN_SPEC - 1` (Ornith: `3`) | drafts per pass at most |
| `QWEN_DRAFT_PARTS` | `2,thresh:25` on qwen3.8-27b, `2` on Ornith | `<n>[,thresh:<x>]`: the draft head reads the lm head's first n column parts; with `thresh:<x>` it reads parts after the first only when the first part's top logit is at most x |
| `QWEN_SPEC_GEOS` | `3,4,6` | verify geometries (row counts) captured; a pass rounds up to the next one |
| `QWEN_SPEC_TREE` | `off` | `rescue2`: every pass verifies 8 rows, the 5-draft chain plus 2 "rescue" rows holding the draft head's second-choice token at the two least confident positions (`fixed:<j1,j2>` for fixed positions). About +1 % on average; see [qwen3.8-27b/README.md](qwen3.8-27b/README.md) |
| `QWEN_SPEC_WARM` | `1` | prepare every verify and draft geometry before the first token |
| `QWEN_SPEC_LOG` | unset | a `.jsonl` path for per-pass telemetry (drafts, acceptance, times, output ids) |
| `QWEN_HEAD_TOP3` | `0` | also log the draft head's top-3 tokens (drafts unchanged) |

### Memory options

| variable | default | what it does |
| --- | --- | --- |
| `QWEN_ZC` | `1` | zero-copy weights; `0` copies each layer (needed when the driver lacks the WBUF / SLOT ioctls) |
| `QWEN_NOPREFETCH` | unset | no read-ahead of layers that are not pinned in RAM |
| `QWEN_PREFILL` | unset | `chunked`: prefill the FP8 model through the verify path (Ornith always does; the server does by default) |

### Alternative kernel paths

The defaults are the fastest paths and the only ones recommended. The alternatives below are kept to compare against and to
narrow down a problem; they produce the same tokens.

| variable | default | alternatives | note |
| --- | --- | --- | --- |
| `QWEN_GDN_PREP` | `dma` | `fast`, `cached`, `dma-lsarr` | how the verify pass's DeltaNet kernel stages its rows. `fast` and `cached` read the rows through the cache; they are the older paths and can stall the NPU under some memory placements |
| `QWEN_ATTN_PREFILL` | `csrc` | `none` | the full-attention prefill (FP8 model). `none` uses tinygrad's generated kernels, the older path, which can stall the NPU |
| `QWEN_FUSE` | `1` | `0` | `0` runs the same-input projections as separate GEMMs (for a cache packed without `--fuse`) |
| `QWEN_ATTN` | `split` | `decm` | the verify pass's attention as one kernel instead of two |
| `QWEN_GDN_TOK` | `3` | `1`, `2` | earlier variants of the one-token DeltaNet kernel |
| `QWEN_RMS_DMA` | `1` | `0` | RMSNorm rows read through the cache instead of streamed by DMA |
| `QWEN_HEAD` | `fp8` when packed, else `fp16` | `fp16`, `both` | the lm head for plain decoding; `both` runs both and compares |

### Diagnostics

| variable | what it does |
| --- | --- |
| `QWEN_DEBUG`, `QWEN_PA`, `QWEN_DUMP_HIDDEN`, `GDN_TOKM_DIAG` | debug prints, buffer addresses, hidden-state dumps, the DeltaNet kernel's diagnostic modes |
| `QWEN_SHARDS`, `QWEN_TMAX`, `QWEN_PROF_POS` | used by the `check/` scripts and diagnostics only (reference shards, a cache length and position) |

### Backend settings

The backend's variables are documented in [`tinygrad/extra/zhouyi/README.md`](tinygrad/extra/zhouyi/README.md). The ones a
user is most likely to need:

| variable | default | what it does |
| --- | --- | --- |
| `ZHOUYI_AIPU_HEADER` | the driver's install locations | the KMD header the bindings are generated from |
| `ZHOUYI_TOOLCHAIN_DIR` | `/usr/share/cix/lib/onnxruntime` | where `libaiputoolchain.so` is |
| `ZHOUYI_HANG_ID` | `0` | `1`: if a job never finishes, report the kernel, its tasks and buffers; `2`: also per-task progress stamps; `1:<path>` / `2:<path>`: also append each report to a file |
| `ZHOUYI_CHAIN_MAX` | `8` | kernel launches fused into one NPU job; `1` runs each launch as its own job (with `ZHOUYI_HANG_ID`, pins a problem to one kernel) |
| `ZHOUYI_GM` | `0` | `1`: GEMMs stage through the cluster's shared GM memory (Ideogram 4 sets it) |
| `ZHOUYI_CORES` | `3` | NPU cores a launch may use |

The packed caches record their weight-scale layout themselves (a `scales` marker file), so no setting is needed for it; the
loader refuses a cache whose layout does not match.

# Ideogram 4 on the Zhouyi NPU

Text-to-image at 1024 x 1024 with [ideogram-ai/ideogram-4-fp8](https://huggingface.co/ideogram-ai/ideogram-4-fp8):
the prompt conditioning (Qwen3-VL-8B, on the host CPU), then both DiT transformers (conditional over
[text | image] tokens, unconditional over image tokens, classifier-free guidance, Euler steps) and the VAE
decoder on the NPU. The transformers' fp8 (E4M3) weights are unpacked exactly to fp16 on the device and every
linear and attention runs as fp16 x fp16 -> fp32 on the TEC matrix unit; the result matches an fp32 reference
(velocity cosine 0.9999+, decoded image PSNR 82 dB against a numpy fp32 decode).

An optional int8 path (`Q8=1`, [below](#the-int8-path-q81)) runs the large linears on the NPU's AIFF int8 engines
instead: an image in ~19 minutes instead of ~33, with the same scene and composition but different small details.

## What you need

| item | size | note |
| --- | ---: | --- |
| `ideogram-4-fp8` checkpoint: `transformer/`, `unconditional_transformer/`, `vae/`, `text_encoder/`, `tokenizer/` | 9.3 + 9.3 + 0.2 + 8.2 GB | gated on Hugging Face: accept the license, put a read token in `~/ideogram4/.token` |
| the packed weight caches (`download.sh` builds them) | 8.4 + 8.4 + 7.2 GB | E4M3 panels in the GEMM's layout: one folder per transformer, one for the text encoder (`te_p48`) |
| tinygrad with the Zhouyi backend: the repository's `tinygrad/` submodule | | `export TG=<path>`; `run.sh` and `download.sh` require it |
| `numpy` | | the only Python dependency besides tinygrad (the text encoder runs on the NPU; its numpy fallback, `NPU_TEXT=0`, wants a numpy linked to an optimised BLAS: see below) |

Everything else on the board is what the backend needs (the driver, the vendor toolchain library).

The files go into `$IDEOGRAM4_DIR` (default `~/ideogram4`; set it to a disk with ~45 GB free). The repacked panels the sampler
writes on first use (~16 GB) go to `P48_DIR` / `W13I_DIR` (default under `~/ideogram4`).

## Run it

```sh
cd ideogram4
export TG=$(cd .. && pwd)/tinygrad             # the submodule
export IDEOGRAM4_DIR=~/ideogram4               # your path
./download.sh                                  # the checkpoint, then the two weight caches (once)
./run.sh "An iPhone photo of a ginger tabby cat wearing a tiny purple wizard hat" out/cat
```

`run.sh` does, in order:

1. `ideogram4_text.py --npu` — the prompt through the text encoder and the text projection, once per image, on the NPU
   (`ideogram4_text_npu.py`, the fp16 GEMM path, nothing quantised): ~4 s for a short prompt (44 tokens) to ~15 s for a
   long JSON prompt (681 tokens), the process start and the kernel builds included (the encode itself 2.3 and 13 s; 2048
   tokens ~40 s). Output: `out/cat_text.npz`, the text rows the conditional transformer reads. `NPU_TEXT=0` runs the numpy
   fallback on the host instead: with a numpy linked to OpenBLAS (`python3 -m venv --system-site-packages ~/tevenv &&
   ~/tevenv/bin/pip install --ignore-installed numpy`, then `TEXT_PY=~/tevenv/bin/python`) ~45 s to 2.2 minutes; with
   Ubuntu's reference-BLAS numpy 4.6 to 63 minutes.
2. `ideogram4_fp_1024.py` — the sampling loop on the NPU. `V4_TURBO_12` (12 steps) takes ~167 s per step at
   1024 x 1024, so ~33 minutes; `V4_DEFAULT_20` and `V4_QUALITY_48` scale with the step count.
   Output: `out/cat_latents.npy` (packed latents [4096, 128]); `--save-every` keeps every step's latents.
3. `vae_npu.py` — the VAE decoder on the NPU, ~20 s warm. Output: `out/cat.png`.

Each stage can be run on its own; the scripts' docstrings give their options. `--size` accepts any multiple of
96 px, `--seed` reproduces the pipeline's CPU `torch.randn` bit-for-bit (`torch_randn.py`), `--preset` picks the
schedule (`V4_TURBO_12`, `V4_DEFAULT_20`, `V4_QUALITY_48`), `--guidance` overrides the per-step CFG scale.

A run cut short can continue: `ideogram4_fp_1024.py ... --resume` (same prompt, seed and schedule). Prompts of any
length up to 2048 tokens fit: the sampler plans its device memory per row count (attention heads per batch, split
partials, weight slots) within the process's 3 GB device address window.

## The int8 path (`Q8=1`)

The qkv, o and w1|w3 linears run as W8A8 on the three AIFF engines: int8 weights per output channel, int8
activations per token, SmoothQuant folded into the weights and the existing per-channel multipliers. w2 stays fp16 on
the TEC (`--q8-w2` puts it on the AIFF too). The quantiser, the AIFF output and the dequantiser are bit-exact against
their numpy references (`GATE=1`).

| JSON prompt, 4704 rows, 12 steps | step | image |
| --- | ---: | ---: |
| fp8 (default) | ~166 s | ~33 min |
| `Q8=1` | ~96 s | ~19 min |
| `Q8=1` + `--q8-w2` | ~73 s | ~15 min |

Quality: against the fp8 image of the same prompt and seed, PSNR ~23 dB / SSIM 0.78. The one-step error against
the fp8 trajectory on prompts outside the calibration set is 0.117 (0.18 with `--q8-w2`).

The calibration (`q8_calib.npz`: SmoothQuant factors and the int16 output shifts) is shipped; it comes from the 15
prompts in `prompts/calib` over steps 0/3/6/9/11 of both transformers (`ideogram4_q8_calib.py`). Setup, once:

```sh
python3 ideogram4_q8_pack.py --weights ~/ideogram4/transformer --branch cond --out ~/ideogram4/q8pack/cond
python3 ideogram4_q8_pack.py --weights ~/ideogram4/unconditional_transformer --branch uncond --out ~/ideogram4/q8pack/uncond
export AIFF_TG=<tinygrad tree with the AIFF register generator (tinygrad/runtime/support/zhouyi/aiff_gen.py)>
Q8=1 ./run.sh "..." out/cat
```

The packs take ~17 GB. `run.sh` builds the AIFF descriptor templates for each new prompt length (~4 s,
`ideogram4_aiff_desc.py`, kept in `~/ideogram4/q8desc`); that step needs `AIFF_TG`. The int8 path runs at 1024 px.

## Files

| file | role |
| --- | --- |
| `download.sh` | fetches the checkpoint into `$IDEOGRAM4_DIR`, then packs both transformers' weights (`ideogram4_fp_pack.py`) |
| `run.sh` | the three stages for one prompt |
| `ideogram4_text.py` | prompt conditioning: tokenizer, then the Qwen3-VL-8B text encoder and the conditional transformer's text projection -- on the NPU (`--npu`) or in numpy |
| `ideogram4_text_npu.py`, `ideogram4_te_pack.py` | the text encoder on the NPU (fp16 GEMMs from the E4M3 codes, the attention and vector kernels, the taps' projection) and its packed weight cache |
| `ideogram4_fp_1024.py` | the sampler: both transformers on the NPU, guidance and the Euler update as device kernels |
| `ideogram4_fp_backend.py` | the transformer blocks on the NPU (E4M3 unpack, fp16 GEMMs, attention, the vector kernels) |
| `ideogram4_ref.py` | the numpy fp32 reference: the transformer, its blocks (used by the `GATE=1` checks) and the sampling schedule |
| `ideogram4_weights.py` | reads the fp8 safetensors with numpy alone |
| `ideogram4_fp_pack.py` | builds the packed weight cache a transformer needs |
| `ideogram4_q8.py`, `ideogram4_q8_pack.py`, `ideogram4_aiff_desc.py` | the int8 path: the AIFF linear plan and kernels, the int8 packs, the descriptor templates |
| `ideogram4_q8_calib.py`, `q8_calib.npz`, `prompts/` | the int8 calibration: the tool, its result, the calibration and evaluation prompts |
| `vae_npu.py`, `ideogram4_vae.py` | the VAE decoder on the NPU and its weights |
| `torch_randn.py` | `torch.manual_seed(s); torch.randn(n)` in numpy, bit-for-bit |
| `zy.py` | resolves the backend's custom ops and kernels from the tinygrad tree `TG` points at |

## Checking a change

`GATE=1 TG=... python3 ideogram4_fp_1024.py --text out/cat_text.npz --steps 1 --layers 0-0 --branch cond --out /tmp/x.npy`
runs layer 0 op by op against numpy. `vae_npu.py LATENTS --ref vae_ref.npz` reports the PSNR against a saved fp32
decode. The expected numbers for the cat prompt at 1024 px: step 1 `|x| 0.782 |v| 2.763`, decode PSNR 82.2 dB.

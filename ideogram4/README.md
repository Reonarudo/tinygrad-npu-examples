# Ideogram 4 on the Zhouyi NPU

Text-to-image at 1024 x 1024 with [ideogram-ai/ideogram-4-fp8](https://huggingface.co/ideogram-ai/ideogram-4-fp8):
the prompt conditioning (Qwen3-VL-8B, on the host CPU), then both DiT transformers (conditional over
[text | image] tokens, unconditional over image tokens, classifier-free guidance, Euler steps) and the VAE
decoder on the NPU. The transformers' fp8 (E4M3) weights are unpacked exactly to fp16 on the device and every
linear and attention runs as fp16 x fp16 -> fp32 on the TEC matrix unit; the result matches an fp32 reference
(velocity cosine 0.9999+, decoded image PSNR 82 dB against a numpy fp32 decode).

## What you need

| item | size | note |
| --- | ---: | --- |
| `ideogram-4-fp8` checkpoint: `transformer/`, `unconditional_transformer/`, `vae/`, `text_encoder/`, `tokenizer/` | 9.3 + 9.3 + 0.2 + 8.2 GB | gated on Hugging Face: accept the license, put a read token in `$IDEOGRAM4_DIR/.token` |
| the packed weight caches (`download.sh` builds them) | 8.4 + 8.4 GB | E4M3 panels in the GEMM's layout, one folder per transformer |
| tinygrad with the Zhouyi backend: the repository's `tinygrad/` submodule | | `export TG=<path>`; `run.sh` and `download.sh` require it |
| `numpy` | | the only Python dependency besides tinygrad |

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

1. `ideogram4_text.py` — the prompt through the text encoder, once per image, host numpy: about 2.5 minutes
   (36 layers at ~4 s each). Output: `out/cat_text.npz`, the text rows the conditional transformer reads.
2. `ideogram4_fp_1024.py` — the sampling loop on the NPU. `V4_TURBO_12` (12 steps) takes ~167 s per step at
   1024 x 1024, so ~33 minutes; `V4_DEFAULT_20` and `V4_QUALITY_48` scale with the step count.
   Output: `out/cat_latents.npy` (packed latents [4096, 128]); `--save-every` keeps every step's latents.
3. `vae_npu.py` — the VAE decoder on the NPU, ~20 s warm. Output: `out/cat.png`.

Each stage can be run on its own; the scripts' docstrings give their options. `--size` accepts any multiple of
96 px, `--seed` reproduces the pipeline's CPU `torch.randn` bit-for-bit (`torch_randn.py`), `--preset` picks the
schedule (`V4_TURBO_12`, `V4_DEFAULT_20`, `V4_QUALITY_48`), `--guidance` overrides the per-step CFG scale.

A run cut short can continue: `ideogram4_fp_1024.py ... --resume` (same prompt, seed and schedule).

## Files

| file | role |
| --- | --- |
| `download.sh` | fetches the checkpoint into `$IDEOGRAM4_DIR`, then packs both transformers' weights (`ideogram4_fp_pack.py`) |
| `run.sh` | the three stages for one prompt |
| `ideogram4_text.py` | prompt conditioning: tokenizer + Qwen3-VL-8B text encoder in numpy, then the conditional transformer's text projection |
| `ideogram4_fp_1024.py` | the sampler: both transformers on the NPU, guidance and the Euler update as device kernels |
| `ideogram4_fp_backend.py` | the transformer blocks on the NPU (E4M3 unpack, fp16 GEMMs, attention, the vector kernels) |
| `ideogram4_ref.py` | the numpy fp32 reference: the transformer, its blocks (used by the `GATE=1` checks) and the sampling schedule |
| `ideogram4_weights.py` | reads the fp8 safetensors with numpy alone |
| `ideogram4_fp_pack.py` | builds the packed weight cache a transformer needs |
| `vae_npu.py`, `ideogram4_vae.py` | the VAE decoder on the NPU and its weights |
| `torch_randn.py` | `torch.manual_seed(s); torch.randn(n)` in numpy, bit-for-bit |
| `zy.py` | resolves the backend's custom ops and kernels from the tinygrad tree `TG` points at |

## Checking a change

`GATE=1 TG=... python3 ideogram4_fp_1024.py --text out/cat_text.npz --steps 1 --layers 0-0 --branch cond --out /tmp/x.npy`
runs layer 0 op by op against numpy. `vae_npu.py LATENTS --ref vae_ref.npz` reports the PSNR against a saved fp32
decode. The expected numbers for the cat prompt at 1024 px: step 1 `|x| 0.782 |v| 2.763`, decode PSNR 82.2 dB.

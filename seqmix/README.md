# seqmix: sequence-mixer kernels for the Zhouyi backend

C kernels (the backend's `register_csrc` route) for the mixers the examples lacked, one module each, with a float64 numpy
Contract, the source generators, the DMA descriptor slots and the LSRAM budget asserted at generation time. Inputs and outputs are
fp32 rows; a model port reads its GEMM tiles instead where it pays (as `qwen3.8-27b`'s gdn_tok3 does).

| module | kernels | what | status |
| --- | --- | --- | --- |
| `mamba2.py` | `mamba2_tok`, `mamba2_gnorm` | Mamba-2 / SSD (Nemotron 3.x, Granite 4.0-H): conv + SiLU over a per-head ring, dt / dA, the state recurrence, the gated group norm | simulator PASS; board 532 us a layer (M 1), 5.9 ms (M 12) at Nemotron's shape |
| `mla.py` | `mla_part`, `mla_comb` | multi-head latent attention over the compressed cache, absorbed decode form (GLM-4.7-Flash, DeepSeek-V2-Lite, Kimi Linear) | simulator PASS; board 447 us a layer (GLM shape, 256 positions) |
| `kda.py` | `kda_tok` | Kimi Delta Attention: the gated delta rule with a per-channel decay (Kimi Linear, Ling-3.0-tiny), both gate forms | simulator PASS; board 253 us a layer |
| `cca.py` | `cca_prep`, `cca_state` | Zyphra CCA / CCGQA preprocessing (ZAYA1-8B); the attention is `attn.py` | simulator PASS; board 101 us a layer |
| `attn.py` | `attn_part`, `attn_comb` | decode GQA attention over fp32 caches with attention sinks and a sliding window (gpt-oss) | simulator PASS; board 147 us (window 128) / 262 us (full, pos 256); in use by gptoss/ |

| `route.py` | `moe_route` | MoE routing on the device: softmax / sigmoid + bias / grouped (Ring) / top-k softmax (gpt-oss), the shared expert's weight | simulator PASS; board 69 us; in use by ringmini/ |
| `rows.py` | `rowadd`, `qrows_gate` | row kernels around the qdot GEMVs (the generic codegen copies cost ms) | in use by ringmini/ |

`bench_seqmix.py` times them on the board (performance governor asserted and restored). Gates: the simulator gates (kept with the hardware notes, outside this repository) (the vendor model, functional only); tickets and evidence in
the hardware notes. Rules learnt: a state kernel's ring / halo is per unit or updated by a separate kernel (a
shared ring raced on the model's sequential cores); a weight block above LSRAM streams in chunks; a norm spanning heads is a second
kernel fed by per-head sums of squares.

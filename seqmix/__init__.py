"""seqmix: sequence-mixer kernels for the Zhouyi backend (C kernels through `register_csrc`), one module a mixer.

  mamba2    the Mamba-2 (SSD) token kernel: conv + SiLU over a ring, dt / dA, the state recurrence, the gated group norm
  mla       DeepSeek-style multi-head latent attention over the compressed cache (absorbed decode form)
  kda       Kimi Delta Attention (the gated delta rule with a per-channel decay)
  cca       Zyphra's compressed convolutional attention
  sinks     attention sinks and a short sliding window in the decode attention (gpt-oss)

Each module has: a float64 numpy Contract (`*_ref`), the source generators (`*_src`), the DMA descriptor slots (`*_desc`) and the
LSRAM budget asserted at generation time. Each kernel has a simulator gate (kept with the hardware notes, outside this repository).
"""

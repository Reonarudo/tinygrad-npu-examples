#!/usr/bin/env python3
"""Ideogram 4's flow-matching transformer in float32 numpy -- the REFERENCE the backend path is
gated against, ported line for line from diffusers' `transformer_ideogram4.py` and
`pipeline_ideogram4.py` at commit 04b197ee. No torch.

  * `MRoPE(position_ids)` -> (cos, sin) `[L, 256]`; interleaved (t, h, w) sections (24, 20, 20).
  * `block(W, l, x, cos, sin, adaln)`: RMSNorm -> * (1 + scale) -> qkv -> q/k RMSNorm -> RoPE ->
    causal-free full attention -> o -> RMSNorm -> * tanh(gate) residual; then the SwiGLU MLP the
    same way. A block's cost at T tokens is 2 * 255M * T FLOP: 0.13 TFLOP at T=256 (a workstation's
    numpy: ~0.4 s; the board's, no BLAS: ~40 s -- the block gate runs there ONCE per block).
  * `forward(W, latents, t, ...)`: the whole transformer for one branch (`unconditional=True`:
    image tokens only, zero text features, the pipeline's `neg_v`).
  * `sigmas(steps, mu, std)`, `euler_step`: the pipeline's logit-normal schedule and the
    FlowMatchEuler update `x <- x + (sigma_next - sigma) * (-v)`.
  * `generate_uncond(W, grid, steps, seed)`: the UNCONDITIONAL image's latents (the pipeline with
    guidance from the negative branch alone is not what CFG does; this is the plain unconditional
    sample: v = neg_v) -- the first milestone, which needs no text encoder.

Positions: image tokens at (0, h, w) + 65536; the unconditional branch's sequence is the image
grid alone (`neg_position_ids = position_ids[:, max_text_tokens:]`).
"""
import math
import numpy as np

HEAD_DIM, NH, HID, MLP, ADALN, IN_CH = 256, 18, 4608, 12288, 512, 128
IMAGE_POSITION_OFFSET = 65536
MROPE = (24, 20, 20)
ROPE_THETA = 5_000_000
NORM_EPS = 1e-5


# ------------------------------------------------------------------ pieces ----
def rmsnorm(x, w, eps):
    x = x.astype(np.float32)
    v = (x * x).mean(-1, keepdims=True, dtype=np.float32)
    return x * (1.0 / np.sqrt(v + np.float32(eps))) * w.astype(np.float32)


def silu(x): return x / (1.0 + np.exp(-x))


def mrope(position_ids: np.ndarray):
    """position_ids `[L, 3]` (t, h, w) int -> cos, sin `[L, 256]` (float32)."""
    inv = (1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))).astype(np.float32)   # [128]
    pos = position_ids.astype(np.float32)                                            # [L, 3]
    freqs = pos.T[:, :, None] * inv[None, None, :]                                    # [3, L, 128]
    ft = freqs[0].copy()
    for axis, off in ((1, 1), (2, 2)):
        idx = np.arange(off, MROPE[axis] * 3, 3)
        ft[:, idx] = freqs[axis][:, idx]
    emb = np.concatenate([ft, ft], -1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], -1)


def image_position_ids(grid_h, grid_w):
    h = np.repeat(np.arange(grid_h), grid_w); w = np.tile(np.arange(grid_w), grid_h)
    return np.stack([np.zeros_like(h), h, w], 1) + IMAGE_POSITION_OFFSET


def sinusoidal(t, dim, scale=1e4):
    half = dim // 2
    f = np.exp(np.arange(half, dtype=np.float32) * -(math.log(scale) / (half - 1)))
    e = np.float32(t) * f
    return np.concatenate([np.sin(e), np.cos(e)]).astype(np.float32)


def attention(q, k, v):
    """q, k, v `[L, NH, D]` -> `[L, NH*D]`; full (segment = one sample) softmax attention."""
    L = q.shape[0]
    out = np.empty((L, NH, HEAD_DIM), np.float32)
    s = np.float32(1.0 / math.sqrt(HEAD_DIM))
    for h in range(NH):
        a = (q[:, h] @ k[:, h].T) * s
        a = a - a.max(-1, keepdims=True); e = np.exp(a); p = e / e.sum(-1, keepdims=True)
        out[:, h] = p @ v[:, h]
    return out.reshape(L, NH * HEAD_DIM)


# ------------------------------------------------------------------ the block ----
class BlockWeights:
    """One block's float32 matrices (dequantized once): ~1 GB at float32."""
    def __init__(self, W, l):
        p = f"layers.{l}."
        self.qkv = W.linear(p + "attention.qkv")                 # [13824, 4608]
        self.o = W.linear(p + "attention.o")                     # [4608, 4608]
        self.w1 = W.linear(p + "feed_forward.w1"); self.w3 = W.linear(p + "feed_forward.w3")   # [12288, 4608]
        self.w2 = W.linear(p + "feed_forward.w2")                # [4608, 12288]
        self.norm_q = W.get(p + "attention.norm_q.weight"); self.norm_k = W.get(p + "attention.norm_k.weight")
        self.an1 = W.get(p + "attention_norm1.weight"); self.an2 = W.get(p + "attention_norm2.weight")
        self.fn1 = W.get(p + "ffn_norm1.weight"); self.fn2 = W.get(p + "ffn_norm2.weight")
        self.ada_w = W.linear(p + "adaln_modulation"); self.ada_b = W.get(p + "adaln_modulation.bias")   # [18432, 512], [18432]


def block_modulation(bw: BlockWeights, adaln):
    """adaln `[512]` -> (scale_msa, gate_msa, scale_mlp, gate_mlp) `[4608]` each."""
    mod = bw.ada_w @ adaln + bw.ada_b
    s1, g1, s2, g2 = np.split(mod, 4)
    return 1.0 + s1, np.tanh(g1), 1.0 + s2, np.tanh(g2)


def block(bw: BlockWeights, x, cos, sin, adaln, parts=None):
    """x `[L, 4608]` float32 -> the block's output; `parts` (a dict) collects the intermediates
    the backend gates against."""
    s1, g1, s2, g2 = block_modulation(bw, adaln)
    h = rmsnorm(x, bw.an1, NORM_EPS) * s1
    qkv = (h @ bw.qkv.T).reshape(-1, 3, NH, HEAD_DIM)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    q = rmsnorm(q, bw.norm_q, 1e-5); k = rmsnorm(k, bw.norm_k, 1e-5)
    c, s = cos[:, None, :], sin[:, None, :]
    q = q * c + rotate_half(q) * s; k = k * c + rotate_half(k) * s
    att = attention(q, k, v)
    ao = att @ bw.o.T
    x = x + g1 * rmsnorm(ao, bw.an2, NORM_EPS)
    h2 = rmsnorm(x, bw.fn1, NORM_EPS) * s2
    m = (silu(h2 @ bw.w1.T) * (h2 @ bw.w3.T)) @ bw.w2.T
    out = x + g2 * rmsnorm(m, bw.fn2, NORM_EPS)
    if parts is not None: parts.update(h=h, q=q, k=k, v=v, att=att, ao=ao, x1=x, h2=h2, m=m)
    return out.astype(np.float32)


# ------------------------------------------------------------------ the transformer ----
class TopWeights:
    def __init__(self, W):
        self.input_w = W.linear("input_proj"); self.input_b = W.get("input_proj.bias")             # [4608, 128]
        self.t_in_w = W.linear("t_embedding.mlp_in"); self.t_in_b = W.get("t_embedding.mlp_in.bias")
        self.t_out_w = W.linear("t_embedding.mlp_out"); self.t_out_b = W.get("t_embedding.mlp_out.bias")
        self.ada_w = W.linear("adaln_proj"); self.ada_b = W.get("adaln_proj.bias")                 # [512, 4608]
        self.ind = W.get("embed_image_indicator.weight")                                            # [2, 4608]
        self.fin_ada_w = W.linear("final_layer.adaln_modulation"); self.fin_ada_b = W.get("final_layer.adaln_modulation.bias")
        self.fin_w = W.linear("final_layer.linear"); self.fin_b = W.get("final_layer.linear.bias")   # [128, 4608]
        self.cond_norm = W.get("llm_cond_norm.weight") if W.has("llm_cond_norm.weight") else None
        self.cond_w = W.linear("llm_cond_proj") if W.has("llm_cond_proj.weight") else None
        self.cond_b = W.get("llm_cond_proj.bias") if W.has("llm_cond_proj.bias") else None


def adaln_input(tw: TopWeights, t_model: float):
    """The block conditioning `[512]` from the model time in [0, 1]."""
    emb = sinusoidal(1e4 * t_model, HID)
    e = silu(tw.t_in_w @ emb + tw.t_in_b)
    t_cond = tw.t_out_w @ e + tw.t_out_b
    return silu(tw.ada_w @ t_cond + tw.ada_b)


def embed_image_tokens(tw: TopWeights, latents):
    """Image tokens `[L, 128]` -> the stream `[L, 4608]` (input_proj + the image indicator embedding)."""
    return (latents.astype(np.float32) @ tw.input_w.T + tw.input_b + tw.ind[1]).astype(np.float32)


def final_layer(tw: TopWeights, x, adaln):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    n = (x - m) / np.sqrt(v + 1e-6)
    scale = 1.0 + (tw.fin_ada_w @ silu(adaln) + tw.fin_ada_b)
    return (n * scale) @ tw.fin_w.T + tw.fin_b


class Calib:
    """Per-layer activation maxima over a reference run: what the backend's per-tensor int8 scales
    are set from (`ideogram4_block_backend.QBlock` reads them instead of calibrating on one input).
    Keys per layer: h, q, k, v, S (the scaled logits), att, h2, gl (the SwiGLU product); and the
    K-chunk partial maxima per linear for a given chunk width (`part[l][name]`)."""
    def __init__(self, kc: int): self.kc = kc; self.act = {}; self.part = {}
    def note(self, l, parts, bw):
        a = self.act.setdefault(l, {}); pm = self.part.setdefault(l, {})
        for k in ("h", "q", "k", "v", "att", "h2"): a[k] = max(a.get(k, 0.0), float(np.abs(parts[k]).max()))
        S = max(float(np.abs(parts["q"][:, hh] @ parts["k"][:, hh].T).max()) for hh in range(NH)) / math.sqrt(HEAD_DIM)
        a["S"] = max(a.get("S", 0.0), S)
        gl = silu(parts["h2"] @ bw.w1.T) * (parts["h2"] @ bw.w3.T); a["gl"] = max(a.get("gl", 0.0), float(np.abs(gl).max()))
        for name, xin, w in (("qkv", parts["h"], bw.qkv), ("o", parts["att"], bw.o), ("w1", parts["h2"], bw.w1), ("w3", parts["h2"], bw.w3), ("w2", gl, bw.w2)):
            m = max(float(np.abs(xin[:, c:c + self.kc] @ w[:, c:c + self.kc].T).max()) for c in range(0, w.shape[1], self.kc))
            pm[name] = max(pm.get(name, 0.0), m)
    def save(self, path): np.savez(path, kc=self.kc, act=np.array([self.act], dtype=object), part=np.array([self.part], dtype=object))
    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True); c = cls(int(d["kc"])); c.act = d["act"][0]; c.part = d["part"][0]; return c


def forward_uncond(W, tw: TopWeights, blocks, latents, t_model, grid, cache=None, calib: "Calib|None" = None):
    """The unconditional branch: image tokens only. `blocks`: a list of BlockWeights (loaded), or
    None to stream them from `W` one at a time (a workstation's 17 GB). `calib` collects the maxima."""
    gh, gw = grid
    cos, sin = mrope(image_position_ids(gh, gw))
    ada = adaln_input(tw, t_model)
    x = embed_image_tokens(tw, latents)
    for l in range(W.config["num_layers"]):
        bw = blocks[l] if blocks is not None else BlockWeights(W, l)
        parts = {} if calib is not None else None
        x = block(bw, x, cos, sin, ada, parts=parts)
        if calib is not None: calib.note(l, parts, bw)
    return final_layer(tw, x, ada)


# ------------------------------------------------------------------ the schedule and the loop ----
def _ndtri(p):
    """Inverse normal CDF (Acklam's rational approximation refined by one Newton step), float64."""
    from math import erf, sqrt, exp, log, pi
    p = np.asarray(p, np.float64); out = np.empty_like(p)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    for i, pi_ in np.ndenumerate(p):
        if pi_ <= 0: out[i] = -np.inf; continue
        if pi_ >= 1: out[i] = np.inf; continue
        if pi_ < 0.02425:
            q = sqrt(-2 * log(pi_)); x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        elif pi_ > 1 - 0.02425:
            q = sqrt(-2 * log(1 - pi_)); x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        else:
            q = pi_ - 0.5; r = q * q
            x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
        e = 0.5 * (1 + erf(x / sqrt(2))) - pi_; u = e * sqrt(2 * pi) * exp(x * x / 2)
        out[i] = x - u / (1 + x * u / 2)
    return out


def sigmas(steps, mu, std, logsnr_min=-15.0, logsnr_max=18.0):
    """The pipeline's `_logit_normal_sigmas`: decreasing sigmas in (0, 1], length `steps`, and the
    terminal 0 the scheduler appends."""
    z = _ndtri(np.linspace(0.0, 1.0, steps + 1))
    y = mu + std * z
    t = 1.0 - 1.0 / (1.0 + np.exp(-y))
    t = np.clip(t, 1.0 / (1.0 + math.exp(0.5 * logsnr_max)), 1.0 / (1.0 + math.exp(0.5 * logsnr_min)))
    s = (1.0 - t)[::-1][:-1]
    return np.concatenate([s, [0.0]])


def resolution_mu(height, width, base_mu):
    return base_mu + 0.5 * math.log(height * width / (512 * 512))


def generate_uncond(W, grid, steps=12, seed=0, mu=0.5, std=1.75, blocks=None, log=print, calib=None):
    """Unconditional latents `[gh*gw, 128]` by the negative branch alone (v = neg_v), the pipeline's
    FlowMatchEuler update. Returns (latents, the per-step velocities)."""
    gh, gw = grid
    tw = TopWeights(W)
    sig = sigmas(steps, resolution_mu(gh * 16, gw * 16, mu), std)
    rng = np.random.RandomState(seed)
    x = rng.randn(gh * gw, IN_CH).astype(np.float32)
    vs = []
    for i in range(steps):
        t_model = 1.0 - sig[i]                                     # diffusers' timestep = sigma * 1000; model time = 1 - sigma
        v = forward_uncond(W, tw, blocks, x, t_model, grid, calib=calib)
        x = x + np.float32(sig[i + 1] - sig[i]) * (-v)               # scheduler.step(-v)
        vs.append(v)
        log("   step %2d/%d sigma %.4f -> %.4f  |x| %.3f  |v| %.3f" % (i + 1, steps, sig[i], sig[i + 1], np.abs(x).mean(), np.abs(v).mean()))
    return x, vs


# ------------------------------------------------------------------ the backend's arithmetic, simulated ----
def quantize_weights_per_unit(w, unit=128):
    """The backend's weight quantization simulated: int8 with one scale per `unit` OUTPUT channels
    (`quantize.quantize_weight_w8` per MTP unit), returned dequantized (float32)."""
    out = np.empty_like(w, dtype=np.float32)
    for o in range(0, w.shape[0], unit):
        blk = w[o:o + unit]; s = np.abs(blk).max() / 127.0
        out[o:o + unit] = np.clip(np.rint(blk / s), -127, 127) * s
    return out


class BlockWeightsQ(BlockWeights):
    """BlockWeights with the five linears' weights quantized as the backend does (per-unit int8)."""
    def __init__(self, W, l):
        super().__init__(W, l)
        for n in ("qkv", "o", "w1", "w3", "w2"): setattr(self, n, quantize_weights_per_unit(getattr(self, n)))


def block_fakequant(bw: BlockWeights, x, cos, sin, adaln, a: dict, pm: dict, kc: int):
    """The block with the BACKEND's quantization points simulated in float: per-tensor int8 at every
    linear / matmul input (scales `a[k]/127` from a `Calib`), int16 partial sums per K-chunk
    (`pm[name]*1.05/32767`), P at 1/127. Whether the NPU's deviation from fp32 is the scheme's or a
    bug's: this reproduces the scheme without the device."""
    q8 = lambda v, s: np.clip(np.rint(v / np.float32(s)), -127, 127).astype(np.float32) * np.float32(s)
    def lin(xin, w, name):
        s16 = np.float32(pm[name] * 1.05 / 32767.0); out = np.zeros((xin.shape[0], w.shape[0]), np.float32)
        for c in range(0, w.shape[1], kc):
            part = xin[:, c:c + kc] @ w[:, c:c + kc].T
            out += np.clip(np.rint(part / s16), -32767, 32767).astype(np.float32) * s16
        return out
    s1, g1, s2, g2 = block_modulation(bw, adaln)
    h = q8(rmsnorm(x, bw.an1, NORM_EPS) * s1, a["h"] / 127.0)
    qkv = lin(h, bw.qkv, "qkv").reshape(-1, 3, NH, HEAD_DIM)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    q = rmsnorm(q, bw.norm_q, 1e-5); k = rmsnorm(k, bw.norm_k, 1e-5)
    c, s = cos[:, None, :], sin[:, None, :]
    q = q8(q * c + rotate_half(q) * s, a["q"] / 127.0); k = q8(k * c + rotate_half(k) * s, a["k"] / 127.0); v = q8(v, a["v"] / 127.0)
    L = q.shape[0]; out = np.empty((L, NH, HEAD_DIM), np.float32); sS = a["S"] / 127.0
    for hh in range(NH):
        lg = q8((q[:, hh] @ k[:, hh].T) / np.float32(math.sqrt(HEAD_DIM)), sS)
        lg = lg - lg.max(-1, keepdims=True); e = np.exp(lg); pr = e / e.sum(-1, keepdims=True)
        pr = np.rint(pr * 127) / 127.0
        out[:, hh] = pr @ v[:, hh]
    att = q8(out.reshape(L, NH * HEAD_DIM), a["att"] / 127.0)
    ao = lin(att, bw.o, "o")
    x = x + g1 * rmsnorm(ao, bw.an2, NORM_EPS)
    h2 = q8(rmsnorm(x, bw.fn1, NORM_EPS) * s2, a["h2"] / 127.0)
    gl = q8(silu(lin(h2, bw.w1, "w1")) * lin(h2, bw.w3, "w3"), a["gl"] / 127.0)
    m = lin(gl, bw.w2, "w2")
    return (x + g2 * rmsnorm(m, bw.fn2, NORM_EPS)).astype(np.float32)


def forward_uncond_fakequant(W, tw, latents, t_model, grid, calib):
    gh, gw = grid; cos, sin = mrope(image_position_ids(gh, gw)); ada = adaln_input(tw, t_model)
    x = embed_image_tokens(tw, latents)
    for l in range(W.config["num_layers"]):
        x = block_fakequant(BlockWeights(W, l), x, cos, sin, ada, calib.act[l], calib.part[l], calib.kc)
    return final_layer(tw, x, ada)

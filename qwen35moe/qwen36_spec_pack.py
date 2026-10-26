#!/usr/bin/env python3
"""Qwen3.6-35B-A3B's cache (qwen35moe_pack.py's) completed for the Qwen3.8 speculative runtime (qwen36_spec.py):
  L{l}_router.bin / L{l}_shg.bin   the router [NE, H] and the shared expert's gate [8, H] (row 0; w_post folded, as the npz had them) as
                                   fp32 files mapped with the layer (the device routing reads them from the slot); their meta entries added
  mtp/L40_*                        the MTP layer (a cache of its own: the runtime's QWEN_MTP_CACHE -- its q8 experts do not fit an attention layer's slot layout) (mtp.* of the HF checkpoint, bf16: --mtp a .safetensors of those tensors) in the layer
                                   layout: q / kv / o / fc as q8f GEMM streams, the 256 experts' gate / up / down and the shared expert as
                                   q8 qdot streams, the router / shg, the norms (L40_small.npz), L40_fc.npz -- every bf16 linear quantised
                                   to Q8_0 (amax / 127 a block of 32)
  lm_head_q8_{p}.bin, lm_head_q8.npz, outside_small.npz   the Q6_K head dequantised and requantised to Q8_0 as q8f GEMM streams in 6 parts
                                   (the runtime's head path runs one GEMM mode per cache; the exact Q6_K qdot head stays qwen35moe_generate's)
  q8f                              the marker: gemm_gs(q8=2) streams
    python3 qwen36_spec_pack.py [--cache ~/npu-packs/qwen36moe-npu] [--gguf ...] [--mtp ~/models/qwen3.6-35b-a3b-mtp.safetensors] [--skip-head]
"""
import argparse, json, os, struct, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); os.environ["QWEN_MODEL"] = "qwen3.5-35b-a3b"
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "tinygrad")))); sys.path[:0] = [HERE, os.path.join(HERE, "..", "qwen3.8-27b"), os.path.join(HERE, "..", "qwen3-30b-a3b"), os.path.join(HERE, "..", "ornith-9b")]
from zy import gemm_fp16 as G                                             # noqa: E402
from gguf_read import GGUF, dequant_q6_k                                  # noqa: E402
import qwen38_ref as R                                                    # noqa: E402

NS, KS = 3, 32
CACHE = os.path.expanduser(os.environ.get("QWEN35MOE_NPU", "~/npu-packs/qwen36moe-npu"))
GGUF_PATH = os.environ.get("QWEN35MOE_GGUF", "/mnt/ssd/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
MTP = os.path.expanduser(os.environ.get("QWEN36_MTP", "~/models/qwen3.6-35b-a3b-mtp.safetensors"))
HEAD_PARTS = 6


def write(path, arr):
  with open(path + ".tmp", "wb") as f: f.write(np.ascontiguousarray(arr).tobytes())
  os.replace(path + ".tmp", path)


def q8(w):
  """fp32 [N, K] -> Q8_0 (int8 codes, fp32 scale a block of 32: amax / 127, as ggml)."""
  N, K = w.shape; b = w.reshape(N, K // 32, 32); d = np.abs(b).max(-1) / 127.0; q = np.rint(b / np.where(d > 0, d, 1)[..., None]).clip(-127, 127).astype(np.int8)
  return q.reshape(N, K), d.astype(np.float32)


def stream_q8(q, d):
  ng = -(-q.shape[0] // 48); return np.concatenate([G.pack_b_group_q8f(q, d, g, NS, KS, qb=32) for g in range(ng)]), 2, 48 * ng, q.shape[1]


class BF16:
  def __init__(self, path):
    self.f = open(path, "rb"); n = struct.unpack("<Q", self.f.read(8))[0]; self.h = json.loads(self.f.read(n)); self.base = 8 + n
  def __call__(self, name):
    v = self.h[name]; lo, hi = v["data_offsets"]; self.f.seek(self.base + lo)
    return (np.frombuffer(self.f.read(hi - lo), np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(v["shape"])


def add_router_files(cache, NL):
  for l in range(NL):
    f = os.path.join(cache, f"L{l}_small.npz"); z = dict(np.load(f)); names = [str(k) for k in z["meta_names"]]
    if "router" in names: continue
    router = np.ascontiguousarray(z["router"], np.float32); shg = np.zeros((8, router.shape[1]), np.float32); shg[0] = z["shexp_gate"].reshape(-1)
    write(os.path.join(cache, f"L{l}_router.bin"), router); write(os.path.join(cache, f"L{l}_shg.bin"), shg)
    meta = [list(m) for m in z["meta"]] + [[0, router.shape[0], 8, router.shape[1], 0], [0, 8, 8, router.shape[1], 0]]; names += ["router", "shg"]
    z["meta_names"] = np.array(names); z["meta"] = np.array(meta, np.int64); np.savez(f, **z)
  print(f"router / shg files for {NL} layers", flush=True)


def pad_uniform(cache, NL):
  """Every layer's files the same size per name (the runtime's slot layout is fixed per layer type): the smaller down streams (q5n
  against the q6n layers') zero-padded at the end (the expert stride S is unchanged)."""
  import glob, collections
  for n in sorted({os.path.basename(f).split("_", 1)[1] for f in glob.glob(os.path.join(cache, "L*_*.bin"))}):
    sizes = {f: os.path.getsize(f) for f in glob.glob(os.path.join(cache, f"L*_{n}")) if int(os.path.basename(f)[1:].split("_")[0]) < NL}
    if len(set(sizes.values())) > 1:
      mx = max(sizes.values()); small = [f for f, s in sizes.items() if s < mx]; print(f"  {n}: {len(small)} files padded to {mx}", flush=True)
      for f in small:
        with open(f, "ab") as fh: fh.write(b"\0" * (mx - sizes[f]))


def pack_mtp(cache, path, L):
  t = BF16(path); p = "mtp.layers.0."; meta = {}
  def lin(name, w):
    q, d = q8(w); st, code, npad, K = stream_q8(q, d); write(os.path.join(cache, f"L{L}_{name}.bin"), st); meta[name] = (code, q.shape[0], npad, K, 0)
  lin("q", t(p + "self_attn.q_proj.weight")); lin("kv", np.concatenate([t(p + "self_attn.k_proj.weight"), t(p + "self_attn.v_proj.weight")])); lin("o", t(p + "self_attn.o_proj.weight"))
  lin("sgate", t(p + "mlp.shared_expert.gate_proj.weight")); lin("sup", t(p + "mlp.shared_expert.up_proj.weight")); lin("sdn", t(p + "mlp.shared_expert.down_proj.weight"))
  gu, dn = t(p + "mlp.experts.gate_up_proj"), t(p + "mlp.experts.down_proj"); E, F2, H = gu.shape; F = F2 // 2
  for name, ws in (("gate", gu[:, :F]), ("up", gu[:, F:]), ("down", dn)):
    sts = [stream_q8(*q8(ws[e]))[0] for e in range(E)]; S = sts[0].size; assert all(s.size == S for s in sts)
    write(os.path.join(cache, f"L{L}_{name}.bin"), np.concatenate(sts)); meta[name] = (2, ws.shape[1], -(-ws.shape[1] // 48) * 48, ws.shape[2], S)
  fc = t("mtp.fc.weight"); q, d = q8(fc); st, _, npad, K = stream_q8(q, d); write(os.path.join(cache, f"L{L}_fc.bin"), st); np.savez(os.path.join(cache, f"L{L}_fc.npz"), meta=np.array([fc.shape[0], npad, K], np.int64))
  wpost = 1.0 + t(p + "post_attention_layernorm.weight")
  router = np.ascontiguousarray(t(p + "mlp.gate.weight") * wpost[None]); shg = np.zeros((8, H), np.float32); shg[0] = t(p + "mlp.shared_expert_gate.weight").reshape(-1) * wpost
  write(os.path.join(cache, f"L{L}_router.bin"), router); write(os.path.join(cache, f"L{L}_shg.bin"), shg); meta["router"] = (0, router.shape[0], 8, H, 0); meta["shg"] = (0, 8, 8, H, 0)
  small = {"input_layernorm_weight": t(p + "input_layernorm.weight"), "post_attention_layernorm_weight": t(p + "post_attention_layernorm.weight"),
           "self_attn_q_norm_weight": t(p + "self_attn.q_norm.weight"), "self_attn_k_norm_weight": t(p + "self_attn.k_norm.weight"),
           "mtp_pre_fc_norm_embedding_weight": t("mtp.pre_fc_norm_embedding.weight"), "mtp_pre_fc_norm_hidden_weight": t("mtp.pre_fc_norm_hidden.weight"), "mtp_norm_weight": t("mtp.norm.weight")}
  names = sorted(meta); np.savez(os.path.join(cache, f"L{L}_small.npz"), meta_names=np.array(names), meta=np.array([list(meta[k]) for k in names], np.int64), **small)
  print(f"MTP layer L{L}: {E} experts of {F} (q8), q / kv / o / fc q8f", flush=True)


def pack_head(cache, gguf):
  g = GGUF(gguf); kind, dims, raw = g.raw("output.weight"); assert kind == "Q6_K", kind; K, N = dims[0], dims[1]; blk = np.asarray(raw).reshape(N, K // 256, 210)
  gs = 16 * NS; ngroups = -(-N // gs); per = -(-ngroups // HEAD_PARTS); per = -(-per // 8) * 8; parts = []
  for p_ in range(HEAD_PARTS):
    g0, g1 = p_ * per, min(ngroups, (p_ + 1) * per); f = os.path.join(cache, f"lm_head_q8_{p_}.bin")
    with open(f + ".tmp", "wb") as fh:
      for c0 in range(g0, g1, 64):
        c1 = min(g1, c0 + 64); r0, r1 = c0 * gs, min(c1 * gs, N)
        w = dequant_q6_k(blk[r0:r1].reshape(-1, 210)).reshape(r1 - r0, K); q, d = q8(w)
        fh.write(np.concatenate([G.pack_b_group_q8f(q, d, gg, NS, KS, qb=32) for gg in range(c1 - c0)]).tobytes())
    os.replace(f + ".tmp", f); parts.append((g0, g1)); print(f"  head part {p_}: groups {g0}-{g1}", flush=True)
  np.savez(os.path.join(cache, "lm_head_q8.npz"), parts=np.array(parts, np.int64), n=np.array([N, K]))
  np.savez(os.path.join(cache, "outside_small.npz"), norm=(g.f32("output_norm.weight") - 1.0).astype(np.float32), lm_head_n=np.array([N, gs * ngroups, K]))


def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--cache", default=CACHE); ap.add_argument("--gguf", default=GGUF_PATH); ap.add_argument("--mtp", default=MTP); ap.add_argument("--skip-head", action="store_true"); a = ap.parse_args()
  t0 = time.perf_counter(); add_router_files(a.cache, R.NL); pad_uniform(a.cache, R.NL); os.makedirs(os.path.join(a.cache, "mtp"), exist_ok=True); pack_mtp(os.path.join(a.cache, "mtp"), a.mtp, R.NL); open(os.path.join(a.cache, "mtp", "q8f"), "w").close()
  if not a.skip_head: pack_head(a.cache, a.gguf)
  open(os.path.join(a.cache, "q8f"), "w").close(); print(f"done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__": main()

"""Ornith 1.0 9B's GGUF Q8_0 checkpoint -> the per-layer NPU cache that qwen3.8-27b's modules run (QWEN_MODEL=ornith-9b).

Per layer, the same files as qwen38_pack.py: L{l}_{linear}.bin (the block-scaled GEMM streams -- here Q8_0: int8 codes and a
scale per 32 weights, `gemm_fp16.pack_b_group_q8`, 32-deep K slices, 3 strips a group) and L{l}_small.npz (norms, DeltaNet
parameters, the linears' meta), all in the Hugging Face conventions (ornith_weights undoes the GGUF converter's reordering); then the
final norm (outside_small.npz) and the output head in 6 parts (lm_head_q8_{p}.bin, lm_head_q8.npz). The embedding stays in
the GGUF (looked up on the host).
`--mtp`: the speculative drafts' MTP head -- Qwen3.5-9B's (Ornith's base model; Ornith's GGUF has none), from a GGUF that keeps
it (default the bf16 conversion) -- as layer 32 in the attention layers' format plus L32_fc (mtp.fc), quantised to Q8_0.
`--q8f`: the streams for gemm_gs(q8=2) -- 32-step K-slices whose weights the kernel builds in fp16 (1024 + u - 1152, x d) --
in their own cache (a file `q8f` marks it; the runtime picks the GEMM mode from it).
`--fuse`: the same-input projections as one stream file per layer alongside the per-linear ones (qwen38_pack.fuse_all: q | k|v
and qkv | z, the MTP head too; no GGUF read), used by the runtime unless QWEN_FUSE=0.
  python3 ornith_pack.py [--gguf path] [--out /mnt/ssd/ornith-9b-npu] [--mtp [--mtp-gguf path]] [--q8f] [--fuse]"""
import argparse, os, sys, time
os.environ["QWEN_MODEL"] = "ornith-9b"
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tinygrad")))); HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "qwen3.8-27b"))
import numpy as np
from zy import gemm_fp16 as G                                             # noqa: E402
import qwen38_ref as R                                                    # noqa: E402
from qwen38_pack import LINEARS, SMALL, fuse_all                          # noqa: E402
from ornith_weights import MTP                                            # noqa: E402

NS, KS = 3, 8
Q8F = "--q8f" in sys.argv                                                 # the q8f format (gemm_gs q8=2): the weights built in fp16, 32-step slices
if Q8F: KS = 32
def pack(q, d, g): return (G.pack_b_group_q8f if Q8F else G.pack_b_group_q8)(q, d, g, NS, KS)

def stream(q, d):
  ngroups = -(-q.shape[0] // (16 * NS))
  return np.concatenate([pack(q, d, g) for g in range(ngroups)]), 16 * NS * ngroups

def pack_layer(W, l, out):
  kind, p = R.LAYER_TYPES[l], f"model.language_model.layers.{l}."; meta = {}
  for name, parts in LINEARS[kind]:
    qd = [W.q8(p + x) for x in parts]; q, d = np.concatenate([a for a, _ in qd]), np.concatenate([b for _, b in qd])
    st, npad = stream(q, d); f = os.path.join(out, f"L{l}_{name}.bin"); st.tofile(f + ".tmp"); os.replace(f + ".tmp", f)
    meta[name] = (q.shape[0], npad, q.shape[1])
  small = {n.replace(".", "_"): W.f32(p + n) for n in SMALL[kind]}
  small["meta_names"] = np.array([k for k in meta]); small["meta"] = np.array([meta[k] for k in meta], np.int64)
  np.savez(os.path.join(out, f"L{l}_small.npz"), **small)

HEAD_PARTS = 6                                                            # each part fits a weight slot (sized to the biggest layer, ~273 MB)
def pack_head(W, out):
  q, d = W.g.q8("output.weight"); N, K = q.shape; gs = 16 * NS; ngroups = -(-N // gs); per = -(-ngroups // HEAD_PARTS); per = -(-per // 8) * 8
  parts = []
  for p_ in range(HEAD_PARTS):
    g0, g1 = p_ * per, min(ngroups, (p_ + 1) * per); f = os.path.join(out, f"{R.HEAD}_{p_}.bin")
    with open(f + ".tmp", "wb") as fh:
      for c0 in range(g0, g1, 64):
        c1 = min(g1, c0 + 64); r0, r1 = c0 * gs, min(c1 * gs, N)
        fh.write(np.concatenate([pack(q[r0:r1], d[r0:r1], g) for g in range(c1 - c0)]).tobytes())
    os.replace(f + ".tmp", f); parts.append((g0, g1))
  np.savez(os.path.join(out, f"{R.HEAD}.npz"), parts=np.array(parts, np.int64), n=np.array([N, K]))
  np.savez(os.path.join(out, "outside_small.npz"), norm=W.f32("model.language_model.norm.weight"), lm_head_n=np.array([N, gs * ngroups, K]))

def pack_mtp(path, out):
  """The MTP head as layer NL (its q / k|v / o / gate|up / down have an attention layer's shapes: the same zero-copy views read it),
  L{NL}_fc (mtp.fc [H, 2H]) with its own meta, the norms under qwen38_pack.pack_mtp's names."""
  M, p, L = MTP(path), "mtp.layers.0.", R.NL; meta = {}
  for name, parts in LINEARS["full"]:
    qd = [M.q8(p + x) for x in parts]; q, d = np.concatenate([a for a, _ in qd]), np.concatenate([b for _, b in qd])
    st, npad = stream(q, d); f = os.path.join(out, f"L{L}_{name}.bin"); st.tofile(f + ".tmp"); os.replace(f + ".tmp", f)
    meta[name] = (q.shape[0], npad, q.shape[1])
  small = {n.replace(".", "_"): M.f32(p + n) for n in SMALL["full"]}
  for n in ("mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight"): small[n.replace(".", "_")] = M.f32(n)
  small["meta_names"] = np.array([k for k in meta]); small["meta"] = np.array([meta[k] for k in meta], np.int64)
  q, d = M.q8("mtp.fc"); st, npad = stream(q, d); f = os.path.join(out, f"L{L}_fc.bin"); st.tofile(f + ".tmp"); os.replace(f + ".tmp", f)
  np.savez(os.path.join(out, f"L{L}_fc.npz"), meta=np.array([q.shape[0], npad, q.shape[1]], np.int64))
  np.savez(os.path.join(out, f"L{L}_small.npz"), **small)                 # last: its presence turns the drafts on

def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--gguf", default=os.environ.get("ORNITH_GGUF", "/mnt/ssd/models/ornith-1.0-9b-Q8_0.gguf"))
  ap.add_argument("--out", default=os.environ.get("QWEN_NPU", "/mnt/ssd/ornith-9b-npu"))
  ap.add_argument("--mtp", action="store_true", help="pack the MTP draft head (layer 32) only")
  ap.add_argument("--q8f", action="store_true", help="the q8f format (the kernel builds the weights in fp16; a separate cache, marked by a file q8f)")
  ap.add_argument("--mtp-gguf", default=os.environ.get("ORNITH_MTP", "/mnt/ssd/qbench/models/Qwen_Qwen3.5-9B-bf16.gguf"))
  ap.add_argument("--fuse", action="store_true", help="the fused q | k|v and qkv | z stream files alongside the packed cache (qwen38_pack.py --fuse)"); a = ap.parse_args()
  os.makedirs(a.out, exist_ok=True)
  if a.fuse: fuse_all(a.out, range(R.NL)); return
  marker = os.path.join(a.out, "q8f"); have = [f for f in os.listdir(a.out) if f.endswith(".bin")]
  assert not have or os.path.exists(marker) == a.q8f, "the cache holds the other format: pack into a fresh --out"
  if a.q8f: open(marker, "w").write("gemm_gs q8=2, ks 32\n")
  if a.mtp:
    t0 = time.perf_counter(); pack_mtp(a.mtp_gguf, a.out); print(f"   the MTP head packed as layer {R.NL} in {time.perf_counter() - t0:.0f} s", flush=True); return
  W = R.Weights(a.gguf)
  for l in range(R.NL):                                                   # (L{NL}_small.npz, the MTP head, is --mtp's)
    if os.path.exists(os.path.join(a.out, f"L{l}_small.npz")): continue
    t0 = time.perf_counter(); pack_layer(W, l, a.out); print(f"   layer {l:2d} ({R.LAYER_TYPES[l]}) packed in {time.perf_counter() - t0:.1f} s", flush=True)
  t0 = time.perf_counter(); pack_head(W, a.out); print(f"   head + final norm packed in {time.perf_counter() - t0:.0f} s", flush=True)

if __name__ == "__main__": main()

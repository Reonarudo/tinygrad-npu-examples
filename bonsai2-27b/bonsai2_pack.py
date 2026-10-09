#!/usr/bin/env python3
"""Ternary Bonsai 2 27B's GGUF (PTQ1_0 + prism.hadamard) -> the per-layer NPU cache that qwen3.8-27b's modules run
(QWEN_MODEL=bonsai2-27b), in qwen38_pack.py's layout: L{l}_{linear}.bin + L{l}_small.npz per layer, the fused q | k|v and
qkv | z files with `--fuse`, outside_small.npz (final norm, head shape) and the head in parts (lm_head_tern_{p}.bin,
lm_head_tern.npz).

Weight format: every linear (and the head) as gemm_fp16.pack_b_group_tern's stream -- the trits as 2-bit codes, the PTQ1_0
block scale d (one per row and 128-wide K block = one K-slice of the ks-32 GEMM) in the single-layout fp32 table; markers `tern`
(the format) and `scales` (single) in the cache, which the runtime reads (qwen38_npu.TERN / SCALES). `--tscale f16` stores the
table as fp16 s x 2^15 instead (gemm_fp16.pack_b_group_tern(tscale="f16"): 32 B a strip and slice instead of 64, 2.125 bits a weight,
the GEMM's C bit-identical; a backend whose gemm_gs takes tscale=), marked by a file `tscale` holding f16
(qwen38_npu.TSCALE); bonsai2_repack_f16.py converts an existing fp32-table cache the same way.

The Hadamard contract (bonsai2_gguf.py): every stored weight is W' = W diag(s) Hb, fed Hb (s * x). The device applies Hb in the A
producers (qwen38_kernels.had_a32: the unnormalised transform), so the cache carries:
  * the normalisation 1 / sqrt(1024) = 2^-5 in every scale table (exact: a power of two on an fp32 value; hadamard.npz fold_norm);
  * the signs s of each input width in hadamard.npz (s5120, s6144, s17408): the runtime folds s5120 into the input / post-attention
    / final norms' 1 + w (the inputs of qkv|z, q|kv, gate|up and the head), and the plain producers of o_proj's input (width 6144)
    multiply by s6144 -- o is linear in neither a norm weight shared across heads (DeltaNet) nor V shared by a GQA group
    (attention), so s6144 cannot fold into weights;
  * s17408 (down's input = silu(gate) * up) in the up rows' scales: row k's scale x s[k] makes u_k s_k exactly.
The DeltaNet's V heads are restored to the HF (grouped) order on qkv's V rows and z's rows (bonsai2_weights.OUT_ROWS); ssm_out
keeps its grouped columns (the GGUF's gdn_v_grouped: the runtime's o rows are grouped already). ssm_alpha / ssm_beta (BF16, not
folded), the norms, the conv, A_log and dt_bias go to L{l}_small.npz under the HF names, as for Qwen3.8 / Ornith.

The token embedding stays in the GGUF: the runtime looks a row up on the host (bonsai2_weights.Weights.embed: one PTQ1_0 row
decoded, then s * (Hb z); ~1 ms a token, bit-identical to the reference implementation's) instead of holding a 2.5 GB fp16 table.

    python3 bonsai2_pack.py [--gguf /mnt/ssd/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf] [--out /mnt/ssd/bonsai2-npu] [--layers 0-63]
    python3 bonsai2_pack.py --head          # outside_small.npz + the head (4 parts)
    python3 bonsai2_pack.py --fuse          # q | k|v and qkv | z as one stream file per layer (qwen38_pack.fuse_all; no GGUF read)
    python3 bonsai2_pack.py --mtp DIR       # the drafter: Qwen3.8-27B's MTP layer from DIR/mtp.safetensors -> <out>/mtp; tokenizer.json -> <out>

`--mtp` (speculative decoding; no GGUF read): the GGUF has no MTP head, so the drafter is Qwen3.8-27B-FP8's (`mtp.safetensors`, 0.48 GB,
the only weights needed from that model). It is packed by qwen38_pack.pack_mtp as layer 64 of a small E4M3 cache of its own, the
subfolder `mtp` (the runtime's default QWEN_MTP_CACHE): its streams are block-scaled E4M3, not ternary, so it carries its own markers
(`scales` = single, as the trunk's, and no `tern`). DIR's tokenizer.json (Qwen3.5's tokenizer, the same ids) is copied into <out>.
"""
import argparse, os, sys, time
os.environ["QWEN_MODEL"] = "bonsai2-27b"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser(os.environ.get("TG", os.path.join(HERE, "..", "tinygrad")))); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "qwen3.8-27b"))
import numpy as np
from zy import gemm_fp16 as G                                             # noqa: E402
import qwen38_ref as R                                                    # noqa: E402
from qwen38_pack import LINEARS, SMALL, fuse_all                          # noqa: E402
from bonsai2_weights import Weights, OUT_ROWS                            # noqa: E402

NS, KS, QK = 3, 32, 128
FOLD = np.float32(2.0 ** -5)                                              # 1 / sqrt(block 1024), exact in the fp32 table
HEAD_PARTS = 4                                                            # as the fp8 head: 1296 groups (62 208 columns) a part, the last shorter
assert R.HAD and R.HEAD == "lm_head_tern"

TSCALE = "f16"                                                            # the scale table's format (--tscale): f16 (default) | f32
def tkw(): return {"tscale": "f16"} if TSCALE == "f16" else {}              # pack_b_group_tern's keyword (older backends lack it)

def stream(t, scale):
  """ternary [N, K] (int8) + scale [N, K / 128] (fp32) -> the whole gemm_gs(tern) stream (all groups), N padded to the group size."""
  ngroups = -(-t.shape[0] // (16 * NS))
  return np.concatenate([G.pack_b_group_tern(t, scale, g, NS, KS, **tkw()) for g in range(ngroups)]), 16 * NS * ngroups

def unpack_stream(buf, npad, K):
  """The inverse of stream(): bytes -> (codes c = w + 1, uint8 [npad, K]; the scales as stored, fp32 [npad, K / 128]). Either
  table format (the record size tells: fp32 s x 2^18, 64 B a strip, or fp16 s x 2^15, 32 B: lane 2i = column i, 2i + 1 = column 8 + i)."""
  ng, nsl = npad // (16 * NS), K // QK; cb = NS * KS * 16; tb = len(buf) // (ng * nsl) - cb
  assert tb in (NS * 64, NS * 32) and len(buf) == ng * nsl * (cb + tb), (len(buf), npad, K)
  b = np.frombuffer(buf, np.uint8).reshape(ng, nsl, cb + tb)
  codes = b[..., :cb].reshape(ng, nsl, NS, KS // 2, 2, 16)
  j = np.stack([(codes >> (6 - 2 * jj)) & 3 for jj in range(4)], 5)       # [g][slice][s][q][h][j][16 = n k]
  c = j.reshape(ng, nsl, NS, KS // 2, 2, 4, 4, 4).transpose(0, 2, 5, 6, 1, 3, 4, 7).reshape(npad, K)   # [g][s][j][n] x [slice][q][h][k]
  if tb == NS * 32:                                                       # fp16 s x 2^15: [s][i][half] -> column 8 half + i
    h = np.ascontiguousarray(b[..., cb:]).view(np.float16).reshape(ng, nsl, NS, 8, 2).transpose(0, 1, 2, 4, 3).reshape(ng, nsl, 16 * NS)
    return c, (h.astype(np.float32) * np.float32(2.0 ** -15)).transpose(0, 2, 1).reshape(npad, nsl)
  sc = np.ascontiguousarray(b[..., cb:]).view(np.float32).reshape(ng, nsl, 16 * NS).transpose(0, 2, 1).reshape(npad, nsl)
  return c, sc / np.float32(2.0 ** 18)

def signs(W):
  """{input width: the +-1 sign vector} of the GGUF's Hadamard contract (5120, 6144, 17408)."""
  return {int(k): np.asarray(v, np.float32) for k, v in W.g.had.signs.items()}

def ternary(W, hf):
  """An HF-named linear -> (trits int8 [N, K], d float32 [N, K / 128]) in the HF row order (the DeltaNet's V heads grouped)."""
  t_, rest = W.gguf_name(hf); t, d = W.g.ptq(t_)
  if rest in OUT_ROWS: r = OUT_ROWS[rest]; t, d = t[r], d[r]
  return t, d.astype(np.float32)

def folded(W, hf, sg):
  """(trits, the scales the stream stores): d x 2^-5, and for the up projection x the down input's signs (row k -> s17408[k])."""
  t, d = ternary(W, hf); sc = d * FOLD
  if hf.endswith("mlp.up_proj"): sc = sc * sg[R.INTER][:, None]
  return t, sc.astype(np.float32)

def pack_layer(W, l, out, sg):
  kind, p = R.LAYER_TYPES[l], f"model.language_model.layers.{l}."; meta = {}
  for name, parts in LINEARS[kind]:
    ts = [folded(W, p + q, sg) for q in parts]; t, sc = np.concatenate([a for a, _ in ts]), np.concatenate([b for _, b in ts])
    st, npad = stream(t, sc); f = os.path.join(out, f"L{l}_{name}.bin"); st.tofile(f + ".tmp"); os.replace(f + ".tmp", f)
    meta[name] = (t.shape[0], npad, t.shape[1])
  small = {n.replace(".", "_"): W.f32(p + n) for n in SMALL[kind]}
  small["meta_names"] = np.array([k for k in meta]); small["meta"] = np.array([meta[k] for k in meta], np.int64)
  np.savez(os.path.join(out, f"L{l}_small.npz"), **small)

def pack_head(W, out):
  """output.weight (ternary [248320, 5120], folded: its input is the final norm's, width 5120) in HEAD_PARTS files of whole groups."""
  name = "output.weight"; assert W.g.folded(name); K, N = W.g.tensors[name][0][:2]
  gs = 16 * NS; ngroups = -(-N // gs); per = -(-ngroups // HEAD_PARTS); per = -(-per // 8) * 8; parts = []
  for p_ in range(HEAD_PARTS):
    g0, g1 = p_ * per, min(ngroups, (p_ + 1) * per); f = os.path.join(out, f"{R.HEAD}_{p_}.bin")
    with open(f + ".tmp", "wb") as fh:
      for c0 in range(g0, g1, 64):                                          # 64 groups (3072 rows) at a time
        c1 = min(g1, c0 + 64); r0, r1 = c0 * gs, min(c1 * gs, N)
        t, d = W.g.ptq(name, slice(r0, r1)); sc = (d.astype(np.float32) * FOLD).astype(np.float32)
        fh.write(np.concatenate([G.pack_b_group_tern(t, sc, g, NS, KS, **tkw()) for g in range(c1 - c0)]).tobytes())
    os.replace(f + ".tmp", f); parts.append((g0, g1))
  np.savez(os.path.join(out, f"{R.HEAD}.npz"), parts=np.array(parts, np.int64), n=np.array([N, K]))
  np.savez(os.path.join(out, "outside_small.npz"), norm=W.f32("model.language_model.norm.weight"), lm_head_n=np.array([N, gs * ngroups, K]))

def markers(out, W):
  """The cache's format markers and the Hadamard signs (written first; a cache holding another format refuses)."""
  have = [f for f in os.listdir(out) if f.endswith(".bin")]
  assert not have or os.path.exists(os.path.join(out, "tern")), f"{out} holds another format's streams: pack into a fresh --out"
  ft = os.path.join(out, "tscale"); was = open(ft).read().split()[0] if os.path.exists(ft) else "f32"
  assert not have or was == TSCALE, f"{out} holds {was}-table streams, not --tscale {TSCALE}: pack into a fresh --out"
  open(os.path.join(out, "tern"), "w").write("gemm_fp16.pack_b_group_tern (ks 32, ns 3), gemm_gs(tern=True); bonsai2_pack.py\n")
  open(os.path.join(out, "scales"), "w").write("single\nthe ternary stream's single-layout scale table (gemm_gs scales='single')\n")
  if TSCALE == "f16": open(ft, "w").write("f16\nthe ternary scale tables as fp16 s x 2^15 (gemm_fp16.pack_b_group_tern(tscale='f16'), gemm_gs(tscale='f16')); bonsai2_pack.py --tscale f16\n")
  h = W.g.had
  np.savez(os.path.join(out, "hadamard.npz"), block=np.array(h.block), fold_norm=np.array(1), **{f"s{w}": v for w, v in signs(W).items()})

def pack_drafter(qdir, out):
  """Qwen3.8-27B-FP8's MTP layer (qdir/mtp.safetensors) -> out/mtp (layer 64, single-layout scales); qdir/tokenizer.json -> out."""
  import shutil
  import qwen38_pack as QP
  qdir = os.path.expanduser(qdir); src = os.path.join(qdir, "mtp.safetensors")
  assert os.path.exists(src), f"{src} not found: --mtp takes the Qwen3.8-27B-FP8 folder holding mtp.safetensors (download.sh)"
  d = os.path.join(out, "mtp"); os.makedirs(d, exist_ok=True)
  QP.SCALES = "single"; QP.scales_marker(d, "single")                   # the trunk's layout: one SCALES for the whole process
  class _W:                                                               # pack_mtp reads one file through W.file
    def file(self, name): return R.SafeFile(os.path.join(qdir, name))
  QP.pack_mtp(_W(), d)
  tok = os.path.join(qdir, "tokenizer.json")
  if os.path.exists(tok) and not os.path.exists(os.path.join(out, "tokenizer.json")): shutil.copy(tok, out)

def main():
  global TSCALE
  ap = argparse.ArgumentParser()
  ap.add_argument("--gguf", default=os.environ.get("BONSAI_GGUF", "/mnt/ssd/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf"))
  ap.add_argument("--out", default=os.environ.get("QWEN_NPU", "/mnt/ssd/bonsai2-npu")); ap.add_argument("--layers", default="0-63")
  ap.add_argument("--head", action="store_true", help="the final norm and the head only")
  ap.add_argument("--fuse", action="store_true", help="the fused q | k|v and qkv | z stream files alongside (qwen38_pack.fuse_all; no GGUF read)")
  ap.add_argument("--tscale", default=TSCALE, choices=("f32", "f16"), help="the ternary scale table: fp16 s x 2^15 (default) or fp32 s x 2^18 (the older format)")
  ap.add_argument("--mtp", metavar="DIR", help="the drafter: Qwen3.8-27B-FP8's MTP layer (DIR/mtp.safetensors) into <out>/mtp, DIR/tokenizer.json into <out>")
  a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); lo, hi = (int(v) for v in a.layers.split("-"))
  TSCALE = a.tscale
  if a.fuse: fuse_all(a.out, range(lo, hi + 1)); return
  if a.mtp:
    t0 = time.perf_counter(); pack_drafter(a.mtp, a.out); print(f"   the drafter (Qwen3.8-27B's MTP layer) packed into {os.path.join(a.out, 'mtp')} in {time.perf_counter() - t0:.0f} s", flush=True); return
  W = Weights(a.gguf); markers(a.out, W); sg = signs(W)
  assert set(sg) == {R.H, R.NV * R.DV, R.INTER} and R.NV * R.DV == R.NH * R.HD, sorted(sg)
  if a.head:
    t0 = time.perf_counter(); pack_head(W, a.out); print(f"   head + final norm packed in {time.perf_counter() - t0:.0f} s", flush=True); return
  for l in range(lo, hi + 1):
    if os.path.exists(os.path.join(a.out, f"L{l}_small.npz")): print(f"   layer {l:2d} present"); continue
    t0 = time.perf_counter(); pack_layer(W, l, a.out, sg); print(f"   layer {l:2d} ({R.LAYER_TYPES[l]}) packed in {time.perf_counter() - t0:.1f} s", flush=True)

if __name__ == "__main__": main()

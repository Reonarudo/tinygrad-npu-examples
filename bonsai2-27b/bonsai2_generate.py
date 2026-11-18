"""Ternary Bonsai 2 27B text generation on the NPU: qwen3.8-27b's generator with QWEN_MODEL=bonsai2-27b (the same Qwen3.5 layer
kernels at Qwen3.8-27B's sizes; every linear and the head on the ternary GEMM, gemm_gs(tern=True); the A producers apply the
model's Hadamard transform), the prompt prefilled in one pass. Greedy speculative decoding by default: the GGUF has no MTP head, so
the drafter is Qwen3.8-27B's MTP layer (bonsai2_pack.py --mtp: the subfolder `mtp` of the cache), fed Bonsai's final-normed hidden
state and drafting through Bonsai's own embedding and head (README "Speculative decoding"). Every token is the verify pass's argmax,
so the ids are plain greedy decoding's (QWEN_SPEC=0: plain decoding, no drafter).

  python3 bonsai2_generate.py "prompt" [--max-new N] [--raw] [--ids 760,6511,...] [--out ids.npz]

Paths (set them to yours): BONSAI_GGUF (the checkpoint; the embedding rows are looked up in it), QWEN_NPU (the packed cache,
bonsai2_pack.py), QWEN_TOK (tokenizer.json's folder; default the cache), QWEN_MTP_CACHE (the drafter's cache; default QWEN_NPU/mtp).
Options: QWEN_MTP_HEAD=qwen drafts through Qwen3.8-27B's fp8 head (QWEN_MTP_CACHE = a Qwen3.8-27B cache packed with --scales single
and --lm-head-fp8; +3 %), QWEN_MTP_EMBED=qwen through its embedding table (QWEN_MTP_DIR = the Qwen3.8-27B-FP8 folder). The drafting
defaults below were tuned on the board for each head (README); each can be overridden."""
import os, sys
os.environ.setdefault("QWEN_MODEL", "bonsai2-27b")       # or bonsai2-9b / -4b / -2b (their drafters: bonsai2_pack.py --mtp-gguf)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "qwen3.8-27b"))
from bonsai2_weights import PATHS                       # noqa: E402
import qwen38_ref as R                                  # noqa: E402  (the profile: NL, the MTP layer's index)
os.environ.setdefault("DEV", "ZHOUYI")                  # tinygrad's default device
if os.environ.get("QWEN_SHARED_NPU") == "1":            # another NPU process runs beside this one (two servers): nothing of ours in the cluster's shared GM
  os.environ.setdefault("ZHOUYI_TEXT_GM", "0"); os.environ.setdefault("QWEN_AGM", "0")   # (kernel text and the GEMMs' A operand stay in DDR; ~5 % slower)
os.environ.setdefault("QWEN_NPU", PATHS[os.environ["QWEN_MODEL"]][1])
os.environ.setdefault("QWEN_DIR", os.environ.get("BONSAI_GGUF", PATHS[os.environ["QWEN_MODEL"]][0]))
os.environ.setdefault("QWEN_TOK", os.environ["QWEN_NPU"])
if os.environ["QWEN_MODEL"] != "bonsai2-27b":           # the one-pass prefill's DeltaNet kernel (gdn_lsr) splits 48 value heads over 12 TECs;
  os.environ.setdefault("QWEN_PREFILL", "chunked")      # 32 (9B / 4B) or 16 (2B) take the chunked prefill, as Ornith-9B does
os.environ.setdefault("ZHOUYI_TEXT_GM", "1")              # kernel code fetched from the cluster's GM, not DDR (4.4x cheaper cold fetch; -17 ms a verify pass)
os.environ.setdefault("ZHOUYI_CHAIN_MAX", "16")           # a layer's launches as one job (8: two); -16 ms a 4-row verify pass
os.environ.setdefault("QWEN_LAYER_BLOCK", "4")            # the verify pass's layers 4 at a time, a block one job (and the head one job)
# verify geometries up to 4 rows: a 5th row still costs ~+80 ms a pass on the 27B (rows 6-8 ~+15 each). The 2B: 8 -- its prompt chunks
# run 8 rows (prefill 131 -> 180 tok/s on 512 tokens); decode still draws <= 3 drafts (QWEN_DRAFT_MAX) on the <= 4-row geometries,
# which keep their own 4-slot deferred commit (qwen38_generate QWEN_MS_SPLIT), so it costs decode nothing measurable (2026-11-07)
os.environ.setdefault("QWEN_SPEC", "8" if os.environ["QWEN_MODEL"] == "bonsai2-2b" else "4")
# The NPU + CPU column split (cpu_split.py): on by default on the 27B and the 2B when the cache has
# its CPU files (cpu_split_pack.py) -- the 27B 6.70 -> 11.42 tok/s on the 10-prompt gate, the 2B +10 % (the same tokens; QWEN_CPU_SPLIT=0:
# the NPU alone). With it: the chain verify (the split's deferred steps run in its kernels), the host on the NPU's interrupt CPU alone and
# the pool on the other big cores (cpu_split.pool_cpus: never the little ones); on the 27B the q4t drafter (its 4-bit linears split too).
# The 2B's GEMMs are smaller against their fixed cost a call, so its best shares are lower (0.30-0.40 flat)
SPLIT_SHARES = {"bonsai2-27b": lambda gt: f"qkvz=0.70,qkv={0.65 if gt else 0.50},gu=0.50,o=0.44,dn=0.44", "bonsai2-2b": lambda gt: "qkvz=0.35,qkv=0.35,gu=0.35,o=0.30,dn=0.30"}
SPLIT_HEAD = {"bonsai2-27b": 0.50, "bonsai2-2b": 0.35}
if os.environ["QWEN_MODEL"] in SPLIT_SHARES and os.environ["QWEN_SPEC"] != "0" and "QWEN_CPU_SPLIT" not in os.environ:
  _c = os.path.expanduser(os.environ["QWEN_NPU"]); _q4t = os.path.join(_c, "mtp_q4t")
  if "QWEN_MTP_CACHE" not in os.environ and os.path.exists(os.path.join(_q4t, f"L{R.NL}_small.npz")): os.environ["QWEN_MTP_CACHE"] = _q4t
  _m = os.path.expanduser(os.environ.get("QWEN_MTP_CACHE", os.path.join(_c, "mtp")))
  if os.path.exists(os.path.join(_c, "cpu_split", "L0_gu.c2c")) and os.path.exists(os.path.join(_m, f"L{R.NL}_small.npz")):
    import glob, zipfile
    _gt = any("gtail.npy" in zipfile.ZipFile(f).namelist() for f in glob.glob(os.path.join(_c, "L*_fused.npz")))   # (the gates at the q|kv tail)
    _hd = os.path.exists(os.path.join(_c, "cpu_split", "head_0.c2c"))
    os.environ["QWEN_CPU_SPLIT"] = SPLIT_SHARES[os.environ["QWEN_MODEL"]](_gt) + (f",head={SPLIT_HEAD[os.environ['QWEN_MODEL']]:.2f}" if _hd else "")
    if os.path.isdir(os.path.join(_m, "cpu_split")): os.environ.setdefault("QWEN_CPU_SPLIT_MTP", "0.55")
if os.environ.get("QWEN_CPU_SPLIT", "0") not in ("0", "off"):
  os.environ.setdefault("QWEN_SPEC_TREE", "off"); os.environ.setdefault("QWEN_CPUS", "irq")
if os.environ["QWEN_MODEL"] != "bonsai2-27b": os.environ.setdefault("QWEN_KV_ASID", "1")    # the K / V caches in the NPU's ASIDs 1-3 (3 GiB each, reached by DMA): contexts past the ASID0 window
if os.environ.get("QWEN_KV_ASID") == "1": os.environ.setdefault("ZHOUYI_ASID_BASES", "0x100000000,0x200000000,0x300000000")   # the chain head's ASID bases (the KMD's layout)
# spare rows take the drafts' rank-2/3 candidates (+3.9 % on 10 prompts, exact) -- not with the caches in ASIDs (its commit moves K / V rows in every layer's cache through ASID0 pointers)
os.environ.setdefault("QWEN_SPEC_TREE", "off" if os.environ.get("QWEN_KV_ASID") == "1" else "leaf")
os.environ.setdefault("QWEN_TREE_TC", "0.5")             # the chain drafts on while its path probability >= 0.5
os.environ.setdefault("QWEN_SPEC_GEOS", "2,3,4,8" if os.environ["QWEN_SPEC"] == "8" else "2,3,4")
os.environ.setdefault("QWEN_DRAFT_MAX", "3")
os.environ.setdefault("QWEN_MTP_EMBED", "bonsai")        # the drafted token's embedding: Bonsai's (the GGUF row)
os.environ.setdefault("QWEN_MTP_HEAD", "bonsai")         # the draft head: Bonsai's ternary head, parts 0-1 (ids < 124416)
QHEAD = os.environ["QWEN_MTP_HEAD"] == "qwen"            # Qwen3.8-27B's fp8 head: part 1 only when part 0's top logit is <= 20
os.environ.setdefault("QWEN_DRAFT_TAU", "0.4" if QHEAD else "0.2")
os.environ.setdefault("QWEN_DRAFT_PARTS", "2,thresh:20" if QHEAD else "2")
mtp = os.path.expanduser(os.environ.get("QWEN_MTP_CACHE", os.path.join(os.environ["QWEN_NPU"], "mtp")))
drafter = os.path.exists(os.path.join(mtp, f"L{R.NL}_small.npz"))
if os.environ["QWEN_SPEC"] != "0" and not drafter:
  print(f"   no drafter in {mtp} (bonsai2_pack.py --mtp <Qwen3.8-27B-FP8 folder>, or QWEN_MTP_CACHE): plain decoding", flush=True)
  # the 27B prefills in one pass; the smaller sizes' chunked prefill runs on the verify geometries, so they stay (no MTP layer loaded)
  if os.environ["QWEN_MODEL"] == "bonsai2-27b": os.environ["QWEN_SPEC"] = "0"
  else: os.environ.setdefault("QWEN_DRAFT", "none")
# the decode loop's token ids chosen, embedded, accepted and drafted on the NPU: with the drafter, by default on the 27B. The 2B / 4B
# (512 Hadamard blocks) take it too (QWEN_NPU_IDS=1: embed_tern_src(hb), 2026-11-17; the 2B +12.8 % at one draft a pass, lossless);
# not the 9B (its 4096 block does not fit the embedding kernel's LSRAM)
os.environ.setdefault("QWEN_NPU_IDS", "1" if os.environ["QWEN_SPEC"] != "0" and drafter and os.environ["QWEN_MODEL"] == "bonsai2-27b" else "0")
import qwen38_generate                                                   # noqa: E402

if __name__ == "__main__": qwen38_generate.main()

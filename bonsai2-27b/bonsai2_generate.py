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
os.environ.setdefault("QWEN_NPU", PATHS[os.environ["QWEN_MODEL"]][1])
os.environ.setdefault("QWEN_DIR", os.environ.get("BONSAI_GGUF", PATHS[os.environ["QWEN_MODEL"]][0]))
os.environ.setdefault("QWEN_TOK", os.environ["QWEN_NPU"])
if os.environ["QWEN_MODEL"] != "bonsai2-27b":           # the one-pass prefill's DeltaNet kernel (gdn_lsr) splits 48 value heads over 12 TECs;
  os.environ.setdefault("QWEN_PREFILL", "chunked")      # 32 (9B / 4B) or 16 (2B) take the chunked prefill, as Ornith-9B does
os.environ.setdefault("ZHOUYI_TEXT_GM", "1")              # kernel code fetched from the cluster's GM, not DDR (4.4x cheaper cold fetch; -17 ms a verify pass)
os.environ.setdefault("ZHOUYI_CHAIN_MAX", "16")           # a layer's launches as one job (8: two); -16 ms a 4-row verify pass
os.environ.setdefault("QWEN_LAYER_BLOCK", "4")            # the verify pass's layers 4 at a time, a block one job (and the head one job)
os.environ.setdefault("QWEN_SPEC", "4")                 # verify geometries up to 4 rows: a 5th row still costs ~+80 ms a pass (rows 6-8 ~+15 each)
if os.environ["QWEN_MODEL"] != "bonsai2-27b": os.environ.setdefault("QWEN_KV_ASID", "1")    # the K / V caches in the NPU's ASIDs 1-3 (3 GiB each, reached by DMA): contexts past the ASID0 window
if os.environ.get("QWEN_KV_ASID") == "1": os.environ.setdefault("ZHOUYI_ASID_BASES", "0x100000000,0x200000000,0x300000000")   # the chain head's ASID bases (the KMD's layout)
# spare rows take the drafts' rank-2/3 candidates (+3.9 % on 10 prompts, exact) -- not with the caches in ASIDs (its commit moves K / V rows in every layer's cache through ASID0 pointers)
os.environ.setdefault("QWEN_SPEC_TREE", "off" if os.environ.get("QWEN_KV_ASID") == "1" else "leaf")
os.environ.setdefault("QWEN_TREE_TC", "0.5")             # the chain drafts on while its path probability >= 0.5
os.environ.setdefault("QWEN_SPEC_GEOS", "2,3,4")
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
# the decode loop's token ids chosen, embedded, accepted and drafted on the NPU: with the drafter, on the 27B (its embedding kernel
# holds one 1024 block; the 9B's is 4096)
os.environ.setdefault("QWEN_NPU_IDS", "1" if os.environ["QWEN_SPEC"] != "0" and drafter and os.environ["QWEN_MODEL"] == "bonsai2-27b" else "0")
import qwen38_generate                                                   # noqa: E402

if __name__ == "__main__": qwen38_generate.main()

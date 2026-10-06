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
os.environ.setdefault("QWEN_MODEL", "bonsai2-27b")
os.environ.setdefault("DEV", "ZHOUYI")                  # tinygrad's default device
os.environ.setdefault("QWEN_NPU", "/mnt/ssd/bonsai2-npu")
os.environ.setdefault("QWEN_DIR", os.environ.get("BONSAI_GGUF", "/mnt/ssd/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf"))
os.environ.setdefault("QWEN_TOK", os.environ["QWEN_NPU"])
os.environ.setdefault("QWEN_NPU_IDS", "1")                 # the decode loop's token ids chosen, embedded, accepted and drafted on the NPU
os.environ.setdefault("ZHOUYI_CHAIN_MAX", "16")           # a layer's launches as one job (8: two); -16 ms a 4-row verify pass
os.environ.setdefault("QWEN_LAYER_BLOCK", "4")            # the verify pass's layers 4 at a time, a block one job (and the head one job)
os.environ.setdefault("QWEN_SPEC", "4")                 # verify geometries up to 4 rows: a 5th row still costs ~+80 ms a pass (rows 6-8 ~+15 each)
os.environ.setdefault("QWEN_SPEC_TREE", "leaf")         # spare rows take the drafts' rank-2/3 candidates (+3.9 % on 10 prompts, exact)
os.environ.setdefault("QWEN_TREE_TC", "0.5")             # the chain drafts on while its path probability >= 0.5
os.environ.setdefault("QWEN_SPEC_GEOS", "2,3,4")
os.environ.setdefault("QWEN_DRAFT_MAX", "3")
os.environ.setdefault("QWEN_MTP_EMBED", "bonsai")        # the drafted token's embedding: Bonsai's (the GGUF row)
os.environ.setdefault("QWEN_MTP_HEAD", "bonsai")         # the draft head: Bonsai's ternary head, parts 0-1 (ids < 124416)
QHEAD = os.environ["QWEN_MTP_HEAD"] == "qwen"            # Qwen3.8-27B's fp8 head: part 1 only when part 0's top logit is <= 20
os.environ.setdefault("QWEN_DRAFT_TAU", "0.4" if QHEAD else "0.2")
os.environ.setdefault("QWEN_DRAFT_PARTS", "2,thresh:20" if QHEAD else "2")
mtp = os.path.expanduser(os.environ.get("QWEN_MTP_CACHE", os.path.join(os.environ["QWEN_NPU"], "mtp")))
if os.environ["QWEN_SPEC"] != "0" and not os.path.exists(os.path.join(mtp, "L64_small.npz")):
  print(f"   no drafter in {mtp} (bonsai2_pack.py --mtp <Qwen3.8-27B-FP8 folder>, or QWEN_MTP_CACHE): plain decoding", flush=True)
  os.environ["QWEN_SPEC"] = "0"
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "qwen3.8-27b")); sys.path.insert(0, HERE)
import qwen38_generate                                                   # noqa: E402

if __name__ == "__main__": qwen38_generate.main()

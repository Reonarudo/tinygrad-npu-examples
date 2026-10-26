"""The numpy reference's greedy ids for a prompt (no device): what the device paths are compared against on a near-tied token.
python3 ref_ids.py [n_new] [prompt]"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import qwen3moe_ref as QR
from tokenizers import Tokenizer
n_new = int(sys.argv[1]) if len(sys.argv) > 1 else 64
prompt = sys.argv[2] if len(sys.argv) > 2 else "The old lighthouse keeper had lived alone on the rocky island for nearly forty years. Continue the story in three sentences."
tok = Tokenizer.from_file(os.path.join(os.environ.get("QWEN3MOE_TOK", "/mnt/ssd/models/qwen3-30b-a3b-tokenizer"), "tokenizer.json"))
M = QR.Model(); ids = tok.encode(QR.chat_text(prompt), add_special_tokens=False).ids; t0 = time.perf_counter()
out = M.greedy(ids, n_new); print(f"reference ids ({len(out)} tokens, {time.perf_counter() - t0:.0f} s): {list(map(int, out))}", flush=True)

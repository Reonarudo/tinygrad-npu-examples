"""The batched one-token attention: attn_dec3 (attn_gqa.py: the decode step's attention over the cache in kv-head units, one row) over M
INDEPENDENT rows in one launch. Row r: q_rows row r (pitch NH 2 HD: [q | gate] a head), kv_rows row r (pitch 2 NKV HD), the cache and
the q / k norm weights at index idx[2 r], position idx[2 r + 1] (the per-user replicas: user x attention layers + layer), its output
row r of o_rows (pitch NH HD). Units = (row, kv head, part) over the 12 tasks; the new k / v rows go into the cache from the unit
itself (part 0) before its own reads, so every row's results are attn_dec3's bits (gate: sim/attn_batch_sim.py).
args: attn_dec3's (o_rows, q_rows, kv_rows, qnw_all, knw_all, rope, Kall, Vall, idx [M][2] ([M][3] with nidx), desc)."""
from qwen38_kernels import NT
import attn_gqa as AG


def attn_decb_src(NH, NKV, HD, TMAX, ROT, eps, M, U=1, BT=2, nidx=False, kvb=0, kvh=False):
  """`nidx`: idx [M][3] -- a row's layer (its norm weights), position and cache index. `kvb` / `kvh`: attn_dec3_src's (the cache's DMA base
  code; fp16 caches)."""
  src = AG.attn_dec3_src(NH, NKV, HD, TMAX, ROT, eps, U=U, BT=BT, nidx=nidx, kvb=kvb, kvh=kvh); CT = "__global half" if kvh else "__global float"; W = 3 if nidx else 2; g = NH // NKV; QR, KR, OR = NH * 2 * HD, 2 * NKV * HD, NH * HD; NU = NKV * U
  def rep(a, b, n=1):
    nonlocal src; assert src.count(a) == n, (a[:70], src.count(a)); src = src.replace(a, b)
  rep("__kernel void attn_dec2(", "__kernel void attn_decb(")
  i0 = src.index(f"  int L = idx[{2 if nidx else 0}], pos = idx[1];"); i1 = src.index("  int kvh = core_id / "); head = src[i0:i1]
  cache_w = head[head.index("    for (int kh = core_id"):head.index("    return;")]
  cache_w = cache_w.replace(f"for (int kh = core_id - {NU}; kh < {NKV}; kh += {NT - NU}) {{", "if (part == 0) { int kh = kvh;")
  cache_w = cache_w.replace(f"(int)(kv_rows + {NKV * HD} + kh *", f"(int)(kv_rows + r * {KR} + {NKV * HD} + kh *").replace("(int)(kv_rows + kh *", f"(int)(kv_rows + r * {KR} + kh *")
  new_head = (f"  for (int un = core_id; un < {M * NU}; un += {NT}) {{ int r = un / {NU}, kvh = (un % {NU}) / {U}, part = un % {U};\n"
              f"  int L = idx[{W} * r + {2 if nidx else 0}], pos = idx[{W} * r + 1];" + (f" int Ln = idx[{W} * r];\n" if nidx else "\n")
              + f"  {CT}* Kc = Kall + (L * {TMAX}) * {NKV * HD}; {CT}* Vc = Vall + (L * {TMAX}) * {NKV * HD};\n"
              + f"  __global float* qnw = qnw_all + {'Ln' if nidx else 'L'} * {HD}; __global float* knw = knw_all + {'Ln' if nidx else 'L'} * {HD}; __global float* cs = rope + pos * {2 * ROT};\n" + cache_w)
  src = src[:i0] + new_head + src[i1:]
  rep(f"  int kvh = core_id / {U}, part = core_id % {U}, h0 = kvh * {g} + part *", f"  int h0 = kvh * {g} + part *")
  rep(f"(int)(q_rows + h0 * {2 * HD})", f"(int)(q_rows + r * {QR} + h0 * {2 * HD})")
  rep(f"(int)(kv_rows + kvh * {HD})", f"(int)(kv_rows + r * {KR} + kvh * {HD})")
  rep(f"(int)(kv_rows + {NKV * HD} + kvh * {HD})", f"(int)(kv_rows + r * {KR} + {NKV * HD} + kvh * {HD})")
  rep(f"(int)(q_rows + (h0 + j0) * {2 * HD} + {HD})", f"(int)(q_rows + r * {QR} + (h0 + j0) * {2 * HD} + {HD})")
  rep(f"(int)(o_rows + h0 * {HD})); DMA_WAIT(2);\n}}", f"(int)(o_rows + r * {OR} + h0 * {HD})); DMA_WAIT(2);\n  }}\n}}")
  assert "core_id" not in src.split("for (int un")[1].split("\n", 1)[1].split("int h0 =")[0], "a task-indexed access left in the unit head"
  return src

#!/usr/bin/env python3
"""Gemma 4's MTP draft head (the GGUF architecture `gemma4-assistant`) on the NPU, and its packer.

The drafter is a 4-layer, 256-wide Gemma 4 decoder without K / V projections: its layers attend, with their own q, the TARGET's
caches -- the local layers the target's last local cache-owning layer (E2B 13, E4B 22), the global layer its last global one
(14 / 23). A draft step at position P (the position of the token just picked, `tok`; the caches hold positions < P):

    x = pre_projection([embed_target(tok) sqrt(H_target) | h])          h: the target's final-normed hidden of the row that picked
    4 x (attention at P over the target's cache positions < P, window W; post-norm residuals; GeGLU MLP; layer scalar)    tok, or
    hn = rms(x) w_out;  draft = argmax(hn @ token_embd_drafter^T)  (no soft-cap);  h_next = post_projection(hn)   the last h_next
Every draft of a round sits at the same position P (gemma4_ref.Drafter; check/mtp_accept.py measures its acceptance).

On the device a step is ONE graph (one job): the pre-projection, the 4 layers (gemma4_npu's kernels at 1 row: rms_a, the Q8_0 GEMM,
rows32d, gattn_part / gattn_comb against the target's caches, pnss / pnap at c 256, geglu), the final norm, the 262144 x 256 head
(head_topd + head_reduce: the draft's id and its probability) and the post-projection; the host writes the input row and reads the
id, the probability and h_next (1536 floats). The attention sees exactly the reference's keys with the kernels as they are: the
position buffer holds P - 1 (the last visible row), the window is W - 1 (keys P - W + 1 .. P - 1) and the RoPE table is shifted by
one row (q rotated at P).

    python3 gemma4_mtp.py [--mtp /mnt/ssd/models/gemma-4/mtp-gemma-4-E2B-it-Q8_0.gguf] [--out /mnt/ssd/gemma4-e2b-npu/mtp]   (pack)
"""
import argparse, functools, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gemma4_npu as GN                                                   # noqa: E402
from gemma4_npu import dev, poke, NS, KS, OA, dtypes, Kernels, LayerW     # noqa: E402
import gemma4_kernels as GK                                              # noqa: E402
import gemma4_ref as GR                                                  # noqa: E402

def default_mtp(gguf): return os.path.join(os.path.dirname(gguf), "mtp-" + os.path.basename(gguf))

# ---- the packer: <cache>/mtp/ D{l}_{q,o,gu,dn}.bin, D{l}_small.npz, pre.bin, post.bin, head.bin, meta.npz, mtp.json (last)
def save(path, arr): arr.tofile(path + ".tmp"); os.replace(path + ".tmp", path)

def pack(mtp_gguf, out):
  os.makedirs(out, exist_ok=True); D = GR.Model(mtp_gguf, cache_layers=False); c = D.c; t0 = time.perf_counter(); meta = {}
  assert c.arch == "gemma4-assistant", c.arch
  def lin(name, tensors):
    qd = [D.g.q8(t) for t in tensors]; q, d = np.concatenate([a for a, _ in qd]), np.concatenate([b for _, b in qd])
    if name == "head": import gemma4_pack as GP; npad = GP.pack_stream(os.path.join(out, "head.bin"), q, d)
    else: st, npad = GN.q8_stream(q, d); save(os.path.join(out, f"{name}.bin"), st)
    meta[name] = (int(npad), int(q.shape[1]), int(q.shape[0]))
  for l in range(c.NL):
    p = f"blk.{l}."
    lin(f"D{l}_q", [p + "attn_q.weight"]); lin(f"D{l}_o", [p + "attn_output.weight"]); lin(f"D{l}_gu", [p + "ffn_gate.weight", p + "ffn_up.weight"])
    lin(f"D{l}_dn", [p + "ffn_down.weight"])
    w = lambda t: D.t(p + t).astype(np.float32)
    np.savez(os.path.join(out, f"D{l}_small.npz"), w_in=w("attn_norm.weight"), qnw=w("attn_q_norm.weight"), w_ffn=w("ffn_norm.weight"),
             wv_attn=np.append(w("post_attention_norm.weight"), np.float32(1.0)),
             wv_ffw=np.append(w("post_ffw_norm.weight"), np.float32(D.t(p + "layer_output_scale.weight")[0])))
  lin("pre", ["nextn.pre_projection.weight"]); lin("post", ["nextn.post_projection.weight"]); lin("head", ["token_embd.weight"])
  names = sorted(meta); np.savez(os.path.join(out, "meta.npz"), names=np.array(names), meta=np.array([meta[k] for k in names], np.int64),
                                 output_norm=D.t("output_norm.weight").astype(np.float32))
  json.dump(dict(gguf=os.path.abspath(mtp_gguf), NL=c.NL, H=c.H, H_OUT=c.H_OUT, packer="gemma4_mtp.py q8=1 ks 8 ns 3"), open(os.path.join(out, "mtp.json"), "w"), indent=1)
  print(f"   drafter {out}: {sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1e6:.0f} MB in {time.perf_counter() - t0:.0f} s", flush=True)

# ---- the drafter on the device
def draft_layer(K, W, c, x, out, Kc, Vc, cs, posd, ring):
  """Drafter layer W.l on 1 row: x fp32 rows [24, 256] -> out; Kc / Vc the target's cache (ring rows when local); posd int32: P - 1."""
  l, sm = W.l, W.sm; hd, nkv, NH, H = c.HD[l], c.NKV[l], c.NH, c.H
  ct = W.gemm(K, K.rms_a(x, sm["w_in"], H, "a_h"), "q", 1)
  q_rows = K.unpack(ct, NH * hd, f"q_rows{hd}")
  HG, BT, P, MR = GK.gattn_cfgr(NH, nkv, hd, 1); win = c.W - 1 if c.SWA[l] else 0
  part = K.buf(f"ga_part|{hd}", GK.gattn_part_size(NH, nkv, hd, 1), dtypes.float32)
  K.call(f"gattn_part|d|{NH}|{nkv}|{hd}|{win}|{ring}", GK.gattn_part_src(NH, nkv, hd, c.EPS, 1, NH * hd, win, ring), part, q_rows, Kc, Vc, sm["qnw"], cs, posd,
         GN._desc(K, f"gattn_part_desc|d|{nkv}|{hd}", lambda: GK.gattn_part_desc(nkv, hd, MR, HG, BT)))
  o_rows = K.rows_buf(f"o_rows{hd}", NH * hd)
  K.call(f"gattn_comb|d|{NH}|{nkv}|{hd}", GK.gattn_comb_src(NH, nkv, hd, 1), o_rows, part, GN._desc(K, f"gattn_comb_desc|d|{nkv}|{hd}", lambda: GK.gattn_comb_desc(NH, nkv, hd, 1)))
  x1 = GN.pnres(K, H, 1, c.EPS, K.rows_buf("x1", H), x, W.gemm(K, K.rms_a(o_rows, None, NH * hd, f"a_o{hd}", norm=False), "o", 1), sm["wv_attn"])
  ff = c.FF[l]
  a_g = K.swiglu_a(W.gemm(K, K.rms_a(x1, sm["w_ffn"], H, "a_f"), "gu", 1), ff, f"a_sw{ff}", act="gelu")
  return GN.pnres(K, H, 1, c.EPS, out, x1, W.gemm(K, a_g, "dn", 1), sm["wv_ffw"])

class Drafter:
  """The packed drafter (`cache`/mtp) on the device next to a gemma4_generate.Model `T` (its weight loader, its caches)."""
  def __init__(self, T, mtp_gguf, mcache):
    t0 = time.perf_counter(); self.T = T; tc = T.c
    self.D = GR.Model(mtp_gguf, cache_layers=False); c = self.c = self.D.c
    assert c.arch == "gemma4-assistant" and c.H_OUT == tc.H, (c.arch, c.H_OUT, tc.H)
    z = np.load(os.path.join(mcache, "meta.npz")); meta = {str(k): tuple(int(v) for v in m) for k, m in zip(z["names"], z["meta"])}
    views = T.Wt.blob({k: os.path.join(mcache, f"{k}.bin") for k in meta})
    lin = lambda k: (views[k], meta[k][0], meta[k][1])
    self.layers = []
    for l in range(c.NL):
      sm = {k: dev(v) for k, v in np.load(os.path.join(mcache, f"D{l}_small.npz")).items()}
      self.layers.append(LayerW(c, l, {n: lin(f"D{l}_{n}") for n in ("q", "o", "gu", "dn")}, {}, sm))
    self.pre, self.post, self.head = lin("pre"), lin("post"), lin("head"); self.vocab = meta["head"][2]
    self.w_out = dev(z["output_norm"])
    # the target's last local / global cache-owning layers (the `share` rule: target layers NL - 2 / NL - 1 resolve to them)
    self.src = {True: tc.KV_SRC[tc.NL - 2] if tc.SWA[tc.NL - 2] else tc.KV_SRC[tc.NL - 1], False: tc.KV_SRC[tc.NL - 1] if not tc.SWA[tc.NL - 1] else tc.KV_SRC[tc.NL - 2]}
    assert tc.SWA[self.src[True]] and not tc.SWA[self.src[False]], self.src
    self.K = GN.SharedK(2, 24, c.EPS, real=1); K = self.K
    self.din = K.rows_buf("din", 2 * tc.H); self.xb = [K.rows_buf("dx0", c.H), K.rows_buf("dx1", c.H)]
    self.posd = GN.dev(np.zeros(16, np.int32))
    l_loc, l_glo = c.SWA.index(True), c.SWA.index(False)
    self.cs = {s: dev(GN.rope_cs(self.D, l, T.tmax + 1)[1:]) for s, l in ((True, l_loc), (False, l_glo))}   # row P - 1 holds P's angles
    self.jit = None; self.sqh = np.float32(math.sqrt(tc.H))
    print(f"   drafter: {mcache} ({sum(meta[k][0] * meta[k][1] for k in meta) * 34 / 32 / 1e6:.0f} MB of Q8_0), target caches of layers {self.src[True]} / {self.src[False]}, "
          f"set up in {time.perf_counter() - t0:.1f} s", flush=True)

  def f_draft(self, _):
    K, c, T = self.K, self.c, self.T; tc = T.c
    b, npad, kk = self.pre; ng = npad // (16 * NS)
    ct = OA.gemm_gs(K.rms_a(self.din, None, 2 * tc.H, "a_din", norm=False), b, ks=KS, ns=NS, nrb=K.nrb, nslices=kk // (4 * KS), ngroups=ng, b8=True, bscale=True,
                    q8=1, scales="dup", piece=K.nrb, rows=1, out=K.buf("ct_pre", ng * K.nrb * NS * 192, dtypes.float32)).realize()
    x = K.unpack(ct, c.H, "dx0")
    for l in range(c.NL):
      s = self.src[c.SWA[l]]; Kc, Vc = T.cache[s]
      x = draft_layer(K, self.layers[l], c, self.xb[l % 2], self.xb[1 - l % 2], Kc, Vc, self.cs[c.SWA[l]], self.posd, T.ring[s])
    a = K.rms_a(x, self.w_out, c.H, "a_dfin")
    b, npad, kk = self.head; ng = npad // (16 * NS)
    ct = OA.gemm_gs(a, b, ks=KS, ns=NS, nrb=K.nrb, nslices=kk // (4 * KS), ngroups=ng, b8=True, bscale=True, q8=1, scales="dup",
                    piece=K.nrb, rows=1, out=K.buf("ct_head", ng * K.nrb * NS * 192, dtypes.float32)).realize()
    K.head_top(ct, self.vocab, "top_head0"); ids = K.head_reduce((0,))
    b, npad, kk = self.post; ng = npad // (16 * NS)
    ct = OA.gemm_gs(a, b, ks=KS, ns=NS, nrb=K.nrb, nslices=kk // (4 * KS), ngroups=ng, b8=True, bscale=True, q8=1, scales="dup",
                    piece=K.nrb, rows=1, out=K.buf("ct_post", ng * K.nrb * NS * 192, dtypes.float32)).realize()
    K.unpack(ct, tc.H, "h_next")
    return ids

  def step(self, tok, h):
    """One draft at the round's position (set_pos) from (tok, h) -> (id, probability, h_next [H_target])."""
    T = self.T; poke(self.din, np.concatenate([T.M.embed([tok])[0], np.asarray(h, np.float32).reshape(-1)])[None])
    from tinygrad.engine.jit import TinyJit
    if T.jit:
      if self.jit is None: self.jit = TinyJit(self.f_draft)
      if self.jit.cnt < 2:
        from gemma4_generate import one_job
        with one_job(): self.jit(T.jd)
      else: self.jit(T.jd)
    else: self.f_draft(None)
    o = OA.host_invalidate(self.K.bufs["ids_head"]).numpy(); hn = OA.host_invalidate(self.K.bufs["h_next"]).numpy().reshape(-1, T.c.H)[0].copy()
    return int(o[0]), float(o[1:2].view(np.float32)[0]), hn

  def set_pos(self, P): poke(self.posd, np.array([P - 1] + [0] * 15, np.int32))

  def chain(self, tok, h, P, k, tau=0.0):
    """Up to k chained drafts at position P from (tok, h): the chain stops after a draft whose probability is below tau (it is
    kept: the verify pass decides) -> (drafts, probabilities)."""
    self.set_pos(P); d, pr = [], []
    for _ in range(k):
      tok, p, h = self.step(tok, h); d.append(tok); pr.append(p)
      if p < tau: break
    return d, pr

def main():
  ap = argparse.ArgumentParser(); ap.add_argument("--mtp", default=default_mtp(GR.DEFAULT)); ap.add_argument("--out", default="/mnt/ssd/gemma4-e2b-npu/mtp")
  a = ap.parse_args(); pack(a.mtp, a.out)

if __name__ == "__main__": main()

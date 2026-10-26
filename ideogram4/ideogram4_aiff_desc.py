#!/usr/bin/env python3
"""The AIFF descriptor templates of Ideogram 4's W8A8 linears (`ideogram4_q8.Plan`): every chain of one linear, built ONCE per
geometry by the AIFF register generator of the `aipu` tinygrad tree (`aiff_gen` / `aiff_blob`, imported from $AIFF_TG at run
time -- nothing of it is copied here), with the addresses left as offsets and their words listed per buffer, and the per-(pass,
unit) shift fields listed: `ideogram4_q8.Desc.build` relocates and patches them per layer in numpy.

    AIFF_TG=~/tgnpu python3 ideogram4_aiff_desc.py --rows 4704 --out ~/ideogram4/q8desc        (all four linears)

The shift fields: a chain's ctrl descriptors are collapsed by the serialiser when consecutive nodes' register configurations are
equal (a loop count), which a later shift patch could not undo. So the template is built with alternating shifts (1, 2 / 2, 1:
consecutive nodes always differ) and the field is located by a second build with every shift + 64; the self-check rebuilds
with random alternating shifts and requires the patched template to be byte-identical.
"""
import argparse, os, struct, sys, time
import numpy as np

sys.path.insert(0, os.path.expanduser(os.environ.get("AIFF_TG", "~/tgnpu")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinygrad.runtime.support.zhouyi import aiff_gen as G, aiff_blob as B   # noqa: E402  (the external aipu tree)
import ideogram4_q8 as Q                                                    # noqa: E402

LINEARS = {"qkv": (4608, 13824), "o": (4608, 4608), "w13h": (4608, 12288), "w13q": (4608, 6144), "qkvh": (4608, 6912), "w2": (12288, 4608)}


def _chain(plan: Q.Plan, e: int, c: int, s: int, A, W, P, trsh, js=None, ntot=None, coff=0):
    """One chain: engine e's nodes (or `js`) over chunk c, K pass s. trsh(j, u) -> the unit's shift. -> (blob, bases).
    With `js` (GM mode: one node a chain) the weights are a node tile's own buffer W [2 x kc x 128]: its address words are
    the unit offsets, relocated to the GM slot the node is staged into."""
    h0, hc = plan.chunks[c]; j0, j1 = plan.engine_nodes[e]
    k0 = s * plan.kc // Q.SURF
    inp = A[:, k0:k0 + plan.kc // Q.SURF, h0:h0 + hc]
    out = P[:, h0:h0 + hc]                                                   # [1, hc, 8, S*N]
    tile = plan.kc * Q.UNIT
    nodes = []
    for j in (range(j0, j1) if js is None else js):
        units = []
        for u in range(2):
            off = ((s * plan.nodes + j) * 2 + u) * tile if js is None else u * tile
            units.append(G.WeightTile(Q.UNIT, W[off:off + tile], mul=1, trsh=trsh(j, u)))
        nodes.append(G.Node(units, cout_offset=s * (ntot or plan.N) + coff + Q.NODE * j))
    cfg = G.GemmConfig(h=hc, w=Q.W_PLANE, k=plan.kc, cout=plan.S * (ntot or plan.N), requant="scale", mul=1, trsh=0, h_step=plan.hstep,
                       nodes=nodes, out_bits=16, ocvt_prec=1, ocvt_dtype=0, iact_rows_total=plan.H, oact_rows_total=plan.H)
    aiff = G.build(cfg, inp, out)
    ext = {id(A): 0, id(W): 0, id(P): 0}
    blob, bases, out_off = B.pack(aiff.gen_descriptor(), [inp] + [t.weight for n in nodes for t in n.units], out, external=ext)
    assert out_off is None and {oid for _, oid in blob.ext_words} <= set(ext)
    return blob, bases


def build(plan: Q.Plan, check: bool = True, gm: bool = False, ntot: int = None, coff: int = 0) -> dict:
    """`gm`: the GM-staged layout -- one node a chain, chains ordered (engine, round, chunk) with round r = (node j0 + r // S,
    pass r % S) of the engine's nodes; `chain_info` [n, 3] = (engine, round, chunk); the weight words relative to the node's
    GM slot (`rel_w_chain`: the chain of each)."""
    A = np.zeros((1, plan.K // Q.SURF, plan.H, Q.W_PLANE, Q.SURF), np.int8)
    W = np.zeros(2 * plan.kc * Q.UNIT if gm else plan.w_bytes, np.int8)
    P = np.zeros((1, plan.H, Q.W_PLANE, plan.S * (ntot or plan.N)), np.int16)       # ntot: the part writes columns [coff, coff + N) of rows ntot wide
    cat = {id(A): "a", id(W): "w", id(P): "o"}
    base_t = lambda j, u: 1 + (j + u) % 2                                     # 1, 2 / 2, 1 alternating: never collapsed
    words, chains, rel = [], [], {k: [] for k in ("self", "a", "w", "o")}
    t_off, t_shift, t_idx = [], [], []
    off = 0
    order = plan.chains if not gm else [(e, c, r % plan.S, plan.engine_nodes[e][0] + r // plan.S, r) for e in range(len(plan.engine_nodes))
                                        for r in range((plan.engine_nodes[e][1] - plan.engine_nodes[e][0]) * plan.S) for c in range(len(plan.chunks))]
    info, wchain = [], []
    for item in order:
        if gm: e, c, s, jj, r = item; js = [jj]; info.append((e, r, c))
        else: e, c, s = item; js = None
        b0, bases = _chain(plan, e, c, s, A, W, P, base_t, js, ntot, coff)
        b1, _ = _chain(plan, e, c, s, A, W, P, lambda j, u: base_t(j, u) + 64, js, ntot, coff)
        assert len(b0.buf) == len(b1.buf) and b0.relocs == b1.relocs, "the shift changed the blob's shape"
        w0 = np.frombuffer(bytes(b0.buf), np.uint32); w1 = np.frombuffer(bytes(b1.buf), np.uint32)
        diff = np.nonzero(w0 != w1)[0]
        j0, j1 = plan.engine_nodes[e]
        want = [(j, u) for j in (range(j0, j1) if js is None else js) for u in range(2)]
        assert len(diff) == len(want), (len(diff), len(want))
        for wi, (j, u) in zip(diff.tolist(), want):
            x = int(w0[wi] ^ w1[wi]); assert x & (x - 1) == 0, hex(x)        # one bit: the field's bit 6
            sh = x.bit_length() - 1 - 6
            assert (int(w0[wi]) >> sh) & 0x7F == base_t(j, u), "the located field does not hold the shift"
            t_off.append(off + 4 * wi); t_shift.append(sh); t_idx.append(s * (plan.N // Q.UNIT) + (Q.NODE * j + Q.UNIT * u) // Q.UNIT)
        w0 = w0.copy()
        for r in b0.relocs: w0[r // 4] += off                                # in-blob offsets -> offsets in the whole buffer
        rel["self"] += [off + r for r in b0.relocs]
        for bo, oid in b0.ext_words:
            rel[cat[oid]].append(off + bo)
            if cat[oid] == "w": wchain.append(len(chains))
        chains.append((e, off + bases[0], off + bases[1], off + bases[2]))
        words.append(w0); pad = (-len(w0)) % 64; words.append(np.zeros(pad, np.uint32))
        off += 4 * (len(w0) + pad)
    out = dict(words=np.concatenate(words), chains=np.array(chains, np.int64), trsh_off=np.array(t_off, np.int64),
               trsh_shift=np.array(t_shift, np.int64), trsh_idx=np.array(t_idx, np.int64),
               M_pad=plan.M_pad, K=plan.K, N=plan.N, hstep=plan.hstep, **{"rel_" + k: np.array(v, np.int64) for k, v in rel.items()})
    if gm: out.update(chain_info=np.array(info, np.int64), rel_w_chain=np.array(wchain, np.int64), gm=1)
    if check and not gm and ntot is None: _self_check(plan, out, A, W, P)
    return out


def _self_check(plan, tpl, A, W, P):
    """Patched template == a fresh build with other (alternating) shifts and other buffer bases, byte for byte (two chains);
    and the packer's weight layout == aiff_gen.dense_weights."""
    rng = np.random.RandomState(5)
    sh = np.zeros((plan.S, plan.N // Q.UNIT), np.int64)
    for s in range(plan.S):
        for j in range(plan.nodes):
            a, b = rng.randint(0, 60, 2); sh[s, 2 * j], sh[s, 2 * j + 1] = 2 * a + (j % 2), 2 * b + 1 - (j % 2)
    path = "/tmp/_aiffdesc_check.npz"; np.savez(path, **tpl); D = Q.Desc(path)
    da, aa, wa, oa = 0x100000, 0x1000000, 0x4000000, 0x8000000
    got = D.build(da, aa, wa, oa, sh).tobytes()
    for ci in (0, len(plan.chains) - 1):
        e, c, s = plan.chains[ci]
        blob, bases = _chain(plan, e, c, s, A, W, P, lambda j, u, s=s: int(sh[s, 2 * j + u]))
        buf = bytearray(blob.buf)
        for r in blob.relocs: struct.pack_into("<I", buf, r, struct.unpack_from("<I", buf, r)[0] + da + int(tpl["chains"][ci][1]) - bases[0])
        base = {id(A): aa, id(W): wa, id(P): oa}
        for bo, oid in blob.ext_words: struct.pack_into("<I", buf, bo, (struct.unpack_from("<I", buf, bo)[0] + base[oid]) & 0xFFFFFFFF)
        o0 = int(tpl["chains"][ci][1]) - bases[0]
        assert got[o0:o0 + len(buf)] == bytes(buf), "chain %d: the patched template differs from a fresh build" % ci
    k, n = plan.kc, Q.UNIT
    wq = rng.randint(-127, 128, (plan.N, plan.K)).astype(np.int8)
    pk = Q.pack_weights(wq, plan).view(np.int8)
    for (s, j, u) in ((0, 0, 0), (plan.S - 1, plan.nodes - 1, 1)):
        ref = G.dense_weights(k, n, np.ascontiguousarray(wq[Q.NODE * j + Q.UNIT * u:Q.NODE * j + Q.UNIT * (u + 1), s * k:(s + 1) * k].T))
        o = ((s * plan.nodes + j) * 2 + u) * k * n
        assert np.array_equal(pk[o:o + k * n], ref), "the weight layout differs from aiff_gen.dense_weights"


def rows_for(n_text: int, n_img: int = 4096) -> int:
    """`ideogram4_fp_1024.Geo`'s row count (B8 weights)."""
    n = n_img + n_text; base = -(-n // 96) * 96
    piece = lambda nrb: max(p_ for p_ in range(2, 29, 2) if nrb % p_ == 0)
    return min((r for r in range(base, base + 960, 96) if (r // 12) % 8 == 0), key=lambda r: r * (32 + 64 / piece(r // 12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--text", default=None, help="the rows of this prompt's sequence (ideogram4_fp_1024.Geo's rule, 1024 px) instead of --rows")
    ap.add_argument("--pipeline", action="store_true", help="every template the default --q8 build needs (GM h16 qkv / o / w1|w3 parts, w2 DDR h32), existing ones skipped")
    ap.add_argument("--hstep", type=int, default=32); ap.add_argument("--only", default=",".join(LINEARS))
    ap.add_argument("--gm", action="store_true", help="the GM-staged layout (one node a chain; weights staged into GM slots)")
    ap.add_argument("--w13cat", action="store_true", help="w1 and w3 as parts of ONE partials buffer [M_pad][S][24576] (swiglu_i16)")
    a = ap.parse_args()
    os.makedirs(os.path.expanduser(a.out), exist_ok=True)
    if a.text: a.rows = rows_for(int(np.load(os.path.expanduser(a.text))["rows"].shape[0]))
    if a.pipeline:
        t0 = time.time(); made = 0
        for name, gm, hs, ntot, coff in (("qkv", True, 16, None, 0), ("o", True, 16, None, 0), ("w13h", True, 16, 24576, 0), ("w13h", True, 16, 24576, 12288),
                                         ("w13q", True, 16, 12288, 0), ("w13q", True, 16, 12288, 6144),
                                         ("qkvh", True, 16, None, 0),
                                         ("w2", False, 32, None, 0)):
            plan = Q.Plan(a.rows, *LINEARS[name], hstep=hs); p = Q.desc_path(plan, a.out, gm=gm, ntot=ntot, coff=coff)
            if os.path.exists(p): continue
            np.savez(p, **build(plan, gm=gm, ntot=ntot, coff=coff)); made += 1
        print("rows %d: %d template sets made in %.1f s" % (a.rows, made, time.time() - t0), flush=True)
        return
    jobs = [(n, None, 0) for n in a.only.split(",")] if not a.w13cat else [("w13h", 24576, 0), ("w13h", 24576, 12288)]
    for name, ntot, coff in jobs:
        K, N = LINEARS[name]
        plan = Q.Plan(a.rows, K, N, hstep=a.hstep)
        tpl = build(plan, gm=a.gm, ntot=ntot, coff=coff)
        p = Q.desc_path(plan, a.out, gm=a.gm, ntot=ntot, coff=coff); np.savez(p, **tpl)
        print("%-5s %s: %d chains (%d chunks x %d passes x 3 engines, nodes/engine %s), %d B of descriptors, %d shift fields -> %s"
              % (name, plan.key, len(plan.chains), len(plan.chunks), plan.S, [b - a_ for a_, b in plan.engine_nodes], tpl["words"].nbytes,
                 len(tpl["trsh_off"]), p), flush=True)


if __name__ == "__main__":
    main()

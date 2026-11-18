"""cpu_split_asm: writes cpu_split_asm.h, the trunk's int8 kernel blocks (cpu_split.c cols_c2c) as hand-allocated assembly.

gcc's own register allocation of the 16 int32 accumulators spilled and copied vector registers inside the hot loop; moves and spills
issue on the same two pipes as USMMLA, so the blocks are written out here with fixed registers (one asm statement a block: the
accumulators cannot be "+w" operands, 16 of them exceed gcc's 30-operand limit).
The weights are the column-interleaved 2-bit layout (.c2c files, cpu_split_pack.py): per slice and column octet [4 b][4 s][16 B],
field f of byte j = column 2f + j/8 of the octet at k = 32b + 8s + j%8 -- the four fields of a code register are four column pairs at
the same 8 k, so they come out with three ANDs (weights 1, 4, 16) and one shift (weight 1), and the weight 4^w(f) folds into the
slice's power-of-two 1 / s. The accumulators start from memory (hi 0, lo -4^w(f) Q), not movi + sub. Each block prefetches the codes
and scales PF bytes ahead.
  i8c_block_asm:  4 rows, 8 columns (one octet: 4 column pairs x 2 row pairs x 2 limbs). Accumulators v0-v7 (hi limb, slot 2 cp + rp),
                  v8-v15 (lo limb); A v16-v19, codes v20-v23, fields v24-v27, masks v28-v30.
  i8c1_block_asm: <= 2 rows, 16 columns (two octets). Accumulators v0-v7 / v8-v15 by column pair.
Both end with the slice's epilogue: (hi 256 + lo) to fp32, times (scale x 4^-w / s), added into the block's fp32 slots.
  qi: {nq[4 f][2 rp] (int32 x4, -4^w(f) Q), iv[4 f][2 rp] (fp32 x4, 4^-w(f) / s), zero[2]}: 288 B.
Run: python3 cpu_split_asm.py (beside cpu_split.c)."""
import os

FW = (0, 1, 2, 0)   # field f's weight exponent (and 3 << 2f for f < 3, ushr 6 for f = 3)
def fields_c(code, regs):
  return [f"and v{regs[0]}.16b, v{code}.16b, v28.16b", f"and v{regs[1]}.16b, v{code}.16b, v29.16b",
          f"and v{regs[2]}.16b, v{code}.16b, v30.16b", f"ushr v{regs[3]}.16b, v{code}.16b, #6"]
MASKS = ["movi v28.16b, #3", "movi v29.16b, #12", "movi v30.16b, #48"]

PF = int(os.environ.get("CS_PF", "1024"))   # software prefetch distance (bytes of codes ahead; the scales PF / 8 ahead)
def pfl(cols):   # prefetch the codes cols columns' worth ahead (32 B a column) and their scales
  return [f"prfm pldl1keep, [%0, #{PF + 64*i}]" for i in range(cols * 32 // 64)] + [f"prfm pldl1keep, [%3, #{PF // 8}]"]
def block4c():
  L = pfl(8) + [f"ldp q{8+2*c}, q{9+2*c}, [%4, #{32*c}]" for c in range(4)] + [f"ldp q{2*i}, q{2*i+1}, [%4, #256]" for i in range(4)] + MASKS
  for b in range(4):
    L += [f"ldr q{20+s_}, [%0, #{64*b + 16*s_}]" for s_ in range(4)]
    for s_ in range(4):
      o = 64 * b + 16 * s_
      L += [f"ldr q16, [%1, #{o}]", f"ldr q17, [%1, #{256+o}]", f"ldr q18, [%1, #{512+o}]", f"ldr q19, [%1, #{768+o}]"]
      L += fields_c(20 + s_, (24, 25, 26, 27))
      for c in range(4):
        u = 24 + c
        L += [f"usmmla v{2*c}.4s, v{u}.16b, v16.16b", f"usmmla v{8+2*c}.4s, v{u}.16b, v17.16b", f"usmmla v{2*c+1}.4s, v{u}.16b, v18.16b", f"usmmla v{9+2*c}.4s, v{u}.16b, v19.16b"]
  L += ["ldp q20, q21, [%3]", "movi v28.4s, #1, lsl #8",
        "zip1 v22.4s, v20.4s, v20.4s", "zip2 v23.4s, v20.4s, v20.4s", "zip1 v24.4s, v21.4s, v21.4s", "zip2 v25.4s, v21.4s, v21.4s"]
  for c in range(4):
    h0, h1, l0, l1 = 2 * c, 2 * c + 1, 8 + 2 * c, 9 + 2 * c
    L += [f"ldp q16, q17, [%4, #{128 + 32*c}]", f"fmul v18.4s, v{22+c}.4s, v16.4s", f"fmul v19.4s, v{22+c}.4s, v17.4s",
          f"mla v{l0}.4s, v{h0}.4s, v28.s[0]", f"mla v{l1}.4s, v{h1}.4s, v28.s[0]", f"scvtf v{l0}.4s, v{l0}.4s", f"scvtf v{l1}.4s, v{l1}.4s",
          f"ldp q26, q27, [%2, #{32*c}]", f"fmla v26.4s, v{l0}.4s, v18.4s", f"fmla v27.4s, v{l1}.4s, v19.4s", f"stp q26, q27, [%2, #{32*c}]"]
  return L, 31

def block1c():
  L = pfl(16) + [f"ldr q{8+c}, [%4, #{32*(c%4)}]" for c in range(8)] + [f"ldp q{2*i}, q{2*i+1}, [%4, #256]" for i in range(4)] + MASKS
  for b in range(4):
    for s_ in range(4):
      o = 64 * b + 16 * s_; cr = 20 + 2 * (s_ % 2)
      L += [f"ldr q{cr}, [%0, #{o}]", f"ldr q{cr+1}, [%0, #{256+o}]", f"ldr q16, [%1, #{o}]", f"ldr q17, [%1, #{256+o}]"]
      for oc in range(2):
        rr = (24, 25, 26, 27) if oc == 0 else (18, 19, 31, cr + 1)   # (the last field may overwrite its own code: the shift reads it last)
        L += fields_c(cr + oc, rr)
        for f in range(4):
          c = 4 * oc + f
          L += [f"usmmla v{c}.4s, v{rr[f]}.16b, v16.16b", f"usmmla v{8+c}.4s, v{rr[f]}.16b, v17.16b"]
  L += ["movi v18.4s, #1, lsl #8", "ldp q20, q21, [%3]", "ldp q22, q23, [%3, #32]"]
  for c in range(8):
    L += [f"{'zip1' if c % 2 == 0 else 'zip2'} v24.4s, v{20 + c // 2}.4s, v{20 + c // 2}.4s", f"ldr q16, [%4, #{128 + 32*(c%4)}]",
          f"mla v{8+c}.4s, v{c}.4s, v18.s[0]", f"scvtf v{8+c}.4s, v{8+c}.4s", "fmul v25.4s, v24.4s, v16.4s",
          f"ldr q26, [%2, #{32*c}]", f"fmla v26.4s, v{8+c}.4s, v25.4s", f"str q26, [%2, #{32*c}]"]
  return L, 32

def fn(name, L, nclob):
  clob = ", ".join(f'"v{i}"' for i in range(nclob))
  return (f"static inline void {name}(const uint8_t *w, const int8_t *a, float32x4_t *ob, const float *sc, const void *qi) {{\n  __asm__ volatile(\n"
          + "".join(f'    "{x}\\n"\n' for x in L) + f'    : : "r"(w), "r"(a), "r"(ob), "r"(sc), "r"(qi) : {clob}, "memory");\n}}\n')

if __name__ == "__main__":
  out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpu_split_asm.h")
  with open(out, "w") as f:
    f.write("// generated by cpu_split_asm.py -- do not edit. w: the block's codes (.c2c); a: the slice's limbs; ob: its fp32 slots;\n"
            "// sc: its scales; qi: {nq[4][2] (int32 x4), iv[4][2] (fp32 x4), zero[2]}.\n")
    f.write(fn("i8c_block_asm", *block4c())); f.write(fn("i8c1_block_asm", *block1c()))
  print("wrote", out)

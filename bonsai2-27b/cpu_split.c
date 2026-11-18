// cpu_split: the CPU half of the NPU + CPU column split of a ternary GEMM (bonsai2-27b's MLP at <= 4 rows).
//
// The NPU runs gemm_gs(tern) on a linear's first column groups; this library computes the remaining columns [n0, N) on the CPU and
// writes them into the same C tiles. A pass is a schedule of steps (one a GEMM, in the chain's order). For each step the NPU chain
// holds two tiny kernels (qwen38_kernels.cpu_req / cpu_ack): after the GEMM's A is in DDR `cpu_req` sets REQ = 1; before the C
// tiles' consumer `cpu_ack` polls ACK until it reads 1 and clears it. The orchestrator (worker 0) spins on REQ, clears it, the pool copies
// A (a share of its lines each) and computes its columns, and worker 0 sets ACK. No Python in the loop. Both words are read / written around the caches (dc civac / dc cvac): the
// NPU's DMA reads and writes DDR.
//
// The kernel: per 128-slice and row a power-of-two scale puts the A (fp16) in 16-bit integers, split in two int8 limbs; USMMLA multiplies
// the weights' u = w + 1 (2-bit fields of the .c2c files) by each limb into exact int32 2x2 tiles; one epilogue a slice (1.3e-5 relative
// rms against exact sums). The blocks are hand-allocated assembly (cpu_split_asm.h, written by cpu_split_asm.py). The drafter's 4-bit
// linears (fmt 1) run the same limbs through cols_q4. The columns are shared among the pool by each core's top clock (cs_set_weights);
// kern_v bit 5 (cs_set_kernel) K-splits the narrow linears in core pairs (run_slice; another order of sums).
//   codes2 uint8 [K/128][npad/8 column octets][4 b][4 s][16 B]  byte j's field f (bits 2f, 2f + 1) = u of column 8 co + 2 f + j / 8 at
//                k = 32 b + 8 s + j % 8                                                        (cpu_split_pack.py: .c2c)
//   sct    fp32  [K/128][N]      the (slice, column) scale
//   a      fp16  the GEMM's compact A at rt = 1: [K/4 k-quads][4 rows][4 k]                       (rows mode, <= 4 rows)
//   ct     fp32  the C tiles: column n, row r at (n / 48) * gstride + ((n % 48) / 16) * 192 + ((n % 16) / 4) * 16 + 4 r + n % 4
// Build: gcc -O3 -march=armv8.2-a+fp16+i8mm -shared -fPIC -pthread -o libcpu_split.so cpu_split.c   (cpu_split_asm.h beside it)
#define _GNU_SOURCE
#include <arm_neon.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { const uint8_t *codes4; const float *sct; const uint16_t *a; float *ct; int32_t K, N, n0, gstride, kind, pad_; const uint8_t *codes2; int32_t fmt, rows, n1, flags; } step_t;
// kind: the stats' bucket (0..15). fmt 0: codes2 the .c2c weights (codes4 unused). fmt 1: 4-bit weights (the drafter's q4t linears,
// w = u - 8, u in 0..15) in codes4 as [K/128][N/2 column pairs][8][16 B]: byte j of block b holds column 2cp + j / 8 at k = 16 b + j % 8
// (bits 3:0) and k = 16 b + 8 + j % 8 (bits 7:4); sct the (slice, column) scale (cols_q4). rows: the geometry's rows (<= 2: one row
// pair computed). n1: the columns' end (0: N). flags bit 0: the previous step's A (no REQ, no copy, no limbs: the CPU split's z
// columns, computed under the NPU's recurrence and ACKed before the gated norm -- QWEN_CPU_SPLIT_ZDEFER). flags bit 1: the A is
// already in the pool's copy (the glue step before it wrote it): no REQ, no copy. bit 2: no ACK after the step (a glue step follows
// and ACKs). bit 3: a glue step (codes4 = its glue_t): after its REQ (the NPU's part of the linear before it is done) the
// orchestrator runs the glue, then ACKs.
// fmt 2: E4M3 weights (an fp8 drafter's linears, the cache's own codes: no requantisation) in codes4 as [K/128][N][128 B] (column n's
// 128 k of a slice together), sct the (slice, column) scale (cols_f8: fp16 products, 8 a lane, summed in fp32).

static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static inline void line_clean(const volatile void *p) { __asm__ volatile("dc cvac, %0" :: "r"(p) : "memory"); }
static inline void line_inval(const volatile void *p) { __asm__ volatile("dc civac, %0" :: "r"(p) : "memory"); }
static inline void dsb(void) { __asm__ volatile("dsb sy" ::: "memory"); }

#define MAXT 16
static int pool_n = 0, pool_cpu[MAXT];
static pthread_t th[MAXT];
static pthread_mutex_t pm = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t pc = PTHREAD_COND_INITIALIZER;
static atomic_int pass_gen = 0, pass_active = 0, step_gen = 0, step_done = 0, step_copied = 0, step_prepped = 0, status = 0;
static step_t *sched = NULL; static int nsteps = 0; static const step_t *cur = NULL;
static volatile int32_t *req = NULL, *ack = NULL;
static int no_compute = 0, no_compute_head = 0, req_lines = 1;   // req_lines: REQ words, one a 64-byte line (cpu_req_copy: one a task)   // cs_set_mode(1): the handshake only, no copy and no compute (the hand-off kernels' own cost)
static uint16_t *acopy = NULL; static size_t acopy_cap = 0;
static int kern_v = 32;   // cs_set_kernel: bit 5 the K-split (run_slice)
static float32x4_t *o_[MAXT]; static size_t o_cap[MAXT];
// stats (ns, summed over the steps of the last pass): waiting for REQ, the A copy, the compute, the longest step's compute; the step
// count; and the wait and the compute by the step's kind (cpu_split.py's linears)
#define NKIND 16
static double st_wait, st_copy, st_comp, st_max_comp, st_wk[NKIND], st_ck[NKIND]; static int st_steps;

// the columns [n0, N) shared out in 16s, in proportion to each worker's weight (cs_init: the core's top clock; the pool's cores run at
// 2.2-2.5 GHz, and an even share waits on the slowest); cs_set_weights(NULL): even. w0, w1: the share's bounds in the prefix sums.
static double pool_w[MAXT + 1];   // the prefix sums, pool_w[pool_n] = 1
static inline int col_end(const step_t *s) { return s->n1 > 0 ? s->n1 : s->N; }
static void col_range(const step_t *s, double w0, double w1, int *a, int *b) {
  int span = (col_end(s) - s->n0) / 16; if (span < 0) span = 0; *a = s->n0 + 16 * (int)(span * w0 + 0.5); *b = s->n0 + 16 * (int)(span * w1 + 0.5);
}

// The int8 matrix-multiply kernel: per 128-slice and row a power-of-two scale s puts the A
// in 16-bit integers q = rint(a s), |q| <= 32639 (exact for every value within 5 binades of the row's largest), split q = 256 hi + lo
// (both int8); USMMLA multiplies the weights' u = w + 1 (unsigned 2-bit fields) by each limb into exact int32 2x2 tiles; per slice
// (hi - sum hi) 256 + (lo - sum lo) is the exact integer sum (< 2^24: exact in fp32), times 1 / s (a power of two: exact), then
// one fma with the scale. The limbs are shared by the pool (each worker preps every pool_n-th slice).
typedef struct { int8_t v[2][2][16][16]; float invs[4]; int32_t S[2][4]; int32_t pad_[4]; } alimb_t;   // [row pair][limb][32 blk + 8 f][2 rows x 8 k]
static alimb_t *limbs = NULL; static size_t limbs_cap = 0;
static void prep_limbs(int sl) {
  alimb_t *L = limbs + sl; const uint16_t *as = acopy + (size_t)sl * 32 * 16;
  for (int r = 0; r < 4; r++) {
    float a[128], mx = 0;
    for (int k = 0; k < 128; k++) { _Float16 h; memcpy(&h, as + (k / 4) * 16 + r * 4 + k % 4, 2); a[k] = (float)h; mx = fmaxf(mx, fabsf(a[k])); }
    int e = 0; if (mx > 0) frexpf(32639.0f / mx, &e); float sc = mx > 0 ? ldexpf(1.0f, e - 1) : 1.0f;
    L->invs[r] = 1.0f / sc; int32_t S0 = 0, S1 = 0;
    for (int k = 0; k < 128; k++) {
      int q = (int)rintf(a[k] * sc), lo = ((q + 128) & 255) - 128, hi = (q - lo) >> 8;
      int8_t *d0 = L->v[r / 2][0][k / 8], *d1 = L->v[r / 2][1][k / 8]; d0[(r % 2) * 8 + k % 8] = (int8_t)hi; d1[(r % 2) * 8 + k % 8] = (int8_t)lo;
      S0 += hi; S1 += lo;
    }
    L->S[0][r] = S0; L->S[1][r] = S1;
  }
}
static void prep_f8(int sl);   // (fmt 2's A: below, with its kernel)
static void prep_shared(const step_t *s, int id) {   // the step's limbs, worker id's slices; then the barrier
  if (s->fmt == 2) { for (int sl = id; sl < s->K / 128; sl += pool_n) prep_f8(sl); }
  else for (int sl = id; sl < s->K / 128; sl += pool_n) prep_limbs(sl);
  atomic_fetch_add(&step_prepped, 1);
  while (atomic_load(&step_prepped) < pool_n) ;
}
// The trunk's kernel on the .c2c weights: eight columns (one octet) a block at 4 rows, 16 at <= 2 (the A720's two USMMLA pipes want
// >= 12 independent accumulators). The four fields of a code register are four column pairs at the same 8 k, weights 1, 4, 16, 1
// (three ANDs, one shift); the weight 4^w folds into the slice's power-of-two 1 / s, and the accumulators start from -4^w Q (the u = w + 1
// offset, both limbs at once). The fp32 sums stay in the accumulators' lane order ([c0 r, c0 r + 1, c1 r, c1 r + 1]) until the step's end.
#include "cpu_split_asm.h"
__attribute__((noinline)) static void cols_c2c(const step_t *s, int n0, int nn, int sl0, int sl1, float32x4_t *o) {
  const int N = s->N, two_rp = s->rows == 0 || s->rows > 2; const float32x4_t z = vdupq_n_f32(0);
  for (int n = 0; n < nn; n++) o[n] = z;
  static const int fw[4] = {0, 1, 2, 0};
  for (int sl = sl0; sl < sl1; sl++) {
    const alimb_t *L = limbs + sl; const uint8_t *w = s->codes2 + ((size_t)sl * N + n0) * 32; const float *sc = s->sct + (size_t)sl * N + n0;
    const int32_t q0 = 256 * L->S[0][0] + L->S[1][0], q1 = 256 * L->S[0][1] + L->S[1][1], q2 = 256 * L->S[0][2] + L->S[1][2], q3 = 256 * L->S[0][3] + L->S[1][3];
    const int32x4_t Q0 = {q0, q1, q0, q1}, Q1 = {q2, q3, q2, q3};
    const float32x4_t I0 = {L->invs[0], L->invs[1], L->invs[0], L->invs[1]}, I1 = {L->invs[2], L->invs[3], L->invs[2], L->invs[3]};
    struct { int32x4_t nq[4][2]; float32x4_t iv[4][2]; int32x4_t zero[2]; } qc;
    for (int f = 0; f < 4; f++) {
      qc.nq[f][0] = vnegq_s32(vmulq_n_s32(Q0, 1 << (2 * fw[f]))); qc.nq[f][1] = vnegq_s32(vmulq_n_s32(Q1, 1 << (2 * fw[f])));
      qc.iv[f][0] = vmulq_n_f32(I0, 1.0f / (float)(1 << (2 * fw[f]))); qc.iv[f][1] = vmulq_n_f32(I1, 1.0f / (float)(1 << (2 * fw[f])));
    }
    qc.zero[0] = qc.zero[1] = vdupq_n_s32(0);
    const int8_t *av = (const int8_t *)L->v;   // [rp][limb][16 k-blocks]
    if (two_rp) for (int n = 0; n < nn; n += 8) i8c_block_asm(w + 32 * n, av, o + n, sc + n, &qc);
    else for (int n = 0; n < nn; n += 16) i8c1_block_asm(w + 32 * n, av, o + n, sc + n, &qc);
  }
  // the accumulators' lane order -> a column's 4 rows: block slots [cp][rp] = [c0 r(2rp), c0 r(2rp+1), c1 .., c1 ..]
  for (int n = 0; n < nn; n += 8) {
    float32x4_t *ob = o + n, t[8]; for (int k = 0; k < 8; k++) t[k] = ob[k];
    for (int cp = 0; cp < 4; cp++) {
      const float32x4_t r0 = t[2 * cp], r1 = t[2 * cp + 1];
      ob[2 * cp] = vcombine_f32(vget_low_f32(r0), vget_low_f32(r1)); ob[2 * cp + 1] = vcombine_f32(vget_high_f32(r0), vget_high_f32(r1));
    }
  }
}

// fmt 1 (4-bit weights): the int8 kernel's limbs, u in 0..15 unsigned against each limb, the sum of w = u - 8 as (sum u a) - 8 (sum a)
static void cols_q4(const step_t *s, int n0, int nn, int sl0, int sl1, float32x4_t *o) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
  const int N = s->N, rp2 = s->rows > 2; const float32x4_t z = vdupq_n_f32(0); const uint8x16_t m15 = vdupq_n_u8(15);
  for (int n = 0; n < nn; n++) o[n] = z;
  for (int sl = sl0; sl < sl1; sl++) {
    const alimb_t *L = limbs + sl; const uint8_t *w = s->codes4 + ((size_t)sl * N + n0) * 64; const float *sc = s->sct + (size_t)sl * N + n0;
    const int32x4_t sh0 = {L->S[0][0], L->S[0][1], L->S[0][0], L->S[0][1]}, sh1 = {L->S[0][2], L->S[0][3], L->S[0][2], L->S[0][3]};
    const int32x4_t sl0v = {L->S[1][0], L->S[1][1], L->S[1][0], L->S[1][1]}, sl1v = {L->S[1][2], L->S[1][3], L->S[1][2], L->S[1][3]};
    const float32x4_t invs = vld1q_f32(L->invs);
    for (int n = 0; n < nn; n += 4, w += 256) {   // two column pairs (128 B each)
      int32x4_t h00 = vdupq_n_s32(0), h01 = h00, l00 = h00, l01 = h00, h10 = h00, h11 = h00, l10 = h00, l11 = h00;   // [col pair][row pair]
      for (int b = 0; b < 8; b++) {
        const uint8x16_t x0 = vld1q_u8(w + 16 * b), x1 = vld1q_u8(w + 128 + 16 * b);
        const uint8x16_t u0[2] = {vandq_u8(x0, m15), vshrq_n_u8(x0, 4)}, u1[2] = {vandq_u8(x1, m15), vshrq_n_u8(x1, 4)};
        for (int f = 0; f < 2; f++) {
          const int kb = 2 * b + f;   // the 8-wide k block
          const int8x16_t a0 = vld1q_s8(L->v[0][0][kb]), b0 = vld1q_s8(L->v[0][1][kb]);
          h00 = vusmmlaq_s32(h00, u0[f], a0); h10 = vusmmlaq_s32(h10, u1[f], a0); l00 = vusmmlaq_s32(l00, u0[f], b0); l10 = vusmmlaq_s32(l10, u1[f], b0);
          if (rp2) {
            const int8x16_t a1 = vld1q_s8(L->v[1][0][kb]), b1 = vld1q_s8(L->v[1][1][kb]);
            h01 = vusmmlaq_s32(h01, u0[f], a1); h11 = vusmmlaq_s32(h11, u1[f], a1); l01 = vusmmlaq_s32(l01, u0[f], b1); l11 = vusmmlaq_s32(l11, u1[f], b1);
          }
        }
      }
#define EX(H, LO, SH, SL) vcvtq_f32_s32(vaddq_s32(vshlq_n_s32(vsubq_s32(H, vshlq_n_s32(SH, 3)), 8), vsubq_s32(LO, vshlq_n_s32(SL, 3))))
      const float32x4_t f00 = EX(h00, l00, sh0, sl0v), f01 = EX(h01, l01, sh1, sl1v), f10 = EX(h10, l10, sh0, sl0v), f11 = EX(h11, l11, sh1, sl1v);
#undef EX
      o[n] = vfmaq_n_f32(o[n], vmulq_f32(vcombine_f32(vget_low_f32(f00), vget_low_f32(f01)), invs), sc[n]);
      o[n + 1] = vfmaq_n_f32(o[n + 1], vmulq_f32(vcombine_f32(vget_high_f32(f00), vget_high_f32(f01)), invs), sc[n + 1]);
      o[n + 2] = vfmaq_n_f32(o[n + 2], vmulq_f32(vcombine_f32(vget_low_f32(f10), vget_low_f32(f11)), invs), sc[n + 2]);
      o[n + 3] = vfmaq_n_f32(o[n + 3], vmulq_f32(vcombine_f32(vget_high_f32(f10), vget_high_f32(f11)), invs), sc[n + 3]);
    }
  }
#else
  (void)s; (void)n0; (void)nn; (void)sl0; (void)sl1; (void)o;
#endif
}
// fmt 2 (E4M3 weights): per slice and row the A scaled by a power of two into [0.5, 1) as fp16 (prep_f8); a code -> its fp16 bits
// (normal: ((x & 127) << 7) + 0x2000; exponent field 0: m 2^-9 from a table; the sign bit on top), products and sums of 8 a lane
// in fp16 (|w| <= 448, |a| < 1: a lane <= 3584), the 16 lanes' sums in fp32, times 2^e and the (slice, column) scale.
typedef struct { float16_t a[4][128]; float invs[4]; float pad_[12]; } af8_t;
static af8_t *fa8 = NULL; static size_t fa8_cap = 0;
static void prep_f8(int sl) {
  af8_t *L = fa8 + sl; const uint16_t *as = acopy + (size_t)sl * 32 * 16;
  for (int r = 0; r < 4; r++) {
    float a[128], mx = 0;
    for (int k = 0; k < 128; k++) { _Float16 h; memcpy(&h, as + (k / 4) * 16 + r * 4 + k % 4, 2); a[k] = (float)h; mx = fmaxf(mx, fabsf(a[k])); }
    int e = 0; if (mx > 0) frexpf(mx, &e); const float sc = ldexpf(1.0f, -e); L->invs[r] = ldexpf(1.0f, e);
    for (int k = 0; k < 128; k++) L->a[r][k] = (float16_t)(a[k] * sc);
  }
}
static inline void f8_dec(uint8x16_t x, float16x8_t *w0, float16x8_t *w1) {
  static const uint8_t sub_[16] = {0x00, 0x18, 0x1C, 0x1E, 0x20, 0x21, 0x22, 0x23, 0, 0, 0, 0, 0, 0, 0, 0};
  const uint8x16_t mag = vandq_u8(x, vdupq_n_u8(0x7f)), issub = vceqzq_u8(vandq_u8(x, vdupq_n_u8(0x78)));
  uint8x16_t hi = vbslq_u8(issub, vqtbl1q_u8(vld1q_u8(sub_), mag), vaddq_u8(vshrq_n_u8(mag, 1), vdupq_n_u8(0x20)));
  const uint8x16_t lo = vbicq_u8(vshlq_n_u8(x, 7), issub);
  hi = vorrq_u8(hi, vandq_u8(x, vdupq_n_u8(0x80)));
  *w0 = vreinterpretq_f16_u8(vzip1q_u8(lo, hi)); *w1 = vreinterpretq_f16_u8(vzip2q_u8(lo, hi));
}
static inline float32x4_t f8_sum(float16x8_t a, float16x8_t b) {   // the 16 lanes in 4 fp32 lanes
  return vaddq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(a)), vcvt_f32_f16(vget_high_f16(a))), vaddq_f32(vcvt_f32_f16(vget_low_f16(b)), vcvt_f32_f16(vget_high_f16(b))));
}
// One (slice, column): 128 codes against nr rows of the slice's A -> the rows' sums in lanes 0..nr-1 (fp32, before 2^e and the
// scale). The fast decode takes every code as a normal number (7 byte operations a 16); the smallest magnitude seen says whether
// the 128 held a code with exponent field 0 (zero or a subnormal: ~1 column-slice in 60 of a drafter's linears) -- those are
// done again with the exact decode (f8_dec).
static inline __attribute__((always_inline)) float32x4_t f8_col(const uint8_t *w, const af8_t *L, const int nr) {
  const uint8x16_t m7f = vdupq_n_u8(0x7f), m80 = vdupq_n_u8(0x80), c20 = vdupq_n_u8(0x20); const float16x8_t z16 = vdupq_n_f16(0);
  float16x8_t c[4][2]; for (int r = 0; r < 4; r++) c[r][0] = c[r][1] = z16;
  uint8x16_t mn = m7f;
  for (int b = 0; b < 128; b += 16) {
    const uint8x16_t x = vld1q_u8(w + b), mag = vandq_u8(x, m7f); mn = vminq_u8(mn, mag);
    const uint8x16_t hi = vbslq_u8(m80, x, vsraq_n_u8(c20, mag, 1)), lo = vshlq_n_u8(x, 7);
    const float16x8_t w0 = vreinterpretq_f16_u8(vzip1q_u8(lo, hi)), w1 = vreinterpretq_f16_u8(vzip2q_u8(lo, hi));
    for (int r = 0; r < nr; r++) { c[r][0] = vfmaq_f16(c[r][0], w0, vld1q_f16(L->a[r] + b)); c[r][1] = vfmaq_f16(c[r][1], w1, vld1q_f16(L->a[r] + b + 8)); }
  }
  if (__builtin_expect(vminvq_u8(mn) < 8, 0)) {
    for (int r = 0; r < 4; r++) c[r][0] = c[r][1] = z16;
    for (int b = 0; b < 128; b += 16) {
      float16x8_t w0, w1; f8_dec(vld1q_u8(w + b), &w0, &w1);
      for (int r = 0; r < nr; r++) { c[r][0] = vfmaq_f16(c[r][0], w0, vld1q_f16(L->a[r] + b)); c[r][1] = vfmaq_f16(c[r][1], w1, vld1q_f16(L->a[r] + b + 8)); }
    }
  }
  const float32x4_t z = vdupq_n_f32(0);   // rows 0..3's sums in lanes 0..3 (two pairwise adds; the rows not computed are zero)
  const float32x4_t s0 = f8_sum(c[0][0], c[0][1]), s1 = nr > 1 ? f8_sum(c[1][0], c[1][1]) : z, s2 = nr > 2 ? f8_sum(c[2][0], c[2][1]) : z, s3 = nr > 3 ? f8_sum(c[3][0], c[3][1]) : z;
  return vpaddq_f32(vpaddq_f32(s0, s1), nr > 2 ? vpaddq_f32(s2, s3) : z);
}
__attribute__((noinline)) static void cols_f8(const step_t *s, int n0, int nn, int sl0, int sl1, float32x4_t *o) {
  const int N = s->N, nr = s->rows == 0 ? 4 : s->rows > 4 ? 4 : s->rows; const float32x4_t z = vdupq_n_f32(0);
  for (int n = 0; n < nn; n++) o[n] = z;
  for (int sl = sl0; sl < sl1; sl++) {
    const af8_t *L = fa8 + sl; const uint8_t *w = s->codes4 + ((size_t)sl * N + n0) * 128; const float *sc = s->sct + (size_t)sl * N + n0;
    const float32x4_t iv = vld1q_f32(L->invs);
    if (nr == 1) for (int n = 0; n < nn; n++) o[n] = vfmaq_n_f32(o[n], vmulq_f32(f8_col(w + 128 * (size_t)n, L, 1), iv), sc[n]);
    else if (nr == 2) for (int n = 0; n < nn; n++) o[n] = vfmaq_n_f32(o[n], vmulq_f32(f8_col(w + 128 * (size_t)n, L, 2), iv), sc[n]);
    else for (int n = 0; n < nn; n++) o[n] = vfmaq_n_f32(o[n], vmulq_f32(f8_col(w + 128 * (size_t)n, L, 4), iv), sc[n]);
  }
}
static void cols_any(const step_t *s, int n0, int nn, int sl0, int sl1, float32x4_t *o) {
  if (s->fmt == 2) cols_f8(s, n0, nn, sl0, sl1, o); else
  if (s->fmt == 1) cols_q4(s, n0, nn, sl0, sl1, o); else cols_c2c(s, n0, nn, sl0, sl1, o);
}

static void store_cols(const step_t *s, int n0, int nn, const float32x4_t *o, const float32x4_t *o2) {   // into the C tiles (o + o2 if o2)
  for (int n = 0; n < nn; n += 4) {       // 4 columns -> one 64-byte line of the tile: [4 rows][4 cols]
    int col = n0 + n; float *f = s->ct + (size_t)(col / 48) * s->gstride + ((col % 48) / 16) * 192 + ((col % 16) / 4) * 16;
    float32x4_t p0 = o[n], p1 = o[n + 1], p2 = o[n + 2], p3 = o[n + 3];
    if (o2) { p0 = vaddq_f32(p0, o2[n]); p1 = vaddq_f32(p1, o2[n + 1]); p2 = vaddq_f32(p2, o2[n + 2]); p3 = vaddq_f32(p3, o2[n + 3]); }
    float32x4_t t0 = vtrn1q_f32(p0, p1), t1 = vtrn2q_f32(p0, p1), t2 = vtrn1q_f32(p2, p3), t3 = vtrn2q_f32(p2, p3);
    vst1q_f32(f, vcombine_f32(vget_low_f32(t0), vget_low_f32(t2))); vst1q_f32(f + 4, vcombine_f32(vget_low_f32(t1), vget_low_f32(t3)));
    vst1q_f32(f + 8, vcombine_f32(vget_high_f32(t0), vget_high_f32(t2))); vst1q_f32(f + 12, vcombine_f32(vget_high_f32(t1), vget_high_f32(t3)));
    line_clean(f);
  }
  dsb();
}

static float32x4_t *obuf(int id, int nn) {
  if (o_cap[id] < (size_t)nn) { free(o_[id]); o_[id] = aligned_alloc(64, (size_t)nn * sizeof(float32x4_t)); o_cap[id] = nn; }
  return o_[id];
}

// kern_v bit 5, a step with fewer than ksplit_cols columns a worker (the narrow linears: o, down): the pool in pairs (workers 2p, 2p + 1:
// CPUs 6/7, 8/9, 10/11 share a clock; an odd pool: worker 0 alone, then the pairs 1/2, 3/4, ...), each pair a column range by its weight, each member half the slices; the even member adds its partner's partial sums (another order of sums than one chain: not bit-identical to the others).
static int ksplit_cols = 512; static atomic_int pair_gen[MAXT];
static void run_slice(const step_t *s, int id) {
  const int nsl = s->K / 128;
  const int off = pool_n % 2;   // an odd pool: worker 0 (the list's first CPU) alone, the others in pairs
  if ((kern_v & 32) && id >= off && (col_end(s) - s->n0) / pool_n < ksplit_cols) {
    const int j = id - off, p = j / 2, first = off + 2 * p; int n0, n1; col_range(s, pool_w[first], pool_w[first + 2], &n0, &n1);
    const int nn = n1 - n0, g = atomic_load(&step_gen);
    float32x4_t *o = obuf(id, nn > 0 ? nn : 1);
    if (nn > 0) { int a = j & 1 ? nsl / 2 : 0, b = j & 1 ? nsl : nsl / 2; cols_any(s, n0, nn, a, b, o); }
    if (j & 1) { atomic_store(&pair_gen[p], g); return; }
    while (atomic_load(&pair_gen[p]) != g) ;
    if (nn > 0) store_cols(s, n0, nn, o, o_[id + 1]);
    return;
  }
  int n0, n1; col_range(s, pool_w[id], pool_w[id + 1], &n0, &n1); const int nn = n1 - n0;
  if (nn <= 0) return;
  float32x4_t *o = obuf(id, nn); cols_any(s, n0, nn, 0, nsl, o); store_cols(s, n0, nn, o, NULL);
}

static void copy_part(const step_t *s, int id) {   // the A out of the device buffer, worker id's 1/pool_n of its lines; then the barrier
  size_t ab = (size_t)s->K * 8, lines = ab / 64, per = (lines + pool_n - 1) / pool_n, l0 = id * per, l1 = l0 + per < lines ? l0 + per : lines;
  if (l0 < l1) {
    for (size_t i = l0; i < l1; i++) line_inval((const char *)s->a + i * 64);
    dsb(); memcpy((char *)acopy + l0 * 64, (const char *)s->a + l0 * 64, (l1 - l0) * 64);
  }
  atomic_fetch_add(&step_copied, 1);
  while (atomic_load(&step_copied) < pool_n) ;
}

// ---- the glue (QWEN_CPU_GLUE): a layer's small kernels between two split linears, on the CPU instead of two NPU launches.
// GLUE_RESID_NORM: after the down projection -- x2 = x1 + C (the residual: resid32q's add), then the next layer's input norm and
// rotation into that layer's first linear's A (had_fast's "rms" producer): the row's sum of squares in 8 lanes (fused, K
// ascending), the lanes summed in order, inv = sqrt(1 / (sum / c + eps)), v = (x inv) w, the unnormalised Walsh-Hadamard transform
// of each hb-block (stages 1, 2, 4, ...: a + b, a - b), times post, to fp16 -- the NPU kernels' operations in their order, so
// the same bits (checked against the device's own buffers over 24 passes: cs_test_glue). Built with -ffp-contract=off:
// only the fma written here is fused.
typedef struct {
  int32_t kind, rows, c, hb, n0, gstride; float post, eps;
  const float *ct;        // the added linear's C tiles (columns < n0 written by the NPU: their lines are dropped first)
  const float *xin;       // the residual rows read (device buffer, NPU-written), pitch c
  float *xout;            // the residual rows written (device buffer), pitch c
  const float *w;         // the next norm's (1 + weight) x signs [c]
  uint16_t *mirror;       // the next linear's A (DDR mirror: the NPU copies it to its A buffer), the compact layout [c/4][4 rows][4 k]
} glue_t;
static float *gl_buf = NULL; static size_t gl_cap = 0;
static double gl_t[6]; static long gl_n;   // the glue's phases (ns, summed): invalidate, the copies in, the compute, the rows out, the A out
void cs_glue_stats(double *out) { for (int i = 0; i < 6; i++) out[i] = gl_n ? gl_t[i] / gl_n : 0; out[5] = (double)gl_n; }
static inline void inval_range(const void *p, size_t n) { const char *a = (const char *)((uintptr_t)p & ~(uintptr_t)63), *e = (const char *)p + n; for (; a < e; a += 64) line_inval(a); }
static inline void clean_range(const void *p, size_t n) { const char *a = (const char *)((uintptr_t)p & ~(uintptr_t)63), *e = (const char *)p + n; for (; a < e; a += 64) line_clean(a); }
static void fwht_block(float *v, int hb) {   // in place, stages 1, 2, 4, .., hb / 2: (a, b) -> (a + b, a - b)
  for (int i = 0; i < hb; i += 8) {           // stages 1, 2, 4 inside 8 values
    float32x4_t a = vld1q_f32(v + i), b = vld1q_f32(v + i + 4);
    float32x4_t e = vuzp1q_f32(a, b), o = vuzp2q_f32(a, b);                 // e = x0 x2 x4 x6, o = x1 x3 x5 x7
    float32x4_t s1 = vaddq_f32(e, o), d1 = vsubq_f32(e, o);                 // stage 1: y(2j) = s1[j], y(2j + 1) = d1[j]
    float32x4_t e2 = vuzp1q_f32(s1, d1), o2 = vuzp2q_f32(s1, d1);           // e2 = y0 y4 y1 y5, o2 = y2 y6 y3 y7
    float32x4_t s2 = vaddq_f32(e2, o2), d2 = vsubq_f32(e2, o2);             // stage 2: z0 z4 z1 z5 | z2 z6 z3 z7
    float32x4_t e4 = vuzp1q_f32(s2, d2), o4 = vuzp2q_f32(s2, d2);           // e4 = z0 z1 z2 z3, o4 = z4 z5 z6 z7
    vst1q_f32(v + i, vaddq_f32(e4, o4)); vst1q_f32(v + i + 4, vsubq_f32(e4, o4));   // stage 4
  }
  for (int h = 8; h < hb; h *= 2)
    for (int i = 0; i < hb; i += 2 * h)
      for (int j = 0; j < h; j += 4) {
        const float32x4_t a = vld1q_f32(v + i + j), b = vld1q_f32(v + i + j + h);
        vst1q_f32(v + i + j, vaddq_f32(a, b)); vst1q_f32(v + i + j + h, vsubq_f32(a, b));
      }
}
// The device buffers are mapped uncached for the host: a load from one is a DRAM access (and a load after a store to one waits
// for it). So the glue copies what it reads into host memory in whole lines first (the residual rows; each 16-column strip's
// first 4 lines: rows 0-3), computes there, and writes its two results (the rows, the A) with one copy each.
static float *gl_x = NULL, *gl_t4 = NULL; static size_t gl_xcap = 0; static const glue_t * volatile gl_cur = NULL;
static void glue_copy(const glue_t *g, int id) {   // worker id's share of the residual rows and of the strips' first 4 lines
  const int c = g->c, nr = g->rows > 4 ? 4 : g->rows, ns = c / 16; const size_t xl = (size_t)nr * c / 16;   // the rows' lines
  const size_t l0 = xl * id / pool_n, l1 = xl * (id + 1) / pool_n;
  memcpy((char *)gl_x + l0 * 64, (const char *)g->xin + l0 * 64, (l1 - l0) * 64);
  for (int st = ns * id / pool_n; st < ns * (id + 1) / pool_n; st++)    // strip st: [4 quads][4 rows][4 cols]
    memcpy(gl_t4 + (size_t)st * 64, g->ct + (size_t)(st / 3) * g->gstride + (st % 3) * 192, 64 * sizeof(float));
}
static void glue_resid_norm(const glue_t *g) {
  const int c = g->c, nr = g->rows > 4 ? 4 : g->rows, ns = c / 16; const double tg0 = now_ns();
  if (gl_cap < (size_t)c) { free(gl_buf); gl_buf = aligned_alloc(64, (size_t)c * sizeof(float)); gl_cap = c; }
  if (gl_xcap < (size_t)c) { free(gl_x); free(gl_t4); gl_x = aligned_alloc(64, (size_t)c * 4 * sizeof(float)); gl_t4 = aligned_alloc(64, (size_t)ns * 64 * sizeof(float)); gl_xcap = c; }
  // the NPU's bytes out of this core's caches: its columns' tiles (row block 0: 576 floats a group) and the residual rows
  for (int gi = 0; gi < g->n0 / 48; gi++) inval_range(g->ct + (size_t)gi * g->gstride, 576 * sizeof(float));
  for (int r = 0; r < nr; r++) inval_range(g->xin + (size_t)r * c, (size_t)c * sizeof(float));
  dsb();
  double tp0 = now_ns(), tp1, tp2, tp3; gl_t[0] += tp0 - tg0;
  // the copies in, a share a worker (the pool is idle under the glue): glue_copy
  gl_cur = g; atomic_store(&step_done, 0); atomic_fetch_add(&step_gen, 1);
  glue_copy(g, 0);
  while (atomic_load(&step_done) < pool_n - 1) ;
  tp1 = now_ns(); gl_t[1] += tp1 - tp0;
  uint16_t *A = acopy;                         // the next step's A (flags bit 1: no REQ, no copy)
  memset(A, 0, (size_t)c * 8);
  for (int r = 0; r < nr; r++) {
    float *x = gl_x + (size_t)r * c, *v = gl_buf;
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
    for (int n = 0; n < c; n += 8) {            // column n, row r: strip n / 16, quad (n % 16) / 4, at 16 quad + 4 r + n % 4
      const float *t0 = gl_t4 + (size_t)(n / 16) * 64 + ((n % 16) / 4) * 16 + 4 * r;
      const float32x4_t x0 = vaddq_f32(vld1q_f32(x + n), vld1q_f32(t0)), x1 = vaddq_f32(vld1q_f32(x + n + 4), vld1q_f32(t0 + 16));
      vst1q_f32(x + n, x0); vst1q_f32(x + n + 4, x1);
      a0 = vfmaq_f32(a0, x0, x0); a1 = vfmaq_f32(a1, x1, x1);
    }
    float l[8]; vst1q_f32(l, a0); vst1q_f32(l + 4, a1);
    float sum = l[0]; for (int i = 1; i < 8; i++) sum = sum + l[i];
    const float m = sum * (1.0f / (float)c) + g->eps, inv = sqrtf(1.0f / m);
    const float32x4_t iv = vdupq_n_f32(inv);
    for (int n = 0; n < c; n += 4) vst1q_f32(v + n, vmulq_f32(vmulq_f32(vld1q_f32(x + n), iv), vld1q_f32(g->w + n)));
    for (int b = 0; b < c; b += g->hb) fwht_block(v + b, g->hb);
    const float32x4_t po = vdupq_n_f32(g->post);
    for (int n = 0; n < c; n += 4) vst1_u16(A + (n / 4) * 16 + r * 4, vreinterpret_u16_f16(vcvt_f16_f32(vmulq_f32(vld1q_f32(v + n), po))));
  }
  gl_cur = NULL; tp2 = now_ns(); gl_t[2] += tp2 - tp1;
  memcpy(g->xout, gl_x, (size_t)nr * c * sizeof(float)); clean_range(g->xout, (size_t)nr * c * sizeof(float)); tp0 = now_ns(); gl_t[3] += tp0 - tp2;
  memcpy(g->mirror, A, (size_t)c * 8); clean_range(g->mirror, (size_t)c * 8); dsb(); tp3 = now_ns(); gl_t[4] += tp3 - tp0; gl_n++;
}
static void do_glue(const step_t *s) {
  size_t ab = (size_t)s->K * 8;
  if (acopy_cap < ab) { free(acopy); acopy = aligned_alloc(64, ab); acopy_cap = ab; }
  double t0 = now_ns(); const glue_t *g = (const glue_t *)s->codes4;
  if (g->kind == 0) glue_resid_norm(g);
  double t1 = now_ns(); st_comp += t1 - t0; st_ck[s->kind & (NKIND - 1)] += t1 - t0;
}
// a test entry: the glue alone (no pool), its A left in `a_out` ([c/4][4 rows][4 k] fp16)
void cs_test_glue(const glue_t *g, uint16_t *a_out) {
  size_t ab = (size_t)g->c * 8; uint16_t *sv = acopy; size_t svc = acopy_cap; acopy = aligned_alloc(64, ab); acopy_cap = ab;
  glue_resid_norm(g); memcpy(a_out, acopy, ab); free(acopy); acopy = sv; acopy_cap = svc;
}

static void do_step(const step_t *s) {      // the orchestrator: A in (all workers, a share each), the pool, C out
  size_t ab = (size_t)s->K * 8;             // K/4 k-quads x 16 halves x 2 B
  if (acopy_cap < ab) { free(acopy); acopy = aligned_alloc(64, ab); acopy_cap = ab; }   // (between steps: no worker reads it now)
  if (limbs_cap < (size_t)(s->K / 128)) { free(limbs); limbs = aligned_alloc(64, (size_t)(s->K / 128) * sizeof(alimb_t)); limbs_cap = s->K / 128; }
  if (s->fmt == 2 && fa8_cap < (size_t)(s->K / 128)) { free(fa8); fa8 = aligned_alloc(64, (size_t)(s->K / 128) * sizeof(af8_t)); fa8_cap = s->K / 128; }
  double t0 = now_ns();
  const int same_a = s->flags & 1, have_a = s->flags & 2;   // have_a: the glue left the A in acopy (no copy; the limbs still to prepare)
  cur = s; atomic_store(&step_done, 0); atomic_store(&step_copied, 0); atomic_store(&step_prepped, 0); atomic_fetch_add(&step_gen, 1);
  if (!same_a && !have_a) copy_part(s, 0);
  double t1 = now_ns();
  if (!same_a) prep_shared(s, 0);
  run_slice(s, 0);
  while (atomic_load(&step_done) < pool_n - 1) ;
  double t2 = now_ns(); st_copy += t1 - t0; st_comp += t2 - t1; if (t2 - t1 > st_max_comp) st_max_comp = t2 - t1;
  st_ck[s->kind & (NKIND - 1)] += t2 - t1;
}

static void *worker(void *p) {
  int id = (int)(intptr_t)p; cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(pool_cpu[id], &cs); pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
  int seen_pass = 0, seen_step = atomic_load(&step_gen);
  for (;;) {
    pthread_mutex_lock(&pm); while (atomic_load(&pass_gen) == seen_pass) pthread_cond_wait(&pc, &pm); seen_pass = atomic_load(&pass_gen); pthread_mutex_unlock(&pm);
    if (id == 0) {
      st_wait = st_copy = st_comp = st_max_comp = 0; st_steps = 0; int bad = 0; memset(st_wk, 0, sizeof st_wk); memset(st_ck, 0, sizeof st_ck);
      for (int i = 0; i < nsteps && !bad; i++) {
        double t0 = now_ns();
        if (req && !(sched[i].flags & 3)) {
          long spins = 0;
          for (;;) {
            for (int j = 0; j < req_lines; j++) line_inval(req + 16 * j);
            dsb(); int all = 1; for (int j = 0; j < req_lines; j++) all &= req[16 * j] == 1;
            if (all) break;
            if ((++spins & 4095) == 0 && now_ns() - t0 > 20e9) { bad = i + 1; break; }
          }
          if (bad) break;
          for (int j = 0; j < req_lines; j++) { req[16 * j] = 0; line_clean(req + 16 * j); }
          dsb();
        }
        double tw = now_ns() - t0; st_wait += tw; st_wk[sched[i].kind & (NKIND - 1)] += tw;
        if (!no_compute && !(no_compute_head && sched[i].kind == 5)) { if (sched[i].flags & 8) do_glue(&sched[i]); else do_step(&sched[i]); }
        st_steps++;
        if (ack && !(sched[i].flags & 4)) {   // the NPU has taken the previous ACK (cpu_ack clears it): a step without a REQ (flags bit 0) can finish before that
          double ta = now_ns();
          for (;;) { line_inval(ack); dsb(); if (*ack == 0) break; if (now_ns() - ta > 5e9) { bad = i + 1; break; } }
          st_wait += now_ns() - ta;
          if (bad) break;
          *ack = 1; line_clean(ack); dsb();
        }
      }
      // the pass is served once the NPU has taken the last ACK (its cpu_ack clears the word): else, with the CPU the short side at
      // the last step, the host may arm the next job (cs_arm clears ACK) before the NPU's last poll sees it -- that poll then runs to
      // its bound (~1.3 s) and the job after waits it out
      if (ack && !bad && nsteps) { double t0 = now_ns(); for (;;) { line_inval(ack); dsb(); if (*ack == 0) break; if (now_ns() - t0 > 5e9) break; } }
      atomic_store(&status, bad); atomic_store(&pass_active, 0);
    } else {
      while (atomic_load(&pass_active)) {
        int g = atomic_load(&step_gen);
        if (g != seen_step) {
          seen_step = g;
          if (gl_cur) glue_copy(gl_cur, id);       // (a glue step: the copies in)
          else { if (!(cur->flags & 1)) { if (!(cur->flags & 2)) copy_part(cur, id); prep_shared(cur, id); } run_slice(cur, id); }
          atomic_fetch_add(&step_done, 1);
        }
      }
      seen_step = atomic_load(&step_gen);
    }
  }
  return NULL;
}

void cs_set_weights(const double *w) {   // NULL: even
  double tot = 0; for (int t = 0; t < pool_n; t++) tot += w ? w[t] : 1;
  pool_w[0] = 0; for (int t = 0; t < pool_n; t++) pool_w[t + 1] = pool_w[t] + (w ? w[t] : 1) / tot;
  pool_w[pool_n] = 1;
}
void cs_init(int nthreads, const int *cpus) {
  if (pool_n) return; pool_n = nthreads < MAXT ? nthreads : MAXT;
  double w[MAXT]; int ok = 1;
  for (int t = 0; t < pool_n; t++) {   // each core's top clock (cpufreq)
    char path[96]; snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpus[t]);
    FILE *f = fopen(path, "r"); long khz = 0; if (!f || fscanf(f, "%ld", &khz) != 1 || khz <= 0) ok = 0; else w[t] = khz; if (f) fclose(f);
  }
  cs_set_weights(ok ? w : NULL);
  for (int t = 0; t < pool_n; t++) { pool_cpu[t] = cpus[t]; pthread_create(&th[t], NULL, worker, (void *)(intptr_t)t); }
}
// Arm a pass: the orchestrator serves `n` steps in order. req / ack: the two flag words (NULL: no handshake -- the steps run at once).
void cs_arm(step_t *steps, int n, int32_t *req_, int32_t *ack_) {
  sched = steps; nsteps = n; req = req_; ack = ack_;
  if (req) for (int j = 0; j < req_lines; j++) { req[16 * j] = 0; line_clean(req + 16 * j); } if (ack) { *ack = 0; line_clean(ack); } dsb();
  atomic_store(&status, 0); atomic_store(&pass_active, 1);
  pthread_mutex_lock(&pm); atomic_fetch_add(&pass_gen, 1); pthread_cond_broadcast(&pc); pthread_mutex_unlock(&pm);
}
// The lines of [p, p + n) out of the CPU's caches (dc civac): before the host reads bytes the NPU wrote (cpu_split.peek)
void cs_inval(const void *p, size_t n) {
  const char *a = (const char *)((uintptr_t)p & ~(uintptr_t)63), *e = (const char *)p + n;
  for (; a < e; a += 64) line_inval(a);
  dsb();
}
// A test entry (no pool): the columns [n0, n0 + nn) of one step from the A at `a` (the compact layout), single-threaded -> out[nn][4 rows]
void cs_test_cols(const step_t *s, const uint16_t *a, int n0, int nn, float *out) {
  const int nsl = s->K / 128; size_t ab = (size_t)s->K * 8; uint16_t *sv = acopy; acopy = aligned_alloc(64, ab); memcpy(acopy, a, ab);
  if (limbs_cap < (size_t)nsl) { free(limbs); limbs = aligned_alloc(64, (size_t)nsl * sizeof(alimb_t)); limbs_cap = nsl; }
  if (fa8_cap < (size_t)nsl) { free(fa8); fa8 = aligned_alloc(64, (size_t)nsl * sizeof(af8_t)); fa8_cap = nsl; }
  for (int sl = 0; sl < nsl; sl++) { if (s->fmt == 2) prep_f8(sl); else prep_limbs(sl); }
  float32x4_t *o = aligned_alloc(64, (size_t)nn * sizeof(float32x4_t)); cols_any(s, n0, nn, 0, nsl, o);
  memcpy(out, o, (size_t)nn * 16); free(o); free(acopy); acopy = sv;
}
void cs_set_mode(int m) { no_compute = m == 1; no_compute_head = m == 2; }   // 2: the head steps acknowledged at once (a timing control)
void cs_set_kernel(int v) { kern_v = v; }   // bit 5: the K-split; no other bit is read
void cs_set_ksplit(int cols_per_worker) { ksplit_cols = cols_per_worker; }   // kern_v bit 5: the steps it applies to
void cs_set_req_lines(int n) { req_lines = n < 1 ? 1 : n > 16 ? 16 : n; }
// Wait for the armed pass's last step. 0: served; i + 1: step i's REQ never came (20 s).
int cs_wait(void) { while (atomic_load(&pass_active)) ; return atomic_load(&status); }
int cs_active(void) { return atomic_load(&pass_active); }
// out[5 + 2 * NKIND]: wait, copy, compute, longest compute, steps, then the wait by kind, then the compute by kind
void cs_stats(double *out) {
  out[0] = st_wait; out[1] = st_copy; out[2] = st_comp; out[3] = st_max_comp; out[4] = st_steps;
  for (int k = 0; k < NKIND; k++) { out[5 + k] = st_wk[k]; out[5 + NKIND + k] = st_ck[k]; }
}

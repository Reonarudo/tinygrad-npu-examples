/* The CPU's share of the output head: logits of Q6_K rows (GGUF super-blocks, 210 B a 256) against an int8 activation (one scale a
   256-block, llama.cpp's Q8_K precision), NEON sdot, T threads pinned to the given cores. A streaming GEMV at the CPU's DDR rate,
   to run beside the NPU's share. gcc -O3 -march=armv8.2-a+dotprod -o libhosthead.so -shared -fPIC host_head.c -lpthread
     int head_q6k(const uint8_t* rows, int n_rows, const int8_t* xq, const float* xs, int K, float* logits, int T, const int* cpus) */
#define _GNU_SOURCE
#include <arm_neon.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <string.h>
typedef struct { const uint8_t* rows; int r0, r1; const int8_t* xq; const float* xs; int K; float* out; int cpu; } job_t;
static inline float fp16f(uint16_t h) { __fp16 v; memcpy(&v, &h, 2); return (float)v; }
static void* run(void* a) {
  job_t* j = a; if (j->cpu >= 0) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(j->cpu, &s); pthread_setaffinity_np(pthread_self(), sizeof s, &s); }
  int nb = j->K / 256; const uint8x16_t m4 = vdupq_n_u8(0xF), m3 = vdupq_n_u8(0x30); const int8x16_t off = vdupq_n_s8(32);
  for (int r = j->r0; r < j->r1; r++) {
    const uint8_t* b = j->rows + (size_t)r * nb * 210; float acc = 0.0f; __builtin_prefetch(b + 2 * nb * 210);
    for (int ib = 0; ib < nb; ib++, b += 210) {
      const uint8_t* ql = b; const uint8_t* qh = b + 128; const int8_t* sc = (const int8_t*)(b + 192); float d = fp16f(*(const uint16_t*)(b + 208));
      const int8_t* q8 = j->xq + ib * 256; int32_t isum = 0;
      for (int h = 0; h < 2; h++, ql += 64, qh += 32, q8 += 128, sc += 8) {   /* 128 weights: 4 groups of 32, 8 blocks of 16 */
        uint8x16_t qh0 = vld1q_u8(qh), qh1 = vld1q_u8(qh + 16);
        uint8x16_t l0 = vld1q_u8(ql), l1 = vld1q_u8(ql + 16), l2 = vld1q_u8(ql + 32), l3 = vld1q_u8(ql + 48);
        int8x16_t q[8];
        q[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l0, m4), vandq_u8(vshlq_n_u8(qh0, 4), m3))), off);   /* k 0..15   */
        q[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l1, m4), vandq_u8(vshlq_n_u8(qh1, 4), m3))), off);   /* k 16..31  */
        q[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l2, m4), vandq_u8(vshlq_n_u8(qh0, 2), m3))), off);   /* k 32..47  */
        q[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(l3, m4), vandq_u8(vshlq_n_u8(qh1, 2), m3))), off);   /* k 48..63  */
        q[4] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l0, 4), vandq_u8(qh0, m3))), off);                   /* k 64..79  */
        q[5] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l1, 4), vandq_u8(qh1, m3))), off);                   /* k 80..95  */
        q[6] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l2, 4), vandq_u8(vshrq_n_u8(qh0, 2), m3))), off);   /* k 96..111 */
        q[7] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(l3, 4), vandq_u8(vshrq_n_u8(qh1, 2), m3))), off);   /* k 112..127 */
        int32x4_t v = vdupq_n_s32(0);
        for (int i = 0; i < 8; i++) v = vmlaq_n_s32(v, vdotq_s32(vdupq_n_s32(0), q[i], vld1q_s8(q8 + 16 * i)), sc[i]);
        isum += vaddvq_s32(v);
      }
      acc += d * j->xs[ib] * (float)isum;
    }
    j->out[r] = acc;
  }
  return 0;
}
int head_q6k(const uint8_t* rows, int n_rows, const int8_t* xq, const float* xs, int K, float* logits, int T, const int* cpus) {
  pthread_t th[32]; job_t jb[32];
  for (int t = 0; t < T; t++) {
    jb[t] = (job_t){rows, n_rows * t / T, n_rows * (t + 1) / T, xq, xs, K, logits, cpus ? cpus[t] : -1};
    pthread_create(&th[t], 0, run, &jb[t]);
  }
  for (int t = 0; t < T; t++) pthread_join(th[t], 0);
  return 0;
}

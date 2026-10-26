/* The CPU's GEMV over GGUF K-quant rows with llama.cpp's own kernels (libggml-cpu: ggml_vec_dot_q4_K_q8_K / q6_K, quantize_row_q8_K),
   T threads pinned to the given cores, each a contiguous range of rows -- a streaming read at the CPU's DDR rate, to run beside the
   NPU. gcc -O2 -shared -fPIC -o libhostgemv.so host_gemv.c -L<llama.cpp>/build/bin -lggml-cpu -lggml-base -lpthread
     void q8k_quantize(const float* x, void* y, int K)                                   -- K / 256 Q8_K blocks (292 B each)
     void gemv_kq(int kind, const uint8_t* rows, int n_rows, const void* yq, int K, float* out, int T, const int* cpus)  -- kind 4 / 6 */
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stddef.h>
void ggml_vec_dot_q4_K_q8_K(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);
void ggml_vec_dot_q6_K_q8_K(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);
void quantize_row_q8_K(const float* x, void* y, int64_t k);
void ggml_cpu_init(void);
typedef struct { int kind; const uint8_t* rows; int r0, r1; const void* yq; int K; float* out; int cpu; } job_t;
static int inited = 0;
static void* run(void* a) {
  job_t* j = a; if (j->cpu >= 0) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(j->cpu, &s); pthread_setaffinity_np(pthread_self(), sizeof s, &s); }
  size_t rb = (size_t)(j->K / 256) * (j->kind == 4 ? 144 : 210);
  for (int r = j->r0; r < j->r1; r++) {
    const uint8_t* row = j->rows + (size_t)r * rb;
    if (j->kind == 4) ggml_vec_dot_q4_K_q8_K(j->K, j->out + r, 0, row, 0, j->yq, 0, 1);
    else ggml_vec_dot_q6_K_q8_K(j->K, j->out + r, 0, row, 0, j->yq, 0, 1);
  }
  return 0;
}
void q8k_quantize(const float* x, void* y, int K) { if (!inited) { ggml_cpu_init(); inited = 1; } quantize_row_q8_K(x, y, K); }
void gemv_kq(int kind, const uint8_t* rows, int n_rows, const void* yq, int K, float* out, int T, const int* cpus) {
  if (!inited) { ggml_cpu_init(); inited = 1; }
  pthread_t th[32]; job_t jb[32];
  for (int t = 0; t < T; t++) { jb[t] = (job_t){kind, rows, n_rows * t / T, n_rows * (t + 1) / T, yq, K, out, cpus ? cpus[t] : -1}; pthread_create(&th[t], 0, run, &jb[t]); }
  for (int t = 0; t < T; t++) pthread_join(th[t], 0);
}

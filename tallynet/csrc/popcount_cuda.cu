// CUDA popcount of packed tally groups. C ABI. Built only when nvcc + CUDA work.

#include <cstdint>
#include <cuda_runtime.h>

namespace {

__global__ void popcount_kernel(
    const uint8_t* __restrict__ packed,
    int32_t* __restrict__ out,
    int64_t n,
    int nbytes,
    uint8_t last_mask) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const uint8_t* row = packed + i * static_cast<int64_t>(nbytes);
  int c = 0;
  int b = 0;
  for (; b + 4 <= nbytes - 1; b += 4) {
    unsigned v = static_cast<unsigned>(row[b]) | (static_cast<unsigned>(row[b + 1]) << 8) |
        (static_cast<unsigned>(row[b + 2]) << 16) | (static_cast<unsigned>(row[b + 3]) << 24);
    c += __popc(v);
  }
  for (; b < nbytes - 1; ++b) {
    c += __popc(static_cast<unsigned>(row[b]));
  }
  c += __popc(static_cast<unsigned>(row[nbytes - 1] & last_mask));
  out[i] = c;
}

}  // namespace

extern "C" {

#if defined(_WIN32)
#define TALLYNET_EXPORT __declspec(dllexport)
#else
#define TALLYNET_EXPORT __attribute__((visibility("default")))
#endif

TALLYNET_EXPORT const char* tallynet_popcount_cuda(
    const uint8_t* packed, int32_t* out, int64_t n, int32_t nbytes, int32_t S) {
  if (n <= 0 || nbytes <= 0 || S <= 0) {
    return nullptr;
  }
  const int valid = S - 8 * (nbytes - 1);
  const uint8_t mask = static_cast<uint8_t>(valid >= 8 ? 0xff : ((1 << valid) - 1));
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  popcount_kernel<<<blocks, threads>>>(packed, out, n, nbytes, mask);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return cudaGetErrorString(err);
  }
  return nullptr;
}

}  // extern "C"

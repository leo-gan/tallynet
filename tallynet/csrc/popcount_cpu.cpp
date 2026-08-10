// CPU popcount of packed tally groups. C ABI, no Python headers.
// packed: n groups, nbytes bytes each. Count the first S bits of each group.

#include <cstdint>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

inline int last_byte_mask(int S, int nbytes) {
  const int valid = S - 8 * (nbytes - 1);
  if (valid >= 8) {
    return 0xff;
  }
  return (1 << valid) - 1;
}

inline int popcnt_u8(uint8_t x) {
  return __builtin_popcount(static_cast<unsigned>(x));
}

#if defined(__AVX2__)
inline __m256i popcnt8_avx2(__m256i v) {
  const __m256i lut = _mm256_setr_epi8(
      0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2,
      2, 3, 2, 3, 3, 4);
  const __m256i nibble = _mm256_set1_epi8(0x0f);
  const __m256i lo = _mm256_and_si256(v, nibble);
  const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), nibble);
  return _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
}

inline void store_u8_as_i32(const uint8_t tmp[32], int32_t* dst, int n) {
  for (int k = 0; k < n; ++k) {
    dst[k] = static_cast<int32_t>(tmp[k]);
  }
}
#endif

void popcount_nbytes1(const uint8_t* packed, int32_t* out, int64_t n, uint8_t mask) {
#if defined(__AVX2__)
  if (mask == 0xff) {
    int64_t i = 0;
#if defined(_OPENMP)
#pragma omp parallel
    {
      int64_t begin, end;
#pragma omp for schedule(static)
      for (int64_t i0 = 0; i0 < n; i0 += 32) {
        begin = i0;
        end = i0 + 32;
        if (end > n) {
          end = n;
        }
        if (end - begin == 32) {
          __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + begin));
          __m256i c = popcnt8_avx2(v);
          alignas(32) uint8_t tmp[32];
          _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), c);
          store_u8_as_i32(tmp, out + begin, 32);
        } else {
          for (int64_t j = begin; j < end; ++j) {
            out[j] = popcnt_u8(packed[j]);
          }
        }
      }
    }
    return;
#else
    for (; i + 32 <= n; i += 32) {
      __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed + i));
      __m256i c = popcnt8_avx2(v);
      alignas(32) uint8_t tmp[32];
      _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), c);
      store_u8_as_i32(tmp, out + i, 32);
    }
    for (; i < n; ++i) {
      out[i] = popcnt_u8(packed[i]);
    }
    return;
#endif
  }
#endif
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (int64_t i = 0; i < n; ++i) {
    out[i] = popcnt_u8(static_cast<uint8_t>(packed[i] & mask));
  }
}

void popcount_generic(
    const uint8_t* packed, int32_t* out, int64_t n, int nbytes, uint8_t mask) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (int64_t i = 0; i < n; ++i) {
    const uint8_t* row = packed + i * nbytes;
    int c = 0;
    int b = 0;
    for (; b + 8 <= nbytes - 1; b += 8) {
      uint64_t w;
      std::memcpy(&w, row + b, 8);
      c += __builtin_popcountll(w);
    }
    for (; b < nbytes - 1; ++b) {
      c += popcnt_u8(row[b]);
    }
    c += popcnt_u8(static_cast<uint8_t>(row[nbytes - 1] & mask));
    out[i] = c;
  }
}

}  // namespace

extern "C" {

#if defined(_WIN32)
#define TALLYNET_EXPORT __declspec(dllexport)
#else
#define TALLYNET_EXPORT __attribute__((visibility("default")))
#endif

TALLYNET_EXPORT void tallynet_popcount_cpu(
    const uint8_t* packed, int32_t* out, int64_t n, int32_t nbytes, int32_t S) {
  if (n <= 0 || nbytes <= 0 || S <= 0) {
    return;
  }
  const uint8_t mask = static_cast<uint8_t>(last_byte_mask(S, nbytes));
  if (nbytes == 1) {
    popcount_nbytes1(packed, out, n, mask);
  } else {
    popcount_generic(packed, out, n, nbytes, mask);
  }
}

}  // extern "C"

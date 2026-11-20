// Standalone driver for the dedispersion kernel with GPU offload.
//
// This small program builds only src/kernels.cpp (no xtensor/HDF5) and
// times the fourier_domain_dedisperse kernel on the GPU with a warm-up
// call, so the first-call runtime init is not counted. It is the
// cross-compiler evidence used in the GPU offloading docs:
//
//   NVIDIA (NVHPC, OpenMP or OpenACC):
//     nvc++ -O3 -std=c++20 -mp=gpu -gpu=cc120,mem:separate:pinnedalloc \
//           -DDEDISP_GPU_TARGET_OMP -Iinclude src/kernels.cpp
//           examples/offload_bench.cpp
//     nvc++ -O3 -std=c++20 -acc  -gpu=cc120,mem:separate:pinnedalloc \
//           -DDEDISP_GPU_TARGET_ACC -Iinclude src/kernels.cpp
//           examples/offload_bench.cpp
//
//   AMD (ROCm clang, OpenMP):
//     clang++ -O3 -std=c++20 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
//             --offload-arch=gfx1201 -DDEDISP_GPU_TARGET_OMP \
//             -Iinclude src/kernels.cpp examples/offload_bench.cpp
//     OMP_TARGET_OFFLOAD=mandatory ./offload_bench
//
// The kernel map clauses are whole-array, so the NVHPC pinned allocator
// (-gpu=mem:separate:pinnedalloc) is safe. Use the default -DDEDISP_GPU_TARGET
// (cc120) on the gpu02 Blackwell card, cc89 on Ada, or cc89,cc120 for a
// multi-arch fatbin.
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "kernels.hpp"

using cplx = std::complex<float>;

int main(int argc, char **argv) {
  int repeats = argc > 1 ? atoi(argv[1]) : 3;
  size_t dm_count = 500, n_frequencies = 119904, n_channels = 32;
  size_t stride_in = n_frequencies, stride_out = n_frequencies;
  float time_resolution = 3.8147115214387945e-4f;

  std::vector<float> spin(n_frequencies), delays(n_channels), dm(dm_count);
  std::vector<cplx> in(n_channels * stride_in);
  std::vector<cplx> out(dm_count * stride_out);
  uint64_t s = 0x9e3779b97f4a7c15ull;
  auto unit = [&]() {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    uint64_t v = s * 0x2545F4914F6CDD1Dull;
    return (float)(v >> 11) * (1.0f / 9007199254740992.0f);
  };
  for (size_t i = 0; i < n_frequencies; ++i)
    spin[i] = i * (1.0f / (n_channels * time_resolution));
  for (size_t i = 0; i < n_channels; ++i)
    delays[i] = 4.0f * (i + 1);
  for (size_t i = 0; i < dm_count; ++i)
    dm[i] = 10.0f * (i + 1);
  for (size_t i = 0; i < in.size(); ++i)
    in[i] = cplx(unit() - 0.5f, unit() - 0.5f);

  double gflop = (double)dm_count * n_frequencies * n_channels * 8.0 / 1e9;

  // Warmup call: pays first-call runtime init (context, first map).
  dedisp::fourier_domain_dedisperse(
      dm_count, n_frequencies, n_channels, time_resolution, spin.data(),
      dm.data(), delays.data(), stride_in, stride_out, in.data(), out.data());
  double best = 1e30, first = 0;
  for (int i = 0; i < repeats; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    dedisp::fourier_domain_dedisperse(
        dm_count, n_frequencies, n_channels, time_resolution, spin.data(),
        dm.data(), delays.data(), stride_in, stride_out, in.data(), out.data());
    double wall = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
    if (i == 0)
      first = wall;
    if (wall < best)
      best = wall;
    printf("call %d: %8.3f ms  %8.1f GFlop/s  first=%.1fms\n", i, wall,
           gflop / (wall / 1e3), first);
  }
  printf("BEST %.3f ms -> %.1f GFlop/s\n", best, gflop / (best / 1e3));
  double chk = 0;
  for (size_t i = 0; i < out.size(); i += 4093)
    chk += out[i].real();
  printf("checksum %.4f\n", chk);
  return 0;
}

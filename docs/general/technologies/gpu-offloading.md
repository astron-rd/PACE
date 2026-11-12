# GPU offloading for C++

## Overview

GPU offloading lets you write standard C++ code and have the compiler generate
the GPU part of it. Instead of writing CUDA kernels and managing device memory
by hand, you annotate loops and regions with pragmas, and the compiler builds
and runs the kernels for you. The two main pragma-based frameworks for this are
OpenMP target offloading and OpenACC.

This page describes what we support in the PACE environment, which compiler to
use for which framework, and the lessons we learned while porting the PACE
dedispersion kernel. Detailed performance numbers are deferred to the
benchmark specification work and are not included here.

## Programming models

### OpenMP target offloading

OpenMP has supported accelerators since version 4. A region is offloaded with:

```cpp
#pragma omp target map(tofrom: result)
{
  result = compute();
}
```

The code in the block is compiled for the GPU and executed there. Data motion
is explicit through `map` clauses. See the
[OpenMP page](openmp.md) for the CPU-side usage we already rely on.

### OpenACC

OpenACC uses a similar pragma approach with a slightly different clause
vocabulary:

```cpp
#pragma acc parallel loop copyout(a[0:n])
for (int i = 0; i < n; ++i) {
  a[i] = i * i;
}
```

OpenACC was designed specifically for accelerators, so its data mapping and
loop constructs are a bit richer than the OpenMP equivalents.

## Available compilers

Both frameworks are available through the Spack stack on the DAS-6 cluster
(environment `offload`, target `x86_64_v3`). Both compilers were verified on
the GPU types in the DAS-6 GPU partition: an RTX 4000 Ada Generation
(compute capability 8.9) and an RTX PRO 4000 Blackwell (compute capability
12.0).

| Compiler | OpenMP offload | OpenACC | sm_89 (Ada) | sm_120 (Blackwell) |
|---|---|---|---|---|
| GCC 16.1.0 (`+nvptx`) | yes | no | yes | runs via PTX JIT, no native SASS |
| NVHPC 26.3 (`nvc`/`nvc++`) | yes (`-mp=gpu`) | yes | yes | yes |

### GCC

`gcc@16.1.0+nvptx` installs a full toolchain with an `nvptx-none` offload
target. Usage:

```sh
gcc -fopenmp -foffload=nvptx-none app.cpp -x c++ -o app
```

The default build embeds PTX, which the CUDA driver JIT-compiles on the device
at runtime. To embed native code for a specific card, add the microarch:

```sh
gcc -fopenmp -foffload=nvptx-none=-misa=sm_89 app.cpp -x c++ -o app
```

Limitations we verified:

- The nvptx backend of GCC 16 only understands up to `sm_89`. It cannot
  generate Blackwell (`sm_120`) code yet.
- GCC's `-fopenacc` is not usable with the nvptx target (the required
  offload runtime pieces are missing), so GCC is our OpenMP compiler only.

### NVHPC

The NVIDIA HPC compilers install as binaries through Spack. The C/C++
compilers are `nvc` and `nvc++`. Usage:

```sh
# OpenACC
nvc++ -acc -gpu=cc89 app.cpp -o app     # or -gpu=cc120 for Blackwell
# OpenMP target offload
nvc++ -mp=gpu -gpu=cc89 app.cpp -o app
# both, letting the runtime pick the device
nvc++ -acc -gpu=cc native app.cpp -o app
```

NVHPC generates native SASS for both Ada and Blackwell. Running OpenACC code
also needs the CUDA driver library available at runtime (loaded automatically
through the Spack module).

## SASS vs PTX JIT

GCC's nvptx backend stops at `sm_89`, so on Blackwell it emits PTX that the
CUDA driver JIT-compiles at runtime, while NVHPC emits native `sm_120` SASS.
There are two separate costs here and they are easy to mix up. The first is
the one-time JIT cost: the driver compiles the PTX once when the program
starts, which takes a few milliseconds and means nothing for a kernel that
runs longer. The second is the quality of the generated code: PTX built for
an older architecture cannot use newer instructions, and there are known
cases where this produces wrong results on newer cards. NVHPC is the only
toolchain here that skips the remap entirely, so the case for native SASS is
as much about correctness as about speed.

## Compiler requirements and the CUDA toolkit

The CUDA toolkit alone is not enough for directive-based offload. The
toolkit's `nvcc` compiles CUDA C++ (`.cu` files) and drives a host compiler.
It does not process `#pragma omp target` or `#pragma acc` code. To offload
directive-based C++ you need a host compiler with a device backend and an
offload runtime. The options are NVHPC (`nvc++`), GCC built with `+nvptx`,
or Clang/LLVM with a working NVIDIA libomptarget plugin. NVHPC's `nvcc` is
the same tool as the toolkit's. The difference is that NVHPC bundles two CUDA
toolkit versions and adds the directive-based `nvc`/`nvc++`/`nvfortran`
compilers that the toolkit alone does not ship.

On the Blackwell support timeline: native Blackwell cubins came with CUDA
12.8 (`sm_100`/`sm_120`). Clang added `sm_120` in LLVM 20, LLVM 19 rejects
it. That does not change the recommendation here, because the spack LLVM
install is built with `~cuda`, so it has no NVIDIA CUDA libomptarget plugin
and cannot offload to NVIDIA anyway. On the AMD side, ROCm's bundled clang is
the working OpenMP path.

## OpenACC on AMD and AOMP

The ROCm toolchain does not provide a production OpenACC implementation, and
the OpenMP compiler is the only directive model AMD ships. So OpenACC remains
NVIDIA-only in practice here: NVHPC is the full OpenACC implementation, and
GCC's OpenACC on nvptx is unusable because the required runtime pieces are
missing.

AOMP (AMD's Clang/LLVM downstream) targets OpenMP offloading, not OpenACC. It
could not directly add OpenACC support. The only OpenACC-on-AMD path is
Clacc, a separate LLVM/Clang fork that translates OpenACC to OpenMP and runs
on the LLVM AMDGCN backend. Clacc is not merged upstream, is research
grade, and is not part of the ROCm or AOMP shipped toolchain. Upstream LLVM
is only starting OpenACC support in Flang/Clang and it is not production
ready. The practical consequence is unchanged: if AMD is in scope, OpenMP is
the framework to standardize on, because it is the only directive model with
a real production compiler on both vendors' GPUs.

## Lessons from the dedispersion port

We ported the PACE dedispersion kernel to both frameworks on the
`gpu-offloading` branch. One `#pragma omp target teams distribute parallel
for` (OpenMP) or a `#pragma acc parallel` with `present` data clauses
(OpenACC) covers the whole kernel. Both frameworks produce bit-identical
results and agree with the CPU reference within float precision.

### Pin the host buffers that feed data clauses

Host-device copies through PCIe are slower than device memory bandwidth, so
for copy-heavy apps the copies, not the math, set the throughput ceiling.
Pinned (page-locked) host buffers transfer materially faster than pageable
ones. With NVHPC this needs no source change at all:

```sh
nvc++ -acc       -gpu=cc120 -gpu=mem:separate:pinnedalloc app.cpp
nvc++ -mp=gpu    -gpu=cc120 -gpu=mem:separate:pinnedalloc app.cpp
```

The flag makes NVHPC allocate the host buffers behind its data clauses
(`map(to:)` / `map(from:)`, `copyin` / `copyout`, `update host`) directly in
pinned memory. Caveats:

- The short spelling `-gpu=pinned` still works but prints a deprecation
  warning. `-gpu=mem:separate:pinnedalloc` is the current form.
- The pinned allocator does not like per-batch data-clause slices. A
  `copyin(a[lo:hi])` inside a per-batch kernel fails at runtime with
  `variable in data clause is partially present on the device`. Use a
  whole-array `enter data` / `present` style instead of slicing per batch.
- If a flag-free source is preferred: `cudaHostAlloc` works with NVHPC
  (its `openacc.h` ships neither `acc_pinned_alloc` nor `acc_host_alloc`),
  and `omp_target_alloc_pinned` from OpenMP 5.1 works with Clang. NVHPC's
  `omp.h` predates OpenMP 5.1, so those functions do not exist there.

### Unified memory is a trap for two-phase apps

NVHPC supports CUDA unified memory via `-gpu=mem:managed` or
`-gpu=mem:unified`, where `map` clauses become no-ops and the driver
migrates pages on demand. This looks fast in a benchmark until the output
must actually come back to the host: page-by-page migration of a large
output buffer costs several times more than an explicit bulk copy, and even
prefetch hints do not close the gap. An apparent win from managed memory is
often just deferred migration hiding the copy cost from the timer. For a
flow that sends a big result back to the host every call, explicit separate
memory with pinned buffers wins. Unified memory is worth revisiting only if
the data can stay resident on the device, or on a coherent-memory system
like Grace.

### Async overlap: OpenACC queues work, NVHPC OpenMP needs an explicit taskwait

OpenACC async queues overlap a batch's DtoH copy with the next batch's
compute, and with pinned buffers the overlap is large enough that the
batched pipeline ends up faster than the single synchronous region.
Batching alone buys nothing while buffers are pageable, because the copy
anchors the pipeline.

The OpenMP equivalent hit a runtime bug: `nowait` target regions work, but
NVHPC's implicit synchronization is broken. Returning from main or running a
later synchronous `target` region after `nowait` batches without an explicit
wait hangs or segfaults. Pairing `nowait` with an explicit `#pragma omp
taskwait` drains the pipeline and everything after it completes correctly,
including `depend`-chained variants. So the guidance is not "avoid
`nowait`" but "always pair it with an explicit `taskwait` and never rely on
implicit synchronization". On NVIDIA the OpenACC async queues remain the
simpler mechanism. OpenMP `nowait` plus `taskwait` is a working,
portability-respecting alternative.

### OpenMP offloading is genuinely portable to AMD

The same OpenMP kernel source, with no target-specific `#ifdef`s, offloads
to NVIDIA with NVHPC and compiles and runs correctly with the ROCm 7.14
bundled clang on the R9700 (Radeon AI PRO, gfx1201) in the DAS-6 GPU
partition, with matching checksums and float accuracy against a CPU
reference:

```sh
clang++ -O3 -std=c++20 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
  --offload-arch=gfx1201 -I include app.cpp kernels.cpp -o app_amd
OMP_TARGET_OFFLOAD=mandatory ./app_amd   # mandatory: host fallback fails loudly
```

Gotchas:

- The R9700's runtime-visible architecture is `gfx1201`, not `gfx1200`.
  Building for the wrong arch (or `--offload-arch=native` on a node with a
  different GPU) quietly leaves you without an offload image.
- Use the ROCm clang install, which ships a working AMD `libomptarget`
  plugin. The Spack LLVM builds do not, and building the plugin from source
  does not work with the stock options.
- `std::numbers::pi_v` requires `-std=c++20` on both compilers. The PACE
  kernel sources use it.

### Pitfalls

Several apparent findings during the port turned out to be measurement
mistakes. Watch for these before trusting a GPU number:

- The first offloaded call includes GPU startup overhead and can take
  several times longer than later calls. Always run the kernel once to warm
  up before timing it.
- The GPU runs at different speeds depending on its state and on what else
  the machine is doing, so two runs of the same code can differ by several
  times. Compare variants against each other inside the same run, not by
  comparing numbers from separate jobs.
- A profiler can show the GPU's main memory almost idle while its cache is
  busy. That means the kernel re-reads data from cache, which is normal and
  not a sign of a slow GPU. If you want it faster, read less from cache per
  thread rather than adding more compute.
- If the offload silently fails, the code falls back to running on the CPU
  and still gives correct answers, so a timing result can look fine while no
  GPU work happened. Set `OMP_TARGET_OFFLOAD=mandatory` so a fallback fails
  loudly, and check results against a CPU reference.

Concrete numbers for all of the above belong with the benchmark
specification work. The qualitative effects are what transfer to other
apps.

## Recommendation

- Use **NVHPC (`nvc++`)** as the default GPU offload compiler. It is the only
  option for OpenACC, it covers both GPU generations natively, and it also
  offers OpenMP offload via `-mp=gpu` if you prefer that pragma style.
- Use **GCC `+nvptx`** when a standard GNU toolchain matters, for example to
  check that offload pragmas are portable across compilers, or to keep CPU
  builds and GPU builds on the same compiler family. Accept that on Blackwell
  the GPU code goes through PTX JIT compilation.
- When the data volume is significant, pin the host buffers that feed `map` /
  `copy` clauses. With NVHPC that is a single compiler flag.
- For GPU-targeted portability to AMD as well as NVIDIA, OpenMP is the
  pragmatic choice. The same source already runs on both vendors'
  toolchains. OpenACC does not have an AMD compiler path here.

Both compilers implement the same OpenMP target pragma syntax, so code can be
compiled with either toolchain and compared directly. That makes a
GCC/NVHPC A/B comparison a cheap first experiment when porting an application
like dedispersion or IDG to the GPU.

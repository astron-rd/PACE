# GPU offloading for C++

## Overview

GPU offloading lets you write standard C++ code and have the compiler generate
the GPU part of it. Instead of writing CUDA kernels and managing device memory
by hand, you annotate loops and regions with pragmas, and the compiler builds
and runs the kernels for you. The two main pragma-based frameworks for this are
OpenMP target offloading and OpenACC.

This page describes what we support in the PACE environment and which compiler
to use for which framework.

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

Both frameworks are available through our private Spack stack
(`source /var/scratch/veenboer/spack/share/spack/setup-env.sh`, environment
`offload`, target `x86_64_v3`). Both compilers were verified on the two GPU
types in `gpu02`: an RTX 4000 Ada Generation (compute capability 8.9, Slurm
gres `AD4000`) and an RTX PRO 4000 Blackwell (compute capability 12.0, gres
`B4000`).

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
at runtime. This verified working on both the Ada and the Blackwell card. To
embed native code for a specific card, add the microarch:

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

## What we measured on the dedispersion port

We ported the PACE dedispersion kernel (`dedisp/cpp`) to OpenMP and OpenACC.
One `#pragma omp target teams distribute parallel for` (OpenMP) or a
`#pragma acc parallel` with `present` data clauses (OpenACC) covers the whole
kernel, which is a pipeline of cheap FFTs followed by the compute-heavy
dedispersion loop. Details of the port live in the `gpu-offloading` branch
of the pace repo.

With the application's real input (119904 frequency channels, 32 sub-bands,
500 dispersion measures) the wall-clock breakdown on Blackwell was:
offload disabled 1.69 s, OpenMP 0.80 s, OpenACC 0.70 s. The two frameworks
produce bit-identical results and agree with the CPU reference to
`1.4e-06` (float precision).

The kernel is memory bound: ncu measured 481 GB/s effective DRAM bandwidth on
Blackwell (75% of peak) and 134 GB/s on Ada (65%). There is no benefit from
tuning the math further; the remaining gap is data motion.

### Host-device copies are the throughput bottleneck

A standalone copy microbenchmark on Blackwell (PCIe Gen5 x16) measured:

| Copy direction | Pageable | Pinned | Speedup |
|---|---|---|---|
| HtoD | 35.9 GB/s | 57.8 GB/s | 1.6x |
| DtoH | 35.8 GB/s | 56.5 GB/s | 1.6x |
| async round-trip | - | 57.2 GB/s effective | - |

So input/output buffers used in `map(to:)` / `map(from:)` or `copyin` /
`copyout` clauses should be pinned (page-locked) host memory when the transfer
size or frequency is significant. In the dedispersion app the buffers are
allocated with plain `new`/`std::vector`, so today they are pageable and the
copies run at the slower ~36 GB/s.

The simplest way to pin with NVHPC is the compiler flag, no API call or
allocator wrapper needed:

```sh
nvc++ -acc  -gpu=cc120 -gpu=mem:separate:pinnedalloc  app.cpp   # OpenACC
nvc++ -mp=gpu -gpu=cc120 -gpu=mem:separate:pinnedalloc  app.cpp  # OpenMP
```

The flag tells NVHPC to allocate the host buffers that back its data
clauses (`map(to:)` / `map(from:)`, `copyin` / `copyout`, `update host`)
directly in pinned memory instead of pageable. Verified on the Blackwell
B4000 with the real 119904 x 32 x 500 OpenMP kernel: `54.7 ms` (281 GFlop/s)
plain vs `49.9 ms` (308 GFlop/s) with the flag, same checksum, no CUDA calls
in the source. The OpenACC async bench (big 32769 x 1024 x 375 shape)
confirms the full copy-bound win: `440 ms` plain vs `311 ms` pinned by the
flag, a 1.42x improvement against a bit-identical checksum.

Two caveats:

- The short spelling `-gpu=pinned` still works but prints a deprecation
  warning. `-gpu=mem:separate:pinnedalloc` is the current form.
- The pinned allocator does not like per-batch data-clause slices. A
  `copyin(a[lo:hi])` inside a per-batch kernel fails at runtime with
  `variable in data clause is partially present on the device`. Use a
  whole-array `enter data` / `present` style (copy the whole `dm` array once)
  instead of slicing it per batch.

### Unified memory and prefetch hints: supported, but a loss for this app

NVHPC also supports CUDA unified memory for offload. Compile with
`-gpu=mem:managed` (dynamic allocations become managed memory) or
`-gpu=mem:unified` (full unified memory, needs HMM-capable hardware. The
B4000 accepted it). In these modes the `map` clauses become no-ops and the
CUDA driver migrates pages on demand. The runtime exposes prefetch and
advice extensions in `openacc.h`: `accx_mem_prefetch` (wraps
`cudaMemPrefetchAsync`), `accx_mem_advise` (read-mostly, preferred
location, accessed-by), and `accx_set_mem_hints` / `accx_set_mem_prefetch`
to enable them automatically at data constructs.

Measured on the real 119904 x 32 x 500 kernel (B4000), warm, 3 calls:

| Build | Output read back on host | BEST |
|---|---|---|
| `mem:separate:pinnedalloc` (reference) | yes | 49.9 ms / 308 GFlop/s |
| `mem:managed`, no hints | no (migration deferred) | 42.0 ms / 366 |
| `mem:managed`, no hints | yes (full readback in timer) | 219 ms / 70 |
| `mem:managed` + host `accx_mem_prefetch` | yes | 73 ms / 211 |
| `mem:unified` | yes | 238 ms / 64 |
| `mem:managed` + device prefetch/advise of input | deferred | 42.0 ms / 365 (no gain) |

Two takeaways. First, the headline "managed is 15% faster" is an illusion
for a two-phase copy app: it only holds while the 480 MB output stays on
the device, and deferred migration hides that cost from the timer. When
the output must actually come back to the host and be consumed, managed
memory's page-by-page migration costs 4.4x more than the explicit
`map(from:)` bulk copy (219 vs 49.9 ms), and even the prefetch hint
(`cudaMemPrefetchAsync` to host, 73 ms) loses to the plain pinned copy.
Second, pinning input with device prefetch/advice gains nothing here,
because the input is already device-resident after the first call, so the
runtime has nothing left to migrate. For dedispersion, where a large
output buffer must return to the host every call, explicit separate
memory wins: pinned host buffers plus a bulk copy beats unified-memory
migration on a PCIe GPU. Unified memory is worth revisiting only if the
data can stay on the device and only deltas need to come back, or on an
HMM/coherent-memory system like Grace.

Pinning methods that work with the compilers here, if a flag-free source is
preferred:

- NVHPC OpenACC: `-gpu=mem:separate:pinnedalloc`, or `cudaHostAlloc` (NVHPC's
  `openacc.h` ships neither `acc_pinned_alloc` nor `acc_host_alloc`).
- NVHPC OpenMP: NVHPC `omp.h` is pre-OpenMP-5.1, so `omp_target_alloc_pinned` /
  `omp_target_alloc_host` do not exist. Pinning is available through
  `-gpu=mem:separate:pinnedalloc` or the CUDA/OpenACC allocator path.
- Clang (LLVM/ROCm): `omp_target_alloc_pinned(..., 0)` from OpenMP 5.1 works,
  and ROCm clang does not need this flag.

### DM-batched async overlap: OpenACC yes, NVHPC OpenMP no

We benchmarked splitting the DM loop into 8 batches, launching the kernel per
batch on an alternating async queue, and for OpenACC issuing the `update host` for the finished batch on the same queue so its DtoH copy overlaps the
next batch's compute. Result buffer pageable vs pinned, Blackwell:

| Mode (OpenACC) | Wall | GFlop/s |
|---|---|---|
| single sync region, pageable (baseline) | 462.7 ms | 217.5 |
| 8 batch async, pageable | 465.6 ms | 216.2 |
| 8 batch async, pinned via `cudaHostAlloc`, per-batch DtoH overlap | 321.9 ms | 312.7 |
| 8 batch async, pinned via compiler flag only, per-batch DtoH overlap | 311 ms | 324 |

Batching alone buys nothing while the buffers are pageable (the DtoH copy
anchors the pipeline and blocks). Pinned plus per-batch overlap gives 1.45x,
and the compiler flag (`-gpu=mem:separate:pinnedalloc`) reaches the same
pinned throughput without a `cudaHostAlloc` call in the source. The flag
run used a whole-array `present` style because per-batch `copyin` slices
crash the pinned allocator.
Notably, OpenACC batching hides the copies so completely that total time
drops below the single sync region even though more work is done.

The equivalent OpenMP experiment with `nowait`-marked per-DM-batch target
regions originally hung the NVHPC runtime (the `nowait` task never completed,
verified standalone with a 200 s timeout, RC 124). A minimal repro in a fresh
driver narrowed the cause: the async `nowait`/`depend` mechanism itself works,
it is NVHPC's implicit synchronization path that is broken. Launching the
batches with `nowait` and then either returning from main (implicit drain at
exit) or running a later synchronous `target` region without an intervening
synchronization point hangs the runtime on 26.3 (RC 124) and segfaults on
26.9 (RC 139). Adding an explicit `#pragma omp taskwait` after the `nowait`
batches drains the pipeline and everything after it, including a follow-up
synchronous `target` region, completes correctly on both 26.3 and 26.9, with
`depend`-chained pipelines also working. So the guidance is narrower than
'do not use nowait with NVHPC': use it, but always pair it with an explicit
`omp taskwait` and never rely on implicit synchronization at program exit or
at a later synchronous target region. On NVIDIA the OpenACC async queues
remain the simpler and more complete overlap mechanism, but OpenMP nowait
plus taskwait is a working portability-respecting alternative if a single
source is preferred.

### OpenMP portability to AMD GPUs: verified on the R9700

The same OpenMP kernel source (no `#ifdef` changes for the target) that
offloads to NVIDIA with NVHPC was compiled with ROCm's bundled clang
(7.14.0, `clang-23`) and executed on the R9700 (Radeon AI PRO, gfx1201,
Slurm gres `R9700`) on `gpu02`:

```sh
/var/software/opt/rocm-7.14.0/core-7.14/lib/llvm/bin/clang++ \
  -O3 -std=c++20 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
  --offload-arch=gfx1201 -I dedisp/cpp/include \
  dedisp_real_omp.cpp dedisp/cpp/src/kernels.cpp -o real_omp_amd
OMP_TARGET_OFFLOAD=mandatory ./real_omp_amd   # mandatory: host fallback fails loudly
```

Result on the R9700 with the real 119904 x 32 x 500 workload,
`OMP_TARGET_OFFLOAD=mandatory` (so no silent host fallback), and a warm-up
call before timing: `212 GFlop/s` (72 ms), checksum identical to a serial
CPU reference, `max |gpu-cpu| = 9.8e-07` (same float accuracy as NVIDIA).
The same source compiled with NVHPC and run on the B4000 produced the same
checksum/cpu-diff and an identical first-call pattern, warming to
`281 GFlop/s` (55 ms). These are the honest steady-state numbers; an earlier
report that claimed 52.5 GFlop/s on the B4000 vs 142.5 on the R9700 was
wrong - both were first-call timings taken inside a single-shot call, and
those pay NVHPC/ROCm runtime init (module load, context, first map). On both
cards the first call takes ~3-8x the steady-state time, so a benchmark must
warm up before timing.

Gotchas we hit:

- The R9700's runtime-visible architecture is `gfx1201`, not `gfx1200`.
  Building for `gfx1200` (or using `--offload-arch=native` on a compile node
  with a different GPU) quietly leaves you without an offload image.
- The AMD toolchain is the ROCm one already present on the node
  (`/var/software/opt/rocm-7.14.0/core-7.14` is shared from das6). The Spack
  LLVM builds (`llvm@20.1.8`) do not ship the AMDGPU `libomptarget` plugin,
  and building it from source does not work with the stock option (the build
  comes out without the plugin). It is unnecessary anyway because the ROCm
  clang install already provides a working OpenMP target runtime for AMD.
- `std::numbers::pi_v` requires `-std=c++20` on both compilers. The PACE
  kernel sources use it.

The conclusion: OpenMP target offloading is a genuinely portable solution
here. The same unmodified kernel source offloads correctly on NVIDIA
(NVHPC, native SASS for Ada and Blackwell) and on AMD (ROCm clang, native
gfx1201 image), with matching float-accuracy results. OpenACC remains the
better performance lever on NVIDIA, mainly for its async queues and pinned
copy overlap.

### What actually limits the kernel: L2, not DRAM, not compute

The user-facing question "is the kernel compute-bound?" measured on the
B4000 with ncu full sections (same-process A/B, pinned clocks): the real
kernel is **cache-bound, not DRAM-bound and not compute-bound**.

| ncu Speed-of-Light (real kernel, Blackwell) | value |
|---|---|
| kernel duration (pinned clocks) | 40 ms |
| Memory Throughput | 90% of peak |
| L2 Cache Throughput | 90% |
| L1/TEX Cache Throughput | 61% |
| DRAM Throughput | only 2% |
| Compute (SM) Throughput | 67% |

The reason DRAM is nearly idle: the 30 MB input spectrum is re-read once per
DM trial (500 times), so it stays resident in L2 and the kernel is bound by
L2 -> L1 read bandwidth, not by HBM. The classic offload flow (stream the
whole input to the device, loop the DM trials on the GPU, bring the result
back) is therefore the right shape - the DM loop already stays on the GPU -
and the remaining lever is per-thread L2-read reduction (e.g. block the DM
loop so each input sample is read fewer times), plus pinned buffers for the
per-batch copies. A first high-level check: ncu DRAM Throughput \<< Memory
Throughput with L2 high is the signature of "cache-resident re-reads", not
a sign the GPU is underutilised.

Also confirmed by the same measurement: the standalone "kernel bench" and
the real `kernels.cpp` produce identical SASS (352 instructions, same
opcode mix) and identical kernel times (73.4 vs 74.4 ms at pinned clocks).
Reports of the bench "surpassing" the kernel were clock-state noise, not a
codegen difference - different GPU clock rates across sbatch jobs (180 MHz
idle vs 2467 MHz boost) inflate or shrink single numbers by 3-5x. Always
time kernels in-process against each other, or pin clock control, before
attributing a difference to code. The ncu-python wrapper runs with default
clock control; pass `--clock-control base` (or none) explicitly and never
compare absolute ms across jobs on shared GPUs.

## SASS vs PTX JIT: how much does it matter

GCC's nvptx backend only understands up to `sm_89`, so even on the Blackwell
B4000 (sm_120) GCC emits Ada PTX that the CUDA driver JIT-compiles at
runtime. NVHPC emits native `sm_120` SASS. The practical question is what
that difference costs, and the honest answer is workload-dependent and
usually small: typically a few percent to low double digits, not the 33x
figure that sometimes circulates.

Two separate costs are usually conflated. First the one-time JIT cost: the
driver compiles PTX to SASS on first load and caches it, a few milliseconds
amortized to nothing on a kernel that runs for tens of ms. Second, codegen
quality across the architecture gap: GCC's sm_89 PTX must be remapped onto
Blackwell by the driver and cannot exploit Blackwell-only instructions,
which is where the real (usually single digit) loss lives. On the L2-bound
dedispersion kernel, where compute is not the bottleneck, that codegen
difference lands mostly in the noise. In this kind of workload the reasons
to prefer native SASS are as much about correctness and guaranteed
Blackwell-optimal code as about raw speed: building sm_89 PTX for a
Blackwell card has known wrong-codegen reports, and NVHPC is the only
toolchain here that removes the generation-gap remap entirely.

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
12.8 (`sm_100`/`sm_120`). Clang added `sm_120` in LLVM 20 (sm_101/sm_120
landed early 2025). LLVM 19 rejects it. That does not change the
recommendation here, because the spack LLVM install is built with `~cuda`,
so it has no NVIDIA CUDA libomptarget plugin and cannot offload to NVIDIA
anyway. On the AMD side, ROCm's bundled clang is the working OpenMP path.

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

## Recommendation

- Use **NVHPC (`nvc++`)** as the default GPU offload compiler. It is the only
  option for OpenACC, it covers both GPU generations natively, and it also
  offers OpenMP offload via `-mp=gpu` if you prefer that pragma style.
- Use **GCC `+nvptx`** when a standard GNU toolchain matters, for example to
  check that offload pragmas are portable across compilers, or to keep CPU
  builds and GPU builds on the same compiler family. Accept that on Blackwell
  the GPU code goes through PTX JIT compilation.
- For performance work on the Blackwell card, prefer NVHPC: it is the only
  compiler here that emits `sm_120` SASS.
- When the data volume is significant, pin the host buffers that feed `map` /
  `copy` clauses. It is a one-line change and buys 1.6x on PCIe copies.
- For GPU-targeted portability to AMD as well as NVIDIA, OpenMP is the
  pragmatic choice. The same source already runs on both vendors'
  toolchains. OpenACC does not have an AMD compiler path here.

Both compilers implement the same OpenMP target pragma syntax, so code can be
compiled with either toolchain and compared directly. That makes a
GCC/NVHPC A/B comparison a cheap first experiment when porting an application
like dedispersion or IDG to the GPU.

## Reproducing the verification

Minimal compile-and-run checks live in `~/gputest*.sh` on das6. Submit with:

```sh
sbatch -p rbq -N1 -w gpu02 --gres=gpu:B4000:1 -t 10 ~/gputest2.sh
```

The async/pinned bench (OpenACC) is `~/gpu-bench/dedisp_gpu_async2.cpp`
(`async_run.sh`), the OpenMP nowait variant is
`~/gpu-bench/dedisp_omp_async.cpp`, and the real kernel portability driver is
`~/gpu-bench/dedisp_real_omp.cpp` with `~/real_omp_r9700.sh` (builds with
NVHPC for NVIDIA and ROCm clang for AMD).

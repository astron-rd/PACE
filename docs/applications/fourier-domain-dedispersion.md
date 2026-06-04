# Fourier-Domain Dedispersion (FDD)

This page combines a conceptual primer with implementation details for
Fourier-domain dedispersion in the PACE context.

## Background

Beamformed radio data are commonly searched for dispersed transient and periodic
signals such as pulsars and fast radio bursts (FRBs). As a radio pulse
propagates through the ionised interstellar medium, lower radio frequencies
travel more slowly than higher frequencies. This creates a
frequency-dependent arrival-time delay that smears the signal across the
observing band.

Dedispersion corrects for this effect by realigning the signal across frequency
channels for a set of trial dispersion measures (DMs). In practical search
pipelines, beamformed data are therefore processed for many DM values to recover
sharp pulses and improve detectability.

## Concepts

Some useful papers and online resources:

- [Fourier-domain dedispersion](https://www.aanda.org/articles/aa/full_html/2022/01/aa42099-21/aa42099-21.html)
- [Dispersion measure: Confusion, Constants & Clarity](https://arxiv.org/abs/2007.02886)
- [Dispersion measure](https://casper.berkeley.edu/astrobaki/index.php/Dispersion_measure)
- [Pulsar Dispersion Measure](https://astronomy.swin.edu.au/cosmos/*/Pulsar+Dispersion+Measure)
- [ASTRON dedisp repository](https://git.astron.nl/RD/dedisp)

### Dispersion

Due to variations in the interstellar medium (ISM) along the line of sight to a
radio transient (e.g. a pulsar), the signal is dispersed as a function of
frequency. There is a time delay between the signal observed at $\nu$ and a
reference frequency $\nu_0$, described by:

$$
\Delta t(\nu, \mathrm{DM}) = \mathrm{DM}\,\kappa_{\mathrm{DM}}\left(\nu^{-2} - \nu_0^{-2}\right)
$$

where $\kappa_{\mathrm{DM}}$ is a proportionality constant and $\mathrm{DM}$
is the dispersion measure, defined as the path integral over electron density
along the line-of-sight:

$$
\mathrm{DM} = \int n_e(\ell)\,\mathrm{d}\ell
$$

### Beamformed input data

The input to dedispersion is typically beamformed filterbank data: a
two-dimensional array with one time series per frequency channel. Each sample
represents detected power in a given channel and time bin. Incoherent
dedispersion operates on these detected intensities rather than on the original
complex voltages.

For a trial DM, the relative delay between two frequencies scales approximately
as:

$$
\Delta t \propto \mathrm{DM}\left(\nu_1^{-2} - \nu_2^{-2}\right)
$$

so lower-frequency channels must be shifted by larger amounts than
higher-frequency channels.

### Time-domain dedispersion

Given a Stokes I spectrum $I(t, \nu)$, time-domain dedispersion applies
channel-dependent delays and sums channels:

$$
I(t, \mathrm{DM}) = \sum_{\nu} I\left(t - \Delta t(\nu, \mathrm{DM}), \nu\right)
$$

Since DM is usually unknown, this process is repeated for many trial DMs (often
100-1000).

Note on performance: the time-domain approach requires significant memory
bandwidth while doing relatively little computation per memory access.

### Fourier-domain dedispersion

Instead of applying time delays directly, Fourier-Domain Dedispersion (FDD)
applies equivalent phase shifts in frequency space. This increases arithmetic
intensity and is often better suited to modern accelerators such as GPUs.

We Fourier-transform each channel (observation frequency) $\nu$ to obtain
intensity as a function of spin frequency $f_s$:

$$
I(f_s, \nu) = \mathcal{F}_{t \rightarrow f_s}\{I(t, \nu)\} = \int I(t, \nu)\,e^{-2\pi i f_s t}\,\mathrm{d}t
$$

In this space, the time delay is applied as a phase rotation:

$$
\mathcal{W}(f_s, \nu, \mathrm{DM}) = e^{-2\pi i f_s\,\Delta t(\nu, \mathrm{DM})}
$$

A discrete-time equivalent for channel spectrum $X_c[k]$ is:

$$
X'_c[k] = X_c[k] e^{-2\pi i k \tau_c / N}
$$

where $N$ is the transform length and $\tau_c$ is the delay for channel $c$ at
the current trial DM.

### Algorithm

For each block of beamformed data, FDD proceeds in three steps:

1. FFT the time series of every frequency channel.
1. For each trial DM, compute per-channel delays and apply corresponding phase
   rotations in the Fourier domain.
1. Sum corrected channels to form a dedispersed output for that DM.

In pseudocode:

```
for each data block:
  for each frequency channel c:
    spectrum[c] = FFT(time_series[c])

  for each trial DM d:
    dedispersed_spectrum = 0
    for each frequency channel c:
      delay = compute_delay(d, c)
      for each Fourier bin k:
        phasor = exp(-2pii k delay / N)
        dedispersed_spectrum[k] += spectrum[c][k] * phasor

    if time_domain_output_required:
      output[d] = IFFT(dedispersed_spectrum)
    else:
      output[d] = dedispersed_spectrum
```

If a search pipeline ultimately needs time-domain output, an inverse FFT can be
applied after summation. However, for FFT-based periodicity searches this step
can be omitted and data can remain in the Fourier domain.

The main motivation for FDD is performance. Time-domain dedispersion is often
memory-bandwidth-limited, while FDD shifts more work into arithmetic on
Fourier-domain data and can be faster for large DM counts.

## PACE simplifications

The PACE reference application focuses on the Fourier-domain dedispersion
algorithm itself rather than on a full real-world dedisp pipeline.

- Only Fourier-domain dedispersion is considered.
- Time-domain dedispersion from dedisp is out of scope.
- Focus is on the dedispersion computation on beamformed data.

## Fourier-Domain Dedispersion data flow

This section describes how data flows through the reference implementation of
frequency-domain dedispersion (FDD), including the main data transformations,
transposes, and FFT operations.

Scope: this page describes the CPU FDD non-segmented execution path
(`execute_cpu`).

**Summary of shapes**:

```
input    : [time, channel]
data_nu  : [channel, padded_time_or_freq]
data_dm  : [dm, padded_time_or_freq]
output   : [dm, nsamps_computed]
```

**Parameters**

Example values for testing/demonstration:

```
nsamps = 120000
nchans = 1024
dm_start = 2.0
dm_end = 100.0
```

**Input data generation**

Create synthetic test data with embedded dispersed pulse signal.

```
float rawdata[nsamps * nchans]  # time-major: data[sample_idx, channel_idx]
```

**Fill with random noise**

```
for ns in [0:nsamps]:
    for nc in [0:nchans]:
        rawdata[ns * nchans + nc] = random()
```

**Embed a dispersed pulse signal**

```
for nc in [0:nchans]:
    delay_s[nc] = <dispersion delay for channel nc>
    ns = (sigT + delay_s[nc]) / dt
    rawdata[ns * nchans + nc] += sigamp
```

**Quantization**

Convert float input to storage format. Float values are quantized to 8-bit
unsigned integers (range `[-127.5, 127.5]`):

```
byte input[nsamps * nchans]
```

**Intermediate dimensions**

Compute output dimensions accounting for dispersion delay.

```
max_delay = dm_list[dm_count - 1] * delay_table[nchans - 1]  # largest dispersion delay in samples
nsamps_computed = nsamps - max_delay  # number of valid output samples
```

**FFT length calculations**

Compute FFT buffer sizes with zero-padding:

```
nsamp_fft = round_up(nsamps + 1, 16384)  # actual FFT length (zero-padded to multiple of 16k)
nfreq = (nsamp_fft / 2 + 1)              # frequency components for an FFT of length nsamp_fft
nsamp_padded = round_up(nsamp_fft + 1, 1024)  # buffer size (holds nfreq complex = 2*nfreq floats)
```

**Preprocessing: transpose and format conversion**

Rearrange data layout and convert from 8-bit to float for FFT processing.

Allocate buffers:

```
float data_nu[nchans * nsamp_padded]    # [channel][time], time dimension is padded
float data_dm[ndm * nsamp_padded]       # [dm][time], time dimension is padded
```

Transpose from `(time, channel)` to `(channel, time)` layout and convert from
8-bit to float. After transpose, only the first `nsamps` time samples are
filled, the rest is zero-padded.

```
dst[channel, time] = (src[time, channel] - 127.5) / nchans
transpose_data(nchans, nsamps, nchans, nsamp_padded, 127.5, nchans, input, data_nu)
```

**Frequency domain transform**

Convert time-domain data to frequency domain to enable fast dedispersion.

Perform in-place Real-to-Complex FFT of length `nsamp_fft` (not
`nsamp_padded`!):

```
fft_r2c_inplace(nsamp_fft, nchans, nsamp_padded, data_nu)
```

- Input: `nsamp_fft` real values per channel (zero-padded after `nsamps`
  samples)
- Output: `nfreq = nsamp_fft/2 + 1` complex values per channel
- Batch size: `nchans` (process all channels)
- Stride: `nsamp_padded` floats between channels = `nsamp_padded/2` complex
  values between channels
- Note: Dedispersion only consumes bins `0..nfreq-1`; higher bins from the
  padded FFT are not used

**Dedispersion in Frequency Domain**

Core algorithm: apply dispersion correction by summing frequency components
across channels with frequency-dependent phase shifts.

Apply frequency-domain dedispersion kernel:

```
dedisperse(ndm, nfreq, nchans,
           dt, spin_frequencies, dm_list, delay_table,
           nsamp_padded/2, nsamp_padded/2,  # strides in complex elements
           (complex<float>*)data_nu, (complex<float>*)data_dm)
```

For each DM and frequency, we compute the complex sum across all channels with
appropriate phase shifts. The phase shift (phasor) accounts for the dispersion
delay at each channel.

Strides are in units of `complex<float>`:

- `in_stride = nsamp_padded / 2` (jump between channels in `data_nu`)
- `out_stride = nsamp_padded / 2` (jump between DMs in `data_dm`)

**Inverse Frequency Domain Transform**

Convert dedispersed data back to the time domain.

Perform in-place Complex-to-Real FFT of length `nsamp_fft` (not
`nsamp_padded`!):

```
fft_c2r_inplace(nsamp_fft, ndm, nsamp_padded, data_dm)
```

- Input: `data_dm` contains nfreq complex values per DM (rest is zero from
  initialisation)
- Output: `nsamp_fft` real values per DM (zero-padded to `nsamp_padded`)
- Batch size: `ndm` (process all output DMs)
- Stride: `nsamp_padded` floats between DMs
- Note: The FFT length is `nsamp_fft` (not `nsamp_padded`), so only the first
  `nsamp_fft` time samples are computed, the remaining padded region is
  undefined

**Copy output (trimmed to valid region)**

Extract the valid output region, discarding samples corrupted by dispersion
delay edge effects.

Copy only the first `nsamps_computed` valid samples from each DM:

```
copy_data(ndm, nsamps_computed, nsamp_padded, nsamps_computed, data_dm, output)
```

- Input: `data_dm[dm * nsamp_padded][time]`, stride=`nsamp_padded`
- Output: `output[dm * nsamps_computed][time]`, stride=`nsamps_computed`

**Why only `nsamps_computed`?**

- We keep the earliest `nsamps_computed = nsamps - max_delay` samples from each
  DM output row
- We drop the last `max_delay` samples, which depend on data beyond the observed
  time window
- This avoids edge effects from delay alignment near the end of the buffer

**Buffer visualisation**

This view shows one row of each 2D buffer (one channel or one DM), how wide it
is in samples/elements, and which part is consumed by each step.

Note: bars are schematic (not exact scale); they show relative used vs
padded/discarded regions.

Legend:

- `=` used/valid
- `.` allocated but unused/padded
- `x` discarded

**1) Input buffer view**

```
input row (one time sample across channels)
type: byte (uint8)
shape: [nsamps, nchans]

time axis (per channel series):
[==============================]
    0                        nsamps-1

all nsamps values are populated from quantized input
```

**2) `data_nu` before R2C FFT**

```
data_nu row before R2C FFT (one channel)
type: float
row width: nsamp_padded floats

[==========================....]
    0      nsamps-1   nsamp_fft-1  nsamp_padded-1

used/populated as real input to FFT:    0 .. nsamp_fft-1
original data lives in:                 0 .. nsamps-1
zero-padding region for FFT:            nsamps .. nsamp_fft-1
extra allocated tail (not FFT input):   nsamp_fft .. nsamp_padded-1
```

**3) `data_nu` after R2C FFT (complex view)**

```
data_nu row after R2C FFT (same memory, reinterpreted)
type: complex<float>
row width: nsamp_padded/2 complex values

[=========================....]
    0      nfreq-1             nsamp_padded/2-1

valid FFT bins produced for length nsamp_fft: 0 .. nfreq-1
remaining complex slots are not consumed by dedispersion
```

**4) `data_dm` after dedispersion (complex view)**

```
data_dm row after dedispersion (one DM)
type: complex<float>
row width: nsamp_padded/2 complex values

[=========================....]
    0      nfreq-1             nsamp_padded/2-1

dedispersion writes/uses bins 0 .. nfreq-1
```

**5) `data_dm` after C2R FFT (float view)**

```
data_dm row after C2R FFT (same memory, reinterpreted)
type: float
row width: nsamp_padded floats

[==========================....]
    0      nsamp_fft-1         nsamp_padded-1

C2R produces nsamp_fft real samples; tail remains outside computed region
```

**6) Final output buffer view**

```
output row (one DM)
type: float
row width: nsamps_computed floats

source in data_dm row:
[==========================xxxx]
    0   nsamps_computed-1      nsamp_fft-1

kept:    0 .. nsamps_computed-1
dropped: nsamps_computed .. (nsamps-1)   (effectively max_delay samples)
```

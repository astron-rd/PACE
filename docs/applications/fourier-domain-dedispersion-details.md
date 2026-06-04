## Fourier-Domain Dedispersion data flow

This describes how data flows through the reference implementation of frequency-domain dedispersion (FDD), including the various data transformations, transposes, and FFT operations.

Scope: this page describes the CPU FDD non-segmented execution path (`execute_cpu`).

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

Convert float input to storage format. Float values are quantized to 8-bit unsigned integers (range `[-127.5, 127.5]`):

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
nfreq = (nsamps / 2 + 1)           # frequency components for an FFT of length nsamps
nsamp_fft = round_up(nsamps + 1, 16384)  # actual FFT length (zero-padded to multiple of 16k)
nsamp_padded = round_up(nsamp_fft + 1, 1024)  # buffer size (holds nfreq complex = 2*nfreq floats)
```

**Preprocessing: transpose and format conversion**

Rearrange data layout and convert from 8-bit to float for FFT processing.

Allocate buffers:

```
float data_nu[nchans * nsamp_padded]    # [channel][time], time dimension is padded
float data_dm[ndm * nsamp_padded]       # [dm][time], time dimension is padded
```

Transpose from `(time, channel)` to `(channel, time)` layout and convert from 8-bit to float. After transpose, only the first `nsamps` time samples are filled, the rest is zero-padded.

```
dst[channel, time] = (src[time, channel] - 127.5) / nchans
transpose_data(nchans, nsamps, nchans, nsamp_padded, 127.5, nchans, input, data_nu)
```

**Frequency domain transform**

Convert time-domain data to frequency domain to enable fast dedispersion.

Perform in-place Real-to-Complex FFT of length `nsamp_fft` (not `nsamp_padded`!):

```
fft_r2c_inplace(nsamp_fft, nchans, nsamp_padded, data_nu)
```

- Input: `nsamp_fft` real values per channel (zero-padded after `nsamps` samples)
- Output: `nfreq = nsamp_fft/2 + 1` complex values per channel
- Batch size: `nchans` (process all channels)
- Stride: `nsamp_padded` floats between channels = `nsamp_padded/2` complex values between channels
- Note: Dedispersion only consumes bins `0..nfreq-1`; higher bins from the padded FFT are not used

**Dedispersion in Frequency Domain**

Core algorithm: apply dispersion correction by summing frequency components across channels with frequency-dependent phase shifts.

Apply frequency-domain dedispersion kernel:

```
dedisperse(ndm, nfreq, nchans,
           dt, spin_frequencies, dm_list, delay_table,
           nsamp_padded/2, nsamp_padded/2,  # strides in complex elements
           (complex<float>*)data_nu, (complex<float>*)data_dm)
```

For each DM and frequency, we compute the complex sum across all channels with appropriate phase shifts. The phase shift (phasor) accounts for the dispersion delay at each channel.

Strides are in units of `complex<float>`:

- `in_stride = nsamp_padded / 2` (jump between channels in `data_nu`)
- `out_stride = nsamp_padded / 2` (jump between DMs in `data_dm`)

**Inverse Frequency Domain Transform**

Convert dedispersed data back to the time domain.

Perform in-place Complex-to-Real FFT of length `nsamp_fft` (not `nsamp_padded`!):

```
fft_c2r_inplace(nsamp_fft, ndm, nsamp_padded, data_dm)
```

- Input: `data_dm` contains nfreq complex values per DM (rest is zero from initialisation)
- Output: `nsamp_fft` real values per DM (zero-padded to `nsamp_padded`)
- Batch size: `ndm` (process all output DMs)
- Stride: `nsamp_padded` floats between DMs
- Note: The FFT length is `nsamp_fft` (not `nsamp_padded`), so only the first `nsamp_fft` time samples are computed, the remaining padded region is undefined

**Copy output (trimmed to valid region)**

Extract the valid output region, discarding samples corrupted by dispersion delay edge effects.

Copy only the first `nsamps_computed` valid samples from each DM:

```
copy_data(ndm, nsamps_computed, nsamp_padded, nsamps_computed, data_dm, output)
```

- Input: `data_dm[dm * nsamp_padded][time]`, stride=`nsamp_padded`
- Output: `output[dm * nsamps_computed][time]`, stride=`nsamps_computed`

**Why only `nsamps_computed`?**

- We keep the earliest `nsamps_computed = nsamps - max_delay` samples from each DM output row
- We drop the last `max_delay` samples, which depend on data beyond the observed time window
- This avoids edge effects from delay alignment near the end of the buffer

**Buffer visualisation**

This view shows one row of each 2D buffer (one channel or one DM), how wide it is in samples/elements, and which part is consumed by each step.

Note: bars are schematic (not exact scale); they show the relative used vs padded/discarded regions.

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

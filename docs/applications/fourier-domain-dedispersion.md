# Fourier-Domain Dedispersion (dedisp)

## Background

Beamformed radio data are commonly searched for dispersed transient and periodic signals such as pulsars and fast radio bursts (FRBs). As a radio pulse propagates through the ionised interstellar medium, lower radio frequencies travel more slowly than higher frequencies. This causes a frequency-dependent arrival-time delay that smears the signal across the observing band.

Dedispersion corrects for this effect by realigning the signal across frequency channels for a set of trial dispersion measures (DMs). In practical search pipelines, beamformed data are therefore processed for many DM values to recover sharp pulses and improve detectability.

### Beamformed input data

The input to dedispersion is typically beamformed filterbank data: a two-dimensional array with one time series per frequency channel. Each sample represents detected power in a given channel and time bin. Incoherent dedispersion operates on these detected intensities rather than on the original complex voltages.

For a trial DM, the relative delay between two frequencies scales approximately as:

$$
\Delta t \propto \mathrm{DM} \left( \nu_1^{-2} - \nu_2^{-2} \right)
$$

so lower-frequency channels must be shifted by larger amounts than higher-frequency channels.

### Time-domain and Fourier-domain dedispersion

Traditional incoherent dedispersion is performed in the time domain. For each trial DM, every frequency channel is shifted by the appropriate number of samples and the aligned channels are summed. This is straightforward, but when many DMs must be evaluated it becomes compute-intensive and often memory-bandwidth limited.

Fourier-Domain Dedispersion (FDD) performs the same alignment in the Fourier domain. Instead of shifting channel time series in time, it Fourier-transforms them and applies the corresponding delay as a phase rotation. This increases arithmetic intensity and makes the algorithm better suited to modern accelerators such as GPUs.

## Fourier-Domain Dedispersion (FDD)

FDD is a brute-force incoherent dedispersion algorithm for beamformed data. It corrects dispersion delays by applying phase rotations to Fourier-transformed time-series data, rather than by shifting samples in the time domain.

A shift in time corresponds to a phase rotation in the Fourier domain. If a channel time series $x_c[t]$ is transformed to $X_c[k]$, then applying a delay $\tau_c$ for channel $c$ amounts to multiplying each Fourier bin by a complex phasor:

$$
X'\_c[k] = X_c[k] e^{-2\pi i k \tau_c / N}
$$

where $N$ is the transform length and $\tau_c$ is the delay for the current trial DM relative to a reference frequency. After this correction, the frequency channels are aligned and can be summed to form the dedispersed result.

### Algorithm

For each block of beamformed data, FDD proceeds in three steps:

1. FFT the time series of every frequency channel.
1. For each trial DM, compute the delay per channel and apply the corresponding phase rotation in the Fourier domain.
1. Sum the corrected channels to form a dedispersed output for that DM.

In pseudocode, the core computation looks like this:

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

If a search pipeline ultimately needs time-domain output, an inverse FFT can be applied after summation. However, for FFT-based periodicity searches this step can be omitted and the dedispersed data can remain in the Fourier domain.

The main motivation for FDD is performance. Time-domain dedispersion performs relatively little computation per byte moved and is therefore often limited by memory bandwidth. FDD moves more of the work into arithmetic operations on Fourier-domain data, making it more compute-dense. This makes FDD competitive with and, for large DM counts, faster than optimised time-domain dedispersion on GPUs.

## PACE simplifications

The PACE reference application focuses on the Fourier-domain dedispersion algorithm itself rather than on the full real-world dedisp pipeline. The main simplifications are:

- Only Fourier-domain dedispersion is considered. The time-domain dedispersion algorithm from dedisp is out of scope.
- The focus is the dedispersion computation on beamformed data, not the broader end-to-end pulsar or transient search pipeline.

## References

- Bassa, C. G., Romein, J. W., Veenboer, B., van der Vlugt, S., Wijnholds, S. J. (2022). Fourier-domain dedispersion. A&A 657, A46.
- ASTRON dedisp repository. https://git.astron.nl/RD/dedisp

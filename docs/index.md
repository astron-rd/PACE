# Introduction

PACE provides simplified reference applications from the radio astronomy domain. The goal is not to reproduce full production pipelines, but to capture the key computational patterns that make these applications interesting for evaluating programming languages and parallelisation or acceleration strategies.

The current PACE applications are:

- Image-Domain Gridding (IDG)\
  A simplified radio interferometric imaging kernel that maps visibilities onto subgrids and forms an image through gridding and FFTs.
- All-sky Imaging (LOFTY)\
  A wide-field imaging application for simple yet efficient all-sky radio image formation.
- Fourier-Domain Dedispersion (FDD/dedisp)\
  A reference application for pulsar and transient search workflows, focused on correcting frequency-dependent dispersion efficiently in the Fourier domain.

______________________________________________________________________

# Image-Domain Gridding

## Background

Radio telescopes detect electromagnetic waves from sources in the universe and use them to construct sky maps showing the positions, intensity, and polarization of those sources. Modern radio telescopes are typically aperture synthesis arrays: collections of receivers whose signals are combined to synthesise a large effective aperture. Examples include LOFAR and SKA1-Low (dipole antenna arrays, ~100 MHz) and the VLA, MeerKAT, and SKA1-Mid (dish arrays, $\ge$ 1 GHz).

### Visibilities and the measurement equation

Each pair of receivers in the array is called a baseline. The receivers measure two orthogonal polarizations (X and Y). Correlating the signals of a receiver pair `(p, q)` over a short integration interval produces a visibility: one 2x2 complex-valued measurement covering all four polarization combinations (XX, XY, YX, YY).

The relationship between visibilities and the sky brightness distribution $B(\ell, m)$ is given by the measurement equation:

$$V\_{pq} = \int_l \int_m \frac{1}{n}, A_p(\ell,m), B(\ell,m), A_q^H(\ell,m); e^{-2\pi i(u\_{pq}\ell + v\_{pq}m + w\_{pq}(n-1))}, d\ell, dm$$

where $\ell, m$ are direction cosines of sky coordinates, $n = \sqrt{1 - \ell^2 - m^2}$, and $A_p(\ell,m),, A_q(\ell,m) \in \mathbb{C}^{2 \times 2}$ describe direction-dependent effects (DDEs) per receiver.

Each visibility has an associated $(u, v, w)$-coordinate determined by receiver positions relative to the observed sky. As the Earth rotates, the $(u, v, w)$-coordinates change continuously, so each baseline traces a track through UV-space. This is known as earth-rotation synthesis. Denser UV coverage generally yields a higher-quality image.

### Imaging pipeline

Creating a sky image involves three main steps:

1. **Correlation**: digitised signals from all receiver pairs are correlated to produce visibilities.
1. **Calibration**: instrumental gain errors are estimated and corrected. Errors are classified as:
   - *Direction-independent effects* (DIEs) - e.g. antenna beam gain; corrected once after calibration.
   - *Direction-dependent effects* (DDEs / A-terms) - e.g. ionospheric variations; time-varying and must be corrected during imaging.
1. **Imaging**: visibilities are converted into a sky image. This step is iterated in major cycles: each cycle consists of gridding (inversion), deconvolution (CLEAN), and degridding (prediction). The CLEAN algorithm iteratively detects peaks in the image (minor cycles) and subtracts them, building up a source model.

## Traditional gridding approaches

A direct non-uniform Fourier transform scales as $O(N\_\text{vis} \cdot N\_\text{pix}^2)$, which is prohibitively expensive for modern arrays. Gridding algorithms project non-uniform visibilities onto a regular grid so that a standard FFT can be applied instead.

### W-projection

W-projection corrects for the $w$-term by convolving each visibility with a 2D kernel before adding it to the UV grid. An inverse FFT then produces the image. The kernels depend solely on the $w$-coordinate. For large $w$-values (long baselines), kernel support grows substantially, making the approach memory-intensive. W-stacking and W-snapshots are extensions that limit kernel support by sorting visibilities into $w$-layers.

### AW-projection

AW-projection extends W-projection to also correct for DDEs (A-terms). Kernels now additionally depend on time, frequency, and baseline. They must be precomputed on an oversampled grid. The resulting data structure scales quadratically with both kernel size and oversampling factor. The high cost of computing and storing these kernels is the main scalability bottleneck of AW-projection.

## Image-Domain Gridding (IDG)

IDG was developed to correct for both W-terms and A-terms without computing large oversampled convolution kernels. The key insight is the **convolution theorem**:

$$\mathcal{F}{f \ast g} = \mathcal{F}{f} \cdot \mathcal{F}{g}$$

This allows the convolution in the Fourier domain to be replaced by a pixel-wise multiplication in the image domain, followed by a Fourier transform. This eliminates the need for oversampled kernels entirely. Both W-correction and A-correction are applied in the image domain.

### Subgrids

The core data structure is the subgrid: a small $\bar{N} \times \bar{N}$ tile (typically $\bar{N} = 32$) representing a low-resolution patch of the sky brightness for a localised subset of visibilities. Rather than convolving each visibility with a large kernel, IDG accumulates visibilities onto subgrids via direct summation, then applies corrections pixel-wise before Fourier-transforming the result onto the full grid.

### Gridding

Gridding proceeds in three steps:

1. **Gridder kernel**: for each subgrid $s$, visibilities are summed into subgrid pixels using a direct DFT:

   ```
   for each pixel i in NbarxNbar:
     offset = compute_offset(s, i)
     for each time step t:
       index = compute_index(s, i, t)
       for each channel c:
         phase  = offset - index x wavenumber[c]
         phasor = exp(i.phase)
         subgrid[p][i] += phasor x visibility[t][c][p]
   apply_aterm(subgrid)   // A-term correction in image domain
   apply_taper(subgrid)   // suppress aliasing
   ```

1. **Inverse FFT**: each subgrid is Fourier-transformed (4 x $\bar{N} \times \bar{N}$ FFTs, one per polarization) to bring it into the Fourier domain.

1. **Adder kernel**: the Fourier-domain subgrid is accumulated into the corresponding region of the full $N \times N$ grid.

**Degridding** (predicting visibilities from a model image) reverses these steps: a splitter kernel extracts a subgrid from the grid, a forward FFT converts it to the image domain, and the degridder kernel computes visibilities by direct summation over subgrid pixels.

### Execution plan

Before gridding or degridding, a greedy execution plan partitions all visibilities $V = V_1 \cup V_2 \cup \cdots \cup V_n$ into tasks. Each task consists of a subgrid $S_j$ (position in the full grid + metadata) and its associated visibilities $V_j$ (with $(u,v,w)$-coordinates). The plan determines:

- **Subgrid position**: covers a contiguous set of visibilities from one baseline and the surrounding AW-kernel support.
- **$\bar{T}$**: number of consecutive time steps on a subgrid; extended until the next step's kernel support falls outside the subgrid.
- **$\bar{C}$**: maximum frequency channels per subgrid; excess channels are split into groups mapped to separate subgrids.
- **One baseline per subgrid**: visibilities from different baselines are placed on separate subgrids, enabling per-baseline A-term correction.
- **$\bar{T} \leq \bar{T}\_\text{max}$** - an architecture-specific cap on time steps per subgrid for load balancing.

The full set of tasks (the work) is subdivided into subsets and processed in parallel by the gridder/degridder kernels. This work-division hierarchy maps naturally onto parallel architectures such as multi-core CPUs and GPUs.

### Complexity

The direct DFT per subgrid is feasible because $\bar{N} \ll N$ - subgrids are orders of magnitude smaller than the full grid. The minimum subgrid size is $\bar{N} \geq N_W$ (the W-kernel support), but larger subgrids (e.g. $\bar{N} = 32$) are used in practice so that multiple visibilities share the FFT cost.

## PACE simplifications

The PACE reference application implements a simplified version of IDG to focus on the core computational patterns relevant for benchmarking programming languages and parallelization/acceleration techniques. The following simplifications are made with respect to full IDG:

- **Gridding only**: only the gridder kernel and the final inverse FFT of the full grid are implemented. Degridding (the splitter, FFT, and degridder kernel) is omitted.
- **No A-term correction**: the `apply_aterm` step is omitted entirely. Subgrids are Fourier-transformed and added to the grid without any direction-dependent correction.
- **Fixed visibilities per subgrid**: the execution plan uses a fixed number of visibilities per subgrid ($\bar{T}$ and $\bar{C}$ are constant), rather than the greedy, baseline-adaptive partitioning of full IDG.
- **Stokes I only**: only a single total-intensity image is produced, instead of the four polarization images (XX, XY, YX, YY) of full IDG.

## References

- van der Tol, S., Veenboer, B., Offringa, A.R. (2018). *Image Domain Gridding: a fast method for convolutional resampling of visibilities.* A&A 616, A27.
- Bhatnagar, S., Cornwell, T.J., Golap, K., Uson, J.M. (2008). *Correcting direction-dependent gains in the deconvolution of radio interferometric images.* A&A 487, 419-429.
- Cornwell, T.J., Golap, K., Bhatnagar, S. (2008). *The noncoplanar baselines effect in radio interferometry.* IEEE J. Sel. Topics Signal Process. 2(5), 647-657.
- Offringa, A.R. et al. (2014). *WSClean: an implementation of a fast, generic wide-field imager for radio astronomy.* MNRAS 444, 606-619.
- ASTRON idg repository. https://git.astron.nl/RD/idg

______________________________________________________________________

# All-sky Imaging (LOFTY)

## Background

All-sky imaging is used to form wide-field radio images from interferometric measurements. For low-frequency aperture arrays such as LOFAR stations, the field of view can be very large, so imaging is not restricted to a small patch around a single phase centre. Instead, the goal is to reconstruct the sky brightness over a large fraction of the visible hemisphere.

In this setting, the measured data are complex visibilities: cross-correlations between pairs of receivers or antenna elements. Each receiver pair forms a baseline, and each baseline contributes a phase term that depends on its geometric separation and on the sky direction being imaged.

### Visibilities and image formation

For a given observing frequency, a visibility can be related to a sky direction $(\ell, m, n)$ through a phase factor of the form

$$
e^{-2\pi i \nu (u\ell + vm + w(n-1))/c}
$$

where $(u,v,w)$ are the baseline coordinates, $\nu$ is the observing frequency, $c$ is the speed of light, and

$$
n = \sqrt{1 - \ell^2 - m^2}
$$

An image pixel is formed by combining the contributions from all baselines for that direction.

Because this page focuses on the computational kernel, it is useful to think of all-sky imaging as a direct mapping from:

- visibilities,
- baseline coordinates,
- observing frequency,
- and a grid of image coordinates,

to a two-dimensional image.

______________________________________________________________________

## The LOFTY imaging algorithm

LOFTY provides tools for working with LOFAR station statistics products, including XST-based imaging. For all-sky imaging, the relevant inputs are a visibility matrix, the corresponding baseline coordinates, and the observing frequency. The implementation then evaluates the imaging equation directly on an $(\ell, m)$ image grid.

### Image grid

The image is sampled on a regular grid in direction cosines $(\ell, m)$, typically covering the square $[-1, 1] \times [-1, 1]$. Only points satisfying

$$
\ell^2 + m^2 < 1
$$

correspond to valid sky directions, so pixels outside this radius are masked or left undefined.

### Direct all-sky imaging

For each image pixel, LOFTY computes the phase for each baseline and combines the visibility contributions. In the simplest form implemented in the code base, the image is computed as the mean over all baselines:

$$
I(\ell, m) = \Re \left\langle V\_{pq} ; e^{-2\pi i \nu (u\_{pq}\ell + v\_{pq}m + w\_{pq}(n-1))/c} \right\rangle\_{p,q}
$$

This is a direct imaging approach rather than a gridding-and-FFT pipeline.

### Algorithm

At a high level, the all-sky imaging kernel proceeds as follows:

1. Build an $(\ell, m)$ image grid.
1. Compute $n = \sqrt{1 - \ell^2 - m^2}$ for valid sky pixels.
1. For each baseline, form the phase term from $(u,v,w)$, frequency, and sky direction.
1. Multiply the visibility by the corresponding complex phasor.
1. Average or sum the contributions from all baselines.
1. Take the real part to obtain the image.

In pseudocode, the core computation looks like this:

```
for each image pixel (l, m):
	if l*l + m*m >= 1:
		image[l, m] = invalid
		continue

	n = sqrt(1 - l*l - m*m)
	acc = 0

	for each baseline (p, q):
		phase = -2pi * freq * (u[p,q]*l + v[p,q]*m + w[p,q]*(n - 1)) / c
		phasor = exp(i * phase)
		acc += visibility[p,q] * phasor

	image[l, m] = real(acc)
```

## Inputs and outputs

In the LOFTY codebase, all-sky imaging is driven by XST data products. The imaging entry point reads the relevant data, optionally applies calibration, and then prepares an image for a selected integration and subband. The resulting output is a two-dimensional all-sky image, typically for Stokes I.

The same tooling also supports near-field imaging and plotting, but the core all-sky imaging kernel is the direct visibility-to-image computation described above.

## PACE simplifications

Compared with the other PACE applications, LOFTY requires relatively few simplifications because the core imaging code is already compact and close to the computational kernel of interest.

The main point for PACE is therefore not to redesign the algorithm, but to isolate the all-sky imaging computation as a reference workload for comparing programming languages and parallelisation or acceleration strategies.

## References

- ASTRON LOFTY repository. https://git.astron.nl/bassa/lofty

______________________________________________________________________

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

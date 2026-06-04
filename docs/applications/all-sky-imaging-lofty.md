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


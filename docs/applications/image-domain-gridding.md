# Image-Domain Gridding

## Background

Radio telescopes detect electromagnetic waves from sources in the universe and
use them to construct sky maps showing the positions, intensity, and
polarization of those sources. Modern radio telescopes are typically aperture
synthesis arrays: collections of receivers whose signals are combined to
synthesise a large effective aperture. Examples include LOFAR and SKA1-Low
(dipole antenna arrays, ~100 MHz) and the VLA, MeerKAT, and SKA1-Mid (dish
arrays, $\\ge$ 1 GHz).

### Visibilities and the measurement equation

Each pair of receivers in the array is called a baseline. The receivers measure
two orthogonal polarizations (X and Y). Correlating the signals of a receiver
pair `(p, q)` over a short integration interval produces a visibility: one 2x2
complex-valued measurement covering all four polarization combinations (XX, XY,
YX, YY).

The relationship between visibilities and the sky brightness distribution
$B(\\ell, m)$ is given by the measurement equation:

$$V\_{pq} = \\int_l \\int_m \\frac{1}{n}, A_p(\\ell,m), B(\\ell,m),
A_q^H(\\ell,m); e^{-2\\pi i(u\_{pq}\\ell + v\_{pq}m + w\_{pq}(n-1))}, d\\ell,
dm$$

where $\\ell, m$ are direction cosines of sky coordinates, $n = \\sqrt{1 -
\\ell^2 - m^2}$, and $A_p(\\ell,m),, A_q(\\ell,m) \\in \\mathbb{C}^{2 \\times
2}$ describe direction-dependent effects (DDEs) per receiver.

Each visibility has an associated $(u, v, w)$-coordinate determined by receiver
positions relative to the observed sky. As the Earth rotates, the $(u, v,
w)$-coordinates change continuously, so each baseline traces a track through
UV-space. This is known as earth-rotation synthesis. Denser UV coverage
generally yields a higher-quality image.

### Imaging pipeline

Creating a sky image involves three main steps:

1. **Correlation**: digitised signals from all receiver pairs are correlated to
   produce visibilities.
1. **Calibration**: instrumental gain errors are estimated and corrected. Errors
   are classified as:
   - *Direction-independent effects* (DIEs) - e.g. antenna beam gain; corrected
     once after calibration.
   - *Direction-dependent effects* (DDEs / A-terms) - e.g. ionospheric
     variations; time-varying and must be corrected during imaging.
1. **Imaging**: visibilities are converted into a sky image. This step is
   iterated in major cycles: each cycle consists of gridding (inversion),
   deconvolution (CLEAN), and degridding (prediction). The CLEAN algorithm
   iteratively detects peaks in the image (minor cycles) and subtracts them,
   building up a source model.

## Traditional gridding approaches

A direct non-uniform Fourier transform scales as $O(N\_\\text{vis} \\cdot
N\_\\text{pix}^2)$, which is prohibitively expensive for modern arrays. Gridding
algorithms project non-uniform visibilities onto a regular grid so that a
standard FFT can be applied instead.

### W-projection

W-projection corrects for the $w$-term by convolving each visibility with a 2D
kernel before adding it to the UV grid. An inverse FFT then produces the image.
The kernels depend solely on the $w$-coordinate. For large $w$-values (long
baselines), kernel support grows substantially, making the approach
memory-intensive. W-stacking and W-snapshots are extensions that limit kernel
support by sorting visibilities into $w$-layers.

### AW-projection

AW-projection extends W-projection to also correct for DDEs (A-terms). Kernels
now additionally depend on time, frequency, and baseline. They must be
precomputed on an oversampled grid. The resulting data structure scales
quadratically with both kernel size and oversampling factor. The high cost of
computing and storing these kernels is the main scalability bottleneck of
AW-projection.

## Image-Domain Gridding (IDG)

IDG was developed to correct for both W-terms and A-terms without computing
large oversampled convolution kernels. The key insight is the **convolution
theorem**:

$$\\mathcal{F}{f \\ast g} = \\mathcal{F}{f} \\cdot \\mathcal{F}{g}$$

This allows the convolution in the Fourier domain to be replaced by a pixel-wise
multiplication in the image domain, followed by a Fourier transform. This
eliminates the need for oversampled kernels entirely. Both W-correction and
A-correction are applied in the image domain.

### Subgrids

The core data structure is the subgrid: a small $\\bar{N} \\times \\bar{N}$ tile
(typically $\\bar{N} = 32$) representing a low-resolution patch of the sky
brightness for a localised subset of visibilities. Rather than convolving each
visibility with a large kernel, IDG accumulates visibilities onto subgrids via
direct summation, then applies corrections pixel-wise before
Fourier-transforming the result onto the full grid.

### Gridding

Gridding proceeds in three steps:

1. **Gridder kernel**: for each subgrid $s$, visibilities are summed into
   subgrid pixels using a direct DFT:

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

1. **Inverse FFT**: each subgrid is Fourier-transformed (4 x $\\bar{N} \\times
   \\bar{N}$ FFTs, one per polarization) to bring it into the Fourier domain.

1. **Adder kernel**: the Fourier-domain subgrid is accumulated into the
   corresponding region of the full $N \\times N$ grid.

**Degridding** (predicting visibilities from a model image) reverses these
steps: a splitter kernel extracts a subgrid from the grid, a forward FFT
converts it to the image domain, and the degridder kernel computes visibilities
by direct summation over subgrid pixels.

### Execution plan

Before gridding or degridding, a greedy execution plan partitions all
visibilities $V = V_1 \\cup V_2 \\cup \\cdots \\cup V_n$ into tasks. Each task
consists of a subgrid $S_j$ (position in the full grid + metadata) and its
associated visibilities $V_j$ (with $(u,v,w)$-coordinates). The plan determines:

- **Subgrid position**: covers a contiguous set of visibilities from one
  baseline and the surrounding AW-kernel support.
- **$\\bar{T}$**: number of consecutive time steps on a subgrid; extended until
  the next step's kernel support falls outside the subgrid.
- **$\\bar{C}$**: maximum frequency channels per subgrid; excess channels are
  split into groups mapped to separate subgrids.
- **One baseline per subgrid**: visibilities from different baselines are placed
  on separate subgrids, enabling per-baseline A-term correction.
- **$\\bar{T} \\leq \\bar{T}\_\\text{max}$** - an architecture-specific cap on
  time steps per subgrid for load balancing.

The full set of tasks (the work) is subdivided into subsets and processed in
parallel by the gridder/degridder kernels. This work-division hierarchy maps
naturally onto parallel architectures such as multi-core CPUs and GPUs.

### Complexity

The direct DFT per subgrid is feasible because $\\bar{N} \\ll N$ - subgrids are
orders of magnitude smaller than the full grid. The minimum subgrid size is
$\\bar{N} \\geq N_W$ (the W-kernel support), but larger subgrids (e.g. $\\bar{N}
= 32$) are used in practice so that multiple visibilities share the FFT cost.

## PACE simplifications

The PACE reference application implements a simplified version of IDG to focus
on the core computational patterns relevant for benchmarking programming
languages and parallelization/acceleration techniques. The following
simplifications are made with respect to full IDG:

- **Gridding only**: only the gridder kernel and the final inverse FFT of the
  full grid are implemented. Degridding (the splitter, FFT, and degridder
  kernel) is omitted.
- **No A-term correction**: the `apply_aterm` step is omitted entirely. Subgrids
  are Fourier-transformed and added to the grid without any direction-dependent
  correction.
- **Fixed visibilities per subgrid**: the execution plan uses a fixed number of
  visibilities per subgrid ($\\bar{T}$ and $\\bar{C}$ are constant), rather than
  the greedy, baseline-adaptive partitioning of full IDG.
- **Stokes I only**: only a single total-intensity image is produced, instead of
  the four polarization images (XX, XY, YX, YY) of full IDG.

## References

- van der Tol, S., Veenboer, B., Offringa, A.R. (2018). *Image Domain Gridding:
  a fast method for convolutional resampling of visibilities.* A&A 616, A27.
- Bhatnagar, S., Cornwell, T.J., Golap, K., Uson, J.M. (2008). *Correcting
  direction-dependent gains in the deconvolution of radio interferometric
  images.* A&A 487, 419-429.
- Cornwell, T.J., Golap, K., Bhatnagar, S. (2008). *The noncoplanar baselines
  effect in radio interferometry.* IEEE J. Sel. Topics Signal Process. 2(5),
  647-657.
- Offringa, A.R. et al. (2014). *WSClean: an implementation of a fast, generic
  wide-field imager for radio astronomy.* MNRAS 444, 606-619.
- ASTRON idg repository. https://git.astron.nl/RD/idg

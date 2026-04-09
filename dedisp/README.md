# Fourier Domain Dedispersion: the PACE Context

This document aims to be a short primer on (Fourier domain) dedispersion and
related concepts that might be useful for understanding the (reference) code.

## Concepts

Some useful papers and online resources:

- https://arxiv.org/abs/2110.03482
- https://arxiv.org/abs/2007.02886
- https://casper.berkeley.edu/astrobaki/index.php/Dispersion_measure
- https://astronomy.swin.edu.au/cosmos/\*/Pulsar+Dispersion+Measure

### Dispersion

Due to the varying ISM along the line-of-sight to a (radio) transient, e.g. a
pulsar, the signal is dispersed as a function of frequency. This means that
there's a time delay between the signal when it's observed at $\\nu$ compared to
say a reference frequency $\\nu_0$, how much of a time delay is described by the
following equation:

$$\\Delta
t(\\nu,,\\text{DM})=\\text{DM},\\kappa\_\\text{DM}\\left(\\nu^{-2}-\\nu_0^{-2}\\right),$$

where $\\kappa\_\\text{DM}$ is a proportionality constant and $\\text{DM}$ is
the *dispersion measure*, defined as the path integral over the electron density
along the line-of-sight:

$$\\text{DM}=\\int n\_\\text{e}(\\ell),\\text{d}\\ell.$$

### Time-domain dedispersion

Imagine an astronomer has a Stokes I spectrum (an incoherent spectrum):
$I(t,\\nu)$, to dedisperse this signal we apply a time delay $\\Delta
t(\\nu,,\\text{DM})$ according to the equation above.

To obtain the highest signal-to-noise ratio (S/N or SNR), we sum over all the
channels such that the resultant spectrum is given by:

$$I(t,\\text{DM})=\\sum\_\\nu I(t-\\Delta t(\\nu,,\\text{DM}),,\\nu),$$

since the DM is most often unknown, we typically repeat this for 100-1000 of
trial DMs.

**Note on performance:** the TDD approach requires significant bandwidth, while
only requiring very small computations while memory is aligned.

### Fourier-domain dedispersion

Instead of applying time delays, we could also apply phase shifts in
frequency-space. In that manner the time delay $\\Delta t(\\nu,,\\text{DM})$ can
be applied as a phasor.

We Fourier-transform each of the channels (or observation frequencies) $\\nu$ to
obtain the intensity as a function of the spin-frequency $f_s$, which is
associated to the periodic signal of, e.g. the pulsar:

$$I(f_s,\\nu)=\\mathcal{F}\_{t\\rightarrow f_s}\\left{I(t,\\nu)\\right}=\\int
I(t,\\nu),\\text{e}^{-2\\pi i f_s t},\\text{dt}.$$

In this space, we can apply the time delay as a phase rotation:

$$\\mathcal{W}(f_s,\\nu,\\text{DM})=\\exp\\left(-2\\pi i f_s \\Delta
t(\\nu,,\\text{DM})\\right),$$

Such that the de-dispersed signal is recovered by:

$$I(t,\\text{DM})=\\sum\_\\nu I(t-\\Delta t(\\nu,,\\text{DM}),,\\nu).$$

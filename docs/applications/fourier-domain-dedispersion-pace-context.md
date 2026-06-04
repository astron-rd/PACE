# Fourier Domain Dedispersion: the PACE Context

This document aims to be a short primer on Fourier-domain dedispersion and related concepts that might be useful for understanding the reference code.

## Concepts

Some useful papers and online resources:

- https://arxiv.org/abs/2110.03482
- https://arxiv.org/abs/2007.02886
- https://casper.berkeley.edu/astrobaki/index.php/Dispersion_measure
- https://astronomy.swin.edu.au/cosmos/*/Pulsar+Dispersion+Measure

### Dispersion

Due to the varying ISM along the line-of-sight to a radio transient, e.g. a pulsar, the signal is dispersed as a function of frequency. This means there is a time delay between the signal when it is observed at $\nu$ compared to a reference frequency $\nu_0$. The amount of time delay is described by:

$$
\Delta t(\nu, \mathrm{DM}) = \mathrm{DM}\,\kappa_{\mathrm{DM}}\left(\nu^{-2} - \nu_0^{-2}\right)
$$

where $\kappa_{\mathrm{DM}}$ is a proportionality constant and $\mathrm{DM}$ is the dispersion measure, defined as the path integral over the electron density along the line-of-sight:

$$
\mathrm{DM} = \int n_e(\ell)\,\mathrm{d}\ell
$$

### Time-domain dedispersion

Imagine an astronomer has a Stokes I spectrum (an incoherent spectrum), $I(t, \nu)$. To dedisperse this signal, we apply a time delay $\Delta t(\nu, \mathrm{DM})$ according to the equation above.

To obtain the highest signal-to-noise ratio (S/N or SNR), we sum over all channels such that the resultant spectrum is given by:

$$
I(t, \mathrm{DM}) = \sum_{\nu} I\left(t - \Delta t(\nu, \mathrm{DM}), \nu\right)
$$

Since the DM is most often unknown, we typically repeat this for 100-1000 trial DMs.

Note on performance: the TDD approach requires significant bandwidth while requiring relatively little computation per memory operation.

### Fourier-domain dedispersion

Instead of applying time delays directly, we can also apply phase shifts in frequency-space. In that manner, the time delay $\Delta t(\nu, \mathrm{DM})$ can be applied as a phasor.

We Fourier-transform each channel (or observation frequency) $\nu$ to obtain the intensity as a function of spin frequency $f_s$, which is associated with the periodic signal of, e.g. the pulsar:

$$
I(f_s, \nu) = \mathcal{F}_{t \rightarrow f_s}\{I(t, \nu)\} = \int I(t, \nu)\,e^{-2\pi i f_s t}\,\mathrm{d}t
$$

In this space, we can apply the time delay as a phase rotation:

$$
\mathcal{W}(f_s, \nu, \mathrm{DM}) = \exp\left(-2\pi i f_s\,\Delta t(\nu, \mathrm{DM})\right)
$$

Such that the dedispersed signal is recovered by:

$$
I(t, \mathrm{DM}) = \sum_{\nu} I\left(t - \Delta t(\nu, \mathrm{DM}), \nu\right)
$$

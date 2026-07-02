"""
Expected kernel output =
{{ 3.000000e+00+0.000000e+00i, -5.960464e-08+2.414214e+00i, -1.000000e+00-5.960464e-08i, -5.960464e-08+4.142137e-01i, -1.000000e+00+6.357301e-08i},
 { 3.000000e+00+0.000000e+00i, -3.596266e-01+2.270593e+00i, -6.535003e-01-2.123351e-01i, -3.201909e-01+6.284094e-01i, 7.298245e-01-5.302490e-01i}}

   [[ 3.0000000e+00+0.0000000e+00j  2.2204460e-16+2.4142137e+00j -1.0000000e+00+2.2204460e-16j  0.0000000e+00+4.1421357e-01j  f-1.0000000e+00+2.4492937e-16j]
   [ 3.0000000e+00+0.0000000e+00j -3.5962659e-01+2.2705929e+00j -6.5350050e-01-2.1233518e-01j -3.2019058e-01+6.2840939e-01j  d-7.2982478e-01-5.3024876e-01j]]
"""

import numpy as np

from fdd.kernels import fourier_domain_dedisperse


def test_fdd_kernel():
    n_samples = 8
    n_channels = 3
    n_dms = 2
    time_res = 0.1

    n_spin = n_samples // 2 + 1
    n_fft_bins = n_samples // 2 + 1

    dm_list = np.array([10.0, 11.0])
    print("dm_list =", dm_list)

    delay_table = 1 / time_res * (np.arange(0, n_channels) + 1)
    print("delay_table =", delay_table)

    spin_table = np.arange(0, n_spin) * (1.0 / (n_samples * time_res) / 100)
    print("spin_table =", spin_table)

    input_data = np.ones((n_channels, n_fft_bins), dtype=complex)
    print("\nkernel input =\n", input_data)

    output_data = np.zeros((n_dms, n_fft_bins), dtype=complex)
    fourier_domain_dedisperse(
        input_data, output_data, time_res, spin_table, dm_list, delay_table
    )
    print("\nKernel output =\n", output_data)

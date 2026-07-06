import numpy as np

from fdd.kernels import fourier_domain_dedisperse, vectorized_fourier_domain_dedisperse

# This is the result from dedisp/cpp/bin/testroutines.cpp.
CPP_KERNEL_RESULT = np.array(
    [
        [
            3.000000e00 + 0.000000e00j,
            -5.960464e-08 + 2.414214e00j,
            -1.000000e00 - 5.960464e-08j,
            -5.960464e-08 + 4.142137e-01j,
            -1.000000e00 + 6.357301e-08j,
        ],
        [
            3.000000e00 + 0.000000e00j,
            -3.596266e-01 + 2.270593e00j,
            -6.535003e-01 - 2.123351e-01j,
            -3.201909e-01 + 6.284094e-01j,
            -7.298245e-01 - 5.302490e-01j,
        ],
    ]
)


def test_fdd_kernel():
    """
    Verify the output of the FDD kernel by comparing it to the output of the C++ implementation.
    """

    n_samples = 8
    n_channels = 3
    n_dms = 2
    time_res = 0.1

    n_spin = n_samples // 2 + 1
    n_fft_bins = n_samples // 2 + 1

    dm_list = np.array([10.0, 11.0])
    delay_table = 1 / time_res * (np.arange(0, n_channels) + 1)
    spin_table = np.arange(0, n_spin) * (1.0 / (n_samples * time_res) / 100)
    input_data = np.ones((n_channels, n_fft_bins), dtype=complex)

    output_data = np.zeros((n_dms, n_fft_bins), dtype=complex)
    fourier_domain_dedisperse(
        input_data, output_data, time_res, spin_table, dm_list, delay_table
    )

    assert np.allclose(output_data, CPP_KERNEL_RESULT)


def test_vectorized_fdd_kernel():
    """
    Verify the output of the vectorized FDD kernel by comparing it to the output of the C++ implementation.
    """
    n_samples = 8
    n_channels = 3
    n_dms = 2
    time_res = 0.1

    n_spin = n_samples // 2 + 1
    n_fft_bins = n_samples // 2 + 1

    dm_list = np.array([10.0, 11.0])
    delay_table = 1 / time_res * (np.arange(0, n_channels) + 1)
    spin_table = np.arange(0, n_spin) * (1.0 / (n_samples * time_res) / 100)
    input_data = np.ones((n_channels, n_fft_bins), dtype=complex)

    output_data = np.zeros((n_dms, n_fft_bins), dtype=complex)
    vectorized_fourier_domain_dedisperse(
        input_data, output_data, time_res, spin_table, dm_list, delay_table
    )

    assert np.allclose(output_data, CPP_KERNEL_RESULT)

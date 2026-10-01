import numpy as np


def fourier_domain_dedisperse(
    input_data: np.ndarray,
    output_data: np.ndarray,
    time_resolution: float,
    spin_frequencies: np.ndarray,
    dispersion_measures: np.ndarray,
    delays: np.ndarray,
) -> None:
    """
    Executes the FDD kernel.

    :param input_data: complex floats with shape (channels, fft bins)
    :param output_data: complex floats with shape (DMs, spin frequencies)
    :param time_resolution: observation integration time in seconds
    :param spin_frequencies: spin frequency table
    :param dispersion_measures: trial dispersion measures
    :param delays: delays per channel
    """
    n_spin_frequencies = spin_frequencies.size
    samples = input_data[:, :n_spin_frequencies]

    for dm_index, dm in enumerate(dispersion_measures):
        dm_delays = dm * delays * time_resolution

        phases = 2.0 * np.pi * np.outer(dm_delays, spin_frequencies)
        phasors = np.exp(1j * phases)

        output_data[dm_index, :n_spin_frequencies] = np.sum(samples * phasors, axis=0)


def fourier_domain_dedisperse_vectorized(
    input_data: np.ndarray,
    output_data: np.ndarray,
    time_resolution: float,
    spin_frequencies: np.ndarray,
    dispersion_measures: np.ndarray,
    delays: np.ndarray,
) -> None:
    """
    Executes a fully vectorised FDD kernel.

    :param input_data: complex floats with shape (channels, fft bins)
    :param output_data: complex floats with shape (DMs, spin frequencies)
    :param time_resolution: observation integration time in seconds
    :param spin_frequencies: spin frequency table
    :param dispersion_measures: trial dispersion measures
    :param delays: delays per channel
    """
    n_spin_frequencies = spin_frequencies.size
    samples = input_data[:, :n_spin_frequencies]

    # Expand all arrays for broadcasting
    delays_expanded = delays[np.newaxis, :, np.newaxis]  # (1, n_channels, 1)
    spin_frequencies_expanded = spin_frequencies[
        np.newaxis, np.newaxis, :
    ]  # (1, 1, n_spin_frequencies)
    dms_expanded = dispersion_measures[:, np.newaxis, np.newaxis]  # (n_dm, 1, 1)
    samples_expanded = samples[np.newaxis, :, :]

    # Compute all phases at once; has shape (n_dms, n_channels, n_spin_frequencies)
    phases = (
        2.0
        * np.pi
        * dms_expanded
        * delays_expanded
        * time_resolution
        * spin_frequencies_expanded
    )
    phasors = np.exp(1j * phases)

    output_data[:, :n_spin_frequencies] = np.sum(samples_expanded * phasors, axis=1)

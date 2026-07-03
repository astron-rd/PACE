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

    :param input_data: complex floats with shape (...)
    :param output_data: complex floats with shape (...)
    :param time_resolution: observation integration time in seconds
    :param spin_frequencies: spin frequency table
    :param dispersion_measures: trial dispersion measures
    :param delays: delays per channel
    """
    n_spin_frequencies = spin_frequencies.size
    samples = input_data[:, :n_spin_frequencies]

    # TODO: could also completely vectorize...!
    # Caveat: might require a lot of memory? Is that smart...?
    for dm_index, dm in enumerate(dispersion_measures):
        dm_delays = dm * delays * time_resolution

        phases = 2.0 * np.pi * np.outer(dm_delays, spin_frequencies)
        phasors = np.exp(1j * phases)

        output_data[dm_index, :n_spin_frequencies] = np.sum(samples * phasors, axis=0)

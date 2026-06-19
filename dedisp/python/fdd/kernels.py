import numpy as np


def fourier_domain_dedisperse(
    input_data: np.ndarray,
    dm_count: int,
    n_frequencies: int,
    n_channels: int,
    time_resolution: float,
    spin_frequencies: np.ndarray,
    dispersion_measures: np.ndarray,
    delays: np.ndarray,
) -> np.ndarray:
    output_data = np.zeros((dm_count, n_frequencies))
    for dm_index, dm in enumerate(dispersion_measures):
        dm_delays = dm * delays * time_resolution

        for frequency_index, spin_frequency in enumerate(spin_frequencies):
            complex_sum = np.sum(
                [
                    input_data[channel_index, frequency_index]
                    * np.exp(2.0j * np.pi * spin_frequency * dm_delays[channel_index])
                    for channel_index in range(0, n_channels)
                ]
            )  # TODO: rename when I'm not suffering from a heat stroke

            output_data[dm_index, frequency_index] = complex_sum

    return output_data

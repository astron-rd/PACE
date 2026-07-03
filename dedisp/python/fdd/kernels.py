import numpy as np


def fourier_domain_dedisperse(
    input_data: np.ndarray,
    output_data: np.ndarray,
    time_resolution: float,
    spin_frequencies: np.ndarray,
    dispersion_measures: np.ndarray,
    delays: np.ndarray,
) -> np.ndarray:
    # TODO: could also completely vectorize...!
    # Caveat: might require a lot of memory? Is that smart...?
    n_spin_frequencies = spin_frequencies.size
    print(f"running the dispersion kernel for {dispersion_measures.size} DMs:", end="")
    for dm_index, dm in enumerate(dispersion_measures):
        print(f" {dm_index}", end="")
        dm_delays = dm * delays * time_resolution

        phases = (
            2.0 * np.pi * spin_frequencies[:, np.newaxis] * dm_delays[np.newaxis, :]
        )
        phasors = np.exp(1j * phases)

        samples = input_data[:, :n_spin_frequencies].T

        output_data[dm_index, :n_spin_frequencies] = np.sum(samples * phasors, axis=1)

    print(";")

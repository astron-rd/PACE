import numpy as np

import initializers
from config import settings


def main():
    uvw = initializers.init_uvw(
        observation_hours=settings.observation_hours,
        nr_baselines=settings.nr_baselines,
        grid_size=settings.grid_size,
    )
    frequencies = initializers.init_frequencies(
        start_frequency=settings.start_frequency,
        frequency_increment=settings.frequency_increment,
        nr_channels=settings.nr_channels,
    )
    metadata = initializers.init_metadata(
        nr_channels=settings.nr_channels,
        subgrid_size=settings.subgrid_size,
        grid_size=settings.grid_size,
        uvw=uvw,
    )
    visibilities = initializers.init_visibilities(
        nr_correlations=settings.nr_correlations_in,
        nr_channels=settings.nr_channels,
        nr_timesteps=settings.nr_timesteps,
        nr_baselines=settings.nr_baselines,
        image_size=settings.image_size,
        grid_size=settings.grid_size,
        frequencies=frequencies,
        uvw=uvw,
    )

    if settings.output_npy:
        np.save("uvw.npy", uvw)
        np.save("frequencies.npy", frequencies)
        np.save("metadata.npy", metadata)
        np.save("visibilities.npy", visibilities)
    else:
        np.savez(
            "artifact.npz",
            uvw=uvw,
            frequencies=frequencies,
            metadata=metadata,
            visibilities=visibilities,
        )


if __name__ == "__main__":
    main()

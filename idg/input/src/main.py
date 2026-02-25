import initializer
from config import settings
from serializer import Serializer


def main():
    targets = {}

    targets["uvw"] = initializer.init_uvw(
        observation_hours=settings.observation_hours,
        nr_baselines=settings.nr_baselines,
        grid_size=settings.grid_size,
    )
    targets["frequencies"] = initializer.init_frequencies(
        start_frequency=settings.start_frequency,
        frequency_increment=settings.frequency_increment,
        nr_channels=settings.nr_channels,
    )
    targets["metadata"] = initializer.init_metadata(
        nr_channels=settings.nr_channels,
        subgrid_size=settings.subgrid_size,
        grid_size=settings.grid_size,
        uvw=targets["uvw"],
    )
    targets["visibilities"] = initializer.init_visibilities(
        nr_correlations=settings.nr_correlations_in,
        nr_channels=settings.nr_channels,
        nr_timesteps=settings.nr_timesteps,
        nr_baselines=settings.nr_baselines,
        image_size=settings.image_size,
        grid_size=settings.grid_size,
        frequencies=targets["frequencies"],
        uvw=targets["uvw"],
    )

    serializer = Serializer()
    for name, array in targets.items():
        serializer.save(name, array)


if __name__ == "__main__":
    main()

import constants
import initializer
from serializer import Serializer


def main():
    targets = {}

    targets["uvw"] = initializer.init_uvw(
        observation_hours=constants.OBSERVATION_HOURS,
        nr_baselines=constants.NR_BASELINES,
        grid_size=constants.GRID_SIZE,
    )
    targets["frequencies"] = initializer.init_frequencies(
        start_frequency=constants.START_FREQUENCY,
        frequency_increment=constants.FREQUENCY_INCREMENT,
        nr_channels=constants.NR_CHANNELS,
    )
    targets["metadata"] = initializer.init_metadata(
        nr_channels=constants.NR_CHANNELS,
        subgrid_size=constants.SUBGRID_SIZE,
        grid_size=constants.GRID_SIZE,
        uvw=targets["uvw"],
    )

    serializer = Serializer()
    for name, array in targets.items():
        serializer.save(name, array)


if __name__ == "__main__":
    main()

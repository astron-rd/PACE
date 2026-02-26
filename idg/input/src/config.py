import argparse
from pydantic import computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Constants
    nr_correlations_in: int = 2  # XX, XY
    nr_correlations_out: int = 1  # I
    w_step: float = 1.0  # w step in wavelengths
    start_frequency: float = 150e6  # 150 MHz
    frequency_increment: float = 1e6  # 1 MHz
    speed_of_light: float = 299792458.0

    # Arguments
    subgrid_size: int = 32  # size of each subgrid
    grid_size: int = 1024  # size of the full grid
    observation_hours: int = 4  # total observation time in hours
    nr_channels: int = 16  # number of frequency channels
    nr_stations: int = 20

    # Computed fields
    @computed_field
    @property
    def nr_timesteps(self) -> int:
        return self.observation_hours * 3600

    @computed_field
    @property
    def end_frequency(self) -> float:
        return self.start_frequency + (self.nr_channels - 1) * self.frequency_increment

    @computed_field
    @property
    def image_size(self) -> float:
        return self.speed_of_light / self.end_frequency

    @computed_field
    @property
    def nr_baselines(self) -> int:
        return self.nr_stations * (self.nr_stations - 1) // 2


settings = Settings()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--subgrid_size",
    type=int,
    default=settings.subgrid_size,
    help="Size of the subgrid in pixels",
)
parser.add_argument(
    "--grid_size",
    type=int,
    default=settings.grid_size,
    help="Size of the grid in pixels",
)
parser.add_argument(
    "--observation_hours",
    type=float,
    default=settings.observation_hours,
    help="Length of the observation in hours",
)
parser.add_argument(
    "--nr_channels",
    type=int,
    default=settings.nr_channels,
    help="Number of frequency channels",
)
parser.add_argument(
    "--nr_stations",
    type=int,
    default=settings.nr_stations,
    help="Number of stations",
)

args = parser.parse_args()

settings.subgrid_size = args.subgrid_size
settings.grid_size = args.grid_size
settings.observation_hours = args.observation_hours
settings.nr_channels = args.nr_channels
settings.nr_stations = args.nr_stations

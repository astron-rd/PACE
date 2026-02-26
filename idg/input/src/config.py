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
    observation_hours: float = 4.0  # total observation time in hours
    nr_channels: int = 16  # number of frequency channels
    nr_stations: int = 20

    # Computed fields
    @computed_field
    @property
    def nr_timesteps(self) -> int:
        return int(self.observation_hours * 3600)

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

    @classmethod
    def from_args(cls) -> "Settings":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--subgrid_size",
            type=int,
            default=cls.model_fields["subgrid_size"].default,
            help="Size of the subgrid in pixels",
        )
        parser.add_argument(
            "--grid_size",
            type=int,
            default=cls.model_fields["grid_size"].default,
            help="Size of the grid in pixels",
        )
        parser.add_argument(
            "--observation_hours",
            type=float,
            default=cls.model_fields["observation_hours"].default,
            help="Length of the observation in hours",
        )
        parser.add_argument(
            "--nr_channels",
            type=int,
            default=cls.model_fields["nr_channels"].default,
            help="Number of frequency channels",
        )
        parser.add_argument(
            "--nr_stations",
            type=int,
            default=cls.model_fields["nr_stations"].default,
            help="Number of stations",
        )

        return cls(**vars(parser.parse_args()))


settings = Settings.from_args()

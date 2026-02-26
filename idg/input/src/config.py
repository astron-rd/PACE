import argparse
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Configuration settings for input data generation process.
    Fields with descriptions are exposed as command-line arguments.
    """

    # Constants
    nr_correlations_in: int = 2  # XX, XY
    nr_correlations_out: int = 1  # I
    w_step: float = 1.0  # w step in wavelengths
    start_frequency: float = 150e6  # 150 MHz
    frequency_increment: float = 1e6  # 1 MHz
    speed_of_light: float = 299792458.0

    # Arguments
    nr_stations: int = Field(20, description="Number of stations")
    nr_channels: int = Field(16, description="Number of frequency channels")
    grid_size: int = Field(1024, description="Size of the grid in pixels")
    subgrid_size: int = Field(32, description="Size of the subgrid in pixels")
    observation_hours: float = Field(4.0, description="Observation length in hours")

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
        for name, field in cls.model_fields.items():
            if not field.description:
                continue

            parser.add_argument(
                f"--{name}",
                type=field.annotation or type(field.default),
                default=field.default,
                help=field.description,
            )

        return cls(**vars(parser.parse_args()))


settings = Settings.from_args()

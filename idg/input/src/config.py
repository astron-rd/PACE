from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path.cwd() / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE)

    nr_correlations_in: int
    nr_correlations_out: int
    subgrid_size: int
    grid_size: int
    observation_hours: int
    nr_channels: int
    w_step: float
    start_frequency: float
    frequency_increment: float
    speed_of_light: float
    nr_stations: int

    @computed_field
    @property
    def nr_timesteps(self) -> int:
        return self.observation_hours * 3600

    @computed_field
    @property
    def end_frequency(self) -> float:
        return self.start_frequency + self.nr_channels * self.frequency_increment

    @computed_field
    @property
    def image_size(self) -> float:
        return self.speed_of_light / self.end_frequency

    @computed_field
    @property
    def nr_baselines(self) -> int:
        return self.nr_stations * (self.nr_stations - 1) // 2


settings = Settings.model_validate({})

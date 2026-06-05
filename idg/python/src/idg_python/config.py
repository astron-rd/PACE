import argparse
from typing import ClassVar

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Configuration settings for the IDG gridding process."""

    # Constants
    nr_correlations_out: ClassVar[int] = 1  # I
    w_step: ClassVar[float] = 1.0  # w step in wavelengths
    speed_of_light: ClassVar[float] = 299792458.0

    # Arguments
    input: str = Field(description="Input HDF5 file")
    store: bool = Field(False, description="Store data in HDF5 format")
    json_output: str | None = Field(
        None, description="Output timings in JSON format (optional: specify filename)"
    )

    @classmethod
    def from_args(cls) -> "Settings":
        parser = argparse.ArgumentParser()
        parser.add_argument("input", help="Input HDF5 file")
        parser.add_argument(
            "--store",
            action="store_true",
            default=False,
            help="Store data in HDF5 format",
        )
        parser.add_argument(
            "--json",
            dest="json_output",
            metavar="JSON",
            const="timings.json",
            nargs="?",
            help="Output timings in JSON format (optional: specify filename)",
        )
        return cls(**vars(parser.parse_args()))


settings = Settings.from_args()

import argparse


class Settings:
    """Configuration settings for the IDG gridding process."""

    nr_correlations_out = 1  # I
    w_step = 1.0  # w step in wavelengths
    speed_of_light = 299792458.0

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("input", help="Input HDF5 file")
        parser.add_argument(
            "--store", action="store_true", help="Store data in HDF5 format"
        )
        parser.add_argument(
            "--json",
            dest="json_output",
            metavar="JSON",
            const="timings.json",
            nargs="?",
            help="Output timings in JSON format (optional: specify filename)",
        )
        parser.parse_args(namespace=self)


settings = Settings()

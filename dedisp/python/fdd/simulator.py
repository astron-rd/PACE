import argparse

import numpy as np


class Simulator:
    def __init__(
        self,
        observation_duration: float,
        time_resolution: float,
        n_channels: int,
        bandwidth: float,
        peak_frequency: float,
        noise_rms: float,
        intensity: float,
        arrival_time: float,
        dispersion_measure: float,
    ):
        self.n_samples = int(observation_duration / time_resolution)
        self.n_channels = n_channels

        self.time_resolution = time_resolution
        self.frequency_resolution = -1.0 * bandwidth / n_channels

        self.peak_frequency = peak_frequency
        self.noise_rms = noise_rms
        self.intensity = intensity
        self.arrival_time = arrival_time
        self.dm = dispersion_measure

    def generate(self, quantise: bool = False, random_seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(random_seed)
        data = rng.normal(
            loc=0.0, scale=self.noise_rms, size=(self.n_samples, self.n_channels)
        )

        channel_indices = np.arange(self.n_channels)
        channel_frequencies = (
            self.peak_frequency + channel_indices * self.frequency_resolution
        )
        inverse_channel_frequencies_squared = 1 / channel_frequencies**2
        inverse_peak_frequency_squared = 1.0 / self.peak_frequency**2

        delay_constant = 4.15e3
        channel_delays = (
            self.dm
            * delay_constant
            * (inverse_channel_frequencies_squared - inverse_peak_frequency_squared)
        )
        sample_indices = (
            (self.arrival_time + channel_delays) / self.time_resolution
        ).astype(int)

        data[sample_indices, channel_indices] += self.intensity

        if quantise:
            # Shift data to the 8-bit range and clip if any value is outside of the [0, 255] range
            shifted_data = data + 127.5

            return np.clip(shifted_data, a_min=0.0, a_max=255.0).astype(np.uint8)

        return data

    def to_hdf5(self, data: np.ndarray, filename: str):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--duration", type=float, default=30.0, help="Observation duration in seconds"
    )
    parser.add_argument(
        "--timeresolution",
        type=float,
        default=250e-6,
        help="Time resolution of the data in seconds",
    )
    parser.add_argument("--channels", type=int, default=1024, help="Number of channels")
    parser.add_argument(
        "--bandwidth", type=float, default=100.0, help="Observation bandwidth in MHz"
    )
    parser.add_argument(
        "--peakfrequency",
        type=float,
        default=1581.0,
        help="Maximum observing frequency",
    )
    parser.add_argument(
        "--noiserms", type=float, default=25.0, help="Background noise level"
    )
    parser.add_argument(
        "--intensity", type=float, default=25.0, help="Signal intensity"
    )
    parser.add_argument(
        "--arrivaltime",
        type=float,
        default=3.14159,
        help="Signal arrival time in seconds",
    )
    parser.add_argument("--dm", type=float, default=41.159, help="Dispersion measure")
    parser.add_argument(
        "--quantise", action="store_true", default=False, help="Quantise the output"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to generate the background noise according to a normal distribution",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="signal.h5",
        help="Filename for the HDF5 dataset containing the simulated signal",
    )
    args = parser.parse_args()

    print(args)

    sim = Simulator(
        args.duration,
        args.timeresolution,
        args.channels,
        args.bandwidth,
        args.peakfrequency,
        args.noiserms,
        args.intensity,
        args.arrivaltime,
        args.dm,
    )

    data = sim.generate(quantise=args.quantise, random_seed=args.seed)

    if args.file:
        print(f"Writing the simulated signal disk: {args.file}")
        sim.to_hdf5(data, args.file)

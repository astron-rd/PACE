import argparse
import logging

import h5py
import numpy as np

logger = logging.getLogger(__name__)


class Signal:
    """
    Class used to simulate a dispersed signal on top of background noise
    and load the signal from a HDF5 into memory.
    """

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
    ) -> None:
        """
        Initialises a Signal object.

        :param observation_duration: duration in seconds
        :param time_resolution: sample integration time in seconds
        :param n_channels: number of channels
        :param bandwidth: frequency range in MHz
        :param peak_frequency: maximum frequency in MHz
        :param noise_rms: background noise level
        :param intensity: signal intensity
        :param arrival_time: pulse arrival time in seconds
        :param dispersion_measure: the dispersion measure associated to the signal
        """
        self.n_samples = int(observation_duration // time_resolution)
        self.n_channels = n_channels

        self.time_resolution = time_resolution
        self.frequency_resolution = bandwidth / n_channels

        self.peak_frequency = peak_frequency
        self.noise_rms = noise_rms
        self.intensity = intensity
        self.arrival_time = arrival_time
        self.dm = dispersion_measure

        self.dynamic_spectrum = None

    def simulate(
        self,
        quantise: bool = False,
        random_seed: int = 0,
    ) -> np.ndarray:
        """
        Generate background and inject a pulse

        :param quantise: duration in seconds
        :param random_seed: sample integration time in seconds
        :returns: dynamic spectrum with shape (samples, channels)
        """
        rng = np.random.default_rng(random_seed)
        data = rng.normal(
            loc=0.0, scale=self.noise_rms, size=(self.n_samples, self.n_channels)
        )

        channel_indices = np.arange(self.n_channels)
        channel_frequencies = self.peak_frequency - channel_indices * abs(
            self.frequency_resolution
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

            self.dynamic_spectrum = np.clip(
                shifted_data, a_min=0.0, a_max=255.0
            ).astype(np.uint8)
        else:
            self.dynamic_spectrum = data

        return self.dynamic_spectrum

    def to_hdf5(self, filename: str) -> None:
        """
        Dump the simulated dynamic spectrum to disk as a HDF5 file.

        :param filename: HDF5 file name
        """
        if self.dynamic_spectrum is None:
            raise RuntimeError("There's no dynamic spectrum to write to HDF5.")

        with h5py.File(filename, "w") as output_file:
            dyn_spec = output_file.create_dataset("dynspec", data=self.dynamic_spectrum)

            # Properties of the dynamic spectrum
            dyn_spec.attrs["samples"] = self.n_samples
            dyn_spec.attrs["channels"] = self.n_channels
            dyn_spec.attrs["integration_time"] = self.time_resolution
            dyn_spec.attrs["channel_width"] = self.frequency_resolution
            dyn_spec.attrs["peak_frequency"] = self.peak_frequency

            # Meta data used to generate the mock signal
            output_file.attrs["noise_rms"] = self.noise_rms
            output_file.attrs["intensity"] = self.intensity
            output_file.attrs["dispersion_measure"] = self.dm
            output_file.attrs["arrival_time"] = self.arrival_time

    def set_dynamic_spectrum(self, dynamic_spectrum: np.ndarray) -> None:
        """
        Set the dynamic spectrum.

        :param dynamic_spectrum: dynamic spectrum with shape (samples, channels)
        """
        expected_shape = (self.n_samples, self.n_channels)
        if dynamic_spectrum.shape != expected_shape:
            raise RuntimeError(
                f"The dynamic spectrum has the incompatible dimensions: expected {expected_shape}, but got {dynamic_spectrum.shape}"
            )

        self.dynamic_spectrum = dynamic_spectrum

    @classmethod
    def from_hdf5(cls, filename: str):
        """
        Read a dynamic spectrum from disk to construct a Signal object.

        :param filename: file name of the dynamic spectrum stored as HDF5
        :returns: a new Signal object
        """
        with h5py.File(filename, "r") as input_file:
            dyn_spec_ds = input_file["dynspec"]
            if not isinstance(dyn_spec_ds, h5py.Dataset):
                raise RuntimeError("Invalid input file: dynamic spectrum not found.")

            # Properties of the dynamic spectrum
            n_samples = dyn_spec_ds.attrs["samples"]
            n_channels = dyn_spec_ds.attrs["channels"]
            time_resolution = dyn_spec_ds.attrs["integration_time"]
            frequency_resolution = dyn_spec_ds.attrs["channel_width"]
            peak_frequency = dyn_spec_ds.attrs["peak_frequency"]

            # Meta data used to generate the mock signal
            noise_rms = input_file.attrs["noise_rms"]
            intensity = input_file.attrs["intensity"]
            dm = input_file.attrs["dispersion_measure"]
            arrival_time = input_file.attrs["arrival_time"]

            # Create a new Signal object and update the dynamic spectrum
            signal_from_hdf5 = cls(
                time_resolution * n_samples,
                time_resolution,
                n_channels,
                frequency_resolution * n_channels,
                peak_frequency,
                noise_rms,
                intensity,
                arrival_time,
                dm,
            )
            signal_from_hdf5.set_dynamic_spectrum(dyn_spec_ds[...])

            return signal_from_hdf5


def main():
    """Entry of 'fdd-sim'."""
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
        "--togglequantisation", action="store_false", help="Toggle quantisation off"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to generate the background noise according to a normal distribution",
    )
    parser.add_argument(
        "--log-level",
        default=logging.INFO,
        choices=logging.getLevelNamesMapping().keys(),
        help="Display timing results",
    )
    parser.add_argument(
        "file",
        type=str,
        help="Filename for the HDF5 dataset containing the simulated signal",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    signal = Signal(
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

    signal.simulate(quantise=args.togglequantisation, random_seed=args.seed)

    logger.info(" writing the simulated signal to disk: %s", args.file)
    signal.to_hdf5(args.file)

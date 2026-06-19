import numpy as np

from fdd.kernels import fourier_domain_dedisperse


class FDDPlan:
    def __init__(
        self,
        n_channels: int,
        time_resolution: float,
        peak_frequency: float,
        frequency_resolution: float,
    ):
        self.dm_count = None
        self.n_channels = n_channels
        self.max_delay = None

        self.time_resolution: float = time_resolution
        self.peak_frequency: float = peak_frequency
        self.frequency_resolution: float = frequency_resolution

        self.dm_table: np.ndarray | None = None
        self.delay_table: np.ndarray | None = None
        self.spin_frequency_table: np.ndarray | None = None

        self.generate_delay_table()

    def execute(self, spectrum: np.ndarray):
        """
        Execute the Fourier Domain Dedispersion algorithm.

        :param spectrum: quantised input spectrum, with shape (samples x channels)
        :return: array with shape (samples x DMs)
        """
        print("spectrum shape = {}".format(spectrum.shape))

        n_samples = spectrum.shape[0]
        n_spin_frequencies = n_samples // 2 + 1
        n_output_samples = n_samples - self.max_delay

        use_zero_padding = True
        n_samples_fft = (
            self.round_up(n_samples + 1, 16384) if use_zero_padding else n_samples
        )
        n_samples_padded = self.round_up(n_samples_fft + 1, 1024)

        print(f"padded samps = {n_samples_padded}")

        # 1. Generate spin table
        self.generate_spin_frequency_table(n_spin_frequencies, n_samples)

        # TODO: 2. Pad the spectrum and transpose the data (convert input bytes to floats)
        padding = n_samples_padded - n_samples
        padded_spectrum = np.pad(spectrum, [(0, padding), (0, 0)], mode="constant")

        byte_offset = 127.5
        transposed_spectrum = self.transpose_data(
            padded_spectrum, byte_offset, self.n_channels
        )

        print("transposed spectrum shape = {}".format(transposed_spectrum.shape))

        # TODO: 3. Real-to-complex FFT: time series data to frequency domain
        print()
        fd_scratch = np.fft.fft(transposed_spectrum, axis=1)

        # TODO: 4. Run dedispersion algorithm (CPU reference or optimised version)
        print(self.dm_table)
        dm_scratch = fourier_domain_dedisperse(
            fd_scratch,
            self.dm_count,
            n_spin_frequencies,
            self.n_channels,
            self.time_resolution,
            self.spin_frequency_table,
            self.dm_table,
            self.delay_table,
        )  # output has shape: DMs x samples
        print("dm_scratch shape = {}".format(dm_scratch.shape))

        # TODO: 5. Complex-to-real FFT: frequency domain back to time series data
        dm_data = np.fft.ifft(dm_scratch, axis=1)
        print("dm_data shape = {}".format(dm_data.shape))

        # 6. Only return n_output_samples samples and transpose the array to match the expected shape (samples x DMs)
        computed_samples = dm_data[:, :n_output_samples].T
        print(
            "computed_samples shape = {} / output samples = {}".format(
                computed_samples.shape, n_output_samples
            )
        )
        return computed_samples

    def generate_dm_list(
        self, dm_start: float, dm_end: float, pulse_width: float, tolerance: float
    ):
        """
        Generate a list of DMs in a linear fashion.

        :param dm_start: first DM value in the interval
        :param dm_end: upper bound of the DM values
        :param pulse_width: ...
        :param tolerance: ...
        """
        pass

    def generate_linear_dm_list(
        self, dm_start: float, dm_end: float, dm_step: float
    ) -> np.ndarray:
        """
        Generate a list of DMs in a linear fashion.

        :param dm_start: first DM value in the interval
        :param dm_end: end of the DM value interval
        :param dm_step: DM step size
        :return: array of trial DMs
        """
        dm_list = np.arange(dm_start, dm_end, dm_step)

        self.dm_table = dm_list
        self.dm_count = dm_list.size
        self.max_delay = int(dm_list[-1] * self.delay_table[-1] + 0.5)

        return dm_list

    def generate_delay_table(self) -> None:
        """
        Calculate the ...
        """
        channel_indices = np.arange(0, self.n_channels)
        inverse_channel_frequency = 1.0 / (
            self.peak_frequency + channel_indices * self.frequency_resolution
        )
        inverse_peak_frequency = 1.0 / self.peak_frequency

        # TODO: document...
        delay_constant = 4.148741601e3

        self.delay_table = (
            delay_constant
            / self.time_resolution
            * (
                inverse_channel_frequency * inverse_channel_frequency
                - inverse_peak_frequency * inverse_peak_frequency
            )
        )

    def generate_spin_frequency_table(
        self, n_spin_frequencies: int, n_samples: int
    ) -> None:
        """
        Initialize the spin frequency table.

        :param n_spin_frequencies: the number of spin frequency values associated with the observed signal
        :param n_samples: the number of input samples
        """
        spin_indices = np.arange(0, n_spin_frequencies)
        observation_duration = n_samples * self.time_resolution

        self.spin_frequency_table = spin_indices / observation_duration

    def show(self) -> None:
        """Display a summary of the FDD plan."""
        frequency_resolution = (
            -1.0 * self.frequency_resolution
        )  # negate since it's negative by definition
        delay_in_seconds = self.max_delay * self.time_resolution

        print(f"""
        FDD Plan Summary
          nr channels:          {self.n_channels}
          nr dm trials:         {self.dm_count}
          max delay:            {delay_in_seconds:.3f} s ({self.max_delay} samples)
          time resolution:      {self.time_resolution:.3f}
          frequency resolution: {frequency_resolution:.3f}
          peak fequency:        {self.peak_frequency:.3f}
        """)

    def transpose_data(self, data: np.ndarray, offset: float, scale: float):
        """
        Transpose and scale the data appropriately.

        :param offset: used to undo quantization, e.g. 128 for 8-bits
        :param scale: use this to prevent overflows when summing the data
        :return: transposed spectrum (with shape channels x samples)
        """
        return (data.T.astype(float) - offset) / scale

    def round_up(self, a: int, b: int):
        """Round up integer a to a multiple of integer b."""
        return ((a + b - 1) // b) * b

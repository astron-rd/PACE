import h5py
import numpy as np

from fdd.kernels import fourier_domain_dedisperse
from fdd.utilities import Timer


class FDDPlan:
    """
    Class that implements the Fourier Domain Dedispersion algorithm.
    """

    def __init__(
        self,
        n_channels: int,
        time_resolution: float,
        peak_frequency: float,
        frequency_resolution: float,
    ) -> None:
        """
        Initialises the Fourier Domain Dedispersion plan.

        :param n_channels: number of frequency channels used in the observation
        :param time_resolution: integration time in seconds
        :param peak_frequency: maximum channel frequency in MHz
        :param frequency_resolution: channel width in MHz
        """
        self.dm_count = None
        self.n_channels = n_channels
        self.max_delay = None

        self.time_resolution: float = time_resolution
        self.peak_frequency: float = peak_frequency
        self.frequency_resolution: float = -abs(frequency_resolution)

        self.dm_table: np.ndarray | None = None
        self.delay_table: np.ndarray | None = None
        self.spin_frequency_table: np.ndarray | None = None

        self.result: np.ndarray | None = None

        self.generate_delay_table()

    def execute(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Execute the Fourier Domain Dedispersion algorithm.

        :param spectrum: quantised input spectrum, with shape (samples x channels)
        :returns: array with shape (samples x DMs)
        """
        print("spectrum shape = {}".format(spectrum.shape))

        n_samples = spectrum.shape[0]
        n_spin_frequencies = n_samples // 2 + 1
        n_output_samples = n_samples - self.max_delay
        print(f"DEBUG: samples        = {n_samples}")
        print(f"DEBUG: spin freq.     = {n_spin_frequencies}")
        print(f"DEBUG: output samples = {n_output_samples}\n")

        use_zero_padding = True
        n_samples_fft = (
            self.round_up(n_samples + 1, 16384) if use_zero_padding else n_samples
        )
        n_samples_padded = self.round_up(n_samples_fft + 1, 1024)
        n_fft_frequency_bins = n_samples_padded // 2 + 1

        print(f"DEBUG: FFT samples   = {n_samples_fft}")
        print(f"DEBUG: padded samps  = {n_samples_padded}")
        print(f"DEBUG: FFT freq bins = {n_fft_frequency_bins}")

        init_timer = Timer()
        preprocessing_timer = Timer()
        dedispersion_timer = Timer()
        postprocessing_timer = Timer()
        output_timer = Timer()

        # 1. Generate spin table
        init_timer.start()

        self.generate_spin_frequency_table(n_spin_frequencies, n_samples)

        init_timer.pause()

        # 2. Pad the spectrum and transpose the data (convert input bytes to floats)
        preprocessing_timer.start()

        padding = n_samples_padded - n_samples
        print("DEBUG: padding = {}".format(padding))
        padded_spectrum = np.pad(spectrum, [(0, padding), (0, 0)], mode="constant")

        byte_offset = 127.5
        transposed_spectrum = self.transpose_data(
            padded_spectrum, byte_offset, self.n_channels
        )

        print("DEBUG: transposed spectrum shape = {}".format(transposed_spectrum.shape))

        # 3. Real-to-complex FFT: time series data to frequency domain
        fd_scratch = np.fft.rfft(transposed_spectrum, axis=1)
        print(
            "DEBUG: real-to-complex FFT output has shape (channels, FFT bins): ",
            fd_scratch.shape,
            "type = ",
            fd_scratch.dtype,
        )

        preprocessing_timer.pause()

        # 4. Run dedispersion algorithm (CPU reference or optimised version)
        init_timer.start()
        dm_scratch = np.zeros((self.dm_count, fd_scratch.shape[1]), dtype=complex)
        init_timer.pause()

        dedispersion_timer.start()
        fourier_domain_dedisperse(
            fd_scratch,
            dm_scratch,
            self.time_resolution,
            self.spin_frequency_table,
            self.dm_table,
            self.delay_table,
        )  # output has shape: DMs x samples
        dedispersion_timer.pause()
        print(
            "DEBUG: kernel output has shape (DMs, spin freq.): {}".format(
                dm_scratch.shape
            ),
            "type = ",
            dm_scratch.dtype,
        )

        # 5. Complex-to-real FFT: frequency domain back to time series data
        postprocessing_timer.start()
        dm_data = np.fft.irfft(dm_scratch, axis=1)
        postprocessing_timer.pause()
        print(
            "DEBUG: complex-to-real FFT output has shape (DMs, padded samples): {}".format(
                dm_data.shape
            )
        )

        # 6. Only return n_output_samples samples and transpose the array to match the expected shape (samples x DMs)
        output_timer.start()
        computed_samples = dm_data[:, :n_output_samples].T
        output_timer.pause()
        print(
            "DEBUG: computed_samples shape = {} / output samples = {}".format(
                computed_samples.shape, n_output_samples
            )
        )

        print(f"""
        Initialization time : {init_timer.duration():.6f} sec.
        Preprocessing time  : {preprocessing_timer.duration():.6f} sec.
        Dedispersion time   : {dedispersion_timer.duration():.6f} sec.
        Postprocessing time : {postprocessing_timer.duration():.6f} sec.
        Output copy time    : {output_timer.duration():.6f} sec.
        """)

        self.result = computed_samples

        return computed_samples

    def generate_dm_list(
        self, dm_start: float, dm_end: float, pulse_width: float, tolerance: float
    ) -> np.ndarray:
        """
        Generate a list of DMs in a linear fashion.

        :param dm_start: first DM value in the interval
        :param dm_end: upper bound of the DM values
        :param pulse_width: expected pulse width in milliseconds
        :param tolerance: smearing tolerance
        :returns: list of DMs
        """
        time_resolution = self.time_resolution * 1e6
        f = (
            self.peak_frequency
            + ((self.n_channels // 2) - 0.5) * self.frequency_resolution
        ) * 1e-3
        a = 8.3 * self.frequency_resolution / (f * f * f)
        a_squared = a**2
        b_squared = a_squared * (self.n_channels**2 / 16.0)
        tolerance_squared = tolerance**2
        c = (time_resolution**2 + pulse_width**2) * (tolerance_squared - 1.0)

        dm_list = [dm_start]
        while dm_list[-1] < dm_end:
            previous_dm = dm_list[-1]
            previous_dm_squared = previous_dm**2
            k = c + tolerance_squared * a_squared * previous_dm_squared
            dm = (
                b_squared * previous_dm
                + np.sqrt(
                    -a_squared * b_squared * previous_dm_squared
                    + (a_squared + b_squared) * k
                )
            ) / (a_squared + b_squared)

            dm_list.append(dm)

        self.dm_table = np.array(dm_list)
        self.dm_count = self.dm_table.size
        self.max_delay = int(dm_list[-1] * self.delay_table[-1] + 0.5)

        return self.dm_table

    def generate_linear_dm_list(
        self, dm_start: float, dm_end: float, dm_step: float
    ) -> np.ndarray:
        """
        Generate a list of DMs in a linear fashion.

        :param dm_start: first DM value in the interval
        :param dm_end: end of the DM value interval
        :param dm_step: DM step size
        :returns: array of trial DMs
        """
        dm_list = np.arange(dm_start, dm_end, dm_step)

        self.dm_table = dm_list
        self.dm_count = dm_list.size
        self.max_delay = int(dm_list[-1] * self.delay_table[-1] + 0.5)

        return dm_list

    def generate_delay_table(self) -> None:
        """
        Calculate the delay for each channel.
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

    def to_hdf5(self, filename: str) -> None:
        """
        Write the result of the Fourier Domain Dedispersion algorithm to disk in the HDF5 format.

        :param filename: HDF5 file name
        """
        if self.result is None:
            raise RuntimeError(
                "There's no results to write to HDF5. Please execute the plan."
            )

        with h5py.File(filename, "w") as output_file:
            fdd_result = output_file.create_dataset("fddresult", data=self.result)

            # Properties of the dynamic spectrum
            fdd_result.attrs["dispersion_measures"] = self.dm_table
            fdd_result.attrs["computed_samples"] = self.result.shape[0]
            fdd_result.attrs["integration_time"] = self.time_resolution

    def transpose_data(
        self, data: np.ndarray, offset: float, scale: float
    ) -> np.ndarray:
        """
        Transpose and scale the data appropriately.

        :param offset: used to undo quantization, e.g. 128 for 8-bits
        :param scale: use this to prevent overflows when summing the data
        :returns: transposed spectrum (with shape channels x samples)
        """
        return (data.T.astype(float) - offset) / scale

    def round_up(self, a: int, b: int) -> int:
        """Round up integer a to a multiple of integer b."""
        return ((a + b - 1) // b) * b

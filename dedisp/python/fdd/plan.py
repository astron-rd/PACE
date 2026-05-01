import numpy as np


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

        :param input: quantised input spectrum, shape (...)
        :return: ...
        """
        pass

    def generate_dm_list(
        self, dm_start: float, dm_end: float, pulse_width: float, tolerance: float
    ):
        pass

    def generate_linear_dm_list(
        self, dm_start: float, dm_end: float, dm_step: float
    ) -> np.ndarray:
        """
        Generate a list of DMs in a linear fashion.

        :param dm_start: first DM value in the interval
        :param dm_end: end of the DM value interval
        :param dm_step: DM step size
        """
        dm_list = np.arange(dm_start, dm_end, dm_step)

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

import numpy as np


def init_frequencies(
    start_frequency: float, frequency_increment: float, nr_channels: int
) -> np.ndarray:
    """
    Generate array of frequencies for each channel.

    :param start_frequency: Starting frequency in Hz
    :param frequency_increment: Increment in Hz between consecutive channels
    :param nr_channels: Number of frequency channels

    :return frequencies array, shape (nr_channels)
    """
    return start_frequency + frequency_increment * np.arange(nr_channels)

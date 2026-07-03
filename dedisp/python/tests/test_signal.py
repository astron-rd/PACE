import numpy as np

from fdd.signal import Signal


def test_signal_simulator():
    """
    Verify that the simulator runs without error and that the data is generated with the expected noise level.
    """
    duration = 30
    timeresolution = 250e-6
    channels = 1024
    bandwidth = 100.0
    peakfrequency = 1581.0
    noiserms = 25.0
    intensity = 25.0
    arrivaltime = 3.14159
    dm = 41.159

    sig = Signal(
        duration,
        timeresolution,
        channels,
        bandwidth,
        peakfrequency,
        noiserms,
        intensity,
        arrivaltime,
        dm,
    )

    data = sig.simulate(quantise=True)
    print(data.shape)

    assert np.isclose(data.std(), noiserms, rtol=1e-3)

    test_fn = "test_signal.h5"
    sig.to_hdf5(test_fn)

    # Try to load the signal from the generated HDF5 file and compare the contents.
    h5_sig = Signal.from_hdf5(test_fn)
    print(h5_sig.dynamic_spectrum.shape)

    assert np.allclose(data, h5_sig.dynamic_spectrum)
    assert np.isclose(h5_sig.n_channels, channels)
    assert np.isclose(h5_sig.noise_rms, noiserms)
    assert np.isclose(h5_sig.peak_frequency, peakfrequency)
    assert np.isclose(h5_sig.intensity, intensity)
    assert np.isclose(h5_sig.dm, dm)
    assert np.isclose(h5_sig.arrival_time, arrivaltime)
    assert np.isclose(h5_sig.time_resolution, timeresolution)

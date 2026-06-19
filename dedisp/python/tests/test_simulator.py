import numpy as np

from fdd.simulator import Simulator


def test_simulator():
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

    sim = Simulator(
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

    data = sim.generate(quantise=False)

    assert np.isclose(data.std(), noiserms, rtol=1e-3)

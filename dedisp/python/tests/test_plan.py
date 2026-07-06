import numpy as np

from fdd.plan import FDDPlan
from fdd.signal import Signal


def test_plan():
    """
    Verify the FDD plan with dummy data.
    """
    n_samples = 64
    n_channels = 16
    integration_time = 250e-6  # sec
    peak_frequency = 1581.0  # MHz
    channel_width = 6  # MHz
    plan = FDDPlan(n_channels, integration_time, peak_frequency, channel_width)

    dm_start = 40
    dm_end = 42
    dm_step = 0.1
    dm_list = plan.generate_linear_dm_list(dm_start, dm_end, dm_step)

    dynamic_spectrum = np.ones((n_samples, n_channels))

    output = plan.execute(dynamic_spectrum)

    n_computed_samples = n_samples - plan.max_delay
    assert output.shape[0] == n_computed_samples
    assert output.shape[1] == dm_list.size


def test_plan_with_simulated_spectrum():
    """
    Test FDD plan execution with a simulated dynamic spectrum.
    """
    # Simulate a dynamic spectrum with a dispersed signal
    duration = 12.0
    timeresolution = 250e-6
    channels = 64
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

    simulated_spectrum = sig.simulate(quantise=True)

    # n_samples = simulated_spectrum.shape[0]
    n_channels = simulated_spectrum.shape[1]
    channel_width = bandwidth / channels
    plan = FDDPlan(n_channels, timeresolution, peakfrequency, channel_width)

    dm_start = 30.0
    dm_end = 60.0
    pulse_width = 4.0
    dm_tolerance = 1.25
    dm_list = plan.generate_dm_list(dm_start, dm_end, pulse_width, dm_tolerance)

    output = plan.execute(simulated_spectrum)

    dm_matrix = np.repeat(dm_list[np.newaxis, :], output.shape[0], axis=0)

    assert output.shape == dm_matrix.shape
    assert np.isclose(dm_matrix.flat[output.argmax()], dm, rtol=0.1)

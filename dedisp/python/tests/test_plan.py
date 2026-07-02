import matplotlib.pyplot as plt
import numpy as np

from fdd.plan import FDDPlan
from fdd.simulator import Simulator


def test_plan():
    """ """
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

    plan.show()


def test_plan_with_simulated_spectrum():
    # Simulate a dynamic spectrum with a dispersed signal
    duration = 30.0
    timeresolution = 250e-6
    channels = 1024
    bandwidth = 100.0
    peakfrequency = 1581.0
    noiserms = 25.0
    intensity = 100.0
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

    simulated_spectrum = sim.generate(quantise=False)

    # n_samples = simulated_spectrum.shape[0]
    n_channels = simulated_spectrum.shape[1]
    channel_width = bandwidth / channels
    plan = FDDPlan(n_channels, timeresolution, peakfrequency, channel_width)

    dm_start = 2.0
    dm_end = 100.0
    pulse_width = 4.0
    dm_tolerance = 1.25
    plan.generate_dm_list(dm_start, dm_end, pulse_width, dm_tolerance)

    output = plan.execute(simulated_spectrum)

    # Plot the input
    dt = 0.05
    select_start = arrivaltime - dt
    select_end = arrivaltime + dt

    samp_start = int(select_start / timeresolution)
    samp_end = int(select_end / timeresolution)

    f_min = peakfrequency - bandwidth
    fig, frames = plt.subplots(
        2, 1, sharex=True, figsize=(8, 8), gridspec_kw=dict(height_ratios=[1, 1])
    )

    frames[0].imshow(
        simulated_spectrum[samp_start:samp_end, :].T,
        aspect="auto",
        extent=(samp_start, samp_end, f_min, peakfrequency),
    )

    # Plot the output
    frames[1].imshow(
        output[samp_start:samp_end, :].T,
        origin="lower",
        aspect="auto",
        extent=(samp_start, samp_end, dm_start, dm_end),
    )
    plt.show()

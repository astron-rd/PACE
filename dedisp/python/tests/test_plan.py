import numpy as np

from fdd.plan import FDDPlan


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

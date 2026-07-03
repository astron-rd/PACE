import argparse

from fdd.plan import FDDPlan
from fdd.signal import Signal


def execute_fdd_plan(
    signal: Signal,
    dm_start=0.0,
    dm_end=1.0,
    dm_step=0.5,
    dm_tolerance=None,
    pulse_width=None,
    filename: str | None = None,
):
    plan = FDDPlan(
        signal.n_channels,
        signal.time_resolution,
        signal.peak_frequency,
        signal.frequency_resolution,
    )

    dm_start = 2.0
    dm_end = 100.0
    pulse_width = 4.0
    dm_tolerance = 1.25

    if dm_tolerance is not None and pulse_width is not None:
        plan.generate_dm_list(dm_start, dm_end, pulse_width, dm_tolerance)
    else:
        plan.generate_linear_dm_list(dm_start, dm_end, dm_step)

    plan.execute(signal.dynamic_spectrum)

    if filename is not None:
        print(f"Writing the Fourier Domain Dedispersion result to disk: {filename}")
        plan.to_hdf5(filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "spectrum",
        type=str,
        help="Path to filterbank file (HDF5) that contains the output of 'fdd-sim'",
    )
    parser.add_argument(
        "--dm-start",
        type=float,
        default=0.0,
        help="Start of the dispersion measure search interval",
    )
    parser.add_argument(
        "--dm-end",
        type=float,
        default=1.0,
        help="End of the dispersion measure search interval",
    )
    parser.add_argument(
        "--dm-step", type=float, default=0.5, help="Dispersion measure stepsize"
    )
    parser.add_argument(
        "--dm-tolerance", type=float, default=1.25, help="Smearing tolerance"
    )
    parser.add_argument(
        "--pulse-width",
        type=float,
        default=4.0,
        help="Expected pulse width in milliseconds",
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Display timing results"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="fdd.h5",
        help="Filename for the HDF5 dataset containing the output of the dedispersion plan",
    )
    args = parser.parse_args()

    sig = Signal.from_hdf5(args.spectrum)

    execute_fdd_plan(
        sig,
        dm_start=args.dm_start,
        dm_end=args.dm_end,
        dm_step=args.dm_step,
        dm_tolerance=args.dm_tolerance,
        pulse_width=args.pulse_width,
        filename=args.file,
    )

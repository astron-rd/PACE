import argparse
import logging

from fdd.plan import FDDPlan
from fdd.signal import Signal

logger = logging.getLogger(__name__)


def execute_fdd_plan(
    signal: Signal,
    dm_start,
    dm_end,
    dm_tolerance,
    pulse_width,
    dm_step: float | None = None,
    filename: str | None = None,
):
    """
    Execute the Fourier Domain Dedispersion plan on a user-provided dynamic spectrum.

    :param signal: Signal object based on a user-prodived dynamic spectrum
    :param dm_start: first DM value on the search interval
    :param dm_start: final DM value on the search interval
    :param dm_step: linear DM step size
    :param dm_tolerance: smearing tolerance
    :param pulse_width: expected pulse width in milliseconds
    :param filename: name of the HDF5 file to write the result to
    """
    plan = FDDPlan(
        signal.n_channels,
        signal.time_resolution,
        signal.peak_frequency,
        signal.frequency_resolution,
    )

    if dm_step is not None:
        logger.warning(" be aware that you are sampling trial DMs on a linear interval")
        plan.generate_linear_dm_list(dm_start, dm_end, dm_step)
    else:
        plan.generate_dm_list(dm_start, dm_end, pulse_width, dm_tolerance)

    result = plan.execute(signal.dynamic_spectrum)

    if filename is not None:
        logger.info(
            " writing the Fourier Domain Dedispersion result to disk: %s", filename
        )
        plan.to_hdf5(result, filename)


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
        default=2.0,
        help="Start of the dispersion measure search interval",
    )
    parser.add_argument(
        "--dm-end",
        type=float,
        default=100.0,
        help="End of the dispersion measure search interval",
    )
    parser.add_argument("--dm-step", type=float, help="Dispersion measure stepsize")
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
        "--log-level",
        default=logging.INFO,
        choices=logging.getLevelNamesMapping().keys(),
        help="Display timing results",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="fdd.h5",
        help="Filename for the HDF5 dataset containing the output of the dedispersion plan",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    sig = Signal.from_hdf5(args.spectrum)

    execute_fdd_plan(
        sig,
        args.dm_start,
        args.dm_end,
        args.dm_tolerance,
        args.pulse_width,
        dm_step=args.dm_step,
        filename=args.file,
    )

import argparse
import logging

from fdd.plan import FDDPlan
from fdd.signal import Signal

logger = logging.getLogger(__name__)


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "spectrum",
        type=str,
        help="Path to filterbank file (HDF5) that contains the output of 'fdd-sim'",
        default="signal.h5",
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
    parser.add_argument("--dm-step", type=float,
                        help="Dispersion measure stepsize")
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

    return parser


def main():
    args = get_argument_parser().parse_args()

    logging.basicConfig(level=args.log_level)

    sig = Signal.from_hdf5(args.spectrum)

    plan = FDDPlan(
        sig.n_channels,
        sig.time_resolution,
        sig.peak_frequency,
        sig.frequency_resolution,
    )

    if args.dm_step is not None:
        logger.warning(
            " be aware that you are sampling trial DMs on a linear interval")
        plan.generate_linear_dm_list(args.dm_start, args.dm_end, args.dm_step)
    else:
        plan.generate_dm_list(args.dm_start, args.dm_end,
                              args.pulse_width, args.dm_tolerance)

    result = plan.execute(sig.dynamic_spectrum)

    if args.file is not None:
        logger.info(
            " writing the Fourier Domain Dedispersion result to disk: %s", args.file
        )
        plan.to_hdf5(result, args.file)

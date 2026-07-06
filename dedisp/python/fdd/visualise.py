import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np

from fdd.signal import Signal


def load_fdd_result(filename: str):
    """
    Load the FDD result from a HDF5 file.

    :param filename: name of the HDF5 file
    """
    with h5py.File(filename, "r") as input_file:
        fdd_result_ds = input_file["fddresult"]
        if not isinstance(fdd_result_ds, h5py.Dataset):
            raise Exception("Invalid input file: FDD result not found.")

        # Properties of the dynamic spectrum
        dm_list = fdd_result_ds.attrs["dispersion_measures"]
        computed_samples = fdd_result_ds.attrs["computed_samples"]
        time_resolution = fdd_result_ds.attrs["integration_time"]

        duration = computed_samples * time_resolution

        return fdd_result_ds[...], dm_list, duration


def plot_burst(
    dynspec_filename: str,
    result_filename: str,
    image_filename: str | None = None,
    zoom_width: float | None = None,
):
    """
    Plot the dedisped burst, dynamic spectrum (with the dispersed signal),
    and the DM search space. The plot is centered around the burst.

    :param dynspec_filename: name of the HDF5 file with the dynamic spectrum
    :param result_filename: name of the HDF5 file with the FDD result
    :param image_filename: name of the image to write the plot to (e.g. burst.png)
    :param zoom_width: duration (in seconds) of the observation to show, centered around the burst
    """
    signal = Signal.from_hdf5(dynspec_filename)
    result, dm_table, duration = load_fdd_result(result_filename)

    # Settings
    t_burst = signal.arrival_time
    t_samp = signal.time_resolution

    f_max = signal.peak_frequency
    chan_width = abs(signal.frequency_resolution)
    n_chans = signal.n_channels

    f_min = f_max - chan_width * n_chans

    if zoom_width:
        t_start = t_burst - zoom_width / 2
        t_end = t_burst + zoom_width / 2
    else:
        t_start = 0
        t_end = t_samp * signal.n_samples

    samp_start = int(t_start / t_samp)
    samp_end = int(t_end / t_samp)

    # Plot the burst and the trial DMs
    fig, frames = plt.subplots(
        3, 1, sharex=True, figsize=(8, 8), gridspec_kw=dict(height_ratios=[0.3, 1, 1])
    )

    # Plot the channel-averaged burst
    mean_intensity = np.mean(result[samp_start:samp_end, :], axis=1)
    time_axis = np.arange(samp_start, samp_end) * t_samp
    frames[0].plot(time_axis, mean_intensity, lw=1, color="black")

    # Plot the input data (samples, channel)
    frames[1].imshow(
        signal.dynamic_spectrum[samp_start:samp_end, :].T,
        aspect="auto",
        extent=(t_start, t_end, f_min, f_max),
    )

    # Plot the output trial DM space (samples, trial DMs)
    frames[2].imshow(
        result[samp_start:samp_end, :].T,
        origin="lower",
        aspect="auto",
        extent=(t_start, t_end, dm_table.min(), dm_table.max()),
    )

    # Axes settings
    frames[0].set_title(f"Signal with a DM of {signal.dm:.3f} at {t_burst:.5f} seconds")
    frames[0].set_ylabel("Mean intensity")
    frames[0].set_ylim(mean_intensity.mean() - 5, mean_intensity.mean() + 5)

    frames[1].axvline(t_burst, color="black", ls="--", lw=0.5)
    frames[1].set_ylabel("frequency (MHz)")

    frames[2].axvline(t_burst, color="black", ls="--", lw=0.5)
    frames[2].set_xlabel("Time from start  of the observation (s)")
    frames[2].set_ylabel(r"DM (pc cm$^{-3}$)")

    plt.tight_layout()

    if image_filename:
        fig.savefig(image_filename, dpi=300)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dynspec", type=str, help="Path to the dynamic spectrum (HDF5)"
    )
    parser.add_argument(
        "--result",
        type=str,
        help="Path to the FDD result (HDF5)",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Filename for the output image",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=0.1,
        help="Zoom-in on the specified number of seconds centered around the pulse",
    )

    args = parser.parse_args()

    if args.dynspec is None or args.result is None:
        print(
            "Expected a dynamic spectrum '--dynspec' and path to the FDD result '--result'."
        )
        return

    plot_burst(
        args.dynspec,
        args.result,
        image_filename=args.image,
        zoom_width=args.zoom,
    )

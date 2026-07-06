r"""TODO: add dedisperse...
def dedisperse(spec, ishift):
    out = np.zeros_like(spec)

    for ichan in range(spec.shape[0]):
        out[ichan, :] = np.roll(spec[ichan, :], ishift[ichan])

    return out

def compute_dmlet(spec, t, f, dms):
    nt, ndms = len(t), len(dms)
    out = np.zeros(nt * ndms).reshape(ndms, nt)

    for i, dm in enumerate(dms):
        dtdm = dm * kdm * (f**(-2) - f.max()**(-2))
        idm = np.round(dtdm / fil.tsamp).astype("int")

        out[i] = np.sum(dedisperse(spec, idm), axis=0)

    return out

def create_dmlet(datfroot, imin, imax):
    # Find dat files
    fnames = sorted(glob.glob(datfname))

    # Find file root
    match = re.search("DM\d", fnames[0])
    idm = match.start() + 2
    froot = fnames[0][:idm]

    # Find DM range
    dms = []
    for fname in fnames:
        dmstr = fname[idm:].replace(".dat", "")
        dms.append(float(dmstr))
    dms = np.asarray(dms)
    ndm = dms.size

    # Output array
    nsamp = imax - imin
    I_t_dm = np.zeros(nsamp * ndm).reshape(ndm, nsamp)

    # Read data
    for i, fname in enumerate(fnames):
        I_t_dm[i] = np.fromfile(fname, dtype="float32")[imin:imax]

    return I_t_dm, dms

if __name__ == "__main__":
    # DM constant
    kdm = 1 / 2.41e-4

    # File to read
    filfname = "./data/pks_frb110220.fil"
    #datfname = "./pks_frb110220_DM*.dat"
    datfname = "./test_DM*.dat"

    # Burst time and DM
    t0 = 209.1
    dm0 = 945
    #t0 = 100
    #dm0 = 700

    # Time width to show
    dt = 0.1

    # Read filterbank file
    fil = fb.FilterbankFile(filfname, "read")

    f = fil.frequencies
    dtdm = dm0 * kdm * (f**(-2) - f.max()**(-2))
    idm = np.round(dtdm / fil.tsamp).astype("int")

    imin = int(np.round((t0 - 0.5 * dt) / fil.tsamp))
    imax = int(np.round((t0 + 0.5 * dt) / fil.tsamp)) + idm.max()
    t = np.arange(imin, imax) * fil.tsamp

    print("Reading filterbank data")
    spec = fil.get_spectra(imin, imax).T

    print("Dedispersing")
    spec_dedisp = dedisperse(spec, -idm)
"""

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np

from fdd.signal import Signal


def load_fdd_result(filename: str):
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
    dynspec_filename: str | None = None,
    result_filename: str | None = None,
    image_filename: str | None = None,
    zoom_width: float | None = None,
):
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

    if args.dynspec is None and args.result is None:
        print(
            "Expected a dynamic spectrum '--dynspec' and/or a path to the FDD result '--result'."
        )
        return

    plot_burst(
        dynspec_filename=args.dynspec,
        result_filename=args.result,
        image_filename=args.image,
        zoom_width=args.zoom,
    )

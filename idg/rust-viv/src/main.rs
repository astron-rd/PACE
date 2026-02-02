use clap::Parser;
use ndarray::{Array1, Array2, Array4};
use num_complex::Complex32;

use crate::{
    cli::Cli,
    constants::{NR_CORRELATIONS_IN, NR_CORRELATIONS_OUT},
    init::*,
    types::{Metadata, UvwArray},
    util::{print_header, print_param},
};

mod cli;
mod constants;
mod gridder;
mod init;
mod types;
mod util;

fn main() {
    let cli = cli::Cli::parse();

    print_parameters(&cli);

    print_header!("INITIALIZATION");
    let my_uvw = generate_uvw(
        cli.timestep_count(),
        cli.baseline_count(),
        cli.grid_size,
        cli.ellipticity,
        cli.random_seed,
    );

    let their_uvw: UvwArray = ndarray_npy::read_npy("../../uvw.npy").unwrap();

    // assert_eq!(my_uvw, their_uvw); // We don't expect these to be the same because the RNG implementation is different.

    let my_frequencies = generate_frequencies(
        cli.start_frequency,
        cli.frequency_increment,
        cli.channel_count,
    );

    let their_frequencies: Array1<f32> = ndarray_npy::read_npy("../../frequencies.npy").unwrap();

    assert_eq!(my_frequencies, their_frequencies);

    let my_metadata = generate_metadata(
        cli.channel_count,
        cli.subgrid_size,
        cli.grid_size,
        &their_uvw,
        None,
    );

    let their_metadata: Array1<Metadata> = ndarray_npy::read_npy("../../metadata.npy").unwrap();

    assert_eq!(my_metadata, their_metadata);

    // let my_visibilities = generate_visibilities(
    //     NR_CORRELATIONS_IN,
    //     cli.channel_count,
    //     cli.timestep_count(),
    //     cli.baseline_count(),
    //     cli.image_size(),
    //     cli.grid_size,
    //     &their_frequencies,
    //     &their_uvw,
    //     None,
    //     None,
    //     None,
    // );

    let their_visibilities: Array4<Complex32> =
        ndarray_npy::read_npy("../../visibilities.npy").unwrap();

    // assert_eq!(my_visibilities, their_visibilities);

    let my_taper = get_taper(cli.subgrid_size);

    let their_taper: Array2<f32> = ndarray_npy::read_npy("../../taper.npy").unwrap();

    // let _subgrids: Array4<Complex32> = Array4::zeros((
    //     subgrid_count,
    //     NR_CORRELATIONS_OUT,
    //     cli.subgrid_size,
    //     cli.subgrid_size,
    // ));

    // let _gridder = Gridder::new(NR_CORRELATIONS_IN, cli.subgrid_size);

    // println!("Done!");
}

fn print_parameters(cli: &Cli) {
    print_header!("PARAMETERS");

    print_param!("nr_correlations_in", NR_CORRELATIONS_IN);
    print_param!("nr_correlations_out", NR_CORRELATIONS_OUT);
    print_param!("start_frequency", cli.start_frequency * 1e-6);
    print_param!("frequency_increment", cli.frequency_increment * 1e-6);
    print_param!("nr_channels", cli.channel_count);
    print_param!("nr_timesteps", cli.timestep_count());
    print_param!("nr_stations", cli.station_count);
    print_param!("nr_baselines", cli.baseline_count());
    print_param!("subgrid_size", cli.subgrid_size);
    print_param!("grid_size", cli.grid_size);
}

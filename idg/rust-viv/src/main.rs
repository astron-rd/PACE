use clap::Parser;
use ndarray::Array4;
use num_complex::Complex64;

use crate::{
    constants::{NR_CORRELATIONS_IN, NR_CORRELATIONS_OUT},
    gridder::Gridder,
    init::{
        generate_frequencies, generate_metadata, generate_uvw, generate_visibilities, get_taper,
    },
    util::print_header,
};

mod cli;
mod constants;
mod gridder;
mod init;
mod types;
mod util;

fn main() {
    let cli = cli::Cli::parse();

    print_header!("INITIALIZATION");
    println!("Get simulated UVW coords.");
    let uvw = generate_uvw(
        cli.timestep_count(),
        cli.baseline_count(),
        cli.grid_size,
        cli.ellipticity,
        cli.random_seed,
    );

    let frequencies = generate_frequencies(
        cli.start_frequency,
        cli.frequency_increment,
        cli.channel_count,
    );

    let metadata = generate_metadata(
        cli.channel_count,
        cli.subgrid_size,
        cli.grid_size,
        &uvw,
        None,
    );
    let subgrid_count = metadata.len();

    let visibilities = generate_visibilities(
        NR_CORRELATIONS_IN,
        cli.channel_count,
        cli.timestep_count(),
        cli.baseline_count(),
        cli.image_size(),
        cli.grid_size,
        &frequencies,
        &uvw,
        None,
        None,
        None,
    );

    let taper = get_taper(cli.subgrid_size);

    let mut subgrids: Array4<Complex64> = Array4::zeros((
        subgrid_count,
        NR_CORRELATIONS_OUT,
        cli.subgrid_size,
        cli.subgrid_size,
    ));

    let gridder = Gridder::new(NR_CORRELATIONS_IN, cli.subgrid_size);

    println!("Done!");
}

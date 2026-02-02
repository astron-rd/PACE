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
    let uvw = generate_uvw(
        cli.timestep_count(),
        cli.baseline_count(),
        cli.grid_size,
        cli.ellipticity,
        cli.random_seed,
    );

    ndarray_npy::write_npy("uvw.npy", &uvw).unwrap();
    
    let metadata = generate_metadata(cli.channel_count, cli.subgrid_size, cli.grid_size, &uvw, None);

    ndarray_npy::write_npy("metadata.npy", &metadata).unwrap();
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

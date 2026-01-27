use std::process::exit;

use clap::Parser;
use ndarray::{Array1, Array4};
use num_complex::Complex32;

use crate::{
    cli::Cli, constants::{NR_CORRELATIONS_IN, NR_CORRELATIONS_OUT}, gridder::Gridder, init::{
        generate_frequencies, generate_metadata, generate_uvw, generate_visibilities, get_taper,
    }, types::UvwArray, util::{print_header, print_param}
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
    // let uvw = generate_uvw(
    //     cli.timestep_count(),
    //     cli.baseline_count(),
    //     cli.grid_size,
    //     cli.ellipticity,
    //     cli.random_seed,
    // );

    let uvw: UvwArray = ndarray_npy::read_npy("../../uvw.npy").unwrap();
    
    // let frequencies = generate_frequencies(
    //     cli.start_frequency,
    //     cli.frequency_increment,
    //     cli.channel_count,
    // );

    let frequencies: Array1<f32> = ndarray_npy::read_npy("../../frequencies.npy").unwrap();

    println!("{}", frequencies);

    exit(0);

    // let metadata = generate_metadata(
    //     cli.channel_count,
    //     cli.subgrid_size,
    //     cli.grid_size,
    //     &uvw,
    //     None,
    // );
    // let subgrid_count = metadata.len();

    // let _visibilities = generate_visibilities(
    //     NR_CORRELATIONS_IN,
    //     cli.channel_count,
    //     cli.timestep_count(),
    //     cli.baseline_count(),
    //     cli.image_size(),
    //     cli.grid_size,
    //     &frequencies,
    //     &uvw,
    //     None,
    //     None,
    //     None,
    // );

    // let _taper = get_taper(cli.subgrid_size);

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

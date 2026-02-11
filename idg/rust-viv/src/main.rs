use std::f32::consts::PI;

use clap::Parser;
use ndarray::prelude::*;
use num_complex::Complex32;

use crate::{
    cli::Cli,
    constants::{NR_CORRELATIONS_IN, NR_CORRELATIONS_OUT, SPEED_OF_LIGHT, W_STEP},
    gridder::Gridder,
    init::*,
    types::{Metadata, UvwArray},
    util::{print_header, print_param, time_function},
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

    let uvw: UvwArray = ndarray_npy::read_npy("../../data/uvw.npy").unwrap();
    let frequencies64: Array1<f64> = ndarray_npy::read_npy("../../data/frequencies.npy").unwrap();
    let frequencies: Array1<f32> = frequencies64.mapv(|x| x as f32);
    let wavenumbers = (frequencies * 2.0 * PI) / SPEED_OF_LIGHT;
    let metadata: Array1<Metadata> = ndarray_npy::read_npy("../../data/metadata.npy").unwrap();
    let subgrid_count = metadata.len();
    let visibilities: Array4<Complex32> =
        ndarray_npy::read_npy("../../data/visibilities.npy").unwrap();
    let mut grid: Array3<Complex32> = Array3::zeros((
        NR_CORRELATIONS_OUT as usize,
        cli.grid_size as usize,
        cli.grid_size as usize,
    ));
    let taper: Array2<f32> = ndarray_npy::read_npy("../../data/taper.npy").unwrap();
    // let mut subgrids: Array4<Complex32> = Array4::zeros((
    //     subgrid_count as usize,
    //     NR_CORRELATIONS_OUT as usize,
    //     cli.subgrid_size as usize,
    //     cli.subgrid_size as usize,
    // ));

    print_header!("MAIN");
    let gridder = Gridder::new(NR_CORRELATIONS_IN, cli.subgrid_size);
    // time_function!(
    //     "grid onto subgrids",
    //     gridder.grid_onto_subgrids(
    //         W_STEP,
    //         cli.image_size(),
    //         cli.grid_size,
    //         &wavenumbers,
    //         &uvw,
    //         &visibilities,
    //         &taper,
    //         &metadata,
    //         subgrids.view_mut()
    //     )
    // );

    let mut subgrids: Array4<Complex32> = ndarray_npy::read_npy("../../subgrids.npy").unwrap();

    time_function!(
        "ifft the subgrids",
        gridder.ifft_subgrids(subgrids.view_mut())
    );

    time_function!(
        "add subgrids to grid",
        gridder.add_subgrids_to_grid(metadata.view(), subgrids.view(), grid.view_mut())
    );

    time_function!(
        "transform grid",
        gridder.transform(fftw::types::Sign::Backward, grid.view_mut())
    );

    ndarray_npy::write_npy("grid.npy", &grid).unwrap();

    println!("{}", &grid);

    println!("done!")
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

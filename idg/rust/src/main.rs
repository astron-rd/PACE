use clap::Parser;

use crate::{
    cli::Cli,
    constants::{NR_CORRELATIONS_IN, NR_CORRELATIONS_OUT, W_STEP},
    gridder::Gridder,
    types::*,
    util::{print_header, print_param, time_function},
};

mod cli;
mod constants;
mod gridder;
mod types;
mod util;

fn main() {
    let cli = cli::Cli::parse();

    print_parameters(&cli);

    print_header!("INITIALIZATION");

    let uvw: UvwArray = time_function!("generate uvws", UvwArray::generate(&cli));

    let frequencies: FrequencyArray =
        time_function!("generate frequencies", FrequencyArray::generate(&cli));

    let wavenumbers: WavenumberArray = time_function!(
        "derive wavenumbers",
        WavenumberArray::from_frequencies(&frequencies)
    );

    let metadata: MetadataArray =
        time_function!("generate metadata", MetadataArray::generate(&cli, &uvw));

    let subgrid_count = metadata.len();

    let visibilities: VisibilityArray = time_function!(
        "generate visibilities",
        VisibilityArray::generate(&cli, &frequencies, &uvw, None, None)
    );

    let taper: Taper = time_function!("get taper", Taper::generate(&cli));

    print_header!("MAIN");
    let mut gridder = Gridder::new_empty(&cli, subgrid_count);

    time_function!(
        "grid onto subgrids",
        gridder.grid_onto_subgrids(
            &cli,
            W_STEP,
            &wavenumbers,
            &uvw,
            &visibilities,
            &taper,
            &metadata,
        )
    );

    time_function!("ifft the subgrids", gridder.ifft_subgrids());

    time_function!(
        "add subgrids to grid",
        gridder.add_subgrids_to_grid(metadata.view())
    );

    time_function!(
        "transform grid",
        gridder.transform(fftw::types::Sign::Backward)
    );

    ndarray_npy::write_npy("grid.npy", gridder.grid()).unwrap();

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

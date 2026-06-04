use anyhow::Result;
use clap::Parser;

use crate::{
    cli::Commands,
    gridder::Gridder,
    input::Input,
    util::{print_header, time_function},
};

mod cli;
mod constants;
mod gridder;
mod input;
mod types;
mod util;

fn main() -> Result<()> {
    let cli = cli::Cli::parse();

    let input = Input::from_cli(&cli)?;

    input.print_parameters();

    print_header!("MAIN");
    let mut gridder = Gridder::new_empty(&input);

    time_function!("grid onto subgrids", gridder.grid_onto_subgrids(&input));

    time_function!("ifft the subgrids", gridder.ifft_subgrids(&input));

    time_function!("add subgrids to grid", gridder.add_subgrids_to_grid(&input));

    time_function!(
        "transform grid",
        gridder.transform(&input, fftw::types::Sign::Backward)
    );

    print_header!("OUTPUT");

    let output_file = hdf5_metno::File::create(
        std::env::current_dir()?.join(cli.output_file.unwrap_or("output.h5".into())),
    )?;

    time_function!("writing grid", {
        let builder = output_file.new_dataset_builder();
        builder.with_data(gridder.grid()).create("grid")?;
    });
    if cli.output_subgrids {
        time_function!("writing subgrids", {
            let builder = output_file.new_dataset_builder();
            builder.with_data(gridder.subgrids()).create("subgrids")?;
        });
    }

    if let Commands::Generate { output_input, .. } = cli.command {
        let inputs_file = hdf5_metno::File::create(std::env::current_dir()?.join("inputs.hdf5"))?;
        if output_input {
            time_function!("writing uvw", {
                let builder = inputs_file.new_dataset_builder();
                builder.with_data(&input.uvw).create("uvws")?;
            });
            time_function!("writing frequencies", {
                let builder = inputs_file.new_dataset_builder();
                builder
                    .with_data(&input.frequencies)
                    .create("frequencies")?;
            });
            time_function!("writing visibilities", {
                let builder = inputs_file.new_dataset_builder();
                builder
                    .with_data(&input.visibilities)
                    .create("visibilities")?;
            });
            time_function!("writing metadata", {
                let builder = inputs_file.new_dataset_builder();
                builder.with_data(&input.metadata).create("metadata")?;
            });
        }
    }

    println!("done!");
    Ok(())
}

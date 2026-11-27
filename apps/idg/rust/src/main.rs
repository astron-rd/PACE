use std::{fs::File, time::Duration};

use anyhow::Result;
use clap::Parser;
use indexmap::IndexMap;
use serde::Serialize;

use crate::{
    cli::Commands, gridder::Gridder, input::Input, timing::time_function, util::print_header,
};

mod cli;
mod constants;
mod gridder;
mod input;
mod timing;
mod types;
mod util;

fn main() -> Result<()> {
    let cli = cli::Cli::parse();

    let input = Input::from_cli(&cli)?;

    input.print_parameters();

    print_header!("MAIN");
    let mut gridder = Gridder::new_empty(&input);

    time_function!(
        "grid visibilities onto subgrids",
        "grid_visibilities",
        gridder.grid_onto_subgrids(&input)
    );

    time_function!("ifft subgrids", gridder.ifft_subgrids(&input));

    time_function!("add subgrids to grid", gridder.add_subgrids_to_grid(&input));

    time_function!(
        "transform grid",
        gridder.transform(&input, fftw::types::Sign::Backward)
    );

    let output_file = hdf5_metno::File::create(
        std::env::current_dir()?.join(cli.output_file.unwrap_or("output.h5".into())),
    )?;

    output_file
        .new_dataset_builder()
        .with_data(gridder.grid())
        .create("grid")?;
    if cli.output_subgrids {
        output_file
            .new_dataset_builder()
            .with_data(gridder.subgrids())
            .create("subgrids")?;
    }

    if let Commands::Generate { output_input, .. } = cli.command
        && output_input
    {
        let inputs_file = hdf5_metno::File::create(std::env::current_dir()?.join("inputs.h5"))?;
        inputs_file
            .new_dataset_builder()
            .with_data(&input.uvw)
            .create("uvws")?;
        inputs_file
            .new_dataset_builder()
            .with_data(&input.frequencies)
            .create("frequencies")?;
        inputs_file
            .new_dataset_builder()
            .with_data(&input.visibilities)
            .create("visibilities")?;
        inputs_file
            .new_dataset_builder()
            .with_data(&input.metadata)
            .create("metadata")?;
    }

    if let Some(path) = &cli.json {
        let mut outfile = File::create(path)?;
        let report = Report::new(&input, timing::get_timings());
        serde_json::to_writer_pretty(&mut outfile, &report)?;
    }

    println!("done!");
    Ok(())
}

#[derive(Serialize)]
struct Report {
    parameters: IndexMap<&'static str, u32>,
    timings: IndexMap<String, f64>,
}

impl Report {
    fn new(input: &Input, timings: Vec<(String, Duration)>) -> Self {
        let mut parameters = IndexMap::new();

        parameters.insert("nr_correlations_in", input.correlation_count_in);
        parameters.insert("nr_correlations_out", input.correlation_count_out);
        parameters.insert("nr_channels", input.channel_count);
        parameters.insert("nr_timesteps", input.timestep_count);
        parameters.insert("nr_baselines", input.baseline_count);
        parameters.insert("subgrid_size", input.subgrid_size);
        parameters.insert("grid_size", input.grid_size);

        let timings = timings
            .into_iter()
            .map(|(name, duration)| (name, duration.as_secs_f64()))
            .collect();

        Self {
            parameters,
            timings,
        }
    }
}

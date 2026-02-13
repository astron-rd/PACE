use anyhow::Result;
use clap::Parser;

use crate::{
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

    let output_dir = cli.numpy_output.clone().unwrap_or(std::env::current_dir()?);
    std::fs::create_dir_all(&output_dir)?;
    
    ndarray_npy::write_npy(output_dir.join("grid.npy"), gridder.grid()).unwrap();
    if cli.output_subgrids {
        ndarray_npy::write_npy(output_dir.join("subgrids.npy"), gridder.subgrids()).unwrap();
    }

    println!("done!");
    Ok(())
}

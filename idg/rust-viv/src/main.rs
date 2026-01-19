use clap::Parser;

use crate::{init::get_simulated_uvw, util::print_header};

mod cli;
mod constants;
mod init;
mod types;
mod util;

fn main() {
    let cli = cli::Cli::parse();

    print_header!("INITIALIZATION");
    println!("Get simulated UVW coords.");
    let uvw = get_simulated_uvw(
        cli.timestep_count(),
        cli.baseline_count(),
        cli.grid_size,
        cli.ellipticity,
        cli.random_seed,
    );

    println!("Done!")
}

use clap::Parser;

mod cli;
mod util;

mod dedisperse;
mod simulate;

mod fdd;

fn main() {
    let cli = cli::Cli::parse();

    match &cli.command {
        cli::Commands::Simulate {
            observation_args,
            signal_args,
        } => simulate::simulate(&cli.args, observation_args, signal_args),
        cli::Commands::Dedisperse {
            observation_args,
            dedisp_args,
        } => dedisperse::dedisperse(&cli.args, observation_args, dedisp_args),
    }
}

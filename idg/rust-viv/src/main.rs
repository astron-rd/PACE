use clap::Parser;

mod cli;
mod constants;
mod types;

fn main() {
    let _cli = cli::Cli::parse();
}

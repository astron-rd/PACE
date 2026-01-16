use clap::Parser;

mod cli;
mod constants;

fn main() {
    let _cli = cli::Cli::parse();
}

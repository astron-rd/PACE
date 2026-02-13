use ndarray::prelude::*;

use crate::{
    cli::Cli,
    constants::{Complex, NR_CORRELATIONS_OUT},
};

pub type Grid = Array3<Complex>;

pub trait GridExtension {
    fn initialize(cli: &Cli) -> Self;
}

impl GridExtension for Grid {
    fn initialize(cli: &Cli) -> Self {
        Array3::zeros((
            NR_CORRELATIONS_OUT as usize,
            cli.grid_size as usize,
            cli.grid_size as usize,
        ))
    }
}

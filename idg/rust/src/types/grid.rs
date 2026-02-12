use ndarray::prelude::*;
use num_complex::Complex32;

use crate::{cli::Cli, constants::NR_CORRELATIONS_OUT};

pub type Grid = Array3<Complex32>;

pub trait GridExtension {
    fn initialize(cli: &Cli) -> Self;
}

impl GridExtension for Grid {
    fn initialize(cli: &Cli) -> Self {
        Array3::zeros((
            NR_CORRELATIONS_OUT as usize,
            cli.subgrid_size as usize,
            cli.subgrid_size as usize,
        ))
    }
}

use ndarray::prelude::*;

use crate::{
    cli::Cli,
    constants::{Complex, NR_CORRELATIONS_OUT},
};

pub type Subgrids = Array4<Complex>;

pub trait SubgridsExtension {
    fn initialize(cli: &Cli, subgrid_count: usize) -> Self;
}

impl SubgridsExtension for Subgrids {
    fn initialize(cli: &Cli, subgrid_count: usize) -> Self {
        Array4::zeros((
            subgrid_count as usize,
            NR_CORRELATIONS_OUT as usize,
            cli.subgrid_size as usize,
            cli.subgrid_size as usize,
        ))
    }
}

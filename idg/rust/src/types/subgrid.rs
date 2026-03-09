use ndarray::prelude::*;

use crate::constants::Complex;

pub type Subgrids = Array4<Complex>;

pub trait SubgridsExtension {
    fn initialize(subgrid_count: usize, correlation_count_out: u32, subgrid_size: u32) -> Self;
}

impl SubgridsExtension for Subgrids {
    fn initialize(subgrid_count: usize, correlation_count_out: u32, subgrid_size: u32) -> Self {
        Array4::zeros((
            subgrid_count,
            correlation_count_out as usize,
            subgrid_size as usize,
            subgrid_size as usize,
        ))
    }
}

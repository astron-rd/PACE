use ndarray::prelude::*;

use crate::constants::Complex;

pub type Grid = Array3<Complex>;

pub trait GridExtension {
    fn initialize(correlation_count_out: u32, grid_size: u32) -> Self;
}

impl GridExtension for Grid {
    fn initialize(correlation_count_out: u32, grid_size: u32) -> Self {
        Array3::zeros((
            correlation_count_out as usize,
            grid_size as usize,
            grid_size as usize,
        ))
    }
}

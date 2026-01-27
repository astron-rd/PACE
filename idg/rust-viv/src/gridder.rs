pub struct Gridder {
    nr_correlations_in: u32,
    nr_correlations_out: u32,
    subgrid_size: u32,
}

impl Gridder {
    pub fn new(nr_correlations_in: u32, subgrid_size: u32) -> Self {
        Self {
            nr_correlations_in,
            nr_correlations_out: if nr_correlations_in == 4 { 4 } else { 1 },
            subgrid_size,
        }
    }
}

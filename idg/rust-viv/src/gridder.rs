pub struct Gridder {
    nr_correlations_in: usize,
    nr_correlations_out: usize,
    subgrid_size: usize,
}

impl Gridder {
    pub fn new(nr_correlations_in: usize, subgrid_size: usize) -> Self {
        Self {
            nr_correlations_in,
            nr_correlations_out: if nr_correlations_in == 4 { 4 } else { 1 },
            subgrid_size,
        }
    }
}

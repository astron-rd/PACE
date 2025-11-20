use std::ops::Add;

use ndarray::Array2;
use ndarray_rand::rand_distr::num_traits::Zero;

pub struct UVW {
    pub u: f64,
    pub v: f64,
    pub w: f64,
}

impl UVW {
    pub fn new(u: f64, v: f64, w: f64) -> Self {
        return UVW { u: u, v: v, w: w };
    }
}

impl Clone for UVW {
    fn clone(&self) -> Self {
        return UVW::new(self.u, self.v, self.w);
    }
}

impl Add for UVW {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        return UVW::new(self.u + other.u, self.v + other.v, self.w + other.w);
    }
}

impl Zero for UVW {
    fn zero() -> Self {
        return UVW::new(0., 0., 0.);
    }

    fn is_zero(&self) -> bool {
        return self.u == 0. && self.v == 0. && self.w == 0.;
    }
}

pub type ArrayUVW = Array2<UVW>;

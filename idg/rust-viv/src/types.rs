use std::ops::Add;

use ndarray::Array2;
use num_traits::identities::Zero;

/// 3-d vector with UVW parameters
#[derive(Clone, Copy)]
pub struct Uvw {
    pub u: f64,
    pub v: f64,
    pub w: f64,
}

impl Uvw {
    pub fn new(u: f64, v: f64, w: f64) -> Self {
        Uvw { u, v, w }
    }
}

impl Add for Uvw {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Uvw {
            u: self.u + rhs.u,
            v: self.v + rhs.u,
            w: self.w + rhs.u,
        }
    }
}

impl Zero for Uvw {
    fn zero() -> Self {
        Self {
            u: 0.0,
            v: 0.0,
            w: 0.0,
        }
    }

    fn is_zero(&self) -> bool {
        self.u == 0.0 && self.v == 0.0 && self.w == 0.0
    }
}

pub type UvwArray = Array2<Uvw>;

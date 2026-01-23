use num_complex::Complex32;
use std::ops::Add;

use ndarray::Array2;
use num_traits::identities::Zero;

/// 3-dimensional vector with UVW parameters
#[derive(Clone, Copy, Debug)]
pub struct Uvw {
    pub u: f32,
    pub v: f32,
    pub w: f32,
}

impl Uvw {
    pub fn new(u: f32, v: f32, w: f32) -> Self {
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

pub struct Metadata {
    pub baseline: usize,
    pub time_index: usize,
    pub timestep_count: usize,
    pub channel_begin: usize,
    pub channel_end: usize,
    pub coordinate: Coordinate,
}

#[derive(Clone, Copy)]
pub struct Coordinate {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

pub type Visibility = Complex32;

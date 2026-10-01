use std::ops::Add;

use crate::constants::{Float, PI};

use hdf5_metno::H5Type;
use ndarray::prelude::*;
use ndarray_rand::{
    RandomExt,
    rand::{SeedableRng, rngs::StdRng},
    rand_distr::{Beta, Uniform},
};
use num_traits::Zero;

/// 3-dimensional array with UVW parameters
#[derive(Clone, Copy, Debug, PartialEq, H5Type)]
#[repr(C)]
pub struct Uvw {
    pub u: Float,
    pub v: Float,
    pub w: Float,
}

impl Uvw {
    pub fn new(u: Float, v: Float, w: Float) -> Self {
        Uvw { u, v, w }
    }
}

impl Add for Uvw {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Uvw {
            u: self.u + rhs.u,
            v: self.v + rhs.v,
            w: self.w + rhs.w,
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

pub trait UvwArrayExtension {
    fn generate(
        ellipticity: Float,
        random_seed: u64,
        timestep_count: u32,
        baseline_count: u32,
        grid_size: u32,
    ) -> Self;
    fn from_file(file: &hdf5_metno::File) -> Result<Self, hdf5_metno::Error>
    where
        Self: Sized;
}

impl UvwArrayExtension for UvwArray {
    /// Generate simulated UVW data
    ///
    ///  Returns a UVW array of size (`baseline_count` * `timestep_count`)
    fn generate(
        ellipticity: Float,
        random_seed: u64,
        timestep_count: u32,
        baseline_count: u32,
        grid_size: u32,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(random_seed);

        // Initialize time_samples array with incrementing floats
        let time_samples = Array::from_iter(0..timestep_count).mapv(|x| x as Float);

        // Initialize uvw array with zeroes
        let mut uvw: UvwArray = UvwArray::zeros((
            baseline_count.try_into().unwrap(),
            timestep_count.try_into().unwrap(),
        ));

        let max_uv = 0.7 * (grid_size / 2) as Float;

        // Generate baseline ratios with more short baselines (beta distribution)
        // Beta distribution with alpha=1, beta=3 peaks at 0 and decreases
        let beta_distribution = Beta::new(1.0, 3.0).expect("Should be a valid distribution.");
        let baseline_ratios =
            Array::random_using(baseline_count as usize, beta_distribution, &mut rng);

        // Generate random starting angles for each baseline
        let start_angles = Array::random_using(
            baseline_count as usize,
            Uniform::new(0.0, 2.0 * PI).expect("Should be a valid distribution."),
            &mut rng,
        );

        // Calculate the UV coordinates for each baseline
        for (baseline, ratio) in baseline_ratios.iter().enumerate() {
            // Calculate radius for this baseline
            let mut u_radius = ratio * max_uv;
            let mut v_radius = ratio * max_uv;

            if ellipticity > 0.0 {
                // Make the ellipse orientation depend on the baseline
                // Longer baselines have more ellipticity
                let ellipse_factor = 1.0 + ellipticity * ratio;
                u_radius *= ellipse_factor;
                v_radius /= ellipse_factor;
            }

            // Calculate angular velocity (complete circle in 24 hours)
            // For shorter observations, we get an arc instead of full circle
            let angular_velocity = (2.0 * PI) / (24 * 3600) as Float;

            // Generate UV coordinates with random starting angle
            let angle = start_angles[baseline] + angular_velocity * &time_samples;
            let u_coords = u_radius * angle.cos();
            let v_coords = v_radius * angle.sin();

            for t in 0..timestep_count as usize {
                uvw[(baseline, t)] = Uvw::new(
                    u_coords[t] + (grid_size / 2) as Float,
                    v_coords[t] + (grid_size / 2) as Float,
                    0.,
                );
            }
        }

        uvw
    }

    /// Read UVW data from HDF5 file
    ///
    /// ## Parameters
    /// - `file`: The HDF5 File
    ///
    ///  Returns a UVW array of size (`baseline_count` * `timestep_count`)
    fn from_file(file: &hdf5_metno::File) -> Result<Self, hdf5_metno::Error> {
        file.dataset("uvws")?.read()
    }
}

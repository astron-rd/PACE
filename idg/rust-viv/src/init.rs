//! Functions to initialize dummy data

use std::f64::consts::PI;

use ndarray::Array;
use ndarray_rand::{
    rand::{rngs::StdRng, SeedableRng},
    rand_distr::{Beta, Uniform},
    RandomExt,
};

use crate::types::{Uvw, UvwArray};

/// Generate simulated UVW data
///
/// ## Parameters
/// `timestep_count`: Number of timesteps
/// `baseline_count`: Number of baselines to simulate
/// `grid_size`: Size of the image in pixels (assumed square)
/// `ellipticity`: Amount of ellipticity (0=circular, 1=highly elliptical) (Optional, default = 0.1)
/// `seed`: Random seed for generating baseline ratios and starting angles (Optional, default = 2)
///
///  Returns a UVW array of size (baseline_count * timestep_count)
pub fn get_simulated_uvw(
    timestep_count: usize,
    baseline_count: usize,
    grid_size: usize,
    ellipticity: Option<f64>,
    seed: Option<u64>,
) -> UvwArray {
    let ellipticity = ellipticity.unwrap_or(0.1);
    let seed = seed.unwrap_or(2);
    let mut rng = StdRng::seed_from_u64(seed);

    let time_samples = Array::from_iter(0..timestep_count).mapv(|x| x as f64);

    let mut uvw: UvwArray = UvwArray::zeros((baseline_count, timestep_count));

    let max_uv = 0.7 * (grid_size / 2) as f64;

    let beta_distribution = Beta::new(1.0f64, 3.0f64).expect("Should be a valid distribution.");
    let baseline_ratios = Array::random_using(baseline_count, beta_distribution, &mut rng);
    let start_angles = Array::random_using(
        baseline_count,
        Uniform::new(0.0, 2.0 * PI).expect("Should be a valid distribution."),
        &mut rng,
    );

    for (baseline, ratio) in baseline_ratios.iter().enumerate() {
        let mut u_radius = ratio * max_uv;
        let mut v_radius = ratio * max_uv;

        if ellipticity > 0.0 {
            let ellipse_factor = 1.0 + ellipticity * ratio;
            u_radius *= ellipse_factor;
            v_radius /= ellipse_factor;
        }

        let angular_velocity = (2.0 * PI) / (24 * 3600) as f64;

        let angle = start_angles[baseline] + angular_velocity * &time_samples;
        let u_coords = u_radius * angle.cos();
        let v_coords = v_radius * angle.sin();

        for t in 0..timestep_count {
            uvw[(baseline, t)] = Uvw::new(u_coords[t], v_coords[t], 0.);
        }
    }

    uvw
}

use crate::idgtypes::{ArrayUVW, UVW};

use ndarray::{Array, Array1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::{SeedableRng, rngs::StdRng};
use ndarray_rand::rand_distr::{Beta, Uniform};
use std::f64::consts::PI;

const DEFAULT_ELLIPTICITY: f64 = 0.1;
const DEFAULT_SEED: u64 = 2;
const DEFAULT_MAX_GROUP_SIZE: usize = 256;

/// Generate simulated UVW data
///
/// # Arguments
/// * `observation_hours` - Duration of observation in hours
/// * `nr_baselines` - Number of baselines to simulate
/// * `grid_size` - Size of the image in pixels (assumed square)
/// * `ellipticity` - Amount of ellipticity (0 = circular, 1 = highly elliptical)
/// * `seed` - Random seed for generating baseline ratios and start angles
///
/// # Returns
/// * `ArrayUVW` - Array with shape (nr_baselines, nr_timesteps)
pub fn get_uvw(
    observation_hours: usize,
    nr_baselines: usize,
    grid_size: usize,
    ellipticity: Option<f64>,
    seed: Option<u64>,
) -> ArrayUVW {
    // Set default values if not provided
    let ellipticity = ellipticity.unwrap_or(DEFAULT_ELLIPTICITY);
    let seed = seed.unwrap_or(DEFAULT_SEED);
    let mut rng = StdRng::seed_from_u64(seed);

    // Convert observation time to seconds (1 sample per second)
    let observation_seconds = observation_hours * 3600;
    let time_samples =
        Array::linspace(0., observation_seconds as f64, observation_seconds as usize);
    let nr_timesteps = observation_seconds;

    // Initialize uvw array
    let mut uvw = ArrayUVW::zeros((nr_baselines, nr_timesteps));

    // Calculate maximum UV distance
    let max_uv = 0.7 * (grid_size as f64) / 2.;

    // Generate baseline ratios with more short baselines (beta distribution)
    // Beta distribution with alpha=1, beta=3 peaks at 0 and decreases
    let beta = Beta::new(1., 3.).unwrap();
    let baseline_ratios = Array::random_using(nr_baselines, beta, &mut rng);

    // Generate random starting angles for each baseline
    let start_angles = Array::random_using(nr_baselines, Uniform::new(0., 2. * PI), &mut rng);

    // Calculate the UV coordinates for each baseline
    for (bl, ratio) in baseline_ratios.iter().enumerate() {
        // Calculate radius for this baseline
        let (mut radius_u, mut radius_v) = (ratio * max_uv, ratio * max_uv);

        // Apply ellipticity if specified
        if ellipticity > 0. {
            // Make the ellipse orientation depend on the baseline
            // Longer baselines have more ellipticity
            let ellipse_factor = 1. + ellipticity * ratio;
            radius_u = radius_u * ellipse_factor;
            radius_v = radius_v / ellipse_factor;
        }

        // Calculate angular velocity (complete circle in 24 hours)
        // For shorter observations, we get an arc instead of a full circle
        let angular_velocity = 2. * PI / (24. * 3600.); // radians per second

        // Generate UV coordinates with random starting angle
        let angle = start_angles[bl] + angular_velocity * &time_samples;
        let coords_u = radius_u * angle.mapv(f64::cos) + grid_size as f64 / 2.;
        let coords_v = radius_v * angle.mapv(f64::sin) + grid_size as f64 / 2.;

        // Store the coordinates
        for t in 0..nr_timesteps {
            uvw[(bl, t)] = UVW::new(coords_u[t], coords_v[t], 0.);
        }
    }

    return uvw;
}

/// Generate array of frequencies for each channel
///
/// # Arguments
/// * `start_frequency` - Starting frequency in Hz
/// * `frequency_increment` - Increment in Hz between consecutive channels
/// * `nr_channels` - Number of frequency channels
///
/// # Returns
/// * `Array1<f64>` - Array of frequencies with shape (nr_channels)
pub fn get_frequencies(
    start_frequency: f64,
    frequency_increment: f64,
    nr_channels: usize,
) -> Array1<f64> {
    Array1::range(
        start_frequency,
        start_frequency + nr_channels as f64 * frequency_increment,
        frequency_increment,
    )
}

/// Compute metadata for all baselines
///
/// # Arguments
/// * `nr_channels` - Number of frequency channels
/// * `subgrid_size` - Size of each subgrid
/// * `uvw` - Array of uvw coordinates, shape (nr_baselines, nr_timesteps, 3)
/// * `max_group_size` - Maximum number of visibilities (timesteps) in a group
///
/// # Returns
/// `Array1<f64>` - Array of metadata with shape (nr_subgrids)
pub fn get_metadata(
    nr_channels: usize,
    subgrid_size: usize,
    uvw: &ArrayUVW,
    max_group_size: Option<usize>,
) {
    let pixels_u = uvw.map(|coord| coord.u);
    let pixels_v = uvw.map(|coord| coord.v);
}

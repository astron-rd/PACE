//! Functions to initialize dummy data

use std::f64::consts::PI;

use code_timing_macros::time_function;
use ndarray::{Array, Array1, Array2, Array4, ArrayView1, linspace, s};
use ndarray_rand::{
    RandomExt,
    rand::{Rng, SeedableRng, rngs::StdRng},
    rand_distr::{Beta, Uniform},
};
use num_complex::Complex64;

use crate::{
    constants::SPEED_OF_LIGHT,
    types::{Coordinate, Metadata, Uvw, UvwArray, Visibility},
};

/// Generate simulated UVW data
///
/// ## Parameters
/// - `timestep_count`: Number of timesteps
/// - `baseline_count`: Number of baselines to simulate
/// - `grid_size`: Size of the image in pixels (assumed square)
/// - `ellipticity`: Amount of ellipticity (0=circular, 1=highly elliptical) (Optional, default = 0.1)
/// - `seed`: Random seed for generating baseline ratios and starting angles (Optional, default = 2)
///
///  Returns a UVW array of size (`baseline_count` * `timestep_count`)
#[time_function]
pub fn generate_uvw(
    timestep_count: usize,
    baseline_count: usize,
    grid_size: usize,
    ellipticity: Option<f64>,
    seed: Option<u64>,
) -> UvwArray {
    let ellipticity = ellipticity.unwrap_or(0.1);
    let seed = seed.unwrap_or(2);
    let mut rng = StdRng::seed_from_u64(seed);

    // Initialize time_samples array with incrementing floats
    let time_samples = Array::from_iter(0..timestep_count).mapv(|x| x as f64);

    // Initialize uvw array with zeroes
    let mut uvw: UvwArray = UvwArray::zeros((baseline_count, timestep_count));

    let max_uv = 0.7 * (grid_size / 2) as f64;

    // Generate baseline ratios with more short baselines (beta distribution)
    // Beta distribution with alpha=1, beta=3 peaks at 0 and decreases
    let beta_distribution = Beta::new(1.0f64, 3.0f64).expect("Should be a valid distribution.");
    let baseline_ratios = Array::random_using(baseline_count, beta_distribution, &mut rng);

    // Generate random starting angles for each baseline
    let start_angles = Array::random_using(
        baseline_count,
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
        let angular_velocity = (2.0 * PI) / f64::from(24 * 3600);

        // Generate UV coordinates with random starting angle
        let angle = start_angles[baseline] + angular_velocity * &time_samples;
        let u_coords = u_radius * angle.cos();
        let v_coords = v_radius * angle.sin();

        for t in 0..timestep_count {
            uvw[(baseline, t)] = Uvw::new(u_coords[t], v_coords[t], 0.);
        }
    }

    uvw
}

/// Generate array of frequencies for each channel.
///
/// ## Parameters
/// - `start_frequency`: Starting frequency in Hz
/// - `frequency_increment`: Increment in Hz between consecutive channels
/// - `channel_count`: Number of frequency channels
///
/// Returns frequencies array, shape (`channel_count`)
#[time_function]
pub fn generate_frequencies(
    start_frequency: f64,
    frequency_increment: f64,
    channel_count: usize,
) -> Array1<f64> {
    Array::range(
        start_frequency,
        start_frequency + (channel_count as f64 * frequency_increment),
        frequency_increment,
    )
}

/// Compute metadata for all baselines.
///
/// ## Parameters
/// - `nr_channels`: number of frequency channels
/// - `subgrid_size`: size of the subgrid
/// - `grid_size`: size of the grid
/// - `uvw`: array of uvw coordinates, shape (nr_baselines, nr_timesteps, 3)
/// - `max_group_size`: maximum number of visibilities (timesteps) in a group
///
/// Returns a metadata array, shape (`nr_subgrids`)
#[time_function]
pub fn generate_metadata(
    channel_count: usize,
    subgrid_size: usize,
    grid_size: usize,
    uvw: &UvwArray,
    max_group_size: Option<usize>,
) -> Vec<Metadata> {
    let max_group_size = max_group_size.unwrap_or(256);

    let u_pixels = uvw.mapv(|x| x.u);
    let v_pixels = uvw.mapv(|x| x.v);

    let baseline_count = uvw.shape()[0];

    let mut metadata = Vec::new();

    for baseline in 0..baseline_count {
        metadata.extend(compute_metadata(
            grid_size,
            subgrid_size,
            channel_count,
            baseline,
            u_pixels.slice(s![baseline, ..]),
            v_pixels.slice(s![baseline, ..]),
            max_group_size,
        ));
    }

    metadata
}

pub fn compute_metadata(
    grid_size: usize,
    subgrid_size: usize,
    channel_count: usize,
    baseline: usize,
    u_pixels: ArrayView1<f64>,
    v_pixels: ArrayView1<f64>,
    max_group_size: usize,
) -> Vec<Metadata> {
    let mut metadata = Vec::new();

    let timestep_count = u_pixels.shape()[0];
    let max_distance = 0.8 * subgrid_size as f64;

    let mut timestep = 0;
    while timestep < timestep_count {
        let current_u = u_pixels[timestep];
        let current_v = v_pixels[timestep];

        // TODO: Add better explanation for what's happening with the group_size here
        let mut group_size = 1;
        while (timestep + group_size < timestep_count)
            && (group_size < max_group_size)
            && (((u_pixels[timestep + group_size] - current_u).powi(2)
                + (v_pixels[timestep + group_size] - current_v).powi(2))
            .sqrt()
                <= max_distance)
        {
            group_size += 1;
        }

        let group_u = u_pixels
            .slice(s![timestep..timestep + group_size])
            .mean()
            .expect("This slice should not be empty");
        let group_v = v_pixels
            .slice(s![timestep..timestep + group_size])
            .mean()
            .expect("This slice should not be empty");

        let subgrid_x = group_u as usize - (subgrid_size / 2);
        let subgrid_y = group_v as usize - (subgrid_size / 2);
        let subgrid_x = subgrid_x.clamp(0, grid_size - subgrid_size);
        let subgrid_y = subgrid_y.clamp(0, grid_size - subgrid_size);

        metadata.push(Metadata {
            baseline,
            time_index: timestep,
            timestep_count: group_size,
            channel_begin: 0,
            channel_end: channel_count,
            coordinate: Coordinate {
                x: subgrid_x,
                y: subgrid_y,
                z: 0,
            },
        });

        timestep += group_size;
    }

    metadata
}

#[time_function]
pub fn generate_visibilities(
    correlation_count: usize,
    channel_count: usize,
    timestep_count: usize,
    baseline_count: usize,
    image_size: f64,
    grid_size: usize,
    frequencies: &Array1<f64>,
    uvw: &UvwArray,
    point_sources_count: Option<usize>,
    max_pixel_offset: Option<usize>,
    seed: Option<u64>,
) -> Array4<Complex64> {
    let point_sources_count = point_sources_count.unwrap_or(4);
    let max_pixel_offset = max_pixel_offset.unwrap_or(grid_size / 3);
    let seed = seed.unwrap_or(2);

    let mut visibilities: Array4<Visibility> = Array4::zeros((
        baseline_count,
        timestep_count,
        channel_count,
        correlation_count,
    ));

    let mut offsets = Vec::new();
    let mut rng = StdRng::seed_from_u64(seed);

    for _ in 0..point_sources_count {
        let x = (rng.random::<f32>() * max_pixel_offset as f32) as usize - (max_pixel_offset / 2);
        let y = (rng.random::<f32>() * max_pixel_offset as f32) as usize - (max_pixel_offset / 2);
        offsets.push((x, y));
    }

    for offset in offsets {
        let amplitude = 1.0;

        // Convert offset from grid cells to radians (l,m)
        let l = offset.0 as f64 * image_size / grid_size as f64;
        let m = offset.1 as f64 * image_size / grid_size as f64;

        for baseline in 0..baseline_count {
            add_point_source_to_baseline(
                baseline,
                timestep_count,
                channel_count,
                amplitude,
                frequencies,
                uvw,
                l,
                m,
                &mut visibilities,
            )
        }
    }

    visibilities
}

pub fn add_point_source_to_baseline(
    baseline: usize,
    timestep_count: usize,
    channel_count: usize,
    amplitude: f64,
    frequencies: &Array1<f64>,
    uvw: &UvwArray,
    l: f64,
    m: f64,
    visibilities: &mut Array4<Visibility>,
) {
    for t in 0..timestep_count {
        for c in 0..channel_count {
            let u = (frequencies[c] / SPEED_OF_LIGHT) * uvw[(baseline, t)].u;
            let v = (frequencies[c] / SPEED_OF_LIGHT) * uvw[(baseline, t)].v;

            let phase = -2.0 * PI * (u * l + v * m);
            let value = amplitude * (phase * Complex64::new(0., 1.)).exp();

            // TODO: This is awful, please Rustify
            visibilities
                .slice_mut(s![baseline, t, c, ..])
                .mapv_inplace(|x| x + value);
        }
    }
}

#[time_function]
pub fn get_taper(subgrid_size: usize) -> Array2<f64> {
    let x: Array1<f64> = linspace(-1.0, 1.0, subgrid_size).collect();
    let spheroidal = x.map(|x| evaluate_spheroidal(*x));

    let mat_1n = spheroidal
        .to_shape((1, spheroidal.len()))
        .expect("1D array should fit in 1xN matrix.");
    let mat_n1 = mat_1n.clone().reversed_axes();

    (mat_1n * mat_n1).to_owned()
}

pub fn evaluate_spheroidal(x: f64) -> f64 {
    #[rustfmt::skip]
    let p: [[f64; 5]; 2] = [
        [8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
        [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2],
    ];
    #[rustfmt::skip]
    let q: [[f64; 3]; 2] = [
        [1.0000000e0, 8.212018e-1, 2.078043e-1],
        [1.0000000e0, 9.599102e-1, 2.918724e-1],
    ];

    let (part, end): (usize, f64) = match x {
        0.0..0.75 => (0, 0.75),
        0.75..=1.0 => (1, 1.0),
        _ => return 0.0,
    };

    // TODO: This bit is kinda ugly, might be able to use some cleaning up
    // TODO: Potentially split off `evaluate_polynomial` function
    let x_squared = x.powi(2);
    let del_x_squared = x_squared - end.powi(2);
    let mut del_x_squared_pow = del_x_squared;
    let mut top = p[part][0];
    for p in p[part].iter().skip(1) {
        top += p * del_x_squared_pow;
        del_x_squared_pow *= del_x_squared;
    }

    let mut btm = q[part][0];
    del_x_squared_pow = del_x_squared;
    for q in q[part].iter().skip(1) {
        btm += q * del_x_squared_pow;
        del_x_squared_pow *= del_x_squared;
    }

    if btm == 0.0 {
        0.0
    } else {
        (1.0 - x_squared) * (top / btm)
    }
}

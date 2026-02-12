use std::f32::consts::PI;

use ndarray::prelude::*;
use ndarray_rand::rand::{Rng, SeedableRng, rngs::StdRng};
use num_complex::Complex32;

use crate::{
    cli::Cli,
    constants::{NR_CORRELATIONS_IN, SPEED_OF_LIGHT},
    types::{FrequencyArray, UvwArray},
};

pub type Visibility = Complex32;

pub type VisibilityArray = Array4<Visibility>;

pub trait VisibilityArrayExtension {
    fn generate(
        cli: &Cli,
        frequencies: &FrequencyArray,
        uvw: &UvwArray,
        point_sources_count: Option<u32>,
        max_pixel_offset: Option<u32>,
    ) -> Self;
    fn from_file(path: &str) -> Result<Self, ndarray_npy::ReadNpyError>
    where
        Self: Sized;
}

impl VisibilityArrayExtension for VisibilityArray {
    fn generate(
        cli: &Cli,
        frequencies: &FrequencyArray,
        uvw: &UvwArray,
        point_sources_count: Option<u32>,
        max_pixel_offset: Option<u32>,
    ) -> Self {
        let point_sources_count = point_sources_count.unwrap_or(4);
        let max_pixel_offset = max_pixel_offset.unwrap_or(cli.grid_size / 3);
        let seed = cli.random_seed.unwrap_or(2);

        let mut visibilities: Array4<Visibility> = Array4::zeros((
            cli.baseline_count().try_into().unwrap(),
            cli.timestep_count().try_into().unwrap(),
            cli.channel_count.try_into().unwrap(),
            NR_CORRELATIONS_IN.try_into().unwrap(),
        ));

        let mut offsets = Vec::new();
        let mut rng = StdRng::seed_from_u64(seed);

        for _ in 0..point_sources_count {
            let x = (rng.random::<f32>() * max_pixel_offset as f32) - (max_pixel_offset / 2) as f32;
            let y = (rng.random::<f32>() * max_pixel_offset as f32) - (max_pixel_offset / 2) as f32;
            offsets.push((x, y));
        }

        for offset in offsets {
            let amplitude = 1.0;

            // Convert offset from grid cells to radians (l,m)
            let l = offset.0 * cli.image_size() / cli.grid_size as f32;
            let m = offset.1 * cli.image_size() / cli.grid_size as f32;

            for baseline in 0..cli.baseline_count() {
                add_point_source_to_baseline(
                    baseline,
                    cli.timestep_count(),
                    cli.channel_count,
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

    fn from_file(path: &str) -> Result<Self, ndarray_npy::ReadNpyError>
    where
        Self: Sized,
    {
        ndarray_npy::read_npy(path)
    }
}

fn add_point_source_to_baseline(
    baseline: u32,
    timestep_count: u32,
    channel_count: u32,
    amplitude: f32,
    frequencies: &Array1<f32>,
    uvw: &UvwArray,
    l: f32,
    m: f32,
    visibilities: &mut Array4<Visibility>,
) {
    let baseline = baseline as usize;
    for t in 0..timestep_count as usize {
        for c in 0..channel_count as usize {
            let u = (frequencies[c] / SPEED_OF_LIGHT) * uvw[(baseline, t)].u;
            let v = (frequencies[c] / SPEED_OF_LIGHT) * uvw[(baseline, t)].v;

            let phase = -2.0 * PI * (u * l + v * m);
            let value = amplitude * (phase * Complex32::new(0., 1.)).exp();

            // TODO: This is awful, please Rustify
            visibilities
                .slice_mut(s![baseline, t, c, ..])
                .mapv_inplace(|x| x + value);
        }
    }
}

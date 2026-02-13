use std::path::Path;

use ndarray::prelude::*;
use ndarray_rand::rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
    constants::{Complex, Float, PI, SPEED_OF_LIGHT},
    types::*,
};

pub type Visibility = Complex;

pub type VisibilityArray = Array4<Visibility>;

pub trait VisibilityArrayExtension {
    fn generate(
        point_sources_count: u32,
        max_pixel_offset: u32,
        random_seed: u64,
        baseline_count: u32,
        timestep_count: u32,
        channel_count: u32,
        correlation_count_in: u32,
        image_size: Float,
        grid_size: u32,
        frequencies: &FrequencyArray,
        uvw: &UvwArray,
    ) -> Self;
    fn from_file(path: &Path) -> Result<Self, ndarray_npy::ReadNpyError>
    where
        Self: Sized;
}

impl VisibilityArrayExtension for VisibilityArray {
    fn generate(
        point_sources_count: u32,
        max_pixel_offset: u32,
        random_seed: u64,
        baseline_count: u32,
        timestep_count: u32,
        channel_count: u32,
        correlation_count_in: u32,
        image_size: Float,
        grid_size: u32,
        frequencies: &FrequencyArray,
        uvw: &UvwArray,
    ) -> Self {
        let mut visibilities: Array4<Visibility> = Array4::zeros((
            baseline_count.try_into().unwrap(),
            timestep_count.try_into().unwrap(),
            channel_count.try_into().unwrap(),
            correlation_count_in.try_into().unwrap(),
        ));

        let mut offsets = Vec::new();
        let mut rng = StdRng::seed_from_u64(random_seed);

        for _ in 0..point_sources_count {
            let x = (rng.random::<Float>() * max_pixel_offset as Float)
                - (max_pixel_offset / 2) as Float;
            let y = (rng.random::<Float>() * max_pixel_offset as Float)
                - (max_pixel_offset / 2) as Float;
            offsets.push((x, y));
        }

        for offset in offsets {
            let amplitude = 1.0;

            // Convert offset from grid cells to radians (l,m)
            let l = offset.0 * image_size / grid_size as Float;
            let m = offset.1 * image_size / grid_size as Float;

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

    fn from_file(path: &Path) -> Result<Self, ndarray_npy::ReadNpyError>
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
    amplitude: Float,
    frequencies: &Array1<Float>,
    uvw: &UvwArray,
    l: Float,
    m: Float,
    visibilities: &mut Array4<Visibility>,
) {
    let baseline = baseline as usize;
    for t in 0..timestep_count as usize {
        for c in 0..channel_count as usize {
            let u = (frequencies[c] / SPEED_OF_LIGHT) * uvw[(baseline, t)].u;
            let v = (frequencies[c] / SPEED_OF_LIGHT) * uvw[(baseline, t)].v;

            let phase = -2.0 * PI * (u * l + v * m);
            let value = amplitude * (phase * Complex::new(0., 1.)).exp();

            // TODO: This is awful, please Rustify
            visibilities
                .slice_mut(s![baseline, t, c, ..])
                .mapv_inplace(|x| x + value);
        }
    }
}

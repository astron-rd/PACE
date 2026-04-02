use std::path::Path;

use itertools::Itertools;
use ndarray::{
    Zip,
    parallel::prelude::{IntoParallelIterator, ParallelIterator},
    prelude::*,
};
use ndarray_rand::{
    rand::{SeedableRng, rngs::StdRng},
    rand_distr::{Distribution, Uniform},
};

use crate::{
    constants::{Complex, Float, PI, SPEED_OF_LIGHT},
    types::*,
};

pub type Visibility = Complex;

/// Shape (baselines, timestep, channels, correlations)
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
        let mut visibilities: VisibilityArray = Array4::zeros((
            baseline_count.try_into().unwrap(),
            timestep_count.try_into().unwrap(),
            channel_count.try_into().unwrap(),
            correlation_count_in.try_into().unwrap(),
        ));

        let mut rng = StdRng::seed_from_u64(random_seed);
        let distribution = Uniform::new_inclusive(
            -(max_pixel_offset as Float / 2.0),
            max_pixel_offset as Float / 2.0,
        )
        .expect("max_pixel_offset is unsigned, so this should not go wrong");

        let freq_div_sol = frequencies / SPEED_OF_LIGHT;

        for (offset_l, offset_m) in distribution
            .sample_iter(&mut rng)
            .map(|x| x * image_size / grid_size as Float)
            .tuples()
            .take(point_sources_count as usize)
        {
            Zip::indexed(visibilities.view_mut())
                .into_par_iter()
                .for_each(|((baseline, timestep, channel, _), visibility)| {
                    let f = freq_div_sol[channel];
                    let u = f * uvw[(baseline, timestep)].u;
                    let v = f * uvw[(baseline, timestep)].v;

                    let phase = -2.0 * PI * (u * offset_l + v * offset_m);
                    let value = Complex::new(0., phase).exp();

                    *visibility += value;
                });
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

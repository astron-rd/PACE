use std::f32::consts::PI;

use code_timing_macros::time_function;
use ndarray::{Zip, prelude::*};
use num_complex::Complex32;

use crate::{
    constants::{NR_CORRELATIONS_IN, NR_CORRELATIONS_OUT},
    types::{Metadata, Uvw, UvwArray},
};

pub struct Gridder {
    nr_correlations_in: u32,
    nr_correlations_out: u32,
    subgrid_size: u32,
}

impl Gridder {
    pub fn new(nr_correlations_in: u32, subgrid_size: u32) -> Self {
        Self {
            nr_correlations_in,
            nr_correlations_out: if nr_correlations_in == 4 { 4 } else { 1 },
            subgrid_size,
        }
    }

    #[time_function]
    pub fn grid_onto_subgrids(
        &self,
        w_step: f32,
        image_size: f32,
        grid_size: u32,
        wavenumbers: &Array1<f32>,
        uvw: &UvwArray,
        visibilities: &Array4<Complex32>,
        taper: &Array2<f32>,
        metadata: &Array1<Metadata>,
        subgrids: ArrayViewMut4<Complex32>,
    ) {
        assert_eq!(self.nr_correlations_in as usize, visibilities.shape()[3]);
        assert_eq!(self.nr_correlations_out as usize, subgrids.shape()[1]);
        assert_eq!(self.subgrid_size as usize, subgrids.shape()[2]);

        visibilities_to_subgrids(
            w_step,
            image_size,
            grid_size,
            wavenumbers,
            uvw,
            visibilities,
            taper,
            metadata,
            subgrids,
        );
    }
}

fn visibilities_to_subgrids(
    w_step: f32,
    image_size: f32,
    grid_size: u32,
    wavenumbers: &Array1<f32>,
    uvw: &UvwArray,
    visibilities: &Array4<Complex32>,
    taper: &Array2<f32>,
    metadata: &Array1<Metadata>,
    mut subgrids: ArrayViewMut4<Complex32>,
) {
    Zip::from(metadata)
        .and(subgrids.axis_iter_mut(ndarray::Axis(0)))
        .par_for_each(|metadata, subgrid| {
            visibility_to_subgrid(
                metadata,
                w_step,
                image_size,
                grid_size,
                wavenumbers,
                uvw,
                visibilities,
                taper,
                subgrid,
            )
        });
}

fn visibility_to_subgrid(
    metadata: &Metadata,
    w_step: f32,
    image_size: f32,
    grid_size: u32,
    wavenumbers: &Array1<f32>,
    uvw: &UvwArray,
    visibilities: &Array4<Complex32>,
    taper: &Array2<f32>,
    mut subgrid: ArrayViewMut3<Complex32>,
) {
    let w_offset_in_lambda = w_step * (metadata.coordinate.z as f32 + 0.5);
    let subgrid_size = subgrid.shape()[1] as u32;

    let u_offset =
        (metadata.coordinate.x as f32 + subgrid_size as f32 / 2.0 - grid_size as f32 / 2.0) * (2.0 * PI / image_size);
    let v_offset =
        (metadata.coordinate.y as f32 + subgrid_size as f32 / 2.0 - grid_size as f32 / 2.0) * (2.0 * PI / image_size);
    let w_offset = 2.0 * PI * w_offset_in_lambda;

    for y in 0..subgrid_size {
        for x in 0..subgrid_size {
            let l = compute_lm(x, subgrid_size, image_size);
            let m = compute_lm(y, subgrid_size, image_size);
            let n = compute_n(l, m);

            let pixels = compute_pixels(
                metadata.timestep_count,
                metadata.time_index,
                uvw,
                metadata.baseline,
                l,
                m,
                n,
                u_offset,
                v_offset,
                w_offset,
                metadata.channel_begin,
                metadata.channel_end,
                wavenumbers,
                visibilities
            );

            let sph = taper[(y as usize, x as usize)];
            let x_dst = (x + (subgrid_size / 2)) % subgrid_size;
            let y_dst = (y + (subgrid_size / 2)) % subgrid_size;

            for pol in 0..NR_CORRELATIONS_OUT {
                subgrid[(pol as usize, y_dst as usize, x_dst as usize)] = pixels[pol as usize] * sph
            }
        }
    }
}

fn compute_lm(x: u32, subgrid_size: u32, image_size: f32) -> f32 {
    (x as f32 + 0.5 - (subgrid_size as f32 / 2.0)) * image_size / subgrid_size as f32
}

fn compute_n(l: f32, m: f32) -> f32 {
    let tmp = l * l + m * m;

    if tmp >= 1.0 {
        return 1.0;
    }

    tmp / (1.0 + f32::sqrt(1.0 - tmp))
}

fn compute_pixels(
    timestep_count: u32,
    offset: u32,
    uvw: &UvwArray,
    baseline: u32,
    l: f32,
    m: f32,
    n: f32,
    u_offset: f32,
    v_offset: f32,
    w_offset: f32,
    channel_begin: u32,
    channel_end: u32,
    wavenumbers: &Array1<f32>,
    visibilities: &Array4<Complex32>,
) -> Array1<Complex32> {
    let mut pixels: Array1<Complex32> = Array1::zeros(NR_CORRELATIONS_OUT as usize);

    for time in 0..timestep_count {
        let idx = offset + time;
        let Uvw { u, v, w } = uvw[(baseline as usize, idx as usize)];

        let phase_index = u * l + v * m + w * n;
        let phase_offset = u_offset * l + v_offset * m + w_offset * n;

        for channel in channel_begin..channel_end {
            let phase = phase_offset - (phase_index * wavenumbers[channel as usize]);
            let phasor = (Complex32::i() * phase).exp();

            for pol in 0..NR_CORRELATIONS_IN {
                pixels[(pol % NR_CORRELATIONS_OUT) as usize] += visibilities[(
                    baseline as usize,
                    idx as usize,
                    channel as usize,
                    pol as usize,
                )] * phasor
            }
        }
    }

    pixels
}

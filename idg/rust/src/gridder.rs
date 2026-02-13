#[cfg(not(feature = "f64"))]
use fftw::plan::C2CPlan32;
#[cfg(feature = "f64")]
use fftw::plan::C2CPlan64;
use fftw::{
    array::AlignedVec,
    plan::C2CPlan,
    types::{Flag, Sign},
};
use ndarray::{Zip, prelude::*};

use crate::{
    constants::{Complex, Float, PI},
    input::Input,
    types::*,
};

pub struct Gridder {
    subgrids: Subgrids,
    grid: Grid,
}

impl Gridder {
    #[allow(unused)] // For future use
    pub fn subgrids(&self) -> &Subgrids {
        &self.subgrids
    }

    pub fn grid(&self) -> &Grid {
        &self.grid
    }
    
    #[allow(unused)] // For future use
    pub fn into_grid_subgrids(self) -> (Grid, Subgrids) {
        (self.grid, self.subgrids)
    }

    pub fn new_empty(input: &Input) -> Self {
        Self {
            subgrids: Subgrids::initialize(
                input.subgrid_count,
                input.correlation_count_out,
                input.subgrid_size,
            ),
            grid: Grid::initialize(input.correlation_count_out, input.grid_size),
        }
    }

    pub fn grid_onto_subgrids(&mut self, input: &Input) {
        assert_eq!(
            input.correlation_count_in as usize,
            input.visibilities.shape()[3]
        );
        assert_eq!(
            input.correlation_count_out as usize,
            self.subgrids.shape()[1]
        );
        assert_eq!(input.subgrid_size as usize, self.subgrids.shape()[2]);

        visibilities_to_subgrids(
            input.w_step,
            input.image_size,
            input.grid_size,
            input.correlation_count_in,
            input.correlation_count_out,
            &input.wavenumbers,
            &input.uvw,
            &input.visibilities,
            &input.taper,
            &input.metadata,
            self.subgrids.view_mut(),
        );
    }

    pub fn ifft_subgrids(&mut self, input: &Input) {
        let subgrid_size = input.subgrid_size.try_into().unwrap();
        #[cfg(not(feature = "f64"))]
        let mut plan: C2CPlan32 =
            C2CPlan::aligned(&[subgrid_size, subgrid_size], Sign::Backward, Flag::MEASURE).unwrap();
        #[cfg(feature = "f64")]
        let mut plan: C2CPlan64 =
            C2CPlan::aligned(&[subgrid_size, subgrid_size], Sign::Backward, Flag::MEASURE).unwrap();

        let mut _in: AlignedVec<Complex> = AlignedVec::new(subgrid_size * subgrid_size);
        let mut _out: AlignedVec<Complex> = AlignedVec::new(subgrid_size * subgrid_size);

        for mut correlations in self.subgrids.outer_iter_mut() {
            for mut subgrid in correlations.outer_iter_mut() {
                for (dst, src) in _in.iter_mut().zip(subgrid.iter()) {
                    *dst = *src;
                }

                plan.c2c(&mut _in, &mut _out).unwrap();

                for (dst, src) in subgrid.iter_mut().zip(_out.iter()) {
                    *dst = *src;
                }
                subgrid /= Complex::new((subgrid_size * subgrid_size) as Float, 0.0); // Normalize
            }
        }
    }

    pub fn add_subgrids_to_grid(&mut self, input: &Input) {
        let phasor = compute_phasor(input.subgrid_size);

        for (subgrid, metadata) in self.subgrids.outer_iter().zip(input.metadata.iter()) {
            add_subgrid_to_grid(subgrid, metadata, self.grid.view_mut(), phasor.view());
        }
    }

    pub fn transform(&mut self, input: &Input, direction: Sign) {
        assert_eq!(input.correlation_count_out, self.grid.shape()[0] as u32);
        let height = self.grid.shape()[1];
        let width = self.grid.shape()[2];
        assert_eq!(height, width);

        #[cfg(not(feature = "f64"))]
        let mut plan: C2CPlan32 =
            C2CPlan::aligned(&[height, width], direction, Flag::MEASURE).unwrap();
        #[cfg(feature = "f64")]
        let mut plan: C2CPlan64 =
            C2CPlan::aligned(&[height, width], direction, Flag::MEASURE).unwrap();

        let mut _in: AlignedVec<Complex> = AlignedVec::new(height * width);
        let mut _out: AlignedVec<Complex> = AlignedVec::new(height * width);

        for mut correlation in self.grid.outer_iter_mut() {
            for ((x, y), src) in correlation.indexed_iter() {
                let dst_x = (x + width / 2) % width;
                let dst_y = (y + height / 2) % height;
                let dst = dst_y * width + dst_x;
                _in[dst] = *src;
            }

            plan.c2c(&mut _in, &mut _out).unwrap();

            for ((x, y), dst) in correlation.indexed_iter_mut() {
                let src_x = (x + width / 2) % width;
                let src_y = (y + height / 2) % height;
                let src = src_y * width + src_x;
                *dst = _out[src];
            }
            correlation /= Complex::new((width * height) as Float, 0.0); // Normalize
            correlation *= Complex::new(2.0, 0.0);
        }
    }
}

fn visibilities_to_subgrids(
    w_step: Float,
    image_size: Float,
    grid_size: u32,
    correlation_count_in: u32,
    correlation_count_out: u32,
    wavenumbers: &WavenumberArray,
    uvw: &UvwArray,
    visibilities: &VisibilityArray,
    taper: &Taper,
    metadata: &MetadataArray,
    mut subgrids: ArrayViewMut4<Complex>,
) {
    Zip::from(metadata)
        .and(subgrids.axis_iter_mut(ndarray::Axis(0)))
        .par_for_each(|metadata, subgrid| {
            visibility_to_subgrid(
                metadata,
                w_step,
                image_size,
                grid_size,
                correlation_count_in,
                correlation_count_out,
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
    w_step: Float,
    image_size: Float,
    grid_size: u32,
    correlation_count_in: u32,
    correlation_count_out: u32,
    wavenumbers: &WavenumberArray,
    uvw: &UvwArray,
    visibilities: &VisibilityArray,
    taper: &Taper,
    mut subgrid: ArrayViewMut3<Complex>,
) {
    let w_offset_in_lambda = w_step * (metadata.coordinate.z as Float + 0.5);
    let subgrid_size = subgrid.shape()[1] as u32;

    let u_offset = (metadata.coordinate.x as Float + subgrid_size as Float / 2.0
        - grid_size as Float / 2.0)
        * (2.0 * PI / image_size);
    let v_offset = (metadata.coordinate.y as Float + subgrid_size as Float / 2.0
        - grid_size as Float / 2.0)
        * (2.0 * PI / image_size);
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
                correlation_count_in,
                correlation_count_out,
                wavenumbers,
                visibilities,
            );

            let sph = taper[(y as usize, x as usize)];
            let x_dst = (x + (subgrid_size / 2)) % subgrid_size;
            let y_dst = (y + (subgrid_size / 2)) % subgrid_size;

            for pol in 0..correlation_count_out {
                subgrid[(pol as usize, y_dst as usize, x_dst as usize)] = pixels[pol as usize] * sph
            }
        }
    }
}

fn compute_lm(x: u32, subgrid_size: u32, image_size: Float) -> Float {
    (x as Float + 0.5 - (subgrid_size as Float / 2.0)) * image_size / subgrid_size as Float
}

fn compute_n(l: Float, m: Float) -> Float {
    let tmp = l * l + m * m;

    if tmp >= 1.0 {
        return 1.0;
    }

    tmp / (1.0 + Float::sqrt(1.0 - tmp))
}

fn compute_pixels(
    timestep_count: u32,
    offset: u32,
    uvw: &UvwArray,
    baseline: u32,
    l: Float,
    m: Float,
    n: Float,
    u_offset: Float,
    v_offset: Float,
    w_offset: Float,
    channel_begin: u32,
    channel_end: u32,
    correlation_count_in: u32,
    correlation_count_out: u32,
    wavenumbers: &Array1<Float>,
    visibilities: &Array4<Complex>,
) -> Array1<Complex> {
    let mut pixels: Array1<Complex> = Array1::zeros(correlation_count_out as usize);

    for time in 0..timestep_count {
        let idx = offset + time;
        let Uvw { u, v, w } = uvw[(baseline as usize, idx as usize)];

        let phase_index = u * l + v * m + w * n;
        let phase_offset = u_offset * l + v_offset * m + w_offset * n;

        for channel in channel_begin..channel_end {
            let phase = phase_offset - (phase_index * wavenumbers[channel as usize]);
            let phasor = (Complex::i() * phase).exp();

            for pol in 0..correlation_count_in {
                pixels[(pol % correlation_count_out) as usize] += visibilities[(
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

fn compute_phasor(subgrid_size: u32) -> Array2<Complex> {
    Array2::<Complex>::from_shape_fn((subgrid_size as usize, subgrid_size as usize), |(x, y)| {
        let phase = PI * (x as Float + y as Float - subgrid_size as Float) / subgrid_size as Float;
        Complex::new(0.0, phase).exp()
    })
}

fn add_subgrid_to_grid(
    subgrid: ArrayView3<Complex>,
    metadata: &Metadata,
    mut grid: ArrayViewMut3<Complex>,
    phasor: ArrayView2<Complex>,
) {
    let grid_size = grid.shape()[1];
    let subgrid_size = subgrid.shape()[1];
    let correlation_count = subgrid.shape()[0];

    assert!(metadata.coordinate.x < (grid_size - subgrid_size) as u32);
    assert!(metadata.coordinate.y < (grid_size - subgrid_size) as u32);

    for y in 0..subgrid_size {
        for x in 0..subgrid_size {
            let x_src = (x + (subgrid_size / 2)) % subgrid_size;
            let y_src = (y + (subgrid_size / 2)) % subgrid_size;

            for p in 0..correlation_count {
                grid[(
                    p,
                    metadata.coordinate.y as usize + y,
                    metadata.coordinate.x as usize + x,
                )] += subgrid[(p, y_src, x_src)] * phasor[(y, x)];
            }
        }
    }
}

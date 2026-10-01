use crate::{constants::Float, types::UvwArray};

use hdf5_metno::H5Type;
use ndarray::prelude::*;

#[derive(Debug, PartialEq, Eq, H5Type)]
#[repr(C)]
pub struct Metadata {
    pub baseline: u32,
    pub time_index: u32,
    pub nr_timesteps: u32,
    pub channel_begin: u32,
    pub channel_end: u32,
    pub coordinate: Coordinate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, H5Type)]
#[repr(C)]
pub struct Coordinate {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

pub type MetadataArray = Array1<Metadata>;

pub trait MetadataArrayExtension {
    fn from_file(file: &hdf5_metno::File) -> Result<Self, hdf5_metno::Error>
    where
        Self: Sized;
    fn generate(grid_size: u32, subgrid_size: u32, channel_count: u32, uvw: &UvwArray) -> Self;
}

impl MetadataArrayExtension for MetadataArray {
    /// Compute metadata for all baselines.
    ///
    /// Returns a metadata array, shape (`subgrid_count`)
    fn generate(grid_size: u32, subgrid_size: u32, channel_count: u32, uvw: &UvwArray) -> Self {
        let max_group_size = 256; // TODO: Add this to CLI struct

        let u_pixels = uvw.mapv(|x| x.u);
        let v_pixels = uvw.mapv(|x| x.v);

        let baseline_count = uvw.shape()[0];

        let mut metadata = Vec::new();

        for baseline in 0..baseline_count {
            metadata.extend(compute_metadata(
                grid_size,
                subgrid_size,
                channel_count,
                baseline.try_into().unwrap(),
                &u_pixels.slice(s![baseline, ..]),
                &v_pixels.slice(s![baseline, ..]),
                max_group_size,
            ));
        }

        metadata.into()
    }

    /// Read metadata data from HDF5 file
    ///
    /// ## Parameters
    /// - `file`: The HDF5 File
    ///
    /// Returns a metadata array, shape (`subgrid_count`)
    fn from_file(file: &hdf5_metno::File) -> Result<Self, hdf5_metno::Error> {
        file.dataset("metadata")?.read()
    }
}

fn compute_metadata(
    grid_size: u32,
    subgrid_size: u32,
    channel_count: u32,
    baseline: u32,
    u_pixels: &ArrayView1<Float>,
    v_pixels: &ArrayView1<Float>,
    max_group_size: u32,
) -> Vec<Metadata> {
    let mut metadata = Vec::new();

    let timestep_count = u_pixels.shape()[0];
    let max_distance = 0.8 * subgrid_size as Float;

    let mut timestep = 0;
    while timestep < timestep_count {
        let current_u = u_pixels[timestep];
        let current_v = v_pixels[timestep];

        // Find the largest group of UVWs that fits in a subgrid.
        let mut group_size = 1;
        while (timestep + group_size < timestep_count)
            && ((group_size as u32) < max_group_size)
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

        let subgrid_x = (group_u as u32)
            .checked_sub(subgrid_size / 2)
            .expect("shouldn't underflow");
        let subgrid_y = (group_v as u32)
            .checked_sub(subgrid_size / 2)
            .expect("shouldn't underflow");
        assert!(
            subgrid_size <= grid_size,
            "subgrid shouldn't be larger than grid"
        );
        let subgrid_x = subgrid_x.clamp(0, grid_size - subgrid_size);
        let subgrid_y = subgrid_y.clamp(0, grid_size - subgrid_size);

        metadata.push(Metadata {
            baseline,
            time_index: timestep.try_into().unwrap(),
            nr_timesteps: group_size.try_into().unwrap(),
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

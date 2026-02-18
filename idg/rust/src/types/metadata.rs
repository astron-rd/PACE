use std::{io, path::Path};

use crate::{constants::Float, types::UvwArray};

use super::{check_for_extra_bytes, check_type_desc};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ndarray::prelude::*;
use ndarray_npy::{ReadDataError, ReadableElement, WritableElement};
use py_literal::Value;

#[derive(Debug, PartialEq, Eq)]
pub struct Metadata {
    pub baseline: u32,
    pub time_index: u32,
    pub timestep_count: u32,
    pub channel_begin: u32,
    pub channel_end: u32,
    pub coordinate: Coordinate,
}

impl ReadableElement for Metadata {
    fn read_to_end_exact_vec<R: io::Read>(
        mut reader: R,
        type_desc: &Value,
        len: usize,
    ) -> Result<Vec<Self>, ReadDataError> {
        check_type_desc(
            type_desc,
            "[('baseline', '<i4'), ('time_index', '<i4'), ('nr_timesteps', '<i4'), ('channel_begin', '<i4'), ('channel_end', '<i4'), ('coordinate', [('x', '<i4'), ('y', '<i4'), ('z', '<i4')])]",
        )?;

        let mut out = Vec::with_capacity(len);

        for _ in 0..len {
            let baseline = reader.read_u32::<LittleEndian>()?;
            let time_index = reader.read_u32::<LittleEndian>()?;
            let timestep_count = reader.read_u32::<LittleEndian>()?;
            let channel_begin = reader.read_u32::<LittleEndian>()?;
            let channel_end = reader.read_u32::<LittleEndian>()?;
            let coordinate_x = reader.read_u32::<LittleEndian>()?;
            let coordinate_y = reader.read_u32::<LittleEndian>()?;
            let coordinate_z = reader.read_u32::<LittleEndian>()?;

            out.push(Metadata {
                baseline,
                time_index,
                timestep_count,
                channel_begin,
                channel_end,
                coordinate: Coordinate {
                    x: coordinate_x,
                    y: coordinate_y,
                    z: coordinate_z,
                },
            });
        }

        check_for_extra_bytes(&mut reader)?;

        Ok(out)
    }
}

impl WritableElement for Metadata {
    fn type_descriptor() -> Value {
        Value::List(vec![
            Value::Tuple(vec![
                Value::String("baseline".into()),
                Value::String("<i4".into()),
            ]),
            Value::Tuple(vec![
                Value::String("time_index".into()),
                Value::String("<i4".into()),
            ]),
            Value::Tuple(vec![
                Value::String("nr_timesteps".into()),
                Value::String("<i4".into()),
            ]),
            Value::Tuple(vec![
                Value::String("channel_begin".into()),
                Value::String("<i4".into()),
            ]),
            Value::Tuple(vec![
                Value::String("channel_end".into()),
                Value::String("<i4".into()),
            ]),
            Value::Tuple(vec![
                Value::String("coordinate".into()),
                Value::List(vec![
                    Value::Tuple(vec![Value::String("x".into()), Value::String("<i4".into())]),
                    Value::Tuple(vec![Value::String("y".into()), Value::String("<i4".into())]),
                    Value::Tuple(vec![Value::String("z".into()), Value::String("<i4".into())]),
                ]),
            ]),
        ])
    }

    fn write<W: io::Write>(&self, mut writer: W) -> Result<(), ndarray_npy::WriteDataError> {
        writer.write_u32::<LittleEndian>(self.baseline)?;
        writer.write_u32::<LittleEndian>(self.time_index)?;
        writer.write_u32::<LittleEndian>(self.timestep_count)?;
        writer.write_u32::<LittleEndian>(self.channel_begin)?;
        writer.write_u32::<LittleEndian>(self.channel_end)?;
        writer.write_u32::<LittleEndian>(self.coordinate.x)?;
        writer.write_u32::<LittleEndian>(self.coordinate.y)?;
        writer.write_u32::<LittleEndian>(self.coordinate.z)?;

        Ok(())
    }

    fn write_slice<W: io::Write>(
        slice: &[Self],
        mut writer: W,
    ) -> Result<(), ndarray_npy::WriteDataError> {
        for item in slice {
            WritableElement::write(item, &mut writer)?;
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coordinate {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

pub type MetadataArray = Array1<Metadata>;

pub trait MetadataArrayExtension {
    fn generate(grid_size: u32, subgrid_size: u32, channel_count: u32, uvw: &UvwArray) -> Self;
    fn from_file(path: &Path) -> Result<Self, ndarray_npy::ReadNpyError>
    where
        Self: Sized;
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

    /// Read metadata data from npy file
    ///
    /// ## Parameters
    /// - `path`: Path to the npy file
    ///
    /// Returns a metadata array, shape (`subgrid_count`)
    fn from_file(path: &Path) -> Result<Self, ndarray_npy::ReadNpyError>
    where
        Self: Sized,
    {
        ndarray_npy::read_npy(path)
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

        // TODO: Add better explanation for what's happening with the group_size here
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

        let subgrid_x = group_u as u32 - (subgrid_size / 2);
        let subgrid_y = group_v as u32 - (subgrid_size / 2);
        let subgrid_x = subgrid_x.clamp(0, grid_size - subgrid_size);
        let subgrid_y = subgrid_y.clamp(0, grid_size - subgrid_size);

        metadata.push(Metadata {
            baseline,
            time_index: timestep.try_into().unwrap(),
            timestep_count: group_size.try_into().unwrap(),
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

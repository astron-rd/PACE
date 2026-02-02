use byteorder::{BigEndian, LittleEndian, ReadBytesExt, WriteBytesExt};
use ndarray_npy::{ReadDataError, ReadableElement, WritableElement};
use num_complex::Complex32;
use py_literal::Value;
use std::{fs::read, io, ops::Add};

use ndarray::Array2;
use num_traits::identities::Zero;

/// 3-dimensional vector with UVW parameters
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uvw {
    pub u: f32,
    pub v: f32,
    pub w: f32,
}

impl Uvw {
    pub fn new(u: f32, v: f32, w: f32) -> Self {
        Uvw { u, v, w }
    }
}

impl Add for Uvw {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Uvw {
            u: self.u + rhs.u,
            v: self.v + rhs.u,
            w: self.w + rhs.u,
        }
    }
}

impl Zero for Uvw {
    fn zero() -> Self {
        Self {
            u: 0.0,
            v: 0.0,
            w: 0.0,
        }
    }

    fn is_zero(&self) -> bool {
        self.u == 0.0 && self.v == 0.0 && self.w == 0.0
    }
}

impl ReadableElement for Uvw {
    fn read_to_end_exact_vec<R: std::io::Read>(
        mut reader: R,
        type_desc: &py_literal::Value,
        len: usize,
    ) -> Result<Vec<Self>, ReadDataError> {
        check_type_desc(type_desc, "[('u', '<f4'), ('v', '<f4'), ('w', '<f4')]")?;

        let mut out = Vec::with_capacity(len);

        for _ in 0..len {
            let u = reader.read_f32::<LittleEndian>()?;
            let v = reader.read_f32::<LittleEndian>()?;
            let w = reader.read_f32::<LittleEndian>()?;

            out.push(Uvw::new(u, v, w));
        }

        check_for_extra_bytes(&mut reader)?;

        Ok(out)
    }
}

impl WritableElement for Uvw {
    fn type_descriptor() -> Value {
        Value::List(vec![
            Value::Tuple(vec![Value::String("u".into()), Value::String("<f4".into())]),
            Value::Tuple(vec![Value::String("v".into()), Value::String("<f4".into())]),
            Value::Tuple(vec![Value::String("w".into()), Value::String("<f4".into())]),
        ])
    }

    fn write<W: io::Write>(&self, mut writer: W) -> Result<(), ndarray_npy::WriteDataError> {
        writer.write_f32::<LittleEndian>(self.u)?;
        writer.write_f32::<LittleEndian>(self.v)?;
        writer.write_f32::<LittleEndian>(self.w)?;
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

fn check_type_desc(type_desc: &Value, expected: &str) -> Result<(), ReadDataError> {
    // Is this cheating? Potentially.
    // Does it really matter in this context? I don't think so.
    let signature = format!("{}", type_desc);

    if signature == expected {
        Ok(())
    } else {
        Err(ReadDataError::WrongDescriptor(type_desc.clone()))
    }
}

/// Returns `Ok(_)` iff the `reader` had no more bytes on entry to this
/// function.
///
/// This function is taken from `ndarray-npy`, file `src/npy/elements/mod.rs`.
///
/// **Warning** This will consume the remainder of the reader.
fn check_for_extra_bytes<R: io::Read>(reader: &mut R) -> Result<(), ReadDataError> {
    let num_extra_bytes = reader.read_to_end(&mut Vec::new())?;
    if num_extra_bytes == 0 {
        Ok(())
    } else {
        Err(ReadDataError::ExtraBytes(num_extra_bytes))
    }
}

pub type UvwArray = Array2<Uvw>;

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

pub type Visibility = Complex32;

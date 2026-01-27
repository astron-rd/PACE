use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use ndarray_npy::{ReadDataError, ReadableElement};
use num_complex::Complex32;
use py_literal::Value;
use std::{fs::read, io, ops::Add};

use ndarray::Array2;
use num_traits::identities::Zero;

/// 3-dimensional vector with UVW parameters
#[derive(Clone, Copy, Debug)]
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
        check_type_desc(type_desc)?;
        
        let mut out = Vec::with_capacity(len);

        for _ in 0..len {
            let u = reader
                .read_f32::<LittleEndian>()
                .map_err(|_| ReadDataError::MissingData)?;
            let v = reader
                .read_f32::<LittleEndian>()
                .map_err(|_| ReadDataError::MissingData)?;
            let w = reader
                .read_f32::<LittleEndian>()
                .map_err(|_| ReadDataError::MissingData)?;

            out.push(Uvw::new(u, v, w));
        }

        check_for_extra_bytes(&mut reader)?;

        Ok(out)
    }
}

fn check_type_desc(type_desc: &Value) -> Result<(), ReadDataError> {
    // Is this cheating? Potentially.
    // Does it really matter in this context? I don't think so.
    let signature = format!("{}", type_desc);

    if signature == "[('u', '<f4'), ('v', '<f4'), ('w', '<f4')]" {
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

pub struct Metadata {
    pub baseline: u32,
    pub time_index: u32,
    pub timestep_count: u32,
    pub channel_begin: u32,
    pub channel_end: u32,
    pub coordinate: Coordinate,
}

#[derive(Clone, Copy)]
pub struct Coordinate {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

pub type Visibility = Complex32;

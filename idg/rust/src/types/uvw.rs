use std::{io, ops::Add};

use crate::{
    cli::Cli,
    constants::{Float, PI},
};

use super::{check_for_extra_bytes, check_type_desc};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ndarray::prelude::*;
use ndarray_npy::{ReadDataError, ReadableElement, WritableElement};
use ndarray_rand::{
    RandomExt,
    rand::{SeedableRng, rngs::StdRng},
    rand_distr::{Beta, Uniform},
};
use num_traits::Zero;
use py_literal::Value;

/// 3-dimensional vector with UVW parameters
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uvw {
    pub u: Float,
    pub v: Float,
    pub w: Float,
}

impl Uvw {
    pub fn new(u: Float, v: Float, w: Float) -> Self {
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
        #[cfg(not(feature = "f64"))]
        check_type_desc(type_desc, "[('u', '<f4'), ('v', '<f4'), ('w', '<f4')]")?;
        #[cfg(feature = "f64")]
        check_type_desc(type_desc, "[('u', '<f8'), ('v', '<f8'), ('w', '<f8')]")?;

        let mut out = Vec::with_capacity(len);

        for _ in 0..len {
            #[cfg(not(feature = "f64"))]
            {
                let u = reader.read_f32::<LittleEndian>()?;
                let v = reader.read_f32::<LittleEndian>()?;
                let w = reader.read_f32::<LittleEndian>()?;

                out.push(Uvw::new(u, v, w));
            }
            #[cfg(feature = "f64")]
            {
                let u = reader.read_f64::<LittleEndian>()?;
                let v = reader.read_f64::<LittleEndian>()?;
                let w = reader.read_f64::<LittleEndian>()?;

                out.push(Uvw::new(u, v, w));
            }
        }

        check_for_extra_bytes(&mut reader)?;

        Ok(out)
    }
}

impl WritableElement for Uvw {
    fn type_descriptor() -> Value {
        #[cfg(not(feature = "f64"))]
        return Value::List(vec![
            Value::Tuple(vec![Value::String("u".into()), Value::String("<f4".into())]),
            Value::Tuple(vec![Value::String("v".into()), Value::String("<f4".into())]),
            Value::Tuple(vec![Value::String("w".into()), Value::String("<f4".into())]),
        ]);
        #[cfg(feature = "f64")]
        return Value::List(vec![
            Value::Tuple(vec![Value::String("u".into()), Value::String("<f8".into())]),
            Value::Tuple(vec![Value::String("v".into()), Value::String("<f8".into())]),
            Value::Tuple(vec![Value::String("w".into()), Value::String("<f8".into())]),
        ]);
    }

    fn write<W: io::Write>(&self, mut writer: W) -> Result<(), ndarray_npy::WriteDataError> {
        #[cfg(not(feature = "f64"))]
        {
            writer.write_f32::<LittleEndian>(self.u)?;
            writer.write_f32::<LittleEndian>(self.v)?;
            writer.write_f32::<LittleEndian>(self.w)?;
        }
        #[cfg(feature = "f64")]
        {
            writer.write_f64::<LittleEndian>(self.u)?;
            writer.write_f64::<LittleEndian>(self.v)?;
            writer.write_f64::<LittleEndian>(self.w)?;
        }
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

pub type UvwArray = Array2<Uvw>;

pub trait UvwArrayExtension {
    fn generate(cli: &Cli) -> Self;
    fn from_file(path: &str) -> Result<Self, ndarray_npy::ReadNpyError>
    where
        Self: Sized;
}

impl UvwArrayExtension for UvwArray {
    /// Generate simulated UVW data
    ///
    ///  Returns a UVW array of size (`baseline_count` * `timestep_count`)
    fn generate(cli: &Cli) -> Self {
        let ellipticity = cli.ellipticity.unwrap_or(0.1);
        let seed = cli.random_seed.unwrap_or(2);
        let mut rng = StdRng::seed_from_u64(seed);

        // Initialize time_samples array with incrementing floats
        let time_samples = Array::from_iter(0..cli.timestep_count()).mapv(|x| x as Float);

        // Initialize uvw array with zeroes
        let mut uvw: UvwArray = UvwArray::zeros((
            cli.baseline_count().try_into().unwrap(),
            cli.timestep_count().try_into().unwrap(),
        ));

        let max_uv = 0.7 * (cli.grid_size / 2) as Float;

        // Generate baseline ratios with more short baselines (beta distribution)
        // Beta distribution with alpha=1, beta=3 peaks at 0 and decreases
        let beta_distribution = Beta::new(1.0, 3.0).expect("Should be a valid distribution.");
        let baseline_ratios =
            Array::random_using(cli.baseline_count() as usize, beta_distribution, &mut rng);

        // Generate random starting angles for each baseline
        let start_angles = Array::random_using(
            cli.baseline_count() as usize,
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
            let angular_velocity = (2.0 * PI) / (24 * 3600) as Float;

            // Generate UV coordinates with random starting angle
            let angle = start_angles[baseline] + angular_velocity * &time_samples;
            let u_coords = u_radius * angle.cos();
            let v_coords = v_radius * angle.sin();

            for t in 0..cli.timestep_count() as usize {
                uvw[(baseline, t)] = Uvw::new(
                    u_coords[t] + (cli.grid_size / 2) as Float,
                    v_coords[t] + (cli.grid_size / 2) as Float,
                    0.,
                );
            }
        }

        uvw
    }

    /// Read UVW data from npy file
    ///
    /// ## Parameters
    /// - `path`: Path to the npy file
    ///
    ///  Returns a UVW array of size (`baseline_count` * `timestep_count`)
    fn from_file(path: &str) -> Result<Self, ndarray_npy::ReadNpyError> {
        ndarray_npy::read_npy(path)
    }
}

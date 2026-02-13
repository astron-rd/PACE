#![allow(unused)]

pub const NR_CORRELATIONS_IN: u32 = 2; // XX, YY
pub const NR_CORRELATIONS_OUT: u32 = 1; // I

#[cfg(not(feature = "f64"))]
pub use f32::*;
#[cfg(feature = "f64")]
pub use f64::*;

mod f32 {
    use num_complex::Complex32;

    pub type Float = f32;
    pub type Complex = Complex32;

    pub const PI: f32 = std::f32::consts::PI;
    pub const W_STEP: f32 = 1.0; // w step in wavelengths
    pub const SPEED_OF_LIGHT: f32 = 299_792_458.0;
}

mod f64 {
    use num_complex::Complex64;

    pub type Float = f64;
    pub type Complex = Complex64;

    pub const PI: f64 = std::f64::consts::PI;
    pub const W_STEP: f64 = 1.0; // w step in wavelengths
    pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;
}

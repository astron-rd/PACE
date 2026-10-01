#[cfg(not(feature = "f64"))]
pub use f32::*;
#[cfg(not(feature = "f64"))]
mod f32 {
    use fftw::plan::C2CPlan32;
    use num_complex::Complex32;

    pub type Float = f32;
    pub type Complex = Complex32;
    pub type C2CPlanFloat = C2CPlan32;

    pub const PI: f32 = std::f32::consts::PI;
    pub const SPEED_OF_LIGHT: f32 = 299_792_458.0;
}

#[cfg(feature = "f64")]
pub use f64::*;
#[cfg(feature = "f64")]
mod f64 {
    use fftw::plan::C2CPlan64;
    use num_complex::Complex64;

    pub type Float = f64;
    pub type Complex = Complex64;
    pub type C2CPlanFloat = C2CPlan64;

    pub const PI: f64 = std::f64::consts::PI;
    pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;
}

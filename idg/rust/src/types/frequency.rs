use std::path::Path;

use ndarray::prelude::*;

use crate::constants::Float;

pub type FrequencyArray = Array1<Float>;

pub trait FrequencyArrayExtension {
    fn generate(start_frequency: Float, channel_count: u32, frequency_increment: Float) -> Self;
    fn from_file(path: &Path) -> Result<Self, ndarray_npy::ReadNpyError>
    where
        Self: Sized;
}

impl FrequencyArrayExtension for FrequencyArray {
    /// Generate array of frequencies for each channel.
    ///
    /// Returns frequencies array, shape (`channel_count`)
    fn generate(start_frequency: Float, channel_count: u32, frequency_increment: Float) -> Self {
        Array::range(
            start_frequency,
            start_frequency + (channel_count as Float * frequency_increment),
            frequency_increment,
        )
    }

    /// Read frequency data from npy file
    ///
    /// ## Parameters
    /// - `path`: Path to the npy file
    ///
    ///  Returns a frequency array, shape (`channel_count`)
    fn from_file(path: &Path) -> Result<Self, ndarray_npy::ReadNpyError>
    where
        Self: Sized,
    {
        ndarray_npy::read_npy(path)
    }
}

use ndarray::prelude::*;

use crate::{cli::Cli, constants::Float};

pub type FrequencyArray = Array1<Float>;

pub trait FrequencyArrayExtension {
    fn generate(cli: &Cli) -> Self;
    fn from_file(path: &str) -> Result<Self, ndarray_npy::ReadNpyError>
    where
        Self: Sized;
}

impl FrequencyArrayExtension for FrequencyArray {
    /// Generate array of frequencies for each channel.
    ///
    /// Returns frequencies array, shape (`channel_count`)
    fn generate(cli: &Cli) -> Self {
        Array::range(
            cli.start_frequency,
            cli.start_frequency + (cli.channel_count as Float * cli.frequency_increment),
            cli.frequency_increment,
        )
    }

    /// Read frequency data from npy file
    ///
    /// ## Parameters
    /// - `path`: Path to the npy file
    ///
    ///  Returns a frequency array, shape (`channel_count`)
    fn from_file(path: &str) -> Result<Self, ndarray_npy::ReadNpyError>
    where
        Self: Sized,
    {
        ndarray_npy::read_npy(path)
    }
}

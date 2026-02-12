use std::f32::consts::PI;

use ndarray::prelude::*;

use crate::{constants::SPEED_OF_LIGHT, types::FrequencyArray};

pub type WavenumberArray = Array1<f32>;

pub trait WavenumberArrayExtension {
    fn from_frequencies(frequencies: &FrequencyArray) -> Self;
}

impl WavenumberArrayExtension for WavenumberArray {
    fn from_frequencies(frequencies: &FrequencyArray) -> Self {
        (frequencies * 2.0 * PI) / SPEED_OF_LIGHT
    }
}

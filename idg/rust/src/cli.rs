use std::path::PathBuf;

use clap::Parser;

use crate::constants::{self, Float};

/// Command-line options
#[derive(Parser)]
#[command(version, about, long_about = Some("IDG is the Image Domain Gridder"))]
pub struct Cli {
    /// Size of the subgrid in pixels
    #[arg(long, default_value = "32")]
    pub subgrid_size: u32,

    /// Size of the grid in pixels
    #[arg(long, default_value = "1024")]
    pub grid_size: u32,

    /// Length of the observation in hours
    #[arg(long, default_value = "4.0")]
    pub observation_hours: Float,

    /// Number of frequency channels
    #[arg(long, default_value = "16")]
    pub channel_count: u32,

    /// Number of stations
    #[arg(long, default_value = "20")]
    pub station_count: u32,

    /// Starting frequency in hertz
    #[arg(long, default_value = "150e6")]
    pub start_frequency: Float,

    /// Frequency increment in hertz
    #[arg(long, default_value = "1e6")]
    pub frequency_increment: Float,

    /// Ellipticity for simulated UVW data
    #[arg(long)]
    pub ellipticity: Option<Float>,

    /// Random seed for RNG
    #[arg(long)]
    pub random_seed: Option<u64>,

    /// Output numpy data
    #[arg(long, default_value = "false", value_name = "OUTPUT_PATH")]
    pub numpy_output: Option<PathBuf>,

    /// Output timing data
    #[arg(long, default_value=None, value_name = "OUTPUT_PATH")]
    pub timing_output: Option<PathBuf>,
}

impl Cli {
    // Derived values:
    /// The number of timesteps, as derived from the observation time.
    pub fn timestep_count(&self) -> u32 {
        (self.observation_hours * 3600.0).floor() as u32
    }

    /// The end frequency, as derived from the start frequency, number of channels, and frequency increment
    pub fn end_frequency(&self) -> Float {
        self.start_frequency + (self.channel_count as Float * self.frequency_increment)
    }

    /// The image size, as derived from the end frequency
    pub fn image_size(&self) -> Float {
        constants::SPEED_OF_LIGHT / self.end_frequency()
    }

    /// The number of baselines, as derived from the number of stations
    pub fn baseline_count(&self) -> u32 {
        (self.station_count * (self.station_count - 1)) / 2
    }
}

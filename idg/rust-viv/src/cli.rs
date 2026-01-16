use std::path::PathBuf;

use clap::Parser;

use crate::constants;

/// Command-line options
#[derive(Parser)]
#[command(version, about, long_about = Some("IDG is the Image Domain Gridder"))]
pub struct Cli {
    /// Size of the subgrid in pixels
    #[arg(long, default_value = "32")]
    subgrid_size: usize,

    /// Size of the grid in pixels
    #[arg(long, default_value = "1024")]
    grid_size: usize,

    /// Length of the observation in hours
    #[arg(long, default_value = "4.0")]
    observation_hours: f32,

    /// Number of frequency channels
    #[arg(long, default_value = "16")]
    channel_count: usize,

    /// Number of stations
    #[arg(long, default_value = "20")]
    station_count: usize,

    /// Starting frequency in hertz
    #[arg(long, default_value = "150e6")]
    start_frequency: f64,

    /// Frequency increment in hertz
    #[arg(long, default_value = "1e6")]
    frequency_increment: f64,

    /// Output numpy data
    #[arg(long, default_value = "false", value_name = "OUTPUT_PATH")]
    numpy_output: Option<PathBuf>,

    /// Output timing data
    #[arg(long, default_value=None, value_name = "OUTPUT_PATH")]
    timing_output: Option<PathBuf>,
}

#[allow(unused)] // These functions are here for potential future use
impl Cli {
    // Derived values:
    /// The number of timesteps, as derived from the observation time.
    pub fn num_timesteps(&self) -> usize {
        (self.observation_hours * 3600.0).floor() as usize
    }

    /// The end frequency, as derived from the start frequency, number of channels, and frequency increment
    pub fn end_frequency(&self) -> f64 {
        self.start_frequency + (self.channel_count as f64 * self.frequency_increment)
    }

    /// The image size, as derived from the end frequency
    pub fn image_size(&self) -> f64 {
        constants::SPEED_OF_LIGHT / self.end_frequency()
    }

    /// The number of baselines, as derived from the number of stations
    pub fn baseline_count(&self) -> usize {
        (self.station_count * (self.station_count - 1)) / 2
    }
}

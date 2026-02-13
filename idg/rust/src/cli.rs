use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::constants::Float;

/// Command-line options
#[derive(Parser)]
#[command(version, about, long_about = Some("IDG is the Image Domain Gridder"))]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    #[arg(long, default_value = "1.0")]
    pub w_step: Float,

    /// Output numpy data
    #[arg(long, value_name = "OUTPUT_PATH")]
    pub numpy_output: Option<PathBuf>,

    /// Output timing data
    #[arg(long, default_value=None, value_name = "OUTPUT_PATH")]
    pub timing_output: Option<PathBuf>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Generate input data from scratch.
    Generate {
        /// Size of the subgrid in pixels
        #[arg(long, default_value = "32")]
        subgrid_size: u32,

        /// Size of the grid in pixels
        #[arg(long, default_value = "1024")]
        grid_size: u32,

        /// Length of the observation in hours
        #[arg(long, default_value = "4.0")]
        observation_hours: Float,

        /// Number of frequency channels
        #[arg(long, default_value = "16")]
        channel_count: u32,

        /// Number of stations
        #[arg(long, default_value = "20")]
        station_count: u32,

        /// Starting frequency in hertz
        #[arg(long, default_value = "150e6")]
        start_frequency: Float,

        /// Frequency increment in hertz
        #[arg(long, default_value = "1e6")]
        frequency_increment: Float,

        /// Ellipticity for simulated UVW data
        #[arg(long, default_value = "0.1")]
        ellipticity: Float,

        /// Number of point sources for simulated visibilities
        #[arg(long, default_value = "4")]
        point_sources_count: u32,

        /// Maximum pixel offset for simulated visibilities
        ///
        /// Defaults to `grid_size / 3`
        #[arg(long)]
        max_pixel_offset: Option<u32>,

        /// Number of correlations in
        #[arg(long, default_value = "2")]
        correlation_count_in: u32,

        /// Number of correlations out
        #[arg(long, default_value = "1")]
        correlation_count_out: u32,

        /// Random seed for RNG
        #[arg(long, default_value = "0")]
        random_seed: u64,
    },
    /// Load the input data from .npy files
    Load {
        /// Location of the data directory, defaults to current working directory.
        #[arg(long, short = 'd')]
        data_dir: Option<PathBuf>,

        // Location of the UVW file, relative to `data_dir`
        #[arg(long, default_value = "uvw.npy")]
        uvw_file: PathBuf,

        // Location of the frequencies file, relative to `data_dir`
        #[arg(long, default_value = "frequencies.npy")]
        frequencies_file: PathBuf,

        // Location of the visibilities file, relative to `data_dir`
        #[arg(long, default_value = "visibilities.npy")]
        visibilities_file: PathBuf,

        /// Size of the subgrid in pixels
        #[arg(long, default_value = "32")]
        subgrid_size: u32,
    },
}

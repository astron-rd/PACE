use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(version, about, long_about = Some("Fourier Domain Dedispersion"))]
pub struct Cli {
    #[command(flatten)]
    pub args: GeneralArgs,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Args)]
pub struct GeneralArgs {
    /// Path to the file containing the signal
    #[arg(long, short('f'), default_value = "signal.h5")]
    pub signal_file: PathBuf,

    /// Name for the file containing the output
    #[arg(long, short, default_value = "fdd.h5")]
    pub output_file: PathBuf,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Simulate a dispersed signal
    Simulate {
        #[command(flatten)]
        observation_args: ObservationArgs,

        #[command(flatten)]
        signal_args: SignalArgs,
    },
    /// Dedisperse a dispersed signal
    Dedisperse {
        #[command(flatten)]
        observation_args: ObservationArgs,

        #[command(flatten)]
        dedisp_args: DedispArgs,
    },
}

#[derive(Args)]
pub struct ObservationArgs {
    /// Duration of the observation in seconds
    #[arg(long, default_value = "30.0")]
    pub duration: f32,

    /// Duration of a sample in seconds
    #[arg(long, default_value = "250.0e-6")]
    pub sampling_period: f32,

    /// Maximum frequency in MHz
    #[arg(long, default_value = "1584.0")]
    pub max_frequency: f32,

    /// Bandwidth in MHz
    #[arg(long, default_value = "100.0")]
    pub bandwidth: f32,

    /// Number of channels
    #[arg(long, default_value = "1024")]
    pub channel_count: usize,
}

#[derive(Args)]
pub struct SignalArgs {
    /// RMS of the noise in the generated data
    #[arg(long, default_value = "25.0")]
    pub noise_rms: f32,

    /// Signal dispersion measure in pc cm^-3
    #[arg(long, default_value = "41.159")]
    pub dispersion_measure: f32,

    /// Arrival time of the pulse in seconds
    #[arg(long, default_value = "3.14")]
    pub arrival_time: f32,

    /// Amplitude of the signal
    #[arg(long, default_value = "25.0")]
    pub amplitude: f32,
}

#[derive(Args)]
pub struct DedispArgs {
    /// Start of dispersion measure search space in pc cm^-3
    #[arg(long, default_value = "2.0")]
    pub dm_start: f32,
    /// End of dispersion measure search space in pc cm^-3
    #[arg(long, default_value = "100.0")]
    pub dm_end: f32,

    /// Expected intrinsic width of the pulse in microseconds
    #[arg(long, default_value = "4.0")]
    pub pulse_width: f32,

    /// Smearing tolerance
    #[arg(long, default_value = "1.25")]
    pub tolerance: f32,
}

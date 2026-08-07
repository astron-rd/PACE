pub struct ObservationInfo {
    /// Duration of the observation in seconds
    pub duration: f32,
    /// Duration of a sample in seconds
    pub sampling_period: f32,
    /// Highest frequency in the observation in MHz
    pub peak_frequency: f32,
    /// Bandwidth in MHz
    pub bandwidth: f32,
    /// Number of channels
    pub channel_count: usize,
}

pub struct SignalInfo {
    /// RMS of the noise in the generated data
    pub noise_rms: f32,
    /// Signal dispersion measure in pc cm^-3
    pub dispersion_measure: f32,
    /// Arrival time of the pulse in seconds
    pub arrival_time: f32,
    /// Amplitude of the signal
    pub intensity: f32,
}

pub struct DedispersionConstraints {
    /// Start of dispersion measure search space in pc cm^-3
    pub dm_start: f32,
    /// End of dispersion measure search space in pc cm^-3
    pub dm_end: f32,

    /// Expected intrinsic width of the pulse in microseconds
    pub pulse_width: f32,

    /// Smearing tolerance
    pub tolerance: f32,
}

use clap::Parser;
use ndarray::Array2;
use ndarray_rand::RandomExt;

use crate::{
    metadata::{ObservationInfo, SignalInfo},
    util::time_function,
};

mod cli;
mod metadata;
mod util;

fn main() {
    let _cli = cli::Cli::parse();

    let observation = ObservationInfo {
        duration: 30.0,
        sampling_period: 250.0e-6,
        peak_frequency: 1581.0,
        bandwidth: 100.0,
        channel_count: 1024,
    };

    let signal_properties = SignalInfo {
        noise_rms: 25.0,
        dispersion_measure: 41.159,
        arrival_time: 3.14159,
        intensity: 25.0,
    };

    println!("Simulating a dispersed signal...");

    let signal = simulate_dispersed_signal(&signal_properties, &observation);

    let quantized_signal = time_function!("quantize signal", signal.map(quantize));

    time_function!("save signal to disk", {
        let output_file = hdf5_metno::File::create("signal.h5").unwrap();

        let builder = output_file.new_dataset_builder();
        builder
            .with_data(&quantized_signal)
            .create("signal")
            .unwrap();
    });
}

fn simulate_dispersed_signal(
    signal: &SignalInfo,
    observation: &ObservationInfo,
) -> ndarray::Array2<f32> {
    let frequency_resolution = -1.0 * observation.bandwidth / observation.channel_count as f32;
    let n_samples = observation.duration / observation.sampling_period;

    let shape = (n_samples as usize, observation.channel_count);
    let mut data = time_function!(
        "create background noise",
        Array2::random(shape, ndarray_rand::rand_distr::StandardNormal) * signal.noise_rms
    );

    time_function!("add dispersed signal", {
        for channel in 0..observation.channel_count {
            let a = 1.0 / (observation.peak_frequency + channel as f32 * frequency_resolution);
            let b = 1.0 / observation.peak_frequency;

            let channel_delay = signal.dispersion_measure * 4.15e3 * (a * a - b * b);
            let sample = (signal.arrival_time + channel_delay) / observation.sampling_period;

            data[(sample as usize, channel)] += signal.intensity;
        }
    });

    data
}

fn quantize(value_in: &f32) -> u8 {
    let value = value_in + 127.5;
    if value > 255.0 {
        255
    } else if value < 0.0 {
        0
    } else {
        value.round() as u8
    }
}

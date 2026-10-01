use ndarray::Array2;
use ndarray_rand::RandomExt;

use crate::cli::{GeneralArgs, ObservationArgs, SignalArgs};

use crate::util::time_function;

/// Simulate a dispersed signal and write it to disk.
pub fn simulate(
    general_args: &GeneralArgs,
    observation_args: &ObservationArgs,
    signal_args: &SignalArgs,
) {
    println!("Simulating a dispersed signal...");

    let signal = simulate_dispersed_signal(observation_args, signal_args);

    let quantized_signal = time_function!("quantize signal", signal.map(|x| quantize(*x)));

    time_function!("save signal to disk", {
        let output_file = hdf5_metno::File::create(&general_args.signal_file).unwrap();

        let builder = output_file.new_dataset_builder();
        builder
            .with_data(&quantized_signal)
            .create("dynspec")
            .unwrap();
    });
}

/// Simulate a dispersed signal based on the observation and signal args.
fn simulate_dispersed_signal(
    observation: &ObservationArgs,
    signal: &SignalArgs,
) -> ndarray::Array2<f32> {
    let frequency_resolution = observation.bandwidth / observation.channel_count as f32;
    let n_samples = observation.duration / observation.sampling_period;

    let shape = (n_samples as usize, observation.channel_count);
    let mut data = time_function!(
        "create background noise",
        Array2::random(shape, ndarray_rand::rand_distr::StandardNormal) * signal.noise_rms
    );

    time_function!("add dispersed signal", {
        for channel in 0..observation.channel_count {
            let a = 1.0 / (observation.max_frequency - channel as f32 * frequency_resolution);
            let b = 1.0 / observation.max_frequency;

            let channel_delay = signal.dispersion_measure * 4.15e3 * (a * a - b * b);
            let sample = (signal.arrival_time + channel_delay) / observation.sampling_period;

            data[(sample as usize, channel)] += signal.amplitude;
        }
    });

    data
}

/// Quantize the value of the `f32` into a `u8`.
///
/// Maps the value from -127.5 - 127.5 to 0 - 255.
fn quantize(value_in: f32) -> u8 {
    let value = value_in + 127.5;
    if value > 255.0 {
        255
    } else if value < 0.0 {
        0
    } else {
        value.round() as u8
    }
}

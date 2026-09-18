use std::f32::consts::PI;

use ndarray::prelude::*;
use ndarray_ndimage::PadMode::Constant;
use ndrustfft::R2cFftHandler;
use num_complex::{Complex, Complex32};

use crate::cli::{DedispArgs, GeneralArgs, ObservationArgs};
use crate::util::time_function;

pub fn dedisperse(
    general_args: &GeneralArgs,
    observation_args: &ObservationArgs,
    dedisp_args: &DedispArgs,
) {
    let signal_file = hdf5_metno::File::open(&general_args.signal_file).unwrap();
    let signal: Array2<u8> = time_function!(
        "read input",
        signal_file
            .dataset("dynspec")
            .expect("dynspec dataset should be in input file")
            .read()
            .unwrap()
    );

    let frequency_resolution =
        -1.0 * observation_args.bandwidth / observation_args.channel_count as f32;
    let mut plan = time_function!(
        "create FDDPlan",
        FDDPlan::new(
            observation_args.channel_count,
            observation_args.sampling_period,
            observation_args.max_frequency,
            frequency_resolution,
        )
    );

    time_function!(
        "generate dm list",
        plan.generate_dm_list(
            dedisp_args.dm_start,
            dedisp_args.dm_end,
            dedisp_args.pulse_width,
            dedisp_args.tolerance
        )
    );

    let output = plan.execute(signal);

    let output_file = hdf5_metno::File::create(&general_args.output_file).unwrap();
    let fdd_result_ds = output_file
        .new_dataset_builder()
        .with_data(&output)
        .create("fddresult")
        .unwrap();
    fdd_result_ds
        .new_attr_builder()
        .with_data(&plan.dm_table)
        .create("dispersion_measures")
        .unwrap();
    fdd_result_ds
        .new_attr_builder()
        .with_data(&ndarray::arr0(output.shape()[0]))
        .create("computed_samples")
        .unwrap();
    fdd_result_ds
        .new_attr_builder()
        .with_data(&ndarray::arr0(plan.time_resolution))
        .create("integration_time")
        .unwrap();
}

pub struct FDDPlan {
    dm_count: usize,
    channel_count: usize,
    max_delay: usize,

    time_resolution: f32,
    max_frequency: f32,
    frequency_resolution: f32,

    dm_table: Array1<f32>,
    delay_table: Array1<f32>,
}

impl FDDPlan {
    fn new(
        channel_count: usize,
        time_resolution: f32,
        max_frequency: f32,
        frequency_resolution: f32,
    ) -> Self {
        let mut zelf = Self {
            dm_count: 0,
            channel_count,
            max_delay: 0,
            time_resolution,
            max_frequency,
            frequency_resolution,
            dm_table: Default::default(),
            delay_table: Default::default(),
        };

        zelf.generate_delay_table();

        zelf
    }

    fn generate_delay_table(&mut self) {
        const MYSTERIOUS_MAGIC_CONSTANT: f32 = 4.148741601e3;

        self.delay_table = Array1::from_iter((0..self.channel_count).map(|channel| {
            let inverse_channel_frequency =
                1.0 / (self.max_frequency + channel as f32 * self.frequency_resolution);
            let inverse_max_frequency = 1.0 / self.max_frequency;

            MYSTERIOUS_MAGIC_CONSTANT / self.time_resolution
                * (inverse_channel_frequency.powi(2) - inverse_max_frequency.powi(2))
        }));
    }

    fn generate_dm_list(&mut self, dm_start: f32, dm_end: f32, pulse_width: f32, tolerance: f32) {
        let time_resolution = self.time_resolution as f64 * 1e6;
        let f = (self.max_frequency as f64
            + ((self.channel_count / 2) as f64 - 0.5) * self.frequency_resolution as f64)
            * 1e-3;
        let a = 8.3 * self.frequency_resolution as f64 / f.powi(3);
        let a_squared = a.powi(2);
        let b_squared = a_squared * (self.channel_count.pow(2) / 16) as f64;
        let tolerance_squared = (tolerance as f64).powi(2);
        let c =
            (time_resolution.powi(2) + (pulse_width as f64).powi(2)) * (tolerance_squared - 1.0);

        let mut dm_table = vec![dm_start];
        while *dm_table.last().unwrap() < dm_end {
            let previous_dm = *dm_table.last().unwrap() as f64;
            let previous_dm_squared = previous_dm.powi(2);
            let k = c + tolerance_squared * a_squared * previous_dm_squared;
            let dm = (b_squared * previous_dm
                + (-a_squared * b_squared * previous_dm_squared + (a_squared + b_squared) * k)
                    .sqrt())
                / (a_squared + b_squared);
            dm_table.push(dm as f32);
        }

        self.dm_table = Array1::from_vec(dm_table);
        self.dm_count = self.dm_table.len();
        self.max_delay =
            (self.dm_table.last().unwrap() * self.delay_table.last().unwrap() + 0.5) as usize
    }

    fn execute(&self, spectrum: Array2<u8>) -> Array2<f32> {
        let n_samples = spectrum.shape()[0];
        let n_spin_frequencies = n_samples / 2 + 1;
        let n_output_samples = n_samples - self.max_delay;

        let n_samples_fft = (n_samples + 1).next_multiple_of(16384);
        let n_samples_padded = (n_samples_fft + 1).next_multiple_of(1024);
        let n_fft_frequency_bins = n_samples_padded / 2 + 1;

        println!("(1) Generate the spin frequency table.");

        let observation_duration = n_samples as f32 * self.time_resolution;
        let spin_frequency_table: Array1<f32> = (0..n_spin_frequencies)
            .into_iter()
            .map(|i| i as f32 / observation_duration)
            .collect();

        println!("(2) Transpose data: int -> float.");

        let padding = n_samples_padded - n_samples;
        let padded_spectrum = ndarray_ndimage::pad(&spectrum, &[[0, padding], [0, 0]], Constant(0));
        let mut transposed_spectrum = padded_spectrum.map(|x| *x as f32);
        transposed_spectrum.reverse_axes();
        transposed_spectrum -= 127.5;
        transposed_spectrum /= self.channel_count as f32;

        let mut fd_scratch = Array2::zeros((self.channel_count, n_fft_frequency_bins));
        time_function!(
            "fft real->complex",
            ndrustfft::ndfft_r2c_par(
                &transposed_spectrum,
                &mut fd_scratch,
                &R2cFftHandler::new(n_samples_padded),
                1,
            )
        );

        let mut dm_scratch: Array2<Complex<f32>> =
            Array2::zeros((self.dm_count, fd_scratch.shape()[1]));

        time_function!(
            "dedisperse",
            Self::fourier_domain_dedispersion(
                &fd_scratch,
                &mut dm_scratch,
                self.time_resolution,
                &spin_frequency_table,
                &self.dm_table,
                &self.delay_table,
            )
        );

        let mut dm_output = Array2::zeros((self.dm_count, n_samples_padded));
        ndrustfft::ndifft_r2c_par(
            &dm_scratch,
            &mut dm_output,
            &R2cFftHandler::new(n_samples_padded),
            1,
        );

        dm_output
            .slice(s![.., ..n_output_samples])
            .reversed_axes()
            .to_owned()
    }

    fn fourier_domain_dedispersion(
        input_data: &Array2<Complex<f32>>,
        output_data: &mut Array2<Complex<f32>>,
        time_resolution: f32,
        spin_frequencies: &Array1<f32>,
        dispersion_measures: &Array1<f32>,
        delays: &Array1<f32>,
    ) {
        let n_spin_frequencies = spin_frequencies.len();
        let samples = input_data.slice(s![.., ..n_spin_frequencies]);
        let spin_freqs_m = spin_frequencies.slice(s![NewAxis, ..]);

        assert_eq!(dispersion_measures.len(), output_data.shape()[0]);

        let output_iter = output_data.axis_iter_mut(Axis(0));

        ndarray::Zip::from(dispersion_measures)
            .and(output_iter)
            .par_for_each(|dm, mut output_axis| {
                let dm_delays = delays * (dm * time_resolution);

                let dm_delays_m = dm_delays.slice(s![.., NewAxis]);
                let phases = 2.0 * PI * (&dm_delays_m * &spin_freqs_m);

                let phasors = phases.map(|x| Complex::new(0.0, *x).exp());

                let result = (&samples * &phasors.view()).sum_axis(Axis(0));

                output_axis
                    .slice_mut(s![..n_spin_frequencies])
                    .assign(&result);
            });
    }
}

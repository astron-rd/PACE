use std::f32::consts::PI;

use ndarray::prelude::*;
use num_complex::Complex;

/// Perform Fourier Domain Dedispersion
pub fn fourier_domain_dedispersion(
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

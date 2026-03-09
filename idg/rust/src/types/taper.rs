use std::path::Path;

use ndarray::{linspace, prelude::*};

use crate::constants::Float;

pub type Taper = Array2<Float>;

pub trait TaperExtension {
    fn generate(subgrid_size: u32) -> Self;
    fn from_file(path: &Path) -> Result<Self, ndarray_npy::ReadNpyError>
    where
        Self: Sized;
}

impl TaperExtension for Taper {
    fn generate(subgrid_size: u32) -> Self {
        let x: Array1<Float> = linspace::<_, Float>(-1.0..1.0, subgrid_size as usize)
            .map(Float::abs)
            .collect();
        let spheroidal = x.map(|x| evaluate_spheroidal(*x));

        let mat_1n = spheroidal
            .to_shape((1, spheroidal.len()))
            .expect("1D array should fit in 1xN matrix.");
        let mat_n1 = mat_1n.clone().reversed_axes();

        (mat_1n * mat_n1).to_owned()
    }

    fn from_file(path: &Path) -> Result<Self, ndarray_npy::ReadNpyError> {
        ndarray_npy::read_npy(path)
    }
}

fn evaluate_spheroidal(x: Float) -> Float {
    #[rustfmt::skip]
    const P: [[Float; 5]; 2] = [
        [8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
        [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2],
    ];
    #[rustfmt::skip]
    const Q: [[Float; 3]; 2] = [
        [1.0000000e0, 8.212018e-1, 2.078043e-1],
        [1.0000000e0, 9.599102e-1, 2.918724e-1],
    ];

    let (part, end): (usize, Float) = match x {
        0.0..0.75 => (0, 0.75),
        0.75..=1.0 => (1, 1.0),
        _ => return 0.0,
    };

    let x_squared = x.powi(2);
    let delta_x_squared = x_squared - end.powi(2);
    let top = evaluate_polynomial(delta_x_squared, &P[part]);
    let btm = evaluate_polynomial(delta_x_squared, &Q[part]);

    if btm == 0.0 {
        0.0
    } else {
        (1.0 - x_squared) * (top / btm)
    }
}

fn evaluate_polynomial(x: Float, coefficients: &[Float]) -> Float {
    let mut val = coefficients[0];
    let mut x_accumulator = x;
    for p in coefficients.iter().skip(1) {
        val += p * x_accumulator;
        x_accumulator *= x;
    }
    val
}

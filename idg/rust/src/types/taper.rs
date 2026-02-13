use ndarray::{linspace, prelude::*};

use crate::constants::Float;

pub type Taper = Array2<Float>;

pub trait TaperExtension {
    fn generate(subgrid_size: u32) -> Self;
}

impl TaperExtension for Taper {
    fn generate(subgrid_size: u32) -> Self {
        let x: Array1<Float> = linspace::<_, Float>(-1.0..1.0, subgrid_size as usize)
            .map(|x| x.abs())
            .collect();
        let spheroidal = x.map(|x| evaluate_spheroidal(*x));

        let mat_1n = spheroidal
            .to_shape((1, spheroidal.len()))
            .expect("1D array should fit in 1xN matrix.");
        let mat_n1 = mat_1n.clone().reversed_axes();

        (mat_1n * mat_n1).to_owned()
    }
}

fn evaluate_spheroidal(x: Float) -> Float {
    #[rustfmt::skip]
    let p: [[Float; 5]; 2] = [
        [8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
        [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2],
    ];
    #[rustfmt::skip]
    let q: [[Float; 3]; 2] = [
        [1.0000000e0, 8.212018e-1, 2.078043e-1],
        [1.0000000e0, 9.599102e-1, 2.918724e-1],
    ];

    let (part, end): (usize, Float) = match x {
        0.0..0.75 => (0, 0.75),
        0.75..=1.0 => (1, 1.0),
        _ => return 0.0,
    };

    // TODO: This bit is kinda ugly, might be able to use some cleaning up
    // TODO: Potentially split off `evaluate_polynomial` function
    let x_squared = x.powi(2);
    let del_x_squared = x_squared - end.powi(2);
    let mut del_x_squared_pow = del_x_squared;
    let mut top = p[part][0];
    for p in p[part].iter().skip(1) {
        top += p * del_x_squared_pow;
        del_x_squared_pow *= del_x_squared;
    }

    let mut btm = q[part][0];
    del_x_squared_pow = del_x_squared;
    for q in q[part].iter().skip(1) {
        btm += q * del_x_squared_pow;
        del_x_squared_pow *= del_x_squared;
    }

    if btm == 0.0 {
        0.0
    } else {
        (1.0 - x_squared) * (top / btm)
    }
}

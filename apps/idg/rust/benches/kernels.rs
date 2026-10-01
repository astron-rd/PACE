use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::linspace;

use idg::types::evaluate_spheroidal;

fn bench_evaluate_spheroidal(criterion: &mut Criterion) {
    let nu: Vec<f32> = linspace(0.0..=1.0, 1024).collect();
    let mut out: Vec<f32> = vec![0.0; 1024];
    criterion.bench_function("evaluate_spheroidal", |bencher| {
        bencher.iter(|| {
            out.iter_mut()
                .zip(&nu)
                .for_each(|(result, &input)| *result = evaluate_spheroidal(black_box(input)))
        })
    });
}

criterion_group!(benches, bench_evaluate_spheroidal);
criterion_main!(benches);

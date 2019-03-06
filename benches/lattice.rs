#![allow(non_snake_case)]

use ::criterion::{criterion_group, criterion_main, Criterion};

use ::ising_lib::prelude::*;

// NOTE
// To keep the results consistent, always run set lattice size to 50.

fn bench_calculate_flip_probability(c: &mut Criterion) {
    let lattice = Lattice::new(50);

    c.bench_function("calculate flip probability", move |b| {
        b.iter(|| {
            let E_diff = lattice.measure_E_diff((10, 42), 1.0);
            let probability = calc_flip_probability(E_diff, 1.0, 1.0);
        })
    });
}

fn bench_measure_E(c: &mut Criterion) {
    let lattice = Lattice::new(50);

    c.bench_function("calculate E", move |b| {
        let E = lattice.measure_E(1.0);
    });
}

criterion_group!(benches, bench_calculate_flip_probability, bench_calculate_E);

criterion_main!(benches);

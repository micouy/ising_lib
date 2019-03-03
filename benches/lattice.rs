#![allow(non_snake_case)]

use ::criterion::{criterion_group, criterion_main, Criterion};

use ::ising_lib::prelude::*;

fn bench_generate_random_spin_index(c: &mut Criterion) {
    let mut lattice = Lattice::new(50);

    c.bench_function("generate random spin index", move |b| {
        b.iter(|| lattice.gen_random_index())
    });
}

fn bench_calculate_flip_probability(c: &mut Criterion) {
    let lattice = Lattice::new(50);

    c.bench_function("calculate flip probability", move |b| {
        b.iter(|| {
            let E_diff = lattice.measure_E_diff((10, 42), 1.0);
            let probability = calc_flip_probability(E_diff, 1.0, 1.0);
        })
    });
}

criterion_group!(
    benches,
    bench_generate_random_spin_index,
    bench_calculate_flip_probability,
);
criterion_main!(benches);

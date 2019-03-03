use ::criterion::{criterion_group, criterion_main, Criterion};

use ::ising_lib::prelude::*;

fn bench_generate_random_spin_index(c: &mut Criterion) {
    let mut lattice = Lattice::new(50);

    c.bench_function("generate random spin index", move |b| {
        b.iter(|| lattice.gen_random_index())
    });
}

fn bench_gen_neighbor_indices(c: &mut Criterion) {
    let mut lattice = Lattice::new(50);

    c.bench_function("generate neighbor indices", move |b| {
        b.iter(|| lattice.gen_neighbor_indices((420, 42), 1))
    });
}

criterion_group!(
    benches,
    bench_generate_random_spin_index,
    bench_gen_neighbor_indices
);
criterion_main!(benches);

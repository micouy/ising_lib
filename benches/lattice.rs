use ::criterion::{Criterion, criterion_group, criterion_main};

use ::ising_lib::prelude::*;

fn bench_generate_random_spin_index(c: &mut Criterion) {
    let mut lattice = Lattice::new(50);

    c.bench_function("generate random spin index", move |b| b.iter(|| lattice.gen_random_index()));
}

criterion_group!(benches, bench_generate_random_spin_index);
criterion_main!(benches);

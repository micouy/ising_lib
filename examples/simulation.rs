#![allow(non_snake_case)]

use ::humantime::format_rfc3339;
use ::pbr::ProgressBar;
use ::rand::prelude::*;

use std::{fs::OpenOptions, io::prelude::*, time::SystemTime};

use ising_lib::prelude::*;

struct Params {
    T_range: (f64, f64),
    flips_to_skip: usize,
    measurements_per_T: usize,
    flips_per_measurement: usize,
    attempts_per_flip: usize,
    lattice_size: usize,
    J: f64,
    K: f64,
}

struct Record {
    T: f64,
    dE: f64,
    I: f64,
    X: f64,
}

fn main() {
    let size = 50;
    let params = Params {
        // the phase transition occurs at ~2.29
        T_range: (0.2, 4.0),
        // allow the spin lattice to "cool down"
        flips_to_skip: 10_000,
        // the more measurements taken at each T, the more precise the results
        // will be
        measurements_per_T: 1_000,
        // just a rule of thumb
        flips_per_measurement: size * size,
        attempts_per_flip: 50,
        lattice_size: size,
        J: 1.0,
        K: 1.0,
    };

    let mut rng = thread_rng();
    let mut lattice = Lattice::new(params.lattice_size);
    let Ts = TRange::new_step(params.T_range.0, params.T_range.1, 0.1)
        .collect::<Vec<f64>>();

    for _ in 0..params.flips_to_skip {
        for _ in 0..params.attempts_per_flip {
            let ix = lattice.gen_random_index();
            let E_diff = lattice.measure_E_diff(ix, params.J);
            let probability =
                calc_flip_probability(E_diff, params.T_range.0, params.K);

            if probability < rng.gen() {
                lattice.flip_spin(ix);

                break;
            }
        }
    }

    let mut pb = ProgressBar::new((params.measurements_per_T * Ts.len()) as u64);
    pb.set_width(Some(80));
    pb.show_message = true;

    let mut records: Vec<Record> = Ts.into_iter()
        .map(|T| {
            pb.message(&format!("At T: {:.2}...", T));

            let mut Es = vec![];
            let mut Is = vec![];

            for _ in 0..params.measurements_per_T {
                for _ in 0..params.flips_per_measurement {
                    for _ in 0..params.attempts_per_flip {
                        let ix = lattice.gen_random_index();
                        let E_diff = lattice.measure_E_diff(ix, params.J);
                        let probability =
                            calc_flip_probability(E_diff, T, params.K);

                        if probability > rng.gen() {
                            lattice.flip_spin(ix);

                            break;
                        }
                    }
                }

                Es.push(lattice.measure_E(params.J));
                Is.push(lattice.measure_I());
                pb.inc();
            }

            let dE = calc_dE(&Es, T);
            let I = Is.iter().sum::<f64>() / Is.len() as f64;
            let X = calc_X(&Es);

            Record { T, dE, I, X }
        })
        .collect();

    pb.finish_print("Finished taking measurements!");

    records.sort_by(|a, b| {
        a.T.partial_cmp(&b.T).unwrap_or(std::cmp::Ordering::Less)
    });

    let contents = {
        let headers = format!("{:>5}{:>30}{:>15}{:>20}", "T", "dE", "I", "X");
        let results = records
            .iter()
            .map(|r| {
                format!("{:>5.2}{:>30.5}{:>15.5}{:>20.10}", r.T, r.dE, r.I, r.X)
            })
            .collect::<Vec<String>>()
            .join("\n");
        let mut contents = vec![headers, results].join("\n");
        contents.push_str("\n");

        contents
    };

    let path = {
        let dir = "results";

        format!("{}/results-{}.txt", dir, format_rfc3339(SystemTime::now()))
    };

    let mut file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(path)
        .unwrap();
    file.write_all(contents.as_bytes()).unwrap();
}

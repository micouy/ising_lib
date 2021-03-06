//! Everything you need to run [Ising model][0] simulation.  Despite its
//! simplicity, the simulation allows us to observe an interesting
//! physical phenomenon - phase transition. [`ndarray`][ndarray]'s arrays
//! are used to store the spin lattice and perform computations. The lattice can
//! either be generated by [`rand`][rand]s RNG or be provided by the user.
//!
//! # Notes on consistency
//! * Spin indices are always `(usize, usize)`.
//! * Spin values are stored using [`i32`], which is the [fastest][1] integer
//!   type.
//! * Energy (`E`), energy fluctuations (`dE`), magnetization (`I`) and magnetic
//!   susceptibility (`X`) are [`f64`].
//! * Variables representing physical properties are denoted using non-snake
//!   case.
//!
//! [0]: https://en.wikipedia.org/wiki/Ising_model
//! [1]: https://doc.rust-lang.org/book/ch03-02-data-types.html#integer-types

#![deny(missing_docs)]
#![allow(non_snake_case)]

pub mod calculations;
pub mod lattice;
pub mod prelude;

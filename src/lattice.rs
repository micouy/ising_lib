//! Stuff related to spin lattice.

use ::ndarray::prelude::*;
use ::rand::prelude::*;

/// A struct encapsulating the spin lattice and all the operations performed
/// on it.
///
/// The lattice behaves like a torus - spins on opposite edges are considered
/// each other's neighbors.
pub struct Lattice<R: RngCore> {
    dims: (usize, usize),
    n_of_spins: i32,
    rng: R,
    inner: Array2<i32>,
    neighbors: Array2<[(usize, usize); 4]>,
}

impl<R: RngCore> Lattice<R> {
    /// Create a new lattice from provided RNG with randomly generated spins.
    pub fn from_rng(size: (usize, usize), mut rng: R) -> Self {
        let inner =
            Array2::from_shape_fn(size, |_| *[-1, 1].choose(&mut rng).unwrap());

        Self::from_array_rng(inner, rng)
    }

    /// View inner array.
    pub fn inner(&self) -> ndarray::ArrayView<i32, ndarray::Dim<[ndarray::Ix; 2]>> {
        self.inner.view()
    }

    /// Create a new lattice from provided array and RNG.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<std::error::Error>> {
    /// # use ::ndarray::prelude::*;
    /// # use ::rand::prelude::*;
    /// # use ising_lib::prelude::*;
    /// let array = Array::from_shape_vec((2, 2), vec![1, -1, 1, -1])?;
    /// let rng = SmallRng::from_entropy();
    /// let lattice = Lattice::from_array_rng(array, rng);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// The function will panic if or if any of the spins has incorrect value
    /// (neither `-1` nor `1`).
    ///
    /// ```should_panic
    /// # fn main() -> Result<(), Box<std::error::Error>> {
    /// # use ::ndarray::prelude::*;
    /// # use ::rand::prelude::*;
    /// # use ising_lib::prelude::*;
    /// let array = Array::from_shape_vec((2, 2), vec![5, -1, 1, -1])?;
    /// //                                             ↑ incorrect spin value
    /// let rng = SmallRng::from_entropy();
    /// let lattice = Lattice::from_array_rng(array, rng);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_array_rng(array: Array2<i32>, rng: R) -> Self {
        assert!(
            array.iter().all(|spin| *spin == 1 || *spin == -1),
            "Invalid spin value."
        );

        let roll_index = |ix: usize, amt: i32, max: usize| {
            let max = max as i32;

            ((ix as i32 + amt + max) % max) as usize
        };

        let (width, height) = array.dim();

        let neighbors = Array2::from_shape_fn((width, height), |ix| {
            [
                (roll_index(ix.0, 1, width), ix.1),   // right
                (ix.0, roll_index(ix.1, 1, height)),  // bottom
                (roll_index(ix.0, -1, width), ix.1),  // left
                (ix.0, roll_index(ix.1, -1, height)), // top
            ]
        });

        Lattice {
            dims: (width, height),
            inner: array,
            n_of_spins: width as i32 * height as i32,
            rng,
            neighbors,
        }
    }

    /// Return lattice's dimensions.
    pub fn dims(&self) -> (usize, usize) {
        self.dims
    }

    /// Return the product of the `(ith, jth)` spin and the sum of all of its
    /// neighbors.
    fn spin_times_all_neighbors(&self, ix: (usize, usize)) -> i32 {
        self.inner[ix]
            * self.neighbors[ix]
                .iter()
                .map(|n_ix| self.inner[*n_ix])
                .sum::<i32>()
    }

    /// Return the product of the `(ith, jth)` spin and the sum of two of its
    /// neighbors (the right one and the bottom one).
    fn spin_times_two_neighbors(&self, ix: (usize, usize)) -> i32 {
        self.inner[ix]
            * self.neighbors[ix][0..2]
                .iter()
                .map(|n_ix| self.inner[*n_ix])
                .sum::<i32>()
    }

    /// Return the difference of energy that would be caused by
    /// flipping the `(ith, jth)` spin without actually doing it.
    /// Used to determine the probability of a flip.
    ///
    /// ```text
    /// Lattice before flip:     Lattice after flip:
    /// ##| a|##                 ##| a|##
    /// --------                 --------
    ///  b| s| c                  b|-s| c
    /// --------                 --------
    /// ##| d|##                 ##| d|##
    ///
    /// E_2 - E_1 =
    ///  = ((-J) * (-s) * (a + b + c + d)) - ((-J) * s * (a + b + c + d)) =
    ///  = -J * ((-s) - s) * (a + b + c + d) =
    ///  = -J * -2 * s * (a + b + c + d) =
    ///  = 2 * J * s * (a + b + c + d)
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if the index is out of bounds.
    ///
    /// ```should_panic
    /// # use ising_lib::prelude::*;
    /// let lattice = Lattice::new((10, 10));
    /// let _ = lattice.measure_E_diff((42, 0));
    /// ```
    pub fn measure_E_diff(&self, ix: (usize, usize)) -> f64 {
        2.0 * f64::from(self.spin_times_all_neighbors(ix))
    }

    /// Return the difference of energy that would be caused by
    /// flipping the `(ith, jth)` spin in the presence of an external magnetic field without actually doing it.
    /// Used to determine the probability of a flip.
    ///
    /// ```text
    /// Lattice:	External magnetic field:
    /// ##| a|##	##|##|##
    /// --------	--------
    ///  b| s| c	##| h|##
    /// --------	--------
    /// ##| d|##	##|##|##
    ///
    /// E_2 - E_1 =
    ///  = ((-J) * (-s) * (a + b + c + d) - h * (-s)) - ((-J) * s * (a + b + c + d) - h * s) =
    ///  = ((-s) - s) * (-J) * (a + b + c + d) + ((-s) - s) * (-h) =
    ///  = -2 * s * ((-J) * (a + b + c + d) - h) =
    ///  = 2 * s * (J * (a + b + c + d) + h)
    ///  ```
    ///
    /// # Panics
    ///
    /// This function will panic if the index is out of bounds.
    ///
    /// ```should_panic
    /// # use ising_lib::prelude::*;
    /// let lattice = Lattice::new((10, 10));
    /// let _ = lattice.measure_E_diff((42, 0));
    /// ```
    pub fn measure_E_diff_with_h(&self, ix: (usize, usize), h: &Array2<f64>) -> f64 {
        2.0 * (f64::from(self.spin_times_all_neighbors(ix)) + f64::from(self.inner[ix]) * h[ix])
    }

    /// Return the energy of the lattice.
    ///
    /// ```text
    /// E = -J * ∑(s_i * s_j)
    /// ```
    pub fn measure_E(&self) -> f64 {
        -f64::from(
            self.inner
                .indexed_iter()
                .map(|(ix, _)| self.spin_times_two_neighbors(ix))
                .sum::<i32>(),
        )
    }

    /// Return the energy of the lattice in the presence of an external magnetic field.
    ///
    /// ```text
    /// E = -J * ∑(s_i * s_j) - ∑(s_i * h_i)
    /// ```
    pub fn measure_E_with_h(&self, h: &Array2<f64>) -> f64 {
        -f64::from(
            self.inner
                .indexed_iter()
                .map(|(ix, _)| self.spin_times_two_neighbors(ix))
                .sum::<i32>(),
        ) - (self.inner.map(|s| f64::from(*s)) * h).sum()
    }

    /// Return the magnetization of the lattice. The magnetization is
    /// a value in range `[0.0, 1.0]` and it is the absolute value of the mean
    /// spin value.
    ///
    /// ```text
    /// I = 1/n * ∑s_i
    /// ```
    pub fn measure_I(&self) -> f64 {
        f64::from(self.inner.sum().abs()) / f64::from(self.n_of_spins)
    }

    /// Flip the `(ith, jth)` spin.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds.
    pub fn flip_spin(&mut self, ix: (usize, usize)) {
        *self.inner.get_mut(ix).unwrap() *= -1;
    }

    /// Return a valid, randomly generated spin index.
    pub fn gen_random_index(&mut self) -> (usize, usize) {
        (
            self.rng.gen_range(0, self.dims.0),
            self.rng.gen_range(0, self.dims.1),
        )
    }
}

impl Lattice<SmallRng> {
    /// Create a new of certain dimensions with randomly generated
    /// spins. [`SmallRng`][rand::prelude::SmallRng] is used as a RNG.
    pub fn new(dims: (usize, usize)) -> Self {
        Self::from_rng(dims, SmallRng::from_entropy())
    }

    /// Create a new lattice from provided array of spins.
    /// [`SmallRng`][rand::prelude::SmallRng] is used as a RNG.
    ///
    /// See [`Lattice::from_array_rng`].
    pub fn from_array(array: Array2<i32>) -> Self {
        Self::from_array_rng(array, SmallRng::from_entropy())
    }
}

#[cfg(test)]
mod test {
    use ::pretty_assertions::assert_eq;

    use super::*;

    fn float_error(x: f64, t: f64) -> f64 {
        (x - t).abs() / t
    }

    #[test]
    fn test_lattice_from_rng() {
        let rng = SmallRng::from_entropy();
        let lattice = Lattice::from_rng((17, 10), rng);

        assert_eq!(lattice.dims(), (17, 10));
    }

    #[test]
    fn test_lattice_from_array_rng() {
        let array = Array::from_shape_vec((2, 2), vec![1, -1, 1, -1]).unwrap();
        let rng = SmallRng::from_entropy();

        let lattice = Lattice::from_array_rng(array, rng);

        assert_eq!(lattice.dims(), (2, 2));
    }

    #[test]
    fn test_lattice_new() {
        let lattice = Lattice::new((40, 20));

        assert_eq!(lattice.dims(), (40, 20));
    }

    #[test]
    fn test_lattice_from_array() {
        let array = Array::from_shape_vec((2, 2), vec![1, -1, 1, -1]).unwrap();

        let lattice = Lattice::from_array(array);

        assert_eq!(lattice.dims(), (2, 2));
    }

    #[test]
    fn test_spin_times_neighbors() {
        let spins = [-1, -1, 1, 1, 1, 1, 1, 1, -1];
        let array = Array::from_shape_vec((3, 3), spins.to_vec()).unwrap();
        let lattice = Lattice::from_array(array);

        let product = lattice.spin_times_all_neighbors((1, 1));

        assert_eq!(product, 2);
    }

    #[test]
    fn test_measure_E_difference() {
        let array =
            Array::from_shape_vec((3, 3), vec![-1, -1, 1, 1, 1, 1, -1, 1, 1])
                .unwrap();
        let lattice = Lattice::from_array(array);

        let E_diff = lattice.measure_E_diff((1, 1));

        assert_eq!(E_diff, 4.0);
    }

    #[test]
    fn test_measure_E_difference_in_magnetic_field() {
        let array =
            Array::from_shape_vec((3, 3), vec![-1, -1, 1, 1, 1, 1, -1, 1, 1])
                .unwrap();
        let h =
            Array::from_shape_vec((3, 3), vec![-1.0, -1.0, 1.0, 1.0, -7.0, 1.0, -1.0, 1.0, 1.0])
                .unwrap();
        let lattice = Lattice::from_array(array);

        let E_diff = lattice.measure_E_diff_with_h((1, 1), &h);

        assert_eq!(E_diff, -10.0);
    }

    #[test]
    fn test_measure_E() {
        let array =
            Array::from_shape_vec((3, 3), vec![-1, -1, -1, 1, 1, -1, 1, 1, -1])
                .unwrap();
        let lattice = Lattice::from_array(array);

        let E = lattice.measure_E();

        assert_eq!(E, -2.0);
    }

    #[test]
    fn test_measure_E_in_magnetic_field() {
        let array =
            Array::from_shape_vec((3, 3), vec![-1, -1, -1, 1, 1, -1, 1, 1, -1])
                .unwrap();
        let h =
            Array::from_shape_vec((3, 3), vec![-1.0, -1.0, 1.0, 1.0, -7.0, 1.0, -1.0, 1.0, 1.0])
                .unwrap();
        let lattice = Lattice::from_array(array);

        let E = lattice.measure_E_with_h(&h);

        assert_eq!(E, 5.0);
    }

    #[test]
    fn test_measure_I() {
        let array = Array::from_shape_vec((2, 2), vec![-1, -1, -1, 1]).unwrap();
        let lattice = Lattice::from_array(array);

        let I = lattice.measure_I();

        assert_eq!(I, 0.5);
    }

    #[test]
    fn test_flip_spin() {
        let array = Array::from_shape_vec(
            (3, 3),
            vec![-1, -1, -1, -1, 1, 1, -1, -1, 1],
        )
        .unwrap();
        let mut lattice = Lattice::from_array(array);

        let E_1 = lattice.measure_E();

        lattice.flip_spin((1, 1));

        let E_2 = lattice.measure_E();

        assert!(float_error(E_2 - E_1, -4.0) < 0.01);
    }
}

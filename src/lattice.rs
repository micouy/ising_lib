//! Stuff related to spin lattice.

use ::ndarray::prelude::*;
use ::rand::prelude::*;

/// Struct which encapsulates the spin lattice and all the operations performed
/// on it.
///
/// The lattice behaves like a torus - spins on opposite edges are considered
/// each other's neighbors.
pub struct Lattice {
    size: usize,
    rng: ThreadRng,
    inner: Array2<i32>,
}

impl Lattice {
    /// Creates a new [`Lattice`] of a certain size with randomly generated
    /// spins.
    pub fn new(size: usize) -> Self {
        let mut rng = thread_rng();
        let spins: [i32; 2] = [-1, 1];
        let inner = Array2::from_shape_fn((size, size), |_| {
            *spins[..].choose(&mut rng).unwrap()
        });

        Self { size, inner, rng }
    }

    /// Creates a new [`Lattice`] from [`Array2<i32>`][ndarray::Array2].
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<std::error::Error>> {
    /// # use ::ndarray::prelude::*;
    /// # use ising_lib::prelude::*;
    /// let array = Array::from_shape_vec((2, 2), vec![1, -1, 1, -1])?;
    /// let lattice = Lattice::from_array(array);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// The function will panic if `array` is not
    /// [`square`][ndarray::ArrayBase::is_square] or if any of the spins
    /// has incorrect value (neither `-1` nor `1`).
    ///
    /// ```should_panic
    /// # fn main() -> Result<(), Box<std::error::Error>> {
    /// # use ::ndarray::prelude::*;
    /// # use ising_lib::prelude::*;
    /// let array = Array::from_shape_vec((2, 2), vec![5, -1, 1, -1])?;
    /// //                                             ↑ incorrect spin value
    /// let lattice = Lattice::from_array(array);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ```should_panic
    /// # fn main() -> Result<(), Box<std::error::Error>> {
    /// # use ::ndarray::prelude::*;
    /// # use ising_lib::prelude::*;
    /// let array = Array::from_shape_vec((1, 4), vec![1, 1, 1, 1])?;
    /// //                                 ↑  ↑ array isn't square
    /// let lattice = Lattice::from_array(array);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_array(array: Array2<i32>) -> Self {
        assert!(array.is_square(), "Array is not square.");
        assert!(
            array.iter().all(|spin| *spin == 1 || *spin == -1),
            "Invalid spin value."
        );

        Lattice {
            size: array.shape()[0],
            inner: array,
            rng: thread_rng(),
        }
    }

    /// Returns the size of the lattice.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns the modulo operation on value `index + amt`, so that it stays
    /// within the bounds.
    fn roll_index(&self, (i, j): (usize, usize), amt: isize) -> (usize, usize) {
        let size = self.size as isize;

        (
            ((i as isize + size + amt) % size) as usize,
            ((j as isize + size + amt) % size) as usize,
        )
    }

    /// Returns the product of the `(ith, jth)` spin and the sum of all of its
    /// neighbors.
    fn spin_times_all_neighbors(&self, (i, j): (usize, usize)) -> i32 {
        assert!(i < self.size && j < self.size);

        let (i_1, j_1) = self.roll_index((i, j), -1);
        let (i_2, j_2) = self.roll_index((i, j), 1);

        self.inner[(i, j)]
            * [(i_1, j), (i_2, j), (i, j_1), (i, j_2)]
                .iter()
                .map(|ix| self.inner[*ix])
                .sum::<i32>()
    }

    /// Returns the product of the `(ith, jth)` spin and the sum of two of its
    /// neighbors (the right one and the bottom one).
    fn spin_times_two_neighbors(&self, (i, j): (usize, usize)) -> i32 {
        assert!(i < self.size && j < self.size);

        let (i_r, j_r) = self.roll_index((i, j), 1);

        self.inner[(i, j)]
            * [(i_r, j), (i, j_r)]
                .iter()
                .map(|ix| self.inner[*ix])
                .sum::<i32>()
    }

    /// Returns the difference of energy that would be caused by
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
    /// The function will panic if the index is out of bounds.
    ///
    /// ```should_panic
    /// # use ising_lib::prelude::*;
    /// let lattice = Lattice::new(10);
    /// let _ = lattice.measure_E_diff((42, 0), 1.0);
    /// ```
    pub fn measure_E_diff(&self, (i, j): (usize, usize), J: f64) -> f64 {
        assert!(i < self.size && j < self.size);

        2.0 * J * f64::from(self.spin_times_all_neighbors((i, j)))
    }

    /// Returns the energy of the lattice.
    pub fn measure_E(&self, J: f64) -> f64 {
        -J * f64::from(
            self.inner
                .indexed_iter()
                .map(|(ix, _)| self.spin_times_two_neighbors(ix))
                .sum::<i32>(),
        )
    }

    /// Returns the magnetization of the lattice. The magnetization is
    /// a value in range `[0.0, 1.0]` and it is the absolute value of the mean
    /// spin value.
    pub fn measure_I(&self) -> f64 {
        f64::from(self.inner.sum().abs()) / self.size.pow(2) as f64
    }

    /// Flips the `(ith, jth)` spin.
    ///
    /// # Panics
    ///
    /// This function panics if the index is out of bounds.
    pub fn flip_spin(&mut self, (i, j): (usize, usize)) {
        assert!(i < self.size && j < self.size);

        *self.inner.get_mut((i, j)).unwrap() *= -1;
    }

    /// Returns a valid, randomly generated spin index.
    pub fn gen_random_index(&mut self) -> (usize, usize) {
        (
            self.rng.gen_range(0, self.size) as usize,
            self.rng.gen_range(0, self.size) as usize,
        )
    }

    /// sraj sie
    pub fn display(&self) {
        let image = self
            .inner
            .genrows()
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|cell| match cell {
                        -1 => " .",
                        1 => "##",
                        _ => unreachable!(),
                    })
                    .collect::<Vec<&str>>()
                    .join("")
            })
            .collect::<Vec<String>>()
            .join("\n");

        println!("{}", image);
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
    fn test_create_lattice() {
        let lattice = Lattice::new(40);

        assert_eq!(lattice.size(), 40);
    }

    #[test]
    fn test_create_lattice_from_array() {
        let array = Array::from_shape_vec((2, 2), vec![1, -1, 1, -1]).unwrap();
        let lattice = Lattice::from_array(array);

        assert_eq!(lattice.size(), 2);
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
        let J = 1.0;

        let E_diff = lattice.measure_E_diff((1, 1), J);

        assert_eq!(E_diff, 4.0);
    }

    #[test]
    fn test_measure_E() {
        let array =
            Array::from_shape_vec((3, 3), vec![-1, -1, -1, 1, 1, -1, 1, 1, -1])
                .unwrap();
        let lattice = Lattice::from_array(array);
        let J = 1.0;

        let E = lattice.measure_E(J);

        assert_eq!(E, -2.0);
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
        let J = 1.0;

        let E_1 = lattice.measure_E(J);

        lattice.flip_spin((1, 1));

        let E_2 = lattice.measure_E(J);

        assert!(float_error(E_2 - E_1, -4.0) < 0.01);
    }
}

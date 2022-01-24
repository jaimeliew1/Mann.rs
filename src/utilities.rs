use ndarray::{concatenate, prelude::*};
use ndarray_rand::{rand::SeedableRng, rand_distr::Normal, RandomExt};
use ndrustfft::{ndfft_par, ndfft_r2c_par, ndifft_par, ndifft_r2c_par, Complex, FftHandler};
use std::f64::consts::{PI, SQRT_2};

use numpy::c64;

/// Various mathematical function implementations.
pub mod Utilities {
    use super::*;

    /// Unnormalized sinc squared function
    pub fn sinc2(x: f64) -> f64 {
        match x {
            x if x == 0.0 => 1.0,
            x => (x.sin() / x).powi(2),
        }
    }

    /// Returns the frequency components for a fft given a signal length (N) and a
    /// sampling distance. This function replicates the behaviour of
    /// `numpy.fft.fftfreq`.
    pub fn fftfreq(N: usize, dx: f64) -> Array1<f64> {
        let df = 1.0 / (N as f64 * dx);
        let _N = (N as i32 - 1) / 2 + 1;
        let f1: Array1<f64> = Array1::from_iter(0.._N).mapv(|elem| elem as f64);
        let f2: Array1<f64> = Array1::from_iter(-(N as i32) / 2..0).mapv(|elem| elem as f64);
        df * concatenate![Axis(0), f1, f2]
    }

    pub fn rfft3d(input: &mut Array3<f64>) -> Array3<c64> {
        let (nx, ny, nz) = input.dim();
        let mut vhat: Array3<c64> = Array3::zeros((nx, ny, nz / 2 + 1));

        let mut handler: FftHandler<f64> = FftHandler::new(nz);
        ndfft_r2c_par(input, &mut vhat, &mut handler, 2);

        let mut vhat2: Array3<c64> = Array3::zeros((nx, ny, nz / 2 + 1));
        let mut handler: FftHandler<f64> = FftHandler::new(nx);

        ndfft_par(&mut vhat, &mut vhat2, &mut handler, 0);

        let mut vhat3: Array3<c64> = Array3::zeros((nx, ny, nz / 2 + 1));
        let mut handler: FftHandler<f64> = FftHandler::new(ny);

        ndfft_par(&mut vhat2, &mut vhat3, &mut handler, 1);

        vhat3
    }
    pub fn irfft3d(input: &mut Array3<c64>) -> Array3<f64> {
        let (nx, ny, _nz) = input.dim();
        let nz = (_nz - 1) * 2;

        let mut handler: FftHandler<f64> = FftHandler::new(nx);
        let mut vhat2: Array3<c64> = Array3::zeros((nx, ny, _nz));
        ndifft_par(input, &mut vhat2, &mut handler, 0);

        let mut handler: FftHandler<f64> = FftHandler::new(ny);
        let mut vhat: Array3<c64> = Array3::zeros((nx, ny, _nz));
        ndifft_par(&mut vhat2, &mut vhat, &mut handler, 1);

        let mut output: Array3<f64> = Array3::zeros((nx, ny, nz));
        let mut handler: FftHandler<f64> = FftHandler::new(nz);
        ndifft_r2c_par(&mut vhat, &mut output, &mut handler, 2);

        output
    }
    /// Returns the frequency components for a real fft given a signal length (N)
    /// and a sampling distance. This function replicates the behaviour of
    /// `numpy.fft.rfftfreq`.
    pub fn rfftfreq(N: usize, dx: f64) -> Array1<f64> {
        let df = 1.0 / (N as f64 * dx);
        let _N = (N as i32) / 2 + 1;
        let f: Array1<f64> = Array1::from_iter(0.._N).mapv(|elem| elem as f64);
        df * f
    }

    /// Returns wave numbers for a turbulence box specification.
    pub fn freq_components(
        Lx: f64,
        Ly: f64,
        Lz: f64,
        Nx: usize,
        Ny: usize,
        Nz: usize,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        (
            fftfreq(Nx, Lx / (2.0 * PI * Nx as f64)),
            fftfreq(Ny, Ly / (2.0 * PI * Ny as f64)),
            rfftfreq(Nz, Lz / (2.0 * PI * Nz as f64)),
        )
    }

    /// Returns Array3 of of complex, gaussian distributed random numbers with
    /// standard deviation if 1/(sqrt(2)).
    pub fn complex_random_gaussian(seed: u64, Nx: usize, Ny: usize, Nz: usize) -> Array4<c64> {
        let mut rng = ndarray_rand::rand::rngs::SmallRng::seed_from_u64(seed);
        let dist = Normal::new(0.0, SQRT_2.recip()).unwrap();
        let real: Array4<c64> = Array4::random_using((Nx, Ny, Nz, 3), dist, &mut rng)
            .mapv(|elem| Complex::new(elem, 0.0));
        let imag: Array4<c64> = Array4::random_using((Nx, Ny, Nz, 3), dist, &mut rng)
            .mapv(|elem| Complex::new(0.0, elem));

        real + imag
    }
}

use ndarray::{concatenate, prelude::*};
use ndarray_rand::{
    rand::SeedableRng,
    rand_distr::{Normal, Uniform},
    RandomExt,
};
use ndrustfft::{
    ndfft, ndfft_par, ndfft_r2c, ndfft_r2c_par, ndifft, ndifft_par, ndifft_r2c, ndifft_r2c_par,
    Complex, FftHandler,
};
use std::f32::consts::{PI, SQRT_2};

use numpy::Complex32;

/// Various mathematical function implementations.
pub mod Utilities {
    use super::*;

    /// Unnormalized sinc squared function
    pub fn sinc2(x: f32) -> f32 {
        match x {
            x if x == 0.0 => 1.0,
            x => (x.sin() / x).powi(2),
        }
    }

    /// Returns the frequency components for a fft given a signal length (N) and a
    /// sampling distance. This function replicates the behaviour of
    /// `numpy.fft.fftfreq`.
    pub fn fftfreq(N: usize, dx: f32) -> Array1<f32> {
        let df = 1.0 / (N as f32 * dx);
        let _N = (N as i32 - 1) / 2 + 1;
        let f1: Array1<f32> = Array1::from_iter(0.._N).mapv(|elem| elem as f32);
        let f2: Array1<f32> = Array1::from_iter(-(N as i32) / 2..0).mapv(|elem| elem as f32);
        df * concatenate![Axis(0), f1, f2]
    }

    pub fn rfft3d_par(input: &mut Array3<f32>) -> Array3<Complex32> {
        let (nx, ny, nz) = input.dim();
        let mut vhat: Array3<Complex32> = Array3::zeros((nx, ny, nz / 2 + 1));

        let mut handler: FftHandler<f32> = FftHandler::new(nz);
        ndfft_r2c_par(input, &mut vhat, &mut handler, 2);

        let mut vhat2: Array3<Complex32> = Array3::zeros((nx, ny, nz / 2 + 1));
        let mut handler: FftHandler<f32> = FftHandler::new(nx);

        ndfft_par(&mut vhat, &mut vhat2, &mut handler, 0);

        let mut vhat3: Array3<Complex32> = Array3::zeros((nx, ny, nz / 2 + 1));
        let mut handler: FftHandler<f32> = FftHandler::new(ny);

        ndfft_par(&mut vhat2, &mut vhat3, &mut handler, 1);

        vhat3
    }
    pub fn irfft3d_par(input: &mut Array3<Complex32>) -> Array3<f32> {
        let (nx, ny, _nz) = input.dim();
        let nz = (_nz - 1) * 2;

        let mut handler: FftHandler<f32> = FftHandler::new(nx);
        let mut vhat2: Array3<Complex32> = Array3::zeros((nx, ny, _nz));
        ndifft_par(input, &mut vhat2, &mut handler, 0);

        let mut handler: FftHandler<f32> = FftHandler::new(ny);
        let mut vhat: Array3<Complex32> = Array3::zeros((nx, ny, _nz));
        ndifft_par(&mut vhat2, &mut vhat, &mut handler, 1);

        let mut output: Array3<f32> = Array3::zeros((nx, ny, nz));
        let mut handler: FftHandler<f32> = FftHandler::new(nz);
        ndifft_r2c_par(&mut vhat, &mut output, &mut handler, 2);

        output
    }
    pub fn rfft3d(input: &mut Array3<f32>) -> Array3<Complex32> {
        let (nx, ny, nz) = input.dim();
        let mut vhat: Array3<Complex32> = Array3::zeros((nx, ny, nz / 2 + 1));

        let mut handler: FftHandler<f32> = FftHandler::new(nz);
        ndfft_r2c(input, &mut vhat, &mut handler, 2);

        let mut vhat2: Array3<Complex32> = Array3::zeros((nx, ny, nz / 2 + 1));
        let mut handler: FftHandler<f32> = FftHandler::new(nx);

        ndfft(&mut vhat, &mut vhat2, &mut handler, 0);

        let mut vhat3: Array3<Complex32> = Array3::zeros((nx, ny, nz / 2 + 1));
        let mut handler: FftHandler<f32> = FftHandler::new(ny);

        ndfft(&mut vhat2, &mut vhat3, &mut handler, 1);

        vhat3
    }
    pub fn irfft3d(input: &mut Array3<Complex32>) -> Array3<f32> {
        let (nx, ny, _nz) = input.dim();
        let nz = (_nz - 1) * 2;

        let mut handler: FftHandler<f32> = FftHandler::new(nx);
        let mut vhat2: Array3<Complex32> = Array3::zeros((nx, ny, _nz));
        ndifft(input, &mut vhat2, &mut handler, 0);

        let mut handler: FftHandler<f32> = FftHandler::new(ny);
        let mut vhat: Array3<Complex32> = Array3::zeros((nx, ny, _nz));
        ndifft(&mut vhat2, &mut vhat, &mut handler, 1);

        let mut output: Array3<f32> = Array3::zeros((nx, ny, nz));
        let mut handler: FftHandler<f32> = FftHandler::new(nz);
        ndifft_r2c(&mut vhat, &mut output, &mut handler, 2);

        output
    }
    /// Returns the frequency components for a real fft given a signal length (N)
    /// and a sampling distance. This function replicates the behaviour of
    /// `numpy.fft.rfftfreq`.
    pub fn rfftfreq(N: usize, dx: f32) -> Array1<f32> {
        let df = 1.0 / (N as f32 * dx);
        let _N = (N as i32) / 2 + 1;
        let f: Array1<f32> = Array1::from_iter(0.._N).mapv(|elem| elem as f32);
        df * f
    }

    /// Returns wave numbers for a turbulence box specification.
    pub fn freq_components(
        Lx: f32,
        Ly: f32,
        Lz: f32,
        Nx: usize,
        Ny: usize,
        Nz: usize,
    ) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
        (
            fftfreq(Nx, Lx / (2.0 * PI * Nx as f32)),
            fftfreq(Ny, Ly / (2.0 * PI * Ny as f32)),
            rfftfreq(Nz, Lz / (2.0 * PI * Nz as f32)),
        )
    }

    /// Returns Array3 of of complex, gaussian distributed random numbers with
    /// unit variance.
    pub fn complex_random_gaussian(
        seed: u64,
        Nx: usize,
        Ny: usize,
        Nz: usize,
    ) -> Array4<Complex32> {
        let mut rng = ndarray_rand::rand::rngs::SmallRng::seed_from_u64(seed);
        let dist = Normal::new(0.0, SQRT_2.recip()).unwrap();
        let real: Array4<Complex32> = Array4::random_using((Nx, Ny, Nz, 3), dist, &mut rng)
            .mapv(|elem| Complex::new(elem, 0.0));
        let imag: Array4<Complex32> = Array4::random_using((Nx, Ny, Nz, 3), dist, &mut rng)
            .mapv(|elem| Complex::new(0.0, elem));

        real + imag
    }

    /// Returns Array3 of of complex, random numbers with unit length.
    pub fn complex_random_unit(seed: u64, Nx: usize, Ny: usize, Nz: usize) -> Array4<Complex32> {
        let mut rng = ndarray_rand::rand::rngs::SmallRng::seed_from_u64(seed);
        let dist = Uniform::new(0.0, 2.0 * PI);
        let phase: Array4<f32> = Array4::random_using((Nx, Ny, Nz, 3), dist, &mut rng);
        let out: Array4<Complex32> = phase.mapv(|elem| Complex::new(elem.cos(), elem.sin()));
        out
    }

    #[derive(Debug)]
    struct SimpsonRange {
        a: f32,
        m: f32,
        b: f32,
        fa: Array2<f32>,
        fm: Array2<f32>,
        fb: Array2<f32>,
        depth: u64,
    }

    pub fn adaptive_quadrature<F>(
        mut func: F,
        A: f32,
        B: f32,
        tol: f32,
        min_depth: u64,
    ) -> (Array2<f32>, u64)
    where
        F: FnMut(f32) -> Array2<f32>,
    {
        let m: f32 = (A + B) / 2.0;
        let mut I: Array2<f32> = Array2::zeros((3, 3));

        let mut S: Vec<SimpsonRange> = Vec::new();
        S.push(SimpsonRange {
            a: A,
            m: m,
            b: B,
            fa: func(A),
            fm: func(m),
            fb: func(B),
            depth: 1,
        });
        let mut neval: u64 = 3;

        while let Some(SimpsonRange {
            a,
            m,
            b,
            fa,
            fm,
            fb,
            depth,
        }) = S.pop()
        {
            // simpson rule
            let I1: Array2<f32> = (b - a) / 6.0 * (&fa + 4.0 * &fm + &fb);
            let m1: f32 = (a + m) / 2.0;
            let m3: f32 = (m + b) / 2.0;
            let fm1: Array2<f32> = func(m1);
            let fm3: Array2<f32> = func(m3);
            neval += 2;

            // Composite trapeszoidal rule with 2 equidistant intervals
            let I2: Array2<f32> =
                (b - a) / 12.0 * (&fa.view() + 4.0 * &fm1 + 2.0 * &fm + 4.0 * &fm3 + &fb);
            if depth >= min_depth && (&I2 - &I1).iter().all(|&x| x.abs() < 15.0 * tol) {
                I += &I2;
            } else {
                S.push(SimpsonRange {
                    a: a,
                    m: m1,
                    b: m,
                    fa: fa,
                    fm: fm1,
                    fb: fm.clone(),
                    depth: depth + 1,
                });
                S.push(SimpsonRange {
                    a: m,
                    m: m3,
                    b: b,
                    fa: fm,
                    fm: fm3,
                    fb: fb,
                    depth: depth + 1,
                });
            }
        }
        (I, neval)
    }

    pub fn adaptive_quadrature_2d<F>(
        mut func: F,
        x0: f32,
        x1: f32,
        y0: f32,
        y1: f32,
        tol: f32,
        min_depth: u64,
    ) -> (Array2<f32>, u64)
    where
        F: FnMut(f32, f32) -> Array2<f32>,
    {
        let mut neval: u64 = 0;
        let g = |x: f32| {
            let f = |y: f32| func(x, y);
            let (I, _neval): (Array2<f32>, u64) = adaptive_quadrature(f, y0, y1, tol, min_depth);
            neval += _neval;
            I
        };
        let (I, _): (Array2<f32>, u64) = adaptive_quadrature(g, x0, x1, tol, min_depth);
        (I, neval)
    }
}

#![allow(non_snake_case)]
//! Coherence turbulence box generation using the Mann turbulence model.
//!
//! `Rustmann` provides a computationally efficient module for generating Mann
//! turbulence boxes for wind turbine simulations. `Rustmann` is designed to be
//! called from Python, however the underlying functions are also available in
//! rust.

use std::f64::consts::{PI, SQRT_2};

use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::{concatenate, Zip};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use num::complex::Complex;
use numpy::c64;

/// Utility functions used for turbulence generation.
pub mod utilities {
    use super::*;
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

    /// Returns the frequency components for a real fft given a signal length (N)
    /// and a sampling distance. This function replicates the behaviour of
    /// `numpy.fft.rfftfreq`.
    pub fn rfftfreq(N: usize, dx: f64) -> Array1<f64> {
        let df = 1.0 / (N as f64 * dx);
        let _N = (N as i32) / 2 + 1;
        let f: Array1<f64> = Array1::from_iter(0.._N).mapv(|elem| elem as f64);
        df * f
    }
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
        let real: Array4<Complex<f64>> = Array4::random_using((Nx, Ny, Nz, 3), dist, &mut rng)
            .mapv(|elem| Complex::new(elem, 0.0));
        let imag: Array4<Complex<f64>> = Array4::random_using((Nx, Ny, Nz, 3), dist, &mut rng)
            .mapv(|elem| Complex::new(0.0, elem));

        real + imag
    }
}
pub fn lifetime_approx(mut kL: f64) -> f64 {
    if kL < 0.005 {
        kL = 0.005;
    }
    let kSqr = kL.powi(2);
    (1.0 + kSqr).powf(1.0 / 6.0) / kL
        * (1.2050983316598936 - 0.04079766636961979 * kL + 1.1050803451576134 * kSqr)
        / (1.0 - 0.04103886513006046 * kL + 1.1050902034670118 * kSqr)
}

pub fn lifetime_exact(_kL: f64) -> f64 {
    unimplemented!();
}

pub fn vonkarman_spectrum(ae: f64, k: f64, L: f64) -> f64 {
    ae * L.powf(5.0 / 3.0) * (L * k).powi(4) / (1.0 + (L * k).powi(2)).powf(17.0 / 6.0)
}

/// Mann tensor calculations
pub mod Tensors {
    use super::*;

    pub fn sqrt_iso_tensor(K: &ArrayView1<f64>, ae: f64, L: f64) -> Array2<f64> {
        let k_norm = K.dot(K).sqrt();
        if k_norm == 0.0 {
            return Array2::zeros((3, 3));
        }
        let E = vonkarman_spectrum(ae, k_norm, L);
        let mut tensor: Array2<f64> =
            arr2(&[[0.0, K[2], -K[1]], [-K[2], 0.0, K[0]], [K[1], -K[0], 0.0]]);
        tensor *= (E / PI).sqrt() / (2.0 * k_norm.powi(2));
        tensor
    }

    pub fn iso_tensor(K: &ArrayView1<f64>, ae: f64, L: f64) -> Array2<f64> {
        let k_norm = K.dot(K).sqrt();
        if k_norm == 0.0 {
            return Array2::zeros((3, 3));
        }
        let E = vonkarman_spectrum(ae, k_norm, L);
        let mut tensor: Array2<f64> = arr2(&[
            [K[1].powi(2) + K[2].powi(2), -K[0] * K[1], -K[0] * K[2]],
            [-K[0] * K[1], K[0].powi(2) + K[2].powi(2), -K[1] * K[2]],
            [-K[0] * K[2], -K[1] * K[2], K[0].powi(2) + K[1].powi(2)],
        ]);
        tensor *= E / (4.0 * PI * k_norm.powi(4));
        tensor
    }

    pub fn sheared_transform(K: &ArrayView1<f64>, _ae: f64, L: f64, gamma: f64) -> Array2<f64> {
        let k_norm2 = K.dot(K);
        if k_norm2 == 0.0 {
            return Array2::zeros((3, 3));
        }
        let beta = gamma * lifetime_approx((k_norm2).sqrt() * L);

        // Equation (12)
        let K0: Array1<f64> = K + arr1(&[0.0, 0.0, beta * K[0]]);
        let k0_norm2 = K0.dot(&K0);
        let (zeta1, zeta2);
        if K[0] == 0.0 {
            zeta1 = -beta;
            zeta2 = 0.0;
        } else {
            // Equation (15)
            let C1 = beta * K[0].powi(2) * (k0_norm2 - 2.0 * K0[2].powi(2) + beta * K[0] * K0[2])
                / (k_norm2 * (K[0].powi(2) + K[1].powi(2)));
            // Equation (16)
            let C2 = K[1] * k0_norm2 / (K[0].powi(2) + K[1].powi(2)).powf(3.0 / 2.0)
                * ((beta * K[0] * (K[0].powi(2) + K[1].powi(2)).sqrt())
                    / (k0_norm2 - K0[2] * K[0] * beta))
                    .atan();
            // Equation (14)
            zeta1 = C1 - K[1] / K[0] * C2;
            zeta2 = K[1] / K[0] * C1 + C2;
        }
        // Equation (13)
        arr2(&[
            [1.0, 0.0, zeta1],
            [0.0, 1.0, zeta2],
            [0.0, 0.0, k0_norm2 / k_norm2],
        ])
    }

    pub fn sqrt_sheared_tensor(K: &ArrayView1<f64>, ae: f64, L: f64, gamma: f64) -> Array2<f64> {
        let k_norm2 = K.dot(K);
        if k_norm2 == 0.0 {
            return Array2::zeros((3, 3));
        }
        let A: Array2<f64> = sheared_transform(K, ae, L, gamma);
        let beta = gamma * lifetime_approx((k_norm2).sqrt() * L);

        // Equation (12)
        let K0: Array1<f64> = K + arr1(&[0.0, 0.0, beta * K[0]]);
        let iso_tensor = sqrt_iso_tensor(&K0.view(), ae, L);

        A.dot(&iso_tensor)
    }

    pub fn sheared_tensor(K: &ArrayView1<f64>, ae: f64, L: f64, gamma: f64) -> Array2<f64> {
        let k_norm2 = K.dot(K);
        if k_norm2 == 0.0 {
            return Array2::zeros((3, 3));
        }
        let A: Array2<f64> = sheared_transform(K, ae, L, gamma);
        let beta = gamma * lifetime_approx((k_norm2).sqrt() * L);

        // Equation (12)
        let K0: Array1<f64> = K + arr1(&[0.0, 0.0, beta * K[0]]);
        let iso_tensor = iso_tensor(&K0.view(), ae, L);

        A.dot(&iso_tensor).dot(&A.t())
    }

    pub fn sheared_tensor_sinc(
        K: &ArrayView1<f64>,
        ae: f64,
        L: f64,
        gamma: f64,
        _Lx: f64,
        Ly: f64,
        Lz: f64,
        Ny: usize,
        Nz: usize,
    ) -> Array2<f64> {
        let kys: Array1<f64> = K[1] * Ly / (2.0 * PI) - Array1::linspace(-1.0, 1.0, Ny);
        let kzs: Array1<f64> = K[2] * Lz / (2.0 * PI) - Array1::linspace(-1.0, 1.0, Nz);

        let mut ans: Array2<f64> = Array2::zeros((3, 3));
        let dA = 1.0 / (Ny as f64 * Nz as f64);
        for (i, ky) in kys.iter().enumerate() {
            for (j, kz) in kzs.iter().enumerate() {
                let mut sinc = 4.0 * utilities::sinc2(ky * PI) * utilities::sinc2(kz * PI);
                if i == 0 || i == Ny - 1 {
                    sinc /= 2.0
                }
                if j == 0 || j == Nz - 1 {
                    sinc /= 2.0
                }
                ans = ans
                    + sinc
                        * sheared_tensor(
                            &arr1(&[K[0], *ky * 2.0 * PI / Ly, *kz * 2.0 * PI / Lz]).view(),
                            ae,
                            L,
                            gamma,
                        );
            }
        }
        ans *= dA / 4.0 * 2.0 * 1.22686;
        ans
    }
}

pub fn stencilate(
    ae: f64,
    L: f64,
    gamma: f64,
    Lx: f64,
    Ly: f64,
    Lz: f64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
) -> Array5<f64> {
    let mut stencil: Array5<f64> = Array5::zeros((Nx, Ny, Nz / 2 + 1, 3, 3));
    let (Kx, Ky, Kz): (Array1<f64>, Array1<f64>, Array1<f64>) =
        utilities::freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
    stencil
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut slice)| {
            for (j, mut column) in slice.outer_iter_mut().enumerate() {
                for (k, mut component) in column.outer_iter_mut().enumerate() {
                    let K: Array1<f64> = arr1(&[Kx[i], Ky[j], Kz[k]]);
                    component.assign(&Tensors::sqrt_sheared_tensor(&K.view(), ae, L, gamma));
                }
            }
        });
    stencil
}

pub fn partial_turbulate(
    stencil: &ArrayView5<f64>,
    seed: u64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    Lx: f64,
    Ly: f64,
    Lz: f64,
) -> (Array3<c64>, Array3<c64>, Array3<c64>) {
    let KVolScaleFac: c64 = Complex::new(
        SQRT_2 * (Nx * Ny * (Nz / 2 + 1)) as f64 * ((8.0 * PI.powi(3)) / (Lx * Ly * Lz)).sqrt(),
        0.0,
    );
    let random: Array4<c64> = utilities::complex_random_gaussian(seed, Nx, Ny, Nz / 2 + 1);

    let mut UVW_f: Array4<c64> = Array4::zeros((Nx, Ny, (Nz / 2 + 1), 3));

    Zip::from(UVW_f.outer_iter_mut())
        .and(stencil.outer_iter())
        .and(random.outer_iter())
        .par_for_each(|mut UVW_slice, stencil_slice, random_slice| {
            Zip::from(UVW_slice.outer_iter_mut())
                .and(stencil_slice.outer_iter())
                .and(random_slice.outer_iter())
                .par_for_each(|mut UVW_col, stencil_col, random_col| {
                    Zip::from(UVW_col.outer_iter_mut())
                        .and(stencil_col.outer_iter())
                        .and(random_col.outer_iter())
                        .for_each(|mut freq_comp, tensor, n| {
                            let _tensor = tensor.mapv(|elem| c64::new(elem, 0.0));
                            freq_comp.assign(&_tensor.dot(&n));
                            freq_comp *= KVolScaleFac;
                        })
                })
        });
    (
        UVW_f.slice(s![.., .., .., 0]).to_owned(),
        UVW_f.slice(s![.., .., .., 1]).to_owned(),
        UVW_f.slice(s![.., .., .., 2]).to_owned(),
    )
}
mod python_interface;
mod tests;

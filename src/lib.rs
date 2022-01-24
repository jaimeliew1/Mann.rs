#![allow(non_snake_case)]
//! Coherence turbulence box generation using the Mann turbulence model.
//!
//! `Rustmann` provides a computationally efficient module for generating Mann
//! turbulence boxes for wind turbine simulations. `Rustmann` is designed to be
//! called from Python, however the underlying functions are also available in
//! rust.
mod python_interface;
mod tensors;
mod tests;
mod utilities;

pub use self::tensors::Tensors;
pub use self::utilities::Utilities;

use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::Zip;
use ndarray_linalg::Norm;
use ndrustfft::Complex;
use numpy::c64;
use std::f64::consts::{PI, SQRT_2};

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
        Utilities::freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
    stencil
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut slice)| {
            for (j, mut column) in slice.outer_iter_mut().enumerate() {
                for (k, mut component) in column.outer_iter_mut().enumerate() {
                    let K: Array1<f64> = arr1(&[Kx[i], Ky[j], Kz[k]]);
                    component.assign(&Tensors::sheared_sqrt(&K.view(), ae, L, gamma));
                }
            }
        });
    stencil
}

pub fn stencilate_sinc(
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
        Utilities::freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
    stencil
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut slice)| {
            for (j, mut column) in slice.outer_iter_mut().enumerate() {
                for (k, mut component) in column.outer_iter_mut().enumerate() {
                    let K: Array1<f64> = arr1(&[Kx[i], Ky[j], Kz[k]]);
                    if K.norm_l2() < 3.0 / L {
                        component.assign(&Tensors::sheared_sqrt_sinc(
                            &K.view(),
                            ae,
                            L,
                            gamma,
                            Ly,
                            Lz,
                            Ny,
                            Nz,
                        ));
                    } else {
                        component.assign(&Tensors::sheared_sqrt(&K.view(), ae, L, gamma));
                    }
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
    let random: Array4<c64> = Utilities::complex_random_gaussian(seed, Nx, Ny, Nz / 2 + 1);

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

pub fn turbulate(
    stencil: &ArrayView5<f64>,
    seed: u64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    Lx: f64,
    Ly: f64,
    Lz: f64,
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let (mut U_f, mut V_f, mut W_f): (Array3<c64>, Array3<c64>, Array3<c64>) =
        partial_turbulate(stencil, seed, Nx, Ny, Nz, Lx, Ly, Lz);

    let U: Array3<f64> = Utilities::irfft3d(&mut U_f);
    let V: Array3<f64> = Utilities::irfft3d(&mut V_f);
    let W: Array3<f64> = Utilities::irfft3d(&mut W_f);
    (U, V, W)
}

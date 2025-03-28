#![allow(non_snake_case)]
//! Coherence turbulence box generation using the Mann turbulence model.
//!
//! `Mannrs` provides a computationally efficient module for generating Mann
//! turbulence boxes for wind turbine simulations. `Mannrs` is designed to be
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
use ndrustfft::Complex;
use numpy::Complex32;
use std::f32::consts::PI;
use std::mem::drop;
use tensors::Tensors::{Sheared, ShearedSinc, TensorGenerator};

/// Returns the normalized spectral component from a stencil of shape `(Nx, Ny,
/// Nz, 3, 3)`.
///
/// This function extracts the `Ruu`, `Rvv`, `Rww`, and `Ruw` components from
/// the stencil. These components are normalized such that their inverse fourier
/// transform as a maximum value of 1.
///
/// - `Ruu_f`: Normalized spectral energy of the `u` component
/// - `Rvv_f`: Normalized spectral energy of the `v` component
/// - `Rww_f`: Normalized spectral energy of the `w` component
/// - `Ruw_f`: Normalized spectral cross-component `uw`
///
/// # Arguments
///
/// * `stencil` - A 5D array containing the velocity correlation tensor across a
///   3D grid.
///
/// # Returns
///
/// A tuple of 3D arrays `(Ruu_f, Rvv_f, Rww_f, Ruw_f)` where each is the real
/// part of the normalized spectral component for the corresponding correlation
/// tensor entry.
pub fn spectral_component_grids(
    stencil: &ArrayView5<f32>,
) -> (Array3<f32>, Array3<f32>, Array3<f32>, Array3<f32>) {
    let mut Ruu_f: Array3<Complex32> = stencil
        .slice(s![.., .., .., 0, 0])
        .mapv(|x| Complex32::new(x, 0.0));
    let mut Rvv_f: Array3<Complex32> = stencil
        .slice(s![.., .., .., 1, 1])
        .mapv(|x| Complex32::new(x, 0.0));
    let mut Rww_f: Array3<Complex32> = stencil
        .slice(s![.., .., .., 2, 2])
        .mapv(|x| Complex32::new(x, 0.0));
    let Ruw_f: Array3<Complex32> = stencil
        .slice(s![.., .., .., 0, 2])
        .mapv(|x| Complex32::new(x, 0.0));

    let Ruu: Array3<f32> = Utilities::irfft3d(&mut Ruu_f);
    let Rvv: Array3<f32> = Utilities::irfft3d(&mut Rvv_f);
    let Rww: Array3<f32> = Utilities::irfft3d(&mut Rww_f);

    // Normalize frequency components
    (
        Ruu_f.mapv(|x| x.re / Ruu[[0, 0, 0]]),
        Rvv_f.mapv(|x| x.re / Rvv[[0, 0, 0]]),
        Rww_f.mapv(|x| x.re / Rww[[0, 0, 0]]),
        Ruw_f.mapv(|x| x.re / (Ruu[[0, 0, 0]] * Rww[[0, 0, 0]]).sqrt()),
    )
}


/// Returns the normalized correlation matrices from a stencil of shape `(Nx,
/// Ny, Nz, 3, 3)`.
///
/// This function extracts the `Ruu`, `Rvv`, `Rww`, and `Ruw` components from
/// the stencil and performs an inverse fourier transform to arrive at the
/// spatial correlation. These components are normalized such that their inverse
/// fourier transform as a maximum value of 1.
///
/// # Arguments
///
/// * `stencil` - A 5D array containing the velocity correlation tensor across a
///   3D grid.
///
/// # Returns
///
/// A tuple of 3D arrays `(Ruu, Rvv, Rww, Ruw)` where each is the spatial
/// correlation matrix for the U, V and W wind components as well as the cross
/// correlation between U and W.
pub fn correlation_grids(
    stencil: &ArrayView5<f32>,
) -> (Array3<f32>, Array3<f32>, Array3<f32>, Array3<f32>) {
    let mut Ruu_f: Array3<Complex32> = stencil
        .slice(s![.., .., .., 0, 0])
        .mapv(|x| Complex32::new(x, 0.0));
    let mut Rvv_f: Array3<Complex32> = stencil
        .slice(s![.., .., .., 1, 1])
        .mapv(|x| Complex32::new(x, 0.0));
    let mut Rww_f: Array3<Complex32> = stencil
        .slice(s![.., .., .., 2, 2])
        .mapv(|x| Complex32::new(x, 0.0));
    let mut Ruw_f: Array3<Complex32> = stencil
        .slice(s![.., .., .., 0, 2])
        .mapv(|x| Complex32::new(x, 0.0));

    let Ruu: Array3<f32> = Utilities::irfft3d(&mut Ruu_f);
    drop(Ruu_f);
    let Rvv: Array3<f32> = Utilities::irfft3d(&mut Rvv_f);
    drop(Rvv_f);
    let Rww: Array3<f32> = Utilities::irfft3d(&mut Rww_f);
    drop(Rww_f);
    let Ruw: Array3<f32> = Utilities::irfft3d(&mut Ruw_f);
    drop(Ruw_f);

    (
        Ruu.mapv(|x| x / Ruu[[0, 0, 0]]),
        Rvv.mapv(|x| x / Rvv[[0, 0, 0]]),
        Rww.mapv(|x| x / Rww[[0, 0, 0]]),
        Ruw.mapv(|x| x / (Ruu[[0, 0, 0]] * Rww[[0, 0, 0]]).sqrt()),
    )
}

pub fn stencilate_par(
    L: f32,
    gamma: f32,
    Lx: f32,
    Ly: f32,
    Lz: f32,
    Nx: usize,
    Ny: usize,
    Nz: usize,
) -> Array5<f32> {
    let mut stencil: Array5<f32> = Array5::zeros((Nx, Ny, Nz / 2 + 1, 3, 3));
    let (Kx, Ky, Kz): (Array1<f32>, Array1<f32>, Array1<f32>) =
        Utilities::freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
    let tensor_gen = Sheared::from_params(1.0, L, gamma);
    stencil
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut slice)| {
            for (j, mut column) in slice.outer_iter_mut().enumerate() {
                for (k, mut component) in column.outer_iter_mut().enumerate() {
                    let K = &[Kx[i], Ky[j], Kz[k]];
                    component.assign(&tensor_gen.decomp(K));
                }
            }
        });
    stencil
}

pub fn stencilate_sinc_par(
    L: f32,
    gamma: f32,
    Lx: f32,
    Ly: f32,
    Lz: f32,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    sinc_thres: f32,
) -> Array5<f32> {
    let mut stencil: Array5<f32> = Array5::zeros((Nx, Ny, Nz / 2 + 1, 3, 3));
    let (Kx, Ky, Kz): (Array1<f32>, Array1<f32>, Array1<f32>) =
        Utilities::freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
    let tensor_gen_sinc = ShearedSinc::from_params(1.0, L, gamma, Ly, Lz, 1.0, 2);
    let tensor_gen = Sheared::from_params(1.0, L, gamma);

    stencil
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut slice)| {
            for (j, mut column) in slice.outer_iter_mut().enumerate() {
                for (k, mut component) in column.outer_iter_mut().enumerate() {
                    let K = &[Kx[i], Ky[j], Kz[k]];
                    let norm = K.iter().fold(0.0, |acc, &x| acc + x * x);
                    if norm < sinc_thres / L {
                        component.assign(&tensor_gen_sinc.decomp(K));
                    } else {
                        component.assign(&tensor_gen.decomp(K));
                    }
                }
            }
        });
    stencil
}

pub fn partial_turbulate_par(
    stencil: &ArrayView5<f32>,
    ae: f32,
    seed: u64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    Lx: f32,
    Ly: f32,
    Lz: f32,
) -> (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) {
    let KVolScaleFac: Complex32 = Complex::new(
        2.0 * (Nx * Ny * (Nz / 2 + 1)) as f32 * ((8.0 * ae * PI.powi(3)) / (Lx * Ly * Lz)).sqrt(),
        0.0,
    );
    let random: Array4<Complex32> = Utilities::complex_random_gaussian(seed, Nx, Ny, Nz / 2 + 1);

    let mut UVW_f: Array4<Complex32> = Array4::zeros((Nx, Ny, (Nz / 2 + 1), 3));

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
                            let _tensor = tensor.mapv(|elem| Complex32::new(elem, 0.0));
                            freq_comp.assign(&_tensor.dot(&n));
                            freq_comp *= KVolScaleFac;
                        })
                })
        });
    UVW_f[[0, 0, 0, 0]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 1]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 2]] = Complex::new(0.0, 0.0);
    (
        UVW_f.slice(s![.., .., .., 0]).to_owned(),
        UVW_f.slice(s![.., .., .., 1]).to_owned(),
        UVW_f.slice(s![.., .., .., 2]).to_owned(),
    )
}

pub fn turbulate_par(
    stencil: &ArrayView5<f32>,
    ae: f32,
    seed: u64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    Lx: f32,
    Ly: f32,
    Lz: f32,
) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
    let (mut U_f, mut V_f, mut W_f): (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) =
        partial_turbulate_par(stencil, ae, seed, Nx, Ny, Nz, Lx, Ly, Lz);

    let U: Array3<f32> = Utilities::irfft3d_par(&mut U_f);
    drop(U_f);
    let V: Array3<f32> = Utilities::irfft3d_par(&mut V_f);
    drop(V_f);
    let W: Array3<f32> = Utilities::irfft3d_par(&mut W_f);
    drop(W_f);
    (U, V, W)
}

pub fn stencilate(
    L: f32,
    gamma: f32,
    Lx: f32,
    Ly: f32,
    Lz: f32,
    Nx: usize,
    Ny: usize,
    Nz: usize,
) -> Array5<f32> {
    let mut stencil: Array5<f32> = Array5::zeros((Nx, Ny, Nz / 2 + 1, 3, 3));
    let (Kx, Ky, Kz): (Array1<f32>, Array1<f32>, Array1<f32>) =
        Utilities::freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
    let tensor_gen = Sheared::from_params(1.0, L, gamma);
    stencil
        .outer_iter_mut()
        .into_iter()
        .enumerate()
        .for_each(|(i, mut slice)| {
            for (j, mut column) in slice.outer_iter_mut().enumerate() {
                for (k, mut component) in column.outer_iter_mut().enumerate() {
                    let K = &[Kx[i], Ky[j], Kz[k]];
                    component.assign(&tensor_gen.decomp(K));
                }
            }
        });
    stencil
}

pub fn stencilate_sinc(
    L: f32,
    gamma: f32,
    Lx: f32,
    Ly: f32,
    Lz: f32,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    sinc_thres: f32,
) -> Array5<f32> {
    let mut stencil: Array5<f32> = Array5::zeros((Nx, Ny, Nz / 2 + 1, 3, 3));
    let (Kx, Ky, Kz): (Array1<f32>, Array1<f32>, Array1<f32>) =
        Utilities::freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
    let tensor_gen_sinc = ShearedSinc::from_params(1.0, L, gamma, Ly, Lz, 1.0, 2);
    let tensor_gen = Sheared::from_params(1.0, L, gamma);

    stencil
        .outer_iter_mut()
        .into_iter()
        .enumerate()
        .for_each(|(i, mut slice)| {
            for (j, mut column) in slice.outer_iter_mut().enumerate() {
                for (k, mut component) in column.outer_iter_mut().enumerate() {
                    let K = &[Kx[i], Ky[j], Kz[k]];
                    let norm = K.iter().fold(0.0, |acc, &x| acc + x * x);
                    if norm < sinc_thres / L {
                        component.assign(&tensor_gen_sinc.decomp(K));
                    } else {
                        component.assign(&tensor_gen.decomp(K));
                    }
                }
            }
        });
    stencil
}

pub fn partial_turbulate(
    stencil: &ArrayView5<f32>,
    ae: f32,
    seed: u64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    Lx: f32,
    Ly: f32,
    Lz: f32,
) -> (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) {
    let KVolScaleFac: Complex32 = Complex::new(
        2.0 * (Nx * Ny * (Nz / 2 + 1)) as f32 * ((8.0 * ae * PI.powi(3)) / (Lx * Ly * Lz)).sqrt(),
        0.0,
    );
    let random: Array4<Complex32> = Utilities::complex_random_gaussian(seed, Nx, Ny, Nz / 2 + 1);

    let mut UVW_f: Array4<Complex32> = Array4::zeros((Nx, Ny, (Nz / 2 + 1), 3));

    Zip::from(UVW_f.outer_iter_mut())
        .and(stencil.outer_iter())
        .and(random.outer_iter())
        .for_each(|mut UVW_slice, stencil_slice, random_slice| {
            Zip::from(UVW_slice.outer_iter_mut())
                .and(stencil_slice.outer_iter())
                .and(random_slice.outer_iter())
                .for_each(|mut UVW_col, stencil_col, random_col| {
                    Zip::from(UVW_col.outer_iter_mut())
                        .and(stencil_col.outer_iter())
                        .and(random_col.outer_iter())
                        .for_each(|mut freq_comp, tensor, n| {
                            let _tensor = tensor.mapv(|elem| Complex32::new(elem, 0.0));
                            freq_comp.assign(&_tensor.dot(&n));
                            freq_comp *= KVolScaleFac;
                        })
                })
        });
    UVW_f[[0, 0, 0, 0]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 1]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 2]] = Complex::new(0.0, 0.0);
    (
        UVW_f.slice(s![.., .., .., 0]).to_owned(),
        UVW_f.slice(s![.., .., .., 1]).to_owned(),
        UVW_f.slice(s![.., .., .., 2]).to_owned(),
    )
}

pub fn turbulate(
    stencil: &ArrayView5<f32>,
    ae: f32,
    seed: u64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    Lx: f32,
    Ly: f32,
    Lz: f32,
) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
    let (mut U_f, mut V_f, mut W_f): (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) =
        partial_turbulate(stencil, ae, seed, Nx, Ny, Nz, Lx, Ly, Lz);

    let U: Array3<f32> = Utilities::irfft3d(&mut U_f);
    drop(U_f);
    let V: Array3<f32> = Utilities::irfft3d(&mut V_f);
    drop(V_f);
    let W: Array3<f32> = Utilities::irfft3d(&mut W_f);
    drop(W_f);
    (U, V, W)
}

pub fn partial_turbulate_unit(
    stencil: &ArrayView5<f32>,
    ae: f32,
    seed: u64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    Lx: f32,
    Ly: f32,
    Lz: f32,
) -> (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) {
    let KVolScaleFac: Complex32 = Complex::new(
        2.0 * (Nx * Ny * (Nz / 2 + 1)) as f32 * ((8.0 * ae * PI.powi(3)) / (Lx * Ly * Lz)).sqrt(),
        0.0,
    );
    let random: Array4<Complex32> = Utilities::complex_random_unit(seed, Nx, Ny, Nz / 2 + 1);

    let mut UVW_f: Array4<Complex32> = Array4::zeros((Nx, Ny, (Nz / 2 + 1), 3));

    Zip::from(UVW_f.outer_iter_mut())
        .and(stencil.outer_iter())
        .and(random.outer_iter())
        .for_each(|mut UVW_slice, stencil_slice, random_slice| {
            Zip::from(UVW_slice.outer_iter_mut())
                .and(stencil_slice.outer_iter())
                .and(random_slice.outer_iter())
                .for_each(|mut UVW_col, stencil_col, random_col| {
                    Zip::from(UVW_col.outer_iter_mut())
                        .and(stencil_col.outer_iter())
                        .and(random_col.outer_iter())
                        .for_each(|mut freq_comp, tensor, n| {
                            let _tensor = tensor.mapv(|elem| Complex32::new(elem, 0.0));
                            freq_comp.assign(&_tensor.dot(&n));
                            freq_comp *= KVolScaleFac;
                        })
                })
        });
    UVW_f[[0, 0, 0, 0]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 1]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 2]] = Complex::new(0.0, 0.0);
    (
        UVW_f.slice(s![.., .., .., 0]).to_owned(),
        UVW_f.slice(s![.., .., .., 1]).to_owned(),
        UVW_f.slice(s![.., .., .., 2]).to_owned(),
    )
}

pub fn turbulate_unit(
    stencil: &ArrayView5<f32>,
    ae: f32,
    seed: u64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    Lx: f32,
    Ly: f32,
    Lz: f32,
) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
    let (mut U_f, mut V_f, mut W_f): (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) =
        partial_turbulate_unit(stencil, ae, seed, Nx, Ny, Nz, Lx, Ly, Lz);

    let U: Array3<f32> = Utilities::irfft3d(&mut U_f);
    let V: Array3<f32> = Utilities::irfft3d(&mut V_f);
    let W: Array3<f32> = Utilities::irfft3d(&mut W_f);
    (U, V, W)
}

struct Constraint {
    x: f32,
    y: f32,
    z: f32,
    u: f32,
    v: f32,
    w: f32,
}

pub fn constrain(constraints: Vec<Constraint>) {}

pub fn partial_forgetful_turbulate_par(
    ae: f32,
    seed: u64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    Lx: f32,
    Ly: f32,
    Lz: f32,
    L: f32,
    gamma: f32,
    sinc_thres: f32,
) -> (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) {
    let KVolScaleFac: Complex32 = Complex::new(
        2.0 * (Nx * Ny * (Nz / 2 + 1)) as f32 * ((8.0 * ae * PI.powi(3)) / (Lx * Ly * Lz)).sqrt(),
        0.0,
    );
    let random: Array4<Complex32> = Utilities::complex_random_gaussian(seed, Nx, Ny, Nz / 2 + 1);
    let (Kx, Ky, Kz): (Array1<f32>, Array1<f32>, Array1<f32>) =
        Utilities::freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
    let mut UVW_f: Array4<Complex32> = Array4::zeros((Nx, Ny, (Nz / 2 + 1), 3));

    let tensor_gen_sinc = ShearedSinc::from_params(1.0, L, gamma, Ly, Lz, 1.0, 2);
    let tensor_gen = Sheared::from_params(1.0, L, gamma);

    UVW_f
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut slice)| {
            for (j, mut column) in slice.outer_iter_mut().enumerate() {
                for (k, mut component) in column.outer_iter_mut().enumerate() {
                    let K = &[Kx[i], Ky[j], Kz[k]];
                    let norm = K.iter().fold(0.0, |acc, &x| acc + x * x);

                    let invol: bool = norm < sinc_thres / L;

                    let coef: Array2<f32> = match invol {
                        true => tensor_gen_sinc.decomp(K),
                        false => tensor_gen.decomp(K),
                    };
                    let coef = coef.mapv(|elem| Complex32::new(elem, 0.0));
                    let n = random.slice(s![i, j, k, ..]);
                    component.assign(&coef.dot(&n));
                    component *= KVolScaleFac;
                }
            }
        });

    UVW_f[[0, 0, 0, 0]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 1]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 2]] = Complex::new(0.0, 0.0);
    (
        UVW_f.slice(s![.., .., .., 0]).to_owned(),
        UVW_f.slice(s![.., .., .., 1]).to_owned(),
        UVW_f.slice(s![.., .., .., 2]).to_owned(),
    )
}

pub fn forgetful_turbulate_par(
    ae: f32,
    seed: u64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    Lx: f32,
    Ly: f32,
    Lz: f32,
    L: f32,
    gamma: f32,
    sinc_thres: f32,
) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
    let (mut U_f, mut V_f, mut W_f): (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) =
        partial_forgetful_turbulate_par(ae, seed, Nx, Ny, Nz, Lx, Ly, Lz, L, gamma, sinc_thres);

    let U: Array3<f32> = Utilities::irfft3d_par(&mut U_f);
    drop(U_f);
    let V: Array3<f32> = Utilities::irfft3d_par(&mut V_f);
    drop(V_f);
    let W: Array3<f32> = Utilities::irfft3d_par(&mut W_f);
    drop(W_f);
    (U, V, W)
}

pub fn partial_forgetful_turbulate(
    ae: f32,
    seed: u64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    Lx: f32,
    Ly: f32,
    Lz: f32,
    L: f32,
    gamma: f32,
    sinc_thres: f32,

) -> (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) {
    let KVolScaleFac: Complex32 = Complex::new(
        2.0 * (Nx * Ny * (Nz / 2 + 1)) as f32 * ((8.0 * ae * PI.powi(3)) / (Lx * Ly * Lz)).sqrt(),
        0.0,
    );
    let random: Array4<Complex32> = Utilities::complex_random_gaussian(seed, Nx, Ny, Nz / 2 + 1);
    let (Kx, Ky, Kz): (Array1<f32>, Array1<f32>, Array1<f32>) =
        Utilities::freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
    let mut UVW_f: Array4<Complex32> = Array4::zeros((Nx, Ny, (Nz / 2 + 1), 3));

    let tensor_gen_sinc = ShearedSinc::from_params(1.0, L, gamma, Ly, Lz, 1.0, 2);
    let tensor_gen = Sheared::from_params(1.0, L, gamma);

    UVW_f
        .outer_iter_mut()
        .into_iter()
        .enumerate()
        .for_each(|(i, mut slice)| {
            for (j, mut column) in slice.outer_iter_mut().enumerate() {
                for (k, mut component) in column.outer_iter_mut().enumerate() {
                    let K = &[Kx[i], Ky[j], Kz[k]];
                    let norm = K.iter().fold(0.0, |acc, &x| acc + x * x);

                    let invol: bool = norm < sinc_thres / L;

                    let coef: Array2<f32> = match invol {
                        true => tensor_gen_sinc.decomp(K),
                        false => tensor_gen.decomp(K),
                    };
                    let coef = coef.mapv(|elem| Complex32::new(elem, 0.0));
                    let n = random.slice(s![i, j, k, ..]);
                    component.assign(&coef.dot(&n));
                    component *= KVolScaleFac;
                }
            }
        });

    UVW_f[[0, 0, 0, 0]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 1]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 2]] = Complex::new(0.0, 0.0);
    (
        UVW_f.slice(s![.., .., .., 0]).to_owned(),
        UVW_f.slice(s![.., .., .., 1]).to_owned(),
        UVW_f.slice(s![.., .., .., 2]).to_owned(),
    )
}

pub fn forgetful_turbulate(
    ae: f32,
    seed: u64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    Lx: f32,
    Ly: f32,
    Lz: f32,
    L: f32,
    gamma: f32,
    sinc_thres: f32,
) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
    let (mut U_f, mut V_f, mut W_f): (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) =
        partial_forgetful_turbulate(ae, seed, Nx, Ny, Nz, Lx, Ly, Lz, L, gamma, sinc_thres);

    let U: Array3<f32> = Utilities::irfft3d(&mut U_f);
    drop(U_f);
    let V: Array3<f32> = Utilities::irfft3d(&mut V_f);
    drop(V_f);
    let W: Array3<f32> = Utilities::irfft3d(&mut W_f);
    drop(W_f);
    (U, V, W)
}
